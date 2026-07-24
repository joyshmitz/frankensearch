//! Latency A/B for the mutual-kNN reciprocity check in graph score-diffusion.
//!
//! `neighbor_smooth` with `mutual: true` (the "no-regret refinement" for recall-bound
//! corpora, `docs/NEGATIVE_EVIDENCE.md`) gates each candidate's in-pool neighbor `n` on
//! whether `n` also points *back* to the candidate `d` (reciprocal k-NN). The LEGACY
//! ORIGINAL realizes that check as `is_similar_neighbor(graph, n, d)`:
//!
//! ```text
//! graph.neighbors(n)                       // a std-SipHash HashMap<String,_> lookup
//!      .iter().any(|e| e.edge_type == Similar && e.neighbor_doc_id == d)  // O(deg(n)) String eq
//! ```
//!
//! called once **per in-pool neighbor per candidate** → O(pool · M · deg) String comparisons
//! plus O(pool · M) `SipHash` lookups on top of the O(pool) diffusion the non-mutual path
//! already pays. That is a genuine algorithmic hot spot the correctness-first landing left in.
//!
//! CANDIDATE (`smooth_v2`, a *succinct-structure* primitive): relabel the pool to dense u32
//! indices and, in **one** pass over each candidate's Similar edges, build a packed-`u64`
//! (`src_idx << 32 | dst_idx`) `AHashSet` of every in-pool directed Similar edge. Reciprocity
//! is then an O(1) `set.contains((n_idx << 32) | d_idx)` — no re-lookup, no String eq, no
//! `SipHash`. Bit-identical to the ORIGINAL (same accumulation order, same reciprocity set).
//!
//! The non-mutual path is carried through unchanged as a regression guard (candidate must not
//! regress the default config).
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-copper \
//!   rch exec -- cargo bench -p frankensearch-fusion --profile release --bench neighbor_smooth_recip_ab
//! ```

use std::hint::black_box;
use std::time::Duration;

use ahash::{AHashMap, AHashSet};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::{DocumentGraph, EdgeType, VectorHit};
use frankensearch_fusion::bench_support::paired_median_ratio;
use frankensearch_fusion::{SmoothConfig, neighbor_smooth};

/// A pool of `pool` candidates with heavy-tailed cosine scores, and an M-nearest `Similar`
/// graph where each doc links to the `m` following docs (+ the reverse edge, so mutual-kNN
/// has real reciprocal work). Mirrors the shipped `neighbor_smooth` bench so the two are
/// directly comparable.
fn build(pool: usize, m: usize) -> (Vec<VectorHit>, DocumentGraph) {
    let hits: Vec<VectorHit> = (0..pool)
        .map(|i| VectorHit {
            index: u32::try_from(i).expect("benchmark pool indices fit in u32"),
            score: 1.0 / ((i as f32) + 1.0),
            doc_id: format!("doc{i:06}").into(),
        })
        .collect();
    let mut graph = DocumentGraph::new();
    for i in 0..pool {
        for j in 1..=m {
            let nbr = (i + j) % pool;
            graph.add_edge(
                format!("doc{i:06}"),
                format!("doc{nbr:06}"),
                EdgeType::Similar,
                1.0 - (j as f32) * 0.01,
            );
            graph.add_edge(
                format!("doc{nbr:06}"),
                format!("doc{i:06}"),
                EdgeType::Similar,
                1.0 - (j as f32) * 0.01,
            );
        }
    }
    (hits, graph)
}

/// Candidate implementation, written exactly as it will land: the **non-mutual** path is the
/// unchanged ORIGINAL (compute-floored single-hop diffusion — no structural headroom); the
/// **mutual** path replaces the per-neighbor `is_similar_neighbor` scan with a pool-restricted
/// integer relabel + packed-`u64` reciprocity `AHashSet`. Semantically identical to
/// `frankensearch_fusion::neighbor_smooth` for both configs.
fn smooth_v2(hits: &[VectorHit], graph: &DocumentGraph, config: &SmoothConfig) -> Vec<VectorHit> {
    if config.is_identity() || graph.is_empty() || hits.is_empty() {
        return hits.to_vec();
    }
    let alpha = config.alpha;
    let keep = 1.0 - alpha;

    if !config.mutual {
        // Unchanged ORIGINAL: pool score map + direct diffusion.
        let pool: AHashMap<&str, f32> = hits.iter().map(|h| (h.doc_id.as_str(), h.score)).collect();
        return hits
            .iter()
            .map(|h| {
                let mut sum = 0.0f32;
                let mut count = 0usize;
                let mut examined = 0usize;
                for edge in graph.neighbors(h.doc_id.as_str()) {
                    if edge.edge_type != EdgeType::Similar {
                        continue;
                    }
                    if examined == config.m {
                        break;
                    }
                    examined += 1;
                    let Some(&nbr_score) = pool.get(edge.neighbor_doc_id.as_str()) else {
                        continue;
                    };
                    sum += nbr_score;
                    count += 1;
                }
                let nbr_mean = if count == 0 {
                    h.score
                } else {
                    sum / count as f32
                };
                VectorHit {
                    index: h.index,
                    score: keep * h.score + alpha * nbr_mean,
                    doc_id: h.doc_id.clone(),
                }
            })
            .collect();
    }

    // Mutual path. Pool: doc_id -> (dense pool index, score). One lookup yields both the neighbor
    // score (diffusion) and its index (reciprocity), so each neighbor string is hashed once.
    let n = hits.len();
    let mut pool: AHashMap<&str, (u32, f32)> = AHashMap::with_capacity(n);
    for (i, h) in hits.iter().enumerate() {
        pool.insert(
            h.doc_id.as_str(),
            (
                u32::try_from(i).expect("benchmark pool indices fit in u32"),
                h.score,
            ),
        );
    }
    // Reciprocity set: every in-pool directed Similar edge, packed `src_idx << 32 | dst_idx`.
    // Built in one full-adjacency pass — matching `is_similar_neighbor`'s un-capped scan.
    let mut recip: AHashSet<u64> = AHashSet::with_capacity(n * config.m);
    for (i, h) in hits.iter().enumerate() {
        let src = (i as u64) << 32;
        for edge in graph.neighbors(h.doc_id.as_str()) {
            if edge.edge_type != EdgeType::Similar {
                continue;
            }
            if let Some(&(dst_idx, _)) = pool.get(edge.neighbor_doc_id.as_str()) {
                recip.insert(src | u64::from(dst_idx));
            }
        }
    }

    hits.iter()
        .enumerate()
        .map(|(i, h)| {
            let doc_idx = i as u64;
            let mut sum = 0.0f32;
            let mut count = 0usize;
            let mut examined = 0usize;
            for edge in graph.neighbors(h.doc_id.as_str()) {
                if edge.edge_type != EdgeType::Similar {
                    continue;
                }
                if examined == config.m {
                    break;
                }
                examined += 1;
                let Some(&(nbr_idx, nbr_score)) = pool.get(edge.neighbor_doc_id.as_str()) else {
                    continue;
                };
                // is_similar_neighbor(graph, neighbor, doc) == "neighbor -> doc" is an edge.
                if !recip.contains(&((u64::from(nbr_idx) << 32) | doc_idx)) {
                    continue;
                }
                sum += nbr_score;
                count += 1;
            }
            let nbr_mean = if count == 0 {
                h.score
            } else {
                sum / count as f32
            };
            VectorHit {
                index: h.index,
                score: keep * h.score + alpha * nbr_mean,
                doc_id: h.doc_id.clone(),
            }
        })
        .collect()
}

/// Candidate v3 (the version to land): a single full-adjacency pass hashes each Similar edge
/// **exactly once**, simultaneously (a) inserting every in-pool directed edge into the packed
/// reciprocity set and (b) capturing the M-capped in-pool forward neighbors as a flat integer
/// CSR. The diffusion pass is then pure integer work — no `graph.neighbors` `SipHash` lookup, no
/// string hashing. Non-mutual is the unchanged ORIGINAL. Bit-identical to `neighbor_smooth`.
fn smooth_v3(hits: &[VectorHit], graph: &DocumentGraph, config: &SmoothConfig) -> Vec<VectorHit> {
    if config.is_identity() || graph.is_empty() || hits.is_empty() {
        return hits.to_vec();
    }
    let alpha = config.alpha;
    let keep = 1.0 - alpha;

    if !config.mutual {
        let pool: AHashMap<&str, f32> = hits.iter().map(|h| (h.doc_id.as_str(), h.score)).collect();
        return hits
            .iter()
            .map(|h| {
                let mut sum = 0.0f32;
                let mut count = 0usize;
                let mut examined = 0usize;
                for edge in graph.neighbors(h.doc_id.as_str()) {
                    if edge.edge_type != EdgeType::Similar {
                        continue;
                    }
                    if examined == config.m {
                        break;
                    }
                    examined += 1;
                    let Some(&nbr_score) = pool.get(edge.neighbor_doc_id.as_str()) else {
                        continue;
                    };
                    sum += nbr_score;
                    count += 1;
                }
                let nbr_mean = if count == 0 {
                    h.score
                } else {
                    sum / count as f32
                };
                VectorHit {
                    index: h.index,
                    score: keep * h.score + alpha * nbr_mean,
                    doc_id: h.doc_id.clone(),
                }
            })
            .collect();
    }

    let n = hits.len();
    let m = config.m;
    let mut pool: AHashMap<&str, (u32, f32)> = AHashMap::with_capacity(n);
    for (i, h) in hits.iter().enumerate() {
        pool.insert(
            h.doc_id.as_str(),
            (
                u32::try_from(i).expect("benchmark pool indices fit in u32"),
                h.score,
            ),
        );
    }

    // One adjacency walk: build the reciprocity set (all in-pool Similar edges, uncapped) and the
    // M-capped in-pool forward CSR. Each neighbor string is hashed exactly once.
    let mut recip: AHashSet<u64> = AHashSet::with_capacity(n * m);
    let mut fwd_flat: Vec<(u32, f32)> = Vec::with_capacity(n * m);
    let mut fwd_off: Vec<u32> = Vec::with_capacity(n + 1);
    fwd_off.push(0);
    for (i, h) in hits.iter().enumerate() {
        let src = (i as u64) << 32;
        let mut examined = 0usize;
        for edge in graph.neighbors(h.doc_id.as_str()) {
            if edge.edge_type != EdgeType::Similar {
                continue;
            }
            let hit = pool.get(edge.neighbor_doc_id.as_str()).copied();
            if let Some((dst_idx, _)) = hit {
                recip.insert(src | u64::from(dst_idx)); // uncapped: matches is_similar_neighbor
            }
            if examined < m {
                examined += 1;
                if let Some((dst_idx, dst_score)) = hit {
                    fwd_flat.push((dst_idx, dst_score));
                }
            }
        }
        fwd_off.push(u32::try_from(fwd_flat.len()).expect("benchmark CSR offsets fit in u32"));
    }

    // Integer-only diffusion: reciprocity is O(1) set membership.
    (0..n)
        .map(|i| {
            let h = &hits[i];
            let doc_idx = i as u64;
            let (lo, hi) = (fwd_off[i] as usize, fwd_off[i + 1] as usize);
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for &(nbr_idx, nbr_score) in &fwd_flat[lo..hi] {
                if !recip.contains(&((u64::from(nbr_idx) << 32) | doc_idx)) {
                    continue;
                }
                sum += nbr_score;
                count += 1;
            }
            let nbr_mean = if count == 0 {
                h.score
            } else {
                sum / count as f32
            };
            VectorHit {
                index: h.index,
                score: keep * h.score + alpha * nbr_mean,
                doc_id: h.doc_id.clone(),
            }
        })
        .collect()
}

fn assert_bit_identical(hits: &[VectorHit], graph: &DocumentGraph, cfg: &SmoothConfig, tag: &str) {
    let a = neighbor_smooth(hits, graph, cfg);
    let b = smooth_v2(hits, graph, cfg);
    let c = smooth_v3(hits, graph, cfg);
    assert_eq!(a.len(), b.len(), "{tag}: v2 length mismatch");
    assert_eq!(a.len(), c.len(), "{tag}: v3 length mismatch");
    for ((x, y), z) in a.iter().zip(&b).zip(&c) {
        assert_eq!(x.index, y.index, "{tag}: v2 index mismatch");
        assert_eq!(x.index, z.index, "{tag}: v3 index mismatch");
        assert_eq!(x.doc_id, y.doc_id, "{tag}: v2 doc_id mismatch");
        assert_eq!(x.doc_id, z.doc_id, "{tag}: v3 doc_id mismatch");
        assert_eq!(
            x.score.to_bits(),
            y.score.to_bits(),
            "{tag}: v2 score mismatch"
        );
        assert_eq!(
            x.score.to_bits(),
            z.score.to_bits(),
            "{tag}: v3 score mismatch ({} vs {})",
            x.score,
            z.score
        );
    }
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("neighbor_smooth_recip");
    g.sample_size(50);
    g.warm_up_time(Duration::from_millis(400));
    g.measurement_time(Duration::from_millis(2000));

    let smooth = SmoothConfig {
        alpha: 0.3,
        m: 10,
        mutual: false,
    };
    let mutual = SmoothConfig {
        alpha: 0.3,
        m: 10,
        mutual: true,
    };

    for &pool in &[50usize, 100, 1000] {
        let (hits, graph) = build(pool, 10);
        // Parity gate: candidate must be bit-identical to the shipped kernel on both configs.
        assert_bit_identical(&hits, &graph, &smooth, &format!("non_mutual/{pool}"));
        assert_bit_identical(&hits, &graph, &mutual, &format!("mutual/{pool}"));

        // Mutual is the target (the O(pool·M·deg) reciprocity scan). v3 (single-hash CSR) is the
        // land candidate; v2 (eager recip re-hash) is kept for the stepping-stone comparison.
        // ORIG is measured first AND last so the first-arm ordering bias is bracketed.
        g.bench_with_input(BenchmarkId::new("mutual_ORIG", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(neighbor_smooth(
                    black_box(&hits),
                    black_box(&graph),
                    &mutual,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("mutual_v3", pool), &(), |b, ()| {
            b.iter(|| black_box(smooth_v3(black_box(&hits), black_box(&graph), &mutual)));
        });
        g.bench_with_input(BenchmarkId::new("mutual_v2", pool), &(), |b, ()| {
            b.iter(|| black_box(smooth_v2(black_box(&hits), black_box(&graph), &mutual)));
        });
        g.bench_with_input(BenchmarkId::new("mutual_ORIG2", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(neighbor_smooth(
                    black_box(&hits),
                    black_box(&graph),
                    &mutual,
                ))
            });
        });
        // Non-mutual regression guard: candidate must not lose the default path.
        g.bench_with_input(BenchmarkId::new("nonmutual_ORIG", pool), &(), |b, ()| {
            b.iter(|| {
                black_box(neighbor_smooth(
                    black_box(&hits),
                    black_box(&graph),
                    &smooth,
                ))
            });
        });
        g.bench_with_input(BenchmarkId::new("nonmutual_v3", pool), &(), |b, ()| {
            b.iter(|| black_box(smooth_v3(black_box(&hits), black_box(&graph), &smooth)));
        });

        // ── DECIDABILITY: alternating-round paired sampler + A/A null control ──
        //
        // The criterion arms above run as separate benchmarks minutes apart, so worker
        // drift between them is not cancelled. The paired sampler runs both arms in ONE
        // routine in alternating rounds; gate on the median against the A/A null's
        // observed spread. One null+lever pair per comparison: the mutual lever (v3 and
        // the v2 stepping-stone) and the non-mutual regression guard.
        let mutual_orig = || {
            black_box(neighbor_smooth(
                black_box(&hits),
                black_box(&graph),
                &mutual,
            ));
        };
        let mutual_v3 = || {
            black_box(smooth_v3(black_box(&hits), black_box(&graph), &mutual));
        };
        let null = paired_median_ratio(41, 8, mutual_orig, mutual_orig);
        let lever = paired_median_ratio(41, 8, mutual_orig, mutual_v3);
        eprintln!(
            "[null]  neighbor_smooth_recip mutual/{pool}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] neighbor_smooth_recip mutual/{pool}: v3/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
        let mutual_orig = || {
            black_box(neighbor_smooth(
                black_box(&hits),
                black_box(&graph),
                &mutual,
            ));
        };
        let mutual_v2 = || {
            black_box(smooth_v2(black_box(&hits), black_box(&graph), &mutual));
        };
        let null = paired_median_ratio(41, 8, mutual_orig, mutual_orig);
        let lever = paired_median_ratio(41, 8, mutual_orig, mutual_v2);
        eprintln!(
            "[null]  neighbor_smooth_recip mutual/{pool}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] neighbor_smooth_recip mutual/{pool}: v2/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
        let nonmutual_orig = || {
            black_box(neighbor_smooth(
                black_box(&hits),
                black_box(&graph),
                &smooth,
            ));
        };
        let nonmutual_v3 = || {
            black_box(smooth_v3(black_box(&hits), black_box(&graph), &smooth));
        };
        let null = paired_median_ratio(41, 8, nonmutual_orig, nonmutual_orig);
        let lever = paired_median_ratio(41, 8, nonmutual_orig, nonmutual_v3);
        eprintln!(
            "[null]  neighbor_smooth_recip nonmutual/{pool}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] neighbor_smooth_recip nonmutual/{pool}: v3/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
