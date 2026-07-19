//! `fuse_expanded_payloads` accumulation A/B: the cross-query RRF fusion over
//! query-expansion variants (runtime.rs) builds FIVE maps keyed by doc path and
//! calls `.entry(hit.path.clone())` per hit across every payload — up to 5 owned
//! `String` allocations per hit, and the fusion case (a doc in ≥2 variants) is
//! exactly when the key already exists so the clone is pure waste. This bench
//! replicates that accumulation loop (+ the final score sort and top-`limit`
//! output build) three ways to attribute the two independent levers:
//!   - `clone_sip`    : current — `HashMap<String,_>` (SipHash), `.entry(clone)`
//!   - `borrow_sip`   : `HashMap<&str,_>` (SipHash), `.entry(&str)` — clone elided
//!   - `borrow_ahash` : `AHashMap<&str,_>` (ahash), `.entry(&str)` — clone + hasher
//! All three produce the identical ranked output (asserted).
use std::collections::HashMap;
use std::hint::black_box;

use ahash::AHashMap;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_fusion::bench_support::paired_median_ratio;

/// Minimal stand-in for `SearchHitPayload` carrying the fields the fusion reads.
#[derive(Clone)]
struct Hit {
    path: String,
    rank: usize,
    snippet: Option<String>,
    lexical_rank: Option<usize>,
    semantic_rank: Option<usize>,
}

struct Payload {
    hits: Vec<Hit>,
}

/// P expansion variants, each ranking H docs sampled (with overlap) from a pool
/// of D distinct paths — so many docs appear in ≥2 variants (the fusion case).
fn make_payloads(p: usize, h: usize, pool: usize) -> Vec<Payload> {
    (0..p)
        .map(|pi| {
            let hits = (0..h)
                .map(|hi| {
                    // Deterministic overlapping sample: stride by variant so
                    // variants share a large common subset of the pool.
                    let doc = (hi * (pi + 1)) % pool;
                    Hit {
                        path: format!("docs/section-{:02}/file-{:04}.md", doc % 16, doc),
                        rank: hi,
                        snippet: Some(format!("snippet for doc {doc}")),
                        lexical_rank: if hi % 2 == 0 { Some(hi) } else { None },
                        semantic_rank: if hi % 3 == 0 { Some(hi) } else { None },
                    }
                })
                .collect();
            Payload { hits }
        })
        .collect()
}

const K: f64 = 60.0;
const LIMIT: usize = 10;

/// Current production shape: owned `String` keys, `.entry(clone())` per hit.
fn fuse_clone_sip(payloads: &[Payload]) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut snippets: HashMap<String, String> = HashMap::new();
    let mut best_lexical_rank: HashMap<String, usize> = HashMap::new();
    let mut best_semantic_rank: HashMap<String, usize> = HashMap::new();
    let mut appeared_in_count: HashMap<String, usize> = HashMap::new();
    for payload in payloads {
        for hit in &payload.hits {
            let contribution = 1.0 / (K + hit.rank as f64);
            *scores.entry(hit.path.clone()).or_default() += contribution;
            *appeared_in_count.entry(hit.path.clone()).or_default() += 1;
            if let Some(snippet) = &hit.snippet {
                snippets
                    .entry(hit.path.clone())
                    .or_insert_with(|| snippet.clone());
            }
            if let Some(lr) = hit.lexical_rank {
                best_lexical_rank
                    .entry(hit.path.clone())
                    .and_modify(|e| *e = (*e).min(lr))
                    .or_insert(lr);
            }
            if let Some(sr) = hit.semantic_rank {
                best_semantic_rank
                    .entry(hit.path.clone())
                    .and_modify(|e| *e = (*e).min(sr))
                    .or_insert(sr);
            }
        }
    }
    let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked
        .into_iter()
        .take(LIMIT)
        .map(|(path, score)| {
            let _ = (
                best_lexical_rank.get(&path),
                best_semantic_rank.get(&path),
                appeared_in_count.get(&path),
                snippets.get(&path),
            );
            (path, score)
        })
        .collect()
}

/// Borrow `&str` keys (payloads outlive the fusion), std SipHash.
fn fuse_borrow_sip(payloads: &[Payload]) -> Vec<(String, f64)> {
    let mut scores: HashMap<&str, f64> = HashMap::new();
    let mut snippets: HashMap<&str, &str> = HashMap::new();
    let mut best_lexical_rank: HashMap<&str, usize> = HashMap::new();
    let mut best_semantic_rank: HashMap<&str, usize> = HashMap::new();
    let mut appeared_in_count: HashMap<&str, usize> = HashMap::new();
    for payload in payloads {
        for hit in &payload.hits {
            let key = hit.path.as_str();
            let contribution = 1.0 / (K + hit.rank as f64);
            *scores.entry(key).or_default() += contribution;
            *appeared_in_count.entry(key).or_default() += 1;
            if let Some(snippet) = &hit.snippet {
                snippets.entry(key).or_insert_with(|| snippet.as_str());
            }
            if let Some(lr) = hit.lexical_rank {
                best_lexical_rank
                    .entry(key)
                    .and_modify(|e| *e = (*e).min(lr))
                    .or_insert(lr);
            }
            if let Some(sr) = hit.semantic_rank {
                best_semantic_rank
                    .entry(key)
                    .and_modify(|e| *e = (*e).min(sr))
                    .or_insert(sr);
            }
        }
    }
    let mut ranked: Vec<(&str, f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    });
    ranked
        .into_iter()
        .take(LIMIT)
        .map(|(path, score)| {
            let _ = (
                best_lexical_rank.get(path),
                best_semantic_rank.get(path),
                appeared_in_count.get(path),
                snippets.get(path),
            );
            (path.to_owned(), score)
        })
        .collect()
}

/// Borrow `&str` keys + `ahash` — the proposed production shape.
fn fuse_borrow_ahash(payloads: &[Payload]) -> Vec<(String, f64)> {
    let mut scores: AHashMap<&str, f64> = AHashMap::new();
    let mut snippets: AHashMap<&str, &str> = AHashMap::new();
    let mut best_lexical_rank: AHashMap<&str, usize> = AHashMap::new();
    let mut best_semantic_rank: AHashMap<&str, usize> = AHashMap::new();
    let mut appeared_in_count: AHashMap<&str, usize> = AHashMap::new();
    for payload in payloads {
        for hit in &payload.hits {
            let key = hit.path.as_str();
            let contribution = 1.0 / (K + hit.rank as f64);
            *scores.entry(key).or_default() += contribution;
            *appeared_in_count.entry(key).or_default() += 1;
            if let Some(snippet) = &hit.snippet {
                snippets.entry(key).or_insert_with(|| snippet.as_str());
            }
            if let Some(lr) = hit.lexical_rank {
                best_lexical_rank
                    .entry(key)
                    .and_modify(|e| *e = (*e).min(lr))
                    .or_insert(lr);
            }
            if let Some(sr) = hit.semantic_rank {
                best_semantic_rank
                    .entry(key)
                    .and_modify(|e| *e = (*e).min(sr))
                    .or_insert(sr);
            }
        }
    }
    let mut ranked: Vec<(&str, f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    });
    ranked
        .into_iter()
        .take(LIMIT)
        .map(|(path, score)| {
            let _ = (
                best_lexical_rank.get(path),
                best_semantic_rank.get(path),
                appeared_in_count.get(path),
                snippets.get(path),
            );
            (path.to_owned(), score)
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("expand_fuse");
    // (variants P, hits-per-variant H, distinct-pool D)
    for &(p, h, pool) in &[(3usize, 20usize, 40usize), (5, 40, 120), (6, 60, 200)] {
        let payloads = make_payloads(p, h, pool);
        // All three arms must yield the identical ranked (path, score) output.
        let a = fuse_clone_sip(&payloads);
        let b = fuse_borrow_sip(&payloads);
        let d = fuse_borrow_ahash(&payloads);
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), d.len());
        for i in 0..a.len() {
            assert_eq!(a[i].0, b[i].0);
            assert_eq!(a[i].0, d[i].0);
            assert_eq!(a[i].1.to_bits(), b[i].1.to_bits());
            assert_eq!(a[i].1.to_bits(), d[i].1.to_bits());
        }
        let id = format!("p{p}_h{h}");
        g.bench_with_input(BenchmarkId::new("clone_sip", &id), &(), |bch, ()| {
            bch.iter(|| black_box(fuse_clone_sip(black_box(&payloads))));
        });
        g.bench_with_input(BenchmarkId::new("borrow_sip", &id), &(), |bch, ()| {
            bch.iter(|| black_box(fuse_borrow_sip(black_box(&payloads))));
        });
        g.bench_with_input(BenchmarkId::new("borrow_ahash", &id), &(), |bch, ()| {
            bch.iter(|| black_box(fuse_borrow_ahash(black_box(&payloads))));
        });

        // ── DECIDABILITY: alternating-round paired sampler + A/A null control ──
        //
        // The criterion arms above CANNOT decide these levers: criterion runs them as
        // separate benchmarks minutes apart, so worker drift between them is not
        // cancelled. The paired sampler runs both arms in ONE routine in alternating
        // rounds and takes the median per-round ratio; gate on the median against the
        // A/A null's observed spread, not on cv. Two independent levers (borrow_sip,
        // borrow_ahash) against the same clone_sip base get one null + two levers.
        let clone_sip = || {
            black_box(fuse_clone_sip(black_box(&payloads)));
        };
        let borrow_sip = || {
            black_box(fuse_borrow_sip(black_box(&payloads)));
        };
        let borrow_ahash = || {
            black_box(fuse_borrow_ahash(black_box(&payloads)));
        };
        let null = paired_median_ratio(41, 8, clone_sip, clone_sip);
        let sip_lever = paired_median_ratio(41, 8, clone_sip, borrow_sip);
        let ahash_lever = paired_median_ratio(41, 8, clone_sip, borrow_ahash);
        eprintln!(
            "[null]  expand_fuse {id}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] expand_fuse {id}: borrow_sip median {:.4} p5 {:.4} p95 {:.4} -> {}",
            sip_lever.median,
            sip_lever.p5,
            sip_lever.p95,
            if sip_lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
        eprintln!(
            "[lever] expand_fuse {id}: borrow_ahash median {:.4} p5 {:.4} p95 {:.4} -> {}",
            ahash_lever.median,
            ahash_lever.p5,
            ahash_lever.p95,
            if ahash_lever.decidable_against(&null) {
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
