//! Graph score-diffusion (kNN neighbor smoothing) over a dense doc-doc similarity graph.
//!
//! This is a *label-propagation* / *manifold-ranking* primitive, structurally distinct
//! from tier fusion (RRF, [`crate::rrf::pool_minmax_fuse`]): instead of combining two
//! rankings for a query, it propagates each candidate's dense score along the doc-doc
//! nearest-neighbor graph. A below-threshold relevant document sitting inside a cluster
//! of confident results is *rescued* by borrowing its neighbors' scores.
//!
//! For a candidate `d` with dense (cosine-to-query) score `s(d)` and dense k-NN neighbors
//! `N(d)`:
//!
//! ```text
//! smoothed(d) = (1 - α) · s(d)  +  α · mean_{n ∈ N(d) ∩ pool} s(n)
//! ```
//!
//! Only the `Similar` (dense-NN) edges of the supplied [`DocumentGraph`] are used, so the
//! same graph may also carry `Reference`/`CoLocation` edges for other consumers (e.g. the
//! graph-PageRank ranker) without interfering. **Neighbor lookup is restricted to the
//! retrieved candidate pool** — the deployable form: the engine only knows `cos(q, ·)` for
//! candidates it retrieved. A candidate with no in-pool neighbors is left unchanged
//! (α collapses to 0 for it), so the transform is a no-op on isolated docs and on an empty
//! graph.
//!
//! ## Measured quality (BEIR, BGE-small hybrid, `docs/NEGATIVE_EVIDENCE.md`)
//!
//! The pool-restricted deployable form measured here is a **recall-mechanism**: it lifts
//! below-threshold relevants near confident clusters. It wins largest on recall-bound
//! corpora (nfcorpus hybrid nDCG@10 up to **+0.0114** at α=0.3) and can slightly regress
//! saturated corpora (scifact ~−0.003) — so it is **recall-saturation-gated**, matching the
//! smoothing tuning arc (`543684e`→`4fd5802`). Default α=0.3 / M=10 is the never-worse
//! knob on recall-bound corpora; `mutual` (reciprocal-edge) is the no-regret refinement.
//!
//! The kernel is O(pool · M): nearly free atop an existing ANN neighbor graph.

use ahash::{AHashMap, AHashSet};
use frankensearch_core::{DocumentGraph, EdgeType, VectorHit};

/// Parameters for [`neighbor_smooth`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothConfig {
    /// Diffusion weight in `[0, 1]`. `0` = identity (no smoothing), `1` = pure neighbor
    /// mean. Default `0.3`: the label-free never-worse setting on recall-bound corpora.
    /// Values `≤ 0` or non-finite are treated as identity.
    pub alpha: f32,
    /// Number of nearest `Similar` neighbors to average per candidate. Default `10`.
    pub m: usize,
    /// When `true`, only count a neighbor `n` of `d` if `d` is *also* a `Similar` neighbor
    /// of `n` (mutual / reciprocal k-NN). Cuts hub and one-way-edge noise; the no-regret
    /// refinement for recall-bound corpora. Default `false`.
    pub mutual: bool,
}

impl Default for SmoothConfig {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            m: 10,
            mutual: false,
        }
    }
}

impl SmoothConfig {
    /// Whether this config is a guaranteed no-op (identity), so callers can skip the pass.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        !self.alpha.is_finite() || self.alpha <= 0.0 || self.m == 0
    }
}

/// Diffuse each candidate's dense score along the dense k-NN graph (pool-restricted).
///
/// Returns a new vector with the same documents (and `index`) but smoothed `score`s, ready
/// to be re-sorted by [`VectorHit::cmp_by_score`]. The input order is preserved; callers
/// that need a ranking should sort the result.
///
/// This is a pure, allocation-bounded transform: it borrows `hits` and `graph` and touches
/// no global state. On an identity config, empty graph, or empty pool it clones `hits`
/// unchanged.
///
/// **Callers that feed a rank-based fusion operator want [`neighbor_smooth_ranked`] instead** —
/// see its docs for why re-sorting is a correctness requirement, not a cosmetic one.
#[must_use]
pub fn neighbor_smooth(
    hits: &[VectorHit],
    graph: &DocumentGraph,
    config: &SmoothConfig,
) -> Vec<VectorHit> {
    if config.is_identity() || graph.is_empty() || hits.is_empty() {
        return hits.to_vec();
    }

    // Reciprocal (mutual-kNN) gating is a distinct, heavier kernel — route it to the
    // integer-relabeled path that answers reciprocity in O(1) instead of an O(deg) String scan.
    if config.mutual {
        return mutual_neighbor_smooth(hits, graph, config);
    }

    // Pool score map: cos(q, ·) for every retrieved candidate. Neighbor lookup is
    // restricted to this set (the engine only scored these docs for this query).
    let pool: AHashMap<&str, f32> = hits
        .iter()
        .map(|h| (h.doc_id.as_str(), h.score))
        .collect();

    let alpha = config.alpha;
    let keep = 1.0 - alpha;
    hits.iter()
        .map(|h| {
            let nbr_mean = in_pool_neighbor_mean(h.doc_id.as_str(), h.score, graph, &pool, config);
            VectorHit {
                index: h.index,
                score: keep * h.score + alpha * nbr_mean,
                doc_id: h.doc_id.clone(),
            }
        })
        .collect()
}

/// Mean of `doc`'s in-pool `Similar` neighbors' scores (up to `m`), or `self_score` when it
/// has none — making α collapse to 0 for isolated candidates.
fn in_pool_neighbor_mean(
    doc: &str,
    self_score: f32,
    graph: &DocumentGraph,
    pool: &AHashMap<&str, f32>,
    config: &SmoothConfig,
) -> f32 {
    // Cap on the `m` NEAREST Similar edges examined (the k-NN set), then average the
    // in-pool subset among them — matching the measured pool-restricted semantics
    // (`nbr[d]` = m nearest; mean over those that were retrieved). Assumes the graph stores
    // each node's Similar edges nearest-first, as a k-NN builder produces.
    let mut sum = 0.0f32;
    let mut count = 0usize;
    let mut examined = 0usize;
    for edge in graph.neighbors(doc) {
        if edge.edge_type != EdgeType::Similar {
            continue;
        }
        if examined == config.m {
            break;
        }
        examined += 1;
        let neighbor = edge.neighbor_doc_id.as_str();
        let Some(&neighbor_score) = pool.get(neighbor) else {
            continue; // out-of-pool: cos(q, neighbor) unknown → skip (pool-restricted)
        };
        sum += neighbor_score;
        count += 1;
    }
    if count == 0 {
        self_score
    } else {
        sum / count as f32
    }
}

/// Mutual (reciprocal-kNN) diffusion — the `config.mutual` kernel.
///
/// Structurally distinct from the direct non-mutual path: a *single* walk over each candidate's
/// `Similar` adjacency hashes every neighbor string exactly once, simultaneously
///
/// 1. recording every **in-pool directed** `Similar` edge `src → dst` in a packed-`u64`
///    (`src_idx << 32 | dst_idx`) reciprocity `AHashSet` (uncapped — matching the ORIGINAL
///    `is_similar_neighbor`'s full-adjacency scan), and
/// 2. capturing the in-pool subset of each candidate's first `m` *examined* `Similar` edges (its
///    diffusion neighbors) as a flat integer CSR of `(neighbor_idx, neighbor_score)`.
///
/// The diffusion is then pure integer work: reciprocity — "does neighbor `n` point back to
/// candidate `d`?" — becomes an O(1) `recip.contains((n_idx << 32) | d_idx)`, replacing the
/// ORIGINAL's O(deg(n)) linear `String`-equality scan per neighbor per candidate. Bit-identical
/// to the direct form (same edge-visit order, same reciprocity relation, same accumulation
/// order). Measured **1.21×/1.35×/1.50× faster** at pool 50/100/1000 (`neighbor_smooth_recip_ab`,
/// `docs/PERF_LEDGER.md`); the win grows with the pool because more reciprocity checks amortize
/// the O(deg) → O(1) shift.
// Pool indices and CSR offsets are bounded by the retrieved candidate count (thousands), always
// well below `u32::MAX`, so the `u32` relabel/packing is exact for any realistic pool.
#[allow(clippy::cast_possible_truncation)]
fn mutual_neighbor_smooth(
    hits: &[VectorHit],
    graph: &DocumentGraph,
    config: &SmoothConfig,
) -> Vec<VectorHit> {
    let n = hits.len();
    let m = config.m;

    // Pool: doc_id → (dense pool index, cos-to-query score). One lookup yields both the neighbor
    // score (diffusion) and its index (reciprocity), so each neighbor string is hashed once.
    let mut pool: AHashMap<&str, (u32, f32)> = AHashMap::with_capacity(n);
    for (i, h) in hits.iter().enumerate() {
        pool.insert(h.doc_id.as_str(), (i as u32, h.score));
    }

    // One adjacency walk builds both structures.
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
                recip.insert(src | u64::from(dst_idx)); // uncapped: full-adjacency reciprocity
            }
            // The M-cap counts every examined `Similar` edge (in-pool or not), matching the
            // ORIGINAL; only in-pool ones become diffusion neighbors.
            if examined < m {
                examined += 1;
                if let Some((dst_idx, dst_score)) = hit {
                    fwd_flat.push((dst_idx, dst_score));
                }
            }
        }
        fwd_off.push(fwd_flat.len() as u32);
    }

    let alpha = config.alpha;
    let keep = 1.0 - alpha;
    (0..n)
        .map(|i| {
            let h = &hits[i];
            let doc_idx = i as u64;
            let (lo, hi) = (fwd_off[i] as usize, fwd_off[i + 1] as usize);
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for &(nbr_idx, nbr_score) in &fwd_flat[lo..hi] {
                // is_similar_neighbor(graph, neighbor, doc) == "neighbor → doc" is an edge.
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

/// [`neighbor_smooth`] followed by a deterministic descending re-sort.
///
/// **Why the re-sort is a correctness requirement.** The rank-based fusion operators
/// (`rrf_fuse_with_graph_merge_unique` and friends) walk the `semantic` slice *once, in order*,
/// and assign each hit's rank from its **position** — they never re-sort it, because a vector
/// index already returns score-sorted hits. Smoothing rewrites those scores, so a doc the
/// diffusion promoted would keep its old, worse rank and the promotion would be silently
/// discarded. Worse, the merge's near-sorted assumption would be violated. Re-sorting restores
/// the invariant "position == rank" that fusion depends on.
///
/// Ties break on `doc_id` so the pool order is a total order and the pipeline stays replayable.
/// Identity config / empty graph / empty pool short-circuits to `neighbor_smooth`'s clone, whose
/// input was already sorted, so the sort is skipped entirely.
#[must_use]
pub fn neighbor_smooth_ranked(
    hits: &[VectorHit],
    graph: &DocumentGraph,
    config: &SmoothConfig,
) -> Vec<VectorHit> {
    if config.is_identity() || graph.is_empty() || hits.is_empty() {
        return hits.to_vec();
    }
    let mut smoothed = neighbor_smooth(hits, graph, config);
    smoothed.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch_core::EdgeType;

    fn hit(doc: &str, score: f32) -> VectorHit {
        VectorHit {
            index: 0,
            score,
            doc_id: doc.into(),
        }
    }

    /// Build a `Similar`-edge graph from `(from, to)` pairs, weight 1.0.
    fn sim_graph(edges: &[(&str, &str)]) -> DocumentGraph {
        let mut g = DocumentGraph::new();
        for (a, b) in edges {
            g.add_edge(*a, *b, EdgeType::Similar, 1.0);
        }
        g
    }

    #[test]
    fn alpha_zero_is_identity() {
        let hits = vec![hit("a", 0.9), hit("b", 0.5)];
        let g = sim_graph(&[("a", "b"), ("b", "a")]);
        let cfg = SmoothConfig {
            alpha: 0.0,
            ..Default::default()
        };
        let out = neighbor_smooth(&hits, &g, &cfg);
        assert_eq!(out, hits);
    }

    // ─── neighbor_smooth_ranked (searcher wiring, bd-kdjr) ──────────────────

    /// PARITY GATE. The disabled config must leave the pool byte-identical — same order, same
    /// score bits. This is what makes wiring the pass into the searcher a no-op by default.
    #[test]
    fn ranked_identity_is_byte_identical() {
        let hits = vec![hit("a", 0.9), hit("b", 0.5), hit("c", 0.4)];
        let g = sim_graph(&[("a", "b"), ("b", "a"), ("b", "c")]);
        let off = SmoothConfig {
            alpha: 0.0,
            ..Default::default()
        };
        let out = neighbor_smooth_ranked(&hits, &g, &off);
        assert_eq!(out.len(), hits.len());
        for (x, y) in out.iter().zip(&hits) {
            assert_eq!(x.doc_id, y.doc_id);
            assert_eq!(x.score.to_bits(), y.score.to_bits(), "doc {}", x.doc_id);
        }
        // An empty graph is the other disabled path.
        let on = SmoothConfig::default();
        let empty = DocumentGraph::new();
        assert_eq!(neighbor_smooth_ranked(&hits, &empty, &on), hits);
    }

    /// CORRECTNESS GATE (the reason `_ranked` exists). Fusion assigns ranks by *position*, so a
    /// doc the diffusion promoted must actually move up the slice. Plain `neighbor_smooth` leaves
    /// it in place with a better score — the promotion would be silently discarded by RRF.
    #[test]
    fn ranked_reorders_a_promoted_doc_where_plain_smooth_does_not() {
        // `c` starts last but sits in a tight cluster with the two top hits, so diffusion lifts
        // it above `b`, a loner.
        let hits = vec![hit("a", 0.90), hit("b", 0.60), hit("c", 0.55)];
        let g = sim_graph(&[("c", "a"), ("a", "c"), ("c", "d"), ("d", "c")]);
        let cfg = SmoothConfig {
            alpha: 0.5,
            m: 10,
            mutual: false,
        };

        let plain = neighbor_smooth(&hits, &g, &cfg);
        let ranked = neighbor_smooth_ranked(&hits, &g, &cfg);

        // Same multiset of scores; only the order differs.
        let plain_c = plain.iter().find(|h| h.doc_id == "c").unwrap().score;
        let ranked_c = ranked.iter().find(|h| h.doc_id == "c").unwrap().score;
        assert_eq!(plain_c.to_bits(), ranked_c.to_bits());

        // Diffusion lifted c above b...
        let b = plain.iter().find(|h| h.doc_id == "b").unwrap().score;
        assert!(
            ranked_c > b,
            "smoothing should lift c ({ranked_c}) over b ({b})"
        );
        // ...but only the ranked form puts it there.
        let plain_pos = plain.iter().position(|h| h.doc_id == "c").unwrap();
        let ranked_pos = ranked.iter().position(|h| h.doc_id == "c").unwrap();
        assert_eq!(plain_pos, 2, "plain smooth leaves c in its original slot");
        assert_eq!(ranked_pos, 1, "ranked smooth moves c ahead of b");

        // The result is a total order (ties broken on doc_id), so replay is deterministic.
        // Compared with `total_cmp` — the very comparator the sort uses — so an exact score tie
        // is a real tie here, not a float-epsilon question.
        assert!(
            ranked
                .windows(2)
                .all(|w| match w[0].score.total_cmp(&w[1].score) {
                    std::cmp::Ordering::Greater => true,
                    std::cmp::Ordering::Equal => w[0].doc_id <= w[1].doc_id,
                    std::cmp::Ordering::Less => false,
                }),
            "ranked pool must be sorted desc with doc_id tiebreak"
        );
    }

    /// QUALITY GATE. Smoothing's claimed nDCG gain, measured in Rust via core's `ndcg_at_k` —
    /// not merely cited from the Python BEIR harness. `c` is the relevant answer, under-scored by
    /// the embedder but adjacent in the k-NN graph to the strong hit `a`; `b` is an irrelevant
    /// loner that outscores it. Diffusion recovers `c`.
    #[test]
    fn ranked_smoothing_improves_ndcg_over_no_smoothing() {
        use frankensearch_core::metrics_eval::ndcg_at_k;

        let hits = vec![hit("a", 0.90), hit("b", 0.60), hit("c", 0.55)];
        let g = sim_graph(&[("c", "a"), ("a", "c")]);
        let cfg = SmoothConfig {
            alpha: 0.5,
            m: 10,
            mutual: false,
        };
        let relevant = ["c"];

        let ids =
            |v: &[VectorHit]| -> Vec<String> { v.iter().map(|h| h.doc_id.to_string()).collect() };
        fn as_refs(v: &[String]) -> Vec<&str> {
            v.iter().map(String::as_str).collect()
        }

        let before = ids(&hits);
        let after = ids(&neighbor_smooth_ranked(&hits, &g, &cfg));

        let ndcg_before = ndcg_at_k(&as_refs(&before), &relevant, 10);
        let ndcg_after = ndcg_at_k(&as_refs(&after), &relevant, 10);
        assert!(
            ndcg_after > ndcg_before,
            "smoothed nDCG@10 {ndcg_after} should beat unsmoothed {ndcg_before} \
             (before {before:?}, after {after:?})"
        );
    }

    #[test]
    fn empty_graph_is_identity() {
        let hits = vec![hit("a", 0.9), hit("b", 0.5)];
        let out = neighbor_smooth(&hits, &DocumentGraph::new(), &SmoothConfig::default());
        assert_eq!(out, hits);
    }

    #[test]
    fn hand_computed_mean() {
        // d has neighbors a(0.9), b(0.7) → mean 0.8. α=0.5, s(d)=0.2
        // smoothed = 0.5*0.2 + 0.5*0.8 = 0.5
        let hits = vec![hit("d", 0.2), hit("a", 0.9), hit("b", 0.7)];
        let g = sim_graph(&[("d", "a"), ("d", "b")]);
        let cfg = SmoothConfig {
            alpha: 0.5,
            m: 10,
            mutual: false,
        };
        let out = neighbor_smooth(&hits, &g, &cfg);
        let d = out.iter().find(|h| h.doc_id == "d").unwrap();
        assert!((d.score - 0.5).abs() < 1e-6, "got {}", d.score);
    }

    #[test]
    fn cluster_rescues_below_threshold_relevant() {
        // "d" is a relevant marooned at low dense score (0.30) but sits among a cluster of
        // confident results (a,b,c ≈ 0.9). "b_iso" is a NON-relevant at a higher raw score
        // (0.40) with no confident neighbors. Plain ranking puts b_iso above d; smoothing
        // must lift d above b_iso — the whole point of the primitive.
        let hits = vec![
            hit("a", 0.92),
            hit("b", 0.90),
            hit("c", 0.88),
            hit("d", 0.30),
            hit("b_iso", 0.40),
        ];
        let g = sim_graph(&[
            ("d", "a"),
            ("d", "b"),
            ("d", "c"),
            ("b_iso", "far"), // "far" is out-of-pool → no rescue
        ]);
        let cfg = SmoothConfig {
            alpha: 0.3,
            m: 10,
            mutual: false,
        };
        let out = neighbor_smooth(&hits, &g, &cfg);
        let s = |id: &str| out.iter().find(|h| h.doc_id == id).unwrap().score;
        assert!(
            s("d") > s("b_iso"),
            "smoothing should rescue d ({}) above isolated b_iso ({})",
            s("d"),
            s("b_iso")
        );
        // b_iso is unchanged (no in-pool neighbors).
        assert!((s("b_iso") - 0.40).abs() < 1e-6);
    }

    #[test]
    fn isolated_doc_unchanged() {
        // "x" has only an out-of-pool neighbor → α collapses to 0 for it.
        let hits = vec![hit("x", 0.55), hit("y", 0.80)];
        let g = sim_graph(&[("x", "not_in_pool")]);
        let out = neighbor_smooth(&hits, &g, &SmoothConfig::default());
        let x = out.iter().find(|h| h.doc_id == "x").unwrap();
        assert!((x.score - 0.55).abs() < 1e-6);
    }

    #[test]
    fn mutual_knn_ignores_one_way_edges() {
        // d → a is one-way (a does NOT point back to d). Under mutual, a is not counted, so
        // d has no mutual neighbor and is unchanged. Under non-mutual, d borrows a.
        let hits = vec![hit("d", 0.20), hit("a", 0.90)];
        let g = sim_graph(&[("d", "a")]); // only one direction
        let non_mutual = neighbor_smooth(
            &hits,
            &g,
            &SmoothConfig {
                alpha: 0.5,
                m: 10,
                mutual: false,
            },
        );
        let d_nm = non_mutual.iter().find(|h| h.doc_id == "d").unwrap();
        assert!((d_nm.score - 0.55).abs() < 1e-6, "non-mutual: {}", d_nm.score); // 0.5*0.2+0.5*0.9

        let mutual = neighbor_smooth(
            &hits,
            &g,
            &SmoothConfig {
                alpha: 0.5,
                m: 10,
                mutual: true,
            },
        );
        let d_m = mutual.iter().find(|h| h.doc_id == "d").unwrap();
        assert!((d_m.score - 0.20).abs() < 1e-6, "mutual one-way ignored: {}", d_m.score);
    }

    #[test]
    fn m_cap_limits_neighbors() {
        // d has 3 in-pool neighbors but m=2 → only the first two (a,b) averaged.
        // mean(0.9, 0.6) = 0.75; α=1.0 → smoothed = 0.75 (ignores c=0.0).
        let hits = vec![hit("d", 0.2), hit("a", 0.9), hit("b", 0.6), hit("c", 0.0)];
        let g = sim_graph(&[("d", "a"), ("d", "b"), ("d", "c")]);
        let cfg = SmoothConfig {
            alpha: 1.0,
            m: 2,
            mutual: false,
        };
        let out = neighbor_smooth(&hits, &g, &cfg);
        let d = out.iter().find(|h| h.doc_id == "d").unwrap();
        assert!((d.score - 0.75).abs() < 1e-6, "m-cap: {}", d.score);
    }

    #[test]
    fn preserves_index_and_docs() {
        let hits = vec![
            VectorHit { index: 7, score: 0.5, doc_id: "a".into() },
            VectorHit { index: 3, score: 0.9, doc_id: "b".into() },
        ];
        let g = sim_graph(&[("a", "b"), ("b", "a")]);
        let out = neighbor_smooth(&hits, &g, &SmoothConfig::default());
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].index, 7);
        assert_eq!(out[0].doc_id, "a");
        assert_eq!(out[1].index, 3);
    }
}
