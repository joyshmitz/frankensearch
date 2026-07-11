//! Query-hubness dense-score correction (demote high-hub docs).
//!
//! High-dimensional cosine retrieval suffers **hubness**: a few "hub" documents sit near
//! *many queries* regardless of relevance and crowd out specific answers (Radovanović et al.;
//! CSLS, Conneau et al.). This module demotes hubs by subtracting a per-doc hubness estimate
//! from its dense score before fusion:
//!
//! ```text
//! s'(q, d) = cos(q, d) − β · r_d
//! ```
//!
//! The load-bearing detail (measured in `docs/NEGATIVE_EVIDENCE.md`): `r_d` must be a
//! **query-distribution** statistic — the doc's mean similarity to its `kq` nearest *sample
//! queries* — estimated from a background **query sample** (e.g. a rolling query log). The
//! cheap *query-free* proxies (doc-doc density, centroid distance, PC removal) were REJECTED:
//! they conflate genuine hubs with docs in tight *relevant* clusters and are corpus-fragile.
//! The query-side estimate, measured leakage-free on a disjoint query split, is all-positive
//! (β=0.2: mean **+0.0033** hybrid nDCG@10 across 4 BEIR corpora) with genuine dense-tier
//! gains where topical centrality anti-correlates with relevance (arguana counter-argument
//! stance +0.0128 dense, scidocs citation +0.0078). The measured value is a *lower bound* — a
//! production query log (thousands of queries) estimates `r_d` better than the 150–500-query
//! sample used in validation.
//!
//! Two pieces:
//! - [`compute_query_hubness`] — offline/amortized: build the per-doc `r_d` table from a
//!   background query sample (recompute periodically as the log grows).
//! - [`apply_hubness_penalty`] — query-time: a trivial O(pool) subtract over the candidate
//!   pool, indexed by [`VectorHit::index`].

use frankensearch_core::VectorHit;
use frankensearch_index::{PARALLEL_THRESHOLD, dot_product_f32_f32};
use rayon::prelude::*;

/// Parameters for [`apply_hubness_penalty`] / [`compute_query_hubness`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HubnessConfig {
    /// Penalty weight in the dense score. `0` = identity (no correction). Default `0.2`: the
    /// never-negative cross-corpus setting; raise toward `0.3` on stance/citation corpora
    /// where topical hubs mislead most. Values `≤ 0` or non-finite are treated as identity.
    pub beta: f32,
    /// Number of nearest sample queries averaged to estimate each doc's hubness `r_d`.
    /// Default `10`.
    pub kq: usize,
}

impl Default for HubnessConfig {
    fn default() -> Self {
        Self { beta: 0.2, kq: 10 }
    }
}

impl HubnessConfig {
    /// Whether this config is a guaranteed no-op, so callers can skip the pass.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        !self.beta.is_finite() || self.beta <= 0.0
    }
}

/// Query-time: demote hub docs in a candidate pool. `hubness[hit.index as usize]` is the doc's
/// precomputed `r_d` (from [`compute_query_hubness`]); a candidate whose index is out of range
/// gets no penalty. Returns a new pool with the same docs/indices and corrected scores, ready
/// to be re-sorted. Pure O(pool) transform; clones unchanged on an identity config.
#[must_use]
pub fn apply_hubness_penalty(
    hits: &[VectorHit],
    hubness: &[f32],
    config: &HubnessConfig,
) -> Vec<VectorHit> {
    if config.is_identity() {
        return hits.to_vec();
    }
    let beta = config.beta;
    hits.iter()
        .map(|h| {
            let r = hubness.get(h.index as usize).copied().unwrap_or(0.0);
            VectorHit {
                index: h.index,
                score: h.score - beta * r,
                doc_id: h.doc_id.clone(),
            }
        })
        .collect()
}

/// Offline/amortized: per-doc query-hubness `r_d` = mean cosine of doc `d` to its `kq` nearest
/// queries in a background sample. Both `doc_vecs` and `query_sample` must be **L2-normalized**
/// (dot = cosine). Returns one `r_d` per doc in `doc_vecs` order (aligned with the vector-store
/// index, so it indexes directly in [`apply_hubness_penalty`]). Empty sample / `kq == 0` → all
/// zeros (identity). O(docs · queries · dim).
///
/// Each doc's `r_d` depends only on that doc and the whole query sample, so the outer loop is
/// embarrassingly parallel. Above [`PARALLEL_THRESHOLD`] dot products of total work it runs on
/// rayon; below it the pool's fork/join overhead would dominate a batch that already finishes in
/// microseconds. The two branches are **bit-identical**, not merely ULP-equal: rayon's *indexed*
/// parallel iterator collects in input order, and no element's arithmetic changes — only the
/// scheduling does. `hubness_par_matches_serial_across_threshold` and the `hubness_dot_ab`
/// `simd_par` vs `simd` gate both assert this.
///
/// The threshold counts **dot products** (`docs · queries`), not docs: one doc's work is
/// `queries · dim` multiply-adds, so a doc-count gate would misjudge a small corpus against a
/// large query log. [`PARALLEL_THRESHOLD`] carries the same "10k dot products" meaning in the
/// vector tier's flat scan.
#[must_use]
pub fn compute_query_hubness(doc_vecs: &[&[f32]], query_sample: &[&[f32]], kq: usize) -> Vec<f32> {
    if query_sample.is_empty() || kq == 0 {
        return vec![0.0; doc_vecs.len()];
    }
    let k = kq.min(query_sample.len());
    let work = doc_vecs.len().saturating_mul(query_sample.len());
    if work >= PARALLEL_THRESHOLD {
        doc_vecs
            .par_iter()
            .map(|d| doc_hubness(d, query_sample, k))
            .collect()
    } else {
        doc_vecs
            .iter()
            .map(|d| doc_hubness(d, query_sample, k))
            .collect()
    }
}

/// One doc's `r_d`: the mean cosine to its `k` nearest queries in the sample.
#[inline]
fn doc_hubness(doc: &[f32], query_sample: &[&[f32]], k: usize) -> f32 {
    let mut sims: Vec<f32> = query_sample.iter().map(|q| dot(doc, q)).collect();
    let n = sims.len();
    // Partition so the k largest sims land in [n-k..]; mean them = mean of the kq
    // nearest queries. `pivot` is the element at n-k, `top` the k-1 above it.
    let (_, pivot, top) = sims.select_nth_unstable_by(n - k, |a, b| a.total_cmp(b));
    let sum: f32 = *pivot + top.iter().sum::<f32>();
    sum / k as f32
}

/// Cosine dot of two L2-normalized vectors.
///
/// Delegates to the vector tier's [`dot_product_f32_f32`] — a hand-written AVX2 kernel carrying
/// four `f32x8` accumulators (32 lanes of ILP) with a portable `wide` fallback off-x86.
/// [`compute_query_hubness`] is O(docs·queries·dim) and wholly dominated by this dot, and the
/// ORIGINAL here was a scalar `iter().zip().map(..).sum()`: one f32 accumulator, a serial add
/// chain LLVM cannot reassociate without fast-math, latency-bound regardless of how wide the
/// multiplies vectorize. Reusing the shipped kernel rather than hand-rolling a reduction is
/// **11.3× on the 384-dim dot / 10.9× on the builder** (`hubness_dot_ab`); an 8-accumulator
/// scalar loop relying on LLVM's SLP vectorizer reached only 2.6× and was rejected.
///
/// Slicing to the common length keeps the ORIGINAL's `zip` truncation semantics, which also makes
/// the kernel's `DimensionMismatch` unreachable. The reassociation is the same accepted
/// search-time ULP trade as [`crate::mmr`]'s `cosine_sim_pre`: `r_d` is an approximate demotion
/// statistic (`β·r_d` moves the dense score by ~1e-2) and its `select_nth` of the kq nearest
/// queries is robust to sub-ULP perturbation — `hubness_dot_ab` gates selection-equality
/// (max `Δr_d` < 1e-4) against the scalar ORIGINAL before timing.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    dot_product_f32_f32(&a[..len], &b[..len]).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(index: u32, score: f32) -> VectorHit {
        VectorHit {
            index,
            score,
            doc_id: format!("d{index}").into(),
        }
    }

    #[test]
    fn beta_zero_is_identity() {
        let hits = vec![hit(0, 0.9), hit(1, 0.5)];
        let hub = vec![0.8, 0.3];
        let out = apply_hubness_penalty(&hits, &hub, &HubnessConfig { beta: 0.0, kq: 10 });
        assert_eq!(out, hits);
    }

    #[test]
    fn penalty_subtracts_beta_times_hubness_by_index() {
        // hit.index selects the hubness entry: index 1 → hub 0.8, β=0.5 → 0.9 − 0.4 = 0.5.
        let hits = vec![hit(0, 0.3), hit(1, 0.9)];
        let hub = vec![0.1, 0.8];
        let out = apply_hubness_penalty(&hits, &hub, &HubnessConfig { beta: 0.5, kq: 10 });
        assert!(
            (out[0].score - (0.3 - 0.05)).abs() < 1e-6,
            "{}",
            out[0].score
        );
        assert!(
            (out[1].score - (0.9 - 0.40)).abs() < 1e-6,
            "{}",
            out[1].score
        );
    }

    #[test]
    fn out_of_range_index_gets_no_penalty() {
        let hits = vec![hit(7, 0.6)]; // hubness has only 2 entries
        let hub = vec![0.9, 0.9];
        let out = apply_hubness_penalty(&hits, &hub, &HubnessConfig { beta: 0.5, kq: 10 });
        assert!((out[0].score - 0.6).abs() < 1e-6);
    }

    #[test]
    fn hub_doc_scores_higher_r_d_than_outlier() {
        // Queries all point along +x. A doc aligned with them (a hub) has r_d≈1; an orthogonal
        // doc has r_d≈0.
        let q0 = [1.0f32, 0.0];
        let q1 = [1.0f32, 0.0];
        let queries: Vec<&[f32]> = vec![&q0, &q1];
        let hub = [1.0f32, 0.0];
        let outlier = [0.0f32, 1.0];
        let docs: Vec<&[f32]> = vec![&hub, &outlier];
        let r = compute_query_hubness(&docs, &queries, 10);
        assert!((r[0] - 1.0).abs() < 1e-6, "hub r_d {}", r[0]);
        assert!(r[1].abs() < 1e-6, "outlier r_d {}", r[1]);
    }

    #[test]
    fn hubness_averages_kq_nearest_queries() {
        // doc near q_a (cos 1.0) and q_b (cos 0.6), far from q_c (cos 0.0). kq=2 → mean(1.0,0.6)=0.8.
        let d = [1.0f32, 0.0];
        let qa = [1.0f32, 0.0];
        let qb = [0.6f32, 0.8]; // cos with d = 0.6
        let qc = [0.0f32, 1.0]; // cos 0.0
        let docs: Vec<&[f32]> = vec![&d];
        let queries: Vec<&[f32]> = vec![&qa, &qb, &qc];
        let r = compute_query_hubness(&docs, &queries, 2);
        assert!((r[0] - 0.8).abs() < 1e-6, "r_d {}", r[0]);
    }

    /// The other tests use 2-dim vectors, which only exercise the dot kernel's scalar tail.
    /// 43 = 32 (one AVX2 group) + 8 (one chunk) + 3 (tail), so this covers every branch of
    /// `dot_product_f32_f32` and pins the reassociated result to a scalar reference.
    #[test]
    fn hubness_matches_scalar_reference_across_kernel_blocks() {
        let dim = 43;
        let mk = |seed: f32| -> Vec<f32> {
            let raw: Vec<f32> = (0..dim).map(|i| ((i as f32) * seed).sin()).collect();
            let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            raw.into_iter().map(|x| x / norm).collect()
        };
        let d0 = mk(0.7);
        let d1 = mk(1.3);
        let q0 = mk(2.1);
        let q1 = mk(0.35);
        let docs: Vec<&[f32]> = vec![&d0, &d1];
        let queries: Vec<&[f32]> = vec![&q0, &q1];

        let scalar_dot =
            |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        let r = compute_query_hubness(&docs, &queries, 2);
        for (i, d) in docs.iter().enumerate() {
            let expect = queries.iter().map(|q| scalar_dot(d, q)).sum::<f32>() / 2.0;
            assert!(
                (r[i] - expect).abs() < 1e-6,
                "doc {i}: {} vs scalar {expect}",
                r[i]
            );
        }
    }

    /// The rayon branch must be **bit-identical** to the serial one, not merely close.
    ///
    /// A doc's `r_d` depends only on that doc and the query sample — never on how many *other*
    /// docs are present. So the same two docs can be evaluated below the parallel threshold (2
    /// docs × 60 queries = 120 dot products, serial) and above it (200 docs × 60 = 12 000,
    /// rayon); their `r_d` must agree to the bit. That pins ordering (rayon's indexed parallel
    /// iterator collects in input order) and arithmetic (unchanged per element) simultaneously.
    #[test]
    fn hubness_par_matches_serial_across_threshold() {
        let dim = 32;
        let mk = |seed: f32| -> Vec<f32> {
            let raw: Vec<f32> = (0..dim).map(|i| ((i as f32) * seed + seed).cos()).collect();
            let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
            raw.into_iter().map(|x| x / norm).collect()
        };
        let queries: Vec<Vec<f32>> = (0..60).map(|i| mk(0.11 * (i as f32 + 1.0))).collect();
        let qrefs: Vec<&[f32]> = queries.iter().map(Vec::as_slice).collect();

        let d0 = mk(3.7);
        let d1 = mk(9.1);

        // Below threshold: 2 × 60 = 120 < PARALLEL_THRESHOLD → serial branch.
        let serial: Vec<&[f32]> = vec![&d0, &d1];
        assert!(
            serial.len() * qrefs.len() < PARALLEL_THRESHOLD,
            "test must stay serial"
        );
        let r_serial = compute_query_hubness(&serial, &qrefs, 10);

        // Above threshold: 200 × 60 = 12 000 ≥ PARALLEL_THRESHOLD → rayon branch. The first two
        // docs are the same two; padding cannot change their r_d.
        let filler = mk(1.9);
        let mut parallel: Vec<&[f32]> = vec![&d0, &d1];
        parallel.extend(std::iter::repeat_n(filler.as_slice(), 198));
        assert!(
            parallel.len() * qrefs.len() >= PARALLEL_THRESHOLD,
            "test must go parallel"
        );
        let r_parallel = compute_query_hubness(&parallel, &qrefs, 10);

        assert_eq!(r_parallel.len(), 200);
        for i in 0..2 {
            assert_eq!(
                r_serial[i].to_bits(),
                r_parallel[i].to_bits(),
                "doc {i}: serial {} vs parallel {}",
                r_serial[i],
                r_parallel[i]
            );
        }
        // Order is preserved: every padded slot holds the filler's r_d, not a shuffled value.
        let filler_rd = r_parallel[2];
        for (i, r) in r_parallel.iter().enumerate().skip(2) {
            assert_eq!(
                r.to_bits(),
                filler_rd.to_bits(),
                "padded slot {i} out of order"
            );
        }
    }

    /// A doc shorter than the query truncates to the common length (the ORIGINAL `zip` semantics),
    /// rather than hitting the kernel's `DimensionMismatch`.
    #[test]
    fn ragged_lengths_truncate_to_common_prefix() {
        let d = [1.0f32, 0.0, 0.0];
        let q = [1.0f32, 0.0]; // shorter than the doc
        let docs: Vec<&[f32]> = vec![&d];
        let queries: Vec<&[f32]> = vec![&q];
        let r = compute_query_hubness(&docs, &queries, 1);
        assert!((r[0] - 1.0).abs() < 1e-6, "r_d {}", r[0]);
    }

    #[test]
    fn empty_sample_is_zero_hubness() {
        let d = [1.0f32, 0.0];
        let docs: Vec<&[f32]> = vec![&d];
        let r = compute_query_hubness(&docs, &[], 10);
        assert_eq!(r, vec![0.0]);
    }

    #[test]
    fn demotes_a_hub_below_a_specific_relevant() {
        // "hub" has a higher raw cosine (0.80) but is a universal hub (r_d 0.75); "rel" is a
        // specific answer (raw 0.72, r_d 0.20). β=0.3 flips them: hub→0.575, rel→0.66.
        let hits = vec![hit(0, 0.80), hit(1, 0.72)];
        let hub = vec![0.75, 0.20];
        let out = apply_hubness_penalty(&hits, &hub, &HubnessConfig { beta: 0.3, kq: 10 });
        assert!(
            out[1].score > out[0].score,
            "rel {} should beat hub {}",
            out[1].score,
            out[0].score
        );
    }
}
