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
#[must_use]
pub fn compute_query_hubness(
    doc_vecs: &[&[f32]],
    query_sample: &[&[f32]],
    kq: usize,
) -> Vec<f32> {
    if query_sample.is_empty() || kq == 0 {
        return vec![0.0; doc_vecs.len()];
    }
    let k = kq.min(query_sample.len());
    doc_vecs
        .iter()
        .map(|d| {
            let mut sims: Vec<f32> = query_sample.iter().map(|q| dot(d, q)).collect();
            let n = sims.len();
            // Partition so the k largest sims land in [n-k..]; mean them = mean of the kq
            // nearest queries. `pivot` is the element at n-k, `top` the k-1 above it.
            let (_, pivot, top) = sims.select_nth_unstable_by(n - k, |a, b| a.total_cmp(b));
            let sum: f32 = *pivot + top.iter().sum::<f32>();
            sum / k as f32
        })
        .collect()
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(index: u32, score: f32) -> VectorHit {
        VectorHit { index, score, doc_id: format!("d{index}").into() }
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
        assert!((out[0].score - (0.3 - 0.05)).abs() < 1e-6, "{}", out[0].score);
        assert!((out[1].score - (0.9 - 0.40)).abs() < 1e-6, "{}", out[1].score);
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
        assert!(out[1].score > out[0].score, "rel {} should beat hub {}", out[1].score, out[0].score);
    }
}
