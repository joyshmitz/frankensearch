//! Information retrieval evaluation metrics.
//!
//! Provides standard IR metrics for evaluating search quality:
//! - **nDCG@K**: Normalized Discounted Cumulative Gain
//! - **MAP@K**: Mean Average Precision at K
//! - **MRR**: Mean Reciprocal Rank
//! - **Recall@K**: Fraction of relevant documents retrieved in top-K

use std::collections::HashSet;

#[inline]
#[allow(clippy::cast_precision_loss)]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Normalized Discounted Cumulative Gain at K.
///
/// Measures ranking quality, giving higher weight to relevant documents
/// appearing earlier in the result list. Uses binary relevance (1.0 if in
/// `relevant`, 0.0 otherwise).
///
/// Returns 0.0 when `relevant` is empty or `k` is 0.
#[must_use]
pub fn ndcg_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    let relevant_set: HashSet<&str> = relevant.iter().copied().collect();
    if relevant_set.is_empty() || k == 0 {
        return 0.0;
    }

    let limit = k.min(retrieved.len());
    let mut seen = HashSet::with_capacity(limit);

    // DCG: sum of 1/log2(rank+1) for relevant docs in retrieved
    let dcg: f64 = retrieved[..limit]
        .iter()
        .enumerate()
        .filter_map(|(i, doc)| {
            if !seen.insert(*doc) {
                return None;
            }
            if relevant_set.contains(doc) {
                Some(1.0 / (usize_to_f64(i) + 2.0).log2())
            } else {
                None
            }
        })
        .sum();

    // Ideal DCG: all relevant docs at top positions
    let ideal_count = k.min(relevant_set.len());
    let idcg: f64 = (0..ideal_count)
        .map(|i| 1.0 / (usize_to_f64(i) + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        return 0.0;
    }

    dcg / idcg
}

/// Mean Average Precision at K.
///
/// Computes precision at each rank position where a relevant document appears,
/// then averages over the total number of relevant documents (capped at K).
///
/// Returns 0.0 when `relevant` is empty or `k` is 0.
#[must_use]
pub fn map_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    let relevant_set: HashSet<&str> = relevant.iter().copied().collect();
    if relevant_set.is_empty() || k == 0 {
        return 0.0;
    }

    let limit = k.min(retrieved.len());
    let mut hits = 0_u32;
    let mut sum_precision = 0.0;
    let mut seen = HashSet::with_capacity(limit);

    for (i, doc) in retrieved[..limit].iter().enumerate() {
        if !seen.insert(*doc) {
            continue;
        }
        if relevant_set.contains(doc) {
            hits += 1;
            sum_precision += f64::from(hits) / (usize_to_f64(i) + 1.0);
        }
    }

    let denominator = usize_to_f64(k.min(relevant_set.len()));
    sum_precision / denominator
}

/// Mean Reciprocal Rank.
///
/// Returns 1/(rank of first relevant document). Returns 0.0 if no relevant
/// document appears in the retrieved list.
#[must_use]
pub fn mrr(retrieved: &[&str], relevant: &[&str]) -> f64 {
    let relevant_set: HashSet<&str> = relevant.iter().copied().collect();
    if relevant_set.is_empty() {
        return 0.0;
    }

    let mut seen = HashSet::new();
    for (i, doc) in retrieved.iter().enumerate() {
        if !seen.insert(*doc) {
            continue;
        }
        if relevant_set.contains(doc) {
            return 1.0 / (usize_to_f64(i) + 1.0);
        }
    }
    0.0
}

/// Recall at K.
///
/// Fraction of relevant documents that appear in the top-K retrieved results.
/// Returns 0.0 when `relevant` is empty or `k` is 0.
#[must_use]
pub fn recall_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    let relevant_set: HashSet<&str> = relevant.iter().copied().collect();
    if relevant_set.is_empty() || k == 0 {
        return 0.0;
    }

    let limit = k.min(retrieved.len());
    let mut seen = HashSet::with_capacity(limit);
    let mut found = 0_usize;

    for doc in &retrieved[..limit] {
        if !seen.insert(*doc) {
            continue;
        }
        if relevant_set.contains(doc) {
            found += 1;
        }
    }

    usize_to_f64(found) / usize_to_f64(relevant_set.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── nDCG@K ─────────────────────────────────────────────────────────

    #[test]
    fn ndcg_perfect_ranking() {
        let retrieved = vec!["a", "b", "c"];
        let relevant = vec!["a", "b", "c"];
        let score = ndcg_at_k(&retrieved, &relevant, 3);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "perfect ranking should be 1.0, got {score}"
        );
    }

    #[test]
    fn ndcg_reversed_ranking() {
        // Relevant docs at bottom positions should score less than at top
        let good = ndcg_at_k(&["a", "b", "x"], &["a", "b"], 3);
        let bad = ndcg_at_k(&["x", "a", "b"], &["a", "b"], 3);
        assert!(
            good > bad,
            "top-ranked relevant docs should score higher: {good} vs {bad}"
        );
    }

    #[test]
    fn ndcg_empty_relevant() {
        assert!((ndcg_at_k(&["a", "b"], &[], 3)).abs() < f64::EPSILON);
    }

    #[test]
    fn ndcg_empty_retrieved() {
        assert!((ndcg_at_k(&[], &["a", "b"], 3)).abs() < f64::EPSILON);
    }

    #[test]
    fn ndcg_k_zero() {
        assert!((ndcg_at_k(&["a"], &["a"], 0)).abs() < f64::EPSILON);
    }

    #[test]
    fn ndcg_single_relevant_at_rank_1() {
        let score = ndcg_at_k(&["a"], &["a"], 10);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "single relevant doc at rank 1 should be 1.0"
        );
    }

    #[test]
    fn ndcg_no_overlap() {
        let score = ndcg_at_k(&["x", "y", "z"], &["a", "b"], 3);
        assert!(
            score.abs() < f64::EPSILON,
            "no overlap should be 0.0, got {score}"
        );
    }

    #[test]
    fn ndcg_duplicate_retrievals_count_once() {
        let score = ndcg_at_k(&["a", "a", "b"], &["a", "b"], 3);
        let expected = 1.5 / (1.0 + 1.0 / 3f64.log2());
        assert!(
            (score - expected).abs() < 1e-12,
            "duplicates in the ranking should not push nDCG above the ideal"
        );
    }

    // ─── MAP@K ──────────────────────────────────────────────────────────

    #[test]
    fn map_perfect_ranking() {
        let score = map_at_k(&["a", "b", "c"], &["a", "b", "c"], 3);
        // P@1=1, P@2=1, P@3=1 → AP = 3/3 = 1.0
        assert!(
            (score - 1.0).abs() < 1e-10,
            "perfect ranking should be 1.0, got {score}"
        );
    }

    #[test]
    fn map_one_relevant_at_top() {
        let score = map_at_k(&["a", "x", "y"], &["a"], 3);
        // P@1=1 → AP = 1/1 = 1.0
        assert!((score - 1.0).abs() < 1e-10, "got {score}");
    }

    #[test]
    fn map_one_relevant_at_rank_3() {
        let score = map_at_k(&["x", "y", "a"], &["a"], 3);
        // P@3=1/3 → AP = (1/3)/1 = 0.333...
        assert!(
            (score - 1.0 / 3.0).abs() < 1e-10,
            "expected 0.333, got {score}"
        );
    }

    #[test]
    fn map_empty_relevant() {
        assert!(map_at_k(&["a", "b"], &[], 3).abs() < f64::EPSILON);
    }

    #[test]
    fn map_k_zero() {
        assert!(map_at_k(&["a"], &["a"], 0).abs() < f64::EPSILON);
    }

    #[test]
    fn map_no_overlap() {
        let score = map_at_k(&["x", "y"], &["a", "b"], 3);
        assert!(
            score.abs() < f64::EPSILON,
            "no overlap should be 0.0, got {score}"
        );
    }

    #[test]
    fn map_ignores_duplicate_docs() {
        let score = map_at_k(&["a", "a", "b"], &["a", "b"], 3);
        let expected = f64::midpoint(1.0, 2.0 / 3.0);
        assert!(
            (score - expected).abs() < 1e-12,
            "duplicate doc hit should only contribute once to average precision"
        );
    }

    // ─── MRR ────────────────────────────────────────────────────────────

    #[test]
    fn mrr_first_relevant_at_rank_1() {
        let score = mrr(&["a", "b", "c"], &["a"]);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mrr_first_relevant_at_rank_3() {
        let score = mrr(&["x", "y", "a"], &["a"]);
        assert!((score - 1.0 / 3.0).abs() < 1e-10, "got {score}");
    }

    #[test]
    fn mrr_no_relevant() {
        let score = mrr(&["x", "y", "z"], &["a"]);
        assert!(score.abs() < f64::EPSILON);
    }

    #[test]
    fn mrr_empty_retrieved() {
        assert!(mrr(&[], &["a"]).abs() < f64::EPSILON);
    }

    // ─── Recall@K ───────────────────────────────────────────────────────

    #[test]
    fn recall_perfect() {
        let score = recall_at_k(&["a", "b", "c"], &["a", "b"], 3);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn recall_partial() {
        let score = recall_at_k(&["a", "x", "y"], &["a", "b"], 3);
        assert!((score - 0.5).abs() < 1e-10, "got {score}");
    }

    #[test]
    fn recall_none() {
        let score = recall_at_k(&["x", "y", "z"], &["a", "b"], 3);
        assert!(score.abs() < f64::EPSILON);
    }

    #[test]
    fn recall_empty_relevant() {
        assert!(recall_at_k(&["a"], &[], 3).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_k_zero() {
        assert!(recall_at_k(&["a"], &["a"], 0).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_k_limits_retrieved() {
        // Only look at top-2, so "b" at position 3 doesn't count
        let score = recall_at_k(&["a", "x", "b"], &["a", "b"], 2);
        assert!((score - 0.5).abs() < 1e-10, "got {score}");
    }

    #[test]
    fn recall_duplicate_documents_count_once() {
        let score = recall_at_k(&["a", "a", "b"], &["a", "b"], 3);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "duplicate hits should not inflate recall beyond 1.0"
        );
    }
}
