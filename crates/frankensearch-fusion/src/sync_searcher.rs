//! Synchronous two-tier search orchestration for low-latency UIs.
//!
//! [`SyncTwoTierSearcher`] mirrors the progressive two-phase contract of
//! [`crate::searcher::TwoTierSearcher`] but operates on precomputed query
//! embeddings and fully in-memory indices.

use std::collections::{HashMap, VecDeque};

// The per-query `&str`-keyed score maps + `seen` dedup set are `.get()`/`.insert()`
// probed only (never iterated for output), so `ahash` is bit-identical to std and
// ~2× faster than SipHash on short doc_ids (`sync_hash_ab` bench: 0.44–0.51 across
// n=30..300), matching the sibling fusion paths (`rrf.rs`, `blend.rs`). `rank_map`
// below stays std `HashMap` — it feeds `blend::compute_rank_changes_with_maps`.
use ahash::{AHashMap, AHashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use frankensearch_core::filter::SearchFilter;
use frankensearch_core::{
    FusedHit, PhaseMetrics, RankChanges, ScoreSource, ScoredResult, SearchError, SearchPhase,
    SearchResult, TwoTierConfig, TwoTierMetrics, VectorHit,
};
use frankensearch_index::{InMemoryTwoTierIndex, SearchParams};

use crate::blend::{blend_two_tier_aligned_vector_index, compute_rank_changes_with_maps};
use crate::normalize::{NqcDenseWeight, nqc_cv_iter};
use crate::rrf::{RrfConfig, RrfTiebreak, candidate_count, fuse_by_strategy};

/// Optional synchronous lexical backend used by [`SyncTwoTierSearcher`].
pub trait SyncLexicalSearch: Send + Sync {
    /// Retrieve lexical candidates for the current query.
    ///
    /// Implementations may ignore `query_vec` when they already have external
    /// query context.
    ///
    /// # Errors
    ///
    /// Returns backend-specific lexical retrieval errors.
    fn search_sync(&self, query_vec: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>>;
}

/// Former enabled-path NQC shape retained for the same-binary allocation A/B.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
#[allow(clippy::needless_collect)]
pub fn bench_nqc_cv_collect(lexical: &[ScoredResult]) -> f32 {
    let scores: Vec<f32> = lexical.iter().map(|hit| hit.score).collect();
    crate::normalize::nqc_cv(&scores)
}

/// Shipping enabled-path NQC shape retained for the same-binary allocation A/B.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn bench_nqc_cv_iter(lexical: &[ScoredResult]) -> f32 {
    nqc_cv_iter(lexical.iter().map(|hit| hit.score))
}

/// Enabled-but-empty NQC path before the neutral-sketch early return.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn bench_nqc_empty_weight_orig(
    lexical: &[ScoredResult],
    weight: &NqcDenseWeight,
    beta: f32,
    w_min: f32,
    semantic_weight: f64,
) -> f64 {
    let cv = nqc_cv_iter(lexical.iter().map(|hit| hit.score));
    let factor = weight.dense_weight(cv, beta, w_min);
    semantic_weight * f64::from(factor)
}

/// Candidate neutral-sketch early return for the enabled NQC path.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn bench_nqc_empty_weight_early(
    lexical: &[ScoredResult],
    weight: &NqcDenseWeight,
    beta: f32,
    w_min: f32,
    semantic_weight: f64,
) -> f64 {
    if beta <= 0.0 {
        return semantic_weight;
    }
    let cv = if weight.is_empty() {
        0.0
    } else {
        nqc_cv_iter(lexical.iter().map(|hit| hit.score))
    };
    let factor = weight.dense_weight(cv, beta, w_min);
    semantic_weight * f64::from(factor)
}

/// Progressive synchronous searcher backed by [`InMemoryTwoTierIndex`].
pub struct SyncTwoTierSearcher {
    index: Arc<InMemoryTwoTierIndex>,
    lexical: Option<Arc<dyn SyncLexicalSearch>>,
    search_params: Option<SearchParams>,
    config: TwoTierConfig,
    rrf_lexical_weight: f64,
    rrf_semantic_weight: f64,
    rrf_tiebreak: RrfTiebreak,
    /// Opt-in NQC dense down-weight (default off, `beta = 0.0`). See
    /// [`Self::with_nqc_dense_downweight`].
    nqc_downweight_beta: f32,
    nqc_downweight_w_min: f32,
    nqc_dense_weight: NqcDenseWeight,
}

impl SyncTwoTierSearcher {
    /// Create a sync searcher over an in-memory two-tier index.
    #[must_use]
    pub const fn new(index: Arc<InMemoryTwoTierIndex>, config: TwoTierConfig) -> Self {
        Self {
            index,
            lexical: None,
            search_params: None,
            config,
            rrf_lexical_weight: 1.0,
            rrf_semantic_weight: 1.0,
            rrf_tiebreak: RrfTiebreak::LexicalThenId,
            nqc_downweight_beta: 0.0,
            nqc_downweight_w_min: 0.0,
            nqc_dense_weight: NqcDenseWeight::new(),
        }
    }

    /// Attach an optional synchronous lexical source for RRF hybrid fusion.
    #[must_use]
    pub fn with_lexical(mut self, lexical: Arc<dyn SyncLexicalSearch>) -> Self {
        self.lexical = Some(lexical);
        self
    }

    /// Override brute-force parallel search parameters for fast-tier retrieval.
    #[must_use]
    pub const fn with_search_params(mut self, params: SearchParams) -> Self {
        self.search_params = Some(params);
        self
    }

    /// Set per-tier RRF fusion weights (default `1.0` / `1.0` = neutral).
    ///
    /// Up-weighting the *stronger* tier for the workload (~1.3×) makes the hybrid strictly
    /// dominate the best single tier (see `docs/NEGATIVE_EVIDENCE.md`). Non-finite or `≤ 0`
    /// values fall back to `1.0`.
    #[must_use]
    pub const fn with_rrf_weights(mut self, lexical_weight: f64, semantic_weight: f64) -> Self {
        self.rrf_lexical_weight = lexical_weight;
        self.rrf_semantic_weight = semantic_weight;
        self
    }

    /// Set the RRF tie-break strategy (default [`RrfTiebreak::LexicalThenId`]).
    ///
    /// [`RrfTiebreak::Hash`] breaks score ties by an unbiased hash of `doc_id` rather than
    /// favoring the lexical tier (see `docs/NEGATIVE_EVIDENCE.md`).
    #[must_use]
    pub const fn with_rrf_tiebreak(mut self, tiebreak: RrfTiebreak) -> Self {
        self.rrf_tiebreak = tiebreak;
        self
    }

    /// Enable the opt-in **NQC dense down-weight** (default OFF).
    ///
    /// Per query, the dense tier's fusion weight is scaled by
    /// `clip(1 − beta·CDF(nqc_cv(lexical scores)), w_min, 1)`, where `CDF` is the empirical
    /// percentile from `weight` — a [`NqcDenseWeight`] built offline from a sample of
    /// observed NQC values (the query stream). High lexical commitment (high NQC), where the
    /// dense tier tends to add little or hurt, gets a lower dense weight. Measured aggregate
    /// gain +0.0022 nDCG@10 (pooled 95% CI `[+0.0008, +0.0035]`); latency-neutral (the NQC is
    /// a single-pass reduction, only computed when enabled). See `docs/SEARCH_QUALITY_FINDINGS.md`.
    ///
    /// `beta <= 0` (the default) or an empty `weight` leaves fusion **byte-identical**.
    /// Use `w_min > 0` (e.g. the measured `beta ≈ 0.5` already floors the multiplier at
    /// `0.5`): a scaled semantic weight that reaches `<= 0` is treated as neutral `1.0` by the
    /// tier-weight sanitizer, which would *undo* the down-weight rather than maximize it.
    #[must_use]
    pub fn with_nqc_dense_downweight(mut self, beta: f32, w_min: f32, weight: NqcDenseWeight) -> Self {
        self.nqc_downweight_beta = beta;
        self.nqc_downweight_w_min = w_min;
        self.nqc_dense_weight = weight;
        self
    }

    /// The dense-tier fusion weight for this query: the static `rrf_semantic_weight`, scaled
    /// by the per-query NQC dense down-weight when enabled (`beta > 0`). Off (default) returns
    /// `rrf_semantic_weight` unchanged with zero extra work.
    fn effective_semantic_weight(&self, lexical: &[ScoredResult]) -> f64 {
        if self.nqc_downweight_beta <= 0.0 {
            return self.rrf_semantic_weight;
        }
        let cv = if self.nqc_dense_weight.is_empty() {
            0.0
        } else {
            nqc_cv_iter(lexical.iter().map(|hit| hit.score))
        };
        let factor =
            self.nqc_dense_weight
                .dense_weight(cv, self.nqc_downweight_beta, self.nqc_downweight_w_min);
        self.rrf_semantic_weight * f64::from(factor)
    }

    /// Execute a synchronous search and return the final result set + metrics.
    ///
    /// # Errors
    ///
    /// Returns dimension/filter errors from vector search and lexical backend
    /// failures (when lexical fusion is enabled).
    pub fn search_collect(
        &self,
        query_vec: &[f32],
        k: usize,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        self.search_collect_with_filter(query_vec, k, None)
    }

    /// Execute a synchronous search with an optional doc-level filter.
    ///
    /// # Errors
    ///
    /// Returns dimension/filter errors from vector search and lexical backend
    /// failures (when lexical fusion is enabled).
    pub fn search_collect_with_filter(
        &self,
        query_vec: &[f32],
        k: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        // `search_collect` discards `outcome.phases`, so skip building them — that
        // avoids cloning the full `Vec<ScoredResult>` (N owned doc_ids each) once
        // per phase (Initial + Refined), pure waste at large `k` (limit_all).
        let outcome = self.search_internal(query_vec, k, filter, false)?;
        Ok((outcome.final_results, outcome.metrics))
    }

    /// Execute a synchronous search and stream progressive phases via iterator.
    ///
    /// When phase-1 retrieval fails (for example dimension mismatch), this
    /// returns an iterator yielding a single `RefinementFailed` phase carrying
    /// an empty `initial_results` payload.
    #[must_use]
    pub fn search_iter(&self, query_vec: &[f32], k: usize) -> SyncSearchIterator {
        self.search_iter_with_filter(query_vec, k, None)
    }

    /// Execute a synchronous filtered search and stream progressive phases.
    #[must_use]
    pub fn search_iter_with_filter(
        &self,
        query_vec: &[f32],
        k: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SyncSearchIterator {
        // The iterator streams the progressive phases, so build them.
        match self.search_internal(query_vec, k, filter, true) {
            Ok(outcome) => SyncSearchIterator::new(outcome.phases),
            Err(error) => SyncSearchIterator::from_error(error),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn search_internal(
        &self,
        query_vec: &[f32],
        k: usize,
        filter: Option<&dyn SearchFilter>,
        want_phases: bool,
    ) -> SearchResult<SyncSearchOutcome> {
        let mut metrics = TwoTierMetrics::default();
        // Only the streaming iterator path consumes `phases`; `search_collect`
        // discards them. When they are not wanted, skip the allocation and the
        // per-phase `Vec<ScoredResult>` clones entirely (see the guarded pushes).
        let mut phases = if want_phases {
            Vec::with_capacity(2)
        } else {
            Vec::new()
        };
        let fetch = candidate_count(k, 0, self.config.candidate_multiplier.max(1)).max(k);

        let phase1_started = Instant::now();
        let fast_hits = self.search_fast_hits(query_vec, fetch, filter)?;
        metrics.phase1_vectors_searched = fast_hits.len();
        metrics.semantic_candidates = fast_hits.len();

        let lexical_started = Instant::now();
        let lexical_hits = self
            .lexical
            .as_ref()
            .map(|lexical| lexical.search_sync(query_vec, fetch))
            .transpose()?;
        let lexical_hits = lexical_hits.map(|hits| filter_lexical_hits(hits, filter));
        metrics.lexical_search_ms = ms(lexical_started.elapsed());
        metrics.lexical_candidates = lexical_hits.as_ref().map_or(0, Vec::len);

        let rrf_started = Instant::now();
        let initial_results = lexical_hits.as_ref().map_or_else(
            || vector_hits_to_scored_results(&fast_hits, k, ScoreSource::SemanticFast, None, None),
            |lexical| {
                fused_hits_to_scored_results(
                    fuse_by_strategy(
                        self.config.fusion_strategy,
                        lexical,
                        &fast_hits,
                        &[],
                        0.0,
                        k,
                        0,
                        &RrfConfig {
                            k: self.config.rrf_k,
                            lexical_weight: self.rrf_lexical_weight,
                            semantic_weight: self.effective_semantic_weight(lexical),
                            tiebreak: self.rrf_tiebreak,
                        },
                    ),
                    k,
                )
            },
        );
        metrics.rrf_fusion_ms = ms(rrf_started.elapsed());

        let phase1_latency = phase1_started.elapsed();
        metrics.vector_search_ms = ms(phase1_latency);
        metrics.phase1_total_ms = ms(phase1_latency);
        metrics.fast_embed_ms = 0.0;

        if want_phases {
            phases.push(SearchPhase::Initial {
                results: initial_results.clone(),
                latency: phase1_latency,
                metrics: PhaseMetrics {
                    embedder_id: "sync-fast-query".to_owned(),
                    vectors_searched: fast_hits.len(),
                    lexical_candidates: metrics.lexical_candidates,
                    fused_count: initial_results.len(),
                },
            });
        }

        if self.config.fast_only || !self.index.has_quality_index() {
            metrics.skip_reason = Some(if self.config.fast_only {
                "fast_only_enabled".to_owned()
            } else {
                "quality_index_unavailable".to_owned()
            });
            return Ok(SyncSearchOutcome {
                phases,
                final_results: initial_results,
                metrics,
            });
        }

        let phase2_started = Instant::now();
        let quality_scores = match self.index.quality_scores_for_hits(query_vec, &fast_hits) {
            Ok(scores) => scores,
            Err(error) => {
                let latency = phase2_started.elapsed();
                metrics.phase2_total_ms = ms(latency);
                metrics.skip_reason = Some(error.to_string());
                if want_phases {
                    phases.push(SearchPhase::RefinementFailed {
                        initial_results: initial_results.clone(),
                        error,
                        latency,
                    });
                }
                return Ok(SyncSearchOutcome {
                    phases,
                    final_results: initial_results,
                    metrics,
                });
            }
        };

        let blend_started = Instant::now();
        // The quality tier is a re-scored subset of `fast_hits` (same doc_ids).
        // Blend straight from the aligned `quality_scores` so we never clone one
        // `String` doc_id per quality hit into an intermediate `Vec<VectorHit>`
        // whose doc_ids are only ever read as `&str` (bit-identical output).
        let quality_count = quality_scores.iter().filter(|s| s.is_some()).count();
        metrics.phase2_vectors_searched = quality_count;
        let blended = blend_two_tier_aligned_vector_index(
            &fast_hits,
            &quality_scores,
            saturating_f64_to_f32(self.config.quality_weight),
        );
        metrics.blend_ms = ms(blend_started.elapsed());
        metrics.quality_search_ms = ms(phase2_started.elapsed());
        metrics.quality_embed_ms = 0.0;

        let refined_results = if let Some(lexical) = lexical_hits.as_ref() {
            fused_hits_to_scored_results(
                fuse_by_strategy(
                    self.config.fusion_strategy,
                    lexical,
                    &blended,
                    &[],
                    0.0,
                    k,
                    0,
                    &RrfConfig {
                        k: self.config.rrf_k,
                        lexical_weight: self.rrf_lexical_weight,
                        semantic_weight: self.effective_semantic_weight(lexical),
                        tiebreak: self.rrf_tiebreak,
                    },
                ),
                k,
            )
        } else {
            unique_vector_hits_to_scored_results_aligned_owned(
                blended,
                k,
                ScoreSource::SemanticQuality,
                &fast_hits,
                &quality_scores,
            )
        };

        let rank_changes = compute_rank_changes_for_scored(&initial_results, &refined_results);
        metrics.rank_changes = rank_changes.clone();
        metrics.phase2_total_ms = ms(phase2_started.elapsed());
        metrics.kendall_tau = None;

        if want_phases {
            phases.push(SearchPhase::Refined {
                results: refined_results.clone(),
                latency: phase2_started.elapsed(),
                metrics: PhaseMetrics {
                    embedder_id: "sync-quality-query".to_owned(),
                    vectors_searched: quality_count,
                    lexical_candidates: metrics.lexical_candidates,
                    fused_count: refined_results.len(),
                },
                rank_changes,
            });
        }

        Ok(SyncSearchOutcome {
            phases,
            final_results: refined_results,
            metrics,
        })
    }

    fn search_fast_hits(
        &self,
        query_vec: &[f32],
        fetch: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        let fast_index = self.index.fast_index();
        self.search_params.map_or_else(
            || {
                // Default: the fast tier is a *reranked candidate generator* (its hits
                // are re-scored by the quality tier + RRF), so use the int8 two-pass
                // (parallel + cutoff) instead of the exact f16 scan.
                //
                // int8 (was 4-bit): on REAL embeddings the int8 candidate set is EXACTLY
                // lossless (candidate-recall@10 = 1.0000 in this fetch=K·3, mult=3 regime,
                // potion-256 + MiniLM-384), whereas 4-bit was only 0.9930–0.9973 (its
                // "recall=1.0" was a synthetic-corpus artifact). And int8 is also FASTER at
                // scale: 0.985 ms vs 4-bit 1.070 ms @ N≈130k (1.09×, separated CIs) — the
                // AVX2 `dot_i8_i8` kernel beats the 4-bit nibble-unpack, so 4-bit's
                // ½-bandwidth edge does not pay when pass-1 is compute-bound (see
                // docs/NEGATIVE_EVIDENCE.md 2026-07-02). Net: strictly-lossless candidate
                // set → identical fused top-k, faster, for ~2× the fast-tier slab bytes
                // (int8 `dim` vs 4-bit `dim/2`; ~12.8 MB extra @100k dim256 — negligible).
                // mult=3 keeps ample margin (int8 is lossless from mult=2); `fetch` is
                // already a candidate over-fetch, so a larger multiplier is wasteful.
                const FAST_TIER_MULT: usize = 3;
                fast_index.search_top_k_int8_two_pass_filtered(
                    query_vec,
                    fetch,
                    FAST_TIER_MULT,
                    filter,
                )
            },
            // Explicit params: honour the exact scan + parallelism configuration.
            |params| fast_index.search_top_k_with_params(query_vec, fetch, filter, params),
        )
    }
}

impl std::fmt::Debug for SyncTwoTierSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncTwoTierSearcher")
            .field("has_lexical", &self.lexical.is_some())
            .field("search_params", &self.search_params)
            .field("has_quality_index", &self.index.has_quality_index())
            .field("config", &self.config)
            .field("rrf_lexical_weight", &self.rrf_lexical_weight)
            .field("rrf_semantic_weight", &self.rrf_semantic_weight)
            .field("rrf_tiebreak", &self.rrf_tiebreak)
            .field("nqc_downweight_beta", &self.nqc_downweight_beta)
            .finish()
    }
}

#[derive(Debug)]
struct SyncSearchOutcome {
    phases: Vec<SearchPhase>,
    final_results: Vec<ScoredResult>,
    metrics: TwoTierMetrics,
}

/// Iterator over progressive phases produced by [`SyncTwoTierSearcher`].
#[derive(Debug)]
pub struct SyncSearchIterator {
    phases: VecDeque<SearchPhase>,
}

impl SyncSearchIterator {
    fn new(phases: Vec<SearchPhase>) -> Self {
        Self {
            phases: phases.into(),
        }
    }

    fn from_error(error: SearchError) -> Self {
        Self::new(vec![SearchPhase::RefinementFailed {
            initial_results: Vec::new(),
            error,
            latency: Duration::from_millis(0),
        }])
    }
}

impl Iterator for SyncSearchIterator {
    type Item = SearchPhase;

    fn next(&mut self) -> Option<Self::Item> {
        self.phases.pop_front()
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[allow(clippy::cast_possible_truncation)]
fn saturating_f64_to_f32(value: f64) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32
}

fn filter_lexical_hits(
    hits: Vec<ScoredResult>,
    filter: Option<&dyn SearchFilter>,
) -> Vec<ScoredResult> {
    let Some(filter) = filter else {
        return hits;
    };
    hits.into_iter()
        .filter(|hit| filter.matches(&hit.doc_id, hit.metadata.as_deref()))
        .collect()
}

fn fused_hits_to_scored_results(hits: Vec<FusedHit>, k: usize) -> Vec<ScoredResult> {
    // Take the `rrf_fuse` result by value and move each `doc_id` into the
    // `ScoredResult` instead of cloning it; the `FusedHit`s are a fresh
    // temporary here, so there is no need to keep them alive.
    hits.into_iter()
        .take(k)
        .map(|hit| ScoredResult {
            doc_id: hit.doc_id,
            score: saturating_f64_to_f32(hit.rrf_score),
            source: ScoreSource::Hybrid,
            index: hit.semantic_index,
            fast_score: hit.semantic_score,
            quality_score: None,
            lexical_score: hit.lexical_score,
            rerank_score: None,
            explanation: None,
            metadata: None,
        })
        .collect()
}

fn vector_hits_to_scored_results(
    hits: &[VectorHit],
    k: usize,
    source: ScoreSource,
    fast_scores: Option<&AHashMap<&str, f32>>,
    quality_scores: Option<&AHashMap<&str, f32>>,
) -> Vec<ScoredResult> {
    let mut seen = AHashSet::with_capacity(hits.len());
    hits.iter()
        .filter(|hit| seen.insert(hit.doc_id.as_str()))
        .take(k)
        .map(|hit| {
            let fast_score = fast_scores
                .and_then(|scores| scores.get(hit.doc_id.as_str()))
                .copied()
                .or(Some(hit.score));
            let quality_score = quality_scores
                .and_then(|scores| scores.get(hit.doc_id.as_str()))
                .copied();
            ScoredResult {
                doc_id: hit.doc_id.clone(),
                score: hit.score,
                source,
                index: Some(hit.index),
                fast_score,
                quality_score,
                lexical_score: None,
                rerank_score: None,
                explanation: None,
                metadata: None,
            }
        })
        .collect()
}

type AlignedScores = (f32, Option<f32>);

enum AlignedScoreLookup {
    Dense {
        base: u32,
        scores: Vec<Option<AlignedScores>>,
    },
    Hash(AHashMap<u32, AlignedScores>),
    Empty,
}

impl AlignedScoreLookup {
    fn new(fast_hits: &[VectorHit], quality_scores: &[Option<f32>]) -> Self {
        let Some(first) = fast_hits.first() else {
            return Self::Empty;
        };
        let (mut min_index, mut max_index) = (first.index, first.index);
        for hit in &fast_hits[1..] {
            min_index = min_index.min(hit.index);
            max_index = max_index.max(hit.index);
        }

        let span = max_index.saturating_sub(min_index).saturating_add(1);
        let dense_limit = u32::try_from(fast_hits.len().saturating_mul(4).saturating_add(1024))
            .unwrap_or(u32::MAX);

        if span <= dense_limit {
            let span = usize::try_from(span).expect("dense score lookup span fits usize");
            let mut scores = vec![None; span];
            for (position, hit) in fast_hits.iter().enumerate() {
                let slot = usize::try_from(hit.index.saturating_sub(min_index))
                    .expect("dense score lookup offset fits usize");
                scores[slot] = Some((hit.score, quality_scores.get(position).copied().flatten()));
            }
            Self::Dense {
                base: min_index,
                scores,
            }
        } else {
            let mut scores = AHashMap::with_capacity(fast_hits.len());
            for (position, hit) in fast_hits.iter().enumerate() {
                scores.insert(
                    hit.index,
                    (hit.score, quality_scores.get(position).copied().flatten()),
                );
            }
            Self::Hash(scores)
        }
    }

    fn get(&self, index: u32) -> Option<AlignedScores> {
        match self {
            Self::Dense { base, scores } => index
                .checked_sub(*base)
                .and_then(|offset| usize::try_from(offset).ok())
                .and_then(|offset| scores.get(offset))
                .copied()
                .flatten(),
            Self::Hash(scores) => scores.get(&index).copied(),
            Self::Empty => None,
        }
    }
}

fn unique_vector_hits_to_scored_results_aligned_owned(
    hits: Vec<VectorHit>,
    k: usize,
    source: ScoreSource,
    fast_hits: &[VectorHit],
    quality_scores: &[Option<f32>],
) -> Vec<ScoredResult> {
    // `blend_two_tier_aligned_vector_index` is only used with vector-index hits,
    // whose `(index, doc_id)` pairs are unique. The blended output preserves the
    // original vector index, so we can recover fast/quality scores through a
    // numeric aligned lookup instead of building two `doc_id`-hashed maps and
    // probing both for every output row.
    let score_lookup = AlignedScoreLookup::new(fast_hits, quality_scores);
    hits.into_iter()
        .take(k)
        .map(|hit| {
            let (fast_score, quality_score) =
                score_lookup.get(hit.index).unwrap_or((hit.score, None));
            ScoredResult {
                doc_id: hit.doc_id,
                score: hit.score,
                source,
                index: Some(hit.index),
                fast_score: Some(fast_score),
                quality_score,
                lexical_score: None,
                rerank_score: None,
                explanation: None,
                metadata: None,
            }
        })
        .collect()
}

fn compute_rank_changes_for_scored(
    initial: &[ScoredResult],
    refined: &[ScoredResult],
) -> RankChanges {
    // Build the doc_id → rank maps directly from the `ScoredResult` slices.
    // `build_borrowed_rank_map` only ever reads `doc_id` (rank = enumerate index;
    // it ignores `VectorHit::index`/`score`), so the previous code allocated two
    // throwaway `Vec<VectorHit>` and cloned every `doc_id` into them per query for
    // nothing. Borrowing `doc_id.as_str()` straight from the input drops those two
    // Vec allocations + 2·N `String` clones on the sync hybrid path. First-occurrence
    // wins (`entry().or_insert`), identical to `build_borrowed_rank_map`, so the
    // resulting maps — and the `RankChanges` — are unchanged.
    fn rank_map(hits: &[ScoredResult]) -> HashMap<&str, usize> {
        let mut ranks = HashMap::with_capacity(hits.len());
        for (rank, hit) in hits.iter().enumerate() {
            ranks.entry(hit.doc_id.as_str()).or_insert(rank);
        }
        ranks
    }
    let initial_map = rank_map(initial);
    let refined_map = rank_map(refined);
    compute_rank_changes_with_maps(&initial_map, &refined_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch_core::ScoreSource;
    use frankensearch_index::{InMemoryTwoTierIndex, InMemoryVectorIndex};

    fn make_index() -> Arc<InMemoryTwoTierIndex> {
        let doc_ids = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let fast_vectors = vec![vec![1.0, 0.0], vec![0.7, 0.3], vec![0.0, 1.0]];
        let quality_vectors = vec![vec![0.2, 0.8], vec![1.0, 0.0], vec![0.0, 1.0]];
        let fast = InMemoryVectorIndex::from_vectors(doc_ids.clone(), fast_vectors, 2).unwrap();
        let quality = InMemoryVectorIndex::from_vectors(doc_ids, quality_vectors, 2).unwrap();
        Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)))
    }

    fn lexical_result(doc_id: &str, score: f32) -> ScoredResult {
        ScoredResult {
            doc_id: doc_id.into(),
            score,
            source: ScoreSource::Lexical,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(score),
            rerank_score: None,
            explanation: None,
            metadata: None,
        }
    }

    struct StaticLexical {
        hits: Vec<ScoredResult>,
    }

    impl SyncLexicalSearch for StaticLexical {
        fn search_sync(&self, _query_vec: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
            Ok(self.hits.iter().take(limit).cloned().collect())
        }
    }

    struct ExcludeB;

    impl SearchFilter for ExcludeB {
        fn matches(&self, doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
            doc_id != "b"
        }

        fn name(&self) -> &'static str {
            "exclude-b"
        }
    }

    fn vector_hit(doc_id: &str, index: u32, score: f32) -> VectorHit {
        VectorHit {
            doc_id: doc_id.into(),
            index,
            score,
        }
    }

    fn assert_aligned_result_scores(indices: [u32; 3]) {
        let fast_hits = vec![
            vector_hit("a", indices[0], 0.9),
            vector_hit("b", indices[1], 0.7),
            vector_hit("c", indices[2], 0.5),
        ];
        let quality_scores = vec![Some(0.2), None, Some(0.95)];
        let blended = vec![
            vector_hit("c", indices[2], 0.91),
            vector_hit("a", indices[0], 0.88),
            vector_hit("b", indices[1], 0.77),
        ];

        let fast_by_doc = fast_hits
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score))
            .collect::<AHashMap<&str, f32>>();
        let quality_by_doc = fast_hits
            .iter()
            .zip(quality_scores.iter())
            .filter_map(|(hit, score)| score.map(|s| (hit.doc_id.as_str(), s)))
            .collect::<AHashMap<&str, f32>>();
        let expected_scores = blended
            .iter()
            .map(|hit| {
                (
                    hit.doc_id.clone(),
                    hit.score,
                    fast_by_doc
                        .get(hit.doc_id.as_str())
                        .copied()
                        .or(Some(hit.score)),
                    quality_by_doc.get(hit.doc_id.as_str()).copied(),
                )
            })
            .collect::<Vec<_>>();

        let actual = unique_vector_hits_to_scored_results_aligned_owned(
            blended,
            3,
            ScoreSource::SemanticQuality,
            &fast_hits,
            &quality_scores,
        );
        assert_eq!(actual.len(), expected_scores.len());
        for (actual, (doc_id, score, fast_score, quality_score)) in
            actual.iter().zip(expected_scores)
        {
            assert_eq!(actual.doc_id, doc_id);
            assert_eq!(actual.score.to_bits(), score.to_bits());
            assert_eq!(actual.fast_score, fast_score);
            assert_eq!(actual.quality_score, quality_score);
            assert_eq!(actual.source, ScoreSource::SemanticQuality);
        }
    }

    #[test]
    fn aligned_numeric_score_lookup_matches_doc_id_maps() {
        assert_aligned_result_scores([10, 11, 12]);
        assert_aligned_result_scores([7, 50_000, 90_000]);
    }

    #[test]
    fn search_collect_returns_refined_results() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let (results, metrics) = searcher.search_collect(&[1.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].source, ScoreSource::SemanticQuality);
        assert!(metrics.phase1_total_ms >= 0.0);
        assert!(metrics.phase2_total_ms >= 0.0);
    }

    #[test]
    fn rrf_weights_flow_through_searcher_to_fusion() {
        // Lexical favors "c" (then "b"); the quality/semantic tier favors a different doc
        // for query [1,0]. Extreme opposite tier weights must therefore flip the top result,
        // proving `with_rrf_weights` / `with_rrf_tiebreak` reach the fusion `RrfConfig`.
        let make_lex = || {
            Arc::new(StaticLexical {
                hits: vec![lexical_result("c", 10.0), lexical_result("b", 5.0)],
            })
        };
        let q = [1.0_f32, 0.0];

        let sem_heavy = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_lexical(make_lex())
            .with_rrf_weights(0.01, 100.0)
            .with_rrf_tiebreak(crate::rrf::RrfTiebreak::Hash);
        let lex_heavy = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_lexical(make_lex())
            .with_rrf_weights(100.0, 0.01);

        let (sem_res, _) = sem_heavy.search_collect(&q, 3).unwrap();
        let (lex_res, _) = lex_heavy.search_collect(&q, 3).unwrap();
        assert!(!sem_res.is_empty() && !lex_res.is_empty());
        assert_ne!(
            sem_res[0].doc_id, lex_res[0].doc_id,
            "opposite tier weights must change the fused top result (weights reach fusion)"
        );
    }

    #[test]
    fn nqc_dense_downweight_empty_sketch_is_bit_identical_to_base_weight() {
        let hits = [lexical_result("a", 10.0), lexical_result("b", 1.0)];
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_rrf_weights(1.0, 1.3)
            .with_nqc_dense_downweight(0.5, 0.1, NqcDenseWeight::new());

        assert_eq!(
            searcher.effective_semantic_weight(&hits).to_bits(),
            searcher.rrf_semantic_weight.to_bits(),
        );
    }

    #[test]
    fn nqc_dense_downweight_flows_through_searcher_to_fusion() {
        // Both searchers up-weight the dense tier (semantic 5×) so it dominates by default;
        // enabling the NQC dense down-weight with a sample below the query's NQC drives the
        // dense weight to 0, so lexical (favoring "c") dominates instead. Different top =>
        // the opt-in down-weight reaches the fusion RrfConfig.
        let make_lex = || {
            Arc::new(StaticLexical {
                hits: vec![lexical_result("c", 10.0), lexical_result("b", 5.0)],
            })
        };
        let q = [1.0_f32, 0.0];

        let neutral = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_lexical(make_lex())
            .with_rrf_weights(1.0, 5.0);
        // Query NQC (cv of lexical scores [10, 5] ≈ 0.333) is above every sampled value, so
        // its percentile is 1.0 and dense_weight(beta=1, w_min=0.05) = clip(1 - 1·1, 0.05, 1)
        // = 0.05 → effective semantic weight 5·0.05 = 0.25 (< lexical 1.0), still > 0 so it is
        // not neutralized by the tier-weight sanitizer.
        let sample = NqcDenseWeight::from_sample(&[0.1, 0.2, 0.3]);
        let downweighted = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_lexical(make_lex())
            .with_rrf_weights(1.0, 5.0)
            .with_nqc_dense_downweight(1.0, 0.05, sample);

        let (neutral_res, _) = neutral.search_collect(&q, 3).unwrap();
        let (down_res, _) = downweighted.search_collect(&q, 3).unwrap();
        assert!(!neutral_res.is_empty() && !down_res.is_empty());
        assert_eq!(down_res[0].doc_id, "c", "zeroing the dense tier lets lexical dominate");
        assert_ne!(
            neutral_res[0].doc_id, down_res[0].doc_id,
            "the NQC dense down-weight must change the fused top (it reaches fusion)"
        );
    }

    #[test]
    fn search_iter_yields_initial_then_refined() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let phases = searcher.search_iter(&[1.0, 0.0], 2).collect::<Vec<_>>();
        assert_eq!(phases.len(), 2);
        assert!(matches!(phases[0], SearchPhase::Initial { .. }));
        assert!(matches!(phases[1], SearchPhase::Refined { .. }));
    }

    #[test]
    fn fast_only_mode_skips_phase_two() {
        let config = TwoTierConfig {
            fast_only: true,
            ..TwoTierConfig::default()
        };
        let searcher = SyncTwoTierSearcher::new(make_index(), config);
        let phases = searcher.search_iter(&[1.0, 0.0], 2).collect::<Vec<_>>();
        assert_eq!(phases.len(), 1);
        assert!(matches!(phases[0], SearchPhase::Initial { .. }));
    }

    #[test]
    fn filter_is_applied_to_fast_and_refined_results() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let (results, _) = searcher
            .search_collect_with_filter(&[1.0, 0.0], 3, Some(&ExcludeB))
            .unwrap();
        assert!(results.iter().all(|result| result.doc_id != "b"));
    }

    #[test]
    fn empty_query_returns_dimension_mismatch() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let err = searcher.search_collect(&[], 3).unwrap_err();
        assert!(matches!(err, SearchError::DimensionMismatch { .. }));
    }

    #[test]
    fn lexical_fusion_can_introduce_lexical_only_hits() {
        let lexical = Arc::new(StaticLexical {
            hits: vec![lexical_result("lex-only", 10.0), lexical_result("a", 9.0)],
        });
        let searcher =
            SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default()).with_lexical(lexical);
        let (results, _) = searcher.search_collect(&[1.0, 0.0], 3).unwrap();
        assert!(results.iter().any(|result| result.doc_id == "lex-only"));
        assert!(
            results
                .iter()
                .all(|result| result.source == ScoreSource::Hybrid)
        );
    }
}
