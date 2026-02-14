//! Progressive two-tier search orchestrator.
//!
//! [`TwoTierSearcher`] coordinates fast-tier and quality-tier search phases,
//! delivering results incrementally via an async callback. Consumers see
//! fast results in ~15ms, then optionally receive quality-refined results.
//!
//! # Callback Protocol
//!
//! The `on_phase` callback fires at most twice:
//! 1. [`SearchPhase::Initial`] — fast-tier results (always fired if search starts).
//! 2. [`SearchPhase::Refined`] or [`SearchPhase::RefinementFailed`] — quality tier
//!    result or graceful degradation (only fired when quality embedder is available
//!    and `fast_only` is false).

use std::sync::Arc;
use std::time::Instant;

use asupersync::Cx;
use tracing::instrument;

use frankensearch_core::canonicalize::{Canonicalizer, DefaultCanonicalizer};
use frankensearch_core::config::{TwoTierConfig, TwoTierMetrics};
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::query_class::QueryClass;
use frankensearch_core::traits::{Embedder, LexicalSearch, Reranker};
use frankensearch_core::types::{
    EmbeddingMetrics, PhaseMetrics, ScoreSource, ScoredResult, SearchMetrics, SearchMode,
    SearchPhase, VectorHit,
};
use frankensearch_index::TwoTierIndex;

use crate::blend::{blend_two_tier, compute_rank_changes, kendall_tau};
use crate::rrf::{RrfConfig, candidate_count, rrf_fuse};

/// Progressive two-tier search orchestrator.
///
/// Coordinates fast-tier embedding, optional lexical search, RRF fusion,
/// optional quality-tier refinement, and optional cross-encoder reranking.
///
/// # Usage
///
/// ```rust,ignore
/// let searcher = TwoTierSearcher::new(index, fast_embedder, config)
///     .with_quality_embedder(quality)
///     .with_lexical(tantivy)
///     .with_reranker(reranker);
///
/// let metrics = searcher.search(&cx, "distributed consensus", 10, text_fn, |phase| {
///     match phase {
///         SearchPhase::Initial { results, .. } => display(&results),
///         SearchPhase::Refined { results, .. } => update_display(&results),
///         SearchPhase::RefinementFailed { .. } => { /* keep initial */ }
///     }
/// }).await?;
/// ```
pub struct TwoTierSearcher {
    index: Arc<TwoTierIndex>,
    fast_embedder: Arc<dyn Embedder>,
    quality_embedder: Option<Arc<dyn Embedder>>,
    lexical: Option<Arc<dyn LexicalSearch>>,
    reranker: Option<Arc<dyn Reranker>>,
    canonicalizer: Box<dyn Canonicalizer>,
    config: TwoTierConfig,
}

impl TwoTierSearcher {
    /// Create a new searcher with a fast-tier embedder.
    #[must_use]
    pub fn new(
        index: Arc<TwoTierIndex>,
        fast_embedder: Arc<dyn Embedder>,
        config: TwoTierConfig,
    ) -> Self {
        Self {
            index,
            fast_embedder,
            quality_embedder: None,
            lexical: None,
            reranker: None,
            canonicalizer: Box::new(DefaultCanonicalizer::default()),
            config,
        }
    }

    /// Set the quality-tier embedder for progressive refinement.
    #[must_use]
    pub fn with_quality_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.quality_embedder = Some(embedder);
        self
    }

    /// Set the lexical search backend for hybrid RRF fusion.
    #[must_use]
    pub fn with_lexical(mut self, lexical: Arc<dyn LexicalSearch>) -> Self {
        self.lexical = Some(lexical);
        self
    }

    /// Set the cross-encoder reranker for Phase 2.
    #[must_use]
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Override the query canonicalizer.
    #[must_use]
    pub fn with_canonicalizer(mut self, canonicalizer: Box<dyn Canonicalizer>) -> Self {
        self.canonicalizer = canonicalizer;
        self
    }

    /// Execute progressive search, calling `on_phase` as results become available.
    ///
    /// Fires [`SearchPhase::Initial`] first, then optionally
    /// [`SearchPhase::Refined`] or [`SearchPhase::RefinementFailed`].
    ///
    /// Returns collected metrics from all phases.
    ///
    /// # Parameters
    ///
    /// * `cx` — Capability context for cancellation.
    /// * `query` — The search query string.
    /// * `k` — Maximum number of results per phase.
    /// * `text_fn` — Retrieves document text by `doc_id` for reranking.
    ///   Pass `|_| None` when reranking is not needed.
    /// * `on_phase` — Callback invoked once per search phase.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Cancelled` if the operation is cancelled via `cx`.
    /// Returns `SearchError::EmbeddingFailed` if fast embedding fails and no
    /// lexical backend is available as fallback.
    #[instrument(skip_all, fields(query_len = query.len(), k))]
    pub async fn search(
        &self,
        cx: &Cx,
        query: &str,
        k: usize,
        text_fn: impl Fn(&str) -> Option<String> + Send + Sync,
        mut on_phase: impl FnMut(SearchPhase) + Send,
    ) -> SearchResult<TwoTierMetrics> {
        let mut metrics = TwoTierMetrics::default();

        if query.is_empty() || k == 0 {
            return Ok(metrics);
        }

        // Canonicalize query.
        let canon_query = self.canonicalizer.canonicalize_query(query);
        let query_class = QueryClass::classify(&canon_query);
        metrics.query_class = Some(query_class);
        metrics.fast_embedder_id = Some(self.fast_embedder.id().to_owned());

        // Phase 1: Initial (fast tier).
        let phase1_start = Instant::now();
        let initial = self
            .run_phase1(cx, &canon_query, k, query_class, &mut metrics)
            .await;
        metrics.phase1_total_ms = phase1_start.elapsed().as_secs_f64() * 1000.0;

        let initial_results = match initial {
            Ok(results) => results,
            Err(err) => {
                self.export_error(&err);
                return Err(err);
            }
        };

        let initial_hits = initial_results.clone();

        on_phase(SearchPhase::Initial {
            results: initial_results,
            latency: phase1_start.elapsed(),
            metrics: PhaseMetrics {
                embedder_id: self.fast_embedder.id().to_owned(),
                vectors_searched: self.index.doc_count(),
                lexical_candidates: metrics.lexical_candidates,
                fused_count: initial_hits.len(),
            },
        });
        self.export_search_metrics(query_class, &metrics, initial_hits.len(), false);

        // Phase 2: Quality refinement (optional).
        if self.should_run_quality() {
            let phase2_start = Instant::now();
            metrics.quality_embedder_id = self.quality_embedder.as_ref().map(|e| e.id().to_owned());

            match self
                .run_phase2(cx, &canon_query, k, &initial_hits, &text_fn, &mut metrics)
                .await
            {
                Ok(refined_results) => {
                    metrics.phase2_total_ms = phase2_start.elapsed().as_secs_f64() * 1000.0;
                    self.export_search_metrics(query_class, &metrics, refined_results.len(), true);
                    on_phase(SearchPhase::Refined {
                        results: refined_results,
                        latency: phase2_start.elapsed(),
                        metrics: PhaseMetrics {
                            embedder_id: self
                                .quality_embedder
                                .as_ref()
                                .map_or("none", |e| e.id())
                                .to_owned(),
                            vectors_searched: self.index.doc_count(),
                            lexical_candidates: metrics.lexical_candidates,
                            fused_count: k,
                        },
                        rank_changes: metrics.rank_changes.clone(),
                    });
                }
                Err(SearchError::Cancelled { phase, reason }) => {
                    return Err(SearchError::Cancelled { phase, reason });
                }
                Err(err) => {
                    metrics.phase2_total_ms = phase2_start.elapsed().as_secs_f64() * 1000.0;
                    metrics.skip_reason = Some(format!("{err}"));
                    self.export_error(&err);
                    on_phase(SearchPhase::RefinementFailed {
                        initial_results: initial_hits,
                        error: err,
                        latency: phase2_start.elapsed(),
                    });
                }
            }
        } else if self.config.fast_only {
            metrics.skip_reason = Some("fast_only".to_owned());
        } else {
            metrics.skip_reason = Some("no_quality_embedder".to_owned());
        }

        Ok(metrics)
    }

    /// Convenience method that collects all phases and returns the best results.
    ///
    /// Returns the refined results if Phase 2 succeeds, otherwise the initial results.
    ///
    /// # Errors
    ///
    /// Same as [`search`](Self::search).
    pub async fn search_collect(
        &self,
        cx: &Cx,
        query: &str,
        k: usize,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        let mut best_results = Vec::new();
        let metrics = self
            .search(
                cx,
                query,
                k,
                |_| None,
                |phase| match phase {
                    SearchPhase::Initial { results, .. } | SearchPhase::Refined { results, .. } => {
                        best_results = results;
                    }
                    SearchPhase::RefinementFailed { .. } => {
                        // Keep the initial results already stored in best_results.
                    }
                },
            )
            .await?;
        Ok((best_results, metrics))
    }

    /// Run Phase 1: fast embedding + optional lexical + RRF fusion.
    async fn run_phase1(
        &self,
        cx: &Cx,
        query: &str,
        k: usize,
        query_class: QueryClass,
        metrics: &mut TwoTierMetrics,
    ) -> SearchResult<Vec<ScoredResult>> {
        let base_candidates = candidate_count(k, 0, self.config.candidate_multiplier);

        // Adaptive budgets: identifiers lean lexical, NL leans semantic.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let semantic_budget =
            (base_candidates as f32 * query_class.semantic_budget_multiplier()) as usize;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let lexical_budget =
            (base_candidates as f32 * query_class.lexical_budget_multiplier()) as usize;

        let rrf_config = RrfConfig {
            k: self.config.rrf_k,
        };

        // Fast embedding.
        let embed_start = Instant::now();
        let fast_embed_result = self.fast_embedder.embed(cx, query).await;
        metrics.fast_embed_ms = embed_start.elapsed().as_secs_f64() * 1000.0;

        // Lexical search (runs regardless of embedding success).
        let lexical_results = self.run_lexical(cx, query, lexical_budget, metrics).await;

        match fast_embed_result {
            Ok(query_vec) => {
                self.export_embedding_metrics(
                    self.fast_embedder.as_ref(),
                    1,
                    metrics.fast_embed_ms,
                );
                // Vector search.
                let search_start = Instant::now();
                let fast_hits = self.index.search_fast(&query_vec, semantic_budget)?;
                metrics.vector_search_ms = search_start.elapsed().as_secs_f64() * 1000.0;
                metrics.semantic_candidates = fast_hits.len();

                // RRF fusion if lexical results are available.
                let fuse_start = Instant::now();
                let results = lexical_results.as_ref().map_or_else(
                    || vector_hits_to_scored_results(&fast_hits, k),
                    |lexical| {
                        let fused = rrf_fuse(lexical, &fast_hits, k, 0, &rrf_config);
                        fused_hits_to_scored_results(&fused)
                    },
                );
                metrics.rrf_fusion_ms = fuse_start.elapsed().as_secs_f64() * 1000.0;

                Ok(results)
            }
            Err(embed_err) => {
                // Graceful degradation: use lexical-only results if available.
                if let Some(ref lexical) = lexical_results {
                    self.export_error(&embed_err);
                    tracing::warn!(
                        error = %embed_err,
                        "fast embedding failed, falling back to lexical-only results"
                    );
                    Ok(lexical.iter().take(k).cloned().collect())
                } else {
                    Err(embed_err)
                }
            }
        }
    }

    /// Run Phase 2: quality embedding + blend + optional rerank.
    #[allow(clippy::too_many_lines)]
    async fn run_phase2(
        &self,
        cx: &Cx,
        query: &str,
        k: usize,
        initial_results: &[ScoredResult],
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
        metrics: &mut TwoTierMetrics,
    ) -> SearchResult<Vec<ScoredResult>> {
        let quality_embedder =
            self.quality_embedder
                .as_ref()
                .ok_or_else(|| SearchError::EmbedderUnavailable {
                    model: "quality".into(),
                    reason: "no quality embedder configured".into(),
                })?;

        // Quality embedding.
        let embed_start = Instant::now();
        let quality_vec = match quality_embedder.embed(cx, query).await {
            Ok(quality_vec) => {
                metrics.quality_embed_ms = embed_start.elapsed().as_secs_f64() * 1000.0;
                self.export_embedding_metrics(
                    quality_embedder.as_ref(),
                    1,
                    metrics.quality_embed_ms,
                );
                quality_vec
            }
            Err(err) => {
                metrics.quality_embed_ms = embed_start.elapsed().as_secs_f64() * 1000.0;
                return Err(err);
            }
        };

        // Get quality scores for top candidates from initial phase.
        let search_start = Instant::now();
        let fast_hits: Vec<VectorHit> = initial_results
            .iter()
            .enumerate()
            .map(|(i, r)| VectorHit {
                #[allow(clippy::cast_possible_truncation)]
                index: i as u32,
                score: r.fast_score.unwrap_or(r.score),
                doc_id: r.doc_id.clone(),
            })
            .collect();

        // Look up indices in the fast index for quality scoring.
        let fast_indices: Vec<usize> = fast_hits
            .iter()
            .filter_map(|h| self.index.doc_ids().iter().position(|id| *id == h.doc_id))
            .collect();

        let quality_scores = self
            .index
            .quality_scores_for_indices(&quality_vec, &fast_indices)?;
        metrics.quality_search_ms = search_start.elapsed().as_secs_f64() * 1000.0;

        // Build quality VectorHits for blending.
        let quality_hits: Vec<VectorHit> = fast_indices
            .iter()
            .zip(quality_scores.iter())
            .map(|(&idx, &score)| VectorHit {
                #[allow(clippy::cast_possible_truncation)]
                index: idx as u32,
                score,
                doc_id: self.index.doc_ids()[idx].clone(),
            })
            .collect();

        // Blend fast + quality scores.
        let blend_start = Instant::now();
        #[allow(clippy::cast_possible_truncation)]
        let blend_factor = self.config.quality_weight as f32;
        let blended = blend_two_tier(&fast_hits, &quality_hits, blend_factor);
        metrics.blend_ms = blend_start.elapsed().as_secs_f64() * 1000.0;

        // Compute rank changes (initial vs refined).
        let rank_changes = compute_rank_changes(&fast_hits, &blended);
        let tau = kendall_tau(&fast_hits, &blended);
        metrics.kendall_tau = tau;
        metrics.rank_changes = rank_changes;

        // Convert blended to scored results.
        #[allow(unused_mut)] // mut needed when `rerank` feature is enabled
        let mut results: Vec<ScoredResult> = blended
            .iter()
            .take(k)
            .map(|hit| {
                let fast_score = fast_hits
                    .iter()
                    .find(|h| h.doc_id == hit.doc_id)
                    .map(|h| h.score);
                let quality_score = quality_hits
                    .iter()
                    .find(|h| h.doc_id == hit.doc_id)
                    .map(|h| h.score);
                ScoredResult {
                    doc_id: hit.doc_id.clone(),
                    score: hit.score,
                    source: ScoreSource::SemanticQuality,
                    fast_score,
                    quality_score,
                    lexical_score: None,
                    rerank_score: None,
                    metadata: None,
                }
            })
            .collect();

        // Optional cross-encoder reranking.
        if let Some(ref reranker) = self.reranker {
            let rerank_start = Instant::now();
            #[cfg(feature = "rerank")]
            {
                if let Err(err) = frankensearch_rerank::pipeline::rerank_step(
                    cx,
                    reranker.as_ref(),
                    query,
                    &mut results,
                    text_fn,
                    k.min(100),
                    5,
                )
                .await
                {
                    self.export_error(&err);
                    return Err(err);
                }
            }
            #[cfg(not(feature = "rerank"))]
            {
                let _ = (reranker, text_fn);
                tracing::debug!("reranker configured but `rerank` feature not enabled");
            }
            metrics.rerank_ms = rerank_start.elapsed().as_secs_f64() * 1000.0;
        }

        Ok(results)
    }

    /// Run optional lexical search, returning results or None.
    async fn run_lexical(
        &self,
        cx: &Cx,
        query: &str,
        candidates: usize,
        metrics: &mut TwoTierMetrics,
    ) -> Option<Vec<ScoredResult>> {
        let lexical = self.lexical.as_ref()?;
        let start = Instant::now();
        match lexical.search(cx, query, candidates).await {
            Ok(results) => {
                metrics.lexical_search_ms = start.elapsed().as_secs_f64() * 1000.0;
                metrics.lexical_candidates = results.len();
                Some(results)
            }
            Err(err) => {
                self.export_error(&err);
                tracing::warn!(error = %err, "lexical search failed, continuing without");
                metrics.lexical_search_ms = start.elapsed().as_secs_f64() * 1000.0;
                None
            }
        }
    }

    fn export_search_metrics(
        &self,
        query_class: QueryClass,
        metrics: &TwoTierMetrics,
        result_count: usize,
        refined: bool,
    ) {
        let Some(exporter) = self.config.metrics_exporter.as_ref() else {
            return;
        };
        let payload = SearchMetrics {
            mode: SearchMode::TwoTier,
            query_class: Some(query_class),
            total_latency_ms: metrics.phase1_total_ms
                + if refined {
                    metrics.phase2_total_ms
                } else {
                    0.0
                },
            phase1_latency_ms: Some(metrics.phase1_total_ms),
            phase2_latency_ms: if refined {
                Some(metrics.phase2_total_ms)
            } else {
                None
            },
            result_count,
            lexical_candidates: metrics.lexical_candidates,
            semantic_candidates: metrics.semantic_candidates,
            refined,
        };
        exporter.on_search_completed(&payload);
    }

    fn export_embedding_metrics(
        &self,
        embedder: &dyn Embedder,
        batch_size: usize,
        duration_ms: f64,
    ) {
        let Some(exporter) = self.config.metrics_exporter.as_ref() else {
            return;
        };
        let payload = EmbeddingMetrics {
            embedder_id: embedder.id().to_owned(),
            batch_size,
            duration_ms,
            dimension: embedder.dimension(),
            is_semantic: embedder.is_semantic(),
        };
        exporter.on_embedding_completed(&payload);
    }

    fn export_error(&self, error: &SearchError) {
        if let Some(exporter) = self.config.metrics_exporter.as_ref() {
            exporter.on_error(error);
        }
    }

    /// Whether quality refinement should run.
    fn should_run_quality(&self) -> bool {
        !self.config.fast_only && self.quality_embedder.is_some()
    }
}

// Implement Debug manually since trait objects don't derive Debug.
impl std::fmt::Debug for TwoTierSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwoTierSearcher")
            .field("fast_embedder", &self.fast_embedder.id())
            .field(
                "quality_embedder",
                &self.quality_embedder.as_ref().map(|e| e.id()),
            )
            .field("has_lexical", &self.lexical.is_some())
            .field("has_reranker", &self.reranker.is_some())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Convert `FusedHit` results to `ScoredResult`.
fn fused_hits_to_scored_results(
    fused: &[frankensearch_core::types::FusedHit],
) -> Vec<ScoredResult> {
    fused
        .iter()
        .map(|fh| {
            #[allow(clippy::cast_possible_truncation)]
            let score = fh.rrf_score as f32;
            ScoredResult {
                doc_id: fh.doc_id.clone(),
                score,
                source: if fh.in_both_sources {
                    ScoreSource::Hybrid
                } else if fh.lexical_rank.is_some() {
                    ScoreSource::Lexical
                } else {
                    ScoreSource::SemanticFast
                },
                fast_score: fh.semantic_score,
                quality_score: None,
                lexical_score: fh.lexical_score,
                rerank_score: None,
                metadata: None,
            }
        })
        .collect()
}

/// Convert `VectorHit` results to `ScoredResult` (semantic-only mode).
fn vector_hits_to_scored_results(hits: &[VectorHit], k: usize) -> Vec<ScoredResult> {
    hits.iter()
        .take(k)
        .map(|h| ScoredResult {
            doc_id: h.doc_id.clone(),
            score: h.score,
            source: ScoreSource::SemanticFast,
            fast_score: Some(h.score),
            quality_score: None,
            lexical_score: None,
            rerank_score: None,
            metadata: None,
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::unnecessary_literal_bound,
    clippy::cast_precision_loss,
    clippy::significant_drop_tightening
)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

    use frankensearch_core::traits::{MetricsExporter, ModelCategory, SearchFuture};
    use frankensearch_core::types::{EmbeddingMetrics, IndexMetrics, SearchMetrics};

    use super::*;

    // ─── Stub Embedder ──────────────────────────────────────────────────

    struct StubEmbedder {
        id: &'static str,
        dimension: usize,
    }

    impl StubEmbedder {
        const fn new(id: &'static str, dimension: usize) -> Self {
            Self { id, dimension }
        }
    }

    impl Embedder for StubEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            let dim = self.dimension;
            Box::pin(async move {
                let mut vec = vec![0.0; dim];
                if !vec.is_empty() {
                    vec[0] = 1.0; // Simple deterministic embedding
                }
                Ok(vec)
            })
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn id(&self) -> &str {
            self.id
        }

        fn model_name(&self) -> &str {
            self.id
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    // ─── Failing Embedder ───────────────────────────────────────────────

    struct FailingEmbedder;

    impl Embedder for FailingEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async {
                Err(SearchError::EmbeddingFailed {
                    model: "failing-embedder".into(),
                    source: Box::new(std::io::Error::other("intentional test failure")),
                })
            })
        }

        fn dimension(&self) -> usize {
            4
        }

        fn id(&self) -> &str {
            "failing-embedder"
        }

        fn model_name(&self) -> &str {
            "failing-embedder"
        }

        fn is_semantic(&self) -> bool {
            false
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    // ─── Stub Lexical Search ────────────────────────────────────────────

    struct StubLexical;

    impl LexicalSearch for StubLexical {
        fn search<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            limit: usize,
        ) -> SearchFuture<'a, Vec<ScoredResult>> {
            Box::pin(async move {
                Ok((0..limit.min(3))
                    .map(|i| ScoredResult {
                        doc_id: format!("lex-doc-{i}"),
                        score: (3 - i) as f32,
                        source: ScoreSource::Lexical,
                        fast_score: None,
                        quality_score: None,
                        lexical_score: Some((3 - i) as f32),
                        rerank_score: None,
                        metadata: None,
                    })
                    .collect())
            })
        }

        fn index_document<'a>(
            &'a self,
            _cx: &'a Cx,
            _doc: &'a frankensearch_core::types::IndexableDocument,
        ) -> SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn index_documents<'a>(
            &'a self,
            _cx: &'a Cx,
            _docs: &'a [frankensearch_core::types::IndexableDocument],
        ) -> SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn commit<'a>(&'a self, _cx: &'a Cx) -> SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn doc_count(&self) -> usize {
            3
        }
    }

    #[derive(Debug, Default)]
    struct RecordingExporter {
        search: Mutex<Vec<SearchMetrics>>,
        embedding: Mutex<Vec<EmbeddingMetrics>>,
        index: Mutex<Vec<IndexMetrics>>,
        errors: Mutex<Vec<String>>,
    }

    impl MetricsExporter for RecordingExporter {
        fn on_search_completed(&self, metrics: &SearchMetrics) {
            self.search
                .lock()
                .expect("search metrics lock")
                .push(metrics.clone());
        }

        fn on_embedding_completed(&self, metrics: &EmbeddingMetrics) {
            self.embedding
                .lock()
                .expect("embedding metrics lock")
                .push(metrics.clone());
        }

        fn on_index_updated(&self, metrics: &IndexMetrics) {
            self.index
                .lock()
                .expect("index metrics lock")
                .push(metrics.clone());
        }

        fn on_error(&self, error: &SearchError) {
            self.errors
                .lock()
                .expect("error metrics lock")
                .push(error.to_string());
        }
    }

    // ─── Test Helpers ───────────────────────────────────────────────────

    fn build_test_index(dimension: usize) -> Arc<TwoTierIndex> {
        let dir = std::env::temp_dir().join(format!(
            "frankensearch-searcher-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let mut builder =
            TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("create index");
        builder.set_fast_embedder_id("stub-fast");
        for i in 0..10 {
            let mut vec = vec![0.0; dimension];
            vec[i % dimension] = 1.0;
            builder
                .add_fast_record(format!("doc-{i}"), &vec)
                .expect("add record");
        }
        Arc::new(builder.finish().expect("finish index"))
    }

    // ─── Tests ──────────────────────────────────────────────────────────

    #[test]
    fn search_empty_query_returns_no_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

            let mut phases = Vec::new();
            let metrics = searcher
                .search(&cx, "", 10, |_| None, |p| phases.push(format!("{p:?}")))
                .await
                .unwrap();

            assert!(phases.is_empty());
            assert!(metrics.phase1_total_ms.abs() < f64::EPSILON);
        });
    }

    #[test]
    fn search_zero_k_returns_no_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

            let mut phases = Vec::new();
            searcher
                .search(&cx, "test", 0, |_| None, |p| phases.push(format!("{p:?}")))
                .await
                .unwrap();

            assert!(phases.is_empty());
        });
    }

    #[test]
    fn search_fast_only_yields_initial_phase() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

            let mut phase_count = 0;
            let mut got_initial = false;
            let metrics = searcher
                .search(
                    &cx,
                    "test query",
                    5,
                    |_| None,
                    |phase| {
                        phase_count += 1;
                        if matches!(phase, SearchPhase::Initial { .. }) {
                            got_initial = true;
                        }
                    },
                )
                .await
                .unwrap();

            assert_eq!(phase_count, 1);
            assert!(got_initial);
            assert!(metrics.phase1_total_ms > 0.0);
            assert!(
                metrics.skip_reason.is_some(),
                "should report skip reason for no quality embedder"
            );
        });
    }

    #[test]
    fn search_with_quality_yields_two_phases() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(StubEmbedder::new("quality", 4));

            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                .with_quality_embedder(quality);

            let mut phase_count = 0;
            let mut got_initial = false;
            let mut got_refined = false;
            let metrics = searcher
                .search(
                    &cx,
                    "test query",
                    5,
                    |_| None,
                    |phase| {
                        phase_count += 1;
                        match phase {
                            SearchPhase::Initial { .. } => got_initial = true,
                            SearchPhase::Refined { .. } => got_refined = true,
                            SearchPhase::RefinementFailed { .. } => {}
                        }
                    },
                )
                .await
                .unwrap();

            assert_eq!(phase_count, 2);
            assert!(got_initial);
            assert!(got_refined);
            assert!(metrics.quality_embed_ms > 0.0);
        });
    }

    #[test]
    fn fast_only_config_skips_quality() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(StubEmbedder::new("quality", 4));

            let config = TwoTierConfig {
                fast_only: true,
                ..TwoTierConfig::default()
            };

            let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);

            let mut phase_count = 0;
            let metrics = searcher
                .search(&cx, "test", 5, |_| None, |_| phase_count += 1)
                .await
                .unwrap();

            assert_eq!(phase_count, 1, "fast_only should skip quality phase");
            assert_eq!(metrics.skip_reason.as_deref(), Some("fast_only"));
        });
    }

    #[test]
    fn fast_embed_failure_with_lexical_degrades_gracefully() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
            let lexical: Arc<dyn LexicalSearch> = Arc::new(StubLexical);

            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default())
                .with_lexical(lexical);

            let mut got_initial = false;
            let mut initial_count = 0;
            searcher
                .search(
                    &cx,
                    "test",
                    5,
                    |_| None,
                    |phase| {
                        if let SearchPhase::Initial { results, .. } = phase {
                            got_initial = true;
                            initial_count = results.len();
                        }
                    },
                )
                .await
                .unwrap();

            assert!(got_initial, "should fall back to lexical-only results");
            assert!(initial_count > 0, "should have lexical results");
        });
    }

    #[test]
    fn fast_embed_failure_without_lexical_returns_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder: Arc<dyn Embedder> = Arc::new(FailingEmbedder);

            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

            let result = searcher.search(&cx, "test", 5, |_| None, |_| {}).await;

            assert!(
                result.is_err(),
                "should propagate error without lexical fallback"
            );
        });
    }

    #[test]
    fn search_collect_returns_best_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));

            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

            let (results, metrics) = searcher.search_collect(&cx, "test", 5).await.unwrap();

            assert!(!results.is_empty());
            assert!(metrics.phase1_total_ms > 0.0);
        });
    }

    #[test]
    fn metrics_track_query_class() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));

            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

            let (_, metrics) = searcher
                .search_collect(&cx, "how does distributed consensus work", 5)
                .await
                .unwrap();

            assert!(metrics.query_class.is_some());
            assert!(metrics.fast_embedder_id.is_some());
        });
    }

    #[test]
    fn debug_impl_works() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
        let debug_str = format!("{searcher:?}");
        assert!(debug_str.contains("TwoTierSearcher"));
        assert!(debug_str.contains("fast"));
    }

    #[test]
    fn metrics_exporter_receives_search_and_embedding_callbacks() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(StubEmbedder::new("quality", 4));
            let exporter = Arc::new(RecordingExporter::default());
            let config = TwoTierConfig::default().with_metrics_exporter(exporter.clone());

            let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);
            let _ = searcher
                .search(&cx, "test query", 5, |_| None, |_| {})
                .await
                .unwrap();

            {
                let search_events = exporter.search.lock().expect("search lock");
                assert_eq!(search_events.len(), 2);
                assert!(search_events.iter().any(|m| !m.refined));
                assert!(search_events.iter().any(|m| m.refined));
            }
            {
                let embedding_events = exporter.embedding.lock().expect("embedding lock");
                assert!(embedding_events.len() >= 2);
            }
            {
                let errors = exporter.errors.lock().expect("errors lock");
                assert!(errors.is_empty());
            }
        });
    }

    #[test]
    fn metrics_exporter_receives_degradation_errors() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
            let lexical: Arc<dyn LexicalSearch> = Arc::new(StubLexical);
            let exporter = Arc::new(RecordingExporter::default());
            let config = TwoTierConfig::default().with_metrics_exporter(exporter.clone());

            let searcher = TwoTierSearcher::new(index, embedder, config).with_lexical(lexical);
            let _ = searcher
                .search(&cx, "test", 5, |_| None, |_| {})
                .await
                .unwrap();

            {
                let errors = exporter.errors.lock().expect("errors lock");
                assert!(!errors.is_empty());
            }
            {
                let search_events = exporter.search.lock().expect("search lock");
                assert_eq!(search_events.len(), 1);
                assert!(!search_events[0].refined);
            }
        });
    }
}
