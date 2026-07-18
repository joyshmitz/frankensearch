//! Optional HNSW approximate nearest-neighbor index (`ann` feature).
//!
//! This module wraps `hnsw_rs` behind a frankensearch-native API.
//!
//! # Persistence
//!
//! The metadata sidecar (e.g. `vector.fast.hnsw`) stores `doc_ids`, config and
//! dimension as JSON. Since format v2 the native `hnsw_rs` graph is also
//! persisted next to it as `vector.fast.hnsw.graph` + `vector.fast.hnsw.data`,
//! so `load()` deserializes the prebuilt graph directly instead of rebuilding
//! it from vectors. Format v4 fingerprints every live source vector so a stale
//! graph cannot survive an unsampled vector change. Legacy sidecars and any
//! load failure fall back to the rebuild-from-`VectorIndex` path.

use std::path::Path;
use std::time::Instant;

use frankensearch_core::{SearchError, SearchResult, VectorHit};
use hnsw_rs::prelude::{AnnT, DistDot, Hnsw, HnswIo};
use serde::{Deserialize, Serialize};

use crate::VectorIndex;
use crate::recall_certificate::{EfCalibration, calibrate_certified_ef};

/// Default HNSW `M` (max connections per node).
pub const HNSW_DEFAULT_M: usize = 16;
/// Default HNSW `ef_construction` (build-time beam width).
pub const HNSW_DEFAULT_EF_CONSTRUCTION: usize = 200;
/// Default HNSW `ef_search` (query-time beam width).
pub const HNSW_DEFAULT_EF_SEARCH: usize = 100;
/// Default HNSW max layer depth.
pub const HNSW_DEFAULT_MAX_LAYER: usize = 16;

/// ANN construction/runtime parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswConfig {
    /// HNSW `M` (max connections per node).
    pub m: usize,
    /// HNSW `ef_construction` (build-time beam width).
    pub ef_construction: usize,
    /// Default HNSW `ef_search` (query-time beam width).
    pub ef_search: usize,
    /// Maximum HNSW layer depth.
    pub max_layer: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: HNSW_DEFAULT_M,
            ef_construction: HNSW_DEFAULT_EF_CONSTRUCTION,
            ef_search: HNSW_DEFAULT_EF_SEARCH,
            max_layer: HNSW_DEFAULT_MAX_LAYER,
        }
    }
}

/// Current on-disk metadata format. v2 added the native graph sidecars
/// (`*.hnsw.graph` + `*.hnsw.data`) alongside the JSON metadata; v3 records
/// graphs built with the dimension-aware `DistDot` roundoff budget; v4 replaces
/// the sampled source fingerprint with a digest of every live vector. Older
/// native graphs must be rebuilt under the current persistence contract.
const HNSW_META_FORMAT_CURRENT: u32 = 4;

// Keep the classical gamma_k floating-point error model in its well-conditioned
// region. With u = f32::EPSILON / 2 and k = 8n + 32, this cap is exactly the
// largest dimension for which k*u <= 1/4.
const DIST_DOT_MAX_DIMENSION: usize = 524_284;

#[derive(Debug, Clone, Copy)]
struct DistDotBudget {
    radius_squared: f32,
    score_tolerance: f32,
}

/// On-disk metadata for the HNSW index.
#[derive(Debug, Serialize, Deserialize)]
struct HnswMeta {
    /// Sidecar format. Absent in legacy v1 metadata (deserializes to 0).
    #[serde(default)]
    format_version: u32,
    doc_ids: Vec<String>,
    config: HnswConfig,
    dimension: usize,
    /// Deterministic fingerprint of the vectors the persisted graph was built
    /// from (FNV-1a 64 over every live f32 vector; see [`fingerprint_vectors`]).
    /// Lets native-graph load detect "doc IDs match
    /// but the underlying vectors were silently swapped" and fall back to a
    /// rebuild rather than serve stale ANN hits.
    ///
    /// Absent in legacy metadata (deserializes to 0). Legacy formats rebuild
    /// before the native fast path; current-format sidecars compare 0 like any
    /// other fingerprint and therefore cannot use omission to bypass validation.
    #[serde(default)]
    vector_fingerprint: u64,
}

/// Diagnostics for one ANN query.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnSearchStats {
    /// Number of vectors indexed.
    pub index_size: usize,
    /// Vector dimensionality.
    pub dimension: usize,
    /// Effective ef used for this query.
    pub ef_search: usize,
    /// Requested `k`.
    pub k_requested: usize,
    /// Returned result count.
    pub k_returned: usize,
    /// Query latency in microseconds.
    pub search_time_us: u64,
    /// Whether this path is approximate ANN.
    pub is_approximate: bool,
    /// Estimated recall@k from the ef/k ratio (see [`estimate_recall`]).
    ///
    /// This is a heuristic point estimate with NO guarantee. For a certified,
    /// distribution-free recall bound (the automated replacement for a human
    /// recall-budget sign-off), use
    /// [`crate::recall_certificate::conformal_recall_lower_bound`] over a measured
    /// calibration sample instead.
    pub estimated_recall: f64,
}

/// HNSW ANN index over vectors aligned to `VectorIndex` row order.
pub struct HnswIndex {
    hnsw: Hnsw<'static, f32, DistDot>,
    doc_ids: Vec<String>,
    dimension: usize,
    config: HnswConfig,
    /// Fingerprint of the vectors the graph was built from. See
    /// [`HnswMeta::vector_fingerprint`].
    vector_fingerprint: u64,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("points", &self.hnsw.get_nb_point())
            .field("doc_ids", &self.doc_ids.len())
            .field("dimension", &self.dimension)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl HnswIndex {
    /// Build a new HNSW index from an opened `VectorIndex`.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `SearchError::InvalidConfig` for invalid HNSW params
    /// - `SearchError::IndexCorrupted` if `vectors/doc_ids` cannot be decoded
    pub fn build_from_vector_index(index: &VectorIndex, config: HnswConfig) -> SearchResult<Self> {
        let dimension = index.dimension();
        let mut doc_ids = Vec::with_capacity(index.record_count());
        let mut vectors = Vec::with_capacity(index.record_count());
        for i in 0..index.record_count() {
            if index.is_deleted(i) {
                continue;
            }
            doc_ids.push(index.doc_id_at(i)?.to_owned());
            vectors.push(index.vector_at_f32(i)?);
        }
        Self::build_from_parts(doc_ids, vectors, dimension, config)
    }

    /// Load an ANN index from disk, rebuilding the graph using vectors from `source_index`.
    ///
    /// The persistent `.hnsw` file only contains metadata and `doc_ids`. The actual
    /// vector data is read from the provided `VectorIndex` to save disk space and
    /// ensure consistency.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::IndexCorrupted` if metadata is missing/malformed or
    /// if referenced documents are missing from `source_index`.
    pub fn load(path: &Path, source_index: &VectorIndex) -> SearchResult<Self> {
        let metadata_bytes = std::fs::read(path).map_err(SearchError::Io)?;
        let meta: HnswMeta = serde_json::from_slice(&metadata_bytes)
            .map_err(|e| ann_corrupted(path, format!("failed to parse HNSW metadata: {e}")))?;

        if meta.dimension != source_index.dimension() {
            return Err(ann_corrupted(
                path,
                format!(
                    "dimension mismatch: hnsw={} source={}",
                    meta.dimension,
                    source_index.dimension()
                ),
            ));
        }

        // Validate before the native fast path. Otherwise a forged current-format sidecar
        // could bypass the dimension bound enforced by graph construction.
        dist_dot_budget(meta.dimension)?;

        // Current format: deserialize the prebuilt native graph directly, skipping the
        // O(n log n) rebuild. Any problem (missing/corrupt sidecars, point-count
        // mismatch, or stale vector fingerprint) returns None and we fall
        // through to the rebuild path, so a bad graph sidecar degrades to
        // "slow load" rather than a hard failure.
        if meta.format_version == HNSW_META_FORMAT_CURRENT
            && let Some(index) = Self::try_load_native_graph(path, &meta, source_index)
        {
            return Ok(index);
        } else if meta.format_version != HNSW_META_FORMAT_CURRENT {
            tracing::warn!(
                path = %path.display(),
                format_version = meta.format_version,
                current_format_version = HNSW_META_FORMAT_CURRENT,
                "rebuilding HNSW sidecar written with a different persistence contract; \
                 re-save to skip rebuild on the next cold load"
            );
        }

        // v1/legacy or fallback: rehydrate vectors from the source index and
        // rebuild the graph.
        let mut vectors = Vec::with_capacity(meta.doc_ids.len());
        for doc_id in &meta.doc_ids {
            let idx = source_index.find_index_by_doc_id(doc_id)?.ok_or_else(|| {
                ann_corrupted(path, format!("doc_id '{doc_id}' missing from source index"))
            })?;
            vectors.push(source_index.vector_at_f32(idx)?);
        }

        Self::build_from_parts(meta.doc_ids, vectors, meta.dimension, meta.config)
    }

    /// Attempt to load the prebuilt native `hnsw_rs` graph for a current-format sidecar.
    ///
    /// Returns `None` (so the caller rebuilds) if any of the following hold —
    /// a degraded sidecar always degrades to "slow load", never a hard error:
    /// - the `.hnsw.graph` / `.hnsw.data` sidecars are absent,
    /// - `hnsw_rs` fails to deserialize them,
    /// - the loaded point count disagrees with the metadata `doc_ids`,
    /// - the metadata `doc_ids` disagree with the live `VectorIndex`'s
    ///   doc-id sequence (live tombstones excluded),
    /// - the persisted vector fingerprint disagrees with the live
    ///   `VectorIndex`'s fingerprint (i.e. vectors were swapped behind the
    ///   same doc ids — the case the prompt explicitly calls out).
    fn try_load_native_graph(
        path: &Path,
        meta: &HnswMeta,
        source_index: &VectorIndex,
    ) -> Option<Self> {
        let parent = path
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let basename = hnsw_sidecar_basename(path).ok()?;
        let graph = parent.join(format!("{basename}.hnsw.graph"));
        let data = parent.join(format!("{basename}.hnsw.data"));
        if !graph.is_file() || !data.is_file() {
            return None;
        }

        // Validate doc-id sequence against the live VectorIndex *before*
        // touching the (potentially expensive) hnsw_rs load.
        if !meta_matches_live_doc_ids(meta, source_index).ok()? {
            tracing::warn!(
                path = %path.display(),
                "HNSW sidecar doc_ids disagree with live VectorIndex; rebuilding"
            );
            return None;
        }

        // Validate the vector-content fingerprint against the live VectorIndex.
        // This is the critical stale-vectors guard: if a caller swaps the FSVI
        // contents while keeping the same doc IDs in the same order, the
        // persisted graph would otherwise silently serve hits against vectors
        // that no longer exist. `try_load_native_graph` is only called for the
        // current format, so a missing fingerprint cannot be treated as a
        // legacy exception: 0 is compared like any other digest value.
        let live_fp =
            fingerprint_live_vector_index(source_index, meta.doc_ids.len(), meta.dimension).ok()?;
        if live_fp != meta.vector_fingerprint {
            tracing::warn!(
                path = %path.display(),
                expected = meta.vector_fingerprint,
                actual = live_fp,
                "HNSW sidecar vector fingerprint disagrees with live VectorIndex \
                 (vectors swapped behind matching doc ids); rebuilding"
            );
            return None;
        }

        // `HnswIo::load_hnsw` returns an `Hnsw` borrowed from the `HnswIo`
        // (`'a: 'b`), so to store it in the `'static` field we must keep the
        // `HnswIo` alive for the program. With the default `ReloadOptions`
        // (`datamap: false`) the `HnswIo` holds no bulk data — the graph and
        // vectors are read into the returned `Hnsw`'s owned storage — so the
        // leaked shell is only a couple of paths plus an `Arc`. Leaking it
        // (rather than a self-referential struct or unsafe lifetime transmute)
        // is the simplest sound way to obtain a `'static` graph, and the cost
        // is negligible because a persisted load happens about once per
        // process.
        let hnsw = Box::leak(Box::new(HnswIo::new(parent, &basename)))
            .load_hnsw::<f32, DistDot>()
            .ok()?;

        // Guard against a graph that doesn't match the metadata it shipped with
        // (e.g. truncated dump, mismatched sidecars). The caller additionally
        // validates doc_ids against the live VectorIndex.
        if hnsw.get_nb_point() != meta.doc_ids.len() {
            return None;
        }

        Some(Self {
            hnsw,
            doc_ids: meta.doc_ids.clone(),
            dimension: meta.dimension,
            config: meta.config,
            vector_fingerprint: meta.vector_fingerprint,
        })
    }

    /// Persist the ANN index to disk.
    ///
    /// Writes the JSON metadata sidecar (`doc_ids`, config, dimension) at
    /// `path`, plus the native `hnsw_rs` graph as `{stem}.hnsw.graph` and
    /// `{stem}.hnsw.data` next to it. Vectors are embedded in the native data
    /// sidecar by `hnsw_rs`; the `VectorIndex` is only consulted on a v1/legacy
    /// or fallback rebuild.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Io` on write/dump failure.
    pub fn save(&self, path: &Path) -> SearchResult<()> {
        let parent = path
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)?;

        // Persist the native graph + data sidecars first; only stamp the
        // metadata as current once the graph is durably written, so a partial dump
        // can never advertise a graph that isn't there.
        let basename = hnsw_sidecar_basename(path)?;
        self.hnsw.file_dump(parent, &basename).map_err(|error| {
            SearchError::Io(std::io::Error::other(format!(
                "failed to dump HNSW graph: {error}"
            )))
        })?;

        let meta = HnswMeta {
            format_version: HNSW_META_FORMAT_CURRENT,
            doc_ids: self.doc_ids.clone(),
            config: self.config,
            dimension: self.dimension,
            vector_fingerprint: self.vector_fingerprint,
        };
        let metadata_bytes = serde_json::to_vec(&meta)
            .map_err(|error| SearchError::Io(std::io::Error::other(error.to_string())))?;
        std::fs::write(path, metadata_bytes).map_err(SearchError::Io)?;

        Ok(())
    }

    /// Run ANN query and return hits plus query diagnostics.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if query dimension differs.
    pub fn knn_search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<(Vec<VectorHit>, AnnSearchStats)> {
        if query.len() != self.dimension {
            return Err(SearchError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }

        if query.iter().any(|value| !value.is_finite()) {
            return Err(SearchError::InvalidConfig {
                field: "query".to_owned(),
                value: "non-finite".to_owned(),
                reason: "all query vector values must be finite".to_owned(),
            });
        }

        if k == 0 || self.doc_ids.is_empty() {
            let stats = AnnSearchStats {
                index_size: self.len(),
                dimension: self.dimension,
                ef_search,
                k_requested: k,
                k_returned: 0,
                search_time_us: 0,
                is_approximate: true,
                estimated_recall: 1.0,
            };
            return Ok((Vec::new(), stats));
        }

        let effective_k = k.min(self.doc_ids.len());
        let effective_ef = ef_search.max(effective_k).max(1);
        let budget = dist_dot_budget(self.dimension)?;
        let normalized_query = normalize_for_dist_dot(query.to_vec(), budget);

        let start = Instant::now();
        let neighbors = self
            .hnsw
            .search(&normalized_query, effective_k, effective_ef);
        let elapsed = start.elapsed();
        let search_time_us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);

        let mut hits = Vec::with_capacity(neighbors.len());
        for neighbor in neighbors {
            let doc_id =
                self.doc_ids
                    .get(neighbor.d_id)
                    .ok_or_else(|| SearchError::InvalidConfig {
                        field: "neighbor_id".to_owned(),
                        value: neighbor.d_id.to_string(),
                        reason: "neighbor id exceeds doc_id table".to_owned(),
                    })?;
            let index = u32::try_from(neighbor.d_id).map_err(|_| SearchError::InvalidConfig {
                field: "neighbor_id".to_owned(),
                value: neighbor.d_id.to_string(),
                reason: "neighbor id exceeds u32 range for VectorHit".to_owned(),
            })?;
            let restored_score = (1.0 - neighbor.distance) / budget.radius_squared;
            let score_envelope = -1.0 - budget.score_tolerance..=1.0 + budget.score_tolerance;
            if !restored_score.is_finite() || !score_envelope.contains(&restored_score) {
                return Err(SearchError::SubsystemError {
                    subsystem: "hnsw",
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "DistDot returned distance {} (restored score {restored_score}); \
                             expected a score within [{}, {}]",
                            neighbor.distance,
                            score_envelope.start(),
                            score_envelope.end()
                        ),
                    )),
                });
            }
            hits.push(VectorHit {
                index,
                // Graph and query vectors use the same deterministic radius.
                // Undo that uniform scale so callers continue receiving cosine
                // similarity rather than a dimension-dependent proxy score.
                // Clamp only after proving the deviation is inside the derived
                // floating-point envelope; materially invalid distances fail.
                score: restored_score.clamp(-1.0, 1.0),
                doc_id: doc_id.as_str().into(),
            });
        }
        hits.sort_by(|left, right| {
            left.cmp_by_score(right)
                .then_with(|| left.index.cmp(&right.index))
        });

        let stats = AnnSearchStats {
            index_size: self.len(),
            dimension: self.dimension,
            ef_search: effective_ef,
            k_requested: k,
            k_returned: hits.len(),
            search_time_us,
            is_approximate: true,
            estimated_recall: estimate_recall(effective_ef, effective_k),
        };
        Ok((hits, stats))
    }

    /// Run ANN query and return only the hits.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DimensionMismatch` if query dimension differs.
    pub fn knn_search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        self.knn_search_with_stats(query, k, ef_search)
            .map(|(hits, _)| hits)
    }

    /// Certify the cheapest `ef_search` whose recall meets `target` — the automated
    /// replacement for the human "recall-budget sign-off" that gated ANN-in-BOLD.
    ///
    /// Measures this ANN index's recall@`k` against exact bruteforce
    /// ([`VectorIndex::search_top_k`]) over `calibration_queries`, and returns the
    /// smallest `ef` in `candidate_efs` whose split-conformal recall **lower bound**
    /// is `≥ target` at confidence `1 − alpha` (distribution-free, finite-sample
    /// valid; see [`crate::recall_certificate`]). If none qualifies, returns the
    /// best-certifiable `ef` with `meets_target = false`.
    ///
    /// The exact top-k for each query is independent of `ef`, so it is computed
    /// **once per query** (not per `ef`); only the ANN search re-runs per candidate,
    /// and the sweep short-circuits at the first certified `ef`, so no ANN search is
    /// run at an `ef` larger than the chosen one.
    ///
    /// # Errors
    ///
    /// Propagates any error from the exact [`VectorIndex::search_top_k`] pass. A
    /// failed ANN search for a single (query, ef) is treated as recall `0.0` for that
    /// query — the conservative direction, which can only *lower* a certified bound
    /// (never over-certify).
    pub fn certify_ef_search(
        &self,
        exact_index: &VectorIndex,
        calibration_queries: &[Vec<f32>],
        candidate_efs: &[usize],
        k: usize,
        target: f64,
        alpha: f64,
    ) -> SearchResult<Option<EfCalibration>> {
        // Exact top-k is ef-independent: compute it once per calibration query.
        let exact: Vec<Vec<VectorHit>> = calibration_queries
            .iter()
            .map(|q| exact_index.search_top_k(q, k, None))
            .collect::<SearchResult<_>>()?;

        Ok(calibrate_certified_ef(
            candidate_efs,
            |ef| {
                calibration_queries
                    .iter()
                    .zip(&exact)
                    .map(|(q, exact_hits)| {
                        let approx = self.knn_search(q, k, ef).unwrap_or_default();
                        recall_at_k_of(&approx, exact_hits)
                    })
                    .collect()
            },
            target,
            alpha,
        ))
    }

    /// Returns true when this ANN index matches row order and shape of a `VectorIndex`.
    ///
    /// Since `HnswIndex` no longer stores vectors, this only checks:
    /// 1. Dimension match.
    /// 2. `doc_id` sequence match (ignoring tombstones in `VectorIndex`).
    ///
    /// # Errors
    ///
    /// Propagates decoding errors from `VectorIndex::doc_id_at`.
    pub fn matches_vector_index(&self, index: &VectorIndex) -> SearchResult<bool> {
        if self.dimension != index.dimension() {
            return Ok(false);
        }
        let mut live_position = 0_usize;
        for i in 0..index.record_count() {
            if index.is_deleted(i) {
                continue;
            }
            let Some(expected_doc_id) = self.doc_ids.get(live_position) else {
                // HNSW has fewer docs than VectorIndex
                return Ok(false);
            };
            if expected_doc_id != index.doc_id_at(i)? {
                return Ok(false);
            }
            // We implicitly assume vectors match if doc_ids match, as HNSW
            // is built *from* the VectorIndex on load.
            live_position = live_position.saturating_add(1);
        }
        // Check if HNSW has extra docs
        Ok(live_position == self.doc_ids.len())
    }

    /// Number of indexed vectors.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Whether ANN index has zero vectors.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Vector dimensionality.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// ANN configuration used to build this index.
    #[must_use]
    pub const fn config(&self) -> HnswConfig {
        self.config
    }

    fn build_from_parts(
        doc_ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        dimension: usize,
        config: HnswConfig,
    ) -> SearchResult<Self> {
        validate_config(config)?;
        if dimension == 0 {
            return Err(SearchError::InvalidConfig {
                field: "dimension".to_owned(),
                value: "0".to_owned(),
                reason: "dimension must be greater than zero".to_owned(),
            });
        }
        let budget = dist_dot_budget(dimension)?;
        if doc_ids.len() != vectors.len() {
            return Err(SearchError::InvalidConfig {
                field: "vectors".to_owned(),
                value: vectors.len().to_string(),
                reason: format!("doc_id count {} must match vector count", doc_ids.len()),
            });
        }
        // Fingerprint the raw (un-normalized) input vectors. This is what
        // VectorIndex::vector_at_f32 returns at load time, so a fresh load from
        // the live VectorIndex will produce the same digest iff the underlying
        // bytes are unchanged. Used by the native-graph load path to detect
        // "doc IDs match but vectors were silently swapped" and trigger a
        // rebuild (see try_load_native_graph).
        let vector_fingerprint = fingerprint_vectors(&doc_ids, &vectors);

        let mut normalized_vectors = Vec::with_capacity(vectors.len());
        for (idx, vector) in vectors.into_iter().enumerate() {
            if vector.len() != dimension {
                return Err(SearchError::DimensionMismatch {
                    expected: dimension,
                    found: vector.len(),
                });
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(SearchError::InvalidConfig {
                    field: "vector".to_owned(),
                    value: idx.to_string(),
                    reason: "all vector values must be finite".to_owned(),
                });
            }
            normalized_vectors.push(normalize_for_dist_dot(vector, budget));
        }

        let hnsw = Hnsw::new(
            config.m,
            doc_ids.len().max(1),
            config.max_layer,
            config.ef_construction,
            DistDot,
        );
        if !normalized_vectors.is_empty() {
            let vectors_with_ids: Vec<(&Vec<f32>, usize)> = normalized_vectors
                .iter()
                .enumerate()
                .map(|(index, vector)| (vector, index))
                .collect();
            hnsw.parallel_insert(&vectors_with_ids);
        }

        Ok(Self {
            hnsw,
            doc_ids,
            // vectors: source_vectors, // Removed!
            dimension,
            config,
            vector_fingerprint,
        })
    }
}

/// Derive the `hnsw_rs` sidecar basename from the metadata path.
///
/// `hnsw_rs` writes `{basename}.hnsw.graph` and `{basename}.hnsw.data`, so for
/// a metadata path of `dir/vector.fast.hnsw` we use the stem `vector.fast`,
/// yielding `dir/vector.fast.hnsw.graph` / `.data` — distinct from the
/// metadata file itself.
fn hnsw_sidecar_basename(path: &Path) -> SearchResult<String> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| ann_corrupted(path, "ANN sidecar path has no usable file stem"))
}

/// FNV-1a 64-bit. Chosen because it is deterministic across processes (unlike
/// `ahash`) and stdlib (`DefaultHasher` is randomized + deprecated for
/// persistence), keeps the fingerprint dependency-free, and is plenty for an
/// integrity-style check — we only need collision resistance against a small
/// number of accidental byte-level edits, not adversarial inputs.
const FNV_OFFSET_BASIS_64: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME_64: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn fnv1a_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME_64);
    }
    h
}

/// Feed an `&[f32]` into the FNV-1a state in little-endian byte order without
/// any unsafe reinterpretation (the crate is `#![forbid(unsafe_code)]`).
/// One `to_le_bytes()` per element compiles to a tight loop.
#[inline]
fn fnv1a_update_f32(mut h: u64, vec: &[f32]) -> u64 {
    for &f in vec {
        h = fnv1a_update(h, &f.to_bits().to_le_bytes());
    }
    h
}

/// Compute the persistence fingerprint from the (raw, un-normalized) vectors
/// and doc IDs used to build the graph.
///
/// Every vector contributes in live-row order. Doc IDs are mixed in alongside
/// their vectors, so either a vector edit or a doc-id permutation changes the
/// digest. The output is stored in the native metadata sidecar and re-derived
/// at load time by [`fingerprint_live_vector_index`] against the live
/// `VectorIndex`; a mismatch means "doc IDs match but the underlying vector
/// bytes were silently swapped" → reject the persisted graph.
fn fingerprint_vectors(doc_ids: &[String], vectors: &[Vec<f32>]) -> u64 {
    // Mix length so a truncated-but-prefix-matching index doesn't collide.
    let mut h = fnv1a_update(FNV_OFFSET_BASIS_64, &(doc_ids.len() as u64).to_le_bytes());
    for (i, (doc_id, vector)) in doc_ids.iter().zip(vectors).enumerate() {
        h = fnv1a_update(h, &(i as u64).to_le_bytes());
        h = fnv1a_update(h, doc_id.as_bytes());
        h = fnv1a_update_f32(h, vector);
    }
    h
}

/// Compute the fingerprint from a live `VectorIndex`, matching the layout
/// [`fingerprint_vectors`] produced at build time.
///
/// `expected_len` and `expected_dim` come from the persisted metadata. If the
/// live index has fewer live records than the persisted graph, the digest
/// will not match and the caller falls back to a rebuild — which is the right
/// behavior.
fn fingerprint_live_vector_index(
    index: &VectorIndex,
    expected_len: usize,
    expected_dim: usize,
) -> SearchResult<u64> {
    let mut h = fnv1a_update(FNV_OFFSET_BASIS_64, &(expected_len as u64).to_le_bytes());

    // Walk live records (tombstones excluded), in row order — same iteration
    // order as `build_from_vector_index`.
    let mut live_idx = 0_usize;
    for raw in 0..index.record_count() {
        if index.is_deleted(raw) {
            continue;
        }
        if live_idx >= expected_len {
            // Live index has more records than the persisted graph.
            break;
        }
        let doc_id = index.doc_id_at(raw)?;
        h = fnv1a_update(h, &(live_idx as u64).to_le_bytes());
        h = fnv1a_update(h, doc_id.as_bytes());

        let vec = index.vector_at_f32(raw)?;
        if vec.len() != expected_dim {
            // Dimension drift — perturb the digest so the caller rejects and
            // rebuilds rather than asserting.
            return Ok(h.wrapping_add(1));
        }
        h = fnv1a_update_f32(h, &vec);
        live_idx += 1;
    }
    Ok(h)
}

/// Verify the metadata `doc_ids` sequence matches the live `VectorIndex`'s
/// live (non-tombstoned) doc IDs in row order. Same semantics as the public
/// `matches_vector_index` but doesn't need a constructed `HnswIndex`, so we
/// can check it before paying for the native graph load.
fn meta_matches_live_doc_ids(meta: &HnswMeta, index: &VectorIndex) -> SearchResult<bool> {
    if meta.dimension != index.dimension() {
        return Ok(false);
    }
    let mut live_position = 0_usize;
    for i in 0..index.record_count() {
        if index.is_deleted(i) {
            continue;
        }
        let Some(expected_doc_id) = meta.doc_ids.get(live_position) else {
            return Ok(false);
        };
        if expected_doc_id != index.doc_id_at(i)? {
            return Ok(false);
        }
        live_position = live_position.saturating_add(1);
    }
    Ok(live_position == meta.doc_ids.len())
}

fn validate_config(config: HnswConfig) -> SearchResult<()> {
    if config.m == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_m".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_m must be greater than zero".to_owned(),
        });
    }
    if config.m > 256 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_m".to_owned(),
            value: config.m.to_string(),
            reason: "hnsw_m must be <= 256".to_owned(),
        });
    }
    if config.ef_construction == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_ef_construction".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_ef_construction must be greater than zero".to_owned(),
        });
    }
    if config.ef_search == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_ef_search".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_ef_search must be greater than zero".to_owned(),
        });
    }
    if config.max_layer == 0 {
        return Err(SearchError::InvalidConfig {
            field: "hnsw_max_layer".to_owned(),
            value: "0".to_owned(),
            reason: "hnsw_max_layer must be greater than zero".to_owned(),
        });
    }
    Ok(())
}

fn ann_corrupted(path: &Path, detail: impl Into<String>) -> SearchError {
    SearchError::IndexCorrupted {
        path: path.to_path_buf(),
        detail: detail.into(),
    }
}

/// Derive the normalization and score-restoration budget for `DistDot`.
///
/// A length-`n` f32 dot product has a forward-error bound conventionally
/// expressed as `gamma_k = k*u/(1-k*u)`, where `u = eps/2`. We budget `8n+32`
/// rounded operations: component rescaling, multiplication, lane-local
/// accumulation, horizontal reduction, score restoration, and a 2x
/// architecture margin. Scaling exact unit vectors to
/// `1/sqrt(1+gamma_k)` makes the worst bounded computed self-dot no greater
/// than one. Dimensions for which `k*u > 1/4` are rejected rather than hiding
/// an ill-conditioned error bound behind an arbitrary shrink factor.
#[allow(clippy::cast_possible_truncation)]
fn dist_dot_budget(dimension: usize) -> SearchResult<DistDotBudget> {
    if dimension == 0 || dimension > DIST_DOT_MAX_DIMENSION {
        return Err(SearchError::InvalidConfig {
            field: "dimension".to_owned(),
            value: dimension.to_string(),
            reason: format!(
                "DistDot requires a dimension in 1..={DIST_DOT_MAX_DIMENSION} \
                 so its f32 roundoff bound remains finite and conservative"
            ),
        });
    }

    let dimension = u32::try_from(dimension).map_err(|_| SearchError::InvalidConfig {
        field: "dimension".to_owned(),
        value: dimension.to_string(),
        reason: "dimension exceeds the DistDot f32 error model".to_owned(),
    })?;
    let rounded_operations = f64::from(dimension).mul_add(8.0, 32.0);
    let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
    let accumulated_roundoff = rounded_operations * unit_roundoff;
    debug_assert!(accumulated_roundoff <= 0.25);
    let gamma = accumulated_roundoff / (1.0 - accumulated_roundoff);
    let radius_squared = 1.0 / (1.0 + gamma);
    // Include a small fixed allowance for the final f32 subtraction and
    // division when converting DistDot's distance back to cosine score.
    let score_tolerance = gamma + 8.0 * f64::from(f32::EPSILON);
    Ok(DistDotBudget {
        radius_squared: radius_squared as f32,
        score_tolerance: score_tolerance as f32,
    })
}

#[allow(clippy::cast_possible_truncation)]
fn normalize_for_dist_dot(mut vector: Vec<f32>, budget: DistDotBudget) -> Vec<f32> {
    // f64 accumulation prevents the normalization pass itself from consuming
    // the f32 error budget intended for hnsw_rs/anndists' distance reduction.
    let norm_squared = vector
        .iter()
        .map(|&value| {
            let value = f64::from(value);
            value * value
        })
        .sum::<f64>();
    if norm_squared > 0.0 && norm_squared.is_finite() {
        let radius = f64::from(budget.radius_squared).sqrt();
        let scale = radius / norm_squared.sqrt();
        for value in &mut vector {
            *value = (f64::from(*value) * scale) as f32;
        }
    }
    vector
}

#[cfg(test)]
#[allow(dead_code)] // retained as utility; direct callers use vector_component_close
fn vectors_close(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(&l, &r)| vector_component_close(l, r))
}

#[cfg(test)]
fn vector_component_close(left: f32, right: f32) -> bool {
    if left.to_bits() == right.to_bits() {
        return true;
    }
    // Non-finite values (NaN, Inf) with different bit patterns are never close.
    if !left.is_finite() || !right.is_finite() {
        return false;
    }
    let diff = (left - right).abs();
    let scale = left.abs().max(right.abs()).max(1.0);
    diff <= (f32::EPSILON * 8.0 * scale)
}

fn estimate_recall(ef_search: usize, k: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let numerator = f64::from(u32::try_from(ef_search.max(1)).unwrap_or(u32::MAX));
    let denominator = f64::from(u32::try_from(k).unwrap_or(u32::MAX));
    let ratio = numerator / denominator;
    0.1_f64.mul_add(ratio.log2(), 0.9_f64).clamp(0.0, 1.0)
}

/// Recall@k of `approx` against exact `exact`: the fraction of exact `doc_id`s that
/// also appear in `approx`. Used by [`HnswIndex::certify_ef_search`] to build the
/// measured calibration sample fed to the conformal certificate. `k` is tiny (top-k),
/// so the nested membership scan is trivial.
fn recall_at_k_of(approx: &[VectorHit], exact: &[VectorHit]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let overlap = exact
        .iter()
        .filter(|e| approx.iter().any(|a| a.doc_id == e.doc_id))
        .count();
    #[allow(clippy::cast_precision_loss)]
    let ratio = overlap as f64 / exact.len() as f64;
    ratio
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::Quantization;

    fn temp_path(label: &str, extension: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-hnsw-{label}-{}-{now}.{extension}",
            std::process::id()
        ))
    }

    fn lcg_next(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        u32::try_from((*state >> 32) & u64::from(u32::MAX)).unwrap_or(u32::MAX)
    }

    fn normalized_vector(seed: usize, dimension: usize) -> Vec<f32> {
        let mut state = u64::try_from(seed).unwrap_or(0).wrapping_add(1);
        let mut out = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            let random = lcg_next(&mut state);
            let upper = u16::try_from((random >> 16) & u32::from(u16::MAX)).unwrap_or(u16::MAX);
            let raw = f32::from(upper) / f32::from(u16::MAX);
            out.push((raw * 2.0_f32) - 1.0_f32);
        }
        let norm = out.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut out {
                *value /= norm;
            }
        }
        out
    }

    fn write_index(path: &Path, vectors: &[Vec<f32>]) -> SearchResult<VectorIndex> {
        let dimension = vectors.first().map_or(8, Vec::len);
        let mut writer =
            VectorIndex::create_with_revision(path, "hash", "test", dimension, Quantization::F32)?;
        for (idx, vector) in vectors.iter().enumerate() {
            writer.write_record(&format!("doc-{idx:04}"), vector)?;
        }
        writer.finish()?;
        VectorIndex::open(path)
    }

    fn recall_at_k(approx: &[VectorHit], exact: &[VectorHit]) -> f64 {
        if exact.is_empty() {
            return 1.0;
        }
        let exact_ids: HashSet<&str> = exact.iter().map(|hit| hit.doc_id.as_str()).collect();
        let overlap = approx
            .iter()
            .filter(|hit| exact_ids.contains(hit.doc_id.as_str()))
            .count();
        f64::from(u32::try_from(overlap).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(exact.len()).unwrap_or(u32::MAX))
    }

    #[test]
    fn empty_index_returns_no_hits() {
        let path = temp_path("empty", "fsvi");
        let writer = VectorIndex::create_with_revision(&path, "hash", "test", 8, Quantization::F16)
            .expect("create writer");
        writer.finish().expect("finish");

        let index = VectorIndex::open(&path).expect("open index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let hits = ann
            .knn_search(&normalized_vector(7, 8), 10, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert!(hits.is_empty());
    }

    #[test]
    fn single_vector_round_trip() {
        let path = temp_path("single", "fsvi");
        let index = write_index(&path, &[normalized_vector(1, 32)]).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let hits = ann
            .knn_search(&normalized_vector(1, 32), 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "doc-0000");
    }

    #[test]
    fn higher_ef_improves_or_matches_recall() {
        let fsvi_path = temp_path("ef", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..256).map(|i| normalized_vector(i, 96)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let mut low_total = 0.0_f64;
        let mut high_total = 0.0_f64;
        let mut count = 0_u32;
        for query_seed in (0..128).step_by(16) {
            let query = normalized_vector(query_seed, 96);
            let exact = index.search_top_k(&query, 10, None).expect("exact");
            let low = ann.knn_search(&query, 10, 10).expect("low ef");
            let high = ann.knn_search(&query, 10, 100).expect("high ef");
            low_total += recall_at_k(&low, &exact);
            high_total += recall_at_k(&high, &exact);
            count += 1;
        }

        let count_f = f64::from(count);
        assert!((high_total / count_f) >= (low_total / count_f));
    }

    #[test]
    fn recall_against_bruteforce_is_high() {
        let fsvi_path = temp_path("recall", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..1_000).map(|i| normalized_vector(i, 384)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let mut total_recall = 0.0_f64;
        let mut query_count = 0_u32;
        for query_seed in (0..1_000).step_by(40) {
            let query = normalized_vector(query_seed, 384);
            let exact = index.search_top_k(&query, 10, None).expect("exact");
            let approx = ann
                .knn_search(&query, 10, HNSW_DEFAULT_EF_SEARCH)
                .expect("approx");
            total_recall += recall_at_k(&approx, &exact);
            query_count += 1;
        }

        let avg_recall = total_recall / f64::from(query_count);
        assert!(
            avg_recall >= 0.95,
            "expected avg recall >= 0.95, got {avg_recall:.4}"
        );
    }

    #[test]
    fn certify_ef_search_wires_conformal_certificate_end_to_end() {
        // Real ANN index + real bruteforce feeding the conformal certificate.
        let fsvi_path = temp_path("certify", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..800).map(|i| normalized_vector(i, 384)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        // Calibration queries disjoint from... they're near corpus points, fine for wiring.
        let calibration: Vec<Vec<f32>> = (5..1_600)
            .step_by(64)
            .map(|s| normalized_vector(s, 384))
            .collect();
        let candidate_efs = [10usize, 40, 100, 200];

        // target=0.0 is always certified => cheapest ef, and the sweep must
        // short-circuit immediately (only the smallest ef is ever ANN-searched).
        let trivial = ann
            .certify_ef_search(&index, &calibration, &candidate_efs, 10, 0.0, 0.1)
            .expect("certify")
            .expect("some");
        assert!(trivial.chosen.meets_target);
        assert_eq!(
            trivial.chosen.ef_search, 10,
            "cheapest ef for a trivial target"
        );
        assert_eq!(
            trivial.sweep.len(),
            1,
            "short-circuits at the first certified ef"
        );

        // An unreachable target => no ef meets it, fall back to the best-certifiable
        // (largest ef here, since recall is non-decreasing in ef), full sweep measured,
        // and the certified bound is a real recall in [0, 1].
        let strict = ann
            .certify_ef_search(&index, &calibration, &candidate_efs, 10, 2.0, 0.1)
            .expect("certify")
            .expect("some");
        assert!(!strict.chosen.meets_target);
        assert_eq!(
            strict.sweep.len(),
            candidate_efs.len(),
            "measures all when none certifies"
        );
        assert!((0.0..=1.0).contains(&strict.chosen.certified_recall));
        // The best-certifiable fallback should be a high ef with a strong bound on
        // this tight synthetic corpus (sanity: ANN recovers most exact neighbours).
        assert!(
            strict.chosen.certified_recall > 0.3,
            "expected a meaningful certified recall, got {}",
            strict.chosen.certified_recall
        );
    }

    // ── Config validation edge cases ──────────────────────────────────

    #[test]
    fn validate_config_rejects_m_zero() {
        let config = HnswConfig {
            m: 0,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_m"),
            "expected InvalidConfig for hnsw_m, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_m_over_256() {
        let config = HnswConfig {
            m: 257,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_m"),
            "expected InvalidConfig for hnsw_m, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_ef_construction_zero() {
        let config = HnswConfig {
            ef_construction: 0,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_ef_construction"),
            "expected InvalidConfig for ef_construction, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_ef_search_zero() {
        let config = HnswConfig {
            ef_search: 0,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_ef_search"),
            "expected InvalidConfig for ef_search, got {error:?}"
        );
    }

    #[test]
    fn validate_config_rejects_max_layer_zero() {
        let config = HnswConfig {
            max_layer: 0,
            ..HnswConfig::default()
        };
        let error = validate_config(config).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "hnsw_max_layer"),
            "expected InvalidConfig for max_layer, got {error:?}"
        );
    }

    #[test]
    fn validate_config_accepts_m_256_boundary() {
        let config = HnswConfig {
            m: 256,
            ..HnswConfig::default()
        };
        assert!(validate_config(config).is_ok());
    }

    // ── build_from_parts error paths ────────────────────────────────────

    #[test]
    fn build_rejects_dimension_zero() {
        let error =
            HnswIndex::build_from_parts(vec![], vec![], 0, HnswConfig::default()).unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "dimension"),
            "expected InvalidConfig for dimension, got {error:?}"
        );
    }

    #[test]
    fn build_rejects_doc_id_vector_count_mismatch() {
        let error = HnswIndex::build_from_parts(
            vec!["a".to_owned(), "b".to_owned()],
            vec![vec![1.0, 0.0]],
            2,
            HnswConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "vectors"),
            "expected InvalidConfig for vectors, got {error:?}"
        );
    }

    #[test]
    fn build_rejects_vector_dimension_mismatch() {
        let error = HnswIndex::build_from_parts(
            vec!["a".to_owned()],
            vec![vec![1.0, 0.0, 0.0]], // 3D but declared 2D
            2,
            HnswConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::DimensionMismatch {
                    expected: 2,
                    found: 3
                }
            ),
            "expected DimensionMismatch, got {error:?}"
        );
    }

    #[test]
    fn build_rejects_nan_in_vector() {
        let error = HnswIndex::build_from_parts(
            vec!["a".to_owned()],
            vec![vec![1.0, f32::NAN]],
            2,
            HnswConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, ref reason, .. }
                     if field == "vector" && reason.contains("finite")),
            "expected InvalidConfig for non-finite vector, got {error:?}"
        );
    }

    #[test]
    fn build_rejects_infinity_in_vector() {
        let error = HnswIndex::build_from_parts(
            vec!["a".to_owned()],
            vec![vec![f32::INFINITY, 0.0]],
            2,
            HnswConfig::default(),
        )
        .unwrap_err();
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "vector"),
            "expected InvalidConfig for non-finite vector, got {error:?}"
        );
    }

    // ── vector_component_close edge cases ────────────────────────────────

    #[test]
    fn vector_component_close_rejects_infinity_vs_finite() {
        // Regression: Inf <= (EPSILON * 8 * Inf) was true, causing false positive.
        assert!(!vector_component_close(f32::INFINITY, 100.0));
        assert!(!vector_component_close(100.0, f32::INFINITY));
        assert!(!vector_component_close(f32::NEG_INFINITY, 0.0));
        assert!(!vector_component_close(0.0, f32::NEG_INFINITY));
    }

    #[test]
    fn vector_component_close_accepts_identical_infinities() {
        assert!(vector_component_close(f32::INFINITY, f32::INFINITY));
        assert!(vector_component_close(f32::NEG_INFINITY, f32::NEG_INFINITY));
    }

    #[test]
    fn vector_component_close_rejects_opposite_infinities() {
        assert!(!vector_component_close(f32::INFINITY, f32::NEG_INFINITY));
    }

    #[test]
    fn vector_component_close_nan_vs_finite_is_rejected() {
        assert!(!vector_component_close(f32::NAN, 0.0));
        assert!(!vector_component_close(0.0, f32::NAN));
    }

    #[test]
    fn vector_component_close_identical_nan_bits_accepted() {
        // Same NaN bit pattern passes the to_bits() fast path. This is fine:
        // NaN vectors are rejected at construction time by build_from_parts().
        assert!(vector_component_close(f32::NAN, f32::NAN));
    }

    #[test]
    fn vector_component_close_accepts_equal_values() {
        assert!(vector_component_close(0.0, 0.0));
        assert!(vector_component_close(1.0, 1.0));
        assert!(vector_component_close(-42.5, -42.5));
    }

    // ── knn_search_with_stats boundary conditions ───────────────────────

    #[test]
    fn search_with_k_zero_returns_empty_with_stats() {
        let path = temp_path("k0", "fsvi");
        let index = write_index(&path, &[normalized_vector(1, 16)]).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let (hits, stats) = ann
            .knn_search_with_stats(&normalized_vector(1, 16), 0, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert!(hits.is_empty());
        assert_eq!(stats.k_requested, 0);
        assert_eq!(stats.k_returned, 0);
        assert!(stats.is_approximate);
        assert!((stats.estimated_recall - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn search_dimension_mismatch_returns_error() {
        let path = temp_path("dimmis", "fsvi");
        let index = write_index(&path, &[normalized_vector(1, 16)]).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let error = ann
            .knn_search_with_stats(&normalized_vector(1, 8), 5, HNSW_DEFAULT_EF_SEARCH)
            .unwrap_err();
        assert!(
            matches!(
                error,
                SearchError::DimensionMismatch {
                    expected: 16,
                    found: 8
                }
            ),
            "expected DimensionMismatch, got {error:?}"
        );
    }

    #[test]
    fn search_stats_fields_are_populated() {
        let path = temp_path("stats", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..50).map(|i| normalized_vector(i, 32)).collect();
        let index = write_index(&path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let (hits, stats) = ann
            .knn_search_with_stats(&normalized_vector(999, 32), 5, 64)
            .expect("search");
        assert_eq!(stats.index_size, 50);
        assert_eq!(stats.dimension, 32);
        assert_eq!(stats.ef_search, 64);
        assert_eq!(stats.k_requested, 5);
        assert_eq!(stats.k_returned, hits.len());
        assert!(stats.is_approximate);
        assert!(stats.estimated_recall > 0.0);
        assert!(stats.estimated_recall <= 1.0);
    }

    #[test]
    fn search_k_larger_than_index_returns_all() {
        let path = temp_path("klarge", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..5).map(|i| normalized_vector(i, 16)).collect();
        let index = write_index(&path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let hits = ann
            .knn_search(&normalized_vector(999, 16), 100, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits.len(), 5);
    }

    // ── matches_vector_index edge cases ─────────────────────────────────

    #[test]
    fn matches_returns_false_for_dimension_mismatch() {
        let path_a = temp_path("match-a", "fsvi");
        let path_b = temp_path("match-b", "fsvi");
        let index_a = write_index(&path_a, &[normalized_vector(1, 16)]).expect("index_a");
        let index_b = write_index(&path_b, &[normalized_vector(1, 32)]).expect("index_b");
        let ann = HnswIndex::build_from_vector_index(&index_a, HnswConfig::default()).expect("ann");
        assert!(!ann.matches_vector_index(&index_b).expect("matches"));
    }

    #[test]
    fn matches_returns_false_for_record_count_mismatch() {
        let path_a = temp_path("match-rc-a", "fsvi");
        let path_b = temp_path("match-rc-b", "fsvi");
        let index_a = write_index(
            &path_a,
            &[normalized_vector(1, 16), normalized_vector(2, 16)],
        )
        .expect("index_a");
        let index_b = write_index(&path_b, &[normalized_vector(1, 16)]).expect("index_b");
        let ann = HnswIndex::build_from_vector_index(&index_a, HnswConfig::default()).expect("ann");
        assert!(!ann.matches_vector_index(&index_b).expect("matches"));
    }

    #[test]
    fn matches_returns_true_when_vectors_change_but_doc_ids_match() {
        // After the HNSW refactoring (vectors no longer stored in metadata),
        // matches_vector_index only checks doc_ids and dimension. When doc_ids
        // match, vectors are assumed correct because HNSW is rebuilt from the
        // VectorIndex's current vectors on load.
        let path_a = temp_path("match-vec-a", "fsvi");
        let path_b = temp_path("match-vec-b", "fsvi");
        let index_a = write_index(
            &path_a,
            &[normalized_vector(1, 16), normalized_vector(2, 16)],
        )
        .expect("index_a");
        let index_b = write_index(
            &path_b,
            &[normalized_vector(3, 16), normalized_vector(4, 16)],
        )
        .expect("index_b");
        let ann = HnswIndex::build_from_vector_index(&index_a, HnswConfig::default()).expect("ann");
        // Same doc_ids (doc-0000, doc-0001) + same dimension → matches
        assert!(ann.matches_vector_index(&index_b).expect("matches"));
    }

    #[test]
    fn build_from_vector_index_excludes_tombstoned_records() {
        let path = temp_path("tombstone-filter", "fsvi");
        let mut index = write_index(
            &path,
            &[
                normalized_vector(1, 16),
                normalized_vector(2, 16),
                normalized_vector(3, 16),
            ],
        )
        .expect("index");
        let deleted = index
            .soft_delete("doc-0001")
            .expect("soft_delete should succeed");
        assert!(deleted);

        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        assert_eq!(ann.len(), 2, "ANN should only index live vectors");

        let query = normalized_vector(2, 16);
        let hits = ann
            .knn_search(&query, 10, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert!(
            !hits.iter().any(|hit| hit.doc_id == "doc-0001"),
            "ANN should never return tombstoned doc IDs"
        );
    }

    // ── normalize_for_dist_dot ──────────────────────────────────────────

    #[test]
    fn normalize_zero_vector_unchanged() {
        let zero = vec![0.0_f32; 8];
        let budget = dist_dot_budget(zero.len()).expect("budget");
        let result = normalize_for_dist_dot(zero.clone(), budget);
        assert_eq!(
            result, zero,
            "zero vector should remain zero after normalize"
        );
    }

    #[test]
    fn dist_dot_roundoff_budget_is_dimension_aware_and_ranking_neutral() {
        let radius_16 = dist_dot_budget(16).expect("budget").radius_squared;
        let radius_384 = dist_dot_budget(384).expect("budget").radius_squared;
        let radius_4096 = dist_dot_budget(4096).expect("budget").radius_squared;

        assert!((0.5..1.0).contains(&radius_16));
        assert!(radius_384 < radius_16);
        assert!(radius_4096 < radius_384);
        assert!(
            radius_384 > 0.999,
            "384-dim safety margin must stay small enough to preserve cosine resolution"
        );

        let error = dist_dot_budget(DIST_DOT_MAX_DIMENSION + 1)
            .expect_err("ill-conditioned f32 bound must fail closed");
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "dimension")
        );
    }

    #[test]
    fn dist_dot_normalization_stays_below_one_across_reduction_orders() {
        fn reassociated_dot(left: &[f32], right: &[f32], lanes: usize) -> f32 {
            let mut partials = vec![0.0_f32; lanes];
            for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
                partials[index % lanes] += left * right;
            }
            partials.into_iter().fold(0.0_f32, |sum, value| sum + value)
        }

        for dimension in [16_usize, 384, 4_096] {
            let budget = dist_dot_budget(dimension).expect("budget");
            let vectors: Vec<Vec<f32>> = (0..16)
                .map(|seed| normalize_for_dist_dot(normalized_vector(seed, dimension), budget))
                .collect();
            for left in &vectors {
                for right in &vectors {
                    for lanes in [1_usize, 2, 4, 8, 16] {
                        let dot = reassociated_dot(left, right, lanes);
                        assert!(
                            dot <= 1.0,
                            "DistDot precondition violated: dim={dimension} lanes={lanes} dot={dot}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn safe_radius_scaling_preserves_valid_cosine_ranking() {
        fn dot(left: &[f32], right: &[f32]) -> f32 {
            left.iter().zip(right).map(|(a, b)| a * b).sum()
        }

        let query = vec![1.0_f32, 0.0, 0.0, 0.0];
        let candidates = [
            vec![1.0_f32, 0.0, 0.0, 0.0],
            vec![0.8_f32, 0.6, 0.0, 0.0],
            vec![0.6_f32, 0.8, 0.0, 0.0],
            vec![-0.2_f32, 0.0, 0.0, 0.98],
        ];
        let budget = dist_dot_budget(query.len()).expect("budget");
        let scaled_query = normalize_for_dist_dot(query.clone(), budget);

        let mut original: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                let candidate_norm = dot(candidate, candidate).sqrt();
                (index, dot(&query, candidate) / candidate_norm)
            })
            .collect();
        let mut restored: Vec<(usize, f32)> = candidates
            .into_iter()
            .enumerate()
            .map(|(index, candidate)| {
                let scaled = normalize_for_dist_dot(candidate, budget);
                (index, dot(&scaled_query, &scaled) / budget.radius_squared)
            })
            .collect();
        original.sort_by(|left, right| right.1.total_cmp(&left.1));
        restored.sort_by(|left, right| right.1.total_cmp(&left.1));

        assert_eq!(
            original.iter().map(|entry| entry.0).collect::<Vec<_>>(),
            restored.iter().map(|entry| entry.0).collect::<Vec<_>>(),
            "uniform safety scaling must not change valid cosine ranking"
        );
        for ((_, expected), (_, actual)) in original.iter().zip(&restored) {
            assert!((expected - actual).abs() <= budget.score_tolerance);
        }
    }

    #[test]
    fn hnsw_restores_cosine_score_after_safe_radius_scaling() {
        let path = temp_path("safe-radius-score", "fsvi");
        let vector = normalized_vector(41, 384);
        let index = write_index(&path, std::slice::from_ref(&vector)).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let hits = ann
            .knn_search(&vector, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert!(
            (hits[0].score - 1.0).abs() <= 1.0e-5,
            "uniform DistDot safety scaling must not leak into public cosine scores: {}",
            hits[0].score
        );
    }

    #[test]
    fn hnsw_rejects_non_finite_query_before_distance_evaluation() {
        let path = temp_path("non-finite-query", "fsvi");
        let vector = normalized_vector(42, 32);
        let index = write_index(&path, std::slice::from_ref(&vector)).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let mut query = vector;
        query[7] = f32::NAN;

        let error = ann
            .knn_search(&query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect_err("non-finite queries must return an error rather than panic in DistDot");
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, ref reason, .. }
                if field == "query" && reason.contains("finite")),
            "expected query InvalidConfig, got {error:?}"
        );
    }

    // ── estimate_recall ─────────────────────────────────────────────────

    #[test]
    fn estimate_recall_k_zero_returns_one() {
        assert!((estimate_recall(100, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn estimate_recall_clamped_between_zero_and_one() {
        // Very low ef relative to k
        let low = estimate_recall(1, 1000);
        assert!((0.0..=1.0).contains(&low), "low recall: {low}");

        // Very high ef relative to k
        let high = estimate_recall(10_000, 1);
        assert!((0.0..=1.0).contains(&high), "high recall: {high}");
    }

    #[test]
    fn estimate_recall_increases_with_ef() {
        let r_low = estimate_recall(10, 10);
        let r_high = estimate_recall(100, 10);
        assert!(
            r_high >= r_low,
            "recall should increase with ef: {r_low} vs {r_high}"
        );
    }

    // ── len / is_empty / dimension / config accessors ───────────────────

    #[test]
    fn accessors_report_correct_values() {
        let path = temp_path("accessors", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| normalized_vector(i, 24)).collect();
        let index = write_index(&path, &vectors).expect("index");
        let config = HnswConfig {
            m: 8,
            ..HnswConfig::default()
        };
        let ann = HnswIndex::build_from_vector_index(&index, config).expect("ann");
        assert_eq!(ann.len(), 10);
        assert!(!ann.is_empty());
        assert_eq!(ann.dimension(), 24);
        assert_eq!(ann.config().m, 8);
    }

    // ── Debug impl ──────────────────────────────────────────────────────

    #[test]
    fn debug_impl_does_not_panic() {
        let path = temp_path("debug", "fsvi");
        let index = write_index(&path, &[normalized_vector(1, 8)]).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");
        let debug_str = format!("{ann:?}");
        assert!(debug_str.contains("HnswIndex"));
        assert!(debug_str.contains("dimension: 8"));
    }

    // ── Original tests ──────────────────────────────────────────────────

    #[test]
    fn scores_are_consistent_with_exact_top_hit() {
        let fsvi_path = temp_path("score", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..128).map(|i| normalized_vector(i, 64)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let query = normalized_vector(7, 64);
        let exact = index.search_top_k(&query, 1, None).expect("exact");
        let approx = ann
            .knn_search(&query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("approx");

        assert_eq!(exact[0].doc_id, approx[0].doc_id);
        assert!((exact[0].score - approx[0].score).abs() < 1e-3);
    }

    #[test]
    fn persistence_round_trip() {
        let fsvi_path = temp_path("persist", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..64).map(|i| normalized_vector(i, 32)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let save_path = temp_path("persist", "hnsw");
        ann.save(&save_path).expect("save");

        let loaded = HnswIndex::load(&save_path, &index).expect("load");
        assert_eq!(loaded.len(), 64);
        assert_eq!(loaded.dimension(), 32);

        let query = normalized_vector(10, 32);
        let hits = loaded
            .knn_search(&query, 5, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits[0].doc_id, "doc-0010");
        assert!((hits[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn persistence_writes_native_graph_sidecars() {
        let fsvi_path = temp_path("persist_native", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..64).map(|i| normalized_vector(i, 32)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let save_path = temp_path("persist_native", "hnsw");
        ann.save(&save_path).expect("save");

        // Current-format save writes the native graph + data sidecars next to the metadata.
        let parent = save_path.parent().expect("parent");
        let basename = save_path.file_stem().unwrap().to_str().unwrap();
        assert!(
            parent.join(format!("{basename}.hnsw.graph")).is_file(),
            "native graph sidecar should exist"
        );
        assert!(
            parent.join(format!("{basename}.hnsw.data")).is_file(),
            "native data sidecar should exist"
        );

        let meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse meta");
        assert_eq!(meta.format_version, HNSW_META_FORMAT_CURRENT);

        // Load goes through the native graph path and still answers correctly.
        let loaded = HnswIndex::load(&save_path, &index).expect("native load");
        assert_eq!(loaded.len(), 64);
        let query = normalized_vector(7, 32);
        let hits = loaded
            .knn_search(&query, 3, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits[0].doc_id, "doc-0007");
    }

    #[test]
    fn load_rebuilds_from_legacy_v1_metadata() {
        let fsvi_path = temp_path("persist_v1", "fsvi");
        let vectors: Vec<Vec<f32>> = (0..64).map(|i| normalized_vector(i, 32)).collect();
        let index = write_index(&fsvi_path, &vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("ann");

        let save_path = temp_path("persist_v1", "hnsw");
        ann.save(&save_path).expect("save");

        // Fabricate a legacy v1 sidecar at a fresh path: identical metadata but
        // with `format_version` stripped (deserializes to 0) and no graph/data
        // sidecars beside it. load() must transparently rebuild from the
        // VectorIndex instead of failing.
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse");
        value
            .as_object_mut()
            .expect("object")
            .remove("format_version");
        let v1_path = temp_path("persist_v1_legacy", "hnsw");
        std::fs::write(&v1_path, serde_json::to_vec(&value).expect("v1 json")).expect("write v1");

        let loaded = HnswIndex::load(&v1_path, &index).expect("v1 rebuild load");
        assert_eq!(loaded.len(), 64);
        let query = normalized_vector(10, 32);
        let hits = loaded
            .knn_search(&query, 5, HNSW_DEFAULT_EF_SEARCH)
            .expect("search");
        assert_eq!(hits[0].doc_id, "doc-0010");
    }

    #[test]
    fn load_never_treats_v3_native_graph_as_v4() {
        // The source index says doc-0000 is e0 and doc-0001 is e1.
        let source_path = temp_path("persist_v3_source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");

        // Build a valid native graph with the same document IDs but the two
        // vectors swapped, then mark its metadata as v3 and disable the vector
        // fingerprint so the format boundary is the sole guard. If load accepts
        // v3 as current, an e0 query returns doc-0001. Correct v4-only loading
        // rebuilds from source and returns doc-0000.
        let swapped_path = temp_path("persist_v3_swapped", "fsvi");
        let swapped_index = write_index(&swapped_path, &[vec![0.0_f32, 1.0], vec![1.0_f32, 0.0]])
            .expect("swapped index");
        let swapped_ann = HnswIndex::build_from_vector_index(&swapped_index, HnswConfig::default())
            .expect("swapped ann");
        let save_path = temp_path("persist_v3_native", "hnsw");
        swapped_ann.save(&save_path).expect("save native graph");

        let mut meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse");
        meta.format_version = 3;
        meta.vector_fingerprint = 0;
        std::fs::write(&save_path, serde_json::to_vec(&meta).expect("serialize v3"))
            .expect("write v3 metadata");

        let loaded = HnswIndex::load(&save_path, &source_index).expect("rebuild v3");
        let hits = loaded
            .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search rebuilt graph");
        assert_eq!(
            hits[0].doc_id, "doc-0000",
            "v3 sampled-fingerprint graph must be rebuilt under v4's full fingerprint contract"
        );
    }

    #[test]
    fn load_rejects_current_sidecar_with_missing_vector_fingerprint() {
        let source_path = temp_path("persist_missing_fp_source", "fsvi");
        let source_index = write_index(&source_path, &[vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]])
            .expect("source index");

        // Persist a current-format native graph whose vectors are swapped
        // behind the same document IDs, then remove only its fingerprint field.
        // Treating the serde default of 0 as an opt-out would admit this stale
        // graph and make an e0 query return doc-0001.
        let swapped_path = temp_path("persist_missing_fp_swapped", "fsvi");
        let swapped_index = write_index(&swapped_path, &[vec![0.0_f32, 1.0], vec![1.0_f32, 0.0]])
            .expect("swapped index");
        let swapped_ann = HnswIndex::build_from_vector_index(&swapped_index, HnswConfig::default())
            .expect("swapped ann");
        let save_path = temp_path("persist_missing_fp_native", "hnsw");
        swapped_ann.save(&save_path).expect("save native graph");

        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse");
        value
            .as_object_mut()
            .expect("metadata object")
            .remove("vector_fingerprint");
        std::fs::write(
            &save_path,
            serde_json::to_vec(&value).expect("serialize metadata without fingerprint"),
        )
        .expect("write metadata without fingerprint");

        let loaded = HnswIndex::load(&save_path, &source_index)
            .expect("missing fingerprint must rebuild from source");
        let hits = loaded
            .knn_search(&[1.0, 0.0], 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search rebuilt graph");
        assert_eq!(
            hits[0].doc_id, "doc-0000",
            "current-format metadata cannot omit its vector fingerprint to admit a stale graph"
        );
    }

    #[test]
    fn load_rejects_native_sidecar_when_previously_unsampled_vector_changes() {
        // With 300 rows, the v3 fingerprint's ceil(300 / 256) stride was 2.
        // Row 1 was therefore neither an even-stride sample nor the final row.
        const VECTOR_COUNT: usize = 300;
        let source_path = temp_path("persist_unsampled_source", "fsvi");
        let original_vectors: Vec<Vec<f32>> = (0..VECTOR_COUNT)
            .map(|i| normalized_vector(i, 32))
            .collect();
        let source_index = write_index(&source_path, &original_vectors).expect("source index");
        let ann = HnswIndex::build_from_vector_index(&source_index, HnswConfig::default())
            .expect("build original graph");
        let save_path = temp_path("persist_unsampled_native", "hnsw");
        ann.save(&save_path).expect("save native graph");

        // Keep the document IDs, count, and dimension identical while changing
        // only the old scheme's unsampled row. Negating the original unit
        // vector makes doc-0001 the worst possible match in the stale graph and
        // the exact match after a rebuild, keeping the result proof decisive.
        let changed_vector: Vec<f32> = original_vectors[1].iter().map(|value| -*value).collect();
        let mut changed_vectors = original_vectors;
        changed_vectors[1].clone_from(&changed_vector);
        let changed_path = temp_path("persist_unsampled_changed", "fsvi");
        let changed_index = write_index(&changed_path, &changed_vectors).expect("changed index");

        let loaded = HnswIndex::load(&save_path, &changed_index)
            .expect("fingerprint mismatch must rebuild from changed source");
        let hits = loaded
            .knn_search(&changed_vector, 1, VECTOR_COUNT)
            .expect("search rebuilt graph");
        assert_eq!(
            hits[0].doc_id, "doc-0001",
            "every live vector must affect persisted source identity; otherwise the stale v3 \
             graph survives a change to row 1"
        );
    }

    /// Stale-vectors validation: the native-graph load path must reject a
    /// sidecar whose persisted vector fingerprint disagrees with the live
    /// `VectorIndex` (i.e. someone swapped the FSVI contents behind matching
    /// doc IDs), and transparently fall back to rebuild rather than silently
    /// serving hits against vectors that no longer exist. This is the exact
    /// scenario the prompt for `frankensearch#25` calls out.
    #[test]
    fn load_rejects_native_sidecar_when_vectors_swapped_under_same_doc_ids() {
        // Build + save a native sidecar over an FSVI where doc-i ≈ basis_i.
        let fsvi_path = temp_path("persist_swap", "fsvi");
        let original_vectors: Vec<Vec<f32>> = (0..16).map(|i| normalized_vector(i, 32)).collect();
        let original_index = write_index(&fsvi_path, &original_vectors).expect("index");
        let ann = HnswIndex::build_from_vector_index(&original_index, HnswConfig::default())
            .expect("build");

        let save_path = temp_path("persist_swap", "hnsw");
        ann.save(&save_path).expect("save native graph");

        // Sanity: a load against the *original* FSVI takes the fast path and
        // returns doc-0007 for query≈doc-0007.
        let loaded_original =
            HnswIndex::load(&save_path, &original_index).expect("native load matches");
        let query = normalized_vector(7, 32);
        let original_hits = loaded_original
            .knn_search(&query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search original");
        assert_eq!(original_hits[0].doc_id, "doc-0007");
        // Sanity that the fingerprint actually got stamped.
        let meta: HnswMeta =
            serde_json::from_slice(&std::fs::read(&save_path).expect("meta")).expect("parse meta");
        assert_ne!(
            meta.vector_fingerprint, 0,
            "native save must stamp a fingerprint"
        );

        // Now swap vectors behind the same doc IDs (doc-0007 now points at
        // basis_99) while leaving the .hnsw.graph / .hnsw.data sidecars
        // unchanged. A naive native fast path would return doc-0007 against the
        // *old* graph; the fingerprint guard forces a rebuild instead.
        let mut swapped_vectors = original_vectors.clone();
        swapped_vectors[7] = normalized_vector(99, 32);
        let swapped_path = temp_path("persist_swap_after", "fsvi");
        let swapped_index = write_index(&swapped_path, &swapped_vectors).expect("swapped");

        // Copy the metadata + graph + data sidecars next to the swapped FSVI
        // so the load path's directory layout is plausible.
        let swap_save_path = temp_path("persist_swap_after", "hnsw");
        std::fs::copy(&save_path, &swap_save_path).expect("copy meta");
        let src_parent = save_path.parent().expect("src parent");
        let dst_parent = swap_save_path.parent().expect("dst parent");
        let src_stem = save_path.file_stem().unwrap().to_str().unwrap();
        let dst_stem = swap_save_path.file_stem().unwrap().to_str().unwrap();
        for ext in ["hnsw.graph", "hnsw.data"] {
            std::fs::copy(
                src_parent.join(format!("{src_stem}.{ext}")),
                dst_parent.join(format!("{dst_stem}.{ext}")),
            )
            .expect("copy sidecar");
        }

        // Load against the swapped FSVI. The fingerprint mismatch must trigger
        // the rebuild fallback, which sees doc-0007 ≈ basis_99 and therefore
        // returns doc-0007 *only* when querying ≈ basis_99 — not when querying
        // basis_7.
        let loaded_swapped =
            HnswIndex::load(&swap_save_path, &swapped_index).expect("load after swap");
        let stale_query = normalized_vector(7, 32);
        let stale_hits = loaded_swapped
            .knn_search(&stale_query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search stale");
        // The persisted graph would have returned doc-0007 (basis_7) here; the
        // rebuilt graph against the swapped FSVI returns whichever doc now
        // *actually* sits near basis_7, which is **not** doc-0007.
        assert_ne!(
            stale_hits[0].doc_id, "doc-0007",
            "fingerprint guard must reject the persisted graph and rebuild against \
             the swapped FSVI; otherwise we'd return stale ANN hits"
        );

        // And the rebuilt graph *can* find the swapped doc by its new vector.
        let new_query = normalized_vector(99, 32);
        let new_hits = loaded_swapped
            .knn_search(&new_query, 1, HNSW_DEFAULT_EF_SEARCH)
            .expect("search new");
        assert_eq!(
            new_hits[0].doc_id, "doc-0007",
            "rebuild path must have picked up the swapped vector for doc-0007"
        );
    }
}
