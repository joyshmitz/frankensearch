//! Embedder auto-detection and fallback stack assembly.

#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use std::collections::BTreeSet;
use std::fmt;
#[cfg(not(any(feature = "model2vec", feature = "fastembed")))]
use std::path::Path;
#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;

use asupersync::Cx;
#[cfg(not(any(feature = "model2vec", feature = "fastembed")))]
use tracing::info;
#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use tracing::{info, warn};

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{Embedder, SearchFuture};

#[cfg(feature = "fastembed")]
use crate::fastembed_embedder::{
    DEFAULT_HF_ID as MINILM_HF_ID, DEFAULT_MODEL_NAME as MINILM_MODEL_NAME, FastEmbedEmbedder,
    find_model_dir_with_hf_id as find_fastembed_model_dir,
};
#[cfg(feature = "hash")]
use crate::hash_embedder::HashEmbedder;
#[cfg(any(feature = "model2vec", feature = "fastembed"))]
use crate::model_manifest::ModelManifest;
#[cfg(feature = "model2vec")]
use crate::model2vec_embedder::{
    Model2VecEmbedder, find_model_dir_with_hf_id as find_model2vec_model_dir,
};

#[cfg(feature = "model2vec")]
const POTION_MODEL_NAME: &str = "potion-multilingual-128M";
#[cfg(feature = "model2vec")]
const POTION_HF_ID: &str = "minishlab/potion-multilingual-128M";

/// Availability classification for two-tier search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoTierAvailability {
    /// Fast + quality embedders available.
    Full,
    /// Semantic fast embedder available, no quality tier.
    FastOnly,
    /// Hash-only fallback path.
    HashOnly,
}

/// Resolved fast/quality embedder stack for progressive search.
#[derive(Clone)]
pub struct EmbedderStack {
    fast: Arc<dyn Embedder>,
    quality: Option<Arc<dyn Embedder>>,
    availability: TwoTierAvailability,
}

impl fmt::Debug for EmbedderStack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbedderStack")
            .field("availability", &self.availability)
            .field("fast_id", &self.fast.id())
            .field("fast_dim", &self.fast.dimension())
            .field(
                "quality_id",
                &self.quality.as_ref().map(|embedder| embedder.id()),
            )
            .finish()
    }
}

impl EmbedderStack {
    /// Build from explicit parts.
    #[must_use]
    pub fn from_parts(fast: Arc<dyn Embedder>, quality: Option<Arc<dyn Embedder>>) -> Self {
        let availability = if quality.is_some() {
            TwoTierAvailability::Full
        } else if fast.is_semantic() {
            TwoTierAvailability::FastOnly
        } else {
            TwoTierAvailability::HashOnly
        };
        Self {
            fast,
            quality,
            availability,
        }
    }

    /// Auto-detect best available embedders from default search paths.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::EmbedderUnavailable` when no usable fast embedder is available.
    pub fn auto_detect() -> SearchResult<Self> {
        Self::auto_detect_with(None)
    }

    /// Auto-detect embedders with an optional explicit model root override.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::EmbedderUnavailable` when no usable fast embedder is available.
    pub fn auto_detect_with(model_root: Option<&Path>) -> SearchResult<Self> {
        let quality = detect_quality_embedder(model_root);
        let fast = detect_fast_embedder(model_root)
            .or_else(hash_fallback_embedder)
            .ok_or_else(|| SearchError::EmbedderUnavailable {
                model: "fast-tier".to_owned(),
                reason: "no model2vec/hash embedder available in this build".to_owned(),
            })?;

        let stack = Self::from_parts(fast, quality);
        info!(
            availability = ?stack.availability,
            fast = stack.fast.id(),
            quality = stack.quality.as_ref().map(|embedder| embedder.id()),
            "embedder stack ready"
        );
        Ok(stack)
    }

    /// Apply MRL-style dimensionality reduction where supported.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when `target_dim` is zero.
    pub fn with_mrl_target_dim(mut self, target_dim: usize) -> SearchResult<Self> {
        if target_dim == 0 {
            return Err(SearchError::InvalidConfig {
                field: "target_dim".to_owned(),
                value: "0".to_owned(),
                reason: "target dimension must be at least 1".to_owned(),
            });
        }

        self.fast = maybe_wrap_mrl(self.fast.clone(), target_dim)?;
        self.quality = self
            .quality
            .clone()
            .map(|embedder| maybe_wrap_mrl(embedder, target_dim))
            .transpose()?;
        self.availability = if self.quality.is_some() {
            TwoTierAvailability::Full
        } else if self.fast.is_semantic() {
            TwoTierAvailability::FastOnly
        } else {
            TwoTierAvailability::HashOnly
        };
        Ok(self)
    }

    /// Fast embedder reference.
    #[must_use]
    pub fn fast(&self) -> &dyn Embedder {
        self.fast.as_ref()
    }

    /// Fast embedder alias for API compatibility.
    #[must_use]
    pub fn fast_embedder(&self) -> &dyn Embedder {
        self.fast()
    }

    /// Cloned fast embedder handle.
    #[must_use]
    pub fn fast_arc(&self) -> Arc<dyn Embedder> {
        self.fast.clone()
    }

    /// Optional quality embedder reference.
    #[must_use]
    pub fn quality(&self) -> Option<&dyn Embedder> {
        self.quality.as_deref()
    }

    /// Optional quality embedder alias for API compatibility.
    #[must_use]
    pub fn quality_embedder(&self) -> Option<&dyn Embedder> {
        self.quality()
    }

    /// Cloned quality embedder handle.
    #[must_use]
    pub fn quality_arc(&self) -> Option<Arc<dyn Embedder>> {
        self.quality.clone()
    }

    /// Availability state.
    #[must_use]
    pub const fn availability(&self) -> TwoTierAvailability {
        self.availability
    }
}

/// MRL dimension reduction wrapper.
pub struct DimReduceEmbedder {
    inner: Arc<dyn Embedder>,
    target_dim: usize,
    id: String,
    model_name: String,
}

impl fmt::Debug for DimReduceEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DimReduceEmbedder")
            .field("inner", &self.inner.id())
            .field("target_dim", &self.target_dim)
            .finish_non_exhaustive()
    }
}

impl DimReduceEmbedder {
    /// Create a dimension-reduced view of an embedder.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when requested dimension is invalid.
    pub fn new(inner: Arc<dyn Embedder>, target_dim: usize) -> SearchResult<Self> {
        if target_dim == 0 {
            return Err(SearchError::InvalidConfig {
                field: "target_dim".to_owned(),
                value: "0".to_owned(),
                reason: "target dimension must be at least 1".to_owned(),
            });
        }
        if target_dim > inner.dimension() {
            return Err(SearchError::InvalidConfig {
                field: "target_dim".to_owned(),
                value: target_dim.to_string(),
                reason: format!(
                    "target dimension cannot exceed embedder dimension {}",
                    inner.dimension()
                ),
            });
        }
        if !inner.supports_mrl() {
            return Err(SearchError::InvalidConfig {
                field: "embedder.supports_mrl".to_owned(),
                value: inner.id().to_owned(),
                reason: "embedder does not support MRL truncation".to_owned(),
            });
        }

        Ok(Self {
            id: format!("{}-mrl-{target_dim}", inner.id()),
            model_name: format!("{} (MRL {target_dim})", inner.model_name()),
            inner,
            target_dim,
        })
    }
}

impl Embedder for DimReduceEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let full = self.inner.embed(cx, text).await?;
            self.inner.truncate_embedding(&full, self.target_dim)
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            let full_batch = self.inner.embed_batch(cx, texts).await?;
            full_batch
                .iter()
                .map(|embedding| self.inner.truncate_embedding(embedding, self.target_dim))
                .collect()
        })
    }

    fn dimension(&self) -> usize {
        self.target_dim
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    fn is_semantic(&self) -> bool {
        self.inner.is_semantic()
    }

    fn category(&self) -> frankensearch_core::traits::ModelCategory {
        self.inner.category()
    }

    fn tier(&self) -> frankensearch_core::traits::ModelTier {
        self.inner.tier()
    }

    fn supports_mrl(&self) -> bool {
        true
    }
}

fn maybe_wrap_mrl(
    embedder: Arc<dyn Embedder>,
    target_dim: usize,
) -> SearchResult<Arc<dyn Embedder>> {
    if target_dim >= embedder.dimension() || !embedder.supports_mrl() {
        return Ok(embedder);
    }
    Ok(Arc::new(DimReduceEmbedder::new(embedder, target_dim)?))
}

#[cfg(feature = "model2vec")]
fn detect_fast_embedder(model_root: Option<&Path>) -> Option<Arc<dyn Embedder>> {
    let manifest = ModelManifest::potion_128m();
    let discovered = find_model2vec_model_dir(POTION_MODEL_NAME, POTION_HF_ID);
    let candidates = candidate_directories(model_root, POTION_MODEL_NAME, discovered.as_deref());
    let checked_paths: Vec<String> = candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect();

    for candidate in candidates {
        if !manifest_files_exist(&manifest, &candidate) {
            continue;
        }
        if manifest.has_verified_checksums()
            && let Err(error) = manifest.verify_dir(&candidate)
        {
            warn!(
                model = POTION_MODEL_NAME,
                path = %candidate.display(),
                error = %error,
                "model2vec manifest verification failed"
            );
            continue;
        }

        match Model2VecEmbedder::load_with_name(&candidate, POTION_MODEL_NAME) {
            Ok(embedder) => {
                info!(
                    model = POTION_MODEL_NAME,
                    tier = "fast",
                    path = %candidate.display(),
                    dimension = embedder.dimension(),
                    "embedder detected"
                );
                return Some(Arc::new(embedder));
            }
            Err(error) => {
                warn!(
                    model = POTION_MODEL_NAME,
                    tier = "fast",
                    path = %candidate.display(),
                    error = %error,
                    "embedder unavailable"
                );
            }
        }
    }

    warn!(
        model = POTION_MODEL_NAME,
        tier = "fast",
        checked_paths = ?checked_paths,
        "embedder unavailable"
    );
    None
}

#[cfg(not(feature = "model2vec"))]
fn detect_fast_embedder(_model_root: Option<&Path>) -> Option<Arc<dyn Embedder>> {
    None
}

#[cfg(feature = "fastembed")]
fn detect_quality_embedder(model_root: Option<&Path>) -> Option<Arc<dyn Embedder>> {
    let manifest = ModelManifest::minilm_v2();
    let discovered = find_fastembed_model_dir(MINILM_MODEL_NAME, MINILM_HF_ID);
    let candidates = candidate_directories(model_root, MINILM_MODEL_NAME, discovered.as_deref());
    let checked_paths: Vec<String> = candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect();

    for candidate in candidates {
        if !manifest_files_exist(&manifest, &candidate) {
            continue;
        }
        if manifest.has_verified_checksums() {
            if let Err(error) = manifest.verify_dir(&candidate) {
                warn!(
                    model = MINILM_MODEL_NAME,
                    path = %candidate.display(),
                    error = %error,
                    "quality manifest verification failed"
                );
                continue;
            }
        }

        match FastEmbedEmbedder::load_with_name(&candidate, MINILM_MODEL_NAME) {
            Ok(embedder) => {
                info!(
                    model = MINILM_MODEL_NAME,
                    tier = "quality",
                    path = %candidate.display(),
                    dimension = embedder.dimension(),
                    "embedder detected"
                );
                return Some(Arc::new(embedder));
            }
            Err(error) => {
                warn!(
                    model = MINILM_MODEL_NAME,
                    tier = "quality",
                    path = %candidate.display(),
                    error = %error,
                    "embedder unavailable"
                );
            }
        }
    }

    warn!(
        model = MINILM_MODEL_NAME,
        tier = "quality",
        checked_paths = ?checked_paths,
        "embedder unavailable"
    );
    None
}

#[cfg(not(feature = "fastembed"))]
fn detect_quality_embedder(_model_root: Option<&Path>) -> Option<Arc<dyn Embedder>> {
    None
}

#[cfg(feature = "hash")]
#[allow(clippy::unnecessary_wraps)]
fn hash_fallback_embedder() -> Option<Arc<dyn Embedder>> {
    Some(Arc::new(HashEmbedder::default_256()))
}

#[cfg(not(feature = "hash"))]
fn hash_fallback_embedder() -> Option<Arc<dyn Embedder>> {
    None
}

#[cfg(any(feature = "model2vec", feature = "fastembed"))]
fn manifest_files_exist(manifest: &ModelManifest, model_dir: &Path) -> bool {
    manifest
        .files
        .iter()
        .all(|file| model_dir.join(&file.name).is_file())
}

#[cfg(any(feature = "model2vec", feature = "fastembed"))]
fn candidate_directories(
    model_root: Option<&Path>,
    model_name: &str,
    discovered: Option<&Path>,
) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(root) = model_root {
        paths.push(root.join(model_name));
        paths.push(root.to_path_buf());
    }
    if let Some(path) = discovered {
        paths.push(path.to_path_buf());
    }

    let mut seen = BTreeSet::new();
    paths
        .into_iter()
        .filter(|path| seen.insert(path.display().to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "model2vec", feature = "hash"))]
    use std::fs;

    use super::*;
    use frankensearch_core::traits::ModelCategory;

    #[cfg(feature = "hash")]
    #[test]
    fn auto_detect_hash_only_when_no_models_present() {
        let temp = tempfile::tempdir().unwrap();
        let stack = EmbedderStack::auto_detect_with(Some(temp.path())).unwrap();
        assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
        assert_eq!(stack.fast().category(), ModelCategory::HashEmbedder);
        assert!(stack.quality().is_none());
    }

    #[cfg(all(feature = "model2vec", feature = "hash"))]
    #[test]
    fn auto_detect_fast_only_when_model2vec_is_available() {
        let temp = tempfile::tempdir().unwrap();
        let model_dir = temp.path().join(POTION_MODEL_NAME);
        fs::create_dir_all(&model_dir).unwrap();
        create_test_model2vec_layout(&model_dir, 16, 8);

        let stack = EmbedderStack::auto_detect_with(Some(temp.path())).unwrap();
        assert_eq!(stack.availability(), TwoTierAvailability::FastOnly);
        assert_eq!(stack.fast().id(), POTION_MODEL_NAME);
    }

    #[cfg(all(feature = "model2vec", feature = "hash"))]
    #[test]
    fn corrupted_model2vec_falls_back_to_hash() {
        let temp = tempfile::tempdir().unwrap();
        let model_dir = temp.path().join(POTION_MODEL_NAME);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
        fs::write(model_dir.join("model.safetensors"), b"not-safetensors").unwrap();

        let stack = EmbedderStack::auto_detect_with(Some(temp.path())).unwrap();
        assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
        assert_eq!(stack.fast().category(), ModelCategory::HashEmbedder);
    }

    #[cfg(all(feature = "model2vec", feature = "hash"))]
    fn create_test_model2vec_layout(dir: &Path, vocab_size: usize, dimensions: usize) {
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [{
                "id": 0,
                "content": "[UNK]",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }],
            "normalizer": { "type": "Lowercase" },
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": create_test_vocab(vocab_size),
                "unk_token": "[UNK]"
            }
        });
        fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_string_pretty(&tokenizer_json).unwrap(),
        )
        .unwrap();
        create_test_safetensors(dir, vocab_size, dimensions);
    }

    #[cfg(all(feature = "model2vec", feature = "hash"))]
    fn create_test_vocab(vocab_size: usize) -> serde_json::Value {
        let mut vocab = serde_json::Map::new();
        vocab.insert("[UNK]".to_owned(), serde_json::Value::from(0));
        for idx in 1..vocab_size {
            vocab.insert(format!("token{idx}"), serde_json::Value::from(idx));
        }
        serde_json::Value::Object(vocab)
    }

    #[cfg(feature = "hash")]
    #[test]
    fn from_parts_hash_only_availability() {
        let hash = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let stack = EmbedderStack::from_parts(hash, None);
        assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
        assert!(stack.quality().is_none());
        assert!(stack.quality_arc().is_none());
        assert_eq!(stack.fast().category(), ModelCategory::HashEmbedder);
        assert_eq!(stack.fast_embedder().id(), stack.fast().id());
    }

    #[cfg(feature = "hash")]
    #[test]
    fn from_parts_with_quality_is_full() {
        let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let quality: Arc<dyn Embedder> =
            Arc::new(crate::hash_embedder::HashEmbedder::default_384());
        let stack = EmbedderStack::from_parts(fast, Some(quality));
        assert_eq!(stack.availability(), TwoTierAvailability::Full);
        assert!(stack.quality().is_some());
        assert!(stack.quality_arc().is_some());
        assert_eq!(
            stack.quality_embedder().unwrap().id(),
            stack.quality().unwrap().id()
        );
    }

    #[cfg(feature = "hash")]
    #[test]
    fn dim_reduce_rejects_zero_target_dim() {
        let inner: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let err = DimReduceEmbedder::new(inner, 0).expect_err("should reject target_dim=0");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn dim_reduce_rejects_target_exceeding_inner_dim() {
        let inner: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let err =
            DimReduceEmbedder::new(inner, 512).expect_err("should reject target_dim > inner dim");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn dim_reduce_rejects_non_mrl_embedder() {
        let inner: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        assert!(!inner.supports_mrl());
        let err = DimReduceEmbedder::new(inner, 64).expect_err("should reject non-MRL embedder");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn with_mrl_target_dim_zero_is_rejected() {
        let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let stack = EmbedderStack::from_parts(fast, None);
        let err = stack
            .with_mrl_target_dim(0)
            .expect_err("should reject target_dim=0");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[cfg(feature = "hash")]
    #[test]
    fn with_mrl_passthrough_when_non_mrl() {
        let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let stack = EmbedderStack::from_parts(fast, None);
        // hash embedder doesn't support MRL, so with_mrl_target_dim should
        // pass through without wrapping (target_dim < dimension but !supports_mrl)
        let stack = stack.with_mrl_target_dim(64).unwrap();
        assert_eq!(stack.availability(), TwoTierAvailability::HashOnly);
        // dimension should be unchanged since wrapping was skipped
        assert_eq!(stack.fast().dimension(), 256);
    }

    #[cfg(feature = "hash")]
    #[test]
    fn embedder_stack_debug_format() {
        let fast: Arc<dyn Embedder> = Arc::new(crate::hash_embedder::HashEmbedder::default_256());
        let stack = EmbedderStack::from_parts(fast, None);
        let debug = format!("{stack:?}");
        assert!(debug.contains("EmbedderStack"));
        assert!(debug.contains("HashOnly"));
    }

    #[cfg(all(feature = "model2vec", feature = "hash"))]
    fn create_test_safetensors(dir: &Path, vocab_size: usize, dimensions: usize) {
        use std::collections::HashMap;

        let mut data = Vec::with_capacity(vocab_size * dimensions * 4);
        for row in 0..vocab_size {
            for col in 0..dimensions {
                #[allow(clippy::cast_precision_loss)]
                let value = (row as f32).mul_add(0.01, (col as f32) * 0.001);
                data.extend_from_slice(&value.to_le_bytes());
            }
        }

        let mut tensors = HashMap::new();
        tensors.insert(
            "embeddings".to_owned(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![vocab_size, dimensions],
                &data,
            )
            .unwrap(),
        );
        let encoded = safetensors::tensor::serialize(&tensors, &None).unwrap();
        fs::write(dir.join("model.safetensors"), encoded).unwrap();
    }
}
