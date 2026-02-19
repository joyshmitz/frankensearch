//! FastEmbed-based cross-encoder reranker (ms-marco-MiniLM-L-6-v2).
//!
//! Loads a local ONNX model + tokenizer bundle via the `fastembed` crate and
//! produces relevance scores for query-document pairs. This implementation never
//! downloads model assets; it expects the model files to be present on disk and
//! returns a clear error when they are missing.
//!
//! Requires the `fastembed-reranker` feature flag.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use fastembed::{RerankInitOptionsUserDefined, TextRerank, UserDefinedRerankingModel};
use frankensearch_core::{RerankDocument, RerankScore, SearchError, SearchResult, SyncRerank};

const MODEL_ID: &str = "ms-marco-minilm-l6-v2";
const MODEL_DIR_NAME: &str = "ms-marco-MiniLM-L-6-v2";
const RERANKER_ID: &str = "ms-marco-minilm-l6-v2";

const MODEL_FILE: &str = "model.onnx";
const TOKENIZER_JSON: &str = "tokenizer.json";
const CONFIG_JSON: &str = "config.json";
const SPECIAL_TOKENS_JSON: &str = "special_tokens_map.json";
const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";

/// FastEmbed-backed cross-encoder reranker using ms-marco-MiniLM-L-6-v2.
///
/// This reranker uses a cross-encoder model that processes query-document pairs
/// simultaneously, providing more accurate relevance scores than bi-encoder similarity.
/// The MiniLM-L-6-v2 model is optimized for fast inference on CPU.
pub struct FastEmbedReranker {
    model: Mutex<TextRerank>,
    id: String,
    model_id: String,
}

impl FastEmbedReranker {
    /// Stable reranker identifier for ms-marco-MiniLM-L-6-v2.
    pub fn reranker_id_static() -> &'static str {
        RERANKER_ID
    }

    /// Stable model identifier for ms-marco-MiniLM-L-6-v2.
    pub fn model_id_static() -> &'static str {
        MODEL_ID
    }

    /// Required model files for the reranker (must all exist locally).
    pub fn required_model_files() -> &'static [&'static str] {
        &[
            MODEL_FILE,
            TOKENIZER_JSON,
            CONFIG_JSON,
            SPECIAL_TOKENS_JSON,
            TOKENIZER_CONFIG_JSON,
        ]
    }

    /// Default model directory relative to a data directory.
    pub fn default_model_dir(data_dir: &Path) -> PathBuf {
        data_dir.join("models").join(MODEL_DIR_NAME)
    }

    /// Load the reranker model + tokenizer from a local directory.
    ///
    /// This never downloads; it returns an error if any required file is missing.
    pub fn load_from_dir(model_dir: &Path) -> SearchResult<Self> {
        if !model_dir.is_dir() {
            return Err(SearchError::RerankFailed {
                model: RERANKER_ID.to_string(),
                source: format!(
                    "reranker model directory not found: {}",
                    model_dir.display()
                )
                .into(),
            });
        }

        let required = Self::required_model_files();
        let mut missing = Vec::new();
        for name in required {
            let path = model_dir.join(name);
            if !path.is_file() {
                missing.push(*name);
            }
        }
        if !missing.is_empty() {
            return Err(SearchError::RerankFailed {
                model: RERANKER_ID.to_string(),
                source: format!(
                    "reranker model files missing in {}: {}",
                    model_dir.display(),
                    missing.join(", ")
                )
                .into(),
            });
        }

        let model_file = Self::read_required(model_dir.join(MODEL_FILE), MODEL_FILE)?;
        let tokenizer_file = Self::read_required(model_dir.join(TOKENIZER_JSON), TOKENIZER_JSON)?;
        let config_file = Self::read_required(model_dir.join(CONFIG_JSON), CONFIG_JSON)?;
        let special_tokens_map_file =
            Self::read_required(model_dir.join(SPECIAL_TOKENS_JSON), SPECIAL_TOKENS_JSON)?;
        let tokenizer_config_file =
            Self::read_required(model_dir.join(TOKENIZER_CONFIG_JSON), TOKENIZER_CONFIG_JSON)?;

        let tokenizer_files = fastembed::TokenizerFiles {
            tokenizer_file,
            config_file,
            special_tokens_map_file,
            tokenizer_config_file,
        };

        let model = UserDefinedRerankingModel::new(model_file, tokenizer_files);
        let init_options = RerankInitOptionsUserDefined::default();

        let model = TextRerank::try_new_from_user_defined(model, init_options).map_err(|e| {
            SearchError::RerankFailed {
                model: RERANKER_ID.to_string(),
                source: format!("fastembed reranker init failed: {e}").into(),
            }
        })?;

        Ok(Self {
            model: Mutex::new(model),
            id: RERANKER_ID.to_string(),
            model_id: MODEL_ID.to_string(),
        })
    }

    /// Stable model identifier for compatibility checks.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    fn read_required(path: PathBuf, label: &str) -> SearchResult<Vec<u8>> {
        fs::read(&path).map_err(|e| SearchError::RerankFailed {
            model: RERANKER_ID.to_string(),
            source: format!("unable to read {label} at {}: {e}", path.display()).into(),
        })
    }
}

impl SyncRerank for FastEmbedReranker {
    fn rerank_sync(
        &self,
        query: &str,
        documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>> {
        if query.is_empty() {
            return Err(SearchError::RerankFailed {
                model: self.id.clone(),
                source: "empty query".into(),
            });
        }
        if documents.is_empty() {
            return Err(SearchError::RerankFailed {
                model: self.id.clone(),
                source: "empty documents list".into(),
            });
        }
        for (i, doc) in documents.iter().enumerate() {
            if doc.text.is_empty() {
                return Err(SearchError::RerankFailed {
                    model: self.id.clone(),
                    source: format!("empty document at index {i}").into(),
                });
            }
        }

        #[allow(unused_mut)]
        let mut model = self.model.lock().map_err(|_| SearchError::RerankFailed {
            model: self.id.clone(),
            source: "fastembed reranker lock poisoned".into(),
        })?;

        // Convert to Vec<String> for fastembed API
        let doc_strings: Vec<String> = documents.iter().map(|d| d.text.clone()).collect();

        // FastEmbed's rerank returns Vec<RerankResult> with index and score
        let rerank_results = model
            .rerank(query.to_string(), doc_strings, false, None)
            .map_err(|e| SearchError::RerankFailed {
                model: self.id.clone(),
                source: format!("fastembed rerank failed: {e}").into(),
            })?;

        // Convert to RerankScore in original document order
        let mut scores: Vec<RerankScore> = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| RerankScore {
                doc_id: doc.doc_id.clone(),
                score: 0.0,
                original_rank: i,
            })
            .collect();

        for result in rerank_results {
            if result.index < scores.len() {
                scores[result.index].score = result.score;
            }
        }

        Ok(scores)
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn is_available(&self) -> bool {
        true // If we got this far, the model is loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reranker_missing_files_returns_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let err = match FastEmbedReranker::load_from_dir(tmp.path()) {
            Ok(_) => panic!("expected missing-model error"),
            Err(err) => err,
        };
        match err {
            SearchError::RerankFailed { model, source } => {
                assert_eq!(model, RERANKER_ID);
                assert!(source.to_string().contains("model files missing"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn reranker_required_files() {
        let files = FastEmbedReranker::required_model_files();
        assert!(files.contains(&"model.onnx"));
        assert!(files.contains(&"tokenizer.json"));
        assert!(files.contains(&"config.json"));
    }
}
