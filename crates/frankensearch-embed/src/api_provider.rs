//! Cloud API embedding provider trait and implementations (OpenAI, Gemini).
//!
//! Each provider knows its endpoint, auth, JSON format, and batch limits.
//! The shared [`super::api_embedder::ApiEmbedder`] handles HTTP, retry, and
//! rate-limiting generically over any `ApiProvider`.
//!
//! Gated behind the `api` feature flag.

use std::fmt;

use frankensearch_core::error::{SearchError, SearchResult};

// ─── ApiProvider trait ──────────────────────────────────────────────────────

/// Abstraction over cloud embedding API differences.
///
/// Implementors encode provider-specific details (URL, auth scheme, JSON
/// schema, batch limits) so that `ApiEmbedder` can drive any provider
/// uniformly.
pub trait ApiProvider: Send + Sync + fmt::Debug {
    /// Human-readable provider name (e.g. `"openai"`, `"gemini"`).
    fn provider_name(&self) -> &str;

    /// Model ID sent to the API (e.g. `"text-embedding-3-small"`).
    fn api_model_id(&self) -> &str;

    /// Stable embedder ID stored in FSVI index headers.
    fn embedder_id(&self) -> &str;

    /// Output embedding dimensionality.
    fn dimension(&self) -> usize;

    /// Maximum texts per single API call.
    fn max_batch_size(&self) -> usize;

    /// Whether this model supports Matryoshka Representation Learning.
    fn supports_mrl(&self) -> bool;

    /// Base endpoint URL for the embedding request.
    fn endpoint_url(&self) -> &str;

    /// Full request URL (may include query parameters like API keys).
    /// Defaults to `endpoint_url()`.
    fn request_url(&self) -> String {
        self.endpoint_url().to_owned()
    }

    /// HTTP headers (excluding content-type which is always application/json).
    fn request_headers(&self) -> Vec<(String, String)>;

    /// Serialize a batch of texts into the provider's JSON request body.
    fn serialize_request(&self, texts: &[&str]) -> SearchResult<Vec<u8>>;

    /// Deserialize the provider's JSON response into embedding vectors.
    /// The returned vectors MUST be in the same order as the input texts.
    fn deserialize_response(&self, body: &[u8]) -> SearchResult<Vec<Vec<f32>>>;
}

// ─── OpenAI ─────────────────────────────────────────────────────────────────

/// OpenAI embeddings API provider (`text-embedding-3-small`, `text-embedding-3-large`).
#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    api_key: String,
    model: String,
    dimension: usize,
    endpoint: String,
    embedder_id: String,
}

impl OpenAiProvider {
    /// Create an OpenAI provider for `text-embedding-3-small`.
    ///
    /// Default dimension is 1536; pass a smaller value for MRL truncation.
    #[must_use]
    pub fn text_embedding_3_small(api_key: impl Into<String>, dimension: Option<usize>) -> Self {
        let dim = dimension.unwrap_or(1536);
        Self {
            api_key: api_key.into(),
            model: "text-embedding-3-small".to_owned(),
            dimension: dim,
            endpoint: "https://api.openai.com/v1/embeddings".to_owned(),
            embedder_id: format!("openai-text-embedding-3-small-{dim}d"),
        }
    }

    /// Create an OpenAI provider for `text-embedding-3-large`.
    ///
    /// Default dimension is 3072; pass a smaller value for MRL truncation.
    #[must_use]
    pub fn text_embedding_3_large(api_key: impl Into<String>, dimension: Option<usize>) -> Self {
        let dim = dimension.unwrap_or(3072);
        Self {
            api_key: api_key.into(),
            model: "text-embedding-3-large".to_owned(),
            dimension: dim,
            endpoint: "https://api.openai.com/v1/embeddings".to_owned(),
            embedder_id: format!("openai-text-embedding-3-large-{dim}d"),
        }
    }

    /// Create a fully custom OpenAI-compatible provider.
    #[must_use]
    pub fn custom(
        api_key: impl Into<String>,
        model: impl Into<String>,
        dimension: usize,
        endpoint: impl Into<String>,
    ) -> Self {
        let model = model.into();
        let embedder_id = format!("openai-{model}-{dimension}d");
        Self {
            api_key: api_key.into(),
            model,
            dimension,
            endpoint: endpoint.into(),
            embedder_id,
        }
    }
}

impl ApiProvider for OpenAiProvider {
    fn provider_name(&self) -> &str {
        "openai"
    }

    fn api_model_id(&self) -> &str {
        &self.model
    }

    fn embedder_id(&self) -> &str {
        &self.embedder_id
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn max_batch_size(&self) -> usize {
        2048
    }

    fn supports_mrl(&self) -> bool {
        self.model.starts_with("text-embedding-3-")
    }

    fn endpoint_url(&self) -> &str {
        &self.endpoint
    }

    fn request_headers(&self) -> Vec<(String, String)> {
        vec![
            (
                "authorization".to_owned(),
                format!("Bearer {}", self.api_key),
            ),
            ("content-type".to_owned(), "application/json".to_owned()),
        ]
    }

    fn serialize_request(&self, texts: &[&str]) -> SearchResult<Vec<u8>> {
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
            "dimensions": self.dimension,
            "encoding_format": "float"
        });
        serde_json::to_vec(&body).map_err(|e| SearchError::EmbeddingFailed {
            model: self.embedder_id.clone(),
            source: e.into(),
        })
    }

    fn deserialize_response(&self, body: &[u8]) -> SearchResult<Vec<Vec<f32>>> {
        let v: serde_json::Value =
            serde_json::from_slice(body).map_err(|e| SearchError::EmbeddingFailed {
                model: self.embedder_id.clone(),
                source: format!("JSON parse error: {e}").into(),
            })?;

        // Check for API-level error.
        if let Some(err) = v.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown API error");
            return Err(SearchError::EmbeddingFailed {
                model: self.embedder_id.clone(),
                source: format!("OpenAI API error: {msg}").into(),
            });
        }

        let data = v.get("data").and_then(|d| d.as_array()).ok_or_else(|| {
            SearchError::EmbeddingFailed {
                model: self.embedder_id.clone(),
                source: "missing 'data' array in response".into(),
            }
        })?;

        // Sort by index field to ensure correct ordering.
        let mut indexed: Vec<(usize, Vec<f32>)> = data
            .iter()
            .map(|item| {
                let idx = item.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let emb = item
                    .get("embedding")
                    .and_then(|e| e.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();
                (idx, emb)
            })
            .collect();
        indexed.sort_by_key(|(idx, _)| *idx);

        Ok(indexed.into_iter().map(|(_, emb)| emb).collect())
    }
}

// ─── Gemini ─────────────────────────────────────────────────────────────────

/// Google Gemini embeddings API provider (`text-embedding-004`, `embedding-001`).
#[derive(Debug, Clone)]
pub struct GeminiProvider {
    api_key: String,
    model: String,
    dimension: usize,
    embedder_id: String,
}

impl GeminiProvider {
    /// Create a Gemini provider for `text-embedding-004` (768-dimensional).
    #[must_use]
    pub fn text_embedding_004(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "text-embedding-004".to_owned(),
            dimension: 768,
            embedder_id: "gemini-text-embedding-004-768d".to_owned(),
        }
    }

    /// Create a Gemini provider for `embedding-001` (768-dimensional).
    #[must_use]
    pub fn embedding_001(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "embedding-001".to_owned(),
            dimension: 768,
            embedder_id: "gemini-embedding-001-768d".to_owned(),
        }
    }
}

impl ApiProvider for GeminiProvider {
    fn provider_name(&self) -> &str {
        "gemini"
    }

    fn api_model_id(&self) -> &str {
        &self.model
    }

    fn embedder_id(&self) -> &str {
        &self.embedder_id
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn max_batch_size(&self) -> usize {
        100
    }

    fn supports_mrl(&self) -> bool {
        false
    }

    fn endpoint_url(&self) -> &str {
        "https://generativelanguage.googleapis.com"
    }

    fn request_url(&self) -> String {
        self.batch_embed_url()
    }

    fn request_headers(&self) -> Vec<(String, String)> {
        vec![("content-type".to_owned(), "application/json".to_owned())]
    }

    fn serialize_request(&self, texts: &[&str]) -> SearchResult<Vec<u8>> {
        let requests: Vec<serde_json::Value> = texts
            .iter()
            .map(|text| {
                serde_json::json!({
                    "model": format!("models/{}", self.model),
                    "content": {
                        "parts": [{"text": text}]
                    }
                })
            })
            .collect();

        let body = serde_json::json!({ "requests": requests });
        serde_json::to_vec(&body).map_err(|e| SearchError::EmbeddingFailed {
            model: self.embedder_id.clone(),
            source: e.into(),
        })
    }

    fn deserialize_response(&self, body: &[u8]) -> SearchResult<Vec<Vec<f32>>> {
        let v: serde_json::Value =
            serde_json::from_slice(body).map_err(|e| SearchError::EmbeddingFailed {
                model: self.embedder_id.clone(),
                source: format!("JSON parse error: {e}").into(),
            })?;

        // Check for API-level error.
        if let Some(err) = v.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown API error");
            return Err(SearchError::EmbeddingFailed {
                model: self.embedder_id.clone(),
                source: format!("Gemini API error: {msg}").into(),
            });
        }

        let embeddings = v
            .get("embeddings")
            .and_then(|e| e.as_array())
            .ok_or_else(|| SearchError::EmbeddingFailed {
                model: self.embedder_id.clone(),
                source: "missing 'embeddings' array in response".into(),
            })?;

        embeddings
            .iter()
            .map(|item| {
                item.get("values")
                    .and_then(|vals| vals.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .ok_or_else(|| SearchError::EmbeddingFailed {
                        model: self.embedder_id.clone(),
                        source: "missing 'values' in embedding entry".into(),
                    })
            })
            .collect()
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

impl GeminiProvider {
    /// Construct the full batch-embed URL including API key.
    #[must_use]
    pub fn batch_embed_url(&self) -> String {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents?key={}",
            self.model, self.api_key
        )
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_serialize_request() {
        let p = OpenAiProvider::text_embedding_3_small("test-key", Some(256));
        let body = p.serialize_request(&["hello", "world"]).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["model"], "text-embedding-3-small");
        assert_eq!(v["dimensions"], 256);
        assert_eq!(v["input"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn openai_deserialize_response() {
        let p = OpenAiProvider::text_embedding_3_small("test-key", Some(3));
        let response = serde_json::json!({
            "data": [
                {"index": 1, "embedding": [0.1, 0.2, 0.3]},
                {"index": 0, "embedding": [0.4, 0.5, 0.6]}
            ]
        });
        let embeddings = p
            .deserialize_response(&serde_json::to_vec(&response).unwrap())
            .unwrap();
        // Should be sorted by index.
        assert_eq!(embeddings.len(), 2);
        assert!((embeddings[0][0] - 0.4).abs() < f32::EPSILON);
        assert!((embeddings[1][0] - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn openai_error_response() {
        let p = OpenAiProvider::text_embedding_3_small("test-key", None);
        let response = serde_json::json!({
            "error": {"message": "Invalid API key", "type": "auth_error"}
        });
        let err = p
            .deserialize_response(&serde_json::to_vec(&response).unwrap())
            .unwrap_err();
        assert!(err.to_string().contains("Invalid API key"));
    }

    #[test]
    fn gemini_serialize_request() {
        let p = GeminiProvider::text_embedding_004("test-key");
        let body = p.serialize_request(&["hello"]).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let requests = v["requests"].as_array().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["model"], "models/text-embedding-004");
    }

    #[test]
    fn gemini_deserialize_response() {
        let p = GeminiProvider::text_embedding_004("test-key");
        let response = serde_json::json!({
            "embeddings": [
                {"values": [0.1, 0.2, 0.3]}
            ]
        });
        let embeddings = p
            .deserialize_response(&serde_json::to_vec(&response).unwrap())
            .unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 3);
    }

    #[test]
    fn gemini_batch_url() {
        let p = GeminiProvider::text_embedding_004("mykey");
        assert!(p.batch_embed_url().contains("text-embedding-004"));
        assert!(p.batch_embed_url().contains("key=mykey"));
    }

    #[test]
    fn openai_supports_mrl() {
        let p = OpenAiProvider::text_embedding_3_small("k", None);
        assert!(p.supports_mrl());
        let p2 = OpenAiProvider::custom("k", "ada-002", 1536, "https://example.com");
        assert!(!p2.supports_mrl());
    }

    #[test]
    fn openai_embedder_id_includes_dimension() {
        let p = OpenAiProvider::text_embedding_3_small("k", Some(512));
        assert_eq!(p.embedder_id(), "openai-text-embedding-3-small-512d");
    }
}
