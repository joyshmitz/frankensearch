//! Pure-Rust cross-encoder reranker backed by frankentorch (no ONNX / no `ort`).
//!
//! Reimplements the `cross-encoder/ms-marco-MiniLM-L6-v2` `BertForSequenceClassification`
//! forward pass (6 layers, hidden 384, 12 heads, exact GELU, LayerNorm eps 1e-12,
//! `[CLS]` pooler + classifier, `sigmoid(logit)`) on frankentorch tensors in **f32**
//! (bd-1nl13.10), validated against the numpy/ONNX reference (bd-1nl13.2/.3). f32 is
//! the dominant inference dtype: half the memory and ~2x the throughput of the
//! original f64 forward, while preserving the reference ranking.
//!
//! Embedding lookups go through `tensor_index_select` rather than `tensor_embedding`:
//! `index_select` preserves the weight dtype (f32 in/f32 out, frankentorch-40i), whereas
//! `tensor_embedding`'s custom gather still materialises f64. The two are semantically
//! identical here (no `padding_idx`). LayerNorm hits frankentorch's f32 fused no-grad
//! fast path. The int8 path (bd-1nl13.15) is tracked separately.
//!
//! The only reranker backend (ort/ONNX was removed in bd-1nl13.6); feature-gated
//! behind `native`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;
use tokenizers::Tokenizer;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{RerankDocument, RerankScore, SyncRerank};

const H: usize = 384;
const L: usize = 6;
const NH: usize = 12;
const HD: usize = H / NH; // 32
const EPS: f64 = 1e-12;
const DEFAULT_MAX_LENGTH: usize = 512;
const MODEL_NAME: &str = "ms-marco-minilm-l-6-v2";
const SAFETENSORS_PRIMARY: &str = "model_f32.safetensors";
const SAFETENSORS_FALLBACK: &str = "model.safetensors";
const TOKENIZER_JSON: &str = "tokenizer.json";

fn rerank_err(ctx: &str, e: impl std::fmt::Display) -> SearchError {
    SearchError::RerankFailed {
        model: MODEL_NAME.to_owned(),
        source: format!("{ctx}: {e}").into(),
    }
}

/// Owns the frankentorch session and the loaded weight tensors. Mutated during the
/// forward pass, so it lives behind a `Mutex` in `NativeReranker`.
struct Model {
    s: FrankenTorchSession,
    w: HashMap<String, TensorNodeId>,
    /// Autograd tape node count captured right after the persistent weights are
    /// loaded. Each forward pass truncates the tape back to this boundary to free
    /// that pass's intermediate activations, so the session does not grow
    /// unbounded across many rerank calls (a single long-doc forward can allocate
    /// ~25 MB attention tensors per layer; without truncation they would
    /// accumulate for the life of the process).
    weights_boundary: usize,
}

impl Model {
    fn g(&self, name: &str) -> SearchResult<TensorNodeId> {
        self.w
            .get(name)
            .copied()
            .ok_or_else(|| rerank_err("weights", format!("missing weight tensor {name}")))
    }

    /// y = x @ Wᵀ + b   (weight stored row-major [out, in], PyTorch convention)
    fn linear(&mut self, x: TensorNodeId, prefix: &str) -> SearchResult<TensorNodeId> {
        let w = self.g(&format!("{prefix}.weight"))?;
        let b = self.g(&format!("{prefix}.bias"))?;
        let wt = self
            .s
            .tensor_transpose(w, 0, 1)
            .map_err(|e| rerank_err("linear.transpose", e))?;
        let y = self
            .s
            .tensor_matmul(x, wt)
            .map_err(|e| rerank_err("linear.matmul", e))?;
        self.s.tensor_add(y, b).map_err(|e| rerank_err("linear.add", e))
    }

    fn ln(&mut self, x: TensorNodeId, prefix: &str) -> SearchResult<TensorNodeId> {
        let w = self.g(&format!("{prefix}.weight"))?;
        let b = self.g(&format!("{prefix}.bias"))?;
        self.s
            .tensor_layer_norm(x, vec![H], Some(w), Some(b), EPS)
            .map_err(|e| rerank_err("layer_norm", e))
    }

    fn idx(&mut self, vals: &[i64]) -> SearchResult<TensorNodeId> {
        let f: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        self.s
            .tensor_variable(f, vec![vals.len()], false)
            .map_err(|e| rerank_err("index_tensor", e))
    }

    /// [S, H] -> [NH, S, HD]
    fn heads(&mut self, x: TensorNodeId, s_len: usize) -> SearchResult<TensorNodeId> {
        let r = self
            .s
            .tensor_reshape(x, vec![s_len, NH, HD])
            .map_err(|e| rerank_err("heads.reshape", e))?;
        self.s
            .tensor_transpose(r, 0, 1)
            .map_err(|e| rerank_err("heads.transpose", e))
    }

    /// Single-pair forward pass (batch = 1, no padding/mask needed). Returns the raw logit.
    ///
    /// Runs entirely in f32: weights are f32 leaves and every op preserves f32, so
    /// embedding/matmul/softmax/layer_norm all stay in the f32 kernels.
    fn forward(&mut self, ids: &[i64], typ: &[i64]) -> SearchResult<f32> {
        let s_len = ids.len();
        // embeddings: word + position + token_type, then LayerNorm.
        // `index_select(weight, dim=0, indices)` is the embedding lookup; unlike
        // `tensor_embedding` it preserves the f32 weight dtype (frankentorch-40i).
        let id_t = self.idx(ids)?;
        let pos: Vec<i64> = (0..s_len as i64).collect();
        let pos_t = self.idx(&pos)?;
        let typ_t = self.idx(typ)?;
        let we = self.g("bert.embeddings.word_embeddings.weight")?;
        let pe = self.g("bert.embeddings.position_embeddings.weight")?;
        let te = self.g("bert.embeddings.token_type_embeddings.weight")?;
        let e_word = self
            .s
            .tensor_index_select(we, 0, id_t)
            .map_err(|e| rerank_err("embed.word", e))?;
        let e_pos = self
            .s
            .tensor_index_select(pe, 0, pos_t)
            .map_err(|e| rerank_err("embed.pos", e))?;
        let e_typ = self
            .s
            .tensor_index_select(te, 0, typ_t)
            .map_err(|e| rerank_err("embed.type", e))?;
        let mut emb = self
            .s
            .tensor_add(e_word, e_pos)
            .map_err(|e| rerank_err("embed.add", e))?;
        emb = self.s.tensor_add(emb, e_typ).map_err(|e| rerank_err("embed.add2", e))?;
        emb = self.ln(emb, "bert.embeddings.LayerNorm")?;

        let scale = 1.0 / (HD as f64).sqrt();
        for i in 0..L {
            let p = format!("bert.encoder.layer.{i}");
            // self-attention
            let q = self.linear(emb, &format!("{p}.attention.self.query"))?;
            let k = self.linear(emb, &format!("{p}.attention.self.key"))?;
            let v = self.linear(emb, &format!("{p}.attention.self.value"))?;
            let q = self.heads(q, s_len)?;
            let k = self.heads(k, s_len)?;
            let v = self.heads(v, s_len)?;
            let kt = self
                .s
                .tensor_transpose(k, 1, 2)
                .map_err(|e| rerank_err("attn.kt", e))?; // [NH, HD, S]
            let mut scores = self.s.tensor_bmm(q, kt).map_err(|e| rerank_err("attn.qk", e))?;
            scores = self
                .s
                .tensor_mul_scalar(scores, scale)
                .map_err(|e| rerank_err("attn.scale", e))?;
            let probs = self
                .s
                .tensor_softmax(scores, 2)
                .map_err(|e| rerank_err("attn.softmax", e))?;
            let ctx = self.s.tensor_bmm(probs, v).map_err(|e| rerank_err("attn.ctx", e))?;
            let ctx = self
                .s
                .tensor_transpose(ctx, 0, 1)
                .map_err(|e| rerank_err("attn.ctx_t", e))?;
            let ctx = self
                .s
                .tensor_reshape(ctx, vec![s_len, H])
                .map_err(|e| rerank_err("attn.ctx_reshape", e))?;
            let attn = self.linear(ctx, &format!("{p}.attention.output.dense"))?;
            let sum1 = self.s.tensor_add(emb, attn).map_err(|e| rerank_err("attn.residual", e))?;
            emb = self.ln(sum1, &format!("{p}.attention.output.LayerNorm"))?;
            // feed-forward
            let inter = self.linear(emb, &format!("{p}.intermediate.dense"))?;
            let inter = self.s.tensor_gelu(inter).map_err(|e| rerank_err("ffn.gelu", e))?;
            let ffn = self.linear(inter, &format!("{p}.output.dense"))?;
            let sum2 = self.s.tensor_add(emb, ffn).map_err(|e| rerank_err("ffn.residual", e))?;
            emb = self.ln(sum2, &format!("{p}.output.LayerNorm"))?;
        }
        // pooler on [CLS] (row 0) + classifier
        let cls = self
            .s
            .tensor_narrow(emb, 0, 0, 1)
            .map_err(|e| rerank_err("pooler.narrow", e))?; // [1, H]
        let pooled = self.linear(cls, "bert.pooler.dense")?;
        let pooled = self.s.tensor_tanh(pooled).map_err(|e| rerank_err("pooler.tanh", e))?;
        let logit_t = self.linear(pooled, "classifier")?; // [1, 1]
        let vals = self
            .s
            .tensor_values_f32(logit_t)
            .map_err(|e| rerank_err("classifier.values", e))?;
        let logit = vals
            .first()
            .copied()
            .ok_or_else(|| rerank_err("classifier", "empty logit output"))?;
        // Free this forward pass's intermediate tape nodes (everything created
        // after the weights), keeping the loaded parameters, so the session's
        // arena does not grow unbounded across rerank calls.
        self.s.truncate_autograd_graph(self.weights_boundary);
        Ok(logit)
    }
}

/// Pure-Rust frankentorch cross-encoder reranker.
pub struct NativeReranker {
    inner: Mutex<Model>,
    tokenizer: Tokenizer,
    max_length: usize,
    name: String,
    id: String,
}

impl std::fmt::Debug for NativeReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeReranker")
            .field("name", &self.name)
            .field("max_length", &self.max_length)
            .finish_non_exhaustive()
    }
}

impl NativeReranker {
    /// Load the reranker from a model directory containing a safetensors weight file
    /// (`model_f32.safetensors` preferred, else `model.safetensors`) and `tokenizer.json`.
    ///
    /// # Errors
    /// `SearchError::ModelNotFound` when required files are missing;
    /// `SearchError::ModelLoadFailed` when the tokenizer or weights fail to load.
    pub fn load(model_dir: impl AsRef<Path>) -> SearchResult<Self> {
        let dir = model_dir.as_ref();

        let tok_path = dir.join(TOKENIZER_JSON);
        if !tok_path.is_file() {
            return Err(SearchError::ModelNotFound {
                name: format!("{MODEL_NAME} (missing {TOKENIZER_JSON} in {})", dir.display()),
            });
        }
        let mut tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| SearchError::ModelLoadFailed {
                path: tok_path.clone(),
                source: format!("tokenizer load failed: {e}").into(),
            })?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: DEFAULT_MAX_LENGTH,
                ..Default::default()
            }))
            .map_err(|e| SearchError::ModelLoadFailed {
                path: tok_path.clone(),
                source: format!("failed to enable truncation: {e}").into(),
            })?;

        let weights_path = {
            let primary = dir.join(SAFETENSORS_PRIMARY);
            if primary.is_file() {
                primary
            } else {
                dir.join(SAFETENSORS_FALLBACK)
            }
        };
        if !weights_path.is_file() {
            return Err(SearchError::ModelNotFound {
                name: format!(
                    "{MODEL_NAME} (missing {SAFETENSORS_PRIMARY} or {SAFETENSORS_FALLBACK} in {})",
                    dir.display()
                ),
            });
        }

        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        session.no_grad_enter();
        let weights = load_f32_safetensors(&mut session, &weights_path)?;
        // Tape boundary AFTER the persistent weights are created; each forward
        // truncates back to here to free intermediates while keeping parameters.
        let weights_boundary = session.autograd_graph_node_count();

        tracing::info!(
            model = MODEL_NAME,
            tensors = weights.len(),
            max_length = DEFAULT_MAX_LENGTH,
            model_dir = %dir.display(),
            "native frankentorch reranker loaded"
        );

        Ok(Self {
            inner: Mutex::new(Model { s: session, w: weights, weights_boundary }),
            tokenizer,
            max_length: DEFAULT_MAX_LENGTH,
            name: MODEL_NAME.to_owned(),
            id: MODEL_NAME.to_owned(),
        })
    }
}

/// Parse a safetensors file, loading every F32 tensor as an f32 leaf and skipping
/// any non-F32 tensor (e.g. the I64 `position_ids` buffer, which we regenerate).
fn load_f32_safetensors(
    session: &mut FrankenTorchSession,
    path: &Path,
) -> SearchResult<HashMap<String, TensorNodeId>> {
    let bytes = fs::read(path).map_err(|e| SearchError::ModelLoadFailed {
        path: path.to_path_buf(),
        source: format!("read safetensors: {e}").into(),
    })?;
    if bytes.len() < 8 {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors file too small".into(),
        });
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")) as usize;
    let header_end = 8usize
        .checked_add(header_len)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors header length out of range".into(),
        })?;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end]).map_err(|e| {
        SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: format!("safetensors header parse: {e}").into(),
        }
    })?;
    let data = &bytes[header_end..];
    let obj = header.as_object().ok_or_else(|| SearchError::ModelLoadFailed {
        path: path.to_path_buf(),
        source: "safetensors header is not an object".into(),
    })?;

    let mut weights = HashMap::new();
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info.get("dtype").and_then(serde_json::Value::as_str).unwrap_or("");
        if dtype != "F32" {
            continue; // skip I64 position_ids and any non-float buffers
        }
        let shape: Vec<usize> = info
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
            .unwrap_or_default();
        let offsets = info
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: format!("safetensors tensor {name} missing data_offsets").into(),
            })?;
        let start = offsets.first().and_then(serde_json::Value::as_u64).unwrap_or(0) as usize;
        let end = offsets.get(1).and_then(serde_json::Value::as_u64).unwrap_or(0) as usize;
        if start > end || end > data.len() {
            return Err(SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: format!("safetensors tensor {name} has out-of-range offsets").into(),
            });
        }
        let raw = &data[start..end];
        // f32 leaves: keep the safetensors values in their native dtype so the whole
        // forward stays in f32 (frankentorch's f32 kernels), not upcast to f64.
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let node = session
            .tensor_variable_f32(vals, shape, false)
            .map_err(|e| SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: format!("create tensor {name}: {e}").into(),
            })?;
        weights.insert(name.clone(), node);
    }
    if weights.is_empty() {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "no F32 tensors found in safetensors".into(),
        });
    }
    Ok(weights)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl SyncRerank for NativeReranker {
    fn rerank_sync(
        &self,
        query: &str,
        documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let mut model = self
            .inner
            .lock()
            .map_err(|e| rerank_err("lock", format!("reranker mutex poisoned: {e}")))?;

        let mut out = Vec::with_capacity(documents.len());
        for (rank, doc) in documents.iter().enumerate() {
            let encoding = self
                .tokenizer
                .encode((query, doc.text.as_str()), true)
                .map_err(|e| rerank_err("tokenize", e))?;
            let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
            let mut typ: Vec<i64> = encoding.get_type_ids().iter().map(|&t| i64::from(t)).collect();
            if ids.len() > self.max_length {
                ids.truncate(self.max_length);
                typ.truncate(self.max_length);
            }
            let logit = model.forward(&ids, &typ)?;
            let (score, raw_logit) = if logit.is_finite() {
                (sigmoid(logit), Some(logit))
            } else {
                (0.0, None)
            };
            out.push(RerankScore {
                doc_id: doc.doc_id.clone(),
                score,
                original_rank: rank,
                raw_logit,
            });
        }
        Ok(out)
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn max_length(&self) -> usize {
        self.max_length
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_DIR: &str = "/private/tmp/ee-reranker-port/model";

    // (query, document, reference logit) from the validated parity_cases.json
    // (numpy reference in f64, itself validated bit-for-ranking against the real ONNX
    // model). The forward runs in f32, so logits match the f64 reference to within a
    // small tolerance (PARITY_TOL) while the ranking is bit-identical.
    const PARITY_TOL: f64 = 0.1;
    const CASES: &[(&str, &str, f64)] = &[
        (
            "how to fix a failing release workflow",
            "the release pipeline builds cross platform binaries and uploads them to github",
            -9.808567,
        ),
        (
            "how to fix a failing release workflow",
            "bananas are a good source of potassium and taste sweet",
            -11.332987,
        ),
        (
            "what is the capital of france",
            "paris is the capital and most populous city of france",
            7.472003,
        ),
        (
            "rust memory safety",
            "the borrow checker enforces ownership rules at compile time",
            -11.367251,
        ),
    ];

    fn model_available() -> bool {
        Path::new(MODEL_DIR).join(TOKENIZER_JSON).is_file()
            && (Path::new(MODEL_DIR).join(SAFETENSORS_PRIMARY).is_file()
                || Path::new(MODEL_DIR).join(SAFETENSORS_FALLBACK).is_file())
    }

    fn doc(id: &str, text: &str) -> RerankDocument {
        RerankDocument { doc_id: id.to_owned(), text: text.to_owned() }
    }

    #[test]
    fn parity_logits_and_ranking_match_reference() {
        if !model_available() {
            eprintln!("[native_reranker] SKIP parity: model dir {MODEL_DIR} not present");
            return;
        }
        let reranker = NativeReranker::load(MODEL_DIR).expect("load native reranker");
        let mut logits = Vec::new();
        let mut max_diff = 0.0_f64;
        eprintln!("[native_reranker] idx |     ft_logit |    ref_logit |     diff");
        for (i, (query, document, ref_logit)) in CASES.iter().enumerate() {
            let scored = reranker
                .rerank_sync(query, &[doc("d", document)])
                .expect("rerank_sync");
            assert_eq!(scored.len(), 1, "one doc in, one score out");
            let logit = scored[0].raw_logit.expect("raw logit present") as f64;
            let diff = (logit - ref_logit).abs();
            max_diff = max_diff.max(diff);
            logits.push(logit);
            eprintln!("[native_reranker] {i:3} | {logit:12.6} | {ref_logit:12.6} | {diff:8.5}");
            assert!(
                diff < PARITY_TOL,
                "case {i} logit {logit} differs from reference {ref_logit} by {diff} (>{PARITY_TOL})"
            );
        }
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        eprintln!("[native_reranker] ranking(desc)={order:?} expected=[2, 0, 1, 3] max_diff={max_diff:.6}");
        assert_eq!(order, vec![2usize, 0, 1, 3], "ranking must match reference");
    }

    #[test]
    fn empty_documents_yield_empty_scores() {
        if !model_available() {
            eprintln!("[native_reranker] SKIP empty-docs: model dir not present");
            return;
        }
        let reranker = NativeReranker::load(MODEL_DIR).expect("load");
        let scored = reranker.rerank_sync("any query", &[]).expect("empty ok");
        assert!(scored.is_empty());
        eprintln!("[native_reranker] empty-docs -> empty scores OK");
    }

    #[test]
    fn whitespace_and_long_documents_do_not_panic() {
        if !model_available() {
            eprintln!("[native_reranker] SKIP whitespace/long: model dir not present");
            return;
        }
        let reranker = NativeReranker::load(MODEL_DIR).expect("load");
        // whitespace-only doc
        let ws = reranker.rerank_sync("q", &[doc("ws", "   ")]).expect("whitespace ok");
        assert_eq!(ws.len(), 1);
        // very long doc (forces truncation well beyond max_length)
        let long_text = "memory safety ".repeat(400);
        let lng = reranker.rerank_sync("rust", &[doc("long", &long_text)]).expect("long ok");
        assert_eq!(lng.len(), 1);
        assert!(lng[0].score.is_finite());
        eprintln!(
            "[native_reranker] whitespace score={:.6}, truncated-long score={:.6} OK",
            ws[0].score, lng[0].score
        );
    }

    #[test]
    fn ranking_is_deterministic_across_runs() {
        if !model_available() {
            eprintln!("[native_reranker] SKIP determinism: model dir not present");
            return;
        }
        let reranker = NativeReranker::load(MODEL_DIR).expect("load");
        let docs: Vec<RerankDocument> = CASES
            .iter()
            .enumerate()
            .map(|(i, (_, d, _))| doc(&format!("d{i}"), d))
            .collect();
        let run1 = reranker.rerank_sync("what is the capital of france", &docs).expect("run1");
        let run2 = reranker.rerank_sync("what is the capital of france", &docs).expect("run2");
        assert_eq!(run1.len(), run2.len());
        for (a, b) in run1.iter().zip(run2.iter()) {
            assert_eq!(a.doc_id, b.doc_id);
            assert_eq!(a.raw_logit, b.raw_logit, "logits must be deterministic");
        }
        eprintln!("[native_reranker] determinism across 2 runs OK");
    }

    #[test]
    fn load_missing_dir_errors() {
        let err = NativeReranker::load("/private/tmp/definitely-not-a-model-dir-xyz");
        assert!(err.is_err(), "loading a missing dir must error, not panic");
        eprintln!("[native_reranker] missing-dir load error OK: {err:?}");
    }
}
