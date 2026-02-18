use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::error::{SearchError, SearchResult};
use tantivy::schema::{
    FAST, Field, INDEXED, STORED, STRING, Schema, TEXT, TextFieldIndexing, TextOptions,
};
use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, TextAnalyzer};
use tantivy::{Index, IndexReader, IndexWriter, doc};
use tracing::{debug, info, warn};

/// Schema version namespace used for cass-compatible Tantivy indexes.
pub const CASS_SCHEMA_VERSION: &str = "v6";
/// Content hash used to detect schema/tokenizer changes that require rebuild.
pub const CASS_SCHEMA_HASH: &str = "tantivy-schema-v6-long-tokens";

/// Minimum time (ms) between merge operations.
const MERGE_COOLDOWN_MS: i64 = 300_000;
/// Segment count threshold above which merge is triggered.
const MERGE_SEGMENT_THRESHOLD: usize = 4;

/// Global last merge timestamp (ms since epoch).
static LAST_MERGE_TS: AtomicI64 = AtomicI64::new(0);

fn tantivy_err<E>(err: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: "tantivy",
        source: Box::new(err),
    }
}

/// Returns true if the given stored hash matches the current schema hash.
#[must_use]
pub fn cass_schema_hash_matches(stored: &str) -> bool {
    stored == CASS_SCHEMA_HASH
}

/// Named fields used by cass-compatible query and indexing code.
#[derive(Clone, Copy, Debug)]
pub struct CassFields {
    pub agent: Field,
    pub workspace: Field,
    pub workspace_original: Field,
    pub source_path: Field,
    pub msg_idx: Field,
    pub created_at: Field,
    pub title: Field,
    pub content: Field,
    pub title_prefix: Field,
    pub content_prefix: Field,
    pub preview: Field,
    pub source_id: Field,
    pub origin_kind: Field,
    pub origin_host: Field,
}

/// Merge status for cass-compatible Tantivy segment optimization.
#[derive(Debug, Clone)]
pub struct CassMergeStatus {
    pub segment_count: usize,
    pub last_merge_ts: i64,
    pub ms_since_last_merge: i64,
    pub merge_threshold: usize,
    pub cooldown_ms: i64,
}

impl CassMergeStatus {
    #[must_use]
    pub fn should_merge(&self) -> bool {
        self.segment_count >= self.merge_threshold
            && (self.ms_since_last_merge < 0 || self.ms_since_last_merge >= self.cooldown_ms)
    }
}

/// Cass-specific lexical document shape for index ingestion.
#[derive(Debug, Clone)]
pub struct CassDocument {
    pub agent: String,
    pub workspace: Option<String>,
    pub workspace_original: Option<String>,
    pub source_path: String,
    pub msg_idx: u64,
    pub created_at: Option<i64>,
    pub title: Option<String>,
    pub content: String,
    pub source_id: String,
    pub origin_kind: String,
    pub origin_host: Option<String>,
}

/// Tantivy index compatible with cass lexical schema and lifecycle.
pub struct CassTantivyIndex {
    index: Index,
    writer: IndexWriter,
    fields: CassFields,
}

impl CassTantivyIndex {
    /// Open existing index or create/rebuild as needed.
    pub fn open_or_create(path: &Path) -> SearchResult<Self> {
        let schema = cass_build_schema();
        std::fs::create_dir_all(path).map_err(tantivy_err)?;

        let meta_path = path.join("schema_hash.json");
        let mut needs_rebuild = true;
        if meta_path.exists()
            && let Ok(meta) = std::fs::read_to_string(&meta_path)
            && let Ok(json) = serde_json::from_str::<serde_json::Value>(&meta)
            && json.get("schema_hash").and_then(|v| v.as_str()) == Some(CASS_SCHEMA_HASH)
        {
            needs_rebuild = false;
        }

        if needs_rebuild {
            let _ = std::fs::remove_dir_all(path);
            std::fs::create_dir_all(path).map_err(tantivy_err)?;
        }

        let mut index = if path.join("meta.json").exists() && !needs_rebuild {
            match Index::open_in_dir(path) {
                Ok(idx) => idx,
                Err(e) => {
                    warn!(
                        error = %e,
                        "failed to open existing cass-compatible index; rebuilding"
                    );
                    let _ = std::fs::remove_dir_all(path);
                    std::fs::create_dir_all(path).map_err(tantivy_err)?;
                    Index::create_in_dir(path, schema.clone()).map_err(tantivy_err)?
                }
            }
        } else {
            Index::create_in_dir(path, schema.clone()).map_err(tantivy_err)?
        };

        cass_ensure_tokenizer(&mut index);
        std::fs::write(
            &meta_path,
            format!("{{\"schema_hash\":\"{CASS_SCHEMA_HASH}\"}}"),
        )
        .map_err(tantivy_err)?;

        let actual_schema = index.schema();
        let writer = index.writer(50_000_000).map_err(tantivy_err)?;
        let fields = cass_fields_from_schema(&actual_schema)?;
        Ok(Self {
            index,
            writer,
            fields,
        })
    }

    #[must_use]
    pub fn fields(&self) -> CassFields {
        self.fields
    }

    pub fn reader(&self) -> SearchResult<IndexReader> {
        self.index.reader().map_err(tantivy_err)
    }

    pub fn delete_all(&mut self) -> SearchResult<()> {
        self.writer.delete_all_documents().map_err(tantivy_err)?;
        Ok(())
    }

    pub fn commit(&mut self) -> SearchResult<()> {
        self.writer.commit().map_err(tantivy_err)?;
        Ok(())
    }

    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.index
            .searchable_segment_ids()
            .map_or(0, |ids| ids.len())
    }

    #[must_use]
    pub fn merge_status(&self) -> CassMergeStatus {
        let last_merge_ts = LAST_MERGE_TS.load(Ordering::Relaxed);
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
        let ms_since_last = if last_merge_ts > 0 {
            now_ms - last_merge_ts
        } else {
            -1
        };
        CassMergeStatus {
            segment_count: self.segment_count(),
            last_merge_ts,
            ms_since_last_merge: ms_since_last,
            merge_threshold: MERGE_SEGMENT_THRESHOLD,
            cooldown_ms: MERGE_COOLDOWN_MS,
        }
    }

    /// Trigger async merge when threshold/cooldown permit.
    pub fn optimize_if_idle(&mut self) -> SearchResult<bool> {
        let segment_ids = self.index.searchable_segment_ids().map_err(tantivy_err)?;
        let segment_count = segment_ids.len();
        if segment_count < MERGE_SEGMENT_THRESHOLD {
            debug!(
                segments = segment_count,
                threshold = MERGE_SEGMENT_THRESHOLD,
                "skipping merge: below threshold"
            );
            return Ok(false);
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
        let last_merge = LAST_MERGE_TS.load(Ordering::Relaxed);
        if last_merge > 0 && (now_ms - last_merge) < MERGE_COOLDOWN_MS {
            debug!(
                ms_since_last = now_ms - last_merge,
                cooldown = MERGE_COOLDOWN_MS,
                "skipping merge: cooldown active"
            );
            return Ok(false);
        }

        info!(
            segments = segment_count,
            "starting cass-compatible segment merge"
        );
        let _merge_future = self.writer.merge(&segment_ids);
        LAST_MERGE_TS.store(now_ms, Ordering::Relaxed);
        Ok(true)
    }

    /// Force immediate merge and block until completion.
    pub fn force_merge(&mut self) -> SearchResult<()> {
        let segment_ids = self.index.searchable_segment_ids().map_err(tantivy_err)?;
        if segment_ids.is_empty() {
            return Ok(());
        }
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));

        let merge_future = self.writer.merge(&segment_ids);
        match merge_future.wait() {
            Ok(_) => {
                LAST_MERGE_TS.store(now_ms, Ordering::Relaxed);
                Ok(())
            }
            Err(err) => Err(tantivy_err(err)),
        }
    }

    /// Add a batch of cass-compatible documents.
    pub fn add_cass_documents(&mut self, docs: &[CassDocument]) -> SearchResult<()> {
        for cass_doc in docs {
            let mut d = doc! {
                self.fields.agent => cass_doc.agent.clone(),
                self.fields.source_path => cass_doc.source_path.clone(),
                self.fields.msg_idx => cass_doc.msg_idx,
                self.fields.content => cass_doc.content.clone(),
                self.fields.source_id => cass_doc.source_id.clone(),
                self.fields.origin_kind => cass_doc.origin_kind.clone(),
            };

            if let Some(host) = &cass_doc.origin_host
                && !host.is_empty()
            {
                d.add_text(self.fields.origin_host, host);
            }
            if let Some(workspace) = &cass_doc.workspace {
                d.add_text(self.fields.workspace, workspace);
            }
            if let Some(workspace_original) = &cass_doc.workspace_original {
                d.add_text(self.fields.workspace_original, workspace_original);
            }
            if let Some(ts) = cass_doc.created_at {
                d.add_i64(self.fields.created_at, ts);
            }
            if let Some(title) = &cass_doc.title {
                d.add_text(self.fields.title, title);
                d.add_text(self.fields.title_prefix, cass_generate_edge_ngrams(title));
            }
            d.add_text(
                self.fields.content_prefix,
                cass_generate_edge_ngrams(&cass_doc.content),
            );
            d.add_text(
                self.fields.preview,
                cass_build_preview(&cass_doc.content, 400),
            );
            self.writer.add_document(d).map_err(tantivy_err)?;
        }
        Ok(())
    }
}

/// Build cass-compatible Tantivy schema.
#[must_use]
pub fn cass_build_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    let text = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("hyphen_normalize")
                .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();

    let text_not_stored = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("hyphen_normalize")
            .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
    );

    schema_builder.add_text_field("agent", STRING | STORED);
    schema_builder.add_text_field("workspace", STRING | STORED);
    schema_builder.add_text_field("workspace_original", STORED);
    schema_builder.add_text_field("source_path", STORED);
    schema_builder.add_u64_field("msg_idx", INDEXED | STORED);
    schema_builder.add_i64_field("created_at", INDEXED | STORED | FAST);
    schema_builder.add_text_field("title", text.clone());
    schema_builder.add_text_field("content", text);
    schema_builder.add_text_field("title_prefix", text_not_stored.clone());
    schema_builder.add_text_field("content_prefix", text_not_stored);
    schema_builder.add_text_field("preview", TEXT | STORED);
    schema_builder.add_text_field("source_id", STRING | STORED);
    schema_builder.add_text_field("origin_kind", STRING | STORED);
    schema_builder.add_text_field("origin_host", STRING | STORED);
    schema_builder.build()
}

/// Extract cass-compatible schema fields from a Tantivy schema handle.
pub fn cass_fields_from_schema(schema: &Schema) -> SearchResult<CassFields> {
    let get = |name: &str| {
        schema
            .get_field(name)
            .map_err(|_| SearchError::InvalidConfig {
                field: "schema".to_string(),
                value: name.to_string(),
                reason: format!("schema missing required field `{name}`"),
            })
    };

    Ok(CassFields {
        agent: get("agent")?,
        workspace: get("workspace")?,
        workspace_original: get("workspace_original")?,
        source_path: get("source_path")?,
        msg_idx: get("msg_idx")?,
        created_at: get("created_at")?,
        title: get("title")?,
        content: get("content")?,
        title_prefix: get("title_prefix")?,
        content_prefix: get("content_prefix")?,
        preview: get("preview")?,
        source_id: get("source_id")?,
        origin_kind: get("origin_kind")?,
        origin_host: get("origin_host")?,
    })
}

/// Resolve cass-compatible index directory under a data root.
pub fn cass_index_dir(base: &Path) -> SearchResult<PathBuf> {
    let dir = base.join("index").join(CASS_SCHEMA_VERSION);
    std::fs::create_dir_all(&dir).map_err(tantivy_err)?;
    Ok(dir)
}

/// Register the tokenizer used by cass-compatible lexical fields.
pub fn cass_ensure_tokenizer(index: &mut Index) {
    let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(RemoveLongFilter::limit(256))
        .build();
    index.tokenizers().register("hyphen_normalize", analyzer);
}

/// Generate edge n-grams from text for prefix search acceleration.
#[must_use]
pub fn cass_generate_edge_ngrams(text: &str) -> String {
    const MAX_NGRAM_INDICES: usize = 21;
    let mut ngrams = String::with_capacity(text.len() * 2);
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        let indices: Vec<usize> = word
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(word.len()))
            .take(MAX_NGRAM_INDICES)
            .collect();

        if indices.len() < 3 {
            continue;
        }
        for &end_idx in &indices[2..] {
            if !ngrams.is_empty() {
                ngrams.push(' ');
            }
            ngrams.push_str(&word[..end_idx]);
        }
    }
    ngrams
}

/// Build a bounded-length preview from message content.
#[must_use]
pub fn cass_build_preview(content: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut chars = content.chars();
    for _ in 0..max_chars {
        if let Some(ch) = chars.next() {
            out.push(ch);
        } else {
            return out;
        }
    }
    if chars.next().is_some() {
        out.push('…');
    }
    out
}
