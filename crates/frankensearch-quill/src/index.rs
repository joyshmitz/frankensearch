//! Shipping scalar Quill index orchestration.
//!
//! This module deliberately joins the already-final Scribe, FSLX, Keeper,
//! parser, cursor, scorer, and collector boundaries without introducing a
//! searchable delta.  Documents become visible only after a MANIFEST publish.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use frankensearch_core::IndexableDocument;
#[cfg(feature = "durability")]
use frankensearch_durability::FileProtector;
use thiserror::Error;
use tracing::Instrument;
use xxhash_rust::xxh3::{Xxh3, xxh3_64};

use crate::argus::{
    ArgusError, Bm25FieldSnapshot, DocSetCollector, FieldNormReader, PhraseScorer, PhraseTerm,
    PositionsHandle, PositionsReader, PostingCursor, ReferenceScorer, ScorerClause, TermScorer,
    TopDocsCollector,
};
use crate::config::QuillConfig;
use crate::error::QuillError;
use crate::grimoire::{ByteSpan, TermDictionary, TermDictionaryError, TermSectionLengths};
use crate::keeper::{
    CURRENT_ENGINE_VERSION, KeeperError, KeeperSnapshot, KeeperWriter, Manifest,
    ManifestFieldStats, ManifestSegment, RecoveredSegment, TombstoneSet,
};
use crate::query::{
    DefaultQueryParser, Query, QueryDiagnostic, QueryParserConfigError, canonicalize_query,
};
use crate::quiver::{
    DocLenCodecError, DocLenSection, PositionCodecError, PositionList, Posting, PostingCodecError,
    PostingList, SnapshotFieldStats,
};
use crate::schema::{DEFAULT_SCHEMA, SchemaDescriptor};
use crate::scribe::{
    AccumulatorError, ColumnarAccumulator, DOC_ORDS_PER_LEASE, FlushDocumentInput, FlushError,
    FlushMode, FlushSegmentInput, IndexedFieldValue, StoredFieldValue, flush_accumulator_with_mode,
};
use crate::segment::{EncodedSegment, SectionKind};

const ID_FIELD: u16 = 0;
const CONTENT_FIELD: u16 = 1;
const TITLE_FIELD: u16 = 2;
const METADATA_FIELD: u16 = 3;
const ORD_FIELD: u16 = 4;
const MAX_GLOBAL_DOCID_EXCLUSIVE: u64 = 1_u64 << 32;

/// Typed failure from the scalar shipping facade.
#[derive(Debug, Error)]
pub enum QuillIndexError {
    /// Engine configuration was rejected before opening resources.
    #[error("invalid Quill index configuration: {0}")]
    Config(#[source] frankensearch_core::SearchError),
    /// Keeper admission, recovery, or publication failed.
    #[error(transparent)]
    Keeper(#[from] KeeperError),
    /// FSLX framing or section access failed.
    #[error(transparent)]
    Quill(#[from] QuillError),
    /// Parser/schema binding failed.
    #[error(transparent)]
    Parser(#[from] QueryParserConfigError),
    /// One document could not be accumulated atomically.
    #[error(transparent)]
    Accumulator(#[from] AccumulatorError),
    /// A scalar accumulator could not be sealed.
    #[error(transparent)]
    Flush(#[from] FlushError),
    /// A term dictionary was malformed or incompatible with its sections.
    #[error(transparent)]
    Dictionary(#[from] TermDictionaryError),
    /// A posting list was malformed.
    #[error(transparent)]
    Postings(#[from] PostingCodecError),
    /// A positions list was malformed.
    #[error(transparent)]
    Positions(#[from] PositionCodecError),
    /// A field-length section was malformed.
    #[error(transparent)]
    DocLen(#[from] DocLenCodecError),
    /// Exhaustive scorer construction or collection failed.
    #[error(transparent)]
    Argus(#[from] ArgusError),
    /// Canonical metadata serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// The requested query shape is outside the scalar G1a surface.
    #[error("unsupported scalar Quill query: {detail}")]
    UnsupportedQuery { detail: String },
    /// An index lifecycle or arithmetic invariant failed.
    #[error("invalid scalar Quill index state: {detail}")]
    InvalidState { detail: String },
    /// Structured cancellation was observed before a state transition.
    #[error("scalar Quill operation cancelled during {phase}")]
    Cancelled { phase: &'static str },
}

/// One final lexical winner.
#[derive(Clone, Debug, PartialEq)]
pub struct QuillHit {
    /// Stable external document identifier.
    pub document_id: String,
    /// Snapshot-global Q1 document identifier used as the native tie key.
    pub global_docid: u32,
    /// Exhaustive BM25 score.
    pub score: f32,
}

/// One globally paginated exhaustive result.
#[derive(Clone, Debug, PartialEq)]
pub struct QuillSearchResult {
    /// Page-local hits in `(score desc, global_docid asc)` order.
    pub hits: Vec<QuillHit>,
    /// Exact match count when requested.
    pub total_count: Option<u64>,
    /// Public live-document count for the committed snapshot.
    pub doc_count: u64,
    /// Lenient parser recovery diagnostics.
    pub diagnostics: Vec<QueryDiagnostic>,
}

#[derive(Debug)]
struct PendingIdentity {
    doc_ord: u32,
    document_id: String,
    canonical_content: Vec<u8>,
}

struct StagedFlush {
    encoded: EncodedSegment,
    manifest_segment: ManifestSegment,
    pending_field_stats: BTreeMap<u16, (u64, u32)>,
    next_seal_seq: u64,
}

/// Scalar, delta-free Quill writer and committed reader view.
pub struct QuillIndex {
    config: QuillConfig,
    schema: SchemaDescriptor,
    parser: DefaultQueryParser,
    backend: IndexBackend,
    accumulator: ColumnarAccumulator,
    identities: Vec<PendingIdentity>,
    uncommitted_ids: BTreeSet<String>,
    current_lease_base: Option<u64>,
    next_lease_base: u64,
    next_seal_seq: u64,
    staged_flush: Option<StagedFlush>,
    pending_segments: Vec<ManifestSegment>,
    pending_owned_segments: Vec<EncodedSegment>,
    pending_field_stats: BTreeMap<u16, (u64, u32)>,
    pending_manifest: Option<Manifest>,
}

enum IndexBackend {
    Durable(KeeperWriter),
    Memory(KeeperSnapshot),
}

impl IndexBackend {
    const fn snapshot(&self) -> &KeeperSnapshot {
        match self {
            Self::Durable(writer) => writer.snapshot(),
            Self::Memory(snapshot) => snapshot,
        }
    }
}

impl QuillIndex {
    /// Create a shipping-schema index or open an existing compatible index.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, or schema failures.
    pub async fn create(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        Self::create_with_schema(cx, directory, DEFAULT_SCHEMA, config).await
    }

    /// Open an existing shipping-schema index for mutation and search.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, or schema failures.
    pub async fn open(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = false,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let instrumented = open_span.clone();
        async move {
            let writer = KeeperWriter::open(cx, directory, DEFAULT_SCHEMA).await?;
            let index = Self::from_backend(IndexBackend::Durable(writer), DEFAULT_SCHEMA, config)?;
            record_snapshot_fields(&open_span, index.snapshot());
            Ok(index)
        }
        .instrument(instrumented)
        .await
    }

    /// Open an existing shipping-schema index with FEC repair enabled.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, schema, or durability
    /// failures.
    #[cfg(feature = "durability")]
    pub async fn open_durable(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
        protector: FileProtector,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = true,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let instrumented = open_span.clone();
        async move {
            let writer =
                KeeperWriter::open_durable(cx, directory, DEFAULT_SCHEMA, protector).await?;
            let index = Self::from_backend(IndexBackend::Durable(writer), DEFAULT_SCHEMA, config)?;
            record_snapshot_fields(&open_span, index.snapshot());
            Ok(index)
        }
        .instrument(instrumented)
        .await
    }

    /// Create or open a shipping-schema index with FEC repair enabled.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, admission, recovery, schema, or durability
    /// failures.
    #[cfg(feature = "durability")]
    pub async fn create_durable(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        config: QuillConfig,
        protector: FileProtector,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = true,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let instrumented = open_span.clone();
        async move {
            let writer =
                KeeperWriter::create_durable(cx, directory, DEFAULT_SCHEMA, protector).await?;
            let index = Self::from_backend(IndexBackend::Durable(writer), DEFAULT_SCHEMA, config)?;
            record_snapshot_fields(&open_span, index.snapshot());
            Ok(index)
        }
        .instrument(instrumented)
        .await
    }

    async fn create_with_schema(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = false,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let instrumented = open_span.clone();
        async move {
            let writer = KeeperWriter::create(cx, directory, schema).await?;
            let index = Self::from_backend(IndexBackend::Durable(writer), schema, config)?;
            record_snapshot_fields(&open_span, index.snapshot());
            Ok(index)
        }
        .instrument(instrumented)
        .await
    }

    /// Construct an owned-buffer genesis index without filesystem I/O.
    ///
    /// # Errors
    ///
    /// Returns typed configuration, schema, or parser failures.
    pub fn in_memory(config: QuillConfig) -> Result<Self, QuillIndexError> {
        validate_config(&config)?;
        let open_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_OPEN,
            phase = "open",
            durability = false,
            generation = tracing::field::Empty,
            segment_count = tracing::field::Empty,
            doc_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
        let _open_entered = open_span.enter();
        let snapshot = KeeperSnapshot::in_memory(DEFAULT_SCHEMA)?;
        let index = Self::from_backend(IndexBackend::Memory(snapshot), DEFAULT_SCHEMA, config)?;
        record_snapshot_fields(&open_span, index.snapshot());
        Ok(index)
    }

    fn from_backend(
        backend: IndexBackend,
        schema: SchemaDescriptor,
        config: QuillConfig,
    ) -> Result<Self, QuillIndexError> {
        let parser = DefaultQueryParser::new(schema)?;
        let accumulator = ColumnarAccumulator::new(schema)?;
        let manifest = &backend.snapshot().loaded_manifest().manifest;
        let next_lease_base = next_lease_boundary(manifest.docid_high_watermark)?;
        let next_seal_seq = manifest
            .segments
            .iter()
            .map(|segment| segment.seal_seq)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        Ok(Self {
            config,
            schema,
            parser,
            backend,
            accumulator,
            identities: Vec::new(),
            uncommitted_ids: BTreeSet::new(),
            current_lease_base: None,
            next_lease_base,
            next_seal_seq,
            staged_flush: None,
            pending_segments: Vec::new(),
            pending_owned_segments: Vec::new(),
            pending_field_stats: BTreeMap::new(),
            pending_manifest: None,
        })
    }

    /// Current committed immutable snapshot.
    #[must_use]
    pub const fn snapshot(&self) -> &KeeperSnapshot {
        self.backend.snapshot()
    }

    /// Durable index directory, or `None` for an owned-buffer index.
    #[must_use]
    pub fn directory(&self) -> Option<&Path> {
        self.backend.snapshot().directory()
    }

    /// Number of committed live documents.
    #[must_use]
    pub const fn doc_count(&self) -> u64 {
        self.backend.snapshot().doc_count()
    }

    /// Whether the writer holds documents or installed segments not yet visible.
    #[must_use]
    pub fn has_uncommitted_changes(&self) -> bool {
        self.accumulator.document_count() != 0
            || self.staged_flush.is_some()
            || !self.pending_segments.is_empty()
            || self.pending_manifest.is_some()
    }

    /// Accumulate one bounded batch into the single scalar shard.
    ///
    /// A budget or lease boundary seals an immutable segment, but no newly
    /// installed segment becomes searchable until [`Self::commit`] publishes
    /// its successor MANIFEST.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, duplicate-id, accumulation, flush, or
    /// publication failures. A failed prior MANIFEST publish must be retried
    /// with `commit` before more documents are accepted.
    pub async fn index_documents(
        &mut self,
        cx: &Cx,
        documents: &[IndexableDocument],
    ) -> Result<(), QuillIndexError> {
        let ingest_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::SCRIBE_INGEST,
            phase = "ingest",
            doc_count = documents.len(),
            result_count = tracing::field::Empty,
            arena_bytes_used_high_water = tracing::field::Empty,
            arena_bytes_reserved_high_water = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _ingest_timer = crate::tracing_conventions::StageTimer::new(&ingest_span);
        let instrumented = ingest_span.clone();
        async move {
            check_cancel(cx, "index")?;
            if self.staged_flush.is_some()
                || !self.pending_segments.is_empty()
                || self.pending_manifest.is_some()
            {
                return Err(invalid_state(
                    "installed segments await MANIFEST publication; retry commit before indexing",
                ));
            }
            let mut arena_bytes_used_high_water = self.accumulator.bytes_used();
            let mut arena_bytes_reserved_high_water = self.accumulator.bytes_reserved();
            for document in documents {
                check_cancel(cx, "index")?;
                if document.id.is_empty() {
                    return Err(invalid_state("document id must be nonempty"));
                }
                if self.uncommitted_ids.contains(&document.id)
                    || self
                        .backend
                        .snapshot()
                        .resolve_document_id(&document.id)?
                        .is_some()
                {
                    return Err(invalid_state(format!(
                        "duplicate live document id {:?}",
                        document.id
                    )));
                }
                let lease_base = self.ensure_lease()?;
                let doc_ord = u32::try_from(self.accumulator.document_count())
                    .map_err(|_| invalid_state("accumulator document count does not fit u32"))?;
                let global_docid = lease_base
                    .checked_add(u64::from(doc_ord))
                    .filter(|docid| *docid < MAX_GLOBAL_DOCID_EXCLUSIVE)
                    .ok_or_else(|| invalid_state("global Q1 document-id space exhausted"))?;

                let metadata = canonical_metadata(&document.metadata)?;
                let ordinal = global_docid.to_le_bytes();
                let title = document.title.as_deref().unwrap_or("");
                let indexed = [
                    IndexedFieldValue::new(ID_FIELD, &document.id),
                    IndexedFieldValue::new(CONTENT_FIELD, &document.content),
                    IndexedFieldValue::new(TITLE_FIELD, title),
                ];
                let stored = [
                    StoredFieldValue::new(METADATA_FIELD, &metadata),
                    StoredFieldValue::new(ORD_FIELD, &ordinal),
                ];
                let accumulated =
                    self.accumulator
                        .add_document_with_values(doc_ord, &indexed, &[], &stored)?;
                arena_bytes_used_high_water =
                    arena_bytes_used_high_water.max(accumulated.bytes_used);
                arena_bytes_reserved_high_water =
                    arena_bytes_reserved_high_water.max(accumulated.bytes_reserved);
                let canonical_content = canonical_document_preimage(document, &metadata)?;
                self.identities.push(PendingIdentity {
                    doc_ord,
                    document_id: document.id.clone(),
                    canonical_content,
                });
                self.uncommitted_ids.insert(document.id.clone());

                if self.accumulator.document_count() == DOC_ORDS_PER_LEASE as usize
                    || self
                        .accumulator
                        .should_flush(self.config.scribe_shard_budget_bytes)
                {
                    self.flush_current(cx).await?;
                }
            }
            ingest_span.record(
                "result_count",
                u64::try_from(documents.len()).unwrap_or(u64::MAX),
            );
            ingest_span.record(
                "arena_bytes_used_high_water",
                u64::try_from(arena_bytes_used_high_water).unwrap_or(u64::MAX),
            );
            ingest_span.record(
                "arena_bytes_reserved_high_water",
                u64::try_from(arena_bytes_reserved_high_water).unwrap_or(u64::MAX),
            );
            Ok(())
        }
        .instrument(instrumented)
        .await
    }

    /// Seal the scalar accumulator and atomically publish the next MANIFEST.
    ///
    /// # Errors
    ///
    /// Returns typed flush, segment-install, manifest-transition, durability,
    /// or cancellation failures. Installed-but-unreferenced segments remain
    /// invisible and are retained for a publication retry.
    pub async fn commit(&mut self, cx: &Cx) -> Result<&KeeperSnapshot, QuillIndexError> {
        check_cancel(cx, "commit")?;
        self.flush_current(cx).await?;
        check_cancel(cx, "commit publish")?;
        if self.pending_segments.is_empty() {
            return Ok(self.backend.snapshot());
        }

        self.prepare_pending_manifest()?;
        let manifest = self
            .pending_manifest
            .as_ref()
            .expect("nonempty pending segments have a retained MANIFEST proposal")
            .clone();
        let commit_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_COMMIT,
            phase = "commit",
            generation = manifest.generation,
            segment_count = manifest.segments.len(),
            doc_count = manifest.field_stats.first().map_or(0, |row| row.doc_count),
            result_count = manifest.segments.len(),
            duration_us = tracing::field::Empty,
        );
        let _commit_timer = crate::tracing_conventions::StageTimer::new(&commit_span);
        let instrumented = commit_span.clone();
        async {
            check_cancel(cx, "commit publish")?;
            let open_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::KEEPER_OPEN,
                phase = "open.committed",
                durability = matches!(&self.backend, IndexBackend::Durable(_)),
                generation = tracing::field::Empty,
                segment_count = tracing::field::Empty,
                doc_count = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _open_timer = crate::tracing_conventions::StageTimer::new(&open_span);
            let instrumented = open_span.clone();
            async {
                match &mut self.backend {
                    IndexBackend::Durable(writer) => {
                        writer.publish(cx, &manifest).await?;
                    }
                    IndexBackend::Memory(snapshot) => {
                        let published = snapshot.publish_owned_segments(
                            &manifest,
                            self.pending_owned_segments.clone(),
                        )?;
                        *snapshot = published;
                    }
                }
                record_snapshot_fields(&open_span, self.backend.snapshot());
                Ok::<(), QuillIndexError>(())
            }
            .instrument(instrumented)
            .await?;
            Ok::<(), QuillIndexError>(())
        }
        .instrument(instrumented)
        .await?;
        self.pending_segments.clear();
        self.pending_owned_segments.clear();
        self.pending_field_stats.clear();
        self.pending_manifest = None;
        self.uncommitted_ids.clear();
        Ok(self.backend.snapshot())
    }

    fn prepare_pending_manifest(&mut self) -> Result<(), QuillIndexError> {
        if self.pending_manifest.is_some() {
            return Ok(());
        }
        let mut manifest = self.backend.snapshot().next_manifest()?;
        manifest
            .segments
            .extend(self.pending_segments.iter().cloned());
        manifest
            .segments
            .sort_unstable_by_key(|segment| segment.docid_lo);
        manifest.docid_high_watermark = self.next_lease_base;
        manifest.field_stats = merge_field_stats(&manifest.field_stats, &self.pending_field_stats)?;
        if matches!(&self.backend, IndexBackend::Durable(_)) {
            // Retain one exact, explicitly stamped image until publication
            // succeeds. A crash-shaped retry may encounter the prior attempt's
            // synced `.tmp-manifest-N`, whose bytes must remain reusable even
            // after the wall clock advances.
            manifest.last_publish_unix_s = wall_clock_unix_s()?;
        }
        self.pending_manifest = Some(manifest);
        Ok(())
    }

    async fn flush_current(&mut self, cx: &Cx) -> Result<(), QuillIndexError> {
        if self.staged_flush.is_none() && self.accumulator.document_count() == 0 {
            return Ok(());
        }
        let seal_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::KEEPER_SEAL,
            phase = "seal",
            segment_id = tracing::field::Empty,
            doc_count = self.accumulator.document_count(),
            token_count = self.accumulator.token_count(),
            result_count = tracing::field::Empty,
            output_bytes = tracing::field::Empty,
            arena_bytes_used_high_water = self.accumulator.bytes_used(),
            arena_bytes_reserved_high_water = self.accumulator.bytes_reserved(),
            duration_us = tracing::field::Empty,
        );
        let _seal_timer = crate::tracing_conventions::StageTimer::new(&seal_span);
        let instrumented = seal_span.clone();
        async {
            check_cancel(cx, "flush")?;
            self.prepare_current_flush()?;
            if let Some(staged) = self.staged_flush.as_ref() {
                seal_span.record("segment_id", staged.manifest_segment.segment_id);
                seal_span.record("result_count", u64::from(staged.manifest_segment.doc_count));
                seal_span.record("output_bytes", staged.encoded.file_len());
            }
            self.install_staged_flush(cx).await
        }
        .instrument(instrumented)
        .await
    }

    fn prepare_current_flush(&mut self) -> Result<(), QuillIndexError> {
        if self.staged_flush.is_some() || self.accumulator.document_count() == 0 {
            return Ok(());
        }
        let lease_docid_base = self
            .current_lease_base
            .ok_or_else(|| invalid_state("nonempty accumulator has no Q1 lease"))?;
        let created_unix_s = self.created_unix_s()?;
        let segment_id = self.derive_segment_id(lease_docid_base, created_unix_s)?;
        let documents = self
            .identities
            .iter()
            .map(|identity| {
                FlushDocumentInput::from_canonical_content(
                    identity.doc_ord,
                    &identity.document_id,
                    &identity.canonical_content,
                )
            })
            .collect::<Vec<_>>();
        let encoded = flush_accumulator_with_mode(
            &self.accumulator,
            FlushSegmentInput {
                segment_id,
                lease_docid_base,
                created_unix_s,
                engine_version: CURRENT_ENGINE_VERSION,
                documents: &documents,
            },
            FlushMode::Scalar,
        )?;
        let manifest_segment = manifest_segment(&encoded, self.next_seal_seq);
        let document_count = u32::try_from(self.accumulator.document_count())
            .map_err(|_| invalid_state("segment document count does not fit u32"))?;
        let mut pending_field_stats = self.pending_field_stats.clone();
        for field in self.accumulator.fields() {
            let entry = pending_field_stats
                .entry(field.field_ord())
                .or_insert((0, 0));
            entry.0 = entry
                .0
                .checked_add(field.total_tokens())
                .ok_or_else(|| invalid_state("pending field token count overflow"))?;
            entry.1 = entry
                .1
                .checked_add(document_count)
                .ok_or_else(|| invalid_state("pending field document count overflow"))?;
        }
        let next_seal_seq = self
            .next_seal_seq
            .checked_add(1)
            .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        self.pending_segments
            .try_reserve(1)
            .map_err(|_| invalid_state("could not reserve pending segment bookkeeping"))?;
        if matches!(&self.backend, IndexBackend::Memory(_)) {
            self.pending_owned_segments
                .try_reserve(1)
                .map_err(|_| invalid_state("could not reserve owned segment bookkeeping"))?;
        }
        self.staged_flush = Some(StagedFlush {
            encoded,
            manifest_segment,
            pending_field_stats,
            next_seal_seq,
        });
        Ok(())
    }

    async fn install_staged_flush(&mut self, cx: &Cx) -> Result<(), QuillIndexError> {
        let Some(staged) = self.staged_flush.as_ref() else {
            return Ok(());
        };
        if let IndexBackend::Durable(writer) = &mut self.backend {
            let directory = writer
                .snapshot()
                .directory()
                .expect("KeeperWriter always owns a durable directory");
            let pending = staged.encoded.write_temp_retryable(directory)?;
            writer.publish_segment(cx, pending).await?;
        }
        let StagedFlush {
            encoded,
            manifest_segment,
            pending_field_stats,
            next_seal_seq,
        } = self
            .staged_flush
            .take()
            .expect("staged flush remains present until installation succeeds");
        if matches!(&self.backend, IndexBackend::Memory(_)) {
            self.pending_owned_segments.push(encoded);
        }
        self.pending_segments.push(manifest_segment);
        self.pending_field_stats = pending_field_stats;
        self.next_seal_seq = next_seal_seq;
        self.accumulator.reset();
        self.identities.clear();
        self.current_lease_base = None;
        Ok(())
    }

    fn ensure_lease(&mut self) -> Result<u64, QuillIndexError> {
        if let Some(base) = self.current_lease_base {
            return Ok(base);
        }
        let base = self.next_lease_base;
        let next = base
            .checked_add(u64::from(DOC_ORDS_PER_LEASE))
            .filter(|next| *next <= MAX_GLOBAL_DOCID_EXCLUSIVE)
            .ok_or_else(|| invalid_state("global Q1 document-id lease space exhausted"))?;
        self.current_lease_base = Some(base);
        self.next_lease_base = next;
        Ok(base)
    }

    fn derive_segment_id(
        &self,
        lease_base: u64,
        created_unix_s: i64,
    ) -> Result<u64, QuillIndexError> {
        let generation = self
            .backend
            .snapshot()
            .loaded_manifest()
            .manifest
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid_state("manifest generation exhausted"))?;
        let schema_id = self.schema.schema_id()?;
        let mut batch_hasher = Xxh3::new();
        for identity in &self.identities {
            let len = u64::try_from(identity.canonical_content.len()).map_err(|_| {
                invalid_state("canonical document preimage length does not fit u64")
            })?;
            batch_hasher.update(&len.to_le_bytes());
            batch_hasher.update(&identity.canonical_content);
        }
        let batch_digest = batch_hasher.digest();
        for salt in 0_u64..=u64::from(u16::MAX) {
            let mut preimage = [0_u8; 52];
            preimage[..8].copy_from_slice(&schema_id.to_le_bytes());
            preimage[8..16].copy_from_slice(&generation.to_le_bytes());
            preimage[16..24].copy_from_slice(&lease_base.to_le_bytes());
            preimage[24..32].copy_from_slice(&created_unix_s.to_le_bytes());
            preimage[32..36].copy_from_slice(&CURRENT_ENGINE_VERSION.to_le_bytes());
            preimage[36..44].copy_from_slice(&batch_digest.to_le_bytes());
            preimage[44..].copy_from_slice(&salt.to_le_bytes());
            let candidate = xxh3_64(&preimage);
            let collision = self
                .backend
                .snapshot()
                .loaded_manifest()
                .manifest
                .segments
                .iter()
                .chain(&self.pending_segments)
                .any(|segment| segment.segment_id == candidate);
            if !collision {
                return Ok(candidate);
            }
        }
        Err(invalid_state(
            "could not derive a collision-free segment id",
        ))
    }

    fn created_unix_s(&self) -> Result<i64, QuillIndexError> {
        if self.config.deterministic_ingest {
            return Ok(0);
        }
        wall_clock_unix_s()
    }

    /// Parse and exhaustively execute one query over the committed snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, section validation, unsupported-query, or
    /// scorer/collector failures. Uncommitted accumulator and installed bytes
    /// are intentionally absent from the searched snapshot.
    pub fn search_paginated(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        offset: usize,
        exact_count: bool,
    ) -> Result<QuillSearchResult, QuillIndexError> {
        let query_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_QUERY,
            phase = "query",
            query_len = query.len(),
            segment_count = self.backend.snapshot().segments().len(),
            doc_count = self.backend.snapshot().doc_count(),
            limit,
            offset,
            exact_count,
            result_count = tracing::field::Empty,
            total_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _query_timer = crate::tracing_conventions::StageTimer::new(&query_span);
        let _query_entered = query_span.enter();
        check_cancel(cx, "search")?;
        let parsed = {
            let parse_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::ARGUS_PARSE,
                phase = "parse",
                query_len = query.len(),
                diagnostic_count = tracing::field::Empty,
                result_count = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _parse_timer = crate::tracing_conventions::StageTimer::new(&parse_span);
            let _parse_entered = parse_span.enter();
            let mut parsed = self.parser.parse_lenient(query);
            let report = canonicalize_query(&mut parsed.query);
            parse_span.record(
                "diagnostic_count",
                u64::try_from(parsed.diagnostics.len()).unwrap_or(u64::MAX),
            );
            parse_span.record("result_count", 1_u64);
            tracing::debug!(
                target: crate::tracing_conventions::TARGET,
                must_not_duplicates_removed = report.must_not_duplicates_removed,
                filter_duplicates_removed = report.filter_duplicates_removed,
                duplicate_fields_removed = report.duplicate_fields_removed,
                boolean_levels_sorted = report.boolean_levels_sorted,
                glob_fields_canonicalized = report.glob_fields_canonicalized,
                "canonicalized parsed Quill query"
            );
            parsed
        };
        let snapshot = self.backend.snapshot();
        let mut collector = if exact_count {
            TopDocsCollector::with_exact_count(limit, offset)?
        } else {
            TopDocsCollector::new(limit, offset)?
        };
        for segment in snapshot.segments() {
            check_cancel(cx, "search")?;
            let score_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::ARGUS_SCORE,
                phase = "score",
                segment_id = segment.manifest().segment_id,
                doc_count = segment.doc_count(),
                duration_us = tracing::field::Empty,
            );
            let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
            let _score_entered = score_span.enter();
            let mut scorer = lower_query(&parsed.query, 1.0, segment, snapshot, self.schema)?;
            collector.collect(&mut scorer, segment)?;
        }
        let collect_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_COLLECT,
            phase = "collect",
            segment_count = snapshot.segments().len(),
            doc_count = snapshot.doc_count(),
            result_count = tracing::field::Empty,
            total_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _collect_timer = crate::tracing_conventions::StageTimer::new(&collect_span);
        let _collect_entered = collect_span.enter();
        let collected = collector.finish()?;
        let total_count = collected.total_count;
        let mut hits = Vec::new();
        hits.try_reserve_exact(collected.hits.len())
            .map_err(|_| invalid_state("could not allocate final hit page"))?;
        for hit in collected.hits {
            let document_id = snapshot.materialize_document_id(hit.global_docid).ok_or(
                ArgusError::MissingExternalDocId {
                    global_docid: hit.global_docid,
                },
            )?;
            hits.push(QuillHit {
                document_id: document_id.to_string(),
                global_docid: hit.global_docid,
                score: hit.score,
            });
        }
        let result_count = u64::try_from(hits.len()).unwrap_or(u64::MAX);
        collect_span.record("result_count", result_count);
        query_span.record("result_count", result_count);
        if let Some(total_count) = total_count {
            collect_span.record("total_count", total_count);
            query_span.record("total_count", total_count);
        }
        Ok(QuillSearchResult {
            hits,
            total_count,
            doc_count: snapshot.doc_count(),
            diagnostics: parsed.diagnostics,
        })
    }

    /// Collect the complete deterministic set of matching global document IDs.
    ///
    /// This is the scoreless collector lane: it lowers the same canonical
    /// default-parser tree as [`Self::search_paginated`], traverses every
    /// committed segment without ranking, and returns sorted unique Q1 IDs.
    /// Uncommitted documents remain invisible.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, parse/lowering, cursor, or allocation
    /// failures.
    pub fn collect_docids(&self, cx: &Cx, query: &str) -> Result<Vec<u32>, QuillIndexError> {
        let query_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_QUERY,
            phase = "query",
            collector = "docset",
            query_len = query.len(),
            segment_count = self.backend.snapshot().segments().len(),
            doc_count = self.backend.snapshot().doc_count(),
            result_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _query_timer = crate::tracing_conventions::StageTimer::new(&query_span);
        let _query_entered = query_span.enter();
        check_cancel(cx, "collect_docids")?;
        let parsed = {
            let parse_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::ARGUS_PARSE,
                phase = "parse",
                query_len = query.len(),
                diagnostic_count = tracing::field::Empty,
                result_count = tracing::field::Empty,
                duration_us = tracing::field::Empty,
            );
            let _parse_timer = crate::tracing_conventions::StageTimer::new(&parse_span);
            let _parse_entered = parse_span.enter();
            let mut parsed = self.parser.parse_lenient(query);
            let _canonicalization = canonicalize_query(&mut parsed.query);
            parse_span.record(
                "diagnostic_count",
                u64::try_from(parsed.diagnostics.len()).unwrap_or(u64::MAX),
            );
            parse_span.record("result_count", 1_u64);
            parsed
        };
        let snapshot = self.backend.snapshot();
        let mut collector = DocSetCollector::new();
        for segment in snapshot.segments() {
            check_cancel(cx, "collect_docids")?;
            let score_span = tracing::info_span!(
                target: crate::tracing_conventions::TARGET,
                crate::tracing_conventions::ARGUS_SCORE,
                phase = "score",
                collector = "docset",
                segment_id = segment.manifest().segment_id,
                doc_count = segment.doc_count(),
                duration_us = tracing::field::Empty,
            );
            let _score_timer = crate::tracing_conventions::StageTimer::new(&score_span);
            let _score_entered = score_span.enter();
            let mut scorer =
                lower_query_unscored(&parsed.query, 1.0, segment, snapshot, self.schema)?;
            collector.collect(&mut scorer, segment)?;
        }
        let collect_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_COLLECT,
            phase = "collect",
            collector = "docset",
            segment_count = snapshot.segments().len(),
            doc_count = snapshot.doc_count(),
            result_count = tracing::field::Empty,
            duration_us = tracing::field::Empty,
        );
        let _collect_timer = crate::tracing_conventions::StageTimer::new(&collect_span);
        let _collect_entered = collect_span.enter();
        let docids = collector.finish();
        let result_count = u64::try_from(docids.len()).unwrap_or(u64::MAX);
        collect_span.record("result_count", result_count);
        query_span.record("result_count", result_count);
        Ok(docids)
    }
}

fn record_snapshot_fields(span: &tracing::Span, snapshot: &KeeperSnapshot) {
    let manifest = &snapshot.loaded_manifest().manifest;
    span.record("generation", manifest.generation);
    span.record(
        "segment_count",
        u64::try_from(manifest.segments.len()).unwrap_or(u64::MAX),
    );
    span.record("doc_count", snapshot.doc_count());
}

fn validate_config(config: &QuillConfig) -> Result<(), QuillIndexError> {
    config.validate().map_err(QuillIndexError::Config)
}

fn check_cancel(cx: &Cx, phase: &'static str) -> Result<(), QuillIndexError> {
    if cx.is_cancel_requested() {
        Err(QuillIndexError::Cancelled { phase })
    } else {
        Ok(())
    }
}

fn invalid_state(detail: impl Into<String>) -> QuillIndexError {
    QuillIndexError::InvalidState {
        detail: detail.into(),
    }
}

fn wall_clock_unix_s() -> Result<i64, QuillIndexError> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid_state("system clock precedes Unix epoch"))?
        .as_secs();
    i64::try_from(seconds).map_err(|_| invalid_state("Unix timestamp does not fit i64"))
}

fn next_lease_boundary(watermark: u64) -> Result<u64, QuillIndexError> {
    if watermark > MAX_GLOBAL_DOCID_EXCLUSIVE {
        return Err(invalid_state(
            "manifest document-id watermark exceeds the Q1 address space",
        ));
    }
    let lease_size = u64::from(DOC_ORDS_PER_LEASE);
    let remainder = watermark % lease_size;
    if remainder == 0 {
        return Ok(watermark);
    }
    watermark
        .checked_add(lease_size - remainder)
        .filter(|boundary| *boundary <= MAX_GLOBAL_DOCID_EXCLUSIVE)
        .ok_or_else(|| invalid_state("manifest document-id watermark cannot reach another lease"))
}

fn canonical_metadata(
    metadata: &std::collections::HashMap<String, String>,
) -> Result<Vec<u8>, serde_json::Error> {
    let ordered = metadata.iter().collect::<BTreeMap<_, _>>();
    serde_json::to_vec(&ordered)
}

fn canonical_document_preimage(
    document: &IndexableDocument,
    metadata: &[u8],
) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(&(
        document.id.as_str(),
        document.content.as_str(),
        document.title.as_deref().unwrap_or(""),
        metadata,
    ))
}

fn manifest_segment(encoded: &EncodedSegment, seal_seq: u64) -> ManifestSegment {
    let header = encoded.header();
    ManifestSegment {
        segment_id: header.segment_id,
        seal_seq,
        file_len: encoded.file_len(),
        file_xxh3: encoded.file_xxh3(),
        docid_lo: header.docid_lo,
        docid_hi: header.docid_hi,
        doc_count: header.doc_count,
        tombstones: TombstoneSet::new(),
    }
}

fn merge_field_stats(
    committed: &[ManifestFieldStats],
    pending: &BTreeMap<u16, (u64, u32)>,
) -> Result<Vec<ManifestFieldStats>, QuillIndexError> {
    let mut merged = committed
        .iter()
        .map(|row| (row.field_ord, (row.total_tokens, row.doc_count)))
        .collect::<BTreeMap<_, _>>();
    for (&field_ord, &(tokens, documents)) in pending {
        let row = merged.entry(field_ord).or_insert((0, 0));
        row.0 = row
            .0
            .checked_add(tokens)
            .ok_or_else(|| invalid_state("manifest field token count overflow"))?;
        row.1 = row
            .1
            .checked_add(documents)
            .ok_or_else(|| invalid_state("manifest field document count overflow"))?;
    }
    Ok(merged
        .into_iter()
        .map(
            |(field_ord, (total_tokens, doc_count))| ManifestFieldStats {
                field_ord,
                total_tokens,
                doc_count,
            },
        )
        .collect())
}

#[derive(Debug)]
struct OwnedFieldNorms {
    field_ord: u16,
    docid_lo: u64,
    values: Vec<u8>,
}

impl FieldNormReader for OwnedFieldNorms {
    fn field_ord(&self) -> u16 {
        self.field_ord
    }

    fn fieldnorm_id(&self, global_docid: u32) -> Option<u8> {
        let offset = u64::from(global_docid).checked_sub(self.docid_lo)?;
        self.values.get(usize::try_from(offset).ok()?).copied()
    }
}

#[derive(Debug)]
struct OwnedPostingCursor {
    postings: Vec<Posting>,
    positions: Option<Vec<Vec<u32>>>,
    cursor: usize,
    segment_num_docs: u32,
}

impl OwnedPostingCursor {
    fn empty(segment_num_docs: u32, positioned: bool) -> Self {
        Self {
            postings: Vec::new(),
            positions: positioned.then(Vec::new),
            cursor: 0,
            segment_num_docs,
        }
    }

    fn current(&self) -> Option<Posting> {
        self.postings.get(self.cursor).copied()
    }
}

impl PostingCursor for OwnedPostingCursor {
    fn doc(&self) -> Option<u32> {
        self.current().map(|posting| posting.doc_id)
    }

    fn freq(&self) -> Option<u32> {
        self.current().map(|posting| posting.freq)
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        self.current()?;
        self.positions.as_ref()?;
        u32::try_from(self.cursor)
            .ok()
            .map(|ordinal| PositionsHandle::new(self, ordinal))
    }

    fn size_hint(&self) -> u32 {
        u32::try_from(self.postings.len()).unwrap_or(u32::MAX)
    }

    fn segment_num_docs(&self) -> u32 {
        self.segment_num_docs
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.cursor < self.postings.len() {
            self.cursor += 1;
        }
        Ok(self.doc())
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.doc());
        }
        let relative = self.postings[self.cursor.min(self.postings.len())..]
            .partition_point(|posting| posting.doc_id < target);
        self.cursor = self
            .cursor
            .saturating_add(relative)
            .min(self.postings.len());
        Ok(self.doc())
    }
}

impl PositionsReader for OwnedPostingCursor {
    fn decode_positions(
        &self,
        posting_ordinal: u32,
        output: &mut Vec<u32>,
    ) -> Result<(), ArgusError> {
        let ordinal = usize::try_from(posting_ordinal)
            .map_err(|_| ArgusError::CursorInvariant("posting ordinal does not fit usize"))?;
        if ordinal != self.cursor {
            return Err(ArgusError::CursorInvariant(
                "positions handle no longer identifies the current posting",
            ));
        }
        let positions = self
            .positions
            .as_ref()
            .and_then(|rows| rows.get(ordinal))
            .ok_or(ArgusError::CursorInvariant(
                "positioned cursor has no current position run",
            ))?;
        output
            .try_reserve_exact(positions.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "owned position run",
                count: positions.len(),
            })?;
        output.extend_from_slice(positions);
        Ok(())
    }
}

fn lower_query(
    query: &Query,
    inherited_boost: f32,
    segment: &RecoveredSegment,
    snapshot: &KeeperSnapshot,
    schema: SchemaDescriptor,
) -> Result<ReferenceScorer<'static>, QuillIndexError> {
    lower_query_with_mode(
        query,
        inherited_boost,
        segment,
        snapshot,
        schema,
        QueryLoweringMode::Scored,
    )
}

fn lower_query_unscored(
    query: &Query,
    inherited_boost: f32,
    segment: &RecoveredSegment,
    snapshot: &KeeperSnapshot,
    schema: SchemaDescriptor,
) -> Result<ReferenceScorer<'static>, QuillIndexError> {
    lower_query_with_mode(
        query,
        inherited_boost,
        segment,
        snapshot,
        schema,
        QueryLoweringMode::Unscored,
    )
}

#[derive(Clone, Copy)]
enum QueryLoweringMode {
    Scored,
    Unscored,
}

fn lower_boolean(
    clauses: Vec<ScorerClause<'static>>,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'static>, QuillIndexError> {
    match mode {
        QueryLoweringMode::Scored => ReferenceScorer::boolean(clauses),
        QueryLoweringMode::Unscored => ReferenceScorer::boolean_unscored(clauses),
    }
    .map_err(QuillIndexError::from)
}

fn lower_query_with_mode(
    query: &Query,
    inherited_boost: f32,
    segment: &RecoveredSegment,
    snapshot: &KeeperSnapshot,
    schema: SchemaDescriptor,
    mode: QueryLoweringMode,
) -> Result<ReferenceScorer<'static>, QuillIndexError> {
    match query {
        Query::Empty => Ok(ReferenceScorer::empty()),
        Query::All => Ok(ReferenceScorer::all_with_boost(
            segment.manifest().docid_lo,
            segment.manifest().docid_hi,
            segment.doc_count(),
            inherited_boost,
        )?),
        Query::Term { fields, text } => {
            let mut clauses = Vec::new();
            clauses
                .try_reserve_exact(fields.len())
                .map_err(|_| invalid_state("could not allocate expanded term clauses"))?;
            for field in fields {
                clauses.push(ScorerClause::should(lower_term(
                    segment,
                    snapshot,
                    schema,
                    field.field_id,
                    text.as_bytes(),
                    inherited_boost * field.boost,
                )?));
            }
            lower_boolean(clauses, mode)
        }
        Query::Phrase {
            fields,
            terms,
            slop,
            prefix,
        } => {
            if *slop != 0 || *prefix {
                return Err(QuillIndexError::UnsupportedQuery {
                    detail: format!("phrase slop={slop} prefix={prefix}"),
                });
            }
            if terms.len() == 1 {
                let term = &terms[0];
                let mut clauses = Vec::new();
                for field in fields {
                    clauses.push(ScorerClause::should(lower_term(
                        segment,
                        snapshot,
                        schema,
                        field.field_id,
                        term.text.as_bytes(),
                        inherited_boost * field.boost,
                    )?));
                }
                return lower_boolean(clauses, mode);
            }
            let mut clauses = Vec::new();
            for field in fields {
                let stats = snapshot_field(snapshot, field.field_id)?;
                let mut phrase_terms = Vec::new();
                phrase_terms
                    .try_reserve_exact(terms.len())
                    .map_err(|_| invalid_state("could not allocate phrase terms"))?;
                for term in terms {
                    let snapshot_doc_freq =
                        snapshot_doc_freq(snapshot, schema, field.field_id, term.text.as_bytes())?;
                    let cursor = open_owned_cursor(
                        segment,
                        schema,
                        field.field_id,
                        term.text.as_bytes(),
                        true,
                    )?;
                    phrase_terms.push(PhraseTerm::new(
                        field.field_id,
                        term.position,
                        cursor,
                        snapshot_doc_freq,
                    ));
                }
                let norms = owned_fieldnorms(segment, schema, field.field_id)?;
                let scorer = PhraseScorer::new(
                    phrase_terms,
                    norms,
                    Bm25FieldSnapshot::new(stats)?,
                    inherited_boost * field.boost,
                )?;
                clauses.push(ScorerClause::should(ReferenceScorer::phrase(scorer)));
            }
            lower_boolean(clauses, mode)
        }
        Query::Boolean { clauses, .. } => {
            let mut lowered = Vec::new();
            lowered
                .try_reserve_exact(clauses.len())
                .map_err(|_| invalid_state("could not allocate Boolean clauses"))?;
            for clause in clauses {
                lowered.push(ScorerClause::new(
                    clause.occur,
                    lower_query_with_mode(
                        &clause.query,
                        inherited_boost,
                        segment,
                        snapshot,
                        schema,
                        mode,
                    )?,
                ));
            }
            lower_boolean(lowered, mode)
        }
        Query::Boost { query, factor } => {
            let boost = inherited_boost * *factor;
            if !boost.is_finite() {
                return Err(QuillIndexError::UnsupportedQuery {
                    detail: format!("non-finite cumulative boost bits 0x{:08x}", boost.to_bits()),
                });
            }
            lower_query_with_mode(query, boost, segment, snapshot, schema, mode)
        }
        Query::Range { .. } | Query::Set { .. } | Query::Glob { .. } => {
            Err(QuillIndexError::UnsupportedQuery {
                detail: "range, set, and glob lowering belongs to a later checkpoint".to_owned(),
            })
        }
    }
}

fn lower_term(
    segment: &RecoveredSegment,
    snapshot: &KeeperSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
    boost: f32,
) -> Result<ReferenceScorer<'static>, QuillIndexError> {
    let stats = snapshot_field(snapshot, field_ord)?;
    let doc_freq = snapshot_doc_freq(snapshot, schema, field_ord, term)?;
    let cursor = open_owned_cursor(segment, schema, field_ord, term, false)?;
    let norms = owned_fieldnorms(segment, schema, field_ord)?;
    Ok(ReferenceScorer::term(TermScorer::new(
        cursor,
        norms,
        Bm25FieldSnapshot::new(stats)?,
        doc_freq,
        boost,
    )?))
}

fn snapshot_field(
    snapshot: &KeeperSnapshot,
    field_ord: u16,
) -> Result<SnapshotFieldStats, QuillIndexError> {
    snapshot
        .loaded_manifest()
        .manifest
        .field_stats
        .iter()
        .find(|row| row.field_ord == field_ord)
        .map(|row| SnapshotFieldStats {
            field_ord,
            total_tokens: row.total_tokens,
            doc_count: u64::from(row.doc_count),
        })
        .ok_or_else(|| invalid_state(format!("manifest has no field statistics for {field_ord}")))
}

fn snapshot_doc_freq(
    snapshot: &KeeperSnapshot,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
) -> Result<u64, QuillIndexError> {
    let mut total = 0_u64;
    for segment in snapshot.segments() {
        let dictionary = open_dictionary(segment, schema)?;
        if let Some(found) = dictionary.lookup(field_ord, term)? {
            total = total
                .checked_add(u64::from(found.metadata.doc_freq))
                .ok_or_else(|| invalid_state("snapshot document frequency overflow"))?;
        }
    }
    Ok(total)
}

fn open_owned_cursor(
    segment: &RecoveredSegment,
    schema: SchemaDescriptor,
    field_ord: u16,
    term: &[u8],
    positioned: bool,
) -> Result<OwnedPostingCursor, QuillIndexError> {
    let dictionary = open_dictionary(segment, schema)?;
    let Some(found) = dictionary.lookup(field_ord, term)? else {
        return Ok(OwnedPostingCursor::empty(segment.doc_count(), positioned));
    };
    let postings_section = required_section(segment, SectionKind::POSTINGS)?;
    let postings_bytes = span(postings_section, found.metadata.postings, "POSTINGS")?;
    let postings = PostingList::parse(postings_bytes, found.metadata.doc_freq)?;
    let rows = postings.decode_all_bounded(found.metadata.doc_freq as usize)?;
    let positions = if positioned {
        let position_span =
            found
                .metadata
                .positions
                .ok_or_else(|| QuillIndexError::UnsupportedQuery {
                    detail: format!("field {field_ord} has no positions"),
                })?;
        let position_section = required_section(segment, SectionKind::POSITIONS)?;
        let position_bytes = span(position_section, position_span, "POSITIONS")?;
        let positions = PositionList::parse(position_bytes, &postings)?;
        let mut decoded = Vec::new();
        decoded
            .try_reserve_exact(rows.len())
            .map_err(|_| invalid_state("could not allocate owned position rows"))?;
        for ordinal in 0..found.metadata.doc_freq {
            let row = positions
                .positions_for_ordinal(ordinal)?
                .collect::<Result<Vec<_>, _>>()?;
            decoded.push(row);
        }
        Some(decoded)
    } else {
        None
    };
    Ok(OwnedPostingCursor {
        postings: rows,
        positions,
        cursor: 0,
        segment_num_docs: segment.doc_count(),
    })
}

fn owned_fieldnorms(
    segment: &RecoveredSegment,
    schema: SchemaDescriptor,
    field_ord: u16,
) -> Result<OwnedFieldNorms, QuillIndexError> {
    let expected = term_field_ords(schema);
    let manifest = segment.manifest();
    let section = DocLenSection::parse(
        required_section(segment, SectionKind::DOCLEN)?,
        manifest.docid_lo,
        manifest.docid_hi,
        &expected,
    )?;
    let field = section
        .field(field_ord)
        .ok_or_else(|| invalid_state(format!("DOCLEN has no field {field_ord}")))?;
    Ok(OwnedFieldNorms {
        field_ord,
        docid_lo: manifest.docid_lo,
        values: field.fieldnorm_ids().to_vec(),
    })
}

fn open_dictionary(
    segment: &RecoveredSegment,
    schema: SchemaDescriptor,
) -> Result<TermDictionary<'_>, QuillIndexError> {
    let postings = required_section(segment, SectionKind::POSTINGS)?;
    let positions = segment.section(SectionKind::POSITIONS)?;
    let blockmax = required_section(segment, SectionKind::BLOCKMAX)?;
    let dictionary = required_section(segment, SectionKind::TERMDICT)?;
    Ok(TermDictionary::parse(
        dictionary,
        schema,
        TermSectionLengths {
            postings: u64::try_from(postings.len())
                .map_err(|_| invalid_state("POSTINGS length does not fit u64"))?,
            positions: positions
                .map(|bytes| u64::try_from(bytes.len()))
                .transpose()
                .map_err(|_| invalid_state("POSITIONS length does not fit u64"))?,
            blockmax: u64::try_from(blockmax.len())
                .map_err(|_| invalid_state("BLOCKMAX length does not fit u64"))?,
        },
    )?)
}

fn required_section(
    segment: &RecoveredSegment,
    kind: SectionKind,
) -> Result<&[u8], QuillIndexError> {
    segment
        .section(kind)?
        .ok_or_else(|| invalid_state(format!("segment is missing section {}", kind.raw())))
}

fn span<'a>(
    section: &'a [u8],
    span: ByteSpan,
    name: &'static str,
) -> Result<&'a [u8], QuillIndexError> {
    let start = usize::try_from(span.offset)
        .map_err(|_| invalid_state(format!("{name} span start does not fit usize")))?;
    let len = usize::try_from(span.len)
        .map_err(|_| invalid_state(format!("{name} span length does not fit usize")))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| invalid_state(format!("{name} span overflows usize")))?;
    section
        .get(start..end)
        .ok_or_else(|| invalid_state(format!("{name} span lies outside its section")))
}

fn term_field_ords(schema: SchemaDescriptor) -> Vec<u16> {
    schema
        .fields
        .iter()
        .filter_map(|field| match field.kind {
            crate::schema::FieldKind::Keyword | crate::schema::FieldKind::Text { .. } => {
                Some(field.id)
            }
            crate::schema::FieldKind::StoredOnly
            | crate::schema::FieldKind::I64 { .. }
            | crate::schema::FieldKind::U64 { .. } => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    #[cfg(feature = "durability")]
    use std::sync::Arc;

    #[cfg(feature = "durability")]
    use frankensearch_durability::{DefaultSymbolCodec, DurabilityConfig};

    use super::*;

    fn run_with_cx<F, Fut>(test: F)
    where
        F: FnOnce(Cx) -> Fut,
        Fut: Future<Output = ()>,
    {
        asupersync::test_utils::run_test_with_cx(test);
    }

    fn deterministic_config() -> QuillConfig {
        QuillConfig {
            deterministic_ingest: true,
            ..QuillConfig::default()
        }
    }

    fn fixture_documents() -> Vec<IndexableDocument> {
        vec![
            IndexableDocument::new("rust-1", "rust ownership prevents data races")
                .with_title("Rust ownership")
                .with_metadata("cluster", "systems"),
            IndexableDocument::new("rust-2", "safe systems programming with rust")
                .with_title("Borrow checker")
                .with_metadata("cluster", "systems"),
            IndexableDocument::new("python-1", "python data science notebooks")
                .with_title("Python guide")
                .with_metadata("cluster", "data"),
        ]
    }

    #[cfg(feature = "durability")]
    fn test_file_protector() -> FileProtector {
        FileProtector::new(
            Arc::new(DefaultSymbolCodec),
            DurabilityConfig {
                symbol_size: 256,
                repair_overhead: 2.0,
                ..DurabilityConfig::default()
            },
        )
        .expect("valid test durability configuration")
    }

    #[test]
    fn scalar_memory_commit_is_visibility_boundary_and_queries_end_to_end() {
        run_with_cx(|cx| async move {
            let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            index
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("accumulate fixture");

            let before = index
                .search_paginated(&cx, "rust", 10, 0, true)
                .expect("search old snapshot");
            assert!(before.hits.is_empty(), "uncommitted docs must be invisible");
            assert_eq!(before.total_count, Some(0));
            assert_eq!(before.doc_count, 0);

            index.commit(&cx).await.expect("publish owned snapshot");
            let term = index
                .search_paginated(&cx, "rust", 10, 0, true)
                .expect("term query");
            assert_eq!(term.total_count, Some(2));
            assert_eq!(term.doc_count, 3);
            assert_eq!(term.hits.len(), 2);
            assert!(
                term.hits
                    .iter()
                    .all(|hit| hit.document_id.starts_with("rust-"))
            );

            let phrase = index
                .search_paginated(&cx, "\"rust ownership\"", 10, 0, true)
                .expect("phrase query");
            assert_eq!(phrase.total_count, Some(1));
            assert_eq!(phrase.hits[0].document_id, "rust-1");

            let boolean = index
                .search_paginated(&cx, "rust AND systems", 10, 0, true)
                .expect("Boolean query");
            assert_eq!(boolean.total_count, Some(1));
            assert_eq!(boolean.hits[0].document_id, "rust-2");

            let page = index
                .search_paginated(&cx, "rust", 1, 1, true)
                .expect("paginated query");
            assert_eq!(page.total_count, Some(2));
            assert_eq!(page.hits.len(), 1);
        });
    }

    #[test]
    fn scoreless_docset_collector_is_wired_across_committed_segments() {
        run_with_cx(|cx| async move {
            let documents = fixture_documents();
            let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            index
                .index_documents(&cx, &documents[..2])
                .await
                .expect("first segment documents");
            index.commit(&cx).await.expect("first segment commit");
            index
                .index_documents(&cx, &documents[2..])
                .await
                .expect("second segment documents");

            assert_eq!(
                index
                    .collect_docids(&cx, "rust OR python")
                    .expect("old committed doc set")
                    .len(),
                2,
                "uncommitted second-segment document must remain invisible",
            );

            index.commit(&cx).await.expect("second segment commit");
            let docids = index
                .collect_docids(&cx, "rust OR python")
                .expect("Boolean doc set");
            let ranked = index
                .search_paginated(&cx, "rust OR python", 10, 0, true)
                .expect("ranked Boolean search");
            let mut ranked_docids = ranked
                .hits
                .iter()
                .map(|hit| hit.global_docid)
                .collect::<Vec<_>>();
            ranked_docids.sort_unstable();
            assert_eq!(docids, ranked_docids);
            assert_eq!(docids.len(), 3);
            assert_eq!(ranked.total_count, Some(3));
        });
    }

    #[test]
    fn deterministic_runs_have_identical_segments_and_results() {
        run_with_cx(|cx| async move {
            let documents = fixture_documents();
            let mut first = QuillIndex::in_memory(deterministic_config()).expect("first index");
            let mut second = QuillIndex::in_memory(deterministic_config()).expect("second index");
            first
                .index_documents(&cx, &documents)
                .await
                .expect("first ingest");
            second
                .index_documents(&cx, &documents)
                .await
                .expect("second ingest");
            first.commit(&cx).await.expect("first commit");
            second.commit(&cx).await.expect("second commit");

            assert_eq!(
                first.snapshot().loaded_manifest().manifest,
                second.snapshot().loaded_manifest().manifest
            );
            let first_segment = &first.snapshot().segments()[0];
            let second_segment = &second.snapshot().segments()[0];
            assert_eq!(first_segment.header(), second_segment.header());
            assert_eq!(first_segment.source_bytes(), second_segment.source_bytes());
            for kind in [
                SectionKind::TERMDICT,
                SectionKind::POSTINGS,
                SectionKind::POSITIONS,
                SectionKind::BLOCKMAX,
                SectionKind::DOCLEN,
                SectionKind::IDMAP,
                SectionKind::IDHASH,
                SectionKind::STOREDMETA,
                SectionKind::STATS,
            ] {
                assert_eq!(
                    first_segment.section(kind).expect("first section"),
                    second_segment.section(kind).expect("second section"),
                    "section {}",
                    kind.raw()
                );
            }

            let first_result = first
                .search_paginated(&cx, "rust", 10, 0, true)
                .expect("first search");
            let second_result = second
                .search_paginated(&cx, "rust", 10, 0, true)
                .expect("second search");
            assert_eq!(first_result, second_result);
        });
    }

    #[test]
    fn unaligned_manifest_watermark_burns_the_partial_lease() {
        let lease_size = u64::from(DOC_ORDS_PER_LEASE);
        assert_eq!(next_lease_boundary(0).expect("genesis"), 0);
        assert_eq!(
            next_lease_boundary(1).expect("partial first lease"),
            lease_size
        );
        assert_eq!(
            next_lease_boundary(lease_size).expect("aligned lease"),
            lease_size
        );
        assert_eq!(
            next_lease_boundary(lease_size + 1).expect("partial second lease"),
            lease_size * 2
        );
        assert!(next_lease_boundary(MAX_GLOBAL_DOCID_EXCLUSIVE - 1).is_ok());
        assert!(next_lease_boundary(MAX_GLOBAL_DOCID_EXCLUSIVE + 1).is_err());
    }

    #[test]
    fn field_stat_overflow_fails_before_segment_installation() {
        run_with_cx(|cx| async move {
            let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            index.pending_field_stats.insert(ID_FIELD, (u64::MAX, 0));
            index
                .index_documents(&cx, &[IndexableDocument::new("overflow", "one token")])
                .await
                .expect("accumulate document");

            let Err(error) = index.commit(&cx).await else {
                panic!("field stats overflow unexpectedly committed");
            };
            assert!(matches!(error, QuillIndexError::InvalidState { .. }));
            assert_eq!(index.accumulator.document_count(), 1);
            assert!(index.staged_flush.is_none());
            assert!(index.pending_segments.is_empty());
            assert!(index.pending_owned_segments.is_empty());
        });
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_manifest_retry_reuses_stamped_bytes_after_clock_advances() {
        let directory = tempfile::tempdir().expect("index tempdir").keep();
        run_with_cx(|cx| async move {
            let mut index = QuillIndex::create(&cx, &directory, deterministic_config())
                .await
                .expect("create filesystem index");
            index
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("accumulate fixture");
            index
                .flush_current(&cx)
                .await
                .expect("install immutable segment");
            index
                .prepare_pending_manifest()
                .expect("retain exact MANIFEST proposal");
            let proposal = index
                .pending_manifest
                .as_ref()
                .expect("proposal retained")
                .clone();
            assert!(proposal.last_publish_unix_s > 0);
            let proposal_bytes = proposal.to_bytes().expect("encode retained proposal");
            let temp_path = directory.join(format!(".tmp-manifest-{}", proposal.generation));
            std::fs::write(&temp_path, &proposal_bytes).expect("seed synced-attempt image");

            std::thread::sleep(std::time::Duration::from_millis(1_100));
            index
                .commit(&cx)
                .await
                .expect("reuse exact prior-attempt bytes");

            assert_eq!(index.snapshot().loaded_manifest().manifest, proposal);
            assert!(index.pending_manifest.is_none());
        });
    }

    #[cfg(unix)]
    #[test]
    fn restarted_ingest_avoids_differing_abandoned_segment_id() {
        let directory = tempfile::tempdir().expect("index tempdir").keep();
        run_with_cx(|cx| async move {
            let abandoned_segment_id = {
                let mut first = QuillIndex::create(&cx, &directory, deterministic_config())
                    .await
                    .expect("create first writer");
                first
                    .index_documents(
                        &cx,
                        &[IndexableDocument::new("abandoned", "old orphan content")],
                    )
                    .await
                    .expect("accumulate abandoned batch");
                first
                    .flush_current(&cx)
                    .await
                    .expect("install segment without MANIFEST");
                assert_eq!(first.snapshot().doc_count(), 0);
                first.pending_segments[0].segment_id
            };
            let abandoned_path = directory.join(format!("seg-{abandoned_segment_id:016x}.fslx"));
            assert!(
                abandoned_path.exists(),
                "orphan must survive writer restart"
            );

            let mut restarted = QuillIndex::open(&cx, &directory, deterministic_config())
                .await
                .expect("reopen old committed generation");
            restarted
                .index_documents(
                    &cx,
                    &[IndexableDocument::new(
                        "replacement",
                        "fresh replacement content",
                    )],
                )
                .await
                .expect("accumulate replacement batch");
            restarted
                .commit(&cx)
                .await
                .expect("publish replacement batch");

            let committed = &restarted.snapshot().loaded_manifest().manifest;
            assert_eq!(committed.segments.len(), 1);
            assert_ne!(committed.segments[0].segment_id, abandoned_segment_id);
            assert!(
                abandoned_path.exists(),
                "grace-window orphan remains intact"
            );
            let replacement = restarted
                .search_paginated(&cx, "replacement", 10, 0, true)
                .expect("search replacement batch");
            assert_eq!(replacement.total_count, Some(1));
            assert_eq!(replacement.hits[0].document_id, "replacement");
            let abandoned = restarted
                .search_paginated(&cx, "abandoned", 10, 0, true)
                .expect("search committed snapshot only");
            assert_eq!(abandoned.total_count, Some(0));
        });
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_facade_protects_commit_and_reopens_searchable_snapshot() {
        let directory = tempfile::tempdir().expect("durable index tempdir").keep();
        run_with_cx(|cx| async move {
            let protector = test_file_protector();
            let mut index = QuillIndex::create_durable(
                &cx,
                &directory,
                deterministic_config(),
                protector.clone(),
            )
            .await
            .expect("create durable index");
            index
                .index_documents(&cx, &fixture_documents())
                .await
                .expect("durable ingest");
            index.commit(&cx).await.expect("durable commit");
            let segment_path = index.snapshot().segments()[0].path().to_path_buf();
            drop(index);

            let manifest_path = directory.join("MANIFEST");
            assert!(
                protector
                    .verify_file(&manifest_path, &FileProtector::sidecar_path(&manifest_path))
                    .expect("verify MANIFEST sidecar")
                    .healthy
            );
            assert!(
                protector
                    .verify_file(&segment_path, &FileProtector::sidecar_path(&segment_path))
                    .expect("verify FSLX sidecar")
                    .healthy
            );

            let reopened =
                QuillIndex::open_durable(&cx, &directory, deterministic_config(), protector)
                    .await
                    .expect("reopen durable index");
            let result = reopened
                .search_paginated(&cx, "rust", 10, 0, true)
                .expect("search reopened durable index");
            assert_eq!(result.total_count, Some(2));
            assert_eq!(result.doc_count, 3);
        });
    }
}
