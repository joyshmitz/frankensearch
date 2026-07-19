//! Shipping Quill index orchestration and process-local snapshot publication.
//!
//! This module deliberately joins the already-final Scribe, FSLX, Keeper,
//! parser, cursor, scorer, and collector boundaries. The scalar writer still
//! searches sealed segments only; immutable Delta epochs and their composite
//! statistics are published here for the E5 cursor integration.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwap;
use asupersync::Cx;
use asupersync::runtime::spawn_blocking;
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
use crate::delta::DeltaSnapshot;
use crate::error::QuillError;
use crate::grimoire::{ByteSpan, TermDictionary, TermDictionaryError, TermSectionLengths};
use crate::keeper::{
    CURRENT_ENGINE_VERSION, KeeperError, KeeperSnapshot, KeeperWriter, Manifest,
    ManifestFieldStats, ManifestSegment, RecoveredSegment, TombstoneSet,
    validate_manifest_successor,
};
use crate::query::{
    DefaultQueryParser, Query, QueryDiagnostic, QueryParserConfigError, canonicalize_query,
};
use crate::quiver::{
    DocLenCodecError, DocLenSection, PositionCodecError, PositionList, Posting, PostingCodecError,
    PostingList, SnapshotFieldStats,
};
use crate::schema::{DEFAULT_SCHEMA, FieldKind, SchemaDescriptor};
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
    /// Composite process-local snapshot construction or publication failed.
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
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

/// Typed failure from process-local Keeper plus Delta snapshot composition.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SnapshotError {
    /// A Delta epoch was frozen against a different schema.
    #[error("delta schema {actual} does not match Keeper schema {expected}")]
    SchemaMismatch {
        /// Keeper schema name.
        expected: &'static str,
        /// Delta schema name.
        actual: &'static str,
    },
    /// A Delta epoch was frozen against a different Keeper generation.
    #[error(
        "delta lease starting at {lease_base} belongs to Keeper generation {actual}, expected {expected}"
    )]
    KeeperGenerationMismatch {
        /// Generation carried by the composite Keeper snapshot.
        expected: u64,
        /// Generation witnessed when the Delta epoch was frozen.
        actual: u64,
        /// Delta lease identifying the stale epoch.
        lease_base: u64,
    },
    /// A Delta epoch's physical covering interval intersects a sealed range.
    #[error(
        "delta occupied range [{delta_lo}, {delta_hi}) overlaps Keeper segment {segment_id:#018x} range [{keeper_lo}, {keeper_hi})"
    )]
    KeeperDeltaDocidOverlap {
        /// Inclusive first physically allocated Delta docid.
        delta_lo: u64,
        /// Exclusive Delta covering upper bound.
        delta_hi: u64,
        /// Conflicting sealed segment identity.
        segment_id: u64,
        /// Inclusive sealed covering lower bound.
        keeper_lo: u64,
        /// Exclusive sealed covering upper bound.
        keeper_hi: u64,
    },
    /// Two independently published shards claimed overlapping Q1 leases.
    #[error("delta leases [{first_base}, {first_end}) and [{second_base}, {second_end}) overlap")]
    OverlappingDeltaLeases {
        /// First inclusive lease base.
        first_base: u64,
        /// First exclusive lease end.
        first_end: u64,
        /// Second inclusive lease base.
        second_base: u64,
        /// Second exclusive lease end.
        second_end: u64,
    },
    /// A complete publication attempted to replace a newer Keeper generation.
    #[error("Keeper generation regressed from current {current} to proposed {proposed}")]
    KeeperGenerationRegression {
        /// Currently published durable generation.
        current: u64,
        /// Stale proposed generation.
        proposed: u64,
    },
    /// One generation number named two different durable MANIFEST images.
    #[error("Keeper generation {generation} identifies two different MANIFEST images")]
    KeeperGenerationCollision {
        /// Reused generation number.
        generation: u64,
    },
    /// A forward complete publication was not a valid Keeper successor.
    #[error("invalid Keeper snapshot transition: {detail}")]
    KeeperTransition {
        /// Underlying Keeper transition rejection.
        detail: String,
    },
    /// A non-genesis Keeper snapshot omitted an indexed field's statistics.
    #[error("Keeper snapshot has no field statistics for indexed field {field_ord}")]
    MissingKeeperFieldStats {
        /// Missing schema field ordinal.
        field_ord: u16,
    },
    /// A schema-compatible Delta failed to expose one indexed field's tokens.
    #[error("delta snapshot has no live token statistics for indexed field {field_ord}")]
    MissingDeltaFieldStats {
        /// Missing schema field ordinal.
        field_ord: u16,
    },
    /// One checked composite counter exceeded its u64 representation.
    #[error("composite snapshot {counter} overflowed")]
    CounterOverflow {
        /// Stable name of the counter being aggregated.
        counter: &'static str,
    },
    /// A per-shard publication named no current Delta slot.
    #[error("delta shard {shard} is outside the published shard count {shard_count}")]
    ShardOutOfBounds {
        /// Requested zero-based shard slot.
        shard: usize,
        /// Current number of published shard slots.
        shard_count: usize,
    },
    /// The process-local publication epoch cannot advance further.
    #[error("process-local snapshot epoch exhausted")]
    EpochExhausted,
    /// A bounded publication sidecar could not be allocated.
    #[error("unable to allocate {additional} entries for {resource}")]
    Allocation {
        /// Stable allocation purpose.
        resource: &'static str,
        /// Requested additional element count.
        additional: usize,
    },
}

/// One immutable, internally consistent process-local search view.
///
/// The Keeper component retains sealed segments and MANIFEST tombstones. Each
/// Delta component is an owner-isolated frozen epoch. BM25 statistics use the
/// hybrid lifecycle contract: sealed tombstones remain in Keeper's at-seal
/// counts until compaction, while Delta-local tombstones are excluded
/// immediately because those rows will never be sealed.
pub struct QuillSearchSnapshot {
    snapshot_epoch: u64,
    keeper: Arc<KeeperSnapshot>,
    deltas: Box<[Arc<DeltaSnapshot>]>,
    field_stats: Box<[SnapshotFieldStats]>,
    bm25_doc_count: u64,
    live_doc_count: u64,
}

impl QuillSearchSnapshot {
    fn compose(
        snapshot_epoch: u64,
        keeper: Arc<KeeperSnapshot>,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Self, SnapshotError> {
        let schema = keeper.schema();
        let manifest = &keeper.loaded_manifest().manifest;
        let keeper_generation = manifest.generation;
        for delta in &deltas {
            if delta.schema() != schema {
                return Err(SnapshotError::SchemaMismatch {
                    expected: schema.name,
                    actual: delta.schema().name,
                });
            }
            if delta.keeper_generation() != keeper_generation {
                return Err(SnapshotError::KeeperGenerationMismatch {
                    expected: keeper_generation,
                    actual: delta.keeper_generation(),
                    lease_base: delta.lease_base(),
                });
            }
            if let Some((delta_lo, delta_hi)) = delta.occupied_docid_range() {
                for segment in &manifest.segments {
                    if delta_lo < segment.docid_hi && segment.docid_lo < delta_hi {
                        return Err(SnapshotError::KeeperDeltaDocidOverlap {
                            delta_lo,
                            delta_hi,
                            segment_id: segment.segment_id,
                            keeper_lo: segment.docid_lo,
                            keeper_hi: segment.docid_hi,
                        });
                    }
                }
            }
        }
        for (index, first) in deltas.iter().enumerate() {
            for second in &deltas[index + 1..] {
                if first.lease_base() < second.lease_end()
                    && second.lease_base() < first.lease_end()
                {
                    return Err(SnapshotError::OverlappingDeltaLeases {
                        first_base: first.lease_base(),
                        first_end: first.lease_end(),
                        second_base: second.lease_base(),
                        second_end: second.lease_end(),
                    });
                }
            }
        }

        let delta_live_doc_count = deltas.iter().try_fold(0_u64, |total, delta| {
            let count = u64::try_from(delta.live_document_count()).map_err(|_| {
                SnapshotError::CounterOverflow {
                    counter: "live Delta document count",
                }
            })?;
            total
                .checked_add(count)
                .ok_or(SnapshotError::CounterOverflow {
                    counter: "live Delta document count",
                })
        })?;
        let bm25_doc_count = keeper
            .at_seal_doc_count()
            .checked_add(delta_live_doc_count)
            .ok_or(SnapshotError::CounterOverflow {
                counter: "BM25 document count",
            })?;
        let live_doc_count = keeper.doc_count().checked_add(delta_live_doc_count).ok_or(
            SnapshotError::CounterOverflow {
                counter: "live document count",
            },
        )?;

        let field_stats = composite_field_stats(
            schema,
            manifest,
            keeper.segments().is_empty(),
            &deltas,
            bm25_doc_count,
        )?;

        Ok(Self {
            snapshot_epoch,
            keeper,
            deltas: deltas.into_boxed_slice(),
            field_stats,
            bm25_doc_count,
            live_doc_count,
        })
    }

    /// Monotone process-local publication epoch.
    #[must_use]
    pub const fn snapshot_epoch(&self) -> u64 {
        self.snapshot_epoch
    }

    /// Durable MANIFEST generation paired with every Delta epoch in this view.
    #[must_use]
    pub fn keeper_generation(&self) -> u64 {
        self.keeper.loaded_manifest().manifest.generation
    }

    /// Pinned immutable Keeper component.
    #[must_use]
    pub fn keeper_snapshot(&self) -> &KeeperSnapshot {
        &self.keeper
    }

    /// Number of frozen shard epochs in this view.
    #[must_use]
    pub fn delta_count(&self) -> usize {
        self.deltas.len()
    }

    /// Snapshot-global BM25 `N`: Keeper at-seal rows plus live Delta rows.
    #[must_use]
    pub const fn bm25_doc_count(&self) -> u64 {
        self.bm25_doc_count
    }

    /// Public live rows across Keeper tombstones and Delta-local tombstones.
    #[must_use]
    pub const fn live_doc_count(&self) -> u64 {
        self.live_doc_count
    }

    /// Checked composite BM25 statistics for one indexed string field.
    #[must_use]
    pub fn bm25_field_stats(&self, field_ord: u16) -> Option<SnapshotFieldStats> {
        self.field_stats
            .binary_search_by_key(&field_ord, |row| row.field_ord)
            .ok()
            .and_then(|index| self.field_stats.get(index))
            .copied()
    }

    /// Keeper physical document frequency plus live Delta document frequency.
    ///
    /// # Errors
    ///
    /// Returns typed dictionary, allocation, or checked-counter failures.
    pub fn bm25_doc_freq(&self, field_ord: u16, term: &[u8]) -> Result<u64, QuillIndexError> {
        if self.bm25_field_stats(field_ord).is_none() {
            return Err(invalid_state(format!(
                "snapshot has no BM25 statistics for field {field_ord}"
            )));
        }
        let mut total = snapshot_doc_freq(&self.keeper, self.keeper.schema(), field_ord, term)?;
        for delta in &self.deltas {
            let delta_doc_freq = delta
                .find_term(field_ord, term)
                .map_or(0, |found| found.live_doc_freq());
            total = total
                .checked_add(u64::try_from(delta_doc_freq).map_err(|_| {
                    SnapshotError::CounterOverflow {
                        counter: "live Delta document frequency",
                    }
                })?)
                .ok_or(SnapshotError::CounterOverflow {
                    counter: "snapshot document frequency",
                })?;
        }
        Ok(total)
    }
}

fn composite_field_stats(
    schema: SchemaDescriptor,
    manifest: &Manifest,
    genesis_without_segments: bool,
    deltas: &[Arc<DeltaSnapshot>],
    bm25_doc_count: u64,
) -> Result<Box<[SnapshotFieldStats]>, SnapshotError> {
    let indexed_field_count = schema
        .fields
        .iter()
        .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
        .count();
    let mut field_stats = Vec::new();
    field_stats
        .try_reserve_exact(indexed_field_count)
        .map_err(|_| SnapshotError::Allocation {
            resource: "composite field statistics",
            additional: indexed_field_count,
        })?;
    for field in schema
        .fields
        .iter()
        .filter(|field| matches!(field.kind, FieldKind::Keyword | FieldKind::Text { .. }))
    {
        let sealed_total_tokens = manifest
            .field_stats
            .iter()
            .find(|row| row.field_ord == field.id)
            .map(|row| row.total_tokens)
            .or_else(|| genesis_without_segments.then_some(0))
            .ok_or(SnapshotError::MissingKeeperFieldStats {
                field_ord: field.id,
            })?;
        let total_tokens = deltas
            .iter()
            .try_fold(sealed_total_tokens, |total, delta| {
                let delta_tokens = delta.live_total_tokens(field.id).ok_or(
                    SnapshotError::MissingDeltaFieldStats {
                        field_ord: field.id,
                    },
                )?;
                total
                    .checked_add(delta_tokens)
                    .ok_or(SnapshotError::CounterOverflow {
                        counter: "field token count",
                    })
            })?;
        field_stats.push(SnapshotFieldStats {
            field_ord: field.id,
            total_tokens,
            doc_count: bm25_doc_count,
        });
    }
    Ok(field_stats.into_boxed_slice())
}

fn manifest_document_counts(manifest: &Manifest) -> Result<(u64, u64), SnapshotError> {
    manifest
        .segments
        .iter()
        .try_fold((0_u64, 0_u64), |(at_seal, live), segment| {
            let at_seal = at_seal.checked_add(u64::from(segment.doc_count)).ok_or(
                SnapshotError::CounterOverflow {
                    counter: "Keeper at-seal document count",
                },
            )?;
            let live = live
                .checked_add(u64::from(segment.live_doc_count()))
                .ok_or(SnapshotError::CounterOverflow {
                    counter: "Keeper live document count",
                })?;
            Ok((at_seal, live))
        })
}

fn validate_complete_keeper_transition(
    current: &Manifest,
    proposed: &Manifest,
) -> Result<(), SnapshotError> {
    if proposed.generation < current.generation {
        return Err(SnapshotError::KeeperGenerationRegression {
            current: current.generation,
            proposed: proposed.generation,
        });
    }
    if proposed.generation == current.generation {
        return if proposed == current {
            Ok(())
        } else {
            Err(SnapshotError::KeeperGenerationCollision {
                generation: proposed.generation,
            })
        };
    }
    validate_manifest_successor(current, proposed).map_err(|error| {
        SnapshotError::KeeperTransition {
            detail: error.to_string(),
        }
    })
}

struct PreparedSealedPublication {
    snapshot_epoch: u64,
    expected_keeper_generation: u64,
    expected_schema_id: u64,
    expected_docid_high_watermark: u64,
    schema: SchemaDescriptor,
    field_stats: Box<[SnapshotFieldStats]>,
    bm25_doc_count: u64,
    live_doc_count: u64,
}

/// Lock-free publisher for one authoritative Keeper plus all-Delta view.
///
/// Readers call [`Self::load`] once and retain that `Arc` for the whole query.
/// Per-shard updates use compare-and-swap so concurrent shard writers cannot
/// lose one another's epochs. A seal transition must use
/// [`Self::publish_complete`] to replace Keeper and every Delta in one swap.
pub struct SnapshotPublisher {
    current: ArcSwap<QuillSearchSnapshot>,
}

impl SnapshotPublisher {
    /// Construct the first process-local view at epoch zero.
    ///
    /// # Errors
    ///
    /// Rejects stale generations, schema drift, overlapping leases, missing
    /// field statistics, allocation failure, and checked-counter overflow.
    pub fn new(
        keeper: Arc<KeeperSnapshot>,
        deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Self, SnapshotError> {
        let initial = Arc::new(QuillSearchSnapshot::compose(0, keeper, deltas)?);
        Ok(Self {
            current: ArcSwap::new(initial),
        })
    }

    /// Clone the current immutable view without taking a reader lock.
    #[must_use]
    pub fn load(&self) -> Arc<QuillSearchSnapshot> {
        self.current.load_full()
    }

    fn prepare_sealed_manifest(
        &self,
        schema: SchemaDescriptor,
        proposed: &Manifest,
    ) -> Result<PreparedSealedPublication, SnapshotError> {
        let current = self.current.load_full();
        validate_complete_keeper_transition(&current.keeper.loaded_manifest().manifest, proposed)?;
        let snapshot_epoch = current
            .snapshot_epoch()
            .checked_add(1)
            .ok_or(SnapshotError::EpochExhausted)?;
        let (bm25_doc_count, live_doc_count) = manifest_document_counts(proposed)?;
        let field_stats = composite_field_stats(
            schema,
            proposed,
            proposed.segments.is_empty(),
            &[],
            bm25_doc_count,
        )?;
        Ok(PreparedSealedPublication {
            snapshot_epoch,
            expected_keeper_generation: proposed.generation,
            expected_schema_id: proposed.schema_id,
            expected_docid_high_watermark: proposed.docid_high_watermark,
            schema,
            field_stats,
            bm25_doc_count,
            live_doc_count,
        })
    }

    fn prepare_equivalent_sealed_successor(
        &self,
    ) -> Result<PreparedSealedPublication, SnapshotError> {
        let current = self.current.load_full();
        if !current.deltas.is_empty() {
            return Err(SnapshotError::KeeperTransition {
                detail: "concat merge cannot discard active Delta epochs".to_owned(),
            });
        }
        let snapshot_epoch = current
            .snapshot_epoch()
            .checked_add(1)
            .ok_or(SnapshotError::EpochExhausted)?;
        let manifest = &current.keeper.loaded_manifest().manifest;
        let expected_keeper_generation =
            manifest
                .generation
                .checked_add(1)
                .ok_or_else(|| SnapshotError::KeeperTransition {
                    detail: "Keeper MANIFEST generation exhausted".to_owned(),
                })?;
        let mut field_stats = Vec::new();
        field_stats
            .try_reserve_exact(current.field_stats.len())
            .map_err(|_| SnapshotError::Allocation {
                resource: "concat-merge field statistics",
                additional: current.field_stats.len(),
            })?;
        field_stats.extend_from_slice(&current.field_stats);
        Ok(PreparedSealedPublication {
            snapshot_epoch,
            expected_keeper_generation,
            expected_schema_id: manifest.schema_id,
            expected_docid_high_watermark: manifest.docid_high_watermark,
            schema: current.keeper.schema(),
            field_stats: field_stats.into_boxed_slice(),
            bm25_doc_count: current.bm25_doc_count,
            live_doc_count: current.live_doc_count,
        })
    }

    fn install_prepared_sealed(
        &self,
        keeper: Arc<KeeperSnapshot>,
        prepared: PreparedSealedPublication,
    ) -> Arc<QuillSearchSnapshot> {
        let manifest = &keeper.loaded_manifest().manifest;
        assert_eq!(
            manifest.generation, prepared.expected_keeper_generation,
            "Keeper installed an unexpected MANIFEST generation after publication"
        );
        assert_eq!(
            manifest.schema_id, prepared.expected_schema_id,
            "Keeper installed an unexpected schema after publication"
        );
        assert_eq!(
            manifest.docid_high_watermark, prepared.expected_docid_high_watermark,
            "Keeper installed an unexpected Q1 watermark after publication"
        );
        assert!(
            keeper.schema() == prepared.schema,
            "Keeper installed a schema descriptor that differs from the prepared publication"
        );
        assert_eq!(
            keeper.at_seal_doc_count(),
            prepared.bm25_doc_count,
            "Keeper at-seal count differs from the prepared publication"
        );
        assert_eq!(
            keeper.doc_count(),
            prepared.live_doc_count,
            "Keeper live count differs from the prepared publication"
        );
        for expected in &prepared.field_stats {
            let actual_total_tokens = manifest
                .field_stats
                .iter()
                .find(|row| row.field_ord == expected.field_ord)
                .map(|row| row.total_tokens)
                .or_else(|| manifest.segments.is_empty().then_some(0));
            assert_eq!(
                actual_total_tokens,
                Some(expected.total_tokens),
                "Keeper field statistics differ from the prepared publication"
            );
        }

        let next = Arc::new(QuillSearchSnapshot {
            snapshot_epoch: prepared.snapshot_epoch,
            keeper,
            deltas: Box::default(),
            field_stats: prepared.field_stats,
            bm25_doc_count: prepared.bm25_doc_count,
            live_doc_count: prepared.live_doc_count,
        });
        self.current.store(Arc::clone(&next));
        next
    }

    /// Atomically replace Keeper and every shard Delta in one publication.
    ///
    /// This is the seal boundary: callers supply a complete authoritative set,
    /// already quiesced against batch writers. The method retries only to keep
    /// the local epoch monotone if another publication wins the CAS first.
    ///
    /// # Errors
    ///
    /// Returns the same composition failures as [`Self::new`], or epoch
    /// exhaustion.
    pub fn publish_complete(
        &self,
        mut keeper: Arc<KeeperSnapshot>,
        mut deltas: Vec<Arc<DeltaSnapshot>>,
    ) -> Result<Arc<QuillSearchSnapshot>, SnapshotError> {
        loop {
            let current = self.current.load_full();
            validate_complete_keeper_transition(
                &current.keeper.loaded_manifest().manifest,
                &keeper.loaded_manifest().manifest,
            )?;
            let epoch = current
                .snapshot_epoch()
                .checked_add(1)
                .ok_or(SnapshotError::EpochExhausted)?;
            let next = Arc::new(QuillSearchSnapshot::compose(epoch, keeper, deltas)?);
            let previous = self.current.compare_and_swap(&current, Arc::clone(&next));
            if Arc::ptr_eq(&current, &previous) {
                return Ok(next);
            }
            keeper = Arc::clone(&next.keeper);
            deltas = clone_delta_arcs(&next.deltas)?;
        }
    }

    /// Atomically replace one shard's frozen Delta epoch.
    ///
    /// Concurrent shard publications are merged through a compare-and-swap
    /// retry; each retry starts from the latest complete composite view.
    ///
    /// # Errors
    ///
    /// Rejects an unknown shard slot and all ordinary composition failures.
    pub fn publish_delta(
        &self,
        shard: usize,
        mut delta: Arc<DeltaSnapshot>,
    ) -> Result<Arc<QuillSearchSnapshot>, SnapshotError> {
        loop {
            let current = self.current.load_full();
            if shard >= current.deltas.len() {
                return Err(SnapshotError::ShardOutOfBounds {
                    shard,
                    shard_count: current.deltas.len(),
                });
            }
            let epoch = current
                .snapshot_epoch()
                .checked_add(1)
                .ok_or(SnapshotError::EpochExhausted)?;
            let mut deltas = clone_delta_arcs(&current.deltas)?;
            deltas[shard] = delta;
            let next = Arc::new(QuillSearchSnapshot::compose(
                epoch,
                Arc::clone(&current.keeper),
                deltas,
            )?);
            let previous = self.current.compare_and_swap(&current, Arc::clone(&next));
            if Arc::ptr_eq(&current, &previous) {
                return Ok(next);
            }
            delta = Arc::clone(&next.deltas[shard]);
        }
    }
}

fn clone_delta_arcs(
    deltas: &[Arc<DeltaSnapshot>],
) -> Result<Vec<Arc<DeltaSnapshot>>, SnapshotError> {
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(deltas.len())
        .map_err(|_| SnapshotError::Allocation {
            resource: "Delta snapshot table",
            additional: deltas.len(),
        })?;
    cloned.extend(deltas.iter().cloned());
    Ok(cloned)
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

/// Scalar Quill writer plus committed and process-local reader views.
pub struct QuillIndex {
    config: QuillConfig,
    schema: SchemaDescriptor,
    parser: DefaultQueryParser,
    backend: IndexBackend,
    published_snapshot: SnapshotPublisher,
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
        let published_snapshot =
            SnapshotPublisher::new(Arc::new(backend.snapshot().clone()), Vec::new())?;
        Ok(Self {
            config,
            schema,
            parser,
            backend,
            published_snapshot,
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

    /// Current process-local Keeper plus Delta snapshot.
    ///
    /// Readers should pin this `Arc` once and retain it for the complete query.
    /// The scalar search implementation consumes only its Keeper component
    /// until E5.3 supplies Delta cursor adapters.
    #[must_use]
    pub fn search_snapshot(&self) -> Arc<QuillSearchSnapshot> {
        self.published_snapshot.load()
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
        let prepared_publication = self
            .published_snapshot
            .prepare_sealed_manifest(self.schema, &manifest)?;
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
                self.published_snapshot.install_prepared_sealed(
                    Arc::new(self.backend.snapshot().clone()),
                    prepared_publication,
                );
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

    /// Replace one exact committed manifest run with a Q1 concat merge.
    ///
    /// The caller supplies source ids in current manifest order plus a
    /// collision-free output identity. Policy-driven candidate selection and
    /// identity generation remain Keeper E3.7 concerns; this method enforces
    /// that no accumulator or staged publication can race the structural
    /// replacement.
    ///
    /// # Errors
    ///
    /// Rejects uncommitted state, cancellation, a nonconsecutive source run,
    /// codec/invariant failures, or durable publication failure.
    pub async fn concat_merge(
        &mut self,
        cx: &Cx,
        source_segment_ids: &[u64],
        output_segment_id: u64,
        created_unix_s: i64,
    ) -> Result<&KeeperSnapshot, QuillIndexError> {
        check_cancel(cx, "concat merge")?;
        if self.has_uncommitted_changes() {
            return Err(invalid_state(
                "concat merge requires a fully committed scalar index",
            ));
        }
        let next_seal_seq = self
            .next_seal_seq
            .checked_add(1)
            .ok_or_else(|| invalid_state("seal sequence exhausted"))?;
        let prepared_publication = self
            .published_snapshot
            .prepare_equivalent_sealed_successor()?;
        match &mut self.backend {
            IndexBackend::Durable(writer) => {
                writer
                    .concat_merge(cx, source_segment_ids, output_segment_id, created_unix_s)
                    .await?;
            }
            IndexBackend::Memory(snapshot) => {
                check_cancel(cx, "concat merge")?;
                let source_snapshot = snapshot.clone();
                let mut source_ids = Vec::new();
                source_ids
                    .try_reserve_exact(source_segment_ids.len())
                    .map_err(|_| invalid_state("could not allocate concat-merge source ids"))?;
                source_ids.extend_from_slice(source_segment_ids);
                let successor = spawn_blocking(move || {
                    source_snapshot.concat_merge_owned(
                        &source_ids,
                        output_segment_id,
                        created_unix_s,
                    )
                })
                .await?;
                check_cancel(cx, "concat merge")?;
                *snapshot = successor;
            }
        }
        self.published_snapshot.install_prepared_sealed(
            Arc::new(self.backend.snapshot().clone()),
            prepared_publication,
        );
        self.next_seal_seq = next_seal_seq;
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
        let published = self.published_snapshot.load();
        let snapshot = published.keeper_snapshot();
        let query_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_QUERY,
            phase = "query",
            query_len = query.len(),
            segment_count = snapshot.segments().len(),
            doc_count = snapshot.doc_count(),
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
        let published = self.published_snapshot.load();
        let snapshot = published.keeper_snapshot();
        let query_span = tracing::info_span!(
            target: crate::tracing_conventions::TARGET,
            crate::tracing_conventions::ARGUS_QUERY,
            phase = "query",
            collector = "docset",
            query_len = query.len(),
            segment_count = snapshot.segments().len(),
            doc_count = snapshot.doc_count(),
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
    use std::collections::BTreeMap;
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use asupersync::runtime::yield_now;
    use asupersync::types::Budget;
    use asupersync::{LabConfig, LabRuntime};
    #[cfg(feature = "durability")]
    use frankensearch_durability::{DefaultSymbolCodec, DurabilityConfig};

    use super::*;
    use crate::contract::fieldnorm_to_id;
    use crate::delta::{DeltaFieldNorm, DeltaSegment, DeltaTermPosting};
    use crate::keeper::ConcatMergeError;
    use crate::quiver::{StatsSection, aggregate_field_stats};
    use crate::schema::FSFS_CHUNK_SCHEMA;

    const CONCAT_MERGE_QUERIES: [&str; 4] =
        ["rust", "python", "rust OR python", "\"rust ownership\""];
    const Q1_OB2A_QUERIES: [&str; 5] = [
        "shared",
        "left",
        "right",
        "left OR right",
        "\"shared left\"",
    ];
    const KNOWN_SECTION_KINDS: [SectionKind; 10] = [
        SectionKind::TERMDICT,
        SectionKind::POSTINGS,
        SectionKind::POSITIONS,
        SectionKind::BLOCKMAX,
        SectionKind::DOCLEN,
        SectionKind::IDMAP,
        SectionKind::IDHASH,
        SectionKind::NUMERIC,
        SectionKind::STOREDMETA,
        SectionKind::STATS,
    ];

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

    fn apply_alpha_delta(
        delta: &mut DeltaSegment,
        global_docid: u32,
        document_id: &str,
        content_frequency: u32,
    ) {
        let positions = (0..content_frequency).collect::<Vec<_>>();
        let fieldnorms = [
            DeltaFieldNorm {
                field_ord: ID_FIELD,
                raw_length: 1,
                fieldnorm_id: fieldnorm_to_id(1),
            },
            DeltaFieldNorm {
                field_ord: CONTENT_FIELD,
                raw_length: content_frequency,
                fieldnorm_id: fieldnorm_to_id(content_frequency),
            },
            DeltaFieldNorm {
                field_ord: TITLE_FIELD,
                raw_length: 0,
                fieldnorm_id: fieldnorm_to_id(0),
            },
        ];
        let postings = [
            DeltaTermPosting {
                field_ord: ID_FIELD,
                term: document_id.as_bytes(),
                frequency: 1,
                positions: None,
            },
            DeltaTermPosting {
                field_ord: CONTENT_FIELD,
                term: b"alpha",
                frequency: content_frequency,
                positions: Some(&positions),
            },
        ];
        delta
            .apply_document(
                global_docid,
                frankensearch_core::DocId::from(document_id),
                &fieldnorms,
                &postings,
            )
            .expect("apply Delta fixture document");
    }

    fn alpha_snapshot_tuple(snapshot: &QuillSearchSnapshot) -> (u64, u64, u64, u64, u64) {
        let stats = snapshot
            .bm25_field_stats(CONTENT_FIELD)
            .expect("content snapshot statistics");
        (
            snapshot.snapshot_epoch(),
            snapshot.live_doc_count(),
            stats.doc_count,
            stats.total_tokens,
            snapshot
                .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                .expect("alpha document frequency"),
        )
    }

    async fn concat_merge_fixture_index(cx: &Cx) -> QuillIndex {
        let documents = fixture_documents();
        let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
        for (batch_index, document) in documents.iter().enumerate() {
            index
                .index_documents(cx, std::slice::from_ref(document))
                .await
                .expect("accumulate concat-merge leaf");
            index.commit(cx).await.expect("commit concat-merge leaf");
            assert_eq!(
                index.snapshot().segments().len(),
                batch_index + 1,
                "each committed fixture batch must seal one leaf segment",
            );
        }
        index
    }

    struct Q1Ob2aSeal {
        encoded: EncodedSegment,
        field_stats: BTreeMap<u16, (u64, u32)>,
    }

    fn q1_ob2a_documents() -> Vec<IndexableDocument> {
        (0_u32..400)
            .map(|ordinal| {
                let cohort = if ordinal < 100 { "left" } else { "right" };
                IndexableDocument::new(
                    format!("q1-ob2a-{ordinal:03}"),
                    format!("shared {cohort} concat parity"),
                )
                .with_title(format!("{cohort} cohort"))
                .with_metadata("cohort", cohort)
            })
            .collect()
    }

    fn q1_ob2a_doc_ord(first_doc_ord: u32, offset: usize) -> u32 {
        first_doc_ord
            .checked_add(u32::try_from(offset).expect("Q1-OB2a offset fits u32"))
            .expect("Q1-OB2a document ordinal stays in one lease")
    }

    fn seal_q1_ob2a_documents(
        documents: &[IndexableDocument],
        first_doc_ord: u32,
        segment_id: u64,
    ) -> Q1Ob2aSeal {
        let mut accumulator =
            ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("Q1-OB2a accumulator");
        let mut canonical_contents = Vec::with_capacity(documents.len());
        for (offset, document) in documents.iter().enumerate() {
            let doc_ord = q1_ob2a_doc_ord(first_doc_ord, offset);
            let metadata = canonical_metadata(&document.metadata).expect("canonical metadata");
            let ordinal = u64::from(doc_ord).to_le_bytes();
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
            accumulator
                .add_document_with_values(doc_ord, &indexed, &[], &stored)
                .expect("accumulate Q1-OB2a document");
            canonical_contents.push(
                canonical_document_preimage(document, &metadata)
                    .expect("canonical Q1-OB2a content"),
            );
        }

        let doc_count = u32::try_from(documents.len()).expect("Q1-OB2a document count fits u32");
        let field_stats = accumulator
            .fields()
            .iter()
            .map(|field| (field.field_ord(), (field.total_tokens(), doc_count)))
            .collect::<BTreeMap<_, _>>();
        let identities = documents
            .iter()
            .zip(&canonical_contents)
            .enumerate()
            .map(|(offset, (document, canonical_content))| {
                FlushDocumentInput::from_canonical_content(
                    q1_ob2a_doc_ord(first_doc_ord, offset),
                    &document.id,
                    canonical_content,
                )
            })
            .collect::<Vec<_>>();
        let encoded = flush_accumulator_with_mode(
            &accumulator,
            FlushSegmentInput {
                segment_id,
                lease_docid_base: 0,
                created_unix_s: 0,
                engine_version: CURRENT_ENGINE_VERSION,
                documents: &identities,
            },
            FlushMode::Scalar,
        )
        .expect("seal Q1-OB2a segment");
        Q1Ob2aSeal {
            encoded,
            field_stats,
        }
    }

    fn q1_ob2a_owned_index(
        encoded_segments: Vec<EncodedSegment>,
        field_stats: Vec<ManifestFieldStats>,
    ) -> QuillIndex {
        let genesis = KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("Q1-OB2a genesis");
        let mut manifest = genesis.next_manifest().expect("Q1-OB2a manifest");
        manifest.docid_high_watermark = u64::from(DOC_ORDS_PER_LEASE);
        manifest.segments = encoded_segments
            .iter()
            .enumerate()
            .map(|(index, encoded)| {
                manifest_segment(
                    encoded,
                    u64::try_from(index + 1).expect("Q1-OB2a seal sequence fits u64"),
                )
            })
            .collect();
        manifest.field_stats = field_stats;
        let snapshot = genesis
            .publish_owned_segments(&manifest, encoded_segments)
            .expect("publish Q1-OB2a fixture");
        QuillIndex::from_backend(
            IndexBackend::Memory(snapshot),
            DEFAULT_SCHEMA,
            deterministic_config(),
        )
        .expect("bind Q1-OB2a fixture index")
    }

    fn q1_ob2a_fixture_indexes() -> (QuillIndex, QuillIndex) {
        let documents = q1_ob2a_documents();
        let left = seal_q1_ob2a_documents(&documents[..100], 0, 0x0b2a_0001);
        let right = seal_q1_ob2a_documents(&documents[100..], 100, 0x0b2a_0002);
        let monolithic = seal_q1_ob2a_documents(&documents, 0, 0x0b2a_0003);

        let first_segment_stats =
            merge_field_stats(&[], &left.field_stats).expect("left Q1-OB2a stats");
        let combined_segment_stats = merge_field_stats(&first_segment_stats, &right.field_stats)
            .expect("aggregate leaf stats");
        let monolithic_stats =
            merge_field_stats(&[], &monolithic.field_stats).expect("monolithic Q1-OB2a stats");
        assert_eq!(combined_segment_stats, monolithic_stats);

        let leaves = q1_ob2a_owned_index(vec![left.encoded, right.encoded], combined_segment_stats);
        let monolithic = q1_ob2a_owned_index(vec![monolithic.encoded], monolithic_stats);
        (leaves, monolithic)
    }

    fn q1_ob2a_decoded_terms(index: &QuillIndex) -> Vec<(u16, Vec<u8>, u32, Vec<Posting>)> {
        let mut decoded = BTreeMap::<(u16, Vec<u8>), (u32, Vec<Posting>)>::new();
        for segment in index.snapshot().segments() {
            let dictionary = open_dictionary(segment, DEFAULT_SCHEMA).expect("Q1-OB2a TERMDICT");
            let limit = usize::try_from(dictionary.term_count()).expect("term count fits usize");
            let terms = dictionary
                .cursor()
                .expect("Q1-OB2a term cursor")
                .collect_bounded(limit)
                .expect("materialize Q1-OB2a terms");
            for term in terms {
                let postings =
                    open_owned_cursor(segment, DEFAULT_SCHEMA, term.field_ord, &term.term, false)
                        .expect("decode Q1-OB2a postings")
                        .postings;
                let entry = decoded.entry((term.field_ord, term.term)).or_default();
                entry.0 = entry
                    .0
                    .checked_add(term.metadata.doc_freq)
                    .expect("Q1-OB2a aggregate df fits u32");
                entry.1.extend(postings);
            }
        }
        decoded
            .into_iter()
            .map(|((field_ord, term), (doc_freq, postings))| (field_ord, term, doc_freq, postings))
            .collect()
    }

    fn q1_ob2a_decoded_stats(segment: &RecoveredSegment) -> StatsSection {
        let expected_fields = term_field_ords(DEFAULT_SCHEMA);
        StatsSection::parse(
            required_section(segment, SectionKind::STATS).expect("Q1-OB2a STATS bytes"),
            &expected_fields,
            segment.manifest().doc_count,
        )
        .expect("decode Q1-OB2a STATS")
    }

    fn q1_ob2a_query_evidence(index: &QuillIndex, cx: &Cx) -> Vec<(QuillSearchResult, Vec<u32>)> {
        Q1_OB2A_QUERIES
            .iter()
            .map(|query| {
                let ranked = index
                    .search_paginated(cx, query, 500, 0, true)
                    .expect("Q1-OB2a ranked query");
                let docids = index
                    .collect_docids(cx, query)
                    .expect("Q1-OB2a scoreless query");
                (ranked, docids)
            })
            .collect()
    }

    fn committed_segment_ids(index: &QuillIndex) -> Vec<u64> {
        index
            .snapshot()
            .segments()
            .iter()
            .map(|segment| segment.manifest().segment_id)
            .collect()
    }

    fn fresh_merge_segment_id(index: &QuillIndex, seed: u64) -> u64 {
        let mut candidate = seed;
        loop {
            if index
                .snapshot()
                .segments()
                .iter()
                .all(|segment| segment.manifest().segment_id != candidate)
            {
                return candidate;
            }
            candidate = candidate
                .checked_add(1)
                .expect("concat-merge test segment-id space exhausted");
        }
    }

    fn concat_merge_query_results(index: &QuillIndex, cx: &Cx) -> Vec<QuillSearchResult> {
        CONCAT_MERGE_QUERIES
            .iter()
            .map(|query| {
                index
                    .search_paginated(cx, query, 10, 0, true)
                    .expect("concat-merge fixture query")
            })
            .collect()
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
    fn snapshot_publication_types_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<DeltaSnapshot>();
        assert_send_sync::<QuillSearchSnapshot>();
        assert_send_sync::<SnapshotPublisher>();
    }

    #[test]
    fn composite_snapshot_uses_keeper_at_seal_and_live_delta_statistics() {
        run_with_cx(|cx| async move {
            let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            let sealed_documents = [
                IndexableDocument::new("sealed-live", "alpha"),
                IndexableDocument::new("sealed-deleted", "alpha"),
            ];
            index
                .index_documents(&cx, &sealed_documents)
                .await
                .expect("accumulate sealed fixtures");
            index.commit(&cx).await.expect("commit sealed fixtures");

            let committed = index.snapshot().clone();
            let mut tombstoned_manifest = committed.next_manifest().expect("next manifest");
            assert!(
                committed
                    .delete_document(&mut tombstoned_manifest, "sealed-deleted")
                    .expect("stage sealed delete")
            );
            let tombstoned = committed
                .publish_owned_segments(&tombstoned_manifest, Vec::new())
                .expect("publish tombstone-only successor");
            assert_eq!(tombstoned.at_seal_doc_count(), 2);
            assert_eq!(tombstoned.doc_count(), 1);
            assert_eq!(
                snapshot_doc_freq(&tombstoned, DEFAULT_SCHEMA, CONTENT_FIELD, b"alpha")
                    .expect("sealed alpha df"),
                2,
                "sealed tombstones remain in BM25 df until compaction"
            );

            let generation = tombstoned.loaded_manifest().manifest.generation;
            let first_base = u64::from(DOC_ORDS_PER_LEASE);
            let second_base = first_base * 2;
            let mut first = DeltaSegment::new(DEFAULT_SCHEMA, first_base, usize::MAX)
                .expect("first Delta shard");
            apply_alpha_delta(
                &mut first,
                u32::try_from(first_base).expect("first Delta docid"),
                "delta-a",
                2,
            );
            let mut second = DeltaSegment::new(DEFAULT_SCHEMA, second_base, usize::MAX)
                .expect("second Delta shard");
            apply_alpha_delta(
                &mut second,
                u32::try_from(second_base).expect("second Delta docid"),
                "delta-b",
                3,
            );

            let snapshot = QuillSearchSnapshot::compose(
                11,
                Arc::new(tombstoned),
                vec![
                    Arc::new(first.freeze(generation)),
                    Arc::new(second.freeze(generation)),
                ],
            )
            .expect("compose Keeper plus Delta snapshot");
            assert_eq!(snapshot.snapshot_epoch(), 11);
            assert_eq!(snapshot.bm25_doc_count(), 4);
            assert_eq!(snapshot.live_doc_count(), 3);
            assert_eq!(
                snapshot
                    .bm25_field_stats(CONTENT_FIELD)
                    .expect("content stats"),
                SnapshotFieldStats {
                    field_ord: CONTENT_FIELD,
                    total_tokens: 7,
                    doc_count: 4,
                }
            );
            assert_eq!(
                snapshot
                    .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                    .expect("composite alpha df"),
                4
            );
        });
    }

    #[test]
    fn three_delta_upserts_contribute_one_live_bm25_row() {
        run_with_cx(|cx| async move {
            let keeper =
                Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
            let generation = keeper.loaded_manifest().manifest.generation;
            let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
            apply_alpha_delta(&mut delta, 0, "same-id", 1);
            apply_alpha_delta(&mut delta, 1, "same-id", 2);
            apply_alpha_delta(&mut delta, 2, "same-id", 3);
            assert_eq!(delta.physical_document_count(), 3);
            assert_eq!(delta.live_document_count(), 1);
            assert_eq!(
                delta
                    .find_term(CONTENT_FIELD, b"alpha")
                    .expect("alpha Delta term")
                    .live_doc_freq(),
                1
            );

            let snapshot =
                QuillSearchSnapshot::compose(0, keeper, vec![Arc::new(delta.freeze(generation))])
                    .expect("compose repeated-upsert snapshot");
            assert_eq!(alpha_snapshot_tuple(&snapshot), (0, 1, 1, 3, 1));
            let pre_stats = snapshot
                .bm25_field_stats(CONTENT_FIELD)
                .expect("pre-seal content stats");
            let pre_df = snapshot
                .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                .expect("pre-seal alpha df");
            let mut pre_scorer = ReferenceScorer::term(
                TermScorer::new(
                    OwnedPostingCursor {
                        postings: vec![Posting::new(2, 3)],
                        positions: None,
                        cursor: 0,
                        segment_num_docs: 1,
                    },
                    OwnedFieldNorms {
                        field_ord: CONTENT_FIELD,
                        docid_lo: 2,
                        values: vec![fieldnorm_to_id(3)],
                    },
                    Bm25FieldSnapshot::new(pre_stats).expect("pre-seal BM25 snapshot"),
                    pre_df,
                    1.0,
                )
                .expect("pre-seal term scorer"),
            );

            let mut sealed = QuillIndex::in_memory(deterministic_config()).expect("sealed index");
            sealed
                .index_documents(
                    &cx,
                    &[IndexableDocument::new("same-id", "alpha alpha alpha")],
                )
                .await
                .expect("accumulate final logical upsert row");
            sealed
                .commit(&cx)
                .await
                .expect("seal final logical upsert row");
            let keeper = sealed.snapshot();
            let post_stats = snapshot_field(keeper, CONTENT_FIELD).expect("post-seal stats");
            let post_df = snapshot_doc_freq(keeper, DEFAULT_SCHEMA, CONTENT_FIELD, b"alpha")
                .expect("post-seal alpha df");
            assert_eq!(pre_stats, post_stats);
            assert_eq!(pre_df, post_df);

            let mut post_scorer = lower_term(
                &keeper.segments()[0],
                keeper,
                DEFAULT_SCHEMA,
                CONTENT_FIELD,
                b"alpha",
                1.0,
            )
            .expect("post-seal term scorer");
            assert_eq!(
                pre_scorer.score().expect("pre-seal score").to_bits(),
                post_scorer.score().expect("post-seal score").to_bits(),
                "the final live upsert must keep bit-identical BM25 across seal"
            );
        });
    }

    #[test]
    fn snapshot_composition_rejects_stale_schema_and_lease_drift() {
        let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
        let generation = keeper.loaded_manifest().manifest.generation;

        let stale = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
            .expect("stale Delta")
            .freeze(generation + 1);
        let Err(error) =
            QuillSearchSnapshot::compose(0, Arc::clone(&keeper), vec![Arc::new(stale)])
        else {
            panic!("stale Delta generation was accepted");
        };
        assert!(matches!(
            error,
            SnapshotError::KeeperGenerationMismatch { .. }
        ));

        let wrong_schema = DeltaSegment::new(FSFS_CHUNK_SCHEMA, 0, usize::MAX)
            .expect("wrong-schema Delta")
            .freeze(generation);
        let Err(error) =
            QuillSearchSnapshot::compose(0, Arc::clone(&keeper), vec![Arc::new(wrong_schema)])
        else {
            panic!("wrong-schema Delta was accepted");
        };
        assert!(matches!(error, SnapshotError::SchemaMismatch { .. }));

        let first = Arc::new(
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
                .expect("first overlapping Delta")
                .freeze(generation),
        );
        let second = Arc::new(
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
                .expect("second overlapping Delta")
                .freeze(generation),
        );
        let Err(error) =
            QuillSearchSnapshot::compose(0, Arc::clone(&keeper), vec![Arc::clone(&first), second])
        else {
            panic!("overlapping Delta leases were accepted");
        };
        assert!(matches!(
            error,
            SnapshotError::OverlappingDeltaLeases { .. }
        ));

        let publisher = SnapshotPublisher::new(keeper, vec![first]).expect("snapshot publisher");
        let replacement = Arc::new(
            DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX)
                .expect("replacement Delta")
                .freeze(generation),
        );
        assert!(matches!(
            publisher.publish_delta(1, replacement),
            Err(SnapshotError::ShardOutOfBounds {
                shard: 1,
                shard_count: 1
            })
        ));
    }

    #[test]
    fn snapshot_composition_rejects_keeper_delta_occupied_range_overlap() {
        run_with_cx(|cx| async move {
            let mut index = QuillIndex::in_memory(deterministic_config()).expect("memory index");
            index
                .index_documents(&cx, &[IndexableDocument::new("sealed", "alpha")])
                .await
                .expect("accumulate sealed row");
            index.commit(&cx).await.expect("commit sealed row");

            let keeper = Arc::new(index.snapshot().clone());
            let generation = keeper.loaded_manifest().manifest.generation;
            let segment = &keeper.loaded_manifest().manifest.segments[0];
            assert_eq!((segment.docid_lo, segment.docid_hi), (0, 1));

            let mut overlapping =
                DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("overlapping Delta");
            apply_alpha_delta(&mut overlapping, 0, "duplicate-docid", 1);
            let Err(error) = QuillSearchSnapshot::compose(
                0,
                Arc::clone(&keeper),
                vec![Arc::new(overlapping.freeze(generation))],
            ) else {
                panic!("occupied Keeper/Delta docid overlap was accepted");
            };
            assert_eq!(
                error,
                SnapshotError::KeeperDeltaDocidOverlap {
                    delta_lo: 0,
                    delta_hi: 1,
                    segment_id: segment.segment_id,
                    keeper_lo: 0,
                    keeper_hi: 1,
                }
            );

            let mut continuation =
                DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("same-lease Delta");
            apply_alpha_delta(&mut continuation, 1, "nonoverlapping-docid", 1);
            QuillSearchSnapshot::compose(
                0,
                keeper,
                vec![Arc::new(continuation.freeze(generation))],
            )
            .expect("actual same-lease continuation range does not overlap Keeper");
        });
    }

    #[test]
    fn failed_complete_publications_leave_the_current_epoch_unchanged() {
        run_with_cx(|cx| async move {
            let genesis = Arc::new(
                KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper snapshot"),
            );
            let genesis_generation = genesis.loaded_manifest().manifest.generation;
            let snapshot_source = SnapshotPublisher::new(Arc::clone(&genesis), Vec::new())
                .expect("genesis publisher");

            let mut first = QuillIndex::in_memory(deterministic_config()).expect("first index");
            first
                .index_documents(&cx, &[IndexableDocument::new("first", "alpha")])
                .await
                .expect("accumulate first successor");
            first.commit(&cx).await.expect("publish first successor");
            let successor = Arc::new(first.snapshot().clone());

            let before = snapshot_source.load();
            let stale_delta = Arc::new(
                DeltaSegment::new(DEFAULT_SCHEMA, DOC_ORDS_PER_LEASE.into(), usize::MAX)
                    .expect("stale Delta")
                    .freeze(genesis_generation),
            );
            assert!(matches!(
                snapshot_source.publish_complete(Arc::clone(&successor), vec![stale_delta]),
                Err(SnapshotError::KeeperGenerationMismatch { .. })
            ));
            assert!(Arc::ptr_eq(&before, &snapshot_source.load()));
            assert_eq!(alpha_snapshot_tuple(&before), (0, 0, 0, 0, 0));

            let accepted = snapshot_source
                .publish_complete(Arc::clone(&successor), Vec::new())
                .expect("publish valid Keeper successor");
            assert_eq!(alpha_snapshot_tuple(&accepted), (1, 1, 1, 1, 1));

            let Err(rollback) = snapshot_source.publish_complete(genesis, Vec::new()) else {
                panic!("stale Keeper rollback was accepted");
            };
            assert_eq!(
                rollback,
                SnapshotError::KeeperGenerationRegression {
                    current: successor.loaded_manifest().manifest.generation,
                    proposed: genesis_generation,
                }
            );
            assert!(Arc::ptr_eq(&accepted, &snapshot_source.load()));

            let mut second = QuillIndex::in_memory(deterministic_config()).expect("second index");
            second
                .index_documents(&cx, &[IndexableDocument::new("second", "beta")])
                .await
                .expect("accumulate colliding successor");
            second
                .commit(&cx)
                .await
                .expect("publish colliding successor fixture");
            let collision = Arc::new(second.snapshot().clone());
            let Err(collision) = snapshot_source.publish_complete(collision, Vec::new()) else {
                panic!("same-generation divergent MANIFEST was accepted");
            };
            assert_eq!(
                collision,
                SnapshotError::KeeperGenerationCollision {
                    generation: successor.loaded_manifest().manifest.generation,
                }
            );
            assert!(Arc::ptr_eq(&accepted, &snapshot_source.load()));
        });
    }

    #[test]
    fn publisher_retains_a_held_epoch_until_its_last_reader_drops() {
        let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
        let generation = keeper.loaded_manifest().manifest.generation;
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
        let publisher = SnapshotPublisher::new(
            Arc::clone(&keeper),
            vec![Arc::new(delta.freeze(generation))],
        )
        .expect("snapshot publisher");
        let held = publisher.load();
        let weak = Arc::downgrade(&held);

        apply_alpha_delta(&mut delta, 0, "new", 1);
        let next_epoch = publisher
            .publish_delta(0, Arc::new(delta.freeze(generation)))
            .expect("publish next Delta epoch");
        assert_eq!(alpha_snapshot_tuple(&held), (0, 0, 0, 0, 0));
        assert_eq!(alpha_snapshot_tuple(&next_epoch), (1, 1, 1, 1, 1));
        assert_eq!(alpha_snapshot_tuple(&publisher.load()), (1, 1, 1, 1, 1));
        assert!(weak.upgrade().is_some());

        drop(held);
        assert!(
            weak.upgrade().is_none(),
            "the old composite must retire after its final reader Arc drops"
        );
    }

    #[test]
    fn labruntime_readers_observe_only_complete_composite_epochs() {
        let keeper = Arc::new(KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper"));
        let generation = keeper.loaded_manifest().manifest.generation;
        let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
        let publisher = Arc::new(
            SnapshotPublisher::new(
                Arc::clone(&keeper),
                vec![Arc::new(delta.freeze(generation))],
            )
            .expect("snapshot publisher"),
        );
        apply_alpha_delta(&mut delta, 0, "new", 1);
        let next_delta = Arc::new(delta.freeze(generation));
        let reader_saw_old = Arc::new(AtomicBool::new(false));
        let writer_done = Arc::new(AtomicBool::new(false));

        let mut lab = LabRuntime::new(LabConfig::new(0xe5_2002).max_steps(100_000));
        let region = lab.state.create_root_region(Budget::INFINITE);

        let reader_publisher = Arc::clone(&publisher);
        let reader_started = Arc::clone(&reader_saw_old);
        let reader_done = Arc::clone(&writer_done);
        let (reader, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                assert_eq!(
                    alpha_snapshot_tuple(&reader_publisher.load()),
                    (0, 0, 0, 0, 0)
                );
                reader_started.store(true, Ordering::SeqCst);
                while !reader_done.load(Ordering::SeqCst) {
                    let observed = alpha_snapshot_tuple(&reader_publisher.load());
                    assert!(
                        observed == (0, 0, 0, 0, 0) || observed == (1, 1, 1, 1, 1),
                        "reader observed a torn composite tuple: {observed:?}"
                    );
                    yield_now().await;
                }
                assert_eq!(
                    alpha_snapshot_tuple(&reader_publisher.load()),
                    (1, 1, 1, 1, 1)
                );
            })
            .expect("create snapshot reader task");

        let writer_publisher = Arc::clone(&publisher);
        let writer_started = Arc::clone(&reader_saw_old);
        let writer_finished = Arc::clone(&writer_done);
        let (writer, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                while !writer_started.load(Ordering::SeqCst) {
                    yield_now().await;
                }
                writer_publisher
                    .publish_delta(0, next_delta)
                    .expect("publish complete Delta epoch");
                writer_finished.store(true, Ordering::SeqCst);
            })
            .expect("create snapshot writer task");

        lab.scheduler.lock().schedule(reader, 0);
        lab.step_for_test();
        assert!(reader_saw_old.load(Ordering::SeqCst));
        lab.scheduler.lock().schedule(writer, 0);
        let report = lab.run_until_quiescent_with_report();

        assert!(report.quiescent, "LabRuntime must reach quiescence");
        assert!(report.oracle_report.all_passed(), "oracles must pass");
        assert!(report.invariant_violations.is_empty());
    }

    #[test]
    fn delta_and_equivalent_sealed_term_have_identical_score_bits() {
        run_with_cx(|cx| async move {
            let keeper = Arc::new(
                KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("genesis Keeper snapshot"),
            );
            let generation = keeper.loaded_manifest().manifest.generation;
            let mut delta = DeltaSegment::new(DEFAULT_SCHEMA, 0, usize::MAX).expect("Delta shard");
            apply_alpha_delta(&mut delta, 0, "doc", 1);
            let composite =
                QuillSearchSnapshot::compose(0, keeper, vec![Arc::new(delta.freeze(generation))])
                    .expect("pre-seal composite snapshot");
            let pre_stats = composite
                .bm25_field_stats(CONTENT_FIELD)
                .expect("pre-seal content stats");
            let pre_df = composite
                .bm25_doc_freq(CONTENT_FIELD, b"alpha")
                .expect("pre-seal alpha df");
            let mut pre_scorer = ReferenceScorer::term(
                TermScorer::new(
                    OwnedPostingCursor {
                        postings: vec![Posting::new(0, 1)],
                        positions: None,
                        cursor: 0,
                        segment_num_docs: 1,
                    },
                    OwnedFieldNorms {
                        field_ord: CONTENT_FIELD,
                        docid_lo: 0,
                        values: vec![fieldnorm_to_id(1)],
                    },
                    Bm25FieldSnapshot::new(pre_stats).expect("pre-seal BM25 snapshot"),
                    pre_df,
                    1.0,
                )
                .expect("pre-seal term scorer"),
            );

            let mut sealed = QuillIndex::in_memory(deterministic_config()).expect("sealed index");
            sealed
                .index_documents(&cx, &[IndexableDocument::new("doc", "alpha")])
                .await
                .expect("accumulate equivalent sealed document");
            sealed.commit(&cx).await.expect("seal equivalent document");
            let keeper = sealed.snapshot();
            let post_stats = snapshot_field(keeper, CONTENT_FIELD).expect("post-seal stats");
            let post_df = snapshot_doc_freq(keeper, DEFAULT_SCHEMA, CONTENT_FIELD, b"alpha")
                .expect("post-seal alpha df");
            assert_eq!(pre_stats, post_stats);
            assert_eq!(pre_df, post_df);

            let mut post_scorer = lower_term(
                &keeper.segments()[0],
                keeper,
                DEFAULT_SCHEMA,
                CONTENT_FIELD,
                b"alpha",
                1.0,
            )
            .expect("post-seal term scorer");
            assert_eq!(
                pre_scorer.score().expect("pre-seal score").to_bits(),
                post_scorer.score().expect("post-seal score").to_bits(),
                "BM25 score must be bit-identical across the seal boundary"
            );
        });
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
    fn in_memory_concat_merge_preserves_query_results_scores_and_docids() {
        run_with_cx(|cx| async move {
            let mut index = concat_merge_fixture_index(&cx).await;
            let source_ids = committed_segment_ids(&index);
            assert_eq!(source_ids.len(), 3);

            let generation_before = index.snapshot().loaded_manifest().manifest.generation;
            let results_before = concat_merge_query_results(&index, &cx);
            let docids_before = index
                .collect_docids(&cx, "rust OR python")
                .expect("collect pre-merge docids");
            let output_segment_id = fresh_merge_segment_id(&index, 0xc011_ca7e_0000_0001);

            index
                .concat_merge(&cx, &source_ids, output_segment_id, 17)
                .await
                .expect("merge the complete in-memory leaf run");

            assert_eq!(index.snapshot().segments().len(), 1);
            assert_eq!(
                index.snapshot().loaded_manifest().manifest.generation,
                generation_before + 1,
            );
            assert_eq!(
                concat_merge_query_results(&index, &cx),
                results_before,
                "ranked hits, exact scores, global docids, and counts must survive merge",
            );
            assert_eq!(
                index
                    .collect_docids(&cx, "rust OR python")
                    .expect("collect post-merge docids"),
                docids_before,
            );
            let merge_seal_seq = index.snapshot().segments()[0].manifest().seal_seq;

            index
                .index_documents(
                    &cx,
                    &[IndexableDocument::new(
                        "after-merge",
                        "ordinary sealing remains available after concat merge",
                    )],
                )
                .await
                .expect("accumulate after concat merge");
            index
                .commit(&cx)
                .await
                .expect("ordinary commit after concat merge");
            assert_eq!(
                index
                    .snapshot()
                    .segments()
                    .iter()
                    .map(|segment| segment.manifest().seal_seq)
                    .collect::<Vec<_>>(),
                [merge_seal_seq, merge_seal_seq + 1],
                "concat output and the next ordinary seal need distinct monotone sequences",
            );
        });
    }

    #[test]
    fn concat_merge_matches_fresh_monolithic_df_100_plus_300_with_same_docids() {
        run_with_cx(|cx| async move {
            let (mut leaves, monolithic) = q1_ob2a_fixture_indexes();
            assert_eq!(
                leaves
                    .snapshot()
                    .segments()
                    .iter()
                    .map(|segment| (segment.manifest().docid_lo, segment.manifest().docid_hi))
                    .collect::<Vec<_>>(),
                [(0, 100), (100, 400)],
                "Q1-OB2a leaves must be adjacent inside one lease",
            );
            assert_eq!(monolithic.snapshot().segments()[0].manifest().docid_lo, 0);
            assert_eq!(monolithic.snapshot().segments()[0].manifest().docid_hi, 400);

            let source_shared_dfs = leaves
                .snapshot()
                .segments()
                .iter()
                .map(|segment| {
                    open_dictionary(segment, DEFAULT_SCHEMA)
                        .expect("source Q1-OB2a TERMDICT")
                        .lookup(CONTENT_FIELD, b"shared")
                        .expect("source shared lookup")
                        .expect("source shared term")
                        .metadata
                        .doc_freq
                })
                .collect::<Vec<_>>();
            assert_eq!(source_shared_dfs, [100, 300]);
            assert_eq!(
                snapshot_doc_freq(
                    monolithic.snapshot(),
                    DEFAULT_SCHEMA,
                    CONTENT_FIELD,
                    b"shared",
                )
                .expect("monolithic shared df"),
                400,
            );

            let source_stats = leaves
                .snapshot()
                .segments()
                .iter()
                .map(q1_ob2a_decoded_stats)
                .collect::<Vec<_>>();
            let source_aggregate =
                aggregate_field_stats(source_stats.iter()).expect("aggregate source STATS");
            let monolithic_stats = q1_ob2a_decoded_stats(&monolithic.snapshot().segments()[0]);
            let monolithic_aggregate = monolithic_stats
                .rows()
                .iter()
                .map(|row| SnapshotFieldStats {
                    field_ord: row.field_ord,
                    total_tokens: row.total_tokens,
                    doc_count: u64::from(row.doc_count),
                })
                .collect::<Vec<_>>();
            assert_eq!(source_aggregate, monolithic_aggregate);
            assert_eq!(
                leaves.snapshot().loaded_manifest().manifest.field_stats,
                monolithic.snapshot().loaded_manifest().manifest.field_stats,
            );

            let monolithic_terms = q1_ob2a_decoded_terms(&monolithic);
            assert_eq!(
                q1_ob2a_decoded_terms(&leaves),
                monolithic_terms,
                "source terms and decoded postings must match fresh monolithic indexing",
            );
            let monolithic_evidence = q1_ob2a_query_evidence(&monolithic, &cx);
            assert_eq!(
                q1_ob2a_query_evidence(&leaves, &cx),
                monolithic_evidence,
                "source leaves and fresh monolithic seal must have exact scores, ids, docids, and counts",
            );
            assert_eq!(
                leaves
                    .collect_docids(&cx, "shared")
                    .expect("source shared docids"),
                (0_u32..400).collect::<Vec<_>>(),
            );

            let source_ids = committed_segment_ids(&leaves);
            leaves
                .concat_merge(&cx, &source_ids, 0x0b2a_0004, 0)
                .await
                .expect("Q1-OB2a concat merge");
            assert_eq!(leaves.snapshot().segments().len(), 1);
            assert_eq!(
                snapshot_doc_freq(leaves.snapshot(), DEFAULT_SCHEMA, CONTENT_FIELD, b"shared")
                    .expect("merged shared df"),
                400,
            );
            assert_eq!(
                q1_ob2a_decoded_terms(&leaves),
                monolithic_terms,
                "every merged term and decoded posting must match fresh monolithic indexing",
            );
            assert_eq!(
                q1_ob2a_decoded_stats(&leaves.snapshot().segments()[0]),
                monolithic_stats,
                "merged decoded STATS must match fresh monolithic indexing",
            );
            assert_eq!(
                leaves.snapshot().loaded_manifest().manifest.field_stats,
                monolithic.snapshot().loaded_manifest().manifest.field_stats,
            );
            assert_eq!(
                q1_ob2a_query_evidence(&leaves, &cx),
                monolithic_evidence,
                "merged and monolithic query scores, ids, docids, counts, and docsets must match",
            );
        });
    }

    #[test]
    fn concat_merge_rejects_reversed_and_skipped_runs_without_publication() {
        run_with_cx(|cx| async move {
            let mut index = concat_merge_fixture_index(&cx).await;
            let source_ids = committed_segment_ids(&index);
            assert_eq!(source_ids.len(), 3);
            let manifest_before = index.snapshot().loaded_manifest().manifest.clone();
            let output_segment_id = fresh_merge_segment_id(&index, 0xc011_ca7e_0000_0010);

            let reversed = [source_ids[1], source_ids[0]];
            let Err(error) = index
                .concat_merge(&cx, &reversed, output_segment_id, 19)
                .await
            else {
                panic!("reversed concat-merge run unexpectedly published");
            };
            assert!(matches!(
                error,
                QuillIndexError::Keeper(KeeperError::ConcatMerge {
                    source: ConcatMergeError::NonConsecutiveSources { .. }
                })
            ));
            assert_eq!(
                &index.snapshot().loaded_manifest().manifest,
                &manifest_before,
                "reversed rejection must not advance the snapshot generation",
            );

            let skipped = [source_ids[0], source_ids[2]];
            let Err(error) = index
                .concat_merge(&cx, &skipped, output_segment_id, 23)
                .await
            else {
                panic!("skipped concat-merge run unexpectedly published");
            };
            assert!(matches!(
                error,
                QuillIndexError::Keeper(KeeperError::ConcatMerge {
                    source: ConcatMergeError::NonConsecutiveSources { .. }
                })
            ));
            assert_eq!(
                &index.snapshot().loaded_manifest().manifest,
                &manifest_before,
                "skipped-run rejection must leave the exact snapshot unchanged",
            );

            let wrapped = [source_ids[1], source_ids[2], source_ids[0]];
            let Err(error) = index
                .concat_merge(&cx, &wrapped, output_segment_id, 29)
                .await
            else {
                panic!("manifest-wrapping concat-merge run unexpectedly published");
            };
            assert!(matches!(
                error,
                QuillIndexError::Keeper(KeeperError::ConcatMerge {
                    source: ConcatMergeError::SourceRunPastManifest {
                        position: 2,
                        actual,
                    }
                }) if actual == source_ids[0]
            ));
            assert_eq!(
                &index.snapshot().loaded_manifest().manifest,
                &manifest_before,
                "wrapped-run rejection must leave the exact snapshot unchanged",
            );
        });
    }

    #[test]
    fn concat_merge_direct_and_associated_schedules_match() {
        run_with_cx(|cx| async move {
            let mut direct = concat_merge_fixture_index(&cx).await;
            let mut left_associated = concat_merge_fixture_index(&cx).await;
            let mut right_associated = concat_merge_fixture_index(&cx).await;
            let leaf_ids = committed_segment_ids(&direct);
            assert_eq!(leaf_ids, committed_segment_ids(&left_associated));
            assert_eq!(leaf_ids, committed_segment_ids(&right_associated));
            assert_eq!(leaf_ids.len(), 3);

            let results_before = concat_merge_query_results(&direct, &cx);
            assert_eq!(
                concat_merge_query_results(&left_associated, &cx),
                results_before,
            );
            assert_eq!(
                concat_merge_query_results(&right_associated, &cx),
                results_before,
            );
            let final_segment_id = fresh_merge_segment_id(&direct, 0xc011_ca7e_0000_0020);
            let intermediate_segment_id =
                fresh_merge_segment_id(&left_associated, 0xc011_ca7e_0000_0030);
            let right_intermediate_segment_id =
                fresh_merge_segment_id(&right_associated, 0xc011_ca7e_0000_0040);

            direct
                .concat_merge(&cx, &leaf_ids, final_segment_id, 31)
                .await
                .expect("direct three-leaf concat merge");
            left_associated
                .concat_merge(&cx, &leaf_ids[..2], intermediate_segment_id, 29)
                .await
                .expect("left-associated first concat merge");
            let left_final_sources = [intermediate_segment_id, leaf_ids[2]];
            left_associated
                .concat_merge(&cx, &left_final_sources, final_segment_id, 31)
                .await
                .expect("left-associated final concat merge");
            right_associated
                .concat_merge(&cx, &leaf_ids[1..], right_intermediate_segment_id, 29)
                .await
                .expect("right-associated first concat merge");
            let right_final_sources = [leaf_ids[0], right_intermediate_segment_id];
            right_associated
                .concat_merge(&cx, &right_final_sources, final_segment_id, 31)
                .await
                .expect("right-associated final concat merge");

            assert_eq!(direct.snapshot().segments().len(), 1);
            assert_eq!(left_associated.snapshot().segments().len(), 1);
            assert_eq!(right_associated.snapshot().segments().len(), 1);
            let direct_segment = &direct.snapshot().segments()[0];
            let left_segment = &left_associated.snapshot().segments()[0];
            let right_segment = &right_associated.snapshot().segments()[0];
            assert_eq!(
                direct_segment.source_bytes(),
                left_segment.source_bytes(),
                "the complete merged FSLX image must be schedule-independent",
            );
            assert_eq!(
                direct_segment.source_bytes(),
                right_segment.source_bytes(),
                "right-associated merge must emit the same complete FSLX image",
            );
            for kind in KNOWN_SECTION_KINDS {
                assert_eq!(
                    direct_segment.section(kind).expect("direct merged section"),
                    left_segment
                        .section(kind)
                        .expect("left-associated merged section"),
                    "section {} differs by merge schedule",
                    kind.raw(),
                );
                assert_eq!(
                    direct_segment.section(kind).expect("direct merged section"),
                    right_segment
                        .section(kind)
                        .expect("right-associated merged section"),
                    "section {} differs under right association",
                    kind.raw(),
                );
            }

            let direct_results = concat_merge_query_results(&direct, &cx);
            let left_results = concat_merge_query_results(&left_associated, &cx);
            let right_results = concat_merge_query_results(&right_associated, &cx);
            assert_eq!(direct_results, results_before);
            assert_eq!(left_results, results_before);
            assert_eq!(right_results, results_before);
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
