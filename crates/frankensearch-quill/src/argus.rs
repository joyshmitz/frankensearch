//! Argus query execution.
//!
//! This module contains the deliberately exhaustive scorer used as the
//! correctness anchor for every later pruning path. It mirrors the pinned
//! Tantivy scorer-tree arithmetic, including buffered `Should` unions, while
//! retaining Quill's explicit `Option<u32>` cursor state so `u32::MAX` remains
//! a searchable global document id.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;
use std::sync::Arc;

use frankensearch_core::DocId;
use thiserror::Error;

use crate::contract::{BM25_K1, compute_tf_cache, idf};
use crate::quiver::{DocLenField, PositionCodecError, PostingCodecError, SnapshotFieldStats};

pub use crate::query::Occur;

const UNION_HORIZON: usize = 4_096;

/// Owner-bound decoder for one cursor's compressed position runs.
///
/// Sealed and delta cursors implement this against their own immutable
/// storage. Callers cannot resolve an ordinal against an unrelated segment.
pub trait PositionsReader {
    /// Decode one posting's absolute positions into an already-cleared buffer.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation, corruption, or cursor-identity failure.
    fn decode_positions(
        &self,
        posting_ordinal: u32,
        output: &mut Vec<u32>,
    ) -> Result<(), ArgusError>;
}

/// Borrowed capability for the current posting's later POSITIONS lookup.
///
/// The immutable borrow prevents its cursor from moving before the handle is
/// resolved. This keeps sealed and future delta position identities local to
/// their owning storage view without a GAT or per-hit allocation.
#[derive(Clone, Copy)]
pub struct PositionsHandle<'a> {
    reader: &'a dyn PositionsReader,
    posting_ordinal: u32,
}

impl fmt::Debug for PositionsHandle<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PositionsHandle")
            .field("posting_ordinal", &self.posting_ordinal)
            .finish_non_exhaustive()
    }
}

impl<'a> PositionsHandle<'a> {
    /// Bind a zero-based posting ordinal to its exact owning reader.
    #[must_use]
    pub fn new(reader: &'a dyn PositionsReader, posting_ordinal: u32) -> Self {
        Self {
            reader,
            posting_ordinal,
        }
    }

    /// Return the zero-based posting ordinal.
    #[must_use]
    pub const fn posting_ordinal(self) -> u32 {
        self.posting_ordinal
    }

    /// Decode this posting into a reusable caller-owned buffer.
    ///
    /// The buffer is cleared before decoding and remains empty on failure.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation, corruption, or cursor-identity failure.
    pub fn decode_into(self, output: &mut Vec<u32>) -> Result<(), ArgusError> {
        output.clear();
        if let Err(error) = self.reader.decode_positions(self.posting_ordinal, output) {
            output.clear();
            return Err(error);
        }
        Ok(())
    }
}

/// Typed failures from exhaustive query evaluation.
#[derive(Debug, Error)]
pub enum ArgusError {
    /// A sealed posting cursor violated its already-validated storage contract.
    #[error(transparent)]
    Posting(#[from] PostingCodecError),
    /// A sealed position cursor violated its validated paired-stream contract.
    #[error(transparent)]
    Position(#[from] PositionCodecError),
    /// Snapshot counters cannot describe a valid scored field.
    #[error("invalid BM25 snapshot for field {field_ord}: {reason}")]
    InvalidSnapshot {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Failed invariant.
        reason: &'static str,
    },
    /// Snapshot term frequency exceeded the snapshot's physical document count.
    #[error("term doc_freq {doc_freq} exceeds snapshot N {doc_count}")]
    InvalidDocFrequency {
        /// Snapshot-level term document frequency.
        doc_freq: u64,
        /// Snapshot-level physical document count.
        doc_count: u64,
    },
    /// A scorer bound statistics and field lengths from different fields.
    #[error("BM25 field mismatch: fieldnorm field {fieldnorm_field}, stats field {stats_field}")]
    FieldMismatch {
        /// Field ordinal supplied by the DOCLEN column.
        fieldnorm_field: u16,
        /// Field ordinal supplied by snapshot STATS.
        stats_field: u16,
    },
    /// A phrase term cursor was opened for a field other than its snapshot.
    #[error("phrase term field {term_field} does not match snapshot field {stats_field}")]
    PhraseTermFieldMismatch {
        /// Field ordinal supplied when the phrase cursor was opened.
        term_field: u16,
        /// Field ordinal supplied by snapshot STATS.
        stats_field: u16,
    },
    /// A programmatic query supplied a non-finite or overflowing field boost.
    #[error("field {field_ord} has invalid boost bits 0x{boost_bits:08x}")]
    InvalidBoost {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Exact rejected f32 representation.
        boost_bits: u32,
    },
    /// A posting had no corresponding DOCLEN byte in its segment.
    #[error("field {field_ord} has no fieldnorm for global docid {global_docid}")]
    MissingFieldnorm {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Global Quill document id.
        global_docid: u32,
    },
    /// A phrase leaf was opened against a frequency-only posting cursor.
    #[error("field {field_ord} has no positions for global docid {global_docid}")]
    MissingPositions {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Global Quill document id whose phrase positions were required.
        global_docid: u32,
    },
    /// A phrase plan cannot describe an exact positional query.
    #[error("invalid exact phrase: {reason}")]
    InvalidPhrase {
        /// Failed phrase-plan invariant.
        reason: &'static str,
    },
    /// A cursor exposed internally inconsistent accessors.
    #[error("posting cursor invariant failed: {0}")]
    CursorInvariant(&'static str),
    /// Score arithmetic escaped the finite BM25 domain.
    #[error("non-finite score for global docid {global_docid}")]
    NonFiniteScore {
        /// Global Quill document id being scored.
        global_docid: u32,
    },
    /// A bounded internal allocation could not be reserved.
    #[error("could not allocate {count} entries for {resource}")]
    Allocation {
        /// Bounded allocation purpose.
        resource: &'static str,
        /// Requested element count.
        count: usize,
    },
    /// A winning global document id could not be resolved externally.
    #[error("no external document id for global docid {global_docid}")]
    MissingExternalDocId {
        /// Global Quill document id.
        global_docid: u32,
    },
    /// Offset pagination overflowed the host collection domain.
    #[error("top-k collector window overflows usize: limit {limit}, offset {offset}")]
    CollectorWindowOverflow {
        /// Requested page size.
        limit: usize,
        /// Requested number of leading hits to skip.
        offset: usize,
    },
    /// An exact match count exceeded the durable collection domain.
    #[error("exact match count overflowed u64")]
    MatchCountOverflow,
    /// A scoreless collector received a scorer that scores while advancing.
    #[error("doc-set collection requires a recursively unscored Boolean scorer")]
    ScoredTreeForUnscoredCollector,
    /// A caller attempted to score a deliberately unscored query tree.
    #[error("an unscored Boolean query has no BM25 score")]
    ScoreUnavailable,
}

/// Forward-only posting access shared by sealed and future delta segments.
///
/// A cursor starts on its first posting. `advance` is an inclusive lower-bound
/// seek and does not move when the current document already satisfies the
/// target. Exhaustion is fused and represented by `None`.
pub trait PostingCursor {
    /// Return the current global document id, or `None` after exhaustion.
    fn doc(&self) -> Option<u32>;

    /// Return the current term frequency, or `None` after exhaustion.
    fn freq(&self) -> Option<u32>;

    /// Return the lazy POSITIONS locator for the current posting, when present.
    fn positions_handle(&self) -> Option<PositionsHandle<'_>>;

    /// Best-effort number of documents matched by this cursor.
    fn size_hint(&self) -> u32;

    /// Relative cost to drive this cursor to exhaustion.
    fn cost(&self) -> u64 {
        u64::from(self.size_hint())
    }

    /// Live document count used by Tantivy's composite size estimators.
    fn segment_num_docs(&self) -> u32;

    /// Move strictly forward by one posting.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the cursor cannot preserve its validated
    /// storage invariants.
    fn next(&mut self) -> Result<Option<u32>, ArgusError>;

    /// Land on the first posting whose document id is at least `target`.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the cursor cannot preserve its validated
    /// storage invariants.
    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError>;
}

impl<T> PostingCursor for Box<T>
where
    T: PostingCursor + ?Sized,
{
    fn doc(&self) -> Option<u32> {
        (**self).doc()
    }

    fn freq(&self) -> Option<u32> {
        (**self).freq()
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        (**self).positions_handle()
    }

    fn size_hint(&self) -> u32 {
        (**self).size_hint()
    }

    fn cost(&self) -> u64 {
        (**self).cost()
    }

    fn segment_num_docs(&self) -> u32 {
        (**self).segment_num_docs()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        (**self).next()
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        (**self).advance(target)
    }
}

enum SealedCursorInner<'a> {
    Docs(crate::quiver::PostingCursor<'a>),
    Positions(crate::quiver::PositionCursor<'a>),
}

/// Owner-safe adapter for one validated sealed posting or position cursor.
pub struct SealedPostingCursor<'a> {
    inner: SealedCursorInner<'a>,
    size_hint: u32,
    segment_num_docs: u32,
}

impl<'a> SealedPostingCursor<'a> {
    /// Open a position-free sealed posting cursor.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the validated posting list can no longer open
    /// its cursor.
    pub fn new(
        postings: &'a crate::quiver::PostingList<'_>,
        segment_num_docs: u32,
    ) -> Result<Self, ArgusError> {
        Ok(Self {
            inner: SealedCursorInner::Docs(postings.cursor()?),
            size_hint: postings.doc_freq(),
            segment_num_docs,
        })
    }

    /// Open a sealed cursor already paired with its validated POSITIONS stream.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the validated paired streams can no longer
    /// open their allocation-free cursor.
    pub fn with_positions(
        positions: &'a crate::quiver::PositionList<'_>,
        segment_num_docs: u32,
    ) -> Result<Self, ArgusError> {
        Ok(Self {
            inner: SealedCursorInner::Positions(positions.cursor()?),
            size_hint: positions.doc_freq(),
            segment_num_docs,
        })
    }
}

impl PostingCursor for SealedPostingCursor<'_> {
    fn doc(&self) -> Option<u32> {
        match &self.inner {
            SealedCursorInner::Docs(cursor) => cursor.doc(),
            SealedCursorInner::Positions(cursor) => cursor.doc(),
        }
    }

    fn freq(&self) -> Option<u32> {
        match &self.inner {
            SealedCursorInner::Docs(cursor) => cursor.freq(),
            SealedCursorInner::Positions(cursor) => cursor.freq(),
        }
    }

    fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
        match &self.inner {
            SealedCursorInner::Docs(_) => None,
            SealedCursorInner::Positions(cursor) => cursor
                .posting_ordinal()
                .map(|ordinal| PositionsHandle::new(self, ordinal)),
        }
    }

    fn size_hint(&self) -> u32 {
        self.size_hint
    }

    fn segment_num_docs(&self) -> u32 {
        self.segment_num_docs
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        match &mut self.inner {
            SealedCursorInner::Docs(cursor) => Ok(cursor.next()?.map(|posting| posting.doc_id)),
            SealedCursorInner::Positions(cursor) => {
                Ok(cursor.next()?.map(|posting| posting.doc_id))
            }
        }
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        match &mut self.inner {
            SealedCursorInner::Docs(cursor) => {
                Ok(cursor.advance(target)?.map(|posting| posting.doc_id))
            }
            SealedCursorInner::Positions(cursor) => {
                Ok(cursor.advance(target)?.map(|posting| posting.doc_id))
            }
        }
    }
}

impl PositionsReader for SealedPostingCursor<'_> {
    fn decode_positions(
        &self,
        posting_ordinal: u32,
        output: &mut Vec<u32>,
    ) -> Result<(), ArgusError> {
        let SealedCursorInner::Positions(cursor) = &self.inner else {
            return Err(ArgusError::CursorInvariant(
                "position-free sealed cursor cannot resolve positions",
            ));
        };
        if cursor.posting_ordinal() != Some(posting_ordinal) {
            return Err(ArgusError::CursorInvariant(
                "positions handle no longer identifies the current posting",
            ));
        }
        let positions = cursor.positions()?.ok_or(ArgusError::CursorInvariant(
            "positioned sealed cursor has no current position run",
        ))?;
        output
            .try_reserve_exact(positions.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "decoded positions",
                count: positions.len(),
            })?;
        for position in positions {
            output.push(position?);
        }
        Ok(())
    }
}

/// Shared, validated BM25 denominator cache for one `(field, snapshot)` pair.
#[derive(Clone, Debug)]
pub struct Bm25FieldSnapshot {
    stats: SnapshotFieldStats,
    average_field_length: Option<f32>,
    tf_cache: Arc<[f32; 256]>,
}

impl Bm25FieldSnapshot {
    /// Validate snapshot counters and build the field's 256-entry tf cache.
    ///
    /// An empty or all-empty field has no scored postings and therefore keeps
    /// a zero cache. A term scorer rejects any posting that would try to use
    /// that cache.
    ///
    /// # Errors
    ///
    /// Returns a typed error when an empty snapshot reports tokens or the raw
    /// average cannot be represented as a finite f32.
    pub fn new(stats: SnapshotFieldStats) -> Result<Self, ArgusError> {
        if stats.doc_count == 0 && stats.total_tokens != 0 {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: stats.field_ord,
                reason: "an empty snapshot cannot contain field tokens",
            });
        }
        let average_field_length = stats.average_field_length();
        if average_field_length.is_some_and(|average| !average.is_finite()) {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: stats.field_ord,
                reason: "raw average field length is not finite",
            });
        }
        let tf_cache = average_field_length
            .filter(|average| *average > 0.0)
            .map_or_else(
                || Arc::new([0.0; 256]),
                |average| Arc::new(compute_tf_cache(average)),
            );
        Ok(Self {
            stats,
            average_field_length,
            tf_cache,
        })
    }

    /// Stable schema field ordinal.
    #[must_use]
    pub const fn field_ord(&self) -> u16 {
        self.stats.field_ord
    }

    /// Snapshot physical document count used as BM25 `N`.
    #[must_use]
    pub const fn doc_count(&self) -> u64 {
        self.stats.doc_count
    }

    /// Snapshot token numerator, including tombstoned documents until merge.
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.stats.total_tokens
    }

    /// Raw `total_tokens / N` average, or `None` for an empty snapshot.
    #[must_use]
    pub const fn average_field_length(&self) -> Option<f32> {
        self.average_field_length
    }
}

/// Backend-neutral fieldnorm access for sealed and delta segment views.
pub trait FieldNormReader {
    /// Stable schema field ordinal.
    fn field_ord(&self) -> u16;

    /// Stored Tantivy-compatible fieldnorm ID for one present global docid.
    fn fieldnorm_id(&self, global_docid: u32) -> Option<u8>;
}

impl FieldNormReader for DocLenField<'_> {
    fn field_ord(&self) -> u16 {
        DocLenField::field_ord(*self)
    }

    fn fieldnorm_id(&self, global_docid: u32) -> Option<u8> {
        DocLenField::fieldnorm_id(*self, u64::from(global_docid))
    }
}

/// One BM25 term leaf bound to a segment cursor and fieldnorm view.
pub struct TermScorer<'a> {
    cursor: Box<dyn PostingCursor + 'a>,
    fieldnorms: Box<dyn FieldNormReader + 'a>,
    snapshot: Bm25FieldSnapshot,
    weight: f32,
    cost: u64,
    size_hint: u32,
    segment_num_docs: u32,
}

impl<'a> TermScorer<'a> {
    /// Build a term scorer from snapshot-level statistics and segment-local
    /// cursor state.
    ///
    /// `snapshot_doc_freq` is the sum across every live segment and therefore
    /// includes tombstoned postings until compaction. Runtime cost, size hint,
    /// and segment `num_docs` come from the cursor so exact required-clause
    /// ordering cannot drift from its storage implementation.
    ///
    /// # Errors
    ///
    /// Rejects field drift, malformed snapshot statistics, inconsistent
    /// cursor accessors, invalid boosts, and `doc_freq > N` without invoking
    /// the vendored `idf` assertion.
    pub fn new<C, F>(
        cursor: C,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        snapshot_doc_freq: u64,
        field_boost: f32,
    ) -> Result<Self, ArgusError>
    where
        C: PostingCursor + 'a,
        F: FieldNormReader + 'a,
    {
        if fieldnorms.field_ord() != snapshot.field_ord() {
            return Err(ArgusError::FieldMismatch {
                fieldnorm_field: fieldnorms.field_ord(),
                stats_field: snapshot.field_ord(),
            });
        }
        if snapshot_doc_freq > snapshot.doc_count() {
            return Err(ArgusError::InvalidDocFrequency {
                doc_freq: snapshot_doc_freq,
                doc_count: snapshot.doc_count(),
            });
        }
        let size_hint = cursor.size_hint();
        let cost = cursor.cost();
        let segment_num_docs = cursor.segment_num_docs();
        if u64::from(segment_num_docs) > snapshot.doc_count() {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: snapshot.field_ord(),
                reason: "segment num_docs exceeds snapshot physical document count",
            });
        }
        if !field_boost.is_finite() {
            return Err(ArgusError::InvalidBoost {
                field_ord: snapshot.field_ord(),
                boost_bits: field_boost.to_bits(),
            });
        }
        if snapshot_doc_freq != 0
            && snapshot
                .average_field_length()
                .is_none_or(|average| average <= 0.0)
        {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: snapshot.field_ord(),
                reason: "a scored term requires a positive raw average field length",
            });
        }
        if cursor.doc().is_some() {
            if snapshot_doc_freq == 0 {
                return Err(ArgusError::InvalidSnapshot {
                    field_ord: snapshot.field_ord(),
                    reason: "a non-empty cursor cannot have zero snapshot doc_freq",
                });
            }
            if size_hint == 0 || cost == 0 {
                return Err(ArgusError::CursorInvariant(
                    "a non-empty cursor must have non-zero size and runtime cost",
                ));
            }
            validate_cursor_position(&cursor, &fieldnorms)?;
        } else {
            if cursor.freq().is_some() || cursor.positions_handle().is_some() {
                return Err(ArgusError::CursorInvariant(
                    "exhausted cursor retained frequency or positions state",
                ));
            }
            if size_hint != 0 || cost != 0 {
                return Err(ArgusError::CursorInvariant(
                    "an empty cursor must have zero size and runtime cost",
                ));
            }
        }

        let mut weight = idf(snapshot_doc_freq, snapshot.doc_count()) * (1.0 + BM25_K1);
        weight *= field_boost;
        if !weight.is_finite() {
            return Err(ArgusError::InvalidBoost {
                field_ord: snapshot.field_ord(),
                boost_bits: field_boost.to_bits(),
            });
        }
        Ok(Self {
            cursor: Box::new(cursor),
            fieldnorms: Box::new(fieldnorms),
            snapshot,
            weight,
            cost,
            size_hint,
            segment_num_docs,
        })
    }

    fn doc(&self) -> Option<u32> {
        self.cursor.doc()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        let previous = self.cursor.doc();
        let moved = self.cursor.next()?;
        let moved =
            validate_cursor_after_move(self.cursor.as_ref(), self.fieldnorms.as_ref(), moved)?;
        match (previous, moved) {
            (None, Some(_)) => Err(ArgusError::CursorInvariant(
                "exhausted cursor resurrected during next",
            )),
            (Some(before), Some(after)) if after <= before => Err(ArgusError::CursorInvariant(
                "next did not move to a strictly greater document",
            )),
            _ => Ok(moved),
        }
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        let previous = self.cursor.doc();
        let moved = self.cursor.advance(target)?;
        let moved =
            validate_cursor_after_move(self.cursor.as_ref(), self.fieldnorms.as_ref(), moved)?;
        match (previous, moved) {
            (None, Some(_)) => Err(ArgusError::CursorInvariant(
                "exhausted cursor resurrected during advance",
            )),
            (Some(before), Some(after)) if before >= target && after != before => {
                Err(ArgusError::CursorInvariant(
                    "advance moved despite the current document satisfying its target",
                ))
            }
            (Some(before), None) if before >= target => Err(ArgusError::CursorInvariant(
                "advance exhausted despite the current document satisfying its target",
            )),
            (_, Some(after)) if after < target => Err(ArgusError::CursorInvariant(
                "advance landed below its requested target",
            )),
            _ => Ok(moved),
        }
    }

    fn score(&self) -> Result<f32, ArgusError> {
        let doc = self.cursor.doc().ok_or(ArgusError::CursorInvariant(
            "cannot score an exhausted cursor",
        ))?;
        let frequency = self.cursor.freq().ok_or(ArgusError::CursorInvariant(
            "current posting has no frequency",
        ))?;
        if frequency == 0 {
            return Err(ArgusError::CursorInvariant(
                "current posting has zero term frequency",
            ));
        }
        let fieldnorm_id =
            self.fieldnorms
                .fieldnorm_id(doc)
                .ok_or_else(|| ArgusError::MissingFieldnorm {
                    field_ord: self.fieldnorms.field_ord(),
                    global_docid: doc,
                })?;
        let frequency = frequency as f32;
        let norm = self.snapshot.tf_cache[usize::from(fieldnorm_id)];
        let tf_factor = frequency / (frequency + norm);
        finite_score(self.weight * tf_factor, doc)
    }
}

/// One analyzed phrase term bound to its positioned posting cursor.
///
/// Terms must be supplied in non-decreasing analyzer-position order. Terms at
/// the same position are treated as alternatives for matching while retaining
/// their original order in the fixed phrase-weight calculation. The field
/// ordinal is trusted lowering metadata and must come from the same posting-open
/// operation as the cursor; the opaque cursor contract cannot recover it.
pub struct PhraseTerm<'a> {
    field_ord: u16,
    position: u32,
    cursor: Box<dyn PostingCursor + 'a>,
    snapshot_doc_freq: u64,
}

impl<'a> PhraseTerm<'a> {
    /// Bind one field, query position, and snapshot document frequency to a cursor.
    ///
    /// `field_ord` must be copied from the trusted posting-open path that
    /// produced `cursor`; construction validates it against snapshot STATS.
    #[must_use]
    pub fn new<C>(field_ord: u16, position: u32, cursor: C, snapshot_doc_freq: u64) -> Self
    where
        C: PostingCursor + 'a,
    {
        Self {
            field_ord,
            position,
            cursor: Box::new(cursor),
            snapshot_doc_freq,
        }
    }

    /// Stable schema field ordinal whose postings back this term.
    #[must_use]
    pub const fn field_ord(&self) -> u16 {
        self.field_ord
    }

    /// Analyzer position retained from the parsed phrase.
    #[must_use]
    pub const fn position(&self) -> u32 {
        self.position
    }

    /// Snapshot document frequency used in the phrase's fixed BM25 weight.
    #[must_use]
    pub const fn snapshot_doc_freq(&self) -> u64 {
        self.snapshot_doc_freq
    }
}

struct PhraseSlot<'a> {
    position: u32,
    alternatives: Vec<Box<dyn PostingCursor + 'a>>,
    positions: Vec<u32>,
    cost: u64,
    size_hint: u32,
}

impl<'a> PhraseSlot<'a> {
    fn new(position: u32, cursor: Box<dyn PostingCursor + 'a>) -> Self {
        let cost = cursor.cost();
        let size_hint = cursor.size_hint();
        Self {
            position,
            alternatives: vec![cursor],
            positions: Vec::new(),
            cost,
            size_hint,
        }
    }

    fn push_alternative(&mut self, cursor: Box<dyn PostingCursor + 'a>) -> Result<(), ArgusError> {
        self.alternatives
            .try_reserve(1)
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase alternatives",
                count: self.alternatives.len().saturating_add(1),
            })?;
        self.cost = self.cost.saturating_add(cursor.cost());
        self.alternatives.push(cursor);
        Ok(())
    }

    fn doc(&self) -> Option<u32> {
        self.alternatives
            .iter()
            .filter_map(|cursor| cursor.doc())
            .min()
    }

    fn seek(
        &mut self,
        target: u32,
        fieldnorms: &dyn FieldNormReader,
    ) -> Result<Option<u32>, ArgusError> {
        for cursor in &mut self.alternatives {
            let previous = cursor.doc();
            if previous.is_some_and(|doc| doc < target) {
                let moved = cursor.advance(target)?;
                validate_phrase_cursor_after_move(cursor.as_ref(), fieldnorms, moved)?;
                match (previous, moved) {
                    (None, Some(_)) => {
                        return Err(ArgusError::CursorInvariant(
                            "exhausted phrase cursor resurrected during advance",
                        ));
                    }
                    (_, Some(after)) if after < target => {
                        return Err(ArgusError::CursorInvariant(
                            "phrase advance landed below its requested target",
                        ));
                    }
                    _ => {}
                }
            }
        }
        Ok(self.doc())
    }

    fn next(&mut self, fieldnorms: &dyn FieldNormReader) -> Result<Option<u32>, ArgusError> {
        let Some(previous) = self.doc() else {
            return Ok(None);
        };
        for cursor in &mut self.alternatives {
            if cursor.doc() == Some(previous) {
                let moved = cursor.next()?;
                validate_phrase_cursor_after_move(cursor.as_ref(), fieldnorms, moved)?;
                if moved.is_some_and(|after| after <= previous) {
                    return Err(ArgusError::CursorInvariant(
                        "phrase next did not move to a strictly greater document",
                    ));
                }
            }
        }
        let moved = self.doc();
        if moved.is_some_and(|after| after <= previous) {
            return Err(ArgusError::CursorInvariant(
                "phrase slot did not move to a strictly greater document",
            ));
        }
        Ok(moved)
    }
}

/// Exact-adjacency phrase scorer over positioned posting cursors.
///
/// Candidate documents are intersected before positions are decoded. The
/// scorer owns and reuses every positional scratch buffer, so advancing after
/// warm-up performs no per-document allocation.
pub struct PhraseScorer<'a> {
    slots: Vec<PhraseSlot<'a>>,
    fieldnorms: Box<dyn FieldNormReader + 'a>,
    snapshot: Bm25FieldSnapshot,
    weight: f32,
    lead_slot: usize,
    cost: u64,
    size_hint: u32,
    segment_num_docs: u32,
    current: Option<u32>,
    current_frequency: u32,
    decode_scratch: Vec<u32>,
    position_indices: Vec<usize>,
}

impl<'a> PhraseScorer<'a> {
    /// Build and initially align one exact phrase scorer.
    ///
    /// Query terms must be ordered by non-decreasing analyzer position and
    /// span at least two distinct positions. Equal-position terms form one OR
    /// slot. Each term needs its own forward-only positioned cursor, including
    /// repeated terms such as `a a b`.
    ///
    /// # Errors
    ///
    /// Rejects malformed positions, non-positioned cursors, field/domain
    /// drift, invalid snapshot frequencies or boosts, and bounded allocation
    /// failures.
    pub fn new<F>(
        terms: Vec<PhraseTerm<'a>>,
        fieldnorms: F,
        snapshot: Bm25FieldSnapshot,
        field_boost: f32,
    ) -> Result<Self, ArgusError>
    where
        F: FieldNormReader + 'a,
    {
        if terms.is_empty() {
            return Err(ArgusError::InvalidPhrase {
                reason: "an exact phrase requires positioned terms",
            });
        }
        if fieldnorms.field_ord() != snapshot.field_ord() {
            return Err(ArgusError::FieldMismatch {
                fieldnorm_field: fieldnorms.field_ord(),
                stats_field: snapshot.field_ord(),
            });
        }
        if !field_boost.is_finite() {
            return Err(ArgusError::InvalidBoost {
                field_ord: snapshot.field_ord(),
                boost_bits: field_boost.to_bits(),
            });
        }
        let num_terms = u32::try_from(terms.len()).map_err(|_| ArgusError::InvalidPhrase {
            reason: "phrase term count exceeds u32",
        })?;
        let phrase_work_factor = num_terms.checked_mul(10).ok_or(ArgusError::InvalidPhrase {
            reason: "phrase term cost factor exceeds u32",
        })?;

        let mut slots: Vec<PhraseSlot<'a>> = Vec::new();
        slots
            .try_reserve_exact(terms.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase position slots",
                count: terms.len(),
            })?;
        let mut segment_num_docs = None;
        let mut idf_sum = 0.0_f32;
        for term in terms {
            if term.field_ord != snapshot.field_ord() {
                return Err(ArgusError::PhraseTermFieldMismatch {
                    term_field: term.field_ord,
                    stats_field: snapshot.field_ord(),
                });
            }
            if term.snapshot_doc_freq > snapshot.doc_count() {
                return Err(ArgusError::InvalidDocFrequency {
                    doc_freq: term.snapshot_doc_freq,
                    doc_count: snapshot.doc_count(),
                });
            }
            let cursor_segment_num_docs = term.cursor.segment_num_docs();
            if u64::from(cursor_segment_num_docs) > snapshot.doc_count() {
                return Err(ArgusError::InvalidSnapshot {
                    field_ord: snapshot.field_ord(),
                    reason: "phrase segment num_docs exceeds snapshot physical document count",
                });
            }
            if segment_num_docs
                .replace(cursor_segment_num_docs)
                .is_some_and(|previous| previous != cursor_segment_num_docs)
            {
                return Err(ArgusError::CursorInvariant(
                    "phrase cursors belong to different segment domains",
                ));
            }
            validate_phrase_cursor_state(term.cursor.as_ref(), &fieldnorms)?;
            if term.cursor.doc().is_none()
                && (term.cursor.size_hint() != 0 || term.cursor.cost() != 0)
            {
                return Err(ArgusError::CursorInvariant(
                    "an initially empty phrase cursor must have zero size and runtime cost",
                ));
            }
            if term.cursor.doc().is_some() && term.snapshot_doc_freq == 0 {
                return Err(ArgusError::InvalidSnapshot {
                    field_ord: snapshot.field_ord(),
                    reason: "a non-empty phrase cursor cannot have zero snapshot doc_freq",
                });
            }
            idf_sum += idf(term.snapshot_doc_freq, snapshot.doc_count());
            if !idf_sum.is_finite() {
                return Err(ArgusError::InvalidSnapshot {
                    field_ord: snapshot.field_ord(),
                    reason: "phrase IDF sum is not finite",
                });
            }

            match slots.last_mut() {
                Some(slot) if term.position < slot.position => {
                    return Err(ArgusError::InvalidPhrase {
                        reason: "phrase positions must be non-decreasing",
                    });
                }
                Some(slot) if term.position == slot.position => {
                    slot.push_alternative(term.cursor)?;
                }
                _ => slots.push(PhraseSlot::new(term.position, term.cursor)),
            }
        }
        if slots.len() < 2 {
            return Err(ArgusError::InvalidPhrase {
                reason: "an exact phrase must span at least two positions",
            });
        }
        if slots.iter().any(|slot| {
            slot.alternatives
                .iter()
                .any(|cursor| cursor.doc().is_some())
        }) && snapshot
            .average_field_length()
            .is_none_or(|average| average <= 0.0)
        {
            return Err(ArgusError::InvalidSnapshot {
                field_ord: snapshot.field_ord(),
                reason: "a scored phrase requires a positive raw average field length",
            });
        }

        let segment_num_docs = segment_num_docs.unwrap_or(0);
        for slot in &mut slots {
            if slot.alternatives.len() > 1 {
                slot.size_hint = estimate_union(
                    slot.alternatives.iter().map(|cursor| cursor.size_hint()),
                    segment_num_docs,
                );
            }
        }
        let lead_slot = slots
            .iter()
            .enumerate()
            .min_by_key(|(_, slot)| slot.cost)
            .map_or(0, |(index, _)| index);
        let mut cost_ordered_estimates = Vec::new();
        cost_ordered_estimates
            .try_reserve_exact(slots.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase cost estimates",
                count: slots.len(),
            })?;
        cost_ordered_estimates.extend(slots.iter().map(|slot| (slot.cost, slot.size_hint)));
        cost_ordered_estimates.sort_by_key(|(cost, _)| *cost);
        let candidate_estimate = estimate_intersection(
            cost_ordered_estimates.into_iter().map(|(_, size)| size),
            segment_num_docs,
        );
        let cost = u64::from(candidate_estimate) * u64::from(phrase_work_factor);
        let size_hint = candidate_estimate / phrase_work_factor;
        let mut position_indices = Vec::new();
        position_indices
            .try_reserve_exact(slots.len())
            .map_err(|_| ArgusError::Allocation {
                resource: "phrase position indices",
                count: slots.len(),
            })?;
        position_indices.resize(slots.len(), 0);

        let mut weight = idf_sum * (1.0 + BM25_K1);
        weight *= field_boost;
        if !weight.is_finite() {
            return Err(ArgusError::InvalidBoost {
                field_ord: snapshot.field_ord(),
                boost_bits: field_boost.to_bits(),
            });
        }
        let mut scorer = Self {
            slots,
            fieldnorms: Box::new(fieldnorms),
            snapshot,
            weight,
            lead_slot,
            cost,
            size_hint,
            segment_num_docs,
            current: None,
            current_frequency: 0,
            decode_scratch: Vec::new(),
            position_indices,
        };
        scorer.align()?;
        Ok(scorer)
    }

    fn doc(&self) -> Option<u32> {
        self.current
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.current.is_none() {
            return Ok(None);
        }
        let fieldnorms = self.fieldnorms.as_ref();
        self.slots[self.lead_slot].next(fieldnorms)?;
        self.align()
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        let fieldnorms = self.fieldnorms.as_ref();
        self.slots[self.lead_slot].seek(target, fieldnorms)?;
        self.align()
    }

    fn align(&mut self) -> Result<Option<u32>, ArgusError> {
        self.current = None;
        self.current_frequency = 0;
        loop {
            let Some(mut target) = self.slots.iter().filter_map(PhraseSlot::doc).max() else {
                return Ok(None);
            };
            loop {
                let mut raised_target = false;
                for slot in &mut self.slots {
                    let Some(doc) = slot.seek(target, self.fieldnorms.as_ref())? else {
                        return Ok(None);
                    };
                    if doc > target {
                        target = doc;
                        raised_target = true;
                    }
                }
                if !raised_target && self.slots.iter().all(|slot| slot.doc() == Some(target)) {
                    break;
                }
            }

            let frequency = self.phrase_frequency(target)?;
            if frequency != 0 {
                self.current = Some(target);
                self.current_frequency = frequency;
                return Ok(self.current);
            }
            let fieldnorms = self.fieldnorms.as_ref();
            self.slots[self.lead_slot].next(fieldnorms)?;
        }
    }

    fn phrase_frequency(&mut self, doc: u32) -> Result<u32, ArgusError> {
        for slot_index in 0..self.slots.len() {
            self.decode_slot_positions(slot_index, doc)?;
        }
        self.position_indices.fill(0);
        let first_position = self.slots[0].position;
        let (slots, position_indices) = (&self.slots, &mut self.position_indices);
        let mut frequency = 0_u32;
        for &base in &slots[0].positions {
            let mut matched = true;
            for slot_index in 1..slots.len() {
                let offset = slots[slot_index].position - first_position;
                let Some(target) = base.checked_add(offset) else {
                    matched = false;
                    break;
                };
                let positions = &slots[slot_index].positions;
                let position_index = &mut position_indices[slot_index];
                while positions
                    .get(*position_index)
                    .is_some_and(|position| *position < target)
                {
                    *position_index += 1;
                }
                if positions.get(*position_index) != Some(&target) {
                    matched = false;
                    break;
                }
            }
            if matched {
                frequency = frequency
                    .checked_add(1)
                    .ok_or(ArgusError::CursorInvariant("phrase frequency exceeds u32"))?;
            }
        }
        Ok(frequency)
    }

    fn decode_slot_positions(&mut self, slot_index: usize, doc: u32) -> Result<(), ArgusError> {
        let (slots, decode_scratch) = (&mut self.slots, &mut self.decode_scratch);
        let slot = &mut slots[slot_index];
        slot.positions.clear();
        for cursor in &slot.alternatives {
            if cursor.doc() != Some(doc) {
                continue;
            }
            let frequency = cursor.freq().ok_or(ArgusError::CursorInvariant(
                "positioned phrase cursor has no frequency",
            ))?;
            let handle = cursor
                .positions_handle()
                .ok_or_else(|| ArgusError::MissingPositions {
                    field_ord: self.snapshot.field_ord(),
                    global_docid: doc,
                })?;
            handle.decode_into(decode_scratch)?;
            let expected = usize::try_from(frequency)
                .map_err(|_| ArgusError::CursorInvariant("phrase frequency does not fit usize"))?;
            if decode_scratch.len() != expected {
                return Err(ArgusError::CursorInvariant(
                    "decoded phrase position count differs from term frequency",
                ));
            }
            let required = slot.positions.len().saturating_add(decode_scratch.len());
            slot.positions
                .try_reserve(decode_scratch.len())
                .map_err(|_| ArgusError::Allocation {
                    resource: "phrase slot positions",
                    count: required,
                })?;
            slot.positions.extend_from_slice(decode_scratch);
        }
        slot.positions.sort_unstable();
        slot.positions.dedup();
        if slot.positions.is_empty() {
            return Err(ArgusError::CursorInvariant(
                "phrase candidate has no decoded positions",
            ));
        }
        Ok(())
    }

    fn score(&self) -> Result<f32, ArgusError> {
        let doc = self.current.ok_or(ArgusError::CursorInvariant(
            "cannot score an exhausted phrase scorer",
        ))?;
        if self.current_frequency == 0 {
            return Err(ArgusError::CursorInvariant(
                "current phrase match has zero frequency",
            ));
        }
        let fieldnorm_id =
            self.fieldnorms
                .fieldnorm_id(doc)
                .ok_or_else(|| ArgusError::MissingFieldnorm {
                    field_ord: self.fieldnorms.field_ord(),
                    global_docid: doc,
                })?;
        let frequency = self.current_frequency as f32;
        let norm = self.snapshot.tf_cache[usize::from(fieldnorm_id)];
        finite_score(self.weight * (frequency / (frequency + norm)), doc)
    }
}

fn validate_phrase_cursor_after_move(
    cursor: &dyn PostingCursor,
    fieldnorms: &dyn FieldNormReader,
    moved_doc: Option<u32>,
) -> Result<(), ArgusError> {
    if cursor.doc() != moved_doc {
        return Err(ArgusError::CursorInvariant(
            "phrase movement result disagrees with current document",
        ));
    }
    validate_phrase_cursor_state(cursor, fieldnorms)
}

fn validate_phrase_cursor_state(
    cursor: &dyn PostingCursor,
    fieldnorms: &dyn FieldNormReader,
) -> Result<(), ArgusError> {
    let Some(doc) = cursor.doc() else {
        if cursor.freq().is_some() || cursor.positions_handle().is_some() {
            return Err(ArgusError::CursorInvariant(
                "exhausted phrase cursor retained frequency or positions state",
            ));
        }
        return Ok(());
    };
    if cursor.freq().is_none_or(|frequency| frequency == 0) {
        return Err(ArgusError::CursorInvariant(
            "positioned phrase cursor has no positive frequency",
        ));
    }
    if cursor.positions_handle().is_none() {
        return Err(ArgusError::MissingPositions {
            field_ord: fieldnorms.field_ord(),
            global_docid: doc,
        });
    }
    if cursor.size_hint() == 0 || cursor.cost() == 0 {
        return Err(ArgusError::CursorInvariant(
            "a non-empty phrase cursor must have non-zero size and runtime cost",
        ));
    }
    if fieldnorms.fieldnorm_id(doc).is_none() {
        return Err(ArgusError::MissingFieldnorm {
            field_ord: fieldnorms.field_ord(),
            global_docid: doc,
        });
    }
    Ok(())
}

fn validate_cursor_after_move(
    cursor: &dyn PostingCursor,
    fieldnorms: &dyn FieldNormReader,
    moved_doc: Option<u32>,
) -> Result<Option<u32>, ArgusError> {
    if cursor.doc() != moved_doc {
        return Err(ArgusError::CursorInvariant(
            "movement result disagrees with current document",
        ));
    }
    if moved_doc.is_none() {
        if cursor.freq().is_some() || cursor.positions_handle().is_some() {
            return Err(ArgusError::CursorInvariant(
                "exhausted cursor retained frequency or positions state",
            ));
        }
        return Ok(None);
    }
    validate_cursor_position(cursor, fieldnorms)?;
    Ok(moved_doc)
}

fn validate_cursor_position<C>(
    cursor: &C,
    fieldnorms: &dyn FieldNormReader,
) -> Result<(), ArgusError>
where
    C: PostingCursor + ?Sized,
{
    let doc = cursor.doc().ok_or(ArgusError::CursorInvariant(
        "positioned cursor has no document",
    ))?;
    if cursor.freq().is_none_or(|frequency| frequency == 0) {
        return Err(ArgusError::CursorInvariant(
            "positioned cursor has no positive frequency",
        ));
    }
    if fieldnorms.fieldnorm_id(doc).is_none() {
        return Err(ArgusError::MissingFieldnorm {
            field_ord: fieldnorms.field_ord(),
            global_docid: doc,
        });
    }
    Ok(())
}

/// One lowered scorer child and its canonical query occurrence.
pub struct ScorerClause<'a> {
    occur: Occur,
    scorer: ReferenceScorer<'a>,
}

impl<'a> ScorerClause<'a> {
    /// Build a lowered scorer child.
    #[must_use]
    pub const fn new(occur: Occur, scorer: ReferenceScorer<'a>) -> Self {
        Self { occur, scorer }
    }

    /// Build a required positive child.
    #[must_use]
    pub const fn must(scorer: ReferenceScorer<'a>) -> Self {
        Self::new(Occur::Must, scorer)
    }

    /// Build an optional positive child.
    #[must_use]
    pub const fn should(scorer: ReferenceScorer<'a>) -> Self {
        Self::new(Occur::Should, scorer)
    }

    /// Build a scoreless exclusion child.
    #[must_use]
    pub const fn must_not(scorer: ReferenceScorer<'a>) -> Self {
        Self::new(Occur::MustNot, scorer)
    }
}

enum ScorerNode<'a> {
    Empty,
    Term(TermScorer<'a>),
    Phrase(PhraseScorer<'a>),
    Intersection(IntersectionScorer<'a>),
    Union(BufferedUnionScorer<'a>),
    UnscoredUnion(UnscoredUnionScorer<'a>),
    RequiredOptional(RequiredOptionalScorer<'a>),
    Exclude(ExcludeScorer<'a>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BooleanMode {
    Scored,
    Unscored,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SeekDangerResult {
    Found,
    SeekLowerBound(Option<u32>),
}

fn classify_seek_result(target: u32, result: Option<u32>) -> Result<SeekDangerResult, ArgusError> {
    match result {
        Some(doc) if doc == target => Ok(SeekDangerResult::Found),
        Some(doc) if doc > target => Ok(SeekDangerResult::SeekLowerBound(Some(doc))),
        None => Ok(SeekDangerResult::SeekLowerBound(None)),
        Some(_) => Err(ArgusError::CursorInvariant(
            "danger seek landed below its requested target",
        )),
    }
}

/// Stateful exhaustive scorer tree.
///
/// Construct one tree per segment and consume it with [`Self::top_k`]. The
/// object intentionally retains the unpruned path for differential testing of
/// future WAND and block-max implementations.
pub struct ReferenceScorer<'a> {
    node: ScorerNode<'a>,
}

impl<'a> ReferenceScorer<'a> {
    /// Build a scorer that never matches.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            node: ScorerNode::Empty,
        }
    }

    /// Wrap one BM25 term leaf.
    #[must_use]
    pub fn term(term: TermScorer<'a>) -> Self {
        if term.doc().is_none() {
            Self::empty()
        } else {
            Self {
                node: ScorerNode::Term(term),
            }
        }
    }

    /// Wrap one exact-adjacency phrase leaf.
    #[must_use]
    pub fn phrase(phrase: PhraseScorer<'a>) -> Self {
        if phrase.doc().is_none() {
            Self::empty()
        } else {
            Self {
                node: ScorerNode::Phrase(phrase),
            }
        }
    }

    fn is_explicit_empty(&self) -> bool {
        matches!(self.node, ScorerNode::Empty)
    }

    fn is_unscored_safe(&self) -> bool {
        match &self.node {
            ScorerNode::Empty | ScorerNode::Term(_) | ScorerNode::Phrase(_) => true,
            ScorerNode::Intersection(scorer) => scorer.scorers.iter().all(Self::is_unscored_safe),
            ScorerNode::Union(_) => false,
            ScorerNode::UnscoredUnion(scorer) => scorer.active.iter().all(Self::is_unscored_safe),
            ScorerNode::RequiredOptional(scorer) => {
                scorer.required.is_unscored_safe() && scorer.optional.is_unscored_safe()
            }
            ScorerNode::Exclude(scorer) => {
                scorer.include.is_unscored_safe()
                    && scorer.excluded.iter().all(Self::is_unscored_safe)
            }
        }
    }

    /// Lower Boolean clauses into the pinned Tantivy scorer-tree shape.
    ///
    /// `Should` is required only when no positive `Must` exists. A raw
    /// `MustNot`-only node matches nothing.
    ///
    /// # Errors
    ///
    /// Returns a typed error if bounded scorer buffers cannot be allocated or
    /// initial cursor alignment detects malformed state.
    pub fn boolean(clauses: Vec<ScorerClause<'a>>) -> Result<Self, ArgusError> {
        Self::boolean_with_mode(clauses, BooleanMode::Scored)
    }

    /// Lower Boolean clauses into a genuinely scoreless matching tree.
    ///
    /// The caller must use this constructor recursively for nested Boolean
    /// children. Unlike the parity-pinned buffered scored union, its `Should`
    /// union advances lazily and never invokes BM25 scoring. Optional `Should`
    /// clauses are omitted when a positive `Must` already determines matching.
    ///
    /// # Errors
    ///
    /// Rejects a child tree containing a scored buffered union, or any ordinary
    /// allocation and cursor-alignment failure.
    pub fn boolean_unscored(clauses: Vec<ScorerClause<'a>>) -> Result<Self, ArgusError> {
        Self::boolean_with_mode(clauses, BooleanMode::Unscored)
    }

    fn boolean_with_mode(
        clauses: Vec<ScorerClause<'a>>,
        mode: BooleanMode,
    ) -> Result<Self, ArgusError> {
        if mode == BooleanMode::Unscored
            && clauses
                .iter()
                .any(|clause| !clause.scorer.is_unscored_safe())
        {
            return Err(ArgusError::ScoredTreeForUnscoredCollector);
        }
        if matches!(
            clauses.as_slice(),
            [clause] if clause.occur == Occur::MustNot
        ) {
            return Ok(Self::empty());
        }
        let must_count = clauses
            .iter()
            .filter(|clause| clause.occur == Occur::Must)
            .count();
        let should_count = clauses
            .iter()
            .filter(|clause| clause.occur == Occur::Should)
            .count();
        let excluded_count = clauses.len() - must_count - should_count;
        let mut must = reserve_scorers("required Boolean clauses", must_count)?;
        let mut should = reserve_scorers("optional Boolean clauses", should_count)?;
        let mut excluded = reserve_scorers("excluded Boolean clauses", excluded_count)?;
        for clause in clauses {
            match clause.occur {
                Occur::Must if clause.scorer.is_explicit_empty() => return Ok(Self::empty()),
                Occur::Must => must.push(clause.scorer),
                Occur::Should if clause.scorer.is_explicit_empty() => {}
                Occur::Should => should.push(clause.scorer),
                Occur::MustNot if clause.scorer.is_explicit_empty() => {}
                Occur::MustNot => excluded.push(clause.scorer),
            }
        }

        let include = if must.is_empty() {
            match mode {
                BooleanMode::Scored => scorer_union(should)?,
                BooleanMode::Unscored => scorer_union_unscored(should)?,
            }
        } else {
            let required = scorer_intersection(must)?;
            if should.is_empty() || mode == BooleanMode::Unscored {
                required
            } else {
                let optional = scorer_union(should)?;
                Self {
                    node: ScorerNode::RequiredOptional(RequiredOptionalScorer::new(
                        required, optional,
                    )?),
                }
            }
        };
        if excluded.is_empty() {
            return Ok(include);
        }
        Ok(Self {
            node: ScorerNode::Exclude(ExcludeScorer::new(include, excluded)?),
        })
    }

    /// Current matching global document id.
    #[must_use]
    pub fn doc(&self) -> Option<u32> {
        match &self.node {
            ScorerNode::Empty => None,
            ScorerNode::Term(scorer) => scorer.doc(),
            ScorerNode::Phrase(scorer) => scorer.doc(),
            ScorerNode::Intersection(scorer) => scorer.doc(),
            ScorerNode::Union(scorer) => scorer.doc(),
            ScorerNode::UnscoredUnion(scorer) => scorer.doc(),
            ScorerNode::RequiredOptional(scorer) => scorer.doc(),
            ScorerNode::Exclude(scorer) => scorer.doc(),
        }
    }

    /// Runtime cost used for stable required-clause ordering.
    #[must_use]
    pub fn cost(&self) -> u64 {
        match &self.node {
            ScorerNode::Empty => 0,
            ScorerNode::Term(scorer) => scorer.cost,
            ScorerNode::Phrase(scorer) => scorer.cost,
            ScorerNode::Intersection(scorer) => scorer.cost(),
            ScorerNode::Union(scorer) => scorer.cost(),
            ScorerNode::UnscoredUnion(scorer) => scorer.cost(),
            ScorerNode::RequiredOptional(scorer) => scorer.cost(),
            ScorerNode::Exclude(scorer) => scorer.cost(),
        }
    }

    /// Tantivy-compatible best-effort match-count estimate.
    #[must_use]
    pub fn size_hint(&self) -> u32 {
        match &self.node {
            ScorerNode::Empty => 0,
            ScorerNode::Term(scorer) => scorer.size_hint,
            ScorerNode::Phrase(scorer) => scorer.size_hint,
            ScorerNode::Intersection(scorer) => scorer.size_hint(),
            ScorerNode::Union(scorer) => scorer.size_hint(),
            ScorerNode::UnscoredUnion(scorer) => scorer.size_hint(),
            ScorerNode::RequiredOptional(scorer) => scorer.size_hint(),
            ScorerNode::Exclude(scorer) => scorer.size_hint(),
        }
    }

    /// Live document count used for Tantivy-compatible composite estimates.
    #[must_use]
    pub fn segment_num_docs(&self) -> u32 {
        match &self.node {
            ScorerNode::Empty => 0,
            ScorerNode::Term(scorer) => scorer.segment_num_docs,
            ScorerNode::Phrase(scorer) => scorer.segment_num_docs,
            ScorerNode::Intersection(scorer) => scorer.segment_num_docs,
            ScorerNode::Union(scorer) => scorer.segment_num_docs,
            ScorerNode::UnscoredUnion(scorer) => scorer.segment_num_docs,
            ScorerNode::RequiredOptional(scorer) => scorer.segment_num_docs,
            ScorerNode::Exclude(scorer) => scorer.segment_num_docs,
        }
    }

    /// Move strictly forward to the next match.
    ///
    /// # Errors
    ///
    /// Propagates typed cursor, fieldnorm, or score-buffer failures.
    #[allow(
        clippy::should_implement_trait,
        reason = "a scorer exposes both next-match and inclusive advance operations"
    )]
    pub fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Ok(None),
            ScorerNode::Term(scorer) => scorer.next(),
            ScorerNode::Phrase(scorer) => scorer.next(),
            ScorerNode::Intersection(scorer) => scorer.next(),
            ScorerNode::Union(scorer) => scorer.next(),
            ScorerNode::UnscoredUnion(scorer) => scorer.next(),
            ScorerNode::RequiredOptional(scorer) => scorer.next(),
            ScorerNode::Exclude(scorer) => scorer.next(),
        }
    }

    /// Seek to the first match at or beyond `target`.
    ///
    /// # Errors
    ///
    /// Propagates typed cursor, fieldnorm, or score-buffer failures.
    pub fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Ok(None),
            ScorerNode::Term(scorer) => scorer.seek(target),
            ScorerNode::Phrase(scorer) => scorer.seek(target),
            ScorerNode::Intersection(scorer) => scorer.seek(target),
            ScorerNode::Union(scorer) => scorer.seek(target),
            ScorerNode::UnscoredUnion(scorer) => scorer.seek(target),
            ScorerNode::RequiredOptional(scorer) => scorer.seek(target),
            ScorerNode::Exclude(scorer) => scorer.seek(target),
        }
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Ok(SeekDangerResult::SeekLowerBound(None)),
            ScorerNode::Term(scorer) => {
                let result = if scorer.doc().is_some_and(|doc| doc < target) {
                    scorer.seek(target)?
                } else {
                    scorer.doc()
                };
                classify_seek_result(target, result)
            }
            ScorerNode::Phrase(scorer) => {
                let result = if scorer.doc().is_some_and(|doc| doc < target) {
                    scorer.seek(target)?
                } else {
                    scorer.doc()
                };
                classify_seek_result(target, result)
            }
            ScorerNode::Intersection(scorer) => scorer.seek_danger(target),
            ScorerNode::Union(scorer) => scorer.seek_danger(target),
            ScorerNode::UnscoredUnion(scorer) => scorer.seek_danger(target),
            ScorerNode::RequiredOptional(scorer) => scorer.seek_danger(target),
            ScorerNode::Exclude(scorer) => {
                let result = if scorer.doc().is_some_and(|doc| doc < target) {
                    scorer.seek(target)?
                } else {
                    scorer.doc()
                };
                classify_seek_result(target, result)
            }
        }
    }

    /// Score the current match in the pinned f32 scorer-tree order.
    ///
    /// # Errors
    ///
    /// Returns a typed error for exhausted or inconsistent cursors, missing
    /// fieldnorms, and non-finite arithmetic.
    pub fn score(&mut self) -> Result<f32, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Err(ArgusError::CursorInvariant("cannot score an empty scorer")),
            ScorerNode::Term(scorer) => scorer.score(),
            ScorerNode::Phrase(scorer) => scorer.score(),
            ScorerNode::Intersection(scorer) => scorer.score(),
            ScorerNode::Union(scorer) => scorer.score(),
            ScorerNode::UnscoredUnion(_) => Err(ArgusError::ScoreUnavailable),
            ScorerNode::RequiredOptional(scorer) => scorer.score(),
            ScorerNode::Exclude(scorer) => scorer.score(),
        }
    }

    /// Exhaustively evaluate this scorer and retain its best `limit` live hits.
    ///
    /// The phase-one heap stores only `(global_docid, score)`. Deleted documents
    /// are filtered here, after physical posting/statistics construction but
    /// before they can affect the heap threshold.
    ///
    /// # Errors
    ///
    /// Propagates typed scorer failures or a bounded heap allocation failure.
    pub fn top_k<L>(&mut self, limit: usize, live_docs: &L) -> Result<Vec<ScoredDoc>, ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        let mut collector = TopDocsCollector::new(limit, 0)?;
        collector.collect(self, live_docs)?;
        Ok(collector.finish()?.hits)
    }

    /// Exhaust this recursively unscored tree into sorted unique global docids.
    ///
    /// Use [`Self::boolean_unscored`] for every Boolean level. Term and phrase
    /// leaves are intrinsically safe because advancing them never evaluates a
    /// BM25 score.
    ///
    /// # Errors
    ///
    /// Rejects a scored buffered union anywhere in the tree and propagates
    /// cursor or bounded-allocation failures.
    pub fn collect_doc_set<L>(&mut self, live_docs: &L) -> Result<Vec<u32>, ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        let mut collector = DocSetCollector::new();
        collector.collect(self, live_docs)?;
        Ok(collector.finish())
    }

    fn collect_with<L>(
        &mut self,
        collector: &mut CollectorState<'_>,
        live_docs: &L,
    ) -> Result<(), ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        if matches!(collector, CollectorState::DocSet(_)) && !self.is_unscored_safe() {
            return Err(ArgusError::ScoredTreeForUnscoredCollector);
        }
        while let Some(doc) = self.doc() {
            match collector {
                CollectorState::TopDocs(top_docs) => {
                    // Scoring physical deleted matches before visibility
                    // filtering preserves the buffered union's f32 state.
                    let score = (top_docs.retained != 0).then(|| self.score()).transpose()?;
                    if live_docs.is_live(doc) {
                        top_docs.record_live(doc, score)?;
                    }
                }
                CollectorState::DocSet(doc_set) => {
                    if live_docs.is_live(doc) {
                        doc_set.push(doc)?;
                    }
                }
            }
            self.next()?;
        }
        Ok(())
    }
}

fn reserve_scorers<'a>(
    resource: &'static str,
    count: usize,
) -> Result<Vec<ReferenceScorer<'a>>, ArgusError> {
    let mut scorers = Vec::new();
    scorers
        .try_reserve_exact(count)
        .map_err(|_| ArgusError::Allocation { resource, count })?;
    Ok(scorers)
}

fn shared_segment_num_docs(scorers: &[ReferenceScorer<'_>]) -> Result<u32, ArgusError> {
    let segment_num_docs = scorers.first().map_or(0, ReferenceScorer::segment_num_docs);
    if scorers
        .iter()
        .any(|scorer| scorer.segment_num_docs() != segment_num_docs)
    {
        return Err(ArgusError::CursorInvariant(
            "Boolean children belong to different segment domains",
        ));
    }
    Ok(segment_num_docs)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the pinned Tantivy estimator clamps its finite non-negative result to u32 input bounds"
)]
fn estimate_intersection(mut sizes: impl Iterator<Item = u32>, segment_num_docs: u32) -> u32 {
    if segment_num_docs == 0 {
        return 0;
    }
    let Some(first) = sizes.next() else {
        return 0;
    };
    let mut co_location_factor = 1.3_f64;
    let mut estimate = f64::from(first);
    let mut smallest = estimate;
    for size in sizes {
        co_location_factor = (co_location_factor - 0.1).max(1.0);
        estimate *= (f64::from(size) / f64::from(segment_num_docs)) * co_location_factor;
        smallest = smallest.min(f64::from(size));
    }
    estimate.round().min(smallest) as u32
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the pinned Tantivy estimator clamps its finite non-negative result to segment_num_docs"
)]
fn estimate_union(sizes: impl Iterator<Item = u32>, segment_num_docs: u32) -> u32 {
    if segment_num_docs == 0 {
        return 0;
    }
    let mut not_in_any_set_probability = 1.0_f64;
    for size in sizes {
        let probability = (f64::from(size) / f64::from(segment_num_docs)) * 0.8;
        not_in_any_set_probability *= 1.0 - probability;
    }
    let estimate = (f64::from(segment_num_docs) * (1.0 - not_in_any_set_probability)).round();
    estimate.min(f64::from(segment_num_docs)) as u32
}

fn scorer_intersection(
    mut scorers: Vec<ReferenceScorer<'_>>,
) -> Result<ReferenceScorer<'_>, ArgusError> {
    match scorers.len() {
        0 => Ok(ReferenceScorer::empty()),
        1 => scorers.pop().ok_or(ArgusError::CursorInvariant(
            "required scorer count changed during lowering",
        )),
        _ => {
            let intersection = IntersectionScorer::new(scorers)?;
            if intersection.doc().is_none() {
                Ok(ReferenceScorer::empty())
            } else {
                Ok(ReferenceScorer {
                    node: ScorerNode::Intersection(intersection),
                })
            }
        }
    }
}

fn scorer_union(mut scorers: Vec<ReferenceScorer<'_>>) -> Result<ReferenceScorer<'_>, ArgusError> {
    match scorers.len() {
        0 => Ok(ReferenceScorer::empty()),
        1 => scorers.pop().ok_or(ArgusError::CursorInvariant(
            "optional scorer count changed during lowering",
        )),
        _ => Ok(ReferenceScorer {
            node: ScorerNode::Union(BufferedUnionScorer::new(scorers)?),
        }),
    }
}

fn scorer_union_unscored(
    mut scorers: Vec<ReferenceScorer<'_>>,
) -> Result<ReferenceScorer<'_>, ArgusError> {
    match scorers.len() {
        0 => Ok(ReferenceScorer::empty()),
        1 => scorers.pop().ok_or(ArgusError::CursorInvariant(
            "unscored optional scorer count changed during lowering",
        )),
        _ => Ok(ReferenceScorer {
            node: ScorerNode::UnscoredUnion(UnscoredUnionScorer::new(scorers)?),
        }),
    }
}

/// Lazy scoreless disjunction used only by doc-set collection.
///
/// The scored parity path keeps [`BufferedUnionScorer`] unchanged. This cursor
/// advances all children at the current minimum document together, so it
/// deduplicates matches without evaluating or retaining any BM25 contribution.
struct UnscoredUnionScorer<'a> {
    active: Vec<ReferenceScorer<'a>>,
    current: Option<u32>,
    segment_num_docs: u32,
}

impl<'a> UnscoredUnionScorer<'a> {
    fn new(mut scorers: Vec<ReferenceScorer<'a>>) -> Result<Self, ArgusError> {
        let segment_num_docs = shared_segment_num_docs(&scorers)?;
        scorers.retain(|scorer| scorer.doc().is_some());
        let current = scorers.iter().filter_map(ReferenceScorer::doc).min();
        Ok(Self {
            active: scorers,
            current,
            segment_num_docs,
        })
    }

    const fn doc(&self) -> Option<u32> {
        self.current
    }

    fn cost(&self) -> u64 {
        self.active
            .iter()
            .map(ReferenceScorer::cost)
            .fold(0_u64, u64::saturating_add)
    }

    fn size_hint(&self) -> u32 {
        estimate_union(
            self.active.iter().map(ReferenceScorer::size_hint),
            self.segment_num_docs,
        )
    }

    fn refresh_current(&mut self) -> Option<u32> {
        self.active.retain(|scorer| scorer.doc().is_some());
        self.current = self.active.iter().filter_map(ReferenceScorer::doc).min();
        self.current
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        let Some(current) = self.current else {
            return Ok(None);
        };
        for scorer in &mut self.active {
            if scorer.doc() == Some(current) {
                scorer.next()?;
            }
        }
        Ok(self.refresh_current())
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        for scorer in &mut self.active {
            if scorer.doc().is_some_and(|doc| doc < target) {
                scorer.advance(target)?;
            }
        }
        Ok(self.refresh_current())
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        let result = self.seek(target)?;
        classify_seek_result(target, result)
    }
}

struct IntersectionScorer<'a> {
    scorers: Vec<ReferenceScorer<'a>>,
    current: Option<u32>,
    segment_num_docs: u32,
}

impl<'a> IntersectionScorer<'a> {
    fn new(mut scorers: Vec<ReferenceScorer<'a>>) -> Result<Self, ArgusError> {
        let segment_num_docs = shared_segment_num_docs(&scorers)?;
        scorers.sort_by_key(ReferenceScorer::cost);
        let mut scorer = Self {
            scorers,
            current: None,
            segment_num_docs,
        };
        scorer.align()?;
        Ok(scorer)
    }

    const fn doc(&self) -> Option<u32> {
        self.current
    }

    fn cost(&self) -> u64 {
        self.scorers.first().map_or(0, ReferenceScorer::cost)
    }

    fn size_hint(&self) -> u32 {
        estimate_intersection(
            self.scorers.iter().map(ReferenceScorer::size_hint),
            self.segment_num_docs,
        )
    }

    fn align(&mut self) -> Result<Option<u32>, ArgusError> {
        let mut candidate = match self.scorers.iter().filter_map(ReferenceScorer::doc).max() {
            Some(candidate) if self.scorers.iter().all(|scorer| scorer.doc().is_some()) => {
                candidate
            }
            _ => {
                self.current = None;
                return Ok(None);
            }
        };
        'outer: loop {
            for scorer in &mut self.scorers {
                let Some(doc) = scorer.advance(candidate)? else {
                    self.current = None;
                    return Ok(None);
                };
                if doc > candidate {
                    candidate = doc;
                    continue 'outer;
                }
            }
            self.current = Some(candidate);
            return Ok(self.current);
        }
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        let Some(current) = self.current else {
            return Ok(None);
        };
        let Some(mut candidate) = current.checked_add(1) else {
            self.scorers[0].next()?;
            self.current = None;
            return Ok(None);
        };

        'outer: loop {
            let Some(driver_doc) = self.scorers[0].advance(candidate)? else {
                self.current = None;
                return Ok(None);
            };
            candidate = driver_doc;
            for scorer in &mut self.scorers[1..] {
                match scorer.seek_danger(candidate)? {
                    SeekDangerResult::Found => {}
                    SeekDangerResult::SeekLowerBound(Some(lower_bound)) => {
                        if lower_bound <= candidate {
                            return Err(ArgusError::CursorInvariant(
                                "danger seek lower bound did not strictly advance",
                            ));
                        }
                        candidate = lower_bound;
                        continue 'outer;
                    }
                    SeekDangerResult::SeekLowerBound(None) => {
                        self.current = None;
                        return Ok(None);
                    }
                }
            }
            self.current = Some(candidate);
            return Ok(self.current);
        }
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        self.scorers[0].advance(target)?;
        self.align()
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        for scorer in &mut self.scorers {
            match scorer.seek_danger(target)? {
                SeekDangerResult::Found => {}
                lower_bound @ SeekDangerResult::SeekLowerBound(_) => return Ok(lower_bound),
            }
        }
        self.current = Some(target);
        Ok(SeekDangerResult::Found)
    }

    fn score(&mut self) -> Result<f32, ArgusError> {
        let doc = self.current.ok_or(ArgusError::CursorInvariant(
            "cannot score exhausted intersection",
        ))?;
        let (left_slice, rest) = self.scorers.split_at_mut(1);
        let left = left_slice[0].score()?;
        let (right_slice, others) = rest.split_at_mut(1);
        let right = right_slice[0].score()?;
        let mut other_sum = 0.0_f32;
        for scorer in others {
            other_sum += scorer.score()?;
        }
        finite_score(left + right + other_sum, doc)
    }
}

struct BufferedUnionScorer<'a> {
    active: Vec<ReferenceScorer<'a>>,
    score_window: Vec<Option<f32>>,
    window_start: Option<u32>,
    scan_offset: usize,
    current: Option<u32>,
    current_score: f32,
    segment_num_docs: u32,
}

impl<'a> BufferedUnionScorer<'a> {
    fn new(mut scorers: Vec<ReferenceScorer<'a>>) -> Result<Self, ArgusError> {
        let segment_num_docs = shared_segment_num_docs(&scorers)?;
        scorers.retain(|scorer| scorer.doc().is_some());
        let mut score_window = Vec::new();
        score_window
            .try_reserve_exact(UNION_HORIZON)
            .map_err(|_| ArgusError::Allocation {
                resource: "buffered union score window",
                count: UNION_HORIZON,
            })?;
        score_window.resize(UNION_HORIZON, None);
        let mut scorer = Self {
            active: scorers,
            score_window,
            window_start: None,
            scan_offset: 0,
            current: None,
            current_score: 0.0,
            segment_num_docs,
        };
        if scorer.refill()? {
            scorer.advance_buffered();
        }
        Ok(scorer)
    }

    const fn doc(&self) -> Option<u32> {
        self.current
    }

    fn cost(&self) -> u64 {
        self.active
            .iter()
            .map(ReferenceScorer::cost)
            .fold(0_u64, u64::saturating_add)
    }

    fn size_hint(&self) -> u32 {
        estimate_union(
            self.active.iter().map(ReferenceScorer::size_hint),
            self.segment_num_docs,
        )
    }

    fn clear_buffer(&mut self) {
        self.score_window.fill(None);
        self.scan_offset = 0;
        self.current = None;
        self.current_score = 0.0;
    }

    fn refill(&mut self) -> Result<bool, ArgusError> {
        self.clear_buffer();
        let Some(window_start) = self.active.iter().filter_map(ReferenceScorer::doc).min() else {
            self.window_start = None;
            return Ok(false);
        };
        self.window_start = Some(window_start);
        let horizon_end = u64::from(window_start) + UNION_HORIZON as u64;
        let mut index = 0;
        while index < self.active.len() {
            loop {
                let Some(doc) = self.active[index].doc() else {
                    self.active.swap_remove(index);
                    break;
                };
                if u64::from(doc) >= horizon_end {
                    index += 1;
                    break;
                }
                let offset = usize::try_from(u64::from(doc) - u64::from(window_start))
                    .map_err(|_| ArgusError::CursorInvariant("union offset does not fit usize"))?;
                let contribution = self.active[index].score()?;
                let total = self.score_window[offset].unwrap_or(0.0) + contribution;
                self.score_window[offset] = Some(finite_score(total, doc)?);
                self.active[index].next()?;
                if self.active[index].doc().is_none() {
                    self.active.swap_remove(index);
                    break;
                }
            }
        }
        Ok(true)
    }

    fn advance_buffered(&mut self) -> Option<u32> {
        let start = self.window_start?;
        while self.scan_offset < self.score_window.len() {
            let offset = self.scan_offset;
            self.scan_offset += 1;
            if let Some(score) = self.score_window[offset].take() {
                let doc = u64::from(start) + offset as u64;
                let doc = u32::try_from(doc).ok()?;
                self.current = Some(doc);
                self.current_score = score;
                return self.current;
            }
        }
        None
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.current.is_none() {
            return Ok(None);
        }
        if self.advance_buffered().is_some() {
            return Ok(self.current);
        }
        if !self.refill()? {
            return Ok(None);
        }
        Ok(self.advance_buffered())
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        if self.window_start.is_some_and(|start| {
            u64::from(target).saturating_sub(u64::from(start)) < UNION_HORIZON as u64
        }) {
            while self.current.is_some_and(|doc| doc < target) {
                self.next()?;
            }
            return Ok(self.current);
        }

        self.clear_buffer();
        let mut index = 0;
        while index < self.active.len() {
            if self.active[index].doc().is_some_and(|doc| doc < target) {
                self.active[index].advance(target)?;
            }
            if self.active[index].doc().is_none() {
                self.active.swap_remove(index);
            } else {
                index += 1;
            }
        }
        if !self.refill()? {
            return Ok(None);
        }
        Ok(self.advance_buffered())
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        if self.current.is_none() {
            return Ok(SeekDangerResult::SeekLowerBound(None));
        }
        if self.window_start.is_some_and(|start| {
            u64::from(target).saturating_sub(u64::from(start)) < UNION_HORIZON as u64
                && target >= start
        }) {
            return classify_seek_result(target, self.seek(target)?);
        }

        let mut minimum_lower_bound = None;
        let mut found = false;
        for scorer in &mut self.active {
            match scorer.seek_danger(target)? {
                SeekDangerResult::Found => {
                    found = true;
                    break;
                }
                SeekDangerResult::SeekLowerBound(Some(lower_bound)) => {
                    minimum_lower_bound = Some(
                        minimum_lower_bound
                            .map_or(lower_bound, |minimum: u32| minimum.min(lower_bound)),
                    );
                }
                SeekDangerResult::SeekLowerBound(None) => {}
            }
        }
        if found {
            let result = self.seek(target)?;
            if result != Some(target) {
                return Err(ArgusError::CursorInvariant(
                    "union danger seek found a child but not the union target",
                ));
            }
            return Ok(SeekDangerResult::Found);
        }
        Ok(SeekDangerResult::SeekLowerBound(minimum_lower_bound))
    }
    fn score(&self) -> Result<f32, ArgusError> {
        let doc = self
            .current
            .ok_or(ArgusError::CursorInvariant("cannot score exhausted union"))?;
        finite_score(self.current_score, doc)
    }
}

struct RequiredOptionalScorer<'a> {
    required: Box<ReferenceScorer<'a>>,
    optional: Box<ReferenceScorer<'a>>,
    score_cache: Option<(u32, f32)>,
    segment_num_docs: u32,
}

impl<'a> RequiredOptionalScorer<'a> {
    fn new(
        required: ReferenceScorer<'a>,
        optional: ReferenceScorer<'a>,
    ) -> Result<Self, ArgusError> {
        if required.segment_num_docs() != optional.segment_num_docs() {
            return Err(ArgusError::CursorInvariant(
                "required and optional scorers belong to different segment domains",
            ));
        }
        let segment_num_docs = required.segment_num_docs();
        Ok(Self {
            required: Box::new(required),
            optional: Box::new(optional),
            score_cache: None,
            segment_num_docs,
        })
    }

    fn doc(&self) -> Option<u32> {
        self.required.doc()
    }

    fn cost(&self) -> u64 {
        self.required.cost()
    }

    fn size_hint(&self) -> u32 {
        self.required.size_hint()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        self.score_cache = None;
        self.required.next()
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.doc());
        }
        self.score_cache = None;
        self.required.advance(target)
    }

    fn seek_danger(&mut self, target: u32) -> Result<SeekDangerResult, ArgusError> {
        self.score_cache = None;
        self.required.seek_danger(target)
    }

    fn score(&mut self) -> Result<f32, ArgusError> {
        let doc = self.doc().ok_or(ArgusError::CursorInvariant(
            "cannot score exhausted required-optional scorer",
        ))?;
        if let Some((cached_doc, score)) = self.score_cache
            && cached_doc == doc
        {
            return Ok(score);
        }
        let mut score = 0.0_f32;
        score += self.required.score()?;
        if self
            .optional
            .doc()
            .is_some_and(|optional_doc| optional_doc <= doc)
            && self.optional.advance(doc)? == Some(doc)
        {
            score += self.optional.score()?;
        }
        let score = finite_score(score, doc)?;
        self.score_cache = Some((doc, score));
        Ok(score)
    }
}

struct ExcludeScorer<'a> {
    include: Box<ReferenceScorer<'a>>,
    excluded: Vec<ReferenceScorer<'a>>,
    segment_num_docs: u32,
}

impl<'a> ExcludeScorer<'a> {
    fn new(
        include: ReferenceScorer<'a>,
        excluded: Vec<ReferenceScorer<'a>>,
    ) -> Result<Self, ArgusError> {
        let segment_num_docs = if include.is_explicit_empty() {
            shared_segment_num_docs(&excluded)?
        } else {
            include.segment_num_docs()
        };
        if !include.is_explicit_empty()
            && excluded
                .iter()
                .any(|scorer| scorer.segment_num_docs() != segment_num_docs)
        {
            return Err(ArgusError::CursorInvariant(
                "include and exclusion scorers belong to different segment domains",
            ));
        }
        let mut scorer = Self {
            include: Box::new(include),
            excluded,
            segment_num_docs,
        };
        scorer.skip_excluded()?;
        Ok(scorer)
    }

    fn doc(&self) -> Option<u32> {
        self.include.doc()
    }

    fn cost(&self) -> u64 {
        u64::from(self.size_hint())
    }

    fn size_hint(&self) -> u32 {
        self.include.size_hint()
    }

    fn contains(&mut self, doc: u32) -> Result<bool, ArgusError> {
        for scorer in &mut self.excluded {
            if scorer.seek_danger(doc)? == SeekDangerResult::Found {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn skip_excluded(&mut self) -> Result<Option<u32>, ArgusError> {
        while let Some(doc) = self.include.doc() {
            if !self.contains(doc)? {
                return Ok(Some(doc));
            }
            self.include.next()?;
        }
        Ok(None)
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        if self.include.doc().is_none() {
            return Ok(None);
        }
        self.include.next()?;
        self.skip_excluded()
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.include.doc().is_some_and(|doc| doc >= target) {
            return Ok(self.include.doc());
        }
        self.include.advance(target)?;
        self.skip_excluded()
    }

    fn score(&mut self) -> Result<f32, ArgusError> {
        self.include.score()
    }
}

fn finite_score(score: f32, global_docid: u32) -> Result<f32, ArgusError> {
    if score.is_finite() {
        Ok(score)
    } else {
        Err(ArgusError::NonFiniteScore { global_docid })
    }
}

/// Live-document membership used only at result collection.
pub trait LiveDocs {
    /// Return whether a physical posting is visible in the current snapshot.
    fn is_live(&self, global_docid: u32) -> bool;
}

impl<F> LiveDocs for F
where
    F: Fn(u32) -> bool,
{
    fn is_live(&self, global_docid: u32) -> bool {
        self(global_docid)
    }
}

/// Membership policy for snapshots without tombstones.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllLiveDocs;

impl LiveDocs for AllLiveDocs {
    fn is_live(&self, _global_docid: u32) -> bool {
        true
    }
}

/// Phase-one exhaustive result carrying no stored metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredDoc {
    /// Stable global Quill document id.
    pub global_docid: u32,
    /// Exact contract-mode f32 score.
    pub score: f32,
}

/// Final globally merged top-doc page.
///
/// `total_count` is present only when the collector was created with
/// [`TopDocsCollector::with_exact_count`]. Hits are already offset-sliced and
/// remain in the canonical `(score desc, global_docid asc)` order.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectedTopDocs {
    /// Page-local ranked hits.
    pub hits: Vec<ScoredDoc>,
    /// Exact number of live matches across every collected scorer.
    pub total_count: Option<u64>,
}

/// Cross-segment exhaustive top-doc accumulator.
///
/// Feed every per-segment scorer into one instance, then call [`Self::finish`]
/// exactly once. Offset slicing happens only after the global heap is sorted,
/// preventing per-segment pagination from discarding valid winners.
#[derive(Debug)]
pub struct TopDocsCollector {
    limit: usize,
    offset: usize,
    retained: usize,
    exact_count: bool,
    total_count: u64,
    heap: BinaryHeap<HeapEntry>,
}

impl TopDocsCollector {
    /// Build a count-free top-k collector with Tantivy-compatible offset.
    ///
    /// # Errors
    ///
    /// Rejects `limit + offset` overflow and bounded heap allocation failure.
    pub fn new(limit: usize, offset: usize) -> Result<Self, ArgusError> {
        Self::build(limit, offset, false)
    }

    /// Build a top-k collector that also exhaustively counts every live match.
    ///
    /// `limit == 0` retains no scores but still counts all matches.
    ///
    /// # Errors
    ///
    /// Rejects `limit + offset` overflow and bounded heap allocation failure.
    pub fn with_exact_count(limit: usize, offset: usize) -> Result<Self, ArgusError> {
        Self::build(limit, offset, true)
    }

    fn build(limit: usize, offset: usize, exact_count: bool) -> Result<Self, ArgusError> {
        let retained = if limit == 0 {
            0
        } else {
            limit
                .checked_add(offset)
                .ok_or(ArgusError::CollectorWindowOverflow { limit, offset })?
        };
        let mut heap = BinaryHeap::new();
        heap.try_reserve_exact(retained)
            .map_err(|_| ArgusError::Allocation {
                resource: "top-k heap",
                count: retained,
            })?;
        Ok(Self {
            limit,
            offset,
            retained,
            exact_count,
            total_count: 0,
            heap,
        })
    }

    /// Consume one segment scorer into this global accumulator.
    ///
    /// # Errors
    ///
    /// Propagates scorer, visibility, count, and allocation failures.
    pub fn collect<L>(
        &mut self,
        scorer: &mut ReferenceScorer<'_>,
        live_docs: &L,
    ) -> Result<(), ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        if self.retained == 0 && !self.exact_count {
            return Ok(());
        }
        let mut state = CollectorState::TopDocs(self);
        scorer.collect_with(&mut state, live_docs)
    }

    fn record_live(&mut self, global_docid: u32, score: Option<f32>) -> Result<(), ArgusError> {
        if self.exact_count {
            self.total_count = self
                .total_count
                .checked_add(1)
                .ok_or(ArgusError::MatchCountOverflow)?;
        }
        if self.retained == 0 {
            return Ok(());
        }
        let score = score.ok_or(ArgusError::CursorInvariant(
            "top-k collector did not score a retained match",
        ))?;
        let entry = HeapEntry {
            global_docid,
            score,
        };
        if self.heap.len() < self.retained {
            self.heap.push(entry);
        } else if self.heap.peek().is_some_and(|cutoff| entry < *cutoff) {
            let _ = self.heap.pop();
            self.heap.push(entry);
        }
        Ok(())
    }

    /// Sort the global heap, apply offset once, and return page winners.
    ///
    /// # Errors
    ///
    /// Returns a typed allocation failure when the final compact hit vector
    /// cannot be reserved.
    pub fn finish(self) -> Result<CollectedTopDocs, ArgusError> {
        let winner_count = self.heap.len();
        let mut hits = Vec::new();
        hits.try_reserve_exact(winner_count)
            .map_err(|_| ArgusError::Allocation {
                resource: "top-k winners",
                count: winner_count,
            })?;
        hits.extend(self.heap.into_iter().map(ScoredDoc::from));
        hits.sort_unstable_by(|left, right| compare_scored_best_first(*left, *right));
        let start = self.offset.min(hits.len());
        let end = start.saturating_add(self.limit).min(hits.len());
        if start != 0 {
            hits.copy_within(start..end, 0);
        }
        hits.truncate(end - start);
        Ok(CollectedTopDocs {
            hits,
            total_count: self.exact_count.then_some(self.total_count),
        })
    }
}

/// Cross-segment scoreless global-document set accumulator.
#[derive(Debug, Default)]
pub struct DocSetCollector {
    docs: Vec<u32>,
}

impl DocSetCollector {
    /// Build an empty scoreless collector.
    #[must_use]
    pub const fn new() -> Self {
        Self { docs: Vec::new() }
    }

    /// Consume one recursively unscored segment scorer.
    ///
    /// # Errors
    ///
    /// Rejects scored buffered unions and propagates cursor or allocation
    /// failures.
    pub fn collect<L>(
        &mut self,
        scorer: &mut ReferenceScorer<'_>,
        live_docs: &L,
    ) -> Result<(), ArgusError>
    where
        L: LiveDocs + ?Sized,
    {
        let mut state = CollectorState::DocSet(self);
        scorer.collect_with(&mut state, live_docs)
    }

    fn push(&mut self, global_docid: u32) -> Result<(), ArgusError> {
        if self.docs.len() == self.docs.capacity() {
            self.docs
                .try_reserve(1)
                .map_err(|_| ArgusError::Allocation {
                    resource: "doc-set matches",
                    count: self.docs.len().saturating_add(1),
                })?;
        }
        self.docs.push(global_docid);
        Ok(())
    }

    /// Return deterministic sorted unique global docids.
    #[must_use]
    pub fn finish(mut self) -> Vec<u32> {
        self.docs.sort_unstable();
        self.docs.dedup();
        self.docs
    }
}

enum CollectorState<'a> {
    TopDocs(&'a mut TopDocsCollector),
    DocSet(&'a mut DocSetCollector),
}

/// Winner after phase-two external-id materialization.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedHit {
    /// Stable global Quill document id.
    pub global_docid: u32,
    /// External document id resolved only after top-k selection.
    pub document_id: DocId,
    /// Exact contract-mode f32 score.
    pub score: f32,
}

/// Resolve external document ids for already-selected winners only.
///
/// # Errors
///
/// Propagates the resolver's typed [`ArgusError`] or a bounded output
/// allocation failure.
pub fn materialize_doc_ids(
    winners: &[ScoredDoc],
    mut resolve: impl FnMut(u32) -> Result<DocId, ArgusError>,
) -> Result<Vec<MaterializedHit>, ArgusError> {
    let mut hits = Vec::new();
    hits.try_reserve_exact(winners.len())
        .map_err(|_| ArgusError::Allocation {
            resource: "materialized hits",
            count: winners.len(),
        })?;
    for winner in winners {
        hits.push(MaterializedHit {
            global_docid: winner.global_docid,
            document_id: resolve(winner.global_docid)?,
            score: winner.score,
        });
    }
    Ok(hits)
}

#[derive(Clone, Copy, Debug)]
struct HeapEntry {
    global_docid: u32,
    score: f32,
}

impl From<HeapEntry> for ScoredDoc {
    fn from(entry: HeapEntry) -> Self {
        Self {
            global_docid: entry.global_docid,
            score: entry.score,
        }
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.global_docid == other.global_docid && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.score.total_cmp(&other.score) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => self.global_docid.cmp(&other.global_docid),
        }
    }
}

fn compare_scored_best_first(left: ScoredDoc, right: ScoredDoc) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.global_docid.cmp(&right.global_docid))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{BM25_B, fieldnorm_to_id, id_to_fieldnorm};
    use crate::quiver::{
        DocLenFieldInput, EncodedDocLenSection, EncodedPositionList, EncodedPostingList, Posting,
    };

    #[derive(Clone, Debug)]
    struct VecCursor {
        postings: Vec<Posting>,
        index: usize,
        cost: u64,
        segment_num_docs: u32,
    }

    impl VecCursor {
        const fn new(postings: Vec<Posting>, cost: u64, segment_num_docs: u32) -> Self {
            Self {
                postings,
                index: 0,
                cost,
                segment_num_docs,
            }
        }

        fn current(&self) -> Option<Posting> {
            self.postings.get(self.index).copied()
        }
    }

    impl PostingCursor for VecCursor {
        fn doc(&self) -> Option<u32> {
            self.current().map(|posting| posting.doc_id)
        }

        fn freq(&self) -> Option<u32> {
            self.current().map(|posting| posting.freq)
        }

        fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
            None
        }

        fn size_hint(&self) -> u32 {
            u32::try_from(self.postings.len()).unwrap_or(u32::MAX)
        }

        fn cost(&self) -> u64 {
            self.cost
        }

        fn segment_num_docs(&self) -> u32 {
            self.segment_num_docs
        }

        fn next(&mut self) -> Result<Option<u32>, ArgusError> {
            if self.index < self.postings.len() {
                self.index += 1;
            }
            Ok(self.doc())
        }

        fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
            let tail = self.postings.get(self.index..).unwrap_or_default();
            self.index += tail.partition_point(|posting| posting.doc_id < target);
            Ok(self.doc())
        }
    }

    #[derive(Clone, Debug)]
    struct PositionedVecCursor {
        postings: Vec<Posting>,
        positions: Vec<Vec<u32>>,
        index: usize,
        cost: u64,
        segment_num_docs: u32,
    }

    impl PositionedVecCursor {
        fn new(rows: Vec<(u32, Vec<u32>)>, cost: u64, segment_num_docs: u32) -> Self {
            let mut postings = Vec::with_capacity(rows.len());
            let mut positions = Vec::with_capacity(rows.len());
            for (doc, row_positions) in rows {
                let frequency = u32::try_from(row_positions.len()).expect("fixture frequency fits");
                postings.push(Posting::new(doc, frequency));
                positions.push(row_positions);
            }
            Self {
                postings,
                positions,
                index: 0,
                cost,
                segment_num_docs,
            }
        }

        fn current(&self) -> Option<Posting> {
            self.postings.get(self.index).copied()
        }
    }

    impl PositionsReader for PositionedVecCursor {
        fn decode_positions(
            &self,
            posting_ordinal: u32,
            output: &mut Vec<u32>,
        ) -> Result<(), ArgusError> {
            let ordinal = usize::try_from(posting_ordinal).map_err(|_| {
                ArgusError::CursorInvariant("test position ordinal does not fit usize")
            })?;
            if ordinal != self.index {
                return Err(ArgusError::CursorInvariant(
                    "test position handle is no longer current",
                ));
            }
            let positions = self
                .positions
                .get(ordinal)
                .ok_or(ArgusError::CursorInvariant(
                    "test position ordinal is out of range",
                ))?;
            output
                .try_reserve_exact(positions.len())
                .map_err(|_| ArgusError::Allocation {
                    resource: "test decoded positions",
                    count: positions.len(),
                })?;
            output.extend_from_slice(positions);
            Ok(())
        }
    }

    impl PostingCursor for PositionedVecCursor {
        fn doc(&self) -> Option<u32> {
            self.current().map(|posting| posting.doc_id)
        }

        fn freq(&self) -> Option<u32> {
            self.current().map(|posting| posting.freq)
        }

        fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
            self.current().and_then(|_| {
                u32::try_from(self.index)
                    .ok()
                    .map(|ordinal| PositionsHandle::new(self, ordinal))
            })
        }

        fn size_hint(&self) -> u32 {
            u32::try_from(self.postings.len()).unwrap_or(u32::MAX)
        }

        fn cost(&self) -> u64 {
            self.cost
        }

        fn segment_num_docs(&self) -> u32 {
            self.segment_num_docs
        }

        fn next(&mut self) -> Result<Option<u32>, ArgusError> {
            if self.index < self.postings.len() {
                self.index += 1;
            }
            Ok(self.doc())
        }

        fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
            let tail = self.postings.get(self.index..).unwrap_or_default();
            self.index += tail.partition_point(|posting| posting.doc_id < target);
            Ok(self.doc())
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum CursorFault {
        StickyNext,
        BackwardNext,
        BelowTargetAdvance,
        MovesDespiteSatisfiedTarget,
        ResurrectsAfterExhaustion,
    }

    #[derive(Clone, Debug)]
    struct FaultCursor {
        current: Option<Posting>,
        fault: CursorFault,
        moves: u8,
    }

    impl FaultCursor {
        const fn new(fault: CursorFault) -> Self {
            let doc_id = match fault {
                CursorFault::BelowTargetAdvance | CursorFault::ResurrectsAfterExhaustion => 1,
                CursorFault::MovesDespiteSatisfiedTarget => 5,
                CursorFault::StickyNext | CursorFault::BackwardNext => 3,
            };
            Self {
                current: Some(Posting::new(doc_id, 1)),
                fault,
                moves: 0,
            }
        }
    }

    impl PostingCursor for FaultCursor {
        fn doc(&self) -> Option<u32> {
            self.current.map(|posting| posting.doc_id)
        }

        fn freq(&self) -> Option<u32> {
            self.current.map(|posting| posting.freq)
        }

        fn positions_handle(&self) -> Option<PositionsHandle<'_>> {
            None
        }

        fn size_hint(&self) -> u32 {
            if matches!(self.fault, CursorFault::ResurrectsAfterExhaustion) {
                2
            } else {
                1
            }
        }

        fn segment_num_docs(&self) -> u32 {
            8
        }

        fn next(&mut self) -> Result<Option<u32>, ArgusError> {
            match self.fault {
                CursorFault::StickyNext => {}
                CursorFault::BackwardNext => self.current = Some(Posting::new(2, 1)),
                CursorFault::ResurrectsAfterExhaustion if self.moves == 0 => {
                    self.current = None;
                }
                CursorFault::ResurrectsAfterExhaustion => {
                    self.current = Some(Posting::new(2, 1));
                }
                CursorFault::BelowTargetAdvance | CursorFault::MovesDespiteSatisfiedTarget => {
                    self.current = None;
                }
            }
            self.moves = self.moves.saturating_add(1);
            Ok(self.doc())
        }

        fn advance(&mut self, _target: u32) -> Result<Option<u32>, ArgusError> {
            match self.fault {
                CursorFault::BelowTargetAdvance => self.current = Some(Posting::new(3, 1)),
                CursorFault::MovesDespiteSatisfiedTarget => {
                    self.current = Some(Posting::new(6, 1));
                }
                CursorFault::ResurrectsAfterExhaustion if self.current.is_none() => {
                    self.current = Some(Posting::new(2, 1));
                }
                CursorFault::StickyNext
                | CursorFault::BackwardNext
                | CursorFault::ResurrectsAfterExhaustion => {}
            }
            self.moves = self.moves.saturating_add(1);
            Ok(self.doc())
        }
    }

    fn snapshot(
        field_ord: u16,
        total_tokens: u64,
        doc_count: u64,
    ) -> Result<Bm25FieldSnapshot, ArgusError> {
        Bm25FieldSnapshot::new(SnapshotFieldStats {
            field_ord,
            total_tokens,
            doc_count,
        })
    }

    fn term<'a>(
        postings: Vec<Posting>,
        fieldnorms: DocLenField<'a>,
        snapshot: &Bm25FieldSnapshot,
        snapshot_doc_freq: u64,
        cost: u64,
        boost: f32,
    ) -> Result<ReferenceScorer<'a>, ArgusError> {
        let segment_num_docs = u32::try_from(snapshot.doc_count()).unwrap_or(u32::MAX);
        Ok(ReferenceScorer::term(TermScorer::new(
            VecCursor::new(postings, cost, segment_num_docs),
            fieldnorms,
            snapshot.clone(),
            snapshot_doc_freq,
            boost,
        )?))
    }

    fn postings(docs: &[u32]) -> Vec<Posting> {
        docs.iter().map(|&doc| Posting::new(doc, 1)).collect()
    }

    fn expected_term_score(
        snapshot: &Bm25FieldSnapshot,
        doc_freq: u64,
        fieldnorm_id: u8,
        frequency: u32,
        boost: f32,
    ) -> f32 {
        let average = snapshot
            .average_field_length()
            .expect("non-empty scored fixture");
        let decoded = id_to_fieldnorm(fieldnorm_id);
        let norm = BM25_K1 * (1.0 - BM25_B + BM25_B * decoded as f32 / average);
        let mut weight = idf(doc_freq, snapshot.doc_count()) * (1.0 + BM25_K1);
        weight *= boost;
        let frequency = frequency as f32;
        weight * (frequency / (frequency + norm))
    }

    fn expected_phrase_score(
        snapshot: &Bm25FieldSnapshot,
        doc_freqs: &[u64],
        fieldnorm_id: u8,
        frequency: u32,
        boost: f32,
    ) -> f32 {
        let average = snapshot
            .average_field_length()
            .expect("non-empty scored fixture");
        let decoded = id_to_fieldnorm(fieldnorm_id);
        let norm = BM25_K1 * (1.0 - BM25_B + BM25_B * decoded as f32 / average);
        let mut idf_sum = 0.0_f32;
        for &doc_freq in doc_freqs {
            idf_sum += idf(doc_freq, snapshot.doc_count());
        }
        let mut weight = idf_sum * (1.0 + BM25_K1);
        weight *= boost;
        let frequency = frequency as f32;
        weight * (frequency / (frequency + norm))
    }

    #[test]
    fn term_scorer_rejects_nonprogress_backward_seek_and_resurrection()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1); 8];
        let encoded =
            EncodedDocLenSection::encode(0, 8, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 8, 8)?;
        let scorer =
            |fault| TermScorer::new(FaultCursor::new(fault), field, snapshot.clone(), 2, 1.0);

        let mut sticky = scorer(CursorFault::StickyNext)?;
        assert!(matches!(sticky.next(), Err(ArgusError::CursorInvariant(_))));
        let mut backward = scorer(CursorFault::BackwardNext)?;
        assert!(matches!(
            backward.next(),
            Err(ArgusError::CursorInvariant(_))
        ));
        let mut below_target = scorer(CursorFault::BelowTargetAdvance)?;
        assert!(matches!(
            below_target.seek(4),
            Err(ArgusError::CursorInvariant(_))
        ));
        let mut moved = scorer(CursorFault::MovesDespiteSatisfiedTarget)?;
        assert!(matches!(moved.seek(4), Err(ArgusError::CursorInvariant(_))));
        let mut resurrecting = scorer(CursorFault::ResurrectsAfterExhaustion)?;
        assert_eq!(resurrecting.next()?, None);
        assert!(matches!(
            resurrecting.next(),
            Err(ArgusError::CursorInvariant(_))
        ));

        assert!(matches!(
            TermScorer::new(VecCursor::new(Vec::new(), 1, 8), field, snapshot, 2, 1.0),
            Err(ArgusError::CursorInvariant(_))
        ));
        Ok(())
    }

    #[test]
    fn sealed_cursor_adapter_preserves_option_state_and_max_docid() -> Result<(), ArgusError> {
        let source = [
            Posting::new(3, 2),
            Posting::new(130, 1),
            Posting::new(u32::MAX, 7),
        ];
        let encoded = EncodedPostingList::encode(&source)?;
        let list = encoded.posting_list()?;
        let expected_positions = [1, 4, 9, 0, 1, 2, 3, 4, 5, 6];
        let encoded_positions = EncodedPositionList::encode(&source, &expected_positions)?;
        let positions = encoded_positions.position_list(&list)?;
        let mut cursor = SealedPostingCursor::with_positions(&positions, 3)?;

        assert_eq!(PostingCursor::doc(&cursor), Some(3));
        assert_eq!(PostingCursor::freq(&cursor), Some(2));
        let handle = PostingCursor::positions_handle(&cursor).expect("position handle");
        assert_eq!(handle.posting_ordinal(), 0);
        let mut decoded_positions = vec![u32::MAX];
        handle.decode_into(&mut decoded_positions)?;
        assert_eq!(decoded_positions, vec![1, 4]);
        assert_eq!(PostingCursor::advance(&mut cursor, 3)?, Some(3));
        assert_eq!(PostingCursor::advance(&mut cursor, 4)?, Some(130));
        let handle = PostingCursor::positions_handle(&cursor).expect("position handle");
        assert_eq!(handle.posting_ordinal(), 1);
        handle.decode_into(&mut decoded_positions)?;
        assert_eq!(decoded_positions, vec![9]);
        assert_eq!(
            PostingCursor::advance(&mut cursor, u32::MAX)?,
            Some(u32::MAX)
        );
        assert_eq!(PostingCursor::freq(&cursor), Some(7));
        let handle = PostingCursor::positions_handle(&cursor).expect("position handle");
        assert_eq!(handle.posting_ordinal(), 2);
        handle.decode_into(&mut decoded_positions)?;
        assert_eq!(decoded_positions, vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(PostingCursor::next(&mut cursor)?, None);
        assert_eq!(PostingCursor::next(&mut cursor)?, None);
        assert_eq!(PostingCursor::doc(&cursor), None);
        assert_eq!(PostingCursor::freq(&cursor), None);
        assert!(PostingCursor::positions_handle(&cursor).is_none());

        let cursor = SealedPostingCursor::new(&list, 3)?;
        assert_eq!(PostingCursor::doc(&cursor), Some(3));
        assert!(PostingCursor::positions_handle(&cursor).is_none());
        Ok(())
    }

    #[test]
    fn phrase_scorer_reads_sealed_postings_and_positions() -> Result<(), Box<dyn std::error::Error>>
    {
        let first_postings = [Posting::new(1, 2), Posting::new(2, 1), Posting::new(3, 1)];
        let encoded_first_postings = EncodedPostingList::encode(&first_postings)?;
        let first_posting_list = encoded_first_postings.posting_list()?;
        let encoded_first_positions = EncodedPositionList::encode(&first_postings, &[0, 4, 0, 1])?;
        let first_positions = encoded_first_positions.position_list(&first_posting_list)?;

        let second_postings = [Posting::new(1, 2), Posting::new(2, 1), Posting::new(3, 1)];
        let encoded_second_postings = EncodedPostingList::encode(&second_postings)?;
        let second_posting_list = encoded_second_postings.posting_list()?;
        let encoded_second_positions =
            EncodedPositionList::encode(&second_postings, &[1, 5, 2, 2])?;
        let second_positions = encoded_second_positions.position_list(&second_posting_list)?;

        let lengths = [Some(8); 4];
        let encoded_doclens =
            EncodedDocLenSection::encode(0, 4, &[6], &[DocLenFieldInput::new(6, &lengths)])?;
        let doclens = encoded_doclens.section(&[6])?;
        let field = doclens.field(6).expect("field exists");
        let snapshot = snapshot(6, 32, 4)?;
        let mut phrase = PhraseScorer::new(
            vec![
                PhraseTerm::new(
                    6,
                    0,
                    SealedPostingCursor::with_positions(&first_positions, 4)?,
                    3,
                ),
                PhraseTerm::new(
                    6,
                    1,
                    SealedPostingCursor::with_positions(&second_positions, 4)?,
                    3,
                ),
            ],
            field,
            snapshot.clone(),
            1.0,
        )?;

        assert_eq!(phrase.doc(), Some(1));
        let expected = expected_phrase_score(
            &snapshot,
            &[3, 3],
            field.fieldnorm_id(1).expect("fieldnorm"),
            2,
            1.0,
        );
        assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
        assert_eq!(phrase.next()?, Some(3));
        let expected = expected_phrase_score(
            &snapshot,
            &[3, 3],
            field.fieldnorm_id(3).expect("fieldnorm"),
            1,
            1.0,
        );
        assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
        assert_eq!(phrase.next()?, None);
        Ok(())
    }

    #[test]
    fn phrase_scorer_filters_candidates_counts_frequency_and_seeks()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(8); 8];
        let encoded =
            EncodedDocLenSection::encode(0, 8, &[5], &[DocLenFieldInput::new(5, &lengths)])?;
        let section = encoded.section(&[5])?;
        let field = section.field(5).expect("field exists");
        let snapshot = snapshot(5, 64, 8)?;
        let build = || {
            let first = PositionedVecCursor::new(
                vec![(1, vec![0, 4]), (2, vec![0]), (3, vec![2]), (5, vec![1])],
                4,
                8,
            );
            let second = PositionedVecCursor::new(
                vec![(1, vec![1, 5]), (2, vec![2]), (3, vec![3]), (5, vec![0])],
                4,
                8,
            );
            PhraseScorer::new(
                vec![
                    PhraseTerm::new(5, 0, first, 4),
                    PhraseTerm::new(5, 1, second, 4),
                ],
                field,
                snapshot.clone(),
                1.0,
            )
        };

        let mut phrase = build()?;
        assert_eq!(phrase.doc(), Some(1));
        let candidate_estimate = estimate_intersection([4, 4].into_iter(), 8);
        assert_eq!(phrase.cost, u64::from(candidate_estimate) * 20);
        assert_eq!(phrase.size_hint, candidate_estimate / 20);
        let expected = expected_phrase_score(
            &snapshot,
            &[4, 4],
            field.fieldnorm_id(1).expect("fieldnorm"),
            2,
            1.0,
        );
        assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
        let decode_capacity = phrase.decode_scratch.capacity();
        let position_capacities = phrase
            .slots
            .iter()
            .map(|slot| slot.positions.capacity())
            .collect::<Vec<_>>();
        assert_eq!(phrase.next()?, Some(3));
        assert_eq!(phrase.decode_scratch.capacity(), decode_capacity);
        assert_eq!(
            phrase
                .slots
                .iter()
                .map(|slot| slot.positions.capacity())
                .collect::<Vec<_>>(),
            position_capacities
        );
        let expected = expected_phrase_score(
            &snapshot,
            &[4, 4],
            field.fieldnorm_id(3).expect("fieldnorm"),
            1,
            1.0,
        );
        assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
        assert_eq!(phrase.next()?, None);
        assert_eq!(phrase.next()?, None);

        let mut phrase = build()?;
        assert_eq!(phrase.seek(2)?, Some(3));
        assert_eq!(phrase.seek(3)?, Some(3));
        assert_eq!(phrase.seek(4)?, None);

        let mut scorer = ReferenceScorer::phrase(build()?);
        let winners = scorer.top_k(10, &AllLiveDocs)?;
        assert_eq!(
            winners
                .iter()
                .map(|winner| winner.global_docid)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        Ok(())
    }

    #[test]
    fn phrase_cost_estimate_follows_tantivy_child_cost_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(2); 12];
        let encoded =
            EncodedDocLenSection::encode(0, 12, &[4], &[DocLenFieldInput::new(4, &lengths)])?;
        let section = encoded.section(&[4])?;
        let field = section.field(4).expect("field exists");
        let snapshot = snapshot(4, 24, 12)?;
        let broad = PositionedVecCursor::new(
            vec![
                (0, vec![0]),
                (1, vec![0]),
                (2, vec![0]),
                (3, vec![0]),
                (4, vec![0]),
            ],
            5,
            12,
        );
        let narrow = PositionedVecCursor::new(vec![(0, vec![1])], 1, 12);
        let phrase = PhraseScorer::new(
            vec![
                PhraseTerm::new(4, 0, broad, 5),
                PhraseTerm::new(4, 1, narrow, 1),
            ],
            field,
            snapshot,
            1.0,
        )?;

        assert_eq!(estimate_intersection([5, 1].into_iter(), 12), 0);
        assert_eq!(estimate_intersection([1, 5].into_iter(), 12), 1);
        let scorer = ReferenceScorer::phrase(phrase);
        assert_eq!(scorer.doc(), Some(0));
        assert_eq!(scorer.cost(), 20);
        assert_eq!(scorer.size_hint(), 0);
        Ok(())
    }

    #[test]
    fn phrase_scorer_handles_repeated_terms_same_position_or_and_query_gaps()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(6); 16];
        let encoded =
            EncodedDocLenSection::encode(0, 16, &[2], &[DocLenFieldInput::new(2, &lengths)])?;
        let section = encoded.section(&[2])?;
        let field = section.field(2).expect("field exists");
        let snapshot = snapshot(2, 96, 16)?;

        let repeated_a = || PositionedVecCursor::new(vec![(7, vec![0, 1, 2])], 1, 16);
        let repeated_b = PositionedVecCursor::new(vec![(7, vec![2, 3])], 1, 16);
        let repeated = PhraseScorer::new(
            vec![
                PhraseTerm::new(2, 0, repeated_a(), 1),
                PhraseTerm::new(2, 1, repeated_a(), 1),
                PhraseTerm::new(2, 2, repeated_b, 1),
            ],
            field,
            snapshot.clone(),
            1.0,
        )?;
        assert_eq!(repeated.doc(), Some(7));
        let expected = expected_phrase_score(
            &snapshot,
            &[1, 1, 1],
            field.fieldnorm_id(7).expect("fieldnorm"),
            2,
            1.0,
        );
        assert_eq!(repeated.score()?.to_bits(), expected.to_bits());

        let alternative_x = PositionedVecCursor::new(vec![(8, vec![5]), (10, vec![1])], 2, 16);
        let alternative_y = PositionedVecCursor::new(vec![(9, vec![10]), (10, vec![1])], 2, 16);
        let following =
            PositionedVecCursor::new(vec![(8, vec![6]), (9, vec![11]), (10, vec![2])], 3, 16);
        let mut alternatives = PhraseScorer::new(
            vec![
                PhraseTerm::new(2, 0, alternative_x, 2),
                PhraseTerm::new(2, 0, alternative_y, 2),
                PhraseTerm::new(2, 1, following, 3),
            ],
            field,
            snapshot.clone(),
            1.0,
        )?;
        for expected_doc in [8, 9, 10] {
            assert_eq!(alternatives.doc(), Some(expected_doc));
            let expected = expected_phrase_score(
                &snapshot,
                &[2, 2, 3],
                field
                    .fieldnorm_id(u64::from(expected_doc))
                    .expect("fieldnorm"),
                1,
                1.0,
            );
            assert_eq!(alternatives.score()?.to_bits(), expected.to_bits());
            alternatives.next()?;
        }
        assert_eq!(alternatives.doc(), None);

        let gap_start = PositionedVecCursor::new(vec![(6, vec![10]), (11, vec![20])], 2, 16);
        let gap_end = PositionedVecCursor::new(vec![(6, vec![12]), (11, vec![21])], 2, 16);
        let mut gapped = PhraseScorer::new(
            vec![
                PhraseTerm::new(2, 4, gap_start, 2),
                PhraseTerm::new(2, 6, gap_end, 2),
            ],
            field,
            snapshot,
            1.0,
        )?;
        assert_eq!(gapped.doc(), Some(6));
        assert_eq!(gapped.next()?, None);
        Ok(())
    }

    #[test]
    fn phrase_scorer_matches_tantivy_corpus_docs_ranks_and_score_bits()
    -> Result<(), Box<dyn std::error::Error>> {
        use frankensearch_lexical::tantivy_crate::{
            Index, TantivyDocument, Term,
            collector::{DocSetCollector, TopDocs},
            doc,
            query::PhraseQuery,
            schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions},
        };

        let mut schema_builder = Schema::builder();
        let indexing = TextFieldIndexing::default()
            .set_tokenizer("default")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let body = schema_builder.add_text_field(
            "body",
            TextOptions::default().set_indexing_options(indexing),
        );
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index.writer_with_num_threads::<TantivyDocument>(1, 15_000_000)?;
        let corpus = [
            "a b",
            "a b a b",
            "a x b",
            "a only",
            "b a",
            "a b filler filler filler filler filler filler",
        ];
        for text in corpus {
            writer.add_document(doc!(body => text))?;
        }
        writer.commit()?;
        drop(writer);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let query = PhraseQuery::new(vec![
            Term::from_field_text(body, "a"),
            Term::from_field_text(body, "b"),
        ]);
        let mut oracle_docset = searcher
            .search(&query, &DocSetCollector)?
            .into_iter()
            .map(|address| {
                assert_eq!(address.segment_ord, 0);
                address.doc_id
            })
            .collect::<Vec<_>>();
        oracle_docset.sort_unstable();
        assert_eq!(oracle_docset, vec![0, 1, 5]);
        let oracle_ranked = searcher
            .search(&query, &TopDocs::with_limit(corpus.len()).order_by_score())?
            .into_iter()
            .map(|(score, address)| {
                assert_eq!(address.segment_ord, 0);
                (address.doc_id, score.to_bits())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            oracle_ranked
                .iter()
                .map(|(doc, _)| *doc)
                .collect::<Vec<_>>(),
            vec![1, 0, 5]
        );

        let lengths = [Some(2), Some(4), Some(3), Some(2), Some(2), Some(8)];
        let encoded =
            EncodedDocLenSection::encode(0, 6, &[0], &[DocLenFieldInput::new(0, &lengths)])?;
        let section = encoded.section(&[0])?;
        let field = section.field(0).expect("field exists");
        let snapshot = snapshot(0, 21, 6)?;
        let a = PositionedVecCursor::new(
            vec![
                (0, vec![0]),
                (1, vec![0, 2]),
                (2, vec![0]),
                (3, vec![0]),
                (4, vec![1]),
                (5, vec![0]),
            ],
            6,
            6,
        );
        let b = PositionedVecCursor::new(
            vec![
                (0, vec![1]),
                (1, vec![1, 3]),
                (2, vec![2]),
                (4, vec![0]),
                (5, vec![1]),
            ],
            5,
            6,
        );
        let phrase = PhraseScorer::new(
            vec![PhraseTerm::new(0, 0, a, 6), PhraseTerm::new(0, 1, b, 5)],
            field,
            snapshot,
            1.0,
        )?;
        let mut scorer = ReferenceScorer::phrase(phrase);
        let quill_ranked = scorer
            .top_k(corpus.len(), &AllLiveDocs)?
            .into_iter()
            .map(|hit| (hit.global_docid, hit.score.to_bits()))
            .collect::<Vec<_>>();
        assert_eq!(quill_ranked, oracle_ranked);
        Ok(())
    }

    #[test]
    fn phrase_same_position_alternatives_are_reviewed_or_divergence()
    -> Result<(), Box<dyn std::error::Error>> {
        use frankensearch_lexical::tantivy_crate::{
            Index, TantivyDocument, Term,
            collector::DocSetCollector,
            doc,
            query::PhraseQuery,
            schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions},
            tokenizer::{PreTokenizedString, Token},
        };

        let mut schema_builder = Schema::builder();
        let indexing = TextFieldIndexing::default()
            .set_tokenizer("default")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let body = schema_builder.add_text_field(
            "body",
            TextOptions::default().set_indexing_options(indexing),
        );
        let index = Index::create_in_ram(schema_builder.build());
        let token = |text: &str, position: usize, offset_from: usize, offset_to: usize| Token {
            offset_from,
            offset_to,
            position,
            text: text.to_owned(),
            position_length: 1,
        };
        let docs = [
            PreTokenizedString {
                text: "a c".to_owned(),
                tokens: vec![token("a", 0, 0, 1), token("c", 1, 2, 3)],
            },
            PreTokenizedString {
                text: "b c".to_owned(),
                tokens: vec![token("b", 0, 0, 1), token("c", 1, 2, 3)],
            },
            PreTokenizedString {
                text: "a/b c".to_owned(),
                tokens: vec![
                    token("a", 0, 0, 3),
                    token("b", 0, 0, 3),
                    token("c", 1, 4, 5),
                ],
            },
        ];
        let mut writer = index.writer_with_num_threads::<TantivyDocument>(1, 15_000_000)?;
        for text in docs {
            writer.add_document(doc!(body => text))?;
        }
        writer.commit()?;
        drop(writer);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let oracle_query = PhraseQuery::new_with_offset(vec![
            (0, Term::from_field_text(body, "a")),
            (0, Term::from_field_text(body, "b")),
            (1, Term::from_field_text(body, "c")),
        ]);
        let mut oracle_docids = searcher
            .search(&oracle_query, &DocSetCollector)?
            .into_iter()
            .map(|address| {
                assert_eq!(address.segment_ord, 0);
                address.doc_id
            })
            .collect::<Vec<_>>();
        oracle_docids.sort_unstable();
        assert_eq!(oracle_docids, vec![2]);

        let lengths = [Some(2), Some(2), Some(3)];
        let encoded =
            EncodedDocLenSection::encode(0, 3, &[0], &[DocLenFieldInput::new(0, &lengths)])?;
        let section = encoded.section(&[0])?;
        let field = section.field(0).expect("field exists");
        let phrase = PhraseScorer::new(
            vec![
                PhraseTerm::new(
                    0,
                    0,
                    PositionedVecCursor::new(vec![(0, vec![0]), (2, vec![0])], 2, 3),
                    2,
                ),
                PhraseTerm::new(
                    0,
                    0,
                    PositionedVecCursor::new(vec![(1, vec![0]), (2, vec![0])], 2, 3),
                    2,
                ),
                PhraseTerm::new(
                    0,
                    1,
                    PositionedVecCursor::new(vec![(0, vec![1]), (1, vec![1]), (2, vec![1])], 3, 3),
                    3,
                ),
            ],
            field,
            snapshot(0, 7, 3)?,
            1.0,
        )?;
        let mut scorer = ReferenceScorer::phrase(phrase);
        let mut quill_docids = scorer
            .top_k(3, &AllLiveDocs)?
            .into_iter()
            .map(|hit| hit.global_docid)
            .collect::<Vec<_>>();
        quill_docids.sort_unstable();
        assert_eq!(quill_docids, vec![0, 1, 2]);
        assert_ne!(quill_docids, oracle_docids);
        Ok(())
    }

    #[test]
    fn phrase_frequency_exceeds_u8_and_matches_u32_position_boundary()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(400); 16];
        let encoded =
            EncodedDocLenSection::encode(0, 16, &[9], &[DocLenFieldInput::new(9, &lengths)])?;
        let section = encoded.section(&[9])?;
        let field = section.field(9).expect("field exists");
        let snapshot = snapshot(9, 6_400, 16)?;
        let first = PositionedVecCursor::new(
            vec![(11, (0..=300).collect()), (12, vec![u32::MAX - 1])],
            2,
            16,
        );
        let second =
            PositionedVecCursor::new(vec![(11, (1..=301).collect()), (12, vec![u32::MAX])], 2, 16);
        let mut phrase = PhraseScorer::new(
            vec![
                PhraseTerm::new(9, 0, first, 2),
                PhraseTerm::new(9, 1, second, 2),
            ],
            field,
            snapshot.clone(),
            1.0,
        )?;

        assert_eq!(phrase.doc(), Some(11));
        let expected = expected_phrase_score(
            &snapshot,
            &[2, 2],
            field.fieldnorm_id(11).expect("fieldnorm"),
            301,
            1.0,
        );
        assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
        assert_eq!(phrase.next()?, Some(12));
        let expected = expected_phrase_score(
            &snapshot,
            &[2, 2],
            field.fieldnorm_id(12).expect("fieldnorm"),
            1,
            1.0,
        );
        assert_eq!(phrase.score()?.to_bits(), expected.to_bits());
        assert_eq!(phrase.next()?, None);
        Ok(())
    }

    #[test]
    fn phrase_constructor_rejects_missing_positions_and_invalid_slot_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(2); 4];
        let encoded =
            EncodedDocLenSection::encode(0, 4, &[3], &[DocLenFieldInput::new(3, &lengths)])?;
        let section = encoded.section(&[3])?;
        let field = section.field(3).expect("field exists");
        let snapshot = snapshot(3, 8, 4)?;

        let wrong_field = PhraseScorer::new(
            vec![
                PhraseTerm::new(2, 0, PositionedVecCursor::new(vec![(0, vec![0])], 1, 4), 1),
                PhraseTerm::new(3, 1, PositionedVecCursor::new(vec![(0, vec![1])], 1, 4), 1),
            ],
            field,
            snapshot.clone(),
            1.0,
        );
        assert!(matches!(
            wrong_field,
            Err(ArgusError::PhraseTermFieldMismatch {
                term_field: 2,
                stats_field: 3
            })
        ));

        let missing_positions = PhraseScorer::new(
            vec![
                PhraseTerm::new(3, 0, VecCursor::new(postings(&[0]), 1, 4), 1),
                PhraseTerm::new(3, 1, PositionedVecCursor::new(vec![(0, vec![1])], 1, 4), 1),
            ],
            field,
            snapshot.clone(),
            1.0,
        );
        assert!(matches!(
            missing_positions,
            Err(ArgusError::MissingPositions {
                field_ord: 3,
                global_docid: 0
            })
        ));

        let descending = PhraseScorer::new(
            vec![
                PhraseTerm::new(3, 2, PositionedVecCursor::new(vec![(0, vec![2])], 1, 4), 1),
                PhraseTerm::new(3, 1, PositionedVecCursor::new(vec![(0, vec![1])], 1, 4), 1),
            ],
            field,
            snapshot.clone(),
            1.0,
        );
        assert!(matches!(descending, Err(ArgusError::InvalidPhrase { .. })));

        let one_slot = PhraseScorer::new(
            vec![
                PhraseTerm::new(3, 0, PositionedVecCursor::new(vec![(0, vec![0])], 1, 4), 1),
                PhraseTerm::new(3, 0, PositionedVecCursor::new(vec![(0, vec![0])], 1, 4), 1),
            ],
            field,
            snapshot,
            1.0,
        );
        assert!(matches!(one_slot, Err(ArgusError::InvalidPhrase { .. })));
        Ok(())
    }

    #[test]
    fn bm25_score_uses_raw_average_cached_norm_and_boost_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(3)];
        let encoded =
            EncodedDocLenSection::encode(0, 1, &[7], &[DocLenFieldInput::new(7, &lengths)])?;
        let section = encoded.section(&[7])?;
        let field = section.field(7).expect("field exists");
        let snapshot = snapshot(7, 9, 3)?;
        let mut scorer = term(vec![Posting::new(0, 2)], field, &snapshot, 2, 1, 1.0)?;

        // Spreadsheet row: N=3, n=2, total_tokens=9, dl=3, f=2.
        // idf=0.47000363, norm=1.2, tf=0.625, score=0.646255.
        let score = scorer.score()?;
        assert_eq!(score.to_bits(), 1_059_418_360);
        let expected = expected_term_score(&snapshot, 2, fieldnorm_to_id(3), 2, 1.0);
        assert_eq!(score.to_bits(), expected.to_bits());

        let mut boosted = term(vec![Posting::new(0, 2)], field, &snapshot, 2, 1, 2.0)?;
        let boosted_score = boosted.score()?;
        let expected_boosted = expected_term_score(&snapshot, 2, fieldnorm_to_id(3), 2, 2.0);
        assert_eq!(boosted_score.to_bits(), expected_boosted.to_bits());
        Ok(())
    }

    #[test]
    fn fieldnorm_boundaries_use_the_shared_snapshot_cache() -> Result<(), Box<dyn std::error::Error>>
    {
        let lengths = [Some(0), Some(40), Some(42), Some(2_013_265_944)];
        let encoded =
            EncodedDocLenSection::encode(0, 4, &[3], &[DocLenFieldInput::new(3, &lengths)])?;
        let section = encoded.section(&[3])?;
        let field = section.field(3).expect("field exists");
        let snapshot = snapshot(3, 2_013_266_026, 4)?;
        let mut scorer = term(
            vec![
                Posting::new(0, 1),
                Posting::new(1, 2),
                Posting::new(2, 3),
                Posting::new(3, 4),
            ],
            field,
            &snapshot,
            4,
            4,
            1.0,
        )?;

        for (doc, fieldnorm_id, frequency) in [(0, 0, 1), (1, 40, 2), (2, 41, 3), (3, 255, 4)] {
            assert_eq!(scorer.doc(), Some(doc));
            let score = scorer.score()?;
            let expected = expected_term_score(&snapshot, 4, fieldnorm_id, frequency, 1.0);
            assert_eq!(score.to_bits(), expected.to_bits());
            scorer.next()?;
        }
        assert_eq!(scorer.doc(), None);
        Ok(())
    }

    #[test]
    fn title_boost_wins_and_equal_scores_tie_by_global_docid()
    -> Result<(), Box<dyn std::error::Error>> {
        let content_lengths = [Some(1), Some(1), Some(1)];
        let title_lengths = [Some(1), Some(1), Some(1)];
        let inputs = [
            DocLenFieldInput::new(1, &content_lengths),
            DocLenFieldInput::new(2, &title_lengths),
        ];
        let encoded = EncodedDocLenSection::encode(0, 3, &[1, 2], &inputs)?;
        let section = encoded.section(&[1, 2])?;
        let content = section.field(1).expect("content field");
        let title = section.field(2).expect("title field");
        let content_snapshot = snapshot(1, 3, 3)?;
        let title_snapshot = snapshot(2, 3, 3)?;
        let content_term = term(postings(&[0, 2]), content, &content_snapshot, 2, 2, 1.0)?;
        let title_term = term(postings(&[1]), title, &title_snapshot, 1, 1, 2.0)?;
        let mut query = ReferenceScorer::boolean(vec![
            ScorerClause::should(content_term),
            ScorerClause::should(title_term),
        ])?;
        let winners = query.top_k(3, &AllLiveDocs)?;

        assert_eq!(
            winners
                .iter()
                .map(|hit| hit.global_docid)
                .collect::<Vec<_>>(),
            vec![1, 0, 2]
        );
        assert_eq!(winners[1].score.to_bits(), winners[2].score.to_bits());
        assert!(winners[0].score > winners[1].score);
        Ok(())
    }

    #[test]
    fn boolean_occurs_match_tantivy_boundaries() -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1); 5];
        let encoded =
            EncodedDocLenSection::encode(0, 5, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 5, 5)?;

        let mut must_with_optional = ReferenceScorer::boolean(vec![
            ScorerClause::must(term(postings(&[1, 2]), field, &snapshot, 2, 2, 1.0)?),
            ScorerClause::should(term(postings(&[2, 3]), field, &snapshot, 2, 2, 1.0)?),
            ScorerClause::must_not(term(postings(&[2]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        let hits = must_with_optional.top_k(5, &AllLiveDocs)?;
        assert_eq!(
            hits.iter().map(|hit| hit.global_docid).collect::<Vec<_>>(),
            vec![1]
        );
        let expected_must = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, 1.0);
        assert_eq!(hits[0].score.to_bits(), expected_must.to_bits());

        let mut should_with_exclusion = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[2, 3]), field, &snapshot, 2, 2, 1.0)?),
            ScorerClause::must_not(term(postings(&[2]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        let hits = should_with_exclusion.top_k(5, &AllLiveDocs)?;
        assert_eq!(
            hits.iter().map(|hit| hit.global_docid).collect::<Vec<_>>(),
            vec![3]
        );

        let mut negative_only = ReferenceScorer::boolean(vec![ScorerClause::must_not(term(
            postings(&[1, 2]),
            field,
            &snapshot,
            2,
            2,
            1.0,
        )?)])?;
        assert!(negative_only.top_k(5, &AllLiveDocs)?.is_empty());

        let mut multiple_negative_only = ReferenceScorer::boolean(vec![
            ScorerClause::must_not(term(postings(&[1]), field, &snapshot, 1, 1, 1.0)?),
            ScorerClause::must_not(term(postings(&[2]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        assert_eq!(multiple_negative_only.segment_num_docs(), 5);
        assert_eq!(multiple_negative_only.size_hint(), 0);
        assert_eq!(multiple_negative_only.cost(), 0);
        assert!(multiple_negative_only.top_k(5, &AllLiveDocs)?.is_empty());
        Ok(())
    }

    #[test]
    fn required_cost_sort_and_optional_composition_pin_f32_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1)];
        let encoded =
            EncodedDocLenSection::encode(0, 1, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 1, 1)?;
        let boosts = [1.0_f32, 1.0e8, -1.0e8, 2.0];
        let costs = [900_u64, 3, 200, 200];
        let mut clauses = Vec::new();
        for (&boost, &cost) in boosts.iter().zip(&costs) {
            clauses.push(ScorerClause::must(term(
                postings(&[0]),
                field,
                &snapshot,
                1,
                cost,
                boost,
            )?));
        }
        clauses.push(ScorerClause::should(term(
            postings(&[0]),
            field,
            &snapshot,
            1,
            1,
            3.0,
        )?));
        let mut query = ReferenceScorer::boolean(clauses)?;
        let actual = query.score()?;

        let score = |boost| expected_term_score(&snapshot, 1, fieldnorm_to_id(1), 1, boost);
        // Required order after stable cost sort: boosts 1e8, -1e8, 2, 1.
        // Tantivy's intersection expression is `(left + right) + sum(others)`;
        // RequiredOptional then adds the optional aggregate.
        let required = score(1.0e8) + score(-1.0e8) + (score(2.0) + score(1.0));
        let mut expected = 0.0_f32;
        expected += required;
        expected += score(3.0);
        assert_eq!(actual.to_bits(), expected.to_bits());
        Ok(())
    }

    #[test]
    fn intersection_danger_seek_preserves_nested_union_score_order()
    -> Result<(), Box<dyn std::error::Error>> {
        const DOC_COUNT: usize = 30_001;
        let lengths = vec![Some(1); DOC_COUNT];
        let encoded = EncodedDocLenSection::encode(
            0,
            DOC_COUNT as u64,
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, DOC_COUNT as u64, DOC_COUNT as u64)?;

        let optional = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[0, 16_000]), field, &snapshot, 2, 2, 0.0)?),
            ScorerClause::should(term(
                postings(&[0, 20_000, 30_000]),
                field,
                &snapshot,
                3,
                2,
                -1.0e8,
            )?),
            ScorerClause::should(term(postings(&[0, 11_000]), field, &snapshot, 2, 2, 0.0)?),
            ScorerClause::should(term(
                postings(&[0, 20_000, 30_000]),
                field,
                &snapshot,
                3,
                2,
                1.0e8,
            )?),
            ScorerClause::should(term(
                postings(&[0, 20_000, 30_000]),
                field,
                &snapshot,
                3,
                2,
                1.0,
            )?),
        ])?;
        let required_optional = ReferenceScorer::boolean(vec![
            ScorerClause::must(term(
                postings(&[0, 11_000, 20_000]),
                field,
                &snapshot,
                3,
                2,
                0.0,
            )?),
            ScorerClause::should(optional),
        ])?;
        let nested_union = ReferenceScorer::boolean(vec![
            ScorerClause::should(required_optional),
            ScorerClause::should(term(postings(&[0, 20_000]), field, &snapshot, 2, 2, 0.0)?),
        ])?;
        let mut query = ReferenceScorer::boolean(vec![
            ScorerClause::must(term(
                postings(&[0, 10_000, 20_000]),
                field,
                &snapshot,
                3,
                1,
                0.0,
            )?),
            ScorerClause::must(nested_union),
        ])?;

        assert_eq!(query.doc(), Some(0));
        assert_eq!(query.next()?, Some(20_000));
        let small = expected_term_score(&snapshot, 3, fieldnorm_to_id(1), 1, 1.0);
        assert_ne!(small.to_bits(), 0.0_f32.to_bits());
        assert_eq!(query.score()?.to_bits(), 0.0_f32.to_bits());
        Ok(())
    }

    #[test]
    fn buffered_union_reproduces_oracle_swap_remove_order_across_horizon()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = vec![Some(1); 5_001];
        let encoded =
            EncodedDocLenSection::encode(0, 5_001, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 5_001, 5_001)?;
        let clauses = vec![
            ScorerClause::should(term(postings(&[0]), field, &snapshot, 1, 1, 1.0)?),
            ScorerClause::should(term(postings(&[5_000]), field, &snapshot, 1, 1, 1.0e8)?),
            ScorerClause::should(term(postings(&[5_000]), field, &snapshot, 1, 1, -1.0e8)?),
            ScorerClause::should(term(postings(&[5_000]), field, &snapshot, 1, 1, 1.0)?),
        ];
        let mut query = ReferenceScorer::boolean(clauses)?;
        assert_eq!(query.doc(), Some(0));
        query.next()?;
        assert_eq!(query.doc(), Some(5_000));

        let small = expected_term_score(&snapshot, 1, fieldnorm_to_id(1), 1, 1.0);
        let positive = expected_term_score(&snapshot, 1, fieldnorm_to_id(1), 1, 1.0e8);
        let negative = expected_term_score(&snapshot, 1, fieldnorm_to_id(1), 1, -1.0e8);
        // The doc-0 scorer exhausts in the first window and `swap_remove`
        // moves the final small scorer ahead of the two large scorers.
        let expected = ((0.0_f32 + small) + positive) + negative;
        let parse_order = ((0.0_f32 + positive) + negative) + small;
        assert_ne!(expected.to_bits(), parse_order.to_bits());
        assert_eq!(query.score()?.to_bits(), expected.to_bits());
        Ok(())
    }

    #[test]
    fn composite_size_hints_and_costs_match_tantivy_runtime_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = vec![Some(1); 10_000];
        let encoded =
            EncodedDocLenSection::encode(0, 10_000, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 10_000, 10_000)?;
        let first_docs = (0_u32..500).collect::<Vec<_>>();
        let second_docs = (0_u32..1_000).collect::<Vec<_>>();
        let inner = ReferenceScorer::boolean(vec![
            ScorerClause::must(term(
                postings(&first_docs),
                field,
                &snapshot,
                500,
                500,
                1.0,
            )?),
            ScorerClause::must(term(
                postings(&second_docs),
                field,
                &snapshot,
                1_000,
                1_000,
                1.0,
            )?),
        ])?;
        assert_eq!(inner.size_hint(), 60);
        assert_eq!(inner.cost(), 500);

        let nested_exclude = ReferenceScorer::boolean(vec![
            ScorerClause::must(inner),
            ScorerClause::must_not(term(postings(&[9_999]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        assert_eq!(nested_exclude.doc(), Some(0));
        assert_eq!(nested_exclude.size_hint(), 60);
        assert_eq!(nested_exclude.cost(), 60);

        let third_docs = (0_u32..100).collect::<Vec<_>>();
        let outer = ReferenceScorer::boolean(vec![
            ScorerClause::must(nested_exclude),
            ScorerClause::must(term(
                postings(&third_docs),
                field,
                &snapshot,
                100,
                100,
                1.0,
            )?),
        ])?;
        assert_eq!(outer.cost(), 60);

        let mut union = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[0]), field, &snapshot, 1, 1, 1.0)?),
            ScorerClause::should(term(postings(&[0, 4_096]), field, &snapshot, 2, 2, 1.0)?),
        ])?;
        assert_eq!(union.doc(), Some(0));
        assert_eq!(union.size_hint(), 2);
        assert_eq!(union.cost(), 2);
        assert_eq!(union.next()?, Some(4_096));
        assert_eq!(union.size_hint(), 0);
        assert_eq!(union.cost(), 0);

        // BooleanWeight passes SegmentReader::num_docs(), not physical BM25 N.
        assert_eq!(estimate_intersection([25, 25].into_iter(), 50), 15);
        assert_eq!(estimate_union([25, 25].into_iter(), 50), 32);
        assert_ne!(estimate_intersection([25, 25].into_iter(), 100), 15);
        assert_ne!(estimate_union([25, 25].into_iter(), 100), 32);
        Ok(())
    }

    #[test]
    fn max_global_docid_survives_union_and_intersection_exhaustion()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1)];
        let encoded = EncodedDocLenSection::encode(
            u64::from(u32::MAX),
            1_u64 << 32,
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 1, 1)?;

        let mut union = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[u32::MAX]), field, &snapshot, 1, 1, 1.0)?),
            ScorerClause::should(term(postings(&[u32::MAX]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        assert_eq!(union.doc(), Some(u32::MAX));
        assert!(union.score()?.is_finite());
        assert_eq!(union.next()?, None);
        assert_eq!(union.next()?, None);

        let mut intersection = ReferenceScorer::boolean(vec![
            ScorerClause::must(term(postings(&[u32::MAX]), field, &snapshot, 1, 1, 1.0)?),
            ScorerClause::must(term(postings(&[u32::MAX]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        assert_eq!(intersection.doc(), Some(u32::MAX));
        assert!(intersection.score()?.is_finite());
        assert_eq!(intersection.next()?, None);
        assert_eq!(intersection.next()?, None);
        Ok(())
    }

    #[test]
    fn randomized_boolean_matches_naive_set_oracle() -> Result<(), Box<dyn std::error::Error>> {
        const DOCS: usize = 48;
        let lengths = [Some(1); DOCS];
        let encoded = EncodedDocLenSection::encode(
            0,
            DOCS as u64,
            &[1],
            &[DocLenFieldInput::new(1, &lengths)],
        )?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, DOCS as u64, DOCS as u64)?;

        for seed in 1_u64..=24 {
            let mut state = seed;
            let mut membership = [[false; DOCS]; 5];
            for doc in 0..DOCS {
                for term_membership in &mut membership {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    term_membership[doc] = state >> 61 != 0;
                }
            }
            let term_docs = |term_index: usize| {
                membership[term_index]
                    .iter()
                    .enumerate()
                    .filter_map(|(doc, present)| {
                        present.then_some(u32::try_from(doc).expect("DOCS fits in u32"))
                    })
                    .collect::<Vec<_>>()
            };
            let docs_by_term = std::array::from_fn::<_, 5, _>(term_docs);
            let leaf = |term_index: usize| {
                term(
                    postings(&docs_by_term[term_index]),
                    field,
                    &snapshot,
                    docs_by_term[term_index].len() as u64,
                    docs_by_term[term_index].len() as u64,
                    1.0,
                )
            };
            let scored_query = || {
                let either = ReferenceScorer::boolean(vec![
                    ScorerClause::should(leaf(0)?),
                    ScorerClause::should(leaf(1)?),
                ])?;
                ReferenceScorer::boolean(vec![
                    ScorerClause::must(either),
                    ScorerClause::must(leaf(2)?),
                    ScorerClause::must_not(leaf(3)?),
                    ScorerClause::should(leaf(4)?),
                ])
            };
            let unscored_query = || {
                let either = ReferenceScorer::boolean_unscored(vec![
                    ScorerClause::should(leaf(0)?),
                    ScorerClause::should(leaf(1)?),
                ])?;
                ReferenceScorer::boolean_unscored(vec![
                    ScorerClause::must(either),
                    ScorerClause::must(leaf(2)?),
                    ScorerClause::must_not(leaf(3)?),
                    ScorerClause::should(leaf(4)?),
                ])
            };
            let mut query = scored_query()?;
            let ranked = query.top_k(DOCS, &AllLiveDocs)?;
            let mut actual = ranked
                .iter()
                .map(|hit| hit.global_docid)
                .collect::<Vec<_>>();
            actual.sort_unstable();
            let expected = (0..DOCS)
                .filter(|&doc| {
                    (membership[0][doc] || membership[1][doc])
                        && membership[2][doc]
                        && !membership[3][doc]
                })
                .map(|doc| u32::try_from(doc).expect("DOCS fits in u32"))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "seed {seed}");

            let offset = usize::try_from(seed % 11).expect("bounded seed offset");
            let limit = 7;
            let mut counted_query = scored_query()?;
            let mut counted_collector = TopDocsCollector::with_exact_count(limit, offset)?;
            counted_collector.collect(&mut counted_query, &AllLiveDocs)?;
            let counted = counted_collector.finish()?;
            let page_start = offset.min(ranked.len());
            let page_end = page_start.saturating_add(limit).min(ranked.len());
            assert_eq!(counted.hits, ranked[page_start..page_end], "seed {seed}");
            assert_eq!(
                counted.total_count,
                Some(expected.len() as u64),
                "seed {seed}"
            );

            let mut unscored = unscored_query()?;
            assert_eq!(
                unscored.collect_doc_set(&AllLiveDocs)?,
                expected,
                "seed {seed}"
            );
        }
        Ok(())
    }

    #[test]
    fn global_collectors_page_count_and_materialize_only_winners()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = vec![Some(1); 10];
        let encoded =
            EncodedDocLenSection::encode(0, 10, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 10, 10)?;
        let left_docs = [0, 2, 4, 6, 8];
        let right_docs = [1, 3, 5, 7, 9];

        let collect = |limit: usize,
                       offset: usize,
                       exact_count: bool,
                       live_docs: &dyn LiveDocs|
         -> Result<CollectedTopDocs, ArgusError> {
            let mut collector = if exact_count {
                TopDocsCollector::with_exact_count(limit, offset)?
            } else {
                TopDocsCollector::new(limit, offset)?
            };
            for docs in [&left_docs[..], &right_docs[..]] {
                let mut scorer = term(postings(docs), field, &snapshot, 10, 5, 1.0)?;
                collector.collect(&mut scorer, live_docs)?;
            }
            collector.finish()
        };

        let full = collect(10, 0, false, &AllLiveDocs)?;
        assert_eq!(
            full.hits
                .iter()
                .map(|hit| hit.global_docid)
                .collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        );
        assert_eq!(full.total_count, None);

        let page = collect(3, 4, false, &AllLiveDocs)?;
        assert_eq!(page.hits, full.hits[4..7]);
        let past_end = collect(3, 20, false, &AllLiveDocs)?;
        assert!(past_end.hits.is_empty());

        let live = |doc| doc % 3 != 0;
        let live_full = collect(10, 0, false, &live)?;
        let counted = collect(2, 1, true, &live)?;
        assert_eq!(counted.total_count, Some(6));
        assert_eq!(counted.hits, live_full.hits[1..3]);
        let count_only = collect(0, usize::MAX, true, &live)?;
        assert!(count_only.hits.is_empty());
        assert_eq!(count_only.total_count, Some(6));

        let mut unconsumed = term(postings(&left_docs), field, &snapshot, 10, 5, 1.0)?;
        let mut empty_page = TopDocsCollector::new(0, usize::MAX)?;
        empty_page.collect(&mut unconsumed, &AllLiveDocs)?;
        assert_eq!(unconsumed.doc(), Some(0));
        assert!(empty_page.finish()?.hits.is_empty());
        assert!(matches!(
            TopDocsCollector::new(1, usize::MAX),
            Err(ArgusError::CollectorWindowOverflow {
                limit: 1,
                offset: usize::MAX
            })
        ));

        let mut materialized = Vec::new();
        let hits = materialize_doc_ids(&page.hits, |global_docid| {
            materialized.push(global_docid);
            Ok(format!("winner-{global_docid}").into())
        })?;
        assert_eq!(materialized, vec![4, 5, 6]);
        assert_eq!(hits.len(), page.hits.len());
        Ok(())
    }

    #[test]
    fn doc_set_uses_recursive_lazy_unscored_unions() -> Result<(), Box<dyn std::error::Error>> {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        #[derive(Clone)]
        struct CountingFieldNorm<'a> {
            field: DocLenField<'a>,
            reads: Arc<AtomicUsize>,
        }

        impl FieldNormReader for CountingFieldNorm<'_> {
            fn field_ord(&self) -> u16 {
                self.field.field_ord()
            }

            fn fieldnorm_id(&self, global_docid: u32) -> Option<u8> {
                self.reads.fetch_add(1, AtomicOrdering::SeqCst);
                self.field.fieldnorm_id(u64::from(global_docid))
            }
        }

        let lengths = vec![Some(1); 5];
        let encoded =
            EncodedDocLenSection::encode(0, 5, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 5, 5)?;
        let reads = Arc::new(AtomicUsize::new(0));
        let leaf = |docs: &[u32]| {
            Ok::<_, ArgusError>(ReferenceScorer::term(TermScorer::new(
                VecCursor::new(postings(docs), docs.len() as u64, 5),
                CountingFieldNorm {
                    field,
                    reads: Arc::clone(&reads),
                },
                snapshot.clone(),
                docs.len() as u64,
                1.0,
            )?))
        };
        let union = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::should(leaf(&[0, 2, 4])?),
            ScorerClause::should(leaf(&[1, 2, 3])?),
        ])?;
        let excluded = leaf(&[3])?;
        let mut query = ReferenceScorer::boolean_unscored(vec![
            ScorerClause::must(union),
            ScorerClause::must_not(excluded),
        ])?;
        assert_eq!(query.collect_doc_set(&AllLiveDocs)?, vec![0, 1, 2, 4]);
        assert_eq!(
            reads.load(AtomicOrdering::SeqCst),
            7,
            "each physical posting is fieldnorm-validated once; none is scored"
        );

        let mut scored_union = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[0, 2]), field, &snapshot, 2, 2, 1.0)?),
            ScorerClause::should(term(postings(&[1, 2]), field, &snapshot, 2, 2, 1.0)?),
        ])?;
        assert!(matches!(
            scored_union.collect_doc_set(&AllLiveDocs),
            Err(ArgusError::ScoredTreeForUnscoredCollector)
        ));
        Ok(())
    }

    #[test]
    fn tombstones_keep_physical_stats_but_never_enter_heap_or_materializer()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(2), Some(2)];
        let encoded =
            EncodedDocLenSection::encode(0, 2, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let pre_compaction = snapshot(1, 4, 2)?;
        let mut query = term(postings(&[0, 1]), field, &pre_compaction, 2, 2, 1.0)?;
        let live = |doc| doc == 1;
        let winners = query.top_k(10, &live)?;
        assert_eq!(winners.len(), 1);
        assert_eq!(winners[0].global_docid, 1);
        let expected = expected_term_score(&pre_compaction, 2, fieldnorm_to_id(2), 1, 1.0);
        assert_eq!(winners[0].score.to_bits(), expected.to_bits());

        let mut calls = Vec::new();
        let hits = materialize_doc_ids(&winners, |doc| {
            calls.push(doc);
            Ok(format!("external-{doc}").into())
        })?;
        assert_eq!(calls, vec![1]);
        assert_eq!(hits[0].document_id, "external-1");

        let compacted = snapshot(1, 2, 1)?;
        let compacted_score = expected_term_score(&compacted, 1, fieldnorm_to_id(2), 1, 1.0);
        assert_ne!(expected.to_bits(), compacted_score.to_bits());
        Ok(())
    }

    #[test]
    fn deleted_matches_are_scored_before_live_filtering_to_preserve_union_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = vec![Some(1); 20_001];
        let encoded =
            EncodedDocLenSection::encode(0, 20_001, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 20_001, 20_001)?;
        let required = term(postings(&[5_000, 10_000]), field, &snapshot, 2, 2, 1.0)?;
        let optional = ReferenceScorer::boolean(vec![
            ScorerClause::should(term(postings(&[0, 9_500]), field, &snapshot, 2, 2, 0.0)?),
            ScorerClause::should(term(
                postings(&[0, 10_000, 20_000]),
                field,
                &snapshot,
                3,
                2,
                -1.0e8,
            )?),
            ScorerClause::should(term(postings(&[0, 5_000]), field, &snapshot, 2, 2, 0.0)?),
            ScorerClause::should(term(
                postings(&[0, 10_000, 20_000]),
                field,
                &snapshot,
                3,
                2,
                1.0e8,
            )?),
            ScorerClause::should(term(
                postings(&[0, 10_000, 20_000]),
                field,
                &snapshot,
                3,
                2,
                1.0,
            )?),
        ])?;
        let mut query = ReferenceScorer::boolean(vec![
            ScorerClause::must(required),
            ScorerClause::should(optional),
        ])?;
        let winners = query.top_k(1, &|doc| doc == 10_000)?;
        assert_eq!(winners.len(), 1);
        assert_eq!(winners[0].global_docid, 10_000);

        let required_score = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, 1.0);
        let optional_score = expected_term_score(&snapshot, 3, fieldnorm_to_id(1), 1, 1.0);
        let expected = required_score + optional_score;
        assert_eq!(winners[0].score.to_bits(), expected.to_bits());
        assert_ne!(winners[0].score.to_bits(), required_score.to_bits());
        Ok(())
    }

    #[test]
    fn malformed_scoring_inputs_fail_without_reaching_idf_assertion()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = [Some(1)];
        let encoded =
            EncodedDocLenSection::encode(0, 1, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let valid = snapshot(1, 1, 1)?;

        assert!(matches!(
            TermScorer::new(
                VecCursor::new(postings(&[0]), 1, 1),
                field,
                valid.clone(),
                2,
                1.0
            ),
            Err(ArgusError::InvalidDocFrequency {
                doc_freq: 2,
                doc_count: 1
            })
        ));
        let empty_field = snapshot(1, 0, 1)?;
        assert!(matches!(
            TermScorer::new(
                VecCursor::new(postings(&[0]), 1, 1),
                field,
                empty_field,
                1,
                1.0
            ),
            Err(ArgusError::InvalidSnapshot { .. })
        ));
        assert!(matches!(
            TermScorer::new(
                VecCursor::new(postings(&[0]), 1, 1),
                field,
                valid,
                1,
                f32::NAN
            ),
            Err(ArgusError::InvalidBoost { .. })
        ));
        Ok(())
    }
}
