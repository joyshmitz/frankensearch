//! Argus query execution.
//!
//! This module contains the deliberately exhaustive scorer used as the
//! correctness anchor for every later pruning path. It mirrors the pinned
//! Tantivy scorer-tree arithmetic, including buffered `Should` unions, while
//! retaining Quill's explicit `Option<u32>` cursor state so `u32::MAX` remains
//! a searchable global document id.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use thiserror::Error;

use crate::contract::{BM25_K1, compute_tf_cache, idf};
use crate::quiver::{DocLenField, PostingCodecError, SnapshotFieldStats};

const UNION_HORIZON: usize = 4_096;

/// Opaque locator for a posting's later POSITIONS lookup.
///
/// The handle deliberately carries only the posting ordinal. Position bytes
/// stay compressed until a phrase scorer asks the owning segment to resolve
/// the handle.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PositionsHandle(u32);

impl PositionsHandle {
    /// Build a handle from a zero-based posting ordinal.
    #[must_use]
    pub const fn new(posting_ordinal: u32) -> Self {
        Self(posting_ordinal)
    }

    /// Return the zero-based posting ordinal.
    #[must_use]
    pub const fn posting_ordinal(self) -> u32 {
        self.0
    }
}

/// Typed failures from exhaustive query evaluation.
#[derive(Debug, Error)]
pub enum ArgusError {
    /// A sealed posting cursor violated its already-validated storage contract.
    #[error(transparent)]
    Posting(#[from] PostingCodecError),
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
    fn positions_handle(&self) -> Option<PositionsHandle>;

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

impl PostingCursor for crate::quiver::PostingCursor<'_> {
    fn doc(&self) -> Option<u32> {
        crate::quiver::PostingCursor::doc(self)
    }

    fn freq(&self) -> Option<u32> {
        crate::quiver::PostingCursor::freq(self)
    }

    fn positions_handle(&self) -> Option<PositionsHandle> {
        crate::quiver::PostingCursor::posting_ordinal(self).map(PositionsHandle::new)
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        Ok(crate::quiver::PostingCursor::next(self)?.map(|posting| posting.doc_id))
    }

    fn advance(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        Ok(crate::quiver::PostingCursor::advance(self, target)?.map(|posting| posting.doc_id))
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

/// One BM25 term leaf bound to a segment cursor and DOCLEN column.
pub struct TermScorer<'a> {
    cursor: Box<dyn PostingCursor + 'a>,
    fieldnorms: DocLenField<'a>,
    snapshot: Bm25FieldSnapshot,
    weight: f32,
    cost: u64,
}

impl<'a> TermScorer<'a> {
    /// Build a term scorer from snapshot-level statistics and segment-local
    /// cursor state.
    ///
    /// `snapshot_doc_freq` is the sum across every live segment and therefore
    /// includes tombstoned postings until compaction. `cursor_cost` is the
    /// runtime estimate for this concrete segment cursor; for sealed terms it
    /// is the segment posting count.
    ///
    /// # Errors
    ///
    /// Rejects field drift, malformed snapshot statistics, inconsistent
    /// cursor accessors, invalid boosts, and `doc_freq > N` without invoking
    /// the vendored `idf` assertion.
    pub fn new<C>(
        cursor: C,
        fieldnorms: DocLenField<'a>,
        snapshot: Bm25FieldSnapshot,
        snapshot_doc_freq: u64,
        cursor_cost: u64,
        field_boost: f32,
    ) -> Result<Self, ArgusError>
    where
        C: PostingCursor + 'a,
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
            if cursor_cost == 0 {
                return Err(ArgusError::CursorInvariant(
                    "a non-empty cursor must have non-zero runtime cost",
                ));
            }
            validate_cursor_position(&cursor, fieldnorms)?;
        } else if cursor.freq().is_some() || cursor.positions_handle().is_some() {
            return Err(ArgusError::CursorInvariant(
                "exhausted cursor retained frequency or positions state",
            ));
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
            fieldnorms,
            snapshot,
            weight,
            cost: cursor_cost,
        })
    }

    fn doc(&self) -> Option<u32> {
        self.cursor.doc()
    }

    fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        let doc = self.cursor.next()?;
        validate_cursor_after_move(self.cursor.as_ref(), self.fieldnorms, doc)
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        let doc = self.cursor.advance(target)?;
        validate_cursor_after_move(self.cursor.as_ref(), self.fieldnorms, doc)
    }

    fn score(&mut self) -> Result<f32, ArgusError> {
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
                .fieldnorm_id(u64::from(doc))
                .ok_or(ArgusError::MissingFieldnorm {
                    field_ord: self.fieldnorms.field_ord(),
                    global_docid: doc,
                })?;
        let frequency = frequency as f32;
        let norm = self.snapshot.tf_cache[usize::from(fieldnorm_id)];
        let tf_factor = frequency / (frequency + norm);
        finite_score(self.weight * tf_factor, doc)
    }
}

fn validate_cursor_after_move(
    cursor: &dyn PostingCursor,
    fieldnorms: DocLenField<'_>,
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

fn validate_cursor_position<C>(cursor: &C, fieldnorms: DocLenField<'_>) -> Result<(), ArgusError>
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
    if fieldnorms.fieldnorm_id(u64::from(doc)).is_none() {
        return Err(ArgusError::MissingFieldnorm {
            field_ord: fieldnorms.field_ord(),
            global_docid: doc,
        });
    }
    Ok(())
}

/// Clause occurrence in a Boolean scorer tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Occur {
    /// Required positive clause.
    Must,
    /// Optional positive clause, or required when no `Must` exists.
    Should,
    /// Scoreless exclusion clause.
    MustNot,
}

/// One owned Boolean child and its occurrence.
pub struct BooleanClause<'a> {
    occur: Occur,
    scorer: ReferenceScorer<'a>,
}

impl<'a> BooleanClause<'a> {
    /// Build a Boolean child.
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
    Intersection(IntersectionScorer<'a>),
    Union(BufferedUnionScorer<'a>),
    RequiredOptional(RequiredOptionalScorer<'a>),
    Exclude(ExcludeScorer<'a>),
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
    pub const fn term(term: TermScorer<'a>) -> Self {
        Self {
            node: ScorerNode::Term(term),
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
    pub fn boolean(clauses: Vec<BooleanClause<'a>>) -> Result<Self, ArgusError> {
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
                Occur::Must => must.push(clause.scorer),
                Occur::Should => should.push(clause.scorer),
                Occur::MustNot => excluded.push(clause.scorer),
            }
        }

        let include = if must.is_empty() {
            scorer_union(should)?
        } else {
            let required = scorer_intersection(must)?;
            if should.is_empty() {
                required
            } else {
                let optional = scorer_union(should)?;
                Self {
                    node: ScorerNode::RequiredOptional(RequiredOptionalScorer::new(
                        required, optional,
                    )),
                }
            }
        };
        if excluded.is_empty() || include.doc().is_none() {
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
            ScorerNode::Intersection(scorer) => scorer.doc(),
            ScorerNode::Union(scorer) => scorer.doc(),
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
            ScorerNode::Intersection(scorer) => scorer.cost(),
            ScorerNode::Union(scorer) => scorer.cost(),
            ScorerNode::RequiredOptional(scorer) => scorer.cost(),
            ScorerNode::Exclude(scorer) => scorer.cost(),
        }
    }

    /// Move strictly forward to the next match.
    ///
    /// # Errors
    ///
    /// Propagates typed cursor, fieldnorm, or score-buffer failures.
    pub fn next(&mut self) -> Result<Option<u32>, ArgusError> {
        match &mut self.node {
            ScorerNode::Empty => Ok(None),
            ScorerNode::Term(scorer) => scorer.next(),
            ScorerNode::Intersection(scorer) => scorer.next(),
            ScorerNode::Union(scorer) => scorer.next(),
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
            ScorerNode::Intersection(scorer) => scorer.seek(target),
            ScorerNode::Union(scorer) => scorer.seek(target),
            ScorerNode::RequiredOptional(scorer) => scorer.seek(target),
            ScorerNode::Exclude(scorer) => scorer.seek(target),
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
            ScorerNode::Intersection(scorer) => scorer.score(),
            ScorerNode::Union(scorer) => scorer.score(),
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
        if limit == 0 {
            return Ok(Vec::new());
        }
        let capacity = limit.saturating_add(1);
        let mut heap = BinaryHeap::new();
        heap.try_reserve(capacity)
            .map_err(|_| ArgusError::Allocation {
                resource: "top-k heap",
                count: capacity,
            })?;
        while let Some(doc) = self.doc() {
            if live_docs.is_live(doc) {
                let entry = HeapEntry {
                    global_docid: doc,
                    score: self.score()?,
                };
                if heap.len() < limit {
                    heap.push(entry);
                } else if heap.peek().is_some_and(|cutoff| entry < *cutoff) {
                    let _ = heap.pop();
                    heap.push(entry);
                }
            }
            self.next()?;
        }
        let mut winners: Vec<_> = heap.into_iter().map(ScoredDoc::from).collect();
        winners.sort_by(compare_scored_best_first);
        Ok(winners)
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

fn scorer_intersection<'a>(
    mut scorers: Vec<ReferenceScorer<'a>>,
) -> Result<ReferenceScorer<'a>, ArgusError> {
    match scorers.len() {
        0 => Ok(ReferenceScorer::empty()),
        1 => scorers.pop().ok_or(ArgusError::CursorInvariant(
            "required scorer count changed during lowering",
        )),
        _ => Ok(ReferenceScorer {
            node: ScorerNode::Intersection(IntersectionScorer::new(scorers)?),
        }),
    }
}

fn scorer_union<'a>(
    mut scorers: Vec<ReferenceScorer<'a>>,
) -> Result<ReferenceScorer<'a>, ArgusError> {
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

struct IntersectionScorer<'a> {
    scorers: Vec<ReferenceScorer<'a>>,
    current: Option<u32>,
}

impl<'a> IntersectionScorer<'a> {
    fn new(mut scorers: Vec<ReferenceScorer<'a>>) -> Result<Self, ArgusError> {
        scorers.sort_by_key(ReferenceScorer::cost);
        let mut scorer = Self {
            scorers,
            current: None,
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
        if self.current.is_none() {
            return Ok(None);
        }
        self.scorers[0].next()?;
        self.align()
    }

    fn seek(&mut self, target: u32) -> Result<Option<u32>, ArgusError> {
        if self.current.is_some_and(|doc| doc >= target) {
            return Ok(self.current);
        }
        self.scorers[0].advance(target)?;
        self.align()
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
    scores: Vec<Option<f32>>,
    window_start: Option<u32>,
    scan_offset: usize,
    current: Option<u32>,
    current_score: f32,
}

impl<'a> BufferedUnionScorer<'a> {
    fn new(mut scorers: Vec<ReferenceScorer<'a>>) -> Result<Self, ArgusError> {
        scorers.retain(|scorer| scorer.doc().is_some());
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(UNION_HORIZON)
            .map_err(|_| ArgusError::Allocation {
                resource: "buffered union score window",
                count: UNION_HORIZON,
            })?;
        scores.resize(UNION_HORIZON, None);
        let mut scorer = Self {
            active: scorers,
            scores,
            window_start: None,
            scan_offset: 0,
            current: None,
            current_score: 0.0,
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

    fn clear_buffer(&mut self) {
        self.scores.fill(None);
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
                let total = self.scores[offset].unwrap_or(0.0) + contribution;
                self.scores[offset] = Some(finite_score(total, doc)?);
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
        while self.scan_offset < self.scores.len() {
            let offset = self.scan_offset;
            self.scan_offset += 1;
            if let Some(score) = self.scores[offset].take() {
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
}

impl<'a> RequiredOptionalScorer<'a> {
    fn new(required: ReferenceScorer<'a>, optional: ReferenceScorer<'a>) -> Self {
        Self {
            required: Box::new(required),
            optional: Box::new(optional),
            score_cache: None,
        }
    }

    fn doc(&self) -> Option<u32> {
        self.required.doc()
    }

    fn cost(&self) -> u64 {
        self.required.cost()
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
}

impl<'a> ExcludeScorer<'a> {
    fn new(
        include: ReferenceScorer<'a>,
        excluded: Vec<ReferenceScorer<'a>>,
    ) -> Result<Self, ArgusError> {
        let mut scorer = Self {
            include: Box::new(include),
            excluded,
        };
        scorer.skip_excluded()?;
        Ok(scorer)
    }

    fn doc(&self) -> Option<u32> {
        self.include.doc()
    }

    fn cost(&self) -> u64 {
        self.include.cost()
    }

    fn contains(&mut self, doc: u32) -> Result<bool, ArgusError> {
        for scorer in &mut self.excluded {
            if scorer.doc().is_some_and(|excluded_doc| excluded_doc <= doc)
                && scorer.advance(doc)? == Some(doc)
            {
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

/// Winner after phase-two external-id materialization.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedHit {
    /// Stable global Quill document id.
    pub global_docid: u32,
    /// External document id resolved only after top-k selection.
    pub document_id: String,
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
    mut resolve: impl FnMut(u32) -> Result<String, ArgusError>,
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

fn compare_scored_best_first(left: &ScoredDoc, right: &ScoredDoc) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.global_docid.cmp(&right.global_docid))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{BM25_B, fieldnorm_to_id, id_to_fieldnorm};
    use crate::quiver::{DocLenFieldInput, EncodedDocLenSection, EncodedPostingList, Posting};

    #[derive(Clone, Debug)]
    struct VecCursor {
        postings: Vec<Posting>,
        index: usize,
    }

    impl VecCursor {
        const fn new(postings: Vec<Posting>) -> Self {
            Self { postings, index: 0 }
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

        fn positions_handle(&self) -> Option<PositionsHandle> {
            self.current()
                .and_then(|_| u32::try_from(self.index).ok())
                .map(PositionsHandle::new)
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
        Ok(ReferenceScorer::term(TermScorer::new(
            VecCursor::new(postings),
            fieldnorms,
            snapshot.clone(),
            snapshot_doc_freq,
            cost,
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

    #[test]
    fn sealed_cursor_adapter_preserves_option_state_and_max_docid() -> Result<(), ArgusError> {
        let source = [
            Posting::new(3, 2),
            Posting::new(130, 1),
            Posting::new(u32::MAX, 7),
        ];
        let encoded = EncodedPostingList::encode(&source)?;
        let list = encoded.posting_list()?;
        let mut cursor = list.cursor()?;

        assert_eq!(PostingCursor::doc(&cursor), Some(3));
        assert_eq!(PostingCursor::freq(&cursor), Some(2));
        assert_eq!(
            PostingCursor::positions_handle(&cursor),
            Some(PositionsHandle::new(0))
        );
        assert_eq!(PostingCursor::advance(&mut cursor, 3)?, Some(3));
        assert_eq!(PostingCursor::advance(&mut cursor, 4)?, Some(130));
        assert_eq!(
            PostingCursor::positions_handle(&cursor),
            Some(PositionsHandle::new(1))
        );
        assert_eq!(
            PostingCursor::advance(&mut cursor, u32::MAX)?,
            Some(u32::MAX)
        );
        assert_eq!(PostingCursor::freq(&cursor), Some(7));
        assert_eq!(
            PostingCursor::positions_handle(&cursor),
            Some(PositionsHandle::new(2))
        );
        assert_eq!(PostingCursor::next(&mut cursor)?, None);
        assert_eq!(PostingCursor::next(&mut cursor)?, None);
        assert_eq!(PostingCursor::doc(&cursor), None);
        assert_eq!(PostingCursor::freq(&cursor), None);
        assert_eq!(PostingCursor::positions_handle(&cursor), None);
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
        assert_eq!(score.to_bits(), 1_054_438_986);
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
            BooleanClause::should(content_term),
            BooleanClause::should(title_term),
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
            BooleanClause::must(term(postings(&[1, 2]), field, &snapshot, 2, 2, 1.0)?),
            BooleanClause::should(term(postings(&[2, 3]), field, &snapshot, 2, 2, 1.0)?),
            BooleanClause::must_not(term(postings(&[2]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        let hits = must_with_optional.top_k(5, &AllLiveDocs)?;
        assert_eq!(
            hits.iter().map(|hit| hit.global_docid).collect::<Vec<_>>(),
            vec![1]
        );
        let expected_must = expected_term_score(&snapshot, 2, fieldnorm_to_id(1), 1, 1.0);
        assert_eq!(hits[0].score.to_bits(), expected_must.to_bits());

        let mut should_with_exclusion = ReferenceScorer::boolean(vec![
            BooleanClause::should(term(postings(&[2, 3]), field, &snapshot, 2, 2, 1.0)?),
            BooleanClause::must_not(term(postings(&[2]), field, &snapshot, 1, 1, 1.0)?),
        ])?;
        let hits = should_with_exclusion.top_k(5, &AllLiveDocs)?;
        assert_eq!(
            hits.iter().map(|hit| hit.global_docid).collect::<Vec<_>>(),
            vec![3]
        );

        let mut negative_only = ReferenceScorer::boolean(vec![BooleanClause::must_not(term(
            postings(&[1, 2]),
            field,
            &snapshot,
            2,
            2,
            1.0,
        )?)])?;
        assert!(negative_only.top_k(5, &AllLiveDocs)?.is_empty());
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
            clauses.push(BooleanClause::must(term(
                postings(&[0]),
                field,
                &snapshot,
                1,
                cost,
                boost,
            )?));
        }
        clauses.push(BooleanClause::should(term(
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
    fn buffered_union_reproduces_oracle_swap_remove_order_across_horizon()
    -> Result<(), Box<dyn std::error::Error>> {
        let lengths = vec![Some(1); 5_001];
        let encoded =
            EncodedDocLenSection::encode(0, 5_001, &[1], &[DocLenFieldInput::new(1, &lengths)])?;
        let section = encoded.section(&[1])?;
        let field = section.field(1).expect("field exists");
        let snapshot = snapshot(1, 5_001, 5_001)?;
        let clauses = vec![
            BooleanClause::should(term(postings(&[0]), field, &snapshot, 1, 1, 1.0)?),
            BooleanClause::should(term(postings(&[5_000]), field, &snapshot, 1, 1, 1.0e8)?),
            BooleanClause::should(term(postings(&[5_000]), field, &snapshot, 1, 1, -1.0e8)?),
            BooleanClause::should(term(postings(&[5_000]), field, &snapshot, 1, 1, 1.0)?),
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
                for term_index in 0..membership.len() {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    membership[term_index][doc] = state >> 61 != 0;
                }
            }
            let term_docs = |term_index: usize| {
                membership[term_index]
                    .iter()
                    .enumerate()
                    .filter_map(|(doc, present)| present.then_some(doc as u32))
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
            let either = ReferenceScorer::boolean(vec![
                BooleanClause::should(leaf(0)?),
                BooleanClause::should(leaf(1)?),
            ])?;
            let mut query = ReferenceScorer::boolean(vec![
                BooleanClause::must(either),
                BooleanClause::must(leaf(2)?),
                BooleanClause::must_not(leaf(3)?),
                BooleanClause::should(leaf(4)?),
            ])?;
            let mut actual = query
                .top_k(DOCS, &AllLiveDocs)?
                .into_iter()
                .map(|hit| hit.global_docid)
                .collect::<Vec<_>>();
            actual.sort_unstable();
            let expected = (0..DOCS)
                .filter(|&doc| {
                    (membership[0][doc] || membership[1][doc])
                        && membership[2][doc]
                        && !membership[3][doc]
                })
                .map(|doc| doc as u32)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "seed {seed}");
        }
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
            Ok(format!("external-{doc}"))
        })?;
        assert_eq!(calls, vec![1]);
        assert_eq!(hits[0].document_id, "external-1");

        let compacted = snapshot(1, 2, 1)?;
        let compacted_score = expected_term_score(&compacted, 1, fieldnorm_to_id(2), 1, 1.0);
        assert_ne!(expected.to_bits(), compacted_score.to_bits());
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
                VecCursor::new(postings(&[0])),
                field,
                valid.clone(),
                2,
                1,
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
                VecCursor::new(postings(&[0])),
                field,
                empty_field,
                1,
                1,
                1.0
            ),
            Err(ArgusError::InvalidSnapshot { .. })
        ));
        assert!(matches!(
            TermScorer::new(VecCursor::new(postings(&[0])), field, valid, 1, 1, f32::NAN),
            Err(ArgusError::InvalidBoost { .. })
        ));
        Ok(())
    }
}
