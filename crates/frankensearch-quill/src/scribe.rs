//! Scribe ingest pipeline.
//!
//! Tokenization, per-shard arenas, columnar accumulation, and segment flush
//! land behind this module in the Quill E1 milestones.
//!
//! Milestones `bd-quill-e1-scribe-bejd.3` and `.4` provide the per-shard
//! foundations the rest of Scribe builds on:
//!
//! - [`ByteArena`]: chunked bump storage with reset-and-reuse (chunk capacity
//!   is retained across flush cycles so steady-state ingest does not touch the
//!   global allocator).
//! - [`TermInterner`]: composite-key interner mapping
//!   `(field_ord, term bytes)` to dense local `u32` ids. Keys are stored once,
//!   as the exact on-disk TERMDICT composite key (big-endian `field_ord`
//!   prefix + term bytes, FSLX §5.1), so [`TermInterner::sorted_ids`] yields
//!   ids in precisely the order the dictionary serializer needs.
//! - Budget accounting ([`TermInterner::bytes_used`]) that feeds the shard
//!   flush trigger (`QuillConfig::scribe_shard_budget_bytes`).
//! - [`FrankensearchTokenizer`]: the allocation-reusing scalar reference for
//!   the shipping `SimpleTokenizer + LowerCaser` semantics.
//! - [`ColumnarAccumulator`]: schema-driven, per-field `SoA` token columns plus
//!   raw document lengths and their Tantivy-compatible fieldnorm bytes.
//!
//! Invariants:
//! - No per-token heap allocation on the intern hot path: an existing term
//!   costs one hash + one arena byte-compare; a new term costs one bump copy.
//! - Term ids are assigned in first-insertion order and are deterministic for
//!   a given ingest order (hasher choice never influences ids or output —
//!   collision buckets only affect probe cost).
//! - `CompactString`/owned allocation happens only at dictionary-build time,
//!   never per token.
//! - Document ordinals in the accumulator are shard-lease-relative and
//!   strictly ascending. Segment build is responsible for the one-time global
//!   docid rebase.
//! - Every indexed string field has parallel term-id and doc-ordinal columns;
//!   only fields whose descriptor enables positions allocate a position
//!   column.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

use thiserror::Error;

use crate::contract::fieldnorm_to_id;
use crate::error::QuillError;
use crate::grimoire::MAX_TERM_BYTES;
use crate::schema::{Analyzer as AnalyzerKind, FieldKind, SchemaDescriptor};

/// Default arena chunk size. 1 MiB amortizes chunk-vector growth while
/// keeping per-shard reset cheap; term corpora that exceed it simply add
/// chunks (each retained across [`ByteArena::reset`]).
pub const DEFAULT_ARENA_CHUNK_BYTES: usize = 1 << 20;

/// Composite dictionary key prefix width: big-endian `field_ord: u16`
/// (FSLX §5.1). Big-endian so raw byte comparison equals `(field, term)`
/// lexicographic order.
pub const FIELD_PREFIX_BYTES: usize = 2;

/// Number of document ordinals in one Keeper lease (Q1 §§3, 5).
///
/// Accumulator ordinals are offsets within one lease, so the valid domain is
/// `0..DOC_ORDS_PER_LEASE`. Crossing the boundary forces a segment cut before
/// a later stage rebases the offsets to global document IDs.
pub const DOC_ORDS_PER_LEASE: u32 = 1 << 16;

/// One token emitted by a Quill analyzer.
///
/// Offsets are half-open byte offsets into the original input. Positions are
/// `u32` end to end, matching the on-disk postings contract and making
/// truncation impossible at the analyzer/accumulator boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnalyzedToken {
    /// Normalized token text.
    pub text: String,
    /// Logical token position, starting at zero.
    pub position: u32,
    /// Source byte offset of the first token byte.
    pub offset_from: usize,
    /// Source byte offset immediately after the token.
    pub offset_to: usize,
    /// Position span. The default analyzer always emits one.
    pub position_length: usize,
}

/// Allocation-free callback surface shared by scalar and future SIMD/CASS
/// analyzer families.
///
/// The token reference is valid only for the duration of the callback. An
/// implementation may reuse one scratch token for the complete stream. The
/// trait is sealed because Quill relies on every implementation preserving
/// the `u32` token-count and stable-support invariants while appending without
/// a staging allocation.
pub trait TokenAnalyzer: sealed::Sealed {
    /// Whether this family implements the requested schema pipeline.
    ///
    /// This answer must remain stable for the lifetime of the implementation.
    /// A family may support several kinds so one accumulator can serve mixed
    /// schemas such as [`crate::schema::CASS_SEMANTIC_SCHEMA`].
    fn supports(&self, analyzer: AnalyzerKind) -> bool;

    /// Analyze `text` in source order and call `sink` once per emitted token.
    ///
    /// Callers invoke this only for kinds accepted by [`Self::supports`].
    fn analyze(&mut self, analyzer: AnalyzerKind, text: &str, sink: &mut dyn FnMut(&AnalyzedToken));

    /// Retained analyzer scratch included in RSS/reuse diagnostics.
    fn bytes_reserved(&self) -> usize {
        0
    }

    /// Reset logical analyzer state while retaining reusable scratch.
    fn reset(&mut self) {}
}

mod sealed {
    pub trait Sealed {}
}

/// Token-admission accounting returned by [`analyze_admitted`].
///
/// Query lowering needs the oversized count to distinguish genuinely empty
/// analysis from an unmatchable required/phrase clause. Document ingest uses
/// `admitted_tokens` for fieldnorms and statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnalysisReport {
    /// Tokens emitted by the analyzer before global length admission.
    pub raw_tokens: usize,
    /// Tokens whose normalized text was at most [`MAX_TERM_BYTES`].
    pub admitted_tokens: usize,
    /// Tokens rejected solely for exceeding [`MAX_TERM_BYTES`].
    pub oversized_tokens: usize,
}

/// A requested pipeline is absent from an injected analyzer family.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
#[error("analyzer family does not implement {analyzer:?}")]
pub struct UnsupportedAnalysis {
    /// Requested schema analyzer.
    pub analyzer: AnalyzerKind,
}

/// Analyze and apply Quill's global 65,530-byte term admission rule.
///
/// Filtering happens after analysis, so retained tokens keep their original
/// positions even when an oversized predecessor is dropped.
///
/// # Errors
///
/// Returns [`UnsupportedAnalysis`] when the family does not implement
/// `analyzer_kind`; the sink is not called in that case.
pub fn analyze_admitted<A: TokenAnalyzer + ?Sized>(
    analyzer: &mut A,
    analyzer_kind: AnalyzerKind,
    text: &str,
    sink: &mut dyn FnMut(&AnalyzedToken),
) -> Result<AnalysisReport, UnsupportedAnalysis> {
    if !analyzer.supports(analyzer_kind) {
        return Err(UnsupportedAnalysis {
            analyzer: analyzer_kind,
        });
    }
    let mut report = AnalysisReport::default();
    analyzer.analyze(analyzer_kind, text, &mut |token| {
        report.raw_tokens += 1;
        if token.text.len() > MAX_TERM_BYTES {
            report.oversized_tokens += 1;
            tracing::warn!(
                token_bytes = token.text.len(),
                max_token_bytes = MAX_TERM_BYTES,
                position = token.position,
                "Quill dropped an oversized analyzed token"
            );
            return;
        }
        report.admitted_tokens += 1;
        sink(token);
    });
    Ok(report)
}

/// Scalar reference implementation of the shipping frankensearch analyzer.
///
/// This fuses Tantivy's `SimpleTokenizer` and `LowerCaser`: split on
/// non-alphanumeric Unicode scalar values, ASCII-lowercase in place, and use
/// the full Unicode lowercase expansion otherwise. It deliberately does not
/// enforce [`MAX_TERM_BYTES`]; admission belongs to document/query consumers
/// so a dropped document token retains its position gap.
#[derive(Debug, Clone, Default)]
pub struct FrankensearchTokenizer {
    token: AnalyzedToken,
}

impl sealed::Sealed for FrankensearchTokenizer {}

#[inline]
fn tokenizer_next_char(text: &str, offset: usize) -> Option<(char, usize)> {
    let remaining = text.get(offset..)?;
    let first = *remaining.as_bytes().first()?;
    if first.is_ascii() {
        Some((char::from(first), offset + 1))
    } else {
        let ch = remaining.chars().next()?;
        Some((ch, offset + ch.len_utf8()))
    }
}

#[inline]
fn tokenizer_is_alphanumeric(ch: char) -> bool {
    if ch.is_ascii() {
        ch.is_ascii_alphanumeric()
    } else {
        ch.is_alphanumeric()
    }
}

#[inline]
fn next_token_position(position: u32) -> u32 {
    position
        .checked_add(1)
        .expect("analyzed token position exceeds the u32 contract")
}

impl TokenAnalyzer for FrankensearchTokenizer {
    fn supports(&self, analyzer: AnalyzerKind) -> bool {
        analyzer == AnalyzerKind::FrankensearchDefault
    }

    fn analyze(
        &mut self,
        analyzer: AnalyzerKind,
        text: &str,
        sink: &mut dyn FnMut(&AnalyzedToken),
    ) {
        debug_assert_eq!(analyzer, AnalyzerKind::FrankensearchDefault);
        let mut cursor = 0;
        let mut position = 0_u32;

        while let Some((ch, next_cursor)) = tokenizer_next_char(text, cursor) {
            if !tokenizer_is_alphanumeric(ch) {
                cursor = next_cursor;
                continue;
            }

            let offset_from = cursor;
            let mut offset_to = next_cursor;
            let mut resume_at = next_cursor;
            let mut all_ascii = ch.is_ascii();
            while let Some((next_ch, after_next)) = tokenizer_next_char(text, resume_at) {
                if !tokenizer_is_alphanumeric(next_ch) {
                    resume_at = after_next;
                    break;
                }
                all_ascii &= next_ch.is_ascii();
                offset_to = after_next;
                resume_at = after_next;
            }

            self.token.text.clear();
            let source = &text[offset_from..offset_to];
            if all_ascii {
                self.token.text.push_str(source);
                self.token.text.make_ascii_lowercase();
            } else {
                for source_char in source.chars() {
                    self.token.text.extend(source_char.to_lowercase());
                }
            }
            self.token.position = position;
            self.token.offset_from = offset_from;
            self.token.offset_to = offset_to;
            self.token.position_length = 1;
            sink(&self.token);

            position = next_token_position(position);
            cursor = resume_at;
        }
    }

    fn bytes_reserved(&self) -> usize {
        self.token.text.capacity()
    }

    fn reset(&mut self) {
        self.token.text.clear();
        self.token.position = 0;
        self.token.offset_from = 0;
        self.token.offset_to = 0;
        self.token.position_length = 0;
    }
}

/// A span into a [`ByteArena`]: which chunk, where, how long.
///
/// 12 bytes, `Copy`; the interner keeps one per term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaSpan {
    chunk: u32,
    offset: u32,
    len: u32,
}

/// Chunked bump allocator for raw bytes.
///
/// Not a general allocator: append-only between resets, no per-item free.
/// [`reset`](Self::reset) retains every standard chunk at full capacity so a
/// steady-state flush cycle performs zero global allocations.
#[derive(Debug)]
pub struct ByteArena {
    chunks: Vec<Vec<u8>>,
    chunk_size: usize,
    /// Index of the chunk currently accepting writes.
    active: usize,
}

impl ByteArena {
    /// Create an arena with the given chunk size (min 4 KiB to bound the
    /// chunk-vector length on adversarial configs).
    #[must_use]
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        let chunk_size = chunk_size.max(4096);
        Self {
            chunks: vec![Vec::with_capacity(chunk_size)],
            chunk_size,
            active: 0,
        }
    }

    /// Copy `bytes` into the arena, returning its span.
    ///
    /// Oversized inputs (longer than the chunk size) get a dedicated
    /// exactly-sized chunk; the previously active chunk stays active so its
    /// remaining capacity is not wasted.
    ///
    /// # Panics
    /// Panics if chunk count, offset, or span length overflow `u32` — at the
    /// 64 MiB shard budget these are unreachable by orders of magnitude.
    pub fn push(&mut self, bytes: &[u8]) -> ArenaSpan {
        if bytes.len() > self.chunk_size {
            let mut chunk = Vec::with_capacity(bytes.len());
            chunk.extend_from_slice(bytes);
            self.chunks.push(chunk);
            let idx = self.chunks.len() - 1;
            return ArenaSpan {
                chunk: u32::try_from(idx).expect("arena chunk count exceeds u32"),
                offset: 0,
                len: u32::try_from(bytes.len()).expect("arena span exceeds u32"),
            };
        }
        let needs_new = {
            let chunk = &self.chunks[self.active];
            chunk.len() + bytes.len() > chunk.capacity()
        };
        if needs_new {
            // Find or create the next reusable standard chunk. Dedicated
            // oversized chunks (capacity != chunk_size) are skipped.
            let mut next = self.active + 1;
            while next < self.chunks.len() && self.chunks[next].capacity() != self.chunk_size {
                next += 1;
            }
            if next == self.chunks.len() {
                self.chunks.push(Vec::with_capacity(self.chunk_size));
            }
            self.active = next;
        }
        let chunk_idx = self.active;
        let chunk = &mut self.chunks[chunk_idx];
        let offset = chunk.len();
        chunk.extend_from_slice(bytes);
        ArenaSpan {
            chunk: u32::try_from(chunk_idx).expect("arena chunk count exceeds u32"),
            offset: u32::try_from(offset).expect("arena offset exceeds u32"),
            len: u32::try_from(bytes.len()).expect("arena span exceeds u32"),
        }
    }

    /// Resolve a span to its bytes.
    ///
    /// # Panics
    /// Panics if the span does not belong to this arena (spans are only ever
    /// produced by [`push`](Self::push) on the same arena; crossing arenas is
    /// a programming error, not a recoverable state).
    #[must_use]
    pub fn resolve(&self, span: ArenaSpan) -> &[u8] {
        let chunk = &self.chunks[span.chunk as usize];
        &chunk[span.offset as usize..span.offset as usize + span.len as usize]
    }

    /// Bytes currently stored (sum of chunk lengths).
    #[must_use]
    pub fn bytes_used(&self) -> usize {
        self.chunks.iter().map(Vec::len).sum()
    }

    /// Bytes currently reserved (sum of chunk capacities) — the RSS-relevant
    /// figure reported in flush-cycle tracing.
    #[must_use]
    pub fn bytes_reserved(&self) -> usize {
        self.chunks.iter().map(Vec::capacity).sum()
    }

    /// Number of chunks (soak tests assert this stabilizes across cycles).
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Clear all content, retaining standard-chunk capacity. Dedicated
    /// oversized chunks are dropped (they are corpus outliers by definition;
    /// retaining them would ratchet RSS on pathological inputs).
    pub fn reset(&mut self) {
        self.chunks.retain(|c| c.capacity() == self.chunk_size);
        if self.chunks.is_empty() {
            self.chunks.push(Vec::with_capacity(self.chunk_size));
        }
        for chunk in &mut self.chunks {
            chunk.clear();
        }
        self.active = 0;
    }
}

impl Default for ByteArena {
    fn default() -> Self {
        Self::with_chunk_size(DEFAULT_ARENA_CHUNK_BYTES)
    }
}

/// Collision bucket: hash → term id(s). The `Many` arm is exercised only on
/// 64-bit hash collisions (or by tests injecting a degenerate hasher).
#[derive(Debug)]
enum Bucket {
    One(u32),
    Many(Vec<u32>),
}

/// Per-shard composite-key term interner.
///
/// Keys are `(field_ord, term bytes)`, stored once in the arena as the
/// on-disk composite form (BE field prefix + term bytes). Ids are dense u32s
/// in first-insertion order.
///
/// The hasher is generic (default [`ahash::RandomState`] — in-memory only;
/// durable hashing elsewhere in Quill is xxh3 by contract, FSLX §2). Tests
/// inject a constant hasher to force every key through the `Many`
/// verification path.
#[derive(Debug)]
pub struct TermInterner<S: BuildHasher = ahash::RandomState> {
    arena: ByteArena,
    spans: Vec<ArenaSpan>,
    buckets: HashMap<u64, Bucket>,
    hasher: S,
    /// Scratch buffer for composite-key assembly (reused, never shrunk).
    key_scratch: Vec<u8>,
}

impl TermInterner<ahash::RandomState> {
    #[must_use]
    pub fn new() -> Self {
        Self::with_hasher(ahash::RandomState::new())
    }
}

impl Default for TermInterner<ahash::RandomState> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: BuildHasher> TermInterner<S> {
    #[must_use]
    pub fn with_hasher(hasher: S) -> Self {
        Self {
            arena: ByteArena::default(),
            spans: Vec::new(),
            buckets: HashMap::new(),
            hasher,
            key_scratch: Vec::with_capacity(64),
        }
    }

    fn hash_key(&self, key: &[u8]) -> u64 {
        let mut h = self.hasher.build_hasher();
        h.write(key);
        h.finish()
    }

    /// Intern `(field_ord, term)`, returning the dense local id.
    ///
    /// Hot path: existing terms cost one hash + one arena compare and perform
    /// zero allocations (the composite key is assembled in a reused scratch
    /// buffer).
    ///
    /// # Panics
    /// Panics if the number of distinct terms exceeds `u32` — unreachable
    /// under the shard budget.
    pub fn intern(&mut self, field_ord: u16, term: &[u8]) -> u32 {
        self.key_scratch.clear();
        self.key_scratch.extend_from_slice(&field_ord.to_be_bytes());
        self.key_scratch.extend_from_slice(term);
        let hash = self.hash_key(&self.key_scratch);

        if let Some(bucket) = self.buckets.get(&hash) {
            match bucket {
                Bucket::One(id) => {
                    if self.arena.resolve(self.spans[*id as usize]) == self.key_scratch.as_slice() {
                        return *id;
                    }
                }
                Bucket::Many(ids) => {
                    for id in ids {
                        if self.arena.resolve(self.spans[*id as usize])
                            == self.key_scratch.as_slice()
                        {
                            return *id;
                        }
                    }
                }
            }
        }

        // New term: copy the composite key into the arena, assign the next id.
        let span = self.arena.push(&self.key_scratch);
        let id = u32::try_from(self.spans.len()).expect("term id space exceeds u32");
        self.spans.push(span);
        match self.buckets.entry(hash) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(Bucket::One(id));
            }
            std::collections::hash_map::Entry::Occupied(mut o) => match o.get_mut() {
                Bucket::One(existing) => {
                    let existing = *existing;
                    *o.get_mut() = Bucket::Many(vec![existing, id]);
                }
                Bucket::Many(ids) => ids.push(id),
            },
        }
        id
    }

    /// Number of distinct interned terms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Resolve an id to its composite key (BE field prefix + term bytes).
    ///
    /// # Panics
    /// Panics on an id not produced by this interner instance.
    #[must_use]
    pub fn composite_key(&self, id: u32) -> &[u8] {
        self.arena.resolve(self.spans[id as usize])
    }

    /// Resolve an id to `(field_ord, term bytes)`.
    ///
    /// # Panics
    /// Panics on an id not produced by this interner instance.
    #[must_use]
    pub fn field_and_term(&self, id: u32) -> (u16, &[u8]) {
        let key = self.composite_key(id);
        let field = u16::from_be_bytes([key[0], key[1]]);
        (field, &key[FIELD_PREFIX_BYTES..])
    }

    /// Ids sorted by composite key bytes — exactly the on-disk TERMDICT order
    /// (FSLX §5.1: the BE field prefix makes byte order equal (field, term)
    /// order). Called once per flush; cost is the sort, not paid per token.
    ///
    /// # Panics
    /// Panics if the id space exceeds `u32` (unreachable; see [`Self::intern`]).
    #[must_use]
    pub fn sorted_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> =
            (0..u32::try_from(self.spans.len()).expect("term id space exceeds u32")).collect();
        ids.sort_unstable_by(|a, b| self.composite_key(*a).cmp(self.composite_key(*b)));
        ids
    }

    /// Approximate live bytes held, for the shard flush trigger.
    ///
    /// This deliberately uses lengths rather than capacities: reset retains
    /// allocations for reuse, and counting those retained capacities as live
    /// would make every post-flush document immediately flush again. Use
    /// [`Self::bytes_reserved`] for RSS diagnostics.
    #[must_use]
    pub fn bytes_used(&self) -> usize {
        const BUCKET_ESTIMATE: usize = 8 + std::mem::size_of::<Bucket>() + 8;
        let collision_ids = self
            .buckets
            .values()
            .map(|bucket| match bucket {
                Bucket::One(_) => 0,
                Bucket::Many(ids) => ids.len() * std::mem::size_of::<u32>(),
            })
            .sum::<usize>();
        self.arena.bytes_used()
            + self.spans.len() * std::mem::size_of::<ArenaSpan>()
            + self.buckets.len() * BUCKET_ESTIMATE
            + collision_ids
    }

    /// Complete retained interner allocation for RSS/reuse diagnostics.
    #[must_use]
    pub fn bytes_reserved(&self) -> usize {
        const BUCKET_ESTIMATE: usize = 8 + std::mem::size_of::<Bucket>() + 8;
        let collision_ids = self
            .buckets
            .values()
            .map(|bucket| match bucket {
                Bucket::One(_) => 0,
                Bucket::Many(ids) => ids.capacity() * std::mem::size_of::<u32>(),
            })
            .sum::<usize>();
        self.arena
            .bytes_reserved()
            .saturating_add(
                self.spans
                    .capacity()
                    .saturating_mul(std::mem::size_of::<ArenaSpan>()),
            )
            .saturating_add(self.buckets.capacity().saturating_mul(BUCKET_ESTIMATE))
            .saturating_add(self.key_scratch.capacity())
            .saturating_add(collision_ids)
    }

    /// Reset for the next flush cycle: clears terms, retains arena chunk and
    /// container capacity (steady-state cycles allocate nothing).
    pub fn reset(&mut self) {
        self.arena.reset();
        self.spans.clear();
        self.buckets.clear();
        self.key_scratch.clear();
    }

    /// Arena diagnostics for flush-cycle tracing:
    /// `(bytes_used, bytes_reserved, chunk_count)`.
    #[must_use]
    pub fn arena_stats(&self) -> (usize, usize, usize) {
        (
            self.arena.bytes_used(),
            self.arena.bytes_reserved(),
            self.arena.chunk_count(),
        )
    }
}

/// One indexed string value supplied for a document.
///
/// Values may name only [`FieldKind::Keyword`] or [`FieldKind::Text`] fields.
/// Omitted indexed string fields receive a zero document length and fieldnorm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexedFieldValue<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Borrowed source text.
    pub text: &'a str,
}

impl<'a> IndexedFieldValue<'a> {
    /// Construct a borrowed indexed field value.
    #[must_use]
    pub const fn new(field_ord: u16, text: &'a str) -> Self {
        Self { field_ord, text }
    }
}

/// Typed validation failures from [`ColumnarAccumulator::add_document`].
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum AccumulatorError {
    /// A lease-relative ordinal cannot name a document in another lease.
    #[error(
        "document ordinal {doc_ord} is outside the lease-relative range 0..{DOC_ORDS_PER_LEASE}"
    )]
    DocumentOutsideLease {
        /// Rejected lease-relative ordinal.
        doc_ord: u32,
    },
    /// Document ordinals must ascend within one shard lease.
    #[error("document ordinal {current} is not greater than prior ordinal {previous}")]
    OutOfOrderDocument {
        /// Last completed document ordinal.
        previous: u32,
        /// Rejected ordinal.
        current: u32,
    },
    /// More documents were supplied than the public u32 counter can express.
    #[error("columnar accumulator document count exceeds u32")]
    TooManyDocuments,
    /// A value named no field in the validated dense descriptor.
    #[error("unknown schema field ordinal {field_ord}")]
    UnknownField {
        /// Rejected field ordinal.
        field_ord: u16,
    },
    /// A document supplied the same indexed field more than once.
    #[error("document {doc_ord} supplies field {field_ord} more than once")]
    DuplicateField {
        /// Current shard-relative document ordinal.
        doc_ord: u32,
        /// Repeated field ordinal.
        field_ord: u16,
    },
    /// A string value named a numeric or stored-only field.
    #[error("field {field_ord} ({field_name}) is not an indexed string field")]
    NonStringField {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
    },
    /// The schema requested an analyzer that has not landed in Scribe yet.
    #[error("field {field_ord} ({field_name}) requires unsupported analyzer {analyzer:?}")]
    UnsupportedAnalyzer {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
        /// Requested schema pipeline.
        analyzer: AnalyzerKind,
    },
    /// Source length exceeded the u32 position proof used by the accumulator.
    #[error("field {field_ord} source is {bytes} bytes and exceeds the u32 position domain")]
    SourceTooLarge {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Source byte length.
        bytes: usize,
    },
}

/// Completion report for one atomically accumulated document.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DocumentAccumulation {
    /// Admitted indexed token occurrences across all supplied fields.
    pub admitted_tokens: u64,
    /// Tokens dropped by the global 65,530-byte admission rule.
    pub oversized_tokens: u64,
    /// Complete retained accumulator footprint after the document.
    pub bytes_reserved: usize,
    /// Live logical accumulator footprint after the document.
    pub bytes_used: usize,
}

/// Per-field structure-of-arrays token and document-length columns.
///
/// `term_ids`, `doc_ords`, and (when present) `positions` are parallel. The
/// document-length and fieldnorm columns align with
/// [`ColumnarAccumulator::document_ords`] rather than with token rows.
#[derive(Debug)]
pub struct FieldTokenColumns {
    field_ord: u16,
    term_ids: Vec<u32>,
    doc_ords: Vec<u32>,
    positions: Option<Vec<u32>>,
    document_lengths: Vec<u32>,
    fieldnorm_ids: Vec<u8>,
    total_tokens: u64,
}

impl FieldTokenColumns {
    fn new(field_ord: u16, positions: bool) -> Self {
        Self {
            field_ord,
            term_ids: Vec::new(),
            doc_ords: Vec::new(),
            positions: positions.then(Vec::new),
            document_lengths: Vec::new(),
            fieldnorm_ids: Vec::new(),
            total_tokens: 0,
        }
    }

    /// Stable schema field ordinal represented by these columns.
    #[must_use]
    pub const fn field_ord(&self) -> u16 {
        self.field_ord
    }

    /// Local composite term IDs in document/token order.
    #[must_use]
    pub fn term_ids(&self) -> &[u32] {
        &self.term_ids
    }

    /// Shard-lease-relative document ordinals parallel to [`Self::term_ids`].
    #[must_use]
    pub fn doc_ords(&self) -> &[u32] {
        &self.doc_ords
    }

    /// Positions parallel to token rows, or `None` when the schema does not
    /// persist positions for this field.
    #[must_use]
    pub fn positions(&self) -> Option<&[u32]> {
        self.positions.as_deref()
    }

    /// Exact admitted token count for each completed document.
    #[must_use]
    pub fn document_lengths(&self) -> &[u32] {
        &self.document_lengths
    }

    /// Tantivy-compatible quantized fieldnorm for each completed document.
    #[must_use]
    pub fn fieldnorm_ids(&self) -> &[u8] {
        &self.fieldnorm_ids
    }

    /// Exact token total used by FSLX STATS/avgdl.
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Number of admitted token triples for this field.
    #[must_use]
    pub fn len(&self) -> usize {
        self.term_ids.len()
    }

    /// Whether this field has no token triples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.term_ids.is_empty()
    }

    fn append_token(&mut self, term_id: u32, doc_ord: u32, position: u32) {
        self.term_ids.push(term_id);
        self.doc_ords.push(doc_ord);
        if let Some(positions) = &mut self.positions {
            positions.push(position);
        }
    }

    fn begin_document(&mut self) {
        self.document_lengths.push(0);
        self.fieldnorm_ids.push(fieldnorm_to_id(0));
    }

    fn finish_document_field(&mut self, length: u32) {
        let raw = self
            .document_lengths
            .last_mut()
            .expect("begin_document creates the document-length slot");
        *raw = length;
        let fieldnorm = self
            .fieldnorm_ids
            .last_mut()
            .expect("begin_document creates the fieldnorm slot");
        *fieldnorm = fieldnorm_to_id(length);
        self.total_tokens += u64::from(length);
    }

    fn bytes_reserved(&self) -> usize {
        let u32_bytes = std::mem::size_of::<u32>();
        self.term_ids
            .capacity()
            .saturating_mul(u32_bytes)
            .saturating_add(self.doc_ords.capacity().saturating_mul(u32_bytes))
            .saturating_add(self.positions.as_ref().map_or(0, |positions| {
                positions.capacity().saturating_mul(u32_bytes)
            }))
            .saturating_add(self.document_lengths.capacity().saturating_mul(u32_bytes))
            .saturating_add(self.fieldnorm_ids.capacity())
    }

    fn bytes_used(&self) -> usize {
        let u32_bytes = std::mem::size_of::<u32>();
        self.term_ids
            .len()
            .saturating_mul(u32_bytes)
            .saturating_add(self.doc_ords.len().saturating_mul(u32_bytes))
            .saturating_add(
                self.positions
                    .as_ref()
                    .map_or(0, |positions| positions.len().saturating_mul(u32_bytes)),
            )
            .saturating_add(self.document_lengths.len().saturating_mul(u32_bytes))
            .saturating_add(self.fieldnorm_ids.len())
    }

    fn reset(&mut self) {
        self.term_ids.clear();
        self.doc_ords.clear();
        if let Some(positions) = &mut self.positions {
            positions.clear();
        }
        self.document_lengths.clear();
        self.fieldnorm_ids.clear();
        self.total_tokens = 0;
    }
}

/// Schema-driven shard-local columnar token accumulator.
///
/// A complete document is validated before any column is changed. Indexed
/// fields are then processed in schema order (not caller order), making local
/// term-ID assignment deterministic. The supplied `doc_ord` remains relative
/// to its Keeper lease until the flush/seal stage rebases it.
#[derive(Debug)]
pub struct ColumnarAccumulator<A = FrankensearchTokenizer> {
    schema: SchemaDescriptor,
    terms: TermInterner,
    fields: Vec<FieldTokenColumns>,
    document_ords: Vec<u32>,
    seen_fields: Vec<bool>,
    last_doc_ord: Option<u32>,
    analyzer: A,
}

impl ColumnarAccumulator<FrankensearchTokenizer> {
    /// Create an empty accumulator for a validated compile-time schema.
    ///
    /// # Errors
    ///
    /// Returns [`QuillError::InvalidDescriptor`] when `schema` violates the
    /// dense-ID and field-shape invariants, or [`QuillError::Resource`] when
    /// the default family does not implement every requested analyzer.
    pub fn new(schema: SchemaDescriptor) -> Result<Self, QuillError> {
        Self::with_analyzer(schema, FrankensearchTokenizer::default())
    }
}

impl<A: TokenAnalyzer> ColumnarAccumulator<A> {
    /// Create an empty accumulator with an injected analyzer implementation.
    ///
    /// This is the stable scalar/SIMD swap seam. One family may advertise
    /// several [`AnalyzerKind`] values, and implementations of a given kind
    /// must emit the same token stream. A schema field requesting a missing
    /// kind is rejected atomically at construction.
    ///
    /// # Errors
    ///
    /// Returns [`QuillError::InvalidDescriptor`] when `schema` violates the
    /// dense-ID and field-shape invariants, or [`QuillError::Resource`] when
    /// the injected family does not implement every requested analyzer.
    pub fn with_analyzer(schema: SchemaDescriptor, analyzer: A) -> Result<Self, QuillError> {
        schema.validate()?;
        if let Some((field, requested)) = schema.fields.iter().find_map(|field| {
            let FieldKind::Text {
                analyzer: requested,
                ..
            } = field.kind
            else {
                return None;
            };
            (!analyzer.supports(requested)).then_some((field, requested))
        }) {
            return Err(QuillError::Resource {
                resource: "analyzer pipeline",
                detail: format!(
                    "field {} ({}) requests unavailable analyzer {requested:?}",
                    field.id, field.name
                ),
            });
        }
        let fields = schema
            .fields
            .iter()
            .filter_map(|field| match field.kind {
                FieldKind::Keyword => Some(FieldTokenColumns::new(field.id, false)),
                FieldKind::Text { positions, .. } => {
                    Some(FieldTokenColumns::new(field.id, positions))
                }
                FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => None,
            })
            .collect();
        Ok(Self {
            schema,
            terms: TermInterner::new(),
            fields,
            document_ords: Vec::new(),
            seen_fields: vec![false; schema.fields.len()],
            last_doc_ord: None,
            analyzer,
        })
    }

    /// Add one complete document.
    ///
    /// `values` may be in any order; Scribe processes them in schema order.
    /// Missing indexed string fields receive length/norm zero. A keyword uses
    /// raw exact bytes and emits one token even when empty. Analyzed text uses
    /// the descriptor's analyzer and the global term admission rule.
    ///
    /// # Errors
    ///
    /// Returns a typed validation error before mutating any column when the
    /// document ordinal, field set, field kind, analyzer, or source-size proof
    /// is invalid.
    pub fn add_document(
        &mut self,
        doc_ord: u32,
        values: &[IndexedFieldValue<'_>],
    ) -> Result<DocumentAccumulation, AccumulatorError> {
        if doc_ord >= DOC_ORDS_PER_LEASE {
            return Err(AccumulatorError::DocumentOutsideLease { doc_ord });
        }
        if let Some(previous) = self.last_doc_ord
            && doc_ord <= previous
        {
            return Err(AccumulatorError::OutOfOrderDocument {
                previous,
                current: doc_ord,
            });
        }
        if self.document_ords.len() == u32::MAX as usize {
            return Err(AccumulatorError::TooManyDocuments);
        }

        self.seen_fields.fill(false);
        for value in values {
            let field_index = usize::from(value.field_ord);
            let Some(field) = self.schema.fields.get(field_index) else {
                return Err(AccumulatorError::UnknownField {
                    field_ord: value.field_ord,
                });
            };
            if field.id != value.field_ord {
                return Err(AccumulatorError::UnknownField {
                    field_ord: value.field_ord,
                });
            }
            if std::mem::replace(&mut self.seen_fields[field_index], true) {
                return Err(AccumulatorError::DuplicateField {
                    doc_ord,
                    field_ord: value.field_ord,
                });
            }
            match field.kind {
                FieldKind::Keyword => {}
                FieldKind::Text { analyzer, .. } if self.analyzer.supports(analyzer) => {}
                FieldKind::Text { analyzer, .. } => {
                    return Err(AccumulatorError::UnsupportedAnalyzer {
                        field_ord: field.id,
                        field_name: field.name,
                        analyzer,
                    });
                }
                FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => {
                    return Err(AccumulatorError::NonStringField {
                        field_ord: field.id,
                        field_name: field.name,
                    });
                }
            }
            if u32::try_from(value.text.len()).is_err() {
                return Err(AccumulatorError::SourceTooLarge {
                    field_ord: field.id,
                    bytes: value.text.len(),
                });
            }
        }

        for field in &mut self.fields {
            field.begin_document();
        }

        let mut admitted_tokens = 0_u64;
        let mut oversized_tokens = 0_u64;
        for field_index in 0..self.fields.len() {
            let field_ord = self.fields[field_index].field_ord;
            let Some(value) = values.iter().find(|value| value.field_ord == field_ord) else {
                continue;
            };
            let descriptor = self.schema.fields[usize::from(field_ord)];
            let (length, dropped) = match descriptor.kind {
                FieldKind::Keyword => {
                    if value.text.len() > MAX_TERM_BYTES {
                        tracing::warn!(
                            field_ord,
                            token_bytes = value.text.len(),
                            max_token_bytes = MAX_TERM_BYTES,
                            "Quill dropped an oversized keyword token"
                        );
                        (0, 1)
                    } else {
                        let term_id = self.terms.intern(field_ord, value.text.as_bytes());
                        self.fields[field_index].append_token(term_id, doc_ord, 0);
                        (1, 0)
                    }
                }
                FieldKind::Text { analyzer, .. } => {
                    debug_assert!(self.analyzer.supports(analyzer));
                    let terms = &mut self.terms;
                    let column = &mut self.fields[field_index];
                    let report =
                        analyze_admitted(&mut self.analyzer, analyzer, value.text, &mut |token| {
                            let term_id = terms.intern(field_ord, token.text.as_bytes());
                            column.append_token(term_id, doc_ord, token.position);
                        })
                        .expect("document validation checked analyzer-family support");
                    (
                        u32::try_from(report.admitted_tokens)
                            .expect("validated source length bounds admitted token count to u32"),
                        u64::try_from(report.oversized_tokens)
                            .expect("token count always fits u64"),
                    )
                }
                FieldKind::StoredOnly | FieldKind::I64 { .. } | FieldKind::U64 { .. } => {
                    unreachable!("constructor excludes non-string fields from token columns")
                }
            };
            self.fields[field_index].finish_document_field(length);
            admitted_tokens += u64::from(length);
            oversized_tokens += dropped;
        }

        self.document_ords.push(doc_ord);
        self.last_doc_ord = Some(doc_ord);
        Ok(DocumentAccumulation {
            admitted_tokens,
            oversized_tokens,
            bytes_reserved: self.bytes_reserved(),
            bytes_used: self.bytes_used(),
        })
    }

    /// Validated schema associated with the accumulator.
    #[must_use]
    pub const fn schema(&self) -> SchemaDescriptor {
        self.schema
    }

    /// Composite field-namespaced term interner.
    #[must_use]
    pub const fn terms(&self) -> &TermInterner {
        &self.terms
    }

    /// Indexed string field columns in stable schema order.
    #[must_use]
    pub fn fields(&self) -> &[FieldTokenColumns] {
        &self.fields
    }

    /// Look up one indexed string field's columns.
    #[must_use]
    pub fn field(&self, field_ord: u16) -> Option<&FieldTokenColumns> {
        self.fields
            .binary_search_by_key(&field_ord, FieldTokenColumns::field_ord)
            .ok()
            .map(|index| &self.fields[index])
    }

    /// Completed shard-relative document ordinals. Per-field length/norm
    /// columns align with this slice.
    #[must_use]
    pub fn document_ords(&self) -> &[u32] {
        &self.document_ords
    }

    /// First completed local document ordinal, if any.
    #[must_use]
    pub fn first_doc_ord(&self) -> Option<u32> {
        self.document_ords.first().copied()
    }

    /// Last completed local document ordinal, if any.
    #[must_use]
    pub const fn last_doc_ord(&self) -> Option<u32> {
        self.last_doc_ord
    }

    /// Number of completed documents.
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.document_ords.len()
    }

    /// Total admitted token triples across indexed string fields.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.fields.iter().map(FieldTokenColumns::len).sum()
    }

    /// Complete retained footprint for RSS and reuse diagnostics.
    #[must_use]
    pub fn bytes_reserved(&self) -> usize {
        self.terms
            .bytes_reserved()
            .saturating_add(
                self.fields
                    .capacity()
                    .saturating_mul(std::mem::size_of::<FieldTokenColumns>()),
            )
            .saturating_add(
                self.fields
                    .iter()
                    .map(FieldTokenColumns::bytes_reserved)
                    .fold(0, usize::saturating_add),
            )
            .saturating_add(
                self.document_ords
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(self.seen_fields.capacity().div_ceil(8))
            .saturating_add(self.analyzer.bytes_reserved())
    }

    /// Live logical footprint used by the post-document flush trigger.
    ///
    /// Retained spare capacity is excluded so a reset shard can refill the
    /// same allocation instead of degenerating into one-document segments.
    #[must_use]
    pub fn bytes_used(&self) -> usize {
        self.terms
            .bytes_used()
            .saturating_add(
                self.fields
                    .len()
                    .saturating_mul(std::mem::size_of::<FieldTokenColumns>()),
            )
            .saturating_add(
                self.fields
                    .iter()
                    .map(FieldTokenColumns::bytes_used)
                    .fold(0, usize::saturating_add),
            )
            .saturating_add(
                self.document_ords
                    .len()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(self.seen_fields.len().div_ceil(8))
    }

    /// Whether a completed document has carried the retained footprint to or
    /// beyond the configured shard budget.
    #[must_use]
    pub fn should_flush(&self, budget_bytes: usize) -> bool {
        !self.document_ords.is_empty() && self.bytes_used() >= budget_bytes
    }

    /// Clear logical contents for the next flush cycle while retaining normal
    /// vector, arena, hash-map, and tokenizer scratch capacity.
    pub fn reset(&mut self) {
        self.terms.reset();
        for field in &mut self.fields {
            field.reset();
        }
        self.document_ords.clear();
        self.seen_fields.fill(false);
        self.last_doc_ord = None;
        self.analyzer.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{CASS_SEMANTIC_SCHEMA, DEFAULT_SCHEMA, FieldDescriptor};
    use frankensearch_lexical::tantivy_crate::tokenizer::TokenStream;
    use serde_json::Value;
    use std::hash::BuildHasherDefault;

    const LANGUAGE_CONTRACT_FIXTURE: &str =
        include_str!("../../../tests/fixtures/quill_language_contract.json");

    /// Degenerate hasher: every key hashes to 0, forcing every intern through
    /// the `Many` collision-verification path.
    #[derive(Default)]
    struct ConstHasher;
    impl Hasher for ConstHasher {
        fn finish(&self) -> u64 {
            0
        }
        fn write(&mut self, _bytes: &[u8]) {}
    }
    type ConstBuild = BuildHasherDefault<ConstHasher>;

    const MIXED_POSITION_FIELDS: [FieldDescriptor; 4] = [
        FieldDescriptor {
            id: 0,
            name: "key",
            kind: FieldKind::Keyword,
            stored: false,
        },
        FieldDescriptor {
            id: 1,
            name: "with_positions",
            kind: FieldKind::Text {
                analyzer: AnalyzerKind::FrankensearchDefault,
                positions: true,
            },
            stored: false,
        },
        FieldDescriptor {
            id: 2,
            name: "without_positions",
            kind: FieldKind::Text {
                analyzer: AnalyzerKind::FrankensearchDefault,
                positions: false,
            },
            stored: false,
        },
        FieldDescriptor {
            id: 3,
            name: "stored",
            kind: FieldKind::StoredOnly,
            stored: true,
        },
    ];
    const MIXED_POSITION_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "scribe-mixed-positions-v1",
        fields: &MIXED_POSITION_FIELDS,
    };

    const UNSUPPORTED_ANALYZER_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
        id: 0,
        name: "cass_text",
        kind: FieldKind::Text {
            analyzer: AnalyzerKind::CassHyphenNormalize,
            positions: true,
        },
        stored: false,
    }];
    const UNSUPPORTED_ANALYZER_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "scribe-unsupported-analyzer-v1",
        fields: &UNSUPPORTED_ANALYZER_FIELDS,
    };

    #[derive(Debug, Default)]
    struct SamePositionAnalyzer;

    impl sealed::Sealed for SamePositionAnalyzer {}

    impl TokenAnalyzer for SamePositionAnalyzer {
        fn supports(&self, analyzer: AnalyzerKind) -> bool {
            analyzer == AnalyzerKind::FrankensearchDefault
        }

        fn analyze(
            &mut self,
            analyzer: AnalyzerKind,
            text: &str,
            sink: &mut dyn FnMut(&AnalyzedToken),
        ) {
            assert_eq!(analyzer, AnalyzerKind::FrankensearchDefault);
            for token_text in ["compound", "part"] {
                sink(&AnalyzedToken {
                    text: token_text.to_owned(),
                    position: 0,
                    offset_from: 0,
                    offset_to: text.len(),
                    position_length: 1,
                });
            }
        }
    }

    #[derive(Debug, Default)]
    struct MixedCassAnalyzer;

    impl sealed::Sealed for MixedCassAnalyzer {}

    impl TokenAnalyzer for MixedCassAnalyzer {
        fn supports(&self, analyzer: AnalyzerKind) -> bool {
            matches!(
                analyzer,
                AnalyzerKind::CassHyphenNormalize | AnalyzerKind::CassPrefixNormalize
            )
        }

        fn analyze(
            &mut self,
            analyzer: AnalyzerKind,
            text: &str,
            sink: &mut dyn FnMut(&AnalyzedToken),
        ) {
            let (token_text, position) = match analyzer {
                AnalyzerKind::CassHyphenNormalize => ("hyphen-dispatch", 7),
                AnalyzerKind::CassPrefixNormalize => ("prefix-dispatch", 11),
                AnalyzerKind::FrankensearchDefault => {
                    unreachable!("fixture does not advertise the default analyzer")
                }
            };
            sink(&AnalyzedToken {
                text: token_text.to_owned(),
                position,
                offset_from: 0,
                offset_to: text.len(),
                position_length: 1,
            });
        }
    }

    #[derive(Debug)]
    struct DeterministicRng(u64);

    impl DeterministicRng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn choose(&mut self, upper: usize) -> usize {
            usize::try_from(self.next() % u64::try_from(upper).expect("small choice bound"))
                .expect("bounded random choice fits usize")
        }
    }

    fn randomized_text(rng: &mut DeterministicRng) -> String {
        const WORDS: [&str; 8] = [
            "Alpha", "BETA", "Rust2024", "ÉCLAIR", "東京", "needle", "tail", "z9",
        ];
        const SEPARATORS: [&str; 5] = [" ", "-", "/", "...", "_"];
        let count = rng.choose(12);
        let mut text = String::new();
        for index in 0..count {
            if index != 0 {
                text.push_str(SEPARATORS[rng.choose(SEPARATORS.len())]);
            }
            text.push_str(WORDS[rng.choose(WORDS.len())]);
        }
        text
    }

    fn phrase_matches(field: &FieldTokenColumns, doc_ord: u32, phrase_term_ids: &[u32]) -> bool {
        let Some(positions) = field.positions() else {
            return false;
        };
        let Some(first_term) = phrase_term_ids.first() else {
            return false;
        };
        field
            .term_ids()
            .iter()
            .zip(field.doc_ords())
            .zip(positions)
            .filter(|((term_id, row_doc), _)| {
                u32::eq(*term_id, first_term) && u32::eq(*row_doc, &doc_ord)
            })
            .any(|(_, start_position)| {
                phrase_term_ids
                    .iter()
                    .enumerate()
                    .skip(1)
                    .all(|(offset, term_id)| {
                        let expected_position = start_position.saturating_add(
                            u32::try_from(offset).expect("test phrase length fits u32"),
                        );
                        field
                            .term_ids()
                            .iter()
                            .zip(field.doc_ords())
                            .zip(positions)
                            .any(|((row_term, row_doc), row_position)| {
                                u32::eq(row_term, term_id)
                                    && u32::eq(row_doc, &doc_ord)
                                    && u32::eq(row_position, &expected_position)
                            })
                    })
            })
    }

    fn analyzed_tokens(text: &str) -> Vec<AnalyzedToken> {
        let mut analyzer = FrankensearchTokenizer::default();
        let mut tokens = Vec::new();
        analyzer.analyze(AnalyzerKind::FrankensearchDefault, text, &mut |token| {
            tokens.push(token.clone());
        });
        tokens
    }

    fn incumbent_tokens(text: &str) -> Vec<AnalyzedToken> {
        let mut analyzer = frankensearch_lexical::default_tokenizer_for_bench();
        let mut stream = analyzer.token_stream(text);
        let mut tokens = Vec::new();
        while stream.advance() {
            let token = stream.token();
            tokens.push(AnalyzedToken {
                text: token.text.clone(),
                position: u32::try_from(token.position)
                    .expect("shipping tokenizer position fits Quill u32 contract"),
                offset_from: token.offset_from,
                offset_to: token.offset_to,
                position_length: token.position_length,
            });
        }
        tokens
    }

    fn fixture_input(case: &Value) -> String {
        if let Some(input) = case.get("input").and_then(Value::as_str) {
            return input.to_owned();
        }
        let generated = case
            .get("generated_input")
            .and_then(Value::as_object)
            .expect("analyzer fixture must contain input or generated_input");
        let repeat = generated
            .get("repeat")
            .and_then(Value::as_str)
            .expect("generated analyzer input must name repeat text");
        let count = ["count", "count_bytes", "count_chars"]
            .into_iter()
            .find_map(|key| generated.get(key).and_then(Value::as_u64))
            .expect("generated analyzer input must contain a repeat count");
        repeat.repeat(usize::try_from(count).expect("fixture repeat count fits usize"))
    }

    fn expected_fixture_tokens(case: &Value) -> Option<Vec<AnalyzedToken>> {
        case.get("expected_tokens")?.as_array().map(|tokens| {
            tokens
                .iter()
                .map(|token| AnalyzedToken {
                    text: token["text"]
                        .as_str()
                        .expect("fixture token text")
                        .to_owned(),
                    position: token["position"]
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .expect("fixture token position fits u32"),
                    offset_from: token["offset_from"]
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .expect("fixture token start fits usize"),
                    offset_to: token["offset_to"]
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .expect("fixture token end fits usize"),
                    position_length: token["position_length"]
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .expect("fixture token position length fits usize"),
                })
                .collect()
        })
    }

    #[test]
    fn scalar_reference_executes_fixture_and_matches_shipping_incumbent() {
        let fixture: Value = serde_json::from_str(LANGUAGE_CONTRACT_FIXTURE)
            .expect("language contract fixture is valid JSON");
        let cases = fixture["analyzer_cases"]
            .as_array()
            .expect("language contract analyzer_cases is an array");
        let mut executed = 0_u32;
        for case in cases {
            if case["analyzer"] != "frankensearch_default" {
                continue;
            }
            executed += 1;
            let case_id = case["id"].as_str().expect("analyzer case has an id");
            let input = fixture_input(case);
            let actual = analyzed_tokens(&input);
            assert_eq!(
                actual,
                incumbent_tokens(&input),
                "Quill token bytes diverged from the shipping incumbent for {case_id}"
            );
            if let Some(expected) = expected_fixture_tokens(case) {
                assert_eq!(actual, expected, "fixture golden diverged for {case_id}");
            }

            if let Some(admission) = case.get("token_admission").and_then(Value::as_str) {
                let mut analyzer = FrankensearchTokenizer::default();
                let mut admitted = Vec::new();
                let report = analyze_admitted(
                    &mut analyzer,
                    AnalyzerKind::FrankensearchDefault,
                    &input,
                    &mut |token| admitted.push(token.clone()),
                )
                .expect("default family supports the fixture analyzer");
                let kept = admission == "kept";
                assert!(kept || admission == "dropped", "{case_id}: {admission}");
                if kept {
                    assert_eq!(report.admitted_tokens, 1, "{case_id}");
                    assert_eq!(report.oversized_tokens, 0, "{case_id}");
                    assert_eq!(admitted.len(), 1, "{case_id}");
                } else {
                    assert_eq!(report.admitted_tokens, 0, "{case_id}");
                    assert_eq!(report.oversized_tokens, 1, "{case_id}");
                    assert!(admitted.is_empty(), "{case_id}");
                }
            }
        }
        assert_eq!(executed, 6, "all default-analyzer fixtures must execute");
    }

    #[test]
    #[should_panic(expected = "analyzed token position exceeds the u32 contract")]
    fn tokenizer_position_increment_cannot_saturate() {
        let _ = next_token_position(u32::MAX);
    }

    #[test]
    fn admission_drops_oversized_token_without_collapsing_position_gap() {
        let oversized = "X".repeat(MAX_TERM_BYTES + 1);
        let input = format!("head {oversized} tail");
        let mut analyzer = FrankensearchTokenizer::default();
        let mut admitted = Vec::new();
        let report = analyze_admitted(
            &mut analyzer,
            AnalyzerKind::FrankensearchDefault,
            &input,
            &mut |token| admitted.push((token.text.clone(), token.position)),
        )
        .expect("default family supports default analysis");
        assert_eq!(
            report,
            AnalysisReport {
                raw_tokens: 3,
                admitted_tokens: 2,
                oversized_tokens: 1,
            }
        );
        assert_eq!(
            admitted,
            vec![("head".to_owned(), 0), ("tail".to_owned(), 2)]
        );

        let maximum = "X".repeat(MAX_TERM_BYTES);
        let mut maximum_tokens = Vec::new();
        let maximum_report = analyze_admitted(
            &mut analyzer,
            AnalyzerKind::FrankensearchDefault,
            &maximum,
            &mut |token| maximum_tokens.push(token.text.len()),
        )
        .expect("default family supports default analysis");
        assert_eq!(maximum_report.admitted_tokens, 1);
        assert_eq!(maximum_report.oversized_tokens, 0);
        assert_eq!(maximum_tokens, vec![MAX_TERM_BYTES]);

        let lowercase_expands = "İ".repeat(MAX_TERM_BYTES / 3 + 1);
        assert!(lowercase_expands.len() < MAX_TERM_BYTES);
        let expansion_report = analyze_admitted(
            &mut analyzer,
            AnalyzerKind::FrankensearchDefault,
            &lowercase_expands,
            &mut |_| {},
        )
        .expect("default family supports default analysis");
        assert_eq!(expansion_report.raw_tokens, 1);
        assert_eq!(expansion_report.admitted_tokens, 0);
        assert_eq!(expansion_report.oversized_tokens, 1);
    }

    #[test]
    fn injected_analyzer_preserves_same_position_duplicate_tokens() {
        let mut accumulator =
            ColumnarAccumulator::with_analyzer(DEFAULT_SCHEMA, SamePositionAnalyzer)
                .expect("valid schema and injected analyzer");
        accumulator
            .add_document(0, &[IndexedFieldValue::new(1, "compound-part")])
            .expect("injected analysis accumulates");
        let content = accumulator.field(1).expect("content column");
        assert_eq!(content.positions(), Some([0, 0].as_slice()));
        assert_eq!(content.document_lengths(), &[2]);
        assert_ne!(content.term_ids()[0], content.term_ids()[1]);
        assert_eq!(
            accumulator.terms().field_and_term(content.term_ids()[0]),
            (1, b"compound".as_slice())
        );
        assert_eq!(
            accumulator.terms().field_and_term(content.term_ids()[1]),
            (1, b"part".as_slice())
        );
    }

    #[test]
    fn analyzer_family_dispatches_each_kind_in_a_mixed_schema() {
        let mut accumulator =
            ColumnarAccumulator::with_analyzer(CASS_SEMANTIC_SCHEMA, MixedCassAnalyzer)
                .expect("mixed CASS analyzer family covers the complete schema");
        accumulator
            .add_document(
                0,
                &[
                    IndexedFieldValue::new(6, "hyphen source"),
                    IndexedFieldValue::new(8, "prefix source"),
                ],
            )
            .expect("both analyzer kinds dispatch through one family");

        let hyphen = accumulator.field(6).expect("title column");
        let prefix = accumulator.field(8).expect("title_prefix column");
        assert_eq!(hyphen.positions(), Some([7].as_slice()));
        assert_eq!(prefix.positions(), None);
        assert_eq!(hyphen.document_lengths(), &[1]);
        assert_eq!(prefix.document_lengths(), &[1]);
        assert_eq!(
            accumulator.terms().field_and_term(hyphen.term_ids()[0]),
            (6, b"hyphen-dispatch".as_slice())
        );
        assert_eq!(
            accumulator.terms().field_and_term(prefix.term_ids()[0]),
            (8, b"prefix-dispatch".as_slice())
        );

        let mut default = FrankensearchTokenizer::default();
        assert_eq!(
            analyze_admitted(
                &mut default,
                AnalyzerKind::CassHyphenNormalize,
                "unsupported",
                &mut |_| {},
            ),
            Err(UnsupportedAnalysis {
                analyzer: AnalyzerKind::CassHyphenNormalize,
            })
        );
    }

    #[test]
    fn accumulator_columns_fieldnorms_and_namespacing_are_schema_driven() {
        let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
        let report = accumulator
            .add_document(
                3,
                &[
                    IndexedFieldValue::new(2, "Rust"),
                    IndexedFieldValue::new(0, ""),
                    IndexedFieldValue::new(1, "Rust POL-358"),
                ],
            )
            .expect("document accumulates");
        assert_eq!(report.admitted_tokens, 5);
        assert_eq!(report.oversized_tokens, 0);
        assert_eq!(accumulator.document_ords(), &[3]);
        assert_eq!(accumulator.token_count(), 5);

        let id = accumulator.field(0).expect("id column");
        assert_eq!(id.doc_ords(), &[3]);
        assert_eq!(id.positions(), None);
        assert_eq!(id.document_lengths(), &[1]);
        assert_eq!(id.fieldnorm_ids(), &[fieldnorm_to_id(1)]);

        let content = accumulator.field(1).expect("content column");
        assert_eq!(content.doc_ords(), &[3, 3, 3]);
        assert_eq!(content.positions(), Some([0, 1, 2].as_slice()));
        assert_eq!(content.document_lengths(), &[3]);
        assert_eq!(content.fieldnorm_ids(), &[fieldnorm_to_id(3)]);
        assert_eq!(content.total_tokens(), 3);

        let title = accumulator.field(2).expect("title column");
        assert_eq!(title.doc_ords(), &[3]);
        assert_eq!(title.positions(), Some([0].as_slice()));
        assert_eq!(title.document_lengths(), &[1]);

        let content_rust = content.term_ids()[0];
        let title_rust = title.term_ids()[0];
        assert_ne!(content_rust, title_rust);
        assert_eq!(
            accumulator.terms().field_and_term(content_rust),
            (1, b"rust".as_slice())
        );
        assert_eq!(
            accumulator.terms().field_and_term(title_rust),
            (2, b"rust".as_slice())
        );
        assert_eq!(
            accumulator.terms().field_and_term(id.term_ids()[0]),
            (0, b"".as_slice()),
            "an empty keyword is one legal raw token"
        );

        accumulator
            .add_document(7, &[IndexedFieldValue::new(1, "---")])
            .expect("gapped ordinal accumulates");
        assert_eq!(accumulator.document_ords(), &[3, 7]);
        for field in accumulator.fields() {
            assert_eq!(field.document_lengths().len(), 2);
            assert_eq!(field.fieldnorm_ids().len(), 2);
        }
        assert_eq!(
            accumulator.field(0).expect("id").document_lengths(),
            &[1, 0]
        );
        assert_eq!(
            accumulator.field(1).expect("content").document_lengths(),
            &[3, 0]
        );
        assert_eq!(
            accumulator.field(2).expect("title").document_lengths(),
            &[1, 0]
        );
    }

    #[test]
    fn positional_and_nonpositional_fields_have_exactly_their_schema_columns() {
        let mut accumulator =
            ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid schema");
        accumulator
            .add_document(
                11,
                &[
                    IndexedFieldValue::new(2, "same term"),
                    IndexedFieldValue::new(1, "same term"),
                    IndexedFieldValue::new(0, "key"),
                ],
            )
            .expect("document accumulates");

        let positioned = accumulator.field(1).expect("positioned field");
        let unpositioned = accumulator.field(2).expect("unpositioned field");
        assert_eq!(positioned.len(), 2);
        assert_eq!(unpositioned.len(), 2);
        assert_eq!(positioned.doc_ords(), unpositioned.doc_ords());
        assert_eq!(positioned.positions(), Some([0, 1].as_slice()));
        assert_eq!(unpositioned.positions(), None);
        assert_eq!(positioned.document_lengths(), &[2]);
        assert_eq!(unpositioned.document_lengths(), &[2]);
        assert_eq!(positioned.fieldnorm_ids(), unpositioned.fieldnorm_ids());
        assert!(accumulator.field(3).is_none(), "stored-only has no columns");
    }

    #[test]
    fn randomized_accumulation_preserves_all_parallel_column_invariants() {
        const SEEDS: [u64; 12] = [
            1,
            0x9e37_79b9_7f4a_7c15,
            0xd1b5_4a32_d192_ed03,
            0x94d0_49bb_1331_11eb,
            0x2545_f491_4f6c_dd1d,
            0x1234_5678_9abc_def0,
            0x0fed_cba9_8765_4321,
            0xa076_1d64_78bd_642f,
            0xe703_7ed1_a0b4_28db,
            0x8ebc_6af0_9c88_c6e3,
            0x5899_65cc_7537_4cc3,
            u64::MAX,
        ];

        for seed in SEEDS {
            let mut rng = DeterministicRng(seed);
            let mut accumulator =
                ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid schema");
            let mut expected_doc_ords = Vec::new();
            let mut expected_lengths = [Vec::new(), Vec::new(), Vec::new()];
            let mut next_doc_ord = u32::try_from(rng.choose(4)).expect("small ordinal");

            for _ in 0..96 {
                let keyword = (rng.choose(4) != 0).then(|| randomized_text(&mut rng));
                let positioned = (rng.choose(5) != 0).then(|| randomized_text(&mut rng));
                let unpositioned = (rng.choose(3) != 0).then(|| randomized_text(&mut rng));

                let mut owned_values = Vec::new();
                if let Some(text) = keyword {
                    expected_lengths[0].push(1);
                    owned_values.push((0_u16, text));
                } else {
                    expected_lengths[0].push(0);
                }
                if let Some(text) = positioned {
                    expected_lengths[1].push(
                        u32::try_from(analyzed_tokens(&text).len()).expect("small token count"),
                    );
                    owned_values.push((1_u16, text));
                } else {
                    expected_lengths[1].push(0);
                }
                if let Some(text) = unpositioned {
                    expected_lengths[2].push(
                        u32::try_from(analyzed_tokens(&text).len()).expect("small token count"),
                    );
                    owned_values.push((2_u16, text));
                } else {
                    expected_lengths[2].push(0);
                }
                if owned_values.len() > 1 {
                    let swap_with = rng.choose(owned_values.len());
                    owned_values.swap(0, swap_with);
                }
                let values: Vec<_> = owned_values
                    .iter()
                    .map(|(field_ord, text)| IndexedFieldValue::new(*field_ord, text))
                    .collect();

                expected_doc_ords.push(next_doc_ord);
                accumulator
                    .add_document(next_doc_ord, &values)
                    .expect("generated document accumulates");
                next_doc_ord += u32::try_from(rng.choose(3) + 1).expect("small ordinal step");
            }

            assert_eq!(
                accumulator.document_ords(),
                expected_doc_ords,
                "seed={seed:#x} document ordinals"
            );
            for (field_index, expected) in expected_lengths.iter().enumerate() {
                let field_ord = u16::try_from(field_index).expect("small field index");
                let field = accumulator
                    .field(field_ord)
                    .expect("generated field column");
                assert_eq!(
                    field.term_ids().len(),
                    field.doc_ords().len(),
                    "seed={seed:#x} field={field_ord} term/doc parallelism"
                );
                if let Some(positions) = field.positions() {
                    assert_eq!(
                        positions.len(),
                        field.term_ids().len(),
                        "seed={seed:#x} field={field_ord} position parallelism"
                    );
                }
                assert_eq!(
                    field.document_lengths(),
                    expected,
                    "seed={seed:#x} field={field_ord} document lengths"
                );
                let expected_norms: Vec<_> =
                    expected.iter().copied().map(fieldnorm_to_id).collect();
                assert_eq!(
                    field.fieldnorm_ids(),
                    expected_norms,
                    "seed={seed:#x} field={field_ord} fieldnorms"
                );
                assert_eq!(
                    field.total_tokens(),
                    expected.iter().copied().map(u64::from).sum::<u64>(),
                    "seed={seed:#x} field={field_ord} total tokens"
                );
                assert!(
                    field.doc_ords().windows(2).all(|pair| pair[0] <= pair[1]),
                    "seed={seed:#x} field={field_ord} row order"
                );

                for ((doc_ord, expected_length), doc_index) in
                    expected_doc_ords.iter().zip(expected).zip(0_usize..)
                {
                    let rows: Vec<_> = field
                        .doc_ords()
                        .iter()
                        .enumerate()
                        .filter_map(|(row, row_doc)| (row_doc == doc_ord).then_some(row))
                        .collect();
                    assert_eq!(
                        rows.len(),
                        usize::try_from(*expected_length).expect("generated length fits usize"),
                        "seed={seed:#x} field={field_ord} doc_index={doc_index}"
                    );
                    if let Some(positions) = field.positions() {
                        for (expected_position, row) in rows.into_iter().enumerate() {
                            assert_eq!(
                                positions[row],
                                u32::try_from(expected_position).expect("small generated position"),
                                "seed={seed:#x} field={field_ord} doc_index={doc_index}"
                            );
                        }
                    }
                }

                for term_id in field.term_ids() {
                    assert_eq!(
                        accumulator.terms().field_and_term(*term_id).0,
                        field_ord,
                        "seed={seed:#x} field={field_ord} term namespace"
                    );
                }
            }
            assert_eq!(
                accumulator.token_count(),
                expected_lengths
                    .iter()
                    .flatten()
                    .map(|length| usize::try_from(*length).expect("generated length fits usize"))
                    .sum::<usize>(),
                "seed={seed:#x} total token count"
            );
            tracing::info!(
                seed,
                documents = accumulator.document_count(),
                tokens = accumulator.token_count(),
                bytes_reserved = accumulator.bytes_reserved(),
                "Scribe randomized accumulation invariant case passed"
            );
        }
    }

    #[test]
    fn intern_roundtrip_and_dense_ids() {
        let mut interner = TermInterner::new();
        let a = interner.intern(0, b"alpha");
        let b = interner.intern(0, b"beta");
        let a2 = interner.intern(0, b"alpha");
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(a2, a, "re-intern must return the same id");
        assert_eq!(interner.len(), 2);
        assert_eq!(interner.field_and_term(a), (0, b"alpha".as_slice()));
        assert_eq!(interner.field_and_term(b), (0, b"beta".as_slice()));
    }

    #[test]
    fn same_term_different_field_is_distinct() {
        let mut interner = TermInterner::new();
        let content = interner.intern(0, b"rust");
        let title = interner.intern(1, b"rust");
        assert_ne!(content, title, "field namespacing must separate terms");
        assert_eq!(interner.field_and_term(content), (0, b"rust".as_slice()));
        assert_eq!(interner.field_and_term(title), (1, b"rust".as_slice()));
    }

    #[test]
    fn sorted_ids_match_composite_byte_order_and_field_grouping() {
        let mut interner = TermInterner::new();
        // Insert deliberately out of (field, term) order.
        let ids = [
            interner.intern(1, b"zebra"),
            interner.intern(0, b"zebra"),
            interner.intern(1, b"alpha"),
            interner.intern(0, b"alpha"),
            interner.intern(258, b"alpha"), // field 0x0102: BE prefix ordering test
        ];
        assert_eq!(ids.len(), 5);
        let sorted = interner.sorted_ids();
        let keys: Vec<Vec<u8>> = sorted
            .iter()
            .map(|id| interner.composite_key(*id).to_vec())
            .collect();
        let mut expect = keys.clone();
        expect.sort();
        assert_eq!(keys, expect, "sorted_ids must equal raw byte order");
        // Field grouping: all field-0 keys precede field-1, which precede 258.
        let fields: Vec<u16> = sorted
            .iter()
            .map(|id| interner.field_and_term(*id).0)
            .collect();
        assert_eq!(fields, vec![0, 0, 1, 1, 258]);
    }

    #[test]
    fn collision_path_verifies_bytes() {
        // Every key collides: correctness must come from byte verification.
        let mut interner: TermInterner<ConstBuild> =
            TermInterner::with_hasher(ConstBuild::default());
        let mut ids = Vec::new();
        for i in 0..200u32 {
            let term = format!("term-{i:03}");
            ids.push(interner.intern((i % 3) as u16, term.as_bytes()));
        }
        // Re-intern everything; ids must be stable.
        for i in 0..200u32 {
            let term = format!("term-{i:03}");
            assert_eq!(
                interner.intern((i % 3) as u16, term.as_bytes()),
                ids[i as usize]
            );
        }
        assert_eq!(interner.len(), 200);
        let sorted = interner.sorted_ids();
        assert_eq!(sorted.len(), 200);
        let keys: Vec<Vec<u8>> = sorted
            .iter()
            .map(|id| interner.composite_key(*id).to_vec())
            .collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]), "strict byte order");
    }

    #[test]
    fn budget_accounting_is_monotone_and_bounded() {
        let mut interner = TermInterner::new();
        let long = vec![b'x'; MAX_TERM_BYTES];
        interner.intern(0, &long);
        let after_long = interner.bytes_used();
        interner.intern(0, b"z");
        let mut prev = interner.bytes_used();
        assert!(
            prev >= after_long,
            "reusing scratch for a shorter key must not reduce live bytes"
        );
        let mut term_bytes_total = long.len() + b"z".len() + 2 * FIELD_PREFIX_BYTES;
        for i in 0..5000u32 {
            let term = format!("budget-term-{i}");
            term_bytes_total += term.len() + FIELD_PREFIX_BYTES;
            interner.intern(0, term.as_bytes());
            let now = interner.bytes_used();
            assert!(now >= prev, "bytes_used must be monotone under inserts");
            prev = now;
        }
        // Accounting must at least cover the raw key bytes and stay within an
        // order of magnitude (approximation contract documented on bytes_used;
        // arena reservation rounds up to chunk granularity).
        assert!(prev >= term_bytes_total, "must cover raw key bytes");
        assert!(
            prev <= term_bytes_total.max(DEFAULT_ARENA_CHUNK_BYTES) * 10,
            "accounting should not wildly overestimate: {prev} vs {term_bytes_total}"
        );
    }

    #[test]
    fn oversized_terms_get_dedicated_chunks_and_reset_drops_them() {
        let mut arena = ByteArena::with_chunk_size(4096);
        let big = vec![0xAB; 100_000];
        let span = arena.push(&big);
        assert_eq!(arena.resolve(span), big.as_slice());
        // Standard chunk still usable after the oversized insert.
        let small = arena.push(b"small");
        assert_eq!(arena.resolve(small), b"small");
        let chunks_with_big = arena.chunk_count();
        arena.reset();
        assert!(
            arena.chunk_count() < chunks_with_big,
            "reset must drop dedicated oversized chunks (RSS ratchet guard)"
        );
        assert_eq!(arena.bytes_used(), 0);
    }

    #[test]
    fn reset_reuse_is_allocation_stable_across_cycles() {
        // Soak: after the first cycle establishes capacity, later identical
        // cycles must not grow reserved bytes or chunk count (the no-leak /
        // RSS-stability contract). Logs per-cycle stats for diagnosis.
        let mut interner = TermInterner::new();
        let mut reserved_after_cycle = Vec::new();
        for cycle in 0..8 {
            for i in 0..20_000u32 {
                let term = format!("cycle-term-{i}");
                interner.intern((i % 4) as u16, term.as_bytes());
            }
            let (used, reserved, chunks) = interner.arena_stats();
            tracing::info!(
                cycle,
                used,
                reserved,
                chunks,
                terms = interner.len(),
                "soak cycle"
            );
            reserved_after_cycle.push((reserved, chunks));
            interner.reset();
            assert_eq!(interner.len(), 0);
            assert!(interner.is_empty());
        }
        let (first_reserved, first_chunks) = reserved_after_cycle[1];
        for (i, (reserved, chunks)) in reserved_after_cycle.iter().enumerate().skip(2) {
            assert_eq!(
                (*reserved, *chunks),
                (first_reserved, first_chunks),
                "cycle {i} changed arena footprint — allocation not stable"
            );
        }
    }

    #[test]
    fn empty_term_and_empty_interner_edges() {
        let mut interner = TermInterner::new();
        assert!(interner.is_empty());
        assert!(interner.sorted_ids().is_empty());
        // Empty term bytes are legal at this layer (the tokenizer never emits
        // them, but the interner must not corrupt on them).
        let id = interner.intern(7, b"");
        assert_eq!(interner.field_and_term(id), (7, b"".as_slice()));
        assert_eq!(interner.intern(7, b""), id);
    }

    #[test]
    fn fieldnorm_columns_retain_exact_lengths_and_quantized_boundaries() {
        assert_eq!(fieldnorm_to_id(40), 40);
        assert_eq!(fieldnorm_to_id(41), 40);
        assert_eq!(fieldnorm_to_id(42), 41);

        let counts = [0_u32, 1, 40, 41, 42, 100];
        let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
        for (doc_ord, count) in counts.into_iter().enumerate() {
            let text = "word ".repeat(count as usize);
            accumulator
                .add_document(
                    u32::try_from(doc_ord).expect("small doc ordinal"),
                    &[IndexedFieldValue::new(1, &text)],
                )
                .expect("document accumulates");
        }
        let content = accumulator.field(1).expect("content field");
        assert_eq!(content.document_lengths(), &counts);
        let expected_ids: Vec<u8> = counts.into_iter().map(fieldnorm_to_id).collect();
        assert_eq!(content.fieldnorm_ids(), expected_ids);
        assert_eq!(
            content.total_tokens(),
            counts.into_iter().map(u64::from).sum::<u64>()
        );
    }

    #[test]
    fn document_validation_is_atomic_and_ordinals_are_strictly_ascending() {
        let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
        assert!(matches!(
            accumulator.add_document(
                4,
                &[
                    IndexedFieldValue::new(1, "first"),
                    IndexedFieldValue::new(1, "duplicate"),
                ],
            ),
            Err(AccumulatorError::DuplicateField {
                doc_ord: 4,
                field_ord: 1
            })
        ));
        assert_eq!(accumulator.document_count(), 0);
        assert_eq!(accumulator.token_count(), 0);
        assert!(accumulator.terms().is_empty());

        assert!(matches!(
            accumulator.add_document(4, &[IndexedFieldValue::new(99, "unknown")]),
            Err(AccumulatorError::UnknownField { field_ord: 99 })
        ));
        assert!(matches!(
            accumulator.add_document(4, &[IndexedFieldValue::new(3, "stored")]),
            Err(AccumulatorError::NonStringField {
                field_ord: 3,
                field_name: "metadata_json"
            })
        ));
        assert_eq!(accumulator.document_count(), 0);

        accumulator
            .add_document(4, &[IndexedFieldValue::new(1, "valid")])
            .expect("first valid document");
        let token_count = accumulator.token_count();
        assert!(matches!(
            accumulator.add_document(4, &[]),
            Err(AccumulatorError::OutOfOrderDocument {
                previous: 4,
                current: 4
            })
        ));
        assert!(matches!(
            accumulator.add_document(3, &[]),
            Err(AccumulatorError::OutOfOrderDocument {
                previous: 4,
                current: 3
            })
        ));
        assert_eq!(accumulator.document_count(), 1);
        assert_eq!(accumulator.token_count(), token_count);

        assert!(matches!(
            ColumnarAccumulator::new(UNSUPPORTED_ANALYZER_SCHEMA),
            Err(QuillError::Resource {
                resource: "analyzer pipeline",
                detail,
            })
                if detail.contains("field 0 (cass_text)")
                    && detail.contains("CassHyphenNormalize")
        ));
    }

    #[test]
    fn lease_relative_document_ordinal_boundary_is_enforced_atomically() {
        let mut at_boundary = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
        at_boundary
            .add_document(
                DOC_ORDS_PER_LEASE - 1,
                &[IndexedFieldValue::new(1, "last legal document")],
            )
            .expect("last lease-relative ordinal is valid");
        assert_eq!(at_boundary.document_ords(), &[DOC_ORDS_PER_LEASE - 1]);

        let mut outside = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
        assert_eq!(
            outside.add_document(
                DOC_ORDS_PER_LEASE,
                &[IndexedFieldValue::new(1, "must not mutate")],
            ),
            Err(AccumulatorError::DocumentOutsideLease {
                doc_ord: DOC_ORDS_PER_LEASE,
            })
        );
        assert_eq!(outside.document_count(), 0);
        assert_eq!(outside.token_count(), 0);
        assert!(outside.terms().is_empty());
    }

    #[test]
    fn oversized_document_token_is_not_counted_but_keeps_following_position() {
        let oversized = "x".repeat(MAX_TERM_BYTES + 1);
        let text = format!("head {oversized} tail");
        let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
        let report = accumulator
            .add_document(0, &[IndexedFieldValue::new(1, &text)])
            .expect("document accumulates");
        assert_eq!(report.admitted_tokens, 2);
        assert_eq!(report.oversized_tokens, 1);
        let content = accumulator.field(1).expect("content field");
        assert_eq!(content.positions(), Some([0, 2].as_slice()));
        assert_eq!(content.document_lengths(), &[2]);
        assert_eq!(content.fieldnorm_ids(), &[fieldnorm_to_id(2)]);
        assert_eq!(content.total_tokens(), 2);
    }

    #[test]
    fn near_two_mib_document_preserves_u32_phrase_tail_positions() {
        const TARGET_BYTES: usize = 2 * 1024 * 1024;
        // Exactly 32 bytes: the 65,535 repeats put `needle` at position 65,535
        // and `tail` at 65,536, so a u16 position would wrap across the phrase.
        const FILLER: &str = "abcdefghijklmnopqrstuvwxyz01234 ";
        const TAIL: &str = "needle tail";
        let repeats = (TARGET_BYTES - TAIL.len()) / FILLER.len();
        let mut text = String::with_capacity(repeats * FILLER.len() + TAIL.len());
        for _ in 0..repeats {
            text.push_str(FILLER);
        }
        text.push_str(TAIL);
        assert!(text.len() <= TARGET_BYTES);
        assert!(TARGET_BYTES - text.len() < FILLER.len());
        assert_eq!(repeats, usize::from(u16::MAX));

        let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
        accumulator
            .add_document(17, &[IndexedFieldValue::new(1, &text)])
            .expect("large document accumulates");
        let content = accumulator.field(1).expect("content field");
        let positions = content.positions().expect("content stores positions");
        let expected_needle = u32::try_from(repeats).expect("fixture position fits u32");
        assert_eq!(
            &positions[positions.len() - 2..],
            &[expected_needle, expected_needle + 1]
        );
        assert_eq!(positions[positions.len() - 2], u32::from(u16::MAX));
        assert!(positions[positions.len() - 1] > u32::from(u16::MAX));
        assert_eq!(&content.doc_ords()[content.len() - 2..], &[17, 17]);
        let needle_id = content.term_ids()[content.len() - 2];
        let tail_id = content.term_ids()[content.len() - 1];
        assert_eq!(
            accumulator.terms().field_and_term(needle_id),
            (1, b"needle".as_slice())
        );
        assert_eq!(
            accumulator.terms().field_and_term(tail_id),
            (1, b"tail".as_slice())
        );
        assert_eq!(
            content.document_lengths(),
            &[u32::try_from(repeats + 2).expect("fixture count fits u32")]
        );
        assert!(
            phrase_matches(content, 17, &[needle_id, tail_id]),
            "tail phrase must match beyond the u16 position boundary"
        );
        assert!(
            !phrase_matches(content, 17, &[tail_id, needle_id]),
            "reversed tail phrase must not match"
        );
    }

    #[test]
    fn low_cardinality_columns_count_toward_budget_and_reset_reuses_capacity() {
        let text = "x ".repeat(20_000);
        let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
        let baseline = accumulator.bytes_reserved();
        let first = accumulator
            .add_document(0, &[IndexedFieldValue::new(1, &text)])
            .expect("first cycle");
        assert!(first.bytes_reserved > baseline + 20_000 * 8);
        assert!(accumulator.should_flush(first.bytes_used));
        assert!(!accumulator.should_flush(first.bytes_used + 1));
        let retained = first.bytes_reserved;
        let live = first.bytes_used;

        accumulator.reset();
        assert_eq!(accumulator.document_count(), 0);
        assert_eq!(accumulator.token_count(), 0);
        assert!(accumulator.terms().is_empty());
        for field in accumulator.fields() {
            assert!(field.document_lengths().is_empty());
            assert!(field.fieldnorm_ids().is_empty());
        }
        assert_eq!(accumulator.bytes_reserved(), retained);
        assert!(
            !accumulator.should_flush(live),
            "retained spare capacity must not force one-document flush cycles"
        );

        let second = accumulator
            .add_document(0, &[IndexedFieldValue::new(1, &text)])
            .expect("second cycle");
        assert_eq!(second.bytes_reserved, retained);
        assert_eq!(second.bytes_used, live);
        assert!(accumulator.should_flush(live));
    }

    #[test]
    fn shorter_final_token_cannot_reduce_accumulator_live_bytes() {
        let long = "x".repeat(MAX_TERM_BYTES);
        let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
        let after_long = accumulator
            .add_document(0, &[IndexedFieldValue::new(1, &long)])
            .expect("maximum admitted token accumulates")
            .bytes_used;
        let after_short = accumulator
            .add_document(1, &[IndexedFieldValue::new(1, "y")])
            .expect("shorter token accumulates")
            .bytes_used;
        assert!(
            after_short >= after_long,
            "reusable analyzer scratch must not make live accounting decrease"
        );
    }

    #[test]
    fn budget_includes_outer_field_column_allocation() {
        let accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
        let outer_fields = accumulator
            .fields
            .capacity()
            .saturating_mul(std::mem::size_of::<FieldTokenColumns>());
        assert!(outer_fields > 0);
        let exact_components = accumulator
            .terms
            .bytes_reserved()
            .saturating_add(outer_fields)
            .saturating_add(
                accumulator
                    .fields
                    .iter()
                    .map(FieldTokenColumns::bytes_reserved)
                    .sum::<usize>(),
            )
            .saturating_add(
                accumulator
                    .document_ords
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(accumulator.seen_fields.capacity().div_ceil(8))
            .saturating_add(accumulator.analyzer.bytes_reserved());
        assert_eq!(accumulator.bytes_reserved(), exact_components);
    }
}
