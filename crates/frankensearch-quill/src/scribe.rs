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
//! - [`CassAnalyzer`]: the native CASS hyphen/CJK analyzer family plus the
//!   matching edge-prefix and bounded-preview helpers.
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
use std::ops::Range;

use rayon::prelude::*;
use thiserror::Error;

use crate::contract::fieldnorm_to_id;
use crate::error::QuillError;
use crate::grimoire::{
    ByteSpan, EncodedTermDictionary, MAX_TERM_BYTES, TermDictionaryError, TermInput, TermMetadata,
    TermSectionLengths,
};
use crate::quiver::{
    BlockMaxError, DocLenCodecError, DocLenFieldInput, EncodedDocLenSection, EncodedIdHashSection,
    EncodedIdMapSection, EncodedNumericSection, EncodedPositionList, EncodedPostingList,
    EncodedStatsSection, EncodedStoredMetaSection, FieldStats, IdHashCodecError, IdMapCodecError,
    IdMapEntryInput, NumericCodecError, NumericValue, PositionCodecError, Posting, StatsCodecError,
    StoredMetaCodecError,
};
use crate::schema::{Analyzer as AnalyzerKind, FieldKind, SchemaDescriptor};
use crate::segment::{EncodedSegment, SectionInput, SectionKind, SegmentHeaderInput};

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

/// Maximum token length retained by the native CASS analyzer pipeline.
///
/// The shipping CASS analyzer retains exactly 256 UTF-8 bytes and drops 257.
/// This inclusive boundary intentionally differs from Tantivy's strict
/// `RemoveLongFilter::limit(256)` predicate and is much smaller than Quill's
/// global [`MAX_TERM_BYTES`] admission ceiling.
pub const CASS_MAX_TOKEN_BYTES: usize = 256;

/// Maximum Unicode-scalar prefix length generated for CASS prefix fields.
pub const CASS_MAX_EDGE_NGRAM_CHARS: usize = 20;

/// Native implementation of the two CASS analyzer pipelines.
///
/// [`AnalyzerKind::CassHyphenNormalize`] applies the shipping
/// `CassTokenizer -> HyphenDecompose -> CjkBigramDecompose ->
/// CassNormalizeAndLimit` stages. [`AnalyzerKind::CassPrefixNormalize`] omits
/// only hyphen decomposition. One scratch token is reused for the complete
/// stream; compound alternatives, parts, and CJK bigrams are sent directly to
/// the callback without a staging allocation.
#[derive(Debug, Clone, Default)]
pub struct CassAnalyzer {
    token: AnalyzedToken,
}

impl sealed::Sealed for CassAnalyzer {}

/// Exact CJK ranges recognized by the incumbent CASS tokenizer.
///
/// Keep this predicate shared inside Quill: broadening it to later Unicode
/// extensions would change durable term bytes and therefore requires an
/// explicit language-contract revision.
#[inline]
pub(crate) fn is_cass_cjk(ch: char) -> bool {
    matches!(
        ch,
        '\u{4E00}'..='\u{9FFF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{3040}'..='\u{309F}'
            | '\u{30A0}'..='\u{30FF}'
            | '\u{AC00}'..='\u{D7AF}'
            | '\u{3100}'..='\u{312F}'
            | '\u{3300}'..='\u{33FF}'
            | '\u{F900}'..='\u{FAFF}'
            | '\u{20000}'..='\u{2A6DF}'
    )
}

fn cass_ascii_token_end(text: &str, mut cursor: usize) -> usize {
    let mut end = cursor;
    let mut last_was_ascii_alphanumeric = false;

    while let Some((ch, next_cursor)) = tokenizer_next_char(text, cursor) {
        if ch.is_ascii_alphanumeric() {
            end = next_cursor;
            cursor = next_cursor;
            last_was_ascii_alphanumeric = true;
            continue;
        }
        if ch == '-'
            && last_was_ascii_alphanumeric
            && let Some((next_ch, _)) = tokenizer_next_char(text, next_cursor)
            && next_ch.is_ascii_alphanumeric()
        {
            end = next_cursor;
            cursor = next_cursor;
            last_was_ascii_alphanumeric = false;
            continue;
        }
        break;
    }
    end
}

fn cass_cjk_token_end(text: &str, mut cursor: usize) -> usize {
    let mut end = cursor;
    while let Some((ch, next_cursor)) = tokenizer_next_char(text, cursor) {
        if !is_cass_cjk(ch) {
            break;
        }
        end = next_cursor;
        cursor = next_cursor;
    }
    end
}

impl CassAnalyzer {
    fn emit_normalized(
        &mut self,
        source: &str,
        position: u32,
        offset_from: usize,
        offset_to: usize,
        sink: &mut dyn FnMut(&AnalyzedToken),
    ) {
        if source.len() > CASS_MAX_TOKEN_BYTES {
            return;
        }
        self.token.text.clear();
        self.token.text.push_str(source);
        self.token.text.make_ascii_lowercase();
        self.token.position = position;
        self.token.offset_from = offset_from;
        self.token.offset_to = offset_to;
        self.token.position_length = 1;
        sink(&self.token);
    }

    fn emit_cjk(
        &mut self,
        source: &str,
        position: u32,
        offset_from: usize,
        offset_to: usize,
        sink: &mut dyn FnMut(&AnalyzedToken),
    ) {
        let mut chars = source.chars();
        let Some(mut left) = chars.next() else {
            return;
        };
        let Some(mut right) = chars.next() else {
            self.emit_normalized(source, position, offset_from, offset_to, sink);
            return;
        };

        loop {
            self.token.text.clear();
            self.token.text.push(left);
            self.token.text.push(right);
            self.token.position = position;
            self.token.offset_from = offset_from;
            self.token.offset_to = offset_to;
            self.token.position_length = 1;
            sink(&self.token);

            left = right;
            let Some(next) = chars.next() else {
                break;
            };
            right = next;
        }
    }

    fn emit_ascii(
        &mut self,
        analyzer: AnalyzerKind,
        source: &str,
        position: u32,
        offset_from: usize,
        offset_to: usize,
        sink: &mut dyn FnMut(&AnalyzedToken),
    ) {
        self.emit_normalized(source, position, offset_from, offset_to, sink);
        if analyzer != AnalyzerKind::CassHyphenNormalize || !source.contains('-') {
            return;
        }
        for part in source.split('-').filter(|part| !part.is_empty()) {
            self.emit_normalized(part, position, offset_from, offset_to, sink);
        }
    }
}

impl TokenAnalyzer for CassAnalyzer {
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
        debug_assert!(self.supports(analyzer));
        let mut cursor = 0;
        let mut position = 0_u32;

        while let Some((ch, next_cursor)) = tokenizer_next_char(text, cursor) {
            let (offset_to, is_cjk) = if ch.is_ascii_alphanumeric() {
                (cass_ascii_token_end(text, cursor), false)
            } else if is_cass_cjk(ch) {
                (cass_cjk_token_end(text, next_cursor), true)
            } else {
                cursor = next_cursor;
                continue;
            };
            let source = &text[cursor..offset_to];
            if is_cjk {
                self.emit_cjk(source, position, cursor, offset_to, sink);
            } else {
                self.emit_ascii(analyzer, source, position, cursor, offset_to, sink);
            }
            position = next_token_position(position);
            cursor = offset_to;
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

fn push_cass_prefix(out: &mut String, prefix: &str) {
    if !out.is_empty() {
        out.push(' ');
    }
    out.push_str(prefix);
}

/// Generate the shipping CASS edge-prefix field value.
///
/// Every alphanumeric word contributes prefixes of 2 through 20 Unicode
/// scalar values, in word and prefix-length order, separated by one ASCII
/// space. Source case is preserved; the prefix analyzer lowercases later.
#[must_use]
pub fn cass_generate_edge_ngrams(text: &str) -> String {
    const MAX_BOUNDARIES: usize = CASS_MAX_EDGE_NGRAM_CHARS + 1;
    let mut prefixes = String::with_capacity(text.len().saturating_mul(2));
    for word in text.split(|ch: char| !ch.is_alphanumeric()) {
        if word.is_ascii() {
            let upper = word.len().min(CASS_MAX_EDGE_NGRAM_CHARS);
            for end in 2..=upper {
                push_cass_prefix(&mut prefixes, &word[..end]);
            }
            continue;
        }

        let mut boundaries = [0_usize; MAX_BOUNDARIES];
        let mut boundary_count = 0;
        for (byte_index, _) in word.char_indices() {
            if boundary_count == MAX_BOUNDARIES {
                break;
            }
            boundaries[boundary_count] = byte_index;
            boundary_count += 1;
        }
        if boundary_count < MAX_BOUNDARIES {
            boundaries[boundary_count] = word.len();
            boundary_count += 1;
        }
        if boundary_count < 3 {
            continue;
        }
        for &end in &boundaries[2..boundary_count] {
            push_cass_prefix(&mut prefixes, &word[..end]);
        }
    }
    prefixes
}

/// Return a Unicode-scalar-bounded CASS preview.
///
/// The first `max_chars` scalar values are copied byte-for-byte and `…` is
/// appended exactly when additional input remains.
#[must_use]
pub fn cass_build_preview(content: &str, max_chars: usize) -> String {
    let mut cut = content.len();
    for (count, (byte_index, _)) in content.char_indices().enumerate() {
        if count == max_chars {
            cut = byte_index;
            break;
        }
    }
    let truncated = cut < content.len();
    let mut preview = String::with_capacity(cut + if truncated { '…'.len_utf8() } else { 0 });
    preview.push_str(&content[..cut]);
    if truncated {
        preview.push('…');
    }
    preview
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

/// One opaque stored-field value supplied for a document.
///
/// Values are retained byte-for-byte for FSLX STOREDMETA. Indexed string
/// fields marked `stored` normally derive these bytes directly from their
/// [`IndexedFieldValue`]; this explicit form supplies stored-only and numeric
/// fields, or a stored indexed field that is intentionally not indexed for one
/// document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredFieldValue<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Borrowed opaque bytes.
    pub bytes: &'a [u8],
}

impl<'a> StoredFieldValue<'a> {
    /// Construct a borrowed stored-field value.
    #[must_use]
    pub const fn new(field_ord: u16, bytes: &'a [u8]) -> Self {
        Self { field_ord, bytes }
    }
}

/// One schema-typed indexed numeric value supplied for a document.
///
/// Numeric values are retained in a dedicated typed column for NUMERIC. When
/// the descriptor also sets `stored=true`, Scribe writes the same canonical
/// eight little-endian bytes into STOREDMETA automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexedNumericValue {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Signed or unsigned value matching the field descriptor.
    pub value: NumericValue,
}

impl IndexedNumericValue {
    /// Construct one signed indexed value.
    #[must_use]
    pub const fn i64(field_ord: u16, value: i64) -> Self {
        Self {
            field_ord,
            value: NumericValue::I64(value),
        }
    }

    /// Construct one unsigned indexed value.
    #[must_use]
    pub const fn u64(field_ord: u16, value: u64) -> Self {
        Self {
            field_ord,
            value: NumericValue::U64(value),
        }
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
    /// A typed numeric value named a string or stored-only field.
    #[error("field {field_ord} ({field_name}) is not numeric")]
    NonNumericField {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
    },
    /// A typed numeric value named a non-indexed numeric field.
    #[error("numeric field {field_ord} ({field_name}) is not indexed")]
    NonIndexedNumericField {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
    },
    /// An in-memory numeric tag disagreed with the schema descriptor.
    #[error("numeric field {field_ord} ({field_name}) expects {expected}, got {actual}")]
    NumericTypeMismatch {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
        /// Schema type name.
        expected: &'static str,
        /// Supplied type name.
        actual: &'static str,
    },
    /// Opaque bytes used as a numeric value must be exactly one scalar wide.
    #[error(
        "numeric field {field_ord} ({field_name}) has {actual} stored bytes, expected {expected}"
    )]
    InvalidNumericBytes {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
        /// Required byte width.
        expected: usize,
        /// Supplied byte width.
        actual: usize,
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
    /// A value named a field whose descriptor does not store original bytes.
    #[error("field {field_ord} ({field_name}) is not stored")]
    NonStoredField {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Stable schema field name.
        field_name: &'static str,
    },
    /// One stored value cannot be represented by the durable u32 offsets.
    #[error("stored field {field_ord} value is {bytes} bytes and exceeds the u32 offset domain")]
    StoredValueTooLarge {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Rejected byte length.
        bytes: usize,
    },
    /// Appending a value would make one field blob exceed the durable u32 domain.
    #[error(
        "stored field {field_ord} blob cannot append {appended} bytes to its current {current} bytes"
    )]
    StoredBlobTooLarge {
        /// Rejected field ordinal.
        field_ord: u16,
        /// Current accumulated blob length.
        current: usize,
        /// Candidate value length.
        appended: usize,
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

/// One schema-ordered column of opaque stored values.
///
/// Offsets align with completed documents rather than lease ordinals. A
/// separate presence byte preserves the distinction between an absent value
/// and a present empty byte string; the FSLX writer packs those bytes into the
/// STOREDMETA presence bitmap.
#[derive(Debug)]
pub struct StoredFieldColumns {
    field_ord: u16,
    offsets: Vec<u32>,
    present: Vec<u8>,
    blob: Vec<u8>,
}

impl StoredFieldColumns {
    fn new(field_ord: u16) -> Self {
        Self {
            field_ord,
            offsets: vec![0],
            present: Vec::new(),
            blob: Vec::new(),
        }
    }

    /// Stable schema field ordinal.
    #[must_use]
    pub const fn field_ord(&self) -> u16 {
        self.field_ord
    }

    /// Number of completed documents represented by this column.
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.present.len()
    }

    /// Monotone blob offsets, with exactly `document_count + 1` entries.
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// One canonical `0`/`1` presence byte per completed document.
    #[must_use]
    pub fn presence(&self) -> &[u8] {
        &self.present
    }

    /// Concatenated opaque value bytes.
    #[must_use]
    pub fn blob(&self) -> &[u8] {
        &self.blob
    }

    /// Borrow one value by completed-document index.
    ///
    /// `None` means the document index is out of range or the field was
    /// absent. `Some(&[])` is a present empty value.
    #[must_use]
    pub fn value(&self, document_index: usize) -> Option<&[u8]> {
        if self.present.get(document_index).copied()? == 0 {
            return None;
        }
        let start = usize::try_from(*self.offsets.get(document_index)?).ok()?;
        let end = usize::try_from(*self.offsets.get(document_index + 1)?).ok()?;
        self.blob.get(start..end)
    }

    fn can_append_len(&self, value_len: Option<usize>) -> bool {
        value_len.is_none_or(|bytes| {
            self.blob
                .len()
                .checked_add(bytes)
                .is_some_and(|len| u32::try_from(len).is_ok())
        })
    }

    fn append_document(&mut self, value: Option<&[u8]>) {
        self.present.push(u8::from(value.is_some()));
        if let Some(bytes) = value {
            self.blob.extend_from_slice(bytes);
        }
        self.offsets.push(
            u32::try_from(self.blob.len())
                .expect("stored blob append was validated before accumulator mutation"),
        );
    }

    fn bytes_reserved(&self) -> usize {
        self.offsets
            .capacity()
            .saturating_mul(std::mem::size_of::<u32>())
            .saturating_add(self.present.capacity())
            .saturating_add(self.blob.capacity())
    }

    fn bytes_used(&self) -> usize {
        self.offsets
            .len()
            .saturating_mul(std::mem::size_of::<u32>())
            .saturating_add(self.present.len())
            .saturating_add(self.blob.len())
    }

    fn reset(&mut self) {
        self.offsets.clear();
        self.offsets.push(0);
        self.present.clear();
        self.blob.clear();
    }
}

/// One schema-ordered indexed numeric column.
///
/// Values align with completed documents, not sparse lease ordinals. `None`
/// represents an absent optional value or a segment-range hole introduced when
/// Scribe seals sparse local ordinals.
#[derive(Debug)]
pub struct NumericFieldColumns {
    field_ord: u16,
    values: Vec<Option<NumericValue>>,
}

impl NumericFieldColumns {
    fn new(field_ord: u16) -> Self {
        Self {
            field_ord,
            values: Vec::new(),
        }
    }

    /// Stable schema field ordinal.
    #[must_use]
    pub const fn field_ord(&self) -> u16 {
        self.field_ord
    }

    /// One optional typed value per completed document.
    #[must_use]
    pub fn values(&self) -> &[Option<NumericValue>] {
        &self.values
    }

    fn append_document(&mut self, value: Option<NumericValue>) {
        self.values.push(value);
    }

    fn bytes_reserved(&self) -> usize {
        self.values
            .capacity()
            .saturating_mul(std::mem::size_of::<Option<NumericValue>>())
    }

    fn bytes_used(&self) -> usize {
        self.values
            .len()
            .saturating_mul(std::mem::size_of::<Option<NumericValue>>())
    }

    fn reset(&mut self) {
        self.values.clear();
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
    numeric_fields: Vec<NumericFieldColumns>,
    stored_fields: Vec<StoredFieldColumns>,
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
        let numeric_fields = schema
            .fields
            .iter()
            .filter_map(|field| match field.kind {
                FieldKind::I64 { indexed: true, .. } | FieldKind::U64 { indexed: true, .. } => {
                    Some(NumericFieldColumns::new(field.id))
                }
                _ => None,
            })
            .collect();
        let stored_fields = schema
            .fields
            .iter()
            .filter(|field| field.stored)
            .map(|field| StoredFieldColumns::new(field.id))
            .collect();
        Ok(Self {
            schema,
            terms: TermInterner::new(),
            fields,
            numeric_fields,
            stored_fields,
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
        self.add_document_with_values(doc_ord, values, &[], &[])
    }

    /// Add one complete document with explicit opaque stored values.
    ///
    /// Indexed string values whose descriptors set `stored=true` are retained
    /// automatically. `stored_values` supplies stored-only/numeric fields, or
    /// a stored indexed field omitted from `values`. Naming the same field in
    /// both slices is rejected rather than admitting divergent indexed and
    /// stored bytes.
    ///
    /// # Errors
    ///
    /// Returns a typed validation error before mutating any column when the
    /// document ordinal, either field set, field shape, analyzer, source-size
    /// proof, or durable stored-blob bound is invalid.
    pub fn add_document_with_stored(
        &mut self,
        doc_ord: u32,
        values: &[IndexedFieldValue<'_>],
        stored_values: &[StoredFieldValue<'_>],
    ) -> Result<DocumentAccumulation, AccumulatorError> {
        self.add_document_with_values(doc_ord, values, &[], stored_values)
    }

    /// Add one complete document with typed indexed numeric values.
    ///
    /// Numeric fields marked `stored` automatically retain the exact canonical
    /// little-endian scalar bytes. Missing numeric fields remain absent.
    ///
    /// # Errors
    ///
    /// Returns a typed validation error before mutating any column when the
    /// document ordinal, field set, numeric type, analyzer, or durable bounds
    /// are invalid.
    pub fn add_document_with_numeric(
        &mut self,
        doc_ord: u32,
        values: &[IndexedFieldValue<'_>],
        numeric_values: &[IndexedNumericValue],
    ) -> Result<DocumentAccumulation, AccumulatorError> {
        self.add_document_with_values(doc_ord, values, numeric_values, &[])
    }

    /// Add one complete document with typed numeric and opaque stored values.
    ///
    /// A field may appear in only one input slice. For a stored indexed numeric
    /// field, prefer `numeric_values`: Scribe derives its canonical stored
    /// bytes automatically. `stored_values` remains accepted for such a field
    /// when the bytes are exactly one schema-typed little-endian scalar.
    ///
    /// # Errors
    ///
    /// Returns a typed validation error before mutating any column when the
    /// document ordinal, any field set, numeric type/width, analyzer, source
    /// size, or durable stored-blob bound is invalid.
    pub fn add_document_with_values(
        &mut self,
        doc_ord: u32,
        values: &[IndexedFieldValue<'_>],
        numeric_values: &[IndexedNumericValue],
        stored_values: &[StoredFieldValue<'_>],
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

        for value in numeric_values {
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
            match (field.kind, value.value) {
                (FieldKind::I64 { indexed: true, .. }, NumericValue::I64(_))
                | (FieldKind::U64 { indexed: true, .. }, NumericValue::U64(_)) => {}
                (
                    FieldKind::I64 { indexed: false, .. } | FieldKind::U64 { indexed: false, .. },
                    _,
                ) => {
                    return Err(AccumulatorError::NonIndexedNumericField {
                        field_ord: field.id,
                        field_name: field.name,
                    });
                }
                (FieldKind::I64 { .. }, actual) => {
                    return Err(AccumulatorError::NumericTypeMismatch {
                        field_ord: field.id,
                        field_name: field.name,
                        expected: "i64",
                        actual: numeric_value_type_name(actual),
                    });
                }
                (FieldKind::U64 { .. }, actual) => {
                    return Err(AccumulatorError::NumericTypeMismatch {
                        field_ord: field.id,
                        field_name: field.name,
                        expected: "u64",
                        actual: numeric_value_type_name(actual),
                    });
                }
                _ => {
                    return Err(AccumulatorError::NonNumericField {
                        field_ord: field.id,
                        field_name: field.name,
                    });
                }
            }
        }

        for value in stored_values {
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
            if !field.stored {
                return Err(AccumulatorError::NonStoredField {
                    field_ord: field.id,
                    field_name: field.name,
                });
            }
            if matches!(field.kind, FieldKind::I64 { .. } | FieldKind::U64 { .. })
                && value.bytes.len() != std::mem::size_of::<u64>()
            {
                return Err(AccumulatorError::InvalidNumericBytes {
                    field_ord: field.id,
                    field_name: field.name,
                    expected: std::mem::size_of::<u64>(),
                    actual: value.bytes.len(),
                });
            }
            if u32::try_from(value.bytes.len()).is_err() {
                return Err(AccumulatorError::StoredValueTooLarge {
                    field_ord: field.id,
                    bytes: value.bytes.len(),
                });
            }
        }

        for field in &self.stored_fields {
            let value_len = stored_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
                .map(|value| value.bytes.len())
                .or_else(|| {
                    values
                        .iter()
                        .find(|value| value.field_ord == field.field_ord)
                        .map(|value| value.text.len())
                })
                .or_else(|| {
                    numeric_values
                        .iter()
                        .find(|value| value.field_ord == field.field_ord)
                        .map(|_| std::mem::size_of::<u64>())
                });
            if !field.can_append_len(value_len) {
                return Err(AccumulatorError::StoredBlobTooLarge {
                    field_ord: field.field_ord,
                    current: field.blob.len(),
                    appended: value_len.unwrap_or(0),
                });
            }
        }

        let resolved_numeric_values = self
            .numeric_fields
            .iter()
            .map(|field| {
                numeric_values
                    .iter()
                    .find(|value| value.field_ord == field.field_ord)
                    .map(|value| value.value)
                    .or_else(|| {
                        stored_values
                            .iter()
                            .find(|value| value.field_ord == field.field_ord)
                            .and_then(|value| {
                                numeric_value_from_le_bytes(
                                    self.schema.fields[usize::from(field.field_ord)].kind,
                                    value.bytes,
                                )
                            })
                    })
            })
            .collect::<Vec<_>>();

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

        for (field, value) in self.numeric_fields.iter_mut().zip(resolved_numeric_values) {
            field.append_document(value);
        }

        for field in &mut self.stored_fields {
            if let Some(value) = numeric_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
            {
                let bytes = value.value.to_le_bytes();
                field.append_document(Some(&bytes));
                continue;
            }
            let value = stored_values
                .iter()
                .find(|value| value.field_ord == field.field_ord)
                .map(|value| value.bytes)
                .or_else(|| {
                    values
                        .iter()
                        .find(|value| value.field_ord == field.field_ord)
                        .map(|value| value.text.as_bytes())
                });
            field.append_document(value);
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

    /// Indexed numeric field columns in stable schema order.
    #[must_use]
    pub fn numeric_fields(&self) -> &[NumericFieldColumns] {
        &self.numeric_fields
    }

    /// Look up one indexed numeric field's typed column.
    #[must_use]
    pub fn numeric_field(&self, field_ord: u16) -> Option<&NumericFieldColumns> {
        self.numeric_fields
            .binary_search_by_key(&field_ord, NumericFieldColumns::field_ord)
            .ok()
            .map(|index| &self.numeric_fields[index])
    }

    /// Stored-field columns in stable schema order.
    #[must_use]
    pub fn stored_fields(&self) -> &[StoredFieldColumns] {
        &self.stored_fields
    }

    /// Look up one stored-field column.
    #[must_use]
    pub fn stored_field(&self, field_ord: u16) -> Option<&StoredFieldColumns> {
        self.stored_fields
            .binary_search_by_key(&field_ord, StoredFieldColumns::field_ord)
            .ok()
            .map(|index| &self.stored_fields[index])
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
                self.numeric_fields
                    .capacity()
                    .saturating_mul(std::mem::size_of::<NumericFieldColumns>()),
            )
            .saturating_add(
                self.numeric_fields
                    .iter()
                    .map(NumericFieldColumns::bytes_reserved)
                    .fold(0, usize::saturating_add),
            )
            .saturating_add(
                self.stored_fields
                    .capacity()
                    .saturating_mul(std::mem::size_of::<StoredFieldColumns>()),
            )
            .saturating_add(
                self.stored_fields
                    .iter()
                    .map(StoredFieldColumns::bytes_reserved)
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
                self.numeric_fields
                    .len()
                    .saturating_mul(std::mem::size_of::<NumericFieldColumns>()),
            )
            .saturating_add(
                self.numeric_fields
                    .iter()
                    .map(NumericFieldColumns::bytes_used)
                    .fold(0, usize::saturating_add),
            )
            .saturating_add(
                self.stored_fields
                    .len()
                    .saturating_mul(std::mem::size_of::<StoredFieldColumns>()),
            )
            .saturating_add(
                self.stored_fields
                    .iter()
                    .map(StoredFieldColumns::bytes_used)
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
        for field in &mut self.numeric_fields {
            field.reset();
        }
        for field in &mut self.stored_fields {
            field.reset();
        }
        self.document_ords.clear();
        self.seen_fields.fill(false);
        self.last_doc_ord = None;
        self.analyzer.reset();
    }
}

fn numeric_value_type_name(value: NumericValue) -> &'static str {
    match value {
        NumericValue::I64(_) => "i64",
        NumericValue::U64(_) => "u64",
    }
}

fn numeric_value_from_le_bytes(kind: FieldKind, bytes: &[u8]) -> Option<NumericValue> {
    let bytes: [u8; 8] = bytes.try_into().ok()?;
    match kind {
        FieldKind::I64 { .. } => Some(NumericValue::I64(i64::from_le_bytes(bytes))),
        FieldKind::U64 { .. } => Some(NumericValue::U64(u64::from_le_bytes(bytes))),
        FieldKind::Keyword | FieldKind::Text { .. } | FieldKind::StoredOnly => None,
    }
}

/// One external-identity row supplied when an accumulator is sealed.
///
/// Scribe deliberately keeps document identifiers out of its token columns.
/// The ordinal makes the sidecar self-checking: positional alignment alone
/// cannot silently attach an identifier to the wrong accumulated document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlushDocumentInput<'a> {
    /// Shard-lease-relative document ordinal.
    pub doc_ord: u32,
    /// Non-empty external document identifier.
    pub document_id: &'a str,
    /// Unseeded xxh3-64 witness of canonical document content.
    pub content_hash: u64,
}

impl<'a> FlushDocumentInput<'a> {
    /// Construct a row from an externally computed content witness.
    #[must_use]
    pub const fn new(doc_ord: u32, document_id: &'a str, content_hash: u64) -> Self {
        Self {
            doc_ord,
            document_id,
            content_hash,
        }
    }

    /// Construct a row while hashing canonical document content.
    #[must_use]
    pub fn from_canonical_content(
        doc_ord: u32,
        document_id: &'a str,
        canonical_content: &[u8],
    ) -> Self {
        let entry = IdMapEntryInput::from_canonical_content(document_id, canonical_content);
        Self::new(doc_ord, entry.document_id, entry.content_hash)
    }
}

/// Fixed metadata and identity sidecar for one deterministic accumulator seal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlushSegmentInput<'a> {
    /// Collision-checked immutable segment identifier.
    pub segment_id: u64,
    /// Global document ID corresponding to shard-local ordinal zero.
    pub lease_docid_base: u64,
    /// Informational creation timestamp persisted in the segment header.
    pub created_unix_s: i64,
    /// Packed engine version persisted in the segment header.
    pub engine_version: u32,
    /// One identity row per completed accumulator document, in the same order.
    pub documents: &'a [FlushDocumentInput<'a>],
}

/// Typed failures from [`flush_accumulator`].
#[derive(Debug, Error)]
pub enum FlushError {
    /// Empty immutable segments are not produced by the shard flush path.
    #[error("cannot flush an empty Scribe accumulator")]
    EmptyAccumulator,
    /// Q1 R1 requires every shard lease to begin on its 65,536-doc boundary.
    #[error(
        "lease document base {lease_docid_base} is not aligned to {DOC_ORDS_PER_LEASE} documents"
    )]
    MisalignedLeaseBase {
        /// Rejected global lease base.
        lease_docid_base: u64,
    },
    /// The external identity sidecar must be one-for-one with completed docs.
    #[error("flush identity sidecar has {actual} rows, expected {expected}")]
    DocumentCountMismatch {
        /// Completed accumulator document count.
        expected: usize,
        /// Supplied sidecar row count.
        actual: usize,
    },
    /// An identity row drifted from its accumulator document.
    #[error(
        "flush identity row {index} names local ordinal {actual}, expected accumulator ordinal {expected}"
    )]
    DocumentOrdinalMismatch {
        /// Completed-document index.
        index: usize,
        /// Accumulator ordinal at this index.
        expected: u32,
        /// Sidecar ordinal at this index.
        actual: u32,
    },
    /// Rebase would leave the FSLX u32 global-document domain.
    #[error(
        "local document ordinal {doc_ord} rebased from lease base {lease_docid_base} exceeds u32"
    )]
    DocumentIdOverflow {
        /// Supplied lease base.
        lease_docid_base: u64,
        /// Rejected local ordinal.
        doc_ord: u32,
    },
    /// A private accumulator column invariant failed before durable bytes existed.
    #[error("invalid token column for field {field_ord}: {detail}")]
    InvalidTokenColumn {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Stable invariant name.
        detail: &'static str,
    },
    /// A token row referenced no term in the accumulator interner.
    #[error("field {field_ord} token row references out-of-range local term id {term_id}")]
    TermIdOutOfRange {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Rejected local term id.
        term_id: u32,
    },
    /// Every interned term must own at least one admitted token row.
    #[error("interned local term id {term_id} has no token rows")]
    TermHasNoRows {
        /// Orphaned local term id.
        term_id: u32,
    },
    /// Stable scatter relies on per-field source rows ascending by document.
    #[error("term {term_id} rows descend at partition index {index}: {previous} then {current}")]
    NonAscendingTermDocuments {
        /// Dense local term id.
        term_id: u32,
        /// Index within the term partition.
        index: usize,
        /// Prior local document ordinal.
        previous: u32,
        /// Current local document ordinal.
        current: u32,
    },
    /// One term frequency exceeded the durable u32 field.
    #[error("term {term_id} frequency for local document {doc_ord} exceeds u32")]
    TermFrequencyOverflow {
        /// Dense local term id.
        term_id: u32,
        /// Local document ordinal.
        doc_ord: u32,
    },
    /// Position deltas must not go backwards within one posting.
    #[error(
        "term {term_id} positions descend in local document {doc_ord}: {previous} then {current}"
    )]
    NonAscendingPositions {
        /// Dense local term id.
        term_id: u32,
        /// Local document ordinal.
        doc_ord: u32,
        /// Prior absolute token position.
        previous: u32,
        /// Current absolute token position.
        current: u32,
    },
    /// Checked host or durable-size arithmetic overflowed.
    #[error("Scribe flush arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Stable computation name.
        field: &'static str,
    },
    /// A fallible flush allocation could not be satisfied.
    #[error("unable to reserve {count} entries for Scribe flush {resource}")]
    Allocation {
        /// Stable allocation purpose.
        resource: &'static str,
        /// Requested entries or bytes.
        count: usize,
    },
    /// POSTINGS/BLOCKMAX encoding failed.
    #[error("Scribe POSTINGS/BLOCKMAX encoding failed: {0}")]
    BlockMax(#[from] BlockMaxError),
    /// POSITIONS encoding failed.
    #[error("Scribe POSITIONS encoding failed: {0}")]
    Positions(#[from] PositionCodecError),
    /// TERMDICT encoding failed.
    #[error("Scribe TERMDICT encoding failed: {0}")]
    TermDictionary(#[from] TermDictionaryError),
    /// DOCLEN encoding failed.
    #[error("Scribe DOCLEN encoding failed: {0}")]
    DocLen(#[from] DocLenCodecError),
    /// IDMAP encoding failed.
    #[error("Scribe IDMAP encoding failed: {0}")]
    IdMap(#[from] IdMapCodecError),
    /// IDHASH encoding failed.
    #[error("Scribe IDHASH encoding failed: {0}")]
    IdHash(#[from] IdHashCodecError),
    /// NUMERIC encoding failed.
    #[error("Scribe NUMERIC encoding failed: {0}")]
    Numeric(#[from] NumericCodecError),
    /// STOREDMETA encoding failed.
    #[error("Scribe STOREDMETA encoding failed: {0}")]
    StoredMeta(#[from] StoredMetaCodecError),
    /// STATS encoding failed.
    #[error("Scribe STATS encoding failed: {0}")]
    Stats(#[from] StatsCodecError),
    /// Final FSLX framing failed.
    #[error("Scribe segment framing failed: {0}")]
    Segment(#[from] QuillError),
}

const RADIX_DIGIT_BITS: u32 = 16;
const RADIX_DIGIT_BUCKETS: usize = 1 << RADIX_DIGIT_BITS;
const PARALLEL_RADIX_ROWS_PER_CHUNK: usize = 4_096;
const MAX_PARALLEL_RADIX_RANGE_BYTES: usize = 64 * 1024 * 1024;

type DocLenColumns = (Vec<u16>, Vec<Vec<Option<u32>>>);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct FlushTokenRow {
    term_id: u32,
    doc_ord: u32,
    position: u32,
}

#[derive(Debug)]
struct RadixPartition {
    rows: Vec<FlushTokenRow>,
    ranges: Vec<Range<usize>>,
}

/// Seal one shard accumulator into a complete deterministic FSLX mini-segment.
///
/// The accumulator is borrowed and is never reset by this function. A caller
/// may therefore publish the returned bytes first and clear the shard only
/// after Keeper makes the generation durable. Token rows are stably radix
/// partitioned by dense local term id; each resulting slice is consumed
/// directly by the posting builder, including stopword-heavy slices, without
/// another per-term triple copy.
///
/// # Errors
///
/// Returns [`FlushError`] for sidecar drift, rebase overflow, accumulator
/// invariant failure, codec failure, or final segment-framing failure.
pub fn flush_accumulator<A: TokenAnalyzer>(
    accumulator: &ColumnarAccumulator<A>,
    input: FlushSegmentInput<'_>,
) -> Result<EncodedSegment, FlushError> {
    let first_doc_ord = accumulator
        .first_doc_ord()
        .ok_or(FlushError::EmptyAccumulator)?;
    let last_doc_ord = accumulator
        .last_doc_ord()
        .ok_or(FlushError::EmptyAccumulator)?;
    let schema = accumulator.schema();
    if !input
        .lease_docid_base
        .is_multiple_of(u64::from(DOC_ORDS_PER_LEASE))
    {
        return Err(FlushError::MisalignedLeaseBase {
            lease_docid_base: input.lease_docid_base,
        });
    }

    validate_flush_documents(accumulator.document_ords(), input.documents)?;
    let docid_lo = rebase_doc_id(input.lease_docid_base, first_doc_ord)?;
    let last_docid = rebase_doc_id(input.lease_docid_base, last_doc_ord)?;
    let docid_hi = last_docid
        .checked_add(1)
        .ok_or(FlushError::ArithmeticOverflow {
            field: "exclusive document high bound",
        })?;
    let span =
        usize::try_from(docid_hi - docid_lo).map_err(|_| FlushError::ArithmeticOverflow {
            field: "document span host size",
        })?;
    let doc_count = u32::try_from(accumulator.document_count()).map_err(|_| {
        FlushError::ArithmeticOverflow {
            field: "segment document count",
        }
    })?;

    let (expected_field_ords, doclen_columns) =
        build_doclen_columns(accumulator, first_doc_ord, span)?;
    let rows = collect_flush_rows(accumulator)?;
    let partition = stable_radix_partition(rows, accumulator.terms().len())?;
    let (postings_bytes, positions_bytes, blockmax_bytes, term_inputs) =
        encode_ordered_term_streams(
            accumulator,
            &partition,
            input.lease_docid_base,
            docid_lo,
            &doclen_columns,
        )?;
    let term_sections = TermSectionLengths {
        postings: durable_len(&postings_bytes, "POSTINGS length")?,
        positions: schema_has_positions(schema)
            .then(|| durable_len(&positions_bytes, "POSITIONS length"))
            .transpose()?,
        blockmax: durable_len(&blockmax_bytes, "BLOCKMAX length")?,
    };
    let termdict = EncodedTermDictionary::encode_sorted(schema, term_sections, &term_inputs)?;

    let doclen_inputs = expected_field_ords
        .iter()
        .copied()
        .zip(&doclen_columns)
        .map(|(field_ord, values)| DocLenFieldInput::new(field_ord, values))
        .collect::<Vec<_>>();
    let doclen =
        EncodedDocLenSection::encode(docid_lo, docid_hi, &expected_field_ords, &doclen_inputs)?;

    let id_map_inputs = build_id_map_inputs(
        input.documents,
        first_doc_ord,
        span,
        accumulator.document_ords(),
    )?;
    let id_map = EncodedIdMapSection::encode(docid_lo, docid_hi, &id_map_inputs)?;
    let id_hash = EncodedIdHashSection::encode(id_map.section()?)?;

    let numeric = if accumulator.numeric_fields().is_empty() {
        None
    } else {
        Some(EncodedNumericSection::encode_accumulator(
            schema,
            docid_lo,
            docid_hi,
            input.lease_docid_base,
            accumulator,
        )?)
    };

    let stored_meta = if accumulator.stored_fields().is_empty() {
        None
    } else {
        Some(EncodedStoredMetaSection::encode_accumulator(
            docid_lo,
            docid_hi,
            input.lease_docid_base,
            accumulator,
        )?)
    };
    let stats_rows = accumulator
        .fields()
        .iter()
        .map(|field| FieldStats::new(field.field_ord(), field.total_tokens(), doc_count))
        .collect::<Vec<_>>();
    let stats = EncodedStatsSection::encode(&expected_field_ords, &stats_rows, doc_count)?;

    let mut sections = Vec::new();
    sections
        .try_reserve_exact(10)
        .map_err(|_| FlushError::Allocation {
            resource: "section inputs",
            count: 10,
        })?;
    sections.push(SectionInput::new(
        SectionKind::TERMDICT,
        termdict.as_bytes(),
    ));
    sections.push(SectionInput::new(SectionKind::POSTINGS, &postings_bytes));
    if schema_has_positions(schema) {
        sections.push(SectionInput::new(SectionKind::POSITIONS, &positions_bytes));
    }
    sections.push(SectionInput::new(SectionKind::BLOCKMAX, &blockmax_bytes));
    sections.push(SectionInput::new(SectionKind::DOCLEN, doclen.as_bytes()));
    sections.push(SectionInput::new(SectionKind::IDMAP, id_map.as_bytes()));
    sections.push(SectionInput::new(SectionKind::IDHASH, id_hash.as_bytes()));
    if let Some(numeric) = &numeric {
        sections.push(SectionInput::new(SectionKind::NUMERIC, numeric.as_bytes()));
    }
    if let Some(stored_meta) = &stored_meta {
        sections.push(SectionInput::new(
            SectionKind::STOREDMETA,
            stored_meta.as_bytes(),
        ));
    }
    sections.push(SectionInput::new(SectionKind::STATS, stats.as_bytes()));

    EncodedSegment::encode(
        SegmentHeaderInput {
            segment_id: input.segment_id,
            schema,
            docid_lo,
            docid_hi,
            doc_count,
            created_unix_s: input.created_unix_s,
            engine_version: input.engine_version,
        },
        &sections,
    )
    .map_err(FlushError::from)
}

fn schema_has_positions(schema: SchemaDescriptor) -> bool {
    schema.fields.iter().any(|field| {
        matches!(
            field.kind,
            FieldKind::Text {
                positions: true,
                ..
            }
        )
    })
}

fn validate_flush_documents(
    document_ords: &[u32],
    documents: &[FlushDocumentInput<'_>],
) -> Result<(), FlushError> {
    if documents.len() != document_ords.len() {
        return Err(FlushError::DocumentCountMismatch {
            expected: document_ords.len(),
            actual: documents.len(),
        });
    }
    for (index, (&expected, document)) in document_ords.iter().zip(documents).enumerate() {
        if document.doc_ord != expected {
            return Err(FlushError::DocumentOrdinalMismatch {
                index,
                expected,
                actual: document.doc_ord,
            });
        }
    }
    Ok(())
}

fn rebase_doc_id(lease_docid_base: u64, doc_ord: u32) -> Result<u64, FlushError> {
    let docid =
        lease_docid_base
            .checked_add(u64::from(doc_ord))
            .ok_or(FlushError::DocumentIdOverflow {
                lease_docid_base,
                doc_ord,
            })?;
    if docid > u64::from(u32::MAX) {
        return Err(FlushError::DocumentIdOverflow {
            lease_docid_base,
            doc_ord,
        });
    }
    Ok(docid)
}

fn durable_len(bytes: &[u8], field: &'static str) -> Result<u64, FlushError> {
    u64::try_from(bytes.len()).map_err(|_| FlushError::ArithmeticOverflow { field })
}

fn build_doclen_columns<A: TokenAnalyzer>(
    accumulator: &ColumnarAccumulator<A>,
    first_doc_ord: u32,
    span: usize,
) -> Result<DocLenColumns, FlushError> {
    let document_ords = accumulator.document_ords();
    let mut field_ords = Vec::new();
    let mut columns = Vec::new();
    field_ords
        .try_reserve_exact(accumulator.fields().len())
        .map_err(|_| FlushError::Allocation {
            resource: "DOCLEN field ordinals",
            count: accumulator.fields().len(),
        })?;
    columns
        .try_reserve_exact(accumulator.fields().len())
        .map_err(|_| FlushError::Allocation {
            resource: "DOCLEN columns",
            count: accumulator.fields().len(),
        })?;

    for field in accumulator.fields() {
        if field.document_lengths().len() != document_ords.len()
            || field.fieldnorm_ids().len() != document_ords.len()
        {
            return Err(FlushError::InvalidTokenColumn {
                field_ord: field.field_ord(),
                detail: "document-length/fieldnorm row count drift",
            });
        }
        let mut column = filled_vec(span, None, "DOCLEN span")?;
        for (index, (&doc_ord, &length)) in document_ords
            .iter()
            .zip(field.document_lengths())
            .enumerate()
        {
            let relative = doc_ord.checked_sub(first_doc_ord).ok_or_else(|| {
                FlushError::InvalidTokenColumn {
                    field_ord: field.field_ord(),
                    detail: "document ordinal precedes segment lower bound",
                }
            })?;
            let relative =
                usize::try_from(relative).map_err(|_| FlushError::ArithmeticOverflow {
                    field: "DOCLEN relative document index",
                })?;
            let slot = column
                .get_mut(relative)
                .ok_or_else(|| FlushError::InvalidTokenColumn {
                    field_ord: field.field_ord(),
                    detail: "document ordinal exceeds segment span",
                })?;
            *slot = Some(length);
            if field.fieldnorm_ids()[index] != fieldnorm_to_id(length) {
                return Err(FlushError::InvalidTokenColumn {
                    field_ord: field.field_ord(),
                    detail: "fieldnorm disagrees with raw document length",
                });
            }
        }
        field_ords.push(field.field_ord());
        columns.push(column);
    }
    Ok((field_ords, columns))
}

fn build_id_map_inputs<'a>(
    documents: &'a [FlushDocumentInput<'a>],
    first_doc_ord: u32,
    span: usize,
    document_ords: &[u32],
) -> Result<Vec<Option<IdMapEntryInput<'a>>>, FlushError> {
    let mut entries = filled_vec(span, None, "IDMAP span")?;
    for (&doc_ord, document) in document_ords.iter().zip(documents) {
        let relative =
            doc_ord
                .checked_sub(first_doc_ord)
                .ok_or(FlushError::ArithmeticOverflow {
                    field: "IDMAP relative document index",
                })?;
        let relative = usize::try_from(relative).map_err(|_| FlushError::ArithmeticOverflow {
            field: "IDMAP relative document host index",
        })?;
        let slot = entries
            .get_mut(relative)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "IDMAP document span",
            })?;
        *slot = Some(IdMapEntryInput::new(
            document.document_id,
            document.content_hash,
        ));
    }
    Ok(entries)
}

fn collect_flush_rows<A: TokenAnalyzer>(
    accumulator: &ColumnarAccumulator<A>,
) -> Result<Vec<FlushTokenRow>, FlushError> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(accumulator.token_count())
        .map_err(|_| FlushError::Allocation {
            resource: "token rows",
            count: accumulator.token_count(),
        })?;
    for field in accumulator.fields() {
        if field.term_ids().len() != field.doc_ords().len()
            || field
                .positions()
                .is_some_and(|positions| positions.len() != field.term_ids().len())
        {
            return Err(FlushError::InvalidTokenColumn {
                field_ord: field.field_ord(),
                detail: "parallel token column length drift",
            });
        }
        let mut previous_doc = None;
        let mut completed_index = 0_usize;
        for index in 0..field.term_ids().len() {
            let term_id = field.term_ids()[index];
            let doc_ord = field.doc_ords()[index];
            if usize::try_from(term_id).map_or(true, |id| id >= accumulator.terms().len()) {
                return Err(FlushError::TermIdOutOfRange {
                    field_ord: field.field_ord(),
                    term_id,
                });
            }
            if accumulator.terms().field_and_term(term_id).0 != field.field_ord() {
                return Err(FlushError::InvalidTokenColumn {
                    field_ord: field.field_ord(),
                    detail: "term interner field namespace drift",
                });
            }
            if previous_doc.is_some_and(|previous| doc_ord < previous) {
                return Err(FlushError::InvalidTokenColumn {
                    field_ord: field.field_ord(),
                    detail: "token document ordinals descend",
                });
            }
            while accumulator
                .document_ords()
                .get(completed_index)
                .is_some_and(|completed| *completed < doc_ord)
            {
                completed_index += 1;
            }
            if accumulator.document_ords().get(completed_index) != Some(&doc_ord) {
                return Err(FlushError::InvalidTokenColumn {
                    field_ord: field.field_ord(),
                    detail: "token references an incomplete document",
                });
            }
            previous_doc = Some(doc_ord);
            rows.push(FlushTokenRow {
                term_id,
                doc_ord,
                position: field.positions().map_or(0, |positions| positions[index]),
            });
        }
    }
    Ok(rows)
}

fn stable_radix_partition(
    rows: Vec<FlushTokenRow>,
    term_count: usize,
) -> Result<RadixPartition, FlushError> {
    let chunk_count = rayon::current_num_threads()
        .min(rows.len().div_ceil(PARALLEL_RADIX_ROWS_PER_CHUNK))
        .max(1);
    stable_radix_partition_with_chunks(rows, term_count, chunk_count)
}

fn stable_radix_partition_with_chunks(
    rows: Vec<FlushTokenRow>,
    term_count: usize,
    chunk_count: usize,
) -> Result<RadixPartition, FlushError> {
    let chunk_count = chunk_count.max(1).min(rows.len().max(1));
    let local_range_bytes = chunk_count
        .checked_mul(term_count)
        .and_then(|count| count.checked_mul(std::mem::size_of::<Range<usize>>()));
    if chunk_count == 1
        || rows.is_empty()
        || local_range_bytes.is_none_or(|bytes| bytes > MAX_PARALLEL_RADIX_RANGE_BYTES)
    {
        return stable_radix_partition_serial(rows, term_count);
    }
    let row_count = rows.len();
    let chunk_len = rows.len().div_ceil(chunk_count);
    let local_partitions: Result<Vec<_>, FlushError> = rows
        .par_chunks(chunk_len)
        .map(|chunk| {
            let mut owned = Vec::new();
            owned
                .try_reserve_exact(chunk.len())
                .map_err(|_| FlushError::Allocation {
                    resource: "thread-local radix rows",
                    count: chunk.len(),
                })?;
            owned.extend_from_slice(chunk);
            stable_radix_partition_serial(owned, term_count)
        })
        .collect();
    let local_partitions = local_partitions?;
    drop(rows);

    let mut counts = filled_vec(term_count, 0_usize, "global radix histogram")?;
    for partition in &local_partitions {
        for (count, range) in counts.iter_mut().zip(&partition.ranges) {
            *count = count
                .checked_add(range.len())
                .ok_or(FlushError::ArithmeticOverflow {
                    field: "global radix histogram count",
                })?;
        }
    }
    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(term_count)
        .map_err(|_| FlushError::Allocation {
            resource: "global radix prefix ranges",
            count: term_count,
        })?;
    let mut cursor = 0_usize;
    for count in counts {
        let end = cursor
            .checked_add(count)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "global radix prefix sum",
            })?;
        ranges.push(cursor..end);
        cursor = end;
    }
    if cursor != row_count {
        return Err(FlushError::ArithmeticOverflow {
            field: "global radix prefix coverage",
        });
    }

    let mut output = filled_vec(row_count, FlushTokenRow::default(), "parallel radix output")?;
    // Split the destination only at term boundaries, then give each disjoint
    // slice to one Rayon callback. Every callback copies its term runs from
    // source chunks in original chunk order: O(rows + chunks * terms), stable,
    // and free of locks, atomics, or unsafe aliasing.
    let term_groups = local_partitions.len().min(term_count).max(1);
    let terms_per_group = term_count.div_ceil(term_groups);
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(term_groups)
        .map_err(|_| FlushError::Allocation {
            resource: "parallel radix output groups",
            count: term_groups,
        })?;
    let mut output_tail = output.as_mut_slice();
    let mut previous_output_end = 0_usize;
    let mut term_start = 0_usize;
    while term_start < term_count {
        let term_end = term_start.saturating_add(terms_per_group).min(term_count);
        let output_end = ranges
            .get(term_end - 1)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "parallel radix group term range",
            })?
            .end;
        let group_len =
            output_end
                .checked_sub(previous_output_end)
                .ok_or(FlushError::ArithmeticOverflow {
                    field: "parallel radix group output length",
                })?;
        let (group, tail) = output_tail.split_at_mut(group_len);
        groups.push((term_start, term_end, group));
        output_tail = tail;
        previous_output_end = output_end;
        term_start = term_end;
    }
    if !output_tail.is_empty() || previous_output_end != row_count {
        return Err(FlushError::ArithmeticOverflow {
            field: "parallel radix output group coverage",
        });
    }
    groups
        .into_par_iter()
        .try_for_each(|(term_start, term_end, destination)| {
            let mut destination_offset = 0_usize;
            for term_id in term_start..term_end {
                for partition in &local_partitions {
                    let source_range =
                        partition
                            .ranges
                            .get(term_id)
                            .ok_or(FlushError::ArithmeticOverflow {
                                field: "thread-local radix term range",
                            })?;
                    let source = partition.rows.get(source_range.clone()).ok_or(
                        FlushError::ArithmeticOverflow {
                            field: "thread-local radix source range",
                        },
                    )?;
                    let destination_end = destination_offset.checked_add(source.len()).ok_or(
                        FlushError::ArithmeticOverflow {
                            field: "parallel radix destination range",
                        },
                    )?;
                    let target = destination
                        .get_mut(destination_offset..destination_end)
                        .ok_or(FlushError::ArithmeticOverflow {
                            field: "parallel radix destination coverage",
                        })?;
                    target.copy_from_slice(source);
                    destination_offset = destination_end;
                }
            }
            if destination_offset != destination.len() {
                return Err(FlushError::ArithmeticOverflow {
                    field: "parallel radix group scatter coverage",
                });
            }
            Ok(())
        })?;
    Ok(RadixPartition {
        rows: output,
        ranges,
    })
}

fn stable_radix_partition_serial(
    rows: Vec<FlushTokenRow>,
    term_count: usize,
) -> Result<RadixPartition, FlushError> {
    if rows.is_empty() {
        return Ok(RadixPartition {
            rows,
            ranges: filled_vec(term_count, 0..0, "empty term ranges")?,
        });
    }
    for row in &rows {
        if usize::try_from(row.term_id).map_or(true, |term_id| term_id >= term_count) {
            return Err(FlushError::TermIdOutOfRange {
                field_ord: 0,
                term_id: row.term_id,
            });
        }
    }

    let sorted = if term_count <= RADIX_DIGIT_BUCKETS {
        stable_digit_scatter(&rows, 0, term_count)?.0
    } else {
        // MSD partition first, then one stable low-digit scatter inside each
        // disjoint high-digit range. This preserves source order among equal
        // ids and avoids a term-count-sized second histogram.
        let high_bucket_count = term_count.div_ceil(RADIX_DIGIT_BUCKETS);
        let (high_sorted, high_ranges) =
            stable_digit_scatter(&rows, RADIX_DIGIT_BITS, high_bucket_count)?;
        let mut output = filled_vec(rows.len(), FlushTokenRow::default(), "radix output")?;
        for (high_digit, range) in high_ranges.into_iter().enumerate() {
            if range.is_empty() {
                continue;
            }
            let remaining_terms = term_count.saturating_sub(high_digit * RADIX_DIGIT_BUCKETS);
            let low_bucket_count = remaining_terms.min(RADIX_DIGIT_BUCKETS);
            let (low_sorted, _) =
                stable_digit_scatter(&high_sorted[range.clone()], 0, low_bucket_count)?;
            output[range].copy_from_slice(&low_sorted);
        }
        output
    };

    let mut ranges = filled_vec(term_count, 0..0, "term ranges")?;
    let mut cursor = 0_usize;
    for (term_id, range) in ranges.iter_mut().enumerate() {
        let start = cursor;
        while sorted
            .get(cursor)
            .is_some_and(|row| row.term_id as usize == term_id)
        {
            cursor += 1;
        }
        *range = start..cursor;
    }
    if cursor != sorted.len() {
        return Err(FlushError::ArithmeticOverflow {
            field: "radix partition coverage",
        });
    }
    Ok(RadixPartition {
        rows: sorted,
        ranges,
    })
}

fn stable_digit_scatter(
    rows: &[FlushTokenRow],
    shift: u32,
    bucket_count: usize,
) -> Result<(Vec<FlushTokenRow>, Vec<Range<usize>>), FlushError> {
    let mut counts = filled_vec(bucket_count, 0_usize, "radix histogram")?;
    for row in rows {
        let digit = ((row.term_id >> shift) & 0xffff) as usize;
        let count = counts
            .get_mut(digit)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "radix digit domain",
            })?;
        *count = count.checked_add(1).ok_or(FlushError::ArithmeticOverflow {
            field: "radix histogram count",
        })?;
    }

    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(bucket_count)
        .map_err(|_| FlushError::Allocation {
            resource: "radix prefix ranges",
            count: bucket_count,
        })?;
    let mut cursor = 0_usize;
    for count in counts {
        let end = cursor
            .checked_add(count)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "radix prefix sum",
            })?;
        ranges.push(cursor..end);
        cursor = end;
    }
    if cursor != rows.len() {
        return Err(FlushError::ArithmeticOverflow {
            field: "radix prefix coverage",
        });
    }
    let mut write_offsets = Vec::new();
    write_offsets
        .try_reserve_exact(ranges.len())
        .map_err(|_| FlushError::Allocation {
            resource: "radix write offsets",
            count: ranges.len(),
        })?;
    write_offsets.extend(ranges.iter().map(|range| range.start));
    let mut output = filled_vec(rows.len(), FlushTokenRow::default(), "radix scatter")?;
    for row in rows {
        let digit = ((row.term_id >> shift) & 0xffff) as usize;
        let destination = write_offsets
            .get_mut(digit)
            .ok_or(FlushError::ArithmeticOverflow {
                field: "radix scatter digit",
            })?;
        output[*destination] = *row;
        *destination += 1;
    }
    Ok((output, ranges))
}

type OrderedTermStreams<'a> = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<TermInput<'a>>);

fn encode_ordered_term_streams<'a, A: TokenAnalyzer>(
    accumulator: &'a ColumnarAccumulator<A>,
    partition: &RadixPartition,
    lease_docid_base: u64,
    docid_lo: u64,
    doclen_columns: &[Vec<Option<u32>>],
) -> Result<OrderedTermStreams<'a>, FlushError> {
    let sorted_ids = accumulator.terms().sorted_ids();
    let mut postings_bytes = Vec::new();
    let mut positions_bytes = Vec::new();
    let mut blockmax_bytes = Vec::new();
    let mut inputs = Vec::new();
    inputs
        .try_reserve_exact(sorted_ids.len())
        .map_err(|_| FlushError::Allocation {
            resource: "TERMDICT inputs",
            count: sorted_ids.len(),
        })?;

    for term_id_u32 in sorted_ids {
        let term_id = usize::try_from(term_id_u32).map_err(|_| FlushError::ArithmeticOverflow {
            field: "local term host index",
        })?;
        let range = partition
            .ranges
            .get(term_id)
            .ok_or(FlushError::TermHasNoRows {
                term_id: term_id_u32,
            })?;
        if range.is_empty() {
            return Err(FlushError::TermHasNoRows {
                term_id: term_id_u32,
            });
        }
        let rows = &partition.rows[range.clone()];
        let field_ord = accumulator.terms().field_and_term(term_id_u32).0;
        let field_index = accumulator
            .fields()
            .binary_search_by_key(&field_ord, FieldTokenColumns::field_ord)
            .map_err(|_| FlushError::InvalidTokenColumn {
                field_ord,
                detail: "term references a non-indexed field",
            })?;
        let field = &accumulator.fields()[field_index];
        let (postings, positions) = build_term_rows(
            term_id_u32,
            rows,
            lease_docid_base,
            field.positions().is_some(),
        )?;
        let field_doclens =
            doclen_columns
                .get(field_index)
                .ok_or(FlushError::InvalidTokenColumn {
                    field_ord,
                    detail: "missing DOCLEN source column",
                })?;
        let (encoded_postings, encoded_blockmax) =
            EncodedPostingList::encode_with_block_max(&postings, |global_docid| {
                let relative = u64::from(global_docid).checked_sub(docid_lo)?;
                let relative = usize::try_from(relative).ok()?;
                field_doclens
                    .get(relative)
                    .copied()
                    .flatten()
                    .map(fieldnorm_to_id)
            })?;
        let encoded_positions = positions
            .as_deref()
            .map(|values| EncodedPositionList::encode(&postings, values))
            .transpose()?;
        let postings_span = append_span(
            &mut postings_bytes,
            encoded_postings.as_bytes(),
            "POSTINGS span",
        )?;
        let positions_span = encoded_positions
            .as_ref()
            .map(|encoded| append_span(&mut positions_bytes, encoded.as_bytes(), "POSITIONS span"))
            .transpose()?;
        let blockmax_span = append_span(
            &mut blockmax_bytes,
            encoded_blockmax.as_bytes(),
            "BLOCKMAX span",
        )?;
        let doc_freq = encoded_postings.doc_freq();
        let metadata = positions_span.map_or_else(
            || TermMetadata::without_positions(doc_freq, postings_span, blockmax_span),
            |positions_span| {
                TermMetadata::with_positions(doc_freq, postings_span, positions_span, blockmax_span)
            },
        );
        let (field_ord, term) = accumulator.terms().field_and_term(term_id_u32);
        inputs.push(TermInput::new(field_ord, term, metadata));
    }
    Ok((postings_bytes, positions_bytes, blockmax_bytes, inputs))
}

fn build_term_rows(
    term_id: u32,
    rows: &[FlushTokenRow],
    lease_docid_base: u64,
    stores_positions: bool,
) -> Result<(Vec<Posting>, Option<Vec<u32>>), FlushError> {
    let posting_count = rows
        .windows(2)
        .filter(|pair| pair[0].doc_ord != pair[1].doc_ord)
        .count()
        .saturating_add(1);
    let mut postings = Vec::new();
    postings
        .try_reserve_exact(posting_count)
        .map_err(|_| FlushError::Allocation {
            resource: "term postings",
            count: posting_count,
        })?;
    let mut positions = if stores_positions {
        let mut values = Vec::new();
        values
            .try_reserve_exact(rows.len())
            .map_err(|_| FlushError::Allocation {
                resource: "term positions",
                count: rows.len(),
            })?;
        Some(values)
    } else {
        None
    };
    let first = rows.first().ok_or(FlushError::TermHasNoRows { term_id })?;
    let mut current_doc = first.doc_ord;
    let mut frequency = 0_u32;
    let mut previous_position = None;
    for (index, row) in rows.iter().enumerate() {
        if row.term_id != term_id {
            return Err(FlushError::TermIdOutOfRange {
                field_ord: 0,
                term_id: row.term_id,
            });
        }
        if row.doc_ord < current_doc {
            return Err(FlushError::NonAscendingTermDocuments {
                term_id,
                index,
                previous: current_doc,
                current: row.doc_ord,
            });
        }
        if row.doc_ord != current_doc {
            postings.push(Posting::new(
                u32::try_from(rebase_doc_id(lease_docid_base, current_doc)?).map_err(|_| {
                    FlushError::DocumentIdOverflow {
                        lease_docid_base,
                        doc_ord: current_doc,
                    }
                })?,
                frequency,
            ));
            current_doc = row.doc_ord;
            frequency = 0;
            previous_position = None;
        }
        if let Some(previous) = previous_position
            && row.position < previous
        {
            return Err(FlushError::NonAscendingPositions {
                term_id,
                doc_ord: row.doc_ord,
                previous,
                current: row.position,
            });
        }
        frequency = frequency
            .checked_add(1)
            .ok_or(FlushError::TermFrequencyOverflow {
                term_id,
                doc_ord: row.doc_ord,
            })?;
        if let Some(positions) = &mut positions {
            positions.push(row.position);
            previous_position = Some(row.position);
        }
    }
    postings.push(Posting::new(
        u32::try_from(rebase_doc_id(lease_docid_base, current_doc)?).map_err(|_| {
            FlushError::DocumentIdOverflow {
                lease_docid_base,
                doc_ord: current_doc,
            }
        })?,
        frequency,
    ));
    Ok((postings, positions))
}

fn append_span(
    output: &mut Vec<u8>,
    bytes: &[u8],
    field: &'static str,
) -> Result<ByteSpan, FlushError> {
    let offset = durable_len(output, field)?;
    output
        .try_reserve(bytes.len())
        .map_err(|_| FlushError::Allocation {
            resource: field,
            count: bytes.len(),
        })?;
    output.extend_from_slice(bytes);
    Ok(ByteSpan::new(offset, durable_len(bytes, field)?))
}

fn filled_vec<T: Clone>(
    len: usize,
    value: T,
    resource: &'static str,
) -> Result<Vec<T>, FlushError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .map_err(|_| FlushError::Allocation {
            resource,
            count: len,
        })?;
    output.resize(len, value);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grimoire::TermDictionary;
    use crate::quiver::{IdHashSection, IdMapSection, PositionList, PostingList, StatsSection};
    use crate::schema::{CASS_SEMANTIC_SCHEMA, DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA, FieldDescriptor};
    use crate::segment::SegmentReader;
    use frankensearch_lexical::tantivy_crate::tokenizer::TokenStream;
    use serde_json::Value;
    use std::hash::BuildHasherDefault;

    const LANGUAGE_CONTRACT_FIXTURE: &str =
        include_str!("../../../tests/fixtures/quill_language_contract.json");
    const SHARED_CORPUS_FIXTURE: &str = include_str!("../../../tests/fixtures/corpus.json");

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

    const INDEXED_NUMERIC_FIELDS: [FieldDescriptor; 1] = [FieldDescriptor {
        id: 0,
        name: "sequence",
        kind: FieldKind::U64 {
            indexed: true,
            fast: false,
        },
        stored: true,
    }];
    const INDEXED_NUMERIC_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "scribe-indexed-numeric-v1",
        fields: &INDEXED_NUMERIC_FIELDS,
    };

    const SIGNED_UNSIGNED_NUMERIC_FIELDS: [FieldDescriptor; 2] = [
        FieldDescriptor {
            id: 0,
            name: "created_at",
            kind: FieldKind::I64 {
                indexed: true,
                fast: false,
            },
            stored: false,
        },
        FieldDescriptor {
            id: 1,
            name: "sequence",
            kind: FieldKind::U64 {
                indexed: true,
                fast: false,
            },
            stored: false,
        },
    ];
    const SIGNED_UNSIGNED_NUMERIC_SCHEMA: SchemaDescriptor = SchemaDescriptor {
        name: "scribe-signed-unsigned-numeric-v1",
        fields: &SIGNED_UNSIGNED_NUMERIC_FIELDS,
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

    fn cass_tokens(analyzer_kind: AnalyzerKind, text: &str) -> Vec<AnalyzedToken> {
        let mut analyzer = CassAnalyzer::default();
        let mut tokens = Vec::new();
        analyzer.analyze(analyzer_kind, text, &mut |token| tokens.push(token.clone()));
        tokens
    }

    fn incumbent_cass_tokens(analyzer_kind: AnalyzerKind, text: &str) -> Vec<AnalyzedToken> {
        let mut index = frankensearch_lexical::tantivy_crate::Index::create_in_ram(
            frankensearch_lexical::cass_compat::cass_build_schema(),
        );
        frankensearch_lexical::cass_compat::cass_ensure_tokenizer(&mut index);
        let tokenizer_name = match analyzer_kind {
            AnalyzerKind::CassHyphenNormalize => "hyphen_normalize",
            AnalyzerKind::CassPrefixNormalize => "prefix_normalize",
            AnalyzerKind::FrankensearchDefault => {
                unreachable!("CASS incumbent helper requires a CASS analyzer")
            }
        };
        let mut analyzer = index
            .tokenizers()
            .get(tokenizer_name)
            .expect("shipping CASS tokenizer is registered");
        let mut stream = analyzer.token_stream(text);
        let mut tokens = Vec::new();
        while stream.advance() {
            let token = stream.token();
            tokens.push(AnalyzedToken {
                text: token.text.clone(),
                position: u32::try_from(token.position)
                    .expect("shipping CASS tokenizer position fits Quill u32 contract"),
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
    fn native_cass_executes_fixture_and_matches_shipping_incumbent() {
        let fixture: Value = serde_json::from_str(LANGUAGE_CONTRACT_FIXTURE)
            .expect("language contract fixture is valid JSON");
        let cases = fixture["analyzer_cases"]
            .as_array()
            .expect("language contract analyzer_cases is an array");
        let mut executed = 0_u32;
        for case in cases {
            let analyzer_kind = match case["analyzer"].as_str() {
                Some("hyphen_normalize") => AnalyzerKind::CassHyphenNormalize,
                Some("prefix_normalize") => AnalyzerKind::CassPrefixNormalize,
                Some("frankensearch_default") => continue,
                Some(other) => panic!("unhandled language-contract analyzer {other}"),
                None => panic!("analyzer fixture has no analyzer name"),
            };
            executed += 1;
            let case_id = case["id"].as_str().expect("analyzer case has an id");
            let input = fixture_input(case);
            let actual = cass_tokens(analyzer_kind, &input);
            assert_eq!(
                actual,
                incumbent_cass_tokens(analyzer_kind, &input),
                "native CASS stream diverged from the shipping incumbent for {case_id}"
            );
            if let Some(expected) = expected_fixture_tokens(case) {
                assert_eq!(actual, expected, "fixture golden diverged for {case_id}");
            }
            if let Some(expected_count) = case.get("expected_token_count").and_then(Value::as_u64) {
                assert_eq!(
                    actual.len(),
                    usize::try_from(expected_count).expect("fixture token count fits usize"),
                    "{case_id}"
                );
            }
            if let Some(expected_bytes) = case.get("expected_token_bytes").and_then(Value::as_u64) {
                assert_eq!(
                    actual.first().map(|token| token.text.len()),
                    Some(usize::try_from(expected_bytes).expect("fixture byte count fits usize")),
                    "{case_id}"
                );
            }
            if let Some(repeat) = case.get("expected_token_repeat").and_then(Value::as_str) {
                let expected_byte = repeat.as_bytes().first().copied();
                assert!(
                    repeat.len() == 1
                        && actual.first().is_some_and(|token| {
                            token
                                .text
                                .as_bytes()
                                .iter()
                                .all(|byte| Some(*byte) == expected_byte)
                        }),
                    "{case_id}"
                );
            }
        }
        assert_eq!(executed, 8, "all CASS analyzer fixtures must execute");
    }

    #[test]
    fn native_cass_matches_incumbent_at_token_boundaries_and_script_edges() {
        for analyzer_kind in [
            AnalyzerKind::CassHyphenNormalize,
            AnalyzerKind::CassPrefixNormalize,
        ] {
            for input in [
                "",
                "-abc abc- a--b a-b a1-b2",
                "ASCII_123/next",
                "搜",
                "搜索引擎",
                "かなカナ",
                "한글검색",
                "𠀀𠀁",
                "東京かな한글",
                "A搜索B",
                "éclair",
            ] {
                assert_eq!(
                    cass_tokens(analyzer_kind, input),
                    incumbent_cass_tokens(analyzer_kind, input),
                    "analyzer={analyzer_kind:?} input={input:?}"
                );
            }
        }
    }

    #[test]
    fn generated_prefix_pipeline_matches_incumbent_composition() {
        for input in [
            "",
            "Hello, happy tax payer!",
            "bd-q3fy foo_bar baz-qux",
            "abc-123 -- def",
            "Hello搜索World",
            "foo搜索-barあいう123",
            "café 𠀀 token",
            "multi---dash and trailing- hyphen",
        ] {
            let generated = cass_generate_edge_ngrams(input);
            let prefix_tokens = cass_tokens(AnalyzerKind::CassPrefixNormalize, &generated);
            assert_eq!(
                prefix_tokens,
                incumbent_cass_tokens(AnalyzerKind::CassPrefixNormalize, &generated),
                "generated prefix stream diverged for input={input:?}"
            );
            assert_eq!(
                prefix_tokens,
                cass_tokens(AnalyzerKind::CassHyphenNormalize, &generated),
                "generated prefixes must not require hyphen decomposition for input={input:?}"
            );
        }
    }

    #[test]
    fn native_cass_matches_incumbent_across_shared_fixture_corpus() {
        let fixture: Value = serde_json::from_str(SHARED_CORPUS_FIXTURE)
            .expect("shared corpus fixture is valid JSON");
        let documents = fixture["documents"]
            .as_array()
            .expect("shared corpus documents are an array");
        assert_eq!(
            documents.len(),
            120,
            "the complete pinned corpus must execute"
        );
        for document in documents {
            let doc_id = document["doc_id"].as_str().expect("fixture document id");
            for field_name in ["title", "content"] {
                let text = document[field_name].as_str().expect("fixture text field");
                assert_eq!(
                    cass_tokens(AnalyzerKind::CassHyphenNormalize, text),
                    incumbent_cass_tokens(AnalyzerKind::CassHyphenNormalize, text),
                    "doc={doc_id} field={field_name}"
                );

                let generated = cass_generate_edge_ngrams(text);
                assert_eq!(
                    cass_tokens(AnalyzerKind::CassPrefixNormalize, &generated),
                    incumbent_cass_tokens(AnalyzerKind::CassPrefixNormalize, &generated),
                    "generated prefix doc={doc_id} field={field_name}"
                );
                assert_eq!(
                    cass_build_preview(text, 400),
                    frankensearch_lexical::cass_compat::cass_build_preview_slow(text, 400),
                    "preview doc={doc_id} field={field_name}"
                );
            }
        }
    }

    #[test]
    fn seeded_cass_differential_matches_incumbent_and_named_slow_oracles() {
        const SEED: u64 = 0xE102_CA55_5C12_1BE5;
        const ATOMS: [&str; 15] = [
            "A", "z9", "-", "--", " ", "_", "/", "搜", "索", "かな", "한글", "𠀀", "é", "🙂", "123",
        ];
        let mut rng = DeterministicRng(SEED);
        for case in 0..128_u32 {
            let mut input = String::new();
            for _ in 0..rng.choose(48) {
                input.push_str(ATOMS[rng.choose(ATOMS.len())]);
            }

            let mut offset = 0;
            let mut native_walk = 0_u64;
            while let Some((ch, next)) = tokenizer_next_char(&input, offset) {
                native_walk = native_walk.wrapping_add(u64::from(ch as u32));
                offset = next;
            }
            assert_eq!(
                native_walk,
                frankensearch_lexical::cass_compat::cass_char_walk_slow(&input),
                "seed={SEED:#x} case={case} input={input:?}"
            );

            let native_cjk = if input.is_empty() || !input.chars().all(is_cass_cjk) {
                None
            } else {
                let chars: Vec<char> = input.chars().collect();
                (chars.len() >= 2).then_some(chars)
            };
            assert_eq!(
                native_cjk,
                frankensearch_lexical::cass_compat::cass_cjk_collect_slow(&input),
                "seed={SEED:#x} case={case} input={input:?}"
            );

            for analyzer_kind in [
                AnalyzerKind::CassHyphenNormalize,
                AnalyzerKind::CassPrefixNormalize,
            ] {
                assert_eq!(
                    cass_tokens(analyzer_kind, &input),
                    incumbent_cass_tokens(analyzer_kind, &input),
                    "seed={SEED:#x} case={case} analyzer={analyzer_kind:?} input={input:?}"
                );
            }
            assert_eq!(
                cass_generate_edge_ngrams(&input),
                frankensearch_lexical::cass_compat::cass_generate_edge_ngrams_slow(&input),
                "seed={SEED:#x} case={case} input={input:?}"
            );
            let preview_bound = rng.choose(32);
            assert_eq!(
                cass_build_preview(&input, preview_bound),
                frankensearch_lexical::cass_compat::cass_build_preview_slow(&input, preview_bound),
                "seed={SEED:#x} case={case} max_chars={preview_bound} input={input:?}"
            );
        }
    }

    #[test]
    fn cass_cjk_range_endpoints_and_neighbors_match_slow_oracle() {
        const RANGES: [(u32, u32); 9] = [
            (0x4E00, 0x9FFF),
            (0x3400, 0x4DBF),
            (0x3040, 0x309F),
            (0x30A0, 0x30FF),
            (0xAC00, 0xD7AF),
            (0x3100, 0x312F),
            (0x3300, 0x33FF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2A6DF),
        ];
        for (start, end) in RANGES {
            assert!(is_cass_cjk(
                char::from_u32(start).expect("valid range start")
            ));
            assert!(is_cass_cjk(char::from_u32(end).expect("valid range end")));
            for codepoint in [start - 1, start, end, end + 1] {
                let Some(ch) = char::from_u32(codepoint) else {
                    continue;
                };
                let input = ch.to_string().repeat(2);
                assert_eq!(
                    is_cass_cjk(ch),
                    frankensearch_lexical::cass_compat::cass_cjk_collect_slow(&input).is_some(),
                    "codepoint=U+{codepoint:04X}"
                );
                assert_eq!(
                    cass_tokens(AnalyzerKind::CassHyphenNormalize, &input),
                    incumbent_cass_tokens(AnalyzerKind::CassHyphenNormalize, &input),
                    "codepoint=U+{codepoint:04X}"
                );
            }
        }
    }

    #[test]
    fn cass_filter_order_preserves_parts_bigrams_and_position_gaps() {
        let oversized_compound = format!("{}-{}", "A".repeat(130), "B".repeat(130));
        let compound_tokens = cass_tokens(AnalyzerKind::CassHyphenNormalize, &oversized_compound);
        assert_eq!(
            compound_tokens,
            incumbent_cass_tokens(AnalyzerKind::CassHyphenNormalize, &oversized_compound)
        );
        assert_eq!(compound_tokens.len(), 2);
        assert!(compound_tokens.iter().all(|token| token.position == 0));
        assert_eq!(compound_tokens[0].text, "a".repeat(130));
        assert_eq!(compound_tokens[1].text, "b".repeat(130));

        let long_cjk = "搜".repeat(100);
        let cjk_tokens = cass_tokens(AnalyzerKind::CassHyphenNormalize, &long_cjk);
        assert_eq!(
            cjk_tokens,
            incumbent_cass_tokens(AnalyzerKind::CassHyphenNormalize, &long_cjk)
        );
        assert_eq!(cjk_tokens.len(), 99);
        assert!(cjk_tokens.iter().all(|token| token.text == "搜搜"));

        let exact_limit = "X".repeat(CASS_MAX_TOKEN_BYTES);
        let limit_tokens = cass_tokens(AnalyzerKind::CassHyphenNormalize, &exact_limit);
        assert_eq!(
            limit_tokens,
            incumbent_cass_tokens(AnalyzerKind::CassHyphenNormalize, &exact_limit)
        );
        assert_eq!(limit_tokens.len(), 1);
        assert_eq!(limit_tokens[0].text, "x".repeat(CASS_MAX_TOKEN_BYTES));

        let dropped_then_kept = format!("{} ok", "X".repeat(CASS_MAX_TOKEN_BYTES + 1));
        let gap_tokens = cass_tokens(AnalyzerKind::CassHyphenNormalize, &dropped_then_kept);
        assert_eq!(
            gap_tokens,
            incumbent_cass_tokens(AnalyzerKind::CassHyphenNormalize, &dropped_then_kept)
        );
        assert_eq!(gap_tokens.len(), 1);
        assert_eq!(gap_tokens[0].text, "ok");
        assert_eq!(gap_tokens[0].position, 1);
    }

    #[test]
    fn cass_helpers_execute_contract_fixture_and_match_slow_oracles() {
        let fixture: Value = serde_json::from_str(LANGUAGE_CONTRACT_FIXTURE)
            .expect("language contract fixture is valid JSON");
        let cases = fixture["helper_cases"]
            .as_array()
            .expect("language contract helper_cases is an array");
        let mut executed = 0_u32;
        for case in cases {
            let case_id = case["id"].as_str().expect("helper case has an id");
            let input = fixture_input(case);
            match case["helper"].as_str() {
                Some("cass_generate_edge_ngrams") => {
                    executed += 1;
                    let actual = cass_generate_edge_ngrams(&input);
                    assert_eq!(
                        actual,
                        frankensearch_lexical::cass_compat::cass_generate_edge_ngrams_slow(&input),
                        "{case_id}"
                    );
                    if let Some(expected) = case.get("expected").and_then(Value::as_str) {
                        assert_eq!(actual, expected, "{case_id}");
                    }
                    if let Some(expected_last) =
                        case.get("last_expected_prefix").and_then(Value::as_str)
                    {
                        assert_eq!(actual.split_whitespace().next_back(), Some(expected_last));
                    }
                    if let Some(expected_count) =
                        case.get("expected_prefix_count").and_then(Value::as_u64)
                    {
                        assert_eq!(
                            actual.split_whitespace().count(),
                            usize::try_from(expected_count)
                                .expect("fixture prefix count fits usize"),
                            "{case_id}"
                        );
                    }
                }
                Some("cass_build_preview") => {
                    executed += 1;
                    let max_chars = case["max_chars"]
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .expect("preview fixture bound fits usize");
                    let actual = cass_build_preview(&input, max_chars);
                    assert_eq!(
                        actual,
                        frankensearch_lexical::cass_compat::cass_build_preview_slow(
                            &input, max_chars
                        ),
                        "{case_id}"
                    );
                    assert_eq!(
                        actual,
                        case["expected"].as_str().expect("preview expected value"),
                        "{case_id}"
                    );
                }
                Some("truncate_query") => {}
                Some(other) => panic!("unhandled language-contract helper {other} in {case_id}"),
                None => panic!("helper fixture {case_id} has no helper name"),
            }
        }
        assert_eq!(executed, 5, "all native CASS helper fixtures must execute");
    }

    #[test]
    fn cass_helper_boundaries_match_slow_oracles() {
        let inputs = [
            String::new(),
            "a".to_owned(),
            "é".to_owned(),
            "é".repeat(20),
            "é".repeat(21),
            "hello,世界/한글".to_owned(),
        ];
        for input in &inputs {
            assert_eq!(
                cass_generate_edge_ngrams(input),
                frankensearch_lexical::cass_compat::cass_generate_edge_ngrams_slow(input),
                "input={input:?}"
            );
            for max_chars in [0, 1, 2, 19, 20, 21, 64] {
                assert_eq!(
                    cass_build_preview(input, max_chars),
                    frankensearch_lexical::cass_compat::cass_build_preview_slow(input, max_chars),
                    "input={input:?} max_chars={max_chars}"
                );
            }
        }
    }

    #[test]
    fn cass_analyzer_reset_retains_reusable_scratch() {
        let mut analyzer = CassAnalyzer::default();
        analyzer.analyze(
            AnalyzerKind::CassHyphenNormalize,
            &"A".repeat(CASS_MAX_TOKEN_BYTES),
            &mut |_| {},
        );
        let retained = analyzer.bytes_reserved();
        assert!(retained >= CASS_MAX_TOKEN_BYTES);
        analyzer.reset();
        assert_eq!(analyzer.bytes_reserved(), retained);
        analyzer.analyze(AnalyzerKind::CassPrefixNormalize, "short", &mut |_| {});
        assert_eq!(analyzer.bytes_reserved(), retained);
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
    fn native_cass_family_accumulates_alternatives_and_prefixes() {
        let mut accumulator =
            ColumnarAccumulator::with_analyzer(CASS_SEMANTIC_SCHEMA, CassAnalyzer::default())
                .expect("native CASS family covers the complete semantic schema");
        let report = accumulator
            .add_document(
                0,
                &[
                    IndexedFieldValue::new(6, "BD-Q3FY search"),
                    IndexedFieldValue::new(8, "BD-Q3FY"),
                ],
            )
            .expect("native CASS analysis accumulates");
        assert_eq!(report.admitted_tokens, 5);
        assert_eq!(report.oversized_tokens, 0);

        let title = accumulator.field(6).expect("title column");
        let title_prefix = accumulator.field(8).expect("title prefix column");
        assert_eq!(title.positions(), Some([0, 0, 0, 1].as_slice()));
        assert_eq!(title.document_lengths(), &[4]);
        assert_eq!(title_prefix.positions(), None);
        assert_eq!(title_prefix.document_lengths(), &[1]);

        let term_id = |field_ord: u16, term: &[u8]| {
            (0..u32::try_from(accumulator.terms().len()).expect("small term table"))
                .find(|id| accumulator.terms().field_and_term(*id) == (field_ord, term))
                .expect("expected CASS term is interned")
        };
        let compound = term_id(6, b"bd-q3fy");
        let left = term_id(6, b"bd");
        let right = term_id(6, b"q3fy");
        let search = term_id(6, b"search");
        // E1 owns the position-column contract. The production phrase-scorer
        // differential that consumes these alternatives belongs to E4.5.
        assert!(phrase_matches(title, 0, &[compound, search]));
        assert!(phrase_matches(title, 0, &[left, search]));
        assert!(phrase_matches(title, 0, &[right, search]));
        assert!(!phrase_matches(title, 0, &[left, right]));
        assert_eq!(
            accumulator
                .terms()
                .field_and_term(title_prefix.term_ids()[0]),
            (8, b"bd-q3fy".as_slice())
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
    fn stored_columns_preserve_opaque_absent_and_present_empty_values() {
        let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
        let ordinal = 41_u64.to_le_bytes();
        accumulator
            .add_document_with_stored(
                5,
                &[
                    IndexedFieldValue::new(0, "doc-a"),
                    IndexedFieldValue::new(1, "body"),
                    IndexedFieldValue::new(2, ""),
                ],
                &[
                    StoredFieldValue::new(3, b""),
                    StoredFieldValue::new(4, &ordinal),
                ],
            )
            .expect("first document accumulates");
        accumulator
            .add_document_with_stored(
                8,
                &[
                    IndexedFieldValue::new(0, "doc-b"),
                    IndexedFieldValue::new(1, ""),
                ],
                &[],
            )
            .expect("second document accumulates");

        assert_eq!(
            accumulator
                .stored_fields()
                .iter()
                .map(StoredFieldColumns::field_ord)
                .collect::<Vec<_>>(),
            [0, 1, 2, 3, 4]
        );
        let ids = accumulator.stored_field(0).expect("stored id");
        assert_eq!(ids.value(0), Some(b"doc-a".as_slice()));
        assert_eq!(ids.value(1), Some(b"doc-b".as_slice()));
        let titles = accumulator.stored_field(2).expect("stored title");
        assert_eq!(titles.value(0), Some(b"".as_slice()));
        assert_eq!(titles.value(1), None);
        let metadata = accumulator.stored_field(3).expect("stored metadata");
        assert_eq!(metadata.value(0), Some(b"".as_slice()));
        assert_eq!(metadata.value(1), None);
        assert_eq!(metadata.presence(), &[1, 0]);
        assert_eq!(metadata.offsets(), &[0, 0, 0]);
        assert_eq!(
            accumulator
                .stored_field(4)
                .expect("stored ordinal")
                .value(0),
            Some(ordinal.as_slice())
        );
    }

    #[test]
    fn stored_columns_follow_each_shipped_descriptor_without_name_hardcoding() {
        fn assert_stored_fields<A: TokenAnalyzer>(
            accumulator: &ColumnarAccumulator<A>,
            expected: &[u16],
        ) {
            assert_eq!(
                accumulator
                    .stored_fields()
                    .iter()
                    .map(StoredFieldColumns::field_ord)
                    .collect::<Vec<_>>(),
                expected,
                "schema={} should derive stored ordinals only from descriptors",
                accumulator.schema().name
            );
        }

        let default = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid default schema");
        assert_stored_fields(&default, &[0, 1, 2, 3, 4]);

        let fsfs = ColumnarAccumulator::new(FSFS_CHUNK_SCHEMA).expect("valid FSFS schema");
        assert_stored_fields(&fsfs, &[0, 1, 2, 3, 4, 5, 7]);

        let cass =
            ColumnarAccumulator::with_analyzer(CASS_SEMANTIC_SCHEMA, CassAnalyzer::default())
                .expect("native analyzer covers the CASS descriptor");
        assert_stored_fields(&cass, &[0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14]);
    }

    #[test]
    fn stored_input_validation_is_atomic_and_rejects_divergent_duplicates() {
        let mut unstored = ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid schema");
        assert_eq!(
            unstored.add_document_with_stored(1, &[], &[StoredFieldValue::new(0, b"not stored")],),
            Err(AccumulatorError::NonStoredField {
                field_ord: 0,
                field_name: "key",
            })
        );
        assert_eq!(unstored.document_count(), 0);
        assert!(unstored.terms().is_empty());
        assert_eq!(
            unstored
                .stored_field(3)
                .expect("stored-only column")
                .document_count(),
            0
        );

        let mut duplicate = ColumnarAccumulator::new(DEFAULT_SCHEMA).expect("valid schema");
        assert_eq!(
            duplicate.add_document_with_stored(
                2,
                &[IndexedFieldValue::new(0, "indexed")],
                &[StoredFieldValue::new(0, b"different stored bytes")],
            ),
            Err(AccumulatorError::DuplicateField {
                doc_ord: 2,
                field_ord: 0,
            })
        );
        assert_eq!(duplicate.document_count(), 0);
        assert!(duplicate.terms().is_empty());
        for field in duplicate.stored_fields() {
            assert_eq!(field.document_count(), 0);
            assert_eq!(field.offsets(), &[0]);
        }
    }

    #[test]
    fn numeric_input_is_typed_stored_and_validation_is_atomic() {
        let mut accumulator =
            ColumnarAccumulator::new(INDEXED_NUMERIC_SCHEMA).expect("valid numeric schema");
        let high_unsigned = (1_u64 << 63) + 17;
        accumulator
            .add_document_with_numeric(3, &[], &[IndexedNumericValue::u64(0, high_unsigned)])
            .expect("typed unsigned value accumulates");

        assert_eq!(accumulator.document_ords(), &[3]);
        assert_eq!(
            accumulator
                .numeric_field(0)
                .expect("indexed numeric column")
                .values(),
            &[Some(NumericValue::U64(high_unsigned))]
        );
        assert_eq!(
            accumulator
                .stored_field(0)
                .expect("stored numeric column")
                .value(0),
            Some(high_unsigned.to_le_bytes().as_slice())
        );

        assert_eq!(
            accumulator.add_document_with_numeric(4, &[], &[IndexedNumericValue::i64(0, -1)]),
            Err(AccumulatorError::NumericTypeMismatch {
                field_ord: 0,
                field_name: "sequence",
                expected: "u64",
                actual: "i64",
            })
        );
        assert_eq!(
            accumulator.add_document_with_stored(4, &[], &[StoredFieldValue::new(0, &[1, 2, 3])],),
            Err(AccumulatorError::InvalidNumericBytes {
                field_ord: 0,
                field_name: "sequence",
                expected: 8,
                actual: 3,
            })
        );
        assert_eq!(
            accumulator.add_document_with_values(
                4,
                &[],
                &[IndexedNumericValue::u64(0, 9)],
                &[StoredFieldValue::new(0, &9_u64.to_le_bytes())],
            ),
            Err(AccumulatorError::DuplicateField {
                doc_ord: 4,
                field_ord: 0,
            })
        );

        assert_eq!(accumulator.document_ords(), &[3]);
        assert_eq!(
            accumulator
                .numeric_field(0)
                .expect("indexed numeric column")
                .values(),
            &[Some(NumericValue::U64(high_unsigned))]
        );
        let stored = accumulator.stored_field(0).expect("stored numeric column");
        assert_eq!(stored.document_count(), 1);
        assert_eq!(
            stored.value(0),
            Some(high_unsigned.to_le_bytes().as_slice())
        );

        let next = 42_u64;
        accumulator
            .add_document_with_stored(4, &[], &[StoredFieldValue::new(0, &next.to_le_bytes())])
            .expect("rejected document did not advance accumulator state");
        assert_eq!(accumulator.document_ords(), &[3, 4]);
        assert_eq!(
            accumulator
                .numeric_field(0)
                .expect("indexed numeric column")
                .values(),
            &[
                Some(NumericValue::U64(high_unsigned)),
                Some(NumericValue::U64(next)),
            ]
        );
    }

    #[test]
    fn sparse_numeric_seal_rebases_both_types_through_u32_max() {
        let mut accumulator = ColumnarAccumulator::new(SIGNED_UNSIGNED_NUMERIC_SCHEMA)
            .expect("valid signed/unsigned numeric schema");
        accumulator
            .add_document_with_numeric(
                0,
                &[],
                &[
                    IndexedNumericValue::i64(0, -7),
                    IndexedNumericValue::u64(1, 0),
                ],
            )
            .expect("first sparse numeric document accumulates");
        accumulator
            .add_document_with_numeric(17, &[], &[IndexedNumericValue::i64(0, i64::MAX)])
            .expect("middle sparse numeric document accumulates");
        accumulator
            .add_document_with_numeric(
                DOC_ORDS_PER_LEASE - 1,
                &[],
                &[
                    IndexedNumericValue::i64(0, i64::MIN),
                    IndexedNumericValue::u64(1, u64::MAX),
                ],
            )
            .expect("lease-boundary numeric document accumulates");

        let documents = [
            FlushDocumentInput::new(0, "first", 1),
            FlushDocumentInput::new(17, "middle", 2),
            FlushDocumentInput::new(DOC_ORDS_PER_LEASE - 1, "last", 3),
        ];
        let lease_docid_base = (1_u64 << 32) - u64::from(DOC_ORDS_PER_LEASE);
        let encoded = flush_accumulator(
            &accumulator,
            FlushSegmentInput {
                segment_id: 91,
                lease_docid_base,
                created_unix_s: 0,
                engine_version: 0,
                documents: &documents,
            },
        )
        .expect("sparse signed/unsigned segment seals");
        let reader =
            SegmentReader::from_owned(encoded.into_bytes(), SIGNED_UNSIGNED_NUMERIC_SCHEMA)
                .expect("sparse signed/unsigned segment reopens");
        assert_eq!(reader.header().docid_lo, lease_docid_base);
        assert_eq!(reader.header().docid_hi, 1_u64 << 32);
        let numeric_bytes = reader
            .section(SectionKind::NUMERIC)
            .expect("NUMERIC checksum verifies")
            .expect("numeric schema requires NUMERIC");
        let section = crate::quiver::NumericSection::parse(
            numeric_bytes,
            SIGNED_UNSIGNED_NUMERIC_SCHEMA,
            lease_docid_base,
            1_u64 << 32,
        )
        .expect("sparse NUMERIC payload validates");
        let base_docid = u32::try_from(lease_docid_base).expect("lease base fits u32");
        assert_eq!(
            section
                .field(0)
                .expect("signed field")
                .entries()
                .collect::<Vec<_>>(),
            vec![
                crate::quiver::NumericEntry::i64(i64::MIN, u32::MAX),
                crate::quiver::NumericEntry::i64(-7, base_docid),
                crate::quiver::NumericEntry::i64(i64::MAX, base_docid + 17),
            ]
        );
        assert_eq!(
            section
                .field(1)
                .expect("unsigned field")
                .entries()
                .collect::<Vec<_>>(),
            vec![
                crate::quiver::NumericEntry::u64(0, base_docid),
                crate::quiver::NumericEntry::u64(u64::MAX, u32::MAX),
            ]
        );
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
        for field in accumulator.stored_fields() {
            assert_eq!(field.document_count(), 0);
            assert_eq!(field.offsets(), &[0]);
            assert!(field.presence().is_empty());
            assert!(field.blob().is_empty());
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
        let outer_stored_fields = accumulator
            .stored_fields
            .capacity()
            .saturating_mul(std::mem::size_of::<StoredFieldColumns>());
        assert!(outer_stored_fields > 0);
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
            .saturating_add(outer_stored_fields)
            .saturating_add(
                accumulator
                    .stored_fields
                    .iter()
                    .map(StoredFieldColumns::bytes_reserved)
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

    #[test]
    fn radix_flush_is_byte_identical_and_reopens_every_section() {
        let mut accumulator =
            ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid mixed schema");
        accumulator
            .add_document_with_stored(
                2,
                &[
                    IndexedFieldValue::new(0, "doc-a"),
                    IndexedFieldValue::new(1, "alpha beta alpha"),
                    IndexedFieldValue::new(2, "alpha alpha"),
                ],
                &[StoredFieldValue::new(3, b"stored-a")],
            )
            .expect("first sparse document accumulates");
        accumulator
            .add_document_with_stored(
                5,
                &[
                    IndexedFieldValue::new(0, "doc-b"),
                    IndexedFieldValue::new(1, "beta alpha"),
                    IndexedFieldValue::new(2, "gamma"),
                ],
                &[StoredFieldValue::new(3, b"stored-b")],
            )
            .expect("second sparse document accumulates");
        let identities = [
            FlushDocumentInput::from_canonical_content(2, "doc-a", b"canonical-a"),
            FlushDocumentInput::from_canonical_content(5, "doc-b", b"canonical-b"),
        ];
        let input = FlushSegmentInput {
            segment_id: 0xfeed_beef,
            lease_docid_base: 65_536,
            created_unix_s: 1_700_000_000,
            engine_version: 0x0001_0002,
            documents: &identities,
        };

        let first = flush_accumulator(&accumulator, input).expect("first flush");
        let second = flush_accumulator(&accumulator, input).expect("second flush");
        assert_eq!(first.as_bytes(), second.as_bytes());
        assert_eq!(accumulator.document_ords(), &[2, 5]);

        let reader = SegmentReader::from_bytes(first.as_bytes(), MIXED_POSITION_SCHEMA)
            .expect("segment reopens");
        reader.verify().expect("all section witnesses verify");
        assert_eq!(reader.header().docid_lo, 65_538);
        assert_eq!(reader.header().docid_hi, 65_542);
        assert_eq!(reader.header().doc_count, 2);
        let postings_bytes = reader
            .section(SectionKind::POSTINGS)
            .expect("POSTINGS checksum")
            .expect("POSTINGS required");
        let positions_bytes = reader
            .section(SectionKind::POSITIONS)
            .expect("POSITIONS checksum")
            .expect("POSITIONS required");
        let blockmax_bytes = reader
            .section(SectionKind::BLOCKMAX)
            .expect("BLOCKMAX checksum")
            .expect("BLOCKMAX required");
        let sections = TermSectionLengths {
            postings: u64::try_from(postings_bytes.len()).expect("POSTINGS length"),
            positions: Some(u64::try_from(positions_bytes.len()).expect("POSITIONS length")),
            blockmax: u64::try_from(blockmax_bytes.len()).expect("BLOCKMAX length"),
        };
        let dictionary = TermDictionary::parse(
            reader
                .section(SectionKind::TERMDICT)
                .expect("TERMDICT checksum")
                .expect("TERMDICT required"),
            MIXED_POSITION_SCHEMA,
            sections,
        )
        .expect("TERMDICT reopens");
        assert_eq!(dictionary.term_count() as usize, accumulator.terms().len());

        for term_id in 0..u32::try_from(accumulator.terms().len()).expect("term count fits u32") {
            let (field_ord, term) = accumulator.terms().field_and_term(term_id);
            let field = accumulator
                .fields()
                .iter()
                .find(|field| field.field_ord() == field_ord)
                .expect("term field exists");
            let mut expected_runs = Vec::<(u32, u32, Vec<u32>)>::new();
            for index in 0..field.term_ids().len() {
                if field.term_ids()[index] != term_id {
                    continue;
                }
                let doc_id =
                    u32::try_from(input.lease_docid_base + u64::from(field.doc_ords()[index]))
                        .expect("fixture document id fits u32");
                if let Some((previous_doc, frequency, run_positions)) = expected_runs.last_mut()
                    && *previous_doc == doc_id
                {
                    *frequency += 1;
                    if let Some(positions) = field.positions() {
                        run_positions.push(positions[index]);
                    }
                } else {
                    expected_runs.push((
                        doc_id,
                        1,
                        field
                            .positions()
                            .map_or_else(Vec::new, |positions| vec![positions[index]]),
                    ));
                }
            }
            let entry = dictionary
                .lookup(field_ord, term)
                .expect("term lookup is valid")
                .expect("every interned term is emitted");
            assert_eq!(
                entry.metadata.doc_freq,
                u32::try_from(expected_runs.len()).expect("fixture doc frequency fits u32"),
                "doc_freq drift for field {field_ord} term {:?}",
                String::from_utf8_lossy(term)
            );
            let posting_start =
                usize::try_from(entry.metadata.postings.offset).expect("posting offset");
            let posting_end = posting_start
                + usize::try_from(entry.metadata.postings.len).expect("posting length");
            let posting_list = PostingList::parse(
                &postings_bytes[posting_start..posting_end],
                entry.metadata.doc_freq,
            )
            .expect("posting list reopens");
            let decoded = posting_list.decode_all().expect("posting list decodes");
            let expected_postings = expected_runs
                .iter()
                .map(|(doc_id, frequency, _)| Posting::new(*doc_id, *frequency))
                .collect::<Vec<_>>();
            assert_eq!(
                decoded,
                expected_postings,
                "posting drift for field {field_ord} term {:?}",
                String::from_utf8_lossy(term)
            );
            if field.positions().is_some() {
                let position_span = entry.metadata.positions.expect("position span required");
                let position_start =
                    usize::try_from(position_span.offset).expect("position offset");
                let position_end =
                    position_start + usize::try_from(position_span.len).expect("position length");
                let position_list = PositionList::parse(
                    &positions_bytes[position_start..position_end],
                    &posting_list,
                )
                .expect("position list reopens");
                for (ordinal, (_, _, expected_positions)) in expected_runs.iter().enumerate() {
                    let actual_positions = position_list
                        .positions_for_ordinal(
                            u32::try_from(ordinal).expect("fixture posting ordinal fits u32"),
                        )
                        .expect("position run exists")
                        .collect::<Result<Vec<_>, _>>()
                        .expect("position run decodes");
                    assert_eq!(actual_positions, *expected_positions);
                }
            } else {
                assert!(entry.metadata.positions.is_none());
            }
        }

        let alpha = dictionary
            .lookup(1, b"alpha")
            .expect("alpha lookup")
            .expect("alpha exists");
        assert_eq!(alpha.metadata.doc_freq, 2);
        let posting_start =
            usize::try_from(alpha.metadata.postings.offset).expect("posting offset");
        let posting_end =
            posting_start + usize::try_from(alpha.metadata.postings.len).expect("posting length");
        let postings = PostingList::parse(
            &postings_bytes[posting_start..posting_end],
            alpha.metadata.doc_freq,
        )
        .expect("alpha postings reopen");
        assert_eq!(
            postings.decode_all().expect("alpha postings decode"),
            [Posting::new(65_538, 2), Posting::new(65_541, 1)]
        );
        let position_span = alpha.metadata.positions.expect("alpha stores positions");
        let position_start = usize::try_from(position_span.offset).expect("position offset");
        let position_end =
            position_start + usize::try_from(position_span.len).expect("position length");
        let positions =
            PositionList::parse(&positions_bytes[position_start..position_end], &postings)
                .expect("alpha positions reopen");
        assert_eq!(
            positions
                .positions_for_ordinal(0)
                .expect("first alpha run")
                .collect::<Result<Vec<_>, _>>()
                .expect("first alpha positions"),
            [0, 2]
        );
        assert_eq!(
            positions
                .positions_for_ordinal(1)
                .expect("second alpha run")
                .collect::<Result<Vec<_>, _>>()
                .expect("second alpha positions"),
            [1]
        );

        let id_map = IdMapSection::parse(
            reader
                .section(SectionKind::IDMAP)
                .expect("IDMAP checksum")
                .expect("IDMAP required"),
            65_538,
            65_542,
        )
        .expect("IDMAP reopens");
        assert_eq!(id_map.get(65_538).expect("first id").document_id(), "doc-a");
        assert!(!id_map.contains(65_539));
        assert!(!id_map.contains(65_540));
        assert_eq!(
            id_map.get(65_541).expect("second id").document_id(),
            "doc-b"
        );
        let id_hash = IdHashSection::parse(
            reader
                .section(SectionKind::IDHASH)
                .expect("IDHASH checksum")
                .expect("IDHASH required"),
            id_map,
        )
        .expect("IDHASH reopens");
        assert_eq!(id_hash.lookup("doc-a"), Some(65_538));
        assert_eq!(id_hash.lookup("doc-b"), Some(65_541));

        let stats = StatsSection::parse(
            reader
                .section(SectionKind::STATS)
                .expect("STATS checksum")
                .expect("STATS required"),
            &[0, 1, 2],
            2,
        )
        .expect("STATS reopens");
        assert_eq!(stats.field(0), Some(FieldStats::new(0, 2, 2)));
        assert_eq!(stats.field(1), Some(FieldStats::new(1, 5, 2)));
        assert_eq!(stats.field(2), Some(FieldStats::new(2, 3, 2)));

        let stored = crate::quiver::StoredMetaSection::parse(
            reader
                .section(SectionKind::STOREDMETA)
                .expect("STOREDMETA checksum")
                .expect("STOREDMETA required"),
            65_538,
            65_542,
            &[3],
        )
        .expect("STOREDMETA reopens");
        let stored_field = stored.field(3).expect("stored field");
        assert_eq!(stored_field.get(65_538), Some(b"stored-a".as_slice()));
        assert_eq!(stored_field.get(65_539), None);
        assert_eq!(stored_field.get(65_541), Some(b"stored-b".as_slice()));
    }

    #[test]
    fn radix_partition_preserves_equal_term_source_order_in_both_widths() {
        let one_pass = stable_radix_partition(
            vec![
                FlushTokenRow {
                    term_id: 2,
                    doc_ord: 1,
                    position: 7,
                },
                FlushTokenRow {
                    term_id: 0,
                    doc_ord: 2,
                    position: 8,
                },
                FlushTokenRow {
                    term_id: 2,
                    doc_ord: 3,
                    position: 9,
                },
            ],
            3,
        )
        .expect("one-pass radix partition");
        assert_eq!(
            one_pass.rows,
            [
                FlushTokenRow {
                    term_id: 0,
                    doc_ord: 2,
                    position: 8,
                },
                FlushTokenRow {
                    term_id: 2,
                    doc_ord: 1,
                    position: 7,
                },
                FlushTokenRow {
                    term_id: 2,
                    doc_ord: 3,
                    position: 9,
                },
            ]
        );

        let two_pass = stable_radix_partition(
            vec![
                FlushTokenRow {
                    term_id: 65_536,
                    doc_ord: 4,
                    position: 1,
                },
                FlushTokenRow {
                    term_id: 1,
                    doc_ord: 5,
                    position: 2,
                },
                FlushTokenRow {
                    term_id: 65_536,
                    doc_ord: 6,
                    position: 3,
                },
                FlushTokenRow {
                    term_id: 65_535,
                    doc_ord: 7,
                    position: 4,
                },
            ],
            65_537,
        )
        .expect("two-pass radix partition");
        assert_eq!(
            two_pass
                .rows
                .iter()
                .map(|row| (row.term_id, row.doc_ord))
                .collect::<Vec<_>>(),
            [(1, 5), (65_535, 7), (65_536, 4), (65_536, 6)]
        );

        let source = (0_u32..10_000)
            .map(|ordinal| FlushTokenRow {
                term_id: ordinal % 17,
                doc_ord: ordinal,
                position: ordinal,
            })
            .collect::<Vec<_>>();
        let serial = stable_radix_partition_with_chunks(source.clone(), 17, 1)
            .expect("serial reference partition");
        let parallel = stable_radix_partition_with_chunks(source, 17, 4)
            .expect("four-chunk parallel partition");
        assert_eq!(parallel.rows, serial.rows);
        assert_eq!(parallel.ranges, serial.ranges);

        let mut seed = 0x7f4a_7c15_d2b7_4407_u64;
        let mut high_cardinality_source = Vec::new();
        for ordinal in 0_u32..12_000 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let term_id = if ordinal == 0 {
                65_536
            } else {
                u32::try_from(seed % 65_537).expect("bounded seeded term id")
            };
            high_cardinality_source.push(FlushTokenRow {
                term_id,
                doc_ord: ordinal,
                position: ordinal,
            });
        }
        let high_cardinality_serial =
            stable_radix_partition_with_chunks(high_cardinality_source.clone(), 65_537, 1)
                .expect("two-pass serial reference partition");
        let high_cardinality_parallel =
            stable_radix_partition_with_chunks(high_cardinality_source, 65_537, 4)
                .expect("two-pass four-chunk parallel partition");
        assert_eq!(high_cardinality_parallel.rows, high_cardinality_serial.rows);
        assert_eq!(
            high_cardinality_parallel.ranges,
            high_cardinality_serial.ranges
        );
    }

    #[test]
    fn flush_rejects_sidecar_drift_duplicates_and_docid_overflow() {
        let empty = ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid schema");
        assert!(matches!(
            flush_accumulator(
                &empty,
                FlushSegmentInput {
                    segment_id: 1,
                    lease_docid_base: 0,
                    created_unix_s: 0,
                    engine_version: 0,
                    documents: &[],
                }
            ),
            Err(FlushError::EmptyAccumulator)
        ));

        let mut accumulator =
            ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid schema");
        accumulator
            .add_document(7, &[IndexedFieldValue::new(1, "same")])
            .expect("first document");
        accumulator
            .add_document(9, &[IndexedFieldValue::new(1, "same")])
            .expect("second document");
        let one_identity = [FlushDocumentInput::new(7, "only-one", 1)];
        assert!(matches!(
            flush_accumulator(
                &accumulator,
                FlushSegmentInput {
                    segment_id: 2,
                    lease_docid_base: 0,
                    created_unix_s: 0,
                    engine_version: 0,
                    documents: &one_identity,
                }
            ),
            Err(FlushError::DocumentCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
        let drifted = [
            FlushDocumentInput::new(7, "first", 1),
            FlushDocumentInput::new(8, "second", 2),
        ];
        assert!(matches!(
            flush_accumulator(
                &accumulator,
                FlushSegmentInput {
                    segment_id: 3,
                    lease_docid_base: 0,
                    created_unix_s: 0,
                    engine_version: 0,
                    documents: &drifted,
                }
            ),
            Err(FlushError::DocumentOrdinalMismatch {
                index: 1,
                expected: 9,
                actual: 8
            })
        ));
        let duplicates = [
            FlushDocumentInput::new(7, "duplicate", 1),
            FlushDocumentInput::new(9, "duplicate", 2),
        ];
        assert!(matches!(
            flush_accumulator(
                &accumulator,
                FlushSegmentInput {
                    segment_id: 4,
                    lease_docid_base: 0,
                    created_unix_s: 0,
                    engine_version: 0,
                    documents: &duplicates,
                }
            ),
            Err(FlushError::IdHash(
                IdHashCodecError::DuplicateDocumentId { .. }
            ))
        ));
        let valid = [
            FlushDocumentInput::new(7, "first", 1),
            FlushDocumentInput::new(9, "second", 2),
        ];
        assert!(matches!(
            flush_accumulator(
                &accumulator,
                FlushSegmentInput {
                    segment_id: 5,
                    lease_docid_base: 1,
                    created_unix_s: 0,
                    engine_version: 0,
                    documents: &valid,
                }
            ),
            Err(FlushError::MisalignedLeaseBase {
                lease_docid_base: 1
            })
        ));
        assert!(matches!(
            flush_accumulator(
                &accumulator,
                FlushSegmentInput {
                    segment_id: 6,
                    lease_docid_base: 1_u64 << 32,
                    created_unix_s: 0,
                    engine_version: 0,
                    documents: &valid,
                }
            ),
            Err(FlushError::DocumentIdOverflow { doc_ord: 7, .. })
        ));

        let mut boundary =
            ColumnarAccumulator::new(MIXED_POSITION_SCHEMA).expect("valid boundary schema");
        boundary
            .add_document(
                DOC_ORDS_PER_LEASE - 1,
                &[IndexedFieldValue::new(1, "last global document")],
            )
            .expect("last lease ordinal accumulates");
        let boundary_identity = [FlushDocumentInput::new(
            DOC_ORDS_PER_LEASE - 1,
            "last-global-document",
            4,
        )];
        let boundary_segment = flush_accumulator(
            &boundary,
            FlushSegmentInput {
                segment_id: 7,
                lease_docid_base: (1_u64 << 32) - u64::from(DOC_ORDS_PER_LEASE),
                created_unix_s: 0,
                engine_version: 0,
                documents: &boundary_identity,
            },
        )
        .expect("u32::MAX remains a valid posting docid");
        assert_eq!(boundary_segment.header().docid_lo, u64::from(u32::MAX));
        assert_eq!(boundary_segment.header().docid_hi, 1_u64 << 32);

        let mut numeric =
            ColumnarAccumulator::new(INDEXED_NUMERIC_SCHEMA).expect("valid numeric schema");
        numeric
            .add_document_with_stored(0, &[], &[StoredFieldValue::new(0, &42_u64.to_le_bytes())])
            .expect("numeric stored bytes accumulate");
        let numeric_identity = [FlushDocumentInput::new(0, "numeric", 3)];
        let numeric_segment = flush_accumulator(
            &numeric,
            FlushSegmentInput {
                segment_id: 8,
                lease_docid_base: 0,
                created_unix_s: 0,
                engine_version: 0,
                documents: &numeric_identity,
            },
        )
        .expect("indexed numeric field seals");
        let numeric_reader = crate::segment::SegmentReader::from_owned(
            numeric_segment.into_bytes(),
            INDEXED_NUMERIC_SCHEMA,
        )
        .expect("numeric segment reopens");
        let numeric_bytes = numeric_reader
            .section(SectionKind::NUMERIC)
            .expect("NUMERIC checksum verifies")
            .expect("indexed numeric schema requires NUMERIC");
        let numeric_section =
            crate::quiver::NumericSection::parse(numeric_bytes, INDEXED_NUMERIC_SCHEMA, 0, 1)
                .expect("NUMERIC payload validates");
        assert_eq!(
            numeric_section
                .field(0)
                .expect("sequence field")
                .entries()
                .collect::<Vec<_>>(),
            vec![crate::quiver::NumericEntry::u64(42, 0)]
        );
    }
}
