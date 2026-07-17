//! Scribe ingest pipeline.
//!
//! Tokenization, per-shard arenas, columnar accumulation, and segment flush
//! land behind this module in the Quill E1 milestones.
//!
//! This milestone (`bd-quill-e1-scribe-bejd.3`) provides the per-shard
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
//!
//! Invariants:
//! - No per-token heap allocation on the intern hot path: an existing term
//!   costs one hash + one arena byte-compare; a new term costs one bump copy.
//! - Term ids are assigned in first-insertion order and are deterministic for
//!   a given ingest order (hasher choice never influences ids or output —
//!   collision buckets only affect probe cost).
//! - `CompactString`/owned allocation happens only at dictionary-build time,
//!   never per token.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

/// Default arena chunk size. 1 MiB amortizes chunk-vector growth while
/// keeping per-shard reset cheap; term corpora that exceed it simply add
/// chunks (each retained across [`ByteArena::reset`]).
pub const DEFAULT_ARENA_CHUNK_BYTES: usize = 1 << 20;

/// Composite dictionary key prefix width: big-endian `field_ord: u16`
/// (FSLX §5.1). Big-endian so raw byte comparison equals `(field, term)`
/// lexicographic order.
pub const FIELD_PREFIX_BYTES: usize = 2;

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
        let mut ids: Vec<u32> = (0..u32::try_from(self.spans.len()).expect("term id space exceeds u32")).collect();
        ids.sort_unstable_by(|a, b| self.composite_key(*a).cmp(self.composite_key(*b)));
        ids
    }

    /// Approximate bytes held, for the shard flush trigger. Counts arena
    /// reservation (the RSS-relevant figure), span table, and a conservative
    /// per-bucket map estimate. Documented as an *accounting approximation*:
    /// the budget test asserts it is monotone under inserts and covers the
    /// raw key bytes, not that it is byte-exact.
    #[must_use]
    pub fn bytes_used(&self) -> usize {
        const BUCKET_ESTIMATE: usize = 8 + std::mem::size_of::<Bucket>() + 8;
        self.arena.bytes_reserved()
            + self.spans.len() * std::mem::size_of::<ArenaSpan>()
            + self.buckets.len() * BUCKET_ESTIMATE
    }

    /// Reset for the next flush cycle: clears terms, retains arena chunk and
    /// container capacity (steady-state cycles allocate nothing).
    pub fn reset(&mut self) {
        self.arena.reset();
        self.spans.clear();
        self.buckets.clear();
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::BuildHasherDefault;

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
        let mut prev = interner.bytes_used();
        let mut term_bytes_total = 0usize;
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
            tracing::info!(cycle, used, reserved, chunks, terms = interner.len(), "soak cycle");
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
}
