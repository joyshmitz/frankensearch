# FSLX On-Disk Format Specification & Registry

**Status:** Normative. **Format version: 1** (unreleased — pre-freeze; changes require only a new row in §12 until the first shipped release, after which changes require a version bump + migration note + gauntlet fixture).
**Owning bead:** `bd-quill-e0-contracts-j53p.2`. **Design of record:** `COMPREHENSIVE_PLAN_FOR_THE_QUILL_LEXICAL_ENGINE.md` §10 (as amended).
**Prime directives:** hand-rolled bytes only (never serde-derived); every durable hash is **xxh3-64** (stable across versions/platforms; **never ahash**); little-endian fixed-width integers except where noted; every artifact independently validatable.

---

## 1. Scope and files

A Quill index directory contains exactly these artifact classes (anything else matching none of them is garbage by definition):

| Artifact | Name schema | Mutability | Spec |
|---|---|---|---|
| Segment | `seg-<hex16>.fslx` | Immutable once referenced by a manifest | §4–§5 |
| Manifest | `MANIFEST`, `MANIFEST.prev` | Replaced only via §6.2 publish protocol | §6 |
| Generation claim | `gen-<u64 decimal>.claim` | Created `O_EXCL`, removed after publish | §7.2 |
| Writer lock | `LOCK` | Created/held by the single writer process | §7.1 |
| Temp | `.tmp-*` prefix | Unreachable; GC-eligible under writer lock | §8 |
| Quarantine | `seg-<hex16>.fslx.quarantine` | Retained, never deleted by the engine | §8 |
| Repair sidecar | `<artifact>.fec` | Managed by frankensearch-durability | durability crate |
| Engine pointer | `CURRENT` (one level above, in the lexical root) | Replaced via §7.3 protocol | §7.3 |

Segment ids are random u64 (hex16 lowercase), collision-checked against the live manifest at creation.

## 2. Common conventions

- **Endianness:** little-endian everywhere **except** the TERMDICT field-ordinal key prefix (§5.1), which is big-endian so that raw byte comparison equals (field, term) lexicographic order.
- **Alignment:** every section starts at a 64-byte boundary (zero padding between sections; padding bytes are not covered by section checksums).
- **Checksums:** each section carries xxh3-64 over its exact `len` bytes (recorded in the section table). The segment header carries `header_crc32` (CRC32 over header bytes). The file trailer carries `file_xxh3` (xxh3-64 over all bytes from offset 0 to the start of the trailer) + `trailer_crc32` (CRC32 over the trailer's preceding 8 bytes). "CRC32" in this specification means the IEEE CRC-32 computed exactly by `crc32fast::hash`. The `.fec` fast-path verify uses `file_xxh3` — one hash pass shared with durability.
- **Varints ("vint"):** LEB128, max 10 bytes, canonical (no over-long encodings; readers reject).
- **Strings/blobs:** raw bytes, never NUL-terminated; lengths always explicit.
- **Docids:** u32 in all postings/section payloads; u64 in header/manifest fields (future-proofing). Every segment has `docid_lo < docid_hi <= 2^32` and `doc_count <= docid_hi - docid_lo`. Docid semantics are governed by Q1 (`docs/contracts/quill-q1-docid-discipline.md`): monotone allocation, session leases, burned tails, **never reused**; R1 = a segment's docid range is a subinterval of one lease block; R2 = merges combine only bound-consecutive runs.
- **Positions:** u32 (a 2 MiB document exceeds u16 token positions).
- **Fieldnorms:** 1 byte, tantivy 0.26.1's exact 256-entry table (vendored: `crates/frankensearch-lexical/src/quill_contract.rs`, landed c5cd8b51).
- **schema_id:** xxh3-64 of the schema descriptor's canonical encoding (see `SchemaDescriptor` in the scaffolding bead `bd-quill-e1-0-crate-scaffold-j1i6`). A reader that sees an unknown `schema_id` returns a typed error; it never guesses.

## 3. Section kinds (u16)

| Kind | Name | Required | Spec |
|---:|---|---|---|
| 1 | TERMDICT | yes | §5.1 |
| 2 | POSTINGS | yes | §5.2 |
| 3 | POSITIONS | iff schema indexes positions | §5.3 |
| 4 | BLOCKMAX | yes | §5.4 |
| 5 | DOCLEN | yes | §5.5 |
| 6 | IDMAP | yes | §5.6 |
| 7 | IDHASH | yes | §5.7 |
| 8 | NUMERIC | iff schema has indexed numerics | §5.8 |
| 9 | STOREDMETA | iff schema stores any field | §5.9 |
| 10 | STATS | yes | §5.10 |

Known section kinds 1..=10 appear exactly once when required by the schema and
are absent when their schema condition is false; their `flags` field is zero.
Kind 0 is invalid. Readers skip an unknown kind > 10 only when its flags are
exactly bit 0 (`OPTIONAL_SKIPPABLE`) and reject it when bit 0 is clear or any
other flag bit is set. This is the forward-compatibility valve, and a registry
row is required before a writer uses it.

## 4. Segment file layout

```
offset  size  field
0       8     magic = "FSLXSEG\0"
8       4     format_version: u32 = 1
12      4     header_len: u32          (bytes of the header block that follows)
16      H     header block:
                segment_id: u64
                schema_id: u64
                docid_lo: u64            # Q1 covering interval [lo, hi)
                docid_hi: u64
                doc_count: u32           # live docs at seal (tombstones excluded at seal time)
                reserved: u32 = 0
                created_unix_s: i64
                engine_version: u32      # packed semver; informational only (§6.1)
                section_count: u16
                reserved2: u16 = 0
                section_table: section_count × {
                  section_kind: u16
                  flags: u16             # bit 0 = OPTIONAL_SKIPPABLE
                  offset: u64            # absolute file offset, 64-aligned
                  len: u64               # exact payload bytes (excl. padding)
                  xxh3: u64              # over payload bytes
                }
16+H    4     header_crc32: u32        (CRC32 over bytes [16, 16+H))
...     ...   sections, each 64-aligned, in ascending section_kind order
EOF-12  8     file_xxh3: u64           (xxh3-64 over bytes [0, EOF-12))
EOF-4   4     trailer_crc32: u32       (CRC32 over the 8 file_xxh3 bytes)
```

For canonical v1 bytes, `H = 56 + 28 * section_count`; readers reject every
other `header_len`. The section table is strictly increasing by `section_kind`
and must contain the unique, exact schema-required known-section set described
in §3, plus only valid optional-skippable unknown entries. The first section
starts at `align_up(16 + H + 4, 64)`. Every later section starts at
`align_up(previous.offset + previous.len, 64)`. All offset/length arithmetic is
checked, payload ranges must not overlap or extend beyond the trailer, and every
byte skipped for alignment is zero. The 12-byte trailer begins immediately
after the final section payload: there is no final alignment padding and no
trailing data after `trailer_crc32`.

Validation on open is structural and eager: readers validate bounds with checked
arithmetic, magic, version, canonical header length, reserved fields, the exact
section set/flags/order, minimal alignment, non-overlap, zero padding, exact
trailer placement, `header_crc32`, and `trailer_crc32`. They do not hash any
section payload during structural open. Each section entry's xxh3 is validated
lazily before that payload is first exposed; skipped unknown sections are not
exposed by normal reads. `verify()` performs the same structural validation,
hashes every section table entry (including unknown optional-skippable
sections), and recomputes `file_xxh3` over `[0, EOF-12)`.
Any failure becomes a typed corruption error (`SearchError`-mapped), never a
panic. Readers are generic over the byte source (`&[u8]`): mmap-backed on disk,
owned buffers for `in_memory()` (bead e3.1 notes). The mmap-backed constructor
accepts only a regular canonical `seg-<hex16>.fslx` path after Keeper has synced
and atomically renamed the temp artifact; the filename id must equal the header
`segment_id`. Published segment inodes are never
mutated or truncated in place while a reader can retain them.

## 5. Section payloads

### 5.1 TERMDICT

Composite key = `field_ord: u16 (BIG-endian)` ++ `term bytes`. Raw-byte ordering of composite keys equals (field, term) lexicographic order; per-field iteration = range scan on the 2-byte prefix.

```
u32 block_count
block index: block_count × { first_key_len: vint, first_key: bytes, block_offset: vint (from blocks-area start) }
blocks: prefix-compressed, target ~4 KiB:
  u16 entry_count
  entries (restart interval R = 16):
    every R-th entry: full key (key_len: vint, key bytes)
    otherwise:        shared_prefix_len: vint, suffix_len: vint, suffix bytes
    each entry's payload:
      doc_freq: vint            # postings-bearing docs in THIS segment (incl. tombstoned; see §5.10 note)
      postings_offset: vint     # from POSTINGS start
      postings_len: vint
      positions_offset: vint    # present iff POSITIONS section exists AND field indexes positions; else omitted
      positions_len: vint       #   (presence decided per-field from schema descriptor — not per-entry flags)
      blockmax_offset: vint     # from BLOCKMAX start
      blockmax_len: vint
```

TERMDICT keys are arbitrary byte strings: the term portion is 0..=65,530 bytes
(so the composite key is 2..=65,532 bytes), and an empty term is legal. `doc_freq`
must fit `u32`; all other TERMDICT vints have the common-convention `u64`
domain. There is no per-entry flags word in v1. Positions presence is derived
from the schema field, so adjacent positional and non-positional fields may use
different payload widths in the same block.

POSTINGS, the positional subset of POSITIONS, and BLOCKMAX ranges are contiguous
in composite-term order: the first range starts at zero, each next offset equals
the prior end, and the final end equals the referenced section length. The
explicit `blockmax_len` is load-bearing because `doc_freq` does not determine
the number of posting blocks after Q1 concat-merges preserve interior partial
blocks.

The blocks area begins immediately after the final block-index entry. The first
relative `block_offset` is zero and later offsets are strictly increasing. This
relative base avoids a self-referential encoding in which absolute offset vint
widths change the size of the index that precedes them.

Canonical writers count the two-byte `entry_count` header toward a 4096-byte
block target. They start a new block before the next entry would exceed that
target, reset the restart ordinal in the new block, and encode every non-restart
entry with the longest shared prefix. A single entry whose legal term bytes make
the block exceed 4096 bytes is retained as one oversized singleton block.
Readers reject oversized multi-entry or prematurely split blocks, non-maximal
shared prefixes, empty blocks, duplicate/out-of-order keys, index/block
first-key disagreement, gaps, and trailing bytes. On open, a reader builds a
bounded in-memory restart directory from the validated full-key entries;
restart offsets are not duplicated on disk.

Lookup binary-searches the block index and the validated restart directory,
then scans at most `R = 16` entries. Ordered iteration is field-major. Per-field
range scans are half-open `[start, end)`. An empty prefix selects the whole
field; other prefix scans stop on the first field or byte-prefix mismatch, so
an all-`0xff` prefix does not require fabricating a successor key.

### 5.2 POSTINGS

Per term: a sequence of independently self-delimiting **doc blocks**. A fresh
seal emits zero or more 128-posting FOR/BITMAP blocks followed by at most one
partial VINT block. A concat-merged stream may retain VINT partial blocks at
interior segment seams; readers therefore consume blocks until the TERMDICT
`postings_len` is exhausted and validate that the sum of `posting_count` equals
`doc_freq`. This is load-bearing for Q1: ordinary merges preserve every input
block byte-for-byte instead of shifting all later 128-posting boundaries.

```
every block:
  u8  block_kind             # 0 = FOR, 1 = BITMAP, 2 = VINT partial
  u8  posting_count          # 1..=128; FOR/BITMAP = 128, VINT = 1..=127
  u16 payload_len            # LE; bytes after this common header

FOR payload (posting_count = 128):
  u32 first_doc              # ABSOLUTE global docid (load-bearing for concat-merge: no rebase)
  u8  doc_width              # 0..=32 bits; 127 deltas (doc[i+1]-doc[i], always >=1, stored minus 1)
  u8  freq_kind              # 0 = all freqs == 1; 1 = bitpacked
  u8  freq_width             # present iff freq_kind == 1; 1..=32
  bitpacked delta payload    # ceil(127*doc_width/8) bytes, LSB-first within bytes
  bitpacked freq payload     # iff freq_kind == 1: 128 values, stored minus 1

BITMAP payload (posting_count = 128; selected by fresh writers when span < 512):
  u32 first_doc
  u16 span                   # last_doc - first_doc + 1; 128..511
  64-byte bitmap (512 bits)  # bit i is bitmap[i/8] & (1 << (i%8)); doc first_doc+i present
  u8  freq_kind              # 0 = all freqs == 1; 1 = bitpacked
  u8  freq_width             # present iff freq_kind == 1; 1..=32
  bitpacked freq payload     # iff freq_kind == 1: 128 freq-1 values in bitmap doc order

VINT partial payload (posting_count = 1..=127):
  posting_count × { doc_delta: vint (first is absolute-from-zero... see note), freq: vint }
```

`payload_len` is canonical: it must exactly equal the bytes implied by the
selected codec/header fields, and readers reject both truncation and trailing
payload bytes. Widths are minimal: `doc_width` is the bit width of the largest
stored `delta-1`; `freq_kind=0` iff every frequency is one, otherwise
`freq_width` is the bit width of the largest stored `freq-1`. FOR/BITMAP counts
other than 128 and VINT counts outside 1..=127 are corrupt. Fresh writers select
BITMAP iff a full block's inclusive span is <512 and FOR otherwise. Readers also
accept a structurally canonical FOR block at any span; this keeps the decoder
domain complete (including width-0 FOR) without changing deterministic writer
output. For BITMAP, bit 0 and bit `span-1` are set, exactly 128 bits are set, and
every bit at index `>= span` is zero.

**Partial-block delta note (normative):** the block's first entry stores the
absolute docid as a vint (u32 range); subsequent entries store `delta-1` vints.
Freqs are stored as-is (≥1). Partial blocks are legal anywhere in a merged term
stream, not only at its end.

**Fragmentation contract:** an ordinary merge never creates a new posting
fragment. Each freshly sealed leaf segment contributes at most one partial block
per term. A merge output preserves exactly the leaf partial blocks already
present in its inputs; it may therefore contain multiple partial blocks, but
repeated merge schedules create none. Optional reblocking is reserved for an
explicit deep-compaction lever and is not part of Q1 concat.

### 5.3 POSITIONS

Per term (fields with positions only), the TERMDICT-referenced span has this
canonical grammar:

```
u32 block_count LE
block directory: block_count × {
  first_posting_ordinal: canonical u32 vint
  block_offset: canonical u64 vint  # relative to blocks-area start
}
blocks area: concatenated whole-document runs {
  first_position: canonical u32 vint
  (freq - 1) × delta: canonical u32 vint
}
```

Runs are doc-aligned with POSTINGS order and contain exactly that posting's
`freq` positions. Positions are u32; the first value in every document is
absolute and later values are checked deltas from the preceding position.
Positions must therefore be nondecreasing, and **zero deltas are legal** for
same-position alternatives such as `HyphenDecompose`. Reconstructing a value
past `u32::MAX`, a non-canonical or overflowing vint, a truncated run, or bytes
remaining after the last frequency-derived value is corruption.

For `doc_freq > 0`, the first directory row is `(0, 0)`. Later posting ordinals
and offsets are strictly increasing and in range. A block's posting interval
ends at the next directory ordinal (or `doc_freq` for the last block), and its
byte interval ends at the next offset (or the term span's end). Blocks are
non-empty and never split a document run. The codec-level empty-list form is
exactly `block_count = 0` with no directory rows or payload bytes. Directory
offsets use the blocks-area-relative base so their vint widths cannot change
the directory base that precedes them.

A fresh seal greedily packs complete document runs into a 4096-byte payload
target: it starts a new block before the next complete run would exceed the
target. A single run larger than 4096 bytes is retained as one oversized
one-document block. Readers validate the directory, frequency alignment,
integer domains, and the oversized-singleton rule, but accept underfilled
interior blocks because they may be preserved Q1 seams.

Ordinary concat-merge creates no new position block and never repacks a run. It
copies every source block payload byte-for-byte in input order, then rewrites
only the directory: `first_posting_ordinal` is rebased by prior source
`doc_freq`, and `block_offset` by prior source payload length. Thus repeated
merge schedules preserve exactly the leaf blocks already present. Optional
reblocking remains an explicit deep-compaction operation.

The POSITIONS section exists iff at least one schema field enables positions;
it is a canonical zero-length section when such a schema has no positional
terms. Only terms from position-enabled fields carry non-empty POSITIONS spans.
Segment open, ordinary term queries, and position-free field reads determine
that POSITIONS is unnecessary from schema and TERMDICT metadata; they must not
call `section(POSITIONS)`, expose its payload, or checksum it. Explicit position
access and phrase evaluation trigger first exposure and checksum validation;
full `verify()` remains intentionally eager.

### 5.4 BLOCKMAX

Per term, one entry per POSTINGS block (including every full or partial block,
regardless of position). The TERMDICT `blockmax_len` span contains entries
back-to-back with **no count prefix**; its exact entry count is derived from the
validated POSTINGS block stream:

```
{ first_doc: u32 LE,
  block_offset: u64 canonical vint (from this term's postings_offset),
  max_freq_q: u8,
  min_fieldnorm: u8 }
```

`max_freq_q` uses the pinned Tantivy 0.26.1 conservative encoding: encode as
`min(max_frequency, 255)`; codes `0..=254` decode exactly, while code `255`
decodes to `u32::MAX`. Code zero is invalid for a real block because posting
frequencies are positive. `min_fieldnorm` is the exact minimum stored DOCLEN
byte among the block's posting docs. Every byte value is valid data; presence
comes from POSTINGS/IDMAP, never from a fieldnorm sentinel.

Readers parse exactly one record per validated POSTINGS block and reject
missing/trailing records, malformed/non-canonical vints, a first doc or byte
offset that disagrees with POSTINGS, or a non-canonical maximum-frequency code.
A merge-only structural view may preserve the fieldnorm byte opaquely, but it
exposes no scoring API. The score-capable view additionally binds DOCLEN and
rejects an incorrect or missing minimum fieldnorm before exposing bounds. That
validation is performed once per immutable term view and the result is cached;
skip/scoring cursors then reuse it and advance without decoding skipped blocks.

**Soundness contract:** impact bounds are computed at **query time** from
`(max_freq_q, min_fieldnorm)` with the live snapshot's idf/avgdl — never stored
as impact scalars (stored impacts are unsound under changing avgdl; see plan
§10.4 and bead e2.3). `max_freq_q` decodes to a value ≥ the true block max
frequency; `min_fieldnorm` decodes to a length ≤ every scored document length
in the block. Together they dominate every `(freq, |d|)` for BM25's tf factor,
which is increasing in frequency and decreasing in document length. Pruning is
disabled for negative/non-finite term weights because they invalidate that
monotonicity argument.

Ordinary Q1 concat re-emits entries in posting-block order. It preserves
`first_doc`, `max_freq_q`, and `min_fieldnorm`, and adds the sum of prior source
term POSTINGS byte lengths to `block_offset`. It never synthesizes a seam entry
or recomputes a bound. Fresh seal derives POSTINGS and BLOCKMAX from the same
source-block pass before the written POSTINGS self-validation; deep compaction
is the re-encoding path.

### 5.5 DOCLEN

Per field with `Text`/`Keyword` indexing: `doc_count_span = docid_hi - docid_lo` fieldnorm bytes, direct-indexed by `docid - docid_lo`. **Holes:** positions for docids absent from the segment (burned/tombstone-folded) hold canonical byte `0x00`. Every u8, including `0x00` and `0xFF`, is a valid fieldnorm for a present document, so readers must derive presence from IDMAP/postings membership and never from the DOCLEN byte. Cursors only score present docids. Post-compaction segments keep positional indexing with holes; space overhead is bounded by the compaction-density threshold.

Field order: ascending `field_ord`; each field's array is 64-aligned within the section; a small directory (`field_count × {field_ord: u16, offset: u32}`) heads the section.

### 5.6 IDMAP

```
directory: { entry_count: u32 = docid span, blob_offset: u32 }
offsets: (span + 1) × u32     # offsets[i]..offsets[i+1] = DocId bytes for docid_lo+i; equal offsets = hole
hashes:  span × u64           # xxh3-64 of the doc's CANONICAL CONTENT bytes (content_hash — resumable-bulk
                              #   witness, bd-quill-duel-resumable-bulk); 0 at holes. 8 bytes/doc, declared
                              #   in the QG-7 bytes/doc budget.
blob: concatenated DocId (CompactString) bytes
```

Materialization = two offset reads + one slice. Holes (equal adjacent offsets) mark absent docids after compaction.

### 5.7 IDHASH

Open-addressed, linear-probe table for `DocId → docid` (upsert/delete/resume probes):

```
{ capacity: u32 (power of two, load factor <= 0.7), entry: capacity × { key_hash: u64, docid_plus1: u32 } }
```

`key_hash` = xxh3-64 of the DocId bytes with **seed = 0x5155_494C_4C31 ("QUILL1")** — a fixed spec constant, never process-random. Empty slot: `docid_plus1 == 0`. Probe: `h & (capacity-1)`, linear. **Every hit must be verified against IDMAP bytes** (64-bit collisions are legal). Cross-segment resolution order: newest segment first (manifest order), skipping tombstoned docids; at most one live docid per DocId (upsert invariant).

### 5.8 NUMERIC

Per indexed numeric field:
`{ field_ord: u16, count: u32, pairs: count × { value_bits: u64, docid: u32 } }`.
The schema is the type tag: `I64` interprets `value_bits` as the little-endian
two's-complement bytes of an `i64`, while `U64` interprets the same eight bytes
as an unsigned `u64`. No redundant per-row tag is encoded. Fields appear in
ascending schema ordinal and pairs are sorted by the field's typed `(value,
docid)` order, so the full `u64` domain (including values above `i64::MAX`)
round-trips losslessly. A range filter binary-searches typed bounds and emits a
docid predicate.

### 5.9 STOREDMETA

The schema-derived stored fields appear in ascending `field_ord`; there is no
redundant count prefix:

```
directory: stored_field_count × {
  field_ord: u16
  field_offset: u32          # section-relative; first = directory length
}

each field payload at field_offset:
  presence: ceil(span / 8) bytes       # LSB-first; 1 = present
  offsets:  (span + 1) × u32           # blob-relative, monotone, first = 0
  blob:     offsets[span] opaque bytes
```

Fields are packed without padding. Every later `field_offset` equals the exact
end of the preceding field blob; the final blob ends at the section boundary.
Unused high bits in the last presence byte are zero. A zero presence bit
requires equal adjacent offsets and represents an absent field or docid hole.
A one bit with equal offsets represents a **present empty value**, which must
remain distinguishable from absence. Per-field offsets and the presence bitmap
remain positional over `docid - docid_lo`, including holes after compaction.

All fields whose descriptor sets `stored=true` are represented; bytes are
opaque and serde_json parsing happens only at hit materialization. Ordinary
query paths do not request this lazy section and therefore never fault it in or
validate its section checksum. Fresh seal maps Scribe's completed-document
columns through the lease base into the exact segment docid span, inserting
holes for sparse local ordinals without constructing dense value objects.
Concat-merge rebases offset tables, inserts zero presence bits for
inter-segment gaps, and copies each source field blob once without interpreting
or re-encoding values.

### 5.10 STATS

Per indexed field: `{ field_ord: u16, total_tokens: u64, doc_count: u32 }`.
**Semantics:** counts are **at-seal** values and include docs later tombstoned
(oracle-mirroring: tantivy does not discount deletes from stats until merge;
snapshot-level aggregation sums these across live segments; compaction
re-derives them). Every indexed-field row's `doc_count` equals the segment
header's at-seal `doc_count`. BM25 `avgdl` is the raw
`total_tokens / doc_count` ratio; fieldnorm decoding applies only to an
individual document's length and never to the STATS aggregate.

## 6. MANIFEST

### 6.1 Layout

```
magic "FSLXMAN\0" | format_version: u32 = 1 | generation: u64
docid_high_watermark: u64 | schema_id: u64 | engine_version: u32
flags: u32 (bit 0 = bulk_mode_in_progress)
segment_count: u32
segments: segment_count × {
  segment_id: u64, seal_seq: u64, file_len: u64, file_xxh3: u64,
  docid_lo: u64, docid_hi: u64, doc_count: u32,
  tombstones: tombstone-set bytes (§6.3)
}
field_count: u32
stats rollup: field_count × { field_ord: u16, total_tokens: u64, doc_count: u32 }   # sums over segments
crc32: u32 (over all preceding bytes)
```

`engine_version` packs the crate semver as `(major:u8 << 24) | (minor:u8 << 16) | patch:u16`; prerelease/build metadata is not encoded. Segments are listed in **ascending docid_lo** order. R2 merge order follows that range order, while IDHASH newest-first probe order uses descending `seal_seq` (the per-segment recency witness added by amendment v1.0.1). Stats entries are listed in strictly ascending `field_ord` order. Across adjacent generations, retained segment metadata is immutable, retained tombstones only grow, and every newly referenced segment has a `seal_seq` greater than the maximum in the prior non-empty generation. An explicit empty generation starts a new seal-sequence epoch; restarting at 1 is valid because no older segment remains queryable or participates in newest-first probing. If the immutable segment set is unchanged, its at-seal stats rollup is unchanged as well; tombstone-only generations do not rewrite BM25 statistics.

### 6.2 Publish protocol (normative, crash-window exact)

1. Write `MANIFEST` (gen N+1) content to `.tmp-manifest-<gen>`; fsync file. A retry after a failed claim or rename may reuse that temp only when a no-follow open proves the directory entry is a regular file and its complete bytes exactly match the canonical proposal, then fsyncs it again. Any symlink, non-regular, or mismatched temp fails closed without overwrite and is left for writer-locked recovery/GC.
2. Claim the generation: create `gen-<N+1>.claim` with `O_CREAT|O_EXCL` (fails ⇒ another writer won; abort with typed error — the CAS of bead `bd-quill-duel-writer-lock`). E3.2 exposes the checked pre-publish seam and implements the in-process mutex plus steps 1 and 3–5; the cross-process claim/release implementation lands with that dependent writer-lock bead.
3. `rename(MANIFEST, MANIFEST.prev)` (skip if no MANIFEST — genesis).
4. `rename(.tmp-manifest-<gen>, MANIFEST)`.
5. fsync directory. 6. Unlink the claim file (best-effort; stale claims for generations ≤ current are GC-eligible under the writer lock).

The E3.2 implementation currently admits publication only on Unix, where replacement rename and directory fsync provide these exact guarantees; unsupported platforms fail closed before creating the temp file until an equivalent safe backend lands.

**Recovery rules (open):** valid `MANIFEST` → use it. Missing/corrupt `MANIFEST` + valid `MANIFEST.prev` → **mid-publish crash**: recover from prev (generation must be exactly current−1 of any claim present, else typed corruption error). Neither present → `IndexNotFound` (distinct error from corruption). Readers never write during recovery; writer-open completes the interrupted publish or rolls forward per Q1 validation.

### 6.3 Tombstone-set encoding (roaring-lite)

```
u32 chunk_count, chunks: chunk_count × {
  chunk_id: u16            # docid >> 16
  kind: u8                 # 0 = ARRAY, 1 = BITMAP
  count: u16               # cardinality within chunk (0 encodes 65536 for kind=1)
  payload: ARRAY -> count × u16 (sorted low-16 bits) | BITMAP -> 8192 bytes
}
```

Promotion threshold 4096 entries (array→bitmap), demotion at 3584 (hysteresis — bead e3.4).

## 7. Coordination artifacts

### 7.1 LOCK (writer lock; bead `bd-quill-duel-writer-lock`)

```
magic "FSLXLCK\0" | format_version: u32 = 1 | pid: u32 | pid_start_nonce: u64 | acquired_unix_s: i64 | crc32
```

`pid_start_nonce` = xxh3 of (pid, process start time from /proc where available, else acquired timestamp) — guards PID reuse. Takeover permitted **only** when `kill(pid, 0)` reports ESRCH (process dead). mtime/heartbeat staleness alone **never** authorizes takeover (SIGSTOP hazard). Readers ignore LOCK except to report `live_writer` in freshness surfaces.

### 7.2 gen-*.claim

Zero-length files; existence is the semantics (O_EXCL create = atomic claim). Named by the generation they claim.

### 7.3 CURRENT (blue-green pointer; bead `bd-quill-duel-blue-green`)

Lives in the lexical root (one level above engine dirs):

```
magic "FSLXCUR\0" | format_version: u32 = 1 | engine_kind: u8 (1=quill, 2=tantivy)
dir_name_len: u16 | dir_name bytes | index_format_version: u32 | crc32
```

Published with the same temp+rename+dir-fsync discipline as MANIFEST. Absent CURRENT + exactly one engine dir → adopt it and write CURRENT (migration bootstrap); absent + multiple dirs → typed error demanding doctor.

## 8. Garbage collection safety

GC runs **only under the writer LOCK** (readers never GC — bead `bd-quill-duel-writer-lock`). Eligible: `.tmp-*`, `seg-*.fslx` not referenced by MANIFEST or MANIFEST.prev **and** older than a grace window (default 300 s), orphaned `.fec` for deleted quill artifacts, stale `gen-*.claim` ≤ current generation. **Never** eligible: `.quarantine` files, anything not matching §1's name schemas, anything outside the index dir (path-safety guard: reject `..`/absolute). RULE-1 posture: the engine deletes only its own garbage by schema; user files are structurally unreachable.

## 9. Cross-references

- Q1 invariant + R1/R2: `docs/contracts/quill-q1-docid-discipline.md` (bead e0.3).
- Language/scoring contract (analyzers, BM25, tie-break, conformance classes): `docs/contracts/quill-language-contract.md` (bead e0.1).
- Divergence Register: `docs/contracts/quill-divergence-register.md` (bead e0.4).
- Vendored fieldnorm + BM25 constants: `crates/frankensearch-lexical/src/quill_contract.rs` (final home `frankensearch_quill::contract`, moves with e1.0).

## 10. Section → implementing bead checklist

| Section/artifact | Bead |
|---|---|
| Segment container (header/table/checksums/mmap-or-owned reader) | quill-e3.1 |
| TERMDICT | quill-e2.1 |
| POSTINGS | quill-e2.2 |
| POSITIONS | quill-e2.4 |
| BLOCKMAX | quill-e2.3 |
| DOCLEN + STATS | quill-e2.5 |
| IDMAP (incl. content_hash) + IDHASH | quill-e2.6 (+ bd-quill-duel-resumable-bulk) |
| NUMERIC | quill-e2.7 |
| STOREDMETA | bd-quill-e2-9-storedmeta-9jkw |
| MANIFEST codec + in-process serialized two-slot publish (steps 1, 3–5) | quill-e3.2 |
| Tombstone sets | quill-e3.4 |
| LOCK + generation claim/release (step 2) + GC ownership | bd-quill-duel-writer-lock (+ e3.3) |
| CURRENT | bd-quill-duel-blue-green |
| v1 golden segment fixture | quill-e2.8 |

## 11. Test fixtures required by this spec

1. **v1 golden segment** (committed, small): parses forever; future versions must read it or ship a registry migration row (e2.8).
2. **Torn-file matrix**: truncate at every 1 KiB boundary → typed errors (e3.1).
3. **Publish crash-window matrix**: kill between every §6.2 step; recovery rules hold (e3.3/e3.9 + writer-lock bead multi-process rows).
4. **Checksum flip sweep**: corrupt each byte class (header/section/trailer/manifest/tombstones) → typed error, never panic (e2.8).
5. **Encoding edge fixtures**: width-0 and width-32 FOR blocks, bitmap threshold spans 511/512, VINT partial blocks of 1 and 127 at final and interior positions, `payload_len` truncation/trailing-byte rejection, the Q1 `df=100 + df=300` raw-block-concat case, zero-delta positions, 256-byte terms, empty DocId rejection, holes in DOCLEN/IDMAP/STOREDMETA after simulated compaction.

## 12. Format registry (append-only)

| Version | Date | Change | Migration | Fixture |
|---|---|---|---|---|
| 1.0.0 | 2026-07-17 | Initial FSLX v1: segment container + sections 1–10; MANIFEST v1 + two-rename/claim publish; tombstone roaring-lite; LOCK/claim/CURRENT artifacts; GC safety rules | — (genesis) | golden segment (e2.8, pending) |
| 1.0.1 | 2026-07-17 | `seal_seq: u64` added to MANIFEST segment entries (recency for IDHASH newest-first probe order and forensics) — folded into v1 pre-freeze | — (pre-freeze fold) | manifest roundtrip test |
| 1.0.2 | 2026-07-17 | IDMAP gains per-doc `content_hash: u64` (resumable-bulk witness + doctor audit; bd-quill-duel-resumable-bulk) — folded into v1 pre-freeze | — (pre-freeze fold) | resume-equivalence fixture (pending) |
| 1.0.3 | 2026-07-17 | POSTINGS blocks gain a common `{kind, posting_count, payload_len}` header; VINT partials are legal at preserved interior seams so Q1 merges copy blocks verbatim without cascading re-encode; `doc_width` is explicitly one byte covering 0..=32 | — (pre-freeze correction) | `df=100 + df=300` concat + scalar/SIMD bitpack differential (e2.2) |
| 1.0.4 | 2026-07-17 | MANIFEST v1 pre-freeze correction: place `seal_seq` in each segment entry, encode `field_count: u32`, define packed-semver `engine_version`, and split E3.2's in-process two-slot publish from the dependent cross-process generation-claim CAS | — (pre-freeze correction) | manifest wire golden + roundtrip/recovery tests (e3.2) |
| 1.0.5 | 2026-07-17 | Tombstone roaring-lite `chunk_count` widened from `u16` to `u32`; a full u32 docid-domain set can contain all 65,536 non-empty chunks, which `u16` could not represent — folded into v1 pre-freeze | — (pre-freeze correction) | manifest hostile-input and tombstone canonicality tests (e3.2/e3.4) |
| 1.0.6 | 2026-07-17 | MANIFEST adjacent-generation invariants pinned: retained segment metadata is immutable, tombstones are monotone, new `seal_seq` values advance the prior non-empty maximum, an explicit empty generation starts a new seal epoch, and tombstone-only generations preserve the at-seal stats rollup | — (pre-freeze contract hardening) | manifest transition publish/reopen tests (e3.2) |
| 1.0.7 | 2026-07-17 | MANIFEST temp retry semantics pinned: a no-follow regular, byte-identical durable `.tmp-manifest-<gen>` is reusable after claim/rename failure; any symlink, non-regular, or mismatched temp fails closed without overwrite until writer-locked recovery/GC | — (pre-freeze protocol hardening) | manifest claim-failure retry, mismatch, and symlink tests (e3.2/e3.3) |
| 1.0.8 | 2026-07-17 | TERMDICT v1 pre-freeze correction: block offsets are relative to the blocks-area start; terms are arbitrary 0..=65,530-byte strings; payload domains, contiguous spans, explicit `blockmax_len`, and schema-derived positions are pinned; no undefined flags word is encoded; canonical greedy 4096-byte splitting with oversized singletons, maximal-prefix compression, half-open ranges, and bounded in-memory restart metadata are required | — (pre-freeze correction) | termdict wire goldens, restart-bound lookup, field/prefix/range, corruption, span-bound, and budget tests (e2.1) |
| 1.0.9 | 2026-07-17 | Segment-container v1 pre-freeze hardening: canonical header length and exact section set/flags are pinned; section ranges use strict kind order, checked non-overlap, minimal 64-byte alignment, and zero padding; the trailer is adjacent to the final payload; IEEE CRC-32 and eager structural/header/trailer validation are explicit; section hashes remain first-touch lazy while `verify()` checks every section plus the whole-file xxh3 | — (pre-freeze validation hardening) | segment wire golden, torn-file matrix, and header/section/trailer corruption sweep (e3.1/e2.8) |
| 1.0.10 | 2026-07-17 | DOCLEN/STATS v1 pre-freeze correction: holes use canonical `0x00` but membership never derives from a fieldnorm byte because all u8 values are valid; each indexed-field STATS row uses the segment at-seal doc count and BM25 avgdl is raw `total_tokens / doc_count`, not a fieldnorm-decoded aggregate | — (pre-freeze correction) | DOCLEN hole/membership fixtures and STATS avgdl/row-count differential (e2.5) |
| 1.0.11 | 2026-07-17 | Segment-container publication hardening: non-empty segment ranges are bounded by the u32 payload domain; public writers reject unregistered extension kinds while readers retain the optional-skippable forward valve; mmap open is restricted to canonical regular published segment paths after sync-and-rename, and no unchecked reader API exposes payload bytes before first-touch verification | — (pre-freeze validation hardening) | pinned inline wire oracle, writer/reader extension differential, u32 boundary, published-mmap parity, and typed torn-file tests (e3.1) |
| 1.0.12 | 2026-07-17 | NUMERIC signedness correction: schema-driven `value_bits: u64` preserves the complete indexed I64 and U64 domains without a redundant row tag; ordering and bounds use the field's schema type | — (pre-freeze correction) | indexed-I64/indexed-U64 section-presence coverage (e3.1); typed range/roundtrip including `u64 > i64::MAX` (e2.7) |
| 1.0.13 | 2026-07-17 | BLOCKMAX v1 pre-freeze hardening: no count prefix; u64 canonical term-relative offsets; Tantivy-compatible conservative max-frequency codes; exact minimum stored fieldnorm IDs; POSTINGS/DOCLEN cross-validation; live-snapshot query bounds; and Q1 concat offset-only rebasing are pinned | — (pre-freeze contract hardening) | BLOCKMAX wire/corruption, bound-soundness, skip, and `df=100 + df=300` concat fixtures (e2.3) |
| 1.0.14 | 2026-07-17 | POSITIONS v1 pre-freeze hardening: fixed-LE block count; canonical u32 posting ordinals and u64 blocks-area-relative offsets; whole-document u32 first-plus-delta runs with zero deltas; frequency-derived boundaries; greedy 4096-byte fresh blocks with oversized singletons; bounded validation; and Q1 payload-preserving directory rebasing are pinned | — (pre-freeze contract hardening) | POSITIONS wire/corruption, zero-delta, >65k, block-boundary, cursor, and concat-seam fixtures (e2.4) |
| 1.0.15 | 2026-07-18 | STOREDMETA v1 pre-freeze correction: every schema-stored field gets a packed presence bitmap before its positional offset table, preserving absent/hole versus present-empty values; exact directory offsets, bitmap canonicality, zero-copy reads, sparse Scribe sealing, and gap-preserving concat are pinned | — (pre-freeze correction) | exact wire golden, absent/empty/non-UTF8 roundtrip, sparse accumulator holes, corruption/resource matrix, lazy first-touch, and direct/left/right concat equivalence (e2.9) |

*Process: pre-freeze changes fold into v1 with a registry row (as above). Post-freeze changes bump `format_version`, add a migration note, and land a reader-compat gauntlet fixture in the same commit.*
