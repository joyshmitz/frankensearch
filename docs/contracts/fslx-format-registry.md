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
- **Checksums:** each section carries xxh3-64 over its exact `len` bytes (recorded in the section table). The segment header carries `header_crc32` (CRC32 over header bytes). The file trailer carries `file_xxh3` (xxh3-64 over all bytes from offset 0 to the start of the trailer) + `trailer_crc32` (CRC32 over the trailer's preceding 8 bytes). The `.fec` fast-path verify uses `file_xxh3` — one hash pass shared with durability.
- **Varints ("vint"):** LEB128, max 10 bytes, canonical (no over-long encodings; readers reject).
- **Strings/blobs:** raw bytes, never NUL-terminated; lengths always explicit.
- **Docids:** u32 in all postings/section payloads; u64 in header/manifest fields (future-proofing). Docid semantics are governed by Q1 (`docs/contracts/quill-q1-docid-discipline.md`): monotone allocation, session leases, burned tails, **never reused**; R1 = a segment's docid range is a subinterval of one lease block; R2 = merges combine only bound-consecutive runs.
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

Unknown section kinds: readers **skip** kinds > 10 whose table entries carry flag bit 0 (`OPTIONAL_SKIPPABLE`) and **reject** otherwise — forward-compatibility valve, registry row required to use it.

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

Validation on open: magic, version, header CRC **eagerly**; section xxh3 **lazily** on first touch (plus an eager `verify()` mode for doctor flows). Any failure → typed corruption error (`SearchError`-mapped), never a panic. Readers are generic over the byte source (`&[u8]`): mmap-backed on disk, owned buffers for `in_memory()` (bead e3.1 notes).

## 5. Section payloads

### 5.1 TERMDICT

Composite key = `field_ord: u16 (BIG-endian)` ++ `term bytes`. Raw-byte ordering of composite keys equals (field, term) lexicographic order; per-field iteration = range scan on the 2-byte prefix.

```
u32 block_count
block index: block_count × { first_key_len: vint, first_key: bytes, block_offset: vint (from dict start) }
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
```

Lookup: binary-search block index (on `first_key`), scan one block (≤ R-entry linear between restarts). Ordered iteration and prefix/range scans are first-class (glob support).

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

Per term (fields with positions only): per-doc position runs, doc-aligned with POSTINGS order; run length for a doc = that doc's freq. Positions are u32, delta-encoded within the doc (first absolute, then `delta` — **zero deltas legal**: same-position duplicate tokens, e.g. HyphenDecompose), vint-encoded, grouped in blocks of 4096 bytes with a vint skip header `{doc_index: vint, offset: vint}` every block for phrase-time seeking.

### 5.4 BLOCKMAX

Per term, one entry per POSTINGS block (including every full or partial block,
regardless of position):

```
{ first_doc: u32, block_offset: vint (from term's postings_offset),
  max_freq_q: u8,        # quantized UP via the monotone freq table (round-up; table in quill contract module)
  min_fieldnorm: u8 }    # minimum fieldnorm id over docs in the block
```

**Soundness contract:** impact bounds are computed at **query time** from `(max_freq_q, min_fieldnorm)` with the live snapshot's idf/avgdl — never stored as impact scalars (stored impacts are unsound under changing avgdl; see plan §10.4 and bead e2.3). `max_freq_q` decodes to a value ≥ the true block max freq; `min_fieldnorm` decodes to a length ≤ the true block min. Together they dominate every (freq, |d|) in the block for BM25's tf_part, which is increasing in freq and decreasing in |d|.

### 5.5 DOCLEN

Per field with `Text`/`Keyword` indexing: `doc_count_span = docid_hi - docid_lo` fieldnorm bytes, direct-indexed by `docid - docid_lo`. **Holes:** positions for docids absent from the segment (burned/tombstone-folded) hold `0xFF` (never a valid norm for a present doc; readers must not score absent docs — cursors only visit present docids, so this is a debugging/audit aid, not a correctness mechanism). Post-compaction segments keep positional indexing with holes; space overhead is bounded by the compaction-density threshold.

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

Per indexed i64 field: `{ field_ord: u16, count: u32, pairs: count × { value: i64, docid: u32 } }`, sorted by (value, docid). Range filter = binary-search bounds → docid predicate. Field directory as in §5.5.

### 5.9 STOREDMETA

Per stored field (schema-descriptor-driven; `metadata_json` always; `content` iff descriptor stores it): field directory, then per-field `(span + 1) × u32` offsets + blob, holes as in IDMAP. Bytes are opaque (serde_json parsing happens only at hit materialization). Lazy section: queries not touching stored fields never fault it in.

### 5.10 STATS

Per field: `{ field_ord: u16, total_tokens: u64, doc_count: u32 }`. **Semantics:** counts are **at-seal** values and include docs later tombstoned (oracle-mirroring: tantivy does not discount deletes from stats until merge; snapshot-level aggregation sums these across live segments; compaction re-derives them). `avgdl` inputs decode through the fieldnorm table (contract: quill_contract.rs conventions).

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

*Process: pre-freeze changes fold into v1 with a registry row (as above). Post-freeze changes bump `format_version`, add a migration note, and land a reader-compat gauntlet fixture in the same commit.*
