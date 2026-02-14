# fsfs Snippet/Highlight/Provenance Rendering Contract v1

Issue: `bd-2hz.5.3`  
Parent: `bd-2hz.5`

## Goal

Define deterministic snippet extraction, highlight semantics, and provenance payloads for trustworthy result interpretation across CLI and TUI surfaces.

This contract is normative for:

- snippet extraction and truncation behavior
- unicode-safe highlight range generation
- provenance fields required for auditability and explainability
- deterministic diagnostics and fallback metadata

## Normative Terms

- `MUST`: hard requirement
- `SHOULD`: expected default unless explicitly justified
- `MUST NOT`: forbidden behavior

## Snippet Extraction Strategy

## Input Classes

`fsfs` snippet generation MUST classify source content into:

- `code`
- `prose`
- `structured`
- `binary_or_unsupported`

For `binary_or_unsupported`, default action MUST be metadata-only rendering, not raw textual snippet generation.

## Windowing Policy

- snippets MUST be centered on best-matching spans
- context MUST be line-aware and deterministic
- snippets MUST be non-overlapping after merge
- per-result snippet count MUST be bounded
- long snippets MUST truncate on configured boundary mode:
  - `word_boundary_ellipsis`
  - `grapheme_boundary_ellipsis`

## Ordering

When multiple snippets exist, ordering MUST be stable:

1. ascending source byte offset
2. ascending line start
3. deterministic segment id tie-break

## Highlight Stability and Unicode Correctness

## Canonical Offset Unit

Highlight ranges MUST be emitted as UTF-8 byte offsets anchored to the original canonicalized text.

## Unicode Safety

- highlight boundaries MUST align to grapheme-cluster boundaries
- highlight generation MUST NOT split a multi-byte code point
- CJK and combining-mark sequences MUST preserve display integrity

## Range Merge Rules

- overlapping or adjacent ranges MUST be merged deterministically
- merged range type precedence MUST be deterministic
- range output order MUST be ascending by `(start_byte, end_byte, highlight_type)`

## Provenance Payload (Required Fields)

Each rendered segment MUST include provenance with at least:

- `path`
- `segment_id`
- `line_range`
- `byte_range`
- `index_revision`
- `content_hash`
- `score_contributors[]`
- deterministic `reason_code` for degraded or exceptional paths

`score_contributors[]` entries MUST preserve stable ordering and include `name`, `weight`, and `raw_score`.

## Degraded and Fallback Semantics

When normal snippet/highlight rendering cannot run safely, `fsfs` MUST emit one of:

- `lexical_only`
- `metadata_only`
- `safe_mode`

Fallback mode MUST be accompanied by:

- stable `reason_code`
- explicit `fallback_action`
- machine-readable diagnostics fields for replay/triage

## Required Diagnostics Fields

Render diagnostics MUST include:

- `event`
- `trace_id`
- `reason_code`
- `render_mode`
- `snippet_status`
- `unicode_safe`
- `offsets_verified`
- `emitted_fields[]`

## Reason-Code Namespace

This contract reserves deterministic reason-code families:

- `snippet.*`
- `highlight.*`
- `provenance.*`

## Validation Artifacts

- `schemas/fsfs-snippet-highlight-provenance-v1.schema.json`
- `schemas/fixtures/fsfs-snippet-highlight-provenance-contract-v1.json`
- `schemas/fixtures/fsfs-snippet-highlight-provenance-decision-v1.json`
- `schemas/fixtures-invalid/fsfs-snippet-highlight-provenance-invalid-*.json`
- `scripts/check_fsfs_snippet_highlight_provenance_contract.sh`

## Validation Command

```bash
scripts/check_fsfs_snippet_highlight_provenance_contract.sh --mode all
```
