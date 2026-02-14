# fsfs File Classification and Encoding Contract v1

Issue: `bd-2hz.2.2`  
Parent: `bd-2hz.2`

## Goal

Define deterministic file eligibility classification for ingestion with explicit semantics for:

- text vs binary sniffing
- encoding detection and normalization fallback
- corrupt/partial file handling
- confidence signals required by downstream utility scoring

## Normative Terms

- `MUST`: hard requirement
- `SHOULD`: expected default
- `MUST NOT`: forbidden behavior

## Classification Pipeline

`fsfs` MUST evaluate files in this order:

1. probe and sniff byte-level signals (null bytes, non-printable ratio, high-bit ratio)
2. classify coarse type (`text`, `binary`, `archive`, `partial`, `corrupt`)
3. for text-like files, detect encoding and apply normalization policy
4. map classification to ingest action (`index`, `skip`, `quarantine`, `index_partial_with_flag`)
5. emit confidence and reason-code fields for auditability

## Binary/Text Sniff Heuristics

The policy MUST include configurable thresholds for:

- `max_probe_bytes`
- `binary_byte_threshold_pct`
- `high_bit_ratio_threshold_pct`
- hard binary trigger on null-byte presence

Binary defaults SHOULD bias toward cost control: uncertain high-cost files default to `skip` or `quarantine`, not full indexing.

## Encoding and Normalization Policy

For text-like content, the detector chain MUST be ordered and deterministic:

1. BOM detection
2. UTF-8 validation
3. heuristic detector fallback

Unknown/low-confidence encodings MUST follow explicit fallback actions:

- `skip`
- `lossy_decode`
- `quarantine`

Normalization policy MUST specify whether NFC normalization is strict (`utf8_nfc`) or lossy (`utf8_nfc_lossy`), and this must be surfaced in emitted decision records.

## Corrupt and Partial Handling

`fsfs` MUST explicitly represent file integrity classes:

- `corrupt`: checksum/decode failures that cannot be trusted
- `partial`: short/truncated inputs where prefix recovery is possible

Corrupt files MUST NOT be indexed as normal text. Partial files MAY be indexed only via `index_partial_with_flag` and MUST carry reason-coded degradation metadata.

For `binary`, `archive`, and `corrupt` classifications:

- `detected_encoding` MUST be `none`
- `normalization_applied` MUST be `none`

For `archive` and `corrupt` classifications, `ingest_action` MUST be `skip` or `quarantine`.

## Required Confidence Signals

Every classification decision MUST include:

- `classification_confidence` in `[0,1]`
- `encoding_confidence` in `[0,1]`
- `reason_code` (`FSFS_*`)
- downstream utility signals (`utility_penalty`, `skip_candidate`, `requires_manual_review`)

These fields are mandatory for downstream utility scoring and operator diagnostics.

## Validation Artifacts

- `schemas/fsfs-file-classification-v1.schema.json`
- `schemas/fixtures/fsfs-file-classification-contract-v1.json`
- `schemas/fixtures/fsfs-file-classification-decision-v1.json`
- `schemas/fixtures/fsfs-file-classification-corrupt-event-v1.json`
- `schemas/fixtures-invalid/fsfs-file-classification-invalid-*.json`
- `scripts/check_fsfs_file_classification_contract.sh`

## Validation Command

```bash
scripts/check_fsfs_file_classification_contract.sh --mode all
```
