# fsfs Incremental Change-Detection Contract v1

Issue: `bd-2hz.2.5`  
Parent: `bd-2hz.2`

## Goal

Define deterministic, crash-safe incremental change detection for file updates so `fsfs` can avoid unnecessary rescans while preserving correctness.

This contract specifies:

- mtime/size/hash tradeoff policy
- rename/move detection semantics
- crash/restart recovery behavior for pending changes
- stale-state reconciliation guarantees

## Normative Terms

- `MUST`: hard requirement
- `SHOULD`: expected default
- `MUST NOT`: forbidden behavior

## Detection Pipeline

`fsfs` MUST evaluate change candidates in this order:

1. `mtime+size` fast path filter
2. hash confirmation for ambiguous updates
3. rename/move identity matching
4. queue decision (`enqueue_embed`, `skip_no_change`, `mark_stale`, `reconcile_full`)
5. durable journal checkpoint update

## mtime / Size / Hash Tradeoff Policy

The policy MUST explicitly define:

- mtime granularity assumptions (`mtime_granularity_ns`)
- when size-only change can skip hashing
- when mtime-only change MUST trigger hash confirmation
- bounded fast-path skipping before forced hash/reconcile

If signals are ambiguous, correctness wins over speed: `fsfs` MUST escalate to hash-confirm or full reconcile.

## Rename and Move Semantics

Rename/move handling MUST support identity continuity via a deterministic key set (for example device/inode/content-hash tuple).

Rules:

- same-device rename SHOULD preserve identity and avoid full re-embedding when content unchanged
- cross-device move MUST be treated as either delete+create or hash-confirmed transfer, per policy
- rename/move decisions MUST emit both `rename_from` and `rename_to`

## Crash / Restart Recovery

Incremental decisions MUST be journaled with monotonic sequence IDs.

On restart, `fsfs` MUST:

- replay unapplied journal entries in ascending sequence order
- expose pending-change count and checkpoint watermark
- detect inconsistent checkpoint state and force reconcile when required

## Stale-State Reconciliation Guarantees

`fsfs` MUST run periodic reconciliation to bound drift between filesystem truth and index metadata.

Contract requires:

- configured full-scan interval
- stale-entry timeout
- deterministic action for orphaned metadata (`delete`, `quarantine`, or `mark_stale`)

## Required Decision Fields

Every change decision MUST include:

- `event_type`
- `detection_mode`
- `queue_action`
- `reason_code` (`FSFS_*`)
- confidence in `[0,1]`

Recovery checkpoints MUST include:

- `last_applied_seq`
- `pending_changes`
- `journal_clean`
- `action_on_restart`

## Validation Artifacts

- `schemas/fsfs-incremental-change-detection-v1.schema.json`
- `schemas/fixtures/fsfs-incremental-change-detection-contract-v1.json`
- `schemas/fixtures/fsfs-incremental-change-detection-decision-v1.json`
- `schemas/fixtures/fsfs-incremental-change-detection-recovery-v1.json`
- `schemas/fixtures-invalid/fsfs-incremental-change-detection-invalid-*.json`
- `scripts/check_fsfs_incremental_change_detection_contract.sh`

## Validation Command

```bash
scripts/check_fsfs_incremental_change_detection_contract.sh --mode all
```
