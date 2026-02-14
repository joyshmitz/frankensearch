# Native Mode Distributed Search Architecture (Future)

Status: Draft (design-only)
Owner bead: `bd-3w1.19`
Scope: Future Tier-4 architecture for distributed frankensearch using FrankenSQLite Native Mode ECS commits.

## 1. Problem Statement

Today, frankensearch is optimized for single-node deployment:
- Source of truth: local metadata/storage + index artifacts.
- Retrieval path: lexical + vector + fusion in one process.
- Durability: local files with optional repair wrappers.

Future multi-node deployments require:
1. Deterministic replication semantics for search state.
2. Self-healing durability under node/storage loss.
3. Bounded-latency query service with local reads.
4. Explicit rollback/degraded modes.

## 2. Goals and Non-Goals

Goals:
1. Define Native Mode storage/object model for distributed search.
2. Define consistency semantics for writes, replication, and reads.
3. Define integration boundaries with compatibility-mode pipeline.
4. Define failure handling, repair, and observability contract.
5. Define staged migration path and follow-up implementation beads.

Non-goals:
1. No implementation code in this bead.
2. No protocol wire-format freeze in this bead.
3. No production SLO commitment for cross-region deployments yet.

## 3. Baseline Concepts

Native Mode model (FrankenSQLite):
- Every mutation is a content-addressed `CommitCapsule` with `ObjectId = BLAKE3(payload)`.
- Durable data is represented as ECS objects with erasure coding metadata.
- Commit finality is driven by commit marker/quorum semantics, not local file writes.

Search system model (frankensearch):
- Semantic artifacts: FSVI vector shards + embedder metadata.
- Lexical artifacts: Tantivy segments.
- Pipeline: canonicalize -> classify -> lexical/vector retrieval -> RRF -> optional rerank.

## 4. Architecture Options

### Option A: Document-Operation Replication (rebuild on each node)

Flow:
1. Writer emits document-level `CommitCapsule` operations.
2. Replicas ingest capsules into local Native Mode state.
3. Each replica runs local embedding + index build/update.

Pros:
- Minimal artifact transfer.
- Full determinism from logical history.
- Natural support for time-travel replay.

Cons:
- Expensive repeated embedding compute on every replica.
- Slower bootstrap and catch-up for large corpora.

### Option B: Index-Artifact Replication (recommended initial distributed path)

Flow:
1. Writer builds validated index artifacts (FSVI/Tantivy) once.
2. Artifact chunks are encoded into deterministic repair symbols.
3. Replicas fetch/repair artifact set and atomically activate generation.

Pros:
- Lower replica compute; faster convergence.
- Predictable bootstrap and backfill times.
- Easier near-term adoption for current architecture.

Cons:
- Higher network/storage transfer volume.
- Requires artifact manifest/version coordination.

Recommendation:
- Phase 1 distributed rollout should prioritize Option B.
- Option A remains strategic for full Native Mode replay and advanced auditing.

## 5. Core Data Model

### 5.1 Commit and Artifact Identifiers

- `commit_seq` (monotonic logical order).
- `commit_id` (content-derived capsule id).
- `generation_id` (search generation built from a commit window).
- `artifact_manifest_id` (hash of artifact manifest).

### 5.2 Search Generation Manifest

Each generation manifest must include:
1. Parent `commit_seq` range.
2. Embedder identity + revision hash.
3. Vector artifact list (FSVI shards, dimensions, quantization).
4. Lexical artifact list (segment files/checksums).
5. Repair symbol descriptors and overhead.
6. Activation invariants (all-or-nothing readiness predicates).

### 5.3 Read Snapshot Contract

At query time, node selects one active `generation_id` and keeps it stable for the request lifecycle.
- No mixed-generation reads in a single query.
- Time-travel reads select historical generation by `commit_seq <= snapshot_high`.

## 6. Consistency Semantics

Write consistency:
1. Writer appends commit capsule.
2. Capsule reaches durability threshold (quorum or local-ack mode by deployment class).
3. Generation build process materializes search artifacts.
4. Activation marker flips atomically from `generation N` to `generation N+1`.

Read consistency:
- Default: read-your-cluster-latest-activated-generation.
- Optional strict mode: block until `commit_seq >= requested_seq` is activated.
- Time-travel mode: explicit `as_of_commit_seq` query option.

Conflict handling:
- Single logical writer per shard in initial model.
- Multi-writer support deferred until Native Mode conflict-resolution primitives are production-ready.

## 7. Replication and Repair

### 7.1 Symbol Strategy

- Deterministic repair symbol generation seeded by artifact/object id.
- Configurable overhead target: default 20% for artifact chunks.
- Receiver can reconstruct once it has sufficient source+repair symbols.

### 7.2 Bootstrap

New node bootstrap sequence:
1. Pull latest generation manifest.
2. Fetch artifact symbols from peers/store.
3. Decode + verify checksums.
4. Atomically activate generation locally.
5. Begin incremental catch-up stream.

### 7.3 Incremental Catch-up

- Consume generation deltas (preferred) or commit stream replay (fallback).
- Enforce bounded lag alerts when generation activation falls behind threshold.

## 8. Failure Handling and Recovery

Failure classes and responses:

1. Partial artifact corruption on node:
- Detect via periodic checksum scan or read-time verification.
- Repair by symbol reconstruction from peers/object store.
- If repair fails: demote node to read-degraded state.

2. Node loss:
- Rehydrate from latest manifest + symbols.
- Resume as replica after generation activation.

3. Writer failure during generation build:
- Incomplete generation remains inactive.
- Last active generation continues serving traffic.
- Recovery resumes from last successful `commit_seq` watermark.

4. Divergent generation manifests:
- Reject activation if manifest hash mismatch against signed control-plane record.
- Emit critical event and quarantine divergent artifacts.

5. Dependency model drift (embedder mismatch):
- Activation blocked when embedder revision in manifest != node runtime contract.
- Requires explicit migration/override workflow.

## 9. Observability Requirements

Required metrics:
1. `distributed.commit.lag_ms`
2. `distributed.generation.activate_ms`
3. `distributed.artifact.repair_ratio`
4. `distributed.snapshot.bootstrap_ms`
5. `distributed.query.generation_skew`
6. `distributed.repair.failures_total`

Required structured events:
1. `generation_build_started`
2. `generation_build_completed`
3. `generation_activation_succeeded`
4. `generation_activation_failed`
5. `artifact_repair_started`
6. `artifact_repair_completed`
7. `artifact_repair_failed`
8. `read_degraded_mode_entered`
9. `read_degraded_mode_exited`

Tracing requirements:
- Span root per generation lifecycle with child spans for encode/transfer/decode/verify/activate.
- Query spans must include `generation_id` and `as_of_commit_seq` tags.

## 10. Performance and Operational Tradeoffs

Expected tradeoffs (directional targets for planning):

| Dimension | Option A (rebuild everywhere) | Option B (artifact replication) |
|---|---|---|
| Replica CPU | High (embedding on each node) | Low-medium |
| Network transfer | Lower | Higher |
| Bootstrap latency | Higher | Lower |
| Deterministic replay depth | Excellent | Good (manifest-driven) |
| Operational complexity | Medium | Medium-high |

Planning assumptions:
1. Option B can reduce replica bootstrap from hours to minutes for large corpora where embedding dominates build time.
2. Option B shifts cost to network/object storage; capacity planning must include repair overhead.
3. Time-travel reads require retained generations; retention policy drives storage growth.

## 11. Integration Boundaries with Current Pipeline

Compatibility mode remains default until Tier-4 is explicitly enabled.

Boundary rules:
1. Existing `TwoTierSearcher` query behavior remains unchanged.
2. Distributed activation feeds existing local index readers via generation pointer swap.
3. Existing storage/index APIs gain optional generation context; no mandatory breaking change for local-only users.
4. Degraded mode must preserve safe local serving from last valid generation.

## 12. Staged Migration Plan

Phase 0: Instrumentation and metadata contracts
- Add generation manifest schema and event taxonomy.

Phase 1: Artifact replication MVP (recommended)
- Build/ship/activate generation artifacts with repair support.

Phase 2: Multi-node operational hardening
- Add repair automation, quorum policies, and health-based routing.

Phase 3: Time-travel query support
- Expose `as_of_commit_seq` against retained generations.

Phase 4: Commit-stream native rebuild path
- Introduce document-operation replay for deep audit/reconstruction.

## 13. Risks and Prerequisites

Prerequisites:
1. Stable Native Mode APIs for commit and object access.
2. Generation manifest/version governance.
3. Deterministic embedder revision pinning policy.

Key risks:
1. Cross-node embedder drift causing ranking inconsistency.
2. Symbol/object store hot spots under repair storms.
3. Activation race conditions without strict manifest validation.
4. Operational complexity if both Option A and B are enabled simultaneously too early.

Risk mitigations:
1. Hard embedder revision checks in activation gate.
2. Backpressure and circuit-breakers on repair workers.
3. Single active generation invariant with rollback pointer.
4. Feature-flagged rollout by environment.

## 14. Rollback and Degraded Mode

Rollback:
- Keep `generation N` artifacts available until `generation N+1` passes verification and warmup checks.
- On activation failure, atomically revert pointer to last healthy generation.

Degraded mode:
- Serve reads from last healthy generation.
- Suspend new activation attempts when corruption/failure thresholds exceed policy.
- Continue ingest in compatibility-safe mode when possible.

## 15. Follow-up Bead Decomposition (proposed)

1. `bd-o26q` (task, P2): Define generation manifest schema + validator.
Depends on: `bd-3w1.1`

2. `bd-163p` (task, P2): Build artifact replication + activation controller MVP.
Depends on: `bd-o26q`

3. `bd-20ic` (task, P2): Implement repair orchestration and degraded-mode routing hooks.
Depends on: `bd-163p`

4. `bd-dbys` (task, P3): Add time-travel query API (`as_of_commit_seq`) across retained generations.
Depends on: `bd-163p`

5. `bd-33zf` (task, P3): Implement commit-stream replay path for full Native Mode reconstruction.
Depends on: `bd-o26q`, `bd-20ic`

6. `bd-a2zj` (task, P2): Add distributed observability package (metrics/events/traces + runbook).
Depends on: `bd-163p`

## 16. Decision Summary

1. Initial distributed architecture should prioritize index-artifact replication (Option B).
2. Native commit-stream replay remains strategic and should be added in later phases.
3. Generation activation invariants, manifest validation, and embedder revision pinning are mandatory correctness controls.
4. Compatibility-mode pipeline remains intact until staged distributed gates are met.
