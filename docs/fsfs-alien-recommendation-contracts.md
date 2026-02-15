# fsfs Alien Recommendation Contracts v1

Issue: `bd-2hz.1.4`  
Parent: `bd-2hz.1`

## Goal

Define reusable recommendation-contract cards for top adaptive fsfs controllers:

- ingestion policy
- degradation scheduler
- ranking policy

Each card is machine-readable and directly consumable by implementation and test-planning workstreams.

## Required Card Fields

Every recommendation card MUST include:

- `ev_score`
- `priority_tier`
- `adoption_wedge`
- budgeted mode + fallback trigger
- baseline comparator
- isomorphism proof plan
- reproducibility artifact requirements
- rollback plan

## Card Catalog

## Ingestion Policy Card

Focus:
- high-cost include/skip/index-later tradeoffs
- utility-sensitive ingest under bounded compute

## Degradation Scheduler Card

Focus:
- pressure-state transitions
- safe fallback ladders and recovery gates

## Ranking Policy Card

Focus:
- quality/latency balancing under constrained resources
- stable tie-break and regression-safe rollout criteria

## Crawl/Ingest Optimization Track (`bd-2hz.9.3`)

Prioritized hotspot candidates and target gains:

1. `ingest.catalog.batch_upsert`
   - stage: `catalog_mutation`
   - target: p50 -16%, p95 -24%, throughput +20%
2. `crawl.classification.policy_batching`
   - stage: `classification`
   - target: p50 -10%, p95 -16%, throughput +12%
3. `ingest.queue.lane_budget_admission`
   - stage: `queue_admission`
   - target: p50 -9%, p95 -14%, throughput +11%
4. `crawl.discovery.path_metadata_cache`
   - stage: `discovery_walk`
   - target: p50 -8%, p95 -13%, throughput +10%
5. `ingest.embed_gate.early_skip`
   - stage: `embedding_gate`
   - target: p50 -7%, p95 -11%, throughput +9%

Isomorphism proof checklist requirements (per lever):

- baseline comparator explicitly names incumbent behavior (discovery/classification/catalog/queue/embed gate)
- replay command: `fsfs profile replay --lane ingest --lever-id <id> --compare baseline`
- invariants include:
  - deterministic scope/classification outcomes
  - monotonic catalog/changelog sequencing
  - bounded queue semantics and stable backpressure reason codes
  - explainability-preserving ingest/degrade reason codes

Rollback guardrails (per optimization class):

- rollback command: `fsfs profile rollback --lever-id <id> --restore baseline`
- abort triggers include class-specific reason codes (scope regressions, idempotency violations, queue starvation/unbounded growth, embed/degrade policy regressions)
- required recovery reason code: `opt.rollback.completed`

## Query Latency Optimization Track (`bd-2hz.9.4`)

Prioritized retrieval/fusion/explanation levers (ICE rank, highest first):

1. `vector_search.scratch_buffer_reuse` (phase: `fast_vector_search`)
   - mechanism: `buffer_reuse`
   - target: lower allocation churn in tight scan loops; expected p95 latency reduction in fast path
2. `fuse.hashmap_capacity` (phase: `fuse`)
   - mechanism: `allocation_reduction`
   - target: fewer HashMap rehashes for typical lexical/semantic overlap ratios
3. `fuse.string_clone_reduction` (phase: `fuse`)
   - mechanism: `data_movement`
   - target: avoid repeated `doc_id` clone work in RRF merge loops
4. `blend.string_clone_reduction` (phase: `blend`)
   - mechanism: `data_movement`
   - target: reduce blend-loop string allocation pressure
5. `blend.rank_map_cache` (phase: `blend`)
   - mechanism: `precomputation`
   - target: reuse phase-1 rank metadata in phase-2 blending
6. `serialize.preallocate_json_buffer` (phase: `serialize`)
   - mechanism: `allocation_reduction`
   - target: reduce serialization reallocations for large result payloads
7. `vector_search.parallel_threshold_tuning` (phase: `fast_vector_search`)
   - mechanism: `parallelism`
   - target: improve p95 crossover by tuning scan parallelization threshold
8. `blend.kendall_tau_approximation` (phase: `blend`)
   - mechanism: `algorithm_replacement`
   - target: replace O(n^2) Kendall tau with O(n log n) equivalent

Latency decomposition and budget contract:

- canonical phase model: `canonicalize`, `classify`, `fast_embed`, `lexical_retrieve`, `fast_vector_search`, `fuse`, `quality_embed`, `quality_vector_search`, `blend`, `rerank`, `explain`, `serialize`
- each phase records `actual_us`, `budget_us`, and `skipped` flag
- decomposition verdict reason codes:
  - `query.latency.on_budget`
  - `query.latency.single_phase_over_budget`
  - `query.latency.multiple_phases_over_budget`

Correctness-preserving verification protocol requirements:

- schema/version contract: `fsfs-query-latency-opt-v1`
- required corpora:
  - `golden_100`
  - `adversarial_unicode`
  - `empty_query`
  - `identifier_query`
  - `natural_language_query`
- proof kinds by lever:
  - `bit_identical` for refactors and allocation/data-movement levers
  - `numerically_equivalent` for floating-point sensitive improvements
  - `rank_preserving` for parallel threshold and ordering-risk levers
- merge gate reason codes:
  - pass: `opt.verify.passed`
  - fail: `opt.verify.failed`

## Contract Semantics

- `ev_score` is numeric expected value (impact-confidence-reuse-effort normalization)
- `priority_tier` uses `A|B|C`
- `adoption_wedge` states where rollout starts first and why
- budgeted mode includes explicit defaults and exhaustion behavior
- fallback trigger includes `condition`, `fallback_action`, and `reason_code`
- baseline comparator names incumbent behavior being outperformed
- isomorphism proof plan defines invariants and replay checks
- reproducibility fields define required artifacts and replay command
- rollback plan defines deterministic rollback command and abort conditions

## Validation Artifacts

- `schemas/fsfs-alien-recommendations-v1.schema.json`
- `schemas/fixtures/fsfs-alien-recommendation-card-ingestion-v1.json`
- `schemas/fixtures/fsfs-alien-recommendation-bundle-v1.json`
- `schemas/fixtures-invalid/fsfs-alien-recommendation-invalid-*.json`
- `scripts/check_fsfs_alien_recommendations.sh`

## Validation Command

```bash
scripts/check_fsfs_alien_recommendations.sh --mode all
```
