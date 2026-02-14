# fsfs Explanation Payload Contract v1

Issue: `bd-2hz.5.4`  
Parent: `bd-2hz.5`

## Goal

Define a deterministic, machine-readable explanation payload that unifies:

- ranking explanation breakdowns
- policy decision explanations
- transport parity across CLI JSON, CLI TOON, and TUI explainability panels

This contract builds on the library-level explanation primitives from `bd-11n`
(`HitExplanation`, `ScoreComponent`, `ExplainedSource`, rank movement semantics)
and specifies the fsfs-facing payload surface.

## Normative Terms

- `MUST`: required for conformance
- `SHOULD`: expected default unless explicitly justified
- `MUST NOT`: forbidden behavior

## Payload Envelope Requirements

Each explanation payload MUST include:

- `schema_version` (`fsfs.explanation.payload.v1`)
- `query` (normalized query text used for planning/execution)
- optional `trace` linkage (`trace_id`, `event_id`, optional causal IDs)
- `ranking` block
- `policy_decisions[]` block

## Ranking Block Requirements

`ranking` MUST include:

- `doc_id`
- `final_score`
- `phase` (`Initial|Refined`)
- `reason_code`
- `confidence_per_mille` (0..1000)
- component breakdown rows in stable order

Optional ranking fields:

- `rank_movement` (phase-1 to phase-2 movement snapshot)
- `fusion` context (lexical/semantic source metadata)

Each component row MUST include:

- source enum (`lexical_bm25|semantic_fast|semantic_quality|rerank`)
- summary string
- raw and normalized scores
- RRF contribution and blend weight
- bounded confidence score (`0..1000`)

## Policy Decision Block Requirements

`policy_decisions[]` MUST carry explicit domain + reason context.

Required fields per decision:

- `domain` (`query_intent|retrieval_budget|query_execution|degradation|discovery`)
- `decision`
- `reason_code`
- `confidence_per_mille` (0..1000)
- `summary`
- `metadata` (string map for machine replay and diagnostics)

## Transport Parity Rules

For equivalent payload input:

- CLI JSON MUST preserve all schema fields losslessly.
- CLI TOON MUST preserve stable enum labels and reason/confidence fields.
- TUI panel projection MUST preserve semantic meaning and reason-code visibility.

## Reason-Code and Confidence Constraints

- `reason_code` values MUST follow canonical fsfs naming conventions.
- `confidence_per_mille` fields MUST be clamped to `[0, 1000]`.
- Unknown policy domains MUST be rejected by schema validation.

## Validation Artifacts

- `schemas/fsfs-explanation-payload-v1.schema.json`
- `schemas/fixtures/fsfs-explanation-payload-contract-v1.json`
- `schemas/fixtures/fsfs-explanation-payload-decision-v1.json`
- `schemas/fixtures-invalid/fsfs-explanation-payload-invalid-*.json`
- `scripts/check_fsfs_explanation_payload_contract.sh`

## Validation Command

```bash
scripts/check_fsfs_explanation_payload_contract.sh --mode all
```
