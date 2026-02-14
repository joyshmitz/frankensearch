# Bead Self-Documentation Rubric v1 (`bd-3qwe.1`)

## Purpose
Define the normative self-documentation contract for active beads so planning artifacts stay self-contained, reviewable, and lintable.

## Stakeholder Summary
This rubric makes each bead answer five questions up front:
1. Why are we doing this now?
2. What is in scope and out of scope?
3. What exact validation evidence is required?
4. What exceptions are allowed and who owns the risk?
5. Can CI detect drift automatically?

## Required Comment Anchors
All implementation and gate beads must carry two machine-readable comment anchors.

### Rationale Anchor
```text
[bd-3qwe self-doc] RATIONALE
PROBLEM_CONTEXT: ...
WHY_NOW: ...
SCOPE_BOUNDARY: ...
PRIMARY_SURFACES: ...
```

### Evidence Anchor
```text
[bd-3qwe self-doc] EVIDENCE
UNIT_TESTS: ...
INTEGRATION_TESTS: ...
E2E_TESTS: ...
PERFORMANCE_VALIDATION: ...
LOGGING_ARTIFACTS: ...
```

`N/A` is allowed only with explicit reason text (for example, `E2E_TESTS: N/A - docs-only policy update`).

## Bead Class Requirements
Class mapping follows `docs/test-matrix-policy.md`.

1. `implementation` (`task`, `feature`, `bug`):
Rationale + evidence anchors required. Missing anchors are `error`.
2. `gate` (`gate`/`release-gate` label):
Rationale + evidence anchors required. Missing anchors are `error`.
3. `program` (`epic`):
Anchors recommended in rollout phase 1. Missing anchors are `warning`.
4. `exploratory` (`question`, `docs`):
Anchors optional. Missing anchors are `info`.

## Exception Policy
Waivers must be explicit, time-bounded, and assigned.

```text
[bd-3qwe self-doc] EXCEPTION
RULE_ID: SDOC-RUBRIC-00X
OWNER: <agent/team>
JUSTIFICATION: <why this waiver is safe>
EXPIRES_ON: YYYY-MM-DD
FOLLOW_UP_BEAD: bd-####
APPROVED_BY: <review authority>
```

Exception rules:
1. Missing any required exception field invalidates the waiver.
2. `EXPIRES_ON` is mandatory.
3. `FOLLOW_UP_BEAD` must reference a concrete bead ID.

## Positive and Negative Examples
Positive examples from existing beads:
1. `bd-17dv`: explicit dependency rationale grammar (`HARD_DEP`/`SOFT_DEP`/`INFO_REF`) and reviewer checklist.
2. `bd-264r`: explicit test-matrix anchor plus required section checklist.
3. `bd-2hz.1`: explicit evidence-lane backfill comment for unit/integration/e2e/logging scope.

Negative examples observed in active backlog (missing explicit rationale/evidence anchors):
1. `bd-2hz.10.11.4`
2. `bd-2hz.10.11.5`
3. `bd-2hz.10.11.6`

## Reviewer Checklist
1. Verify rationale anchor exists and all four fields are present.
2. Verify evidence anchor exists and all five evidence lanes are present.
3. Verify each `N/A` lane includes a concrete reason.
4. Verify exception markers (if present) include all required exception fields.
5. Reject vague justifications without concrete artifact ownership.

## Machine-Checkable Rule IDs
The checker emits JSON findings with deterministic IDs:
1. `SDOC-RUBRIC-000` policy self-test drift.
2. `SDOC-RUBRIC-001` missing rationale anchor.
3. `SDOC-RUBRIC-002` missing rationale fields.
4. `SDOC-RUBRIC-003` missing evidence anchor.
5. `SDOC-RUBRIC-004` missing evidence fields.
6. `SDOC-RUBRIC-005` malformed exception payload.

## Validation Command
```bash
scripts/check_bead_self_documentation.sh --mode all
```

## CI and Operations
1. CI gate workflow: `.github/workflows/selfdoc-lint.yml`
2. Reporter formats:
   - `--format human` (terminal summary)
   - `--format json --report-path <path>` (machine ingest)
   - `--format ci` (GitHub annotations)
3. Rollout phases:
   - `--phase audit` (report-only)
   - `--phase default` (fail on `error`)
   - `--phase strict` (fail on `warning` and `error`)
4. Operational ownership and lifecycle guidance:
   - `docs/self-documentation-lint-playbook.md`
