# Self-Documentation Lint Maintenance Playbook (`bd-3qwe.7.9`)

## Purpose
Keep backlog self-documentation quality enforceable and stable as rules evolve.

## Ownership Model
1. Primary owner: current release/quality lane maintainer for backlog hygiene (`bd-3qwe.*`).
2. Secondary owner: CI gate maintainer for workflow health and artifact retention.
3. Review authority for waivers: designated maintainer approving `[bd-3qwe self-doc] EXCEPTION` payloads.

## Rule Lifecycle
1. Add rule:
   - Reserve `.beads/issues.jsonl` and lint surfaces.
   - Assign stable rule ID.
   - Add deterministic fixture coverage (`unit`, `integration`, `e2e`).
   - Update policy docs and CI remediation hints.
2. Change rule:
   - Document compatibility impact and migration path.
   - Run audit phase first (`--phase audit`) and inspect findings.
   - Promote to enforced phase only after false-positive review.
3. Deprecate rule:
   - Mark rule as deprecated with sunset date.
   - Keep parser/report compatibility until sunset is complete.
   - Remove only after one clean strict cycle.

## CI Rollout Policy
1. `audit`:
   - Findings reported, no gate failure.
   - Used during new rule introduction and large migrations.
2. `default`:
   - Gate fails on `error`.
3. `strict`:
   - Gate fails on `warning` and `error`.

## Operator Runbook
1. Local full run:
   - `scripts/check_bead_self_documentation.sh --mode all --phase default --format human`
2. CI annotation output:
   - `scripts/check_bead_self_documentation.sh --mode all --phase default --format ci`
3. Machine-readable export:
   - `scripts/check_bead_self_documentation.sh --mode all --format json --report-path artifacts/selfdoc-lint.json`

## Remediation Wave Execution
1. Triage:
   - Use inventory artifacts and `bv --robot-triage` to prioritize high-impact beads.
2. Claim:
   - Set bead to `in_progress`, reserve `.beads/issues.jsonl`, announce in Agent Mail.
3. Apply:
   - Add required rationale/evidence anchors (or explicit exception payload).
4. Verify:
   - Re-run lint in `default` phase.
   - Confirm missing counts reach target.
5. Close and handoff:
   - Add completion evidence comment.
   - Post Agent Mail thread update and release reservations.

## Fixture and Regression Hygiene
1. Keep positive and negative fixtures for each rule class.
2. Add a regression fixture for every production false positive/false negative.
3. Ensure deterministic ordering in reports for stable CI diffs.

## Telemetry and Health Signals
1. Track per-run counts by severity (`info`, `warning`, `error`).
2. Track unresolved findings age and reopen rate.
3. Alert on:
   - sudden `error` spikes,
   - repeated exception renewals without follow-up closure,
   - strict-phase regressions on `main`.
