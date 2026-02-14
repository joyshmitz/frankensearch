# Dependency-Semantics Lint Playbook (`bd-2yu.9.4.7`)

## Purpose
Keep bead dependency semantics (`HARD_DEP`, `SOFT_DEP`, `INFO_REF`) consistent, auditable, and CI-enforced.

## Ownership
1. Primary owner: backlog quality/governance maintainer for dependency policy.
2. Secondary owner: CI maintainer for lint workflow health and artifact retention.
3. Waiver authority: release-quality maintainer approving temporary policy exceptions.

## Operator Commands
1. Local lint run:
   - `scripts/check_dependency_semantics.sh --mode all`
2. Unit-only policy checks:
   - `scripts/check_dependency_semantics.sh --mode unit`
3. Integration drift scan:
   - `scripts/check_dependency_semantics.sh --mode integration`

## CI Gate
1. Workflow:
   - `.github/workflows/dependency-semantics-lint.yml`
2. Gate behavior:
   - Fails build on any unit/integration violation reported by the checker.
3. Review artifacts:
   - CI logs include explicit failing bead IDs and remediation messages.

## Remediation Process
1. Reproduce locally with `--mode all`.
2. For each failing bead:
   - confirm edge semantics are correct (`blocks` vs `parent-child` vs prose-only),
   - add/repair explicit `DEP_SEMANTICS` annotation details,
   - re-run lint and ensure deterministic pass.
3. Post remediation evidence in bead comments and thread update in Agent Mail.

## Rule Evolution
1. Introduce new rules with deterministic fixtures first.
2. Run at least one clean cycle in CI before tightening policy scope.
3. Keep policy examples synchronized with actual checker behavior.

## Review Cadence
1. Weekly drift scan:
   - run integration mode and triage global candidate blockers.
2. Pre-release:
   - require clean `--mode all` output.
3. Post-incident:
   - add regression checks for newly discovered ambiguity patterns.
