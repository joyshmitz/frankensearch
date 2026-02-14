# Per-Bead Test Matrix Policy (bd-264r)

## Goal
Make test scope explicit at bead level so implementation work is auditable before coding starts.

## Required Test Matrix Sections
Each implementation bead must include these sections (either in description or explicit policy comment):
1. `Unit tests`
2. `Integration tests`
3. `E2E tests` (or explicit `N/A` + rationale)
4. `Performance/bench` (or explicit `N/A` + rationale)
5. `Logs/artifacts assertions`

## Standard Template
```text
TEST_MATRIX
Unit tests:
- ...
Integration tests:
- ...
E2E tests:
- ... (or N/A: <reason>)
Performance/bench:
- ... (or N/A: <reason>)
Logs/artifacts:
- ...
```

## Wave-1 Retrofit (completed)
Updated with explicit test-matrix annotations:
- `bd-3un.31`
- `bd-3un.32`
- `bd-3un.40`
- `bd-3un.52`

## Wave-2 Retrofit (completed with justified exceptions)
Workstream aggregators marked with explicit exception rationale (matrix inherited by child implementation beads):
- `bd-2hz.10`
- `bd-2yu.8`

## Lint/CI Checker
Checker script:
- `scripts/check_bead_test_matrix.sh`

Modes:
- `--mode unit`: validates wave-1 explicit matrix annotations and wave-2 exception annotations.
- `--mode integration`: surfaces additional open implementation beads likely missing explicit matrix sections.
- `--mode all`: runs both.

Replay command:
```bash
scripts/check_bead_test_matrix.sh --mode all
```

## Change Log (bd-264r pass)
- Added explicit matrix annotations to 4 wave-1 high-priority beads.
- Added explicit exception markers/rationales to 2 wave-2 workstream beads.
- Added CI-checkable script to prevent silent regressions.
