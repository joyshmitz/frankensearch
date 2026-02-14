# fsfs Scope, Privacy, and Redaction Boundaries v1

Issue: `bd-2hz.1.3`  
Parent: `bd-2hz.1`

## Goal

Define non-negotiable boundaries for what `fsfs` may:

- scan
- persist
- emit (telemetry/logs/evidence)
- display (CLI/TUI/explain surfaces)

This contract is normative for all downstream indexing, policy, and evidence beads.

## Normative Terms

- `MUST`: hard requirement
- `SHOULD`: default expectation unless explicitly justified
- `MUST NOT`: forbidden behavior

## Scope Defaults + Opt-In/Opt-Out Semantics

## Default Discovery Scope

`fsfs` MUST default to user-owned, high-value textual roots only:

- `$HOME` source trees, docs, config directories
- explicit caller-provided roots within user-owned paths

`fsfs` MUST NOT scan system/global roots by default:

- `/etc`, `/var`, `/proc`, `/sys`, `/dev`, `/run`
- other users' home directories

## Opt-In Rules

Paths outside the default safe scope require explicit opt-in.

- opt-in MUST be path-explicit (no implicit escalation)
- opt-in MUST be auditable in config/evidence output
- deny-list entries retain precedence even when broad roots are opt-in

## Opt-Out Rules

Users MUST be able to exclude any path via explicit globs.

Precedence:

1. hard deny rules (`MUST NOT scan`)
2. explicit opt-out
3. explicit opt-in
4. defaults

## Sensitive Path/Data-Class Handling

Sensitive classes (minimum required):

- credentials/tokens
- private keys (`.ssh`, keychains, auth stores)
- browser secret stores/session artifacts
- personal/financial/health content markers

Hard deny examples (default):

- `~/.ssh/**`
- `~/.gnupg/**`
- browser profile secret DBs
- cloud credentials (`~/.aws/credentials`, `~/.config/gcloud/**`)

If a path is sensitive-classed:

- raw content persistence is forbidden
- raw content emission is forbidden
- display of secret-bearing spans is forbidden

Only redacted/hashed metadata MAY be retained where explicitly allowed.

## Redaction Contract (Logs, Explain, Replay)

## Global Rule

`fsfs` MUST emit only redacted-safe artifacts in logs, explain payloads, and replay bundles.

Required behavior:

- `raw_content_allowed = false`
- `reason_code` required for every redaction or deny decision
- deterministic redaction profile/version included in artifacts

## Artifact-Specific Policy

- logs: mask/hash sensitive tokens; no raw secret literals
- explain payloads: include policy reason codes and class labels, not raw secret spans
- replay artifacts: retain deterministic decision metadata, never raw protected content

## Threat Model (Local Multi-User)

Assumptions:

- local multi-user environment is possible
- same-host users/processes may inspect logs/artifacts if filesystem controls are weak

Required controls:

- local-only operation boundaries
- least-privilege file permissions for evidence/artifact outputs
- redaction before persistence/emission
- explicit reason-coded denials for unsafe outputs

## Required Decision Fields (for Auditing/Testability)

Every scope/privacy decision MUST include:

- `path`
- `decision` (`include`, `exclude`, `require_opt_in`)
- `reason_code` (machine-stable)
- `sensitive_classes[]`
- `persist_allowed` / `emit_allowed` / `display_allowed`
- `redaction_profile`

## Validation Artifacts

- `schemas/fsfs-scope-privacy-v1.schema.json`
- `schemas/fixtures/fsfs-scope-privacy-*.json`
- `schemas/fixtures-invalid/fsfs-scope-privacy-*.json`

## Validation Commands

```bash
for f in schemas/fixtures/fsfs-scope-privacy-*.json; do
  jsonschema -i "$f" schemas/fsfs-scope-privacy-v1.schema.json
done

for f in schemas/fixtures-invalid/fsfs-scope-privacy-*.json; do
  if jsonschema -i "$f" schemas/fsfs-scope-privacy-v1.schema.json; then
    echo "unexpected pass: $f" && exit 1
  fi
done
```
