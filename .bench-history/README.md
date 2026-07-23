# Quill performance history

`QG-<n>.<machine-class>.latest.json` is the committed pass-over-pass baseline for
one normative Quill performance gate. A baseline may advance only when
`quill-perf-ratchet` emits `Allow`; `Block` and `Quarantine` never overwrite it.
Every Allow also writes a dated sibling. History files are retained—automation
does not delete older evidence under the repository's Rule 1.

The committed `*.unmeasured.latest.json` files are explicit bootstrap
placeholders, not performance evidence. They contain no cells, have
`laws_attested=false`, and force PR regression alarms to `Quarantine`. Once a
gate is activated, a full, stable candidate/rerun from distinct passes in one
measurement window may establish the first real machine-class baseline. No QG
number may be cited as kept before that first `Allow`.
