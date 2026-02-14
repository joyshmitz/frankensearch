#!/usr/bin/env bash
set -euo pipefail

run_check() {
  echo "[feature-matrix] $*"
  "$@"
}

# Default/dev configuration (includes tests/examples/benches for the facade crate).
run_check cargo check -p frankensearch --all-targets

# Representative feature combinations required by bd-3w1.14.
run_check cargo check -p frankensearch --lib --no-default-features --features storage
run_check cargo check -p frankensearch --lib --no-default-features --features durability
run_check cargo check -p frankensearch --lib --no-default-features --features persistent
run_check cargo check -p frankensearch --lib --no-default-features --features durable
run_check cargo check -p frankensearch --lib --no-default-features --features full
run_check cargo check -p frankensearch --lib --no-default-features --features full-fts5

echo "[feature-matrix] PASS"
