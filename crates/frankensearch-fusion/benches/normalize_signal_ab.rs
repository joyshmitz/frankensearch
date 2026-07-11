//! `code_structure_sidecar::normalize_signal_value` A/B (fsfs).
//!
//! The current normaliser makes THREE allocations to collapse whitespace runs to
//! single spaces, trim, and ASCII-lowercase a signal value:
//!   `value.split_whitespace().collect::<Vec<_>>().join(" ").trim().to_ascii_lowercase()`
//! (a `Vec<&str>`, the joined `String`, and the lowercased `String`). It runs per
//! extracted declaration at index time and once per (query, document) at search
//! time. The proposed single-pass version writes directly into one `String`:
//! skip leading whitespace, emit one space between non-whitespace runs, drop
//! trailing whitespace, ASCII-lowercase each char — one allocation, byte-identical
//! output. Identical results asserted.
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

/// Current: split_whitespace -> Vec -> join -> trim -> to_ascii_lowercase (3 allocs).
fn old_normalize(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_ascii_lowercase()
}

/// Proposed: single pass into one `String`. Collapses `char::is_whitespace` runs
/// to a single ASCII space, strips leading/trailing whitespace, ASCII-lowercases.
fn new_normalize(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut pending_space = false;
    for ch in value.chars() {
        if ch.is_whitespace() {
            pending_space = true;
        } else {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            pending_space = false;
            out.push(ch.to_ascii_lowercase());
        }
    }
    out
}

/// Realistic signal values: file paths, imports (with internal whitespace runs),
/// declarations, and markdown headings with mixed case + padding.
fn values() -> Vec<&'static str> {
    vec![
        "src/auth/login_handler.rs",
        "use   std::collections :: HashMap",
        "pub use  crate::fusion::rrf",
        "AuthenticateUserSession",
        "async def   refresh_access_token",
        "##   Authentication  Guide   ",
        "  class    UserSessionManager  ",
        "import bcrypt  hashlib   os",
        "frankensearch-fusion/src/rrf.rs",
        "MODULE_NAME",
        "GET /api/v2/users/{id}/profile",
        "\ttrait   TokenStore\t",
    ]
}

fn bench(c: &mut Criterion) {
    let vals = values();
    // Parity: byte-identical output on every value.
    for v in &vals {
        assert_eq!(old_normalize(v), new_normalize(v), "diverged on {v:?}");
    }
    let mut g = c.benchmark_group("normalize_signal");
    for &mult in &[16usize, 128] {
        // Build a batch to amortise per-call criterion overhead (mirrors a
        // document with many extracted signals / a search over many docs).
        let batch: Vec<&str> = vals
            .iter()
            .cloned()
            .cycle()
            .take(vals.len() * mult)
            .collect();
        let id = format!("n{}", batch.len());
        g.bench_with_input(BenchmarkId::new("old", &id), &(), |b, ()| {
            b.iter(|| {
                let mut acc = 0usize;
                for v in black_box(&batch) {
                    acc += old_normalize(v).len();
                }
                black_box(acc)
            });
        });
        g.bench_with_input(BenchmarkId::new("new", &id), &(), |b, ()| {
            b.iter(|| {
                let mut acc = 0usize;
                for v in black_box(&batch) {
                    acc += new_normalize(v).len();
                }
                black_box(acc)
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
