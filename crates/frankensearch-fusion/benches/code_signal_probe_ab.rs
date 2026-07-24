//! `code_structure_sidecar::score_document` inner-loop A/B (fsfs).
//!
//! For each code-structure signal on a document, the current path builds a full
//! `BTreeSet<String>` of the signal's tokens (`tokenize`) purely to compute
//! `query_tokens.intersection(&signal_tokens).next()` — i.e. it only needs the
//! lexicographically smallest query token that appears in the signal, but it
//! materialises the entire signal token set + one heap `String` per token first.
//!
//! Proposed: stream the signal's tokens through a single reused scratch buffer and
//! probe `query_tokens` directly, tracking the smallest match. No intermediate
//! `BTreeSet`, and a token is cloned onto the heap only when it becomes the new
//! smallest match (0–1 allocs/signal instead of one per token). Bit-identical:
//! `min(query ∩ signal_set)` == `min over signal token occurrences in query`
//! (set dedup does not affect the minimum); non-ASCII delegates to the exact old
//! path. Identical results asserted below.
use std::collections::BTreeSet;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_fusion::bench_support::paired_median_ratio;

/// Reference tokenizer (the current `code_structure_sidecar::tokenize`, ASCII
/// fast path + Unicode fallback).
fn tokenize(value: &str) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    let mut current = String::new();
    if value.is_ascii() {
        for &b in value.as_bytes() {
            let lowered = b.to_ascii_lowercase();
            if lowered.is_ascii_alphanumeric() {
                current.push(lowered as char);
            } else if !current.is_empty() {
                tokens.insert(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            tokens.insert(current);
        }
        return tokens;
    }
    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.insert(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.insert(current);
    }
    tokens
}

/// Current: materialise the signal token set, then take the intersection min.
fn old_match(query_tokens: &BTreeSet<String>, value: &str) -> Option<String> {
    let signal_tokens = tokenize(value);
    query_tokens.intersection(&signal_tokens).next().cloned()
}

#[inline]
fn consider(best: &mut Option<String>, current: &str, query_tokens: &BTreeSet<String>) {
    if query_tokens.contains(current) && best.as_deref().is_none_or(|b| current < b) {
        *best = Some(current.to_owned());
    }
}

/// Proposed: stream tokens through one reused buffer, probe the query set, keep
/// the smallest match. No intermediate `BTreeSet`; clone only on a new minimum.
fn new_match(query_tokens: &BTreeSet<String>, value: &str) -> Option<String> {
    if !value.is_ascii() {
        // Rare: delegate to the exact reference computation for full parity.
        let signal_tokens = tokenize(value);
        return query_tokens.intersection(&signal_tokens).next().cloned();
    }
    let mut best: Option<String> = None;
    let mut current = String::new();
    for &b in value.as_bytes() {
        let lowered = b.to_ascii_lowercase();
        if lowered.is_ascii_alphanumeric() {
            current.push(lowered as char);
        } else if !current.is_empty() {
            consider(&mut best, &current, query_tokens);
            current.clear();
        }
    }
    if !current.is_empty() {
        consider(&mut best, &current, query_tokens);
    }
    best
}

/// A realistic per-document signal set (file paths, module/function/class names,
/// imports) — the kind `CodeStructureDocument` carries. `mult` scales the count.
fn signals(mult: usize) -> Vec<String> {
    let base = [
        "src/auth/login_handler.rs",
        "authenticate_user_session",
        "class UserSessionManager",
        "import bcrypt hashlib",
        "fn validate_bearer_token",
        "middleware::auth::jwt",
        "def refresh_access_token",
        "TokenStore trait impl",
        "pub struct SessionCookieJar",
        "GET /api/v2/users/{id}/profile",
        "markdown heading Authentication Guide",
        "type AuthResult Result Error",
        "crates/frankensearch-fusion/src/rrf.rs",
        "compute_reciprocal_rank_fusion",
        "NoMatchHereJustFillerIdentifiers",
        "another_unrelated_symbol_name_xyz",
    ];
    let mut out = Vec::with_capacity(base.len() * mult);
    for _ in 0..mult {
        for s in &base {
            out.push((*s).to_owned());
        }
    }
    out
}

fn query_set() -> BTreeSet<String> {
    // A few query tokens; some hit early signals, most signals miss entirely
    // (the realistic case: the query matches a handful of a document's signals).
    ["auth", "user", "token", "session", "login"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

fn bench(c: &mut Criterion) {
    let query = query_set();
    let mut g = c.benchmark_group("code_signal_probe");
    for &mult in &[8usize, 40, 160] {
        let sigs = signals(mult);
        // Parity: old and new must agree on every signal.
        for s in &sigs {
            assert_eq!(
                old_match(&query, s),
                new_match(&query, s),
                "diverged on {s:?}"
            );
        }
        let id = format!("signals{}", sigs.len());
        g.bench_with_input(BenchmarkId::new("old", &id), &(), |b, ()| {
            b.iter(|| {
                let mut acc = 0usize;
                for s in black_box(&sigs) {
                    if let Some(t) = old_match(black_box(&query), s) {
                        acc += t.len();
                    }
                }
                black_box(acc)
            });
        });
        g.bench_with_input(BenchmarkId::new("new", &id), &(), |b, ()| {
            b.iter(|| {
                let mut acc = 0usize;
                for s in black_box(&sigs) {
                    if let Some(t) = new_match(black_box(&query), s) {
                        acc += t.len();
                    }
                }
                black_box(acc)
            });
        });

        // ── DECIDABILITY: alternating-round paired sampler + A/A null control ──
        //
        // The criterion arms above CANNOT decide this lever: criterion runs them as
        // separate benchmarks minutes apart, so worker drift between them is not
        // cancelled. The paired sampler runs both arms in ONE routine in alternating
        // rounds and takes the median per-round ratio; gate on the median against the
        // A/A null's observed spread, not on cv.
        let old = || {
            let mut acc = 0usize;
            for s in black_box(&sigs) {
                if let Some(t) = old_match(black_box(&query), s) {
                    acc += t.len();
                }
            }
            black_box(acc);
        };
        let new = || {
            let mut acc = 0usize;
            for s in black_box(&sigs) {
                if let Some(t) = new_match(black_box(&query), s) {
                    acc += t.len();
                }
            }
            black_box(acc);
        };
        let null = paired_median_ratio(41, 8, old, old);
        let lever = paired_median_ratio(41, 8, old, new);
        eprintln!(
            "[null]  code_signal_probe {id}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] code_signal_probe {id}: new median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
