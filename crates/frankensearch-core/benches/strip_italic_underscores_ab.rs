//! Paired A/B for `canonicalize::strip_italic_underscores`.
//!
//! In `strip_markdown_line`, the italic-underscore strip is guarded by a trigger
//! scan and only runs when the line contains a `_`. But `_` is ubiquitous in
//! technical text (`snake_case`, `variable_name`, `fn foo_bar`), where **no**
//! underscore is an italic word boundary — yet the shipping version always
//! allocated a fresh `String` and copied the whole line char-by-char, producing an
//! exact copy of the input. It was the only strip in the chain that returned
//! `String` instead of borrowing when unchanged (its siblings —
//! `strip_markdown_line`, `strip_prefixes_and_list_marker`, `strip_list_marker`,
//! and even `strip_markdown_links` — all borrow / reuse).
//!
//! The new form returns `Option<String>` (`None` = unchanged, lazily allocating
//! from the borrowed prefix only at the first dropped marker). This bench mirrors
//! both variants exactly and asserts a byte-identical materialization before
//! timing:
//!
//! - `alloc`  : always build a `String` (the shipping path).
//! - `cow`    : `Option<String>`, `None` when nothing is dropped (the new path).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/<lane> \
//!   rch exec -- cargo bench -p frankensearch-core --bench strip_italic_underscores_ab
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

/// Shipping path: always materialize a `String` (mirrors the former impl).
fn strip_alloc(text: &str) -> String {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let mut result = String::with_capacity(text.len());
    let mut prev: Option<char> = None;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        let drop_marker = c == '_' && {
            let prev_is_word = prev.is_some_and(|p| is_word(p) && p != '_');
            let next_is_word = chars.peek().is_some_and(|&n| is_word(n) && n != '_');
            (!prev_is_word && next_is_word) || (prev_is_word && !next_is_word)
        };
        if !drop_marker {
            result.push(c);
        }
        prev = Some(c);
    }
    result
}

/// New path: `Option<String>`, `None` when nothing is dropped (lazy alloc).
fn strip_cow(text: &str) -> Option<String> {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    let mut result: Option<String> = None;
    let mut prev: Option<char> = None;
    let mut chars = text.char_indices().peekable();
    while let Some((idx, c)) = chars.next() {
        let drop_marker = c == '_' && {
            let prev_is_word = prev.is_some_and(|p| is_word(p) && p != '_');
            let next_is_word = chars.peek().is_some_and(|&(_, n)| is_word(n) && n != '_');
            (!prev_is_word && next_is_word) || (prev_is_word && !next_is_word)
        };
        if drop_marker {
            result.get_or_insert_with(|| {
                let mut buf = String::with_capacity(text.len());
                buf.push_str(&text[..idx]);
                buf
            });
        } else if let Some(buf) = result.as_mut() {
            buf.push(c);
        }
        prev = Some(c);
    }
    result
}

fn run_alloc(lines: &[&str]) -> usize {
    let mut total = 0;
    for &line in lines {
        let s = strip_alloc(line);
        total += s.len();
        black_box(&s);
    }
    total
}

fn run_cow(lines: &[&str]) -> usize {
    let mut total = 0;
    for &line in lines {
        match strip_cow(line) {
            Some(s) => {
                total += s.len();
                black_box(&s);
            }
            None => total += line.len(),
        }
    }
    total
}

/// Code / technical text: every line has `_`, none is an italic marker → the new
/// path returns `None` for all of them (the common corpus case).
const SNAKE: &[&str] = &[
    "fn compute_value(a_b: i32, c_d: i32) -> retry_count",
    "let user_id = fetch_account_row(db_conn, request_id);",
    "pub struct HttpResponseWriter { status_code: u16, body_len: usize }",
    "SELECT user_name, created_at FROM audit_log WHERE tenant_id = ?",
    "const MAX_RETRY_ATTEMPTS: usize = 5; // default_backoff_ms below",
    "impl<T> DefaultHasherBuilder for FastHashMap<T> where T: Send",
];

/// Prose italics: each line drops markers → the new path still allocates.
const ITALIC: &[&str] = &[
    "this is _emphasized_ and _important_ prose text here",
    "a single _italic_ word in the middle of a sentence",
    "_leading emphasis_ then normal then _trailing emphasis_",
    "we _really_ mean it, and _you_ should _read_ carefully",
    "the _quick_ brown _fox_ jumps over the _lazy_ dog today",
    "mixing _one_ and two and _three_ emphasized spans total",
];

fn bench(c: &mut Criterion) {
    // Realistic mix: technical corpora are snake_case-dominant.
    let mixed: Vec<&str> = SNAKE
        .iter()
        .chain(SNAKE)
        .chain(SNAKE)
        .chain(SNAKE)
        .chain(ITALIC)
        .copied()
        .collect();

    let mut group = c.benchmark_group("strip_italic_underscores");
    for (name, lines) in [
        ("snake_only", SNAKE.to_vec()),
        ("italic_only", ITALIC.to_vec()),
        ("mixed_4to1", mixed),
    ] {
        // Byte-identical materialization gate.
        assert_eq!(
            run_alloc(&lines),
            run_cow(&lines),
            "kept-byte parity ({name})"
        );
        for (i, &line) in lines.iter().enumerate() {
            let a = strip_alloc(line);
            let b = strip_cow(line).unwrap_or_else(|| line.to_owned());
            assert_eq!(a, b, "line parity ({name}[{i}])");
        }

        group.bench_with_input(BenchmarkId::new("alloc", name), &lines, |b, lines| {
            b.iter(|| black_box(run_alloc(black_box(lines))));
        });
        group.bench_with_input(BenchmarkId::new("cow", name), &lines, |b, lines| {
            b.iter(|| black_box(run_cow(black_box(lines))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
