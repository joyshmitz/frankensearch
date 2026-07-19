//! `tokenize_lexical` ASCII fast-path A/B (fsfs lexical_pipeline.rs). The lexical
//! tokenizer runs a per-character loop over EVERY document's full text at index
//! time. The current impl uses `text.char_indices()` (UTF-8 decode per char) +
//! `is_token_char` (Unicode `is_alphanumeric()`). Its sibling `count_lexical_tokens`
//! already has an ASCII byte fast path via a 256-byte LUT (won ~1.5-1.8×), but
//! `tokenize_lexical` never got it. For all-ASCII text (the common case: code,
//! English prose) a byte-iteration + LUT path is bit-identical — byte index == char
//! index, and the LUT equals `is_token_char` for every ASCII byte. This bench
//! measures char-path (current) vs byte-path (proposed) on realistic ASCII docs.
//! Identical token output asserted.
use std::hint::black_box;

use compact_str::CompactString;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_fusion::bench_support::paired_median_ratio;

#[derive(PartialEq, Eq, Debug)]
struct Token {
    text: String,
    line: u32,
    byte_start: usize,
    byte_end: usize,
}

/// SSO-token variant: identical to `Token` but the lowercased text is a
/// `CompactString`. Lexical tokens (code identifiers, prose words) are almost
/// always <=24 bytes, so they live inline with ZERO heap allocation — the
/// per-token `to_ascii_lowercase` heap alloc that the ledger flagged as the
/// dominant remaining emission cost simply disappears for the common case.
#[derive(PartialEq, Eq, Debug)]
struct CompactToken {
    text: CompactString,
    line: u32,
    byte_start: usize,
    byte_end: usize,
}

fn is_token_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | ':')
}

const fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'/' | b':')
}

const TOKEN_BYTE: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        t[i] = is_token_byte(i as u8) as u8;
        i += 1;
    }
    t
};

/// Current: char_indices + Unicode is_alphanumeric.
fn tokenize_char(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut token_start: Option<usize> = None;
    let mut line = 1_u32;
    let mut token_line = 1_u32;
    for (idx, ch) in text.char_indices() {
        if is_token_char(ch) {
            if token_start.is_none() {
                token_start = Some(idx);
                token_line = line;
            }
        } else if let Some(start) = token_start.take() {
            tokens.push(Token {
                text: text[start..idx].to_ascii_lowercase(),
                line: token_line,
                byte_start: start,
                byte_end: idx,
            });
        }
        if ch == '\n' {
            line = line.saturating_add(1);
        }
    }
    if let Some(start) = token_start {
        tokens.push(Token {
            text: text[start..].to_ascii_lowercase(),
            line: token_line,
            byte_start: start,
            byte_end: text.len(),
        });
    }
    tokens
}

/// Proposed: ASCII byte fast path (LUT), falling back to the char path for
/// non-ASCII text so behaviour is preserved for every input.
fn tokenize_fast(text: &str) -> Vec<Token> {
    if !text.is_ascii() {
        return tokenize_char(text);
    }
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut token_start: Option<usize> = None;
    let mut line = 1_u32;
    let mut token_line = 1_u32;
    for (idx, &b) in bytes.iter().enumerate() {
        if TOKEN_BYTE[b as usize] == 1 {
            if token_start.is_none() {
                token_start = Some(idx);
                token_line = line;
            }
        } else if let Some(start) = token_start.take() {
            tokens.push(Token {
                text: text[start..idx].to_ascii_lowercase(),
                line: token_line,
                byte_start: start,
                byte_end: idx,
            });
        }
        if b == b'\n' {
            line = line.saturating_add(1);
        }
    }
    if let Some(start) = token_start {
        tokens.push(Token {
            text: text[start..].to_ascii_lowercase(),
            line: token_line,
            byte_start: start,
            byte_end: text.len(),
        });
    }
    tokens
}

/// Proposed SSO variant: same ASCII byte fast path as `tokenize_fast`, but each
/// token's lowercased text is built directly into a `CompactString` via
/// `CompactString::new(slice)` + in-place `make_ascii_lowercase()`. For ASCII
/// input this is byte-identical to `slice.to_ascii_lowercase()` (ASCII-only
/// lowercasing), and short tokens (<=24 bytes) never touch the heap.
fn tokenize_compact(text: &str) -> Vec<CompactToken> {
    fn lower(slice: &str) -> CompactString {
        let mut cs = CompactString::new(slice);
        cs.as_mut_str().make_ascii_lowercase();
        cs
    }
    if !text.is_ascii() {
        // Match the char-path lowercasing (ASCII-only) for parity with the other arms.
        let string = tokenize_char(text);
        return string
            .into_iter()
            .map(|t| CompactToken {
                text: CompactString::new(&t.text),
                line: t.line,
                byte_start: t.byte_start,
                byte_end: t.byte_end,
            })
            .collect();
    }
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut token_start: Option<usize> = None;
    let mut line = 1_u32;
    let mut token_line = 1_u32;
    for (idx, &b) in bytes.iter().enumerate() {
        if TOKEN_BYTE[b as usize] == 1 {
            if token_start.is_none() {
                token_start = Some(idx);
                token_line = line;
            }
        } else if let Some(start) = token_start.take() {
            tokens.push(CompactToken {
                text: lower(&text[start..idx]),
                line: token_line,
                byte_start: start,
                byte_end: idx,
            });
        }
        if b == b'\n' {
            line = line.saturating_add(1);
        }
    }
    if let Some(start) = token_start {
        tokens.push(CompactToken {
            text: lower(&text[start..]),
            line: token_line,
            byte_start: start,
            byte_end: text.len(),
        });
    }
    tokens
}

/// Realistic ASCII document text (code + prose + paths), repeated to `reps`.
fn doc(reps: usize) -> String {
    let unit = "fn compute_search_score(query: &str, doc_id: usize) -> f64 {\n    \
        let normalized = query.trim().to_ascii_lowercase();\n    // see docs/design/scoring.md \
        for the RRF fusion details and the two-tier blend.\n    let weight = 0.7_f64; \
        return weight * bm25 + (1.0 - weight) * cosine_similarity;\n}\n\n";
    unit.repeat(reps)
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("tokenize_ascii");
    for &reps in &[20usize, 100, 400] {
        let text = doc(reps);
        assert_eq!(tokenize_char(&text), tokenize_fast(&text)); // bit-identical
        // The CompactString arm must emit the same token texts/offsets/lines.
        let string_tokens = tokenize_fast(&text);
        let compact_tokens = tokenize_compact(&text);
        assert_eq!(string_tokens.len(), compact_tokens.len());
        for (s, c) in string_tokens.iter().zip(compact_tokens.iter()) {
            assert_eq!(s.text, c.text.as_str());
            assert_eq!(s.line, c.line);
            assert_eq!(s.byte_start, c.byte_start);
            assert_eq!(s.byte_end, c.byte_end);
        }
        let id = format!("bytes{}", text.len());
        g.bench_with_input(BenchmarkId::new("char", &id), &(), |b, ()| {
            b.iter(|| black_box(tokenize_char(black_box(&text))));
        });
        g.bench_with_input(BenchmarkId::new("fast", &id), &(), |b, ()| {
            b.iter(|| black_box(tokenize_fast(black_box(&text))));
        });
        g.bench_with_input(BenchmarkId::new("compact", &id), &(), |b, ()| {
            b.iter(|| black_box(tokenize_compact(black_box(&text))));
        });

        // ── DECIDABILITY: alternating-round paired sampler + A/A null control ──
        //
        // The criterion arms above CANNOT decide these levers: criterion runs them as
        // separate benchmarks minutes apart, so worker drift between them is not
        // cancelled. The paired sampler runs both arms in ONE routine in alternating
        // rounds and takes the median per-round ratio; gate on the median against the
        // A/A null's observed spread, not on cv.
        let char_path = || {
            black_box(tokenize_char(black_box(&text)));
        };
        let fast_path = || {
            black_box(tokenize_fast(black_box(&text)));
        };
        let compact_path = || {
            black_box(tokenize_compact(black_box(&text)));
        };
        let null = paired_median_ratio(41, 8, char_path, char_path);
        let lever_fast = paired_median_ratio(41, 8, char_path, fast_path);
        let lever_compact = paired_median_ratio(41, 8, char_path, compact_path);
        eprintln!(
            "[null]  tokenize_ascii {id}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] tokenize_ascii {id}: fast median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever_fast.median,
            lever_fast.p5,
            lever_fast.p95,
            if lever_fast.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
        eprintln!(
            "[lever] tokenize_ascii {id}: compact median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever_compact.median,
            lever_compact.p5,
            lever_compact.p95,
            if lever_compact.decidable_against(&null) {
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
