//! Default-analyzer tokenizer A/B: Quill's SWAR byte classifier vs the scalar
//! char-walk reference, with the shipping fused scalar and the legacy Tantivy
//! `SimpleTokenizer + LowerCaser` chain as orientation arms (bd-quill-e1-scribe-bejd.1).
//!
//! Tokenization is a *full-scan classify*: every input byte is visited to find
//! token boundaries, which is precisely the shape where SWAR/SIMD pays — unlike
//! the `memchr`/`contains` early-exit scans that regress when fused (bd-5hz0).
//! The decisive lever is `quill_scalar` vs `quill_simd`: identical token model,
//! identical emission, so the only difference is boundary-finding (char-walk vs
//! 8-bytes-per-word SWAR). The A/A null is scalar-vs-scalar; the ratio is
//! `simd / scalar`, so a median `< 1.0` outside the null floor is a decidable win.
//!
//! Two corpora, because the payoff is length-dependent: `corpus` is realistic
//! short (~6-byte) space-separated tokens, where SWAR is ≈parity (the scalar
//! char-walk already has an ASCII byte fast-path, so ~6-byte tokens barely fill
//! one 8-lane window — the run-to-run sign is inside the fleet's noise);
//! `long_token_corpus` is 24–48 byte tokens (hashes/base64/UUIDs/identifiers)
//! with long separator runs, where SWAR amortizes its per-window mask and wins
//! decidably (~1.2–1.5×). Adopting SWAR as the default is therefore neutral on
//! prose and a real win on the long tokens common in code/log/data corpora.
//!
//! All four arms are asserted byte-identical (offsets + text) before timing, so
//! any ranking/recall-affecting divergence fails the bench rather than shipping.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-quill --features bench-internals \
//!     --profile release --bench tokenizer_simd_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_lexical::default_tokenizer_for_bench;
use frankensearch_quill::Analyzer;
use frankensearch_quill::scribe::{
    FrankensearchTokenizer, TokenAnalyzer, analyze_default_scalar_reference,
};
use tantivy::tokenizer::{LowerCaser, SimpleTokenizer, TextAnalyzer, TokenStream};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold one token's identity (source offsets + normalized bytes) into a running
/// digest. Shared across all arms so cross-implementation parity is one compare.
#[inline]
fn fold_token(mut digest: u64, offset_from: usize, offset_to: usize, text: &str) -> u64 {
    digest ^= u64::try_from(offset_from).unwrap_or(u64::MAX);
    digest = digest.wrapping_mul(FNV_PRIME);
    digest ^= u64::try_from(offset_to).unwrap_or(u64::MAX);
    digest = digest.wrapping_mul(FNV_PRIME);
    for byte in text.bytes() {
        digest ^= u64::from(byte);
        digest = digest.wrapping_mul(FNV_PRIME);
    }
    digest
}

fn quill_simd_digest(analyzer: &mut FrankensearchTokenizer, text: &str) -> u64 {
    let mut digest = FNV_OFFSET;
    analyzer.analyze(Analyzer::FrankensearchDefault, text, &mut |token| {
        digest = fold_token(digest, token.offset_from, token.offset_to, &token.text);
    });
    digest
}

fn quill_scalar_digest(text: &str) -> u64 {
    let mut digest = FNV_OFFSET;
    analyze_default_scalar_reference(text, &mut |token| {
        digest = fold_token(digest, token.offset_from, token.offset_to, &token.text);
    });
    digest
}

fn tantivy_digest(analyzer: &mut TextAnalyzer, text: &str) -> u64 {
    let mut digest = FNV_OFFSET;
    let mut stream = analyzer.token_stream(text);
    while stream.advance() {
        let token = stream.token();
        digest = fold_token(digest, token.offset_from, token.offset_to, &token.text);
    }
    digest
}

fn legacy_tantivy_tokenizer() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .build()
}

/// Realistic mostly-ASCII corpus (English prose + code identifiers + IDs) with a
/// small non-ASCII fraction — what the default analyzer actually sees at index
/// and query time — sized so the classifier, not allocation, dominates.
fn corpus() -> String {
    let words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "search",
        "index",
        "tokenizer",
        "bd-q3fy",
        "ID_42",
        "camelCase",
        "snake_case_name",
        "http",
        "vector",
        "embedding",
        "rerank",
        "POL-358",
        "Rust2024",
        "café", // one accented word so the corpus is not 100% ASCII
    ];
    let mut text = String::with_capacity(64 * 1024);
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    while text.len() < 48 * 1024 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let idx = usize::try_from(state % words.len() as u64).unwrap_or(0);
        text.push_str(words[idx]);
        text.push(' ');
    }
    text
}

/// Long-run corpus: 24–48 byte alphanumeric tokens (hashes, base64, UUIDs,
/// long identifiers) separated by long whitespace runs — the shape where an
/// 8-lanes-per-op SWAR classifier amortizes its per-window mask setup, unlike
/// the short space-separated tokens of [`corpus`].
fn long_token_corpus() -> String {
    let words = [
        "9f8c2a1b7e4d6035af19cd82b73e05461fa9c7d20e8b34a6",
        "aGVsbG8gd29ybGQgdGhpcyBpcyBhIGxvbmcgYmFzZTY0IHN0cmluZw",
        "550e8400e29b41d4a716446655440000deadbeefcafef00d",
        "extremely_long_snake_case_identifier_for_the_tokenizer_benchmark",
        "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKl",
    ];
    let mut text = String::with_capacity(64 * 1024);
    let mut state = 0x106c_9b1f_2a37_d45e_u64;
    while text.len() < 48 * 1024 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let idx = usize::try_from(state % words.len() as u64).unwrap_or(0);
        text.push_str(words[idx]);
        text.push_str("   \n\t  "); // long separator run
    }
    text
}

fn bench_default_tokenizer(c: &mut Criterion) {
    let text = corpus();

    // Parity before timing: every arm must emit the identical token stream.
    let mut simd = FrankensearchTokenizer::default();
    let mut fused = default_tokenizer_for_bench();
    let mut legacy = legacy_tantivy_tokenizer();
    let scalar_ref = quill_scalar_digest(&text);
    assert_eq!(
        quill_simd_digest(&mut simd, &text),
        scalar_ref,
        "Quill SWAR tokenizer diverged from the scalar char-walk reference"
    );
    assert_eq!(
        tantivy_digest(&mut fused, &text),
        scalar_ref,
        "shipping fused scalar tokenizer diverged from the Quill reference"
    );
    assert_eq!(
        tantivy_digest(&mut legacy, &text),
        scalar_ref,
        "legacy SimpleTokenizer+LowerCaser diverged from the Quill reference"
    );

    // NULL (scalar vs scalar), then the lever (scalar=ORIG vs SWAR).
    // Ratio = simd/scalar, < 1.0 = SWAR classifier wins. `inner` batches ~16
    // full-corpus passes (~4ms) so per-batch scheduler jitter on a shared worker
    // does not dominate a ~250µs single pass (an earlier inner=4 run produced an
    // unusably wide ~0.86..1.16 A/A null floor).
    let null = paired_median_ratio(
        41,
        16,
        || {
            black_box(quill_scalar_digest(black_box(&text)));
        },
        || {
            black_box(quill_scalar_digest(black_box(&text)));
        },
    );
    let mut simd_a = FrankensearchTokenizer::default();
    let lever = paired_median_ratio(
        41,
        16,
        || {
            black_box(quill_scalar_digest(black_box(&text)));
        },
        || {
            black_box(quill_simd_digest(&mut simd_a, black_box(&text)));
        },
    );
    eprintln!(
        "[null]  tokenizer_simd/{}KiB: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        text.len() / 1024,
        null.median,
        null.p5,
        null.p95,
        null.rounds
    );
    eprintln!(
        "[lever] tokenizer_simd/{}KiB: simd/scalar median {:.4} p5 {:.4} p95 {:.4} -> {}",
        text.len() / 1024,
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (SWAR classifier faster)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut group = c.benchmark_group("default_tokenizer");
    group.sample_size(30);
    group.bench_function("legacy_tantivy/48KiB", |b| {
        let mut analyzer = legacy_tantivy_tokenizer();
        b.iter(|| black_box(tantivy_digest(&mut analyzer, black_box(&text))));
    });
    group.bench_function("shipping_fused_scalar/48KiB", |b| {
        let mut analyzer = default_tokenizer_for_bench();
        b.iter(|| black_box(tantivy_digest(&mut analyzer, black_box(&text))));
    });
    group.bench_function("quill_scalar/48KiB", |b| {
        b.iter(|| black_box(quill_scalar_digest(black_box(&text))));
    });
    group.bench_function("quill_simd/48KiB", |b| {
        let mut analyzer = FrankensearchTokenizer::default();
        b.iter(|| black_box(quill_simd_digest(&mut analyzer, black_box(&text))));
    });
    group.finish();
}

fn bench_long_token_tokenizer(c: &mut Criterion) {
    let text = long_token_corpus();

    let scalar_ref = quill_scalar_digest(&text);
    let mut simd = FrankensearchTokenizer::default();
    assert_eq!(
        quill_simd_digest(&mut simd, &text),
        scalar_ref,
        "Quill SWAR tokenizer diverged from the scalar reference on the long-token corpus"
    );

    let null = paired_median_ratio(
        41,
        16,
        || {
            black_box(quill_scalar_digest(black_box(&text)));
        },
        || {
            black_box(quill_scalar_digest(black_box(&text)));
        },
    );
    let mut simd_a = FrankensearchTokenizer::default();
    let lever = paired_median_ratio(
        41,
        16,
        || {
            black_box(quill_scalar_digest(black_box(&text)));
        },
        || {
            black_box(quill_simd_digest(&mut simd_a, black_box(&text)));
        },
    );
    eprintln!(
        "[null]  tokenizer_simd_long/{}KiB: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        text.len() / 1024,
        null.median,
        null.p5,
        null.p95,
        null.rounds
    );
    eprintln!(
        "[lever] tokenizer_simd_long/{}KiB: simd/scalar median {:.4} p5 {:.4} p95 {:.4} -> {}",
        text.len() / 1024,
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (SWAR classifier faster)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut group = c.benchmark_group("long_token_tokenizer");
    group.sample_size(30);
    group.bench_function("quill_scalar/48KiB", |b| {
        b.iter(|| black_box(quill_scalar_digest(black_box(&text))));
    });
    group.bench_function("quill_simd/48KiB", |b| {
        let mut analyzer = FrankensearchTokenizer::default();
        b.iter(|| black_box(quill_simd_digest(&mut analyzer, black_box(&text))));
    });
    group.finish();
}

criterion_group!(benches, bench_default_tokenizer, bench_long_token_tokenizer);
criterion_main!(benches);
