//! CJK bigram-decompose A/B: one fused UTF-8 decode pass vs the two-pass form.
//!
//! `CjkBigramDecomposeStream::decompose_cjk` runs at index time (per doc) and
//! query time (per query) on every all-CJK token. CJK isn't whitespace
//! delimited, so these tokens are *long runs* — exactly why the bigram
//! decomposer exists. The naive form decoded the run twice: a guard scan
//! `chars().all(is_cjk)` then `chars().collect::<Vec<char>>()`. The shipped
//! `cass_cjk_collect_fast` fuses both into one pass (validate while collecting,
//! bail on the first non-CJK char with no allocation for ASCII tokens). It is
//! byte-identical (`cass_compat::tests::cass_cjk_collect_fast_matches_slow`) so
//! token boundaries — hence recall/ordering — are unchanged. This measures the
//! speedup on a realistic CJK token corpus.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-lexical --features bench-internals --bench cjk_bigram_decompose_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_lexical::cass_compat::{
    cass_cjk_bigrams_direct, cass_cjk_bigrams_staged, cass_cjk_collect_fast, cass_cjk_collect_slow,
};
use tantivy::tokenizer::Token;

/// Realistic all-CJK token corpus: runs of Han / Hiragana / Katakana / Hangul
/// with no whitespace (the shape the bigram decomposer actually receives).
fn cjk_tokens() -> Vec<String> {
    let pool: Vec<char> = "搜索引擎向量嵌入検索とうきょうデータ한국어처리文档分析語彙"
        .chars()
        .collect();
    let mut tokens = Vec::with_capacity(2048);
    let mut r = 0x1234_5678_9abc_def0_u64;
    for _ in 0..2048 {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        let len = 2 + (r % 30) as usize; // 2..=31 CJK chars per token
        let mut s = String::new();
        for _ in 0..len {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            s.push(pool[(r as usize) % pool.len()]);
        }
        tokens.push(s);
    }
    tokens
}

fn assert_token_parity(original: &[Token], candidate: &[Token]) {
    assert_eq!(original.len(), candidate.len());
    for (original, candidate) in original.iter().zip(candidate) {
        assert_eq!(original.text, candidate.text);
        assert_eq!(original.position, candidate.position);
        assert_eq!(original.offset_from, candidate.offset_from);
        assert_eq!(original.offset_to, candidate.offset_to);
        assert_eq!(original.position_length, candidate.position_length);
    }
}

fn bench(c: &mut Criterion) {
    let tokens = cjk_tokens();
    let template = Token {
        text: String::new(),
        position: 7,
        offset_from: 11,
        offset_to: 42,
        position_length: 3,
    };

    // Parity before timing: fused == two-pass on the decompose path...
    for t in &tokens {
        assert_eq!(cass_cjk_collect_fast(t), cass_cjk_collect_slow(t));
    }
    // ...and on every reject path.
    for t in ["", "a", "hello world", "搜", "搜a", "a搜", "搜 索"] {
        assert_eq!(
            cass_cjk_collect_fast(t),
            cass_cjk_collect_slow(t),
            "parity for {t:?}"
        );
    }

    let mut staged = Vec::new();
    let mut direct = Vec::new();
    for t in &tokens {
        staged.clear();
        direct.clear();
        cass_cjk_bigrams_staged(t, &template, &mut staged);
        cass_cjk_bigrams_direct(t, &template, &mut direct);
        assert_token_parity(&staged, &direct);
    }

    let run_fast = || {
        let mut acc = 0usize;
        for t in &tokens {
            acc += black_box(cass_cjk_collect_fast(black_box(t))).map_or(0, |v| v.len());
        }
        black_box(acc);
    };
    let run_slow = || {
        let mut acc = 0usize;
        for t in &tokens {
            acc += black_box(cass_cjk_collect_slow(black_box(t))).map_or(0, |v| v.len());
        }
        black_box(acc);
    };
    let run_staged = || {
        let mut pending = Vec::new();
        let mut acc = 0usize;
        for t in &tokens {
            pending.clear();
            cass_cjk_bigrams_staged(black_box(t), black_box(&template), &mut pending);
            acc = acc.wrapping_add(pending.len());
            black_box(&pending);
        }
        black_box(acc);
    };
    let run_direct = || {
        let mut pending = Vec::new();
        let mut acc = 0usize;
        for t in &tokens {
            pending.clear();
            cass_cjk_bigrams_direct(black_box(t), black_box(&template), &mut pending);
            acc = acc.wrapping_add(pending.len());
            black_box(&pending);
        }
        black_box(acc);
    };

    let pending_null = paired_median_ratio(41, 4, run_staged, run_staged);
    let pending_lever = paired_median_ratio(41, 4, run_staged, run_direct);
    eprintln!(
        "[null]  cjk_pending/{}tok: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        tokens.len(),
        pending_null.median,
        pending_null.p5,
        pending_null.p95,
        pending_null.rounds
    );
    eprintln!(
        "[lever] cjk_pending/{}tok: direct/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        tokens.len(),
        pending_lever.median,
        pending_lever.p5,
        pending_lever.p95,
        if pending_lever.decidable_against(&pending_null) {
            if pending_lever.median < 1.0 {
                "DECIDABLE WIN (direct pending)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    // NULL (slow vs slow) then lever (slow=ORIG vs fast). Ratio = fast/ORIG, <1.0 = fused wins.
    let null = paired_median_ratio(41, 4, run_slow, run_slow);
    let lever = paired_median_ratio(41, 4, run_slow, run_fast);
    eprintln!(
        "[null]  cjk_decompose/{}tok: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        tokens.len(),
        null.median,
        null.p5,
        null.p95,
        null.rounds
    );
    eprintln!(
        "[lever] cjk_decompose/{}tok: fast/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        tokens.len(),
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (one-pass fused)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut g = c.benchmark_group("cjk_bigram_decompose");
    g.sample_size(30);
    g.bench_function("two_pass", |b| b.iter(run_slow));
    g.bench_function("one_pass", |b| b.iter(run_fast));
    g.bench_function("staged_pending", |b| b.iter(run_staged));
    g.bench_function("direct_pending", |b| b.iter(run_direct));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
