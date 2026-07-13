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
use frankensearch_lexical::cass_compat::{cass_cjk_collect_fast, cass_cjk_collect_slow};

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

fn bench(c: &mut Criterion) {
    let tokens = cjk_tokens();

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
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
