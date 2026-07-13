//! Index-time A/B: one-pass `collapse_code_block` vs the `format!`/`join` form.
//!
//! `collapse_code_block` runs during canonicalization for **every fenced code
//! block in every document** (collapse a block to first N + last M lines). The
//! original built an intermediate joined `String` (`lines.join("\n")` — a full
//! copy of the kept bytes) that `format!` then copied a *second* time into the
//! returned `String`; the collapse branch did this twice (head + tail). The
//! shipped version writes the kept lines straight into the output buffer via
//! `push_joined`, copying each byte once. Byte-identical
//! (`canonicalize::tests::collapse_code_block_matches_slow`), so canonicalized
//! text — hence tokens/embeddings — is unchanged.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-core --features bench-internals --bench collapse_code_block_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_core::canonicalize::{
    code_block_language_borrowed_bench, code_block_language_owned_bench,
    collapse_code_block_fast_bench, collapse_code_block_slow, push_collapsed_code_block_fast_bench,
};

const HEAD: usize = 20;
const TAIL: usize = 10;

/// A realistic mix of code blocks: some short (kept in full) and some long
/// (collapsed to head+tail), the shape canonicalization sees from technical docs
/// and chat logs.
fn code_blocks() -> Vec<Vec<String>> {
    let template = [
        "fn parse_rfc3339(raw: &str) -> Option<i64> {",
        "    let bytes = raw.as_bytes();",
        "    if bytes.len() < 20 { return None; }",
        "    let year = parse_u32(&bytes[0..4])?;",
        "    let month = parse_u32(&bytes[5..7])?;",
        "    // accumulate days since the civil epoch",
        "    let days = days_from_civil(year, month, day);",
        "    Some(days * 86_400 + secs)",
        "}",
        "",
    ];
    let mut blocks = Vec::with_capacity(512);
    let mut r = 0x2545_f491_4f6c_dd1d_u64;
    for _ in 0..512 {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        // Half the blocks are short (<= head+tail, full keep), half are long (collapse).
        let len = if r & 1 == 0 {
            2 + (r >> 8) as usize % (HEAD + TAIL - 2) // 2..=29 -> full keep
        } else {
            (HEAD + TAIL) + 1 + (r >> 8) as usize % 200 // 31..=230 -> collapse
        };
        let mut block = Vec::with_capacity(len);
        for i in 0..len {
            block.push(template[i % template.len()].to_string());
        }
        blocks.push(block);
    }
    blocks
}

fn fence_headers() -> Vec<&'static str> {
    let pool = [
        "```",
        "```rust",
        "``` python ",
        "````typescript",
        "```   ",
        "```python-with-a-long-language-name",
    ];
    (0..4096).map(|i| pool[i % pool.len()]).collect()
}

fn bench(c: &mut Criterion) {
    let blocks = code_blocks();
    let headers = fence_headers();
    let refs: Vec<Vec<&str>> = blocks
        .iter()
        .map(|b| b.iter().map(String::as_str).collect())
        .collect();

    // Parity before timing.
    for lines in &refs {
        assert_eq!(
            collapse_code_block_fast_bench("rust", lines, HEAD, TAIL),
            collapse_code_block_slow("rust", lines, HEAD, TAIL),
        );
    }

    for header in &headers {
        assert_eq!(
            code_block_language_owned_bench(header),
            code_block_language_borrowed_bench(header),
        );
    }
    let run_owned_lang = || {
        let mut acc = 0usize;
        for header in &headers {
            let lang = code_block_language_owned_bench(black_box(header));
            acc = acc.wrapping_add(black_box(lang.len()));
        }
        black_box(acc);
    };
    let run_borrowed_lang = || {
        let mut acc = 0usize;
        for header in &headers {
            let lang = code_block_language_borrowed_bench(black_box(header));
            acc = acc.wrapping_add(black_box(lang.len()));
        }
        black_box(acc);
    };
    let lang_null = paired_median_ratio(41, 16, run_owned_lang, run_owned_lang);
    let lang_lever = paired_median_ratio(41, 16, run_owned_lang, run_borrowed_lang);
    eprintln!(
        "[null]  code_block_lang/{}hdr: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        headers.len(),
        lang_null.median,
        lang_null.p5,
        lang_null.p95,
        lang_null.rounds
    );
    eprintln!(
        "[lever] code_block_lang/{}hdr: borrowed/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        headers.len(),
        lang_lever.median,
        lang_lever.p5,
        lang_lever.p95,
        if lang_lever.decidable_against(&lang_null) {
            if lang_lever.median < 1.0 {
                "DECIDABLE WIN (borrow language)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let caller_capacity: usize = refs
        .iter()
        .map(|lines| {
            "```rust\n".len()
                + lines.iter().map(|line| line.len() + 1).sum::<usize>()
                + "```\n".len()
        })
        .sum();
    let build_via_returned = || {
        let mut out = String::with_capacity(caller_capacity);
        for lines in &refs {
            out.push_str(&collapse_code_block_fast_bench(
                black_box("rust"),
                black_box(lines),
                HEAD,
                TAIL,
            ));
            out.push('\n');
        }
        out
    };
    let build_via_append = || {
        let mut out = String::with_capacity(caller_capacity);
        for lines in &refs {
            push_collapsed_code_block_fast_bench(
                &mut out,
                black_box("rust"),
                black_box(lines),
                HEAD,
                TAIL,
            );
            out.push('\n');
        }
        out
    };
    assert_eq!(build_via_returned(), build_via_append());

    let run_via_returned = || {
        black_box(build_via_returned());
    };
    let run_via_append = || {
        black_box(build_via_append());
    };

    let append_null = paired_median_ratio(41, 4, run_via_returned, run_via_returned);
    let append_lever = paired_median_ratio(41, 4, run_via_returned, run_via_append);
    eprintln!(
        "[null]  collapse_append/{}blk: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        refs.len(),
        append_null.median,
        append_null.p5,
        append_null.p95,
        append_null.rounds
    );
    eprintln!(
        "[lever] collapse_append/{}blk: append/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        refs.len(),
        append_lever.median,
        append_lever.p5,
        append_lever.p95,
        if append_lever.decidable_against(&append_null) {
            if append_lever.median < 1.0 {
                "DECIDABLE WIN (direct append)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let run_fast = || {
        let mut acc = 0usize;
        for lines in &refs {
            acc += black_box(collapse_code_block_fast_bench(
                black_box("rust"),
                black_box(lines),
                HEAD,
                TAIL,
            ))
            .len();
        }
        black_box(acc);
    };
    let run_slow = || {
        let mut acc = 0usize;
        for lines in &refs {
            acc += black_box(collapse_code_block_slow(
                black_box("rust"),
                black_box(lines),
                HEAD,
                TAIL,
            ))
            .len();
        }
        black_box(acc);
    };

    // NULL (slow vs slow) then lever (slow=ORIG vs fast). Ratio = fast/ORIG, <1.0 = one-pass wins.
    let null = paired_median_ratio(41, 4, run_slow, run_slow);
    let lever = paired_median_ratio(41, 4, run_slow, run_fast);
    eprintln!(
        "[null]  collapse_code/{}blk: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        refs.len(),
        null.median,
        null.p5,
        null.p95,
        null.rounds
    );
    eprintln!(
        "[lever] collapse_code/{}blk: fast/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        refs.len(),
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (one-pass build)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut g = c.benchmark_group("collapse_code_block");
    g.sample_size(30);
    g.bench_function("join_format", |b| b.iter(run_slow));
    g.bench_function("one_pass", |b| b.iter(run_fast));
    g.bench_function("caller_return_push", |b| b.iter(run_via_returned));
    g.bench_function("caller_direct_append", |b| b.iter(run_via_append));
    g.bench_function("language_owned", |b| b.iter(run_owned_lang));
    g.bench_function("language_borrowed", |b| b.iter(run_borrowed_lang));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
