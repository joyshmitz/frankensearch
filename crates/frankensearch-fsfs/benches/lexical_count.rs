//! `count_lexical_tokens` ASCII fast-path benchmark.
//! Old: `chars()` UTF-8 decode loop. New: byte loop for ASCII text. Bit-identical
//! for ASCII (is_token_byte(b) == is_token_char(b as char)); the count state
//! machine is unchanged.
use std::hint::black_box;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[inline]
fn is_token_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | ':')
}
#[inline]
fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'/' | b':')
}
// OLD: chars() loop.
fn count_old(text: &str) -> usize {
    let mut count = 0;
    let mut in_token = false;
    for ch in text.chars() {
        if is_token_char(ch) { in_token = true; }
        else if in_token { in_token = false; count += 1; }
    }
    count + usize::from(in_token)
}
// NEW: ASCII byte fast path.
fn count_new(text: &str) -> usize {
    if text.is_ascii() {
        let mut count = 0;
        let mut in_token = false;
        for &b in text.as_bytes() {
            if is_token_byte(b) { in_token = true; }
            else if in_token { in_token = false; count += 1; }
        }
        return count + usize::from(in_token);
    }
    count_old(text)
}
// LUT candidate: 256-byte class table + branchless transition counting. Each byte
// is one table load; a token is counted at every token→non-token transition via
// `prev & !cur` (no data-dependent `in_token` branch). Bit-identical to count_new.
const TOKEN_BYTE: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        let b = i as u8;
        t[i] = (b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'/' | b':')) as u8;
        i += 1;
    }
    t
};
fn count_lut(text: &str) -> usize {
    if text.is_ascii() {
        let mut count = 0usize;
        let mut prev = 0u8;
        for &b in text.as_bytes() {
            let cur = TOKEN_BYTE[b as usize];
            count += (prev & !cur) as usize;
            prev = cur;
        }
        return count + prev as usize;
    }
    count_old(text)
}
// Realistic ASCII code/doc chunk.
fn make_text(n: usize) -> String {
    let base = "src/main.rs -> fn run_fast(x: i32) { return x + foo.bar.baz(qux); } // http://example.com:8080/path token_count ";
    base.repeat(n / base.len() + 1)
}
fn bench_count(c: &mut Criterion) {
    let mut g = c.benchmark_group("lexical_count");
    for n in [1024usize, 4096, 16384] {
        let text = make_text(n);
        debug_assert_eq!(count_old(&text), count_new(&text));
        assert_eq!(count_new(&text), count_lut(&text), "LUT count must match byte path (n{n})");
        let id = format!("ascii_{n}");
        g.bench_with_input(BenchmarkId::new("chars", &id), &(), |b, ()| b.iter(|| black_box(count_old(black_box(&text)))));
        g.bench_with_input(BenchmarkId::new("bytes", &id), &(), |b, ()| b.iter(|| black_box(count_new(black_box(&text)))));
        g.bench_with_input(BenchmarkId::new("lut", &id), &(), |b, ()| b.iter(|| black_box(count_lut(black_box(&text)))));
    }
    g.finish();
}
criterion_group!(benches, bench_count);
criterion_main!(benches);
