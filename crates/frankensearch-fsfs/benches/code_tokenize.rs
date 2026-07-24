//! `code_structure_sidecar::tokenize` ASCII fast-path bench.
//! Old: `chars().flat_map(char::to_lowercase)` (per-char `ToLowercase` iterator).
//! New: byte loop with `to_ascii_lowercase` for ASCII input. Bit-identical for
//! ASCII; non-ASCII falls back to the Unicode path.
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::collections::BTreeSet;
use std::hint::black_box;

fn tok_old(value: &str) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    let mut current = String::new();
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
fn tok_new(value: &str) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    let mut current = String::new();
    if value.is_ascii() {
        for &b in value.as_bytes() {
            let lo = b.to_ascii_lowercase();
            if lo.is_ascii_alphanumeric() {
                current.push(lo as char);
            } else if !current.is_empty() {
                tokens.insert(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            tokens.insert(current);
        }
        return tokens;
    }
    tok_old(value)
}
fn make(n: usize) -> String {
    let base = "pub fn run_fast(x: i32) -> Result<Vec<Symbol>, Error> { self.rank_symbols(query).map(Into::into) } async def Rank_Symbols ";
    base.repeat(n / base.len() + 1)
}
fn bench_tok(c: &mut Criterion) {
    let mut g = c.benchmark_group("code_tokenize");
    for n in [256usize, 1024, 4096] {
        let s = make(n);
        debug_assert_eq!(tok_old(&s), tok_new(&s));
        let id = format!("ascii_{n}");
        g.bench_with_input(BenchmarkId::new("unicode", &id), &(), |b, ()| {
            b.iter(|| black_box(tok_old(black_box(&s))));
        });
        g.bench_with_input(BenchmarkId::new("ascii", &id), &(), |b, ()| {
            b.iter(|| black_box(tok_new(black_box(&s))));
        });
    }
    g.finish();
}
criterion_group!(benches, bench_tok);
criterion_main!(benches);
