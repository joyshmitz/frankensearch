//! File-classification byte-sniff benchmark (per-file content scan).
//! Old: per-byte `u32::saturating_add` (blocks vectorization). New: branchless
//! `u64` accumulators saturate-cast to `u32` (SIMD histograms). Bit-identical.
use std::hint::black_box;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[inline]
fn is_non_printable(byte: u8) -> bool {
    !matches!(byte, b'\t' | b'\n' | b'\r') && (byte < 0x20 || byte == 0x7f)
}
fn sniff_old(bytes: &[u8]) -> (u32, f64, f64) {
    if bytes.is_empty() { return (0, 0.0, 0.0); }
    let (mut nb, mut np, mut hb) = (0u32, 0u32, 0u32);
    for &b in bytes {
        if b == 0 { nb = nb.saturating_add(1); }
        if is_non_printable(b) { np = np.saturating_add(1); }
        if b >= 0x80 { hb = hb.saturating_add(1); }
    }
    let l = bytes.len() as f64;
    (nb, f64::from(np) / l, f64::from(hb) / l)
}
fn sniff_new(bytes: &[u8]) -> (u32, f64, f64) {
    if bytes.is_empty() { return (0, 0.0, 0.0); }
    let (mut nb, mut np, mut hb) = (0u64, 0u64, 0u64);
    for &b in bytes {
        nb += u64::from(b == 0);
        np += u64::from(is_non_printable(b));
        hb += u64::from(b >= 0x80);
    }
    let nb = u32::try_from(nb).unwrap_or(u32::MAX);
    let np = u32::try_from(np).unwrap_or(u32::MAX);
    let hb = u32::try_from(hb).unwrap_or(u32::MAX);
    let l = bytes.len() as f64;
    (nb, f64::from(np) / l, f64::from(hb) / l)
}
fn make_probe(n: usize) -> Vec<u8> {
    let mut s = 0x9e37_79b9_7f4a_7c15_u64 ^ (n as u64);
    (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; let r = (s >> 56) as u8; if r < 205 { 0x20 + (r % 0x5f) } else { r } }).collect()
}
fn bench_sniff(c: &mut Criterion) {
    let mut g = c.benchmark_group("sniff_features");
    for n in [4096usize, 16384, 65536] {
        let probe = make_probe(n);
        debug_assert_eq!(sniff_old(&probe), sniff_new(&probe));
        let id = format!("probe_{n}");
        g.bench_with_input(BenchmarkId::new("old", &id), &(), |b, ()| b.iter(|| black_box(sniff_old(black_box(&probe)))));
        g.bench_with_input(BenchmarkId::new("new", &id), &(), |b, ()| b.iter(|| black_box(sniff_new(black_box(&probe)))));
    }
    g.finish();
}
criterion_group!(benches, bench_sniff);
criterion_main!(benches);
