//! Model2Vec embed GATHER software-prefetch A/B.
//!
//! `Model2VecEmbedder::embed_sync` mean-pools token vectors: for each token id it
//! indexes a random row `emb[id*DIM .. id*DIM+DIM]` of the `[vocab, DIM]` static
//! table and AVX2-accumulates it into `sum`. For a real model (potion-base-8M:
//! vocab ~30k, DIM 256 → a ~30 MB table) each token's row is a RANDOM location that
//! the hardware stride-prefetcher cannot predict → a first-touch cache miss per
//! token. The accumulate itself is cheap (DIM/8 AVX2 adds), so the loop is
//! MEMORY-LATENCY-bound on the indirect gather — the same shape as the CLS-attention
//! prefetch win, but on an indirect scatter instead of a fixed stride.
//!
//! The whole token-id sequence is known upfront, so `emb`'s row for token `i+PF` can
//! be `_mm_prefetch`'d while token `i`'s row accumulates — hiding the miss latency.
//! Prefetch is a hint → the accumulated `sum` is BIT-IDENTICAL to the base loop
//! (parity asserts max-delta 0), so this is exact and distribution-independent in
//! correctness.
//!
//! Arms: `base` (= shipped loop) vs `pf_head` (prefetch the row's first line only —
//! the accumulate then streams the rest via the HW next-line prefetcher) vs `pf_row`
//! (prefetch all DIM/16 lines of the row). Swept over token count T (query→doc).
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-embed --profile release \
//!     --bench model2vec_gather_prefetch
//! ```

use std::hint::black_box;
use std::time::Duration;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_embed::simd::accumulate_f32_into;

const VOCAB: usize = 30_000; // potion-base-8M-class vocab
const DIM: usize = 256; // potion-base-8M dimension → 30k*256*4 ≈ 30 MB table
const PF: usize = 4; // prefetch distance (tokens ahead)
const LINE_F32: usize = 16; // 64-byte cache line = 16 f32

/// Prefetch the row starting at `row_start` (f32 offset). `full` = every cache line
/// of the DIM-wide row; else just the first line (HW streams the rest).
#[inline(always)]
#[allow(unsafe_code)]
fn prefetch_row(_emb: &[f32], _row_start: usize, _full: bool) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: _mm_prefetch is a hint; any address is architecturally valid to
    // prefetch and we bound the offset by the slice length anyway.
    unsafe {
        if !_full {
            if _row_start < _emb.len() {
                _mm_prefetch(_emb.as_ptr().add(_row_start) as *const i8, _MM_HINT_T0);
            }
            return;
        }
        let mut off = _row_start;
        let end = (_row_start + DIM).min(_emb.len());
        while off < end {
            _mm_prefetch(_emb.as_ptr().add(off) as *const i8, _MM_HINT_T0);
            off += LINE_F32;
        }
    }
}

/// SHIPPED: gather each token's row and accumulate, no prefetch (embed_sync loop).
fn gather_base(emb: &[f32], ids: &[u32], sum: &mut [f32]) {
    sum.fill(0.0);
    for &id in ids {
        let idx = id as usize;
        if idx < VOCAB {
            let start = idx * DIM;
            accumulate_f32_into(sum, &emb[start..start + DIM]);
        }
    }
}

/// CANDIDATE: prefetch token `i+PF`'s row while accumulating token `i`.
fn gather_prefetch(emb: &[f32], ids: &[u32], sum: &mut [f32], full: bool) {
    sum.fill(0.0);
    let n = ids.len();
    for i in 0..n {
        if i + PF < n {
            let pidx = ids[i + PF] as usize;
            if pidx < VOCAB {
                prefetch_row(emb, pidx * DIM, full);
            }
        }
        let idx = ids[i] as usize;
        if idx < VOCAB {
            let start = idx * DIM;
            accumulate_f32_into(sum, &emb[start..start + DIM]);
        }
    }
}

fn emb_fixture() -> Vec<f32> {
    let mut out = vec![0.0f32; VOCAB * DIM];
    let mut s = 0x9e37_79b9_7f4a_7c15_u64;
    for v in &mut out {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
    }
    out
}

fn ids_fixture(t: usize) -> Vec<u32> {
    // Uniform-random token ids = the cache-cold regime (broad-vocab doc embedding
    // over a 30 MB table). Realistic Zipfian text keeps hot tokens cached and would
    // see a smaller gather benefit; this is the gather-heavy case.
    let mut out = Vec::with_capacity(t);
    let mut s = 0x1234_5678_9abc_def0_u64;
    for _ in 0..t {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.push(((s >> 33) as usize % VOCAB) as u32);
    }
    out
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("model2vec_gather_prefetch");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_millis(1000));

    let emb = emb_fixture();

    for &t in &[16usize, 64, 256] {
        let ids = ids_fixture(t);

        // Parity: prefetch is a hint → sum is bit-identical to the base loop.
        let mut a = vec![0.0f32; DIM];
        let mut bh = vec![0.0f32; DIM];
        let mut br = vec![0.0f32; DIM];
        gather_base(&emb, &ids, &mut a);
        gather_prefetch(&emb, &ids, &mut bh, false);
        gather_prefetch(&emb, &ids, &mut br, true);
        let dh = a.iter().zip(&bh).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let dr = a.iter().zip(&br).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(dh == 0.0, "pf_head diverged from base by {dh} (t={t}) — must be bit-identical");
        assert!(dr == 0.0, "pf_row diverged from base by {dr} (t={t}) — must be bit-identical");

        group.bench_with_input(BenchmarkId::new("base", t), &ids, |bn, ids| {
            let mut sum = vec![0.0f32; DIM];
            bn.iter(|| {
                gather_base(black_box(&emb), black_box(ids), black_box(&mut sum));
                black_box(&sum);
            });
        });
        group.bench_with_input(BenchmarkId::new("pf_head", t), &ids, |bn, ids| {
            let mut sum = vec![0.0f32; DIM];
            bn.iter(|| {
                gather_prefetch(black_box(&emb), black_box(ids), black_box(&mut sum), false);
                black_box(&sum);
            });
        });
        group.bench_with_input(BenchmarkId::new("pf_row", t), &ids, |bn, ids| {
            let mut sum = vec![0.0f32; DIM];
            bn.iter(|| {
                gather_prefetch(black_box(&emb), black_box(ids), black_box(&mut sum), true);
                black_box(&sum);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
