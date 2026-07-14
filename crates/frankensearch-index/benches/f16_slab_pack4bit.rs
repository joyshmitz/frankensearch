//! mmap f16-byte slab → signed-4-bit packing: shipped portable path vs AVX2+F16C.
//!
//! The file-backed FSVI 4-bit scan lazily builds its packed pass-1 slab directly
//! from mmap little-endian f16 bytes. The shipped path SIMD-widens decode but still
//! rounds, clamps, and packs every nibble scalarly; the candidate reuses the
//! in-memory path's AVX2+F16C quantize-and-pack pipeline directly over those bytes.
//!
//! Arms: two shipped-path null controls around the runtime-dispatched candidate.
//! The pre-timing assertion requires byte-identical output at dim=384.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p frankensearch-index \
//!   --profile release --bench f16_slab_pack4bit -- f16_slab_pack4bit --noplot
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_index::{pack_f16_le_bytes_to_4bit, pack_f16_le_bytes_to_4bit_generic};
use half::f16;

const DIM: usize = 384;

fn slab_fixture(vectors: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(vectors * DIM * 2);
    let mut s = 0x2545_f491_4f6c_dd1d_u64;
    for _ in 0..vectors * DIM {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let x = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.12;
        out.extend_from_slice(&f16::from_f32(x).to_le_bytes());
    }
    out
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("f16_slab_pack4bit");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(600));

    let vectors = 10_000_usize;
    let slab = slab_fixture(vectors);
    assert_eq!(
        pack_f16_le_bytes_to_4bit_generic(&slab, DIM),
        pack_f16_le_bytes_to_4bit(&slab, DIM),
        "AVX2 pack must be byte-identical to the shipped mmap path"
    );

    group.throughput(criterion::Throughput::Elements((vectors * DIM) as u64));
    group.bench_with_input(BenchmarkId::new("current_a", vectors), &slab, |bn, slab| {
        bn.iter(|| black_box(pack_f16_le_bytes_to_4bit_generic(black_box(slab), DIM)));
    });
    group.bench_with_input(
        BenchmarkId::new("avx2_dispatch", vectors),
        &slab,
        |bn, slab| {
            bn.iter(|| black_box(pack_f16_le_bytes_to_4bit(black_box(slab), DIM)));
        },
    );
    group.bench_with_input(BenchmarkId::new("current_b", vectors), &slab, |bn, slab| {
        bn.iter(|| black_box(pack_f16_le_bytes_to_4bit_generic(black_box(slab), DIM)));
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
