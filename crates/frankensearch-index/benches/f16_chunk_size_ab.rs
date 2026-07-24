//! Scan-tuning A/B (cc lane, bd-b5wl-adjacent): does widening the **f16** flat-scan parallel chunk
//! size cut merge fan-in enough to win, the way cod's int8 widening did (`fd06f77`, 4×, ~1.12–1.17×)?
//!
//! The f16 scan uses `PARALLEL_CHUNK_SIZE = 1024`; the int8 scan uses `4096`. The int8 dot is cheap,
//! so its merge fan-in dominated and widening paid. The f16 dot is `cvtph2ps`-decode-bound (more
//! expensive per vector), so merge fan-in is a smaller fraction — widening may win less, or wash.
//! This decides it. Chunk size does NOT affect results (the scan is exact and chunk-invariant: any
//! global top-k item is in the top-k of its own chunk regardless of boundaries), so recall/ordering
//! is trivially preserved — asserted identical before timing. Measured via `search_top_k_with_params`
//! (a runtime `SearchParams.parallel_chunk_size`), so NO source change is needed to A/B.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-index --features bench-internals --bench f16_chunk_size_ab
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_index::{SearchParams, VectorIndex};

const DIM: usize = 384;
const N: usize = 40_000;
const CLUSTERS: usize = 64;
const QUERIES: usize = 16;
const K: usize = 10;
const ORIG_CHUNK: usize = 1_024; // current f16 default
const WIDE_CHUNK: usize = 4_096; // int8's widened value

fn raw(seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..DIM)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let sample = u16::try_from(s >> 48).expect("upper 16 bits fit u16");
            f32::from(sample) / 65_536.0 - 0.5
        })
        .collect()
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    v
}

fn make_vector(centroids: &[Vec<f32>], cluster: usize, seed: u64) -> Vec<f32> {
    let c = &centroids[cluster];
    let j = raw(seed);
    normalize(c.iter().zip(&j).map(|(a, b)| a + 0.15 * b).collect())
}

fn params(chunk: usize) -> SearchParams {
    SearchParams {
        parallel_chunk_size: chunk,
        ..SearchParams::default()
    }
}

fn bench(c: &mut Criterion) {
    let dir = std::env::temp_dir().join(format!("f16_chunk_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("bench dir");
    let path = dir.join("index.idx");

    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(raw(0xc000_0000 + i as u64)))
        .collect();
    let mut writer = VectorIndex::create(&path, "bench-384", DIM).expect("create fsvi");
    for i in 0..N {
        let v = make_vector(&centroids, i % CLUSTERS, i as u64 + 1);
        writer
            .write_record(&format!("doc-{i:06}"), &v)
            .expect("write record");
    }
    writer.finish().expect("finish fsvi");
    let index = VectorIndex::open(&path).expect("open fsvi");

    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    // PARITY GATE: chunk size must not change results (exact + chunk-invariant).
    for q in &queries {
        let a: Vec<(String, f32)> = index
            .search_top_k_with_params(q, K, None, params(ORIG_CHUNK))
            .expect("orig")
            .into_iter()
            .map(|h| (h.doc_id.to_string(), h.score))
            .collect();
        let b: Vec<(String, f32)> = index
            .search_top_k_with_params(q, K, None, params(WIDE_CHUNK))
            .expect("wide")
            .into_iter()
            .map(|h| (h.doc_id.to_string(), h.score))
            .collect();
        assert_eq!(a.len(), b.len());
        for ((ad, asc), (bd, bsc)) in a.iter().zip(&b) {
            assert_eq!(ad, bd, "chunk size changed the ranking");
            assert_eq!(asc.to_bits(), bsc.to_bits(), "chunk size changed a score");
        }
    }

    let do_scan = |chunk: usize| {
        for q in &queries {
            black_box(
                index
                    .search_top_k_with_params(black_box(q), K, None, params(chunk))
                    .expect("scan"),
            );
        }
    };

    // NULL (orig vs orig) then lever (orig vs wide). Ratio = wide/ORIG, <1.0 = wider is faster.
    let null = paired_median_ratio(41, 2, || do_scan(ORIG_CHUNK), || do_scan(ORIG_CHUNK));
    let lever = paired_median_ratio(41, 2, || do_scan(ORIG_CHUNK), || do_scan(WIDE_CHUNK));
    eprintln!(
        "[null]  f16_scan/{N}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] f16_scan/{N}: chunk{WIDE_CHUNK}/chunk{ORIG_CHUNK} median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (wider faster)"
            } else {
                "DECIDABLE REGRESSION (wider slower)"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    let mut g = c.benchmark_group("f16_chunk_size");
    g.sample_size(20);
    g.bench_function("chunk1024", |b| b.iter(|| do_scan(ORIG_CHUNK)));
    g.bench_function("chunk4096", |b| b.iter(|| do_scan(WIDE_CHUNK)));
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
