//! S3-FIFO vs FIFO query-embedding-cache replay (bd-tjkm).
//!
//! The hot-path query-embedding cache (`CachedEmbedder` → `CacheState`) uses naive
//! **FIFO** eviction (evict oldest by insertion order; `get` does not reorder). The
//! codebase already implements **S3-FIFO** (`frankensearch_core::S3FifoCache`, Yang
//! et al. SOSP 2023) but it is not wired into the embedder cache. On skewed (Zipf)
//! and scan-polluted query streams FIFO evicts hot keys and admits one-hit-wonders;
//! S3-FIFO keeps the hot set (Main queue) and filters cold singletons (Small queue).
//!
//! A cache **miss** is an embedding recompute (~0.5 ms for a real embedder), so the
//! decisive metric is **miss count at matched capacity** — the per-op cost (ns) is
//! dwarfed by the embed it avoids. This bench replays both policies over identical
//! traces at the same effective capacity and reports hits/misses (eprintln) plus a
//! criterion timing of the replay loop. It is the replay step bd-tjkm requires
//! before any S3-FIFO wiring decision.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-embed --bench cache_replay
//! ```

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::zero_sized_map_values
)]

use std::collections::{HashMap, VecDeque};
use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::{CachePolicy, S3FifoCache, S3FifoConfig};

const DIM: usize = 384;
const ENTRY_BYTES: usize = DIM * 4; // f32 embedding
const CAP: usize = 128; // CachedEmbedder DEFAULT_CAPACITY
const ACCESSES: usize = 100_000;

/// Plain FIFO mirroring `CachedEmbedder::CacheState`: evict the oldest key on
/// overflow; `get` is membership-only (no reordering).
struct Fifo {
    set: HashMap<u64, ()>,
    order: VecDeque<u64>,
    cap: usize,
}
impl Fifo {
    fn new(cap: usize) -> Self {
        Self {
            set: HashMap::with_capacity(cap),
            order: VecDeque::with_capacity(cap),
            cap,
        }
    }
    fn get(&self, k: u64) -> bool {
        self.set.contains_key(&k)
    }
    fn insert(&mut self, k: u64) {
        if self.set.contains_key(&k) {
            return;
        }
        if self.set.len() >= self.cap
            && let Some(old) = self.order.pop_front()
        {
            self.set.remove(&old);
        }
        self.set.insert(k, ());
        self.order.push_back(k);
    }
}

/// Deterministic Zipf-ish key stream over `universe` keys: `rank = U·u^skew`
/// (skew>1 concentrates mass on low ranks). xorshift uniform, no rng dep.
fn zipf_stream(n: usize, universe: u64, skew: f64, seed: u64) -> Vec<u64> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = ((s >> 11) as f64 / (1u64 << 53) as f64).max(1e-12); // (0,1]
        let k = ((universe as f64) * u.powf(skew)) as u64 % universe;
        out.push(k);
    }
    out
}

/// Scan-polluted stream: a hot working set (`hot` keys, Zipf) interleaved with long
/// runs of unique cold keys (one-hit-wonders) — the pattern S3-FIFO's Small queue
/// is designed to absorb without evicting the hot set.
fn scan_polluted_stream(n: usize, hot: u64, seed: u64) -> Vec<u64> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n);
    let mut cold = 1_000_000u64;
    while out.len() < n {
        // burst of hot accesses (Zipf over the hot set)
        for _ in 0..8 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
            out.push(((hot as f64) * u.powf(2.0)) as u64 % hot);
        }
        // a scan run of unique cold keys
        for _ in 0..16 {
            out.push(cold);
            cold += 1;
        }
    }
    out.truncate(n);
    out
}

fn replay_fifo(stream: &[u64]) -> (usize, usize) {
    let mut c = Fifo::new(CAP);
    let (mut hits, mut misses) = (0usize, 0usize);
    for &k in stream {
        if c.get(k) {
            hits += 1;
        } else {
            misses += 1;
            c.insert(k);
        }
    }
    (hits, misses)
}

fn replay_s3fifo(stream: &[u64]) -> (usize, usize) {
    // Match the FIFO entry-capacity by byte budget: CAP entries × ENTRY_BYTES.
    let cache: S3FifoCache<u64, ()> = S3FifoCache::new(S3FifoConfig {
        max_bytes: CAP * ENTRY_BYTES,
        ..Default::default()
    });
    let (mut hits, mut misses) = (0usize, 0usize);
    for &k in stream {
        if cache.get(&k).is_some() {
            hits += 1;
        } else {
            misses += 1;
            cache.insert(k, (), ENTRY_BYTES);
        }
    }
    (hits, misses)
}

fn embedding(seed: usize) -> Vec<f32> {
    (0..DIM)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let x = ((seed * 131) ^ (i * 17)) as f32;
            x.sin()
        })
        .collect()
}

fn batch_slots_distinct(n: usize) -> Vec<Option<usize>> {
    (0..n).map(Some).collect()
}

fn batch_slots_repeated(n: usize, unique: usize) -> Vec<Option<usize>> {
    (0..n).map(|i| Some(i % unique)).collect()
}

fn fanout_clone_all(slot_miss: &[Option<usize>], embedded: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(slot_miss.len());
    out.resize_with(slot_miss.len(), || None);
    for (slot, maybe_idx) in slot_miss.iter().enumerate() {
        if let Some(idx) = *maybe_idx {
            out[slot] = Some(embedded[idx].clone());
        }
    }
    out.into_iter()
        .map(|v| v.expect("every miss slot filled"))
        .collect()
}

fn fanout_move_last(slot_miss: &[Option<usize>], embedded: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(slot_miss.len());
    out.resize_with(slot_miss.len(), || None);
    let mut miss_use_counts = vec![0_usize; embedded.len()];
    for maybe_idx in slot_miss {
        if let Some(idx) = *maybe_idx {
            miss_use_counts[idx] += 1;
        }
    }
    let mut embedded_slots: Vec<Option<Vec<f32>>> = embedded.into_iter().map(Some).collect();
    for (slot, maybe_idx) in slot_miss.iter().enumerate() {
        if let Some(idx) = *maybe_idx {
            miss_use_counts[idx] = miss_use_counts[idx].saturating_sub(1);
            let vec = if miss_use_counts[idx] == 0 {
                embedded_slots[idx].take().expect("embedding slot filled")
            } else {
                embedded_slots[idx]
                    .as_ref()
                    .expect("embedding slot filled")
                    .clone()
            };
            out[slot] = Some(vec);
        }
    }
    out.into_iter()
        .map(|v| v.expect("every miss slot filled"))
        .collect()
}

fn admit_and_fanout_current(
    keys: &[String],
    slot_miss: &[Option<usize>],
    embedded: Vec<Vec<f32>>,
) -> Vec<Vec<f32>> {
    let mut cache = HashMap::with_capacity(keys.len());
    for (idx, vec) in embedded.iter().enumerate() {
        cache.insert(keys[idx].as_str(), vec.clone());
    }
    black_box(cache.len());
    fanout_move_last(slot_miss, embedded)
}

fn admit_and_return_distinct(keys: &[String], embedded: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut cache = HashMap::with_capacity(keys.len());
    for (idx, vec) in embedded.iter().enumerate() {
        cache.insert(keys[idx].as_str(), vec.clone());
    }
    black_box(cache.len());
    embedded
}

fn bench_cache_replay(c: &mut Criterion) {
    let traces: [(&str, Vec<u64>); 3] = [
        ("zipf_s2_u2000", zipf_stream(ACCESSES, 2000, 2.0, 0xa11ce)),
        ("zipf_s3_u5000", zipf_stream(ACCESSES, 5000, 3.0, 0xb0b)),
        (
            "scan_polluted_hot256",
            scan_polluted_stream(ACCESSES, 256, 0xcafe),
        ),
    ];

    for (name, stream) in &traces {
        let (fh, fm) = replay_fifo(stream);
        let (sh, sm) = replay_s3fifo(stream);
        let fifo_hr = fh as f64 / (fh + fm) as f64;
        let s3_hr = sh as f64 / (sh + sm) as f64;
        let miss_ratio = sm as f64 / fm.max(1) as f64;
        eprintln!(
            "[cache_replay] {name} cap={CAP}: FIFO hit={:.3} ({fm} miss) | S3FIFO hit={:.3} ({sm} miss) | miss_ratio(s3/fifo)={miss_ratio:.3}",
            fifo_hr, s3_hr
        );
    }

    {
        let mut g = c.benchmark_group("cache_replay");
        for (name, stream) in &traces {
            g.bench_function(format!("fifo/{name}"), |b| {
                b.iter(|| black_box(replay_fifo(black_box(stream))));
            });
            g.bench_function(format!("s3fifo/{name}"), |b| {
                b.iter(|| black_box(replay_s3fifo(black_box(stream))));
            });
        }
        g.finish();
    }

    {
        let mut fanout = c.benchmark_group("batch_miss_fanout");
        for (label, slots, unique) in [
            ("distinct_256", batch_slots_distinct(256), 256_usize),
            ("repeat4_256", batch_slots_repeated(256, 64), 64_usize),
        ] {
            let embedded: Vec<Vec<f32>> = (0..unique).map(embedding).collect();
            assert_eq!(
                fanout_clone_all(&slots, &embedded),
                fanout_move_last(&slots, embedded.clone())
            );

            fanout.bench_with_input(BenchmarkId::new("clone_all", label), &(), |b, ()| {
                b.iter_batched(
                    || embedded.clone(),
                    |e| black_box(fanout_clone_all(black_box(&slots), &e)),
                    BatchSize::SmallInput,
                );
            });
            fanout.bench_with_input(BenchmarkId::new("move_last", label), &(), |b, ()| {
                b.iter_batched(
                    || embedded.clone(),
                    |e| black_box(fanout_move_last(black_box(&slots), e)),
                    BatchSize::SmallInput,
                );
            });
        }
        fanout.finish();
    }

    {
        let mut distinct_return = c.benchmark_group("batch_all_distinct_miss_return");
        for (label, slots, unique) in [
            ("distinct_64", batch_slots_distinct(64), 64_usize),
            ("distinct_256", batch_slots_distinct(256), 256_usize),
        ] {
            let keys: Vec<String> = (0..unique).map(|i| format!("query-{i}")).collect();
            let embedded: Vec<Vec<f32>> = (0..unique).map(embedding).collect();
            assert_eq!(
                admit_and_fanout_current(&keys, &slots, embedded.clone()),
                admit_and_return_distinct(&keys, embedded.clone())
            );

            distinct_return.bench_with_input(
                BenchmarkId::new("current_generic", label),
                &(),
                |b, ()| {
                    b.iter_batched(
                        || embedded.clone(),
                        |e| {
                            black_box(admit_and_fanout_current(
                                black_box(&keys),
                                black_box(&slots),
                                e,
                            ))
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            distinct_return.bench_with_input(
                BenchmarkId::new("direct_return", label),
                &(),
                |b, ()| {
                    b.iter_batched(
                        || embedded.clone(),
                        |e| black_box(admit_and_return_distinct(black_box(&keys), e)),
                        BatchSize::SmallInput,
                    );
                },
            );
        }
        distinct_return.finish();
    }
}

criterion_group!(benches, bench_cache_replay);
criterion_main!(benches);
