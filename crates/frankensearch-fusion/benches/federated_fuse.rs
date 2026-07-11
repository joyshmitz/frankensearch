//! Federated cross-shard RRF fuse hash-map benchmark.
//!
//! `federated::fuse_rrf` accumulates per-doc aggregates in a
//! `std::collections::HashMap<String, AggregateDoc>` — i.e. the DoS-resistant
//! **SipHash** hasher — while the sibling single-node `rrf_fuse_with_graph` path
//! uses `ahash::AHashMap`. This bench isolates the hasher swap (SipHash → aHash)
//! on the real federated merge shape: a `String` doc-id key (cloned per hit), a
//! `BTreeSet<String>` `appeared_in`, and a small template clone — so the measured
//! ratio reflects the real proportion of hash-compute vs the per-hit allocations
//! (`doc_id.clone()`, `shard_name.to_owned()`), not an inflated hash fraction.
//! The accumulate/rank logic is private, so it is replicated here with the same
//! field shape (cf. the `rrf_fuse` bench, which does the same for `FusedHitScratch`).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench federated_fuse
//! ```

use std::collections::{BTreeSet, HashMap};
use std::hint::black_box;

use ahash::RandomState;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const K: f64 = 60.0;

#[inline]
fn rank_contribution(rank: usize) -> f32 {
    (1.0 / (K + rank as f64 + 1.0)) as f32
}

/// Mirrors the allocation profile of the real `ScoredResult` template clone:
/// an owned `doc_id` plus a score (metadata is `None` in the common path, so a
/// lightweight clone is faithful).
#[derive(Clone)]
struct Hit {
    doc_id: String,
    score: f32,
}

/// Mirrors the real `AggregateDoc` field shape.
struct AggDoc {
    template: Hit,
    primary_index: String,
    primary_rank: usize,
    primary_contribution: f32,
    fused_score: f32,
    appeared_in: BTreeSet<String>,
}

#[inline]
fn accumulate<S: std::hash::BuildHasher>(
    docs: &mut HashMap<String, AggDoc, S>,
    hit: &Hit,
    shard_name: &str,
    rank: usize,
    contribution: f32,
) {
    let entry = docs.entry(hit.doc_id.to_string()).or_insert_with(|| {
        let mut template = hit.clone();
        template.score = 0.0;
        AggDoc {
            template,
            primary_index: shard_name.to_owned(),
            primary_rank: rank,
            primary_contribution: contribution,
            fused_score: 0.0,
            appeared_in: BTreeSet::new(),
        }
    });
    entry.fused_score += contribution;
    entry.appeared_in.insert(shard_name.to_owned());
    let update_primary = match contribution.total_cmp(&entry.primary_contribution) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => {
            rank < entry.primary_rank
                || (rank == entry.primary_rank && shard_name < entry.primary_index.as_str())
        }
    };
    if update_primary {
        shard_name.clone_into(&mut entry.primary_index);
        entry.primary_rank = rank;
        entry.primary_contribution = contribution;
        entry.template = hit.clone();
    }
}

fn rank_and_drain<S: std::hash::BuildHasher>(mut docs: HashMap<String, AggDoc, S>) -> usize {
    let mut output: Vec<(String, f32, usize)> = docs
        .drain()
        .map(|(id, agg)| (id, agg.fused_score, agg.appeared_in.len()))
        .collect();
    // Real path: total-order sort; replicate an unstable sort on the fused score.
    output.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    output.len()
}

struct Shard {
    name: String,
    hits: Vec<Hit>,
}

/// Build `shards` shards of `per_shard` hits each, drawing doc ids from a shared
/// pool of `universe` ids so the same doc appears across shards (the dedup the
/// fuse map exists to do). Deterministic — no RNG.
fn make_shards(shards: usize, per_shard: usize, universe: usize) -> Vec<Shard> {
    (0..shards)
        .map(|s| {
            let hits = (0..per_shard)
                .map(|i| {
                    // Stride the pool per shard so overlap is ~50% across shards.
                    let id = (s * (per_shard / 2) + i) % universe;
                    Hit {
                        doc_id: format!("doc_{id:07}").into(),
                        score: 1.0 - (i as f32) / (per_shard as f32),
                    }
                })
                .collect();
            Shard {
                name: format!("shard_{s:02}"),
                hits,
            }
        })
        .collect()
}

fn fuse_sip(shards: &[Shard]) -> usize {
    let mut docs: HashMap<String, AggDoc> = HashMap::new();
    for shard in shards {
        for (rank, hit) in shard.hits.iter().enumerate() {
            accumulate(&mut docs, hit, &shard.name, rank, rank_contribution(rank));
        }
    }
    rank_and_drain(docs)
}

fn fuse_ahash(shards: &[Shard]) -> usize {
    let mut docs: HashMap<String, AggDoc, RandomState> = HashMap::with_hasher(RandomState::new());
    for shard in shards {
        for (rank, hit) in shard.hits.iter().enumerate() {
            accumulate(&mut docs, hit, &shard.name, rank, rank_contribution(rank));
        }
    }
    rank_and_drain(docs)
}

fn bench_federated(c: &mut Criterion) {
    let mut g = c.benchmark_group("federated_fuse");
    // (shards, per_shard, universe) — universe < shards*per_shard forces overlap.
    for &(s, h, u) in &[(5usize, 200usize, 600usize), (10, 500, 2500)] {
        let shards = make_shards(s, h, u);
        let id = format!("s{s}_h{h}_u{u}");
        debug_assert_eq!(fuse_sip(&shards), fuse_ahash(&shards));
        g.bench_with_input(BenchmarkId::new("sip", &id), &shards, |b, sh| {
            b.iter(|| black_box(fuse_sip(black_box(sh))));
        });
        g.bench_with_input(BenchmarkId::new("ahash", &id), &shards, |b, sh| {
            b.iter(|| black_box(fuse_ahash(black_box(sh))));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_federated);
criterion_main!(benches);
