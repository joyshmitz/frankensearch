//! Federated `appeared_in` representation A/B: `BTreeSet<String>` (current) vs a
//! shard-name-interned integer bitset (the untested route-next from the reverted
//! key-clone lever — NEGATIVE_EVIDENCE 2026-06-27, "attack the BTreeSet<String>
//! churn ... dedup shard names to a small interned id set").
//!
//! `accumulate_doc` runs once per (shard, hit) — thousands of times — and each
//! call did `appeared_in.insert(shard_name.to_owned())`: a `String` heap alloc
//! plus a `BTreeSet` node alloc + str-comparison tree descent. The bitset arm
//! interns each distinct shard name to its sorted rank (so ascending id order ==
//! lexicographic name order) and does `appeared_in |= 1 << id` — O(1), no alloc,
//! natural dedup. **Both arms materialize the same sorted `Vec<String>`
//! `appeared_in` at output** (faithful to `into_ranked_hits`, which the SipHash
//! bench omitted), so the measured delta is honest: the bitset relocates the
//! output `String` allocs but eliminates the per-accumulate BTree churn.
//!
//! Bit-identity (full output, not just len) is asserted before timing.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench federated_appeared_in
//! ```

use std::collections::BTreeSet;
use std::hint::black_box;

use ahash::{AHashMap, RandomState};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const K: f64 = 60.0;

#[inline]
fn rank_contribution(rank: usize) -> f32 {
    (1.0 / (K + rank as f64 + 1.0)) as f32
}

#[derive(Clone)]
struct Hit {
    doc_id: String,
    score: f32,
}

struct Shard {
    name: String,
    hits: Vec<Hit>,
}

/// One output row: (doc_id, fused_score, sorted appeared_in names). Used to
/// assert both arms produce byte-identical results.
type Row = (String, f32, Vec<String>);

// --- current: BTreeSet<String> appeared_in -------------------------------------

struct AggDocSet {
    template: Hit,
    primary_index: String,
    primary_rank: usize,
    primary_contribution: f32,
    fused_score: f32,
    appeared_in: BTreeSet<String>,
}

fn accumulate_set(
    docs: &mut AHashMap<String, AggDocSet, RandomState>,
    hit: &Hit,
    shard_name: &str,
    rank: usize,
    contribution: f32,
) {
    let entry = docs.entry(hit.doc_id.to_string()).or_insert_with(|| {
        let mut template = hit.clone();
        template.score = 0.0;
        AggDocSet {
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
    if contribution.total_cmp(&entry.primary_contribution) == std::cmp::Ordering::Greater {
        shard_name.clone_into(&mut entry.primary_index);
        entry.primary_rank = rank;
        entry.primary_contribution = contribution;
        entry.template = hit.clone();
    }
}

fn fuse_set(shards: &[Shard]) -> Vec<Row> {
    let mut docs: AHashMap<String, AggDocSet, RandomState> =
        AHashMap::with_hasher(RandomState::new());
    for shard in shards {
        for (rank, hit) in shard.hits.iter().enumerate() {
            accumulate_set(&mut docs, hit, &shard.name, rank, rank_contribution(rank));
        }
    }
    let mut out: Vec<Row> = docs
        .drain()
        .map(|(id, agg)| (id, agg.fused_score, agg.appeared_in.into_iter().collect()))
        .collect();
    out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

// --- new: interned shard-id bitset ---------------------------------------------

struct AggDocBits {
    template: Hit,
    primary_index: String,
    primary_rank: usize,
    primary_contribution: f32,
    fused_score: f32,
    appeared_in: u64,
}

fn accumulate_bits(
    docs: &mut AHashMap<String, AggDocBits, RandomState>,
    hit: &Hit,
    shard_name: &str,
    shard_id: u32,
    rank: usize,
    contribution: f32,
) {
    let entry = docs.entry(hit.doc_id.to_string()).or_insert_with(|| {
        let mut template = hit.clone();
        template.score = 0.0;
        AggDocBits {
            template,
            primary_index: shard_name.to_owned(),
            primary_rank: rank,
            primary_contribution: contribution,
            fused_score: 0.0,
            appeared_in: 0,
        }
    });
    entry.fused_score += contribution;
    entry.appeared_in |= 1u64 << shard_id;
    if contribution.total_cmp(&entry.primary_contribution) == std::cmp::Ordering::Greater {
        shard_name.clone_into(&mut entry.primary_index);
        entry.primary_rank = rank;
        entry.primary_contribution = contribution;
        entry.template = hit.clone();
    }
}

fn fuse_bits(shards: &[Shard]) -> Vec<Row> {
    // Intern distinct shard names to their sorted rank so ascending id order ==
    // lexicographic name order (⇒ output appeared_in is sorted, as BTreeSet gave).
    let mut names: Vec<&str> = shards.iter().map(|s| s.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert!(names.len() <= 64, "bitset arm caps at 64 distinct shards");
    let name_to_id: AHashMap<&str, u32> = names
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i as u32))
        .collect();

    let mut docs: AHashMap<String, AggDocBits, RandomState> =
        AHashMap::with_hasher(RandomState::new());
    for shard in shards {
        let shard_id = name_to_id[shard.name.as_str()];
        for (rank, hit) in shard.hits.iter().enumerate() {
            accumulate_bits(
                &mut docs,
                hit,
                &shard.name,
                shard_id,
                rank,
                rank_contribution(rank),
            );
        }
    }
    let mut out: Vec<Row> = docs
        .drain()
        .map(|(id, agg)| {
            let mut appeared: Vec<String> =
                Vec::with_capacity(agg.appeared_in.count_ones() as usize);
            let mut bits = agg.appeared_in;
            while bits != 0 {
                let id_bit = bits.trailing_zeros();
                appeared.push(names[id_bit as usize].to_owned());
                bits &= bits - 1;
            }
            (id, agg.fused_score, appeared)
        })
        .collect();
    out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

// --- new (robust, no shard cap): interned shard-id Vec<u32> --------------------

struct AggDocVec {
    template: Hit,
    primary_index: String,
    primary_rank: usize,
    primary_contribution: f32,
    fused_score: f32,
    appeared_in: Vec<u32>,
}

fn accumulate_vec(
    docs: &mut AHashMap<String, AggDocVec, RandomState>,
    hit: &Hit,
    shard_name: &str,
    shard_id: u32,
    rank: usize,
    contribution: f32,
) {
    let entry = docs.entry(hit.doc_id.to_string()).or_insert_with(|| {
        let mut template = hit.clone();
        template.score = 0.0;
        AggDocVec {
            template,
            primary_index: shard_name.to_owned(),
            primary_rank: rank,
            primary_contribution: contribution,
            fused_score: 0.0,
            appeared_in: Vec::new(),
        }
    });
    entry.fused_score += contribution;
    entry.appeared_in.push(shard_id);
    if contribution.total_cmp(&entry.primary_contribution) == std::cmp::Ordering::Greater {
        shard_name.clone_into(&mut entry.primary_index);
        entry.primary_rank = rank;
        entry.primary_contribution = contribution;
        entry.template = hit.clone();
    }
}

fn fuse_vec(shards: &[Shard]) -> Vec<Row> {
    let mut names: Vec<&str> = shards.iter().map(|s| s.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    let name_to_id: AHashMap<&str, u32> = names
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i as u32))
        .collect();

    let mut docs: AHashMap<String, AggDocVec, RandomState> =
        AHashMap::with_hasher(RandomState::new());
    for shard in shards {
        let shard_id = name_to_id[shard.name.as_str()];
        for (rank, hit) in shard.hits.iter().enumerate() {
            accumulate_vec(
                &mut docs,
                hit,
                &shard.name,
                shard_id,
                rank,
                rank_contribution(rank),
            );
        }
    }
    let mut out: Vec<Row> = docs
        .drain()
        .map(|(id, mut agg)| {
            agg.appeared_in.sort_unstable();
            agg.appeared_in.dedup();
            let appeared: Vec<String> = agg
                .appeared_in
                .iter()
                .map(|&sid| names[sid as usize].to_owned())
                .collect();
            (id, agg.fused_score, appeared)
        })
        .collect();
    out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

fn make_shards(shards: usize, per_shard: usize, universe: usize) -> Vec<Shard> {
    (0..shards)
        .map(|s| {
            let hits = (0..per_shard)
                .map(|i| {
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

fn bench_federated_appeared_in(c: &mut Criterion) {
    let mut g = c.benchmark_group("federated_appeared_in");
    for &(s, h, u) in &[(5usize, 200usize, 600usize), (10, 500, 2500)] {
        let shards = make_shards(s, h, u);
        let id = format!("s{s}_h{h}_u{u}");
        assert_eq!(
            fuse_set(&shards),
            fuse_bits(&shards),
            "bitset output differs ({id})"
        );
        assert_eq!(
            fuse_set(&shards),
            fuse_vec(&shards),
            "vec output differs ({id})"
        );
        g.bench_with_input(BenchmarkId::new("btreeset", &id), &shards, |b, sh| {
            b.iter(|| black_box(fuse_set(black_box(sh))));
        });
        g.bench_with_input(BenchmarkId::new("bitset", &id), &shards, |b, sh| {
            b.iter(|| black_box(fuse_bits(black_box(sh))));
        });
        g.bench_with_input(BenchmarkId::new("vec", &id), &shards, |b, sh| {
            b.iter(|| black_box(fuse_vec(black_box(sh))));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_federated_appeared_in);
criterion_main!(benches);
