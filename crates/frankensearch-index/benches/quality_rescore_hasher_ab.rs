//! Paired A/B for the per-hit quality-rescore doc_id lookup hasher.
//!
//! `InMemoryTwoTierIndex::quality_scores_for_hits` resolves every fast-tier hit's
//! `doc_id` to a quality-index position through `InMemoryVectorIndex::index_of_doc_id`,
//! whose lazily-built `doc_id_index` map ships as a std `HashMap<String, usize>`
//! (SipHash). The map is `OnceLock`-built once, but the *lookup* — one SipHash of
//! the hit's `doc_id` plus a probe — runs for **every** hit on **every** quality
//! query. This bench mirrors that inner loop exactly:
//!
//! ```ignore
//! map.get(hit.doc_id).map(|&idx| dot_product_f16_f32(vector_slice(idx), query))
//! ```
//!
//! over a realistic fixture (50k medium-length doc_ids, f16 dim-384 slab), comparing
//! the shipping SipHash `HashMap` against an `ahash::AHashMap`. The f16 dot work is
//! byte-identical in both arms (same positions, same vectors, same query); only the
//! hasher used for the per-hit lookup differs. Both arms assert a bit-identical
//! `Vec<Option<f32>>` before timing.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/<lane> \
//!   rch exec -- cargo bench -p frankensearch-index --bench quality_rescore_hasher_ab
//! ```

#![allow(clippy::doc_markdown)]

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::hint::black_box;

use ahash::AHashMap;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_index::dot_product_f16_f32;
use half::f16;

const DIM: usize = 384;
const N_DOCS: usize = 50_000;

fn next(state: &mut u64) -> u64 {
    // xorshift64* — deterministic, no external RNG.
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Map a raw draw to `[-1.0, 1.0)`. The top-24 bits fit f32's mantissa exactly.
fn unit_f32(bits: u64) -> f32 {
    ((bits >> 40) as f32) / 16_777_216.0 * 2.0 - 1.0
}

/// A realistic medium-length doc_id (~34 bytes): SipHash cost is per-byte, so a
/// path-shaped id (not a tiny `doc_0`) reflects the deployed key length.
fn make_doc_id(i: usize) -> String {
    format!("repository/src/module_{:04}/handler_{:08}.rs", i % 97, i)
}

fn build_fixture() -> (Vec<String>, Vec<f16>) {
    let mut state = 0x1234_5678_9abc_def0_u64;
    let doc_ids: Vec<String> = (0..N_DOCS).map(make_doc_id).collect();
    let mut vectors = Vec::with_capacity(N_DOCS * DIM);
    for _ in 0..N_DOCS * DIM {
        // f16 values in roughly [-1, 1).
        vectors.push(f16::from_f32(unit_f32(next(&mut state))));
    }
    (doc_ids, vectors)
}

fn make_query(state: &mut u64) -> Vec<f32> {
    (0..DIM).map(|_| unit_f32(next(state))).collect()
}

/// Pick `m` present hit doc_ids spread across the corpus (the fast tier's top-m
/// are effectively arbitrary corpus positions; all present in the quality map for
/// the common two-tier case where both tiers index the same corpus).
fn make_hits(doc_ids: &[String], m: usize) -> Vec<String> {
    let stride = (doc_ids.len() / m).max(1);
    (0..m)
        .map(|j| doc_ids[(j * stride) % doc_ids.len()].clone())
        .collect()
}

/// The exact per-hit lookup + dot loop, generic over the map's hasher.
#[inline]
fn rescore<S: BuildHasher>(
    map: &HashMap<String, usize, S>,
    hits: &[String],
    vectors: &[f16],
    query: &[f32],
) -> Vec<Option<f32>> {
    let mut scores = Vec::with_capacity(hits.len());
    for id in hits {
        let score = map
            .get(id.as_str())
            .map(|&idx| dot_product_f16_f32(&vectors[idx * DIM..idx * DIM + DIM], query).unwrap());
        scores.push(score);
    }
    scores
}

fn bench(c: &mut Criterion) {
    let (doc_ids, vectors) = build_fixture();

    // Arm 1: shipping std SipHash map. Arm 2: ahash map. Both first-insert-wins
    // (no duplicate ids here), keyed by the same owned Strings.
    let mut sip: HashMap<String, usize> = HashMap::with_capacity(doc_ids.len());
    for (i, id) in doc_ids.iter().enumerate() {
        sip.entry(id.clone()).or_insert(i);
    }
    let mut aha: AHashMap<String, usize> = AHashMap::with_capacity(doc_ids.len());
    for (i, id) in doc_ids.iter().enumerate() {
        aha.entry(id.clone()).or_insert(i);
    }
    let aha_std: HashMap<String, usize, ahash::RandomState> = aha.into_iter().collect();

    let mut qstate = 0x0f0f_0f0f_dead_beef_u64;
    let query = make_query(&mut qstate);

    let mut group = c.benchmark_group("quality_rescore_hasher");
    for &m in &[32usize, 128, 300] {
        let hits = make_hits(&doc_ids, m);

        // Bit-identical parity gate before timing.
        let base = rescore(&sip, &hits, &vectors, &query);
        let cand = rescore(&aha_std, &hits, &vectors, &query);
        assert_eq!(base.len(), cand.len(), "length parity (m={m})");
        for (a, b) in base.iter().zip(cand.iter()) {
            match (a, b) {
                (Some(x), Some(y)) => assert_eq!(x.to_bits(), y.to_bits(), "score parity (m={m})"),
                (None, None) => {}
                _ => panic!("presence parity (m={m})"),
            }
        }

        group.bench_with_input(BenchmarkId::new("siphash", m), &hits, |b, hits| {
            b.iter(|| black_box(rescore(black_box(&sip), black_box(hits), &vectors, &query)));
        });
        group.bench_with_input(BenchmarkId::new("ahash", m), &hits, |b, hits| {
            b.iter(|| {
                black_box(rescore(
                    black_box(&aha_std),
                    black_box(hits),
                    &vectors,
                    &query,
                ))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
