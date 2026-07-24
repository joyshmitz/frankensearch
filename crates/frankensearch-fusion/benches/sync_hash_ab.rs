//! Sync-searcher per-query map/set hasher A/B: std SipHash vs `ahash`.
//!
//! `vector_hits_to_scored_results` (sync_searcher.rs) builds, per query, two
//! `HashMap<&str, f32>` score maps (fast + quality, one entry per candidate) and
//! a `seen` dedup `HashSet<&str>`, then probes every candidate's doc_id in both
//! maps. Today those are std `HashMap`/`HashSet` (SipHash), while the sibling
//! fusion paths (`rrf.rs`, `blend.rs`) and `search.rs` already use `ahash`.
//! SipHash is a cryptographic hash; for short non-adversarial `&str` doc_ids the
//! non-crypto `ahash` is materially faster on both insert and lookup. This bench
//! measures the *full* per-query shape (build both maps + seen set, then one
//! fast + one quality `.get()` per candidate) so the ratio reflects the real
//! path, not just the build. Bit-identical: same keys, same values, same order.
use std::collections::{HashMap, HashSet};
use std::hint::black_box;

use ahash::{AHashMap, AHashSet};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_fusion::bench_support::paired_median_ratio;

struct VHit {
    score: f32,
    doc_id: String,
}

fn make(n: usize, off: usize) -> Vec<VHit> {
    (0..n)
        .map(|i| VHit {
            score: 1.0 / (i as f32 + 1.0),
            doc_id: format!("doc-{:06}", off + i),
        })
        .collect()
}

/// std SipHash arm — mirrors the current sync_searcher.rs code exactly.
fn run_sip(fast: &[VHit], quality: &[VHit]) -> f32 {
    let fast_scores: HashMap<&str, f32> =
        fast.iter().map(|h| (h.doc_id.as_str(), h.score)).collect();
    let quality_scores: HashMap<&str, f32> = quality
        .iter()
        .map(|h| (h.doc_id.as_str(), h.score))
        .collect();
    let mut seen: HashSet<&str> = HashSet::with_capacity(fast.len());
    let mut acc = 0.0_f32;
    for hit in fast {
        if seen.insert(hit.doc_id.as_str()) {
            let f = fast_scores
                .get(hit.doc_id.as_str())
                .copied()
                .unwrap_or(hit.score);
            let q = quality_scores
                .get(hit.doc_id.as_str())
                .copied()
                .unwrap_or(0.0);
            acc += f + q;
        }
    }
    acc
}

/// `ahash` arm — identical logic, non-crypto hasher for the short `&str` keys.
fn run_ahash(fast: &[VHit], quality: &[VHit]) -> f32 {
    let fast_scores: AHashMap<&str, f32> =
        fast.iter().map(|h| (h.doc_id.as_str(), h.score)).collect();
    let quality_scores: AHashMap<&str, f32> = quality
        .iter()
        .map(|h| (h.doc_id.as_str(), h.score))
        .collect();
    let mut seen: AHashSet<&str> = AHashSet::with_capacity(fast.len());
    let mut acc = 0.0_f32;
    for hit in fast {
        if seen.insert(hit.doc_id.as_str()) {
            let f = fast_scores
                .get(hit.doc_id.as_str())
                .copied()
                .unwrap_or(hit.score);
            let q = quality_scores
                .get(hit.doc_id.as_str())
                .copied()
                .unwrap_or(0.0);
            acc += f + q;
        }
    }
    acc
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("sync_hash");
    for n in [30usize, 100, 300] {
        let fast = make(n, 0);
        let quality = make(n, n / 3); // quality overlaps a subset of fast doc_ids
        // Same accumulated value from both arms (bit-identical map contents).
        debug_assert_eq!(
            run_sip(&fast, &quality).to_bits(),
            run_ahash(&fast, &quality).to_bits()
        );
        let id = format!("n{n}");
        g.bench_with_input(BenchmarkId::new("sip", &id), &(), |b, ()| {
            b.iter(|| black_box(run_sip(black_box(&fast), black_box(&quality))));
        });
        g.bench_with_input(BenchmarkId::new("ahash", &id), &(), |b, ()| {
            b.iter(|| black_box(run_ahash(black_box(&fast), black_box(&quality))));
        });

        // ── DECIDABILITY: alternating-round paired sampler + A/A null control ──
        //
        // The criterion arms above run as separate benchmarks minutes apart, so worker
        // drift between them is not cancelled. The paired sampler runs both arms in ONE
        // routine in alternating rounds; gate on the median against the A/A null's
        // observed spread.
        let sip = || {
            black_box(run_sip(black_box(&fast), black_box(&quality)));
        };
        let ahash = || {
            black_box(run_ahash(black_box(&fast), black_box(&quality)));
        };
        let null = paired_median_ratio(41, 8, sip, sip);
        let lever = paired_median_ratio(41, 8, sip, ahash);
        eprintln!(
            "[null]  sync_hash n{n}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] sync_hash n{n}: ahash/sip median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                "DECIDABLE"
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
