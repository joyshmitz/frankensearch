//! `merge_with_lexical_tail` dedup A/B: the default fsfs result-assembly merge
//! (runtime.rs) builds a `HashSet<&str>` of the fused head's doc_ids, then probes
//! it once per lexical-tail candidate (O(tail), which is the FULL lexical result
//! set — large on big corpora) to skip duplicates, cloning each kept candidate
//! into the merged output. The set is std SipHash; siblings use `ahash`. Keys are
//! already borrowed (`&str`), so the only lever here is the hasher — but the
//! per-candidate `FusedCandidate` clone (a `String` doc_id alloc) runs in BOTH
//! arms and may dominate, so this bench measures the REAL merge shape to decide
//! honestly whether the hasher swap survives end-to-end (not just the isolated
//! set op). Identical merged output asserted across arms.
use std::collections::HashSet;
use std::hint::black_box;

use ahash::AHashSet;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_fusion::bench_support::paired_median_ratio;

/// Faithful stand-in for `FusedCandidate`: only `doc_id` allocates on clone;
/// every other field is `Copy` (matches the real struct's clone cost).
#[derive(Clone)]
struct Cand {
    doc_id: String,
    fused_score: f64,
    prior_boost: f64,
    lexical_rank: Option<usize>,
    semantic_rank: Option<usize>,
    lexical_score: Option<f32>,
    semantic_score: Option<f32>,
    in_both_sources: bool,
}

fn cand(id: usize, score: f64) -> Cand {
    Cand {
        doc_id: format!("docs/section-{:02}/file-{:05}.md", id % 32, id),
        fused_score: score,
        prior_boost: 0.0,
        lexical_rank: None,
        semantic_rank: Some(id % 7),
        lexical_score: Some(score as f32),
        semantic_score: Some(score as f32),
        in_both_sources: false,
    }
}

/// head: the fused semantic head (≈limit). tail: the full lexical result set;
/// the first `head_len` ids overlap the head (dedup hits), the rest are new.
fn make(head_len: usize, tail_len: usize) -> (Vec<Cand>, Vec<Cand>) {
    let head: Vec<Cand> = (0..head_len)
        .map(|i| cand(i, 1.0 / (i as f64 + 1.0)))
        .collect();
    // tail overlaps head on ids [0, head_len) then continues into fresh ids.
    let tail: Vec<Cand> = (0..tail_len)
        .map(|i| cand(i, 1.0 / (i as f64 + 2.0)))
        .collect();
    (head, tail)
}

fn merge_sip(head: &[Cand], tail: &[Cand]) -> Vec<Cand> {
    let mut merged = Vec::with_capacity(head.len() + tail.len());
    let mut seen: HashSet<&str> = HashSet::with_capacity(head.len());
    for c in head {
        seen.insert(c.doc_id.as_str());
        merged.push(c.clone());
    }
    for (rank, c) in tail.iter().enumerate() {
        if seen.contains(c.doc_id.as_str()) {
            continue;
        }
        let mut nc = c.clone();
        nc.lexical_rank = Some(rank);
        merged.push(nc);
    }
    merged
}

fn merge_ahash(head: &[Cand], tail: &[Cand]) -> Vec<Cand> {
    let mut merged = Vec::with_capacity(head.len() + tail.len());
    let mut seen: AHashSet<&str> = AHashSet::with_capacity(head.len());
    for c in head {
        seen.insert(c.doc_id.as_str());
        merged.push(c.clone());
    }
    for (rank, c) in tail.iter().enumerate() {
        if seen.contains(c.doc_id.as_str()) {
            continue;
        }
        let mut nc = c.clone();
        nc.lexical_rank = Some(rank);
        merged.push(nc);
    }
    merged
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("merge_dedup");
    for &(h, t) in &[(50usize, 200usize), (50, 1000), (100, 2000)] {
        let (head, tail) = make(h, t);
        let a = merge_sip(&head, &tail);
        let b = merge_ahash(&head, &tail);
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a[i].doc_id, b[i].doc_id);
        }
        let id = format!("h{h}_t{t}");
        g.bench_with_input(BenchmarkId::new("sip", &id), &(), |bch, ()| {
            bch.iter(|| black_box(merge_sip(black_box(&head), black_box(&tail))));
        });
        g.bench_with_input(BenchmarkId::new("ahash", &id), &(), |bch, ()| {
            bch.iter(|| black_box(merge_ahash(black_box(&head), black_box(&tail))));
        });

        // ── DECIDABILITY: alternating-round paired sampler + A/A null control ──
        //
        // The criterion arms above run as separate benchmarks minutes apart, so worker
        // drift between them is not cancelled. The paired sampler runs both arms in ONE
        // routine in alternating rounds; gate on the median against the A/A null's
        // observed spread.
        let mut sip = || {
            black_box(merge_sip(black_box(&head), black_box(&tail)));
        };
        let mut ahash = || {
            black_box(merge_ahash(black_box(&head), black_box(&tail)));
        };
        let null = paired_median_ratio(41, 8, sip, sip);
        let lever = paired_median_ratio(41, 8, sip, ahash);
        eprintln!(
            "[null]  merge_dedup h{h}_t{t}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] merge_dedup h{h}_t{t}: ahash/sip median {:.4} p5 {:.4} p95 {:.4} -> {}",
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
