//! Stratified (Mondrian-conformal) certified `ef` selection for ANN.
//!
//! On a HETEROGENEOUS corpus (some clusters tight/easy, some diffuse/hard) a single
//! *global* certified `ef` must satisfy the HARDEST stratum, forcing a high `ef` on
//! every query. If queries can be routed to a difficulty stratum — here by
//! nearest-centroid (tight vs diffuse half) — each stratum gets its OWN conformally
//! certified `ef`, so easy queries run at a cheap `ef` while the recall guarantee
//! still holds *per stratum*. This bench measures global-`ef` latency vs
//! stratified-routed latency, and reports the per-stratum certified `ef`s + held-out
//! recall under both policies. It operationalises the recall certificate's group-
//! conditional (Mondrian) extension.
//!
//! Run: `rch exec -- cargo bench -p frankensearch-index --features ann --bench ann_stratified_ef`

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "ann")]
fn bench_ann_stratified_ef(c: &mut Criterion) {
    use std::hint::black_box;

    use frankensearch_index::{
        HnswConfig, HnswIndex, Quantization, VectorIndex, certified_min_ef,
    };

    const DIM: usize = 128;
    // Corpus size (env-overridable so the same bench can probe the steeper
    // ef->latency regime at larger N, where stratified routing should pay off).
    let n: usize = std::env::var("FS_STRAT_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40_000);
    const K: usize = 10;
    const CLUSTERS: usize = 128; // first half tight, second half diffuse
    const TIGHT_NOISE: f32 = 0.08;
    const DIFFUSE_NOISE: f32 = 0.35;
    const CAL_PER_STRATUM: usize = 400;
    const HOLDOUT: usize = 400;
    const TARGET: f64 = 0.90;
    const ALPHA: f64 = 0.1;
    const CANDIDATE_EFS: [usize; 6] = [10, 20, 40, 80, 160, 320];

    fn raw(seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..DIM)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }
    fn normalize(mut v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 1e-12 {
            for x in &mut v {
                *x /= n;
            }
        }
        v
    }
    let centroids: Vec<Vec<f32>> =
        (0..CLUSTERS).map(|cl| normalize(raw(0xC000_0000 + cl as u64))).collect();
    let make = |cluster: usize, seed: u64| -> Vec<f32> {
        let noise = if cluster < CLUSTERS / 2 { TIGHT_NOISE } else { DIFFUSE_NOISE };
        let r = raw(seed);
        normalize(centroids[cluster].iter().zip(&r).map(|(a, n)| a + noise * n).collect())
    };

    // Corpus + index + HNSW.
    let path = std::env::temp_dir().join(format!("fs_strat_{}.fsvi", std::process::id()));
    {
        let mut w = VectorIndex::create_with_revision(&path, "hash", "bench", DIM, Quantization::F32)
            .expect("create writer");
        for i in 0..n {
            let v = make(i % CLUSTERS, i as u64 + 1);
            w.write_record(&format!("doc-{i:06}"), &v).expect("write");
        }
        w.finish().expect("finish");
    }
    let index = VectorIndex::open(&path).expect("open");
    let hnsw = HnswIndex::build_from_vector_index(&index, HnswConfig { m: 32, ..HnswConfig::default() })
        .expect("build hnsw");

    let ann_ids = |q: &[f32], ef: usize| -> Vec<String> {
        hnsw.knn_search(q, K, ef)
            .expect("ann")
            .into_iter()
            .map(|h| h.doc_id.to_string())
            .collect()
    };
    let exact_ids = |q: &[f32]| -> Vec<String> {
        index
            .search_top_k(q, K, None)
            .expect("flat")
            .into_iter()
            .map(|h| h.doc_id.to_string())
            .collect()
    };
    let recall_of = |ann: &[String], exact: &[String]| -> f64 {
        if exact.is_empty() {
            return 1.0;
        }
        let hits = ann.iter().filter(|id| exact.contains(id)).count();
        hits as f64 / exact.len() as f64
    };

    // Per-ef recall sample for a query set (exact top-k computed ONCE per query).
    let samples = |queries: &[Vec<f32>]| -> Vec<(usize, Vec<f64>)> {
        let exact: Vec<Vec<String>> = queries.iter().map(|q| exact_ids(q)).collect();
        CANDIDATE_EFS
            .iter()
            .map(|&ef| {
                let recalls: Vec<f64> = queries
                    .iter()
                    .zip(&exact)
                    .map(|(q, ex)| recall_of(&ann_ids(q, ef), ex))
                    .collect();
                (ef, recalls)
            })
            .collect()
    };

    let tight_cal: Vec<Vec<f32>> =
        (0..CAL_PER_STRATUM).map(|i| make(i % (CLUSTERS / 2), 0x00A1_0000 + i as u64)).collect();
    let diffuse_cal: Vec<Vec<f32>> = (0..CAL_PER_STRATUM)
        .map(|i| make(CLUSTERS / 2 + i % (CLUSTERS / 2), 0x00B2_0000 + i as u64))
        .collect();
    let mut global_cal = tight_cal.clone();
    global_cal.extend(diffuse_cal.clone());

    let tight_ef = certified_min_ef(&samples(&tight_cal), TARGET, ALPHA).expect("tight").ef_search;
    let diffuse_ef =
        certified_min_ef(&samples(&diffuse_cal), TARGET, ALPHA).expect("diffuse").ef_search;
    // Population-conformal global: certifies a fresh MIXED query at target — but the
    // mixed tail is diluted by easy queries, so it does NOT certify the diffuse
    // stratum on its own.
    let global_pop_ef =
        certified_min_ef(&samples(&global_cal), TARGET, ALPHA).expect("global").ef_search;
    // Per-group global: the ef a NON-stratified system must run for EVERYONE to
    // certify EVERY stratum (the apples-to-apples baseline for the stratified router,
    // which also certifies every stratum).
    let pergroup_ef = tight_ef.max(diffuse_ef);

    // Nearest-centroid stratum router: tight iff nearest centroid is in the first half.
    let route_ef = |q: &[f32]| -> usize {
        let mut best_dot = f32::NEG_INFINITY;
        let mut best_ci = 0usize;
        for (ci, cen) in centroids.iter().enumerate() {
            let dot: f32 = cen.iter().zip(q).map(|(a, b)| a * b).sum();
            if dot > best_dot {
                best_dot = dot;
                best_ci = ci;
            }
        }
        if best_ci < CLUSTERS / 2 { tight_ef } else { diffuse_ef }
    };

    // Held-out mixed queries + their exact top-k (for honest recall under routing).
    let holdout: Vec<Vec<f32>> =
        (0..HOLDOUT).map(|i| make(i % CLUSTERS, 0x00D0_0000 + i as u64)).collect();
    let holdout_exact: Vec<Vec<String>> = holdout.iter().map(|q| exact_ids(q)).collect();
    let routed: Vec<usize> = holdout.iter().map(|q| route_ef(q)).collect();

    let denom = HOLDOUT as f64;
    let pop_recall: f64 = holdout
        .iter()
        .zip(&holdout_exact)
        .map(|(q, ex)| recall_of(&ann_ids(q, global_pop_ef), ex))
        .sum::<f64>()
        / denom;
    let pergroup_recall: f64 = holdout
        .iter()
        .zip(&holdout_exact)
        .map(|(q, ex)| recall_of(&ann_ids(q, pergroup_ef), ex))
        .sum::<f64>()
        / denom;
    let strat_recall: f64 = holdout
        .iter()
        .zip(&holdout_exact)
        .zip(&routed)
        .map(|((q, ex), &ef)| recall_of(&ann_ids(q, ef), ex))
        .sum::<f64>()
        / denom;

    println!(
        "STRATIFIED_RESULT target={TARGET} alpha={ALPHA} tight_ef={tight_ef} diffuse_ef={diffuse_ef} \
         global_pop_ef={global_pop_ef} pergroup_ef={pergroup_ef} \
         holdout_recall(pop)={pop_recall:.4} holdout_recall(pergroup)={pergroup_recall:.4} \
         holdout_recall(stratified)={strat_recall:.4} \
         (per-group guarantee: compare stratified_routed vs global_pergroup_ef{pergroup_ef})"
    );

    let mut g = c.benchmark_group("ann_stratified_ef");
    let mut qi = 0usize;
    g.bench_function(format!("global_population_ef{global_pop_ef}"), |b| {
        b.iter(|| {
            let i = qi % HOLDOUT;
            qi += 1;
            black_box(hnsw.knn_search(black_box(&holdout[i]), K, global_pop_ef).expect("ann"))
        });
    });
    let mut qp = 0usize;
    g.bench_function(format!("global_pergroup_ef{pergroup_ef}"), |b| {
        b.iter(|| {
            let i = qp % HOLDOUT;
            qp += 1;
            black_box(hnsw.knn_search(black_box(&holdout[i]), K, pergroup_ef).expect("ann"))
        });
    });
    let mut qj = 0usize;
    g.bench_function("stratified_routed", |b| {
        b.iter(|| {
            let i = qj % HOLDOUT;
            qj += 1;
            black_box(hnsw.knn_search(black_box(&holdout[i]), K, routed[i]).expect("ann"))
        });
    });
    g.finish();

    let _ = std::fs::remove_file(&path);
}

#[cfg(not(feature = "ann"))]
fn bench_ann_stratified_ef(_c: &mut Criterion) {
    // HNSW lives behind the `ann` feature; build with `--features ann` to run it.
}

criterion_group!(benches, bench_ann_stratified_ef);
criterion_main!(benches);
