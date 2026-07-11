//! Profile and paired gate for the buffered-record handoff into the FSVI writer.
//!
//! The fixture models one 20k-document, 384-dimension tier. Fixture construction
//! and per-arm source cloning stay outside the timer: the measured production work
//! begins with records already owned by `TwoTierIndexBuilder`.

use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use frankensearch_index::{VectorIndex, VectorIndexWriter};

const DOCUMENTS: usize = 20_000;
const DIMENSION: usize = 384;
const PROFILE_ROUNDS: usize = 21;
const PAIRED_ROUND_PAIRS: usize = 21;
const PARITY_QUERIES: usize = 8;

#[derive(Clone, Copy)]
struct Distribution {
    median_ms: f64,
    p5_ms: f64,
    p95_ms: f64,
}

#[derive(Clone, Copy)]
struct RatioDistribution {
    median: f64,
    p5: f64,
    p95: f64,
    round_pairs: usize,
}

impl RatioDistribution {
    fn null_contains_one(self) -> bool {
        self.p5 <= 1.0 && 1.0 <= self.p95
    }

    fn verdict_against(self, null: Self) -> &'static str {
        if !null.null_contains_one() {
            "BIASED_NULL_UNDECIDABLE"
        } else if self.median < null.p5 {
            "CANDIDATE_FASTER"
        } else if self.median > null.p95 {
            "CANDIDATE_SLOWER"
        } else {
            "INSIDE_NULL_FLOOR"
        }
    }
}

fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    let index = ((sorted.len() - 1) * pct + 50) / 100;
    sorted[index]
}

fn distribution(mut samples: Vec<Duration>) -> Distribution {
    samples.sort_unstable();
    Distribution {
        median_ms: percentile(&samples, 50).as_secs_f64() * 1_000.0,
        p5_ms: percentile(&samples, 5).as_secs_f64() * 1_000.0,
        p95_ms: percentile(&samples, 95).as_secs_f64() * 1_000.0,
    }
}

fn ratio_distribution(mut samples: Vec<f64>) -> RatioDistribution {
    samples.sort_unstable_by(f64::total_cmp);
    let index = |pct: usize| ((samples.len() - 1) * pct + 50) / 100;
    RatioDistribution {
        median: samples[index(50)],
        p5: samples[index(5)],
        p95: samples[index(95)],
        round_pairs: samples.len(),
    }
}

fn fixture() -> Vec<(String, Vec<f32>)> {
    (0..DOCUMENTS)
        .map(|document| {
            let vector = (0..DIMENSION)
                .map(|axis| {
                    let value = document.wrapping_mul(131).wrapping_add(axis * 17) % 2_003;
                    let value = u16::try_from(value).expect("fixture value fits u16");
                    (f32::from(value) - 1_001.0) / 1_001.0
                })
                .collect();
            (format!("document-{document:08}"), vector)
        })
        .collect()
}

fn writer_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "fsvi_builder_record_transfer_{}_{}.fsvi",
        std::process::id(),
        label
    ))
}

fn create_writer(label: &str) -> VectorIndexWriter {
    VectorIndex::create(&writer_path(label), "profile-384", DIMENSION)
        .expect("create in-memory writer state")
}

fn time_borrowed_ingest(source: &[(String, Vec<f32>)], label: &str) -> Duration {
    let records = source.to_vec();
    let mut writer = create_writer(label);
    let started = Instant::now();
    for (doc_id, embedding) in &records {
        writer
            .write_record(doc_id, embedding)
            .expect("buffer valid record");
    }
    let elapsed = started.elapsed();
    black_box(&writer);
    drop(writer);
    drop(records);
    elapsed
}

fn time_owned_ingest(source: &[(String, Vec<f32>)], label: &str) -> Duration {
    let records = source.to_vec();
    let mut writer = create_writer(label);
    let started = Instant::now();
    for (doc_id, embedding) in records {
        writer
            .write_record_owned_for_benchmark(doc_id, embedding)
            .expect("buffer valid owned record");
    }
    let elapsed = started.elapsed();
    black_box(&writer);
    drop(writer);
    elapsed
}

fn time_borrowed_finish(source: &[(String, Vec<f32>)], label: &str) -> Duration {
    let records = source.to_vec();
    let mut writer = create_writer(label);
    let started = Instant::now();
    for (doc_id, embedding) in &records {
        writer
            .write_record(doc_id, embedding)
            .expect("buffer valid record");
    }
    writer.finish().expect("finish borrowed writer");
    let elapsed = started.elapsed();
    drop(records);
    elapsed
}

fn time_owned_finish(source: &[(String, Vec<f32>)], label: &str) -> Duration {
    let records = source.to_vec();
    let mut writer = create_writer(label);
    let started = Instant::now();
    for (doc_id, embedding) in records {
        writer
            .write_record_owned_for_benchmark(doc_id, embedding)
            .expect("buffer valid owned record");
    }
    writer.finish().expect("finish owned writer");
    started.elapsed()
}

fn timed_pair<A, B>(
    original: &mut A,
    candidate: &mut B,
    original_first: bool,
) -> (Duration, Duration)
where
    A: FnMut() -> Duration,
    B: FnMut() -> Duration,
{
    if original_first {
        let original_elapsed = original();
        let candidate_elapsed = candidate();
        (original_elapsed, candidate_elapsed)
    } else {
        let candidate_elapsed = candidate();
        let original_elapsed = original();
        (original_elapsed, candidate_elapsed)
    }
}

fn paired_ratio<A, B>(mut original: A, mut candidate: B) -> RatioDistribution
where
    A: FnMut() -> Duration,
    B: FnMut() -> Duration,
{
    for _ in 0..2 {
        black_box(timed_pair(&mut original, &mut candidate, true));
        black_box(timed_pair(&mut original, &mut candidate, false));
    }

    let mut ratios = Vec::with_capacity(PAIRED_ROUND_PAIRS);
    for pair in 0..PAIRED_ROUND_PAIRS {
        let (ab, ba) = if pair.is_multiple_of(2) {
            (
                timed_pair(&mut original, &mut candidate, true),
                timed_pair(&mut original, &mut candidate, false),
            )
        } else {
            (
                timed_pair(&mut original, &mut candidate, false),
                timed_pair(&mut original, &mut candidate, true),
            )
        };
        let ab_ratio = ab.1.as_secs_f64() / ab.0.as_secs_f64();
        let ba_ratio = ba.1.as_secs_f64() / ba.0.as_secs_f64();
        ratios.push((ab_ratio * ba_ratio).sqrt());
    }
    ratio_distribution(ratios)
}

fn profile(source: &[(String, Vec<f32>)]) {
    let mut clone_samples = Vec::with_capacity(PROFILE_ROUNDS);
    let mut writer_samples = Vec::with_capacity(PROFILE_ROUNDS);

    for round in 0..PROFILE_ROUNDS {
        let started = Instant::now();
        let cloned = source.to_vec();
        let clone_elapsed = started.elapsed();
        black_box(&cloned);
        clone_samples.push(clone_elapsed);
        drop(cloned);

        // Model the builder's already-owned source records outside the timed handoff.
        writer_samples.push(time_borrowed_ingest(source, &format!("baseline-{round}")));
    }

    let clone_profile = distribution(clone_samples);
    let writer_profile = distribution(writer_samples);
    eprintln!(
        "[profile] record_deep_clone median_ms={:.3} p5_ms={:.3} p95_ms={:.3}",
        clone_profile.median_ms, clone_profile.p5_ms, clone_profile.p95_ms
    );
    eprintln!(
        "[profile] borrowed_writer_ingest median_ms={:.3} p5_ms={:.3} p95_ms={:.3}",
        writer_profile.median_ms, writer_profile.p5_ms, writer_profile.p95_ms
    );
    eprintln!(
        "[profile] clone_to_ingest_median_fraction={:.4}",
        clone_profile.median_ms / writer_profile.median_ms
    );
}

fn prove_parity(source: &[(String, Vec<f32>)]) {
    let borrowed_path = writer_path("parity-borrowed");
    let owned_path = writer_path("parity-owned");

    let mut borrowed = VectorIndex::create(&borrowed_path, "profile-384", DIMENSION)
        .expect("create borrowed parity writer");
    for (doc_id, embedding) in source {
        borrowed
            .write_record(doc_id, embedding)
            .expect("write borrowed parity record");
    }
    borrowed.finish().expect("finish borrowed parity writer");

    let mut owned = VectorIndex::create(&owned_path, "profile-384", DIMENSION)
        .expect("create owned parity writer");
    for (doc_id, embedding) in source.to_vec() {
        owned
            .write_record_owned_for_benchmark(doc_id, embedding)
            .expect("write owned parity record");
    }
    owned.finish().expect("finish owned parity writer");

    let borrowed_bytes = fs::read(&borrowed_path).expect("read borrowed parity file");
    let owned_bytes = fs::read(&owned_path).expect("read owned parity file");
    assert_eq!(borrowed_bytes, owned_bytes, "FSVI bytes must be identical");

    let borrowed = VectorIndex::open(&borrowed_path).expect("open borrowed parity index");
    let owned = VectorIndex::open(&owned_path).expect("open owned parity index");
    assert_eq!(borrowed.record_count(), owned.record_count());
    for index in (0..DOCUMENTS).step_by(DOCUMENTS / 32) {
        assert_eq!(
            borrowed.doc_id_at(index).expect("borrowed doc id"),
            owned.doc_id_at(index).expect("owned doc id")
        );
        let borrowed_bits: Vec<u32> = borrowed
            .vector_at_f32(index)
            .expect("borrowed vector")
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let owned_bits: Vec<u32> = owned
            .vector_at_f32(index)
            .expect("owned vector")
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(borrowed_bits, owned_bits);
    }

    for query in source.iter().step_by(DOCUMENTS / PARITY_QUERIES) {
        let borrowed_hits = borrowed
            .search_top_k(&query.1, 10, None)
            .expect("search borrowed index");
        let owned_hits = owned
            .search_top_k(&query.1, 10, None)
            .expect("search owned index");
        assert_eq!(borrowed_hits.len(), owned_hits.len());
        for (borrowed_hit, owned_hit) in borrowed_hits.iter().zip(&owned_hits) {
            assert_eq!(borrowed_hit.index, owned_hit.index);
            assert_eq!(borrowed_hit.doc_id, owned_hit.doc_id);
            assert_eq!(borrowed_hit.score.to_bits(), owned_hit.score.to_bits());
        }
    }

    eprintln!(
        "[parity] fsvi_bytes_identical=true bytes={} sampled_records=32 queries={PARITY_QUERIES} top10_identical=true recall_preserved=true ndcg_preserved=true score_bits_identical=true",
        borrowed_bytes.len()
    );
}

fn print_ratio(label: &str, result: RatioDistribution, null: Option<RatioDistribution>) {
    if let Some(null) = null {
        eprintln!(
            "[paired-lever] {label} candidate/original median={:.6} p5={:.6} p95={:.6} calibrated_ratio={:.6} rounds={} verdict={}",
            result.median,
            result.p5,
            result.p95,
            result.median / null.median,
            result.round_pairs,
            result.verdict_against(null)
        );
    } else {
        eprintln!(
            "[paired-null] {label} median={:.6} p5={:.6} p95={:.6} contains_one={} rounds={}",
            result.median,
            result.p5,
            result.p95,
            result.null_contains_one(),
            result.round_pairs
        );
    }
}

fn main() {
    let payload_elements = u32::try_from(DOCUMENTS * DIMENSION).expect("payload fits u32");
    eprintln!(
        "[profile-config] documents={DOCUMENTS} dimension={DIMENSION} profile_rounds={PROFILE_ROUNDS} paired_round_pairs={PAIRED_ROUND_PAIRS} payload_mib={:.3}",
        f64::from(payload_elements) * 4.0 / (1024.0 * 1024.0)
    );

    let source = fixture();
    profile(&source);
    prove_parity(&source);

    let ingest_null = paired_ratio(
        || time_borrowed_ingest(&source, "ingest-null-a"),
        || time_borrowed_ingest(&source, "ingest-null-b"),
    );
    let ingest_lever = paired_ratio(
        || time_borrowed_ingest(&source, "ingest-original"),
        || time_owned_ingest(&source, "ingest-candidate"),
    );
    print_ratio("writer_ingest", ingest_null, None);
    print_ratio("writer_ingest", ingest_lever, Some(ingest_null));

    let finish_null = paired_ratio(
        || time_borrowed_finish(&source, "finish-null-a"),
        || time_borrowed_finish(&source, "finish-null-b"),
    );
    let finish_lever = paired_ratio(
        || time_borrowed_finish(&source, "finish-original"),
        || time_owned_finish(&source, "finish-candidate"),
    );
    print_ratio("writer_transfer_plus_finish", finish_null, None);
    print_ratio(
        "writer_transfer_plus_finish",
        finish_lever,
        Some(finish_null),
    );

    let ingest_verdict = ingest_lever.verdict_against(ingest_null);
    let finish_verdict = finish_lever.verdict_against(finish_null);
    eprintln!(
        "[paired-gate] parity=true ingest_verdict={ingest_verdict} finish_verdict={finish_verdict} gate={}",
        if finish_verdict == "CANDIDATE_FASTER" {
            "KEEP"
        } else {
            "HOLD"
        }
    );
}
