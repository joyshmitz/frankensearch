//! Ingest-path dual-hash A/B (bd-0j5e neighbor): the storage ingest pipeline needs BOTH the raw
//! `[u8; 32]` content hash (for `DocumentRecord` + dedup) and its lowercase-hex form (for the
//! `content_hashes` seen-count table). The former code computed them independently —
//! `ContentHasher::hash(text)` then `ContentHasher::hash_hex(text)` — running SHA-256 over the
//! canonical text TWICE. The lever computes the digest once and hex-encodes it
//! (`ContentHasher::to_hex(&digest)`), eliminating the redundant SHA-256 per document.
//!
//! Byte-identical: `to_hex(&hash(t)) == hash_hex(t)` (proven by `to_hex_matches_write_format`).
//! Gate on the MEDIAN of the paired candidate/ORIG ratio against the A/A null floor. Single-
//! threaded scalar SHA-256, so the paired alternating-round ratio is robust to fleet contention.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-storage --bench content_hash_dual
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_storage::ContentHasher;

/// Build a deterministic ASCII document of roughly `bytes` length (representative canonical text).
fn make_text(bytes: usize) -> String {
    let mut s = String::with_capacity(bytes + 16);
    let mut x = 0x2545_f491_4f6c_dd1d_u64;
    while s.len() < bytes {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        // Map to a printable ASCII word + space (keeps it text-like, all single-byte).
        let word = (x % 9973) as u32;
        let _ = std::fmt::Write::write_fmt(&mut s, format_args!("token{word} "));
    }
    s.truncate(bytes);
    s
}

fn bench(c: &mut Criterion) {
    // A few representative canonical-text sizes; SHA-256 cost scales with length, so the
    // redundant-hash saving grows with document size.
    let sizes = [256_usize, 2_048, 16_384];
    let inner: u32 = std::env::var("HASH_AB_INNER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let texts: Vec<String> = sizes.iter().map(|&s| make_text(s)).collect();

    // ── Paired MEDIAN gate (authoritative): per size, A/A null then ORIG-vs-candidate. ──
    for (text, &size) in texts.iter().zip(&sizes) {
        // Byte-identity assert before timing (cheap; makes the bench self-checking).
        assert_eq!(
            ContentHasher::to_hex(&ContentHasher::hash(text)),
            ContentHasher::hash_hex(text),
            "to_hex(hash) must equal hash_hex for size {size}"
        );

        // ORIG arm = the former pipeline shape: two independent SHA-256 passes.
        let run_orig = || {
            black_box(ContentHasher::hash(black_box(text)));
            black_box(ContentHasher::hash_hex(black_box(text)));
        };
        // CANDIDATE arm = hash once, hex-encode the digest.
        let run_cand = || {
            let d = ContentHasher::hash(black_box(text));
            black_box(d);
            black_box(ContentHasher::to_hex(&d));
        };

        let null = paired_median_ratio(41, inner, run_orig, run_orig);
        let lever = paired_median_ratio(41, inner, run_orig, run_cand);
        eprintln!(
            "[null]  dual_hash/{size}B: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
            null.median, null.p5, null.p95, null.rounds
        );
        eprintln!(
            "[lever] dual_hash/{size}B: cand/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
            lever.median,
            lever.p5,
            lever.p95,
            if lever.decidable_against(&null) {
                if lever.median < 1.0 {
                    "DECIDABLE WIN"
                } else {
                    "DECIDABLE REGRESSION"
                }
            } else {
                "INSIDE NULL FLOOR (not decidable)"
            }
        );
    }

    // ── Criterion central-estimate cross-check (one group, per-size functions). ──
    let mut g = c.benchmark_group("content_hash_dual");
    g.sample_size(20);
    for (text, &size) in texts.iter().zip(&sizes) {
        g.bench_function(format!("orig_{size}"), |b| {
            b.iter(|| {
                black_box(ContentHasher::hash(black_box(text)));
                black_box(ContentHasher::hash_hex(black_box(text)));
            });
        });
        g.bench_function(format!("cand_{size}"), |b| {
            b.iter(|| {
                let d = ContentHasher::hash(black_box(text));
                black_box(d);
                black_box(ContentHasher::to_hex(&d));
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
