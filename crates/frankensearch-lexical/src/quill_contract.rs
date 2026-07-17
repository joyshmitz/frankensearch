//! Quill scoring-contract constants vendored from the tantivy oracle.
//!
//! Quill (the ground-up lexical engine that replaces tantivy) must reproduce
//! tantivy's BM25 rank order exactly for equal-stats configurations (the
//! `RankExact` conformance class). BM25's `|d|` term comes from tantivy's
//! 1-byte log-scale fieldnorm (document length) quantization, so rank parity
//! requires reproducing that table and its encode/decode functions bit-for-bit.
//!
//! Provenance — vendored from tantivy 0.26.1 (pinned oracle, quill-e0.7):
//! - [`FIELD_NORMS_TABLE`], [`id_to_fieldnorm`], [`fieldnorm_to_id`]:
//!   `src/fieldnorm/code.rs` (table at line 13).
//! - [`BM25_K1`], [`BM25_B`], [`idf`], [`compute_tf_cache`]:
//!   `src/query/bm25.rs` (constants at lines 8-9, `idf` at line 52,
//!   tf cache at lines 58-66).
//!
//! Contract notes:
//! - `avgdl` and `|d|` in BM25 are computed from **decoded** table values
//!   ([`id_to_fieldnorm`]), never from raw token counts. The Language Contract
//!   (quill-e0.1) cross-references this module for the scoring section.
//! - The per-snapshot tf cache ([`compute_tf_cache`]) is tantivy's O(1)
//!   per-doc `tf_part` trick: one 256-entry table per (field, snapshot).
//!   Quill's evaluator (quill-e4.3) mirrors it for both parity and cost.
//! - Staging home: this module lives in `frankensearch-lexical` (the
//!   tantivy-facing crate, so pin tests can diff against the real oracle).
//!   Moving it to `frankensearch_quill::contract` is part of quill-e1.0's
//!   acceptance criteria.

/// BM25 `k1` parameter, pinned to tantivy 0.26.1 `src/query/bm25.rs:8`.
pub const BM25_K1: f32 = 1.2;

/// BM25 `b` parameter, pinned to tantivy 0.26.1 `src/query/bm25.rs:9`.
pub const BM25_B: f32 = 0.75;

/// Fieldnorm id → decoded fieldnorm (document length) table.
///
/// Vendored verbatim from tantivy 0.26.1 `src/fieldnorm/code.rs:13`
/// (`FIELD_NORMS_TABLE`), which uses the exact same log-scale scheme as
/// Lucene: exact lengths 0..=39, then geometrically widening buckets up to
/// `2_013_265_944`. Strictly increasing, so [`fieldnorm_to_id`] can bucket
/// via binary search.
pub const FIELD_NORMS_TABLE: [u32; 256] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    42,
    44,
    46,
    48,
    50,
    52,
    54,
    56,
    60,
    64,
    68,
    72,
    76,
    80,
    84,
    88,
    96,
    104,
    112,
    120,
    128,
    136,
    144,
    152,
    168,
    184,
    200,
    216,
    232,
    248,
    264,
    280,
    312,
    344,
    376,
    408,
    440,
    472,
    504,
    536,
    600,
    664,
    728,
    792,
    856,
    920,
    984,
    1_048,
    1_176,
    1_304,
    1_432,
    1_560,
    1_688,
    1_816,
    1_944,
    2_072,
    2_328,
    2_584,
    2_840,
    3_096,
    3_352,
    3_608,
    3_864,
    4_120,
    4_632,
    5_144,
    5_656,
    6_168,
    6_680,
    7_192,
    7_704,
    8_216,
    9_240,
    10_264,
    11_288,
    12_312,
    13_336,
    14_360,
    15_384,
    16_408,
    18_456,
    20_504,
    22_552,
    24_600,
    26_648,
    28_696,
    30_744,
    32_792,
    36_888,
    40_984,
    45_080,
    49_176,
    53_272,
    57_368,
    61_464,
    65_560,
    73_752,
    81_944,
    90_136,
    98_328,
    106_520,
    114_712,
    122_904,
    131_096,
    147_480,
    163_864,
    180_248,
    196_632,
    213_016,
    229_400,
    245_784,
    262_168,
    294_936,
    327_704,
    360_472,
    393_240,
    426_008,
    458_776,
    491_544,
    524_312,
    589_848,
    655_384,
    720_920,
    786_456,
    851_992,
    917_528,
    983_064,
    1_048_600,
    1_179_672,
    1_310_744,
    1_441_816,
    1_572_888,
    1_703_960,
    1_835_032,
    1_966_104,
    2_097_176,
    2_359_320,
    2_621_464,
    2_883_608,
    3_145_752,
    3_407_896,
    3_670_040,
    3_932_184,
    4_194_328,
    4_718_616,
    5_242_904,
    5_767_192,
    6_291_480,
    6_815_768,
    7_340_056,
    7_864_344,
    8_388_632,
    9_437_208,
    10_485_784,
    11_534_360,
    12_582_936,
    13_631_512,
    14_680_088,
    15_728_664,
    16_777_240,
    18_874_392,
    20_971_544,
    23_068_696,
    25_165_848,
    27_263_000,
    29_360_152,
    31_457_304,
    33_554_456,
    37_748_760,
    41_943_064,
    46_137_368,
    50_331_672,
    54_525_976,
    58_720_280,
    62_914_584,
    67_108_888,
    75_497_496,
    83_886_104,
    92_274_712,
    100_663_320,
    109_051_928,
    117_440_536,
    125_829_144,
    134_217_752,
    150_994_968,
    167_772_184,
    184_549_400,
    201_326_616,
    218_103_832,
    234_881_048,
    251_658_264,
    268_435_480,
    301_989_912,
    335_544_344,
    369_098_776,
    402_653_208,
    436_207_640,
    469_762_072,
    503_316_504,
    536_870_936,
    603_979_800,
    671_088_664,
    738_197_528,
    805_306_392,
    872_415_256,
    939_524_120,
    1_006_632_984,
    1_073_741_848,
    1_207_959_576,
    1_342_177_304,
    1_476_395_032,
    1_610_612_760,
    1_744_830_488,
    1_879_048_216,
    2_013_265_944,
];

/// Decodes a 1-byte fieldnorm id to its fieldnorm (document length) value.
///
/// Mirrors tantivy 0.26.1 `src/fieldnorm/code.rs::id_to_fieldnorm`.
#[inline]
#[must_use]
pub fn id_to_fieldnorm(id: u8) -> u32 {
    FIELD_NORMS_TABLE[usize::from(id)]
}

/// Encodes a fieldnorm (document length) to its 1-byte id, rounding down to
/// the containing bucket.
///
/// Mirrors tantivy 0.26.1 `src/fieldnorm/code.rs::fieldnorm_to_id`:
/// exact matches return their index; misses land in the preceding bucket.
// The binary-search index over a 256-entry table always fits in u8: an exact
// hit is <= 255, and a miss's insertion point is >= 1 (table starts at 0), so
// `idx - 1` is <= 255 and never underflows.
#[allow(clippy::cast_possible_truncation)]
#[inline]
#[must_use]
pub fn fieldnorm_to_id(fieldnorm: u32) -> u8 {
    FIELD_NORMS_TABLE
        .binary_search(&fieldnorm)
        .unwrap_or_else(|idx| idx - 1) as u8
}

/// BM25 inverse document frequency.
///
/// Mirrors tantivy 0.26.1 `src/query/bm25.rs::idf` including the f32
/// (`tantivy::Score`) intermediate arithmetic and the `doc_count >= doc_freq`
/// assertion.
///
/// # Panics
///
/// Panics if `doc_freq > doc_count`, matching the oracle.
// `x.ln_1p()` would be more accurate (and is what clippy suggests), but the
// oracle computes `(1.0 + x).ln()` in f32; matching its exact op sequence —
// including the small-x cancellation — is the contract.
#[allow(clippy::imprecise_flops)]
#[must_use]
pub fn idf(doc_freq: u64, doc_count: u64) -> f32 {
    assert!(doc_count >= doc_freq, "{doc_count} >= {doc_freq}");
    let x = ((doc_count - doc_freq) as f32 + 0.5) / (doc_freq as f32 + 0.5);
    (1.0 + x).ln()
}

/// One entry of the BM25 tf cache: `k1 * (1 - b + b * |d| / avgdl)`.
///
/// Mirrors tantivy 0.26.1 `src/query/bm25.rs::cached_tf_component`. The
/// operation order is part of the contract — reassociating or fusing
/// (`mul_add`) changes f32 results and breaks bit parity with the oracle.
#[inline]
#[must_use]
pub fn cached_tf_component(fieldnorm: u32, average_fieldnorm: f32) -> f32 {
    BM25_K1 * (1.0 - BM25_B + BM25_B * fieldnorm as f32 / average_fieldnorm)
}

/// Precomputes the 256-entry BM25 tf cache for one (field, snapshot) pair.
///
/// Mirrors tantivy 0.26.1 `src/query/bm25.rs::compute_tf_cache`: entry `i` is
/// [`cached_tf_component`] of the *decoded* fieldnorm for id `i`. Scoring a
/// doc then costs one table lookup per (term, doc) instead of a division.
#[must_use]
pub fn compute_tf_cache(average_fieldnorm: f32) -> [f32; 256] {
    let mut cache = [0.0_f32; 256];
    for (fieldnorm_id, slot) in cache.iter_mut().enumerate() {
        // Enumerating a [_; 256] array: index is always <= 255.
        #[allow(clippy::cast_possible_truncation)]
        let fieldnorm = id_to_fieldnorm(fieldnorm_id as u8);
        *slot = cached_tf_component(fieldnorm, average_fieldnorm);
    }
    cache
}

#[cfg(test)]
mod tests {
    use tantivy::fieldnorm::FieldNormReader;

    use super::*;

    #[test]
    fn decode_matches_tantivy_for_all_256_ids() {
        for id in 0..=u8::MAX {
            assert_eq!(
                id_to_fieldnorm(id),
                FieldNormReader::id_to_fieldnorm(id),
                "decode diverges from oracle at id {id}"
            );
        }
    }

    /// Landmarks + checksum keep the pin meaningful after the tantivy
    /// dev-dependency is eventually dropped (a botched re-vendor cannot
    /// satisfy both the oracle diff and these constants).
    #[test]
    fn table_landmarks_pinned_independent_of_oracle() {
        assert_eq!(FIELD_NORMS_TABLE[0], 0);
        assert_eq!(FIELD_NORMS_TABLE[39], 39);
        assert_eq!(FIELD_NORMS_TABLE[40], 40);
        assert_eq!(FIELD_NORMS_TABLE[41], 42);
        assert_eq!(FIELD_NORMS_TABLE[64], 152);
        assert_eq!(FIELD_NORMS_TABLE[128], 32_792);
        assert_eq!(FIELD_NORMS_TABLE[200], 16_777_240);
        assert_eq!(FIELD_NORMS_TABLE[255], 2_013_265_944);
        let sum: u64 = FIELD_NORMS_TABLE.iter().map(|&v| u64::from(v)).sum();
        assert_eq!(sum, 24_696_067_732, "table checksum drifted");
    }

    #[test]
    fn table_is_strictly_increasing() {
        for window in FIELD_NORMS_TABLE.windows(2) {
            assert!(
                window[0] < window[1],
                "table not strictly increasing at {window:?}"
            );
        }
    }

    #[test]
    fn encode_decode_roundtrip_all_ids() {
        for id in 0..=u8::MAX {
            assert_eq!(fieldnorm_to_id(id_to_fieldnorm(id)), id);
        }
    }

    #[test]
    fn encode_matches_tantivy_for_lengths_0_to_10k() {
        for len in 0..10_000_u32 {
            let id = fieldnorm_to_id(len);
            assert_eq!(
                id,
                FieldNormReader::fieldnorm_to_id(len),
                "encode diverges from oracle at length {len}"
            );
            let decoded = id_to_fieldnorm(id);
            assert!(decoded <= len, "bucket floor exceeds length {len}");
            if id < u8::MAX {
                assert!(
                    id_to_fieldnorm(id + 1) > len,
                    "length {len} not in tightest bucket"
                );
            }
        }
    }

    #[test]
    fn encode_saturates_at_max_bucket() {
        assert_eq!(fieldnorm_to_id(u32::MAX), u8::MAX);
        assert_eq!(fieldnorm_to_id(2_013_265_944), u8::MAX);
        assert_eq!(fieldnorm_to_id(2_013_265_943), 254);
    }

    #[test]
    fn tf_cache_is_bitwise_identical_to_oracle_formula() {
        for &avg in &[1.0_f32, 10.5, 487.3, 65_530.0] {
            let cache = compute_tf_cache(avg);
            for id in 0..=u8::MAX {
                let fieldnorm = FieldNormReader::id_to_fieldnorm(id) as f32;
                let expected = 1.2_f32 * (1.0 - 0.75 + 0.75 * fieldnorm / avg);
                assert_eq!(
                    cache[usize::from(id)].to_bits(),
                    expected.to_bits(),
                    "tf cache diverges at id {id}, avgdl {avg}"
                );
            }
        }
    }

    #[test]
    fn idf_matches_f64_reference_within_f32_tolerance() {
        for &(doc_freq, doc_count) in &[
            (1_u64, 1_u64),
            (1, 2),
            (1, 1_000),
            (37, 1_000),
            (500, 1_000),
            (999, 1_000),
        ] {
            let reference = {
                let x = ((doc_count - doc_freq) as f64 + 0.5) / (doc_freq as f64 + 0.5);
                x.ln_1p()
            };
            let got = f64::from(idf(doc_freq, doc_count));
            // The oracle's f32 `(1.0 + x).ln()` cancels low bits of x for
            // small x, bounding ABSOLUTE error near f32 epsilon (~1.2e-7)
            // regardless of how small the result is; the relative term covers
            // the large-idf regime. Both get ~10x headroom.
            assert!(
                (got - reference).abs() <= 1e-6 + reference.abs() * 1e-5,
                "idf({doc_freq}, {doc_count}) = {got}, f64 reference {reference}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "2 >= 3")]
    fn idf_rejects_doc_freq_above_doc_count() {
        let _ = idf(3, 2);
    }
}
