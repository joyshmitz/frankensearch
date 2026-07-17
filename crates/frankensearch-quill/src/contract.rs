//! Quill scoring constants vendored from the pinned Tantivy 0.26.1 oracle.
//!
//! Rank-exact BM25 conformance requires the same fieldnorm quantization and
//! f32 operation order. Per-document `|d|` is decoded through
//! [`FIELD_NORMS_TABLE`], while `avgdl` remains Tantivy's raw
//! `total_num_tokens / total_num_docs` value. Averaging decoded buckets would
//! change scores and is not conformant.

/// BM25 `k1`, pinned to Tantivy 0.26.1.
pub const BM25_K1: f32 = 1.2;
/// BM25 `b`, pinned to Tantivy 0.26.1.
pub const BM25_B: f32 = 0.75;

/// Fieldnorm ID to decoded document-length table.
///
/// Vendored verbatim from Tantivy 0.26.1 `src/fieldnorm/code.rs`. It is
/// strictly increasing: exact lengths 0..=40 followed by geometrically wider
/// buckets ending at `2_013_265_944`.
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

/// Decode a one-byte fieldnorm ID to its quantized document length.
#[inline]
#[must_use]
pub fn id_to_fieldnorm(id: u8) -> u32 {
    FIELD_NORMS_TABLE[usize::from(id)]
}

/// Encode a document length, rounding down to its fieldnorm bucket.
// A search hit is at most index 255; a miss inserts after table[0], so the
// preceding bucket is also at most 255 and cannot underflow.
#[allow(clippy::cast_lossless, clippy::cast_possible_truncation)]
#[inline]
#[must_use]
pub fn fieldnorm_to_id(fieldnorm: u32) -> u8 {
    FIELD_NORMS_TABLE
        .binary_search(&fieldnorm)
        .unwrap_or_else(|index| index - 1) as u8
}

/// Encode a posting block's maximum term frequency into one byte.
///
/// Codes `0..=254` are exact. Every frequency at or above 255 uses the
/// saturating code `255`, whose decoder deliberately returns [`u32::MAX`].
/// This is the conservative Tantivy 0.26.1 Block-WAND encoding: decoding an
/// encoded frequency can never under-estimate the original value.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub const fn block_max_frequency_to_code(max_frequency: u32) -> u8 {
    if max_frequency >= 255 {
        u8::MAX
    } else {
        max_frequency as u8
    }
}

/// Decode a posting block's conservative maximum-frequency code.
///
/// Code `255` is an unbounded sentinel, not the exact frequency 255.
#[inline]
#[must_use]
#[allow(clippy::cast_lossless)]
pub const fn block_max_frequency_from_code(code: u8) -> u32 {
    if code == u8::MAX {
        u32::MAX
    } else {
        code as u32
    }
}

/// Compute the conservative BM25 tf-factor for one durable BLOCKMAX pair.
///
/// The average field length is supplied from the live snapshot. Returning
/// `None` disables pruning for invalid snapshot inputs or for code zero, which
/// cannot describe a real non-empty posting block. The arithmetic order
/// matches the term scorer: first compute `norm`, then `f / (f + norm)`.
#[must_use]
pub(crate) fn block_max_tf_factor(
    max_frequency_code: u8,
    min_fieldnorm_id: u8,
    live_avgdl: f32,
) -> Option<f32> {
    if max_frequency_code == 0 || !live_avgdl.is_finite() || live_avgdl <= 0.0 {
        return None;
    }
    let frequency = block_max_frequency_from_code(max_frequency_code) as f32;
    let norm = cached_tf_component(id_to_fieldnorm(min_fieldnorm_id), live_avgdl);
    let factor = frequency / (frequency + norm);
    factor.is_finite().then_some(factor)
}

/// Apply a non-negative live BM25 weight to a conservative BLOCKMAX tf-factor.
///
/// Negative or non-finite weights return `None`: callers must disable pruning
/// in those regimes because multiplying by a negative weight reverses the
/// max-frequency/min-length monotonicity argument.
#[must_use]
pub(crate) fn block_max_score(
    max_frequency_code: u8,
    min_fieldnorm_id: u8,
    live_avgdl: f32,
    nonnegative_weight: f32,
) -> Option<f32> {
    if !nonnegative_weight.is_finite() || nonnegative_weight < 0.0 {
        return None;
    }
    let score =
        nonnegative_weight * block_max_tf_factor(max_frequency_code, min_fieldnorm_id, live_avgdl)?;
    score.is_finite().then_some(score)
}

/// Tantivy-compatible BM25 inverse document frequency.
///
/// # Panics
///
/// Panics when `doc_freq > doc_count`, matching the pinned oracle.
#[allow(clippy::imprecise_flops)]
#[must_use]
pub fn idf(doc_freq: u64, doc_count: u64) -> f32 {
    assert!(doc_count >= doc_freq, "{doc_count} >= {doc_freq}");
    let ratio = ((doc_count - doc_freq) as f32 + 0.5) / (doc_freq as f32 + 0.5);
    (1.0 + ratio).ln()
}

/// One Tantivy-compatible BM25 tf-cache entry.
///
/// `average_fieldnorm` is the raw average field length; the historical name is
/// retained to mirror the oracle. The operation order is part of the contract.
#[inline]
#[must_use]
pub fn cached_tf_component(fieldnorm: u32, average_fieldnorm: f32) -> f32 {
    BM25_K1 * (1.0 - BM25_B + BM25_B * fieldnorm as f32 / average_fieldnorm)
}

/// Precompute the 256 BM25 tf-cache entries for one field and snapshot.
#[must_use]
pub fn compute_tf_cache(average_fieldnorm: f32) -> [f32; 256] {
    let mut cache = [0.0_f32; 256];
    for (fieldnorm_id, slot) in cache.iter_mut().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let fieldnorm = id_to_fieldnorm(fieldnorm_id as u8);
        *slot = cached_tf_component(fieldnorm, average_fieldnorm);
    }
    cache
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_landmarks_and_checksum_are_pinned() {
        assert_eq!(FIELD_NORMS_TABLE[0], 0);
        assert_eq!(FIELD_NORMS_TABLE[39], 39);
        assert_eq!(FIELD_NORMS_TABLE[40], 40);
        assert_eq!(FIELD_NORMS_TABLE[41], 42);
        assert_eq!(FIELD_NORMS_TABLE[64], 152);
        assert_eq!(FIELD_NORMS_TABLE[128], 32_792);
        assert_eq!(FIELD_NORMS_TABLE[200], 16_777_240);
        assert_eq!(FIELD_NORMS_TABLE[255], 2_013_265_944);
        let checksum: u64 = FIELD_NORMS_TABLE
            .iter()
            .map(|&value| u64::from(value))
            .sum();
        assert_eq!(checksum, 24_696_067_732);
    }

    #[test]
    fn table_is_strictly_increasing_and_roundtrips() {
        assert!(
            FIELD_NORMS_TABLE
                .windows(2)
                .all(|window| window[0] < window[1])
        );
        for id in 0..=u8::MAX {
            assert_eq!(fieldnorm_to_id(id_to_fieldnorm(id)), id);
        }
    }

    #[test]
    fn encoding_floors_and_saturates() {
        assert_eq!(fieldnorm_to_id(41), 40);
        assert_eq!(fieldnorm_to_id(u32::MAX), u8::MAX);
        assert_eq!(fieldnorm_to_id(2_013_265_943), 254);
        assert_eq!(fieldnorm_to_id(2_013_265_944), u8::MAX);
    }

    #[test]
    fn block_max_frequency_encoding_is_exact_then_conservative() {
        for frequency in 0..u32::from(u8::MAX) {
            let code = block_max_frequency_to_code(frequency);
            assert_eq!(u32::from(code), frequency);
            assert_eq!(block_max_frequency_from_code(code), frequency);
        }
        for frequency in [255, 256, 65_535, u32::MAX] {
            let code = block_max_frequency_to_code(frequency);
            assert_eq!(code, u8::MAX);
            assert_eq!(block_max_frequency_from_code(code), u32::MAX);
            assert!(block_max_frequency_from_code(code) >= frequency);
        }
    }

    #[test]
    fn block_max_score_dominates_componentwise_smaller_inputs() {
        let max_frequency = 37;
        let min_fieldnorm_id = 4;
        let code = block_max_frequency_to_code(max_frequency);
        for average in [0.25_f32, 1.0, 17.5, 1_000.0, 1_000_000.0] {
            let weight = 3.25_f32;
            let bound = block_max_score(code, min_fieldnorm_id, average, weight)
                .expect("valid positive scoring regime");
            for frequency in 1..=max_frequency {
                for fieldnorm_id in [min_fieldnorm_id, 5, 40, 128, u8::MAX] {
                    let frequency = frequency as f32;
                    let norm = cached_tf_component(id_to_fieldnorm(fieldnorm_id), average);
                    let score = weight * (frequency / (frequency + norm));
                    assert!(
                        bound >= score,
                        "avg={average} frequency={frequency} fieldnorm_id={fieldnorm_id} bound={bound} score={score}"
                    );
                }
            }
        }
        assert_eq!(block_max_tf_factor(0, 0, 1.0), None);
        assert_eq!(block_max_tf_factor(1, 0, 0.0), None);
        assert_eq!(block_max_tf_factor(1, 0, f32::NAN), None);
        assert_eq!(block_max_score(1, 0, 1.0, -1.0), None);
        assert_eq!(block_max_score(1, 0, 1.0, f32::INFINITY), None);
    }

    #[test]
    fn tf_cache_preserves_oracle_operation_order() {
        for average in [1.0_f32, 10.5, 487.3, 65_530.0] {
            let cache = compute_tf_cache(average);
            for id in 0..=u8::MAX {
                let fieldnorm = id_to_fieldnorm(id) as f32;
                let expected = 1.2_f32 * (1.0 - 0.75 + 0.75 * fieldnorm / average);
                assert_eq!(cache[usize::from(id)].to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    #[allow(clippy::imprecise_flops)]
    fn idf_preserves_oracle_f32_operation_order() {
        for (doc_freq, doc_count) in [(0, 0), (1, 1), (1, 10), (59, 100), (1_000, 65_535)] {
            let ratio = ((doc_count - doc_freq) as f32 + 0.5) / (doc_freq as f32 + 0.5);
            let expected = (1.0_f32 + ratio).ln();
            assert_eq!(idf(doc_freq, doc_count).to_bits(), expected.to_bits());
        }
    }

    #[test]
    #[should_panic(expected = "2 >= 3")]
    fn idf_rejects_invalid_stats() {
        let _ = idf(3, 2);
    }
}
