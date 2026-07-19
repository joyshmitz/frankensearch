use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::GauntletError;

const ORACLE_VERSION_CONTRACT_JSON: &str = include_str!("../oracle-version-contract.json");
const Q1_FIXTURE_CATALOG_JSON: &str = include_str!("../fixtures/q1-obligations.json");
const WORKSPACE_MANIFEST: &str = include_str!("../../../Cargo.toml");
const TANTIVY_VERSION: &str = "0.26.1";
const TANTIVY_CHECKSUM_SHA256: &str =
    "edde6a10743fff00a4e1a8c9ef020bf5f3cbad301b7d2d39f2b07f123c4eac07";
const QUIVER_DIFFERENTIAL_FIXTURE_ID: &str = "quiver-postings-bitpack-scalar-wide-v1";
const Q1_FIXTURE_CATALOG_SHA256: [u8; 32] = [
    0x32, 0x04, 0xe3, 0x59, 0xee, 0x73, 0x23, 0xc7, 0x46, 0x51, 0xd7, 0xd8, 0x95, 0xbf, 0x5f, 0xdf,
    0x8a, 0xf6, 0xe9, 0x4d, 0x9b, 0x5c, 0xb1, 0x28, 0x93, 0x37, 0x3e, 0xcc, 0x59, 0x53, 0x94, 0xa4,
];

/// Committed provenance contract for the shipping Tantivy oracle adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleVersionContract {
    pub schema_version: u32,
    pub tantivy_version: String,
    pub tantivy_checksum_sha256: String,
    pub lexical_package: String,
    pub lexical_package_version: String,
    pub lexical_git_revision: String,
    pub source_dirty_allowed: bool,
}

impl OracleVersionContract {
    /// Validate the source state supplied by a runner before admitting evidence.
    ///
    /// # Errors
    ///
    /// Returns an error for a mismatched revision or a dirty source tree. Build
    /// workers do not infer Git state; the runner must supply it explicitly.
    pub fn validate_source_state(
        &self,
        observed_revision: &str,
        source_dirty: bool,
    ) -> Result<(), GauntletError> {
        if observed_revision != self.lexical_git_revision {
            return Err(GauntletError::InvalidContract {
                reason: format!(
                    "lexical revision {observed_revision} does not match {}",
                    self.lexical_git_revision
                ),
            });
        }
        if source_dirty && !self.source_dirty_allowed {
            return Err(GauntletError::InvalidContract {
                reason: "dirty lexical source is not admissible evidence".to_owned(),
            });
        }
        Ok(())
    }
}

/// One Q1 obligation with an honest executable-or-deferred state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Q1Fixture {
    pub id: String,
    pub status: String,
    pub enforcement: String,
    pub assertion: String,
}

/// Committed catalog of pending Q1 obligations and live internal differentials.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Q1FixtureCatalog {
    pub schema_version: u32,
    pub source_contract: String,
    pub fixtures: Vec<Q1Fixture>,
    pub internal_differentials: Vec<InternalDifferentialFixture>,
}

/// One registered same-engine kernel differential.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InternalDifferentialFixture {
    pub id: String,
    pub status: String,
    pub source_contract: String,
    pub enforcement: String,
    pub assertion: String,
}

/// Parse and validate the embedded oracle version contract.
///
/// # Errors
///
/// Returns an error when a pin is absent, malformed, or no longer exact.
pub fn oracle_version_contract() -> Result<OracleVersionContract, GauntletError> {
    let contract: OracleVersionContract = serde_json::from_str(ORACLE_VERSION_CONTRACT_JSON)?;
    if contract.schema_version != 1
        || contract.tantivy_version != TANTIVY_VERSION
        || contract.tantivy_checksum_sha256 != TANTIVY_CHECKSUM_SHA256
        || contract.lexical_package != "frankensearch-lexical"
        || contract.lexical_package_version != "0.2.1"
        || contract.source_dirty_allowed
        || !is_lower_hex(&contract.tantivy_checksum_sha256, 64)
        || !is_lower_hex(&contract.lexical_git_revision, 40)
        || !WORKSPACE_MANIFEST
            .lines()
            .any(|line| line.trim() == "tantivy = \"=0.26.1\"")
    {
        return Err(GauntletError::InvalidContract {
            reason: "oracle version contract pins are incomplete or malformed".to_owned(),
        });
    }
    Ok(contract)
}

/// Parse and validate the complete Q1 ownership catalog and live differentials.
///
/// # Errors
///
/// Returns an error when IDs are missing, duplicated, reordered, assigned an
/// unknown status, or when a full normative Q1 obligation is falsely marked
/// executable before its downstream owner closes it.
pub fn q1_fixture_catalog() -> Result<Q1FixtureCatalog, GauntletError> {
    let catalog: Q1FixtureCatalog = serde_json::from_str(Q1_FIXTURE_CATALOG_JSON)?;
    let catalog_hash = Sha256::digest(Q1_FIXTURE_CATALOG_JSON.as_bytes());
    const EXPECTED: [&str; 8] = [
        "Q1-OB1", "Q1-OB2a", "Q1-OB2b", "Q1-OB2c", "Q1-OB3", "Q1-OB4", "Q1-OB5", "Q1-OB6",
    ];
    let ids = catalog
        .fixtures
        .iter()
        .map(|fixture| fixture.id.as_str())
        .collect::<Vec<_>>();
    let unique = ids.iter().copied().collect::<BTreeSet<_>>();
    let internal = catalog.internal_differentials.as_slice();
    // ubs:ignore — this SHA-256 pins public fixture bytes, not a secret or authenticator.
    if catalog_hash.as_slice() != Q1_FIXTURE_CATALOG_SHA256
        || catalog.schema_version != 1
        || catalog.source_contract
            != "docs/contracts/quill-q1-docid-discipline.md#6-obligations-each-becomes-a-gauntlet-fixture"
        || ids != EXPECTED
        || unique.len() != EXPECTED.len()
        || catalog.fixtures.iter().any(|fixture| {
            !matches!(fixture.status.as_str(), "executable" | "deferred")
                || fixture.enforcement.is_empty()
                || fixture.assertion.is_empty()
        })
        || catalog
            .fixtures
            .iter()
            .any(|fixture| fixture.status == "executable")
        || internal.len() != 1
        || internal[0].id != QUIVER_DIFFERENTIAL_FIXTURE_ID
        || internal[0].status != "executable"
        || internal[0].source_contract != "docs/contracts/fslx-format-registry.md#52-postings"
        || internal[0].enforcement.is_empty()
        || internal[0].assertion.is_empty()
    {
        return Err(GauntletError::InvalidContract {
            reason: "fixture catalog must contain eight ordered deferred normative Q1 rows and the executable Quiver differential"
                .to_owned(),
        });
    }
    Ok(catalog)
}

/// Execute the bounded Q1 sub-fixtures owned by the scalar G1a checkpoint.
///
/// Downstream merge, compaction, and multi-shard obligations remain explicit
/// `deferred` rows rather than receiving false-green placeholder assertions.
///
/// # Errors
///
/// Returns a contract error if interval validation, lease disjointness,
/// boundary cutting, burn accounting, or watermark monotonicity regresses.
pub fn run_q1_live_fixtures() -> Result<Vec<String>, GauntletError> {
    use frankensearch_quill::scribe::{DOCID_LEASE_BLOCK, DocIdAllocator};
    use frankensearch_quill::{
        CURRENT_ENGINE_VERSION, DEFAULT_SCHEMA, Manifest, ManifestSegment, TombstoneSet,
    };

    let schema_id = DEFAULT_SCHEMA
        .schema_id()
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("default schema is invalid while running Q1 fixtures: {error}"),
        })?;
    let segment = |segment_id, seal_seq, docid_lo, docid_hi| ManifestSegment {
        segment_id,
        seal_seq,
        file_len: 1,
        file_xxh3: segment_id,
        docid_lo,
        docid_hi,
        doc_count: 1,
        tombstones: TombstoneSet::new(),
    };
    let mut manifest = Manifest {
        generation: 2,
        docid_high_watermark: 2 * DOCID_LEASE_BLOCK,
        schema_id,
        engine_version: CURRENT_ENGINE_VERSION,
        flags: 0,
        last_publish_unix_s: 0,
        segments: vec![
            segment(1, 1, 0, 1),
            segment(2, 2, DOCID_LEASE_BLOCK, DOCID_LEASE_BLOCK + 1),
        ],
        field_stats: Vec::new(),
    };
    manifest
        .validate()
        .map_err(|error| GauntletError::InvalidContract {
            reason: format!("Q1-OB1 valid disjoint intervals failed: {error}"),
        })?;
    manifest.segments[1].docid_lo = 0;
    if manifest.validate().is_ok() {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB1 admitted overlapping segment intervals".to_owned(),
        });
    }

    let mut allocator = DocIdAllocator::open(0, 2).map_err(q1_allocator_error)?;
    let first = allocator.alloc_batch(0, 10).map_err(q1_allocator_error)?;
    let second = allocator.alloc_batch(1, 10).map_err(q1_allocator_error)?;
    if first.spans()[0].global_end() > second.spans()[0].global_first() {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB5 granted overlapping concurrent leases".to_owned(),
        });
    }
    let lease_batch =
        u32::try_from(DOCID_LEASE_BLOCK).map_err(|_| GauntletError::InvalidContract {
            reason: "Q1 lease block does not fit the batch-allocation contract".to_owned(),
        })?;
    let boundary = allocator
        .alloc_batch(0, lease_batch)
        .map_err(q1_allocator_error)?;
    if !boundary.crossed_lease() {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB5 did not cut an allocation at lease exhaustion".to_owned(),
        });
    }
    let burn = allocator.end_session();
    if burn.total_burned == 0 || burn.final_watermark < 2 * DOCID_LEASE_BLOCK {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB5 failed burn accounting or watermark monotonicity".to_owned(),
        });
    }
    let mut next = DocIdAllocator::open(burn.final_watermark, 1).map_err(q1_allocator_error)?;
    let next_batch = next.alloc_batch(0, 1).map_err(q1_allocator_error)?;
    if next_batch.spans()[0].global_first() < burn.final_watermark {
        return Err(GauntletError::InvalidContract {
            reason: "Q1-OB5 reused a burned document-id tail".to_owned(),
        });
    }
    Ok(vec![
        "Q1-OB1/manifest-overlap".to_owned(),
        "Q1-OB5/allocator-boundary".to_owned(),
    ])
}

fn q1_allocator_error(error: impl std::fmt::Display) -> GauntletError {
    GauntletError::InvalidContract {
        reason: format!("Q1 live allocator fixture failed: {error}"),
    }
}

fn is_lower_hex(value: &str, width: usize) -> bool {
    value.len() == width
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch_quill::quiver::differential::{
        BitpackError, FIXTURE_ID, SPEC_SECTION, pack_values, unpack_scalar_into, unpack_wide_into,
    };

    fn bitpack_fixture_values(width: u8, count: usize) -> Vec<u32> {
        let mask = match width {
            0 => 0,
            32 => u32::MAX,
            _ => (1_u32 << width) - 1,
        };
        let mut state = 0x9e37_79b9_u32
            ^ u32::from(width)
            ^ u32::try_from(count).expect("fixture count fits in u32");
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            values.push(state & mask);
        }
        if let Some(first) = values.first_mut() {
            *first = 0;
        }
        if let Some(last) = values.last_mut() {
            *last = mask;
        }
        values
    }

    fn matching_decode_error(input: &[u8], width: u8, count: usize) -> BitpackError {
        let untouched = vec![0xa5a5_a5a5; count];
        let mut scalar = untouched.clone();
        let mut wide = untouched.clone();
        let scalar_result = unpack_scalar_into(input, width, &mut scalar);
        let wide_result = unpack_wide_into(input, width, &mut wide);
        assert_eq!(scalar_result, wide_result, "typed error mismatch");
        assert_eq!(scalar, untouched, "scalar mutated output before rejecting");
        assert_eq!(wide, untouched, "wide mutated output before rejecting");
        scalar_result.expect_err("malformed fixture must be rejected")
    }

    #[test]
    fn embedded_oracle_contract_is_exact_and_rejects_dirty_source() {
        let contract = oracle_version_contract().expect("valid oracle contract");
        assert_eq!(contract.tantivy_version, TANTIVY_VERSION);
        assert_eq!(contract.tantivy_checksum_sha256, TANTIVY_CHECKSUM_SHA256);
        assert_eq!(contract.lexical_git_revision.len(), 40);
        assert!(
            contract
                .validate_source_state(&contract.lexical_git_revision, false)
                .is_ok()
        );
        assert!(
            contract
                .validate_source_state(&contract.lexical_git_revision, true)
                .is_err()
        );
        assert!(
            contract
                .validate_source_state(&"0".repeat(40), false)
                .is_err()
        );
    }

    #[test]
    fn q1_catalog_keeps_full_obligations_deferred_and_runs_owned_subfixtures() {
        let catalog = q1_fixture_catalog().expect("valid Q1 catalog");
        assert_eq!(
            Sha256::digest(Q1_FIXTURE_CATALOG_JSON.as_bytes()).as_slice(),
            Q1_FIXTURE_CATALOG_SHA256
        );
        assert_eq!(catalog.fixtures.len(), 8);
        assert_eq!(
            catalog
                .fixtures
                .iter()
                .filter(|fixture| fixture.status == "executable")
                .map(|fixture| fixture.id.as_str())
                .collect::<Vec<_>>(),
            Vec::<&str>::new()
        );
        assert_eq!(
            run_q1_live_fixtures().expect("live Q1 fixtures"),
            ["Q1-OB1/manifest-overlap", "Q1-OB5/allocator-boundary"]
        );
        assert_eq!(catalog.internal_differentials.len(), 1);
        assert_eq!(catalog.internal_differentials[0].id, FIXTURE_ID);
        assert_eq!(catalog.internal_differentials[0].status, "executable");
    }

    #[test]
    fn quiver_bitpack_scalar_and_wide_fixture_is_executable() -> Result<(), BitpackError> {
        let catalog = q1_fixture_catalog().expect("valid fixture catalog");
        let fixture = &catalog.internal_differentials[0];
        assert_eq!(fixture.id, FIXTURE_ID);
        assert_eq!(fixture.status, "executable");
        assert_eq!(
            fixture.source_contract,
            "docs/contracts/fslx-format-registry.md#52-postings"
        );
        assert_eq!(FIXTURE_ID, "quiver-postings-bitpack-scalar-wide-v1");
        assert_eq!(
            SPEC_SECTION,
            "FSLX v1 section 5.2 LSB-first bitpacked payloads"
        );

        let known_answers: [(&[u8], u8, &[u32]); 3] = [
            (&[0x51, 0x01], 3, &[1, 2, 5]),
            (&[0x39], 2, &[1, 2, 3]),
            (
                &[0x12, 0x34, 0x56, 0x78, 0xef, 0xbe, 0xad, 0xde],
                32,
                &[0x7856_3412, 0xdead_beef],
            ),
        ];
        for (packed, width, expected) in known_answers {
            assert_eq!(pack_values(expected, width)?, packed);
            let mut scalar = vec![u32::MAX; expected.len()];
            let mut wide = scalar.clone();
            unpack_scalar_into(packed, width, &mut scalar)?;
            unpack_wide_into(packed, width, &mut wide)?;
            assert_eq!(scalar, expected, "known scalar answer width={width}");
            assert_eq!(wide, expected, "known wide answer width={width}");
        }

        for (shape, count) in [("doc-delta", 127), ("frequency", 128)] {
            for width in 0..=32 {
                let expected = bitpack_fixture_values(width, count);
                let packed = pack_values(&expected, width)?;
                for offset in 0..32 {
                    let mut storage = vec![0x5a; offset];
                    let payload_start = storage.len();
                    storage.extend_from_slice(&packed);
                    let payload_end = storage.len();
                    storage.extend_from_slice(&[0xa5; 32]);
                    let input = &storage[payload_start..payload_end];
                    let mut scalar = vec![u32::MAX; count];
                    let mut wide = scalar.clone();

                    let scalar_result = unpack_scalar_into(input, width, &mut scalar);
                    let wide_result = unpack_wide_into(input, width, &mut wide);
                    assert_eq!(
                        scalar_result, wide_result,
                        "result mismatch for {shape} width={width} offset={offset}"
                    );
                    scalar_result?;
                    assert_eq!(
                        scalar, expected,
                        "scalar mismatch for {shape} width={width} offset={offset}"
                    );
                    assert_eq!(
                        wide, expected,
                        "wide mismatch for {shape} width={width} offset={offset}"
                    );
                }
            }
        }

        let expected = bitpack_fixture_values(7, 127);
        let packed = pack_values(&expected, 7)?;
        let truncated = &packed[..packed.len() - 1];
        assert_eq!(
            matching_decode_error(truncated, 7, expected.len()),
            BitpackError::LengthMismatch {
                expected: packed.len(),
                actual: truncated.len(),
            }
        );

        let mut overlong = packed.clone();
        overlong.push(0);
        assert_eq!(
            matching_decode_error(&overlong, 7, expected.len()),
            BitpackError::LengthMismatch {
                expected: packed.len(),
                actual: overlong.len(),
            }
        );

        let mut noncanonical = pack_values(&bitpack_fixture_values(3, 127), 3)?;
        let final_byte = noncanonical
            .last_mut()
            .expect("127 three-bit values have a final byte");
        *final_byte |= 0x80;
        let final_byte = *final_byte;
        assert_eq!(
            matching_decode_error(&noncanonical, 3, 127),
            BitpackError::NonCanonicalPadding {
                byte: final_byte,
                used_bits: 5,
            }
        );
        assert_eq!(
            matching_decode_error(&[], 33, 128),
            BitpackError::InvalidWidth { width: 33 }
        );
        assert_eq!(
            matching_decode_error(&[0], 0, 128),
            BitpackError::LengthMismatch {
                expected: 0,
                actual: 1,
            }
        );
        assert_eq!(
            pack_values(&[1], 0),
            Err(BitpackError::ValueOutOfRange {
                index: 0,
                value: 1,
                width: 0,
            })
        );
        assert_eq!(
            pack_values(&[0, 8], 3),
            Err(BitpackError::ValueOutOfRange {
                index: 1,
                value: 8,
                width: 3,
            })
        );

        Ok(())
    }
}
