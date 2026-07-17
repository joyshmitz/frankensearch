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
    0x9b, 0x59, 0x99, 0xd8, 0x3f, 0xc5, 0x38, 0xeb, 0x39, 0xae, 0x12, 0xf1, 0xe8, 0x35, 0x32, 0xd4,
    0x55, 0x0b, 0x4d, 0x89, 0x01, 0xfa, 0xf4, 0x9f, 0x93, 0x13, 0x70, 0xd8, 0xa7, 0xee, 0xa9, 0x3d,
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

/// One future Q1 fixture. `stub` is deliberately not a passing state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Q1FixtureStub {
    pub id: String,
    pub status: String,
    pub future_enforcement: String,
    pub assertion: String,
}

/// Committed catalog of pending Q1 obligations and live internal differentials.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Q1FixtureCatalog {
    pub schema_version: u32,
    pub source_contract: String,
    pub fixtures: Vec<Q1FixtureStub>,
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

/// Parse and validate all seven pending Q1 stubs and the live Quiver fixture.
///
/// # Errors
///
/// Returns an error when IDs are missing, duplicated, reordered, or accidentally
/// promoted from `stub` before executable enforcement exists, or when the
/// registered Quiver differential is not executable.
pub fn q1_fixture_catalog() -> Result<Q1FixtureCatalog, GauntletError> {
    let catalog: Q1FixtureCatalog = serde_json::from_str(Q1_FIXTURE_CATALOG_JSON)?;
    let catalog_hash = Sha256::digest(Q1_FIXTURE_CATALOG_JSON.as_bytes());
    const EXPECTED: [&str; 7] = [
        "Q1-OB1", "Q1-OB2a", "Q1-OB2b", "Q1-OB3", "Q1-OB4", "Q1-OB5", "Q1-OB6",
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
            fixture.status != "stub"
                || fixture.future_enforcement.is_empty()
                || fixture.assertion.is_empty()
        })
        || internal.len() != 1
        || internal[0].id != QUIVER_DIFFERENTIAL_FIXTURE_ID
        || internal[0].status != "executable"
        || internal[0].source_contract != "docs/contracts/fslx-format-registry.md#52-postings"
        || internal[0].enforcement.is_empty()
        || internal[0].assertion.is_empty()
    {
        return Err(GauntletError::InvalidContract {
            reason: "fixture catalog must contain seven ordered Q1 stubs and the executable Quiver differential"
                .to_owned(),
        });
    }
    Ok(catalog)
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
    fn q1_catalog_has_seven_stubs_and_one_executable_internal_differential() {
        let catalog = q1_fixture_catalog().expect("valid Q1 catalog");
        assert_eq!(
            Sha256::digest(Q1_FIXTURE_CATALOG_JSON.as_bytes()).as_slice(),
            Q1_FIXTURE_CATALOG_SHA256
        );
        assert_eq!(catalog.fixtures.len(), 7);
        assert!(
            catalog
                .fixtures
                .iter()
                .all(|fixture| fixture.status == "stub")
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
