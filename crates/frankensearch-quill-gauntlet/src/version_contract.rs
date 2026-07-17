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
const Q1_FIXTURE_CATALOG_SHA256: [u8; 32] = [
    0xe1, 0xb4, 0x26, 0xc0, 0x69, 0xb3, 0x76, 0x9d, 0x22, 0xb2, 0x04, 0xdf, 0x79, 0x92, 0xe2, 0xae,
    0x62, 0x45, 0x62, 0xf1, 0x7f, 0x34, 0x26, 0x28, 0x82, 0x40, 0x3b, 0x7e, 0x56, 0xd7, 0x88, 0xae,
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

/// Committed catalog of Q1 obligation fixtures waiting for executable engines.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Q1FixtureCatalog {
    pub schema_version: u32,
    pub source_contract: String,
    pub fixtures: Vec<Q1FixtureStub>,
    pub internal_differentials: Vec<InternalDifferentialStub>,
}

/// One future same-engine kernel differential. `stub` is non-passing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InternalDifferentialStub {
    pub id: String,
    pub status: String,
    pub source_contract: String,
    pub future_enforcement: String,
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

/// Parse and validate all seven pending Q1 fixture stubs.
///
/// # Errors
///
/// Returns an error when IDs are missing, duplicated, reordered, or accidentally
/// promoted from `stub` before executable enforcement exists.
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
        || internal[0].id != "quiver-postings-bitpack-scalar-wide-v1"
        || internal[0].status != "stub"
        || internal[0].source_contract != "docs/contracts/fslx-format-registry.md#52-postings"
        || internal[0].future_enforcement.is_empty()
        || internal[0].assertion.is_empty()
    {
        return Err(GauntletError::InvalidContract {
            reason: "fixture catalog must contain seven ordered Q1 stubs and the Quiver differential stub"
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
    fn q1_catalog_has_exactly_seven_non_passing_stubs() {
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
        assert_eq!(
            catalog.internal_differentials[0].id,
            "quiver-postings-bitpack-scalar-wide-v1"
        );
        assert_eq!(catalog.internal_differentials[0].status, "stub");
    }
}
