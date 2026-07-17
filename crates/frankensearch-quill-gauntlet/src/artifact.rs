use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

use crate::GauntletError;
use crate::comparator::{ComparatorConfig, compare_observations};
use crate::engine::{EnginePairIdentity, HarnessRun, TANTIVY_ORACLE_CONFIG_HASH};
use crate::version_contract::{OracleVersionContract, oracle_version_contract};

pub const OBJECT_SCHEMA_VERSION: u32 = 1;
pub const CANONICALIZATION_VERSION: u32 = 1;
const HASH_DOMAIN: &[u8] = b"frankensearch-quill-gauntlet:artifact-object:v1\0";

/// Immutable comparison object. Run-local provenance is deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactObject {
    pub object_schema_version: u32,
    pub canonicalization_version: u32,
    pub oracle_version: OracleVersionContract,
    pub engines: EnginePairIdentity,
    pub case: crate::DifferentialCase,
    pub comparator_config: ComparatorConfig,
    pub comparison: crate::ComparisonReport,
}

impl ArtifactObject {
    /// Build an immutable object from one completed harness run.
    ///
    /// # Errors
    ///
    /// Returns an error when the committed oracle version contract is invalid.
    pub fn from_run(run: HarnessRun) -> Result<Self, GauntletError> {
        Ok(Self {
            object_schema_version: OBJECT_SCHEMA_VERSION,
            canonicalization_version: CANONICALIZATION_VERSION,
            oracle_version: oracle_version_contract()?,
            engines: run.engines,
            case: run.case,
            comparator_config: run.comparator_config,
            comparison: run.comparison,
        })
    }

    /// Canonical compact JSON bytes used as the immutable object body.
    ///
    /// Hashed DTOs use fixed struct field order, typed metadata, integer score
    /// bits, and preserved vector order. The output has no trailing newline.
    ///
    /// # Errors
    ///
    /// Returns a JSON serialization error if the schema stops being encodable.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, GauntletError> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Compute the domain-separated xxh3-64 object address.
    ///
    /// # Errors
    ///
    /// Returns an error when canonical serialization fails.
    pub fn object_hash(&self) -> Result<String, GauntletError> {
        let bytes = self.canonical_bytes()?;
        Ok(hash_object_bytes(&bytes))
    }

    fn validate(&self) -> Result<(), GauntletError> {
        if self.object_schema_version != OBJECT_SCHEMA_VERSION
            || self.canonicalization_version != CANONICALIZATION_VERSION
            || self.oracle_version != oracle_version_contract()?
        {
            return Err(GauntletError::InvalidContract {
                reason: "artifact object schema or embedded oracle contract is invalid".to_owned(),
            });
        }
        let rebuilt = EnginePairIdentity::new(
            self.engines.comparison_mode,
            self.engines.subject.clone(),
            self.engines.oracle.clone(),
        )?;
        if rebuilt != self.engines {
            return Err(GauntletError::InvalidContract {
                reason: "artifact engine identity is not self-consistent".to_owned(),
            });
        }
        if self.engines.comparison_mode == crate::ComparisonMode::CrossEngine
            && (self.engines.oracle.implementation != "frankensearch-lexical/tantivy-index"
                || self.engines.oracle.crate_version != self.oracle_version.lexical_package_version
                || self.engines.oracle.source_revision != self.oracle_version.lexical_git_revision
                || self.engines.oracle.config_hash != TANTIVY_ORACLE_CONFIG_HASH
                || self.engines.oracle.source_dirty)
        {
            return Err(GauntletError::InvalidContract {
                reason: "oracle descriptor does not match the embedded lexical version contract"
                    .to_owned(),
            });
        }
        self.comparator_config.validate_contract()?;
        self.case.validate_observations(
            &self.engines,
            &self.comparison.subject,
            &self.comparison.oracle,
        )?;
        let recomputed = compare_observations(
            self.comparison.subject.clone(),
            self.comparison.oracle.clone(),
            self.comparator_config,
        )?;
        if recomputed != self.comparison {
            return Err(GauntletError::InvalidContract {
                reason: "artifact comparison report does not match its observations".to_owned(),
            });
        }
        Ok(())
    }
}

/// Mutable run provenance referencing one immutable object hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunManifest {
    pub schema_version: u32,
    pub run_id: String,
    pub object_hash: String,
    pub provenance: BTreeMap<String, String>,
}

/// Fully encoded paths and bytes, prepared without filesystem mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedArtifact {
    object_hash: String,
    object_path: PathBuf,
    object_bytes: Vec<u8>,
    run_path: PathBuf,
    run_manifest: RunManifest,
    run_manifest_bytes: Vec<u8>,
}

impl PreparedArtifact {
    #[must_use]
    pub fn object_hash(&self) -> &str {
        &self.object_hash
    }

    #[must_use]
    pub fn object_path(&self) -> &Path {
        &self.object_path
    }

    #[must_use]
    pub fn object_bytes(&self) -> &[u8] {
        &self.object_bytes
    }

    #[must_use]
    pub fn run_path(&self) -> &Path {
        &self.run_path
    }

    #[must_use]
    pub const fn run_manifest(&self) -> &RunManifest {
        &self.run_manifest
    }

    #[must_use]
    pub fn run_manifest_bytes(&self) -> &[u8] {
        &self.run_manifest_bytes
    }
}

/// Store rooted at `.gauntlet`, with immutable objects separated from runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactStore {
    root: PathBuf,
}

impl Default for ArtifactStore {
    fn default() -> Self {
        Self::new(".gauntlet")
    }
}

impl ArtifactStore {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Encode an object and run manifest without writing files.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid identities/contracts, unsafe run IDs, or
    /// serialization failures.
    pub fn prepare(
        &self,
        run_id: &str,
        object: &ArtifactObject,
        provenance: BTreeMap<String, String>,
    ) -> Result<PreparedArtifact, GauntletError> {
        validate_run_id(run_id)?;
        object.validate()?;
        let object_bytes = object.canonical_bytes()?;
        let object_hash = object.object_hash()?;
        let run_manifest = RunManifest {
            schema_version: 1,
            run_id: run_id.to_owned(),
            object_hash: object_hash.clone(),
            provenance,
        };
        let run_manifest_bytes = serde_json::to_vec(&run_manifest)?;
        Ok(PreparedArtifact {
            object_path: self
                .root
                .join("objects")
                .join(format!("{object_hash}.json")),
            run_path: self.root.join("runs").join(format!("{run_id}.json")),
            object_hash,
            object_bytes,
            run_manifest,
            run_manifest_bytes,
        })
    }

    /// Persist an already prepared object and run reference without overwrites.
    ///
    /// Existing object bytes must match exactly. Existing run IDs must reference
    /// exactly the same manifest. The store never deletes or replaces files.
    ///
    /// # Errors
    ///
    /// Returns I/O, object-collision, or run-conflict errors.
    pub fn persist(&self, prepared: &PreparedArtifact) -> Result<(), GauntletError> {
        self.validate_prepared(prepared)?;
        ensure_real_directory(&self.root)?;
        ensure_real_directory(&self.root.join("objects"))?;
        ensure_real_directory(&self.root.join("runs"))?;
        write_once_or_verify(
            &prepared.object_path,
            &prepared.object_bytes,
            ExistingFileKind::Object,
        )?;
        write_once_or_verify(
            &prepared.run_path,
            &prepared.run_manifest_bytes,
            ExistingFileKind::Run,
        )?;
        Ok(())
    }

    fn validate_prepared(&self, prepared: &PreparedArtifact) -> Result<(), GauntletError> {
        let object: ArtifactObject = serde_json::from_slice(&prepared.object_bytes)?;
        object.validate()?;
        if object.canonical_bytes()? != prepared.object_bytes
            || hash_object_bytes(&prepared.object_bytes) != prepared.object_hash
            || prepared.object_path
                != self
                    .root
                    .join("objects")
                    .join(format!("{}.json", prepared.object_hash))
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "object bytes, hash, or store path are inconsistent".to_owned(),
            });
        }
        validate_run_id(&prepared.run_manifest.run_id)?;
        if prepared.run_manifest.schema_version != 1
            || prepared.run_manifest.object_hash != prepared.object_hash
            || serde_json::to_vec(&prepared.run_manifest)? != prepared.run_manifest_bytes
            || prepared.run_path
                != self
                    .root
                    .join("runs")
                    .join(format!("{}.json", prepared.run_manifest.run_id))
        {
            return Err(GauntletError::InvalidPreparedArtifact {
                reason: "run manifest bytes, object reference, or store path are inconsistent"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ExistingFileKind {
    Object,
    Run,
}

fn write_once_or_verify(
    path: &Path,
    bytes: &[u8],
    kind: ExistingFileKind,
) -> Result<(), GauntletError> {
    match publish_atomic_no_clobber(path, bytes) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = read_regular_file(path)?;
            if existing == bytes {
                sync_parent_directory(path)?;
                Ok(())
            } else {
                Err(match kind {
                    ExistingFileKind::Object => GauntletError::ArtifactCollision {
                        path: path.to_path_buf(),
                    },
                    ExistingFileKind::Run => GauntletError::RunManifestConflict {
                        path: path.to_path_buf(),
                    },
                })
            }
        }
        Err(error) => Err(error.into()),
    }
}

fn hash_object_bytes(bytes: &[u8]) -> String {
    let mut hash_input = Vec::with_capacity(HASH_DOMAIN.len() + bytes.len());
    hash_input.extend_from_slice(HASH_DOMAIN);
    hash_input.extend_from_slice(bytes);
    format!("{:016x}", xxh3_64(&hash_input))
}

fn ensure_real_directory(path: &Path) -> Result<(), GauntletError> {
    ensure_real_ancestors(path)?;
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            sync_parent_directory(path)?;
            Ok(())
        }
        Ok(_) => Err(GauntletError::UnsafeStorePath {
            path: path.to_path_buf(),
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match std::fs::create_dir(path) {
                Ok(()) => {
                    sync_parent_directory(path)?;
                    Ok(())
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    ensure_real_directory(path)
                }
                Err(error) => Err(error.into()),
            }
        }
        Err(error) => Err(error.into()),
    }
}

fn sync_parent_directory(path: &Path) -> Result<(), GauntletError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn ensure_real_ancestors(path: &Path) -> Result<(), GauntletError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    let mut current = PathBuf::new();
    for component in parent.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(GauntletError::UnsafeStorePath {
                path: parent.to_path_buf(),
            });
        }
        current.push(component.as_os_str());
        if current.as_os_str().is_empty() {
            continue;
        }
        let metadata = std::fs::symlink_metadata(&current)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(GauntletError::UnsafeStorePath { path: current });
        }
    }
    Ok(())
}

fn read_regular_file(path: &Path) -> Result<Vec<u8>, GauntletError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(GauntletError::UnsafeStorePath {
            path: path.to_path_buf(),
        });
    }
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
    File::open(path)?.read_to_end(&mut bytes)?;
    Ok(bytes)
}

#[cfg(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "redox"
))]
fn publish_atomic_no_clobber(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use rustix::fs::{FlockOperation, Mode, OFlags, RenameFlags, flock, openat, renameat_with};

    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing parent"))?;
    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing file name")
    })?;
    let directory = File::open(parent)?;
    flock(&directory, FlockOperation::LockExclusive).map_err(std::io::Error::from)?;
    match std::fs::symlink_metadata(path) {
        Ok(_) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "artifact already exists",
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let mut pending_name = OsString::from(".");
    pending_name.push(file_name);
    pending_name.push(".pending");
    let temporary = openat(
        &directory,
        &pending_name,
        OFlags::RDWR | OFlags::CREATE | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(std::io::Error::from)?;
    let mut temporary = File::from(temporary);
    temporary.seek(SeekFrom::Start(0))?;
    let mut existing = Vec::new();
    temporary.read_to_end(&mut existing)?;
    if !bytes.starts_with(&existing) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "staged artifact is not a prefix of the canonical bytes",
        ));
    }
    temporary.seek(SeekFrom::End(0))?;
    temporary.write_all(&bytes[existing.len()..])?;
    temporary.sync_all()?;
    renameat_with(
        &directory,
        &pending_name,
        &directory,
        file_name,
        RenameFlags::NOREPLACE,
    )
    .map_err(std::io::Error::from)?;
    directory.sync_all()
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "redox"
)))]
fn publish_atomic_no_clobber(_path: &Path, _bytes: &[u8]) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "atomic no-clobber artifact publication is unsupported on this platform",
    ))
}

fn validate_run_id(run_id: &str) -> Result<(), GauntletError> {
    let safe = !run_id.is_empty()
        && run_id.len() <= 128
        && run_id != "."
        && run_id != ".."
        && run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'));
    if safe {
        Ok(())
    } else {
        Err(GauntletError::InvalidRunId {
            run_id: run_id.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ComparisonMode, CountState, DifferentialCase, EngineDescriptor, EngineFamily,
        EngineObservation, NativeTieKey, RankedHit, compare_observations,
    };

    fn representative_observation(
        hits: Vec<RankedHit>,
        snippets: BTreeMap<String, String>,
    ) -> EngineObservation {
        EngineObservation {
            hits,
            cutoff_tie_group: Vec::new(),
            cutoff_tie_complete: true,
            snippets,
            match_count: CountState::Value(2),
            doc_count: 2,
        }
    }

    fn sample_object() -> ArtifactObject {
        let oracle_version = oracle_version_contract().expect("version contract");
        let subject = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: "quill-stub".to_owned(),
            crate_version: "0.2.1".to_owned(),
            source_revision: "stub".to_owned(),
            source_dirty: false,
            config_hash: "01".to_owned(),
        };
        let oracle = EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "frankensearch-lexical/tantivy-index".to_owned(),
            crate_version: oracle_version.lexical_package_version.clone(),
            source_revision: oracle_version.lexical_git_revision.clone(),
            source_dirty: false,
            config_hash: TANTIVY_ORACLE_CONFIG_HASH.to_owned(),
        };
        let subject_observation = representative_observation(
            vec![
                RankedHit {
                    doc_id: "β/~second".to_owned(),
                    score_bits: 4.0_f32.to_bits(),
                    native_tie_key: NativeTieKey::QuillDocId { doc_id: 1 },
                },
                RankedHit {
                    doc_id: "α-first".to_owned(),
                    score_bits: 4.0_f32.to_bits(),
                    native_tie_key: NativeTieKey::QuillDocId { doc_id: 2 },
                },
            ],
            BTreeMap::from([
                ("α-first".to_owned(), "<b>α</b> body".to_owned()),
                ("β/~second".to_owned(), "<b>β</b> body".to_owned()),
            ]),
        );
        let oracle_observation = representative_observation(
            vec![
                RankedHit {
                    doc_id: "α-first".to_owned(),
                    score_bits: 4.0_f32.to_bits(),
                    native_tie_key: NativeTieKey::TantivyDocAddress {
                        segment_ord: 3,
                        doc_id: 8,
                    },
                },
                RankedHit {
                    doc_id: "β/~second".to_owned(),
                    score_bits: 4.0_f32.to_bits(),
                    native_tie_key: NativeTieKey::TantivyDocAddress {
                        segment_ord: 3,
                        doc_id: 9,
                    },
                },
            ],
            BTreeMap::from([
                ("β/~second".to_owned(), "<b>β</b> body".to_owned()),
                ("α-first".to_owned(), "<b>α</b> body".to_owned()),
            ]),
        );
        let comparator_config = ComparatorConfig::default();
        let comparison =
            compare_observations(subject_observation, oracle_observation, comparator_config)
                .expect("representative comparison");
        let mut case = DifferentialCase::new("artifact-β/~smoke", "rust β", 2);
        case.metadata.generator_id = Some("quill-generator-v1".to_owned());
        case.metadata.generator_seed = Some(42);
        case.metadata.corpus_hash = Some("0123456789abcdef".to_owned());
        ArtifactObject {
            object_schema_version: OBJECT_SCHEMA_VERSION,
            canonicalization_version: CANONICALIZATION_VERSION,
            oracle_version,
            engines: EnginePairIdentity::new(ComparisonMode::CrossEngine, subject, oracle)
                .expect("distinct engines"),
            case,
            comparator_config,
            comparison,
        }
    }

    #[test]
    fn run_ids_reference_one_immutable_object() {
        let object = sample_object();
        let store = ArtifactStore::default();
        let first = store
            .prepare("run-one", &object, BTreeMap::new())
            .expect("first preparation");
        let second = store
            .prepare("run-two", &object, BTreeMap::new())
            .expect("second preparation");

        assert_eq!(first.object_hash, second.object_hash);
        assert_eq!(first.object_bytes, second.object_bytes);
        assert_ne!(first.run_manifest_bytes, second.run_manifest_bytes);
        assert_eq!(
            first.object_path,
            Path::new(".gauntlet")
                .join("objects")
                .join(format!("{}.json", first.object_hash))
        );
    }

    #[test]
    fn canonical_bytes_and_hash_are_repeatable() {
        let mut first = sample_object();
        first
            .case
            .metadata
            .generator_id
            .clone_from(&Some("quill-generator-v1".to_owned()));
        let second = first.clone();
        assert_eq!(
            first.canonical_bytes().unwrap(),
            second.canonical_bytes().unwrap()
        );
        assert_eq!(first.object_hash().unwrap(), second.object_hash().unwrap());
    }

    #[test]
    fn prepare_rejects_oracle_descriptor_outside_version_contract() {
        let mut object = sample_object();
        object.engines.oracle.source_revision = "f".repeat(40);
        assert!(
            ArtifactStore::default()
                .prepare("bad-oracle-pin", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn prepare_rejects_unpinned_oracle_configuration() {
        let mut object = sample_object();
        object.engines.oracle.config_hash = "different-schema".to_owned();
        assert!(
            ArtifactStore::default()
                .prepare("bad-oracle-config", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn prepare_rejects_fabricated_exact_report() {
        let mut object = sample_object();
        object.comparison.status = crate::ComparisonStatus::Exact;
        object.comparison.rank_class = crate::RankClass::RankExact;
        object.comparison.divergences.clear();
        object.comparison.first_divergence = None;
        assert!(
            ArtifactStore::default()
                .prepare("forged-report", &object, BTreeMap::new())
                .is_err()
        );
    }

    #[test]
    fn comparator_policy_changes_object_hash() {
        let first = sample_object();
        let mut second = first.clone();
        second.comparator_config = second
            .comparator_config
            .with_score_epsilon_reason(crate::ScoreEpsilonReason::PlatformLibm);
        second.comparison = compare_observations(
            second.comparison.subject.clone(),
            second.comparison.oracle.clone(),
            second.comparator_config,
        )
        .expect("comparison under recorded policy");
        assert_ne!(first.object_hash().unwrap(), second.object_hash().unwrap());
    }

    #[test]
    fn canonical_object_golden_bytes_and_hash_are_pinned() {
        let object = sample_object();
        let canonical = object.canonical_bytes().unwrap();
        let golden_with_newline = include_bytes!("../fixtures/artifact-object-v1.json");
        let golden = golden_with_newline
            .strip_suffix(b"\n")
            .expect("golden fixture must end in exactly one LF");
        assert_eq!(object.object_hash().unwrap(), "46cf5c8d37641b7e");
        assert_eq!(canonical, golden);
    }

    #[test]
    fn object_embeds_both_engines_and_the_exact_version_contract() {
        let object = sample_object();
        assert_eq!(object.engines.subject.family, EngineFamily::Quill);
        assert_eq!(object.engines.oracle.family, EngineFamily::Tantivy);
        assert_eq!(object.oracle_version, oracle_version_contract().unwrap());
        assert_eq!(object.object_hash().unwrap().len(), 16);
        let encoded = serde_json::to_value(&object).expect("serialize object");
        let pointer = object
            .comparison
            .first_divergence
            .as_deref()
            .expect("representative divergence");
        assert!(encoded.pointer(pointer).is_some());
    }

    #[test]
    fn score_bit_dtos_preserve_negative_zero_and_nan_payloads() {
        let values = [(-0.0_f32).to_bits(), f32::from_bits(0x7fc0_1234).to_bits()];
        let json = serde_json::to_vec(&values).expect("serialize bits");
        let decoded: [u32; 2] = serde_json::from_slice(&json).expect("deserialize bits");
        assert_eq!(decoded, values);
    }

    #[test]
    fn unsafe_run_ids_are_rejected() {
        let object = sample_object();
        let store = ArtifactStore::default();
        assert!(
            store
                .prepare("../escape", &object, BTreeMap::new())
                .is_err()
        );
        assert!(store.prepare("", &object, BTreeMap::new()).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn persist_is_idempotent_and_rejects_cross_store_preparation() {
        let temp = tempfile::tempdir().expect("temporary parent");
        let first_store = ArtifactStore::new(temp.path().join("first"));
        let second_store = ArtifactStore::new(temp.path().join("second"));
        let prepared = first_store
            .prepare("run-one", &sample_object(), BTreeMap::new())
            .expect("prepare artifact");

        first_store.persist(&prepared).expect("first publication");
        first_store
            .persist(&prepared)
            .expect("idempotent publication");
        assert_eq!(
            std::fs::read(prepared.object_path()).expect("read object"),
            prepared.object_bytes()
        );
        assert!(second_store.persist(&prepared).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn concurrent_identical_publications_both_succeed() {
        use std::sync::{Arc, Barrier};

        let temp = tempfile::tempdir().expect("temporary parent");
        let store = Arc::new(ArtifactStore::new(temp.path().join("gauntlet")));
        let prepared = Arc::new(
            store
                .prepare("concurrent", &sample_object(), BTreeMap::new())
                .expect("prepare artifact"),
        );
        let barrier = Arc::new(Barrier::new(2));
        let workers = (0..2)
            .map(|_| {
                let store = Arc::clone(&store);
                let prepared = Arc::clone(&prepared);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    store.persist(&prepared)
                })
            })
            .collect::<Vec<_>>();

        for worker in workers {
            worker.join().expect("worker did not panic").unwrap();
        }
        assert_eq!(
            std::fs::read_dir(store.root().join("objects"))
                .expect("read object directory")
                .count(),
            1
        );
        assert_eq!(
            std::fs::read_dir(store.root().join("runs"))
                .expect("read run directory")
                .count(),
            1
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn partial_staging_file_is_resumed_before_atomic_publish() {
        let temp = tempfile::tempdir().expect("temporary parent");
        let store = ArtifactStore::new(temp.path().join("gauntlet"));
        let prepared = store
            .prepare("resume-staging", &sample_object(), BTreeMap::new())
            .expect("prepare artifact");
        std::fs::create_dir(store.root()).expect("create store root");
        std::fs::create_dir(store.root().join("objects")).expect("create objects directory");
        std::fs::create_dir(store.root().join("runs")).expect("create runs directory");
        let final_name = prepared
            .object_path()
            .file_name()
            .expect("object file name");
        let mut pending_name = OsString::from(".");
        pending_name.push(final_name);
        pending_name.push(".pending");
        let split = prepared.object_bytes().len() / 2;
        std::fs::write(
            store.root().join("objects").join(pending_name),
            &prepared.object_bytes()[..split],
        )
        .expect("write partial staging prefix");

        store.persist(&prepared).expect("resume publication");
        assert_eq!(
            std::fs::read(prepared.object_path()).expect("read published object"),
            prepared.object_bytes()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn reused_run_id_with_different_provenance_is_rejected() {
        let temp = tempfile::tempdir().expect("temporary parent");
        let store = ArtifactStore::new(temp.path().join("gauntlet"));
        let first = store
            .prepare(
                "same-run",
                &sample_object(),
                BTreeMap::from([("worker".to_owned(), "one".to_owned())]),
            )
            .expect("first preparation");
        let second = store
            .prepare(
                "same-run",
                &sample_object(),
                BTreeMap::from([("worker".to_owned(), "two".to_owned())]),
            )
            .expect("second preparation");
        store.persist(&first).expect("first publication");
        assert!(matches!(
            store.persist(&second),
            Err(GauntletError::RunManifestConflict { .. })
        ));
    }

    #[cfg(all(target_os = "linux", unix))]
    #[test]
    fn symlinked_store_subdirectory_is_rejected() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().expect("temporary parent");
        let root = temp.path().join("gauntlet");
        std::fs::create_dir(&root).expect("create root");
        let redirect = temp.path().join("redirect");
        std::fs::create_dir(&redirect).expect("create redirect");
        symlink(&redirect, root.join("objects")).expect("create symlink");
        let store = ArtifactStore::new(root);
        let prepared = store
            .prepare("symlink", &sample_object(), BTreeMap::new())
            .expect("prepare artifact");
        assert!(matches!(
            store.persist(&prepared),
            Err(GauntletError::UnsafeStorePath { .. })
        ));
    }

    #[cfg(all(target_os = "linux", unix))]
    #[test]
    fn symlinked_store_ancestor_is_rejected() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().expect("temporary parent");
        let redirect = temp.path().join("redirect");
        std::fs::create_dir(&redirect).expect("create redirect");
        let link = temp.path().join("link");
        symlink(&redirect, &link).expect("create ancestor symlink");
        let store = ArtifactStore::new(link.join("gauntlet"));
        let prepared = store
            .prepare("symlink-ancestor", &sample_object(), BTreeMap::new())
            .expect("prepare artifact");
        assert!(matches!(
            store.persist(&prepared),
            Err(GauntletError::UnsafeStorePath { .. })
        ));
    }
}
