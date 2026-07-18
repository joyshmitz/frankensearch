#[cfg(any(feature = "tantivy-oracle", test))]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;

use asupersync::Cx;
use frankensearch_quill::QuillConfig;
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

use crate::GauntletError;
#[cfg(any(feature = "tantivy-oracle", test))]
use crate::comparator::RankedHit;
use crate::comparator::{
    ComparatorConfig, ComparisonReport, CountState, EngineObservation, NativeTieKey,
    compare_observations,
};
#[cfg(feature = "tantivy-oracle")]
use crate::version_contract::oracle_version_contract;

const MAX_ORACLE_LIMIT: u64 = 100_000;
const MAX_TIE_EXPANSION: u64 = 100_000;
const MAX_ORACLE_FETCH: u64 = 200_000;
pub const TANTIVY_ORACLE_CONFIG_HASH: &str = "shipping-schema-and-parser-v1";

/// Closed engine family used by the cross-engine false-green guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineFamily {
    Quill,
    Tantivy,
}

/// Whether the harness compares separate engines or two variants of one engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonMode {
    CrossEngine,
    InternalDifferential,
}

/// Build identity stamped into every immutable gauntlet object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineDescriptor {
    pub family: EngineFamily,
    pub implementation: String,
    pub crate_version: String,
    pub source_revision: String,
    pub source_dirty: bool,
    pub config_hash: String,
}

impl EngineDescriptor {
    fn implementation_fingerprint(&self) -> (&str, &str, &str, bool, &str) {
        (
            &self.implementation,
            &self.crate_version,
            &self.source_revision,
            self.source_dirty,
            &self.config_hash,
        )
    }
}

/// Subject/oracle pair with mode-specific distinctness validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnginePairIdentity {
    pub comparison_mode: ComparisonMode,
    pub subject: EngineDescriptor,
    pub oracle: EngineDescriptor,
}

impl EnginePairIdentity {
    /// Validate and construct an identity pair before either engine executes.
    ///
    /// # Errors
    ///
    /// Cross-engine comparisons reject the same closed engine family even when
    /// instances or configs differ. Internal differentials allow one family but
    /// require distinct implementation build fingerprints.
    pub fn new(
        comparison_mode: ComparisonMode,
        subject: EngineDescriptor,
        oracle: EngineDescriptor,
    ) -> Result<Self, GauntletError> {
        let invalid = match comparison_mode {
            ComparisonMode::CrossEngine => {
                subject.family != EngineFamily::Quill || oracle.family != EngineFamily::Tantivy
            }
            ComparisonMode::InternalDifferential => {
                subject.family != oracle.family
                    || subject.implementation_fingerprint() == oracle.implementation_fingerprint()
            }
        };
        if invalid {
            return Err(GauntletError::EngineIdentityCollision {
                comparison_mode,
                subject: subject.implementation.clone(),
                oracle: oracle.implementation.clone(),
            });
        }
        Ok(Self {
            comparison_mode,
            subject,
            oracle,
        })
    }
}

/// Engine-neutral query case consumed by both adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DifferentialCase {
    pub fixture_id: String,
    pub query: String,
    pub limit: u64,
    pub tie_expansion_limit: u64,
    pub count_requested: bool,
    pub snippet_max_chars: Option<u64>,
    pub metadata: DifferentialCaseMetadata,
}

/// Deterministic fixture-generation inputs allowed in the object hash basis.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DifferentialCaseMetadata {
    pub generator_id: Option<String>,
    pub generator_seed: Option<u64>,
    pub corpus_hash: Option<String>,
}

impl DifferentialCase {
    #[must_use]
    pub fn new(fixture_id: impl Into<String>, query: impl Into<String>, limit: u64) -> Self {
        Self {
            fixture_id: fixture_id.into(),
            query: query.into(),
            limit,
            tie_expansion_limit: 256,
            count_requested: true,
            snippet_max_chars: Some(200),
            metadata: DifferentialCaseMetadata::default(),
        }
    }

    pub(crate) fn validate_observations(
        &self,
        engines: &EnginePairIdentity,
        subject: &EngineObservation,
        oracle: &EngineObservation,
    ) -> Result<(), GauntletError> {
        self.validate_shape()?;
        let counts_match_request = if self.count_requested {
            matches!(subject.match_count, CountState::Value(_))
                && matches!(oracle.match_count, CountState::Value(_))
        } else {
            subject.match_count == CountState::NotRequested
                && oracle.match_count == CountState::NotRequested
        };
        if !counts_match_request {
            return Err(GauntletError::InvalidObservation {
                reason: "count evidence does not match the differential case request".to_owned(),
            });
        }
        self.validate_observation_shape("subject", subject)?;
        self.validate_observation_shape("oracle", oracle)?;
        validate_observation_family("subject", engines.subject.family, subject)?;
        validate_observation_family("oracle", engines.oracle.family, oracle)?;
        Ok(())
    }

    fn validate_observation_shape(
        &self,
        label: &str,
        observation: &EngineObservation,
    ) -> Result<(), GauntletError> {
        let hit_count = u64::try_from(observation.hits.len()).map_err(|_| {
            GauntletError::InvalidObservation {
                reason: format!("{label} top-k length does not fit u64"),
            }
        })?;
        let cutoff_count = u64::try_from(observation.cutoff_tie_group.len()).map_err(|_| {
            GauntletError::InvalidObservation {
                reason: format!("{label} cutoff tie-group length does not fit u64"),
            }
        })?;
        if hit_count > self.limit
            || hit_count > observation.doc_count
            || cutoff_count > observation.doc_count
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} result lengths exceed the case or document count"),
            });
        }
        if let CountState::Value(match_count) = observation.match_count
            && (match_count > observation.doc_count
                || hit_count != self.limit.min(match_count)
                || cutoff_count > match_count)
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} top-k evidence is inconsistent with its exact count"),
            });
        }
        let observed_ids = observation
            .hits
            .iter()
            .chain(&observation.cutoff_tie_group)
            .map(|hit| hit.doc_id.as_str())
            .collect::<BTreeSet<_>>();
        let observed_id_count =
            u64::try_from(observed_ids.len()).map_err(|_| GauntletError::InvalidObservation {
                reason: format!("{label} observed ID count does not fit u64"),
            })?;
        let exceeds_exact_count = matches!(
            observation.match_count,
            CountState::Value(match_count) if observed_id_count > match_count
        );
        if observed_id_count > observation.doc_count || exceeds_exact_count {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} observed IDs exceed the exact count evidence"),
            });
        }

        let Some(cutoff) = observation.hits.last() else {
            if observation.cutoff_tie_group.is_empty() {
                return Ok(());
            }
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} has cutoff evidence without any top-k hit"),
            });
        };
        if observation.cutoff_tie_group.is_empty() {
            return Ok(());
        }
        let cutoff_score = f32::from_bits(cutoff.score_bits);
        let cutoff_keys = observation
            .cutoff_tie_group
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
            .collect::<BTreeSet<_>>();
        let group_is_exact = observation.cutoff_tie_group.iter().all(|hit| {
            f32::from_bits(hit.score_bits)
                .total_cmp(&cutoff_score)
                .is_eq()
        }) && observation.hits.iter().all(|hit| {
            !f32::from_bits(hit.score_bits)
                .total_cmp(&cutoff_score)
                .is_eq()
                || cutoff_keys.contains(&(hit.doc_id.as_str(), hit.score_bits))
        });
        if !group_is_exact {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} cutoff tie-group does not describe the top-k cutoff"),
            });
        }
        Ok(())
    }

    fn validate_shape(&self) -> Result<(), GauntletError> {
        if self.fixture_id.is_empty() {
            return Err(GauntletError::InvalidCase {
                reason: "fixture ID must not be empty".to_owned(),
            });
        }
        let fetch = self.limit.checked_add(self.tie_expansion_limit);
        if self.limit > MAX_ORACLE_LIMIT
            || self.tie_expansion_limit > MAX_TIE_EXPANSION
            || fetch.is_none_or(|value| value > MAX_ORACLE_FETCH)
        {
            return Err(GauntletError::InvalidCase {
                reason: "top-k and tie expansion exceed the bounded oracle budget".to_owned(),
            });
        }
        Ok(())
    }
}

fn validate_observation_family(
    label: &str,
    family: EngineFamily,
    observation: &EngineObservation,
) -> Result<(), GauntletError> {
    let valid = observation
        .hits
        .iter()
        .chain(&observation.cutoff_tie_group)
        .all(|hit| {
            matches!(
                (family, &hit.native_tie_key),
                (EngineFamily::Quill, NativeTieKey::QuillDocId { .. })
                    | (
                        EngineFamily::Tantivy,
                        NativeTieKey::TantivyDocAddress { .. }
                    )
            )
        });
    if valid {
        Ok(())
    } else {
        Err(GauntletError::InvalidObservation {
            reason: format!("{label} native tie keys do not match its engine family"),
        })
    }
}

/// Future returned by object-safe engine adapters.
pub type GauntletFuture<'a> =
    Pin<Box<dyn Future<Output = Result<EngineObservation, GauntletError>> + Send + 'a>>;

/// Minimal adapter boundary. A real `QuillIndex` can replace the stub unchanged.
pub trait GauntletEngine: Send + Sync {
    fn descriptor(&self) -> EngineDescriptor;

    fn observe<'a>(&'a self, cx: &'a Cx, case: &'a DifferentialCase) -> GauntletFuture<'a>;
}

/// Result of one harness execution before it is encoded as an artifact object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessRun {
    pub engines: EnginePairIdentity,
    pub case: DifferentialCase,
    pub comparator_config: ComparatorConfig,
    pub comparison: ComparisonReport,
}

/// Pure orchestration shell around engine adapters and the comparator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DifferentialHarness {
    comparison_mode: ComparisonMode,
    comparator_config: ComparatorConfig,
}

impl DifferentialHarness {
    #[must_use]
    pub const fn new(comparison_mode: ComparisonMode, comparator_config: ComparatorConfig) -> Self {
        Self {
            comparison_mode,
            comparator_config,
        }
    }

    /// Execute one subject/oracle case.
    ///
    /// Identity validation occurs before either adapter's `observe` method.
    ///
    /// # Errors
    ///
    /// Returns identity, engine, or comparator failures without producing a
    /// false-green report.
    pub async fn run(
        &self,
        cx: &Cx,
        subject: &dyn GauntletEngine,
        oracle: &dyn GauntletEngine,
        case: &DifferentialCase,
    ) -> Result<HarnessRun, GauntletError> {
        let engines = EnginePairIdentity::new(
            self.comparison_mode,
            subject.descriptor(),
            oracle.descriptor(),
        )?;
        self.comparator_config.validate_contract()?;
        case.validate_shape()?;
        let subject_observation = subject.observe(cx, case).await?;
        let oracle_observation = oracle.observe(cx, case).await?;
        case.validate_observations(&engines, &subject_observation, &oracle_observation)?;
        let comparison = compare_observations(
            subject_observation,
            oracle_observation,
            self.comparator_config,
        )?;
        Ok(HarnessRun {
            engines,
            case: case.clone(),
            comparator_config: self.comparator_config,
            comparison,
        })
    }
}

impl Default for DifferentialHarness {
    fn default() -> Self {
        Self::new(ComparisonMode::CrossEngine, ComparatorConfig::default())
    }
}

/// Quill-side placeholder until the first executable `QuillIndex` lands.
#[derive(Debug, Clone)]
pub struct QuillSubjectStub {
    config: QuillConfig,
    descriptor: EngineDescriptor,
}

impl QuillSubjectStub {
    #[must_use]
    pub fn new(config: QuillConfig) -> Self {
        let config_hash = quill_config_hash(&config);
        Self {
            config,
            descriptor: EngineDescriptor {
                family: EngineFamily::Quill,
                implementation: "frankensearch-quill/subject-stub".to_owned(),
                crate_version: env!("CARGO_PKG_VERSION").to_owned(),
                source_revision: "unimplemented-subject-stub".to_owned(),
                source_dirty: false,
                config_hash,
            },
        }
    }

    #[must_use]
    pub const fn config(&self) -> &QuillConfig {
        &self.config
    }
}

impl Default for QuillSubjectStub {
    fn default() -> Self {
        Self::new(QuillConfig::default())
    }
}

impl GauntletEngine for QuillSubjectStub {
    fn descriptor(&self) -> EngineDescriptor {
        self.descriptor.clone()
    }

    fn observe<'a>(&'a self, _cx: &'a Cx, _case: &'a DifferentialCase) -> GauntletFuture<'a> {
        Box::pin(async {
            Err(GauntletError::SubjectUnavailable {
                reason: "QuillIndex has not reached the executable G1 checkpoint".to_owned(),
            })
        })
    }
}

fn quill_config_hash(config: &QuillConfig) -> String {
    let canonical = format!(
        "scribe={};delta={};fanout={};compact={:016x};holes={:016x};glob={};shards={};deterministic={}",
        config.scribe_shard_budget_bytes,
        config.delta_budget_bytes,
        config.tier_fanout,
        config.compaction_tombstone_density.to_bits(),
        config.merge_max_hole_ratio.to_bits(),
        config.glob_expansion_limit,
        config.max_ingest_shards,
        config.deterministic_ingest
    );
    format!("{:016x}", xxh3_64(canonical.as_bytes()))
}

/// Tantivy oracle adapter over the shipping lexical implementation.
#[cfg(feature = "tantivy-oracle")]
pub struct TantivyOracle {
    index: frankensearch_lexical::TantivyIndex,
    descriptor: EngineDescriptor,
}

#[cfg(feature = "tantivy-oracle")]
impl TantivyOracle {
    /// Create an in-memory oracle using the shipping schema/parser.
    ///
    /// # Errors
    ///
    /// Returns an error when the embedded version contract or Tantivy index
    /// cannot be initialized.
    pub fn in_memory(
        observed_lexical_revision: &str,
        source_dirty: bool,
    ) -> Result<Self, GauntletError> {
        Self::from_index(
            frankensearch_lexical::TantivyIndex::in_memory()?,
            observed_lexical_revision,
            source_dirty,
        )
    }

    /// Wrap an existing shipping Tantivy index.
    ///
    /// # Errors
    ///
    /// Returns an error when the committed oracle version contract is invalid.
    pub fn from_index(
        index: frankensearch_lexical::TantivyIndex,
        observed_lexical_revision: &str,
        source_dirty: bool,
    ) -> Result<Self, GauntletError> {
        let contract = oracle_version_contract()?;
        contract.validate_source_state(observed_lexical_revision, source_dirty)?;
        Ok(Self {
            index,
            descriptor: EngineDescriptor {
                family: EngineFamily::Tantivy,
                implementation: "frankensearch-lexical/tantivy-index".to_owned(),
                crate_version: contract.lexical_package_version,
                source_revision: contract.lexical_git_revision,
                source_dirty,
                config_hash: TANTIVY_ORACLE_CONFIG_HASH.to_owned(),
            },
        })
    }

    /// Index and commit a corpus through the shipping lexical trait.
    ///
    /// # Errors
    ///
    /// Propagates lexical indexing or commit failures.
    pub async fn index_documents(
        &self,
        cx: &Cx,
        documents: &[frankensearch_core::IndexableDocument],
    ) -> Result<(), GauntletError> {
        use frankensearch_core::LexicalSearch;

        self.index.index_documents(cx, documents).await?;
        self.index.commit(cx).await?;
        Ok(())
    }

    #[must_use]
    pub const fn index(&self) -> &frankensearch_lexical::TantivyIndex {
        &self.index
    }
}

#[cfg(feature = "tantivy-oracle")]
impl GauntletEngine for TantivyOracle {
    fn descriptor(&self) -> EngineDescriptor {
        self.descriptor.clone()
    }

    fn observe<'a>(&'a self, cx: &'a Cx, case: &'a DifferentialCase) -> GauntletFuture<'a> {
        Box::pin(async move {
            case.validate_shape()?;
            let limit = usize::try_from(case.limit).map_err(|_| GauntletError::InvalidCase {
                reason: "limit does not fit usize".to_owned(),
            })?;
            let tie_expansion_limit = usize::try_from(case.tie_expansion_limit).map_err(|_| {
                GauntletError::InvalidCase {
                    reason: "tie expansion limit does not fit usize".to_owned(),
                }
            })?;
            let mut snippet_config = frankensearch_lexical::SnippetConfig::default();
            if let Some(max_chars) = case.snippet_max_chars {
                snippet_config.max_chars =
                    usize::try_from(max_chars).map_err(|_| GauntletError::InvalidCase {
                        reason: "snippet character limit does not fit usize".to_owned(),
                    })?;
            }
            let observation = self.index.oracle_observe_query(
                cx,
                &case.query,
                limit,
                tie_expansion_limit,
                &snippet_config,
            )?;
            let mut snippets = BTreeMap::new();
            let hits = observation
                .hits
                .into_iter()
                .map(|hit| {
                    if let Some(snippet) = hit.snippet {
                        snippets.insert(hit.doc_id.clone(), snippet);
                    }
                    RankedHit {
                        doc_id: hit.doc_id,
                        score_bits: hit.score_bits,
                        native_tie_key: NativeTieKey::TantivyDocAddress {
                            segment_ord: hit.segment_ord,
                            doc_id: hit.segment_doc_id,
                        },
                    }
                })
                .collect();
            let cutoff_tie_group = observation
                .cutoff_tie_group
                .into_iter()
                .map(|hit| RankedHit {
                    doc_id: hit.doc_id,
                    score_bits: hit.score_bits,
                    native_tie_key: NativeTieKey::TantivyDocAddress {
                        segment_ord: hit.segment_ord,
                        doc_id: hit.segment_doc_id,
                    },
                })
                .collect();
            Ok(EngineObservation {
                hits,
                cutoff_tie_group,
                cutoff_tie_complete: observation.cutoff_tie_complete,
                offset_tie_group: Vec::new(),
                offset_tie_complete: false,
                snippets,
                match_count: if case.count_requested {
                    CountState::Value(u64::try_from(observation.total_count).unwrap_or(u64::MAX))
                } else {
                    CountState::NotRequested
                },
                doc_count: u64::try_from(observation.doc_count).unwrap_or(u64::MAX),
                ast_differences: Vec::new(),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct CountingEngine {
        descriptor: EngineDescriptor,
        observe_calls: Arc<AtomicUsize>,
    }

    impl GauntletEngine for CountingEngine {
        fn descriptor(&self) -> EngineDescriptor {
            self.descriptor.clone()
        }

        fn observe<'a>(&'a self, _cx: &'a Cx, _case: &'a DifferentialCase) -> GauntletFuture<'a> {
            Box::pin(async move {
                self.observe_calls.fetch_add(1, Ordering::Relaxed);
                Err(GauntletError::SubjectUnavailable {
                    reason: "counting test engine executed".to_owned(),
                })
            })
        }
    }

    #[test]
    fn subject_stub_is_a_trait_object_with_quill_identity() {
        let subject: Box<dyn GauntletEngine> = Box::new(QuillSubjectStub::default());
        assert_eq!(subject.descriptor().family, EngineFamily::Quill);
        assert_eq!(subject.descriptor().config_hash.len(), 16);
    }

    #[test]
    fn cross_engine_guard_rejects_family_even_when_configs_differ() {
        let first = EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "tantivy".to_owned(),
            crate_version: "0.26.1".to_owned(),
            source_revision: "a".repeat(40),
            source_dirty: false,
            config_hash: "one".to_owned(),
        };
        let mut second = first.clone();
        second.config_hash = "two".to_owned();
        assert!(matches!(
            EnginePairIdentity::new(ComparisonMode::CrossEngine, first, second),
            Err(GauntletError::EngineIdentityCollision { .. })
        ));
    }

    #[test]
    fn identity_guard_rejects_before_engine_execution() {
        let observe_calls = Arc::new(AtomicUsize::new(0));
        let descriptor = EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "counting-tantivy".to_owned(),
            crate_version: "0.26.1".to_owned(),
            source_revision: "test".to_owned(),
            source_dirty: false,
            config_hash: "one".to_owned(),
        };
        let first = CountingEngine {
            descriptor: descriptor.clone(),
            observe_calls: Arc::clone(&observe_calls),
        };
        let mut second_descriptor = descriptor;
        second_descriptor.config_hash = "two".to_owned();
        let second = CountingEngine {
            descriptor: second_descriptor,
            observe_calls: Arc::clone(&observe_calls),
        };
        let harness = DifferentialHarness::default();
        let case = DifferentialCase::new("identity-preflight", "anything", 10);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(matches!(
                harness.run(&cx, &first, &second, &case).await,
                Err(GauntletError::EngineIdentityCollision { .. })
            ));
        });
        assert_eq!(observe_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn case_rejects_underfilled_and_overfilled_exact_top_k_evidence() {
        let subject_descriptor = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: "quill-test".to_owned(),
            crate_version: "0.2.1".to_owned(),
            source_revision: "test".to_owned(),
            source_dirty: false,
            config_hash: "subject".to_owned(),
        };
        let oracle_descriptor = EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "tantivy-test".to_owned(),
            crate_version: "0.26.1".to_owned(),
            source_revision: "test".to_owned(),
            source_dirty: false,
            config_hash: "oracle".to_owned(),
        };
        let engines = EnginePairIdentity::new(
            ComparisonMode::CrossEngine,
            subject_descriptor,
            oracle_descriptor,
        )
        .expect("distinct engines");
        let underfilled = EngineObservation {
            hits: Vec::new(),
            cutoff_tie_group: Vec::new(),
            cutoff_tie_complete: true,
            offset_tie_group: Vec::new(),
            offset_tie_complete: false,
            snippets: BTreeMap::new(),
            match_count: CountState::Value(2),
            doc_count: 2,
            ast_differences: Vec::new(),
        };
        let case = DifferentialCase::new("underfilled", "query", 10);
        assert!(
            case.validate_observations(&engines, &underfilled, &underfilled)
                .is_err()
        );

        let quill_hit = RankedHit {
            doc_id: "one".to_owned(),
            score_bits: 1.0_f32.to_bits(),
            native_tie_key: NativeTieKey::QuillDocId { doc_id: 1 },
        };
        let subject_overfilled = EngineObservation {
            hits: vec![quill_hit.clone()],
            cutoff_tie_group: vec![quill_hit],
            cutoff_tie_complete: true,
            offset_tie_group: Vec::new(),
            offset_tie_complete: false,
            snippets: BTreeMap::new(),
            match_count: CountState::Value(1),
            doc_count: 1,
            ast_differences: Vec::new(),
        };
        let oracle_empty = EngineObservation {
            hits: Vec::new(),
            cutoff_tie_group: Vec::new(),
            cutoff_tie_complete: true,
            offset_tie_group: Vec::new(),
            offset_tie_complete: false,
            snippets: BTreeMap::new(),
            match_count: CountState::Value(0),
            doc_count: 1,
            ast_differences: Vec::new(),
        };
        let zero_limit = DifferentialCase::new("overfilled", "query", 0);
        assert!(
            zero_limit
                .validate_observations(&engines, &subject_overfilled, &oracle_empty)
                .is_err()
        );
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn separate_tantivy_instances_fail_before_execution() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_git_revision;
        let first = TantivyOracle::in_memory(&revision, false).expect("first oracle");
        let second = TantivyOracle::in_memory(&revision, false).expect("second oracle");
        let harness = DifferentialHarness::default();
        let case = DifferentialCase::new("identity-guard", "anything", 10);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let error = harness
                .run(&cx, &first, &second, &case)
                .await
                .expect_err("oracle-vs-oracle must fail before observation");
            assert!(matches!(
                error,
                GauntletError::EngineIdentityCollision { .. }
            ));
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_observation_retains_full_tie_evidence_and_exact_count() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_git_revision;
        let oracle = TantivyOracle::in_memory(&revision, false).expect("oracle");
        let documents = vec![
            frankensearch_core::IndexableDocument::new("a", "shared token"),
            frankensearch_core::IndexableDocument::new("b", "shared token"),
            frankensearch_core::IndexableDocument::new("c", "shared token"),
        ];
        let mut case = DifferentialCase::new("oracle-observation", "shared", 2);
        case.tie_expansion_limit = 8;
        let mut exhausted_case = case.clone();
        exhausted_case.fixture_id = "oracle-exhausted-tie-expansion".to_owned();
        exhausted_case.tie_expansion_limit = 0;
        let mut zero_limit_case = case.clone();
        zero_limit_case.fixture_id = "oracle-zero-limit-count".to_owned();
        zero_limit_case.limit = 0;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            oracle
                .index_documents(&cx, &documents)
                .await
                .expect("index oracle corpus");
            let observation = oracle.observe(&cx, &case).await.expect("observe query");
            assert_eq!(observation.hits.len(), 2);
            assert_eq!(observation.cutoff_tie_group.len(), 3);
            assert!(observation.cutoff_tie_complete);
            assert_eq!(observation.match_count, CountState::Value(3));
            assert_eq!(observation.doc_count, 3);
            assert!(
                observation.hits.iter().all(|hit| matches!(
                    hit.native_tie_key,
                    NativeTieKey::TantivyDocAddress { .. }
                ))
            );

            let exhausted = oracle
                .observe(&cx, &exhausted_case)
                .await
                .expect("observe exhausted tie expansion");
            assert_eq!(exhausted.hits.len(), 2);
            assert_eq!(exhausted.cutoff_tie_group.len(), 2);
            assert!(!exhausted.cutoff_tie_complete);
            assert_eq!(exhausted.match_count, CountState::Value(3));

            let zero_limit = oracle
                .observe(&cx, &zero_limit_case)
                .await
                .expect("observe zero-limit exact count");
            assert!(zero_limit.hits.is_empty());
            assert!(zero_limit.cutoff_tie_group.is_empty());
            assert!(zero_limit.cutoff_tie_complete);
            assert_eq!(zero_limit.match_count, CountState::Value(3));
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_lower_score_sentinel_completes_cutoff_tie_group() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_git_revision;
        let oracle = TantivyOracle::in_memory(&revision, false).expect("oracle");
        let documents = vec![
            frankensearch_core::IndexableDocument::new("a", "alpha beta"),
            frankensearch_core::IndexableDocument::new("b", "alpha beta"),
            frankensearch_core::IndexableDocument::new("c", "alpha"),
            frankensearch_core::IndexableDocument::new("d", "alpha"),
            frankensearch_core::IndexableDocument::new("e", "alpha"),
        ];
        let mut case = DifferentialCase::new("oracle-lower-score-sentinel", "alpha beta", 1);
        case.tie_expansion_limit = 2;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            oracle
                .index_documents(&cx, &documents)
                .await
                .expect("index oracle corpus");
            let observation = oracle.observe(&cx, &case).await.expect("observe query");
            assert_eq!(observation.hits.len(), 1);
            assert_eq!(observation.cutoff_tie_group.len(), 2);
            assert!(observation.cutoff_tie_complete);
            assert_eq!(observation.match_count, CountState::Value(5));
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn fieldnorm_codec_matches_tantivy_decodes_and_encode_boundaries() {
        use frankensearch_lexical::tantivy_crate::fieldnorm::FieldNormReader;
        use frankensearch_quill::contract::{FIELD_NORMS_TABLE, fieldnorm_to_id, id_to_fieldnorm};

        for id in 0..=u8::MAX {
            assert_eq!(
                id_to_fieldnorm(id),
                FieldNormReader::id_to_fieldnorm(id),
                "decode id={id}"
            );
        }
        // The encoder is constant between consecutive table boundaries, so
        // each boundary and both adjacent values cover every output interval.
        let mut probes = vec![0, u32::MAX];
        for &boundary in &FIELD_NORMS_TABLE {
            probes.push(boundary);
            probes.push(boundary.saturating_sub(1));
            probes.push(boundary.saturating_add(1));
        }
        probes.sort_unstable();
        probes.dedup();
        for length in probes {
            assert_eq!(
                fieldnorm_to_id(length),
                FieldNormReader::fieldnorm_to_id(length),
                "encode length={length}"
            );
        }
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn doclen_and_raw_stats_match_tantivy_before_and_after_delete() {
        use frankensearch_lexical::tantivy_crate::indexer::NoMergePolicy;
        use frankensearch_lexical::tantivy_crate::query::Bm25StatisticsProvider;
        use frankensearch_lexical::tantivy_crate::schema::{STORED, STRING, Schema, TEXT};
        use frankensearch_lexical::tantivy_crate::{Index, Term, doc};
        use frankensearch_quill::contract::id_to_fieldnorm;
        use frankensearch_quill::quiver::{
            DocLenFieldInput, EncodedDocLenSection, EncodedStatsSection, FieldStats,
            aggregate_field_stats,
        };

        fn tokens(count: usize) -> String {
            std::iter::repeat_n("x", count)
                .collect::<Vec<_>>()
                .join(" ")
        }

        assert_eq!(
            oracle_version_contract()
                .expect("version contract")
                .tantivy_version,
            "0.26.1"
        );
        let raw_lengths = [41_u32, 42, 65];
        let mut schema_builder = Schema::builder();
        let id = schema_builder.add_text_field("id", STRING | STORED);
        let content = schema_builder.add_text_field("content", TEXT | STORED);
        let index = Index::create_in_ram(schema_builder.build());
        let mut writer = index
            .writer_with_num_threads(1, 50_000_000)
            .expect("single-segment oracle writer");
        writer.set_merge_policy(Box::new(NoMergePolicy));
        for (document_index, &length) in raw_lengths.iter().enumerate() {
            writer
                .add_document(doc!(
                    id => format!("stats-{document_index}"),
                    content => tokens(usize::try_from(length).unwrap_or(usize::MAX)),
                ))
                .expect("add oracle document");
        }
        writer.commit().expect("commit oracle fixture");
        let reader = index.reader().expect("oracle reader");
        reader.reload().expect("reload committed oracle");
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);

        let oracle_tokens = Bm25StatisticsProvider::total_num_tokens(&searcher, content)
            .expect("oracle token count");
        let oracle_docs =
            Bm25StatisticsProvider::total_num_docs(&searcher).expect("oracle document count");
        assert_eq!(oracle_tokens, 148);
        assert_eq!(oracle_docs, 3);
        let mut oracle_ids = searcher
            .segment_readers()
            .iter()
            .flat_map(|segment| {
                let norms = segment
                    .get_fieldnorms_reader(content)
                    .expect("content fieldnorms");
                (0..segment.max_doc())
                    .map(move |doc| norms.fieldnorm_id(doc))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        oracle_ids.sort_unstable();

        let lengths = raw_lengths.map(Some);
        let inputs = [DocLenFieldInput::new(1, &lengths)];
        let encoded_doclen =
            EncodedDocLenSection::encode(0, 3, &[1], &inputs).expect("Quill DOCLEN");
        let mut quill_ids = encoded_doclen
            .section(&[1])
            .expect("parse Quill DOCLEN")
            .field(1)
            .expect("Quill content field")
            .fieldnorm_ids()
            .to_vec();
        quill_ids.sort_unstable();
        assert_eq!(quill_ids, oracle_ids);

        let segment_stats = searcher
            .segment_readers()
            .iter()
            .map(|segment| {
                let row = [FieldStats::new(
                    1,
                    segment
                        .inverted_index(content)
                        .expect("inverted index")
                        .total_num_tokens(),
                    segment.max_doc(),
                )];
                EncodedStatsSection::encode(&[1], &row, segment.max_doc())
                    .expect("encode segment STATS")
                    .section(&[1])
                    .expect("parse segment STATS")
            })
            .collect::<Vec<_>>();
        let aggregate = aggregate_field_stats(segment_stats.iter())
            .expect("aggregate multi-segment Quill STATS");
        let raw_avgdl = aggregate[0]
            .average_field_length()
            .expect("non-empty average");
        assert_eq!(raw_avgdl.to_bits(), (148.0_f32 / 3.0).to_bits());
        let decoded_avgdl =
            oracle_ids.iter().copied().map(id_to_fieldnorm).sum::<u32>() as f32 / 3.0;
        assert_ne!(raw_avgdl.to_bits(), decoded_avgdl.to_bits());

        drop(searcher);
        writer.delete_term(Term::from_field_text(id, "stats-1"));
        writer.commit().expect("commit oracle delete");
        reader.reload().expect("reload oracle deletion");
        let deleted_searcher = reader.searcher();
        assert_eq!(deleted_searcher.num_docs(), 2);
        assert_eq!(
            Bm25StatisticsProvider::total_num_docs(&deleted_searcher)
                .expect("post-delete oracle document count"),
            3
        );
        assert_eq!(
            Bm25StatisticsProvider::total_num_tokens(&deleted_searcher, content)
                .expect("post-delete oracle token count"),
            148
        );
        let mut post_delete_ids = deleted_searcher
            .segment_readers()
            .iter()
            .flat_map(|segment| {
                let norms = segment
                    .get_fieldnorms_reader(content)
                    .expect("post-delete fieldnorms");
                (0..segment.max_doc())
                    .map(move |doc| norms.fieldnorm_id(doc))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        post_delete_ids.sort_unstable();
        assert_eq!(post_delete_ids, oracle_ids);
        assert!(
            deleted_searcher
                .segment_readers()
                .iter()
                .any(|segment| { (0..segment.max_doc()).any(|doc| segment.is_deleted(doc)) })
        );

        let post_delete_sections = deleted_searcher
            .segment_readers()
            .iter()
            .map(|segment| {
                let row = [FieldStats::new(
                    1,
                    segment
                        .inverted_index(content)
                        .expect("post-delete inverted index")
                        .total_num_tokens(),
                    segment.max_doc(),
                )];
                EncodedStatsSection::encode(&[1], &row, segment.max_doc())
                    .expect("encode post-delete STATS")
                    .section(&[1])
                    .expect("parse post-delete STATS")
            })
            .collect::<Vec<_>>();
        let post_delete = aggregate_field_stats(post_delete_sections.iter())
            .expect("aggregate post-delete STATS");
        assert_eq!(post_delete[0].total_tokens, 148);
        assert_eq!(post_delete[0].doc_count, 3);
        assert_eq!(
            post_delete[0].average_field_length().map(f32::to_bits),
            Some(raw_avgdl.to_bits())
        );
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_constructor_rejects_dirty_or_mismatched_source() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_git_revision;
        assert!(TantivyOracle::in_memory(&revision, true).is_err());
        assert!(TantivyOracle::in_memory(&"0".repeat(40), false).is_err());
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_rejects_oversized_case_before_query_execution() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_git_revision;
        let oracle = TantivyOracle::in_memory(&revision, false).expect("oracle");
        let mut case = DifferentialCase::new("oversized", "anything", MAX_ORACLE_LIMIT + 1);
        case.tie_expansion_limit = 0;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            assert!(matches!(
                oracle.observe(&cx, &case).await,
                Err(GauntletError::InvalidCase { .. })
            ));
        });
    }
}
