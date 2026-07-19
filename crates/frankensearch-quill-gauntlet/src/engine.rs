use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;

use asupersync::Cx;
use frankensearch_quill::{QuillConfig, QuillIndex};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

use crate::GauntletError;
use crate::comparator::{
    ComparatorConfig, ComparisonReport, CountState, EngineObservation, NativeTieKey, RankedHit,
    compare_observations,
};
use crate::generator::MAX_DOCUMENT_ID_BYTES;
#[cfg(feature = "tantivy-oracle")]
use crate::runner::SemanticContract;
use crate::version_contract::oracle_version_contract;

const MAX_ORACLE_LIMIT: u64 = 100_000;
const MAX_TIE_EXPANSION: u64 = 100_000;
const MAX_ORACLE_FETCH: u64 = 200_000;
const MAX_CASE_ID_BYTES: usize = 1_024;
const MAX_CASE_QUERY_BYTES: usize = 1024 * 1024;
const MAX_CASE_METADATA_BYTES: usize = 1_024;
const MAX_AST_DIFFERENCES: usize = 1_024;
const MAX_OBSERVATION_TEXT_BYTES: usize = 1024 * 1024;
const MAX_OBSERVATION_AGGREGATE_TEXT_BYTES: usize = 64 * 1024 * 1024;
/// Maximum snippet budget accepted at every harness and campaign boundary.
pub const MAX_SNIPPET_CHARS: u64 = 1_000_000;
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
    fn validate(&self) -> Result<(), GauntletError> {
        for (label, value) in [
            ("implementation", self.implementation.as_str()),
            ("crate_version", self.crate_version.as_str()),
            ("source_revision", self.source_revision.as_str()),
            ("config_hash", self.config_hash.as_str()),
        ] {
            if value.is_empty()
                || value.len() > 256
                || value.trim() != value
                || value.chars().any(char::is_control)
            {
                return Err(GauntletError::InvalidContract {
                    reason: format!(
                        "engine descriptor {label} must be nonempty, canonical text of at most 256 bytes"
                    ),
                });
            }
        }
        Ok(())
    }

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
    /// Shared campaign semantic contract declared by both adapters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_contract: Option<crate::runner::SemanticContract>,
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
        subject.validate()?;
        oracle.validate()?;
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
            semantic_contract: None,
        })
    }

    pub(crate) fn bind_semantic_contract(
        &mut self,
        semantic_contract: crate::runner::SemanticContract,
    ) -> Result<(), GauntletError> {
        semantic_contract.validate()?;
        self.semantic_contract = Some(semantic_contract);
        Ok(())
    }

    pub(crate) fn validate_gauntlet_contract(&self) -> Result<(), GauntletError> {
        let mut rebuilt = Self::new(
            self.comparison_mode,
            self.subject.clone(),
            self.oracle.clone(),
        )?;
        if let Some(semantic_contract) = &self.semantic_contract {
            rebuilt.bind_semantic_contract(semantic_contract.clone())?;
        }
        if &rebuilt != self {
            return Err(GauntletError::InvalidContract {
                reason: "engine identity is not self-consistent".to_owned(),
            });
        }
        if self.comparison_mode == ComparisonMode::CrossEngine {
            let oracle_version = oracle_version_contract()?;
            if self.oracle.implementation != "frankensearch-lexical/tantivy-index"
                || self.oracle.crate_version != oracle_version.lexical_package_version
                || self.oracle.source_revision != oracle_version.lexical_git_revision
                || self.oracle.config_hash != TANTIVY_ORACLE_CONFIG_HASH
                || self.oracle.source_dirty
            {
                return Err(GauntletError::InvalidContract {
                    reason: "oracle descriptor does not match the lexical version contract"
                        .to_owned(),
                });
            }
        }
        Ok(())
    }
}

/// Engine-neutral query case consumed by both adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DifferentialCase {
    pub fixture_id: String,
    pub query: String,
    pub limit: u64,
    /// Number of ranked matches skipped before the returned page.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub offset: u64,
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
            offset: 0,
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
        let observation_text_is_bounded = observation
            .hits
            .iter()
            .chain(&observation.cutoff_tie_group)
            .chain(&observation.offset_tie_group)
            .all(|hit| hit.doc_id.len() <= MAX_DOCUMENT_ID_BYTES)
            && observation.ast_differences.len() <= MAX_AST_DIFFERENCES
            && observation.ast_differences.iter().all(|difference| {
                difference.oracle.len() <= MAX_OBSERVATION_TEXT_BYTES
                    && difference.subject.len() <= MAX_OBSERVATION_TEXT_BYTES
            });
        let aggregate_text_bytes = observation
            .hits
            .iter()
            .chain(&observation.cutoff_tie_group)
            .chain(&observation.offset_tie_group)
            .map(|hit| hit.doc_id.len())
            .chain(
                observation
                    .snippets
                    .iter()
                    .map(|(doc_id, snippet)| doc_id.len().saturating_add(snippet.len())),
            )
            .chain(observation.ast_differences.iter().map(|difference| {
                difference
                    .oracle
                    .len()
                    .saturating_add(difference.subject.len())
            }))
            .try_fold(0_usize, usize::checked_add);
        let snippets_are_bounded = observation.snippets.iter().all(|(doc_id, snippet)| {
            doc_id.len() <= MAX_DOCUMENT_ID_BYTES && snippet.len() <= MAX_OBSERVATION_TEXT_BYTES
        });
        if !observation_text_is_bounded
            || !snippets_are_bounded
            || aggregate_text_bytes.is_none_or(|bytes| bytes > MAX_OBSERVATION_AGGREGATE_TEXT_BYTES)
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} contains oversized result evidence"),
            });
        }
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
        let offset_count = u64::try_from(observation.offset_tie_group.len()).map_err(|_| {
            GauntletError::InvalidObservation {
                reason: format!("{label} offset tie-group length does not fit u64"),
            }
        })?;
        let evidence_budget = self
            .offset
            .checked_add(self.limit)
            .and_then(|value| value.checked_add(self.tie_expansion_limit))
            .ok_or_else(|| GauntletError::InvalidObservation {
                reason: format!("{label} evidence budget overflowed"),
            })?;
        if hit_count > self.limit
            || hit_count > observation.doc_count
            || cutoff_count > observation.doc_count
            || offset_count > observation.doc_count
            || cutoff_count > evidence_budget
            || offset_count > evidence_budget
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} result lengths exceed the case or document count"),
            });
        }
        if let CountState::Value(match_count) = observation.match_count
            && (match_count > observation.doc_count
                || hit_count != self.limit.min(match_count.saturating_sub(self.offset))
                || cutoff_count > match_count
                || offset_count > match_count)
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} top-k evidence is inconsistent with its exact count"),
            });
        }
        let observed_ids = observation
            .hits
            .iter()
            .chain(&observation.cutoff_tie_group)
            .chain(&observation.offset_tie_group)
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
            if observation.cutoff_tie_group.is_empty() && observation.offset_tie_group.is_empty() {
                return Ok(());
            }
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label} has cutoff evidence without any top-k hit"),
            });
        };
        if !observation.cutoff_tie_group.is_empty() {
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
        }
        if !observation.offset_tie_group.is_empty() {
            if self.offset == 0 {
                return Err(GauntletError::InvalidObservation {
                    reason: format!("{label} has offset tie-group evidence for a zero-offset case"),
                });
            }
            let leading = &observation.hits[0];
            let leading_score = f32::from_bits(leading.score_bits);
            let leading_keys = observation
                .offset_tie_group
                .iter()
                .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
                .collect::<BTreeSet<_>>();
            let leading_group_is_exact = observation.offset_tie_group.iter().all(|hit| {
                f32::from_bits(hit.score_bits)
                    .total_cmp(&leading_score)
                    .is_eq()
            }) && observation.hits.iter().all(|hit| {
                !f32::from_bits(hit.score_bits)
                    .total_cmp(&leading_score)
                    .is_eq()
                    || leading_keys.contains(&(hit.doc_id.as_str(), hit.score_bits))
            });
            let page_ids = observation
                .hits
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<BTreeSet<_>>();
            let proves_skipped_member = observation
                .offset_tie_group
                .iter()
                .any(|hit| !page_ids.contains(hit.doc_id.as_str()));
            if !leading_group_is_exact || !proves_skipped_member {
                return Err(GauntletError::InvalidObservation {
                    reason: format!(
                        "{label} offset tie-group does not describe the page's leading boundary"
                    ),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn validate_shape(&self) -> Result<(), GauntletError> {
        let metadata_is_bounded = [
            self.metadata.generator_id.as_deref(),
            self.metadata.corpus_hash.as_deref(),
        ]
        .into_iter()
        .flatten()
        .all(|value| value.len() <= MAX_CASE_METADATA_BYTES);
        if self.fixture_id.is_empty()
            || self.fixture_id.len() > MAX_CASE_ID_BYTES
            || self.query.len() > MAX_CASE_QUERY_BYTES
            || !metadata_is_bounded
        {
            return Err(GauntletError::InvalidCase {
                reason: "fixture ID, query, or metadata exceed the bounded case contract"
                    .to_owned(),
            });
        }
        let fetch = self
            .offset
            .checked_add(self.limit)
            .and_then(|value| value.checked_add(self.tie_expansion_limit));
        if self.limit > MAX_ORACLE_LIMIT
            || self.offset > MAX_ORACLE_LIMIT
            || self.tie_expansion_limit > MAX_TIE_EXPANSION
            || self
                .snippet_max_chars
                .is_some_and(|value| value > MAX_SNIPPET_CHARS)
            || fetch.is_none_or(|value| value > MAX_ORACLE_FETCH)
        {
            return Err(GauntletError::InvalidCase {
                reason: "top-k, tie expansion, or snippets exceed the bounded oracle budget"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde skip_serializing_if protocol
const fn is_zero(value: &u64) -> bool {
    *value == 0
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
        .chain(&observation.offset_tie_group)
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

/// Live scalar Quill subject used by the G1a campaign.
pub struct QuillSubject {
    config: QuillConfig,
    descriptor: EngineDescriptor,
    index: Option<QuillIndex>,
    state: QuillCampaignState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QuillCampaignState {
    Fresh,
    Ingesting,
    Committed,
    Aborted,
}

impl QuillSubject {
    /// Construct a fresh owned-buffer scalar subject.
    ///
    /// # Errors
    ///
    /// Returns a typed Quill configuration/schema failure or invalid engine
    /// descriptor input.
    pub fn in_memory(
        config: QuillConfig,
        source_revision: impl Into<String>,
        source_dirty: bool,
    ) -> Result<Self, GauntletError> {
        let config_hash = quill_config_hash(&config);
        let descriptor = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: "frankensearch-quill/scalar-index".to_owned(),
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            source_revision: source_revision.into(),
            source_dirty,
            config_hash,
        };
        descriptor.validate()?;
        Ok(Self {
            index: Some(QuillIndex::in_memory(config.clone())?),
            config,
            descriptor,
            state: QuillCampaignState::Fresh,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &QuillConfig {
        &self.config
    }

    pub(crate) fn index(&self) -> Result<&QuillIndex, GauntletError> {
        self.index
            .as_ref()
            .ok_or_else(|| GauntletError::SubjectUnavailable {
                reason: "Quill campaign subject was aborted".to_owned(),
            })
    }

    pub(crate) fn index_mut(&mut self) -> Result<&mut QuillIndex, GauntletError> {
        self.index
            .as_mut()
            .ok_or_else(|| GauntletError::SubjectUnavailable {
                reason: "Quill campaign subject was aborted".to_owned(),
            })
    }

    pub(crate) fn claim_fresh_campaign(&mut self) -> Result<(), GauntletError> {
        if self.state != QuillCampaignState::Fresh {
            return Err(GauntletError::InvalidCampaign {
                reason: "Quill subject may execute only one campaign".to_owned(),
            });
        }
        self.state = QuillCampaignState::Ingesting;
        Ok(())
    }

    pub(crate) fn require_ingesting(&self) -> Result<(), GauntletError> {
        if self.state != QuillCampaignState::Ingesting {
            return Err(GauntletError::InvalidCampaign {
                reason: "Quill indexing and commit require an active ingest session".to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn mark_committed(&mut self) -> Result<(), GauntletError> {
        self.require_ingesting()?;
        self.state = QuillCampaignState::Committed;
        Ok(())
    }

    pub(crate) fn require_committed(&self) -> Result<(), GauntletError> {
        if self.state != QuillCampaignState::Committed {
            return Err(GauntletError::InvalidCampaign {
                reason: "Quill observation requires a committed campaign snapshot".to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn abort(&mut self) {
        self.state = QuillCampaignState::Aborted;
        self.index = None;
    }
}

impl GauntletEngine for QuillSubject {
    fn descriptor(&self) -> EngineDescriptor {
        self.descriptor.clone()
    }

    fn observe<'a>(&'a self, cx: &'a Cx, case: &'a DifferentialCase) -> GauntletFuture<'a> {
        Box::pin(async move {
            self.require_committed()?;
            quill_observe(self.index()?, cx, case)
        })
    }
}

fn quill_observe(
    index: &QuillIndex,
    cx: &Cx,
    case: &DifferentialCase,
) -> Result<EngineObservation, GauntletError> {
    case.validate_shape()?;
    if case.snippet_max_chars.is_some() {
        return Err(GauntletError::InvalidCase {
            reason: "the scalar G1a Quill adapter requires snippets to be disabled".to_owned(),
        });
    }
    let limit = usize::try_from(case.limit).map_err(|_| GauntletError::InvalidCase {
        reason: "limit does not fit usize".to_owned(),
    })?;
    let offset = usize::try_from(case.offset).map_err(|_| GauntletError::InvalidCase {
        reason: "offset does not fit usize".to_owned(),
    })?;
    let tie_expansion =
        usize::try_from(case.tie_expansion_limit).map_err(|_| GauntletError::InvalidCase {
            reason: "tie expansion limit does not fit usize".to_owned(),
        })?;
    let page_end = offset
        .checked_add(limit)
        .ok_or_else(|| GauntletError::InvalidCase {
            reason: "offset plus limit does not fit usize".to_owned(),
        })?;
    let fetch_limit =
        page_end
            .checked_add(tie_expansion)
            .ok_or_else(|| GauntletError::InvalidCase {
                reason: "expanded Quill observation window does not fit usize".to_owned(),
            })?;
    // The first call is the collector mode under test. Keep the expanded
    // exact-count call separate: it exists only to furnish comparator tie
    // evidence and must never stand in for pagination or count-free execution.
    let observed = index.search_paginated(cx, &case.query, limit, offset, case.count_requested)?;
    let evidence = index.search_paginated(cx, &case.query, fetch_limit, 0, true)?;
    if observed.doc_count != evidence.doc_count {
        return Err(GauntletError::InvalidObservation {
            reason: "Quill collector modes disagreed on the committed document count".to_owned(),
        });
    }
    let total_count = evidence
        .total_count
        .ok_or_else(|| GauntletError::InvalidObservation {
            reason: "Quill tie-evidence observation omitted its exact count".to_owned(),
        })?;
    let match_count = match (case.count_requested, observed.total_count) {
        (true, Some(observed_count)) if observed_count == total_count => {
            CountState::Value(observed_count)
        }
        (true, Some(_)) => {
            return Err(GauntletError::InvalidObservation {
                reason: "Quill counted page disagreed with its expanded tie evidence".to_owned(),
            });
        }
        (true, None) => {
            return Err(GauntletError::InvalidObservation {
                reason: "Quill counted page omitted its exact count".to_owned(),
            });
        }
        (false, None) => CountState::NotRequested,
        (false, Some(_)) => {
            return Err(GauntletError::InvalidObservation {
                reason: "Quill count-free page unexpectedly executed exact-count work".to_owned(),
            });
        }
    };
    let ranked = evidence
        .hits
        .iter()
        .map(|hit| RankedHit {
            doc_id: hit.document_id.clone(),
            score_bits: hit.score.to_bits(),
            native_tie_key: NativeTieKey::QuillDocId {
                doc_id: hit.global_docid,
            },
        })
        .collect::<Vec<_>>();
    let top_len = page_end.min(ranked.len());
    let page_window = &ranked[..top_len];
    let (cutoff_tie_group, cutoff_tie_complete) =
        cutoff_tie_group(&ranked, top_len, total_count, limit > 0 && top_len > offset);
    let (offset_tie_group, offset_tie_complete) = if limit == 0 {
        (Vec::new(), false)
    } else {
        offset_tie_group(
            page_window,
            offset,
            total_count,
            &cutoff_tie_group,
            cutoff_tie_complete,
        )
    };
    let hits = observed
        .hits
        .iter()
        .map(|hit| RankedHit {
            doc_id: hit.document_id.clone(),
            score_bits: hit.score.to_bits(),
            native_tie_key: NativeTieKey::QuillDocId {
                doc_id: hit.global_docid,
            },
        })
        .collect();
    Ok(EngineObservation {
        hits,
        cutoff_tie_group,
        cutoff_tie_complete,
        offset_tie_group,
        offset_tie_complete,
        snippets: BTreeMap::new(),
        match_count,
        doc_count: observed.doc_count,
        ast_differences: Vec::new(),
    })
}

fn cutoff_tie_group(
    hits: &[RankedHit],
    boundary: usize,
    total_count: u64,
    relevant: bool,
) -> (Vec<RankedHit>, bool) {
    if !relevant || boundary == 0 || boundary > hits.len() {
        return (Vec::new(), false);
    }
    let score_bits = hits[boundary - 1].score_bits;
    let group = hits
        .iter()
        .filter(|hit| hit.score_bits == score_bits)
        .cloned()
        .collect::<Vec<_>>();
    let complete = u64::try_from(hits.len()).is_ok_and(|fetched| fetched >= total_count)
        || hits
            .last()
            .is_some_and(|last| last.score_bits != score_bits);
    (group, complete)
}

fn offset_tie_group(
    hits: &[RankedHit],
    offset: usize,
    total_count: u64,
    cutoff_group: &[RankedHit],
    cutoff_complete: bool,
) -> (Vec<RankedHit>, bool) {
    if offset == 0 || offset >= hits.len() {
        return (Vec::new(), false);
    }
    let previous = &hits[offset - 1];
    let leading = &hits[offset];
    if previous.score_bits != leading.score_bits {
        return (Vec::new(), false);
    }
    if cutoff_group
        .first()
        .is_some_and(|hit| hit.score_bits == leading.score_bits)
    {
        return (cutoff_group.to_vec(), cutoff_complete);
    }
    let group = hits
        .iter()
        .filter(|hit| hit.score_bits == leading.score_bits)
        .cloned()
        .collect::<Vec<_>>();
    let complete = hits
        .iter()
        .skip(offset + 1)
        .any(|hit| hit.score_bits != leading.score_bits)
        || u64::try_from(hits.len()).is_ok_and(|fetched| fetched >= total_count);
    (group, complete)
}

fn quill_config_hash(config: &QuillConfig) -> String {
    let canonical = format!(
        "scribe={};delta={};fanout={};compact={:016x};holes={:016x};glob={};shards={};deterministic={};visibility_ms={}",
        config.scribe_shard_budget_bytes,
        config.delta_budget_bytes,
        config.tier_fanout,
        config.compaction_tombstone_density.to_bits(),
        config.merge_max_hole_ratio.to_bits(),
        config.glob_expansion_limit,
        config.max_ingest_shards,
        config.deterministic_ingest,
        config.max_visibility_lag_ms
    );
    format!("{:016x}", xxh3_64(canonical.as_bytes()))
}

/// Tantivy oracle adapter over the shipping lexical implementation.
#[cfg(feature = "tantivy-oracle")]
pub struct TantivyOracle {
    index: frankensearch_lexical::TantivyIndex,
    descriptor: EngineDescriptor,
    semantic_contract: SemanticContract,
    campaign_freshness_verified: bool,
    campaign_state: TantivyCampaignState,
}

#[cfg(feature = "tantivy-oracle")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TantivyCampaignState {
    Fresh,
    Ingesting,
    Committed,
    Aborted,
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
        Self::from_index_with_campaign_freshness(
            frankensearch_lexical::TantivyIndex::in_memory()?,
            observed_lexical_revision,
            source_dirty,
            true,
            SemanticContract::shipping_default(),
        )
    }

    /// Create a fresh in-memory oracle for the snippet-free scalar G1a profile.
    ///
    /// # Errors
    ///
    /// Returns the same provenance or index-construction errors as [`Self::in_memory`].
    pub fn in_memory_scalar_g1a(
        observed_lexical_revision: &str,
        source_dirty: bool,
    ) -> Result<Self, GauntletError> {
        Self::from_index_with_campaign_freshness(
            frankensearch_lexical::TantivyIndex::in_memory_single_threaded_oracle()?,
            observed_lexical_revision,
            source_dirty,
            true,
            SemanticContract::scalar_g1a(),
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
        Self::from_index_with_campaign_freshness(
            index,
            observed_lexical_revision,
            source_dirty,
            false,
            SemanticContract::shipping_default(),
        )
    }

    fn from_index_with_campaign_freshness(
        index: frankensearch_lexical::TantivyIndex,
        observed_lexical_revision: &str,
        source_dirty: bool,
        campaign_freshness_verified: bool,
        semantic_contract: SemanticContract,
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
            semantic_contract,
            campaign_freshness_verified,
            campaign_state: TantivyCampaignState::Fresh,
        })
    }

    pub(crate) const fn campaign_semantic_contract(&self) -> &SemanticContract {
        &self.semantic_contract
    }

    pub(crate) fn claim_fresh_campaign(&mut self) -> Result<(), GauntletError> {
        if !self.campaign_freshness_verified {
            return Err(GauntletError::InvalidContract {
                reason: "Tantivy campaigns require a newly constructed one-shot oracle".to_owned(),
            });
        }
        if self.campaign_state != TantivyCampaignState::Fresh {
            return Err(GauntletError::InvalidCampaign {
                reason: "Tantivy oracle may execute only one campaign".to_owned(),
            });
        }
        self.campaign_state = TantivyCampaignState::Ingesting;
        Ok(())
    }

    pub(crate) fn require_ingesting(&self) -> Result<(), GauntletError> {
        if self.campaign_state != TantivyCampaignState::Ingesting {
            return Err(GauntletError::InvalidCampaign {
                reason: "Tantivy indexing and commit require an active ingest session".to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn mark_committed(&mut self) -> Result<(), GauntletError> {
        self.require_ingesting()?;
        self.campaign_state = TantivyCampaignState::Committed;
        Ok(())
    }

    pub(crate) fn require_committed(&self) -> Result<(), GauntletError> {
        if self.campaign_state != TantivyCampaignState::Committed {
            return Err(GauntletError::InvalidCampaign {
                reason: "Tantivy observation requires a committed campaign snapshot".to_owned(),
            });
        }
        Ok(())
    }

    pub(crate) fn abort_campaign(&mut self) {
        self.campaign_state = TantivyCampaignState::Aborted;
        self.campaign_freshness_verified = false;
    }

    /// Index and commit a corpus through the shipping lexical trait.
    ///
    /// # Errors
    ///
    /// Propagates lexical indexing or commit failures.
    pub async fn index_documents(
        &mut self,
        cx: &Cx,
        documents: &[frankensearch_core::IndexableDocument],
    ) -> Result<(), GauntletError> {
        use frankensearch_core::LexicalSearch;

        self.campaign_freshness_verified = false;
        self.index.index_documents(cx, documents).await?;
        self.index.commit(cx).await?;
        Ok(())
    }

    #[must_use]
    pub(crate) const fn index(&self) -> &frankensearch_lexical::TantivyIndex {
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
            let offset = usize::try_from(case.offset).map_err(|_| GauntletError::InvalidCase {
                reason: "offset does not fit usize".to_owned(),
            })?;
            let fetch_limit =
                offset
                    .checked_add(limit)
                    .ok_or_else(|| GauntletError::InvalidCase {
                        reason: "offset plus limit does not fit usize".to_owned(),
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
                fetch_limit,
                tie_expansion_limit,
                &snippet_config,
            )?;
            let (offset_tie_group, offset_tie_complete) = if offset > 0
                && offset < observation.hits.len()
                && observation
                    .hits
                    .get(offset - 1)
                    .zip(observation.hits.get(offset))
                    .is_some_and(|(previous, first)| {
                        f32::from_bits(previous.score_bits)
                            .total_cmp(&f32::from_bits(first.score_bits))
                            .is_eq()
                    }) {
                let leading_bits = observation.hits[offset].score_bits;
                let same_leading_score = |score_bits| {
                    f32::from_bits(score_bits)
                        .total_cmp(&f32::from_bits(leading_bits))
                        .is_eq()
                };
                let cutoff_is_leading = observation
                    .cutoff_tie_group
                    .first()
                    .is_some_and(|hit| same_leading_score(hit.score_bits));
                if cutoff_is_leading {
                    (
                        observation.cutoff_tie_group.clone(),
                        observation.cutoff_tie_complete,
                    )
                } else {
                    let group = observation
                        .hits
                        .iter()
                        .filter(|hit| same_leading_score(hit.score_bits))
                        .cloned()
                        .collect::<Vec<_>>();
                    let complete = observation
                        .hits
                        .iter()
                        .skip(offset + 1)
                        .any(|hit| !same_leading_score(hit.score_bits))
                        || observation.total_count <= observation.hits.len();
                    (group, complete)
                }
            } else {
                (Vec::new(), false)
            };
            let mut snippets = BTreeMap::new();
            let hits: Vec<RankedHit> = observation
                .hits
                .into_iter()
                .skip(offset)
                .take(limit)
                .map(|hit| {
                    if case.snippet_max_chars.is_some()
                        && let Some(snippet) = hit.snippet
                    {
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
            let cutoff_tie_group = if hits.is_empty() {
                Vec::new()
            } else {
                observation
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
                    .collect()
            };
            let offset_tie_group = offset_tie_group
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
                offset_tie_group,
                offset_tie_complete,
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
    fn live_subject_is_a_trait_object_with_quill_identity() {
        let subject: Box<dyn GauntletEngine> = Box::new(
            QuillSubject::in_memory(QuillConfig::default(), "test-revision", false)
                .expect("live Quill subject"),
        );
        assert_eq!(subject.descriptor().family, EngineFamily::Quill);
        assert_eq!(subject.descriptor().config_hash.len(), 16);
    }

    #[test]
    fn quill_config_hash_covers_every_public_knob() {
        let baseline_config = QuillConfig::default();
        let baseline_hash = quill_config_hash(&baseline_config);
        let variants = [
            (
                "scribe_shard_budget_bytes",
                QuillConfig {
                    scribe_shard_budget_bytes: baseline_config.scribe_shard_budget_bytes + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "delta_budget_bytes",
                QuillConfig {
                    delta_budget_bytes: baseline_config.delta_budget_bytes + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "tier_fanout",
                QuillConfig {
                    tier_fanout: baseline_config.tier_fanout + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "compaction_tombstone_density",
                QuillConfig {
                    compaction_tombstone_density: 0.21,
                    ..baseline_config.clone()
                },
            ),
            (
                "merge_max_hole_ratio",
                QuillConfig {
                    merge_max_hole_ratio: 0.51,
                    ..baseline_config.clone()
                },
            ),
            (
                "glob_expansion_limit",
                QuillConfig {
                    glob_expansion_limit: baseline_config.glob_expansion_limit + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "max_ingest_shards",
                QuillConfig {
                    max_ingest_shards: baseline_config.max_ingest_shards + 1,
                    ..baseline_config.clone()
                },
            ),
            (
                "deterministic_ingest",
                QuillConfig {
                    deterministic_ingest: !baseline_config.deterministic_ingest,
                    ..baseline_config.clone()
                },
            ),
            (
                "max_visibility_lag_ms",
                QuillConfig {
                    max_visibility_lag_ms: baseline_config.max_visibility_lag_ms + 1,
                    ..baseline_config.clone()
                },
            ),
        ];

        let mut observed_hashes = BTreeSet::from([baseline_hash.clone()]);
        for (field, variant) in variants {
            let variant_hash = quill_config_hash(&variant);
            assert_ne!(variant_hash, baseline_hash, "hash omitted {field}");
            assert!(
                observed_hashes.insert(variant_hash),
                "hash collision while mutating {field}"
            );
        }
    }

    #[test]
    fn case_shape_rejects_snippet_budget_at_every_entry_point() {
        let mut case = DifferentialCase::new("snippet-budget", "anything", 1);
        case.snippet_max_chars = Some(MAX_SNIPPET_CHARS + 1);
        assert!(matches!(
            case.validate_shape(),
            Err(GauntletError::InvalidCase { .. })
        ));
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
    fn identity_guard_rejects_empty_subject_provenance() {
        let subject = EngineDescriptor {
            family: EngineFamily::Quill,
            implementation: String::new(),
            crate_version: "0.2.1".to_owned(),
            source_revision: "subject-revision".to_owned(),
            source_dirty: false,
            config_hash: "subject-config".to_owned(),
        };
        let oracle = EngineDescriptor {
            family: EngineFamily::Tantivy,
            implementation: "tantivy".to_owned(),
            crate_version: "0.26.1".to_owned(),
            source_revision: "oracle-revision".to_owned(),
            source_dirty: false,
            config_hash: "oracle-config".to_owned(),
        };
        assert!(matches!(
            EnginePairIdentity::new(ComparisonMode::CrossEngine, subject, oracle),
            Err(GauntletError::InvalidContract { .. })
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

        let malformed_offset = EngineObservation {
            hits: vec![RankedHit {
                doc_id: "page".to_owned(),
                score_bits: 1.0_f32.to_bits(),
                native_tie_key: NativeTieKey::QuillDocId { doc_id: 2 },
            }],
            cutoff_tie_group: Vec::new(),
            cutoff_tie_complete: true,
            offset_tie_group: vec![RankedHit {
                doc_id: "prefix".to_owned(),
                score_bits: 2.0_f32.to_bits(),
                native_tie_key: NativeTieKey::QuillDocId { doc_id: 1 },
            }],
            offset_tie_complete: true,
            snippets: BTreeMap::new(),
            match_count: CountState::Value(2),
            doc_count: 2,
            ast_differences: Vec::new(),
        };
        let mut paginated = DifferentialCase::new("malformed-offset", "query", 1);
        paginated.offset = 1;
        assert!(
            paginated
                .validate_observations(&engines, &malformed_offset, &malformed_offset)
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
        let mut oracle = TantivyOracle::in_memory(&revision, false).expect("oracle");
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
        let mut paginated_case = case.clone();
        paginated_case.fixture_id = "oracle-offset-tie".to_owned();
        paginated_case.offset = 1;
        paginated_case.limit = 1;

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

            let paginated = oracle
                .observe(&cx, &paginated_case)
                .await
                .expect("observe offset inside tie");
            assert_eq!(paginated.hits.len(), 1);
            assert_eq!(paginated.offset_tie_group.len(), 3);
            assert!(paginated.offset_tie_complete);
            assert_eq!(paginated.match_count, CountState::Value(3));
        });
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn oracle_lower_score_sentinel_completes_cutoff_tie_group() {
        let revision = oracle_version_contract()
            .expect("version contract")
            .lexical_git_revision;
        let mut oracle = TantivyOracle::in_memory(&revision, false).expect("oracle");
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
