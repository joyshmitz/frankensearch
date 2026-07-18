#![forbid(unsafe_code)]
//! Dev-only conformance harness for Quill and the pinned Tantivy oracle.
//!
//! This crate is a workspace member but intentionally excluded from
//! `default-members` and cannot be published. Shipping crates must never depend
//! on it. The current G0 milestone provides engine identity guards, pure
//! comparators, immutable content-addressed artifacts, pending Q1 fixtures, and
//! an executable same-engine Quiver codec differential, and deterministic,
//! content-addressed E6 corpus/query generators. The Quill search subject
//! remains an explicit non-executable stub until its adapter lands.

mod artifact;
mod comparator;
mod engine;
mod generator;
mod version_contract;

use std::path::PathBuf;

use thiserror::Error;

pub use artifact::{
    ArtifactObject, ArtifactStore, CANONICALIZATION_VERSION, OBJECT_SCHEMA_VERSION,
    PreparedArtifact, RunManifest,
};
pub use comparator::{
    ComparatorConfig, ComparisonReport, ComparisonStatus, CountState, Divergence, DivergenceClass,
    EngineObservation, NativeTieKey, RankClass, RankedHit, SCORE_EPSILON, ScoreEpsilonReason,
    compare_observations,
};
#[cfg(feature = "tantivy-oracle")]
pub use engine::TantivyOracle;
pub use engine::{
    ComparisonMode, DifferentialCase, DifferentialCaseMetadata, DifferentialHarness,
    EngineDescriptor, EngineFamily, EnginePairIdentity, GauntletEngine, GauntletFuture, HarnessRun,
    QuillSubjectStub,
};
pub use generator::{
    CORE_RELEVANCE_DOCUMENT_COUNT, CassDocumentFields, CorpusManifest, CorpusSourceManifest,
    FULL_SHARED_DOCUMENT_COUNT, GENERATOR_ID, GENERATOR_SCHEMA_VERSION, GeneratedDocument,
    GeneratedQueryCase, GeneratedQueryFilters, GeneratedQueryKind, GeneratedQuerySuite,
    GlobPatternClass, HarvestedContractQuery, MAX_DOCUMENT_BYTES, Pathology, QueryGeneratorSpec,
    QueryManifest, QuerySyntax, RangeClass, RepositoryEntry, RepositoryFileDigest,
    RepositorySkipReason, RepositorySnapshot, SharedCorpusView, SharedEdgeCase, SharedFixtureSuite,
    SharedRelevanceQuery, SkippedRepositoryEntry, SourceFileDigest, SyntheticCorpus,
    SyntheticCorpusIter, SyntheticCorpusSpec, UnicodeLane, XLARGE_DOCUMENT_COUNT, ZipfExponent,
};
pub use version_contract::{
    InternalDifferentialFixture, OracleVersionContract, Q1FixtureCatalog, Q1FixtureStub,
    oracle_version_contract, q1_fixture_catalog,
};

/// Typed failure surface for harness setup, execution, comparison, and storage.
#[derive(Debug, Error)]
pub enum GauntletError {
    #[error("engine identity collision in {comparison_mode:?}: {subject} vs {oracle}")]
    EngineIdentityCollision {
        comparison_mode: ComparisonMode,
        subject: String,
        oracle: String,
    },
    #[error("invalid comparator configuration: {reason}")]
    InvalidComparatorConfig { reason: String },
    #[error("invalid engine observation: {reason}")]
    InvalidObservation { reason: String },
    #[error("invalid differential case: {reason}")]
    InvalidCase { reason: String },
    #[error("invalid deterministic generator input: {reason}")]
    InvalidGenerator { reason: String },
    #[error("content-addressed replay mismatch: {reason}")]
    ManifestMismatch { reason: String },
    #[error("subject is unavailable: {reason}")]
    SubjectUnavailable { reason: String },
    #[error("invalid committed contract: {reason}")]
    InvalidContract { reason: String },
    #[error("invalid run ID {run_id:?}")]
    InvalidRunId { run_id: String },
    #[error("invalid prepared artifact: {reason}")]
    InvalidPreparedArtifact { reason: String },
    #[error("unsafe gauntlet store path: {path}")]
    UnsafeStorePath { path: PathBuf },
    #[error("content-address collision at {path}")]
    ArtifactCollision { path: PathBuf },
    #[error("run manifest already points at different content: {path}")]
    RunManifestConflict { path: PathBuf },
    #[error(transparent)]
    Search(#[from] frankensearch_core::SearchError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
