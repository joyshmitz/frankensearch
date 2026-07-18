#![forbid(unsafe_code)]
//! Quill is frankensearch's native, deterministic lexical engine.
//!
//! This crate owns the schema and scoring contracts shared by the ingest,
//! storage, query, and lifecycle subsystems. The current milestone establishes
//! those contracts; later milestones fill in each subsystem behind them.
//!
//! ```
//! use frankensearch_quill::{DEFAULT_SCHEMA, QuillConfig};
//!
//! let config = QuillConfig::default();
//! assert_eq!(config.tier_fanout, 8);
//! assert_ne!(DEFAULT_SCHEMA.schema_id().unwrap(), 0);
//! ```

pub mod argus;
pub mod config;
pub mod contract;
pub mod error;
pub mod grimoire;
pub mod keeper;
pub mod query;
pub mod quiver;
pub mod schema;
pub mod scribe;
pub mod segment;
pub mod stats;
pub mod tracing_conventions;

pub use config::QuillConfig;
pub use error::{QuillError, map_lock_error};
pub use grimoire::{
    ByteSpan, EncodedTermDictionary, MAX_COMPOSITE_KEY_BYTES, MAX_TERM_BYTES, OwnedTerm,
    TERM_BLOCK_TARGET_BYTES, TERM_RESTART_INTERVAL, TermCursor, TermDictionary,
    TermDictionaryError, TermDictionaryLimits, TermInput, TermMatch, TermMetadata, TermRef,
    TermScratch, TermSectionLengths,
};
pub use keeper::{
    CURRENT_ENGINE_VERSION, DEFAULT_GARBAGE_GRACE, EMPTY_TOMBSTONES, GarbageCollectionOptions,
    GarbageCollectionReport, KeeperError, KeeperSnapshot, KeeperWriter, LoadedManifest,
    MANIFEST_FORMAT_VERSION, MANIFEST_MAGIC, Manifest, ManifestCodecError, ManifestFieldStats,
    ManifestSegment, ManifestSource, RecoveredSegment, WRITER_LOCK_FORMAT_VERSION,
    WRITER_LOCK_MAGIC, WRITER_LOCK_RECORD_BYTES, load_manifest_pair, pack_engine_version,
    unpack_engine_version,
};
pub use query::{
    BooleanClause, BooleanOperator, CassQueryFilters, CassQueryParser, CassQueryParserConfigError,
    CassSourceFilter, CassWildcardPattern, DefaultQueryParser, MAX_QUERY_DEPTH, MAX_QUERY_LENGTH,
    Occur, ParsedQuery, PositionedTerm, Query, QueryDiagnostic, QueryDiagnosticKind,
    QueryExplanation, QueryField, QueryParserConfigError, QueryValue, cass_sanitize_query,
    classify_query, truncate_query,
};
pub use schema::{
    Analyzer, CASS_SEMANTIC_SCHEMA, DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA, FieldDescriptor, FieldKind,
    SchemaDescriptor,
};
pub use segment::{
    EncodedSegment, FSLX_FORMAT_VERSION, FSLX_SECTION_ALIGNMENT, FSLX_SEGMENT_MAGIC,
    PendingSegmentFile, SectionEntry, SectionInput, SectionKind, SegmentHeader, SegmentHeaderInput,
    SegmentLimits, SegmentReader,
};
pub use stats::{SegmentStats, SegmentStatsProvider};
