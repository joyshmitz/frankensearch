//! Keeper lifecycle and durability.
//!
//! This module owns the hand-rolled MANIFEST v1 wire format, Q1 range
//! validation, two-slot recovery, cross-process writer ownership, staged
//! durability repair, serialized publication, and writer-admitted garbage
//! collection. Segment I/O and the remaining merge/compaction policy live in
//! adjacent Keeper modules and milestones.

#[cfg(unix)]
use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime};

use asupersync::Cx;
use asupersync::runtime::spawn_blocking;
use asupersync::sync::{LockError, Mutex, OwnedMutexGuard};
use frankensearch_core::SearchError;
#[cfg(feature = "durability")]
use frankensearch_durability::{FileProtector, FileRecoveryOutcome, FileSourceWitness};
use frankensearch_index::mapped_file::ReadOnlyMappedFile;
use thiserror::Error;

use crate::error::QuillError;
use crate::schema::SchemaDescriptor;
use crate::segment::{PendingSegmentFile, SectionKind, SegmentHeader, SegmentReader};

pub use crate::stats::{SegmentStats, SegmentStatsProvider};

/// Eight-byte MANIFEST magic, including its trailing NUL.
pub const MANIFEST_MAGIC: [u8; 8] = *b"FSLXMAN\0";
/// Current durable MANIFEST format version.
pub const MANIFEST_FORMAT_VERSION: u32 = 1;
/// Maximum MANIFEST accepted by an eager reader.
pub const MAX_MANIFEST_BYTES: usize = 64 * 1024 * 1024;
/// Maximum segment records admitted by one MANIFEST.
///
/// This is deliberately below the byte-format's theoretical limit so hostile
/// inputs cannot amplify a bounded file into millions of heap allocations.
pub const MAX_MANIFEST_SEGMENTS: usize = 65_536;
/// Maximum stats rows. `field_ord` is a `u16`, so no larger set is useful.
pub const MAX_MANIFEST_FIELDS: usize = 65_536;
/// Known MANIFEST flag: a bulk build is in progress.
pub const MANIFEST_FLAG_BULK_MODE_IN_PROGRESS: u32 = 1;
/// All flag bits understood by this reader.
pub const MANIFEST_KNOWN_FLAGS: u32 = MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
/// Canonical empty roaring-lite tombstone set (`chunk_count = 0`).
pub const EMPTY_TOMBSTONES: [u8; 4] = [0, 0, 0, 0];
/// Minimum age of an unreachable Quill artifact before writer GC may remove it.
pub const DEFAULT_GARBAGE_GRACE: Duration = Duration::from_secs(300);
/// Eight-byte writer-admission magic, including its trailing NUL.
pub const WRITER_LOCK_MAGIC: [u8; 8] = *b"FSLXLCK\0";
/// Current durable writer-lock record version.
pub const WRITER_LOCK_FORMAT_VERSION: u32 = 1;
/// Exact byte length of a v1 writer-lock record, including its CRC32.
pub const WRITER_LOCK_RECORD_BYTES: usize = 36;

/// Current Quill crate version in the MANIFEST packed-semver representation.
///
/// The build-time assertion in this module's tests intentionally forces this
/// value to change when `Cargo.toml` changes.
pub const CURRENT_ENGINE_VERSION: u32 = pack_engine_version(0, 2, 1);

const MANIFEST_MIN_BYTES: usize = 8 + 4 + 8 + 8 + 8 + 4 + 4 + 4 + 4 + 4;
const SEGMENT_FIXED_BYTES: usize = 8 + 8 + 8 + 8 + 8 + 8 + 4;
const FIELD_STATS_BYTES: usize = 2 + 8 + 4;
const MAX_DOCID_EXCLUSIVE: u64 = 4_294_967_296;
const TOMBSTONE_ARRAY_MAX_CARDINALITY: u16 = 4_096;
const TOMBSTONE_BITMAP_MIN_CARDINALITY: u64 = 3_584;
const MAX_TOMBSTONE_CHUNKS: usize = 65_536;

// Construction initializes this once outside the async publication path. The
// asupersync mutex serializes every filesystem mutation within the process;
// every public mutation path additionally requires a retained cross-process
// `KeeperWriter` admission. Its owned guard is moved into blocking work so a
// cancelled async caller cannot release serialization before that work exits.
static PUBLISH_LOCK: OnceLock<Arc<Mutex<()>>> = OnceLock::new();

/// Pack semantic-version components into the durable `engine_version` word.
///
/// Layout: `major:u8 << 24 | minor:u8 << 16 | patch:u16`. Prerelease and
/// build metadata are intentionally not durable.
#[must_use]
pub const fn pack_engine_version(major: u8, minor: u8, patch: u16) -> u32 {
    (major as u32) << 24 | (minor as u32) << 16 | patch as u32
}

/// Decode a packed `engine_version` into `(major, minor, patch)`.
#[must_use]
pub const fn unpack_engine_version(version: u32) -> (u8, u8, u16) {
    (
        version.to_be_bytes()[0],
        version.to_be_bytes()[1],
        u16::from_be_bytes([version.to_be_bytes()[2], version.to_be_bytes()[3]]),
    )
}

/// Typed failures from MANIFEST byte validation.
#[derive(Debug, Error)]
pub enum ManifestCodecError {
    /// An in-memory manifest violates a writer invariant.
    #[error("invalid manifest: {detail}")]
    Invalid {
        /// Violated invariant.
        detail: String,
    },
    /// The byte stream ended before a field was complete.
    #[error("truncated manifest at byte {offset}: need {needed} bytes, only {remaining} remain")]
    Truncated {
        /// Byte offset where the read began.
        offset: usize,
        /// Bytes required by the field.
        needed: usize,
        /// Bytes still available.
        remaining: usize,
    },
    /// The magic did not identify a Quill MANIFEST.
    #[error("invalid manifest magic")]
    BadMagic,
    /// The reader does not implement the encoded version.
    #[error("unsupported manifest format version {found}")]
    UnsupportedVersion {
        /// Encoded version.
        found: u32,
    },
    /// The trailing CRC did not authenticate the preceding bytes.
    #[error("manifest CRC mismatch: stored {stored:#010x}, computed {computed:#010x}")]
    ChecksumMismatch {
        /// CRC stored in the trailer.
        stored: u32,
        /// CRC recomputed by the reader.
        computed: u32,
    },
    /// A count or file length exceeds an eager-reader resource budget.
    #[error("manifest {resource} limit exceeded: {actual} > {limit}")]
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Observed value.
        actual: u64,
        /// Admitted maximum.
        limit: u64,
    },
    /// Structurally parseable bytes violate a canonical or Q1 invariant.
    #[error("non-canonical manifest: {detail}")]
    NonCanonical {
        /// Violated invariant.
        detail: String,
    },
    /// Valid fields were followed by bytes not owned by the format.
    #[error("manifest has {remaining} trailing bytes before its CRC")]
    TrailingBytes {
        /// Unconsumed byte count.
        remaining: usize,
    },
}

/// Keeper I/O, recovery, locking, and publication failures.
#[derive(Debug, Error)]
pub enum KeeperError {
    /// Neither MANIFEST slot exists.
    #[error("Quill index not found at {directory}")]
    IndexNotFound {
        /// Index directory that was inspected.
        directory: PathBuf,
    },
    /// One durable slot contains invalid bytes.
    #[error("corrupt manifest at {path}: {source}")]
    ManifestCorrupted {
        /// Invalid slot.
        path: PathBuf,
        /// Codec diagnosis.
        #[source]
        source: ManifestCodecError,
    },
    /// Proposed in-memory state failed writer-side validation.
    #[error("invalid proposed manifest: {source}")]
    InvalidManifest {
        /// Writer invariant diagnosis.
        #[source]
        source: ManifestCodecError,
    },
    /// Neither existing slot is valid.
    #[error("no valid manifest in {directory}: current={current}; previous={previous}")]
    NoValidManifest {
        /// Index directory that was inspected.
        directory: PathBuf,
        /// Primary-slot diagnosis.
        current: String,
        /// Previous-slot diagnosis.
        previous: String,
    },
    /// Two valid slots do not form an admitted generation pair.
    #[error(
        "invalid manifest generation pair in {directory}: current={current}, previous={previous}"
    )]
    InvalidGenerationPair {
        /// Index directory containing the pair.
        directory: PathBuf,
        /// Current generation.
        current: u64,
        /// Previous generation.
        previous: u64,
    },
    /// Two individually valid slots disagree on durable snapshot identity.
    #[error(
        "invalid manifest pair in {directory}: current generation {current}, previous generation {previous}: {detail}"
    )]
    InvalidManifestPair {
        /// Index directory containing the pair.
        directory: PathBuf,
        /// Current generation.
        current: u64,
        /// Previous generation.
        previous: u64,
        /// Cross-slot invariant that failed.
        detail: String,
    },
    /// The proposed publication is not the next monotone generation.
    #[error("manifest generation conflict: expected {expected}, proposed {proposed}")]
    GenerationConflict {
        /// Required next generation.
        expected: u64,
        /// Proposed generation.
        proposed: u64,
    },
    /// Another process still owns the cross-process writer admission.
    #[error("Quill writer is busy at {path}; owner pid={owner_pid:?}")]
    WriterBusy {
        /// Canonical writer-lock path.
        path: PathBuf,
        /// Best-effort owner diagnosis from a complete lock record.
        owner_pid: Option<u32>,
    },
    /// The persistent writer-lock record is malformed and cannot be reclaimed.
    #[error("corrupt Quill writer lock at {path}: {detail}")]
    WriterLockCorrupted {
        /// Canonical writer-lock path.
        path: PathBuf,
        /// Failed wire, type, or liveness invariant.
        detail: String,
    },
    /// Cancellation was observed while acquiring or initializing a writer.
    #[error("Quill writer admission cancelled")]
    WriterAdmissionCancelled,
    /// An extant `O_EXCL` artifact already owns the proposed generation.
    #[error("Quill generation {generation} is already claimed at {path}")]
    GenerationClaimConflict {
        /// Existing canonical claim path.
        path: PathBuf,
        /// Claimed MANIFEST generation.
        generation: u64,
    },
    /// The current generation cannot be incremented.
    #[error("manifest generation space exhausted at {current}")]
    GenerationExhausted {
        /// Last representable generation.
        current: u64,
    },
    /// A proposed generation would roll back durable index identity or state.
    #[error("invalid manifest transition: {detail}")]
    InvalidTransition {
        /// Violated monotonicity invariant.
        detail: String,
    },
    /// The caller's expected schema does not match the durable MANIFEST.
    #[error("Quill schema mismatch at {path}: expected {expected:#018x}, found {found:#018x}")]
    SchemaMismatch {
        /// MANIFEST path carrying the durable schema identity.
        path: PathBuf,
        /// Hash of the schema requested by the caller.
        expected: u64,
        /// Hash recorded in the selected MANIFEST.
        found: u64,
    },
    /// The supplied schema descriptor itself is invalid.
    #[error("invalid Quill schema descriptor: {source}")]
    InvalidSchema {
        /// Descriptor validation failure.
        #[source]
        source: QuillError,
    },
    /// A MANIFEST-referenced segment could not be opened structurally.
    #[error("cannot open Quill segment at {path}: {source}")]
    SegmentOpen {
        /// Canonical published segment path.
        path: PathBuf,
        /// Segment reader diagnosis.
        #[source]
        source: QuillError,
    },
    /// A segment disagrees with its immutable MANIFEST witnesses.
    #[error("Quill segment metadata mismatch at {path}: {detail}")]
    SegmentMetadataMismatch {
        /// Canonical published segment path.
        path: PathBuf,
        /// Failed identity, range, length, or checksum-witness comparison.
        detail: String,
    },
    /// A fallback MANIFEST and an extant generation claim disagree.
    #[error(
        "invalid interrupted-publish claim at {path}: recovered generation {recovered}, claimed generation {claimed}"
    )]
    InvalidRecoveryClaim {
        /// Claim path carrying the incompatible generation.
        path: PathBuf,
        /// Generation selected from `MANIFEST.prev`.
        recovered: u64,
        /// Generation named by the claim.
        claimed: u64,
    },
    /// A canonical generation claim is not the required zero-length file.
    #[error("invalid Quill generation claim at {path}: {detail}")]
    InvalidClaimArtifact {
        /// Canonical claim path.
        path: PathBuf,
        /// Failed no-follow type or length invariant.
        detail: String,
    },
    /// A claimed future generation must be resolved before garbage collection.
    #[error(
        "writer recovery required at {path}: current generation {current}, claimed generation {claimed}"
    )]
    ClaimedGenerationPending {
        /// Future claim blocking a garbage sweep.
        path: PathBuf,
        /// Selected durable generation.
        current: u64,
        /// Future claimed generation.
        claimed: u64,
    },
    /// A garbage-collection target was not one safe direct child name.
    #[error("unsafe Quill garbage path rejected: {path}")]
    UnsafeGarbagePath {
        /// Rejected relative or absolute target.
        path: PathBuf,
    },
    /// The supplied index-directory path stopped naming the opened directory.
    #[error("Quill index directory changed during garbage collection: {directory}")]
    GarbageDirectoryChanged {
        /// Directory whose identity was not stable for the sweep.
        directory: PathBuf,
    },
    /// Publishing over a corrupt primary requires the later writer-recovery lane.
    #[error("manifest recovery required before publish at {path}")]
    RecoveryRequired {
        /// Corrupt primary slot.
        path: PathBuf,
    },
    /// A stale temp file exists but is not byte-identical to this proposal.
    #[error("manifest temp conflicts with proposed bytes at {path}")]
    TempConflict {
        /// Temp file that cannot be safely reused or overwritten.
        path: PathBuf,
    },
    /// The cancel-aware in-process publish mutex was not acquired.
    #[error("manifest publish lock failed: {source}")]
    PublishLock {
        /// Asupersync lock failure.
        #[source]
        source: LockError,
    },
    /// Optional repair-sidecar processing failed.
    #[cfg(feature = "durability")]
    #[error("keeper durability {operation} failed at {path}: {source}")]
    Durability {
        /// Durability operation that failed.
        operation: &'static str,
        /// Source or sidecar path being processed.
        path: PathBuf,
        /// Generic durability-layer diagnosis.
        #[source]
        source: SearchError,
    },
    /// A filesystem operation failed with path and operation context.
    #[error("keeper {operation} failed at {path}: {source}")]
    Io {
        /// Operation label.
        operation: &'static str,
        /// Affected path.
        path: PathBuf,
        /// Operating-system error.
        #[source]
        source: io::Error,
    },
}

impl From<KeeperError> for SearchError {
    fn from(error: KeeperError) -> Self {
        match error {
            KeeperError::IndexNotFound { directory } => Self::IndexNotFound { path: directory },
            KeeperError::ManifestCorrupted { path, source } => Self::IndexCorrupted {
                path,
                detail: source.to_string(),
            },
            KeeperError::NoValidManifest {
                directory,
                current,
                previous,
            } => Self::IndexCorrupted {
                path: directory,
                detail: format!("no valid manifest: current={current}; previous={previous}"),
            },
            KeeperError::InvalidGenerationPair {
                directory,
                current,
                previous,
            } => Self::IndexCorrupted {
                path: directory,
                detail: format!(
                    "invalid manifest generation pair: current={current}, previous={previous}"
                ),
            },
            KeeperError::InvalidManifestPair {
                directory,
                current,
                previous,
                detail,
            } => Self::IndexCorrupted {
                path: directory,
                detail: format!(
                    "invalid manifest pair at generations {current}/{previous}: {detail}"
                ),
            },
            KeeperError::SegmentOpen {
                path,
                source: QuillError::IndexCorrupted { detail, .. },
            } => Self::IndexCorrupted { path, detail },
            KeeperError::SegmentOpen {
                path,
                source: QuillError::Io(source),
            } if source.kind() == io::ErrorKind::NotFound => Self::IndexCorrupted {
                path,
                detail: "MANIFEST-referenced segment is missing".to_owned(),
            },
            KeeperError::SegmentOpen {
                path,
                source: QuillError::UnknownSchema { schema_id },
            } => Self::IndexCorrupted {
                path,
                detail: format!("MANIFEST-referenced segment has unknown schema {schema_id:#018x}"),
            },
            KeeperError::SegmentOpen { path, source } => Self::SubsystemError {
                subsystem: "quill",
                source: Box::new(KeeperError::SegmentOpen { path, source }),
            },
            KeeperError::SegmentMetadataMismatch { path, detail } => {
                Self::IndexCorrupted { path, detail }
            }
            KeeperError::InvalidRecoveryClaim {
                path,
                recovered,
                claimed,
            } => Self::IndexCorrupted {
                path,
                detail: format!(
                    "recovered manifest generation {recovered} conflicts with claim {claimed}"
                ),
            },
            KeeperError::InvalidClaimArtifact { path, detail } => {
                Self::IndexCorrupted { path, detail }
            }
            KeeperError::WriterAdmissionCancelled => Self::Cancelled {
                phase: "keeper.writer_admission".to_owned(),
                reason: "writer admission cancelled".to_owned(),
            },
            KeeperError::PublishLock { source } => match source {
                LockError::Cancelled => Self::Cancelled {
                    phase: "keeper.publish".to_owned(),
                    reason: "manifest publish lock cancelled".to_owned(),
                },
                LockError::TimedOut(deadline) => Self::Cancelled {
                    phase: "keeper.publish".to_owned(),
                    reason: format!("manifest publish lock timed out at {deadline:?}"),
                },
                other => Self::SubsystemError {
                    subsystem: "quill",
                    source: Box::new(KeeperError::PublishLock { source: other }),
                },
            },
            other => Self::SubsystemError {
                subsystem: "quill",
                source: Box::new(other),
            },
        }
    }
}

/// One immutable segment referenced by a MANIFEST generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestSegment {
    /// Random, collision-checked segment identifier.
    pub segment_id: u64,
    /// Monotone seal order used for newest-first IDHASH probing.
    pub seal_seq: u64,
    /// Exact FSLX file length.
    pub file_len: u64,
    /// xxh3-64 over the FSLX prefix before the trailer.
    pub file_xxh3: u64,
    /// Inclusive lower bound of the segment's Q1 range.
    pub docid_lo: u64,
    /// Exclusive upper bound of the segment's Q1 range.
    pub docid_hi: u64,
    /// Live-at-seal document count.
    pub doc_count: u32,
    /// Canonical roaring-lite tombstone bytes.
    pub tombstones: Vec<u8>,
}

/// Snapshot-level field statistics rolled up over referenced segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifestFieldStats {
    /// Schema field ordinal.
    pub field_ord: u16,
    /// Sum of at-seal token counts.
    pub total_tokens: u64,
    /// Sum of at-seal document counts for this field.
    pub doc_count: u32,
}

/// Fully owned MANIFEST v1 contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Monotone publication generation. Genesis is generation 1.
    pub generation: u64,
    /// First docid that has never been leased; never rolls back.
    pub docid_high_watermark: u64,
    /// Durable schema descriptor hash.
    pub schema_id: u64,
    /// Informational packed Quill crate version.
    pub engine_version: u32,
    /// Versioned manifest flags.
    pub flags: u32,
    /// Segments in strictly ascending `docid_lo` order.
    pub segments: Vec<ManifestSegment>,
    /// Field statistics in strictly ascending `field_ord` order.
    pub field_stats: Vec<ManifestFieldStats>,
}

impl Manifest {
    /// Construct an empty generation using the current engine version.
    #[must_use]
    pub const fn empty(generation: u64, schema_id: u64, docid_high_watermark: u64) -> Self {
        Self {
            generation,
            docid_high_watermark,
            schema_id,
            engine_version: CURRENT_ENGINE_VERSION,
            flags: 0,
            segments: Vec::new(),
            field_stats: Vec::new(),
        }
    }

    /// Whether the durable bulk-mode flag is set.
    #[must_use]
    pub const fn bulk_mode_in_progress(&self) -> bool {
        self.flags & MANIFEST_FLAG_BULK_MODE_IN_PROGRESS != 0
    }

    /// Validate writer-side canonical form and Q1 range invariants.
    ///
    /// # Errors
    ///
    /// Returns a typed invariant or resource error before any bytes are
    /// written.
    pub fn validate(&self) -> Result<(), ManifestCodecError> {
        validate_manifest(self, ErrorClass::Invalid)
    }

    /// Encode canonical MANIFEST v1 bytes including the trailing CRC32.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid in-memory state, count conversion, or an
    /// output exceeding the eager-reader budget.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ManifestCodecError> {
        validate_manifest_shape(self, ErrorClass::Invalid)?;
        let estimated = manifest_encoded_len(self)?;
        self.validate()?;

        let segment_count =
            u32::try_from(self.segments.len()).map_err(|_| ManifestCodecError::ResourceLimit {
                resource: "segment count",
                actual: usize_to_u64(self.segments.len()),
                limit: u64::from(u32::MAX),
            })?;
        let field_count = u32::try_from(self.field_stats.len()).map_err(|_| {
            ManifestCodecError::ResourceLimit {
                resource: "field count",
                actual: usize_to_u64(self.field_stats.len()),
                limit: u64::from(u32::MAX),
            }
        })?;

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(estimated)
            .map_err(|error| invalid(format!("manifest allocation failed: {error}")))?;
        bytes.extend_from_slice(&MANIFEST_MAGIC);
        put_u32(&mut bytes, MANIFEST_FORMAT_VERSION);
        put_u64(&mut bytes, self.generation);
        put_u64(&mut bytes, self.docid_high_watermark);
        put_u64(&mut bytes, self.schema_id);
        put_u32(&mut bytes, self.engine_version);
        put_u32(&mut bytes, self.flags);
        put_u32(&mut bytes, segment_count);
        for segment in &self.segments {
            put_u64(&mut bytes, segment.segment_id);
            put_u64(&mut bytes, segment.seal_seq);
            put_u64(&mut bytes, segment.file_len);
            put_u64(&mut bytes, segment.file_xxh3);
            put_u64(&mut bytes, segment.docid_lo);
            put_u64(&mut bytes, segment.docid_hi);
            put_u32(&mut bytes, segment.doc_count);
            bytes.extend_from_slice(&segment.tombstones);
        }
        put_u32(&mut bytes, field_count);
        for stats in &self.field_stats {
            put_u16(&mut bytes, stats.field_ord);
            put_u64(&mut bytes, stats.total_tokens);
            put_u32(&mut bytes, stats.doc_count);
        }
        let checksum = crc32fast::hash(&bytes);
        put_u32(&mut bytes, checksum);
        debug_assert_eq!(bytes.len(), estimated);
        Ok(bytes)
    }

    /// Parse and eagerly validate canonical MANIFEST v1 bytes.
    ///
    /// # Errors
    ///
    /// Rejects unsupported versions, checksum failures, hostile counts,
    /// non-canonical tombstones, Q1 violations, truncation, and trailing bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ManifestCodecError> {
        check_manifest_byte_limit(bytes.len())?;
        if bytes.len() < MANIFEST_MIN_BYTES {
            return Err(ManifestCodecError::Truncated {
                offset: 0,
                needed: MANIFEST_MIN_BYTES,
                remaining: bytes.len(),
            });
        }

        let body_len = bytes.len() - 4;
        let stored_checksum = u32::from_le_bytes([
            bytes[body_len],
            bytes[body_len + 1],
            bytes[body_len + 2],
            bytes[body_len + 3],
        ]);
        let computed_checksum = crc32fast::hash(&bytes[..body_len]);
        if stored_checksum != computed_checksum {
            return Err(ManifestCodecError::ChecksumMismatch {
                stored: stored_checksum,
                computed: computed_checksum,
            });
        }

        let mut cursor = ByteCursor::new(&bytes[..body_len]);
        if cursor.take(8)? != MANIFEST_MAGIC {
            return Err(ManifestCodecError::BadMagic);
        }
        let version = cursor.u32()?;
        if version != MANIFEST_FORMAT_VERSION {
            return Err(ManifestCodecError::UnsupportedVersion { found: version });
        }
        let generation = cursor.u64()?;
        let docid_high_watermark = cursor.u64()?;
        let schema_id = cursor.u64()?;
        let engine_version = cursor.u32()?;
        let flags = cursor.u32()?;
        let segment_count = count_to_usize(cursor.u32()?, "segment count", MAX_MANIFEST_SEGMENTS)?;
        let minimum_segment_bytes = segment_count
            .checked_mul(SEGMENT_FIXED_BYTES + EMPTY_TOMBSTONES.len())
            .ok_or_else(|| non_canonical("segment byte count overflow"))?;
        if minimum_segment_bytes > cursor.remaining() {
            return Err(ManifestCodecError::Truncated {
                offset: cursor.position(),
                needed: minimum_segment_bytes,
                remaining: cursor.remaining(),
            });
        }

        let mut segments = Vec::new();
        segments
            .try_reserve_exact(segment_count)
            .map_err(|error| non_canonical(format!("segment allocation failed: {error}")))?;
        for _ in 0..segment_count {
            let segment_id = cursor.u64()?;
            let seal_seq = cursor.u64()?;
            let file_len = cursor.u64()?;
            let file_xxh3 = cursor.u64()?;
            let docid_lo = cursor.u64()?;
            let docid_hi = cursor.u64()?;
            let doc_count = cursor.u32()?;
            let tombstone_start = cursor.position();
            consume_tombstone_set(&mut cursor, None)?;
            let tombstones = copy_bytes(
                &bytes[tombstone_start..cursor.position()],
                "tombstone bytes",
            )?;
            segments.push(ManifestSegment {
                segment_id,
                seal_seq,
                file_len,
                file_xxh3,
                docid_lo,
                docid_hi,
                doc_count,
                tombstones,
            });
        }

        let field_count = count_to_usize(cursor.u32()?, "field count", MAX_MANIFEST_FIELDS)?;
        let minimum_stats_bytes = field_count
            .checked_mul(FIELD_STATS_BYTES)
            .ok_or_else(|| non_canonical("field-stats byte count overflow"))?;
        if minimum_stats_bytes > cursor.remaining() {
            return Err(ManifestCodecError::Truncated {
                offset: cursor.position(),
                needed: minimum_stats_bytes,
                remaining: cursor.remaining(),
            });
        }
        let mut field_stats = Vec::new();
        field_stats
            .try_reserve_exact(field_count)
            .map_err(|error| non_canonical(format!("field-stats allocation failed: {error}")))?;
        for _ in 0..field_count {
            field_stats.push(ManifestFieldStats {
                field_ord: cursor.u16()?,
                total_tokens: cursor.u64()?,
                doc_count: cursor.u32()?,
            });
        }
        if cursor.remaining() != 0 {
            return Err(ManifestCodecError::TrailingBytes {
                remaining: cursor.remaining(),
            });
        }

        let manifest = Self {
            generation,
            docid_high_watermark,
            schema_id,
            engine_version,
            flags,
            segments,
            field_stats,
        };
        validate_manifest(&manifest, ErrorClass::NonCanonical)?;
        Ok(manifest)
    }
}

/// Slot selected by read-only two-slot recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestSource {
    /// The primary `MANIFEST` was valid.
    Current,
    /// `MANIFEST` was absent, as in the crash window between the two renames.
    PreviousAfterMissingCurrent,
    /// `MANIFEST` existed but was corrupt, so only `MANIFEST.prev` was usable.
    PreviousAfterCorruptCurrent,
    /// No filesystem is attached; this is an owned-buffer genesis snapshot.
    InMemory,
}

/// A validated MANIFEST together with its recovery provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedManifest {
    /// Validated contents.
    pub manifest: Manifest,
    /// Durable slot that supplied the contents.
    pub source: ManifestSource,
}

/// One immutable, mmap-backed segment admitted by a recovered snapshot.
///
/// Structural framing and MANIFEST witnesses are checked during open. Section
/// payload hashes remain lazy and are checked on first access.
#[derive(Clone)]
pub struct RecoveredSegment {
    path: PathBuf,
    manifest: ManifestSegment,
    reader: Arc<SegmentReader<ReadOnlyMappedFile>>,
}

impl RecoveredSegment {
    /// Canonical published segment path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Immutable MANIFEST metadata for this segment.
    #[must_use]
    pub const fn manifest(&self) -> &ManifestSegment {
        &self.manifest
    }

    /// Structurally validated FSLX header.
    #[must_use]
    pub fn header(&self) -> SegmentHeader {
        self.reader.header()
    }

    /// Verify and borrow one section payload on first access.
    ///
    /// # Errors
    ///
    /// Returns a typed segment-corruption error when the payload checksum does
    /// not match its section-table witness.
    pub fn section(&self, kind: SectionKind) -> Result<Option<&[u8]>, QuillError> {
        self.reader.section(kind)
    }

    /// Eagerly recompute every segment checksum for doctor-style verification.
    ///
    /// # Errors
    ///
    /// Returns a typed corruption error for any checksum mismatch.
    pub fn verify(&self) -> Result<(), QuillError> {
        self.reader.verify()
    }
}

/// A read-only, internally consistent Keeper snapshot.
///
/// `open` performs recovery by selecting the admitted MANIFEST slot and
/// validating every referenced segment. It never repairs, renames, or removes
/// filesystem entries. Keeping this value alive keeps its immutable mmaps
/// alive as well.
#[derive(Clone)]
pub struct KeeperSnapshot {
    directory: Option<PathBuf>,
    schema: SchemaDescriptor,
    loaded: LoadedManifest,
    segments: Vec<RecoveredSegment>,
}

impl KeeperSnapshot {
    /// Open one durable snapshot without mutating the index directory.
    ///
    /// # Errors
    ///
    /// Returns typed not-found, manifest, schema, segment, or metadata errors.
    pub fn open(
        directory: impl AsRef<Path>,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        let directory = directory.as_ref();
        match Self::open_once(directory, schema) {
            Err(error) if recovery_retryable(&error) => Self::open_once(directory, schema),
            result => result,
        }
    }

    fn open_once(directory: &Path, schema: SchemaDescriptor) -> Result<Self, KeeperError> {
        let expected_schema_id = schema
            .schema_id()
            .map_err(|source| KeeperError::InvalidSchema { source })?;
        let loaded = load_manifest_pair(directory)?;
        validate_loaded_schema(directory, expected_schema_id, &loaded)?;
        validate_recovery_claims(directory, &loaded)?;

        let mut segments = Vec::new();
        segments
            .try_reserve_exact(loaded.manifest.segments.len())
            .map_err(|error| KeeperError::Io {
                operation: "allocate recovered segment table",
                path: directory.to_path_buf(),
                source: io::Error::other(error.to_string()),
            })?;
        for manifest_segment in &loaded.manifest.segments {
            let path = directory.join(canonical_segment_name(manifest_segment.segment_id));
            let reader = SegmentReader::open_published(&path, schema).map_err(|source| {
                KeeperError::SegmentOpen {
                    path: path.clone(),
                    source,
                }
            })?;
            validate_segment_witnesses(&path, manifest_segment, &reader)?;
            segments.push(RecoveredSegment {
                path,
                manifest: manifest_segment.clone(),
                reader: Arc::new(reader),
            });
        }

        Ok(Self {
            directory: Some(directory.to_path_buf()),
            schema,
            loaded,
            segments,
        })
    }

    /// Create a genesis index or open an existing index with the same schema.
    ///
    /// Creation is a writer operation: it acquires the cross-process admission,
    /// publishes genesis through an `O_EXCL` generation claim, and releases the
    /// writer capability before returning this read-only snapshot.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, I/O, schema mismatch, publication, or open
    /// errors. Existing durable state is never rebuilt on mismatch.
    pub async fn create(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        let writer = KeeperWriter::create(cx, directory, schema).await?;
        Ok(writer.snapshot.clone())
    }

    /// Acquire the sole cross-process writer capability for this index.
    ///
    /// Reader [`Self::open`] remains lock-free and filesystem-touchless.
    /// Writer open acquires `LOCK` before interrupted-publish recovery and GC.
    ///
    /// # Errors
    ///
    /// Returns a typed busy, corrupt-lock, recovery, schema, or I/O failure.
    pub async fn open_writer(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
    ) -> Result<KeeperWriter, KeeperError> {
        KeeperWriter::open(cx, directory, schema).await
    }

    /// Construct an owned-buffer genesis snapshot without creating files.
    ///
    /// Later ingest milestones attach mutable deltas and owned FSLX segments;
    /// recovery and garbage collection remain no-ops for this backend.
    ///
    /// # Errors
    ///
    /// Returns an invalid-schema error when the descriptor is not canonical.
    pub fn in_memory(schema: SchemaDescriptor) -> Result<Self, KeeperError> {
        let schema_id = schema
            .schema_id()
            .map_err(|source| KeeperError::InvalidSchema { source })?;
        Ok(Self {
            directory: None,
            schema,
            loaded: LoadedManifest {
                manifest: Manifest::empty(1, schema_id, 0),
                source: ManifestSource::InMemory,
            },
            segments: Vec::new(),
        })
    }

    /// Durable index directory, or `None` for an in-memory snapshot.
    #[must_use]
    pub fn directory(&self) -> Option<&Path> {
        self.directory.as_deref()
    }

    /// Compile-time schema admitted by this snapshot.
    #[must_use]
    pub const fn schema(&self) -> SchemaDescriptor {
        self.schema
    }

    /// Selected MANIFEST and its recovery provenance.
    #[must_use]
    pub const fn loaded_manifest(&self) -> &LoadedManifest {
        &self.loaded
    }

    /// Referenced immutable segments in ascending Q1 range order.
    #[must_use]
    pub fn segments(&self) -> &[RecoveredSegment] {
        &self.segments
    }
}

#[derive(Clone)]
enum WriterProtection {
    Disabled,
    #[cfg(feature = "durability")]
    Enabled(FileProtector),
}

/// Sole cross-process mutation capability for one Quill index directory.
///
/// This type is intentionally not `Clone`. Its internal admission token may be
/// retained by an in-flight blocking publication so cancellation cannot release
/// the OS lock while filesystem mutation is still running.
///
/// Every process that mutates the index directory must honor Quill's `LOCK`.
/// Crash consistency does not cover an out-of-band process replacing admitted
/// directory entries while a writer operation is in flight.
pub struct KeeperWriter {
    admission: Arc<WriterAdmissionInner>,
    snapshot: KeeperSnapshot,
    garbage_options: GarbageCollectionOptions,
    protection: WriterProtection,
}

impl KeeperWriter {
    /// Open an existing index for mutation with the default GC grace period.
    ///
    /// # Errors
    ///
    /// Returns a typed admission, recovery, schema, segment, or I/O failure.
    pub async fn open(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        Self::open_inner(
            cx,
            directory.into(),
            schema,
            false,
            GarbageCollectionOptions::default(),
            WriterProtection::Disabled,
        )
        .await
    }

    /// Create a genesis index or open an existing one for mutation.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::open`] plus directory creation or
    /// genesis-publication failures.
    pub async fn create(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
    ) -> Result<Self, KeeperError> {
        Self::open_inner(
            cx,
            directory.into(),
            schema,
            true,
            GarbageCollectionOptions::default(),
            WriterProtection::Disabled,
        )
        .await
    }

    /// Open an existing index and activate writer-only FEC recovery.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::open`] plus typed durability
    /// decode, validation, or staged-install failures.
    #[cfg(feature = "durability")]
    pub async fn open_durable(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        protector: FileProtector,
    ) -> Result<Self, KeeperError> {
        Self::open_inner(
            cx,
            directory.into(),
            schema,
            false,
            GarbageCollectionOptions::default(),
            WriterProtection::Enabled(protector),
        )
        .await
    }

    /// Create or open an index with writer-only FEC recovery enabled.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::open_durable`] plus genesis
    /// directory or publication failures.
    #[cfg(feature = "durability")]
    pub async fn create_durable(
        cx: &Cx,
        directory: impl Into<PathBuf>,
        schema: SchemaDescriptor,
        protector: FileProtector,
    ) -> Result<Self, KeeperError> {
        Self::open_inner(
            cx,
            directory.into(),
            schema,
            true,
            GarbageCollectionOptions::default(),
            WriterProtection::Enabled(protector),
        )
        .await
    }

    async fn open_inner(
        cx: &Cx,
        directory: PathBuf,
        schema: SchemaDescriptor,
        create: bool,
        garbage_options: GarbageCollectionOptions,
        protection: WriterProtection,
    ) -> Result<Self, KeeperError> {
        if cx.is_cancel_requested() {
            return Err(KeeperError::WriterAdmissionCancelled);
        }
        let mut directory = normalize_publish_directory(directory);
        if create {
            let create_path = directory.clone();
            spawn_blocking(move || {
                std::fs::create_dir_all(&create_path).map_err(|source| KeeperError::Io {
                    operation: "create index directory",
                    path: create_path,
                    source,
                })
            })
            .await?;
            directory = normalize_publish_directory(directory);
        }
        let admission_directory = directory.clone();
        let admission =
            spawn_blocking(move || acquire_writer_admission(&admission_directory)).await?;
        if cx.is_cancel_requested() {
            return Err(KeeperError::WriterAdmissionCancelled);
        }

        let recovery_admission = Arc::clone(&admission);
        let recovery_protection = protection.clone();
        spawn_blocking(move || {
            recover_writer_directory(&recovery_admission, schema, &recovery_protection)
        })
        .await?;

        match open_snapshot_blocking(directory.clone(), schema).await {
            Ok(_) => {}
            Err(KeeperError::IndexNotFound { .. }) if create => {
                let schema_id = schema
                    .schema_id()
                    .map_err(|source| KeeperError::InvalidSchema { source })?;
                let genesis = Manifest::empty(1, schema_id, 0);
                let claim_admission = Arc::clone(&admission);
                let publisher = ManifestPublisher::new(&directory);
                match &protection {
                    WriterProtection::Disabled => {
                        publisher
                            .publish_with_generation_claim(cx, &genesis, move |_, generation| {
                                GenerationClaimGuard::acquire(claim_admission, generation)
                            })
                            .await?;
                    }
                    #[cfg(feature = "durability")]
                    WriterProtection::Enabled(protector) => {
                        publisher
                            .publish_durable_with_generation_claim(
                                cx,
                                &genesis,
                                protector,
                                move |_, generation| {
                                    GenerationClaimGuard::acquire(claim_admission, generation)
                                },
                            )
                            .await?;
                    }
                }
                open_snapshot_blocking(directory.clone(), schema).await?;
            }
            Err(error) => return Err(error),
        }

        let gc_admission = Arc::clone(&admission);
        let gc_directory = directory.clone();
        spawn_blocking(move || {
            gc_admission.ensure_directory_identity()?;
            collect_writer_garbage_under_lock(&gc_directory, schema, garbage_options)
        })
        .await?;
        // GC cannot remove reachable bytes, but reopen once so the writer's
        // snapshot is proven from the post-recovery directory state.
        let snapshot = open_snapshot_blocking(directory, schema).await?;
        Ok(Self {
            admission,
            snapshot,
            garbage_options,
            protection,
        })
    }

    /// Current immutable reader view held by this writer.
    #[must_use]
    pub const fn snapshot(&self) -> &KeeperSnapshot {
        &self.snapshot
    }

    /// Publish exactly the next MANIFEST generation through an `O_EXCL` claim.
    ///
    /// # Errors
    ///
    /// Returns typed cancellation, claim, transition, durability, or I/O
    /// failures. The prior snapshot remains installed on error.
    pub async fn publish(
        &mut self,
        cx: &Cx,
        manifest: &Manifest,
    ) -> Result<&KeeperSnapshot, KeeperError> {
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        self.admission.ensure_directory_identity()?;
        validate_manifest_shape(manifest, ErrorClass::Invalid)
            .map_err(|source| KeeperError::InvalidManifest { source })?;
        manifest_encoded_len(manifest).map_err(|source| KeeperError::InvalidManifest { source })?;
        manifest
            .validate()
            .map_err(|source| KeeperError::InvalidManifest { source })?;
        validate_manifest_successor(&self.snapshot.loaded_manifest().manifest, manifest)?;
        let directory = self.admission.directory.clone();
        let preflight_directory = directory.clone();
        let preflight_manifest = manifest.clone();
        let preflight_protection = self.protection.clone();
        let schema = self.snapshot.schema();
        spawn_blocking(move || {
            validate_proposed_manifest_segments(
                &preflight_directory,
                &preflight_manifest,
                schema,
                &preflight_protection,
            )
        })
        .await?;
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        let claim_admission = Arc::clone(&self.admission);
        let publisher = ManifestPublisher::new(&directory);
        match &self.protection {
            WriterProtection::Disabled => {
                publisher
                    .publish_with_generation_claim(cx, manifest, move |_, generation| {
                        GenerationClaimGuard::acquire(claim_admission, generation)
                    })
                    .await?;
            }
            #[cfg(feature = "durability")]
            WriterProtection::Enabled(protector) => {
                publisher
                    .publish_durable_with_generation_claim(
                        cx,
                        manifest,
                        protector,
                        move |_, generation| {
                            GenerationClaimGuard::acquire(claim_admission, generation)
                        },
                    )
                    .await?;
            }
        }
        self.admission.ensure_directory_identity()?;
        self.snapshot = open_snapshot_blocking(directory, self.snapshot.schema()).await?;
        Ok(&self.snapshot)
    }

    /// Atomically install one synced immutable segment while holding `LOCK`.
    ///
    /// # Errors
    ///
    /// Returns a typed path, collision, sidecar, or fsync failure.
    pub async fn publish_segment(
        &mut self,
        cx: &Cx,
        pending: PendingSegmentFile,
    ) -> Result<PathBuf, KeeperError> {
        let guard = writer_mutation_guard(cx).await?;
        let admission = Arc::clone(&self.admission);
        let protection = self.protection.clone();
        spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            let parent = pending.path().parent().ok_or_else(|| KeeperError::Io {
                operation: "resolve pending segment directory",
                path: pending.path().to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "pending segment path has no parent directory",
                ),
            })?;
            let parent = std::fs::canonicalize(parent).map_err(|source| KeeperError::Io {
                operation: "canonicalize pending segment directory",
                path: parent.to_path_buf(),
                source,
            })?;
            if parent != admission.directory {
                return Err(KeeperError::Io {
                    operation: "verify pending segment directory",
                    path: pending.path().to_path_buf(),
                    source: io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "pending segment belongs to a different index directory",
                    ),
                });
            }
            let published = match &protection {
                WriterProtection::Disabled => publish_pending_segment(pending),
                #[cfg(feature = "durability")]
                WriterProtection::Enabled(protector) => {
                    publish_pending_segment_durable(pending, protector)
                }
            }?;
            admission.ensure_directory_identity()?;
            Ok(published)
        })
        .await
    }

    /// Run one grace-window garbage sweep under the held writer admission.
    ///
    /// # Errors
    ///
    /// Returns before removal on any recovery, identity, metadata, or path
    /// validation failure.
    pub async fn collect_garbage(
        &mut self,
        cx: &Cx,
    ) -> Result<GarbageCollectionReport, KeeperError> {
        let guard = writer_mutation_guard(cx).await?;
        let admission = Arc::clone(&self.admission);
        let schema = self.snapshot.schema();
        let options = self.garbage_options;
        spawn_blocking(move || {
            let _guard = guard;
            admission.ensure_directory_identity()?;
            collect_writer_garbage_under_lock(&admission.directory, schema, options)
        })
        .await
    }
}

async fn writer_mutation_guard(cx: &Cx) -> Result<OwnedMutexGuard<()>, KeeperError> {
    if cx.is_cancel_requested() {
        return Err(KeeperError::WriterAdmissionCancelled);
    }
    let guard = OwnedMutexGuard::lock(global_publish_lock(), cx)
        .await
        .map_err(|source| KeeperError::PublishLock { source })?;
    if cx.is_cancel_requested() {
        return Err(KeeperError::WriterAdmissionCancelled);
    }
    Ok(guard)
}

fn recovery_retryable(error: &KeeperError) -> bool {
    matches!(
        error,
        KeeperError::SegmentOpen { .. } | KeeperError::SegmentMetadataMismatch { .. }
    )
}

async fn open_snapshot_blocking(
    directory: PathBuf,
    schema: SchemaDescriptor,
) -> Result<KeeperSnapshot, KeeperError> {
    spawn_blocking(move || KeeperSnapshot::open(directory, schema)).await
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WriterLockRecord {
    pid: u32,
    pid_start_nonce: u64,
    acquired_unix_s: i64,
}

impl WriterLockRecord {
    fn current(path: &Path) -> Result<Self, KeeperError> {
        let pid = std::process::id();
        if i32::try_from(pid).is_err() || pid == 0 {
            return Err(KeeperError::WriterLockCorrupted {
                path: path.to_path_buf(),
                detail: format!("current process id {pid} is outside the POSIX pid range"),
            });
        }
        let acquired_unix_s = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()
            .and_then(|elapsed| i64::try_from(elapsed.as_secs()).ok())
            .ok_or_else(|| KeeperError::WriterLockCorrupted {
                path: path.to_path_buf(),
                detail: "current wall clock is outside the v1 lock-record range".to_owned(),
            })?;
        Ok(Self {
            pid,
            pid_start_nonce: writer_pid_start_nonce(pid, acquired_unix_s),
            acquired_unix_s,
        })
    }

    fn to_bytes(self) -> [u8; WRITER_LOCK_RECORD_BYTES] {
        let mut bytes = [0_u8; WRITER_LOCK_RECORD_BYTES];
        bytes[..8].copy_from_slice(&WRITER_LOCK_MAGIC);
        bytes[8..12].copy_from_slice(&WRITER_LOCK_FORMAT_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.pid.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.pid_start_nonce.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.acquired_unix_s.to_le_bytes());
        let checksum = crc32fast::hash(&bytes[..32]);
        bytes[32..36].copy_from_slice(&checksum.to_le_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != WRITER_LOCK_RECORD_BYTES {
            return Err(format!(
                "v1 record must be exactly {WRITER_LOCK_RECORD_BYTES} bytes, found {}",
                bytes.len()
            ));
        }
        if bytes[..8] != WRITER_LOCK_MAGIC {
            return Err("invalid writer-lock magic".to_owned());
        }
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version != WRITER_LOCK_FORMAT_VERSION {
            return Err(format!("unsupported writer-lock version {version}"));
        }
        let stored_checksum = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        let computed_checksum = crc32fast::hash(&bytes[..32]);
        if stored_checksum != computed_checksum {
            return Err(format!(
                "writer-lock CRC mismatch: stored {stored_checksum:#010x}, computed {computed_checksum:#010x}"
            ));
        }
        let pid = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        if pid == 0 || i32::try_from(pid).is_err() {
            return Err(format!(
                "writer-lock pid {pid} is outside the POSIX pid range"
            ));
        }
        let pid_start_nonce = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        let acquired_unix_s = i64::from_le_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        if acquired_unix_s < 0 {
            return Err(format!(
                "writer-lock acquisition timestamp {acquired_unix_s} predates the Unix epoch"
            ));
        }
        Ok(Self {
            pid,
            pid_start_nonce,
            acquired_unix_s,
        })
    }
}

fn writer_pid_start_nonce(pid: u32, acquired_unix_s: i64) -> u64 {
    #[cfg(target_os = "linux")]
    if let Some(start_time) = linux_process_start_time(pid) {
        let mut identity = [0_u8; 12];
        identity[..4].copy_from_slice(&pid.to_le_bytes());
        identity[4..].copy_from_slice(&start_time.to_le_bytes());
        return xxhash_rust::xxh3::xxh3_64(&identity);
    }

    let mut identity = [0_u8; 12];
    identity[..4].copy_from_slice(&pid.to_le_bytes());
    identity[4..].copy_from_slice(&acquired_unix_s.to_le_bytes());
    xxhash_rust::xxh3::xxh3_64(&identity)
}

#[cfg(target_os = "linux")]
fn linux_process_start_time(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_command = stat.get(stat.rfind(')')?.checked_add(1)?..)?;
    // The first token after the command is field 3 (`state`); starttime is
    // field 22, hence zero-based token 19 in this suffix.
    after_command.split_whitespace().nth(19)?.parse().ok()
}

struct WriterAdmissionInner {
    directory: PathBuf,
    directory_file: File,
    lock_path: PathBuf,
    lock_file: File,
    #[cfg(unix)]
    lock_device: u64,
    #[cfg(unix)]
    lock_inode: u64,
    record: WriterLockRecord,
}

impl WriterAdmissionInner {
    #[cfg(unix)]
    fn ensure_directory_identity(&self) -> Result<(), KeeperError> {
        ensure_gc_directory_identity(&self.directory, &self.directory_file)?;
        ensure_writer_lock_identity(self)
    }

    #[cfg(not(unix))]
    fn ensure_directory_identity(&self) -> Result<(), KeeperError> {
        Err(KeeperError::Io {
            operation: "verify writer-lock directory identity",
            path: self.directory.clone(),
            source: io::Error::new(
                io::ErrorKind::Unsupported,
                "cross-process Quill writer admission requires Unix directory identity",
            ),
        })
    }
}

impl Drop for WriterAdmissionInner {
    fn drop(&mut self) {
        release_writer_admission(self);
    }
}

#[cfg(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
))]
fn acquire_writer_admission(directory: &Path) -> Result<Arc<WriterAdmissionInner>, KeeperError> {
    use rustix::fs::{FlockOperation, Mode, OFlags, flock, openat};

    let directory_file = open_gc_directory(directory)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    let lock_path = directory.join("LOCK");
    let lock = openat(
        &directory_file,
        "LOCK",
        OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::CREATE,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(io::Error::from)
    .map_err(|source| KeeperError::Io {
        operation: "open no-follow writer lock",
        path: lock_path.clone(),
        source,
    })?;
    let mut lock_file = File::from(lock);
    let metadata = lock_file.metadata().map_err(|source| KeeperError::Io {
        operation: "inspect writer lock",
        path: lock_path.clone(),
        source,
    })?;
    if !metadata.file_type().is_file() {
        return Err(KeeperError::WriterLockCorrupted {
            path: lock_path,
            detail: "LOCK must be a no-follow regular file".to_owned(),
        });
    }
    use std::os::unix::fs::MetadataExt;
    let lock_device = metadata.dev();
    let lock_inode = metadata.ino();
    if let Err(source) = flock(&lock_file, FlockOperation::NonBlockingLockExclusive) {
        if source == rustix::io::Errno::AGAIN {
            let owner_pid = read_writer_lock_record(&lock_path, &mut lock_file)
                .ok()
                .flatten()
                .map(|record| record.pid);
            return Err(KeeperError::WriterBusy {
                path: lock_path,
                owner_pid,
            });
        }
        return Err(KeeperError::Io {
            operation: "acquire writer flock",
            path: lock_path,
            source: io::Error::from(source),
        });
    }

    if let Some(previous) = read_writer_lock_record(&lock_path, &mut lock_file)?
        && !writer_pid_is_dead(previous.pid)
    {
        return Err(KeeperError::WriterBusy {
            path: lock_path,
            owner_pid: Some(previous.pid),
        });
    }

    let record = WriterLockRecord::current(&lock_path)?;
    if let Err(error) = write_writer_lock_record(&lock_path, &mut lock_file, record) {
        // The flock proves this descriptor is the only cooperating writer for
        // this inode. Best-effort truncation prevents a short failed write from
        // becoming a permanent corrupt residual record.
        let _ = lock_file.set_len(0);
        let _ = lock_file.sync_all();
        return Err(error);
    }
    let admission = Arc::new(WriterAdmissionInner {
        directory: directory.to_path_buf(),
        directory_file,
        lock_path,
        lock_file,
        lock_device,
        lock_inode,
        record,
    });
    admission
        .directory_file
        .sync_all()
        .map_err(|source| KeeperError::Io {
            operation: "fsync writer-lock directory",
            path: directory.to_path_buf(),
            source,
        })?;
    admission.ensure_directory_identity()?;
    Ok(admission)
}

#[cfg(not(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
)))]
fn acquire_writer_admission(directory: &Path) -> Result<Arc<WriterAdmissionInner>, KeeperError> {
    Err(KeeperError::Io {
        operation: "verify writer-lock support",
        path: directory.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            "cross-process Quill writer admission requires flock semantics",
        ),
    })
}

fn read_writer_lock_record(
    path: &Path,
    file: &mut File,
) -> Result<Option<WriterLockRecord>, KeeperError> {
    let length = file
        .metadata()
        .map_err(|source| KeeperError::Io {
            operation: "inspect writer-lock record",
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if length == 0 {
        return Ok(None);
    }
    if length != usize_to_u64(WRITER_LOCK_RECORD_BYTES) {
        return Err(KeeperError::WriterLockCorrupted {
            path: path.to_path_buf(),
            detail: format!(
                "v1 record must be exactly {WRITER_LOCK_RECORD_BYTES} bytes, found {length}"
            ),
        });
    }
    let mut bytes = [0_u8; WRITER_LOCK_RECORD_BYTES];
    file.seek(SeekFrom::Start(0))
        .and_then(|_| file.read_exact(&mut bytes))
        .map_err(|source| KeeperError::Io {
            operation: "read writer-lock record",
            path: path.to_path_buf(),
            source,
        })?;
    WriterLockRecord::from_bytes(&bytes)
        .map(Some)
        .map_err(|detail| KeeperError::WriterLockCorrupted {
            path: path.to_path_buf(),
            detail,
        })
}

fn write_writer_lock_record(
    path: &Path,
    file: &mut File,
    record: WriterLockRecord,
) -> Result<(), KeeperError> {
    let bytes = record.to_bytes();
    file.set_len(0)
        .and_then(|()| file.seek(SeekFrom::Start(0)).map(|_| ()))
        .and_then(|()| file.write_all(&bytes))
        .and_then(|()| file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "persist writer-lock record",
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(unix)]
fn writer_pid_is_dead(pid: u32) -> bool {
    let Ok(raw_pid) = i32::try_from(pid) else {
        return false;
    };
    let Some(pid) = rustix::process::Pid::from_raw(raw_pid) else {
        return false;
    };
    matches!(
        rustix::process::test_kill_process(pid),
        Err(source) if source == rustix::io::Errno::SRCH
    )
}

#[cfg(not(unix))]
fn writer_pid_is_dead(_: u32) -> bool {
    false
}

#[cfg(unix)]
fn ensure_writer_lock_identity(admission: &WriterAdmissionInner) -> Result<(), KeeperError> {
    use std::os::unix::fs::MetadataExt;

    let metadata =
        std::fs::symlink_metadata(&admission.lock_path).map_err(|source| KeeperError::Io {
            operation: "verify writer-lock pathname",
            path: admission.lock_path.clone(),
            source,
        })?;
    if !metadata.file_type().is_file()
        || metadata.dev() != admission.lock_device
        || metadata.ino() != admission.lock_inode
    {
        return Err(KeeperError::WriterLockCorrupted {
            path: admission.lock_path.clone(),
            detail: "LOCK pathname no longer resolves to the flocked inode".to_owned(),
        });
    }
    Ok(())
}

#[cfg(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
))]
fn release_writer_admission(admission: &mut WriterAdmissionInner) {
    use rustix::fs::{FlockOperation, flock};

    if read_writer_lock_record(&admission.lock_path, &mut admission.lock_file)
        .is_ok_and(|record| record == Some(admission.record))
    {
        let _ = admission.lock_file.set_len(0);
        let _ = admission.lock_file.sync_all();
    }
    let _ = flock(&admission.lock_file, FlockOperation::Unlock);
}

#[cfg(not(all(
    unix,
    not(any(
        target_os = "espidf",
        target_os = "horizon",
        target_os = "solaris",
        target_os = "vita",
        target_os = "wasi"
    ))
)))]
fn release_writer_admission(_: &mut WriterAdmissionInner) {}

fn recover_writer_directory(
    admission: &Arc<WriterAdmissionInner>,
    schema: SchemaDescriptor,
    protection: &WriterProtection,
) -> Result<(), KeeperError> {
    admission.ensure_directory_identity()?;
    #[cfg(feature = "durability")]
    if let WriterProtection::Enabled(protector) = protection {
        recover_durable_manifest_slots(admission, schema, protector)?;
    }
    recover_interrupted_generation_claims(admission)?;
    recover_corrupt_primary_slot(admission)?;
    #[cfg(feature = "durability")]
    if let WriterProtection::Enabled(protector) = protection {
        recover_durable_writer_files(admission, schema, protector)?;
    }
    #[cfg(not(feature = "durability"))]
    {
        let _ = schema;
        let _ = protection;
    }
    admission.ensure_directory_identity()
}

fn recover_interrupted_generation_claims(
    admission: &Arc<WriterAdmissionInner>,
) -> Result<(), KeeperError> {
    let recovered_generation = match load_manifest_pair(&admission.directory) {
        Ok(loaded) => loaded.manifest.generation,
        Err(KeeperError::IndexNotFound { .. }) => 0,
        Err(error) => return Err(error),
    };
    let next_generation = recovered_generation.checked_add(1);
    for (path, generation) in scan_generation_claims(&admission.directory)? {
        if generation > recovered_generation && Some(generation) != next_generation {
            return Err(KeeperError::InvalidRecoveryClaim {
                path,
                recovered: recovered_generation,
                claimed: generation,
            });
        }
        if Some(generation) == next_generation {
            retire_interrupted_manifest_temp(admission, generation)?;
        }
        remove_stale_generation_claim(admission, &path)?;
    }
    Ok(())
}

fn retire_interrupted_manifest_temp(
    admission: &WriterAdmissionInner,
    generation: u64,
) -> Result<(), KeeperError> {
    let source = admission
        .directory
        .join(format!(".tmp-manifest-{generation}"));
    let retired = admission.directory.join(format!(
        ".tmp-abandoned-manifest-{generation}-{:016x}",
        admission.record.pid_start_nonce
    ));
    let changed = retire_regular_artifact(admission, &source, &retired)?;
    let source_sidecar = append_path_suffix(&source, ".fec");
    let retired_sidecar = append_path_suffix(&retired, ".fec");
    let changed = retire_regular_artifact(admission, &source_sidecar, &retired_sidecar)? || changed;
    if changed {
        admission
            .directory_file
            .sync_all()
            .map_err(|source| KeeperError::Io {
                operation: "fsync abandoned MANIFEST temp retirement",
                path: admission.directory.clone(),
                source,
            })?;
    }
    Ok(())
}

fn recover_corrupt_primary_slot(admission: &WriterAdmissionInner) -> Result<(), KeeperError> {
    let current = admission.directory.join("MANIFEST");
    let previous = admission.directory.join("MANIFEST.prev");
    let current_slot = read_manifest_slot(&current)?;
    let previous_slot = read_manifest_slot(&previous)?;
    if !matches!(current_slot, ManifestSlot::Invalid(_))
        || !matches!(previous_slot, ManifestSlot::Valid(_))
    {
        return Ok(());
    }

    let retired = admission.directory.join(format!(
        ".tmp-corrupt-manifest-{:016x}",
        admission.record.pid_start_nonce
    ));
    let changed = retire_regular_artifact(admission, &current, &retired)?;
    let current_sidecar = append_path_suffix(&current, ".fec");
    let retired_sidecar = append_path_suffix(&retired, ".fec");
    let changed =
        retire_regular_artifact(admission, &current_sidecar, &retired_sidecar)? || changed;
    if changed {
        admission
            .directory_file
            .sync_all()
            .map_err(|source| KeeperError::Io {
                operation: "fsync corrupt MANIFEST retirement",
                path: admission.directory.clone(),
                source,
            })?;
    }
    Ok(())
}

#[cfg(feature = "durability")]
fn recover_durable_manifest_slots(
    admission: &WriterAdmissionInner,
    schema: SchemaDescriptor,
    protector: &FileProtector,
) -> Result<(), KeeperError> {
    let current_path = admission.directory.join("MANIFEST");
    let previous_path = admission.directory.join("MANIFEST.prev");
    let expected_schema = schema
        .schema_id()
        .map_err(|source| KeeperError::InvalidSchema { source })?;

    let (mut current, mut current_repair, current_recovery_error) =
        durable_manifest_candidate(protector, &current_path, expected_schema)?.into_parts();
    let (mut previous, mut previous_repair, previous_recovery_error) =
        durable_manifest_candidate(protector, &previous_path, expected_schema)?.into_parts();
    if current.is_none() && previous.is_none() {
        if let Some(error) = current_recovery_error {
            return Err(*error);
        }
        if let Some(error) = previous_recovery_error {
            return Err(*error);
        }
    }
    if let (Some(current_manifest), Some(previous_manifest)) = (&current, &previous) {
        let pair = validate_manifest_pair(
            &admission.directory,
            current_manifest.clone(),
            previous_manifest,
        );
        if let Err(error) = pair {
            match (current_repair.is_some(), previous_repair.is_some()) {
                // Two pre-existing valid slots disagree: retain the existing
                // fail-closed corruption signal.
                (false, false) => return Err(error),
                // Reconstructed current is not allowed to displace the usable
                // pre-existing fallback when its transition is inconsistent.
                (true, false) => {
                    current = None;
                    current_repair = None;
                }
                // A reconstructed previous is optional while a pre-existing
                // current is usable. Leave the invalid on-disk slot untouched.
                (false, true) => {
                    previous = None;
                    previous_repair = None;
                }
                // If neither slot existed as valid authority, an inconsistent
                // pair of authenticated reconstructions has no safe winner.
                // Stale or swapped sidecars can make either generation look
                // plausible in isolation, so preserve the fail-closed signal.
                (true, true) => return Err(error),
            }
        }
    }
    // A reconstructed MANIFEST is not authority until every referenced
    // segment is readable or has itself been reconstructed and validated.
    // Prefer current, but never let an unusable optional fallback block a
    // usable current or let an unusable current displace a usable previous.
    match (&current, &previous) {
        (Some(current_manifest), _) => {
            match recover_durable_manifest_segments(admission, schema, protector, current_manifest)
            {
                Ok(()) => {
                    // Install a reconstructed fallback only when its own
                    // dependencies are sound. It is optional while current is
                    // usable, so a bad previous-only segment leaves the prior
                    // slot untouched instead of bricking writer admission.
                    if let (Some(bytes), Some(previous_manifest)) =
                        (previous_repair.as_deref(), previous.as_ref())
                        && recover_durable_manifest_segments(
                            admission,
                            schema,
                            protector,
                            previous_manifest,
                        )
                        .is_ok()
                    {
                        install_recovered_bytes(admission, "MANIFEST.prev", &previous_path, bytes)?;
                    }
                    if let Some(bytes) = &current_repair {
                        install_recovered_bytes(admission, "MANIFEST", &current_path, bytes)?;
                    }
                }
                Err(current_error) => {
                    // A pre-existing valid current was already reader-visible;
                    // do not mask that corruption here. A reconstructed current
                    // can instead be skipped when previous is usable.
                    if current_repair.is_none() {
                        return Err(current_error);
                    }
                    let Some(previous_manifest) = &previous else {
                        return Err(current_error);
                    };
                    recover_durable_manifest_segments(
                        admission,
                        schema,
                        protector,
                        previous_manifest,
                    )?;
                    if let Some(bytes) = &previous_repair {
                        install_recovered_bytes(admission, "MANIFEST.prev", &previous_path, bytes)?;
                    }
                }
            }
        }
        (None, Some(previous_manifest)) => {
            recover_durable_manifest_segments(admission, schema, protector, previous_manifest)?;
            if let Some(bytes) = &previous_repair {
                install_recovered_bytes(admission, "MANIFEST.prev", &previous_path, bytes)?;
            }
        }
        (None, None) => {}
    }

    for path in [&current_path, &previous_path] {
        if let ManifestSlot::Valid(_) = read_manifest_slot(path)? {
            let bytes = std::fs::read(path).map_err(|source| KeeperError::Io {
                operation: "read valid MANIFEST for sidecar witness",
                path: path.clone(),
                source,
            })?;
            ensure_matching_durability_sidecar(admission, protector, path, &bytes)?;
        }
    }
    Ok(())
}

#[cfg(feature = "durability")]
enum DurableManifestCandidate {
    Unavailable,
    Existing(Manifest),
    Recovered { manifest: Manifest, bytes: Vec<u8> },
    RecoveryFailed(Box<KeeperError>),
}

#[cfg(feature = "durability")]
impl DurableManifestCandidate {
    fn into_parts(self) -> (Option<Manifest>, Option<Vec<u8>>, Option<Box<KeeperError>>) {
        match self {
            Self::Unavailable => (None, None, None),
            Self::Existing(manifest) => (Some(manifest), None, None),
            Self::Recovered { manifest, bytes } => (Some(manifest), Some(bytes), None),
            Self::RecoveryFailed(error) => (None, None, Some(error)),
        }
    }
}

#[cfg(feature = "durability")]
fn durable_manifest_candidate(
    protector: &FileProtector,
    path: &Path,
    expected_schema: u64,
) -> Result<DurableManifestCandidate, KeeperError> {
    if let ManifestSlot::Valid(manifest) = read_manifest_slot(path)? {
        if manifest.schema_id != expected_schema {
            return Err(KeeperError::SchemaMismatch {
                path: path.to_path_buf(),
                expected: expected_schema,
                found: manifest.schema_id,
            });
        }
        return Ok(DurableManifestCandidate::Existing(manifest));
    }

    let recovered = match recover_manifest_bytes(protector, path) {
        Ok(recovered) => recovered,
        Err(error) => {
            return Ok(DurableManifestCandidate::RecoveryFailed(Box::new(error)));
        }
    };
    let Some((recovered, bytes)) = recovered else {
        return Ok(DurableManifestCandidate::Unavailable);
    };
    if recovered.schema_id != expected_schema {
        return Ok(DurableManifestCandidate::RecoveryFailed(Box::new(
            KeeperError::SchemaMismatch {
                path: path.to_path_buf(),
                expected: expected_schema,
                found: recovered.schema_id,
            },
        )));
    }
    Ok(DurableManifestCandidate::Recovered {
        manifest: recovered,
        bytes,
    })
}

#[cfg(feature = "durability")]
fn recover_manifest_bytes(
    protector: &FileProtector,
    source: &Path,
) -> Result<Option<(Manifest, Vec<u8>)>, KeeperError> {
    let sidecar = FileProtector::sidecar_path(source);
    if !regular_artifact_exists(&sidecar, "inspect MANIFEST repair sidecar")? {
        return Ok(None);
    }
    match protector
        .recover_file_bytes(source, &sidecar)
        .map_err(|source_error| KeeperError::Durability {
            operation: "recover MANIFEST bytes",
            path: source.to_path_buf(),
            source: source_error,
        })? {
        FileRecoveryOutcome::Recovered { bytes, .. } => {
            let witness = FileSourceWitness::from_bytes(&bytes);
            if !protector
                .sidecar_matches_witness(&sidecar, witness)
                .map_err(|source_error| KeeperError::Durability {
                    operation: "validate staged MANIFEST repair witness",
                    path: sidecar.clone(),
                    source: source_error,
                })?
            {
                return Err(KeeperError::Io {
                    operation: "validate staged MANIFEST repair witness",
                    path: source.to_path_buf(),
                    source: io::Error::new(
                        io::ErrorKind::InvalidData,
                        "repaired MANIFEST does not match its complete-source sidecar",
                    ),
                });
            }
            let manifest = Manifest::from_bytes(&bytes).map_err(|source_error| {
                KeeperError::ManifestCorrupted {
                    path: source.to_path_buf(),
                    source: source_error,
                }
            })?;
            Ok(Some((manifest, bytes)))
        }
        FileRecoveryOutcome::NotNeeded | FileRecoveryOutcome::Unrecoverable { .. } => Ok(None),
    }
}

#[cfg(feature = "durability")]
fn recover_durable_writer_files(
    admission: &WriterAdmissionInner,
    schema: SchemaDescriptor,
    protector: &FileProtector,
) -> Result<(), KeeperError> {
    let expected_schema = schema
        .schema_id()
        .map_err(|source| KeeperError::InvalidSchema { source })?;
    let loaded = match load_manifest_pair(&admission.directory) {
        Ok(loaded) => loaded,
        Err(KeeperError::IndexNotFound { .. }) => return Ok(()),
        Err(error) => return Err(error),
    };
    validate_loaded_schema(&admission.directory, expected_schema, &loaded)?;
    validate_recovery_claims(&admission.directory, &loaded)?;

    recover_durable_manifest_segments(admission, schema, protector, &loaded.manifest)
}

#[cfg(feature = "durability")]
fn recover_durable_manifest_segments(
    admission: &WriterAdmissionInner,
    schema: SchemaDescriptor,
    protector: &FileProtector,
    manifest: &Manifest,
) -> Result<(), KeeperError> {
    for manifest_segment in &manifest.segments {
        let path = admission
            .directory
            .join(canonical_segment_name(manifest_segment.segment_id));
        match open_verified_segment(&path, manifest_segment, schema) {
            Ok(()) => {
                let bytes = std::fs::read(&path).map_err(|source| KeeperError::Io {
                    operation: "read valid segment for sidecar witness",
                    path: path.clone(),
                    source,
                })?;
                ensure_matching_durability_sidecar(admission, protector, &path, &bytes)?;
            }
            Err(original_error) => {
                let sidecar = FileProtector::sidecar_path(&path);
                if !regular_artifact_exists(&sidecar, "inspect segment repair sidecar")? {
                    return Err(original_error);
                }
                let label = format!("segment-{:016x}", manifest_segment.segment_id);
                let bytes =
                    match protector
                        .recover_file_bytes(&path, &sidecar)
                        .map_err(|source| KeeperError::Durability {
                            operation: "recover segment bytes",
                            path: path.clone(),
                            source,
                        })? {
                        FileRecoveryOutcome::Recovered { bytes, .. } => bytes,
                        FileRecoveryOutcome::NotNeeded
                        | FileRecoveryOutcome::Unrecoverable { .. } => {
                            return Err(original_error);
                        }
                    };
                let witness = FileSourceWitness::from_bytes(&bytes);
                if !protector
                    .sidecar_matches_witness(&sidecar, witness)
                    .map_err(|source| KeeperError::Durability {
                        operation: "validate staged segment repair witness",
                        path: sidecar.clone(),
                        source,
                    })?
                {
                    return Err(KeeperError::SegmentMetadataMismatch {
                        path: path.clone(),
                        detail: "repaired segment does not match its complete-source sidecar"
                            .to_owned(),
                    });
                }
                let reader = SegmentReader::from_bytes(&bytes, schema).map_err(|source| {
                    KeeperError::SegmentOpen {
                        path: path.clone(),
                        source,
                    }
                })?;
                reader.verify().map_err(|source| KeeperError::SegmentOpen {
                    path: path.clone(),
                    source,
                })?;
                validate_segment_witnesses(&path, manifest_segment, &reader)?;
                install_recovered_bytes(admission, &label, &path, &bytes)?;
                open_verified_segment(&path, manifest_segment, schema)?;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "durability")]
fn open_verified_segment(
    path: &Path,
    manifest: &ManifestSegment,
    schema: SchemaDescriptor,
) -> Result<(), KeeperError> {
    let reader =
        SegmentReader::open_published(path, schema).map_err(|source| KeeperError::SegmentOpen {
            path: path.to_path_buf(),
            source,
        })?;
    reader.verify().map_err(|source| KeeperError::SegmentOpen {
        path: path.to_path_buf(),
        source,
    })?;
    validate_segment_witnesses(path, manifest, &reader)
}

#[cfg(feature = "durability")]
fn ensure_matching_durability_sidecar(
    admission: &WriterAdmissionInner,
    protector: &FileProtector,
    source: &Path,
    bytes: &[u8],
) -> Result<(), KeeperError> {
    let witness = FileSourceWitness::from_bytes(bytes);
    let sidecar = FileProtector::sidecar_path(source);
    let sidecar_exists = regular_artifact_exists(&sidecar, "inspect durability sidecar")?;
    let sidecar_matches = sidecar_exists
        && protector
            .sidecar_matches_witness(&sidecar, witness)
            .unwrap_or(false);
    if sidecar_matches {
        return Ok(());
    }
    if sidecar_exists {
        let file_name = source
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("artifact");
        let retired = admission.directory.join(format!(
            ".tmp-stale-fec-{file_name}-{:016x}",
            admission.record.pid_start_nonce
        ));
        if retire_regular_artifact(admission, &sidecar, &retired)? {
            admission
                .directory_file
                .sync_all()
                .map_err(|source_error| KeeperError::Io {
                    operation: "fsync stale sidecar retirement",
                    path: admission.directory.clone(),
                    source: source_error,
                })?;
        }
    }
    protector
        .protect_file_with_witness(source, witness)
        .map_err(|source_error| KeeperError::Durability {
            operation: "regenerate durability sidecar",
            path: source.to_path_buf(),
            source: source_error,
        })?;
    if !protector
        .sidecar_matches_witness(&sidecar, witness)
        .map_err(|source_error| KeeperError::Durability {
            operation: "verify regenerated durability sidecar",
            path: sidecar.clone(),
            source: source_error,
        })?
    {
        return Err(KeeperError::Io {
            operation: "verify regenerated durability sidecar",
            path: sidecar,
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "regenerated sidecar does not match authoritative source bytes",
            ),
        });
    }
    Ok(())
}

#[cfg(feature = "durability")]
fn regular_artifact_exists(path: &Path, operation: &'static str) -> Result<bool, KeeperError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(true),
        Ok(_) => Err(KeeperError::Io {
            operation,
            path: path.to_path_buf(),
            source: io::Error::new(io::ErrorKind::InvalidData, "artifact is not a regular file"),
        }),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(KeeperError::Io {
            operation,
            path: path.to_path_buf(),
            source,
        }),
    }
}

#[cfg(all(
    feature = "durability",
    any(target_os = "linux", target_os = "android", target_vendor = "apple")
))]
fn install_recovered_bytes(
    admission: &WriterAdmissionInner,
    label: &str,
    destination: &Path,
    bytes: &[u8],
) -> Result<(), KeeperError> {
    install_recovered_bytes_with_hook(admission, label, destination, bytes, |_| Ok(()))
}

#[cfg(all(
    feature = "durability",
    any(target_os = "linux", target_os = "android", target_vendor = "apple")
))]
fn install_recovered_bytes_with_hook(
    admission: &WriterAdmissionInner,
    label: &str,
    destination: &Path,
    bytes: &[u8],
    before_atomic_install: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<(), KeeperError> {
    use rustix::fs::{
        AtFlags, CWD, FileType, Mode, OFlags, RenameFlags, linkat, openat, renameat_with, statat,
    };
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::MetadataExt;

    admission.ensure_directory_identity()?;
    if destination.parent() != Some(admission.directory.as_path()) {
        return Err(KeeperError::Io {
            operation: "validate recovered-byte destination",
            path: destination.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "repair destination is outside the admitted index directory",
            ),
        });
    }
    let expected_stat_size = i64::try_from(bytes.len()).map_err(|_| KeeperError::Io {
        operation: "validate recovered-byte length",
        path: destination.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::InvalidInput,
            "recovered bytes exceed the platform file-size range",
        ),
    })?;
    let destination_name =
        destination
            .file_name()
            .ok_or_else(|| KeeperError::UnsafeGarbagePath {
                path: destination.to_path_buf(),
            })?;

    let base = format!(
        ".tmp-repair-{label}-{:016x}",
        admission.record.pid_start_nonce
    );
    let mut collision = 0_u64;
    let (temporary_name, mut temporary_file, temporary_metadata) = loop {
        let name = if collision == 0 {
            OsString::from(&base)
        } else {
            OsString::from(format!("{base}.{collision}"))
        };
        safe_direct_child(&admission.directory, Path::new(&name))?;
        match openat(
            &admission.directory_file,
            &name,
            OFlags::WRONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR | Mode::RGRP | Mode::WGRP | Mode::ROTH | Mode::WOTH,
        ) {
            Ok(file) => {
                let file = File::from(file);
                let metadata = file.metadata().map_err(|source| KeeperError::Io {
                    operation: "inspect owned recovered-byte temp",
                    path: admission.directory.join(&name),
                    source,
                })?;
                break (name, file, metadata);
            }
            Err(source) if source == rustix::io::Errno::EXIST => {
                collision = collision.checked_add(1).ok_or_else(|| KeeperError::Io {
                    operation: "allocate recovered-byte temp",
                    path: admission.directory.join(&base),
                    source: io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "recovered-byte temp suffix space is exhausted",
                    ),
                })?;
            }
            Err(source) => {
                return Err(KeeperError::Io {
                    operation: "create recovered-byte temp",
                    path: admission.directory.join(&name),
                    source: io::Error::from(source),
                });
            }
        }
    };
    let temporary_path = admission.directory.join(&temporary_name);
    temporary_file
        .write_all(bytes)
        .and_then(|()| temporary_file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "persist recovered-byte temp",
            path: temporary_path.clone(),
            source,
        })?;
    if !temporary_metadata.file_type().is_file()
        || temporary_file
            .metadata()
            .map_err(|source| KeeperError::Io {
                operation: "reinspect recovered-byte temp",
                path: temporary_path.clone(),
                source,
            })?
            .len()
            != usize_to_u64(bytes.len())
    {
        return Err(KeeperError::Io {
            operation: "validate recovered-byte temp",
            path: temporary_path,
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "recovered-byte temp is not a regular file with the expected length",
            ),
        });
    }
    let descriptor_root = if cfg!(target_vendor = "apple") {
        "/dev/fd"
    } else {
        "/proc/self/fd"
    };
    let descriptor_path =
        PathBuf::from(format!("{descriptor_root}/{}", temporary_file.as_raw_fd()));
    let ready_base = format!(
        ".tmp-ready-repair-{label}-{:016x}",
        admission.record.pid_start_nonce
    );
    let mut ready_collision = 0_u64;
    let ready_name = loop {
        let name = if ready_collision == 0 {
            OsString::from(&ready_base)
        } else {
            OsString::from(format!("{ready_base}.{ready_collision}"))
        };
        safe_direct_child(&admission.directory, Path::new(&name))?;
        match linkat(
            CWD,
            &descriptor_path,
            &admission.directory_file,
            &name,
            AtFlags::SYMLINK_FOLLOW,
        ) {
            Ok(()) => break name,
            Err(error) if error == rustix::io::Errno::EXIST => {
                ready_collision =
                    ready_collision
                        .checked_add(1)
                        .ok_or_else(|| KeeperError::Io {
                            operation: "allocate recovered-byte ready link",
                            path: admission.directory.join(&ready_base),
                            source: io::Error::new(
                                io::ErrorKind::AlreadyExists,
                                "recovered-byte ready-link suffix space is exhausted",
                            ),
                        })?;
            }
            Err(error) => {
                return Err(KeeperError::Io {
                    operation: "create descriptor-bound recovered-byte ready link",
                    path: admission.directory.join(&name),
                    source: io::Error::from(error),
                });
            }
        }
    };
    let ready_path = admission.directory.join(&ready_name);
    let ready_stat = statat(
        &admission.directory_file,
        &ready_name,
        AtFlags::SYMLINK_NOFOLLOW,
    )
    .map_err(io::Error::from)
    .map_err(|source| KeeperError::Io {
        operation: "verify recovered-byte ready link",
        path: ready_path.clone(),
        source,
    })?;
    if ready_stat.st_dev != temporary_metadata.dev()
        || ready_stat.st_ino != temporary_metadata.ino()
        || ready_stat.st_size != expected_stat_size
    {
        return Err(KeeperError::Io {
            operation: "verify recovered-byte ready link",
            path: ready_path.clone(),
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "recovered-byte ready link does not name the owned inode",
            ),
        });
    }
    admission
        .directory_file
        .sync_all()
        .map_err(|source| KeeperError::Io {
            operation: "fsync recovered-byte staging links",
            path: admission.directory.clone(),
            source,
        })?;

    let destination_identity = match statat(
        &admission.directory_file,
        destination_name,
        AtFlags::SYMLINK_NOFOLLOW,
    ) {
        Ok(stat) if FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile => {
            Some((stat.st_dev, stat.st_ino, stat.st_size))
        }
        Ok(_) => {
            return Err(KeeperError::Io {
                operation: "inspect recovered-byte destination",
                path: destination.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "recovered-byte destination is not a regular file",
                ),
            });
        }
        Err(error) if error == rustix::io::Errno::NOENT => None,
        Err(error) => {
            return Err(KeeperError::Io {
                operation: "inspect recovered-byte destination",
                path: destination.to_path_buf(),
                source: io::Error::from(error),
            });
        }
    };
    before_atomic_install(&ready_path).map_err(|source| KeeperError::Io {
        operation: "run recovered-byte install checkpoint",
        path: ready_path.clone(),
        source,
    })?;
    let destination_exists = destination_identity.is_some();
    if destination_exists {
        // Threat model: writer admission excludes cooperating directory
        // mutators. A same-directory process that ignores LOCK can replace the
        // ready pathname; the inode check below detects and rolls that back
        // during an uninterrupted call, but POSIX has no inode-conditioned
        // exchange primitive that also makes such hostile substitution
        // crash-safe. Non-cooperating mutation of an admitted index directory
        // is therefore outside Keeper's crash-consistency contract.
        renameat_with(
            &admission.directory_file,
            &ready_name,
            &admission.directory_file,
            destination_name,
            RenameFlags::EXCHANGE,
        )
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "atomically exchange recovered bytes",
            path: destination.to_path_buf(),
            source,
        })?;
    } else {
        // With no canonical destination, link the still-open validated inode
        // directly into place. The ready pathname is deliberately not the
        // source, so substitution of that staging name cannot affect the bytes
        // installed at the canonical name.
        linkat(
            CWD,
            &descriptor_path,
            &admission.directory_file,
            destination_name,
            AtFlags::SYMLINK_FOLLOW,
        )
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "link descriptor-bound recovered bytes into place",
            path: destination.to_path_buf(),
            source,
        })?;
    }
    let destination_stat = statat(
        &admission.directory_file,
        destination_name,
        AtFlags::SYMLINK_NOFOLLOW,
    );
    let installed_owned_inode = destination_stat.as_ref().is_ok_and(|stat| {
        stat.st_dev == temporary_metadata.dev()
            && stat.st_ino == temporary_metadata.ino()
            && stat.st_size == expected_stat_size
    });
    if !installed_owned_inode {
        if !destination_exists {
            return Err(KeeperError::Io {
                operation: "verify descriptor-bound recovered-byte installation",
                path: destination.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "descriptor-bound recovered-byte destination changed after installation",
                ),
            });
        }
        renameat_with(
            &admission.directory_file,
            destination_name,
            &admission.directory_file,
            &ready_name,
            RenameFlags::EXCHANGE,
        )
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "roll back substituted recovered-byte installation",
            path: destination.to_path_buf(),
            source,
        })?;
        let restored = destination_identity.is_some_and(|(device, inode, size)| {
            statat(
                &admission.directory_file,
                destination_name,
                AtFlags::SYMLINK_NOFOLLOW,
            )
            .is_ok_and(|stat| stat.st_dev == device && stat.st_ino == inode && stat.st_size == size)
        });
        if !restored {
            return Err(KeeperError::Io {
                operation: "verify recovered-byte rollback",
                path: destination.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "recovered-byte rollback did not restore the prior authority",
                ),
            });
        }
        admission
            .directory_file
            .sync_all()
            .map_err(|source| KeeperError::Io {
                operation: "fsync recovered-byte rollback",
                path: admission.directory.clone(),
                source,
            })?;
        return Err(KeeperError::Io {
            operation: "verify installed recovered bytes",
            path: destination.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "installed recovered-byte pathname was substituted; prior authority restored",
            ),
        });
    }
    if destination_exists {
        let retired = admission.directory.join(format!(
            ".tmp-corrupt-{label}-{:016x}",
            admission.record.pid_start_nonce
        ));
        retire_regular_artifact(admission, &ready_path, &retired)?;
    }
    // Keep the exclusively-created temp name as a second link until ordinary
    // grace-period garbage collection retires it. Conditional unlink is not
    // available on Unix; leaving the temp avoids a stat-then-unlink race that
    // could remove a substituted pathname while consuming no extra data blocks.
    admission
        .directory_file
        .sync_all()
        .map_err(|source| KeeperError::Io {
            operation: "fsync staged repair installation",
            path: admission.directory.clone(),
            source,
        })
}

#[cfg(all(
    feature = "durability",
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple"))
))]
fn install_recovered_bytes(
    admission: &WriterAdmissionInner,
    _: &str,
    destination: &Path,
    _: &[u8],
) -> Result<(), KeeperError> {
    Err(KeeperError::Io {
        operation: "install recovered bytes",
        path: destination.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "no-replace recovered-byte installation is unsupported for {}",
                admission.directory.display()
            ),
        ),
    })
}

fn append_path_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn retire_regular_artifact(
    admission: &WriterAdmissionInner,
    source: &Path,
    destination: &Path,
) -> Result<bool, KeeperError> {
    admission.ensure_directory_identity()?;
    if source.parent() != Some(admission.directory.as_path())
        || destination.parent() != Some(admission.directory.as_path())
    {
        return Err(KeeperError::Io {
            operation: "validate retirement paths",
            path: source.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "retirement paths must be direct children of the admitted directory",
            ),
        });
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    {
        use rustix::fs::{AtFlags, FileType, RenameFlags, renameat_with, statat};

        let source_name = source
            .file_name()
            .ok_or_else(|| KeeperError::UnsafeGarbagePath {
                path: source.to_path_buf(),
            })?;
        let source_stat = match statat(
            &admission.directory_file,
            source_name,
            AtFlags::SYMLINK_NOFOLLOW,
        ) {
            Ok(stat) => stat,
            Err(error) if error == rustix::io::Errno::NOENT => return Ok(false),
            Err(error) => {
                return Err(KeeperError::Io {
                    operation: "inspect retirement source",
                    path: source.to_path_buf(),
                    source: io::Error::from(error),
                });
            }
        };
        if FileType::from_raw_mode(source_stat.st_mode) != FileType::RegularFile {
            return Err(KeeperError::Io {
                operation: "retire writer artifact",
                path: source.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "retirement source is not a regular file",
                ),
            });
        }

        let destination_name =
            destination
                .file_name()
                .ok_or_else(|| KeeperError::UnsafeGarbagePath {
                    path: destination.to_path_buf(),
                })?;
        let mut collision = 0_u64;
        loop {
            let candidate = if collision == 0 {
                destination_name.to_os_string()
            } else {
                let mut name = destination_name.to_os_string();
                name.push(format!(".{collision}"));
                name
            };
            match renameat_with(
                &admission.directory_file,
                source_name,
                &admission.directory_file,
                &candidate,
                RenameFlags::NOREPLACE,
            ) {
                Ok(()) => return Ok(true),
                Err(error) if error == rustix::io::Errno::EXIST => {
                    collision = collision.checked_add(1).ok_or_else(|| KeeperError::Io {
                        operation: "allocate retirement destination",
                        path: destination.to_path_buf(),
                        source: io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            "retirement destination suffix space is exhausted",
                        ),
                    })?;
                }
                Err(error) if error == rustix::io::Errno::NOENT => return Ok(false),
                Err(error) => {
                    return Err(KeeperError::Io {
                        operation: "atomically retire writer artifact",
                        path: admission.directory.join(candidate),
                        source: io::Error::from(error),
                    });
                }
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    Err(KeeperError::Io {
        operation: "retire writer artifact",
        path: source.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            "atomic no-replace retirement is unsupported on this platform",
        ),
    })
}

#[cfg(unix)]
fn remove_stale_generation_claim(
    admission: &WriterAdmissionInner,
    path: &Path,
) -> Result<(), KeeperError> {
    use rustix::fs::{AtFlags, Mode, OFlags, openat, statat, unlinkat};
    use std::os::unix::fs::MetadataExt;

    admission.ensure_directory_identity()?;
    let name = path
        .file_name()
        .ok_or_else(|| KeeperError::UnsafeGarbagePath {
            path: path.to_path_buf(),
        })?;
    let claim = openat(
        &admission.directory_file,
        name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(io::Error::from)
    .map_err(|source| KeeperError::Io {
        operation: "open stale generation claim",
        path: path.to_path_buf(),
        source,
    })?;
    let claim_file = File::from(claim);
    let metadata = claim_file.metadata().map_err(|source| KeeperError::Io {
        operation: "inspect stale generation claim",
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.file_type().is_file() || metadata.len() != 0 {
        return Err(KeeperError::InvalidClaimArtifact {
            path: path.to_path_buf(),
            detail: "claim must be a zero-length regular file".to_owned(),
        });
    }
    let stat = statat(&admission.directory_file, name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "revalidate stale generation claim",
            path: path.to_path_buf(),
            source,
        })?;
    if stat.st_dev != metadata.dev() || stat.st_ino != metadata.ino() || stat.st_size != 0 {
        return Err(KeeperError::InvalidClaimArtifact {
            path: path.to_path_buf(),
            detail: "claim pathname changed during stale recovery".to_owned(),
        });
    }
    unlinkat(&admission.directory_file, Path::new(name), AtFlags::empty())
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "remove stale generation claim",
            path: path.to_path_buf(),
            source,
        })?;
    admission
        .directory_file
        .sync_all()
        .map_err(|source| KeeperError::Io {
            operation: "fsync stale-claim removal",
            path: admission.directory.clone(),
            source,
        })
}

#[cfg(not(unix))]
fn remove_stale_generation_claim(
    admission: &WriterAdmissionInner,
    path: &Path,
) -> Result<(), KeeperError> {
    Err(KeeperError::Io {
        operation: "remove stale generation claim",
        path: path.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "stale claim recovery is unsupported for {}",
                admission.directory.display()
            ),
        ),
    })
}

/// Writer-owned garbage collection configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GarbageCollectionOptions {
    /// Minimum artifact age required before deletion.
    pub grace_period: Duration,
}

impl Default for GarbageCollectionOptions {
    fn default() -> Self {
        Self {
            grace_period: DEFAULT_GARBAGE_GRACE,
        }
    }
}

/// Deterministic report from one writer-locked garbage sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GarbageCollectionReport {
    /// Removed direct-child names, sorted bytewise by platform string order.
    pub removed: Vec<PathBuf>,
}

impl GarbageCollectionReport {
    /// Whether the sweep made no filesystem changes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.removed.is_empty()
    }
}

/// Load the current MANIFEST, falling back to `MANIFEST.prev` without writing.
///
/// A valid current slot wins. When both slots are valid, the previous
/// generation must equal the current generation (idempotent republish) or be
/// exactly one less. Missing current plus valid previous is the expected
/// between-renames crash window and is not `IndexNotFound`.
///
/// # Errors
///
/// Returns a typed not-found, corruption, invalid-pair, or I/O error.
pub fn load_manifest_pair(directory: impl AsRef<Path>) -> Result<LoadedManifest, KeeperError> {
    let directory = directory.as_ref();
    let current_path = directory.join("MANIFEST");
    let previous_path = directory.join("MANIFEST.prev");
    let current = match read_manifest_slot(&current_path)? {
        ManifestSlot::Valid(current) => {
            return load_valid_current(directory, &current_path, &previous_path, current);
        }
        current => current,
    };

    let previous = read_manifest_slot(&previous_path)?;
    match (current, previous) {
        (ManifestSlot::Missing, ManifestSlot::Valid(previous)) => Ok(LoadedManifest {
            manifest: previous,
            source: ManifestSource::PreviousAfterMissingCurrent,
        }),
        (ManifestSlot::Invalid(_), ManifestSlot::Valid(previous)) => Ok(LoadedManifest {
            manifest: previous,
            source: ManifestSource::PreviousAfterCorruptCurrent,
        }),
        (ManifestSlot::Missing, ManifestSlot::Missing) => Err(KeeperError::IndexNotFound {
            directory: directory.to_path_buf(),
        }),
        (ManifestSlot::Invalid(source), ManifestSlot::Missing) => {
            Err(KeeperError::ManifestCorrupted {
                path: current_path,
                source,
            })
        }
        (ManifestSlot::Missing, ManifestSlot::Invalid(source)) => {
            Err(KeeperError::ManifestCorrupted {
                path: previous_path,
                source,
            })
        }
        (ManifestSlot::Invalid(current), ManifestSlot::Invalid(previous)) => {
            Err(KeeperError::NoValidManifest {
                directory: directory.to_path_buf(),
                current: current.to_string(),
                previous: previous.to_string(),
            })
        }
        (ManifestSlot::Valid(_), _) => unreachable!("valid current returned above"),
    }
}

fn load_valid_current(
    directory: &Path,
    current_path: &Path,
    previous_path: &Path,
    current: Manifest,
) -> Result<LoadedManifest, KeeperError> {
    let previous = match read_manifest_slot(previous_path) {
        Ok(ManifestSlot::Valid(previous)) => previous,
        Ok(ManifestSlot::Missing | ManifestSlot::Invalid(_)) | Err(_) => {
            return Ok(LoadedManifest {
                manifest: current,
                source: ManifestSource::Current,
            });
        }
    };

    // Readers intentionally take no writer lock. Re-read the primary after the
    // secondary so a commit that advanced both slots cannot manufacture a
    // false pair error from two different publication instants.
    let stable_current = match read_manifest_slot(current_path) {
        Ok(ManifestSlot::Valid(stable_current)) if stable_current == current => stable_current,
        Ok(ManifestSlot::Valid(newer_current)) => {
            return Ok(LoadedManifest {
                manifest: newer_current,
                source: ManifestSource::Current,
            });
        }
        Ok(ManifestSlot::Missing | ManifestSlot::Invalid(_)) | Err(_) => {
            return Ok(LoadedManifest {
                manifest: current,
                source: ManifestSource::Current,
            });
        }
    };

    validate_manifest_pair(directory, stable_current, &previous)
}

fn validate_manifest_pair(
    directory: &Path,
    current: Manifest,
    previous: &Manifest,
) -> Result<LoadedManifest, KeeperError> {
    let adjacent = previous.generation == current.generation
        || previous
            .generation
            .checked_add(1)
            .is_some_and(|generation| generation == current.generation);
    if !adjacent {
        return Err(KeeperError::InvalidGenerationPair {
            directory: directory.to_path_buf(),
            current: current.generation,
            previous: previous.generation,
        });
    }
    if previous.generation == current.generation && previous != &current {
        return Err(KeeperError::InvalidManifestPair {
            directory: directory.to_path_buf(),
            current: current.generation,
            previous: previous.generation,
            detail: "equal generations have different canonical contents".to_owned(),
        });
    }
    if previous.schema_id != current.schema_id {
        return Err(KeeperError::InvalidManifestPair {
            directory: directory.to_path_buf(),
            current: current.generation,
            previous: previous.generation,
            detail: format!(
                "schema_id changed from {:#018x} to {:#018x}",
                previous.schema_id, current.schema_id
            ),
        });
    }
    if current.docid_high_watermark < previous.docid_high_watermark {
        return Err(KeeperError::InvalidManifestPair {
            directory: directory.to_path_buf(),
            current: current.generation,
            previous: previous.generation,
            detail: format!(
                "docid_high_watermark rolled back from {} to {}",
                previous.docid_high_watermark, current.docid_high_watermark
            ),
        });
    }
    if previous.generation != current.generation {
        validate_segment_transitions(previous, &current).map_err(|error| {
            KeeperError::InvalidManifestPair {
                directory: directory.to_path_buf(),
                current: current.generation,
                previous: previous.generation,
                detail: error.to_string(),
            }
        })?;
    }
    Ok(LoadedManifest {
        manifest: current,
        source: ManifestSource::Current,
    })
}

fn validate_loaded_schema(
    directory: &Path,
    expected: u64,
    loaded: &LoadedManifest,
) -> Result<(), KeeperError> {
    if loaded.manifest.schema_id == expected {
        return Ok(());
    }
    let slot = match loaded.source {
        ManifestSource::Current => "MANIFEST",
        ManifestSource::PreviousAfterMissingCurrent
        | ManifestSource::PreviousAfterCorruptCurrent => "MANIFEST.prev",
        ManifestSource::InMemory => "<in-memory>",
    };
    Err(KeeperError::SchemaMismatch {
        path: directory.join(slot),
        expected,
        found: loaded.manifest.schema_id,
    })
}

fn validate_recovery_claims(directory: &Path, loaded: &LoadedManifest) -> Result<(), KeeperError> {
    if loaded.source == ManifestSource::Current || loaded.source == ManifestSource::InMemory {
        return Ok(());
    }
    let admitted_claim = loaded.manifest.generation.checked_add(1);
    for (path, claimed) in scan_generation_claims(directory)? {
        if claimed <= loaded.manifest.generation || admitted_claim == Some(claimed) {
            continue;
        }
        return Err(KeeperError::InvalidRecoveryClaim {
            path,
            recovered: loaded.manifest.generation,
            claimed,
        });
    }
    Ok(())
}

fn scan_generation_claims(directory: &Path) -> Result<Vec<(PathBuf, u64)>, KeeperError> {
    let mut claims = Vec::new();
    let entries = std::fs::read_dir(directory).map_err(|source| KeeperError::Io {
        operation: "scan recovery claims",
        path: directory.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| KeeperError::Io {
            operation: "read recovery claim entry",
            path: directory.to_path_buf(),
            source,
        })?;
        let Some(claimed) = parse_claim_name(&entry.file_name()) else {
            continue;
        };
        let path = entry.path();
        let metadata = std::fs::symlink_metadata(&path).map_err(|source| KeeperError::Io {
            operation: "inspect generation claim",
            path: path.clone(),
            source,
        })?;
        if !metadata.file_type().is_file() || metadata.len() != 0 {
            return Err(KeeperError::InvalidClaimArtifact {
                path,
                detail: format!(
                    "claim must be a zero-length regular file, found type {:?} and length {}",
                    metadata.file_type(),
                    metadata.len()
                ),
            });
        }
        claims.try_reserve(1).map_err(|error| KeeperError::Io {
            operation: "allocate recovery claim inventory",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
        claims.push((path, claimed));
    }
    claims.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    Ok(claims)
}

fn validate_proposed_manifest_segments(
    directory: &Path,
    manifest: &Manifest,
    schema: SchemaDescriptor,
    protection: &WriterProtection,
) -> Result<(), KeeperError> {
    for manifest_segment in &manifest.segments {
        let path = directory.join(canonical_segment_name(manifest_segment.segment_id));
        let reader = SegmentReader::open_published(&path, schema).map_err(|source| {
            KeeperError::SegmentOpen {
                path: path.clone(),
                source,
            }
        })?;
        reader.verify().map_err(|source| KeeperError::SegmentOpen {
            path: path.clone(),
            source,
        })?;
        validate_segment_witnesses(&path, manifest_segment, &reader)?;
        #[cfg(feature = "durability")]
        if let WriterProtection::Enabled(protector) = protection {
            let sidecar = FileProtector::sidecar_path(&path);
            let verification = protector.verify_file(&path, &sidecar).map_err(|source| {
                KeeperError::Durability {
                    operation: "preflight durable segment sidecar",
                    path: path.clone(),
                    source,
                }
            })?;
            if !verification.healthy {
                return Err(KeeperError::SegmentMetadataMismatch {
                    path,
                    detail: "durable segment sidecar does not match the proposed FSLX authority"
                        .to_owned(),
                });
            }
        }
        #[cfg(not(feature = "durability"))]
        let _ = protection;
    }
    Ok(())
}

fn validate_segment_witnesses(
    path: &Path,
    manifest: &ManifestSegment,
    reader: &SegmentReader<impl AsRef<[u8]>>,
) -> Result<(), KeeperError> {
    let header = reader.header();
    let mismatch = if header.segment_id != manifest.segment_id {
        Some(format!(
            "header segment_id {:#018x} != manifest {:#018x}",
            header.segment_id, manifest.segment_id
        ))
    } else if reader.file_len() != manifest.file_len {
        Some(format!(
            "file length {} != manifest {}",
            reader.file_len(),
            manifest.file_len
        ))
    } else if reader.file_xxh3() != manifest.file_xxh3 {
        Some(format!(
            "trailer file_xxh3 {:#018x} != manifest {:#018x}",
            reader.file_xxh3(),
            manifest.file_xxh3
        ))
    } else if header.docid_lo != manifest.docid_lo || header.docid_hi != manifest.docid_hi {
        Some(format!(
            "header range [{}, {}) != manifest [{}, {})",
            header.docid_lo, header.docid_hi, manifest.docid_lo, manifest.docid_hi
        ))
    } else if header.doc_count != manifest.doc_count {
        Some(format!(
            "header doc_count {} != manifest {}",
            header.doc_count, manifest.doc_count
        ))
    } else {
        None
    };
    if let Some(detail) = mismatch {
        return Err(KeeperError::SegmentMetadataMismatch {
            path: path.to_path_buf(),
            detail,
        });
    }
    Ok(())
}

/// Remove unreachable Quill artifacts while the crate-internal caller holds
/// the writer lock.
///
/// This seam stays crate-private because callers must already hold the
/// `KeeperWriter` admission that owns the directory handle. The function first
/// opens and validates the selected snapshot, so a failed recovery never
/// performs garbage collection. Reader [`KeeperSnapshot::open`] never calls it.
/// Reachability is the union of every individually valid MANIFEST slot.
///
/// # Errors
///
/// Returns before deletion when recovery, directory enumeration, metadata, or
/// path-safety validation fails. An unlink or final directory fsync failure is
/// reported with the exact affected path.
pub(crate) fn collect_writer_garbage_under_lock(
    directory: impl AsRef<Path>,
    schema: SchemaDescriptor,
    options: GarbageCollectionOptions,
) -> Result<GarbageCollectionReport, KeeperError> {
    collect_writer_garbage_at(directory.as_ref(), schema, options, SystemTime::now())
}

fn collect_writer_garbage_at(
    directory: &Path,
    schema: SchemaDescriptor,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    collect_writer_garbage_at_platform(directory, schema, options, now)
}

#[cfg(not(unix))]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn collect_writer_garbage_at_platform(
    directory: &Path,
    _: SchemaDescriptor,
    _: GarbageCollectionOptions,
    _: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    ensure_atomic_publish_supported(directory)?;
    unreachable!("unsupported-platform guard always returns an error")
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn collect_writer_garbage_at_platform(
    directory: &Path,
    schema: SchemaDescriptor,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    let directory_file = open_gc_directory(directory)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    let snapshot = KeeperSnapshot::open(directory, schema)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    let live_segments = live_segment_names_at(&directory_file, directory, &snapshot)?;
    let current_generation = snapshot.loaded_manifest().manifest.generation;
    sweep_garbage_directory(
        &directory_file,
        directory,
        &live_segments,
        current_generation,
        options,
        now,
    )
}

/// Remove abandoned genesis artifacts when both durable slots are absent.
///
/// This is a separate writer-locked state because ordinary snapshot recovery
/// correctly returns [`KeeperError::IndexNotFound`] when neither slot exists.
/// Any canonical generation claim blocks deletion for writer-open recovery to
/// resolve. This helper remains reserved for a future writer-only doctor path.
#[allow(dead_code, reason = "reserved for a writer-only doctor path")]
pub(crate) fn collect_abandoned_genesis_garbage_under_lock(
    directory: impl AsRef<Path>,
    options: GarbageCollectionOptions,
) -> Result<GarbageCollectionReport, KeeperError> {
    collect_abandoned_genesis_garbage_at(directory.as_ref(), options, SystemTime::now())
}

#[allow(dead_code, reason = "reserved for a writer-only doctor path")]
fn collect_abandoned_genesis_garbage_at(
    directory: &Path,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    collect_abandoned_genesis_garbage_at_platform(directory, options, now)
}

#[cfg(not(unix))]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn collect_abandoned_genesis_garbage_at_platform(
    directory: &Path,
    _: GarbageCollectionOptions,
    _: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    ensure_atomic_publish_supported(directory)?;
    unreachable!("unsupported-platform guard always returns an error")
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn collect_abandoned_genesis_garbage_at_platform(
    directory: &Path,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    let directory_file = open_gc_directory(directory)?;
    ensure_gc_directory_identity(directory, &directory_file)?;
    for slot_name in ["MANIFEST", "MANIFEST.prev"] {
        let path = directory.join(slot_name);
        match read_manifest_slot_at(&directory_file, OsStr::new(slot_name), &path)? {
            ManifestSlot::Missing => {}
            ManifestSlot::Valid(_) => return Err(KeeperError::RecoveryRequired { path }),
            ManifestSlot::Invalid(source) => {
                return Err(KeeperError::ManifestCorrupted { path, source });
            }
        }
    }
    ensure_gc_directory_identity(directory, &directory_file)?;
    sweep_garbage_directory(&directory_file, directory, &HashSet::new(), 0, options, now)
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn sweep_garbage_directory(
    directory_file: &File,
    directory: &Path,
    live_segments: &HashSet<OsString>,
    current_generation: u64,
    options: GarbageCollectionOptions,
    now: SystemTime,
) -> Result<GarbageCollectionReport, KeeperError> {
    use rustix::fs::{AtFlags, Dir, FileType, statat, unlinkat};
    use std::os::unix::ffi::OsStringExt;

    let mut candidates = Vec::<(OsString, GarbageCandidate, rustix::fs::Stat)>::new();
    let mut entries = Dir::read_from(directory_file)
        .map_err(io::Error::from)
        .map_err(|source| KeeperError::Io {
            operation: "scan garbage candidates",
            path: directory.to_path_buf(),
            source,
        })?;
    while let Some(entry) = entries.read() {
        let entry = entry
            .map_err(io::Error::from)
            .map_err(|source| KeeperError::Io {
                operation: "read garbage candidate",
                path: directory.to_path_buf(),
                source,
            })?;
        let name = OsString::from_vec(entry.file_name().to_bytes().to_vec());
        let Some(candidate) = classify_garbage_candidate(&name) else {
            continue;
        };
        let candidate_path = directory.join(&name);
        let stat = match statat(directory_file, entry.file_name(), AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) => stat,
            Err(source) if source == rustix::io::Errno::NOENT => continue,
            Err(source) => {
                return Err(KeeperError::Io {
                    operation: "inspect garbage candidate",
                    path: candidate_path,
                    source: io::Error::from(source),
                });
            }
        };
        let is_regular = FileType::from_raw_mode(stat.st_mode) == FileType::RegularFile;
        if let GarbageCandidate::Claim { generation } = &candidate {
            if !is_regular || stat.st_size != 0 {
                return Err(KeeperError::InvalidClaimArtifact {
                    path: candidate_path,
                    detail: format!(
                        "claim must be a zero-length regular file, found type {:?} and length {}",
                        FileType::from_raw_mode(stat.st_mode),
                        stat.st_size
                    ),
                });
            }
            if *generation > current_generation {
                return Err(KeeperError::ClaimedGenerationPending {
                    path: candidate_path,
                    current: current_generation,
                    claimed: *generation,
                });
            }
        }
        if is_regular {
            candidates.try_reserve(1).map_err(|error| KeeperError::Io {
                operation: "allocate garbage candidate inventory",
                path: directory.to_path_buf(),
                source: io::Error::other(error.to_string()),
            })?;
            candidates.push((name, candidate, stat));
        }
    }

    let mut removable_segments = HashSet::<OsString>::new();
    removable_segments
        .try_reserve(candidates.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate removable segment inventory",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    for (name, candidate, stat) in &candidates {
        if matches!(candidate, GarbageCandidate::Segment)
            && !live_segments.contains(name)
            && stat_old_enough(stat, now, options.grace_period)
        {
            removable_segments.insert(name.clone());
        }
    }

    let mut removals = Vec::<OsString>::new();
    removals
        .try_reserve_exact(candidates.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate garbage removal inventory",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    for (name, candidate, stat) in candidates {
        let old_enough = stat_old_enough(&stat, now, options.grace_period);
        let remove = match candidate {
            GarbageCandidate::Temporary => old_enough,
            GarbageCandidate::Segment => removable_segments.contains(&name),
            GarbageCandidate::Sidecar { base } => {
                old_enough
                    && sidecar_is_orphan_at(directory_file, directory, &removable_segments, &base)?
            }
            GarbageCandidate::Claim { .. } => old_enough,
        };
        if remove {
            removals.push(name);
        }
    }
    removals.sort_unstable();

    let mut removed = Vec::new();
    removed
        .try_reserve_exact(removals.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate garbage report",
            path: directory.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    for name in removals {
        let relative = Path::new(&name);
        let path = safe_direct_child(directory, relative)?;
        match unlinkat(directory_file, relative, AtFlags::empty()) {
            Ok(()) => removed.push(relative.to_path_buf()),
            Err(source) if source == rustix::io::Errno::NOENT => {}
            Err(source) => {
                if !removed.is_empty() {
                    sync_gc_directory(directory_file, directory)?;
                }
                return Err(KeeperError::Io {
                    operation: "remove garbage candidate",
                    path,
                    source: io::Error::from(source),
                });
            }
        }
    }
    if !removed.is_empty() {
        sync_gc_directory(directory_file, directory)?;
    }
    Ok(GarbageCollectionReport { removed })
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn live_segment_names_at(
    directory_file: &File,
    directory: &Path,
    snapshot: &KeeperSnapshot,
) -> Result<HashSet<OsString>, KeeperError> {
    let mut live = HashSet::new();
    for slot_name in ["MANIFEST", "MANIFEST.prev"] {
        let path = directory.join(slot_name);
        if let ManifestSlot::Valid(manifest) =
            read_manifest_slot_at(directory_file, OsStr::new(slot_name), &path)?
        {
            live.try_reserve(manifest.segments.len())
                .map_err(|error| KeeperError::Io {
                    operation: "allocate live segment set",
                    path: directory.to_path_buf(),
                    source: io::Error::other(error.to_string()),
                })?;
            live.extend(
                manifest
                    .segments
                    .iter()
                    .map(|segment| OsString::from(canonical_segment_name(segment.segment_id))),
            );
        }
    }
    let selected_name = match snapshot.loaded_manifest().source {
        ManifestSource::Current => "MANIFEST",
        ManifestSource::PreviousAfterMissingCurrent
        | ManifestSource::PreviousAfterCorruptCurrent => "MANIFEST.prev",
        ManifestSource::InMemory => {
            return Err(KeeperError::GarbageDirectoryChanged {
                directory: directory.to_path_buf(),
            });
        }
    };
    let selected_path = directory.join(selected_name);
    match read_manifest_slot_at(directory_file, OsStr::new(selected_name), &selected_path)? {
        ManifestSlot::Valid(manifest) if manifest == snapshot.loaded_manifest().manifest => {}
        ManifestSlot::Missing | ManifestSlot::Invalid(_) | ManifestSlot::Valid(_) => {
            return Err(KeeperError::GarbageDirectoryChanged {
                directory: directory.to_path_buf(),
            });
        }
    }
    Ok(live)
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
#[derive(Debug)]
enum GarbageCandidate {
    Temporary,
    Segment,
    Sidecar { base: OsString },
    Claim { generation: u64 },
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn classify_garbage_candidate(name: &OsStr) -> Option<GarbageCandidate> {
    let name = name.to_str()?;
    if name.starts_with(".tmp-") {
        return Some(GarbageCandidate::Temporary);
    }
    if let Some(base) = name.strip_suffix(".fec.tmp")
        && (parse_segment_name(base).is_some() || matches!(base, "MANIFEST" | "MANIFEST.prev"))
    {
        return Some(GarbageCandidate::Temporary);
    }
    if parse_segment_name(name).is_some() {
        return Some(GarbageCandidate::Segment);
    }
    if let Some(base) = name.strip_suffix(".fec")
        && (parse_segment_name(base).is_some() || matches!(base, "MANIFEST" | "MANIFEST.prev"))
    {
        return Some(GarbageCandidate::Sidecar {
            base: OsString::from(base),
        });
    }
    parse_claim_name(OsStr::new(name)).map(|generation| GarbageCandidate::Claim { generation })
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn sidecar_is_orphan_at(
    directory_file: &File,
    directory: &Path,
    removable_segments: &HashSet<OsString>,
    base: &OsStr,
) -> Result<bool, KeeperError> {
    use rustix::fs::{AtFlags, statat};

    let base_text = base.to_str();
    if base_text.is_some_and(|name| parse_segment_name(name).is_some())
        && removable_segments.contains(base)
    {
        return Ok(true);
    }
    match statat(directory_file, base, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(_) => Ok(false),
        Err(source) if source == rustix::io::Errno::NOENT => Ok(true),
        Err(source) => Err(KeeperError::Io {
            operation: "inspect sidecar base",
            path: directory.join(base),
            source: io::Error::from(source),
        }),
    }
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn stat_old_enough(stat: &rustix::fs::Stat, now: SystemTime, grace_period: Duration) -> bool {
    stat_modified_time(stat).is_some_and(|modified| {
        now.duration_since(modified)
            .is_ok_and(|age| age >= grace_period)
    })
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn stat_modified_time(stat: &rustix::fs::Stat) -> Option<SystemTime> {
    let seconds = stat.st_mtime;
    let nanoseconds = u32::try_from(stat.st_mtime_nsec).ok()?;
    if nanoseconds >= 1_000_000_000 {
        return None;
    }
    if seconds >= 0 {
        return SystemTime::UNIX_EPOCH
            .checked_add(Duration::new(u64::try_from(seconds).ok()?, nanoseconds));
    }
    let magnitude = seconds.unsigned_abs();
    let before_epoch = if nanoseconds == 0 {
        Duration::from_secs(magnitude)
    } else {
        Duration::new(magnitude.checked_sub(1)?, 1_000_000_000 - nanoseconds)
    };
    SystemTime::UNIX_EPOCH.checked_sub(before_epoch)
}

#[cfg(unix)]
fn open_gc_directory(directory: &Path) -> Result<File, KeeperError> {
    use rustix::fs::{Mode, OFlags, open};

    open(
        directory,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map(File::from)
    .map_err(|source| KeeperError::Io {
        operation: "open no-follow garbage-collection directory",
        path: directory.to_path_buf(),
        source: io::Error::from(source),
    })
}

#[cfg(unix)]
fn ensure_gc_directory_identity(directory: &Path, opened: &File) -> Result<(), KeeperError> {
    use std::os::unix::fs::MetadataExt;

    let path_metadata = std::fs::metadata(directory).map_err(|source| KeeperError::Io {
        operation: "verify garbage-collection directory",
        path: directory.to_path_buf(),
        source,
    })?;
    let opened_metadata = opened.metadata().map_err(|source| KeeperError::Io {
        operation: "inspect opened garbage-collection directory",
        path: directory.to_path_buf(),
        source,
    })?;
    if path_metadata.dev() != opened_metadata.dev()
        || path_metadata.ino() != opened_metadata.ino()
        || !opened_metadata.is_dir()
    {
        return Err(KeeperError::GarbageDirectoryChanged {
            directory: directory.to_path_buf(),
        });
    }
    Ok(())
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn sync_gc_directory(opened: &File, directory: &Path) -> Result<(), KeeperError> {
    opened.sync_all().map_err(|source| KeeperError::Io {
        operation: "fsync garbage-collection directory",
        path: directory.to_path_buf(),
        source,
    })
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn safe_direct_child(directory: &Path, relative: &Path) -> Result<PathBuf, KeeperError> {
    let mut components = relative.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(KeeperError::UnsafeGarbagePath {
            path: relative.to_path_buf(),
        });
    }
    Ok(directory.join(relative))
}

fn canonical_segment_name(segment_id: u64) -> String {
    format!("seg-{segment_id:016x}.fslx")
}

/// Internal fault-injection points in immutable segment publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SegmentPublishCheckpoint {
    /// The synced temp file was renamed to its canonical published name.
    SegmentRenamed,
    /// The index directory was fsynced after the segment rename.
    DirectorySynced,
    /// The complete-source repair sidecar was emitted and made durable.
    SidecarEmitted,
}

/// Publish one already-synced segment without durability sidecar I/O.
///
/// The first public accumulator-to-seal commit path activates this primitive
/// in the dependent G1a milestone. Until then it remains a tested Keeper
/// checkpoint building block.
#[allow(dead_code, reason = "activated by the dependent G1a commit milestone")]
pub(crate) fn publish_pending_segment(pending: PendingSegmentFile) -> Result<PathBuf, KeeperError> {
    publish_pending_segment_with_observer(pending, |_, _| Ok(()), |_, _| Ok(None), |_, _| Ok(()))
}

/// Publish one already-synced segment and emit its complete-source sidecar.
#[cfg(feature = "durability")]
#[allow(dead_code, reason = "activated by the dependent G1a commit milestone")]
pub(crate) fn publish_pending_segment_durable(
    pending: PendingSegmentFile,
    protector: &FileProtector,
) -> Result<PathBuf, KeeperError> {
    publish_pending_segment_durable_with_observer(pending, protector, |_, _| Ok(()))
}

#[cfg(feature = "durability")]
fn publish_pending_segment_durable_with_observer<O>(
    pending: PendingSegmentFile,
    protector: &FileProtector,
    observe: O,
) -> Result<PathBuf, KeeperError>
where
    O: FnMut(SegmentPublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    publish_pending_segment_with_observer(
        pending,
        |_, published| ensure_durable_segment_sidecar_absent(published),
        |pending, published| {
            let witness = FileSourceWitness::new(pending.file_len(), pending.source_xxh3());
            let result = protector
                .protect_file_with_witness(published, witness)
                .map_err(|source| KeeperError::Durability {
                    operation: "protect published segment",
                    path: published.to_path_buf(),
                    source,
                })?;
            Ok(Some(result.sidecar_path))
        },
        observe,
    )
}

fn publish_pending_segment_with_observer<B, A, O>(
    pending: PendingSegmentFile,
    mut before_rename: B,
    mut after_directory_sync: A,
    mut observe: O,
) -> Result<PathBuf, KeeperError>
where
    B: FnMut(&PendingSegmentFile, &Path) -> Result<(), KeeperError>,
    A: FnMut(&PendingSegmentFile, &Path) -> Result<Option<PathBuf>, KeeperError>,
    O: FnMut(SegmentPublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    let pending_path = pending.path().to_path_buf();
    let expected_temp = format!(".tmp-segment-{:016x}", pending.segment_id());
    if pending_path.file_name() != Some(OsStr::new(&expected_temp)) {
        return Err(KeeperError::Io {
            operation: "validate segment temp name",
            path: pending_path,
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "pending segment does not have its canonical temp name",
            ),
        });
    }
    let metadata = std::fs::symlink_metadata(&pending_path).map_err(|source| KeeperError::Io {
        operation: "inspect segment temp",
        path: pending_path.clone(),
        source,
    })?;
    if !metadata.file_type().is_file() || metadata.len() != pending.file_len() {
        return Err(KeeperError::Io {
            operation: "validate segment temp",
            path: pending_path.clone(),
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "pending segment is not a regular file with the encoded length",
            ),
        });
    }

    let directory = pending_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let published = directory.join(canonical_segment_name(pending.segment_id()));
    match std::fs::symlink_metadata(&published) {
        Err(source) if source.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect published segment destination",
                path: published,
                source,
            });
        }
        Ok(_) => {
            return Err(KeeperError::Io {
                operation: "publish segment",
                path: published,
                source: io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "published segment destination already exists",
                ),
            });
        }
    }

    before_rename(&pending, &published)?;
    std::fs::rename(&pending_path, &published).map_err(|source| KeeperError::Io {
        operation: "rename segment temp to published",
        path: pending_path,
        source,
    })?;
    observe(SegmentPublishCheckpoint::SegmentRenamed, &published)?;
    sync_directory(&directory)?;
    observe(SegmentPublishCheckpoint::DirectorySynced, &directory)?;
    if let Some(sidecar) = after_directory_sync(&pending, &published)? {
        observe(SegmentPublishCheckpoint::SidecarEmitted, &sidecar)?;
    }
    // Publication consumes the temp-file capability: after rename, callers
    // must not be able to reuse the pending handle for a second publish.
    drop(pending);
    Ok(published)
}

#[cfg(feature = "durability")]
fn ensure_durable_segment_sidecar_absent(published: &Path) -> Result<(), KeeperError> {
    let sidecar = FileProtector::sidecar_path(published);
    match std::fs::symlink_metadata(&sidecar) {
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(KeeperError::Io {
            operation: "inspect published segment sidecar destination",
            path: sidecar,
            source,
        }),
        Ok(_) => Err(KeeperError::Io {
            operation: "publish segment sidecar",
            path: sidecar,
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "published segment sidecar destination already exists",
            ),
        }),
    }
}

#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn parse_segment_name(name: &str) -> Option<u64> {
    let hex = name.strip_prefix("seg-")?.strip_suffix(".fslx")?;
    if hex.len() != 16
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

fn parse_claim_name(name: &OsStr) -> Option<u64> {
    let name = name.to_str()?;
    let digits = name.strip_prefix("gen-")?.strip_suffix(".claim")?;
    if digits.is_empty()
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
        || (digits.len() > 1 && digits.starts_with('0'))
    {
        return None;
    }
    digits.parse().ok()
}

struct GenerationClaimGuard {
    admission: Arc<WriterAdmissionInner>,
    name: OsString,
    claim_file: File,
    device: u64,
    inode: u64,
}

impl GenerationClaimGuard {
    #[cfg(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    ))]
    fn acquire(admission: Arc<WriterAdmissionInner>, generation: u64) -> Result<Self, KeeperError> {
        use rustix::fs::{AtFlags, Mode, OFlags, openat, statat};
        use std::os::unix::fs::MetadataExt;

        admission.ensure_directory_identity()?;
        let name = OsString::from(format!("gen-{generation}.claim"));
        let path = admission.directory.join(&name);
        let claim = match openat(
            &admission.directory_file,
            &name,
            OFlags::WRONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::CREATE | OFlags::EXCL,
            Mode::RUSR | Mode::WUSR,
        ) {
            Ok(claim) => claim,
            Err(source) if source == rustix::io::Errno::EXIST => {
                let stat = statat(&admission.directory_file, &name, AtFlags::SYMLINK_NOFOLLOW)
                    .map_err(io::Error::from)
                    .map_err(|source| KeeperError::Io {
                        operation: "inspect conflicting generation claim",
                        path: path.clone(),
                        source,
                    })?;
                let regular = rustix::fs::FileType::from_raw_mode(stat.st_mode)
                    == rustix::fs::FileType::RegularFile;
                if !regular || stat.st_size != 0 {
                    return Err(KeeperError::InvalidClaimArtifact {
                        path,
                        detail: format!(
                            "claim must be a zero-length regular file, found type {:?} and length {}",
                            rustix::fs::FileType::from_raw_mode(stat.st_mode),
                            stat.st_size
                        ),
                    });
                }
                return Err(KeeperError::GenerationClaimConflict { path, generation });
            }
            Err(source) => {
                return Err(KeeperError::Io {
                    operation: "create O_EXCL generation claim",
                    path,
                    source: io::Error::from(source),
                });
            }
        };
        let claim_file = File::from(claim);
        let metadata = claim_file.metadata().map_err(|source| KeeperError::Io {
            operation: "inspect owned generation claim",
            path: path.clone(),
            source,
        })?;
        if !metadata.file_type().is_file() || metadata.len() != 0 {
            return Err(KeeperError::InvalidClaimArtifact {
                path,
                detail: "new claim is not a zero-length regular file".to_owned(),
            });
        }
        claim_file.sync_all().map_err(|source| KeeperError::Io {
            operation: "fsync generation claim",
            path: path.clone(),
            source,
        })?;
        admission
            .directory_file
            .sync_all()
            .map_err(|source| KeeperError::Io {
                operation: "fsync generation-claim directory",
                path: admission.directory.clone(),
                source,
            })?;
        admission.ensure_directory_identity()?;
        Ok(Self {
            admission,
            name,
            device: metadata.dev(),
            inode: metadata.ino(),
            claim_file,
        })
    }

    #[cfg(not(all(
        unix,
        not(any(
            target_os = "espidf",
            target_os = "horizon",
            target_os = "solaris",
            target_os = "vita",
            target_os = "wasi"
        ))
    )))]
    fn acquire(admission: Arc<WriterAdmissionInner>, _: u64) -> Result<Self, KeeperError> {
        Err(KeeperError::Io {
            operation: "create generation claim",
            path: admission.directory.clone(),
            source: io::Error::new(
                io::ErrorKind::Unsupported,
                "O_EXCL generation claims require Unix filesystem semantics",
            ),
        })
    }
}

impl Drop for GenerationClaimGuard {
    fn drop(&mut self) {
        release_generation_claim(self);
    }
}

#[cfg(unix)]
fn release_generation_claim(claim: &GenerationClaimGuard) {
    use rustix::fs::{AtFlags, statat, unlinkat};

    // Keep the owned descriptor alive through the identity check. If an
    // external actor replaced the pathname, leave that replacement untouched.
    let _ = claim.claim_file.metadata();
    let Ok(stat) = statat(
        &claim.admission.directory_file,
        &claim.name,
        AtFlags::SYMLINK_NOFOLLOW,
    ) else {
        return;
    };
    if stat.st_dev != claim.device || stat.st_ino != claim.inode || stat.st_size != 0 {
        return;
    }
    if unlinkat(
        &claim.admission.directory_file,
        Path::new(&claim.name),
        AtFlags::empty(),
    )
    .is_ok()
    {
        let _ = claim.admission.directory_file.sync_all();
    }
}

#[cfg(not(unix))]
fn release_generation_claim(_: &GenerationClaimGuard) {}

/// In-process serializer for the two-slot MANIFEST publication protocol.
///
/// The owned asupersync guard moves into the blocking I/O closure. If the
/// awaiting task is cancelled after I/O starts, the guard remains held until
/// the atomic publication finishes, so a second writer cannot interleave the
/// two rename steps.
#[derive(Debug, Clone)]
struct ManifestPublisher {
    directory: PathBuf,
    publish_lock: Arc<Mutex<()>>,
}

#[derive(Debug, Clone)]
enum ManifestProtection {
    Disabled,
    #[cfg(feature = "durability")]
    Enabled(FileProtector),
}

impl ManifestPublisher {
    /// Bind a publisher to one Quill index directory.
    #[must_use]
    fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            publish_lock: global_publish_lock(),
        }
    }

    /// Publish one manifest using only the in-process serializer.
    ///
    /// This is a private substrate-test seam. Public mutation always goes
    /// through [`Self::publish_with_generation_claim`] while `KeeperWriter`
    /// retains `LOCK`. This ordinary path performs no sidecar I/O. If it follows
    /// a durable publication, an existing `MANIFEST.fec` may therefore be stale
    /// and must never be used as repair authority without matching the current
    /// source witness first.
    ///
    /// # Errors
    ///
    /// Returns typed validation, cancellation, transition, recovery, and I/O
    /// errors. A cancelled caller may observe an ambiguous result after the
    /// blocking phase starts; retrying is safe because generation validation
    /// reports whether the prior publication won.
    #[allow(dead_code, reason = "exercised by inline publication substrate tests")]
    async fn publish(&self, cx: &Cx, manifest: &Manifest) -> Result<LoadedManifest, KeeperError> {
        self.publish_with_generation_claim(cx, manifest, |_, _| Ok(()))
            .await
    }

    /// Publish with a caller-supplied cross-process generation claim.
    ///
    /// `claim` runs after the temp file is durable and before either slot is
    /// renamed. Its returned guard stays alive through the directory fsync;
    /// `KeeperWriter` can therefore release its claim from `Drop`.
    ///
    /// # Errors
    ///
    /// In addition to [`Self::publish`] failures, returns any typed error from
    /// the claim callback.
    async fn publish_with_generation_claim<C, F>(
        &self,
        cx: &Cx,
        manifest: &Manifest,
        claim: F,
    ) -> Result<LoadedManifest, KeeperError>
    where
        C: Send + 'static,
        F: FnOnce(&Path, u64) -> Result<C, KeeperError> + Send + 'static,
    {
        self.publish_with_generation_claim_and_protection(
            cx,
            manifest,
            claim,
            ManifestProtection::Disabled,
        )
        .await
    }

    /// Publish a MANIFEST and its complete-file FEC sidecar as one
    /// writer-serialized checkpoint.
    ///
    /// The new sidecar is encoded beside the durable temp MANIFEST before any
    /// slot replacement. The old previous sidecar is retired, then the current
    /// MANIFEST and its matching sidecar move to the previous slot before the
    /// new current slot and sidecar are installed ahead of the final directory
    /// fsync. This prevents a completed checkpoint from pairing a newer
    /// MANIFEST with an older current-slot sidecar. A failure after the
    /// MANIFEST rename but before sidecar installation can leave the new slot
    /// unprotected; the next `open_durable` recovery finishes that state while
    /// retaining its cross-process lock. The ordinary [`Self::publish`] path
    /// remains sidecar-free and private to substrate tests.
    ///
    /// # Errors
    ///
    /// Returns typed publication or durability errors. This method is private
    /// to substrate tests; public callers use the claimed variant below.
    #[cfg(feature = "durability")]
    #[allow(dead_code, reason = "exercised by inline durability substrate tests")]
    async fn publish_durable(
        &self,
        cx: &Cx,
        manifest: &Manifest,
        protector: &FileProtector,
    ) -> Result<LoadedManifest, KeeperError> {
        self.publish_durable_with_generation_claim(cx, manifest, protector, |_, _| Ok(()))
            .await
    }

    /// Durable publication with a caller-supplied cross-process generation
    /// claim.
    ///
    /// # Errors
    ///
    /// In addition to [`Self::publish_durable`] failures, returns any typed
    /// error from the claim callback.
    #[cfg(feature = "durability")]
    async fn publish_durable_with_generation_claim<C, F>(
        &self,
        cx: &Cx,
        manifest: &Manifest,
        protector: &FileProtector,
        claim: F,
    ) -> Result<LoadedManifest, KeeperError>
    where
        C: Send + 'static,
        F: FnOnce(&Path, u64) -> Result<C, KeeperError> + Send + 'static,
    {
        self.publish_with_generation_claim_and_protection(
            cx,
            manifest,
            claim,
            ManifestProtection::Enabled(protector.clone()),
        )
        .await
    }

    async fn publish_with_generation_claim_and_protection<C, F>(
        &self,
        cx: &Cx,
        manifest: &Manifest,
        claim: F,
        protection: ManifestProtection,
    ) -> Result<LoadedManifest, KeeperError>
    where
        C: Send + 'static,
        F: FnOnce(&Path, u64) -> Result<C, KeeperError> + Send + 'static,
    {
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        let directory = self.directory.clone();
        let publish_lock = Arc::clone(&self.publish_lock);
        let guard = OwnedMutexGuard::lock(publish_lock, cx)
            .await
            .map_err(|source| KeeperError::PublishLock { source })?;
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        // Encode only after admission to the process-global critical section.
        // Otherwise every waiter could retain its own 64 MiB output buffer.
        let bytes = manifest
            .to_bytes()
            .map_err(|source| KeeperError::InvalidManifest { source })?;
        if cx.is_cancel_requested() {
            return Err(KeeperError::PublishLock {
                source: LockError::Cancelled,
            });
        }
        spawn_blocking(move || {
            // Filesystem resolution is blocking I/O. Resolve the publication
            // target once inside the blocking phase, then use that same path
            // for inspection, the generation claim, renames, and directory
            // fsync so a late-created alias cannot split the operation.
            let directory = normalize_publish_directory(directory);
            match protection {
                ManifestProtection::Disabled => {
                    publish_manifest_locked(directory, &bytes, guard, claim)
                }
                #[cfg(feature = "durability")]
                ManifestProtection::Enabled(protector) => {
                    publish_manifest_durable_locked(directory, &bytes, guard, claim, &protector)
                }
            }
        })
        .await
    }

    #[cfg(test)]
    fn publish_lock_for_test(&self) -> Arc<Mutex<()>> {
        Arc::clone(&self.publish_lock)
    }
}

fn global_publish_lock() -> Arc<Mutex<()>> {
    Arc::clone(
        PUBLISH_LOCK.get_or_init(|| Arc::new(Mutex::with_name("quill.manifest_publish", ()))),
    )
}

fn normalize_publish_directory(directory: PathBuf) -> PathBuf {
    let absolute = if directory.is_absolute() {
        directory
    } else if let Ok(current_directory) = std::env::current_dir() {
        current_directory.join(directory)
    } else {
        directory
    };
    if let Ok(canonical) = std::fs::canonicalize(&absolute) {
        return canonical;
    }

    // Canonicalize the longest existing prefix so aliases through an existing
    // symlink resolve to one stable I/O target even when the final index
    // directory has not been created yet. Normalize the missing suffix
    // lexically afterwards.
    for ancestor in absolute.ancestors().skip(1) {
        let Ok(mut canonical) = std::fs::canonicalize(ancestor) else {
            continue;
        };
        let Ok(suffix) = absolute.strip_prefix(ancestor) else {
            continue;
        };
        canonical.push(suffix);
        return lexical_normalize(&canonical);
    }
    lexical_normalize(&absolute)
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if normalized
                    .file_name()
                    .is_some_and(|part| part != std::ffi::OsStr::new(".."))
                {
                    let _ = normalized.pop();
                } else if !path.is_absolute() {
                    normalized.push("..");
                }
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

fn publish_manifest_locked<C, F>(
    directory: PathBuf,
    bytes: &[u8],
    _guard: OwnedMutexGuard<()>,
    claim: F,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
{
    publish_manifest_choreography(directory, bytes, claim, |_, _| Ok(()))
}

#[cfg(feature = "durability")]
fn publish_manifest_durable_locked<C, F>(
    directory: PathBuf,
    bytes: &[u8],
    _guard: OwnedMutexGuard<()>,
    claim: F,
    protector: &FileProtector,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
{
    publish_manifest_durable_choreography(directory, bytes, claim, protector, |_, _| Ok(()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PublishCheckpoint {
    TempWritten,
    TempSynced,
    GenerationClaimed,
    CurrentMovedToPrevious,
    TempMovedToCurrent,
    DirectorySynced,
}

fn publish_manifest_choreography<C, F, O>(
    directory: PathBuf,
    bytes: &[u8],
    claim: F,
    observe: O,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
    O: FnMut(PublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    publish_manifest_choreography_with_hooks(
        directory,
        bytes,
        claim,
        |_, _, _| Ok(()),
        |_, _, _, _| Ok(()),
        |_, _| Ok(()),
        observe,
    )
}

#[cfg(feature = "durability")]
fn publish_manifest_durable_choreography<C, F, O>(
    directory: PathBuf,
    bytes: &[u8],
    claim: F,
    protector: &FileProtector,
    observe: O,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
    O: FnMut(PublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    let witness = FileSourceWitness::from_bytes(bytes);
    let proposed_generation = Manifest::from_bytes(bytes)
        .map_err(|source| KeeperError::InvalidManifest { source })?
        .generation;
    publish_manifest_choreography_with_hooks(
        directory,
        bytes,
        claim,
        |directory, rename_current, generation| {
            debug_assert_eq!(generation, proposed_generation);
            let temp_path = directory.join(format!(".tmp-manifest-{generation}"));
            protector
                .protect_file_with_witness(&temp_path, witness)
                .map_err(|source| KeeperError::Durability {
                    operation: "protect temp MANIFEST",
                    path: temp_path,
                    source,
                })?;
            prepare_manifest_sidecar_rotation(directory, rename_current, generation, protector)
        },
        complete_manifest_sidecar_rotation,
        |directory, current_path| {
            install_manifest_sidecar(directory, current_path, proposed_generation)
        },
        observe,
    )
}

fn publish_manifest_choreography_with_hooks<C, F, B, M, A, O>(
    directory: PathBuf,
    bytes: &[u8],
    claim: F,
    mut before_slot_renames: B,
    mut after_current_move: M,
    mut before_directory_sync: A,
    mut observe: O,
) -> Result<LoadedManifest, KeeperError>
where
    F: FnOnce(&Path, u64) -> Result<C, KeeperError>,
    B: FnMut(&Path, bool, u64) -> Result<(), KeeperError>,
    M: FnMut(&Path, &Path, &Path, u64) -> Result<(), KeeperError>,
    A: FnMut(&Path, &Path) -> Result<(), KeeperError>,
    O: FnMut(PublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    #[cfg(not(unix))]
    ensure_atomic_publish_supported(&directory)?;
    let proposed =
        Manifest::from_bytes(bytes).map_err(|source| KeeperError::InvalidManifest { source })?;
    let metadata = std::fs::metadata(&directory).map_err(|source| KeeperError::Io {
        operation: "inspect directory",
        path: directory.clone(),
        source,
    })?;
    if !metadata.is_dir() {
        return Err(KeeperError::Io {
            operation: "inspect directory",
            path: directory,
            source: io::Error::new(
                io::ErrorKind::NotADirectory,
                "index path is not a directory",
            ),
        });
    }

    let (expected_generation, rename_current, previous_manifest) =
        match load_manifest_pair(&directory) {
            Ok(loaded) => {
                if loaded.source == ManifestSource::PreviousAfterCorruptCurrent {
                    return Err(KeeperError::RecoveryRequired {
                        path: directory.join("MANIFEST"),
                    });
                }
                let expected = loaded.manifest.generation.checked_add(1).ok_or(
                    KeeperError::GenerationExhausted {
                        current: loaded.manifest.generation,
                    },
                )?;
                (
                    expected,
                    loaded.source == ManifestSource::Current,
                    Some(loaded.manifest),
                )
            }
            Err(KeeperError::IndexNotFound { .. }) => (1, false, None),
            Err(error) => return Err(error),
        };
    if proposed.generation != expected_generation {
        return Err(KeeperError::GenerationConflict {
            expected: expected_generation,
            proposed: proposed.generation,
        });
    }
    if let Some(previous) = &previous_manifest {
        if proposed.schema_id != previous.schema_id {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "schema_id changed from {:#018x} to {:#018x}",
                    previous.schema_id, proposed.schema_id
                ),
            });
        }
        if proposed.docid_high_watermark < previous.docid_high_watermark {
            return Err(KeeperError::InvalidTransition {
                detail: format!(
                    "docid_high_watermark rolled back from {} to {}",
                    previous.docid_high_watermark, proposed.docid_high_watermark
                ),
            });
        }
        validate_segment_transitions(previous, &proposed)?;
    }

    let temp_path = directory.join(format!(".tmp-manifest-{}", proposed.generation));
    let current_path = directory.join("MANIFEST");
    let previous_path = directory.join("MANIFEST.prev");
    prepare_manifest_temp(&temp_path, bytes, &mut observe)?;

    let _claim_guard = claim(&directory, proposed.generation)?;
    observe(PublishCheckpoint::GenerationClaimed, &directory)?;
    before_slot_renames(&directory, rename_current, proposed.generation)?;
    if rename_current {
        std::fs::rename(&current_path, &previous_path).map_err(|source| KeeperError::Io {
            operation: "rename current to previous",
            path: current_path.clone(),
            source,
        })?;
        after_current_move(
            &directory,
            &current_path,
            &previous_path,
            proposed.generation,
        )?;
        observe(PublishCheckpoint::CurrentMovedToPrevious, &previous_path)?;
    }
    std::fs::rename(&temp_path, &current_path).map_err(|source| KeeperError::Io {
        operation: "rename temp to current",
        path: temp_path,
        source,
    })?;
    observe(PublishCheckpoint::TempMovedToCurrent, &current_path)?;
    before_directory_sync(&directory, &current_path)?;
    sync_directory(&directory)?;
    observe(PublishCheckpoint::DirectorySynced, &directory)?;

    Ok(LoadedManifest {
        manifest: proposed,
        source: ManifestSource::Current,
    })
}

#[cfg(feature = "durability")]
fn prepare_manifest_sidecar_rotation(
    directory: &Path,
    rename_current: bool,
    proposed_generation: u64,
    protector: &FileProtector,
) -> Result<(), KeeperError> {
    let current = directory.join("MANIFEST.fec");
    let previous = directory.join("MANIFEST.prev.fec");
    let current_exists = regular_sidecar_exists(&current)?;
    let mut changed = false;

    if rename_current && regular_sidecar_exists(&previous)? {
        let retired = directory.join(format!(".tmp-manifest-previous-fec-{proposed_generation}"));
        changed = retire_manifest_sidecar(directory, &previous, &retired)? || changed;
    } else if !rename_current && current_exists {
        let retired = directory.join(format!(".tmp-manifest-current-fec-{proposed_generation}"));
        changed = retire_manifest_sidecar(directory, &current, &retired)? || changed;
    }

    if changed {
        sync_directory(directory)?;
    }
    if rename_current {
        let current_manifest = directory.join("MANIFEST");
        let current_bytes = std::fs::read(&current_manifest).map_err(|source| KeeperError::Io {
            operation: "read current MANIFEST for sidecar rotation",
            path: current_manifest.clone(),
            source,
        })?;
        let current_witness = FileSourceWitness::from_bytes(&current_bytes);
        let sidecar_matches = current_exists
            && protector
                .sidecar_matches_witness(&current, current_witness)
                .unwrap_or(false);
        if current_exists && !sidecar_matches {
            let retired =
                directory.join(format!(".tmp-manifest-current-fec-{proposed_generation}"));
            if retire_manifest_sidecar(directory, &current, &retired)? {
                sync_directory(directory)?;
            }
        }
        if !sidecar_matches {
            protector
                .protect_file_with_witness(&current_manifest, current_witness)
                .map_err(|source| KeeperError::Durability {
                    operation: "protect prior MANIFEST before sidecar rotation",
                    path: current_manifest,
                    source,
                })?;
        }
    }
    Ok(())
}

#[cfg(feature = "durability")]
fn complete_manifest_sidecar_rotation(
    directory: &Path,
    current_path: &Path,
    previous_path: &Path,
    _: u64,
) -> Result<(), KeeperError> {
    let current_sidecar = FileProtector::sidecar_path(current_path);
    if !regular_sidecar_exists(&current_sidecar)? {
        return Ok(());
    }
    let previous_sidecar = FileProtector::sidecar_path(previous_path);
    ensure_sidecar_absent(&previous_sidecar)?;
    std::fs::rename(&current_sidecar, &previous_sidecar).map_err(|source| KeeperError::Io {
        operation: "move current MANIFEST sidecar to previous",
        path: current_sidecar,
        source,
    })?;
    sync_directory(directory)
}

#[cfg(feature = "durability")]
fn install_manifest_sidecar(
    directory: &Path,
    current_path: &Path,
    proposed_generation: u64,
) -> Result<(), KeeperError> {
    let temp_path = directory.join(format!(".tmp-manifest-{proposed_generation}"));
    let temp_sidecar = FileProtector::sidecar_path(&temp_path);
    let current_sidecar = FileProtector::sidecar_path(current_path);
    ensure_sidecar_absent(&current_sidecar)?;
    std::fs::rename(&temp_sidecar, &current_sidecar).map_err(|source| KeeperError::Io {
        operation: "install current MANIFEST sidecar",
        path: temp_sidecar,
        source,
    })
}

#[cfg(feature = "durability")]
fn regular_sidecar_exists(path: &Path) -> Result<bool, KeeperError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(true),
        Ok(_) => Err(KeeperError::Io {
            operation: "inspect MANIFEST sidecar",
            path: path.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidData,
                "MANIFEST sidecar is not a regular file",
            ),
        }),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(KeeperError::Io {
            operation: "inspect MANIFEST sidecar",
            path: path.to_path_buf(),
            source,
        }),
    }
}

#[cfg(feature = "durability")]
fn retire_manifest_sidecar(
    directory: &Path,
    source: &Path,
    destination: &Path,
) -> Result<bool, KeeperError> {
    if source.parent() != Some(directory) || destination.parent() != Some(directory) {
        return Err(KeeperError::Io {
            operation: "validate MANIFEST sidecar retirement paths",
            path: source.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "sidecar retirement paths must be direct children of the publish directory",
            ),
        });
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    {
        use rustix::fs::{AtFlags, FileType, RenameFlags, renameat_with, statat};

        let directory_file = File::open(directory).map_err(|source| KeeperError::Io {
            operation: "open MANIFEST publish directory",
            path: directory.to_path_buf(),
            source,
        })?;
        let source_name = source
            .file_name()
            .ok_or_else(|| KeeperError::UnsafeGarbagePath {
                path: source.to_path_buf(),
            })?;
        let source_stat = match statat(&directory_file, source_name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(stat) => stat,
            Err(error) if error == rustix::io::Errno::NOENT => return Ok(false),
            Err(error) => {
                return Err(KeeperError::Io {
                    operation: "inspect MANIFEST sidecar retirement source",
                    path: source.to_path_buf(),
                    source: io::Error::from(error),
                });
            }
        };
        if FileType::from_raw_mode(source_stat.st_mode) != FileType::RegularFile {
            return Err(KeeperError::Io {
                operation: "retire MANIFEST sidecar",
                path: source.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "MANIFEST sidecar retirement source is not a regular file",
                ),
            });
        }

        let destination_name =
            destination
                .file_name()
                .ok_or_else(|| KeeperError::UnsafeGarbagePath {
                    path: destination.to_path_buf(),
                })?;
        let mut collision = 0_u64;
        loop {
            let candidate = if collision == 0 {
                destination_name.to_os_string()
            } else {
                let mut name = destination_name.to_os_string();
                name.push(format!(".{collision}"));
                name
            };
            match renameat_with(
                &directory_file,
                source_name,
                &directory_file,
                &candidate,
                RenameFlags::NOREPLACE,
            ) {
                Ok(()) => return Ok(true),
                Err(error) if error == rustix::io::Errno::EXIST => {
                    collision = collision.checked_add(1).ok_or_else(|| KeeperError::Io {
                        operation: "allocate MANIFEST sidecar retirement destination",
                        path: destination.to_path_buf(),
                        source: io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            "MANIFEST sidecar retirement suffix space is exhausted",
                        ),
                    })?;
                }
                Err(error) if error == rustix::io::Errno::NOENT => return Ok(false),
                Err(error) => {
                    return Err(KeeperError::Io {
                        operation: "atomically retire MANIFEST sidecar",
                        path: directory.join(candidate),
                        source: io::Error::from(error),
                    });
                }
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    Err(KeeperError::Io {
        operation: "retire MANIFEST sidecar",
        path: source.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            "atomic no-replace MANIFEST sidecar retirement is unsupported on this platform",
        ),
    })
}

#[cfg(feature = "durability")]
fn ensure_sidecar_absent(path: &Path) -> Result<(), KeeperError> {
    if regular_sidecar_exists(path)? {
        return Err(KeeperError::Io {
            operation: "install MANIFEST sidecar",
            path: path.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                "stale MANIFEST sidecar survived its rotation checkpoint",
            ),
        });
    }
    Ok(())
}

fn prepare_manifest_temp<O>(path: &Path, bytes: &[u8], observe: &mut O) -> Result<(), KeeperError>
where
    O: FnMut(PublishCheckpoint, &Path) -> Result<(), KeeperError>,
{
    let mut temp_file = match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
            verify_reusable_manifest_temp(path, bytes)?;
            observe(PublishCheckpoint::TempSynced, path)?;
            return Ok(());
        }
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "create temp",
                path: path.to_path_buf(),
                source,
            });
        }
    };
    temp_file
        .write_all(bytes)
        .map_err(|source| KeeperError::Io {
            operation: "write temp",
            path: path.to_path_buf(),
            source,
        })?;
    observe(PublishCheckpoint::TempWritten, path)?;
    temp_file.sync_all().map_err(|source| KeeperError::Io {
        operation: "fsync temp",
        path: path.to_path_buf(),
        source,
    })?;
    observe(PublishCheckpoint::TempSynced, path)
}

fn verify_reusable_manifest_temp(path: &Path, expected: &[u8]) -> Result<(), KeeperError> {
    let mut temp_file = open_existing_manifest_temp(path).map_err(|source| KeeperError::Io {
        operation: "open existing temp",
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = temp_file.metadata().map_err(|source| KeeperError::Io {
        operation: "stat existing temp",
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() || metadata.len() != usize_to_u64(expected.len()) {
        return Err(KeeperError::TempConflict {
            path: path.to_path_buf(),
        });
    }

    let mut existing = Vec::new();
    existing
        .try_reserve_exact(expected.len())
        .map_err(|error| KeeperError::Io {
            operation: "allocate existing temp buffer",
            path: path.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    Read::by_ref(&mut temp_file)
        .take(usize_to_u64(expected.len()).saturating_add(1))
        .read_to_end(&mut existing)
        .map_err(|source| KeeperError::Io {
            operation: "read existing temp",
            path: path.to_path_buf(),
            source,
        })?;
    if existing.as_slice() != expected {
        return Err(KeeperError::TempConflict {
            path: path.to_path_buf(),
        });
    }
    temp_file.sync_all().map_err(|source| KeeperError::Io {
        operation: "fsync existing temp",
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(unix)]
fn open_existing_manifest_temp(path: &Path) -> io::Result<File> {
    use rustix::fs::{Mode, OFlags, openat};

    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "temp path has no parent"))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "temp path has no file name"))?;
    let directory = File::open(parent)?;
    let temp_file = openat(
        &directory,
        file_name,
        OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(io::Error::from)?;
    Ok(File::from(temp_file))
}

#[cfg(not(unix))]
fn open_existing_manifest_temp(path: &Path) -> io::Result<File> {
    OpenOptions::new().read(true).write(true).open(path)
}

fn validate_manifest_successor(
    previous: &Manifest,
    proposed: &Manifest,
) -> Result<(), KeeperError> {
    let expected = previous
        .generation
        .checked_add(1)
        .ok_or(KeeperError::GenerationExhausted {
            current: previous.generation,
        })?;
    if proposed.generation != expected {
        return Err(KeeperError::GenerationConflict {
            expected,
            proposed: proposed.generation,
        });
    }
    if proposed.schema_id != previous.schema_id {
        return Err(KeeperError::InvalidTransition {
            detail: format!(
                "schema_id changed from {:#018x} to {:#018x}",
                previous.schema_id, proposed.schema_id
            ),
        });
    }
    if proposed.docid_high_watermark < previous.docid_high_watermark {
        return Err(KeeperError::InvalidTransition {
            detail: format!(
                "docid_high_watermark rolled back from {} to {}",
                previous.docid_high_watermark, proposed.docid_high_watermark
            ),
        });
    }
    validate_segment_transitions(previous, proposed)
}

fn validate_segment_transitions(
    previous: &Manifest,
    proposed: &Manifest,
) -> Result<(), KeeperError> {
    let previous_max_seal_seq = previous
        .segments
        .iter()
        .map(|segment| segment.seal_seq)
        .max();
    let mut previous_by_id = Vec::new();
    previous_by_id
        .try_reserve_exact(previous.segments.len())
        .map_err(|error| KeeperError::InvalidTransition {
            detail: format!("previous segment-index allocation failed: {error}"),
        })?;
    previous_by_id.extend(&previous.segments);
    previous_by_id.sort_unstable_by_key(|segment| segment.segment_id);

    let mut proposed_by_id = Vec::new();
    proposed_by_id
        .try_reserve_exact(proposed.segments.len())
        .map_err(|error| KeeperError::InvalidTransition {
            detail: format!("proposed segment-index allocation failed: {error}"),
        })?;
    proposed_by_id.extend(&proposed.segments);
    proposed_by_id.sort_unstable_by_key(|segment| segment.segment_id);

    let mut previous_index = 0;
    let mut proposed_index = 0;
    let mut segment_set_changed = false;
    while previous_index < previous_by_id.len() && proposed_index < proposed_by_id.len() {
        let old = previous_by_id[previous_index];
        let new = proposed_by_id[proposed_index];
        match old.segment_id.cmp(&new.segment_id) {
            std::cmp::Ordering::Less => {
                segment_set_changed = true;
                previous_index += 1;
            }
            std::cmp::Ordering::Greater => {
                validate_new_segment_seal_seq(previous_max_seal_seq, new)?;
                segment_set_changed = true;
                proposed_index += 1;
            }
            std::cmp::Ordering::Equal => {
                if old.seal_seq != new.seal_seq
                    || old.file_len != new.file_len
                    || old.file_xxh3 != new.file_xxh3
                    || old.docid_lo != new.docid_lo
                    || old.docid_hi != new.docid_hi
                    || old.doc_count != new.doc_count
                {
                    return Err(KeeperError::InvalidTransition {
                        detail: format!(
                            "immutable metadata changed for segment {:#018x}",
                            old.segment_id
                        ),
                    });
                }
                let monotone =
                    tombstones_are_subset(&old.tombstones, &new.tombstones).map_err(|error| {
                        KeeperError::InvalidTransition {
                            detail: format!(
                                "cannot compare tombstones for segment {:#018x}: {error}",
                                old.segment_id
                            ),
                        }
                    })?;
                if !monotone {
                    return Err(KeeperError::InvalidTransition {
                        detail: format!(
                            "tombstones shrank or changed for retained segment {:#018x}",
                            old.segment_id
                        ),
                    });
                }
                previous_index += 1;
                proposed_index += 1;
            }
        }
    }
    if previous_index != previous_by_id.len() {
        segment_set_changed = true;
    }
    for new in &proposed_by_id[proposed_index..] {
        validate_new_segment_seal_seq(previous_max_seal_seq, new)?;
        segment_set_changed = true;
    }
    if !segment_set_changed && previous.field_stats != proposed.field_stats {
        return Err(KeeperError::InvalidTransition {
            detail: "field-stat rollup changed while the immutable segment set was unchanged"
                .to_owned(),
        });
    }
    Ok(())
}

fn validate_new_segment_seal_seq(
    previous_max_seal_seq: Option<u64>,
    segment: &ManifestSegment,
) -> Result<(), KeeperError> {
    if let Some(previous_max) = previous_max_seal_seq
        && segment.seal_seq <= previous_max
    {
        return Err(KeeperError::InvalidTransition {
            detail: format!(
                "new segment {:#018x} seal_seq {} does not advance prior maximum {}",
                segment.segment_id, segment.seal_seq, previous_max
            ),
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct TombstoneContainer<'a> {
    chunk_id: u16,
    kind: u8,
    cardinality: u64,
    payload: &'a [u8],
}

struct TombstoneContainers<'a> {
    cursor: ByteCursor<'a>,
    remaining: usize,
}

impl<'a> TombstoneContainers<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self, ManifestCodecError> {
        let mut cursor = ByteCursor::new(bytes);
        let remaining =
            count_to_usize(cursor.u32()?, "tombstone chunk count", MAX_TOMBSTONE_CHUNKS)?;
        Ok(Self { cursor, remaining })
    }

    fn next_container(&mut self) -> Result<Option<TombstoneContainer<'a>>, ManifestCodecError> {
        if self.remaining == 0 {
            if self.cursor.remaining() != 0 {
                return Err(non_canonical(format!(
                    "tombstone set has {} trailing bytes",
                    self.cursor.remaining()
                )));
            }
            return Ok(None);
        }

        let chunk_id = self.cursor.u16()?;
        let kind = self.cursor.u8()?;
        let encoded_count = self.cursor.u16()?;
        let (cardinality, payload_length) = match kind {
            0 => (u64::from(encoded_count), usize::from(encoded_count) * 2),
            1 => (
                if encoded_count == 0 {
                    65_536
                } else {
                    u64::from(encoded_count)
                },
                8_192,
            ),
            other => {
                return Err(non_canonical(format!(
                    "tombstone container has unknown kind {other}"
                )));
            }
        };
        let payload = self.cursor.take(payload_length)?;
        self.remaining -= 1;
        Ok(Some(TombstoneContainer {
            chunk_id,
            kind,
            cardinality,
            payload,
        }))
    }
}

fn tombstones_are_subset(previous: &[u8], proposed: &[u8]) -> Result<bool, ManifestCodecError> {
    if previous == proposed {
        return Ok(true);
    }
    let mut previous = TombstoneContainers::new(previous)?;
    let mut proposed = TombstoneContainers::new(proposed)?;
    let mut old = previous.next_container()?;
    let mut new = proposed.next_container()?;

    while let Some(old_container) = old {
        while new.is_some_and(|container| container.chunk_id < old_container.chunk_id) {
            new = proposed.next_container()?;
        }
        let Some(new_container) = new else {
            return Ok(false);
        };
        if new_container.chunk_id != old_container.chunk_id
            || !tombstone_container_is_subset(old_container, new_container)
        {
            return Ok(false);
        }
        old = previous.next_container()?;
        new = proposed.next_container()?;
    }
    Ok(true)
}

fn tombstone_container_is_subset(
    previous: TombstoneContainer<'_>,
    proposed: TombstoneContainer<'_>,
) -> bool {
    if previous.cardinality > proposed.cardinality {
        return false;
    }
    match (previous.kind, proposed.kind) {
        (0, 0) => array_is_subset_of_array(previous.payload, proposed.payload),
        (0, 1) => array_is_subset_of_bitmap(previous.payload, proposed.payload),
        (1, 0) => bitmap_is_subset_of_array(previous.payload, proposed.payload),
        (1, 1) => previous
            .payload
            .iter()
            .zip(proposed.payload)
            .all(|(old, new)| old & !new == 0),
        _ => false,
    }
}

fn array_is_subset_of_array(previous: &[u8], proposed: &[u8]) -> bool {
    let mut proposed_index = 0;
    for previous_index in 0..previous.len() / 2 {
        let old = array_value(previous, previous_index);
        while proposed_index < proposed.len() / 2 && array_value(proposed, proposed_index) < old {
            proposed_index += 1;
        }
        if proposed_index == proposed.len() / 2 || array_value(proposed, proposed_index) != old {
            return false;
        }
        proposed_index += 1;
    }
    true
}

fn array_is_subset_of_bitmap(previous: &[u8], proposed: &[u8]) -> bool {
    (0..previous.len() / 2).all(|index| bitmap_contains(proposed, array_value(previous, index)))
}

fn bitmap_is_subset_of_array(previous: &[u8], proposed: &[u8]) -> bool {
    let mut proposed_index = 0;
    for (byte_index, byte) in previous.iter().copied().enumerate() {
        let mut bits = byte;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            let Ok(old) = u16::try_from(byte_index * 8 + bit) else {
                return false;
            };
            while proposed_index < proposed.len() / 2 && array_value(proposed, proposed_index) < old
            {
                proposed_index += 1;
            }
            if proposed_index == proposed.len() / 2 || array_value(proposed, proposed_index) != old
            {
                return false;
            }
            proposed_index += 1;
            bits &= bits - 1;
        }
    }
    true
}

fn array_value(bytes: &[u8], index: usize) -> u16 {
    let offset = index * 2;
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn bitmap_contains(bitmap: &[u8], value: u16) -> bool {
    let value = usize::from(value);
    bitmap[value / 8] & (1 << (value % 8)) != 0
}

enum ManifestSlot {
    Missing,
    Valid(Manifest),
    Invalid(ManifestCodecError),
}

fn read_manifest_slot(path: &Path) -> Result<ManifestSlot, KeeperError> {
    let path_metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(ManifestSlot::Missing);
        }
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect slot path",
                path: path.to_path_buf(),
                source,
            });
        }
    };
    if !path_metadata.file_type().is_file() {
        return Ok(ManifestSlot::Invalid(non_canonical(
            "manifest slot is not a regular file",
        )));
    }
    let file = match open_manifest_slot(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(ManifestSlot::Missing),
        #[cfg(unix)]
        Err(error) if error.raw_os_error() == Some(rustix::io::Errno::LOOP.raw_os_error()) => {
            return Ok(ManifestSlot::Invalid(non_canonical(
                "manifest slot became a symlink while opening",
            )));
        }
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "open",
                path: path.to_path_buf(),
                source,
            });
        }
    };
    read_manifest_file(path, file)
}

fn read_manifest_file(path: &Path, file: File) -> Result<ManifestSlot, KeeperError> {
    let file_length = file.metadata().map_err(|source| KeeperError::Io {
        operation: "stat",
        path: path.to_path_buf(),
        source,
    })?;
    if !file_length.file_type().is_file() {
        return Ok(ManifestSlot::Invalid(non_canonical(
            "manifest slot is not a regular file",
        )));
    }
    if file_length.len() > usize_to_u64(MAX_MANIFEST_BYTES) {
        return Ok(ManifestSlot::Invalid(ManifestCodecError::ResourceLimit {
            resource: "byte length",
            actual: file_length.len(),
            limit: usize_to_u64(MAX_MANIFEST_BYTES),
        }));
    }
    let reserve = usize::try_from(file_length.len()).unwrap_or(MAX_MANIFEST_BYTES);
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(reserve)
        .map_err(|error| KeeperError::Io {
            operation: "allocate read buffer",
            path: path.to_path_buf(),
            source: io::Error::other(error.to_string()),
        })?;
    let read_limit = usize_to_u64(MAX_MANIFEST_BYTES).saturating_add(1);
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|source| KeeperError::Io {
            operation: "read",
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Ok(ManifestSlot::Invalid(ManifestCodecError::ResourceLimit {
            resource: "byte length",
            actual: usize_to_u64(bytes.len()),
            limit: usize_to_u64(MAX_MANIFEST_BYTES),
        }));
    }
    Ok(match Manifest::from_bytes(&bytes) {
        Ok(manifest) => ManifestSlot::Valid(manifest),
        Err(error) => ManifestSlot::Invalid(error),
    })
}

#[cfg(unix)]
#[allow(dead_code, reason = "wired by the dependent writer-lock milestone")]
fn read_manifest_slot_at(
    directory: &File,
    name: &OsStr,
    path: &Path,
) -> Result<ManifestSlot, KeeperError> {
    use rustix::fs::{AtFlags, FileType, Mode, OFlags, openat, statat};

    let stat = match statat(directory, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(stat) => stat,
        Err(source) if source == rustix::io::Errno::NOENT => return Ok(ManifestSlot::Missing),
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "inspect slot path",
                path: path.to_path_buf(),
                source: io::Error::from(source),
            });
        }
    };
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Ok(ManifestSlot::Invalid(non_canonical(
            "manifest slot is not a regular file",
        )));
    }
    let file = match openat(
        directory,
        name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    ) {
        Ok(file) => File::from(file),
        Err(source) if source == rustix::io::Errno::NOENT => return Ok(ManifestSlot::Missing),
        Err(source) if source == rustix::io::Errno::LOOP => {
            return Ok(ManifestSlot::Invalid(non_canonical(
                "manifest slot became a symlink while opening",
            )));
        }
        Err(source) => {
            return Err(KeeperError::Io {
                operation: "open",
                path: path.to_path_buf(),
                source: io::Error::from(source),
            });
        }
    };
    read_manifest_file(path, file)
}

#[cfg(unix)]
fn open_manifest_slot(path: &Path) -> io::Result<File> {
    use rustix::fs::{Mode, OFlags, openat};

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "slot path has no file name"))?;
    let directory = File::open(parent)?;
    let slot = openat(
        &directory,
        file_name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(io::Error::from)?;
    Ok(File::from(slot))
}

#[cfg(not(unix))]
fn open_manifest_slot(path: &Path) -> io::Result<File> {
    File::open(path)
}

#[cfg(not(unix))]
fn ensure_atomic_publish_supported(directory: &Path) -> Result<(), KeeperError> {
    Err(KeeperError::Io {
        operation: "verify atomic publish support",
        path: directory.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::Unsupported,
            "MANIFEST replacement and directory fsync require Unix semantics",
        ),
    })
}

#[cfg(unix)]
fn sync_directory(directory: &Path) -> Result<(), KeeperError> {
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|source| KeeperError::Io {
            operation: "fsync directory",
            path: directory.to_path_buf(),
            source,
        })
}

#[cfg(not(unix))]
fn sync_directory(directory: &Path) -> Result<(), KeeperError> {
    ensure_atomic_publish_supported(directory)
}

#[derive(Debug, Clone, Copy)]
enum ErrorClass {
    Invalid,
    NonCanonical,
}

fn validate_manifest_shape(
    manifest: &Manifest,
    error_class: ErrorClass,
) -> Result<(), ManifestCodecError> {
    let reject = |detail: String| match error_class {
        ErrorClass::Invalid => ManifestCodecError::Invalid { detail },
        ErrorClass::NonCanonical => ManifestCodecError::NonCanonical { detail },
    };

    if manifest.generation == 0 {
        return Err(reject("generation zero is reserved".to_owned()));
    }
    if manifest.flags & !MANIFEST_KNOWN_FLAGS != 0 {
        return Err(reject(format!(
            "unknown flag bits {:#010x}",
            manifest.flags & !MANIFEST_KNOWN_FLAGS
        )));
    }
    if manifest.segments.len() > MAX_MANIFEST_SEGMENTS {
        return Err(ManifestCodecError::ResourceLimit {
            resource: "segment count",
            actual: usize_to_u64(manifest.segments.len()),
            limit: usize_to_u64(MAX_MANIFEST_SEGMENTS),
        });
    }
    if manifest.field_stats.len() > MAX_MANIFEST_FIELDS {
        return Err(ManifestCodecError::ResourceLimit {
            resource: "field count",
            actual: usize_to_u64(manifest.field_stats.len()),
            limit: usize_to_u64(MAX_MANIFEST_FIELDS),
        });
    }
    Ok(())
}

fn validate_manifest(
    manifest: &Manifest,
    error_class: ErrorClass,
) -> Result<(), ManifestCodecError> {
    validate_manifest_shape(manifest, error_class)?;
    let reject = |detail: String| match error_class {
        ErrorClass::Invalid => ManifestCodecError::Invalid { detail },
        ErrorClass::NonCanonical => ManifestCodecError::NonCanonical { detail },
    };

    let mut segment_ids = Vec::new();
    segment_ids
        .try_reserve_exact(manifest.segments.len())
        .map_err(|error| reject(format!("segment-id validation allocation failed: {error}")))?;
    segment_ids.extend(manifest.segments.iter().map(|segment| segment.segment_id));
    segment_ids.sort_unstable();
    if let Some([duplicate, ..]) = segment_ids.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(reject(format!("duplicate segment_id {duplicate:#018x}")));
    }

    let mut seal_sequences = Vec::new();
    seal_sequences
        .try_reserve_exact(manifest.segments.len())
        .map_err(|error| {
            reject(format!(
                "seal-sequence validation allocation failed: {error}"
            ))
        })?;
    seal_sequences.extend(manifest.segments.iter().map(|segment| segment.seal_seq));
    seal_sequences.sort_unstable();
    if let Some([duplicate, ..]) = seal_sequences.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(reject(format!("duplicate seal_seq {duplicate}")));
    }

    let mut previous_hi = None;
    let mut maximum_hi = 0_u64;
    let mut total_documents = 0_u64;
    for (index, segment) in manifest.segments.iter().enumerate() {
        if segment.file_len == 0 {
            return Err(reject(format!("segment {index} has zero file_len")));
        }
        if segment.docid_lo >= segment.docid_hi {
            return Err(reject(format!(
                "segment {index} has empty or reversed range [{}, {})",
                segment.docid_lo, segment.docid_hi
            )));
        }
        if segment.docid_hi > MAX_DOCID_EXCLUSIVE {
            return Err(reject(format!(
                "segment {index} range exceeds the u32 payload domain"
            )));
        }
        if previous_hi.is_some_and(|hi| segment.docid_lo < hi) {
            return Err(reject(format!(
                "segment {index} overlaps or is out of docid_lo order"
            )));
        }
        let span = segment.docid_hi - segment.docid_lo;
        if u64::from(segment.doc_count) > span {
            return Err(reject(format!(
                "segment {index} doc_count {} exceeds range span {span}",
                segment.doc_count
            )));
        }
        let tombstone_count = validate_tombstone_bytes(
            &segment.tombstones,
            Some((segment.docid_lo, segment.docid_hi)),
        )?;
        if tombstone_count > u64::from(segment.doc_count) {
            return Err(reject(format!(
                "segment {index} tombstone count {tombstone_count} exceeds doc_count {}",
                segment.doc_count
            )));
        }
        total_documents = total_documents
            .checked_add(u64::from(segment.doc_count))
            .ok_or_else(|| reject("aggregate segment doc_count overflow".to_owned()))?;
        maximum_hi = maximum_hi.max(segment.docid_hi);
        previous_hi = Some(segment.docid_hi);
    }
    if manifest.docid_high_watermark < maximum_hi {
        return Err(reject(format!(
            "docid_high_watermark {} is below maximum docid_hi {maximum_hi}",
            manifest.docid_high_watermark
        )));
    }

    let mut previous_field = None;
    for stats in &manifest.field_stats {
        if previous_field.is_some_and(|field| stats.field_ord <= field) {
            return Err(reject(format!(
                "field stats are not strictly ordered at field_ord {}",
                stats.field_ord
            )));
        }
        if u64::from(stats.doc_count) > total_documents {
            return Err(reject(format!(
                "field {} doc_count {} exceeds aggregate segment count {total_documents}",
                stats.field_ord, stats.doc_count
            )));
        }
        previous_field = Some(stats.field_ord);
    }
    Ok(())
}

fn validate_tombstone_bytes(
    bytes: &[u8],
    range: Option<(u64, u64)>,
) -> Result<u64, ManifestCodecError> {
    let mut cursor = ByteCursor::new(bytes);
    let count = consume_tombstone_set(&mut cursor, range)?;
    if cursor.remaining() != 0 {
        return Err(non_canonical(format!(
            "tombstone set has {} trailing bytes",
            cursor.remaining()
        )));
    }
    Ok(count)
}

fn consume_tombstone_set(
    cursor: &mut ByteCursor<'_>,
    range: Option<(u64, u64)>,
) -> Result<u64, ManifestCodecError> {
    let chunk_count = count_to_usize(cursor.u32()?, "tombstone chunk count", MAX_TOMBSTONE_CHUNKS)?;
    let mut previous_chunk = None;
    let mut total = 0_u64;
    for chunk_index in 0..chunk_count {
        let chunk_id = cursor.u16()?;
        if previous_chunk.is_some_and(|previous| chunk_id <= previous) {
            return Err(non_canonical(format!(
                "tombstone chunk {chunk_index} is not strictly ordered"
            )));
        }
        previous_chunk = Some(chunk_id);
        let kind = cursor.u8()?;
        let encoded_count = cursor.u16()?;
        let cardinality = match kind {
            0 => {
                if encoded_count == 0 {
                    return Err(non_canonical(format!(
                        "array tombstone chunk {chunk_index} is empty"
                    )));
                }
                if encoded_count > TOMBSTONE_ARRAY_MAX_CARDINALITY {
                    return Err(non_canonical(format!(
                        "array tombstone chunk {chunk_index} cardinality {encoded_count} exceeds promotion threshold {TOMBSTONE_ARRAY_MAX_CARDINALITY}"
                    )));
                }
                let count = usize::from(encoded_count);
                let mut previous_low = None;
                for _ in 0..count {
                    let low = cursor.u16()?;
                    if previous_low.is_some_and(|previous| low <= previous) {
                        return Err(non_canonical(format!(
                            "array tombstone chunk {chunk_index} is not strictly ordered"
                        )));
                    }
                    validate_tombstone_docid(chunk_id, low, range)?;
                    previous_low = Some(low);
                }
                u64::from(encoded_count)
            }
            1 => {
                let bitmap = cursor.take(8_192)?;
                let actual = bitmap
                    .iter()
                    .map(|byte| u64::from(byte.count_ones()))
                    .sum::<u64>();
                let expected = if encoded_count == 0 {
                    65_536
                } else {
                    u64::from(encoded_count)
                };
                if actual != expected {
                    return Err(non_canonical(format!(
                        "bitmap tombstone chunk {chunk_index} cardinality {actual} != {expected}"
                    )));
                }
                if actual < TOMBSTONE_BITMAP_MIN_CARDINALITY {
                    return Err(non_canonical(format!(
                        "bitmap tombstone chunk {chunk_index} cardinality {actual} is below demotion threshold {TOMBSTONE_BITMAP_MIN_CARDINALITY}"
                    )));
                }
                if let Some(bounds) = range {
                    validate_bitmap_range(bitmap, chunk_id, bounds)?;
                }
                actual
            }
            other => {
                return Err(non_canonical(format!(
                    "tombstone chunk {chunk_index} has unknown kind {other}"
                )));
            }
        };
        total = total
            .checked_add(cardinality)
            .ok_or_else(|| non_canonical("tombstone cardinality overflow"))?;
    }
    Ok(total)
}

fn validate_bitmap_range(
    bitmap: &[u8],
    chunk_id: u16,
    bounds: (u64, u64),
) -> Result<(), ManifestCodecError> {
    let (lo, hi) = bounds;
    let chunk_lo = u64::from(chunk_id) << 16;
    let chunk_hi = chunk_lo + 65_536;
    if lo <= chunk_lo && chunk_hi <= hi {
        return Ok(());
    }
    if hi <= chunk_lo || lo >= chunk_hi {
        return Err(non_canonical(format!(
            "bitmap tombstone chunk {chunk_id} is outside segment range [{lo}, {hi})"
        )));
    }

    for (byte_index, byte) in bitmap.iter().copied().enumerate() {
        let mut bits = byte;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            let low = u16::try_from(byte_index * 8 + bit as usize)
                .map_err(|_| non_canonical("bitmap low-bit overflow"))?;
            validate_tombstone_docid(chunk_id, low, Some(bounds))?;
            bits &= bits - 1;
        }
    }
    Ok(())
}

fn validate_tombstone_docid(
    chunk_id: u16,
    low: u16,
    range: Option<(u64, u64)>,
) -> Result<(), ManifestCodecError> {
    if let Some((lo, hi)) = range {
        let docid = (u64::from(chunk_id) << 16) | u64::from(low);
        if !(lo..hi).contains(&docid) {
            return Err(non_canonical(format!(
                "tombstoned docid {docid} is outside segment range [{lo}, {hi})"
            )));
        }
    }
    Ok(())
}

fn check_manifest_byte_limit(length: usize) -> Result<(), ManifestCodecError> {
    if length > MAX_MANIFEST_BYTES {
        return Err(ManifestCodecError::ResourceLimit {
            resource: "byte length",
            actual: usize_to_u64(length),
            limit: usize_to_u64(MAX_MANIFEST_BYTES),
        });
    }
    Ok(())
}

fn manifest_encoded_len(manifest: &Manifest) -> Result<usize, ManifestCodecError> {
    let length = MANIFEST_MIN_BYTES
        .checked_add(
            manifest
                .segments
                .len()
                .checked_mul(SEGMENT_FIXED_BYTES)
                .ok_or_else(|| invalid("manifest encoded length overflow"))?,
        )
        .and_then(|length| {
            manifest.segments.iter().try_fold(length, |total, segment| {
                total.checked_add(segment.tombstones.len())
            })
        })
        .and_then(|length| {
            manifest
                .field_stats
                .len()
                .checked_mul(FIELD_STATS_BYTES)
                .and_then(|stats| length.checked_add(stats))
        })
        .ok_or_else(|| invalid("manifest encoded length overflow"))?;
    check_manifest_byte_limit(length)?;
    Ok(length)
}

fn count_to_usize(
    count: u32,
    resource: &'static str,
    limit: usize,
) -> Result<usize, ManifestCodecError> {
    let count = usize::try_from(count).map_err(|_| ManifestCodecError::ResourceLimit {
        resource,
        actual: u64::from(count),
        limit: usize_to_u64(limit),
    })?;
    if count > limit {
        return Err(ManifestCodecError::ResourceLimit {
            resource,
            actual: usize_to_u64(count),
            limit: usize_to_u64(limit),
        });
    }
    Ok(count)
}

fn copy_bytes(bytes: &[u8], label: &'static str) -> Result<Vec<u8>, ManifestCodecError> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(bytes.len())
        .map_err(|error| non_canonical(format!("{label} allocation failed: {error}")))?;
    owned.extend_from_slice(bytes);
    Ok(owned)
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn invalid(detail: impl Into<String>) -> ManifestCodecError {
    ManifestCodecError::Invalid {
        detail: detail.into(),
    }
}

fn non_canonical(detail: impl Into<String>) -> ManifestCodecError {
    ManifestCodecError::NonCanonical {
        detail: detail.into(),
    }
}

fn put_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ByteCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    const fn position(&self) -> usize {
        self.position
    }

    const fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], ManifestCodecError> {
        let end =
            self.position
                .checked_add(length)
                .ok_or_else(|| ManifestCodecError::Truncated {
                    offset: self.position,
                    needed: length,
                    remaining: self.remaining(),
                })?;
        let Some(bytes) = self.bytes.get(self.position..end) else {
            return Err(ManifestCodecError::Truncated {
                offset: self.position,
                needed: length,
                remaining: self.remaining(),
            });
        };
        self.position = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, ManifestCodecError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ManifestCodecError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, ManifestCodecError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, ManifestCodecError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use asupersync::runtime::yield_now;
    use asupersync::types::Budget;
    use asupersync::{LabConfig, LabRuntime};
    #[cfg(feature = "durability")]
    use frankensearch_durability::{DefaultSymbolCodec, DurabilityConfig, FileHealth};
    use tempfile::tempdir;

    use crate::schema::{DEFAULT_SCHEMA, FSFS_CHUNK_SCHEMA};
    #[cfg(feature = "durability")]
    use crate::segment::SegmentWriteCheckpoint;
    use crate::segment::{EncodedSegment, SectionInput, SegmentHeaderInput};

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn array_tombstones(chunk_id: u16, lows: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::new();
        put_u32(&mut bytes, 1);
        put_u16(&mut bytes, chunk_id);
        bytes.push(0);
        put_u16(
            &mut bytes,
            u16::try_from(lows.len()).expect("test tombstone count fits u16"),
        );
        for low in lows {
            put_u16(&mut bytes, *low);
        }
        bytes
    }

    fn bitmap_tombstones(chunk_id: u16, lows: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::new();
        put_u32(&mut bytes, 1);
        put_u16(&mut bytes, chunk_id);
        bytes.push(1);
        let encoded_count = if lows.len() == 65_536 {
            0
        } else {
            u16::try_from(lows.len()).expect("test bitmap count fits u16")
        };
        put_u16(&mut bytes, encoded_count);
        bytes.resize(bytes.len() + 8_192, 0);
        let payload_start = bytes.len() - 8_192;
        for low in lows {
            let low = usize::from(*low);
            bytes[payload_start + low / 8] |= 1 << (low % 8);
        }
        bytes
    }

    fn sample_manifest(generation: u64) -> Manifest {
        Manifest {
            generation,
            docid_high_watermark: 70_000 + generation,
            schema_id: 0x1122_3344_5566_7788,
            engine_version: CURRENT_ENGINE_VERSION,
            flags: 0,
            segments: vec![
                ManifestSegment {
                    segment_id: 0x1001,
                    seal_seq: 7,
                    file_len: 4_096,
                    file_xxh3: 0xa1a2_a3a4_a5a6_a7a8,
                    docid_lo: 100,
                    docid_hi: 200,
                    doc_count: 3,
                    tombstones: array_tombstones(0, &[101, 150]),
                },
                ManifestSegment {
                    segment_id: 0x1002,
                    seal_seq: 8,
                    file_len: 8_192,
                    file_xxh3: 0xb1b2_b3b4_b5b6_b7b8,
                    docid_lo: 65_536,
                    docid_hi: 65_600,
                    doc_count: 2,
                    tombstones: EMPTY_TOMBSTONES.to_vec(),
                },
            ],
            field_stats: vec![
                ManifestFieldStats {
                    field_ord: 1,
                    total_tokens: 12,
                    doc_count: 5,
                },
                ManifestFieldStats {
                    field_ord: 3,
                    total_tokens: 42,
                    doc_count: 4,
                },
            ],
        }
    }

    fn rewrite_crc(bytes: &mut [u8]) {
        let body_len = bytes.len() - 4;
        let checksum = crc32fast::hash(&bytes[..body_len]);
        bytes[body_len..].copy_from_slice(&checksum.to_le_bytes());
    }

    fn write_manifest(path: &Path, manifest: &Manifest) -> TestResult {
        let bytes = manifest.to_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    fn encoded_test_segment(
        segment_id: u64,
        docid_lo: u64,
        docid_hi: u64,
        doc_count: u32,
    ) -> Result<EncodedSegment, QuillError> {
        EncodedSegment::encode(
            SegmentHeaderInput {
                segment_id,
                schema: DEFAULT_SCHEMA,
                docid_lo,
                docid_hi,
                doc_count,
                created_unix_s: 1_700_000_000,
                engine_version: CURRENT_ENGINE_VERSION,
            },
            &[
                SectionInput::new(SectionKind::TERMDICT, b"termdict"),
                SectionInput::new(SectionKind::POSTINGS, b"postings"),
                SectionInput::new(SectionKind::POSITIONS, b"positions"),
                SectionInput::new(SectionKind::BLOCKMAX, b"blockmax"),
                SectionInput::new(SectionKind::DOCLEN, b"doclen"),
                SectionInput::new(SectionKind::IDMAP, b"idmap"),
                SectionInput::new(SectionKind::IDHASH, b"idhash"),
                SectionInput::new(SectionKind::STOREDMETA, b"storedmeta"),
                SectionInput::new(SectionKind::STATS, b"stats"),
            ],
        )
    }

    fn write_test_segment(
        directory: &Path,
        segment_id: u64,
        seal_seq: u64,
        docid_lo: u64,
        docid_hi: u64,
    ) -> Result<ManifestSegment, Box<dyn std::error::Error>> {
        let encoded = encoded_test_segment(segment_id, docid_lo, docid_hi, 1)?;
        std::fs::write(
            directory.join(canonical_segment_name(segment_id)),
            encoded.as_bytes(),
        )?;
        Ok(manifest_segment(&encoded, seal_seq))
    }

    fn manifest_segment(encoded: &EncodedSegment, seal_seq: u64) -> ManifestSegment {
        let header = encoded.header();
        ManifestSegment {
            segment_id: header.segment_id,
            seal_seq,
            file_len: encoded.file_len(),
            file_xxh3: encoded.file_xxh3(),
            docid_lo: header.docid_lo,
            docid_hi: header.docid_hi,
            doc_count: header.doc_count,
            tombstones: EMPTY_TOMBSTONES.to_vec(),
        }
    }

    fn durable_test_manifest(generation: u64, segments: Vec<ManifestSegment>) -> Manifest {
        Manifest {
            generation,
            docid_high_watermark: segments
                .iter()
                .map(|segment| segment.docid_hi)
                .max()
                .unwrap_or(0),
            schema_id: DEFAULT_SCHEMA.schema_id().expect("valid test schema"),
            engine_version: CURRENT_ENGINE_VERSION,
            flags: 0,
            segments,
            field_stats: Vec::new(),
        }
    }

    #[cfg(feature = "durability")]
    fn test_file_protector() -> FileProtector {
        FileProtector::new(
            Arc::new(DefaultSymbolCodec),
            DurabilityConfig {
                symbol_size: 256,
                repair_overhead: 2.0,
                ..DurabilityConfig::default()
            },
        )
        .expect("valid test durability configuration")
    }

    fn directory_bytes(directory: &Path) -> Result<Vec<(OsString, Vec<u8>)>, io::Error> {
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                entries.push((entry.file_name(), std::fs::read(entry.path())?));
            }
        }
        entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        Ok(entries)
    }

    #[test]
    fn packed_engine_version_matches_crate_semver() {
        let major = env!("CARGO_PKG_VERSION_MAJOR")
            .parse::<u8>()
            .expect("crate major fits packed engine version");
        let minor = env!("CARGO_PKG_VERSION_MINOR")
            .parse::<u8>()
            .expect("crate minor fits packed engine version");
        let patch = env!("CARGO_PKG_VERSION_PATCH")
            .parse::<u16>()
            .expect("crate patch fits packed engine version");
        assert_eq!(
            CURRENT_ENGINE_VERSION,
            pack_engine_version(major, minor, patch)
        );
        assert_eq!(
            unpack_engine_version(CURRENT_ENGINE_VERSION),
            (major, minor, patch)
        );
    }

    #[test]
    fn referenced_segment_unknown_schema_maps_to_public_corruption() {
        let path = PathBuf::from("index/seg-0000000000000001.fslx");
        let error: SearchError = KeeperError::SegmentOpen {
            path: path.clone(),
            source: QuillError::UnknownSchema { schema_id: 0x55 },
        }
        .into();
        assert!(matches!(
            error,
            SearchError::IndexCorrupted { path: actual, detail }
                if actual == path && detail.contains("unknown schema")
        ));
    }

    #[test]
    fn empty_manifest_has_stable_wire_golden() -> TestResult {
        let manifest = Manifest::empty(1, 0x1122_3344_5566_7788, 0);
        let bytes = manifest.to_bytes()?;
        let expected = hex_bytes(
            "46534c584d414e0001000000010000000000000000000000000000008877665544332211\
             010002000000000000000000000000005846c4c1",
        );
        assert_eq!(bytes, expected);
        assert_eq!(Manifest::from_bytes(&bytes)?, manifest);
        Ok(())
    }

    #[test]
    fn zero_schema_hash_is_a_valid_roundtrip() -> TestResult {
        let manifest = Manifest::empty(1, 0, 0);
        assert_eq!(Manifest::from_bytes(&manifest.to_bytes()?)?, manifest);
        Ok(())
    }

    #[test]
    fn nonempty_manifest_has_stable_wire_golden() -> TestResult {
        let manifest = Manifest {
            generation: 0x0102_0304_0506_0708,
            docid_high_watermark: 20,
            schema_id: 0x1122_3344_5566_7788,
            engine_version: 0x0102_0304,
            flags: MANIFEST_FLAG_BULK_MODE_IN_PROGRESS,
            segments: vec![ManifestSegment {
                segment_id: 0x1020_3040_5060_7080,
                seal_seq: 0x1112_1314_1516_1718,
                file_len: 0x2122_2324_2526_2728,
                file_xxh3: 0x3132_3334_3536_3738,
                docid_lo: 10,
                docid_hi: 20,
                doc_count: 2,
                tombstones: array_tombstones(0, &[12, 18]),
            }],
            field_stats: vec![ManifestFieldStats {
                field_ord: 3,
                total_tokens: 0x4142_4344_4546_4748,
                doc_count: 2,
            }],
        };
        let bytes = manifest.to_bytes()?;
        let expected = hex_bytes(
            "46534c584d414e0001000000080706050403020114000000000000008877665544332211\
             040302010100000001000000807060504030201018171615141312112827262524232221\
             38373635343332310a0000000000000014000000000000000200000001000000000000\
             02000c0012000100000003004847464544434241020000007bcad46b",
        );
        assert_eq!(
            bytes, expected,
            "nonempty wire golden pins segment, tombstone, and stats field order"
        );
        assert_eq!(Manifest::from_bytes(&bytes)?, manifest);
        Ok(())
    }

    #[test]
    fn nonempty_manifest_roundtrips_exactly() -> TestResult {
        let manifest = sample_manifest(9);
        let bytes = manifest.to_bytes()?;
        assert_eq!(Manifest::from_bytes(&bytes)?, manifest);
        Ok(())
    }

    #[test]
    fn every_truncation_and_checksum_flip_is_rejected_without_panic() -> TestResult {
        let bytes = sample_manifest(3).to_bytes()?;
        for length in 0..bytes.len() {
            let parsed = std::panic::catch_unwind(|| Manifest::from_bytes(&bytes[..length]));
            assert!(parsed.is_ok(), "parser panicked at truncation {length}");
            assert!(parsed.expect("catch_unwind result").is_err());
        }
        let body = &bytes[..bytes.len() - 4];
        for length in 0..body.len() {
            let mut valid_crc_prefix = body[..length].to_vec();
            put_u32(&mut valid_crc_prefix, crc32fast::hash(&body[..length]));
            let parsed = std::panic::catch_unwind(|| Manifest::from_bytes(&valid_crc_prefix));
            assert!(
                parsed.is_ok(),
                "parser panicked at valid-CRC body truncation {length}"
            );
            assert!(parsed.expect("catch_unwind result").is_err());
        }
        for index in [0, 8, 44, bytes.len() / 2, bytes.len() - 1] {
            let mut corrupt = bytes.clone();
            corrupt[index] ^= 0x80;
            assert!(matches!(
                Manifest::from_bytes(&corrupt),
                Err(ManifestCodecError::ChecksumMismatch { .. })
            ));
        }
        Ok(())
    }

    #[test]
    fn valid_crc_cannot_hide_hostile_counts_or_unknown_flags() -> TestResult {
        let mut bytes = Manifest::empty(1, 7, 0).to_bytes()?;
        bytes[40..44].copy_from_slice(&0x8000_0000_u32.to_le_bytes());
        rewrite_crc(&mut bytes);
        assert!(matches!(
            Manifest::from_bytes(&bytes),
            Err(ManifestCodecError::NonCanonical { .. })
        ));

        let mut bytes = Manifest::empty(1, 7, 0).to_bytes()?;
        let hostile_segment_count =
            u32::try_from(MAX_MANIFEST_SEGMENTS + 1).expect("segment cap fits u32");
        bytes[44..48].copy_from_slice(&hostile_segment_count.to_le_bytes());
        rewrite_crc(&mut bytes);
        assert!(matches!(
            Manifest::from_bytes(&bytes),
            Err(ManifestCodecError::ResourceLimit {
                resource: "segment count",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn q1_validation_rejects_overlap_duplicates_and_watermark_rollback() {
        let mut manifest = sample_manifest(1);
        manifest.segments[1].docid_lo = 150;
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest(1);
        manifest.segments[1].segment_id = manifest.segments[0].segment_id;
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest(1);
        manifest.segments[1].seal_seq = manifest.segments[0].seal_seq;
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest(1);
        manifest.docid_high_watermark = manifest.segments[1].docid_hi - 1;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn tombstones_must_be_canonical_and_inside_the_segment_range() {
        let mut manifest = sample_manifest(1);
        manifest.segments[0].tombstones = array_tombstones(0, &[99]);
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest(1);
        manifest.segments[0].tombstones = array_tombstones(0, &[150, 150]);
        assert!(manifest.validate().is_err());

        let mut bitmap = vec![1, 0, 0, 0, 0, 0, 1, 1, 0];
        bitmap.extend_from_slice(&[0; 8_192]);
        bitmap[9] = 1;
        let mut manifest = sample_manifest(1);
        manifest.segments[0].tombstones = bitmap;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn tombstone_container_thresholds_and_chunk_count_are_bounded() {
        let oversized_lows = (0..=TOMBSTONE_ARRAY_MAX_CARDINALITY).collect::<Vec<_>>();
        assert!(matches!(
            validate_tombstone_bytes(&array_tombstones(0, &oversized_lows), None),
            Err(ManifestCodecError::NonCanonical { .. })
        ));

        assert!(matches!(
            validate_tombstone_bytes(&bitmap_tombstones(0, &[7]), None),
            Err(ManifestCodecError::NonCanonical { .. })
        ));

        let dense_lows = (0..u16::try_from(TOMBSTONE_BITMAP_MIN_CARDINALITY)
            .expect("bitmap threshold fits u16"))
            .collect::<Vec<_>>();
        let dense = bitmap_tombstones(0, &dense_lows);
        assert_eq!(
            validate_tombstone_bytes(&dense, Some((0, 65_536))).expect("dense bitmap is canonical"),
            TOMBSTONE_BITMAP_MIN_CARDINALITY
        );
        assert!(validate_tombstone_bytes(&dense, Some((100, 65_536))).is_err());

        let mut too_many_chunks = Vec::new();
        put_u32(
            &mut too_many_chunks,
            u32::try_from(MAX_TOMBSTONE_CHUNKS + 1).expect("chunk cap fits u32"),
        );
        assert!(matches!(
            validate_tombstone_bytes(&too_many_chunks, None),
            Err(ManifestCodecError::ResourceLimit {
                resource: "tombstone chunk count",
                ..
            })
        ));

        let sparse = array_tombstones(0, &[1, 100]);
        let dense_lows = (0..u16::try_from(TOMBSTONE_BITMAP_MIN_CARDINALITY)
            .expect("bitmap threshold fits u16"))
            .collect::<Vec<_>>();
        let dense = bitmap_tombstones(0, &dense_lows);
        assert!(tombstones_are_subset(&sparse, &dense).expect("compare array to bitmap"));
        let missing = array_tombstones(0, &[1, 4_000]);
        assert!(!tombstones_are_subset(&missing, &dense).expect("detect missing bitmap bit"));
    }

    #[test]
    fn stats_must_be_ordered_and_bounded_by_snapshot_docs() {
        let mut manifest = sample_manifest(1);
        manifest.field_stats.swap(0, 1);
        assert!(manifest.validate().is_err());

        let mut manifest = sample_manifest(1);
        manifest.field_stats[0].doc_count = 6;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn arbitrary_bytes_never_panic() {
        let mut state = 0x5e31_9a72_d0c4_b611_u64;
        for case in 0..1_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let length = usize::try_from(state % 257).expect("small generated length");
            let mut bytes = Vec::with_capacity(length);
            for _ in 0..length {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                bytes.push(state.to_le_bytes()[0]);
            }
            assert!(
                std::panic::catch_unwind(|| Manifest::from_bytes(&bytes)).is_ok(),
                "parser panic in case {case}"
            );
        }
    }

    #[test]
    fn pair_loader_distinguishes_not_found_and_both_fallback_windows() -> TestResult {
        let directory = tempdir()?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::IndexNotFound { .. })
        ));

        let previous = sample_manifest(4);
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, previous);
        assert_eq!(loaded.source, ManifestSource::PreviousAfterMissingCurrent);

        std::fs::write(directory.path().join("MANIFEST"), b"corrupt")?;
        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, previous);
        assert_eq!(loaded.source, ManifestSource::PreviousAfterCorruptCurrent);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn pair_loader_never_follows_manifest_symlinks_or_special_entries() -> TestResult {
        use std::os::unix::fs::symlink;

        let directory = tempdir()?;
        let previous = sample_manifest(4);
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        let symlink_target = directory.path().join("manifest-target");
        write_manifest(&symlink_target, &sample_manifest(5))?;
        symlink(&symlink_target, directory.path().join("MANIFEST"))?;

        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, previous);
        assert_eq!(loaded.source, ManifestSource::PreviousAfterCorruptCurrent);

        let directory = tempdir()?;
        let previous = sample_manifest(7);
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        std::fs::create_dir(directory.path().join("MANIFEST"))?;
        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, previous);
        assert_eq!(loaded.source, ManifestSource::PreviousAfterCorruptCurrent);

        let directory = tempdir()?;
        let current = sample_manifest(9);
        write_manifest(&directory.path().join("MANIFEST"), &current)?;
        let previous_target = directory.path().join("previous-target");
        write_manifest(&previous_target, &sample_manifest(8))?;
        symlink(&previous_target, directory.path().join("MANIFEST.prev"))?;
        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, current);
        assert_eq!(loaded.source, ManifestSource::Current);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pair_loader_treats_fifo_manifest_as_invalid_without_blocking() -> TestResult {
        use rustix::fs::{Mode, mkfifoat};

        let directory = tempdir()?;
        let previous = sample_manifest(4);
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        let directory_file = File::open(directory.path())?;
        mkfifoat(&directory_file, "MANIFEST", Mode::RUSR | Mode::WUSR)?;

        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, previous);
        assert_eq!(loaded.source, ManifestSource::PreviousAfterCorruptCurrent);
        Ok(())
    }

    #[test]
    fn pair_loader_rejects_nonadjacent_valid_generations() -> TestResult {
        let directory = tempdir()?;
        write_manifest(&directory.path().join("MANIFEST"), &sample_manifest(9))?;
        write_manifest(&directory.path().join("MANIFEST.prev"), &sample_manifest(3))?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::InvalidGenerationPair {
                current: 9,
                previous: 3,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn pair_loader_rejects_stable_identity_and_watermark_disagreement() -> TestResult {
        let directory = tempdir()?;
        let previous = sample_manifest(8);

        let mut equal_but_different = previous.clone();
        equal_but_different.flags = MANIFEST_FLAG_BULK_MODE_IN_PROGRESS;
        write_manifest(&directory.path().join("MANIFEST"), &equal_but_different)?;
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::InvalidManifestPair {
                current: 8,
                previous: 8,
                ..
            })
        ));

        let mut schema_mismatch = sample_manifest(9);
        schema_mismatch.schema_id ^= 1;
        write_manifest(&directory.path().join("MANIFEST"), &schema_mismatch)?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::InvalidManifestPair {
                current: 9,
                previous: 8,
                ..
            })
        ));

        let mut watermark_rollback = sample_manifest(9);
        watermark_rollback.docid_high_watermark = previous.docid_high_watermark - 1;
        write_manifest(&directory.path().join("MANIFEST"), &watermark_rollback)?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::InvalidManifestPair {
                current: 9,
                previous: 8,
                ..
            })
        ));

        let mut resurrected = sample_manifest(9);
        resurrected.segments[0].tombstones = array_tombstones(0, &[101]);
        write_manifest(&directory.path().join("MANIFEST"), &resurrected)?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::InvalidManifestPair {
                current: 9,
                previous: 8,
                ..
            })
        ));

        let mut changed_stats = sample_manifest(9);
        changed_stats.field_stats[0].total_tokens += 1;
        write_manifest(&directory.path().join("MANIFEST"), &changed_stats)?;
        assert!(matches!(
            load_manifest_pair(directory.path()),
            Err(KeeperError::InvalidManifestPair {
                current: 9,
                previous: 8,
                ..
            })
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn valid_current_wins_when_previous_cannot_be_read_as_a_file() -> TestResult {
        let directory = tempdir()?;
        let current = sample_manifest(1);
        write_manifest(&directory.path().join("MANIFEST"), &current)?;
        std::fs::create_dir(directory.path().join("MANIFEST.prev"))?;

        let loaded = load_manifest_pair(directory.path())?;
        assert_eq!(loaded.manifest, current);
        assert_eq!(loaded.source, ManifestSource::Current);
        Ok(())
    }

    #[test]
    fn publish_reopens_monotone_current_and_previous_generations() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let first = sample_manifest(1);
            publisher
                .publish(&cx, &first)
                .await
                .expect("publish genesis");

            let mut second = sample_manifest(2);
            second.docid_high_watermark = first.docid_high_watermark + 100;
            publisher
                .publish(&cx, &second)
                .await
                .expect("publish second");

            let loaded = load_manifest_pair(directory.path()).expect("reopen pair");
            assert_eq!(loaded.manifest, second);
            assert_eq!(loaded.source, ManifestSource::Current);
            let previous_bytes =
                std::fs::read(directory.path().join("MANIFEST.prev")).expect("read previous");
            assert_eq!(
                Manifest::from_bytes(&previous_bytes).expect("parse previous"),
                first
            );
        });
    }

    #[test]
    fn publish_rejects_generation_gap_and_watermark_rollback() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let first = sample_manifest(1);
            publisher
                .publish(&cx, &first)
                .await
                .expect("publish genesis");

            assert!(matches!(
                publisher.publish(&cx, &sample_manifest(3)).await,
                Err(KeeperError::GenerationConflict {
                    expected: 2,
                    proposed: 3
                })
            ));
            let mut rollback = sample_manifest(2);
            rollback.docid_high_watermark = first.docid_high_watermark - 1;
            assert!(matches!(
                publisher.publish(&cx, &rollback).await,
                Err(KeeperError::InvalidTransition { .. })
            ));
        });
    }

    #[test]
    fn publish_preserves_retained_segment_identity_and_monotone_tombstones() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let first = sample_manifest(1);
            publisher
                .publish(&cx, &first)
                .await
                .expect("publish genesis");

            let mut mutated = sample_manifest(2);
            mutated.segments[0].file_xxh3 ^= 1;
            assert!(matches!(
                publisher.publish(&cx, &mutated).await,
                Err(KeeperError::InvalidTransition { .. })
            ));

            let mut resurrected = sample_manifest(2);
            resurrected.segments[0].tombstones = array_tombstones(0, &[101]);
            assert!(matches!(
                publisher.publish(&cx, &resurrected).await,
                Err(KeeperError::InvalidTransition { .. })
            ));

            let mut stale_seal = sample_manifest(2);
            stale_seal.segments.push(ManifestSegment {
                segment_id: 0x1003,
                seal_seq: 6,
                file_len: 512,
                file_xxh3: 0xc1c2_c3c4_c5c6_c7c8,
                docid_lo: 70_000,
                docid_hi: 70_001,
                doc_count: 1,
                tombstones: EMPTY_TOMBSTONES.to_vec(),
            });
            assert!(matches!(
                publisher.publish(&cx, &stale_seal).await,
                Err(KeeperError::InvalidTransition { .. })
            ));

            let mut changed_stats = sample_manifest(2);
            changed_stats.field_stats[0].total_tokens += 1;
            assert!(matches!(
                publisher.publish(&cx, &changed_stats).await,
                Err(KeeperError::InvalidTransition { .. })
            ));

            let mut monotone = sample_manifest(2);
            monotone.segments[0].tombstones = array_tombstones(0, &[101, 150, 175]);
            publisher
                .publish(&cx, &monotone)
                .await
                .expect("monotone tombstone growth publishes");
            assert_eq!(
                load_manifest_pair(directory.path())
                    .expect("reopen monotone publish")
                    .manifest,
                monotone
            );
        });
    }

    #[test]
    fn empty_manifest_starts_a_new_seal_sequence_epoch() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let first = sample_manifest(1);
            publisher
                .publish(&cx, &first)
                .await
                .expect("publish genesis");

            let empty = Manifest::empty(
                2,
                first.schema_id,
                first.docid_high_watermark.saturating_add(1),
            );
            publisher
                .publish(&cx, &empty)
                .await
                .expect("publish explicit empty generation");

            let mut restarted = Manifest::empty(3, first.schema_id, empty.docid_high_watermark);
            let mut segment = first.segments[0].clone();
            segment.segment_id = 0x2001;
            segment.seal_seq = 1;
            restarted.segments.push(segment);
            restarted.field_stats = vec![
                ManifestFieldStats {
                    field_ord: 1,
                    total_tokens: 12,
                    doc_count: 3,
                },
                ManifestFieldStats {
                    field_ord: 3,
                    total_tokens: 42,
                    doc_count: 2,
                },
            ];
            publisher
                .publish(&cx, &restarted)
                .await
                .expect("restart seal sequence after empty epoch boundary");
            assert_eq!(
                load_manifest_pair(directory.path())
                    .expect("reopen restarted seal epoch")
                    .manifest,
                restarted
            );
        });
    }

    #[test]
    fn invalid_or_cancelled_proposal_creates_no_manifest_slot() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            assert!(matches!(
                publisher.publish(&cx, &Manifest::empty(0, 7, 0)).await,
                Err(KeeperError::InvalidManifest { .. })
            ));

            cx.set_cancel_requested(true);
            assert!(matches!(
                publisher.publish(&cx, &Manifest::empty(1, 7, 0)).await,
                Err(KeeperError::PublishLock {
                    source: LockError::Cancelled
                })
            ));
            assert!(!directory.path().join("MANIFEST").exists());
            assert!(!directory.path().join("MANIFEST.prev").exists());
        });
    }

    #[test]
    fn generation_claim_runs_after_temp_fsync_and_before_renames() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            publisher
                .publish(&cx, &sample_manifest(1))
                .await
                .expect("publish genesis");
            let saw_expected_window = Arc::new(AtomicBool::new(false));
            let witness = Arc::clone(&saw_expected_window);
            publisher
                .publish_with_generation_claim(&cx, &sample_manifest(2), move |path, generation| {
                    let temp = path.join(format!(".tmp-manifest-{generation}"));
                    let current = path.join("MANIFEST");
                    let previous = path.join("MANIFEST.prev");
                    witness.store(
                        temp.is_file() && current.is_file() && !previous.exists(),
                        Ordering::SeqCst,
                    );
                    Ok(())
                })
                .await
                .expect("claimed publish");
            assert!(saw_expected_window.load(Ordering::SeqCst));
        });
    }

    #[test]
    fn ordinary_segment_publish_performs_zero_sidecar_io() -> TestResult {
        let directory = tempdir()?;
        let encoded = encoded_test_segment(0x50, 0, 2, 1)?;
        let pending = encoded.write_temp(directory.path())?;
        let published = directory.path().join(canonical_segment_name(0x50));
        let mut sidecar_name = published.as_os_str().to_os_string();
        sidecar_name.push(".fec");
        let sidecar = PathBuf::from(sidecar_name);
        let sentinel = b"opaque feature-off segment sidecar";
        std::fs::write(&sidecar, sentinel)?;

        assert_eq!(publish_pending_segment(pending)?, published);
        assert_eq!(std::fs::read(&published)?, encoded.as_bytes());
        assert_eq!(std::fs::read(sidecar)?, sentinel);
        Ok(())
    }

    #[test]
    fn ordinary_manifest_publish_performs_zero_sidecar_io() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            publisher
                .publish(&cx, &sample_manifest(1))
                .await
                .expect("publish genesis");
            let sidecar = directory.path().join("MANIFEST.fec");
            std::fs::write(&sidecar, b"opaque feature-off artifact").expect("write sentinel");

            publisher
                .publish(&cx, &sample_manifest(2))
                .await
                .expect("publish without durability");

            assert_eq!(
                std::fs::read(&sidecar).expect("sentinel remains readable"),
                b"opaque feature-off artifact"
            );
            assert!(!directory.path().join("MANIFEST.prev.fec").exists());
        });
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_segment_publish_refuses_an_existing_sidecar() -> TestResult {
        let directory = tempdir()?;
        let protector = test_file_protector();
        let encoded = encoded_test_segment(0x51, 0, 2, 1)?;
        let pending = encoded.write_temp(directory.path())?;
        let pending_path = pending.path().to_path_buf();
        let published = directory.path().join(canonical_segment_name(0x51));
        let sidecar = FileProtector::sidecar_path(&published);
        let sentinel = b"preserved orphan sidecar";
        std::fs::write(&sidecar, sentinel)?;

        assert!(matches!(
            publish_pending_segment_durable(pending, &protector),
            Err(KeeperError::Io {
                operation: "publish segment sidecar",
                ..
            })
        ));
        assert_eq!(std::fs::read(&pending_path)?, encoded.as_bytes());
        assert!(!published.exists());
        assert_eq!(std::fs::read(sidecar)?, sentinel);
        Ok(())
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_segment_publish_rejects_same_length_temp_mutation() -> TestResult {
        let directory = tempdir()?;
        let protector = test_file_protector();
        let encoded = encoded_test_segment(0x52, 0, 2, 1)?;
        let pending = encoded.write_temp(directory.path())?;
        let pending_path = pending.path().to_path_buf();
        let published = directory.path().join(canonical_segment_name(0x52));
        let sidecar = FileProtector::sidecar_path(&published);

        let mut mutated = std::fs::read(&pending_path)?;
        let mutation_offset = mutated.len() / 2;
        mutated[mutation_offset] ^= 0x80;
        std::fs::write(&pending_path, &mutated)?;
        assert_eq!(std::fs::metadata(&pending_path)?.len(), pending.file_len());

        assert!(matches!(
            publish_pending_segment_durable(pending, &protector),
            Err(KeeperError::Durability {
                operation: "protect published segment",
                ..
            })
        ));
        assert!(!pending_path.exists());
        assert_eq!(std::fs::read(&published)?, mutated);
        assert!(!sidecar.exists());
        Ok(())
    }

    #[cfg(feature = "durability")]
    #[test]
    fn sidecar_retirement_uses_a_collision_safe_suffix() -> TestResult {
        let directory = tempdir()?;
        let source = directory.path().join("MANIFEST.fec");
        let retired = directory.path().join(".tmp-manifest-current-fec-2");
        let sentinel = b"operator-preserved crash artifact";
        let sidecar = b"active sidecar";
        std::fs::write(&source, sidecar)?;
        std::fs::write(&retired, sentinel)?;

        assert!(retire_manifest_sidecar(
            directory.path(),
            &source,
            &retired
        )?);
        assert!(!source.exists());
        assert_eq!(std::fs::read(&retired)?, sentinel);
        assert_eq!(std::fs::read(append_path_suffix(&retired, ".1"))?, sidecar);
        Ok(())
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_manifest_publish_retries_generation_after_retirement_collision() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let protector = test_file_protector();
            let mut writer = KeeperWriter::create_durable(
                &cx,
                directory.path(),
                DEFAULT_SCHEMA,
                protector.clone(),
            )
            .await
            .expect("create durable writer");
            writer
                .publish(&cx, &durable_test_manifest(2, Vec::new()))
                .await
                .expect("publish durable generation two");
            drop(writer);

            let previous = directory.path().join("MANIFEST.prev");
            let previous_sidecar = FileProtector::sidecar_path(&previous);
            let retired = directory.path().join(".tmp-manifest-previous-fec-3");
            let retired_bytes = std::fs::read(&previous_sidecar).expect("read prior sidecar");
            std::fs::rename(&previous_sidecar, &retired)
                .expect("simulate interrupted generation-three retirement");

            let mut writer = KeeperWriter::open_durable(
                &cx,
                directory.path(),
                DEFAULT_SCHEMA,
                protector.clone(),
            )
            .await
            .expect("recover active previous sidecar");
            assert_eq!(
                std::fs::read(&retired).expect("preserve crash artifact"),
                retired_bytes
            );
            assert!(
                protector
                    .verify_file(&previous, &previous_sidecar)
                    .expect("verify regenerated previous sidecar")
                    .healthy
            );

            writer
                .publish(&cx, &durable_test_manifest(3, Vec::new()))
                .await
                .expect("retry durable generation three");
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 3);
            assert_eq!(
                std::fs::read(&retired).expect("retain first retirement artifact"),
                retired_bytes
            );
            assert!(append_path_suffix(&retired, ".1").is_file());
            for source in ["MANIFEST", "MANIFEST.prev"] {
                let path = directory.path().join(source);
                assert!(
                    protector
                        .verify_file(&path, &FileProtector::sidecar_path(&path))
                        .expect("verify retried sidecar")
                        .healthy,
                    "{source}"
                );
            }
        });
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_manifest_publish_rotates_matching_sidecars() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let protector = test_file_protector();
            publisher
                .publish_durable(&cx, &sample_manifest(1), &protector)
                .await
                .expect("durable genesis");
            publisher
                .publish_durable(&cx, &sample_manifest(2), &protector)
                .await
                .expect("durable successor");

            for source in ["MANIFEST", "MANIFEST.prev"] {
                let path = directory.path().join(source);
                let sidecar = FileProtector::sidecar_path(&path);
                let verified = protector
                    .verify_file(&path, &sidecar)
                    .expect("verify matching sidecar");
                assert!(verified.healthy, "{source}");
            }
            assert_eq!(
                load_manifest_pair(directory.path())
                    .expect("load durable pair")
                    .manifest
                    .generation,
                2
            );
        });
    }

    #[cfg(feature = "durability")]
    #[test]
    fn manifest_sidecar_rotation_has_no_wrong_source_pairing() -> TestResult {
        let directory = tempdir()?;
        let protector = test_file_protector();
        for generation in [1, 2] {
            let bytes = sample_manifest(generation).to_bytes()?;
            publish_manifest_durable_choreography(
                directory.path().to_path_buf(),
                &bytes,
                |_, _| Ok(()),
                &protector,
                |_, _| Ok(()),
            )?;
        }

        let third = sample_manifest(3).to_bytes()?;
        let temp_path = directory.path().join(".tmp-manifest-3");
        std::fs::write(&temp_path, &third)?;
        protector.protect_file_with_witness(&temp_path, FileSourceWitness::from_bytes(&third))?;

        prepare_manifest_sidecar_rotation(directory.path(), true, 3, &protector)?;
        let current = directory.path().join("MANIFEST");
        let previous = directory.path().join("MANIFEST.prev");
        assert!(
            protector
                .verify_file(&current, &FileProtector::sidecar_path(&current))?
                .healthy
        );
        assert!(!FileProtector::sidecar_path(&previous).exists());

        std::fs::rename(&current, &previous)?;
        assert!(!current.exists());
        assert!(FileProtector::sidecar_path(&current).exists());
        assert!(!FileProtector::sidecar_path(&previous).exists());
        assert_eq!(load_manifest_pair(directory.path())?.manifest.generation, 2);

        complete_manifest_sidecar_rotation(directory.path(), &current, &previous, 3)?;
        assert!(!FileProtector::sidecar_path(&current).exists());
        assert!(
            protector
                .verify_file(&previous, &FileProtector::sidecar_path(&previous))?
                .healthy
        );

        std::fs::rename(&temp_path, &current)?;
        assert!(!FileProtector::sidecar_path(&current).exists());
        install_manifest_sidecar(directory.path(), &current, 3)?;
        for source in [&current, &previous] {
            assert!(
                protector
                    .verify_file(source, &FileProtector::sidecar_path(source))?
                    .healthy
            );
        }
        assert_eq!(load_manifest_pair(directory.path())?.manifest.generation, 3);
        Ok(())
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_publish_replaces_a_stale_mixed_mode_sidecar() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let protector = test_file_protector();
            publisher
                .publish_durable(&cx, &sample_manifest(1), &protector)
                .await
                .expect("durable generation one");
            publisher
                .publish(&cx, &sample_manifest(2))
                .await
                .expect("ordinary generation two");

            let current = directory.path().join("MANIFEST");
            let stale_sidecar = FileProtector::sidecar_path(&current);
            let current_bytes = std::fs::read(&current).expect("read generation two");
            assert!(
                !protector
                    .sidecar_matches_witness(
                        &stale_sidecar,
                        FileSourceWitness::from_bytes(&current_bytes),
                    )
                    .expect("compare stale sidecar")
            );

            publisher
                .publish_durable(&cx, &sample_manifest(3), &protector)
                .await
                .expect("durable generation three");
            for source in ["MANIFEST", "MANIFEST.prev"] {
                let path = directory.path().join(source);
                assert!(
                    protector
                        .verify_file(&path, &FileProtector::sidecar_path(&path))
                        .expect("verify regenerated sidecar")
                        .healthy,
                    "{source}"
                );
            }
            assert_eq!(
                load_manifest_pair(directory.path())
                    .expect("load mixed-mode pair")
                    .manifest
                    .generation,
                3
            );
        });
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durable_manifest_crash_windows_never_pair_stale_sidecars() -> TestResult {
        const CHECKPOINTS: [PublishCheckpoint; 6] = [
            PublishCheckpoint::TempWritten,
            PublishCheckpoint::TempSynced,
            PublishCheckpoint::GenerationClaimed,
            PublishCheckpoint::CurrentMovedToPrevious,
            PublishCheckpoint::TempMovedToCurrent,
            PublishCheckpoint::DirectorySynced,
        ];

        for fault in CHECKPOINTS {
            let directory = tempdir()?;
            let protector = test_file_protector();
            let first = sample_manifest(1).to_bytes()?;
            publish_manifest_durable_choreography(
                directory.path().to_path_buf(),
                &first,
                |_, _| Ok(()),
                &protector,
                |_, _| Ok(()),
            )?;

            let second = sample_manifest(2).to_bytes()?;
            let result = publish_manifest_durable_choreography(
                directory.path().to_path_buf(),
                &second,
                |_, _| Ok(()),
                &protector,
                |checkpoint, path| {
                    if checkpoint == fault {
                        return Err(KeeperError::Io {
                            operation: "inject durable manifest crash",
                            path: path.to_path_buf(),
                            source: io::Error::other(format!(
                                "injected durable crash at {checkpoint:?}"
                            )),
                        });
                    }
                    Ok(())
                },
            );
            assert!(matches!(
                result,
                Err(KeeperError::Io {
                    operation: "inject durable manifest crash",
                    ..
                })
            ));

            let loaded = load_manifest_pair(directory.path())?;
            let expected_generation = match fault {
                PublishCheckpoint::TempWritten
                | PublishCheckpoint::TempSynced
                | PublishCheckpoint::GenerationClaimed
                | PublishCheckpoint::CurrentMovedToPrevious => 1,
                PublishCheckpoint::TempMovedToCurrent | PublishCheckpoint::DirectorySynced => 2,
            };
            assert_eq!(loaded.manifest.generation, expected_generation, "{fault:?}");

            let current_path = directory.path().join("MANIFEST");
            let current_sidecar = FileProtector::sidecar_path(&current_path);
            if fault == PublishCheckpoint::TempMovedToCurrent {
                assert!(!current_sidecar.exists());
            } else if fault == PublishCheckpoint::DirectorySynced {
                assert!(current_sidecar.exists());
                assert!(
                    protector
                        .verify_file(&current_path, &current_sidecar)?
                        .healthy
                );
            }

            for source in ["MANIFEST", "MANIFEST.prev"] {
                let path = directory.path().join(source);
                let sidecar = FileProtector::sidecar_path(&path);
                if sidecar.exists() {
                    assert!(
                        path.exists(),
                        "orphan active sidecar at {fault:?}: {source}"
                    );
                    assert!(
                        protector.verify_file(&path, &sidecar)?.healthy,
                        "stale sidecar at {fault:?}: {source}"
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "durability")]
    #[test]
    fn real_durable_segment_checkpoints_recover_and_gc_exactly() -> TestResult {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum Fault {
            TempWritten,
            TempSynced,
            SegmentRenamed,
            DirectorySynced,
            SidecarEmitted,
        }

        const FAULTS: [Fault; 5] = [
            Fault::TempWritten,
            Fault::TempSynced,
            Fault::SegmentRenamed,
            Fault::DirectorySynced,
            Fault::SidecarEmitted,
        ];

        for fault in FAULTS {
            let directory = tempdir()?;
            let protector = test_file_protector();

            let base_encoded = encoded_test_segment(0x70, 0, 2, 1)?;
            let base_manifest_segment = manifest_segment(&base_encoded, 1);
            let base_pending = base_encoded.write_temp(directory.path())?;
            let base_path = publish_pending_segment_durable(base_pending, &protector)?;
            assert!(
                protector
                    .verify_file(&base_path, &FileProtector::sidecar_path(&base_path))?
                    .healthy
            );
            let base_manifest = durable_test_manifest(1, vec![base_manifest_segment]);
            publish_manifest_durable_choreography(
                directory.path().to_path_buf(),
                &base_manifest.to_bytes()?,
                |_, _| Ok(()),
                &protector,
                |_, _| Ok(()),
            )?;

            let next_encoded = encoded_test_segment(0x71, 2, 4, 1)?;
            let write_result =
                next_encoded.write_temp_with_observer(directory.path(), |checkpoint, path| {
                    let injected = matches!(
                        (fault, checkpoint),
                        (Fault::TempWritten, SegmentWriteCheckpoint::TempWritten)
                            | (Fault::TempSynced, SegmentWriteCheckpoint::TempSynced)
                    );
                    if injected {
                        return Err(QuillError::Io(io::Error::other(format!(
                            "injected segment writer crash at {checkpoint:?}: {}",
                            path.display()
                        ))));
                    }
                    Ok(())
                });

            if matches!(fault, Fault::TempWritten | Fault::TempSynced) {
                assert!(write_result.is_err(), "{fault:?}");
            } else {
                let pending = write_result?;
                assert_ne!(pending.file_xxh3(), pending.source_xxh3());
                let publish_result = publish_pending_segment_durable_with_observer(
                    pending,
                    &protector,
                    |checkpoint, path| {
                        let injected = matches!(
                            (fault, checkpoint),
                            (
                                Fault::SegmentRenamed,
                                SegmentPublishCheckpoint::SegmentRenamed
                            ) | (
                                Fault::DirectorySynced,
                                SegmentPublishCheckpoint::DirectorySynced
                            ) | (
                                Fault::SidecarEmitted,
                                SegmentPublishCheckpoint::SidecarEmitted
                            )
                        );
                        if injected {
                            return Err(KeeperError::Io {
                                operation: "inject segment publish crash",
                                path: path.to_path_buf(),
                                source: io::Error::other(format!(
                                    "injected segment publication crash at {checkpoint:?}"
                                )),
                            });
                        }
                        Ok(())
                    },
                );
                assert!(publish_result.is_err(), "{fault:?}");
            }

            let before_open = directory_bytes(directory.path())?;
            let first = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            let second = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            assert_eq!(
                first.loaded_manifest(),
                second.loaded_manifest(),
                "{fault:?}"
            );
            assert_eq!(first.loaded_manifest().manifest, base_manifest, "{fault:?}");
            assert_eq!(directory_bytes(directory.path())?, before_open, "{fault:?}");

            let segment_name = canonical_segment_name(0x71);
            let segment_path = directory.path().join(&segment_name);
            if fault == Fault::SidecarEmitted {
                assert!(
                    protector
                        .verify_file(&segment_path, &FileProtector::sidecar_path(&segment_path))?
                        .healthy
                );
            }

            let now = SystemTime::now()
                .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
                .expect("test clock remains representable");
            let report = collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            )?;
            let mut expected_removed = match fault {
                Fault::TempWritten | Fault::TempSynced => {
                    vec![PathBuf::from(".tmp-segment-0000000000000071")]
                }
                Fault::SegmentRenamed | Fault::DirectorySynced => {
                    vec![PathBuf::from(&segment_name)]
                }
                Fault::SidecarEmitted => vec![
                    PathBuf::from(&segment_name),
                    PathBuf::from(format!("{segment_name}.fec")),
                ],
            };
            expected_removed.sort_unstable();
            assert_eq!(report.removed, expected_removed, "{fault:?}");

            let mut expected_inventory = before_open;
            expected_inventory.retain(|(name, _)| {
                !report
                    .removed
                    .iter()
                    .any(|removed| removed.as_os_str() == name)
            });
            assert_eq!(
                directory_bytes(directory.path())?,
                expected_inventory,
                "{fault:?}"
            );
            let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            assert_eq!(
                reopened.loaded_manifest(),
                first.loaded_manifest(),
                "{fault:?}"
            );
            assert_eq!(
                collect_writer_garbage_at(
                    directory.path(),
                    DEFAULT_SCHEMA,
                    GarbageCollectionOptions::default(),
                    now,
                )?,
                GarbageCollectionReport {
                    removed: Vec::new()
                },
                "{fault:?}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "durability")]
    #[test]
    fn real_fslx_and_manifest_repair_restore_exact_bytes() -> TestResult {
        let directory = tempdir()?;
        let protector = test_file_protector();
        let encoded = encoded_test_segment(0x71, 0, 2, 1)?;
        let segment = manifest_segment(&encoded, 1);
        let pending = encoded.write_temp(directory.path())?;
        assert_eq!(
            pending.source_xxh3(),
            xxhash_rust::xxh3::xxh3_64(encoded.as_bytes())
        );
        let segment_path = publish_pending_segment_durable(pending, &protector)?;
        let manifest = durable_test_manifest(1, vec![segment]);
        let manifest_path = directory.path().join("MANIFEST");
        let manifest_bytes = manifest.to_bytes()?;
        publish_manifest_durable_choreography(
            directory.path().to_path_buf(),
            &manifest_bytes,
            |_, _| Ok(()),
            &protector,
            |_, _| Ok(()),
        )?;

        let segment_bytes = std::fs::read(&segment_path)?;
        assert_eq!(std::fs::read(&manifest_path)?, manifest_bytes);

        let mut corrupted_segment = segment_bytes.clone();
        let corruption_offset = corrupted_segment.len() / 2;
        corrupted_segment[corruption_offset] ^= 0x80;
        std::fs::write(&segment_path, &corrupted_segment)?;
        assert!(!protector.sidecar_matches_witness(
            &FileProtector::sidecar_path(&segment_path),
            FileSourceWitness::from_bytes(&corrupted_segment),
        )?);
        let segment_health = protector.verify_and_repair_file(&segment_path)?;
        assert!(matches!(segment_health.status, FileHealth::Repaired { .. }));
        assert_eq!(std::fs::read(&segment_path)?, segment_bytes);

        File::options()
            .write(true)
            .open(&segment_path)?
            .set_len(u64::try_from(segment_bytes.len() / 3)?)?;
        let truncated = std::fs::read(&segment_path)?;
        assert!(!protector.sidecar_matches_witness(
            &FileProtector::sidecar_path(&segment_path),
            FileSourceWitness::from_bytes(&truncated),
        )?);
        let segment_health = protector.verify_and_repair_file(&segment_path)?;
        assert!(matches!(segment_health.status, FileHealth::Repaired { .. }));
        assert_eq!(std::fs::read(&segment_path)?, segment_bytes);

        let displaced_segment = directory.path().join(".tmp-missing-segment-fixture");
        std::fs::rename(&segment_path, &displaced_segment)?;
        let segment_health = protector.verify_and_repair_file(&segment_path)?;
        assert!(matches!(segment_health.status, FileHealth::Repaired { .. }));
        assert_eq!(std::fs::read(&segment_path)?, segment_bytes);
        assert_eq!(std::fs::read(displaced_segment)?, segment_bytes);

        let mut corrupted_manifest = manifest_bytes.clone();
        corrupted_manifest[16] ^= 0x40;
        std::fs::write(&manifest_path, &corrupted_manifest)?;
        assert!(!protector.sidecar_matches_witness(
            &FileProtector::sidecar_path(&manifest_path),
            FileSourceWitness::from_bytes(&corrupted_manifest),
        )?);
        let manifest_health = protector.verify_and_repair_file(&manifest_path)?;
        assert!(matches!(
            manifest_health.status,
            FileHealth::Repaired { .. }
        ));
        assert_eq!(std::fs::read(&manifest_path)?, manifest_bytes);

        let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        assert_eq!(reopened.segments().len(), 1);
        reopened.segments()[0].verify()?;
        Ok(())
    }

    #[test]
    fn real_manifest_choreography_recovers_at_every_checkpoint() -> TestResult {
        const CHECKPOINTS: [PublishCheckpoint; 6] = [
            PublishCheckpoint::TempWritten,
            PublishCheckpoint::TempSynced,
            PublishCheckpoint::GenerationClaimed,
            PublishCheckpoint::CurrentMovedToPrevious,
            PublishCheckpoint::TempMovedToCurrent,
            PublishCheckpoint::DirectorySynced,
        ];

        let success_directory = tempdir()?;
        let schema_id = DEFAULT_SCHEMA.schema_id()?;
        write_manifest(
            &success_directory.path().join("MANIFEST"),
            &Manifest::empty(1, schema_id, 0),
        )?;
        let proposed = Manifest::empty(2, schema_id, 0);
        let proposed_bytes = proposed.to_bytes()?;
        let mut observed = Vec::new();
        publish_manifest_choreography(
            success_directory.path().to_path_buf(),
            &proposed_bytes,
            |_, _| Ok(()),
            |checkpoint, _| {
                observed.push(checkpoint);
                Ok(())
            },
        )?;
        assert_eq!(observed, CHECKPOINTS);

        for fault in CHECKPOINTS {
            let directory = tempdir()?;
            let initial = Manifest::empty(1, schema_id, 0);
            write_manifest(&directory.path().join("MANIFEST"), &initial)?;
            let result = publish_manifest_choreography(
                directory.path().to_path_buf(),
                &proposed_bytes,
                |_, _| Ok(()),
                |checkpoint, path| {
                    if checkpoint == fault {
                        return Err(KeeperError::Io {
                            operation: "inject manifest publish crash",
                            path: path.to_path_buf(),
                            source: io::Error::other(format!("injected crash at {checkpoint:?}")),
                        });
                    }
                    Ok(())
                },
            );
            assert!(
                matches!(
                    result,
                    Err(KeeperError::Io {
                        operation: "inject manifest publish crash",
                        ..
                    })
                ),
                "{fault:?}"
            );

            let before_open = directory_bytes(directory.path())?;
            let first = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            let second = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            assert_eq!(directory_bytes(directory.path())?, before_open, "{fault:?}");
            assert_eq!(
                first.loaded_manifest(),
                second.loaded_manifest(),
                "{fault:?}"
            );
            let expected_generation = match fault {
                PublishCheckpoint::TempWritten
                | PublishCheckpoint::TempSynced
                | PublishCheckpoint::GenerationClaimed
                | PublishCheckpoint::CurrentMovedToPrevious => 1,
                PublishCheckpoint::TempMovedToCurrent | PublishCheckpoint::DirectorySynced => 2,
            };
            assert_eq!(
                first.loaded_manifest().manifest.generation,
                expected_generation,
                "{fault:?}"
            );
            let expected_source = if fault == PublishCheckpoint::CurrentMovedToPrevious {
                ManifestSource::PreviousAfterMissingCurrent
            } else {
                ManifestSource::Current
            };
            assert_eq!(first.loaded_manifest().source, expected_source, "{fault:?}");

            let now = SystemTime::now()
                .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
                .expect("test clock remains representable");
            let report = collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            )?;
            let temp_survived_fault = matches!(
                fault,
                PublishCheckpoint::TempWritten
                    | PublishCheckpoint::TempSynced
                    | PublishCheckpoint::GenerationClaimed
                    | PublishCheckpoint::CurrentMovedToPrevious
            );
            assert_eq!(
                report.removed,
                if temp_survived_fault {
                    vec![PathBuf::from(".tmp-manifest-2")]
                } else {
                    Vec::new()
                },
                "{fault:?}"
            );
            let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            assert_eq!(
                reopened.loaded_manifest(),
                first.loaded_manifest(),
                "{fault:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn retry_reuses_byte_identical_temp_after_claim_failure() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            publisher
                .publish(&cx, &sample_manifest(1))
                .await
                .expect("publish genesis");
            let proposed = sample_manifest(2);
            let expected_bytes = proposed.to_bytes().expect("encode proposal");

            let first_result = publisher
                .publish_with_generation_claim::<(), _>(&cx, &proposed, |path, generation| {
                    Err(KeeperError::Io {
                        operation: "test generation claim",
                        path: path.join(format!("gen-{generation}.claim")),
                        source: io::Error::other("injected claim failure"),
                    })
                })
                .await;
            assert!(matches!(
                first_result,
                Err(KeeperError::Io {
                    operation: "test generation claim",
                    ..
                })
            ));
            let temp_path = directory.path().join(".tmp-manifest-2");
            assert_eq!(
                std::fs::read(&temp_path).expect("read durable temp"),
                expected_bytes
            );

            publisher
                .publish(&cx, &proposed)
                .await
                .expect("retry byte-identical durable temp");
            assert_eq!(
                load_manifest_pair(directory.path())
                    .expect("reopen retried publish")
                    .manifest,
                proposed
            );
        });
    }

    #[test]
    fn mismatched_existing_temp_fails_closed_without_overwrite() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let temp_path = directory.path().join(".tmp-manifest-1");
            let stale_bytes = b"stale temp from another proposal";
            std::fs::write(&temp_path, stale_bytes).expect("write stale temp fixture");

            assert!(matches!(
                publisher.publish(&cx, &sample_manifest(1)).await,
                Err(KeeperError::TempConflict { .. })
            ));
            assert_eq!(
                std::fs::read(&temp_path).expect("read preserved stale temp"),
                stale_bytes
            );
            assert!(!directory.path().join("MANIFEST").exists());
        });
    }

    #[cfg(unix)]
    #[test]
    fn existing_temp_symlink_fails_closed_without_publication() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempdir().expect("temp directory");
            let publisher = ManifestPublisher::new(directory.path());
            let expected_bytes = sample_manifest(1).to_bytes().expect("encode proposal");
            let symlink_target = directory.path().join("byte-identical-target");
            std::fs::write(&symlink_target, expected_bytes).expect("write symlink target");
            let temp_path = directory.path().join(".tmp-manifest-1");
            std::os::unix::fs::symlink(&symlink_target, &temp_path)
                .expect("create temp symlink fixture");

            assert!(matches!(
                publisher.publish(&cx, &sample_manifest(1)).await,
                Err(KeeperError::Io {
                    operation: "open existing temp",
                    ..
                })
            ));
            assert!(
                std::fs::symlink_metadata(&temp_path)
                    .expect("stat preserved temp symlink")
                    .file_type()
                    .is_symlink()
            );
            assert!(!directory.path().join("MANIFEST").exists());
        });
    }

    #[cfg(unix)]
    #[test]
    fn labruntime_serializes_concurrent_publishers_across_a_late_symlink_alias() {
        let root = tempdir().expect("temp directory");
        let directory = root.path().join("target-index");
        let alias = root.path().join("index-alias");
        std::fs::create_dir(&directory).expect("create target index directory");
        let first_publisher = ManifestPublisher::new(&alias);
        let second_publisher = ManifestPublisher::new(&directory);
        std::os::unix::fs::symlink(&directory, &alias).expect("create alias after first publisher");
        let lock = first_publisher.publish_lock_for_test();
        assert!(Arc::ptr_eq(
            &lock,
            &second_publisher.publish_lock_for_test()
        ));
        let saw_two_waiters = Arc::new(AtomicBool::new(false));
        let successes = Arc::new(AtomicUsize::new(0));
        let conflicts = Arc::new(AtomicUsize::new(0));

        let mut lab = LabRuntime::new(LabConfig::new(0x51a1).max_steps(100_000));
        let region = lab.state.create_root_region(Budget::INFINITE);
        let holder_lock = Arc::clone(&lock);
        let holder_witness = Arc::clone(&saw_two_waiters);
        let (holder, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                let cx = Cx::for_testing();
                let guard = OwnedMutexGuard::lock(Arc::clone(&holder_lock), &cx)
                    .await
                    .expect("holder lock");
                while holder_lock.waiters() < 2 {
                    yield_now().await;
                }
                holder_witness.store(true, Ordering::SeqCst);
                drop(guard);
            })
            .expect("create holder task");

        let first_successes = Arc::clone(&successes);
        let first_conflicts = Arc::clone(&conflicts);
        let (first_task, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                let cx = Cx::for_testing();
                match first_publisher.publish(&cx, &sample_manifest(1)).await {
                    Ok(_) => {
                        first_successes.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(KeeperError::GenerationConflict {
                        expected: 2,
                        proposed: 1,
                    }) => {
                        first_conflicts.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(error) => panic!("unexpected first publish result: {error}"),
                }
            })
            .expect("create first publisher");

        let second_successes = Arc::clone(&successes);
        let second_conflicts = Arc::clone(&conflicts);
        let (second_task, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                let cx = Cx::for_testing();
                match second_publisher.publish(&cx, &sample_manifest(1)).await {
                    Ok(_) => {
                        second_successes.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(KeeperError::GenerationConflict {
                        expected: 2,
                        proposed: 1,
                    }) => {
                        second_conflicts.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(error) => panic!("unexpected second publish result: {error}"),
                }
            })
            .expect("create second publisher");

        lab.scheduler.lock().schedule(holder, 0);
        lab.step_for_test();
        lab.scheduler.lock().schedule(first_task, 0);
        lab.scheduler.lock().schedule(second_task, 0);
        let report = lab.run_until_quiescent_with_report();

        assert!(saw_two_waiters.load(Ordering::SeqCst));
        assert_eq!(successes.load(Ordering::SeqCst), 1);
        assert_eq!(conflicts.load(Ordering::SeqCst), 1);
        assert!(report.quiescent, "LabRuntime must reach quiescence");
        assert!(report.oracle_report.all_passed(), "oracles must pass");
        assert!(report.invariant_violations.is_empty());
        assert_eq!(
            load_manifest_pair(&directory)
                .expect("reopen after concurrent publishes")
                .manifest
                .generation,
            1
        );
    }

    #[cfg(unix)]
    #[test]
    fn labruntime_writer_admission_blocks_a_second_writer_but_not_a_reader() -> TestResult {
        let index = tempdir()?;
        create_test_index(index.path()).map_err(io::Error::other)?;
        let directory = index.path().to_path_buf();
        let holder_ready = Arc::new(AtomicBool::new(false));
        let contender_was_busy = Arc::new(AtomicBool::new(false));
        let reader_was_touchless = Arc::new(AtomicBool::new(false));
        let observers_done = Arc::new(AtomicUsize::new(0));

        let mut lab = LabRuntime::new(LabConfig::new(0x10c6).max_steps(100_000));
        let region = lab.state.create_root_region(Budget::INFINITE);

        let holder_directory = directory.clone();
        let holder_ready_task = Arc::clone(&holder_ready);
        let holder_done = Arc::clone(&observers_done);
        let (holder, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                let cx = Cx::for_testing();
                let writer = KeeperWriter::open(&cx, holder_directory, DEFAULT_SCHEMA)
                    .await
                    .expect("first writer admission");
                holder_ready_task.store(true, Ordering::SeqCst);
                while holder_done.load(Ordering::SeqCst) < 2 {
                    yield_now().await;
                }
                drop(writer);
            })
            .expect("create holder task");

        let contender_directory = directory.clone();
        let contender_ready = Arc::clone(&holder_ready);
        let contender_result = Arc::clone(&contender_was_busy);
        let contender_done = Arc::clone(&observers_done);
        let (contender, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                while !contender_ready.load(Ordering::SeqCst) {
                    yield_now().await;
                }
                let cx = Cx::for_testing();
                match KeeperWriter::open(&cx, contender_directory, DEFAULT_SCHEMA).await {
                    Err(KeeperError::WriterBusy { .. }) => {
                        contender_result.store(true, Ordering::SeqCst);
                    }
                    Ok(writer) => {
                        drop(writer);
                        panic!("second writer was admitted while the first held LOCK");
                    }
                    Err(error) => panic!("unexpected second-writer result: {error}"),
                }
                contender_done.fetch_add(1, Ordering::SeqCst);
            })
            .expect("create contender task");

        let reader_directory = directory.clone();
        let reader_ready = Arc::clone(&holder_ready);
        let reader_result = Arc::clone(&reader_was_touchless);
        let reader_done = Arc::clone(&observers_done);
        let (reader, _) = lab
            .state
            .create_task(region, Budget::INFINITE, async move {
                while !reader_ready.load(Ordering::SeqCst) {
                    yield_now().await;
                }
                let before = directory_bytes(&reader_directory).expect("reader inventory before");
                let snapshot = KeeperSnapshot::open(&reader_directory, DEFAULT_SCHEMA)
                    .expect("reader opens while writer is live");
                assert_eq!(snapshot.loaded_manifest().manifest.generation, 1);
                let after = directory_bytes(&reader_directory).expect("reader inventory after");
                assert_eq!(after, before, "reader open must not touch the directory");
                reader_result.store(true, Ordering::SeqCst);
                reader_done.fetch_add(1, Ordering::SeqCst);
            })
            .expect("create reader task");

        lab.scheduler.lock().schedule(holder, 0);
        lab.step_for_test();
        assert!(holder_ready.load(Ordering::SeqCst));
        lab.scheduler.lock().schedule(contender, 0);
        lab.scheduler.lock().schedule(reader, 0);
        let report = lab.run_until_quiescent_with_report();

        assert!(contender_was_busy.load(Ordering::SeqCst));
        assert!(reader_was_touchless.load(Ordering::SeqCst));
        assert!(report.quiescent, "LabRuntime must reach quiescence");
        assert!(report.oracle_report.all_passed(), "oracles must pass");
        assert!(report.invariant_violations.is_empty());
        assert_eq!(std::fs::metadata(directory.join("LOCK"))?.len(), 0);
        Ok(())
    }

    #[test]
    fn keeper_create_is_create_or_open_and_in_memory_creates_no_files() {
        let memory = KeeperSnapshot::in_memory(DEFAULT_SCHEMA).expect("in-memory genesis");
        assert_eq!(memory.directory(), None);
        assert_eq!(memory.loaded_manifest().source, ManifestSource::InMemory);
        assert_eq!(memory.loaded_manifest().manifest.generation, 1);
        assert!(memory.segments().is_empty());

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let root = tempdir().expect("temp directory");
            let directory = root.path().join("quill-index");
            let created = KeeperSnapshot::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .expect("create genesis");
            assert_eq!(created.directory(), Some(directory.as_path()));
            assert_eq!(created.loaded_manifest().manifest.generation, 1);
            let manifest_bytes = std::fs::read(directory.join("MANIFEST")).expect("manifest");

            let reopened = KeeperSnapshot::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .expect("create-or-open existing index");
            assert_eq!(reopened.loaded_manifest().manifest.generation, 1);
            assert_eq!(
                std::fs::read(directory.join("MANIFEST")).expect("manifest after reopen"),
                manifest_bytes
            );
            assert!(matches!(
                KeeperSnapshot::create(&cx, &directory, FSFS_CHUNK_SCHEMA).await,
                Err(KeeperError::SchemaMismatch { .. })
            ));
        });
    }

    #[test]
    fn recovered_snapshot_validates_witnesses_and_keeps_section_hashes_lazy() -> TestResult {
        let directory = tempdir()?;
        let encoded = encoded_test_segment(0xabc, 10, 20, 1)?;
        let postings = encoded
            .section_entries()
            .iter()
            .find(|entry| entry.kind == SectionKind::POSTINGS)
            .expect("postings entry");
        let postings_offset = usize::try_from(postings.offset)?;
        let mut bytes = encoded.as_bytes().to_vec();
        bytes[postings_offset] ^= 0x80;
        std::fs::write(directory.path().join(canonical_segment_name(0xabc)), bytes)?;
        let segment = ManifestSegment {
            segment_id: 0xabc,
            seal_seq: 1,
            file_len: encoded.file_len(),
            file_xxh3: encoded.file_xxh3(),
            docid_lo: 10,
            docid_hi: 20,
            doc_count: 1,
            tombstones: EMPTY_TOMBSTONES.to_vec(),
        };
        let manifest = durable_test_manifest(1, vec![segment]);
        write_manifest(&directory.path().join("MANIFEST"), &manifest)?;
        let before = directory_bytes(directory.path())?;

        let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        assert_eq!(directory_bytes(directory.path())?, before);
        assert_eq!(snapshot.segments().len(), 1);
        assert_eq!(
            snapshot.segments()[0].section(SectionKind::TERMDICT)?,
            Some(b"termdict".as_slice())
        );
        assert!(matches!(
            snapshot.segments()[0].section(SectionKind::POSTINGS),
            Err(QuillError::IndexCorrupted { .. })
        ));

        let mut mismatched = manifest;
        mismatched.segments[0].file_len += 1;
        write_manifest(&directory.path().join("MANIFEST"), &mismatched)?;
        assert!(matches!(
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
            Err(KeeperError::SegmentMetadataMismatch { .. })
        ));
        assert!(matches!(
            KeeperSnapshot::open(directory.path(), FSFS_CHUNK_SCHEMA),
            Err(KeeperError::SchemaMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn fallback_ignores_manifest_temp_and_rejects_incompatible_claim() -> TestResult {
        let directory = tempdir()?;
        let segment = write_test_segment(directory.path(), 0x11, 1, 0, 2)?;
        let previous = durable_test_manifest(1, vec![segment]);
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        let temp = durable_test_manifest(2, Vec::new());
        write_manifest(&directory.path().join(".tmp-manifest-2"), &temp)?;
        let before = directory_bytes(directory.path())?;

        let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
        assert_eq!(
            snapshot.loaded_manifest().source,
            ManifestSource::PreviousAfterMissingCurrent
        );
        assert_eq!(snapshot.loaded_manifest().manifest.generation, 1);
        assert_eq!(directory_bytes(directory.path())?, before);

        std::fs::write(directory.path().join("gen-3.claim"), [])?;
        assert!(matches!(
            KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA),
            Err(KeeperError::InvalidRecoveryClaim {
                recovered: 1,
                claimed: 3,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn deterministic_publish_crash_states_recover_read_only_and_gc_exactly() -> TestResult {
        const CASES: [&str; 10] = [
            "segment temp written",
            "segment temp fsynced",
            "segment renamed",
            "segment directory fsynced",
            "uncommitted sidecar emitted",
            "manifest temp written",
            "manifest temp fsynced",
            "between manifest renames",
            "new manifest current",
            "committed sidecar emitted",
        ];

        for (case_index, label) in CASES.into_iter().enumerate() {
            let directory = tempdir()?;
            let segment_a = write_test_segment(directory.path(), 0xa, 1, 0, 2)?;
            let segment_b_name = canonical_segment_name(0xb);
            let segment_b = if case_index >= 2 {
                Some(write_test_segment(directory.path(), 0xb, 2, 2, 4)?)
            } else {
                let encoded = encoded_test_segment(0xb, 2, 4, 1)?;
                let temp_path = directory.path().join(".tmp-segment-000000000000000b");
                std::fs::write(&temp_path, encoded.as_bytes())?;
                if case_index == 1 {
                    File::open(&temp_path)?.sync_all()?;
                }
                None
            };

            let previous = durable_test_manifest(1, vec![segment_a.clone()]);
            if case_index == 7 {
                write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
            } else if case_index >= 8 {
                let current = durable_test_manifest(
                    2,
                    vec![segment_a.clone(), segment_b.clone().expect("segment B")],
                );
                write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
                write_manifest(&directory.path().join("MANIFEST"), &current)?;
            } else {
                write_manifest(&directory.path().join("MANIFEST"), &previous)?;
            }

            if matches!(case_index, 5..=7) {
                let next = durable_test_manifest(
                    2,
                    vec![segment_a.clone(), segment_b.clone().expect("segment B")],
                );
                let temp_manifest = directory.path().join(".tmp-manifest-2");
                write_manifest(&temp_manifest, &next)?;
                if case_index == 6 {
                    File::open(&temp_manifest)?.sync_all()?;
                }
            }
            if case_index == 3 {
                sync_directory(directory.path())?;
            }
            if case_index == 4 || case_index == 9 {
                std::fs::write(
                    directory.path().join(format!("{segment_b_name}.fec")),
                    b"sidecar",
                )?;
            }

            let before = directory_bytes(directory.path())?;
            let first = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .unwrap_or_else(|error| panic!("{label}: first recovery failed: {error}"));
            let second = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .unwrap_or_else(|error| panic!("{label}: second recovery failed: {error}"));
            assert_eq!(first.loaded_manifest(), second.loaded_manifest(), "{label}");
            assert_eq!(directory_bytes(directory.path())?, before, "{label}");
            let expected_committed = if case_index >= 8 { 2 } else { 1 };
            assert_eq!(first.segments().len(), expected_committed, "{label}");
            for segment in first.segments() {
                assert_eq!(
                    segment.section(SectionKind::TERMDICT)?,
                    Some(b"termdict".as_slice()),
                    "{label}"
                );
            }

            let now = SystemTime::now()
                .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
                .expect("test clock remains representable");
            let report = collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            )?;
            let mut expected_removed = match case_index {
                0 | 1 => vec![PathBuf::from(".tmp-segment-000000000000000b")],
                2 | 3 => vec![PathBuf::from(&segment_b_name)],
                4 => vec![
                    PathBuf::from(&segment_b_name),
                    PathBuf::from(format!("{segment_b_name}.fec")),
                ],
                5..=7 => vec![
                    PathBuf::from(".tmp-manifest-2"),
                    PathBuf::from(&segment_b_name),
                ],
                8 | 9 => Vec::new(),
                _ => unreachable!(),
            };
            expected_removed.sort_unstable();
            assert_eq!(report.removed, expected_removed, "{label}");

            let mut expected_inventory = before;
            expected_inventory.retain(|(name, _)| {
                !report
                    .removed
                    .iter()
                    .any(|removed| removed.as_os_str() == name)
            });
            assert_eq!(
                directory_bytes(directory.path())?,
                expected_inventory,
                "{label}"
            );
            let reopened = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)
                .unwrap_or_else(|error| panic!("{label}: post-GC recovery failed: {error}"));
            assert_eq!(
                reopened.loaded_manifest(),
                first.loaded_manifest(),
                "{label}"
            );
            assert_eq!(
                collect_writer_garbage_at(
                    directory.path(),
                    DEFAULT_SCHEMA,
                    GarbageCollectionOptions::default(),
                    now,
                )?,
                GarbageCollectionReport {
                    removed: Vec::new()
                },
                "{label}"
            );
        }
        Ok(())
    }

    #[test]
    fn writer_gc_unions_both_slots_honors_grace_and_is_idempotent() -> TestResult {
        let directory = tempdir()?;
        let segment_a = write_test_segment(directory.path(), 0xa, 1, 0, 2)?;
        let segment_b = write_test_segment(directory.path(), 0xb, 3, 2, 4)?;
        let segment_c = write_test_segment(directory.path(), 0xc, 2, 4, 6)?;
        let segment_e = write_test_segment(directory.path(), 0xe, 4, 6, 8)?;
        let segment_f = write_test_segment(directory.path(), 0x10, 5, 8, 10)?;
        let previous = durable_test_manifest(1, vec![segment_a.clone(), segment_c.clone()]);
        let mut current = durable_test_manifest(2, vec![segment_a, segment_b]);
        current.docid_high_watermark = previous.docid_high_watermark;
        write_manifest(&directory.path().join("MANIFEST.prev"), &previous)?;
        write_manifest(&directory.path().join("MANIFEST"), &current)?;

        std::fs::write(directory.path().join(".tmp-manifest-3"), b"uncommitted")?;
        std::fs::write(
            directory.path().join(format!(
                "{}.fec",
                canonical_segment_name(segment_e.segment_id)
            )),
            b"orphan sidecar",
        )?;
        let fresh_base = directory
            .path()
            .join(canonical_segment_name(segment_f.segment_id));
        let old_sidecar_with_fresh_base = directory.path().join(format!(
            "{}.fec",
            canonical_segment_name(segment_f.segment_id)
        ));
        std::fs::write(&old_sidecar_with_fresh_base, b"sidecar older than base")?;
        File::options().write(true).open(&fresh_base)?.set_times(
            std::fs::FileTimes::new().set_modified(
                SystemTime::now()
                    .checked_add(Duration::from_secs(3_600))
                    .expect("test clock remains representable"),
            ),
        )?;
        std::fs::write(
            directory.path().join(format!(
                "{}.fec",
                canonical_segment_name(segment_c.segment_id)
            )),
            b"previous sidecar",
        )?;
        std::fs::write(
            directory.path().join("MANIFEST.fec"),
            b"live manifest sidecar",
        )?;
        std::fs::write(directory.path().join("notes.fec"), b"user data")?;
        std::fs::write(directory.path().join(".quarantine"), b"operator data")?;
        let quarantine_name = format!("{}.quarantine", canonical_segment_name(0xf));
        std::fs::write(
            directory.path().join(&quarantine_name),
            b"quarantined operator data",
        )?;
        std::fs::write(
            directory.path().join("SEG-000000000000000f.fslx"),
            b"user data",
        )?;
        std::fs::create_dir(directory.path().join(".tmp-directory"))?;

        #[cfg(unix)]
        let outside_fixture = {
            use std::os::unix::ffi::OsStringExt;

            let outside = tempdir()?;
            let target = outside.path().join("outside-data");
            std::fs::write(&target, b"outside")?;
            std::os::unix::fs::symlink(
                &target,
                directory.path().join("seg-000000000000000f.fslx"),
            )?;
            std::os::unix::fs::symlink(&target, directory.path().join(".tmp-link"))?;
            std::fs::write(
                directory
                    .path()
                    .join(OsString::from_vec(b".tmp-nonutf8-\xff".to_vec())),
                b"non-UTF-8 user data",
            )?;

            let before_open = directory_bytes(directory.path())?;
            let snapshot = KeeperSnapshot::open(directory.path(), DEFAULT_SCHEMA)?;
            assert_eq!(directory_bytes(directory.path())?, before_open);
            assert_eq!(snapshot.segments().len(), 2);
            assert_eq!(std::fs::read(&target)?, b"outside");
            (outside, target)
        };

        let fresh = collect_writer_garbage_under_lock(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
        )?;
        assert!(
            fresh.is_empty(),
            "fresh garbage must survive the grace window"
        );
        assert!(
            directory
                .path()
                .join(canonical_segment_name(segment_e.segment_id))
                .exists()
        );

        std::fs::write(directory.path().join("gen-2.claim"), [])?;
        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");
        let report = collect_writer_garbage_at(
            directory.path(),
            DEFAULT_SCHEMA,
            GarbageCollectionOptions::default(),
            now,
        )?;
        let orphan_name = canonical_segment_name(segment_e.segment_id);
        assert_eq!(
            report.removed,
            vec![
                PathBuf::from(".tmp-manifest-3"),
                PathBuf::from("gen-2.claim"),
                PathBuf::from(&orphan_name),
                PathBuf::from(format!("{orphan_name}.fec")),
            ]
        );
        assert!(directory.path().join(canonical_segment_name(0xc)).exists());
        assert!(fresh_base.exists());
        assert!(old_sidecar_with_fresh_base.exists());
        assert!(
            directory
                .path()
                .join(format!("{}.fec", canonical_segment_name(0xc)))
                .exists()
        );
        assert!(directory.path().join("notes.fec").exists());
        assert!(directory.path().join(".quarantine").exists());
        assert!(directory.path().join(&quarantine_name).exists());
        assert!(directory.path().join("SEG-000000000000000f.fslx").exists());
        assert!(directory.path().join(".tmp-directory").is_dir());
        #[cfg(unix)]
        {
            assert!(directory.path().join(".tmp-link").is_symlink());
            assert!(
                directory
                    .path()
                    .join("seg-000000000000000f.fslx")
                    .is_symlink()
            );
            use std::os::unix::ffi::OsStringExt;
            assert!(
                directory
                    .path()
                    .join(OsString::from_vec(b".tmp-nonutf8-\xff".to_vec()))
                    .exists()
            );
            assert_eq!(std::fs::read(&outside_fixture.1)?, b"outside");
        }
        assert!(
            collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            )?
            .is_empty()
        );
        Ok(())
    }

    #[test]
    fn future_claim_blocks_gc_before_any_deletion() -> TestResult {
        let directory = tempdir()?;
        write_manifest(
            &directory.path().join("MANIFEST"),
            &durable_test_manifest(2, Vec::new()),
        )?;
        std::fs::write(directory.path().join(".tmp-old"), b"must survive")?;
        std::fs::write(directory.path().join("gen-3.claim"), [])?;
        let before = directory_bytes(directory.path())?;
        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");

        assert!(matches!(
            collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::ClaimedGenerationPending {
                current: 2,
                claimed: 3,
                ..
            })
        ));
        assert_eq!(directory_bytes(directory.path())?, before);
        Ok(())
    }

    #[test]
    fn stale_claim_still_honors_the_gc_grace_period() -> TestResult {
        let directory = tempdir()?;
        write_manifest(
            &directory.path().join("MANIFEST"),
            &durable_test_manifest(2, Vec::new()),
        )?;
        std::fs::write(directory.path().join("gen-2.claim"), [])?;

        assert!(
            collect_writer_garbage_under_lock(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
            )?
            .is_empty()
        );
        assert!(directory.path().join("gen-2.claim").exists());

        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");
        assert_eq!(
            collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            )?
            .removed,
            vec![PathBuf::from("gen-2.claim")]
        );
        Ok(())
    }

    #[test]
    fn malformed_claim_blocks_gc_before_any_deletion() -> TestResult {
        let directory = tempdir()?;
        write_manifest(
            &directory.path().join("MANIFEST"),
            &durable_test_manifest(2, Vec::new()),
        )?;
        std::fs::write(directory.path().join(".tmp-old"), b"must survive")?;
        std::fs::create_dir(directory.path().join("gen-2.claim"))?;
        let before = directory_bytes(directory.path())?;
        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");

        assert!(matches!(
            collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::InvalidClaimArtifact { .. })
        ));
        assert_eq!(directory_bytes(directory.path())?, before);
        assert!(directory.path().join("gen-2.claim").is_dir());

        let nonzero = tempdir()?;
        write_manifest(
            &nonzero.path().join("MANIFEST"),
            &durable_test_manifest(2, Vec::new()),
        )?;
        std::fs::write(nonzero.path().join(".tmp-old"), b"must survive")?;
        std::fs::write(nonzero.path().join("gen-2.claim"), b"not empty")?;
        let before = directory_bytes(nonzero.path())?;
        assert!(matches!(
            collect_writer_garbage_at(
                nonzero.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::InvalidClaimArtifact { .. })
        ));
        assert_eq!(directory_bytes(nonzero.path())?, before);

        #[cfg(unix)]
        {
            let symlinked = tempdir()?;
            write_manifest(
                &symlinked.path().join("MANIFEST"),
                &durable_test_manifest(2, Vec::new()),
            )?;
            std::fs::write(symlinked.path().join(".tmp-old"), b"must survive")?;
            let target = symlinked.path().join("claim-target");
            std::fs::write(&target, [])?;
            std::os::unix::fs::symlink(&target, symlinked.path().join("gen-2.claim"))?;
            let before = directory_bytes(symlinked.path())?;
            assert!(matches!(
                collect_writer_garbage_at(
                    symlinked.path(),
                    DEFAULT_SCHEMA,
                    GarbageCollectionOptions::default(),
                    now,
                ),
                Err(KeeperError::InvalidClaimArtifact { .. })
            ));
            assert_eq!(directory_bytes(symlinked.path())?, before);
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn gc_rejects_symlinked_directory_alias_without_touching_target() -> TestResult {
        let target = tempdir()?;
        std::fs::write(target.path().join(".tmp-old"), b"must survive")?;
        let alias_root = tempdir()?;
        let alias = alias_root.path().join("index-alias");
        std::os::unix::fs::symlink(target.path(), &alias)?;
        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");

        assert!(matches!(
            collect_abandoned_genesis_garbage_at(&alias, GarbageCollectionOptions::default(), now,),
            Err(KeeperError::Io {
                operation: "open no-follow garbage-collection directory",
                ..
            })
        ));
        assert_eq!(
            std::fs::read(target.path().join(".tmp-old"))?,
            b"must survive"
        );
        Ok(())
    }

    #[test]
    fn genesis_gc_requires_absent_slots_and_no_pending_claim() -> TestResult {
        let directory = tempdir()?;
        let segment = write_test_segment(directory.path(), 0x44, 1, 0, 2)?;
        write_manifest(
            &directory.path().join(".tmp-manifest-1"),
            &durable_test_manifest(1, vec![segment.clone()]),
        )?;
        std::fs::write(directory.path().join(".tmp-write-buffer"), b"uncommitted")?;
        std::fs::write(directory.path().join("MANIFEST.fec"), b"orphan sidecar")?;
        std::fs::write(
            directory.path().join("MANIFEST.fec.tmp"),
            b"interrupted manifest sidecar",
        )?;
        let segment_sidecar_temp = directory.path().join(format!(
            "{}.fec.tmp",
            canonical_segment_name(segment.segment_id)
        ));
        std::fs::write(&segment_sidecar_temp, b"interrupted segment sidecar")?;
        std::fs::write(directory.path().join("notes"), b"operator data")?;
        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");

        let report = collect_abandoned_genesis_garbage_at(
            directory.path(),
            GarbageCollectionOptions::default(),
            now,
        )?;
        let segment_name = canonical_segment_name(segment.segment_id);
        assert_eq!(
            report.removed,
            vec![
                PathBuf::from(".tmp-manifest-1"),
                PathBuf::from(".tmp-write-buffer"),
                PathBuf::from("MANIFEST.fec"),
                PathBuf::from("MANIFEST.fec.tmp"),
                PathBuf::from(segment_name),
                segment_sidecar_temp
                    .file_name()
                    .expect("sidecar temp has a file name")
                    .into(),
            ]
        );
        assert_eq!(
            std::fs::read(directory.path().join("notes"))?,
            b"operator data"
        );
        assert!(
            collect_abandoned_genesis_garbage_at(
                directory.path(),
                GarbageCollectionOptions::default(),
                now,
            )?
            .is_empty()
        );

        let claimed = tempdir()?;
        std::fs::write(claimed.path().join(".tmp-manifest-1"), b"must survive")?;
        std::fs::write(claimed.path().join("gen-1.claim"), [])?;
        let before = directory_bytes(claimed.path())?;
        assert!(matches!(
            collect_abandoned_genesis_garbage_at(
                claimed.path(),
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::ClaimedGenerationPending {
                current: 0,
                claimed: 1,
                ..
            })
        ));
        assert_eq!(directory_bytes(claimed.path())?, before);

        let corrupt = tempdir()?;
        std::fs::write(corrupt.path().join("MANIFEST"), b"corrupt")?;
        std::fs::write(corrupt.path().join(".tmp-old"), b"must survive")?;
        let before = directory_bytes(corrupt.path())?;
        assert!(matches!(
            collect_abandoned_genesis_garbage_at(
                corrupt.path(),
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::ManifestCorrupted { .. })
        ));
        assert_eq!(directory_bytes(corrupt.path())?, before);
        Ok(())
    }

    #[test]
    fn failed_writer_recovery_never_collects_and_path_guard_rejects_escape() -> TestResult {
        let directory = tempdir()?;
        let missing = ManifestSegment {
            segment_id: 0xdead,
            seal_seq: 1,
            file_len: 100,
            file_xxh3: 7,
            docid_lo: 0,
            docid_hi: 2,
            doc_count: 1,
            tombstones: EMPTY_TOMBSTONES.to_vec(),
        };
        write_manifest(
            &directory.path().join("MANIFEST"),
            &durable_test_manifest(1, vec![missing]),
        )?;
        std::fs::write(directory.path().join(".tmp-old"), b"must survive")?;
        let before = directory_bytes(directory.path())?;
        let now = SystemTime::now()
            .checked_add(DEFAULT_GARBAGE_GRACE + Duration::from_secs(1))
            .expect("test clock remains representable");
        assert!(matches!(
            collect_writer_garbage_at(
                directory.path(),
                DEFAULT_SCHEMA,
                GarbageCollectionOptions::default(),
                now,
            ),
            Err(KeeperError::SegmentOpen { .. })
        ));
        assert_eq!(directory_bytes(directory.path())?, before);

        for unsafe_path in [
            Path::new("../outside"),
            Path::new("nested/file"),
            Path::new("/absolute"),
        ] {
            assert!(matches!(
                safe_direct_child(directory.path(), unsafe_path),
                Err(KeeperError::UnsafeGarbagePath { .. })
            ));
        }
        assert_eq!(
            safe_direct_child(directory.path(), Path::new(".tmp-owned"))?,
            directory.path().join(".tmp-owned")
        );
        assert!(classify_garbage_candidate(OsStr::new("notes.fec")).is_none());
        assert!(classify_garbage_candidate(OsStr::new("notes.fec.tmp")).is_none());
        assert!(classify_garbage_candidate(OsStr::new("SEG-0000000000000001.fslx")).is_none());
        assert!(matches!(
            classify_garbage_candidate(OsStr::new("seg-0000000000000001.fslx.fec.tmp")),
            Some(GarbageCandidate::Temporary)
        ));
        assert!(matches!(
            classify_garbage_candidate(OsStr::new("MANIFEST.fec.tmp")),
            Some(GarbageCandidate::Temporary)
        ));
        Ok(())
    }

    fn run_with_test_cx<T, F, Fut>(operation: F) -> T
    where
        T: Send + 'static,
        F: FnOnce(Cx) -> Fut + 'static,
        Fut: std::future::Future<Output = T> + 'static,
    {
        let output = Arc::new(std::sync::Mutex::new(None));
        let task_output = Arc::clone(&output);
        asupersync::test_utils::run_test_with_cx(move |cx| async move {
            let value = operation(cx).await;
            let mut slot = task_output
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *slot = Some(value);
        });
        let output = Arc::try_unwrap(output)
            .unwrap_or_else(|_| panic!("test runtime retained its result slot"));
        output
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .expect("test runtime produced a result")
    }

    #[cfg(unix)]
    struct WriterChild {
        child: Option<std::process::Child>,
    }

    #[cfg(unix)]
    impl WriterChild {
        fn child_mut(&mut self) -> &mut std::process::Child {
            self.child.as_mut().expect("child is still owned")
        }

        fn wait_success(mut self) -> TestResult {
            let status = self.child.take().expect("child is still owned").wait()?;
            if !status.success() {
                return Err(format!("writer child exited with {status}").into());
            }
            Ok(())
        }

        fn kill_and_reap(mut self) -> TestResult {
            let mut child = self.child.take().expect("child is still owned");
            let kill_result = child.kill();
            let wait_result = child.wait();
            kill_result?;
            let status = wait_result?;
            if status.success() {
                return Err("killed writer child unexpectedly exited successfully".into());
            }
            Ok(())
        }
    }

    #[cfg(unix)]
    impl Drop for WriterChild {
        fn drop(&mut self) {
            if let Some(child) = &mut self.child {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    #[cfg(unix)]
    fn spawn_writer_child(
        role: &str,
        directory: &Path,
        control: &Path,
        identifier: &str,
    ) -> Result<WriterChild, Box<dyn std::error::Error>> {
        let child = std::process::Command::new(std::env::current_exe()?)
            .arg("--exact")
            .arg("keeper::tests::writer_lock_child_dispatch")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env("QUILL_WRITER_LOCK_ROLE", role)
            .env("QUILL_WRITER_LOCK_DIRECTORY", directory)
            .env("QUILL_WRITER_LOCK_CONTROL", control)
            .env("QUILL_WRITER_LOCK_ID", identifier)
            .spawn()?;
        Ok(WriterChild { child: Some(child) })
    }

    #[cfg(unix)]
    fn wait_for_child_marker(child: &mut WriterChild, marker: &Path, label: &str) -> TestResult {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if marker.exists() {
                return Ok(());
            }
            if let Some(status) = child.child_mut().try_wait()? {
                return Err(format!("{label} exited before its marker: {status}").into());
            }
            if std::time::Instant::now() >= deadline {
                return Err(format!("timed out waiting for {label}").into());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(unix)]
    fn create_test_index(directory: &Path) -> Result<(), String> {
        let directory = directory.to_path_buf();
        run_with_test_cx(move |cx| async move {
            KeeperSnapshot::create(&cx, directory, DEFAULT_SCHEMA)
                .await
                .map(|_| ())
                .map_err(|error| error.to_string())
        })
    }

    #[cfg(unix)]
    #[test]
    fn writer_lock_child_dispatch() {
        let Ok(role) = std::env::var("QUILL_WRITER_LOCK_ROLE") else {
            return;
        };
        let directory = PathBuf::from(
            std::env::var_os("QUILL_WRITER_LOCK_DIRECTORY")
                .expect("child index directory is configured"),
        );
        let control = PathBuf::from(
            std::env::var_os("QUILL_WRITER_LOCK_CONTROL")
                .expect("child control directory is configured"),
        );
        let identifier = std::env::var("QUILL_WRITER_LOCK_ID").expect("child id is configured");
        let result: Result<(), String> = run_with_test_cx(move |cx| async move {
            if role == "contend" {
                let start = control.join("START");
                while !start.exists() {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            match KeeperWriter::open(&cx, &directory, DEFAULT_SCHEMA).await {
                Ok(writer) => {
                    if role == "hold_unpublished" {
                        std::fs::write(
                            directory.join("seg-feedfacefeedface.fslx"),
                            b"sealed but not yet published",
                        )
                        .map_err(|error| error.to_string())?;
                    }
                    let marker = if role == "contend" {
                        control.join(format!("winner-{identifier}"))
                    } else {
                        control.join(format!("ready-{identifier}"))
                    };
                    std::fs::write(&marker, []).map_err(|error| error.to_string())?;
                    let release = control.join("RELEASE");
                    while !release.exists() {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    drop(writer);
                    Ok(())
                }
                Err(KeeperError::WriterBusy { .. }) if role == "contend" => {
                    std::fs::write(control.join(format!("busy-{identifier}")), [])
                        .map_err(|error| error.to_string())
                }
                Err(error) => Err(error.to_string()),
            }
        });
        result.unwrap_or_else(|error| panic!("writer child failed: {error}"));
    }

    #[test]
    fn writer_lock_wire_golden_and_corruption_matrix() {
        let record = WriterLockRecord {
            pid: 42,
            pid_start_nonce: 0x1122_3344_5566_7788,
            acquired_unix_s: 1_700_000_000,
        };
        let golden = hex_bytes(
            "46534c584c434b00010000002a0000008877665544332211\
             00f153650000000044f89139",
        );
        assert_eq!(record.to_bytes().as_slice(), golden);
        assert_eq!(WriterLockRecord::from_bytes(&golden), Ok(record));
        for length in 0..WRITER_LOCK_RECORD_BYTES {
            assert!(WriterLockRecord::from_bytes(&golden[..length]).is_err());
        }
        let mut bad_magic = golden.clone();
        bad_magic[0] ^= 0xff;
        assert!(WriterLockRecord::from_bytes(&bad_magic).is_err());
        let mut bad_version = golden.clone();
        bad_version[8] = 2;
        let checksum = crc32fast::hash(&bad_version[..32]);
        bad_version[32..].copy_from_slice(&checksum.to_le_bytes());
        assert!(WriterLockRecord::from_bytes(&bad_version).is_err());
        let mut zero_pid = golden.clone();
        zero_pid[12..16].copy_from_slice(&0_u32.to_le_bytes());
        let checksum = crc32fast::hash(&zero_pid[..32]);
        zero_pid[32..].copy_from_slice(&checksum.to_le_bytes());
        assert!(WriterLockRecord::from_bytes(&zero_pid).is_err());
        let mut bad_crc = golden;
        bad_crc[20] ^= 1;
        assert!(WriterLockRecord::from_bytes(&bad_crc).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn writer_admission_is_exclusive_reusable_and_corruption_fails_closed() -> TestResult {
        let directory = tempdir()?;
        let first = acquire_writer_admission(directory.path())?;
        let lock_path = directory.path().join("LOCK");
        let active_bytes = std::fs::read(&lock_path)?;
        assert_eq!(active_bytes.len(), WRITER_LOCK_RECORD_BYTES);
        assert!(matches!(
            acquire_writer_admission(directory.path()),
            Err(KeeperError::WriterBusy {
                owner_pid: Some(pid),
                ..
            }) if pid == std::process::id()
        ));
        assert_eq!(std::fs::read(&lock_path)?, active_bytes);
        drop(first);
        assert_eq!(std::fs::metadata(&lock_path)?.len(), 0);

        let second = acquire_writer_admission(directory.path())?;
        drop(second);
        std::fs::write(&lock_path, b"truncated")?;
        let before = std::fs::read(&lock_path)?;
        assert!(matches!(
            acquire_writer_admission(directory.path()),
            Err(KeeperError::WriterLockCorrupted { .. })
        ));
        assert_eq!(std::fs::read(lock_path)?, before);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn old_guards_preserve_replacement_lock_and_claim_names() -> TestResult {
        let directory = tempdir()?;
        let admission = acquire_writer_admission(directory.path())?;
        let original_lock = directory.path().join("LOCK");
        let moved_lock = directory.path().join("LOCK.old");
        std::fs::rename(&original_lock, &moved_lock)?;
        let replacement = WriterLockRecord {
            pid: std::process::id(),
            pid_start_nonce: 0xfeed_face,
            acquired_unix_s: 1_700_000_001,
        }
        .to_bytes();
        std::fs::write(&original_lock, replacement)?;
        assert!(matches!(
            admission.ensure_directory_identity(),
            Err(KeeperError::WriterLockCorrupted { .. })
        ));
        drop(admission);
        assert_eq!(std::fs::read(&original_lock)?, replacement);
        assert_eq!(std::fs::metadata(&moved_lock)?.len(), 0);

        let claims = tempdir()?;
        let admission = acquire_writer_admission(claims.path())?;
        let claim = GenerationClaimGuard::acquire(Arc::clone(&admission), 1)?;
        assert!(matches!(
            GenerationClaimGuard::acquire(Arc::clone(&admission), 1),
            Err(KeeperError::GenerationClaimConflict { generation: 1, .. })
        ));
        let claim_path = claims.path().join("gen-1.claim");
        let moved_claim = claims.path().join("gen-1.claim.old");
        std::fs::rename(&claim_path, moved_claim)?;
        std::fs::write(&claim_path, [])?;
        drop(claim);
        assert!(
            claim_path.exists(),
            "old guard must preserve replacement claim"
        );
        drop(admission);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn sigstop_writer_is_never_reaped_and_readers_remain_touchless() -> TestResult {
        let index = tempdir()?;
        create_test_index(index.path()).map_err(io::Error::other)?;
        let control = tempdir()?;
        let ready = control.path().join("ready-holder");
        let mut child =
            spawn_writer_child("hold_unpublished", index.path(), control.path(), "holder")?;
        wait_for_child_marker(&mut child, &ready, "writer holder")?;
        assert!(index.path().join("seg-feedfacefeedface.fslx").exists());

        let lock_path = index.path().join("LOCK");
        let lock_file = OpenOptions::new().write(true).open(&lock_path)?;
        lock_file.set_times(std::fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))?;
        let before = directory_bytes(index.path())?;
        rustix::process::kill_process(
            rustix::process::Pid::from_child(child.child_mut()),
            rustix::process::Signal::STOP,
        )?;

        let contender_path = index.path().to_path_buf();
        let blocked: Result<(), String> = run_with_test_cx(move |cx| async move {
            match KeeperWriter::open(&cx, contender_path, DEFAULT_SCHEMA).await {
                Err(KeeperError::WriterBusy { .. }) => Ok(()),
                Ok(writer) => {
                    drop(writer);
                    Err("SIGSTOP holder was incorrectly taken over".to_owned())
                }
                Err(error) => Err(format!("unexpected contender failure: {error}")),
            }
        });
        blocked.map_err(io::Error::other)?;
        KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?;
        KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?;
        assert_eq!(directory_bytes(index.path())?, before);

        rustix::process::kill_process(
            rustix::process::Pid::from_child(child.child_mut()),
            rustix::process::Signal::CONT,
        )?;
        std::fs::write(control.path().join("RELEASE"), [])?;
        child.wait_success()?;
        let reopen_path = index.path().to_path_buf();
        let reopened: Result<(), String> = run_with_test_cx(move |cx| async move {
            KeeperWriter::open(&cx, reopen_path, DEFAULT_SCHEMA)
                .await
                .map(drop)
                .map_err(|error| error.to_string())
        });
        reopened.map_err(io::Error::other)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn two_processes_racing_dead_owner_takeover_admit_exactly_one() -> TestResult {
        let index = tempdir()?;
        create_test_index(index.path()).map_err(io::Error::other)?;
        let dead_control = tempdir()?;
        let mut dead = spawn_writer_child("hold", index.path(), dead_control.path(), "dead")?;
        wait_for_child_marker(
            &mut dead,
            &dead_control.path().join("ready-dead"),
            "dead-owner fixture",
        )?;
        dead.kill_and_reap()?;
        assert_eq!(
            std::fs::metadata(index.path().join("LOCK"))?.len(),
            usize_to_u64(WRITER_LOCK_RECORD_BYTES)
        );

        let control = tempdir()?;
        let mut first = spawn_writer_child("contend", index.path(), control.path(), "a")?;
        let mut second = spawn_writer_child("contend", index.path(), control.path(), "b")?;
        std::fs::write(control.path().join("START"), [])?;
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let first_done =
                control.path().join("winner-a").exists() || control.path().join("busy-a").exists();
            let second_done =
                control.path().join("winner-b").exists() || control.path().join("busy-b").exists();
            if first_done && second_done {
                break;
            }
            if std::time::Instant::now() >= deadline {
                return Err("timed out waiting for takeover contenders".into());
            }
            if let Some(status) = first.child_mut().try_wait()?
                && !first_done
            {
                return Err(format!("first contender exited early: {status}").into());
            }
            if let Some(status) = second.child_mut().try_wait()?
                && !second_done
            {
                return Err(format!("second contender exited early: {status}").into());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let winners = usize::from(control.path().join("winner-a").exists())
            + usize::from(control.path().join("winner-b").exists());
        let busy = usize::from(control.path().join("busy-a").exists())
            + usize::from(control.path().join("busy-b").exists());
        assert_eq!(winners, 1);
        assert_eq!(busy, 1);
        std::fs::write(control.path().join("RELEASE"), [])?;
        first.wait_success()?;
        second.wait_success()?;
        assert_eq!(std::fs::metadata(index.path().join("LOCK"))?.len(), 0);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn keeper_writer_claims_generations_and_abandons_interrupted_temp() -> TestResult {
        let index = tempdir()?;
        let directory = index.path().to_path_buf();
        let first: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            assert_eq!(
                std::fs::metadata(directory.join("LOCK"))
                    .map_err(|error| error.to_string())?
                    .len(),
                usize_to_u64(WRITER_LOCK_RECORD_BYTES)
            );
            let schema_id = DEFAULT_SCHEMA
                .schema_id()
                .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &Manifest::empty(2, schema_id, 0))
                .await
                .map_err(|error| error.to_string())?;
            assert!(!directory.join("gen-2.claim").exists());
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 2);
            drop(writer);
            Ok(())
        });
        first.map_err(io::Error::other)?;
        assert_eq!(std::fs::metadata(index.path().join("LOCK"))?.len(), 0);

        let schema_id = DEFAULT_SCHEMA.schema_id()?;
        let interrupted = Manifest::empty(3, schema_id, 0).to_bytes()?;
        std::fs::write(index.path().join(".tmp-manifest-3"), interrupted)?;
        std::fs::write(index.path().join("gen-3.claim"), [])?;
        let directory = index.path().to_path_buf();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::open(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 2);
            assert!(!directory.join(".tmp-manifest-3").exists());
            assert!(!directory.join("gen-3.claim").exists());
            writer
                .publish(&cx, &Manifest::empty(3, schema_id, 0))
                .await
                .map_err(|error| error.to_string())?;
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 3);
            Ok(())
        });
        recovered.map_err(io::Error::other)?;
        assert!(std::fs::read_dir(index.path())?.any(|entry| {
            entry
                .ok()
                .and_then(|entry| entry.file_name().into_string().ok())
                .is_some_and(|name| name.starts_with(".tmp-abandoned-manifest-3-"))
        }));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn writer_preflights_missing_segment_before_manifest_publication() -> TestResult {
        let index = tempdir()?;
        let encoded = encoded_test_segment(0xdead, 0, 1, 1)?;
        let proposed = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
        let directory = index.path().to_path_buf();
        let rejected: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            let manifest_path = directory.join("MANIFEST");
            let before = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            match writer.publish(&cx, &proposed).await {
                Err(KeeperError::SegmentOpen { .. }) => {}
                Err(error) => return Err(format!("unexpected publish failure: {error}")),
                Ok(_) => return Err("missing segment was published as authority".to_owned()),
            }
            let after = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            if after != before {
                return Err("MANIFEST changed before segment preflight failed".to_owned());
            }
            if writer.snapshot().loaded_manifest().manifest.generation != 1 {
                return Err("writer snapshot advanced after rejected publication".to_owned());
            }
            Ok(())
        });
        rejected.map_err(io::Error::other)?;
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            1
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn writer_validates_manifest_before_segment_preflight() -> TestResult {
        let index = tempdir()?;
        let encoded = encoded_test_segment(0xbad, 0, 1, 1)?;
        let mut proposed = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
        proposed.generation = 0;
        let directory = index.path().to_path_buf();
        let rejected: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            match writer.publish(&cx, &proposed).await {
                Err(KeeperError::InvalidManifest { .. }) => Ok(()),
                Err(error) => Err(format!("unexpected publish failure: {error}")),
                Ok(_) => Err("invalid manifest reached segment preflight".to_owned()),
            }
        });
        rejected.map_err(io::Error::other)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn writer_preflights_corrupt_segment_before_manifest_publication() -> TestResult {
        let index = tempdir()?;
        let encoded = encoded_test_segment(0xcafe, 0, 1, 1)?;
        let section_offset = usize::try_from(encoded.section_entries()[0].offset)?;
        let proposed = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
        let directory = index.path().to_path_buf();
        let rejected: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            let pending = encoded
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            let published = writer
                .publish_segment(&cx, pending)
                .await
                .map_err(|error| error.to_string())?;
            let mut bytes = std::fs::read(&published).map_err(|error| error.to_string())?;
            bytes[section_offset] ^= 0x80;
            std::fs::write(&published, bytes).map_err(|error| error.to_string())?;

            let manifest_path = directory.join("MANIFEST");
            let before = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            match writer.publish(&cx, &proposed).await {
                Err(KeeperError::SegmentOpen { .. }) => {}
                Err(error) => return Err(format!("unexpected publish failure: {error}")),
                Ok(_) => return Err("corrupt segment was published as authority".to_owned()),
            }
            let after = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            if after != before {
                return Err("MANIFEST changed before segment preflight failed".to_owned());
            }
            if writer.snapshot().loaded_manifest().manifest.generation != 1 {
                return Err("writer snapshot advanced after rejected publication".to_owned());
            }
            Ok(())
        });
        rejected.map_err(io::Error::other)?;
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            1
        );
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn durable_writer_preflights_segment_sidecar_before_manifest_publication() -> TestResult {
        let index = tempdir()?;
        let encoded = encoded_test_segment(0xfec, 0, 1, 1)?;
        let proposed = durable_test_manifest(2, vec![manifest_segment(&encoded, 1)]);
        let directory = index.path().to_path_buf();
        let protector = test_file_protector();
        let rejected: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, protector)
                    .await
                    .map_err(|error| error.to_string())?;
            let pending = encoded
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            let published = publish_pending_segment(pending).map_err(|error| error.to_string())?;
            if FileProtector::sidecar_path(&published).exists() {
                return Err(
                    "ordinary segment publication unexpectedly created a sidecar".to_owned(),
                );
            }

            let manifest_path = directory.join("MANIFEST");
            let before = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            match writer.publish(&cx, &proposed).await {
                Err(KeeperError::Durability {
                    operation: "preflight durable segment sidecar",
                    ..
                }) => {}
                Err(error) => return Err(format!("unexpected publish failure: {error}")),
                Ok(_) => return Err("unprotected segment became durable authority".to_owned()),
            }
            let after = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            if after != before {
                return Err("durable MANIFEST changed before sidecar preflight failed".to_owned());
            }
            Ok(())
        });
        rejected.map_err(io::Error::other)?;
        Ok(())
    }

    #[cfg(all(
        feature = "durability",
        any(target_os = "linux", target_os = "android", target_vendor = "apple")
    ))]
    #[test]
    fn recovered_byte_install_skips_stale_temps_and_quarantine_collisions() -> TestResult {
        let index = tempdir()?;
        let admission = acquire_writer_admission(index.path())?;
        let destination = index.path().join("MANIFEST");
        std::fs::write(&destination, b"first corrupt authority")?;
        let base = format!(
            ".tmp-repair-MANIFEST-{:016x}",
            admission.record.pid_start_nonce
        );
        let stale_temp = index.path().join(&base);
        std::fs::write(&stale_temp, b"interrupted prior repair")?;

        install_recovered_bytes(&admission, "MANIFEST", &destination, b"first recovery")?;
        assert_eq!(std::fs::read(&destination)?, b"first recovery");
        assert_eq!(std::fs::read(&stale_temp)?, b"interrupted prior repair");

        std::fs::write(&destination, b"second corrupt authority")?;
        install_recovered_bytes(&admission, "MANIFEST", &destination, b"second recovery")?;
        assert_eq!(std::fs::read(&destination)?, b"second recovery");
        assert_eq!(std::fs::read(&stale_temp)?, b"interrupted prior repair");
        assert_eq!(
            std::fs::read_dir(index.path())?
                .filter_map(Result::ok)
                .filter_map(|entry| entry.file_name().into_string().ok())
                .filter(|name| name.starts_with(".tmp-corrupt-MANIFEST-"))
                .count(),
            2
        );
        Ok(())
    }

    #[cfg(all(
        feature = "durability",
        any(target_os = "linux", target_os = "android", target_vendor = "apple")
    ))]
    #[test]
    fn recovered_byte_install_rolls_back_a_substituted_ready_link() -> TestResult {
        let index = tempdir()?;
        let admission = acquire_writer_admission(index.path())?;
        let destination = index.path().join("MANIFEST");
        let prior = Manifest::empty(1, DEFAULT_SCHEMA.schema_id()?, 0).to_bytes()?;
        std::fs::write(&destination, &prior)?;
        let displaced_ready = index.path().join("displaced-owned-ready-link");

        let error = install_recovered_bytes_with_hook(
            &admission,
            "MANIFEST",
            &destination,
            b"validated recovery",
            |ready| {
                std::fs::rename(ready, &displaced_ready)?;
                std::fs::write(ready, b"substituted bytes")
            },
        )
        .expect_err("a substituted ready link must not become authority");
        assert!(matches!(
            error,
            KeeperError::Io {
                operation: "verify installed recovered bytes",
                ..
            }
        ));
        let reopened = std::fs::read(&destination)?;
        assert_eq!(reopened, prior);
        assert_eq!(Manifest::from_bytes(&reopened)?.generation, 1);
        assert_eq!(std::fs::read(&displaced_ready)?, b"validated recovery");
        Ok(())
    }

    #[cfg(all(
        feature = "durability",
        any(target_os = "linux", target_os = "android", target_vendor = "apple")
    ))]
    #[test]
    fn recovered_byte_install_ignores_a_substituted_ready_link_when_destination_is_missing()
    -> TestResult {
        let index = tempdir()?;
        let admission = acquire_writer_admission(index.path())?;
        let destination = index.path().join("MANIFEST");
        let displaced_ready = index.path().join("displaced-owned-ready-link");

        install_recovered_bytes_with_hook(
            &admission,
            "MANIFEST",
            &destination,
            b"validated recovery",
            |ready| {
                std::fs::rename(ready, &displaced_ready)?;
                std::fs::write(ready, b"substituted bytes")
            },
        )?;
        assert_eq!(std::fs::read(&destination)?, b"validated recovery");
        assert_eq!(std::fs::read(&displaced_ready)?, b"validated recovery");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn writer_retires_corrupt_primary_and_recovers_from_previous() -> TestResult {
        let index = tempdir()?;
        let directory = index.path().to_path_buf();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::create(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            let schema_id = DEFAULT_SCHEMA
                .schema_id()
                .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &Manifest::empty(2, schema_id, 0))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;
        std::fs::write(index.path().join("MANIFEST"), b"corrupt primary")?;
        let before_reader = directory_bytes(index.path())?;
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            1
        );
        assert_eq!(directory_bytes(index.path())?, before_reader);

        let directory = index.path().to_path_buf();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer = KeeperWriter::open(&cx, &directory, DEFAULT_SCHEMA)
                .await
                .map_err(|error| error.to_string())?;
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 1);
            let schema_id = DEFAULT_SCHEMA
                .schema_id()
                .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &Manifest::empty(2, schema_id, 0))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        recovered.map_err(io::Error::other)?;
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            2
        );
        assert!(std::fs::read_dir(index.path())?.any(|entry| {
            entry
                .ok()
                .and_then(|entry| entry.file_name().into_string().ok())
                .is_some_and(|name| name.starts_with(".tmp-corrupt-manifest-"))
        }));
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn durable_writer_recovers_both_missing_or_corrupt_manifest_slots() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, Vec::new()))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let current_path = index.path().join("MANIFEST");
        let previous_path = index.path().join("MANIFEST.prev");
        let expected_current = std::fs::read(&current_path)?;
        let expected_previous = std::fs::read(&previous_path)?;
        let mut corrupt_current = expected_current.clone();
        corrupt_current[16] ^= 0xff;
        let mut corrupt_previous = expected_previous.clone();
        corrupt_previous[16] ^= 0xff;
        std::fs::write(&current_path, corrupt_current)?;
        std::fs::write(&previous_path, corrupt_previous)?;

        let directory = index.path().to_path_buf();
        let corrupt_protector = protector.clone();
        let recovered_corrupt: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, directory, DEFAULT_SCHEMA, corrupt_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 2);
            Ok(())
        });
        recovered_corrupt.map_err(io::Error::other)?;
        assert_eq!(std::fs::read(&current_path)?, expected_current);
        assert_eq!(std::fs::read(&previous_path)?, expected_previous);

        let saved_current = index.path().join("saved-MANIFEST");
        let saved_previous = index.path().join("saved-MANIFEST.prev");
        std::fs::rename(&current_path, &saved_current)?;
        std::fs::rename(&previous_path, &saved_previous)?;
        let directory = index.path().to_path_buf();
        let missing_protector = protector.clone();
        let recovered_missing: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::create_durable(&cx, directory, DEFAULT_SCHEMA, missing_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 2);
            Ok(())
        });
        recovered_missing.map_err(io::Error::other)?;
        assert_eq!(std::fs::read(&current_path)?, expected_current);
        assert_eq!(std::fs::read(&previous_path)?, expected_previous);
        assert_eq!(std::fs::read(saved_current)?, expected_current);
        assert_eq!(std::fs::read(saved_previous)?, expected_previous);
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn unrecoverable_reconstructed_current_preserves_previous_authority() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let encoded = encoded_test_segment(0xfeed, 0, 2, 1)?;
        let manifest_segment = manifest_segment(&encoded, 1);
        let setup_manifest_segment = manifest_segment.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            let pending = encoded
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            writer
                .publish_segment(&cx, pending)
                .await
                .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, vec![setup_manifest_segment]))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let current_path = index.path().join("MANIFEST");
        let segment_path = index
            .path()
            .join(canonical_segment_name(manifest_segment.segment_id));
        let mut corrupt_current = std::fs::read(&current_path)?;
        corrupt_current[16] ^= 0xff;
        std::fs::write(&current_path, &corrupt_current)?;
        let mut corrupt_segment = std::fs::read(&segment_path)?;
        corrupt_segment[64] ^= 0xff;
        std::fs::write(&segment_path, corrupt_segment)?;
        let segment_sidecar = FileProtector::sidecar_path(&segment_path);
        let saved_sidecar = index.path().join("saved-unrecoverable-segment.fec");
        std::fs::rename(&segment_sidecar, &saved_sidecar)?;

        let directory = index.path().to_path_buf();
        let repair_protector = protector.clone();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, repair_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            if writer.snapshot().loaded_manifest().manifest.generation != 1 {
                return Err("unrecoverable reconstructed current displaced previous".to_owned());
            }
            Ok(())
        });
        recovered.map_err(io::Error::other)?;

        assert!(!current_path.exists());
        let retired_current = std::fs::read_dir(index.path())?
            .filter_map(Result::ok)
            .find(|entry| {
                entry.file_name().into_string().is_ok_and(|name| {
                    name.starts_with(".tmp-corrupt-manifest-")
                        && !Path::new(&name)
                            .extension()
                            .is_some_and(|extension| extension.eq_ignore_ascii_case("fec"))
                })
            })
            .ok_or_else(|| io::Error::other("corrupt current was not quarantined"))?;
        assert_eq!(std::fs::read(retired_current.path())?, corrupt_current);
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            1,
            "the usable previous generation must remain reader-visible"
        );
        assert!(saved_sidecar.is_file());
        assert!(!segment_sidecar.exists());
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn unusable_reconstructed_fallback_does_not_block_healthy_current() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let encoded = encoded_test_segment(0xfa11, 0, 2, 1)?;
        let manifest_segment = manifest_segment(&encoded, 1);
        let setup_manifest_segment = manifest_segment.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            let pending = encoded
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            writer
                .publish_segment(&cx, pending)
                .await
                .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, vec![setup_manifest_segment]))
                .await
                .map_err(|error| error.to_string())?;
            let mut current = durable_test_manifest(3, Vec::new());
            current.docid_high_watermark = 2;
            writer
                .publish(&cx, &current)
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let previous_path = index.path().join("MANIFEST.prev");
        let mut corrupt_previous = std::fs::read(&previous_path)?;
        corrupt_previous[16] ^= 0xff;
        std::fs::write(&previous_path, &corrupt_previous)?;
        let segment_path = index
            .path()
            .join(canonical_segment_name(manifest_segment.segment_id));
        let mut corrupt_segment = std::fs::read(&segment_path)?;
        corrupt_segment[64] ^= 0xff;
        std::fs::write(&segment_path, corrupt_segment)?;
        let segment_sidecar = FileProtector::sidecar_path(&segment_path);
        let saved_sidecar = index.path().join("saved-fallback-segment.fec");
        std::fs::rename(&segment_sidecar, &saved_sidecar)?;

        let directory = index.path().to_path_buf();
        let repair_protector = protector.clone();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, repair_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            if writer.snapshot().loaded_manifest().manifest.generation != 3 {
                return Err("unusable optional fallback displaced healthy current".to_owned());
            }
            Ok(())
        });
        recovered.map_err(io::Error::other)?;

        assert_eq!(std::fs::read(&previous_path)?, corrupt_previous);
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            3
        );
        assert!(saved_sidecar.is_file());
        assert!(!segment_sidecar.exists());
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn stale_reconstructed_fallback_does_not_block_healthy_current() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, Vec::new()))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let previous_path = index.path().join("MANIFEST.prev");
        let stale = Manifest::empty(50, DEFAULT_SCHEMA.schema_id()?, 0).to_bytes()?;
        std::fs::write(&previous_path, &stale)?;
        protector.protect_file(&previous_path)?;
        let mut corrupt_stale = stale;
        corrupt_stale[16] ^= 0xff;
        std::fs::write(&previous_path, &corrupt_stale)?;

        let directory = index.path().to_path_buf();
        let repair_protector = protector.clone();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, repair_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            if writer.snapshot().loaded_manifest().manifest.generation != 2 {
                return Err("stale optional fallback displaced healthy current".to_owned());
            }
            Ok(())
        });
        recovered.map_err(io::Error::other)?;

        assert_eq!(std::fs::read(&previous_path)?, corrupt_stale);
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            2
        );
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn mismatched_dual_reconstructed_manifests_fail_closed() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, Vec::new()))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let current_path = index.path().join("MANIFEST");
        let previous_path = index.path().join("MANIFEST.prev");
        let current_sidecar = FileProtector::sidecar_path(&current_path);
        let saved_current_sidecar = index.path().join("saved-current-generation-two.fec");
        std::fs::rename(&current_sidecar, &saved_current_sidecar)?;

        let synthetic_current = index.path().join(".tmp-synthetic-current-five");
        let synthetic_bytes = durable_test_manifest(5, Vec::new()).to_bytes()?;
        std::fs::write(&synthetic_current, &synthetic_bytes)?;
        protector.protect_file_with_witness(
            &synthetic_current,
            FileSourceWitness::from_bytes(&synthetic_bytes),
        )?;
        std::fs::rename(
            FileProtector::sidecar_path(&synthetic_current),
            &current_sidecar,
        )?;

        let mut corrupt_current = std::fs::read(&current_path)?;
        corrupt_current[16] ^= 0xff;
        std::fs::write(&current_path, &corrupt_current)?;
        let mut corrupt_previous = std::fs::read(&previous_path)?;
        corrupt_previous[16] ^= 0xff;
        std::fs::write(&previous_path, &corrupt_previous)?;

        let directory = index.path().to_path_buf();
        let open_result: Result<KeeperWriter, KeeperError> =
            run_with_test_cx(move |cx| async move {
                KeeperWriter::open_durable(&cx, directory, DEFAULT_SCHEMA, protector).await
            });
        assert!(matches!(
            open_result,
            Err(KeeperError::InvalidGenerationPair {
                current: 5,
                previous: 1,
                ..
            })
        ));
        assert_eq!(std::fs::read(&current_path)?, corrupt_current);
        assert_eq!(std::fs::read(&previous_path)?, corrupt_previous);
        assert!(saved_current_sidecar.is_file());
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn malformed_current_sidecar_falls_back_to_usable_previous() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, Vec::new()))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let current_path = index.path().join("MANIFEST");
        let mut corrupt_current = std::fs::read(&current_path)?;
        corrupt_current[16] ^= 0xff;
        std::fs::write(&current_path, corrupt_current)?;
        std::fs::write(
            FileProtector::sidecar_path(&current_path),
            b"malformed current sidecar",
        )?;

        let directory = index.path().to_path_buf();
        let repair_protector = protector.clone();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, repair_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            if writer.snapshot().loaded_manifest().manifest.generation != 1 {
                return Err("malformed current sidecar blocked previous fallback".to_owned());
            }
            Ok(())
        });
        recovered.map_err(io::Error::other)?;
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            1
        );
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn malformed_previous_sidecar_does_not_block_usable_current() -> TestResult {
        let index = tempdir()?;
        let protector = test_file_protector();
        let directory = index.path().to_path_buf();
        let setup_protector = protector.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, Vec::new()))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let previous_path = index.path().join("MANIFEST.prev");
        let mut corrupt_previous = std::fs::read(&previous_path)?;
        corrupt_previous[16] ^= 0xff;
        std::fs::write(&previous_path, &corrupt_previous)?;
        std::fs::write(
            FileProtector::sidecar_path(&previous_path),
            b"malformed previous sidecar",
        )?;

        let directory = index.path().to_path_buf();
        let repair_protector = protector.clone();
        let recovered: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, repair_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            if writer.snapshot().loaded_manifest().manifest.generation != 2 {
                return Err("malformed optional fallback blocked healthy current".to_owned());
            }
            Ok(())
        });
        recovered.map_err(io::Error::other)?;

        assert_eq!(std::fs::read(&previous_path)?, corrupt_previous);
        assert_eq!(
            KeeperSnapshot::open(index.path(), DEFAULT_SCHEMA)?
                .loaded_manifest()
                .manifest
                .generation,
            2
        );
        Ok(())
    }

    #[cfg(all(unix, feature = "durability"))]
    #[test]
    fn durable_writer_stages_and_validates_manifest_and_segment_repairs() -> TestResult {
        let index = tempdir()?;
        let directory = index.path().to_path_buf();
        let protector = test_file_protector();
        let setup_protector = protector.clone();
        let expected = encoded_test_segment(0xabc, 0, 2, 1)?;
        let expected_segment = expected.as_bytes().to_vec();
        let segment_manifest = manifest_segment(&expected, 1);
        let setup_manifest = segment_manifest.clone();
        let setup: Result<(), String> = run_with_test_cx(move |cx| async move {
            let mut writer =
                KeeperWriter::create_durable(&cx, &directory, DEFAULT_SCHEMA, setup_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            let pending = expected
                .write_temp(&directory)
                .map_err(|error| error.to_string())?;
            writer
                .publish_segment(&cx, pending)
                .await
                .map_err(|error| error.to_string())?;
            writer
                .publish(&cx, &durable_test_manifest(2, vec![setup_manifest]))
                .await
                .map_err(|error| error.to_string())?;
            Ok(())
        });
        setup.map_err(io::Error::other)?;

        let manifest_path = index.path().join("MANIFEST");
        let expected_manifest = std::fs::read(&manifest_path)?;
        let segment_path = index
            .path()
            .join(canonical_segment_name(segment_manifest.segment_id));
        let mut corrupt_manifest = expected_manifest.clone();
        corrupt_manifest[16] ^= 0xff;
        std::fs::write(&manifest_path, corrupt_manifest)?;
        let mut corrupt_segment = expected_segment.clone();
        corrupt_segment[64] ^= 0xff;
        std::fs::write(&segment_path, corrupt_segment)?;

        let directory = index.path().to_path_buf();
        let repair_protector = protector.clone();
        let repaired: Result<(), String> = run_with_test_cx(move |cx| async move {
            let writer =
                KeeperWriter::open_durable(&cx, &directory, DEFAULT_SCHEMA, repair_protector)
                    .await
                    .map_err(|error| error.to_string())?;
            assert_eq!(writer.snapshot().loaded_manifest().manifest.generation, 2);
            Ok(())
        });
        repaired.map_err(io::Error::other)?;
        assert_eq!(std::fs::read(&manifest_path)?, expected_manifest);
        assert_eq!(std::fs::read(&segment_path)?, expected_segment);
        assert!(
            protector
                .verify_file(&manifest_path, &FileProtector::sidecar_path(&manifest_path))?
                .healthy
        );
        assert!(
            protector
                .verify_file(&segment_path, &FileProtector::sidecar_path(&segment_path))?
                .healthy
        );

        let authoritative_manifest = std::fs::read(&manifest_path)?;
        std::fs::write(
            FileProtector::sidecar_path(&manifest_path),
            b"malformed stale sidecar",
        )?;
        let directory = index.path().to_path_buf();
        let stale_protector = protector.clone();
        let regenerated: Result<(), String> = run_with_test_cx(move |cx| async move {
            KeeperWriter::open_durable(&cx, directory, DEFAULT_SCHEMA, stale_protector)
                .await
                .map(drop)
                .map_err(|error| error.to_string())
        });
        regenerated.map_err(io::Error::other)?;
        assert_eq!(std::fs::read(&manifest_path)?, authoritative_manifest);
        assert!(
            protector
                .verify_file(&manifest_path, &FileProtector::sidecar_path(&manifest_path))?
                .healthy
        );

        std::fs::write(
            FileProtector::sidecar_path(&manifest_path),
            b"second malformed stale sidecar",
        )?;
        let directory = index.path().to_path_buf();
        let repeated_protector = protector.clone();
        let repeated: Result<(), String> = run_with_test_cx(move |cx| async move {
            KeeperWriter::open_durable(&cx, directory, DEFAULT_SCHEMA, repeated_protector)
                .await
                .map(drop)
                .map_err(|error| error.to_string())
        });
        repeated.map_err(io::Error::other)?;
        assert_eq!(std::fs::read(&manifest_path)?, authoritative_manifest);
        assert_eq!(
            std::fs::read_dir(index.path())?
                .filter_map(Result::ok)
                .filter_map(|entry| entry.file_name().into_string().ok())
                .filter(|name| name.starts_with(".tmp-stale-fec-MANIFEST-"))
                .count(),
            2,
            "each stale sidecar must retain a distinct no-replace quarantine"
        );
        Ok(())
    }

    fn hex_bytes(source: &str) -> Vec<u8> {
        let compact = source
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        assert_eq!(compact.len() % 2, 0, "hex input must have even length");
        let (pairs, remainder) = compact.as_chunks::<2>();
        assert!(remainder.is_empty());
        pairs
            .iter()
            .map(|&[high, low]| {
                let high = hex_nibble(high);
                let low = hex_nibble(low);
                high << 4 | low
            })
            .collect()
    }

    fn hex_nibble(byte: u8) -> u8 {
        match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            b'A'..=b'F' => byte - b'A' + 10,
            _ => panic!("invalid hex byte"),
        }
    }
}
