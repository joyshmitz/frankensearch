//! Quill error taxonomy and bridge to the workspace-wide search error.

use std::path::{Path, PathBuf};

use asupersync::sync::LockError;
use frankensearch_core::SearchError;
use thiserror::Error;

/// Errors carrying Quill-specific diagnostic context.
#[derive(Debug, Error)]
pub enum QuillError {
    /// A compile-time schema descriptor is internally inconsistent.
    #[error("invalid schema descriptor: {detail}")]
    InvalidDescriptor {
        /// Validation failure.
        detail: String,
    },
    /// An FSLX header references a schema unknown to this build.
    #[error("unknown Quill schema id {schema_id:#018x}")]
    UnknownSchema {
        /// Durable schema identifier from the file header.
        schema_id: u64,
    },
    /// Neither the primary nor previous manifest exists.
    #[error("Quill index not found at {path}")]
    IndexNotFound {
        /// Expected index directory.
        path: PathBuf,
    },
    /// FSLX or manifest validation failed.
    #[error("Quill index corrupted at {path}: {detail}")]
    IndexCorrupted {
        /// Corrupted artifact path.
        path: PathBuf,
        /// Failed invariant or checksum.
        detail: String,
    },
    /// A cancel-aware operation stopped before completion.
    #[error("Quill operation cancelled during {phase}: {reason}")]
    Cancelled {
        /// Operation phase.
        phase: String,
        /// Cancellation or deadline detail.
        reason: String,
    },
    /// An internal invariant failed without identifying corrupt durable bytes.
    #[error("Quill invariant failed: {detail}")]
    Invariant {
        /// Failed invariant.
        detail: String,
    },
    /// A configured ceiling or allocation request could not be satisfied.
    #[error("Quill resource failure for {resource}: {detail}")]
    Resource {
        /// Bounded resource or allocation category.
        resource: &'static str,
        /// Requested size, configured ceiling, or allocation diagnostic.
        detail: String,
    },
    /// Underlying filesystem I/O failed.
    #[error("Quill I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<QuillError> for SearchError {
    fn from(error: QuillError) -> Self {
        match error {
            QuillError::IndexNotFound { path } => Self::IndexNotFound { path },
            QuillError::IndexCorrupted { path, detail } => Self::IndexCorrupted { path, detail },
            QuillError::Cancelled { phase, reason } => Self::Cancelled { phase, reason },
            other => Self::SubsystemError {
                subsystem: "quill",
                source: Box::new(other),
            },
        }
    }
}

/// Construct the standard missing-index result used by `QuillIndex::open`.
#[must_use]
pub fn index_not_found(path: impl AsRef<Path>) -> SearchError {
    QuillError::IndexNotFound {
        path: path.as_ref().to_path_buf(),
    }
    .into()
}

/// Map an asupersync cancel-aware lock failure into the public error taxonomy.
#[must_use]
pub fn map_lock_error(phase: &str, error: LockError) -> SearchError {
    match error {
        LockError::Cancelled => QuillError::Cancelled {
            phase: phase.to_owned(),
            reason: "writer lock cancelled".to_owned(),
        }
        .into(),
        LockError::TimedOut(deadline) => QuillError::Cancelled {
            phase: phase.to_owned(),
            reason: format!("writer lock timed out at {deadline:?}"),
        }
        .into(),
        LockError::Poisoned => QuillError::Invariant {
            detail: format!("writer mutex poisoned during {phase}"),
        }
        .into(),
        LockError::PolledAfterCompletion => QuillError::Invariant {
            detail: format!("writer mutex future reused after completion during {phase}"),
        }
        .into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn general_errors_map_to_quill_subsystem() {
        let error: SearchError = QuillError::UnknownSchema { schema_id: 7 }.into();
        assert!(matches!(
            error,
            SearchError::SubsystemError {
                subsystem: "quill",
                ..
            }
        ));
    }

    #[test]
    fn missing_index_preserves_path() {
        let path = PathBuf::from("missing/quill");
        assert!(matches!(
            index_not_found(&path),
            SearchError::IndexNotFound { path: actual } if actual == path
        ));
    }

    #[test]
    fn cancelled_lock_preserves_phase() {
        assert!(matches!(
            map_lock_error("keeper.commit", LockError::Cancelled),
            SearchError::Cancelled { phase, reason }
                if phase == "keeper.commit" && reason.contains("lock cancelled")
        ));
    }

    #[test]
    fn poisoned_lock_is_a_quill_subsystem_error() {
        assert!(matches!(
            map_lock_error("scribe.ingest", LockError::Poisoned),
            SearchError::SubsystemError {
                subsystem: "quill",
                ..
            }
        ));
    }

    #[test]
    fn corruption_preserves_the_public_typed_variant() {
        let error: SearchError = QuillError::IndexCorrupted {
            path: PathBuf::from("index/seg-1.fslx"),
            detail: "section checksum mismatch".to_owned(),
        }
        .into();
        assert!(matches!(
            error,
            SearchError::IndexCorrupted { path, detail }
                if path == PathBuf::from("index/seg-1.fslx")
                    && detail == "section checksum mismatch"
        ));
    }
}
