//! Incremental change detection contract types for fsfs.
//!
//! This module defines the data structures for the fsfs incremental change
//! detection contract v1, which specifies:
//! - mtime/size/hash tradeoff policy
//! - rename/move detection semantics
//! - crash/restart recovery behavior
//! - stale-state reconciliation guarantees

use serde::{Deserialize, Serialize};

// ─── Kind Constants ──────────────────────────────────────────────────────────

pub const KIND_CONTRACT_DEFINITION: &str = "fsfs_incremental_change_detection_contract_definition";
pub const KIND_CHANGE_DECISION: &str = "fsfs_incremental_change_decision";
pub const KIND_RECOVERY_CHECKPOINT: &str = "fsfs_incremental_recovery_checkpoint";
pub const CONTRACT_VERSION: u32 = 1;

// ─── Event Types ─────────────────────────────────────────────────────────────

pub const EVENT_TYPE_CREATE: &str = "create";
pub const EVENT_TYPE_MODIFY: &str = "modify";
pub const EVENT_TYPE_DELETE: &str = "delete";
pub const EVENT_TYPE_RENAME: &str = "rename";

// ─── Detection Modes ─────────────────────────────────────────────────────────

pub const DETECTION_MODE_FASTPATH: &str = "fastpath";
pub const DETECTION_MODE_HASH_CONFIRM: &str = "hash_confirm";
pub const DETECTION_MODE_FULL_RECONCILE: &str = "full_reconcile";

// ─── Queue Actions ───────────────────────────────────────────────────────────

pub const QUEUE_ACTION_ENQUEUE_EMBED: &str = "enqueue_embed";
pub const QUEUE_ACTION_SKIP_NO_CHANGE: &str = "skip_no_change";
pub const QUEUE_ACTION_MARK_STALE: &str = "mark_stale";
pub const QUEUE_ACTION_RECONCILE_FULL: &str = "reconcile_full";
pub const QUEUE_ACTION_DROP_MISSING: &str = "drop_missing";

// ─── Policy Structs ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FastpathPolicy {
    pub mtime_granularity_ns: u64,
    pub require_size_change: bool,
    pub hash_on_mtime_only: bool,
    pub max_fastpath_skips: u32,
}

impl Default for FastpathPolicy {
    fn default() -> Self {
        Self {
            mtime_granularity_ns: 1_000_000, // 1ms
            require_size_change: false,
            hash_on_mtime_only: true,
            max_fastpath_skips: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HashPolicy {
    pub algorithm: String,
    pub sample_prefix_bytes: u32,
    pub full_hash_threshold_bytes: u32,
}

impl Default for HashPolicy {
    fn default() -> Self {
        Self {
            algorithm: "sha256".to_owned(),
            sample_prefix_bytes: 4096,
            full_hash_threshold_bytes: 1_048_576, // 1MB
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RenameMovePolicy {
    pub identity_keys: Vec<String>,
    pub same_device_rename_preserves_identity: bool,
    pub cross_device_move: String,
}

impl Default for RenameMovePolicy {
    fn default() -> Self {
        Self {
            identity_keys: vec!["inode".to_owned(), "content_hash".to_owned()],
            same_device_rename_preserves_identity: true,
            cross_device_move: "hash_confirm".to_owned(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecoveryPolicy {
    pub journal_required: bool,
    pub replay_order: String,
    pub pending_ttl_seconds: u32,
}

impl Default for RecoveryPolicy {
    fn default() -> Self {
        Self {
            journal_required: true,
            replay_order: "sequence_asc".to_owned(),
            pending_ttl_seconds: 3600,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReconciliationPolicy {
    pub full_scan_interval_seconds: u32,
    pub stale_after_seconds: u32,
    pub orphan_entry_action: String,
}

impl Default for ReconciliationPolicy {
    fn default() -> Self {
        Self {
            full_scan_interval_seconds: 86400, // 24 hours
            stale_after_seconds: 3600,
            orphan_entry_action: "mark_stale".to_owned(),
        }
    }
}

// ─── Contract Definition ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IncrementalChangeDetectionContractDefinition {
    pub kind: String,
    pub v: u32,
    pub fastpath_policy: FastpathPolicy,
    pub hash_policy: HashPolicy,
    pub rename_move_policy: RenameMovePolicy,
    pub recovery_policy: RecoveryPolicy,
    pub reconciliation_policy: ReconciliationPolicy,
}

impl Default for IncrementalChangeDetectionContractDefinition {
    fn default() -> Self {
        Self {
            kind: KIND_CONTRACT_DEFINITION.to_owned(),
            v: CONTRACT_VERSION,
            fastpath_policy: FastpathPolicy::default(),
            hash_policy: HashPolicy::default(),
            rename_move_policy: RenameMovePolicy::default(),
            recovery_policy: RecoveryPolicy::default(),
            reconciliation_policy: ReconciliationPolicy::default(),
        }
    }
}

// ─── File State ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileState {
    pub file_id: String,
    pub size_bytes: u64,
    pub mtime_ns: u64,
    pub content_hash: String,
}

impl FileState {
    /// Check if mtime changed beyond the granularity threshold.
    #[must_use]
    pub fn mtime_changed(&self, other: &Self, granularity_ns: u64) -> bool {
        let diff = self.mtime_ns.abs_diff(other.mtime_ns);
        diff >= granularity_ns
    }

    /// Check if size changed.
    #[must_use]
    pub fn size_changed(&self, other: &Self) -> bool {
        self.size_bytes != other.size_bytes
    }

    /// Check if content hash differs.
    #[must_use]
    pub fn content_changed(&self, other: &Self) -> bool {
        self.content_hash != other.content_hash
    }
}

// ─── Change Decision ─────────────────────────────────────────────────────────

/// Decision artifact emitted after evaluating a file change event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IncrementalChangeDecision {
    pub kind: String,
    pub v: u32,
    pub path: String,
    pub event_type: String,
    pub detection_mode: String,
    pub previous_state: FileState,
    pub current_state: FileState,
    pub queue_action: String,
    pub reason_code: String,
    pub confidence: f64,
}

impl IncrementalChangeDecision {
    /// Create a new decision with default kind and version.
    #[must_use]
    pub fn new(
        path: String,
        event_type: String,
        detection_mode: String,
        previous_state: FileState,
        current_state: FileState,
        queue_action: String,
        reason_code: String,
        confidence: f64,
    ) -> Self {
        Self {
            kind: KIND_CHANGE_DECISION.to_owned(),
            v: CONTRACT_VERSION,
            path,
            event_type,
            detection_mode,
            previous_state,
            current_state,
            queue_action,
            reason_code,
            confidence,
        }
    }

    /// Returns true if this decision requires embedding work.
    #[must_use]
    pub fn requires_embedding(&self) -> bool {
        self.queue_action == QUEUE_ACTION_ENQUEUE_EMBED
    }
}

// ─── Recovery Checkpoint ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IncrementalRecoveryCheckpoint {
    pub kind: String,
    pub v: u32,
    pub checkpoint_id: String,
    pub last_applied_seq: u64,
    pub pending_changes: u32,
    pub journal_clean: bool,
    pub stale_entries: u32,
    pub action_on_restart: String,
    pub reason_code: String,
}

impl IncrementalRecoveryCheckpoint {
    /// Create a new checkpoint with default kind and version.
    #[must_use]
    pub fn new(
        checkpoint_id: String,
        last_applied_seq: u64,
        pending_changes: u32,
        journal_clean: bool,
        stale_entries: u32,
        action_on_restart: String,
        reason_code: String,
    ) -> Self {
        Self {
            kind: KIND_RECOVERY_CHECKPOINT.to_owned(),
            v: CONTRACT_VERSION,
            checkpoint_id,
            last_applied_seq,
            pending_changes,
            journal_clean,
            stale_entries,
            action_on_restart,
            reason_code,
        }
    }

    /// Returns true if journal replay is needed on restart.
    #[must_use]
    pub fn needs_replay(&self) -> bool {
        !self.journal_clean || self.pending_changes > 0
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_contract_has_correct_kind_and_version() {
        let contract = IncrementalChangeDetectionContractDefinition::default();
        assert_eq!(contract.kind, KIND_CONTRACT_DEFINITION);
        assert_eq!(contract.v, CONTRACT_VERSION);
    }

    #[test]
    fn default_fastpath_policy_values() {
        let policy = FastpathPolicy::default();
        assert_eq!(policy.mtime_granularity_ns, 1_000_000);
        assert!(policy.hash_on_mtime_only);
        assert_eq!(policy.max_fastpath_skips, 10);
    }

    #[test]
    fn default_hash_policy_uses_sha256() {
        let policy = HashPolicy::default();
        assert_eq!(policy.algorithm, "sha256");
    }

    #[test]
    fn file_state_mtime_changed_respects_granularity() {
        let state1 = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "abc".to_owned(),
        };
        let state2 = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_500_000, // 500μs difference
            content_hash: "abc".to_owned(),
        };

        // 1ms granularity: 500μs difference is below threshold
        assert!(!state1.mtime_changed(&state2, 1_000_000));

        // 100μs granularity: 500μs difference exceeds threshold
        assert!(state1.mtime_changed(&state2, 100_000));
    }

    #[test]
    fn file_state_size_changed() {
        let state1 = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "abc".to_owned(),
        };
        let state2 = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 200,
            mtime_ns: 1_000_000_000,
            content_hash: "abc".to_owned(),
        };

        assert!(state1.size_changed(&state2));
        assert!(!state1.size_changed(&state1));
    }

    #[test]
    fn file_state_content_changed() {
        let state1 = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "abc".to_owned(),
        };
        let state2 = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "def".to_owned(),
        };

        assert!(state1.content_changed(&state2));
        assert!(!state1.content_changed(&state1));
    }

    #[test]
    fn change_decision_requires_embedding() {
        let state = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "abc".to_owned(),
        };

        let decision = IncrementalChangeDecision::new(
            "/path/to/file".to_owned(),
            EVENT_TYPE_MODIFY.to_owned(),
            DETECTION_MODE_HASH_CONFIRM.to_owned(),
            state.clone(),
            state,
            QUEUE_ACTION_ENQUEUE_EMBED.to_owned(),
            "FSFS_CONTENT_CHANGED".to_owned(),
            0.95,
        );

        assert!(decision.requires_embedding());
        assert_eq!(decision.kind, KIND_CHANGE_DECISION);
        assert_eq!(decision.v, CONTRACT_VERSION);
    }

    #[test]
    fn recovery_checkpoint_needs_replay() {
        let dirty = IncrementalRecoveryCheckpoint::new(
            "cp-1".to_owned(),
            100,
            5,
            false,
            0,
            "replay_pending".to_owned(),
            "FSFS_PENDING_CHANGES".to_owned(),
        );
        assert!(dirty.needs_replay());

        let clean = IncrementalRecoveryCheckpoint::new(
            "cp-2".to_owned(),
            100,
            0,
            true,
            0,
            "none".to_owned(),
            "FSFS_CLEAN".to_owned(),
        );
        assert!(!clean.needs_replay());
    }

    #[test]
    fn contract_roundtrip_serialization() {
        let contract = IncrementalChangeDetectionContractDefinition::default();
        let json = serde_json::to_string(&contract).unwrap();
        let parsed: IncrementalChangeDetectionContractDefinition =
            serde_json::from_str(&json).unwrap();
        assert_eq!(contract, parsed);
    }

    #[test]
    fn decision_roundtrip_serialization() {
        let state = FileState {
            file_id: "f1".to_owned(),
            size_bytes: 100,
            mtime_ns: 1_000_000_000,
            content_hash: "abc123def456abc123def456abc123def456abc123def456abc123def456abcd".to_owned(),
        };
        let decision = IncrementalChangeDecision::new(
            "/path".to_owned(),
            EVENT_TYPE_MODIFY.to_owned(),
            DETECTION_MODE_FASTPATH.to_owned(),
            state.clone(),
            state,
            QUEUE_ACTION_SKIP_NO_CHANGE.to_owned(),
            "FSFS_NO_CHANGE".to_owned(),
            1.0,
        );
        let json = serde_json::to_string(&decision).unwrap();
        let parsed: IncrementalChangeDecision = serde_json::from_str(&json).unwrap();
        assert_eq!(decision, parsed);
    }

    #[test]
    fn checkpoint_roundtrip_serialization() {
        let checkpoint = IncrementalRecoveryCheckpoint::new(
            "cp-test".to_owned(),
            42,
            0,
            true,
            0,
            "none".to_owned(),
            "FSFS_CLEAN".to_owned(),
        );
        let json = serde_json::to_string(&checkpoint).unwrap();
        let parsed: IncrementalRecoveryCheckpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(checkpoint, parsed);
    }
}
