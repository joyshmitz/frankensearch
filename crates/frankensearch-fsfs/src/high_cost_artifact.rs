//! High-cost artifact detection contract types for fsfs.
//!
//! This module defines the data structures for the fsfs high-cost artifact
//! detection contract v1, which specifies:
//! - Giant log file detection (size/churn/redundancy)
//! - Vendor/generated code detection
//! - Archive and transient build artifact detection
//! - Override policies for user force-include

use serde::{Deserialize, Serialize};

// ─── Kind Constants ──────────────────────────────────────────────────────────

pub const KIND_CONTRACT_DEFINITION: &str = "fsfs_high_cost_artifact_contract_definition";
pub const KIND_DECISION: &str = "fsfs_high_cost_artifact_decision";
pub const KIND_OVERRIDE_EVENT: &str = "fsfs_high_cost_override_event";
pub const CONTRACT_VERSION: u32 = 1;

// ─── Detector Names ──────────────────────────────────────────────────────────

pub const DETECTOR_GIANT_LOG: &str = "giant_log";
pub const DETECTOR_VENDOR_TREE: &str = "vendor_tree";
pub const DETECTOR_GENERATED_FILE: &str = "generated_file";
pub const DETECTOR_ARCHIVE_CONTAINER: &str = "archive_container";
pub const DETECTOR_TRANSIENT_BUILD: &str = "transient_build_artifact";

// ─── Reason Codes ────────────────────────────────────────────────────────────

pub const REASON_SIZE_EXCEEDED: &str = "FSFS_HIGH_COST_SIZE_EXCEEDED";
pub const REASON_CHURN_DETECTED: &str = "FSFS_HIGH_COST_CHURN_DETECTED";
pub const REASON_REDUNDANCY_HIGH: &str = "FSFS_HIGH_COST_REDUNDANCY_HIGH";
pub const REASON_VENDOR_PATH: &str = "FSFS_HIGH_COST_VENDOR_PATH";
pub const REASON_GENERATED_MARKER: &str = "FSFS_HIGH_COST_GENERATED_MARKER";
pub const REASON_ARCHIVE_EXT: &str = "FSFS_HIGH_COST_ARCHIVE_EXT";
pub const REASON_TRANSIENT_DIR: &str = "FSFS_HIGH_COST_TRANSIENT_DIR";
pub const REASON_BUILD_ARTIFACT: &str = "FSFS_HIGH_COST_BUILD_ARTIFACT";
pub const REASON_OVERRIDE_APPLIED: &str = "FSFS_HIGH_COST_OVERRIDE_APPLIED";
pub const REASON_MANUAL_REVIEW: &str = "FSFS_HIGH_COST_MANUAL_REVIEW";

// ─── Detector Structs ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GiantLogDetector {
    pub max_size_mb: u32,
    pub churn_window_minutes: u32,
    pub redundancy_ratio_threshold: f64,
}

impl Default for GiantLogDetector {
    fn default() -> Self {
        Self {
            max_size_mb: 100,
            churn_window_minutes: 60,
            redundancy_ratio_threshold: 0.8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VendorGeneratedDetector {
    pub vendor_path_patterns: Vec<String>,
    pub generated_markers: Vec<String>,
    pub library_tree_depth_threshold: u32,
}

impl Default for VendorGeneratedDetector {
    fn default() -> Self {
        Self {
            vendor_path_patterns: vec![
                "vendor/".to_owned(),
                "node_modules/".to_owned(),
                "third_party/".to_owned(),
                ".cargo/registry/".to_owned(),
            ],
            generated_markers: vec![
                "// Code generated".to_owned(),
                "// DO NOT EDIT".to_owned(),
                "# Auto-generated".to_owned(),
            ],
            library_tree_depth_threshold: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArchiveTransientDetector {
    pub archive_extensions: Vec<String>,
    pub transient_directories: Vec<String>,
    pub build_artifact_patterns: Vec<String>,
}

impl Default for ArchiveTransientDetector {
    fn default() -> Self {
        Self {
            archive_extensions: vec![
                ".zip".to_owned(),
                ".tar".to_owned(),
                ".tar.gz".to_owned(),
                ".tgz".to_owned(),
                ".rar".to_owned(),
                ".7z".to_owned(),
            ],
            transient_directories: vec![
                "target/".to_owned(),
                "build/".to_owned(),
                "dist/".to_owned(),
                ".cache/".to_owned(),
                "__pycache__/".to_owned(),
            ],
            build_artifact_patterns: vec![
                "*.o".to_owned(),
                "*.a".to_owned(),
                "*.so".to_owned(),
                "*.dylib".to_owned(),
                "*.dll".to_owned(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OverridePolicy {
    pub allow_user_force_include: bool,
    pub requires_reason: bool,
    pub max_override_ttl_seconds: u32,
}

impl Default for OverridePolicy {
    fn default() -> Self {
        Self {
            allow_user_force_include: true,
            requires_reason: true,
            max_override_ttl_seconds: 86400, // 24 hours
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DefaultAction {
    IndexMetadataOnly,
    Skip,
    IndexFull,
}

impl Default for DefaultAction {
    fn default() -> Self {
        Self::IndexMetadataOnly
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DefaultActions {
    pub giant_log: DefaultAction,
    pub vendor_tree: DefaultAction,
    pub generated_file: DefaultAction,
    pub archive_container: DefaultAction,
    pub transient_build_artifact: DefaultAction,
}

impl Default for DefaultActions {
    fn default() -> Self {
        Self {
            giant_log: DefaultAction::IndexMetadataOnly,
            vendor_tree: DefaultAction::Skip,
            generated_file: DefaultAction::IndexMetadataOnly,
            archive_container: DefaultAction::Skip,
            transient_build_artifact: DefaultAction::Skip,
        }
    }
}

// ─── Contract Definition ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HighCostArtifactContractDefinition {
    pub kind: String,
    pub v: u32,
    pub giant_log_detector: GiantLogDetector,
    pub vendor_generated_detector: VendorGeneratedDetector,
    pub archive_transient_detector: ArchiveTransientDetector,
    pub override_policy: OverridePolicy,
    pub default_actions: DefaultActions,
}

impl Default for HighCostArtifactContractDefinition {
    fn default() -> Self {
        Self {
            kind: KIND_CONTRACT_DEFINITION.to_owned(),
            v: CONTRACT_VERSION,
            giant_log_detector: GiantLogDetector::default(),
            vendor_generated_detector: VendorGeneratedDetector::default(),
            archive_transient_detector: ArchiveTransientDetector::default(),
            override_policy: OverridePolicy::default(),
            default_actions: DefaultActions::default(),
        }
    }
}

// ─── Evidence ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Evidence {
    pub size_mb: u32,
    pub churn_rate_per_hour: u32,
    pub redundancy_ratio: f64,
    pub path_depth: u32,
    pub extension: String,
}

// ─── Decision ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HighCostArtifactDecision {
    pub kind: String,
    pub v: u32,
    pub path: String,
    pub detectors_fired: Vec<String>,
    pub evidence: Evidence,
    pub final_action: DefaultAction,
    pub reason_code: String,
    pub cost_score: f64,
    pub override_applied: bool,
    pub needs_manual_review: bool,
}

impl HighCostArtifactDecision {
    /// Create a new decision with default kind and version.
    #[must_use]
    pub fn new(
        path: String,
        detectors_fired: Vec<String>,
        evidence: Evidence,
        final_action: DefaultAction,
        reason_code: String,
        cost_score: f64,
    ) -> Self {
        Self {
            kind: KIND_DECISION.to_owned(),
            v: CONTRACT_VERSION,
            path,
            detectors_fired,
            evidence,
            final_action,
            reason_code,
            cost_score,
            override_applied: false,
            needs_manual_review: false,
        }
    }

    /// Returns true if any detector fired.
    #[must_use]
    pub fn has_detections(&self) -> bool {
        !self.detectors_fired.is_empty()
    }

    /// Returns true if the decision results in skipping the file.
    #[must_use]
    pub fn is_skipped(&self) -> bool {
        self.final_action == DefaultAction::Skip
    }

    /// Returns true if only metadata should be indexed.
    #[must_use]
    pub fn is_metadata_only(&self) -> bool {
        self.final_action == DefaultAction::IndexMetadataOnly
    }
}

// ─── Override Event ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HighCostOverrideEvent {
    pub kind: String,
    pub v: u32,
    pub path: String,
    pub requested_action: DefaultAction,
    pub approved: bool,
    pub expires_at: String,
    pub reason: String,
    pub reason_code: String,
}

impl HighCostOverrideEvent {
    /// Create a new override event with default kind and version.
    #[must_use]
    pub fn new(
        path: String,
        requested_action: DefaultAction,
        approved: bool,
        expires_at: String,
        reason: String,
        reason_code: String,
    ) -> Self {
        Self {
            kind: KIND_OVERRIDE_EVENT.to_owned(),
            v: CONTRACT_VERSION,
            path,
            requested_action,
            approved,
            expires_at,
            reason,
            reason_code,
        }
    }

    /// Returns true if the override was approved.
    #[must_use]
    pub fn is_approved(&self) -> bool {
        self.approved
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_contract_has_correct_kind_and_version() {
        let contract = HighCostArtifactContractDefinition::default();
        assert_eq!(contract.kind, KIND_CONTRACT_DEFINITION);
        assert_eq!(contract.v, CONTRACT_VERSION);
    }

    #[test]
    fn default_giant_log_detector_values() {
        let detector = GiantLogDetector::default();
        assert_eq!(detector.max_size_mb, 100);
        assert_eq!(detector.churn_window_minutes, 60);
        assert!((detector.redundancy_ratio_threshold - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn default_vendor_generated_detector_has_patterns() {
        let detector = VendorGeneratedDetector::default();
        assert!(!detector.vendor_path_patterns.is_empty());
        assert!(!detector.generated_markers.is_empty());
        assert_eq!(detector.library_tree_depth_threshold, 5);
    }

    #[test]
    fn default_archive_transient_detector_has_extensions() {
        let detector = ArchiveTransientDetector::default();
        assert!(detector.archive_extensions.contains(&".zip".to_owned()));
        assert!(
            detector
                .transient_directories
                .contains(&"target/".to_owned())
        );
    }

    #[test]
    fn default_override_policy_allows_force_include() {
        let policy = OverridePolicy::default();
        assert!(policy.allow_user_force_include);
        assert!(policy.requires_reason);
        assert_eq!(policy.max_override_ttl_seconds, 86400);
    }

    #[test]
    fn default_actions_skip_vendor_and_archives() {
        let actions = DefaultActions::default();
        assert_eq!(actions.vendor_tree, DefaultAction::Skip);
        assert_eq!(actions.archive_container, DefaultAction::Skip);
        assert_eq!(actions.transient_build_artifact, DefaultAction::Skip);
        assert_eq!(actions.giant_log, DefaultAction::IndexMetadataOnly);
    }

    #[test]
    fn decision_new_sets_kind_and_version() {
        let evidence = Evidence {
            size_mb: 150,
            churn_rate_per_hour: 10,
            redundancy_ratio: 0.9,
            path_depth: 2,
            extension: ".log".to_owned(),
        };
        let decision = HighCostArtifactDecision::new(
            "/var/log/app.log".to_owned(),
            vec![DETECTOR_GIANT_LOG.to_owned()],
            evidence,
            DefaultAction::IndexMetadataOnly,
            REASON_SIZE_EXCEEDED.to_owned(),
            0.85,
        );

        assert_eq!(decision.kind, KIND_DECISION);
        assert_eq!(decision.v, CONTRACT_VERSION);
        assert!(decision.has_detections());
        assert!(decision.is_metadata_only());
        assert!(!decision.is_skipped());
    }

    #[test]
    fn decision_skipped_detection() {
        let evidence = Evidence {
            size_mb: 0,
            churn_rate_per_hour: 0,
            redundancy_ratio: 0.0,
            path_depth: 6,
            extension: ".js".to_owned(),
        };
        let decision = HighCostArtifactDecision::new(
            "node_modules/lodash/index.js".to_owned(),
            vec![DETECTOR_VENDOR_TREE.to_owned()],
            evidence,
            DefaultAction::Skip,
            REASON_VENDOR_PATH.to_owned(),
            1.0,
        );

        assert!(decision.is_skipped());
        assert!(!decision.is_metadata_only());
    }

    #[test]
    fn override_event_new_sets_kind_and_version() {
        let event = HighCostOverrideEvent::new(
            "/important/large_file.log".to_owned(),
            DefaultAction::IndexFull,
            true,
            "2026-04-16T00:00:00Z".to_owned(),
            "Required for compliance audit".to_owned(),
            REASON_OVERRIDE_APPLIED.to_owned(),
        );

        assert_eq!(event.kind, KIND_OVERRIDE_EVENT);
        assert_eq!(event.v, CONTRACT_VERSION);
        assert!(event.is_approved());
    }

    #[test]
    fn contract_roundtrip_serialization() {
        let contract = HighCostArtifactContractDefinition::default();
        let json = serde_json::to_string(&contract).unwrap();
        let parsed: HighCostArtifactContractDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(contract, parsed);
    }

    #[test]
    fn decision_roundtrip_serialization() {
        let evidence = Evidence {
            size_mb: 50,
            churn_rate_per_hour: 5,
            redundancy_ratio: 0.5,
            path_depth: 3,
            extension: ".tmp".to_owned(),
        };
        let decision = HighCostArtifactDecision::new(
            "/tmp/build/cache.tmp".to_owned(),
            vec![DETECTOR_TRANSIENT_BUILD.to_owned()],
            evidence,
            DefaultAction::Skip,
            REASON_TRANSIENT_DIR.to_owned(),
            0.75,
        );
        let json = serde_json::to_string(&decision).unwrap();
        let parsed: HighCostArtifactDecision = serde_json::from_str(&json).unwrap();
        assert_eq!(decision, parsed);
    }

    #[test]
    fn override_event_roundtrip_serialization() {
        let event = HighCostOverrideEvent::new(
            "/data/archive.zip".to_owned(),
            DefaultAction::IndexFull,
            false,
            "".to_owned(),
            "Rejected - no justification".to_owned(),
            "FSFS_OVERRIDE_REJECTED".to_owned(),
        );
        let json = serde_json::to_string(&event).unwrap();
        let parsed: HighCostOverrideEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, parsed);
    }
}
