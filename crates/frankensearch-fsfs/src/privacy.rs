use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const CORPUS_PRIVACY_PREFLIGHT_SCHEMA_VERSION: u32 = 1;
pub const CORPUS_PRIVACY_PREFLIGHT_CONTRACT_KIND: &str = "fsfs_corpus_privacy_preflight_contract";
pub const CORPUS_PRIVACY_PREFLIGHT_REPORT_KIND: &str = "fsfs_corpus_privacy_preflight_report";
pub const CORPUS_PRIVACY_PREFLIGHT_REDACTION_PROFILE: &str = "privacy-preflight-v1-default";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScopeDefaults {
    pub default_roots: Vec<String>,
    pub requires_explicit_opt_in_outside_defaults: bool,
    pub opt_out_globs_supported: bool,
    pub precedence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PathPolicies {
    pub deny_always_globs: Vec<String>,
    pub allow_with_opt_in_globs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RedactionConfig {
    pub logs_default_action: String,
    pub explain_default_action: String,
    pub replay_default_action: String,
    pub deterministic_profile_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThreatModel {
    pub local_multi_user_assumed: bool,
    pub same_host_read_risk: bool,
    pub mitigations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetryEmissionRules {
    pub raw_content_allowed: bool,
    pub reason_code_required: bool,
    pub redaction_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScopePrivacyContractDefinition {
    pub kind: String, // "fsfs_scope_privacy_contract_definition"
    pub v: u32,       // 1
    pub scope_defaults: ScopeDefaults,
    pub sensitive_classes: Vec<String>,
    pub path_policies: PathPolicies,
    pub redaction: RedactionConfig,
    pub threat_model: ThreatModel,
    pub telemetry_emission_rules: TelemetryEmissionRules,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RedactedArtifact {
    pub kind: String, // "fsfs_scope_redacted_artifact"
    pub v: u32,       // 1
    pub artifact_type: String,
    pub path: String,
    pub reason_code: String,
    pub redaction_applied: bool,
    pub raw_content_present: bool,
    pub redaction_profile: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScopeScanDecision {
    pub kind: String, // "fsfs_scope_scan_decision"
    pub v: u32,       // 1
    pub path: String,
    pub decision: String,
    pub reason_code: String,
    pub sensitive_classes: Vec<String>,
    pub persist_allowed: bool,
    pub emit_allowed: bool,
    pub display_allowed: bool,
    pub redaction_profile: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum CorpusPreflightSignal {
    CredentialToken,
    PrivateKey,
    GeneratedArtifact,
    OversizedBinary,
    SensitivePath,
    PersonalData,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CorpusPreflightDecision {
    Include,
    Skip,
    Defer,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CorpusPreflightRedactionAction {
    DropContent,
    HashPath,
    MaskExcerpt,
    MetadataOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusPreflightRule {
    pub rule_id: String,
    pub signal: CorpusPreflightSignal,
    pub default_decision: CorpusPreflightDecision,
    pub reason_code: String,
    pub redaction_action: CorpusPreflightRedactionAction,
    pub override_allowed: bool,
    pub false_positive_suppressions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusPrivacyPreflightContractDefinition {
    pub kind: String,
    pub v: u32,
    pub dry_run_only: bool,
    pub destructive_cleanup_allowed: bool,
    pub redaction_profile: String,
    pub rule_matrix: Vec<CorpusPreflightRule>,
    pub required_report_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusPreflightEvidence {
    pub evidence_id: String,
    pub sample_hash: String,
    pub redacted_excerpt: String,
    pub raw_content_present: bool,
    pub redaction_applied: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusPreflightOverride {
    pub requested_decision: CorpusPreflightDecision,
    pub approved: bool,
    pub reason: String,
    pub reason_code: String,
    pub expires_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusPreflightFileDecision {
    pub path: String,
    pub decision: CorpusPreflightDecision,
    pub reason_code: String,
    pub signals: Vec<CorpusPreflightSignal>,
    pub sensitive_classes: Vec<String>,
    pub redaction_action: CorpusPreflightRedactionAction,
    pub evidence: CorpusPreflightEvidence,
    pub semantic_index_allowed: bool,
    pub lexical_index_allowed: bool,
    pub evidence_emit_allowed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub false_positive_suppression_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub override_event: Option<CorpusPreflightOverride>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusPreflightSummary {
    pub included: u32,
    pub skipped: u32,
    pub deferred: u32,
    pub overrides_applied: u32,
    pub false_positive_suppressions: u32,
    pub raw_content_present: bool,
    pub destructive_cleanup_allowed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusPrivacyPreflightReport {
    pub kind: String,
    pub v: u32,
    pub run_id: String,
    pub generated_at: String,
    pub dry_run: bool,
    pub root: String,
    pub config_hash: String,
    pub redaction_profile: String,
    pub replay_command: String,
    pub destructive_cleanup_allowed: bool,
    pub decisions: Vec<CorpusPreflightFileDecision>,
    pub summary: CorpusPreflightSummary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusPrivacyPreflightViolation {
    pub field: String,
    pub reason: String,
}

impl CorpusPrivacyPreflightViolation {
    fn new(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for CorpusPrivacyPreflightViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.reason)
    }
}

impl std::error::Error for CorpusPrivacyPreflightViolation {}

/// Builds the canonical v1 corpus privacy preflight contract definition.
#[must_use]
pub fn corpus_privacy_preflight_contract_definition() -> CorpusPrivacyPreflightContractDefinition {
    use CorpusPreflightDecision::{Defer, Include, Skip};
    use CorpusPreflightRedactionAction::{DropContent, HashPath, MaskExcerpt, MetadataOnly};
    use CorpusPreflightSignal::{
        CredentialToken, GeneratedArtifact, OversizedBinary, PersonalData, PrivateKey,
        SensitivePath,
    };

    CorpusPrivacyPreflightContractDefinition {
        kind: CORPUS_PRIVACY_PREFLIGHT_CONTRACT_KIND.to_owned(),
        v: CORPUS_PRIVACY_PREFLIGHT_SCHEMA_VERSION,
        dry_run_only: true,
        destructive_cleanup_allowed: false,
        redaction_profile: CORPUS_PRIVACY_PREFLIGHT_REDACTION_PROFILE.to_owned(),
        rule_matrix: vec![
            CorpusPreflightRule {
                rule_id: "credential-token".to_owned(),
                signal: CredentialToken,
                default_decision: Skip,
                reason_code: "privacy.secret.token".to_owned(),
                redaction_action: MaskExcerpt,
                override_allowed: true,
                false_positive_suppressions: vec![
                    "documented-placeholder".to_owned(),
                    "test-fixture-token".to_owned(),
                ],
            },
            CorpusPreflightRule {
                rule_id: "private-key".to_owned(),
                signal: PrivateKey,
                default_decision: Skip,
                reason_code: "privacy.secret.private_key".to_owned(),
                redaction_action: DropContent,
                override_allowed: false,
                false_positive_suppressions: Vec::new(),
            },
            CorpusPreflightRule {
                rule_id: "generated-artifact".to_owned(),
                signal: GeneratedArtifact,
                default_decision: Defer,
                reason_code: "privacy.artifact.generated".to_owned(),
                redaction_action: MetadataOnly,
                override_allowed: true,
                false_positive_suppressions: vec!["checked-in-golden".to_owned()],
            },
            CorpusPreflightRule {
                rule_id: "oversized-binary".to_owned(),
                signal: OversizedBinary,
                default_decision: Defer,
                reason_code: "privacy.artifact.oversized_binary".to_owned(),
                redaction_action: MetadataOnly,
                override_allowed: false,
                false_positive_suppressions: Vec::new(),
            },
            CorpusPreflightRule {
                rule_id: "sensitive-path".to_owned(),
                signal: SensitivePath,
                default_decision: Skip,
                reason_code: "privacy.path.sensitive".to_owned(),
                redaction_action: HashPath,
                override_allowed: false,
                false_positive_suppressions: Vec::new(),
            },
            CorpusPreflightRule {
                rule_id: "personal-data".to_owned(),
                signal: PersonalData,
                default_decision: Skip,
                reason_code: "privacy.personal_data.detected".to_owned(),
                redaction_action: MaskExcerpt,
                override_allowed: true,
                false_positive_suppressions: vec!["public-contact-page".to_owned()],
            },
            CorpusPreflightRule {
                rule_id: "token-doc-false-positive".to_owned(),
                signal: CredentialToken,
                default_decision: Include,
                reason_code: "privacy.false_positive.suppressed".to_owned(),
                redaction_action: MaskExcerpt,
                override_allowed: false,
                false_positive_suppressions: vec!["documented-placeholder".to_owned()],
            },
        ],
        required_report_fields: vec![
            "run_id".to_owned(),
            "dry_run".to_owned(),
            "config_hash".to_owned(),
            "redaction_profile".to_owned(),
            "decisions".to_owned(),
            "summary".to_owned(),
            "replay_command".to_owned(),
        ],
    }
}

/// Builds the canonical dry-run report fixture for privacy preflight behavior.
#[must_use]
pub fn corpus_privacy_preflight_report_fixture() -> CorpusPrivacyPreflightReport {
    CorpusPrivacyPreflightReport {
        kind: CORPUS_PRIVACY_PREFLIGHT_REPORT_KIND.to_owned(),
        v: CORPUS_PRIVACY_PREFLIGHT_SCHEMA_VERSION,
        run_id: "privacy-preflight-smoke-001".to_owned(),
        generated_at: "2026-05-08T00:00:00Z".to_owned(),
        dry_run: true,
        root: "/home/ubuntu/project".to_owned(),
        config_hash: fixed_hash("11"),
        redaction_profile: CORPUS_PRIVACY_PREFLIGHT_REDACTION_PROFILE.to_owned(),
        replay_command: "scripts/check_fsfs_corpus_privacy_preflight.sh --mode smoke".to_owned(),
        destructive_cleanup_allowed: false,
        decisions: vec![
            CorpusPreflightFileDecision {
                path: "/home/ubuntu/.ssh/id_ed25519".to_owned(),
                decision: CorpusPreflightDecision::Skip,
                reason_code: "privacy.secret.private_key".to_owned(),
                signals: vec![
                    CorpusPreflightSignal::PrivateKey,
                    CorpusPreflightSignal::SensitivePath,
                ],
                sensitive_classes: vec!["private_keys".to_owned(), "ssh_material".to_owned()],
                redaction_action: CorpusPreflightRedactionAction::DropContent,
                evidence: evidence("ssh-key", "22", "<redacted:private_key>"),
                semantic_index_allowed: false,
                lexical_index_allowed: false,
                evidence_emit_allowed: true,
                false_positive_suppression_id: None,
                override_event: None,
            },
            CorpusPreflightFileDecision {
                path: "/home/ubuntu/project/.env".to_owned(),
                decision: CorpusPreflightDecision::Skip,
                reason_code: "privacy.secret.token".to_owned(),
                signals: vec![CorpusPreflightSignal::CredentialToken],
                sensitive_classes: vec!["credentials".to_owned(), "tokens".to_owned()],
                redaction_action: CorpusPreflightRedactionAction::MaskExcerpt,
                evidence: evidence("env-token", "33", "<redacted:credential_token>"),
                semantic_index_allowed: false,
                lexical_index_allowed: false,
                evidence_emit_allowed: true,
                false_positive_suppression_id: None,
                override_event: None,
            },
            CorpusPreflightFileDecision {
                path: "/home/ubuntu/project/target/debug/app".to_owned(),
                decision: CorpusPreflightDecision::Defer,
                reason_code: "privacy.artifact.generated".to_owned(),
                signals: vec![
                    CorpusPreflightSignal::GeneratedArtifact,
                    CorpusPreflightSignal::OversizedBinary,
                ],
                sensitive_classes: Vec::new(),
                redaction_action: CorpusPreflightRedactionAction::MetadataOnly,
                evidence: evidence("target-binary", "44", "<metadata-only>"),
                semantic_index_allowed: false,
                lexical_index_allowed: false,
                evidence_emit_allowed: true,
                false_positive_suppression_id: None,
                override_event: None,
            },
            CorpusPreflightFileDecision {
                path: "/home/ubuntu/project/docs/token-format.md".to_owned(),
                decision: CorpusPreflightDecision::Include,
                reason_code: "privacy.false_positive.suppressed".to_owned(),
                signals: vec![CorpusPreflightSignal::CredentialToken],
                sensitive_classes: vec!["tokens".to_owned()],
                redaction_action: CorpusPreflightRedactionAction::MaskExcerpt,
                evidence: evidence(
                    "token-doc",
                    "55",
                    "example token shape <redacted:placeholder>",
                ),
                semantic_index_allowed: true,
                lexical_index_allowed: true,
                evidence_emit_allowed: true,
                false_positive_suppression_id: Some("documented-placeholder".to_owned()),
                override_event: None,
            },
            CorpusPreflightFileDecision {
                path: "/home/ubuntu/project/export/customer_dump.bin".to_owned(),
                decision: CorpusPreflightDecision::Defer,
                reason_code: "privacy.artifact.oversized_binary".to_owned(),
                signals: vec![
                    CorpusPreflightSignal::OversizedBinary,
                    CorpusPreflightSignal::PersonalData,
                ],
                sensitive_classes: vec!["personal_data".to_owned()],
                redaction_action: CorpusPreflightRedactionAction::MetadataOnly,
                evidence: evidence("customer-dump", "66", "<metadata-only>"),
                semantic_index_allowed: false,
                lexical_index_allowed: false,
                evidence_emit_allowed: true,
                false_positive_suppression_id: None,
                override_event: None,
            },
        ],
        summary: CorpusPreflightSummary {
            included: 1,
            skipped: 2,
            deferred: 2,
            overrides_applied: 0,
            false_positive_suppressions: 1,
            raw_content_present: false,
            destructive_cleanup_allowed: false,
        },
    }
}

/// Builds the canonical explicit override fixture for privacy preflight behavior.
#[must_use]
pub fn corpus_privacy_preflight_override_fixture() -> CorpusPrivacyPreflightReport {
    CorpusPrivacyPreflightReport {
        kind: CORPUS_PRIVACY_PREFLIGHT_REPORT_KIND.to_owned(),
        v: CORPUS_PRIVACY_PREFLIGHT_SCHEMA_VERSION,
        run_id: "privacy-preflight-override-001".to_owned(),
        generated_at: "2026-05-08T00:00:00Z".to_owned(),
        dry_run: true,
        root: "/home/ubuntu/project".to_owned(),
        config_hash: fixed_hash("77"),
        redaction_profile: CORPUS_PRIVACY_PREFLIGHT_REDACTION_PROFILE.to_owned(),
        replay_command: "scripts/check_fsfs_corpus_privacy_preflight.sh --mode e2e".to_owned(),
        destructive_cleanup_allowed: false,
        decisions: vec![CorpusPreflightFileDecision {
            path: "/home/ubuntu/project/docs/example.env".to_owned(),
            decision: CorpusPreflightDecision::Include,
            reason_code: "privacy.override.user_approved".to_owned(),
            signals: vec![CorpusPreflightSignal::CredentialToken],
            sensitive_classes: vec!["tokens".to_owned()],
            redaction_action: CorpusPreflightRedactionAction::MaskExcerpt,
            evidence: evidence("override-example-env", "88", "TOKEN=<redacted:placeholder>"),
            semantic_index_allowed: true,
            lexical_index_allowed: true,
            evidence_emit_allowed: true,
            false_positive_suppression_id: None,
            override_event: Some(CorpusPreflightOverride {
                requested_decision: CorpusPreflightDecision::Include,
                approved: true,
                reason: "example fixture contains documented placeholder tokens".to_owned(),
                reason_code: "privacy.override.user_approved".to_owned(),
                expires_at: "2026-05-09T00:00:00Z".to_owned(),
            }),
        }],
        summary: CorpusPreflightSummary {
            included: 1,
            skipped: 0,
            deferred: 0,
            overrides_applied: 1,
            false_positive_suppressions: 0,
            raw_content_present: false,
            destructive_cleanup_allowed: false,
        },
    }
}

impl CorpusPrivacyPreflightContractDefinition {
    /// Validates the privacy preflight rule matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when required contract invariants are missing or unsafe.
    pub fn validate(&self) -> Result<(), CorpusPrivacyPreflightViolation> {
        if self.kind != CORPUS_PRIVACY_PREFLIGHT_CONTRACT_KIND {
            return Err(CorpusPrivacyPreflightViolation::new(
                "kind",
                "unexpected contract kind",
            ));
        }
        if self.v != CORPUS_PRIVACY_PREFLIGHT_SCHEMA_VERSION {
            return Err(CorpusPrivacyPreflightViolation::new(
                "v",
                "unsupported contract version",
            ));
        }
        if !self.dry_run_only {
            return Err(CorpusPrivacyPreflightViolation::new(
                "dry_run_only",
                "preflight contract must remain dry-run only",
            ));
        }
        if self.destructive_cleanup_allowed {
            return Err(CorpusPrivacyPreflightViolation::new(
                "destructive_cleanup_allowed",
                "privacy preflight must never authorize cleanup",
            ));
        }

        let mut signals = BTreeSet::new();
        let mut rule_ids = BTreeSet::new();
        for rule in &self.rule_matrix {
            require_nonempty(&rule.rule_id, "rule_id")?;
            require_reason_code(&rule.reason_code, "rule.reason_code")?;
            if !rule_ids.insert(rule.rule_id.as_str()) {
                return Err(CorpusPrivacyPreflightViolation::new(
                    "rule_id",
                    "duplicate preflight rule id",
                ));
            }
            signals.insert(rule.signal);
        }

        for required in [
            CorpusPreflightSignal::CredentialToken,
            CorpusPreflightSignal::PrivateKey,
            CorpusPreflightSignal::GeneratedArtifact,
            CorpusPreflightSignal::OversizedBinary,
            CorpusPreflightSignal::SensitivePath,
            CorpusPreflightSignal::PersonalData,
        ] {
            if !signals.contains(&required) {
                return Err(CorpusPrivacyPreflightViolation::new(
                    "rule_matrix",
                    "missing required preflight signal",
                ));
            }
        }
        Ok(())
    }
}

impl CorpusPrivacyPreflightReport {
    /// Validates dry-run report safety and summary consistency.
    ///
    /// # Errors
    ///
    /// Returns an error when the report can leak raw content, authorize cleanup,
    /// or misstate include/skip/defer outcomes.
    pub fn validate(&self) -> Result<(), CorpusPrivacyPreflightViolation> {
        if self.kind != CORPUS_PRIVACY_PREFLIGHT_REPORT_KIND {
            return Err(CorpusPrivacyPreflightViolation::new(
                "kind",
                "unexpected report kind",
            ));
        }
        if self.v != CORPUS_PRIVACY_PREFLIGHT_SCHEMA_VERSION {
            return Err(CorpusPrivacyPreflightViolation::new(
                "v",
                "unsupported report version",
            ));
        }
        if !self.dry_run {
            return Err(CorpusPrivacyPreflightViolation::new(
                "dry_run",
                "privacy preflight reports must be dry-run",
            ));
        }
        if self.destructive_cleanup_allowed || self.summary.destructive_cleanup_allowed {
            return Err(CorpusPrivacyPreflightViolation::new(
                "destructive_cleanup_allowed",
                "privacy preflight must not plan cleanup",
            ));
        }
        if self.summary.raw_content_present {
            return Err(CorpusPrivacyPreflightViolation::new(
                "summary.raw_content_present",
                "summary reports raw content",
            ));
        }

        require_nonempty(&self.run_id, "run_id")?;
        require_nonempty(&self.generated_at, "generated_at")?;
        require_nonempty(&self.root, "root")?;
        require_sha256(&self.config_hash, "config_hash")?;
        require_nonempty(&self.redaction_profile, "redaction_profile")?;
        require_nonempty(&self.replay_command, "replay_command")?;
        if self.decisions.is_empty() {
            return Err(CorpusPrivacyPreflightViolation::new(
                "decisions",
                "preflight report must include at least one decision",
            ));
        }

        let mut included = 0;
        let mut skipped = 0;
        let mut deferred = 0;
        let mut overrides_applied = 0;
        let mut false_positive_suppressions = 0;
        let mut paths = BTreeSet::new();
        let mut evidence_ids = BTreeSet::new();

        for decision in &self.decisions {
            decision.validate()?;
            if !paths.insert(decision.path.as_str()) {
                return Err(CorpusPrivacyPreflightViolation::new(
                    "path",
                    "duplicate file decision path",
                ));
            }
            if !evidence_ids.insert(decision.evidence.evidence_id.as_str()) {
                return Err(CorpusPrivacyPreflightViolation::new(
                    "evidence_id",
                    "duplicate evidence id",
                ));
            }

            match decision.decision {
                CorpusPreflightDecision::Include => included += 1,
                CorpusPreflightDecision::Skip => skipped += 1,
                CorpusPreflightDecision::Defer => deferred += 1,
            }
            if decision
                .override_event
                .as_ref()
                .is_some_and(|event| event.approved)
            {
                overrides_applied += 1;
            }
            if decision.false_positive_suppression_id.is_some() {
                false_positive_suppressions += 1;
            }
        }

        let expected = (
            self.summary.included,
            self.summary.skipped,
            self.summary.deferred,
            self.summary.overrides_applied,
            self.summary.false_positive_suppressions,
        );
        let actual = (
            included,
            skipped,
            deferred,
            overrides_applied,
            false_positive_suppressions,
        );
        if actual != expected {
            return Err(CorpusPrivacyPreflightViolation::new(
                "summary",
                "summary counts do not match decisions",
            ));
        }
        Ok(())
    }
}

impl CorpusPreflightFileDecision {
    fn validate(&self) -> Result<(), CorpusPrivacyPreflightViolation> {
        require_nonempty(&self.path, "decision.path")?;
        require_reason_code(&self.reason_code, "decision.reason_code")?;
        if self.signals.is_empty() {
            return Err(CorpusPrivacyPreflightViolation::new(
                "decision.signals",
                "decision must include at least one signal",
            ));
        }
        self.evidence.validate()?;
        if self.semantic_index_allowed && self.decision != CorpusPreflightDecision::Include {
            return Err(CorpusPrivacyPreflightViolation::new(
                "semantic_index_allowed",
                "only include decisions may enter semantic index",
            ));
        }
        if self.lexical_index_allowed && self.decision != CorpusPreflightDecision::Include {
            return Err(CorpusPrivacyPreflightViolation::new(
                "lexical_index_allowed",
                "only include decisions may enter lexical index",
            ));
        }
        if self.reason_code == "privacy.false_positive.suppressed"
            && self.false_positive_suppression_id.is_none()
        {
            return Err(CorpusPrivacyPreflightViolation::new(
                "false_positive_suppression_id",
                "false-positive suppression must name the suppressor",
            ));
        }
        if let Some(override_event) = &self.override_event {
            override_event.validate()?;
        }
        Ok(())
    }
}

impl CorpusPreflightEvidence {
    fn validate(&self) -> Result<(), CorpusPrivacyPreflightViolation> {
        require_nonempty(&self.evidence_id, "evidence.evidence_id")?;
        require_sha256(&self.sample_hash, "evidence.sample_hash")?;
        require_nonempty(&self.redacted_excerpt, "evidence.redacted_excerpt")?;
        if self.raw_content_present {
            return Err(CorpusPrivacyPreflightViolation::new(
                "evidence.raw_content_present",
                "raw content must not be present in preflight evidence",
            ));
        }
        if !self.redaction_applied {
            return Err(CorpusPrivacyPreflightViolation::new(
                "evidence.redaction_applied",
                "preflight evidence must prove redaction",
            ));
        }
        Ok(())
    }
}

impl CorpusPreflightOverride {
    fn validate(&self) -> Result<(), CorpusPrivacyPreflightViolation> {
        require_reason_code(&self.reason_code, "override.reason_code")?;
        require_nonempty(&self.expires_at, "override.expires_at")?;
        if self.approved && self.reason.trim().is_empty() {
            return Err(CorpusPrivacyPreflightViolation::new(
                "override.reason",
                "approved overrides require a reason",
            ));
        }
        Ok(())
    }
}

fn evidence(id: &str, hex: &str, redacted_excerpt: &str) -> CorpusPreflightEvidence {
    CorpusPreflightEvidence {
        evidence_id: id.to_owned(),
        sample_hash: fixed_hash(hex),
        redacted_excerpt: redacted_excerpt.to_owned(),
        raw_content_present: false,
        redaction_applied: true,
    }
}

fn fixed_hash(pair: &str) -> String {
    format!("sha256:{}", pair.repeat(32))
}

fn require_nonempty(value: &str, field: &str) -> Result<(), CorpusPrivacyPreflightViolation> {
    if value.trim().is_empty() {
        return Err(CorpusPrivacyPreflightViolation::new(
            field,
            "value must not be empty",
        ));
    }
    Ok(())
}

fn require_reason_code(value: &str, field: &str) -> Result<(), CorpusPrivacyPreflightViolation> {
    require_nonempty(value, field)?;
    if !value.starts_with("privacy.") {
        return Err(CorpusPrivacyPreflightViolation::new(
            field,
            "reason code must use privacy namespace",
        ));
    }
    if !value
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '.')
    {
        return Err(CorpusPrivacyPreflightViolation::new(
            field,
            "reason code must be stable lowercase ascii",
        ));
    }
    Ok(())
}

fn require_sha256(value: &str, field: &str) -> Result<(), CorpusPrivacyPreflightViolation> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(CorpusPrivacyPreflightViolation::new(
            field,
            "hash must use sha256 prefix",
        ));
    };
    if hex.len() != 64
        || !hex
            .chars()
            .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase())
    {
        return Err(CorpusPrivacyPreflightViolation::new(
            field,
            "hash must contain 64 lowercase hex characters",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_preflight_contract_covers_required_rule_families() {
        let contract = corpus_privacy_preflight_contract_definition();
        contract.validate().expect("contract should validate");

        let signals: BTreeSet<CorpusPreflightSignal> = contract
            .rule_matrix
            .iter()
            .map(|rule| rule.signal)
            .collect();
        assert!(signals.contains(&CorpusPreflightSignal::CredentialToken));
        assert!(signals.contains(&CorpusPreflightSignal::PrivateKey));
        assert!(signals.contains(&CorpusPreflightSignal::GeneratedArtifact));
        assert!(signals.contains(&CorpusPreflightSignal::OversizedBinary));
        assert!(signals.contains(&CorpusPreflightSignal::SensitivePath));
        assert!(signals.contains(&CorpusPreflightSignal::PersonalData));
        assert!(!contract.destructive_cleanup_allowed);
    }

    #[test]
    fn corpus_preflight_report_covers_include_skip_defer_without_raw_content() {
        let report = corpus_privacy_preflight_report_fixture();
        report.validate().expect("report should validate");

        assert!(report.dry_run);
        assert!(!report.destructive_cleanup_allowed);
        assert!(!report.summary.raw_content_present);
        assert_eq!(report.summary.included, 1);
        assert_eq!(report.summary.skipped, 2);
        assert_eq!(report.summary.deferred, 2);
    }

    #[test]
    fn corpus_preflight_false_positive_suppression_allows_safe_include() {
        let report = corpus_privacy_preflight_report_fixture();
        let decision = report
            .decisions
            .iter()
            .find(|decision| decision.reason_code == "privacy.false_positive.suppressed")
            .expect("false-positive decision");

        assert_eq!(decision.decision, CorpusPreflightDecision::Include);
        assert_eq!(
            decision.false_positive_suppression_id.as_deref(),
            Some("documented-placeholder")
        );
        assert!(decision.semantic_index_allowed);
        assert!(decision.lexical_index_allowed);
    }

    #[test]
    fn corpus_preflight_override_requires_reason_and_redacted_evidence() {
        let report = corpus_privacy_preflight_override_fixture();
        report.validate().expect("override report should validate");

        let mut invalid = report.clone();
        let decision = invalid
            .decisions
            .first_mut()
            .expect("override decision should exist");
        let override_event = decision.override_event.as_mut().expect("override event");
        override_event.reason.clear();
        let error = invalid.validate().expect_err("missing reason should fail");
        assert_eq!(error.field, "override.reason");
    }

    #[test]
    fn corpus_preflight_rejects_raw_content_and_cleanup_plans() {
        let mut report = corpus_privacy_preflight_report_fixture();
        report
            .decisions
            .first_mut()
            .expect("preflight decision should exist")
            .evidence
            .raw_content_present = true;
        let error = report.validate().expect_err("raw content should fail");
        assert_eq!(error.field, "evidence.raw_content_present");

        let mut cleanup = corpus_privacy_preflight_report_fixture();
        cleanup.destructive_cleanup_allowed = true;
        let error = cleanup.validate().expect_err("cleanup should fail");
        assert_eq!(error.field, "destructive_cleanup_allowed");
    }
}
