//! Host adapter SDK and conformance harness for telemetry integrations.
//!
//! This module provides:
//! - A stable adapter interface for host identity handshake, telemetry emission,
//!   and lifecycle hooks.
//! - Shared conformance validators for schema-version and redaction compliance.
//! - A harness that runs fixture-based contract checks with actionable diagnostics.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::collectors::{
    TELEMETRY_SCHEMA_VERSION, TelemetryCorrelation, TelemetryEnvelope, TelemetryEvent,
    TelemetryInstance,
};
use crate::error::{SearchError, SearchResult};

/// Default forbidden substrings used by redaction conformance checks.
///
/// These patterns intentionally target high-signal secret formats to minimize false positives
/// on normal telemetry payloads.
pub const DEFAULT_REDACTION_FORBIDDEN_PATTERNS: &[&str] = &[
    "BEGIN PRIVATE KEY",
    "AWS_SECRET_ACCESS_KEY=",
    "ghp_",
    "xoxb-",
];

/// Adapter identity handshake payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterIdentity {
    /// Stable adapter identifier (e.g. `xf-host-adapter`).
    pub adapter_id: String,
    /// Adapter implementation version.
    pub adapter_version: String,
    /// Host project identifier (required).
    pub host_project: String,
    /// Optional runtime role label (e.g. `indexer`, `query`, `control-plane`).
    pub runtime_role: Option<String>,
    /// Optional stable instance UUID advertised by the host integration.
    pub instance_uuid: Option<String>,
    /// Telemetry schema version emitted by this adapter.
    pub telemetry_schema_version: u8,
    /// Redaction policy version enforced by this adapter.
    pub redaction_policy_version: String,
}

/// Canonical host projects with first-class adapter support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalHostProject {
    CodingAgentSessionSearch,
    Xf,
    McpAgentMailRust,
    Frankenterm,
}

impl CanonicalHostProject {
    /// Deterministic host iteration order used by tests and harnesses.
    pub const ALL: [Self; 4] = [
        Self::CodingAgentSessionSearch,
        Self::Xf,
        Self::McpAgentMailRust,
        Self::Frankenterm,
    ];

    /// Canonical host project key.
    #[must_use]
    pub const fn host_project_key(self) -> &'static str {
        match self {
            Self::CodingAgentSessionSearch => "coding_agent_session_search",
            Self::Xf => "xf",
            Self::McpAgentMailRust => "mcp_agent_mail_rust",
            Self::Frankenterm => "frankenterm",
        }
    }

    /// Stable adapter identifier for this host.
    #[must_use]
    pub const fn adapter_id(self) -> &'static str {
        match self {
            Self::CodingAgentSessionSearch => "cass-host-adapter",
            Self::Xf => "xf-host-adapter",
            Self::McpAgentMailRust => "mcp-agent-mail-host-adapter",
            Self::Frankenterm => "frankenterm-host-adapter",
        }
    }

    /// Default runtime role label for this host adapter.
    #[must_use]
    pub const fn default_runtime_role(self) -> &'static str {
        match self {
            Self::McpAgentMailRust => "control-plane",
            Self::CodingAgentSessionSearch | Self::Xf | Self::Frankenterm => "query",
        }
    }
}

/// Canonical host-project attribution result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostProjectAttribution {
    /// Canonical project key or `unknown` when unresolved.
    pub resolved_project_key: String,
    /// Confidence score in `[0, 100]`.
    pub confidence_score: u8,
    /// Machine-stable reason code for diagnostics.
    pub reason_code: String,
    /// Whether multiple conflicting project candidates were observed.
    pub collision: bool,
}

impl HostProjectAttribution {
    #[must_use]
    pub fn unknown(reason_code: impl Into<String>) -> Self {
        Self {
            resolved_project_key: "unknown".to_owned(),
            confidence_score: 20,
            reason_code: reason_code.into(),
            collision: false,
        }
    }
}

/// Resolve canonical host-project attribution from adapter identity + telemetry hints.
///
/// The algorithm is deterministic with source-precedence tie breaks:
/// adapter identity > telemetry project key > host name hint.
#[must_use]
pub fn resolve_host_project_attribution(
    identity_host_project: Option<&str>,
    telemetry_project_key: Option<&str>,
    host_name_hint: Option<&str>,
) -> HostProjectAttribution {
    #[derive(Debug, Clone, Copy)]
    struct Candidate {
        project: &'static str,
        weight: u8,
        reason: &'static str,
    }

    let mut candidates: Vec<Candidate> = Vec::new();

    if let Some(hint) = identity_host_project {
        for project in canonical_projects_from_hint(hint) {
            candidates.push(Candidate {
                project,
                weight: 4,
                reason: "adapter_identity",
            });
        }
    }

    if let Some(hint) = telemetry_project_key {
        for project in canonical_projects_from_hint(hint) {
            candidates.push(Candidate {
                project,
                weight: 3,
                reason: "telemetry_project_key",
            });
        }
    }

    if let Some(hint) = host_name_hint {
        for project in canonical_projects_from_hint(hint) {
            candidates.push(Candidate {
                project,
                weight: 1,
                reason: "host_name",
            });
        }
    }

    if candidates.is_empty() {
        return HostProjectAttribution::unknown("attribution.unknown");
    }

    let unique_projects: BTreeSet<&str> = candidates
        .iter()
        .map(|candidate| candidate.project)
        .collect();
    let collision = unique_projects.len() > 1;

    let Some(winner) = candidates.into_iter().max_by(|left, right| {
        left.weight
            .cmp(&right.weight)
            .then_with(|| right.project.cmp(left.project))
            .then_with(|| right.reason.cmp(left.reason))
    }) else {
        return HostProjectAttribution::unknown("attribution.unknown");
    };

    let mut confidence_score: u8 = match winner.weight {
        4 => 95,
        3 => 85,
        1 => 60,
        _ => 50,
    };
    if collision {
        confidence_score = confidence_score.saturating_sub(25);
    }

    let reason_code = if collision {
        "attribution.collision".to_owned()
    } else {
        format!("attribution.{}", winner.reason)
    };

    HostProjectAttribution {
        resolved_project_key: winner.project.to_owned(),
        confidence_score,
        reason_code,
        collision,
    }
}

/// Lifecycle hooks exposed to host adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AdapterLifecycleEvent {
    /// Adapter session startup.
    SessionStart { ts: String },
    /// Adapter session shutdown.
    SessionStop { ts: String },
    /// Periodic health tick.
    HealthTick { ts: String },
}

/// Host adapter interface used by integrations.
pub trait HostAdapter: fmt::Debug + Send + Sync {
    /// Identity handshake called during initialization and conformance checks.
    fn identity(&self) -> AdapterIdentity;

    /// Emit one canonical telemetry envelope.
    ///
    /// Implementations must be non-blocking and should preserve envelope fidelity.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when emission fails due to adapter transport,
    /// serialization, or sink-level validation problems.
    fn emit_telemetry(&self, envelope: &TelemetryEnvelope) -> SearchResult<()>;

    /// Lifecycle hook invoked by orchestration and conformance harnesses.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when lifecycle processing fails.
    fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()>;
}

/// Non-blocking sink abstraction used by concrete host adapters.
pub trait AdapterSink: Send + Sync {
    /// Emit a canonical telemetry envelope.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when sink delivery fails.
    fn emit(&self, envelope: &TelemetryEnvelope) -> SearchResult<()>;

    /// Process one lifecycle event.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when lifecycle handling fails.
    fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()>;
}

/// No-op sink used by default when callers only need conformance-safe identities.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopAdapterSink;

impl AdapterSink for NoopAdapterSink {
    fn emit(&self, _: &TelemetryEnvelope) -> SearchResult<()> {
        Ok(())
    }

    fn on_lifecycle_event(&self, _: &AdapterLifecycleEvent) -> SearchResult<()> {
        Ok(())
    }
}

/// Reusable host adapter implementation that forwards to an injected sink.
///
/// This gives host projects one deterministic adapter path:
/// - canonical identity handshake
/// - non-blocking sink forwarding
/// - shared conformance harness compatibility
pub struct ForwardingHostAdapter {
    identity: AdapterIdentity,
    sink: Arc<dyn AdapterSink>,
}

impl fmt::Debug for ForwardingHostAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardingHostAdapter")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

impl ForwardingHostAdapter {
    /// Construct an adapter with an explicit identity and no-op sink.
    #[must_use]
    pub fn new(identity: AdapterIdentity) -> Self {
        Self {
            identity,
            sink: Arc::new(NoopAdapterSink),
        }
    }

    /// Construct a canonical host adapter profile with shared defaults.
    #[must_use]
    pub fn for_host(
        host: CanonicalHostProject,
        adapter_version: impl Into<String>,
        runtime_role: Option<String>,
        instance_uuid: Option<String>,
    ) -> Self {
        let runtime_role = runtime_role.or_else(|| Some(host.default_runtime_role().to_owned()));
        Self::new(AdapterIdentity {
            adapter_id: host.adapter_id().to_owned(),
            adapter_version: adapter_version.into(),
            host_project: host.host_project_key().to_owned(),
            runtime_role,
            instance_uuid,
            telemetry_schema_version: TELEMETRY_SCHEMA_VERSION,
            redaction_policy_version: "v1".to_owned(),
        })
    }

    /// Canonical profile for `coding_agent_session_search` (cass).
    #[must_use]
    pub fn for_cass(adapter_version: impl Into<String>, instance_uuid: Option<String>) -> Self {
        Self::for_host(
            CanonicalHostProject::CodingAgentSessionSearch,
            adapter_version,
            None,
            instance_uuid,
        )
    }

    /// Canonical profile for `xf`.
    #[must_use]
    pub fn for_xf(adapter_version: impl Into<String>, instance_uuid: Option<String>) -> Self {
        Self::for_host(
            CanonicalHostProject::Xf,
            adapter_version,
            None,
            instance_uuid,
        )
    }

    /// Canonical profile for `mcp_agent_mail_rust`.
    #[must_use]
    pub fn for_mcp_agent_mail(
        adapter_version: impl Into<String>,
        instance_uuid: Option<String>,
    ) -> Self {
        Self::for_host(
            CanonicalHostProject::McpAgentMailRust,
            adapter_version,
            None,
            instance_uuid,
        )
    }

    /// Canonical profile for `frankenterm`.
    #[must_use]
    pub fn for_frankenterm(
        adapter_version: impl Into<String>,
        instance_uuid: Option<String>,
    ) -> Self {
        Self::for_host(
            CanonicalHostProject::Frankenterm,
            adapter_version,
            None,
            instance_uuid,
        )
    }

    /// Replace the sink implementation used for forwarding.
    #[must_use]
    pub fn with_sink(mut self, sink: Arc<dyn AdapterSink>) -> Self {
        self.sink = sink;
        self
    }

    /// Borrow this adapter's identity.
    #[must_use]
    pub const fn identity_ref(&self) -> &AdapterIdentity {
        &self.identity
    }
}

impl HostAdapter for ForwardingHostAdapter {
    fn identity(&self) -> AdapterIdentity {
        self.identity.clone()
    }

    fn emit_telemetry(&self, envelope: &TelemetryEnvelope) -> SearchResult<()> {
        self.sink.emit(envelope)
    }

    fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()> {
        self.sink.on_lifecycle_event(event)
    }
}

/// One machine-readable conformance violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceViolation {
    /// Stable violation code.
    pub code: String,
    /// Field/path associated with the violation.
    pub field: String,
    /// Human-readable diagnostic.
    pub message: String,
}

/// Summary result for an adapter conformance run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceReport {
    /// Whether all checks passed.
    pub passed: bool,
    /// Number of fixture events validated.
    pub fixtures_checked: usize,
    /// Number of successful adapter `emit_telemetry` calls.
    pub emitted_events: usize,
    /// Number of successful lifecycle hook invocations.
    pub lifecycle_hooks_checked: usize,
    /// Validation failures.
    pub violations: Vec<ConformanceViolation>,
}

impl ConformanceReport {
    const fn with_violations(
        fixtures_checked: usize,
        emitted_events: usize,
        lifecycle_hooks_checked: usize,
        violations: Vec<ConformanceViolation>,
    ) -> Self {
        Self {
            passed: violations.is_empty(),
            fixtures_checked,
            emitted_events,
            lifecycle_hooks_checked,
            violations,
        }
    }
}

/// Configuration for adapter conformance checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceConfig {
    /// Required telemetry schema version.
    pub expected_schema_version: u8,
    /// Required redaction policy version label.
    pub required_redaction_policy_version: String,
    /// Forbidden substrings that must not appear in serialized telemetry payloads.
    pub forbidden_substrings: Vec<String>,
}

impl Default for ConformanceConfig {
    fn default() -> Self {
        Self {
            expected_schema_version: TELEMETRY_SCHEMA_VERSION,
            required_redaction_policy_version: "v1".to_owned(),
            forbidden_substrings: DEFAULT_REDACTION_FORBIDDEN_PATTERNS
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        }
    }
}

/// Conformance harness for host adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceHarness {
    config: ConformanceConfig,
}

impl Default for ConformanceHarness {
    fn default() -> Self {
        Self::new(ConformanceConfig::default())
    }
}

impl ConformanceHarness {
    /// Construct a harness with explicit configuration.
    #[must_use]
    pub const fn new(config: ConformanceConfig) -> Self {
        Self { config }
    }

    /// Access harness configuration.
    #[must_use]
    pub const fn config(&self) -> &ConformanceConfig {
        &self.config
    }

    /// Validate adapter identity handshake.
    #[must_use]
    pub fn validate_identity(&self, identity: &AdapterIdentity) -> Vec<ConformanceViolation> {
        let mut violations = Vec::new();

        if identity.adapter_id.trim().is_empty() {
            violations.push(violation(
                "adapter.identity.missing_adapter_id",
                "identity.adapter_id",
                "adapter_id must be non-empty",
            ));
        }
        if identity.adapter_version.trim().is_empty() {
            violations.push(violation(
                "adapter.identity.missing_adapter_version",
                "identity.adapter_version",
                "adapter_version must be non-empty",
            ));
        }
        if identity.host_project.trim().is_empty() {
            violations.push(violation(
                "adapter.identity.missing_host_project",
                "identity.host_project",
                "host_project must be non-empty",
            ));
        }
        if identity
            .runtime_role
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            violations.push(violation(
                "adapter.identity.empty_runtime_role",
                "identity.runtime_role",
                "runtime_role must be non-empty when provided",
            ));
        }
        if identity
            .instance_uuid
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            violations.push(violation(
                "adapter.identity.empty_instance_uuid",
                "identity.instance_uuid",
                "instance_uuid must be non-empty when provided",
            ));
        }

        if identity.telemetry_schema_version != self.config.expected_schema_version {
            violations.push(ConformanceViolation {
                code: "adapter.identity.schema_version_mismatch".to_owned(),
                field: "identity.telemetry_schema_version".to_owned(),
                message: format!(
                    "expected schema v{}, got v{}",
                    self.config.expected_schema_version, identity.telemetry_schema_version
                ),
            });
        }

        if identity.redaction_policy_version != self.config.required_redaction_policy_version {
            violations.push(ConformanceViolation {
                code: "adapter.identity.redaction_policy_mismatch".to_owned(),
                field: "identity.redaction_policy_version".to_owned(),
                message: format!(
                    "expected redaction policy {}, got {}",
                    self.config.required_redaction_policy_version,
                    identity.redaction_policy_version
                ),
            });
        }

        violations
    }

    /// Validate one telemetry envelope for schema/version and redaction compliance.
    #[must_use]
    pub fn validate_envelope(&self, envelope: &TelemetryEnvelope) -> Vec<ConformanceViolation> {
        let mut violations = Vec::new();

        if envelope.v != self.config.expected_schema_version {
            violations.push(ConformanceViolation {
                code: "adapter.envelope.schema_version_mismatch".to_owned(),
                field: "envelope.v".to_owned(),
                message: format!(
                    "expected schema v{}, got v{}",
                    self.config.expected_schema_version, envelope.v
                ),
            });
        }

        if envelope.ts.trim().is_empty() {
            violations.push(violation(
                "adapter.envelope.missing_timestamp",
                "envelope.ts",
                "timestamp must be non-empty",
            ));
        }

        match &envelope.event {
            TelemetryEvent::Search {
                instance,
                correlation,
                query,
                ..
            } => validate_search_event(instance, correlation, query.text.as_str(), &mut violations),
            TelemetryEvent::Embedding {
                instance,
                correlation,
                duration_ms,
                ..
            } => validate_duration_event(
                "embedding",
                instance,
                correlation,
                *duration_ms,
                &mut violations,
            ),
            TelemetryEvent::Index {
                instance,
                correlation,
                duration_ms,
                ..
            } => validate_duration_event(
                "index",
                instance,
                correlation,
                *duration_ms,
                &mut violations,
            ),
            TelemetryEvent::Resource {
                instance,
                correlation,
                sample,
            } => {
                validate_resource_event(instance, correlation, sample.interval_ms, &mut violations);
            }
            TelemetryEvent::Lifecycle {
                instance,
                correlation,
                ..
            } => validate_lifecycle_event(instance, correlation, &mut violations),
        }

        violations.extend(validate_redaction_forbidden_substrings(
            envelope,
            &self.config.forbidden_substrings,
        ));

        violations
    }

    /// Execute a full conformance run over identity, lifecycle hooks, and fixture envelopes.
    ///
    /// Lifecycle order:
    /// 1. `SessionStart`
    /// 2. per-envelope `HealthTick`
    /// 3. `SessionStop`
    #[must_use]
    pub fn run(
        &self,
        adapter: &dyn HostAdapter,
        fixtures: &[TelemetryEnvelope],
    ) -> ConformanceReport {
        let identity = adapter.identity();
        let mut violations = self.validate_identity(&identity);
        let mut lifecycle_hooks_checked = 0usize;
        let mut emitted_events = 0usize;

        let start_event = AdapterLifecycleEvent::SessionStart {
            ts: "conformance-start".to_owned(),
        };
        match adapter.on_lifecycle_event(&start_event) {
            Ok(()) => lifecycle_hooks_checked += 1,
            Err(err) => violations.push(adapter_error_violation("lifecycle_start", &err)),
        }

        for (idx, envelope) in fixtures.iter().enumerate() {
            let event_violations = self.validate_envelope(envelope);
            if !event_violations.is_empty() {
                for violation in event_violations {
                    violations.push(ConformanceViolation {
                        code: violation.code,
                        field: format!("fixtures[{idx}].{}", violation.field),
                        message: violation.message,
                    });
                }
            }

            match adapter.emit_telemetry(envelope) {
                Ok(()) => emitted_events += 1,
                Err(err) => violations.push(adapter_error_violation("emit_telemetry", &err)),
            }

            let tick_event = AdapterLifecycleEvent::HealthTick {
                ts: format!("conformance-tick-{idx}"),
            };
            match adapter.on_lifecycle_event(&tick_event) {
                Ok(()) => lifecycle_hooks_checked += 1,
                Err(err) => violations.push(adapter_error_violation("lifecycle_tick", &err)),
            }
        }

        let stop_event = AdapterLifecycleEvent::SessionStop {
            ts: "conformance-stop".to_owned(),
        };
        match adapter.on_lifecycle_event(&stop_event) {
            Ok(()) => lifecycle_hooks_checked += 1,
            Err(err) => violations.push(adapter_error_violation("lifecycle_stop", &err)),
        }

        ConformanceReport::with_violations(
            fixtures.len(),
            emitted_events,
            lifecycle_hooks_checked,
            violations,
        )
    }
}

fn validate_instance(
    prefix: &str,
    instance: &TelemetryInstance,
    violations: &mut Vec<ConformanceViolation>,
) {
    if !is_valid_ulid(&instance.instance_id) {
        violations.push(ConformanceViolation {
            code: "adapter.event.instance.invalid_instance_id".to_owned(),
            field: format!("{prefix}.instance_id"),
            message: "instance_id must be a 26-char ULID".to_owned(),
        });
    }
    if instance.project_key.trim().is_empty() {
        violations.push(ConformanceViolation {
            code: "adapter.event.instance.missing_project_key".to_owned(),
            field: format!("{prefix}.project_key"),
            message: "project_key must be non-empty".to_owned(),
        });
    }
    if instance.host_name.trim().is_empty() {
        violations.push(ConformanceViolation {
            code: "adapter.event.instance.missing_host_name".to_owned(),
            field: format!("{prefix}.host_name"),
            message: "host_name must be non-empty".to_owned(),
        });
    }
}

fn validate_search_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    query_text: &str,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance("search.instance", instance, violations);
    validate_correlation("search.correlation", correlation, violations);
    if query_text.trim().is_empty() {
        violations.push(violation(
            "adapter.event.search.empty_query",
            "search.query.text",
            "query text must be non-empty",
        ));
    }
    let char_count = query_text.chars().count();
    if char_count > 500 {
        violations.push(ConformanceViolation {
            code: "adapter.event.search.query_too_long".to_owned(),
            field: "search.query.text".to_owned(),
            message: format!("query length {char_count} exceeds 500"),
        });
    }
}

fn validate_duration_event(
    prefix: &str,
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    duration_ms: u64,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance(&format!("{prefix}.instance"), instance, violations);
    validate_correlation(&format!("{prefix}.correlation"), correlation, violations);
    if duration_ms == 0 {
        violations.push(ConformanceViolation {
            code: format!("adapter.event.{prefix}.zero_duration"),
            field: format!("{prefix}.duration_ms"),
            message: "duration_ms should be > 0 for completed telemetry".to_owned(),
        });
    }
}

fn validate_resource_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    interval_ms: u64,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance("resource.instance", instance, violations);
    validate_correlation("resource.correlation", correlation, violations);
    if interval_ms == 0 {
        violations.push(violation(
            "adapter.event.resource.zero_interval",
            "resource.sample.interval_ms",
            "interval_ms must be greater than zero",
        ));
    }
}

fn validate_lifecycle_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance("lifecycle.instance", instance, violations);
    validate_correlation("lifecycle.correlation", correlation, violations);
}

fn validate_correlation(
    prefix: &str,
    correlation: &TelemetryCorrelation,
    violations: &mut Vec<ConformanceViolation>,
) {
    if !is_valid_ulid(&correlation.event_id) {
        violations.push(ConformanceViolation {
            code: "adapter.event.correlation.invalid_event_id".to_owned(),
            field: format!("{prefix}.event_id"),
            message: "event_id must be a 26-char ULID".to_owned(),
        });
    }
    if !is_valid_ulid(&correlation.root_request_id) {
        violations.push(ConformanceViolation {
            code: "adapter.event.correlation.invalid_root_request_id".to_owned(),
            field: format!("{prefix}.root_request_id"),
            message: "root_request_id must be a 26-char ULID".to_owned(),
        });
    }
    if let Some(parent_event_id) = &correlation.parent_event_id
        && !is_valid_ulid(parent_event_id)
    {
        violations.push(ConformanceViolation {
            code: "adapter.event.correlation.invalid_parent_event_id".to_owned(),
            field: format!("{prefix}.parent_event_id"),
            message: "parent_event_id must be a 26-char ULID when provided".to_owned(),
        });
    }
}

fn validate_redaction_forbidden_substrings(
    envelope: &TelemetryEnvelope,
    forbidden_substrings: &[String],
) -> Vec<ConformanceViolation> {
    let mut violations = Vec::new();
    let payload = match serde_json::to_string(envelope) {
        Ok(payload) => payload,
        Err(source) => {
            violations.push(ConformanceViolation {
                code: "adapter.redaction.serialization_failed".to_owned(),
                field: "envelope".to_owned(),
                message: format!("failed to serialize envelope for redaction check: {source}"),
            });
            return violations;
        }
    };

    let payload_lower = payload.to_ascii_lowercase();
    for pattern in forbidden_substrings {
        if pattern.is_empty() {
            continue;
        }
        let pattern_lower = pattern.to_ascii_lowercase();
        if payload_lower.contains(&pattern_lower) {
            violations.push(ConformanceViolation {
                code: "adapter.redaction.forbidden_pattern".to_owned(),
                field: "envelope".to_owned(),
                message: format!("payload contains forbidden pattern '{pattern}'"),
            });
        }
    }

    violations
}

fn is_valid_ulid(candidate: &str) -> bool {
    if candidate.len() != 26 {
        return false;
    }
    candidate.bytes().all(|byte| {
        matches!(
            byte,
            b'0'..=b'9'
                | b'A'..=b'H'
                | b'J'..=b'K'
                | b'M'..=b'N'
                | b'P'..=b'T'
                | b'V'..=b'Z'
                | b'a'..=b'h'
                | b'j'..=b'k'
                | b'm'..=b'n'
                | b'p'..=b't'
                | b'v'..=b'z'
        )
    })
}

fn violation(code: &str, field: &str, message: &str) -> ConformanceViolation {
    ConformanceViolation {
        code: code.to_owned(),
        field: field.to_owned(),
        message: message.to_owned(),
    }
}

fn adapter_error_violation(context: &str, error: &SearchError) -> ConformanceViolation {
    ConformanceViolation {
        code: "adapter.hook.error".to_owned(),
        field: context.to_owned(),
        message: error.to_string(),
    }
}

fn canonical_projects_from_hint(hint: &str) -> Vec<&'static str> {
    const CANONICAL_ALIASES: &[(&str, &[&str])] = &[
        (
            "coding_agent_session_search",
            &[
                "coding_agent_session_search",
                "codingagentsessionsearch",
                "cass",
            ],
        ),
        ("xf", &["xf"]),
        (
            "mcp_agent_mail_rust",
            &[
                "mcp_agent_mail_rust",
                "mcpagentmailrust",
                "mcpagentmail",
                "agent_mail",
                "agentmail",
                "amail",
            ],
        ),
        ("frankenterm", &["frankenterm"]),
    ];

    let normalized = normalize_project_hint(hint);
    if normalized.is_empty() {
        return Vec::new();
    }
    let tokens: BTreeSet<&str> = normalized
        .split('_')
        .filter(|token| !token.is_empty())
        .collect();

    let mut matches = Vec::new();
    for (canonical, aliases) in CANONICAL_ALIASES {
        if aliases
            .iter()
            .any(|alias| normalized == *alias || tokens.contains(alias))
        {
            matches.push(*canonical);
        }
    }

    matches.sort_unstable();
    matches.dedup();
    matches
}

fn normalize_project_hint(hint: &str) -> String {
    let mut normalized = String::with_capacity(hint.len());
    let mut pending_separator = false;

    for ch in hint.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_separator && !normalized.is_empty() {
                normalized.push('_');
            }
            normalized.push(ch.to_ascii_lowercase());
            pending_separator = false;
        } else {
            pending_separator = true;
        }
    }

    normalized
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Debug)]
    struct MemoryHostAdapter {
        identity: AdapterIdentity,
        emitted: Arc<Mutex<Vec<TelemetryEnvelope>>>,
        lifecycle: Arc<Mutex<Vec<AdapterLifecycleEvent>>>,
        fail_emit: bool,
        fail_lifecycle: bool,
    }

    impl MemoryHostAdapter {
        fn new(identity: AdapterIdentity) -> Self {
            Self {
                identity,
                emitted: Arc::new(Mutex::new(Vec::new())),
                lifecycle: Arc::new(Mutex::new(Vec::new())),
                fail_emit: false,
                fail_lifecycle: false,
            }
        }
    }

    impl HostAdapter for MemoryHostAdapter {
        fn identity(&self) -> AdapterIdentity {
            self.identity.clone()
        }

        fn emit_telemetry(&self, envelope: &TelemetryEnvelope) -> SearchResult<()> {
            if self.fail_emit {
                return Err(SearchError::InvalidConfig {
                    field: "adapter.emit".to_owned(),
                    value: "forced_failure".to_owned(),
                    reason: "emit failure requested in test".to_owned(),
                });
            }
            self.emitted.lock().unwrap().push(envelope.clone());
            Ok(())
        }

        fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()> {
            if self.fail_lifecycle {
                return Err(SearchError::InvalidConfig {
                    field: "adapter.lifecycle".to_owned(),
                    value: "forced_failure".to_owned(),
                    reason: "lifecycle failure requested in test".to_owned(),
                });
            }
            self.lifecycle.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct RecordingSink {
        emitted: Arc<Mutex<Vec<TelemetryEnvelope>>>,
        lifecycle: Arc<Mutex<Vec<AdapterLifecycleEvent>>>,
    }

    impl AdapterSink for RecordingSink {
        fn emit(&self, envelope: &TelemetryEnvelope) -> SearchResult<()> {
            self.emitted.lock().unwrap().push(envelope.clone());
            Ok(())
        }

        fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()> {
            self.lifecycle.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    fn default_identity() -> AdapterIdentity {
        AdapterIdentity {
            adapter_id: "sample-host-adapter".to_owned(),
            adapter_version: "0.1.0".to_owned(),
            host_project: "sample-host".to_owned(),
            runtime_role: Some("query".to_owned()),
            instance_uuid: Some("sample-instance-uuid".to_owned()),
            telemetry_schema_version: TELEMETRY_SCHEMA_VERSION,
            redaction_policy_version: "v1".to_owned(),
        }
    }

    fn load_fixture(name: &str) -> TelemetryEnvelope {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../schemas/fixtures")
            .join(name);
        let raw = fs::read_to_string(path).unwrap();
        serde_json::from_str(&raw).unwrap()
    }

    #[test]
    fn sample_host_adapter_passes_conformance_with_golden_fixtures() {
        let harness = ConformanceHarness::default();
        let adapter = MemoryHostAdapter::new(default_identity());
        let fixtures = vec![
            load_fixture("telemetry-search-v1.json"),
            load_fixture("telemetry-embedding-v1.json"),
            load_fixture("telemetry-index-v1.json"),
            load_fixture("telemetry-resource-v1.json"),
            load_fixture("telemetry-lifecycle-v1.json"),
        ];

        let report = harness.run(&adapter, &fixtures);
        assert!(report.passed, "{:?}", report.violations);
        assert_eq!(report.fixtures_checked, fixtures.len());
        assert_eq!(report.emitted_events, fixtures.len());
        assert!(report.lifecycle_hooks_checked >= 2);
        assert!(report.violations.is_empty());
    }

    #[test]
    fn canonical_host_profiles_emit_expected_identity_handshake() {
        let cases = [
            (
                CanonicalHostProject::CodingAgentSessionSearch,
                "cass-host-adapter",
                "coding_agent_session_search",
            ),
            (CanonicalHostProject::Xf, "xf-host-adapter", "xf"),
            (
                CanonicalHostProject::McpAgentMailRust,
                "mcp-agent-mail-host-adapter",
                "mcp_agent_mail_rust",
            ),
            (
                CanonicalHostProject::Frankenterm,
                "frankenterm-host-adapter",
                "frankenterm",
            ),
        ];

        for (host, expected_adapter_id, expected_project) in cases {
            let adapter = ForwardingHostAdapter::for_host(
                host,
                "1.2.3",
                None,
                Some("instance-xyz".to_owned()),
            );
            let identity = adapter.identity();
            assert_eq!(identity.adapter_id, expected_adapter_id);
            assert_eq!(identity.host_project, expected_project);
            assert_eq!(identity.adapter_version, "1.2.3");
            assert_eq!(identity.telemetry_schema_version, TELEMETRY_SCHEMA_VERSION);
            assert_eq!(identity.redaction_policy_version, "v1");
            assert_eq!(
                identity.runtime_role.as_deref(),
                Some(host.default_runtime_role())
            );
        }
    }

    #[test]
    fn forwarding_host_adapter_routes_events_to_sink() {
        let sink = Arc::new(RecordingSink::default());
        let sink_trait: Arc<dyn AdapterSink> = sink.clone();
        let adapter = ForwardingHostAdapter::for_cass("0.9.0", None).with_sink(sink_trait);

        let fixture = load_fixture("telemetry-search-v1.json");
        adapter.emit_telemetry(&fixture).unwrap();
        adapter
            .on_lifecycle_event(&AdapterLifecycleEvent::SessionStart {
                ts: "now".to_owned(),
            })
            .unwrap();

        {
            let emitted = sink.emitted.lock().unwrap();
            assert_eq!(emitted.len(), 1);
            assert_eq!(emitted[0], fixture);
            drop(emitted);
        }

        {
            let lifecycle = sink.lifecycle.lock().unwrap();
            assert_eq!(lifecycle.len(), 1);
            assert!(matches!(
                lifecycle[0],
                AdapterLifecycleEvent::SessionStart { .. }
            ));
            drop(lifecycle);
        }
    }

    #[test]
    fn forwarding_host_adapter_conformance_passes_for_all_canonical_hosts() {
        let harness = ConformanceHarness::default();
        let fixtures = vec![
            load_fixture("telemetry-search-v1.json"),
            load_fixture("telemetry-embedding-v1.json"),
            load_fixture("telemetry-index-v1.json"),
            load_fixture("telemetry-resource-v1.json"),
            load_fixture("telemetry-lifecycle-v1.json"),
        ];

        for host in CanonicalHostProject::ALL {
            let adapter = ForwardingHostAdapter::for_host(host, "2.0.0", None, None);
            let report = harness.run(&adapter, &fixtures);
            assert!(
                report.passed,
                "host {:?} failed conformance: {:?}",
                host, report.violations
            );
            assert_eq!(report.fixtures_checked, fixtures.len());
            assert_eq!(report.emitted_events, fixtures.len());
            assert!(report.lifecycle_hooks_checked >= 2);
        }
    }

    #[test]
    fn identity_schema_version_mismatch_is_reported() {
        let harness = ConformanceHarness::default();
        let mut identity = default_identity();
        identity.telemetry_schema_version = 9;

        let violations = harness.validate_identity(&identity);
        assert!(violations.iter().any(|violation| {
            violation.code == "adapter.identity.schema_version_mismatch"
                && violation.field == "identity.telemetry_schema_version"
        }));
    }

    #[test]
    fn identity_optional_handshake_fields_require_nonempty_values_when_present() {
        let harness = ConformanceHarness::default();
        let mut identity = default_identity();
        identity.runtime_role = Some("   ".to_owned());
        identity.instance_uuid = Some(String::new());

        let violations = harness.validate_identity(&identity);
        assert!(violations.iter().any(|violation| {
            violation.code == "adapter.identity.empty_runtime_role"
                && violation.field == "identity.runtime_role"
        }));
        assert!(violations.iter().any(|violation| {
            violation.code == "adapter.identity.empty_instance_uuid"
                && violation.field == "identity.instance_uuid"
        }));
    }

    #[test]
    fn redaction_forbidden_pattern_is_detected() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        if let TelemetryEvent::Search { query, .. } = &mut envelope.event {
            query.text = "BEGIN PRIVATE KEY".to_owned();
        } else {
            panic!("fixture shape changed");
        }

        let violations = harness.validate_envelope(&envelope);
        assert!(
            violations
                .iter()
                .any(|violation| violation.code == "adapter.redaction.forbidden_pattern")
        );
    }

    #[test]
    fn harness_reports_adapter_hook_errors_with_diagnostics() {
        let harness = ConformanceHarness::default();
        let mut adapter = MemoryHostAdapter::new(default_identity());
        adapter.fail_emit = true;
        adapter.fail_lifecycle = true;

        let fixtures = vec![load_fixture("telemetry-search-v1.json")];
        let report = harness.run(&adapter, &fixtures);

        assert!(!report.passed);
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.code == "adapter.hook.error")
        );
    }

    #[test]
    fn host_project_attribution_prefers_adapter_identity_hint() {
        let attribution = resolve_host_project_attribution(
            Some("mcp_agent_mail_rust"),
            Some("agent-mail"),
            Some("mail-host-01"),
        );
        assert_eq!(attribution.resolved_project_key, "mcp_agent_mail_rust");
        assert_eq!(attribution.reason_code, "attribution.adapter_identity");
        assert!(!attribution.collision);
        assert!(attribution.confidence_score >= 90);
    }

    #[test]
    fn host_project_attribution_falls_back_to_unknown_bucket() {
        let attribution =
            resolve_host_project_attribution(None, Some("custom-app"), Some("odd-host-name"));
        assert_eq!(attribution.resolved_project_key, "unknown");
        assert_eq!(attribution.reason_code, "attribution.unknown");
        assert_eq!(attribution.confidence_score, 20);
        assert!(!attribution.collision);
    }

    #[test]
    fn host_project_attribution_flags_collisions() {
        let attribution =
            resolve_host_project_attribution(Some("cass"), Some("xf"), Some("mixed-host"));
        assert_eq!(
            attribution.resolved_project_key,
            "coding_agent_session_search"
        );
        assert_eq!(attribution.reason_code, "attribution.collision");
        assert!(attribution.collision);
        assert!(attribution.confidence_score < 95);
    }

    #[test]
    fn host_project_attribution_uses_host_name_hint_when_needed() {
        let attribution =
            resolve_host_project_attribution(None, None, Some("frankenterm-prod-runner"));
        assert_eq!(attribution.resolved_project_key, "frankenterm");
        assert_eq!(attribution.reason_code, "attribution.host_name");
        assert!(!attribution.collision);
    }

    // ── is_valid_ulid edge cases ────────────────────────────────────────

    #[test]
    fn ulid_valid_26_char_crockford() {
        assert!(is_valid_ulid("01JAH9A2W8F8Q6GQ4C7M3N2P1R"));
    }

    #[test]
    fn ulid_empty_string() {
        assert!(!is_valid_ulid(""));
    }

    #[test]
    fn ulid_too_short() {
        assert!(!is_valid_ulid("01JAH9A2W8F8Q6GQ4C7M3N2P1"));
    }

    #[test]
    fn ulid_too_long() {
        assert!(!is_valid_ulid("01JAH9A2W8F8Q6GQ4C7M3N2P1RX"));
    }

    #[test]
    fn ulid_invalid_chars_i_l_o_u() {
        // Crockford base32 excludes I, L, O, U
        assert!(!is_valid_ulid("01JAH9A2W8F8Q6GQ4C7M3N2PIR")); // I at pos 24
        assert!(!is_valid_ulid("01JAH9A2W8F8Q6GQ4C7M3N2PLR")); // L at pos 24
        assert!(!is_valid_ulid("01JAH9A2W8F8Q6GQ4C7M3N2POR")); // O at pos 24
        assert!(!is_valid_ulid("01JAH9A2W8F8Q6GQ4C7M3N2PUR")); // U at pos 24
    }

    #[test]
    fn ulid_lowercase_valid() {
        assert!(is_valid_ulid("01jah9a2w8f8q6gq4c7m3n2p1r"));
    }

    // ── normalize_project_hint edge cases ───────────────────────────────

    #[test]
    fn normalize_hint_strips_special_chars() {
        assert_eq!(normalize_project_hint("Hello-World!"), "hello_world");
    }

    #[test]
    fn normalize_hint_empty_input() {
        assert_eq!(normalize_project_hint(""), "");
    }

    #[test]
    fn normalize_hint_only_special_chars() {
        assert_eq!(normalize_project_hint("---"), "");
    }

    #[test]
    fn normalize_hint_preserves_alphanumeric() {
        assert_eq!(normalize_project_hint("abc123"), "abc123");
    }

    #[test]
    fn normalize_hint_collapses_consecutive_separators() {
        assert_eq!(normalize_project_hint("a--b__c"), "a_b_c");
    }

    #[test]
    fn normalize_hint_no_trailing_separator() {
        let result = normalize_project_hint("hello-");
        assert!(!result.ends_with('_'), "no trailing separator: {result}");
    }

    // ── canonical_projects_from_hint coverage ───────────────────────────

    #[test]
    fn hint_resolves_cass_aliases() {
        for alias in [
            "coding_agent_session_search",
            "codingagentsessionsearch",
            "cass",
            "CASS",
        ] {
            let matches = canonical_projects_from_hint(alias);
            assert!(
                matches.contains(&"coding_agent_session_search"),
                "alias '{alias}' should resolve to cass"
            );
        }
    }

    #[test]
    fn hint_resolves_xf() {
        let matches = canonical_projects_from_hint("xf");
        assert!(matches.contains(&"xf"));
    }

    #[test]
    fn hint_resolves_mcp_agent_mail_aliases() {
        for alias in [
            "mcp_agent_mail_rust",
            "mcpagentmailrust",
            "mcpagentmail",
            "agent_mail",
            "agentmail",
            "amail",
        ] {
            let matches = canonical_projects_from_hint(alias);
            assert!(
                matches.contains(&"mcp_agent_mail_rust"),
                "alias '{alias}' should resolve to mcp_agent_mail_rust"
            );
        }
    }

    #[test]
    fn hint_resolves_frankenterm() {
        let matches = canonical_projects_from_hint("frankenterm");
        assert!(matches.contains(&"frankenterm"));
    }

    #[test]
    fn hint_unknown_returns_empty() {
        let matches = canonical_projects_from_hint("totally_unknown_project");
        assert!(matches.is_empty());
    }

    #[test]
    fn hint_empty_returns_empty() {
        let matches = canonical_projects_from_hint("");
        assert!(matches.is_empty());
    }

    // ── HostProjectAttribution::unknown defaults ────────────────────────

    #[test]
    fn attribution_unknown_has_expected_defaults() {
        let attr = HostProjectAttribution::unknown("test.reason");
        assert_eq!(attr.resolved_project_key, "unknown");
        assert_eq!(attr.confidence_score, 20);
        assert_eq!(attr.reason_code, "test.reason");
        assert!(!attr.collision);
    }

    // ── attribution with all hints None ─────────────────────────────────

    #[test]
    fn attribution_all_none_returns_unknown() {
        let attr = resolve_host_project_attribution(None, None, None);
        assert_eq!(attr.resolved_project_key, "unknown");
        assert_eq!(attr.reason_code, "attribution.unknown");
    }

    // ── AdapterIdentity serde roundtrip ─────────────────────────────────

    #[test]
    fn adapter_identity_serde_roundtrip() {
        let identity = default_identity();
        let json = serde_json::to_string(&identity).unwrap();
        let decoded: AdapterIdentity = serde_json::from_str(&json).unwrap();
        assert_eq!(identity, decoded);
    }

    #[test]
    fn adapter_identity_without_optional_fields_roundtrip() {
        let identity = AdapterIdentity {
            adapter_id: "test".to_owned(),
            adapter_version: "0.1.0".to_owned(),
            host_project: "test-host".to_owned(),
            runtime_role: None,
            instance_uuid: None,
            telemetry_schema_version: TELEMETRY_SCHEMA_VERSION,
            redaction_policy_version: "v1".to_owned(),
        };
        let json = serde_json::to_string(&identity).unwrap();
        let decoded: AdapterIdentity = serde_json::from_str(&json).unwrap();
        assert_eq!(identity, decoded);
    }

    // ── ConformanceViolation serde roundtrip ─────────────────────────────

    #[test]
    fn conformance_violation_serde_roundtrip() {
        let v = ConformanceViolation {
            code: "test.code".to_owned(),
            field: "test.field".to_owned(),
            message: "test message".to_owned(),
        };
        let json = serde_json::to_string(&v).unwrap();
        let decoded: ConformanceViolation = serde_json::from_str(&json).unwrap();
        assert_eq!(v, decoded);
    }

    // ── ConformanceReport serde roundtrip and passed logic ───────────────

    #[test]
    fn conformance_report_passed_when_no_violations() {
        let report = ConformanceReport::with_violations(5, 5, 3, Vec::new());
        assert!(report.passed);
        assert_eq!(report.fixtures_checked, 5);
        assert_eq!(report.emitted_events, 5);
        assert_eq!(report.lifecycle_hooks_checked, 3);
    }

    #[test]
    fn conformance_report_failed_when_violations_present() {
        let violations = vec![ConformanceViolation {
            code: "test".to_owned(),
            field: "f".to_owned(),
            message: "m".to_owned(),
        }];
        let report = ConformanceReport::with_violations(1, 1, 1, violations);
        assert!(!report.passed);
        assert_eq!(report.violations.len(), 1);
    }

    #[test]
    fn conformance_report_serde_roundtrip() {
        let report = ConformanceReport::with_violations(3, 3, 2, Vec::new());
        let json = serde_json::to_string(&report).unwrap();
        let decoded: ConformanceReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report, decoded);
    }

    // ── ConformanceConfig defaults ──────────────────────────────────────

    #[test]
    fn conformance_config_default_values() {
        let config = ConformanceConfig::default();
        assert_eq!(config.expected_schema_version, TELEMETRY_SCHEMA_VERSION);
        assert_eq!(config.required_redaction_policy_version, "v1");
        assert_eq!(
            config.forbidden_substrings.len(),
            DEFAULT_REDACTION_FORBIDDEN_PATTERNS.len()
        );
    }

    // ── NoopAdapterSink ─────────────────────────────────────────────────

    #[test]
    fn noop_adapter_sink_accepts_all_operations() {
        let sink = NoopAdapterSink;
        let fixture = load_fixture("telemetry-search-v1.json");
        sink.emit(&fixture).expect("noop emit should succeed");
        sink.on_lifecycle_event(&AdapterLifecycleEvent::HealthTick {
            ts: "test".to_owned(),
        })
        .expect("noop lifecycle should succeed");
    }

    // ── validate_identity missing required fields ────────────────────────

    #[test]
    fn validate_identity_missing_adapter_id() {
        let harness = ConformanceHarness::default();
        let mut identity = default_identity();
        identity.adapter_id = String::new();
        let violations = harness.validate_identity(&identity);
        assert!(
            violations
                .iter()
                .any(|v| v.code == "adapter.identity.missing_adapter_id")
        );
    }

    #[test]
    fn validate_identity_missing_adapter_version() {
        let harness = ConformanceHarness::default();
        let mut identity = default_identity();
        identity.adapter_version = "  ".to_owned();
        let violations = harness.validate_identity(&identity);
        assert!(
            violations
                .iter()
                .any(|v| v.code == "adapter.identity.missing_adapter_version")
        );
    }

    #[test]
    fn validate_identity_missing_host_project() {
        let harness = ConformanceHarness::default();
        let mut identity = default_identity();
        identity.host_project = String::new();
        let violations = harness.validate_identity(&identity);
        assert!(
            violations
                .iter()
                .any(|v| v.code == "adapter.identity.missing_host_project")
        );
    }

    #[test]
    fn validate_identity_redaction_policy_mismatch() {
        let harness = ConformanceHarness::default();
        let mut identity = default_identity();
        identity.redaction_policy_version = "v99".to_owned();
        let violations = harness.validate_identity(&identity);
        assert!(
            violations
                .iter()
                .any(|v| v.code == "adapter.identity.redaction_policy_mismatch")
        );
    }

    // ── validate_envelope edge cases ─────────────────────────────────────

    #[test]
    fn validate_envelope_wrong_schema_version() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");
        envelope.v = 99;
        let violations = harness.validate_envelope(&envelope);
        assert!(
            violations
                .iter()
                .any(|v| v.code == "adapter.envelope.schema_version_mismatch")
        );
    }

    #[test]
    fn validate_envelope_empty_timestamp() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");
        envelope.ts = "  ".to_owned();
        let violations = harness.validate_envelope(&envelope);
        assert!(
            violations
                .iter()
                .any(|v| v.code == "adapter.envelope.missing_timestamp")
        );
    }

    // ── CanonicalHostProject::ALL ───────────────────────────────────────

    #[test]
    fn canonical_host_project_all_has_four_entries() {
        assert_eq!(CanonicalHostProject::ALL.len(), 4);
    }

    #[test]
    fn canonical_host_project_keys_are_unique() {
        let keys: Vec<&str> = CanonicalHostProject::ALL
            .iter()
            .map(|h| h.host_project_key())
            .collect();
        let unique: std::collections::HashSet<&str> = keys.iter().copied().collect();
        assert_eq!(keys.len(), unique.len(), "host project keys must be unique");
    }

    #[test]
    fn canonical_host_project_adapter_ids_are_unique() {
        let ids: Vec<&str> = CanonicalHostProject::ALL
            .iter()
            .map(|h| h.adapter_id())
            .collect();
        let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "adapter IDs must be unique");
    }

    #[test]
    fn canonical_host_project_runtime_roles_are_nonempty() {
        for host in CanonicalHostProject::ALL {
            let role = host.default_runtime_role();
            assert!(!role.is_empty(), "{host:?} has empty default_runtime_role");
        }
    }

    // ── AdapterLifecycleEvent serde roundtrip ────────────────────────────

    #[test]
    fn lifecycle_event_serde_roundtrip() {
        for event in [
            AdapterLifecycleEvent::SessionStart {
                ts: "t1".to_owned(),
            },
            AdapterLifecycleEvent::SessionStop {
                ts: "t2".to_owned(),
            },
            AdapterLifecycleEvent::HealthTick {
                ts: "t3".to_owned(),
            },
        ] {
            let json = serde_json::to_string(&event).unwrap();
            let decoded: AdapterLifecycleEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(event, decoded);
        }
    }

    // ── CanonicalHostProject serde roundtrip ─────────────────────────────

    #[test]
    fn canonical_host_project_serde_roundtrip() {
        for host in CanonicalHostProject::ALL {
            let json = serde_json::to_string(&host).unwrap();
            let decoded: CanonicalHostProject = serde_json::from_str(&json).unwrap();
            assert_eq!(host, decoded);
        }
    }

    // ── HostProjectAttribution serde roundtrip ──────────────────────────

    #[test]
    fn host_project_attribution_serde_roundtrip() {
        let attr = HostProjectAttribution {
            resolved_project_key: "xf".to_owned(),
            confidence_score: 85,
            reason_code: "attribution.telemetry_project_key".to_owned(),
            collision: false,
        };
        let json = serde_json::to_string(&attr).unwrap();
        let decoded: HostProjectAttribution = serde_json::from_str(&json).unwrap();
        assert_eq!(attr, decoded);
    }

    // ── ForwardingHostAdapter identity_ref ───────────────────────────────

    #[test]
    fn forwarding_adapter_identity_ref_matches_identity() {
        let adapter = ForwardingHostAdapter::for_xf("1.0.0", None);
        let ref_identity = adapter.identity_ref();
        let cloned_identity = adapter.identity();
        assert_eq!(*ref_identity, cloned_identity);
    }

    // ── DEFAULT_REDACTION_FORBIDDEN_PATTERNS ─────────────────────────────

    #[test]
    fn default_forbidden_patterns_are_nonempty() {
        assert!(!DEFAULT_REDACTION_FORBIDDEN_PATTERNS.is_empty());
        for pattern in DEFAULT_REDACTION_FORBIDDEN_PATTERNS {
            assert!(!pattern.is_empty(), "forbidden pattern must be nonempty");
        }
    }

    // ── ForwardingHostAdapter Debug impl ─────────────────────────────────

    #[test]
    fn forwarding_adapter_debug_works() {
        let adapter = ForwardingHostAdapter::for_cass("0.1.0", None);
        let debug = format!("{adapter:?}");
        assert!(debug.contains("ForwardingHostAdapter"));
        assert!(debug.contains("cass-host-adapter"));
    }

    // ── MCP Agent Mail has control-plane role ────────────────────────────

    #[test]
    fn mcp_agent_mail_default_role_is_control_plane() {
        assert_eq!(
            CanonicalHostProject::McpAgentMailRust.default_runtime_role(),
            "control-plane"
        );
    }

    #[test]
    fn non_mcp_hosts_default_role_is_query() {
        for host in [
            CanonicalHostProject::CodingAgentSessionSearch,
            CanonicalHostProject::Xf,
            CanonicalHostProject::Frankenterm,
        ] {
            assert_eq!(
                host.default_runtime_role(),
                "query",
                "{host:?} should default to query role"
            );
        }
    }
}
