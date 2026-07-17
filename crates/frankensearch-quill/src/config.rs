//! Explicit Quill configuration.
//!
//! Quill never reads environment variables. Applications may populate this
//! structure from their own configuration layers, but the engine sees only
//! explicit values.

use frankensearch_core::{SearchError, SearchResult};

/// Default per-shard Scribe arena budget (64 MiB).
pub const DEFAULT_SCRIBE_SHARD_BUDGET_BYTES: usize = 64 * 1024 * 1024;
/// Default searchable delta budget per shard (8 MiB).
pub const DEFAULT_DELTA_BUDGET_BYTES: usize = 8 * 1024 * 1024;
/// Default number of same-tier segments that triggers a merge.
pub const DEFAULT_TIER_FANOUT: usize = 8;
/// Default live-segment tombstone density that triggers compaction.
pub const DEFAULT_COMPACTION_TOMBSTONE_DENSITY: f64 = 0.20;
/// Default maximum hole ratio accepted by the concat-merge policy.
pub const DEFAULT_MERGE_MAX_HOLE_RATIO: f64 = 0.50;
/// Default maximum number of terms produced by one glob expansion.
pub const DEFAULT_GLOB_EXPANSION_LIMIT: usize = 16_384;
/// Default cap on independent ingest shards.
pub const DEFAULT_MAX_INGEST_SHARDS: usize = 32;

/// All engine-level Quill tuning knobs.
///
/// Defaults are deterministic constants, not environment-dependent values.
#[derive(Debug, Clone, PartialEq)]
pub struct QuillConfig {
    /// Maximum bytes in one Scribe shard's accumulation arenas before flush.
    pub scribe_shard_budget_bytes: usize,
    /// Maximum bytes in one shard's searchable delta before seal.
    pub delta_budget_bytes: usize,
    /// Number of segments at one tier that triggers promotion/merge.
    pub tier_fanout: usize,
    /// Per-segment tombstone density that triggers compaction.
    pub compaction_tombstone_density: f64,
    /// Maximum fraction of holes tolerated by concat-merge policy.
    pub merge_max_hole_ratio: f64,
    /// Maximum terms a glob may expand into before returning a typed error.
    pub glob_expansion_limit: usize,
    /// Upper bound on independent ingest shards.
    pub max_ingest_shards: usize,
    /// Force one deterministic shard for replay and conformance runs.
    pub deterministic_ingest: bool,
}

impl Default for QuillConfig {
    fn default() -> Self {
        Self {
            scribe_shard_budget_bytes: DEFAULT_SCRIBE_SHARD_BUDGET_BYTES,
            delta_budget_bytes: DEFAULT_DELTA_BUDGET_BYTES,
            tier_fanout: DEFAULT_TIER_FANOUT,
            compaction_tombstone_density: DEFAULT_COMPACTION_TOMBSTONE_DENSITY,
            merge_max_hole_ratio: DEFAULT_MERGE_MAX_HOLE_RATIO,
            glob_expansion_limit: DEFAULT_GLOB_EXPANSION_LIMIT,
            max_ingest_shards: DEFAULT_MAX_INGEST_SHARDS,
            deterministic_ingest: false,
        }
    }
}

impl QuillConfig {
    /// Validate every knob before the engine allocates resources.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for zero budgets/counts or for
    /// non-finite/out-of-range density values.
    pub fn validate(&self) -> SearchResult<()> {
        require_positive("scribe_shard_budget_bytes", self.scribe_shard_budget_bytes)?;
        require_positive("delta_budget_bytes", self.delta_budget_bytes)?;
        if self.tier_fanout < 2 {
            return Err(invalid_config(
                "tier_fanout",
                &self.tier_fanout,
                "must be at least 2",
            ));
        }
        require_fraction_open_closed(
            "compaction_tombstone_density",
            self.compaction_tombstone_density,
        )?;
        require_fraction_closed("merge_max_hole_ratio", self.merge_max_hole_ratio)?;
        require_positive("glob_expansion_limit", self.glob_expansion_limit)?;
        require_positive("max_ingest_shards", self.max_ingest_shards)?;
        Ok(())
    }

    /// Resolve an externally detected parallelism count to Quill's shard count.
    ///
    /// The caller chooses how parallelism is detected. Deterministic mode always
    /// returns one; otherwise the count is clamped to `1..=max_ingest_shards`.
    #[must_use]
    pub fn resolved_ingest_shards(&self, detected_parallelism: usize) -> usize {
        if self.deterministic_ingest {
            1
        } else {
            detected_parallelism
                .max(1)
                .min(self.max_ingest_shards.max(1))
        }
    }
}

fn invalid_config(field: &str, value: &impl ToString, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_string(),
        reason: reason.to_owned(),
    }
}

fn require_positive(field: &str, value: usize) -> SearchResult<()> {
    if value == 0 {
        Err(invalid_config(field, &value, "must be greater than zero"))
    } else {
        Ok(())
    }
}

fn require_fraction_open_closed(field: &str, value: f64) -> SearchResult<()> {
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(())
    } else {
        Err(invalid_config(
            field,
            &value,
            "must be finite and in (0, 1]",
        ))
    }
}

fn require_fraction_closed(field: &str, value: f64) -> SearchResult<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(invalid_config(
            field,
            &value,
            "must be finite and in [0, 1]",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_pinned() {
        assert_eq!(
            QuillConfig::default(),
            QuillConfig {
                scribe_shard_budget_bytes: 64 * 1024 * 1024,
                delta_budget_bytes: 8 * 1024 * 1024,
                tier_fanout: 8,
                compaction_tombstone_density: 0.20,
                merge_max_hole_ratio: 0.50,
                glob_expansion_limit: 16_384,
                max_ingest_shards: 32,
                deterministic_ingest: false,
            }
        );
        assert!(QuillConfig::default().validate().is_ok());
    }

    #[test]
    fn shard_resolution_clamps_and_honors_determinism() {
        let mut config = QuillConfig::default();
        assert_eq!(config.resolved_ingest_shards(0), 1);
        assert_eq!(config.resolved_ingest_shards(12), 12);
        assert_eq!(config.resolved_ingest_shards(128), 32);
        config.deterministic_ingest = true;
        assert_eq!(config.resolved_ingest_shards(128), 1);
    }

    #[test]
    fn validation_rejects_boundaries_and_non_finite_values() {
        let mut config = QuillConfig {
            delta_budget_bytes: 0,
            ..QuillConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(SearchError::InvalidConfig { field, .. }) if field == "delta_budget_bytes"
        ));

        config = QuillConfig::default();
        config.compaction_tombstone_density = f64::NAN;
        assert!(config.validate().is_err());
        config.compaction_tombstone_density = 0.0;
        assert!(config.validate().is_err());
        config.compaction_tombstone_density = 1.0;
        assert!(config.validate().is_ok());

        config.merge_max_hole_ratio = 0.0;
        assert!(config.validate().is_ok());
        config.merge_max_hole_ratio = 1.01;
        assert!(config.validate().is_err());
    }
}
