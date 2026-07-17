//! Stable operational statistics surfaced by Keeper snapshots.

/// Point-in-time Quill segment and footprint statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SegmentStats {
    /// Schema used by all segments in the snapshot.
    pub schema_id: u64,
    /// Published manifest generation.
    pub published_generation: u64,
    /// Number of immutable FSLX segments.
    pub sealed_segments: usize,
    /// Number of currently visible in-memory delta segments.
    pub delta_segments: usize,
    /// Number of live documents visible to readers.
    pub live_docs: usize,
    /// Number of manifest tombstones not yet folded by compaction.
    pub tombstones: usize,
    /// Bytes occupied by every Quill-managed disk artifact.
    ///
    /// Includes FSLX segments, manifests, repair sidecars, and lifecycle
    /// metadata so callers do not need a separate filesystem walk.
    pub managed_disk_bytes: u64,
    /// Bytes occupied by searchable delta structures.
    pub delta_memory_bytes: u64,
    /// Wall-clock publish time from the current manifest, when available.
    pub last_publish_unix: Option<i64>,
    /// Whether this process currently owns a live writer.
    pub live_writer: bool,
}

impl SegmentStats {
    /// Total sealed and delta segments in the snapshot.
    #[must_use]
    pub const fn total_segments(self) -> usize {
        self.sealed_segments.saturating_add(self.delta_segments)
    }

    /// Fraction of known documents represented by tombstones.
    #[must_use]
    pub fn tombstone_density(self) -> f64 {
        let total = self.live_docs.saturating_add(self.tombstones);
        if total == 0 {
            0.0
        } else {
            self.tombstones as f64 / total as f64
        }
    }
}

/// Implemented by an index handle once Keeper owns a published snapshot.
pub trait SegmentStatsProvider {
    /// Read lock-free point-in-time segment statistics.
    fn segment_stats(&self) -> SegmentStats;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_cover_empty_and_nonempty_snapshots() {
        assert_eq!(SegmentStats::default().total_segments(), 0);
        assert!(SegmentStats::default().tombstone_density().abs() < f64::EPSILON);

        let stats = SegmentStats {
            sealed_segments: 3,
            delta_segments: 2,
            live_docs: 75,
            tombstones: 25,
            ..SegmentStats::default()
        };
        assert_eq!(stats.total_segments(), 5);
        assert!((stats.tombstone_density() - 0.25).abs() < f64::EPSILON);
    }
}
