//! Keeper lifecycle and durability.
//!
//! Manifest publication, mmap-backed snapshots, recovery, merge, compaction,
//! and writer ownership land in the Quill E3 milestones.

pub use crate::stats::{SegmentStats, SegmentStatsProvider};
