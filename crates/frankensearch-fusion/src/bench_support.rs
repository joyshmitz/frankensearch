//! Re-export of the shared bench harness. **Canonical home: [`frankensearch_core::bench_support`]**.
//!
//! The paired-sampler + null-control harness was promoted to `frankensearch-core` so that
//! `frankensearch-index` (which cannot depend on this crate — that would be a cycle) can use the
//! same decidability machinery for the int8-ADC-scan A/B. This module re-exports it so existing
//! `frankensearch_fusion::bench_support::*` bench imports keep working unchanged.

pub use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio};
