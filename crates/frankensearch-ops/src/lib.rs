//! Operations control-plane TUI for frankensearch fleet monitoring.
//!
//! This crate builds on the shared [`frankensearch_tui`] framework to provide
//! an operations console that discovers running frankensearch instances,
//! displays real-time metrics, and provides fleet-wide observability.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  frankensearch-ops (this crate)                 │
//! │  ├─ app: OpsApp entry point and event loop      │
//! │  ├─ data_source: DataSource trait + mock impl   │
//! │  ├─ state: Shared AppState for async bridge     │
//! │  ├─ screens: Fleet, Search, Index, Resource     │
//! │  └─ category: Screen groupings for navigation   │
//! ├─────────────────────────────────────────────────┤
//! │  frankensearch-tui (shared framework)           │
//! │  Screen, ScreenRegistry, AppShell, Keymap, ...  │
//! ├─────────────────────────────────────────────────┤
//! │  ratatui + crossterm                            │
//! └─────────────────────────────────────────────────┘
//! ```

#![forbid(unsafe_code)]

pub mod app;
pub mod category;
pub mod data_source;
pub mod screens;
pub mod state;

// ─── Re-exports ─────────────────────────────────────────────────────────────

pub use app::OpsApp;
pub use category::ScreenCategory;
pub use data_source::{DataSource, MockDataSource};
pub use state::AppState;
