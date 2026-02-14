use asupersync::Cx;
use frankensearch_core::SearchResult;
use tracing::info;

use crate::config::FsfsConfig;

/// Supported fsfs interfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceMode {
    Cli,
    Tui,
}

/// Shared runtime entrypoint used by interface adapters.
#[derive(Debug, Clone)]
pub struct FsfsRuntime {
    config: FsfsConfig,
}

impl FsfsRuntime {
    #[must_use]
    pub const fn new(config: FsfsConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub const fn config(&self) -> &FsfsConfig {
        &self.config
    }

    /// Dispatch by interface mode using the caller-provided `Cx`.
    ///
    /// # Errors
    ///
    /// Returns any surfaced `SearchError` from the selected runtime lane.
    pub async fn run_mode(&self, cx: &Cx, mode: InterfaceMode) -> SearchResult<()> {
        match mode {
            InterfaceMode::Cli => self.run_cli(cx).await,
            InterfaceMode::Tui => self.run_tui(cx).await,
        }
    }

    /// CLI runtime lane scaffold.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when downstream CLI runtime logic fails.
    pub async fn run_cli(&self, _cx: &Cx) -> SearchResult<()> {
        std::future::ready(()).await;
        info!(profile = ?self.config.pressure.profile, "fsfs cli runtime scaffold invoked");
        Ok(())
    }

    /// TUI runtime lane scaffold.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when downstream TUI runtime logic fails.
    pub async fn run_tui(&self, _cx: &Cx) -> SearchResult<()> {
        std::future::ready(()).await;
        info!(theme = ?self.config.tui.theme, "fsfs tui runtime scaffold invoked");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use asupersync::test_utils::run_test_with_cx;

    use super::{FsfsRuntime, InterfaceMode};
    use crate::config::FsfsConfig;

    #[test]
    fn runtime_modes_are_callable() {
        run_test_with_cx(|cx| async move {
            let runtime = FsfsRuntime::new(FsfsConfig::default());
            runtime
                .run_mode(&cx, InterfaceMode::Cli)
                .await
                .expect("cli mode");
            runtime
                .run_mode(&cx, InterfaceMode::Tui)
                .await
                .expect("tui mode");
        });
    }
}
