//! Tracing subscriber initialization for the fsfs binary.
//!
//! Wires CLI flags (`--verbose`, `--quiet`, `--no-color`) and environment
//! variables (`FRANKENSEARCH_LOG`, `RUST_LOG`) into a single `tracing-subscriber`
//! stack that writes structured logs to stderr.
//!
//! # Priority (highest to lowest)
//!
//! 1. `FRANKENSEARCH_LOG` env var (per-target directives, e.g. `frankensearch=debug,warn`)
//! 2. `RUST_LOG` env var (standard fallback)
//! 3. CLI flags (`-v` → debug, `-q` → error)
//! 4. Default level: `warn`

use tracing::Level;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

use frankensearch_core::tracing_config;

/// Verbosity level derived from CLI flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    /// `--quiet` / `-q`: only errors.
    Quiet,
    /// Default: warnings and above.
    Normal,
    /// `--verbose` / `-v`: debug-level output.
    Verbose,
}

impl Verbosity {
    /// Determine verbosity from the parsed CLI flags.
    #[must_use]
    pub const fn from_flags(verbose: bool, quiet: bool) -> Self {
        // If both are set, verbose wins (more information is safer for debugging).
        if verbose {
            Self::Verbose
        } else if quiet {
            Self::Quiet
        } else {
            Self::Normal
        }
    }

    /// Map to a default `tracing::Level`.
    #[must_use]
    pub const fn default_level(self) -> Level {
        match self {
            Self::Quiet => Level::ERROR,
            Self::Normal => Level::WARN,
            Self::Verbose => Level::DEBUG,
        }
    }
}

/// Initialize the global tracing subscriber.
///
/// Must be called exactly once, as early as possible in `main()` — before
/// config loading so that config-parsing traces are captured.
///
/// # Arguments
///
/// * `verbosity` — derived from `--verbose` / `--quiet` CLI flags.
/// * `no_color` — if `true`, ANSI escape codes are suppressed regardless of
///   terminal detection.
///
/// # Panics
///
/// Panics if a global subscriber has already been set (double-init).
pub fn init_subscriber(verbosity: Verbosity, no_color: bool) {
    let filter = build_env_filter(verbosity);

    let stderr_is_tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let use_ansi = !no_color && stderr_is_tty;

    let fmt_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(use_ansi)
        .with_target(true)
        .with_thread_ids(false)
        .with_level(true);

    // Verbose mode: full timestamps and span events for diagnostics.
    // Normal/quiet: compact output without timestamps (less noise for end-users).
    if verbosity == Verbosity::Verbose {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer.with_timer(fmt::time::uptime()))
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer.without_time().compact())
            .init();
    }
}

/// Build an `EnvFilter` respecting the priority chain:
/// `FRANKENSEARCH_LOG` > `RUST_LOG` > CLI verbosity default.
fn build_env_filter(verbosity: Verbosity) -> EnvFilter {
    // Try FRANKENSEARCH_LOG first (project-specific).
    // If the value is unparseable, fall through to RUST_LOG / default
    // rather than failing hard.
    if let Ok(directives) = std::env::var("FRANKENSEARCH_LOG")
        && let Ok(filter) = EnvFilter::try_new(&directives)
    {
        return filter;
    }

    // Try RUST_LOG (standard ecosystem convention).
    if let Ok(filter) = EnvFilter::try_from_default_env() {
        return filter;
    }

    // Fall back to CLI-derived level. In verbose mode, elevate frankensearch
    // internals to debug so users see search pipeline activity. In normal and
    // quiet mode, keep output clean — only warnings/errors reach the terminal.
    let level = verbosity.default_level();

    let directive = if verbosity == Verbosity::Verbose {
        format!(
            "{level},{prefix}=debug",
            prefix = tracing_config::TARGET_PREFIX,
        )
    } else {
        level.to_string()
    };

    EnvFilter::try_new(&directive).unwrap_or_else(|_| EnvFilter::new(level.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbosity_from_flags_default() {
        assert_eq!(Verbosity::from_flags(false, false), Verbosity::Normal);
    }

    #[test]
    fn verbosity_from_flags_verbose() {
        assert_eq!(Verbosity::from_flags(true, false), Verbosity::Verbose);
    }

    #[test]
    fn verbosity_from_flags_quiet() {
        assert_eq!(Verbosity::from_flags(false, true), Verbosity::Quiet);
    }

    #[test]
    fn verbosity_from_flags_both_prefers_verbose() {
        assert_eq!(Verbosity::from_flags(true, true), Verbosity::Verbose);
    }

    #[test]
    fn default_level_mapping() {
        assert_eq!(Verbosity::Quiet.default_level(), Level::ERROR);
        assert_eq!(Verbosity::Normal.default_level(), Level::WARN);
        assert_eq!(Verbosity::Verbose.default_level(), Level::DEBUG);
    }

    #[test]
    fn build_env_filter_produces_valid_filter() {
        // Ensure the fallback path doesn't panic.
        let _filter = build_env_filter(Verbosity::Normal);
        let _filter = build_env_filter(Verbosity::Verbose);
        let _filter = build_env_filter(Verbosity::Quiet);
    }

    // Note: init_subscriber can only be called once per process, so we test
    // it only via the e2e harness rather than in unit tests.
}
