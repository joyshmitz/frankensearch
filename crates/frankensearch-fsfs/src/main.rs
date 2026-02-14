use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use asupersync::Cx;
use asupersync::runtime::RuntimeBuilder;
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_fsfs::{
    CliCommand, FsfsRuntime, InterfaceMode, ShutdownCoordinator, ShutdownReason, Verbosity,
    default_project_config_file_path, default_user_config_file_path, detect_auto_mode,
    emit_config_loaded, exit_code, init_subscriber, load_from_layered_sources, load_from_sources,
    parse_cli_args,
};
use tracing::info;

#[allow(clippy::too_many_lines)]
fn main() -> SearchResult<()> {
    let cli_input = parse_cli_args(std::env::args().skip(1))?;

    // Version is handled immediately, before config loading.
    if cli_input.command == CliCommand::Version {
        println!(
            "fsfs {} (frankensearch {})",
            env!("CARGO_PKG_VERSION"),
            env!("CARGO_PKG_VERSION"),
        );
        std::process::exit(exit_code::OK);
    }

    // Initialize tracing subscriber before anything else that emits events.
    // CLI flags are already parsed, so we can derive verbosity.
    let verbosity = Verbosity::from_flags(cli_input.verbose, cli_input.quiet);
    init_subscriber(verbosity, cli_input.no_color);

    let env_map: HashMap<String, String> = std::env::vars().collect();

    let home_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
    let loaded = if let Some(path) = cli_input.overrides.config_path.as_deref() {
        let config_path = expand_cli_config_path(path, &home_dir);
        if !config_path.exists() {
            return Err(SearchError::InvalidConfig {
                field: "config_file".to_owned(),
                value: config_path.display().to_string(),
                reason: "explicitly provided --config path does not exist".to_owned(),
            });
        }
        load_from_sources(
            Some(config_path.as_path()),
            &env_map,
            &cli_input.overrides,
            &home_dir,
        )?
    } else {
        let cwd = std::env::current_dir().map_err(SearchError::Io)?;
        let project_config_path = default_project_config_file_path(&cwd);
        let user_config_path = default_user_config_file_path(&home_dir);
        load_from_layered_sources(
            Some(project_config_path.as_path()),
            Some(user_config_path.as_path()),
            &env_map,
            &cli_input.overrides,
            &home_dir,
        )?
    };

    let event = loaded.to_loaded_event();
    emit_config_loaded(&event);

    let is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let Some(command) = detect_auto_mode(cli_input.command, is_tty, cli_input.command_source)
    else {
        eprintln!("usage: fsfs <command> [flags]");
        eprintln!("commands: {}", CliCommand::ALL_NAMES.join(", "));
        std::process::exit(exit_code::USAGE_ERROR);
    };

    let mut resolved_config = loaded.config;
    if cli_input.watch {
        resolved_config.indexing.watch_mode = true;
    }

    let mut runtime_cli_input = cli_input;
    runtime_cli_input.command = command;
    let app_runtime = FsfsRuntime::new(resolved_config).with_cli_input(runtime_cli_input);
    let interface_mode = match command {
        CliCommand::Tui => InterfaceMode::Tui,
        CliCommand::Search
        | CliCommand::Index
        | CliCommand::Watch
        | CliCommand::Status
        | CliCommand::Explain
        | CliCommand::Config
        | CliCommand::Download
        | CliCommand::Doctor
        | CliCommand::Update
        | CliCommand::Completions
        | CliCommand::Uninstall
        | CliCommand::Help
        | CliCommand::Version => InterfaceMode::Cli,
    };

    info!(
        command = ?command,
        interface_mode = ?interface_mode,
        pressure_profile = ?app_runtime.config().pressure.profile,
        "fsfs command parsed and runtime wired"
    );

    let scheduler =
        RuntimeBuilder::current_thread()
            .build()
            .map_err(|error| SearchError::SubsystemError {
                subsystem: "fsfs",
                source: Box::new(io::Error::other(format!(
                    "failed to initialize asupersync runtime: {error}"
                ))),
            })?;
    let shutdown = Arc::new(ShutdownCoordinator::new());
    shutdown.register_signals()?;
    let cx = Cx::for_request();
    let run_with_shutdown =
        matches!(interface_mode, InterfaceMode::Tui) || app_runtime.config().indexing.watch_mode;
    let shutdown_for_run = Arc::clone(&shutdown);

    let run_result = scheduler.block_on(async move {
        let run_result = if run_with_shutdown {
            app_runtime
                .run_mode_with_shutdown(&cx, interface_mode, shutdown_for_run.as_ref())
                .await
        } else {
            app_runtime.run_mode(&cx, interface_mode).await
        };

        if let Err(error) = &run_result {
            shutdown_for_run.request_shutdown(ShutdownReason::Error(error.to_string()));
        }

        run_result
    });
    shutdown.stop_signal_listener();

    if shutdown.is_force_exit_requested() {
        std::process::exit(exit_code::INTERRUPTED);
    }

    if let Some(reason) = shutdown.current_reason()
        && shutdown.is_shutting_down()
    {
        info!(reason = ?reason, "fsfs shutdown completed at process boundary");
    }

    run_result
}

fn expand_cli_config_path(path: &std::path::Path, home_dir: &std::path::Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        home_dir.to_path_buf()
    } else if let Some(rest) = raw.strip_prefix("~/") {
        home_dir.join(rest)
    } else {
        path.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::expand_cli_config_path;
    use std::path::{Path, PathBuf};

    #[test]
    fn expand_cli_config_path_expands_tilde_prefix() {
        let expanded =
            expand_cli_config_path(Path::new("~/cfg/fsfs.toml"), Path::new("/home/alex"));
        assert_eq!(expanded, PathBuf::from("/home/alex/cfg/fsfs.toml"));
    }

    #[test]
    fn expand_cli_config_path_keeps_absolute_paths() {
        let expanded = expand_cli_config_path(Path::new("/tmp/fsfs.toml"), Path::new("/home/alex"));
        assert_eq!(expanded, PathBuf::from("/tmp/fsfs.toml"));
    }

    #[test]
    fn expand_cli_config_path_bare_tilde() {
        let expanded = expand_cli_config_path(Path::new("~"), Path::new("/home/alex"));
        assert_eq!(expanded, PathBuf::from("/home/alex"));
    }

    #[test]
    fn expand_cli_config_path_keeps_relative_paths() {
        let expanded =
            expand_cli_config_path(Path::new("relative/config.toml"), Path::new("/home/alex"));
        assert_eq!(expanded, PathBuf::from("relative/config.toml"));
    }
}
