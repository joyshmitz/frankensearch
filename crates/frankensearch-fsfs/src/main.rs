use std::collections::HashMap;
use std::path::PathBuf;

use frankensearch_core::SearchResult;
use frankensearch_fsfs::{
    CliCommand, FsfsRuntime, InterfaceMode, default_config_file_path, detect_auto_mode,
    emit_config_loaded, exit_code, load_from_sources, parse_cli_args,
};
use tracing::info;

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

    let env_map: HashMap<String, String> = std::env::vars().collect();

    let home_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
    let config_path = cli_input
        .overrides
        .config_path
        .clone()
        .unwrap_or_else(|| default_config_file_path(&home_dir));

    let loaded = load_from_sources(
        Some(config_path.as_path()),
        &env_map,
        &cli_input.overrides,
        &home_dir,
    )?;

    let event = loaded.to_loaded_event();
    emit_config_loaded(&event);

    let is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let Some(command) = detect_auto_mode(cli_input.command, is_tty, cli_input.command_explicit)
    else {
        eprintln!("usage: fsfs <command> [flags]");
        eprintln!("commands: {}", CliCommand::ALL_NAMES.join(", "));
        std::process::exit(exit_code::USAGE_ERROR);
    };

    let runtime = FsfsRuntime::new(loaded.config);
    let interface_mode = match command {
        CliCommand::Tui => InterfaceMode::Tui,
        CliCommand::Search
        | CliCommand::Index
        | CliCommand::Status
        | CliCommand::Explain
        | CliCommand::Config
        | CliCommand::Download
        | CliCommand::Doctor
        | CliCommand::Version => InterfaceMode::Cli,
    };

    info!(
        command = ?command,
        interface_mode = ?interface_mode,
        pressure_profile = ?runtime.config().pressure.profile,
        "fsfs command parsed and runtime wired"
    );
    Ok(())
}
