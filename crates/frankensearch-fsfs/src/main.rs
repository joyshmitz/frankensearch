use std::collections::HashMap;
use std::path::PathBuf;

use frankensearch_core::SearchResult;
use frankensearch_fsfs::{
    FsfsRuntime, InterfaceMode, default_config_file_path, emit_config_loaded, load_from_sources,
    parse_cli_args,
};
use tracing::info;

fn main() -> SearchResult<()> {
    let cli_input = parse_cli_args(std::env::args().skip(1))?;
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

    let runtime = FsfsRuntime::new(loaded.config);
    let interface_mode = match cli_input.command {
        frankensearch_fsfs::CliCommand::Search
        | frankensearch_fsfs::CliCommand::Index
        | frankensearch_fsfs::CliCommand::Status
        | frankensearch_fsfs::CliCommand::Explain
        | frankensearch_fsfs::CliCommand::Config => InterfaceMode::Cli,
    };

    info!(
        command = ?cli_input.command,
        interface_mode = ?interface_mode,
        pressure_profile = ?runtime.config().pressure.profile,
        "fsfs scaffold command parsed and runtime wired"
    );
    Ok(())
}
