use std::error::Error;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use frankensearch_ops::{
    DataSource, MockDataSource, OpsApp, OpsStorage, OpsStorageConfig, StorageDataSource,
};
use frankensearch_tui::InputEvent;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalGuard {
    fn enter() -> Result<Self, Box<dyn Error>> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            crossterm::event::EnableMouseCapture
        )?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            crossterm::event::DisableMouseCapture
        );
        let _ = self.terminal.show_cursor();
    }
}

const fn map_event(event: &Event) -> Option<InputEvent> {
    match event {
        Event::Key(key) => Some(InputEvent::Key(key.code, key.modifiers)),
        Event::Mouse(mouse) => Some(InputEvent::Mouse(mouse.kind, mouse.column, mouse.row)),
        Event::Resize(width, height) => Some(InputEvent::Resize(*width, *height)),
        Event::FocusGained | Event::FocusLost | Event::Paste(_) => None,
    }
}

#[derive(Debug, Default, Clone)]
struct RuntimeOptions {
    demo_mode: bool,
    db_path: Option<PathBuf>,
}

fn parse_runtime_options() -> Result<RuntimeOptions, Box<dyn Error>> {
    let mut options = RuntimeOptions::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--demo" => {
                options.demo_mode = true;
            }
            "--db-path" => {
                let value = args.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "--db-path requires a filesystem path argument",
                    )
                })?;
                options.db_path = Some(resolve_db_path(&value));
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument: {other}"),
                )
                .into());
            }
        }
    }

    if !options.demo_mode
        && let Ok(value) = std::env::var("FRANKENSEARCH_OPS_DEMO")
    {
        let value = value.trim();
        options.demo_mode = matches!(value, "1" | "true" | "TRUE" | "True");
    }
    if options.db_path.is_none()
        && let Ok(value) = std::env::var("FRANKENSEARCH_OPS_DB_PATH")
    {
        let value = value.trim();
        if !value.is_empty() {
            options.db_path = Some(resolve_db_path(value));
        }
    }

    Ok(options)
}

fn resolve_db_path(raw: &str) -> PathBuf {
    if raw == ":memory:" {
        return PathBuf::from(raw);
    }
    if let Some(rest) = raw.strip_prefix("~/").or_else(|| raw.strip_prefix("~\\")) {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(raw)
}

fn print_help() {
    println!("frankensearch-ops");
    println!();
    println!("Usage:");
    println!("  frankensearch-ops [--db-path <path>] [--demo]");
    println!();
    println!("Flags:");
    println!("  --db-path <path>  Path to ops telemetry database");
    println!("  --demo            Use synthetic mock data source");
    println!("  -h, --help        Show this help message");
    println!();
    println!("Environment:");
    println!("  FRANKENSEARCH_OPS_DB_PATH=<path>");
    println!("  FRANKENSEARCH_OPS_DEMO=true|false");
}

fn build_data_source(options: &RuntimeOptions) -> Result<Box<dyn DataSource>, Box<dyn Error>> {
    if options.demo_mode {
        return Ok(Box::new(MockDataSource::sample()));
    }

    let mut config = OpsStorageConfig::default();
    if let Some(path) = options.db_path.as_ref() {
        config.db_path.clone_from(path);
    }
    if config.db_path.as_os_str() != ":memory:"
        && let Some(parent) = config.db_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let storage = OpsStorage::open(config)?;
    Ok(Box::new(StorageDataSource::new(storage)))
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_runtime_options()?;
    let mut terminal = TerminalGuard::enter()?;
    let mut app = OpsApp::new(build_data_source(&options)?);
    app.refresh_data();

    let refresh_every = Duration::from_millis(500);
    let mut last_refresh = Instant::now();

    loop {
        terminal.terminal.draw(|frame| app.render(frame))?;
        let timeout = refresh_every.saturating_sub(last_refresh.elapsed());

        if event::poll(timeout)? {
            let event = event::read()?;
            if let Some(input) = map_event(&event) {
                let quit = app.handle_input(&input);
                if quit || app.should_quit() {
                    break;
                }
            }
        }

        if last_refresh.elapsed() >= refresh_every {
            app.refresh_data();
            last_refresh = Instant::now();
        }
    }

    Ok(())
}
