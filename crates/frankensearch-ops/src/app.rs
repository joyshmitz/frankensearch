//! Ops TUI application entry point.
//!
//! [`OpsApp`] assembles the app shell, registers screens, wires up the
//! data source, and drives the event loop. Product binaries create an
//! `OpsApp` and call `run()`.

use frankensearch_tui::palette::{Action, ActionCategory};
use frankensearch_tui::shell::{AppShell, ShellConfig};
use frankensearch_tui::theme::Theme;
use frankensearch_tui::Screen;

use crate::data_source::DataSource;
use crate::preferences::DisplayPreferences;
use crate::screens::FleetOverviewScreen;
use crate::state::AppState;

// ─── Ops App ─────────────────────────────────────────────────────────────────

/// Ops control-plane TUI application.
///
/// Wires the shared TUI framework with ops-specific screens and data sources.
pub struct OpsApp {
    /// The app shell (owns registry, keymap, overlays).
    pub shell: AppShell,
    /// Shared application state.
    pub state: AppState,
    /// Data source for populating state.
    data_source: Box<dyn DataSource>,
    /// Fleet screen (kept outside registry for direct state updates).
    fleet_screen: FleetOverviewScreen,
    /// Accessibility and display preferences.
    pub preferences: DisplayPreferences,
}

impl OpsApp {
    /// Create a new ops app with the given data source.
    #[must_use]
    pub fn new(data_source: Box<dyn DataSource>) -> Self {
        let config = ShellConfig {
            title: "frankensearch ops".to_string(),
            theme: Theme::dark(),
            show_status_bar: true,
            show_breadcrumbs: true,
        };

        let mut shell = AppShell::new(config);

        // Register palette actions.
        for action in Self::palette_actions() {
            shell.palette.register(action);
        }

        // Create and register the fleet screen.
        let fleet_screen = FleetOverviewScreen::new();
        let fleet_id = fleet_screen.id().clone();
        shell.registry.register(Box::new(FleetOverviewScreen::new()));

        // Set initial active screen.
        shell.navigate_to(&fleet_id);

        Self {
            shell,
            state: AppState::new(),
            data_source,
            fleet_screen,
            preferences: DisplayPreferences::new(),
        }
    }

    /// Refresh state from the data source.
    pub fn refresh_data(&mut self) {
        let snapshot = self.data_source.fleet_snapshot();
        self.state.update_fleet(snapshot);

        // Update status line with connection info.
        self.shell.status_line = self
            .shell
            .status_line
            .clone()
            .with_center(self.state.connection_status().to_string());

        // Update the fleet screen's state.
        self.fleet_screen.update_state(&self.state);
    }

    /// Process an input event. Returns `true` if the app should quit.
    pub fn handle_input(&mut self, event: &frankensearch_tui::InputEvent) -> bool {
        let quit = self.shell.handle_input(event);

        // Handle palette action confirmations.
        if let Some(action_id) = self.shell.last_palette_action() {
            self.dispatch_palette_action(action_id);
        }

        quit
    }

    /// Dispatch a confirmed palette action by ID.
    fn dispatch_palette_action(&mut self, action_id: &str) {
        match action_id {
            "nav.fleet" => {
                let id = frankensearch_tui::screen::ScreenId::new("ops.fleet");
                self.shell.navigate_to(&id);
            }
            "debug.refresh" => {
                self.refresh_data();
            }
            "settings.theme" => {
                self.shell.config.theme = if self.shell.config.theme == Theme::dark() {
                    Theme::light()
                } else {
                    Theme::dark()
                };
            }
            "settings.contrast" => {
                self.preferences.toggle_contrast();
            }
            "settings.motion" => {
                self.preferences.toggle_motion();
            }
            "settings.focus" => {
                self.preferences.toggle_focus_visibility();
            }
            "settings.hints" => {
                self.preferences.toggle_shortcut_hints();
            }
            _ => {}
        }
    }

    /// Whether the app should quit.
    #[must_use]
    pub const fn should_quit(&self) -> bool {
        self.shell.should_quit
    }

    /// Get a reference to the fleet screen (for testing/inspection).
    #[must_use]
    pub const fn fleet_screen(&self) -> &FleetOverviewScreen {
        &self.fleet_screen
    }

    /// Get a list of all registered palette actions (for help screen).
    #[must_use]
    pub fn palette_actions() -> Vec<Action> {
        vec![
            Action::new("nav.fleet", "Go to Fleet Overview", ActionCategory::Navigation)
                .with_shortcut("1"),
            Action::new("nav.search", "Go to Search Stream", ActionCategory::Navigation)
                .with_shortcut("2"),
            Action::new("nav.index", "Go to Index Status", ActionCategory::Navigation)
                .with_shortcut("3"),
            Action::new(
                "debug.refresh",
                "Force Refresh Data",
                ActionCategory::Debug,
            )
            .with_shortcut("F5"),
            Action::new(
                "settings.theme",
                "Toggle Theme",
                ActionCategory::Settings,
            )
            .with_shortcut("T")
            .with_description("Switch between dark and light theme"),
            Action::new(
                "settings.contrast",
                "Toggle High Contrast",
                ActionCategory::Settings,
            )
            .with_description("Switch between normal and high contrast mode"),
            Action::new(
                "settings.motion",
                "Toggle Reduced Motion",
                ActionCategory::Settings,
            )
            .with_description("Enable or disable non-essential animations"),
            Action::new(
                "settings.focus",
                "Toggle Focus Visibility",
                ActionCategory::Settings,
            )
            .with_description("Switch between normal and enhanced focus indicators"),
            Action::new(
                "settings.hints",
                "Toggle Shortcut Hints",
                ActionCategory::Settings,
            )
            .with_description("Show or hide inline keyboard shortcut hints"),
        ]
    }

    /// Get the current display preferences.
    #[must_use]
    pub const fn preferences(&self) -> &DisplayPreferences {
        &self.preferences
    }
}

#[cfg(test)]
mod tests {
    use frankensearch_tui::palette::PaletteState;
    use frankensearch_tui::screen::ScreenId;

    use super::*;
    use crate::data_source::MockDataSource;
    use crate::preferences::ContrastMode;

    #[test]
    fn ops_app_creation() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert!(!app.should_quit());
        assert!(app.shell.active_screen.is_some());
    }

    #[test]
    fn ops_app_refresh_data() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        app.refresh_data();
        assert!(app.state.has_data());
        assert_eq!(app.state.fleet().instance_count(), 3);
    }

    #[test]
    fn ops_app_empty_data_source() {
        let mut app = OpsApp::new(Box::new(MockDataSource::empty()));
        app.refresh_data();
        assert!(app.state.has_data());
        assert_eq!(app.state.fleet().instance_count(), 0);
    }

    #[test]
    fn ops_app_quit() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        let event = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = app.handle_input(&event);
        assert!(quit);
        assert!(app.should_quit());
    }

    #[test]
    fn palette_actions_nonempty() {
        let actions = OpsApp::palette_actions();
        assert!(!actions.is_empty());
        assert!(actions.iter().any(|a| a.id == "nav.fleet"));
    }

    #[test]
    fn palette_actions_include_accessibility() {
        let actions = OpsApp::palette_actions();
        assert!(actions.iter().any(|a| a.id == "settings.contrast"));
        assert!(actions.iter().any(|a| a.id == "settings.motion"));
        assert!(actions.iter().any(|a| a.id == "settings.focus"));
        assert!(actions.iter().any(|a| a.id == "settings.hints"));
    }

    #[test]
    fn fleet_screen_accessible() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert_eq!(app.fleet_screen().id(), &ScreenId::new("ops.fleet"));
    }

    #[test]
    fn palette_registered_on_creation() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert!(!app.shell.palette.is_empty());
        assert_eq!(app.shell.palette.len(), OpsApp::palette_actions().len());
    }

    #[test]
    fn palette_toggle_via_ctrl_p() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert_eq!(app.shell.palette.state(), &PaletteState::Closed);

        let event = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char('p'),
            crossterm::event::KeyModifiers::CONTROL,
        );
        app.handle_input(&event);
        assert_eq!(app.shell.palette.state(), &PaletteState::Open);

        // Esc closes.
        let esc = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Esc,
            crossterm::event::KeyModifiers::NONE,
        );
        app.handle_input(&esc);
        assert_eq!(app.shell.palette.state(), &PaletteState::Closed);
    }

    #[test]
    fn palette_text_input() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));

        // Open palette.
        let open = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char(':'),
            crossterm::event::KeyModifiers::NONE,
        );
        app.handle_input(&open);
        assert_eq!(app.shell.palette.state(), &PaletteState::Open);

        // Type "fleet".
        for ch in "fleet".chars() {
            let event = frankensearch_tui::InputEvent::Key(
                crossterm::event::KeyCode::Char(ch),
                crossterm::event::KeyModifiers::NONE,
            );
            app.handle_input(&event);
        }
        assert_eq!(app.shell.palette.query(), "fleet");

        // Filtered results should include fleet action.
        let filtered = app.shell.palette.filtered();
        assert!(filtered.iter().any(|a| a.id == "nav.fleet"));
    }

    #[test]
    fn dispatch_contrast_toggle() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert_eq!(app.preferences.contrast, ContrastMode::Normal);
        app.dispatch_palette_action("settings.contrast");
        assert_eq!(app.preferences.contrast, ContrastMode::High);
    }

    #[test]
    fn dispatch_refresh() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert!(!app.state.has_data());
        app.dispatch_palette_action("debug.refresh");
        assert!(app.state.has_data());
    }

    #[test]
    fn preferences_default() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        let prefs = app.preferences();
        assert_eq!(prefs.contrast, ContrastMode::Normal);
        assert!(prefs.show_shortcut_hints);
    }
}
