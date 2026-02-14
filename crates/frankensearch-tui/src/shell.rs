//! App shell: status bar, breadcrumbs, screen lifecycle.
//!
//! The [`AppShell`] owns the [`ScreenRegistry`], manages navigation between
//! screens, renders the chrome (status bar, breadcrumbs), and dispatches
//! input events to the active screen.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs};
use ratatui::Frame;
use serde::{Deserialize, Serialize};

use crate::input::{InputEvent, KeyAction, Keymap};
use crate::overlay::OverlayManager;
use crate::screen::{ScreenAction, ScreenContext, ScreenId, ScreenRegistry};
use crate::theme::Theme;

// ─── Shell Config ────────────────────────────────────────────────────────────

/// Configuration for the app shell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellConfig {
    /// Application title shown in the status bar.
    pub title: String,
    /// Theme preset to use.
    pub theme: Theme,
    /// Whether to show the status bar.
    pub show_status_bar: bool,
    /// Whether to show breadcrumbs (tab bar).
    pub show_breadcrumbs: bool,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            title: "frankensearch".to_string(),
            theme: Theme::dark(),
            show_status_bar: true,
            show_breadcrumbs: true,
        }
    }
}

// ─── Status Line ─────────────────────────────────────────────────────────────

/// Status line content rendered at the bottom of the shell.
#[derive(Debug, Clone, Default)]
pub struct StatusLine {
    /// Left-aligned status text.
    pub left: String,
    /// Center status text.
    pub center: String,
    /// Right-aligned status text.
    pub right: String,
}

impl StatusLine {
    /// Create a new status line.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the left-aligned text.
    #[must_use]
    pub fn with_left(mut self, text: impl Into<String>) -> Self {
        self.left = text.into();
        self
    }

    /// Set the center text.
    #[must_use]
    pub fn with_center(mut self, text: impl Into<String>) -> Self {
        self.center = text.into();
        self
    }

    /// Set the right-aligned text.
    #[must_use]
    pub fn with_right(mut self, text: impl Into<String>) -> Self {
        self.right = text.into();
        self
    }
}

// ─── App Shell ───────────────────────────────────────────────────────────────

/// The main app shell that manages screens, chrome, and input dispatch.
pub struct AppShell {
    /// Shell configuration.
    pub config: ShellConfig,
    /// Screen registry.
    pub registry: ScreenRegistry,
    /// Currently active screen ID.
    pub active_screen: Option<ScreenId>,
    /// Keymap for input resolution.
    pub keymap: Keymap,
    /// Overlay manager.
    pub overlays: OverlayManager,
    /// Status line content.
    pub status_line: StatusLine,
    /// Whether the app should quit.
    pub should_quit: bool,
}

impl AppShell {
    /// Create a new app shell with the given config.
    #[must_use]
    pub fn new(config: ShellConfig) -> Self {
        Self {
            config,
            registry: ScreenRegistry::new(),
            active_screen: None,
            keymap: Keymap::default_bindings(),
            overlays: OverlayManager::new(),
            status_line: StatusLine::new(),
            should_quit: false,
        }
    }

    /// Navigate to a screen by ID.
    pub fn navigate_to(&mut self, id: &ScreenId) {
        if self.registry.get(id).is_some() {
            // Blur the old screen.
            if let Some(old_id) = &self.active_screen {
                let old_id = old_id.clone();
                if let Some(screen) = self.registry.get_mut(&old_id) {
                    screen.on_blur();
                }
            }
            // Focus the new screen.
            self.active_screen = Some(id.clone());
            if let Some(screen) = self.registry.get_mut(id) {
                screen.on_focus();
            }
        }
    }

    /// Navigate to the next screen in tab order.
    pub fn next_screen(&mut self) {
        if let Some(current) = &self.active_screen {
            if let Some(next) = self.registry.next_screen(current).cloned() {
                self.navigate_to(&next);
            }
        }
    }

    /// Navigate to the previous screen in tab order.
    pub fn prev_screen(&mut self) {
        if let Some(current) = &self.active_screen {
            if let Some(prev) = self.registry.prev_screen(current).cloned() {
                self.navigate_to(&prev);
            }
        }
    }

    /// Build the screen context for the current state.
    #[must_use]
    pub fn screen_context(&self, area: Rect) -> ScreenContext {
        ScreenContext {
            active_screen: self
                .active_screen
                .clone()
                .unwrap_or_else(|| ScreenId::new("")),
            terminal_width: area.width,
            terminal_height: area.height,
            focused: true,
        }
    }

    /// Handle an input event. Returns `true` if the app should quit.
    pub fn handle_input(&mut self, event: &InputEvent) -> bool {
        // If an overlay is active, let it handle first.
        if self.overlays.has_active() {
            if let InputEvent::Key(key, mods) = event {
                if let Some(action) = self.keymap.resolve(*key, *mods) {
                    if action == &KeyAction::Dismiss {
                        self.overlays.dismiss();
                        return false;
                    }
                }
            }
            return false;
        }

        // Resolve key actions.
        if let InputEvent::Key(key, mods) = event {
            if let Some(action) = self.keymap.resolve(*key, *mods).cloned() {
                match action {
                    KeyAction::Quit => {
                        self.should_quit = true;
                        return true;
                    }
                    KeyAction::NextScreen => {
                        self.next_screen();
                        return false;
                    }
                    KeyAction::PrevScreen => {
                        self.prev_screen();
                        return false;
                    }
                    KeyAction::ToggleHelp => {
                        self.overlays
                            .push(crate::overlay::OverlayRequest::new(
                                crate::overlay::OverlayKind::Help,
                                "Help".to_string(),
                            ));
                        return false;
                    }
                    _ => {}
                }
            }
        }

        // Forward to active screen.
        if let Some(screen_id) = &self.active_screen {
            let screen_id = screen_id.clone();
            let area = Rect::new(0, 0, 80, 24); // Default for context.
            let ctx = self.screen_context(area);
            if let Some(screen) = self.registry.get_mut(&screen_id) {
                match screen.handle_input(event, &ctx) {
                    ScreenAction::Quit => {
                        self.should_quit = true;
                        return true;
                    }
                    ScreenAction::Navigate(target) => {
                        self.navigate_to(&target);
                    }
                    ScreenAction::OpenOverlay(name) => {
                        self.overlays
                            .push(crate::overlay::OverlayRequest::new(
                                crate::overlay::OverlayKind::Custom(name.clone()),
                                name,
                            ));
                    }
                    ScreenAction::Consumed | ScreenAction::Ignored => {}
                }
            }
        }

        false
    }

    /// Render the shell chrome and active screen.
    #[allow(clippy::too_many_lines)]
    pub fn render(&self, frame: &mut Frame<'_>) {
        let area = frame.area();
        let ctx = self.screen_context(area);

        // Layout: optional breadcrumbs + content + optional status bar.
        let mut constraints = Vec::new();
        if self.config.show_breadcrumbs && self.registry.len() > 1 {
            constraints.push(Constraint::Length(1));
        }
        constraints.push(Constraint::Min(1));
        if self.config.show_status_bar {
            constraints.push(Constraint::Length(1));
        }

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area);

        let mut chunk_idx = 0;

        // Breadcrumbs / tabs.
        if self.config.show_breadcrumbs && self.registry.len() > 1 {
            let titles: Vec<Line<'_>> = self
                .registry
                .screen_ids()
                .iter()
                .map(|id| {
                    let title = self
                        .registry
                        .get(id)
                        .map_or(id.0.as_str(), |s| s.title());
                    Line::from(title.to_string())
                })
                .collect();

            let selected = self
                .active_screen
                .as_ref()
                .and_then(|active| {
                    self.registry
                        .screen_ids()
                        .iter()
                        .position(|id| id == active)
                })
                .unwrap_or(0);

            let tabs = Tabs::new(titles)
                .select(selected)
                .highlight_style(
                    Style::default()
                        .fg(self.config.theme.highlight_fg.to_ratatui())
                        .bg(self.config.theme.highlight_bg.to_ratatui())
                        .add_modifier(Modifier::BOLD),
                )
                .style(
                    Style::default()
                        .fg(self.config.theme.muted.to_ratatui())
                        .bg(self.config.theme.bg.to_ratatui()),
                );

            frame.render_widget(tabs, chunks[chunk_idx]);
            chunk_idx += 1;
        }

        // Main content area.
        let content_area = chunks[chunk_idx];
        chunk_idx += 1;

        if let Some(screen_id) = &self.active_screen {
            if let Some(screen) = self.registry.get(screen_id) {
                screen.render(frame, &ctx);
            }
        } else {
            // No screen active — render placeholder.
            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(
                    Style::default().fg(self.config.theme.border.to_ratatui()),
                )
                .style(
                    Style::default()
                        .bg(self.config.theme.bg.to_ratatui())
                        .fg(self.config.theme.fg.to_ratatui()),
                );
            let placeholder =
                Paragraph::new("No screens registered").block(block);
            frame.render_widget(placeholder, content_area);
        }

        // Status bar.
        if self.config.show_status_bar {
            let status_area = chunks[chunk_idx];
            let status_text = if self.status_line.center.is_empty() {
                format!(" {} ", self.config.title)
            } else {
                format!(
                    " {} │ {} ",
                    self.config.title, self.status_line.center
                )
            };

            let status_spans = vec![
                Span::styled(
                    &self.status_line.left,
                    Style::default()
                        .fg(self.config.theme.status_bar_fg.to_ratatui()),
                ),
                Span::styled(
                    status_text,
                    Style::default()
                        .fg(self.config.theme.status_bar_fg.to_ratatui())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    &self.status_line.right,
                    Style::default()
                        .fg(self.config.theme.status_bar_fg.to_ratatui()),
                ),
            ];

            let status = Paragraph::new(Line::from(status_spans)).style(
                Style::default()
                    .bg(self.config.theme.status_bar_bg.to_ratatui()),
            );

            frame.render_widget(status, status_area);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_config_default() {
        let config = ShellConfig::default();
        assert_eq!(config.title, "frankensearch");
        assert!(config.show_status_bar);
        assert!(config.show_breadcrumbs);
    }

    #[test]
    fn status_line_builder() {
        let status = StatusLine::new()
            .with_left("left")
            .with_center("center")
            .with_right("right");
        assert_eq!(status.left, "left");
        assert_eq!(status.center, "center");
        assert_eq!(status.right, "right");
    }

    #[test]
    fn shell_creation() {
        let shell = AppShell::new(ShellConfig::default());
        assert!(!shell.should_quit);
        assert!(shell.active_screen.is_none());
        assert!(shell.registry.is_empty());
    }

    #[test]
    fn shell_config_serde_roundtrip() {
        let config = ShellConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let decoded: ShellConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.title, config.title);
    }

    #[test]
    fn shell_quit_handling() {
        let mut shell = AppShell::new(ShellConfig::default());
        let event = InputEvent::Key(
            crossterm::event::KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = shell.handle_input(&event);
        assert!(quit);
        assert!(shell.should_quit);
    }
}
