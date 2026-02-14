//! Fleet overview screen — primary dashboard showing all discovered instances.
//!
//! Displays instance list with health status, document counts, pending jobs,
//! and resource utilization at a glance.

use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};
use ratatui::Frame;

use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{ScreenAction, ScreenContext, ScreenId};
use frankensearch_tui::Screen;

use crate::state::AppState;

/// Fleet overview screen showing all discovered instances.
pub struct FleetOverviewScreen {
    id: ScreenId,
    state: AppState,
    selected_row: usize,
}

impl FleetOverviewScreen {
    /// Create a new fleet overview screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.fleet"),
            state: AppState::new(),
            selected_row: 0,
        }
    }

    /// Update the screen's data from shared state.
    pub fn update_state(&mut self, state: &AppState) {
        self.state = state.clone();
    }

    /// Build the instance table rows.
    fn build_rows(&self) -> Vec<Row<'_>> {
        let fleet = self.state.fleet();
        fleet
            .instances
            .iter()
            .enumerate()
            .map(|(i, inst)| {
                let health = if inst.healthy { "OK" } else { "WARN" };
                let resources = fleet
                    .resources
                    .get(&inst.id)
                    .map_or_else(|| "-".to_string(), |r| format!("{:.1}%", r.cpu_percent));

                let style = if i == self.selected_row {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else if !inst.healthy {
                    Style::default().fg(ratatui::style::Color::Red)
                } else {
                    Style::default()
                };

                Row::new(vec![
                    health.to_string(),
                    inst.project.clone(),
                    inst.id.clone(),
                    format!("{}", inst.doc_count),
                    format!("{}", inst.pending_jobs),
                    resources,
                ])
                .style(style)
            })
            .collect()
    }
}

impl Default for FleetOverviewScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for FleetOverviewScreen {
    fn id(&self) -> &ScreenId {
        &self.id
    }

    fn title(&self) -> &'static str {
        "Fleet Overview"
    }

    fn render(&self, frame: &mut Frame<'_>, _ctx: &ScreenContext) {
        let area = frame.area();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(5)])
            .split(area);

        // Header with summary stats.
        let fleet = self.state.fleet();
        let summary = format!(
            " {} instances | {} healthy | {} docs | {} pending",
            fleet.instance_count(),
            fleet.healthy_count(),
            fleet.total_docs(),
            fleet.total_pending_jobs(),
        );
        let header = Paragraph::new(Line::from(vec![
            Span::styled("Fleet: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(summary),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Fleet Overview "),
        );
        frame.render_widget(header, chunks[0]);

        // Instance table.
        let header_row = Row::new(vec![
            "Health", "Project", "Instance", "Docs", "Pending", "CPU",
        ])
        .style(Style::default().add_modifier(Modifier::BOLD));

        let rows = self.build_rows();
        let table = Table::new(
            rows,
            [
                Constraint::Length(6),
                Constraint::Length(12),
                Constraint::Length(15),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(8),
            ],
        )
        .header(header_row)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Instances "),
        );

        frame.render_widget(table, chunks[1]);
    }

    fn handle_input(&mut self, event: &InputEvent, _ctx: &ScreenContext) -> ScreenAction {
        if let InputEvent::Key(key, _mods) = event {
            match key {
                crossterm::event::KeyCode::Up | crossterm::event::KeyCode::Char('k') => {
                    if self.selected_row > 0 {
                        self.selected_row -= 1;
                    }
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Down | crossterm::event::KeyCode::Char('j') => {
                    let count = self.state.fleet().instance_count();
                    if count > 0 && self.selected_row < count - 1 {
                        self.selected_row += 1;
                    }
                    return ScreenAction::Consumed;
                }
                _ => {}
            }
        }
        ScreenAction::Ignored
    }

    fn semantic_role(&self) -> &'static str {
        "grid"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::FleetSnapshot;

    #[test]
    fn fleet_screen_default() {
        let screen = FleetOverviewScreen::new();
        assert_eq!(screen.id(), &ScreenId::new("ops.fleet"));
        assert_eq!(screen.title(), "Fleet Overview");
        assert_eq!(screen.semantic_role(), "grid");
    }

    #[test]
    fn fleet_screen_empty_state() {
        let screen = FleetOverviewScreen::new();
        let rows = screen.build_rows();
        assert!(rows.is_empty());
    }

    #[test]
    fn fleet_screen_with_data() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "a".to_string(),
                    project: "test".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 100,
                    pending_jobs: 0,
                },
                crate::state::InstanceInfo {
                    id: "b".to_string(),
                    project: "test2".to_string(),
                    pid: None,
                    healthy: false,
                    doc_count: 200,
                    pending_jobs: 50,
                },
            ],
            resources: std::collections::HashMap::new(),
            search_metrics: std::collections::HashMap::new(),
        });
        screen.update_state(&state);
        let rows = screen.build_rows();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn fleet_screen_navigation() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "a".to_string(),
                    project: "p".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 1,
                    pending_jobs: 0,
                },
                crate::state::InstanceInfo {
                    id: "b".to_string(),
                    project: "p".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 2,
                    pending_jobs: 0,
                },
            ],
            resources: std::collections::HashMap::new(),
            search_metrics: std::collections::HashMap::new(),
        });
        screen.update_state(&state);

        assert_eq!(screen.selected_row, 0);

        let ctx = ScreenContext {
            active_screen: ScreenId::new("ops.fleet"),
            terminal_width: 80,
            terminal_height: 24,
            focused: true,
        };

        // Move down.
        let event = InputEvent::Key(
            crossterm::event::KeyCode::Down,
            crossterm::event::KeyModifiers::NONE,
        );
        let result = screen.handle_input(&event, &ctx);
        assert_eq!(result, ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);

        // Don't go past end.
        let result = screen.handle_input(&event, &ctx);
        assert_eq!(result, ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);

        // Move up.
        let event = InputEvent::Key(
            crossterm::event::KeyCode::Up,
            crossterm::event::KeyModifiers::NONE,
        );
        let result = screen.handle_input(&event, &ctx);
        assert_eq!(result, ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 0);
    }
}
