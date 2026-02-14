//! Fleet overview screen — primary dashboard showing all discovered instances.
//!
//! Displays instance list with health status, document counts, pending jobs,
//! and resource utilization at a glance.

use std::any::Any;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};

use frankensearch_tui::Screen;
use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{ScreenAction, ScreenContext, ScreenId};

use crate::presets::ViewState;
use crate::state::AppState;

/// Fleet overview screen showing all discovered instances.
pub struct FleetOverviewScreen {
    id: ScreenId,
    state: AppState,
    view: ViewState,
    selected_row: usize,
}

impl FleetOverviewScreen {
    /// Create a new fleet overview screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.fleet"),
            state: AppState::new(),
            view: ViewState::default(),
            selected_row: 0,
        }
    }

    /// Update the screen's data from shared state.
    pub fn update_state(&mut self, state: &AppState, view: &ViewState) {
        self.state = state.clone();
        self.view = view.clone();
        let visible = self.visible_instances().len();
        if visible == 0 {
            self.selected_row = 0;
        } else if self.selected_row >= visible {
            self.selected_row = visible - 1;
        }
    }

    fn visible_instances(&self) -> Vec<&crate::state::InstanceInfo> {
        let fleet = self.state.fleet();
        let mut visible: Vec<_> = fleet
            .instances
            .iter()
            .filter(|inst| !self.view.hide_healthy || !inst.healthy)
            .filter(|inst| {
                self.view
                    .project_filter
                    .as_deref()
                    .is_none_or(|project| inst.project.eq_ignore_ascii_case(project))
            })
            .collect();

        if self.view.unhealthy_first {
            visible.sort_by(|left, right| {
                (left.healthy, left.project.as_str(), left.id.as_str()).cmp(&(
                    right.healthy,
                    right.project.as_str(),
                    right.id.as_str(),
                ))
            });
        }

        visible
    }

    /// Build the instance table rows.
    fn build_rows(&self) -> Vec<Row<'_>> {
        let fleet = self.state.fleet();
        self.visible_instances()
            .into_iter()
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

                let cells = if self.view.density.show_inline_metrics() {
                    vec![
                        health.to_string(),
                        inst.project.clone(),
                        inst.id.clone(),
                        format!("{}", inst.doc_count),
                        format!("{}", inst.pending_jobs),
                        resources,
                    ]
                } else {
                    vec![
                        health.to_string(),
                        inst.project.clone(),
                        inst.id.clone(),
                        format!("{}", inst.doc_count),
                        format!("{}", inst.pending_jobs),
                    ]
                };

                Row::new(cells)
                    .style(style)
                    .height(self.view.density.row_height())
            })
            .collect()
    }

    /// Number of instances currently visible to this screen.
    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.visible_instances().len()
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
        let visible = self.visible_instances();
        let visible_count = visible.len();
        let visible_healthy = visible.iter().filter(|inst| inst.healthy).count();
        let visible_docs: u64 = visible.iter().map(|inst| inst.doc_count).sum();
        let visible_pending: u64 = visible.iter().map(|inst| inst.pending_jobs).sum();
        let summary = format!(
            " {visible_count}/{} instances | {visible_healthy} healthy | {visible_docs} docs | {visible_pending} pending | {} density",
            fleet.instance_count(),
            self.view.density,
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
        let show_metrics = self.view.density.show_inline_metrics();
        let header_row = if show_metrics {
            Row::new(vec![
                "Health", "Project", "Instance", "Docs", "Pending", "CPU",
            ])
            .style(Style::default().add_modifier(Modifier::BOLD))
        } else {
            Row::new(vec!["Health", "Project", "Instance", "Docs", "Pending"])
                .style(Style::default().add_modifier(Modifier::BOLD))
        };

        let rows = self.build_rows();
        let table = if show_metrics {
            Table::new(
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
            .block(Block::default().borders(Borders::ALL).title(" Instances "))
        } else {
            Table::new(
                rows,
                [
                    Constraint::Length(6),
                    Constraint::Length(14),
                    Constraint::Length(16),
                    Constraint::Length(10),
                    Constraint::Length(10),
                ],
            )
            .header(header_row)
            .block(Block::default().borders(Borders::ALL).title(" Instances "))
        };

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
                    let count = self.instance_count();
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::ViewState;
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
        screen.update_state(&state, &ViewState::default());
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
        screen.update_state(&state, &ViewState::default());

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

    #[test]
    fn hide_healthy_filter_applies_to_rows() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "healthy".to_string(),
                    project: "p".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 1,
                    pending_jobs: 0,
                },
                crate::state::InstanceInfo {
                    id: "warn".to_string(),
                    project: "p".to_string(),
                    pid: None,
                    healthy: false,
                    doc_count: 2,
                    pending_jobs: 1,
                },
            ],
            resources: std::collections::HashMap::new(),
            search_metrics: std::collections::HashMap::new(),
        });
        let view = ViewState {
            hide_healthy: true,
            ..ViewState::default()
        };
        screen.update_state(&state, &view);

        let rows = screen.build_rows();
        assert_eq!(rows.len(), 1);
    }
}
