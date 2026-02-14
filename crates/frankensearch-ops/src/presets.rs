//! Operator view presets and density modes for progressive disclosure.
//!
//! View presets configure which information is visible and at what density,
//! letting operators switch between high-level triage and deep-dive modes
//! without reconfiguring individual panels.

use serde::{Deserialize, Serialize};

// ─── Density ────────────────────────────────────────────────────────────────

/// Display density mode controlling the amount of detail shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Density {
    /// Compact: minimal chrome, one-line rows, maximized data density.
    Compact,
    /// Normal: balanced layout with reasonable spacing.
    #[default]
    Normal,
    /// Expanded: extra detail panes, wider spacing, inline metrics.
    Expanded,
}

impl Density {
    /// All densities in order.
    pub const ALL: &'static [Self] = &[Self::Compact, Self::Normal, Self::Expanded];

    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Compact => "Compact",
            Self::Normal => "Normal",
            Self::Expanded => "Expanded",
        }
    }

    /// Cycle to the next density mode.
    #[must_use]
    pub const fn cycle_next(self) -> Self {
        match self {
            Self::Compact => Self::Normal,
            Self::Normal => Self::Expanded,
            Self::Expanded => Self::Compact,
        }
    }

    /// Cycle to the previous density mode.
    #[must_use]
    pub const fn cycle_prev(self) -> Self {
        match self {
            Self::Compact => Self::Expanded,
            Self::Normal => Self::Compact,
            Self::Expanded => Self::Normal,
        }
    }

    /// Whether detail panes should be shown.
    #[must_use]
    pub const fn show_details(self) -> bool {
        matches!(self, Self::Expanded)
    }

    /// Whether inline metrics should be shown (expanded + normal).
    #[must_use]
    pub const fn show_inline_metrics(self) -> bool {
        !matches!(self, Self::Compact)
    }

    /// Suggested table row height.
    #[must_use]
    pub const fn row_height(self) -> u16 {
        match self {
            Self::Compact | Self::Normal => 1,
            Self::Expanded => 2,
        }
    }
}

impl std::fmt::Display for Density {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ─── View Preset ────────────────────────────────────────────────────────────

/// Predefined view configurations for common operational workflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewPreset {
    /// Fleet triage: compact overview of all instances, health-first sorting.
    #[default]
    FleetTriage,
    /// Project deep-dive: expanded view of a single project with full metrics.
    ProjectDeepDive,
    /// Incident mode: high-contrast, alerts prominent, unhealthy instances first.
    IncidentMode,
    /// Low-noise mode: hide healthy instances, only show actionable items.
    LowNoise,
}

impl ViewPreset {
    /// All presets.
    pub const ALL: &'static [Self] = &[
        Self::FleetTriage,
        Self::ProjectDeepDive,
        Self::IncidentMode,
        Self::LowNoise,
    ];

    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::FleetTriage => "Fleet Triage",
            Self::ProjectDeepDive => "Project Deep Dive",
            Self::IncidentMode => "Incident Mode",
            Self::LowNoise => "Low Noise",
        }
    }

    /// Short description for the command palette.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::FleetTriage => "Compact fleet overview, health-first sorting",
            Self::ProjectDeepDive => "Expanded single-project view with full metrics",
            Self::IncidentMode => "High-contrast, alerts prominent, unhealthy first",
            Self::LowNoise => "Hide healthy instances, show only actionable items",
        }
    }

    /// Recommended density for this preset.
    #[must_use]
    pub const fn default_density(self) -> Density {
        match self {
            Self::ProjectDeepDive => Density::Expanded,
            Self::IncidentMode => Density::Normal,
            Self::FleetTriage | Self::LowNoise => Density::Compact,
        }
    }

    /// Whether to use high-contrast theme by default.
    #[must_use]
    pub const fn prefer_high_contrast(self) -> bool {
        matches!(self, Self::IncidentMode)
    }

    /// Whether to hide healthy instances by default.
    #[must_use]
    pub const fn hide_healthy(self) -> bool {
        matches!(self, Self::LowNoise)
    }

    /// Whether to sort unhealthy instances first.
    #[must_use]
    pub const fn unhealthy_first(self) -> bool {
        matches!(self, Self::IncidentMode | Self::LowNoise)
    }
}

impl std::fmt::Display for ViewPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ─── View State ─────────────────────────────────────────────────────────────

/// Current view configuration combining preset and overrides.
///
/// The `ViewState` holds the active preset, density, and filtering options.
/// These can be changed at runtime via keyboard shortcuts or the command palette.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewState {
    /// Active view preset.
    pub preset: ViewPreset,
    /// Current density mode (may differ from preset default if user overrode it).
    pub density: Density,
    /// Whether to hide healthy instances.
    pub hide_healthy: bool,
    /// Whether to sort unhealthy instances first.
    pub unhealthy_first: bool,
    /// Optional project filter (only show instances matching this project).
    pub project_filter: Option<String>,
}

impl Default for ViewState {
    fn default() -> Self {
        Self::from_preset(ViewPreset::default())
    }
}

impl ViewState {
    /// Create a view state from a preset with its default settings.
    #[must_use]
    pub const fn from_preset(preset: ViewPreset) -> Self {
        Self {
            preset,
            density: preset.default_density(),
            hide_healthy: preset.hide_healthy(),
            unhealthy_first: preset.unhealthy_first(),
            project_filter: None,
        }
    }

    /// Apply a new preset, resetting overrides to preset defaults.
    pub fn apply_preset(&mut self, preset: ViewPreset) {
        *self = Self::from_preset(preset);
    }

    /// Cycle density mode forward.
    pub const fn cycle_density(&mut self) {
        self.density = self.density.cycle_next();
    }

    /// Toggle the hide-healthy filter.
    pub const fn toggle_hide_healthy(&mut self) {
        self.hide_healthy = !self.hide_healthy;
    }

    /// Toggle unhealthy-first sorting.
    pub const fn toggle_unhealthy_first(&mut self) {
        self.unhealthy_first = !self.unhealthy_first;
    }

    /// Set a project filter.
    pub fn set_project_filter(&mut self, project: impl Into<String>) {
        self.project_filter = Some(project.into());
    }

    /// Clear the project filter.
    pub fn clear_project_filter(&mut self) {
        self.project_filter = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_all_values() {
        assert_eq!(Density::ALL.len(), 3);
    }

    #[test]
    fn density_cycle_next() {
        assert_eq!(Density::Compact.cycle_next(), Density::Normal);
        assert_eq!(Density::Normal.cycle_next(), Density::Expanded);
        assert_eq!(Density::Expanded.cycle_next(), Density::Compact);
    }

    #[test]
    fn density_cycle_prev() {
        assert_eq!(Density::Compact.cycle_prev(), Density::Expanded);
        assert_eq!(Density::Normal.cycle_prev(), Density::Compact);
        assert_eq!(Density::Expanded.cycle_prev(), Density::Normal);
    }

    #[test]
    fn density_display() {
        assert_eq!(Density::Compact.to_string(), "Compact");
        assert_eq!(Density::Normal.to_string(), "Normal");
        assert_eq!(Density::Expanded.to_string(), "Expanded");
    }

    #[test]
    fn density_show_details() {
        assert!(!Density::Compact.show_details());
        assert!(!Density::Normal.show_details());
        assert!(Density::Expanded.show_details());
    }

    #[test]
    fn density_show_inline_metrics() {
        assert!(!Density::Compact.show_inline_metrics());
        assert!(Density::Normal.show_inline_metrics());
        assert!(Density::Expanded.show_inline_metrics());
    }

    #[test]
    fn density_row_height() {
        assert_eq!(Density::Compact.row_height(), 1);
        assert_eq!(Density::Normal.row_height(), 1);
        assert_eq!(Density::Expanded.row_height(), 2);
    }

    #[test]
    fn density_serde_roundtrip() {
        for density in Density::ALL {
            let json = serde_json::to_string(density).unwrap();
            let decoded: Density = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, *density);
        }
    }

    #[test]
    fn view_preset_all_values() {
        assert_eq!(ViewPreset::ALL.len(), 4);
    }

    #[test]
    fn view_preset_labels() {
        assert_eq!(ViewPreset::FleetTriage.label(), "Fleet Triage");
        assert_eq!(ViewPreset::IncidentMode.label(), "Incident Mode");
    }

    #[test]
    fn view_preset_descriptions() {
        for preset in ViewPreset::ALL {
            assert!(!preset.description().is_empty());
        }
    }

    #[test]
    fn view_preset_default_densities() {
        assert_eq!(ViewPreset::FleetTriage.default_density(), Density::Compact);
        assert_eq!(
            ViewPreset::ProjectDeepDive.default_density(),
            Density::Expanded
        );
        assert_eq!(ViewPreset::IncidentMode.default_density(), Density::Normal);
        assert_eq!(ViewPreset::LowNoise.default_density(), Density::Compact);
    }

    #[test]
    fn view_preset_contrast_preferences() {
        assert!(ViewPreset::IncidentMode.prefer_high_contrast());
        assert!(!ViewPreset::FleetTriage.prefer_high_contrast());
    }

    #[test]
    fn view_preset_filtering() {
        assert!(ViewPreset::LowNoise.hide_healthy());
        assert!(!ViewPreset::FleetTriage.hide_healthy());
        assert!(ViewPreset::IncidentMode.unhealthy_first());
        assert!(ViewPreset::LowNoise.unhealthy_first());
        assert!(!ViewPreset::FleetTriage.unhealthy_first());
    }

    #[test]
    fn view_preset_serde_roundtrip() {
        for preset in ViewPreset::ALL {
            let json = serde_json::to_string(preset).unwrap();
            let decoded: ViewPreset = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, *preset);
        }
    }

    #[test]
    fn view_preset_display() {
        assert_eq!(ViewPreset::FleetTriage.to_string(), "Fleet Triage");
        assert_eq!(ViewPreset::LowNoise.to_string(), "Low Noise");
    }

    #[test]
    fn view_state_default() {
        let state = ViewState::default();
        assert_eq!(state.preset, ViewPreset::FleetTriage);
        assert_eq!(state.density, Density::Compact);
        assert!(!state.hide_healthy);
        assert!(!state.unhealthy_first);
        assert!(state.project_filter.is_none());
    }

    #[test]
    fn view_state_from_preset() {
        let state = ViewState::from_preset(ViewPreset::IncidentMode);
        assert_eq!(state.preset, ViewPreset::IncidentMode);
        assert_eq!(state.density, Density::Normal);
        assert!(!state.hide_healthy);
        assert!(state.unhealthy_first);
    }

    #[test]
    fn view_state_apply_preset() {
        let mut state = ViewState {
            density: Density::Expanded,
            ..ViewState::default()
        };
        state.apply_preset(ViewPreset::LowNoise);
        assert_eq!(state.preset, ViewPreset::LowNoise);
        assert_eq!(state.density, Density::Compact); // Reset to preset default.
        assert!(state.hide_healthy);
    }

    #[test]
    fn view_state_cycle_density() {
        let mut state = ViewState::default();
        assert_eq!(state.density, Density::Compact);
        state.cycle_density();
        assert_eq!(state.density, Density::Normal);
        state.cycle_density();
        assert_eq!(state.density, Density::Expanded);
        state.cycle_density();
        assert_eq!(state.density, Density::Compact);
    }

    #[test]
    fn view_state_toggle_filters() {
        let mut state = ViewState::default();
        assert!(!state.hide_healthy);
        state.toggle_hide_healthy();
        assert!(state.hide_healthy);
        state.toggle_hide_healthy();
        assert!(!state.hide_healthy);
    }

    #[test]
    fn view_state_project_filter() {
        let mut state = ViewState::default();
        state.set_project_filter("cass");
        assert_eq!(state.project_filter.as_deref(), Some("cass"));
        state.clear_project_filter();
        assert!(state.project_filter.is_none());
    }

    #[test]
    fn view_state_serde_roundtrip() {
        let mut state = ViewState::from_preset(ViewPreset::ProjectDeepDive);
        state.set_project_filter("xf");
        let json = serde_json::to_string(&state).unwrap();
        let decoded: ViewState = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, state);
    }
}
