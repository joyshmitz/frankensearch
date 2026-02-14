//! Accessibility and display preferences with persistence.
//!
//! Manages user preferences for contrast, reduced motion, focus visibility,
//! and density mode. Preferences are serializable for persistence across sessions.

use serde::{Deserialize, Serialize};

// ─── Contrast Mode ──────────────────────────────────────────────────────────

/// Contrast level for accessibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContrastMode {
    /// Standard contrast (default).
    #[default]
    Normal,
    /// High contrast for better visibility.
    High,
}

impl ContrastMode {
    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::High => "High Contrast",
        }
    }

    /// Toggle between modes.
    #[must_use]
    pub const fn toggle(self) -> Self {
        match self {
            Self::Normal => Self::High,
            Self::High => Self::Normal,
        }
    }
}

// ─── Motion Preference ──────────────────────────────────────────────────────

/// Motion preference for animations and transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionPreference {
    /// Full animations (default).
    #[default]
    Full,
    /// Reduced motion — disable non-essential animations.
    Reduced,
}

impl MotionPreference {
    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Full => "Full Motion",
            Self::Reduced => "Reduced Motion",
        }
    }

    /// Toggle between modes.
    #[must_use]
    pub const fn toggle(self) -> Self {
        match self {
            Self::Full => Self::Reduced,
            Self::Reduced => Self::Full,
        }
    }

    /// Whether animations should be shown.
    #[must_use]
    pub const fn animations_enabled(self) -> bool {
        matches!(self, Self::Full)
    }
}

// ─── Focus Visibility ───────────────────────────────────────────────────────

/// Focus indicator visibility level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FocusVisibility {
    /// Standard focus indicator.
    #[default]
    Normal,
    /// Enhanced focus indicators (bolder, larger).
    Enhanced,
}

impl FocusVisibility {
    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Enhanced => "Enhanced",
        }
    }

    /// Toggle between modes.
    #[must_use]
    pub const fn toggle(self) -> Self {
        match self {
            Self::Normal => Self::Enhanced,
            Self::Enhanced => Self::Normal,
        }
    }
}

// ─── Display Preferences ────────────────────────────────────────────────────

/// Complete set of accessibility and display preferences.
///
/// Serializable for persistence. Product code reads these preferences
/// to adjust rendering behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisplayPreferences {
    /// Contrast mode.
    pub contrast: ContrastMode,
    /// Motion preference.
    pub motion: MotionPreference,
    /// Focus indicator visibility.
    pub focus_visibility: FocusVisibility,
    /// Whether to show shortcut hints inline (e.g., "[j] down [k] up").
    pub show_shortcut_hints: bool,
}

impl Default for DisplayPreferences {
    fn default() -> Self {
        Self {
            contrast: ContrastMode::default(),
            motion: MotionPreference::default(),
            focus_visibility: FocusVisibility::default(),
            show_shortcut_hints: true,
        }
    }
}

impl DisplayPreferences {
    /// Create default preferences.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Toggle contrast mode.
    pub fn toggle_contrast(&mut self) {
        self.contrast = self.contrast.toggle();
    }

    /// Toggle motion preference.
    pub fn toggle_motion(&mut self) {
        self.motion = self.motion.toggle();
    }

    /// Toggle focus visibility.
    pub fn toggle_focus_visibility(&mut self) {
        self.focus_visibility = self.focus_visibility.toggle();
    }

    /// Toggle shortcut hints.
    pub fn toggle_shortcut_hints(&mut self) {
        self.show_shortcut_hints = !self.show_shortcut_hints;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contrast_mode_toggle() {
        let mode = ContrastMode::Normal;
        assert_eq!(mode.toggle(), ContrastMode::High);
        assert_eq!(mode.toggle().toggle(), ContrastMode::Normal);
    }

    #[test]
    fn contrast_mode_labels() {
        assert_eq!(ContrastMode::Normal.label(), "Normal");
        assert_eq!(ContrastMode::High.label(), "High Contrast");
    }

    #[test]
    fn motion_preference_toggle() {
        let pref = MotionPreference::Full;
        assert!(pref.animations_enabled());
        let reduced = pref.toggle();
        assert_eq!(reduced, MotionPreference::Reduced);
        assert!(!reduced.animations_enabled());
    }

    #[test]
    fn focus_visibility_toggle() {
        let vis = FocusVisibility::Normal;
        assert_eq!(vis.toggle(), FocusVisibility::Enhanced);
    }

    #[test]
    fn display_preferences_default() {
        let prefs = DisplayPreferences::new();
        assert_eq!(prefs.contrast, ContrastMode::Normal);
        assert_eq!(prefs.motion, MotionPreference::Full);
        assert_eq!(prefs.focus_visibility, FocusVisibility::Normal);
        assert!(prefs.show_shortcut_hints);
    }

    #[test]
    fn display_preferences_toggles() {
        let mut prefs = DisplayPreferences::new();
        prefs.toggle_contrast();
        assert_eq!(prefs.contrast, ContrastMode::High);
        prefs.toggle_motion();
        assert_eq!(prefs.motion, MotionPreference::Reduced);
        prefs.toggle_focus_visibility();
        assert_eq!(prefs.focus_visibility, FocusVisibility::Enhanced);
        prefs.toggle_shortcut_hints();
        assert!(!prefs.show_shortcut_hints);
    }

    #[test]
    fn display_preferences_serde_roundtrip() {
        let mut prefs = DisplayPreferences::new();
        prefs.toggle_contrast();
        prefs.toggle_motion();
        let json = serde_json::to_string(&prefs).unwrap();
        let decoded: DisplayPreferences = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, prefs);
    }

    #[test]
    fn contrast_mode_serde_roundtrip() {
        for mode in [ContrastMode::Normal, ContrastMode::High] {
            let json = serde_json::to_string(&mode).unwrap();
            let decoded: ContrastMode = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, mode);
        }
    }

    #[test]
    fn motion_preference_serde_roundtrip() {
        for pref in [MotionPreference::Full, MotionPreference::Reduced] {
            let json = serde_json::to_string(&pref).unwrap();
            let decoded: MotionPreference = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, pref);
        }
    }

    #[test]
    fn motion_preference_labels() {
        assert_eq!(MotionPreference::Full.label(), "Full Motion");
        assert_eq!(MotionPreference::Reduced.label(), "Reduced Motion");
    }

    #[test]
    fn focus_visibility_labels() {
        assert_eq!(FocusVisibility::Normal.label(), "Normal");
        assert_eq!(FocusVisibility::Enhanced.label(), "Enhanced");
    }
}
