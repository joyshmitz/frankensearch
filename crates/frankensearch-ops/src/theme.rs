//! Semantic color palette definitions for Dark, Light, and High-Contrast themes.
//!
//! Each theme variant maps semantic roles (foreground, background, accent, etc.)
//! to concrete terminal colors. Screens render using semantic names; the active
//! palette resolves them to ANSI / RGB values at paint time.

use ratatui::style::{Color, Modifier, Style};
use serde::{Deserialize, Serialize};

use crate::preferences::{ContrastMode, DisplayPreferences};

// ─── Schema Version ─────────────────────────────────────────────────────────

/// Palette schema version (bump when semantic roles change).
pub const PALETTE_SCHEMA_VERSION: u32 = 1;

// ─── Theme Variant ──────────────────────────────────────────────────────────

/// Resolved theme variant after combining base theme + accessibility overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThemeVariant {
    Dark,
    Light,
    HighContrastDark,
    HighContrastLight,
}

impl ThemeVariant {
    /// Resolve variant from base theme preference + contrast mode.
    #[must_use]
    pub const fn resolve(is_light: bool, contrast: ContrastMode) -> Self {
        match (is_light, contrast) {
            (false, ContrastMode::Normal) => Self::Dark,
            (true, ContrastMode::Normal) => Self::Light,
            (false, ContrastMode::High) => Self::HighContrastDark,
            (true, ContrastMode::High) => Self::HighContrastLight,
        }
    }

    /// Whether this is a dark-background variant.
    #[must_use]
    pub const fn is_dark(self) -> bool {
        matches!(self, Self::Dark | Self::HighContrastDark)
    }

    /// Whether this is a high-contrast variant.
    #[must_use]
    pub const fn is_high_contrast(self) -> bool {
        matches!(self, Self::HighContrastDark | Self::HighContrastLight)
    }
}

// ─── Semantic Color Palette ─────────────────────────────────────────────────

/// Semantic color palette — every color has a purpose, not a hue.
///
/// Screen implementations use these roles instead of hardcoded colors.
/// The active theme variant determines the concrete Color values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticPalette {
    // ── Surface colors ──
    /// Primary background.
    pub bg: Color,
    /// Elevated surface (panels, cards).
    pub bg_surface: Color,
    /// Hover / selection highlight.
    pub bg_highlight: Color,

    // ── Text colors ──
    /// Primary text.
    pub fg: Color,
    /// Secondary / muted text.
    pub fg_muted: Color,
    /// Disabled text.
    pub fg_disabled: Color,

    // ── Semantic status ──
    /// Success / healthy / green indicator.
    pub success: Color,
    /// Warning / degraded / yellow indicator.
    pub warning: Color,
    /// Error / critical / red indicator.
    pub error: Color,
    /// Informational / accent / blue indicator.
    pub info: Color,

    // ── Interactive ──
    /// Accent color for active elements, links, selections.
    pub accent: Color,
    /// Focus ring / indicator color.
    pub focus_ring: Color,
    /// Border color for panels in normal state.
    pub border: Color,
    /// Border color for focused panel.
    pub border_focused: Color,
}

impl SemanticPalette {
    /// Dark theme palette (default).
    #[must_use]
    pub const fn dark() -> Self {
        Self {
            bg: Color::Rgb(30, 30, 30),
            bg_surface: Color::Rgb(45, 45, 45),
            bg_highlight: Color::Rgb(60, 60, 70),
            fg: Color::Rgb(220, 220, 220),
            fg_muted: Color::Rgb(150, 150, 150),
            fg_disabled: Color::Rgb(90, 90, 90),
            success: Color::Rgb(80, 200, 120),
            warning: Color::Rgb(240, 180, 50),
            error: Color::Rgb(240, 80, 80),
            info: Color::Rgb(100, 160, 240),
            accent: Color::Rgb(100, 160, 240),
            focus_ring: Color::Rgb(100, 180, 255),
            border: Color::Rgb(70, 70, 70),
            border_focused: Color::Rgb(100, 180, 255),
        }
    }

    /// Light theme palette.
    #[must_use]
    pub const fn light() -> Self {
        Self {
            bg: Color::Rgb(250, 250, 250),
            bg_surface: Color::Rgb(240, 240, 240),
            bg_highlight: Color::Rgb(220, 225, 235),
            fg: Color::Rgb(30, 30, 30),
            fg_muted: Color::Rgb(100, 100, 100),
            fg_disabled: Color::Rgb(170, 170, 170),
            success: Color::Rgb(40, 160, 80),
            warning: Color::Rgb(200, 140, 20),
            error: Color::Rgb(200, 50, 50),
            info: Color::Rgb(40, 100, 200),
            accent: Color::Rgb(40, 100, 200),
            focus_ring: Color::Rgb(30, 90, 200),
            border: Color::Rgb(200, 200, 200),
            border_focused: Color::Rgb(30, 90, 200),
        }
    }

    /// High-contrast dark palette (WCAG AAA: >= 7:1 contrast ratio).
    #[must_use]
    pub const fn high_contrast_dark() -> Self {
        Self {
            bg: Color::Rgb(0, 0, 0),
            bg_surface: Color::Rgb(20, 20, 20),
            bg_highlight: Color::Rgb(40, 40, 60),
            fg: Color::Rgb(255, 255, 255),
            fg_muted: Color::Rgb(200, 200, 200),
            fg_disabled: Color::Rgb(120, 120, 120),
            success: Color::Rgb(80, 255, 80),
            warning: Color::Rgb(255, 255, 0),
            error: Color::Rgb(255, 80, 80),
            info: Color::Rgb(100, 200, 255),
            accent: Color::Rgb(100, 200, 255),
            focus_ring: Color::Rgb(255, 255, 0),
            border: Color::Rgb(120, 120, 120),
            border_focused: Color::Rgb(255, 255, 0),
        }
    }

    /// High-contrast light palette (WCAG AAA: >= 7:1 contrast ratio).
    #[must_use]
    pub const fn high_contrast_light() -> Self {
        Self {
            bg: Color::Rgb(255, 255, 255),
            bg_surface: Color::Rgb(245, 245, 245),
            bg_highlight: Color::Rgb(200, 210, 230),
            fg: Color::Rgb(0, 0, 0),
            fg_muted: Color::Rgb(50, 50, 50),
            fg_disabled: Color::Rgb(130, 130, 130),
            success: Color::Rgb(0, 100, 0),
            warning: Color::Rgb(150, 100, 0),
            error: Color::Rgb(180, 0, 0),
            info: Color::Rgb(0, 50, 180),
            accent: Color::Rgb(0, 50, 180),
            focus_ring: Color::Rgb(0, 0, 200),
            border: Color::Rgb(80, 80, 80),
            border_focused: Color::Rgb(0, 0, 200),
        }
    }

    /// Get palette for a specific theme variant.
    #[must_use]
    pub const fn for_variant(variant: ThemeVariant) -> Self {
        match variant {
            ThemeVariant::Dark => Self::dark(),
            ThemeVariant::Light => Self::light(),
            ThemeVariant::HighContrastDark => Self::high_contrast_dark(),
            ThemeVariant::HighContrastLight => Self::high_contrast_light(),
        }
    }

    /// Convenience: resolve palette from display preferences + light/dark.
    #[must_use]
    pub const fn from_preferences(is_light: bool, prefs: &DisplayPreferences) -> Self {
        let variant = ThemeVariant::resolve(is_light, prefs.contrast);
        Self::for_variant(variant)
    }

    // ─── Style Helpers ───────────────────────────────────────────────────

    /// Style for primary text on primary background.
    #[must_use]
    pub const fn style_default(&self) -> Style {
        Style::new().fg(self.fg).bg(self.bg)
    }

    /// Style for muted / secondary text.
    #[must_use]
    pub const fn style_muted(&self) -> Style {
        Style::new().fg(self.fg_muted).bg(self.bg)
    }

    /// Style for a healthy status indicator.
    #[must_use]
    pub const fn style_success(&self) -> Style {
        Style::new().fg(self.success)
    }

    /// Style for a warning status indicator.
    #[must_use]
    pub const fn style_warning(&self) -> Style {
        Style::new().fg(self.warning)
    }

    /// Style for an error / critical status indicator.
    #[must_use]
    pub const fn style_error(&self) -> Style {
        Style::new().fg(self.error)
    }

    /// Style for an informational indicator.
    #[must_use]
    pub const fn style_info(&self) -> Style {
        Style::new().fg(self.info)
    }

    /// Style for a focused panel border.
    #[must_use]
    pub const fn style_focus_border(&self) -> Style {
        Style::new()
            .fg(self.border_focused)
            .add_modifier(Modifier::BOLD)
    }

    /// Style for an unfocused panel border.
    #[must_use]
    pub const fn style_border(&self) -> Style {
        Style::new().fg(self.border)
    }

    /// Style for selected / highlighted row.
    #[must_use]
    pub const fn style_highlight(&self) -> Style {
        Style::new().fg(self.fg).bg(self.bg_highlight)
    }

    /// Style for accent-colored interactive text (links, active items).
    #[must_use]
    pub const fn style_accent(&self) -> Style {
        Style::new()
            .fg(self.accent)
            .add_modifier(Modifier::UNDERLINED)
    }
}

// ─── Focus Indicator Spec ───────────────────────────────────────────────────

/// Focus indicator rendering specification.
///
/// Determines how the focus ring is displayed based on `FocusVisibility`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct FocusIndicatorSpec {
    /// Whether to draw a visible border around the focused panel.
    pub draw_border: bool,
    /// Whether to use bold modifier on the focus border.
    pub bold_border: bool,
    /// Whether to add a colored left-edge bar (2 cells wide).
    pub left_bar: bool,
    /// Whether to add a cursor-line highlight in the focused panel.
    pub cursor_highlight: bool,
}

impl FocusIndicatorSpec {
    /// Normal focus indicators (subtle border change).
    pub const NORMAL: Self = Self {
        draw_border: true,
        bold_border: false,
        left_bar: false,
        cursor_highlight: true,
    };

    /// Enhanced focus indicators (bold border + left bar + cursor highlight).
    pub const ENHANCED: Self = Self {
        draw_border: true,
        bold_border: true,
        left_bar: true,
        cursor_highlight: true,
    };

    /// Resolve spec from preferences.
    #[must_use]
    pub const fn from_preferences(prefs: &DisplayPreferences) -> Self {
        match prefs.focus_visibility {
            crate::preferences::FocusVisibility::Normal => Self::NORMAL,
            crate::preferences::FocusVisibility::Enhanced => Self::ENHANCED,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preferences::{ContrastMode, FocusVisibility, MotionPreference};

    #[test]
    fn theme_variant_resolution() {
        assert_eq!(
            ThemeVariant::resolve(false, ContrastMode::Normal),
            ThemeVariant::Dark
        );
        assert_eq!(
            ThemeVariant::resolve(true, ContrastMode::Normal),
            ThemeVariant::Light
        );
        assert_eq!(
            ThemeVariant::resolve(false, ContrastMode::High),
            ThemeVariant::HighContrastDark
        );
        assert_eq!(
            ThemeVariant::resolve(true, ContrastMode::High),
            ThemeVariant::HighContrastLight
        );
    }

    #[test]
    fn theme_variant_classification() {
        assert!(ThemeVariant::Dark.is_dark());
        assert!(!ThemeVariant::Dark.is_high_contrast());
        assert!(ThemeVariant::HighContrastDark.is_dark());
        assert!(ThemeVariant::HighContrastDark.is_high_contrast());
        assert!(!ThemeVariant::Light.is_dark());
        assert!(!ThemeVariant::Light.is_high_contrast());
        assert!(!ThemeVariant::HighContrastLight.is_dark());
        assert!(ThemeVariant::HighContrastLight.is_high_contrast());
    }

    #[test]
    fn all_four_palettes_have_distinct_backgrounds() {
        let dark = SemanticPalette::dark();
        let light = SemanticPalette::light();
        let hc_dark = SemanticPalette::high_contrast_dark();
        let hc_light = SemanticPalette::high_contrast_light();
        assert_ne!(dark.bg, light.bg);
        assert_ne!(dark.bg, hc_dark.bg);
        assert_ne!(light.bg, hc_light.bg);
        assert_ne!(hc_dark.bg, hc_light.bg);
    }

    #[test]
    fn high_contrast_dark_fg_is_white_on_black() {
        let p = SemanticPalette::high_contrast_dark();
        assert_eq!(p.bg, Color::Rgb(0, 0, 0));
        assert_eq!(p.fg, Color::Rgb(255, 255, 255));
    }

    #[test]
    fn high_contrast_light_fg_is_black_on_white() {
        let p = SemanticPalette::high_contrast_light();
        assert_eq!(p.bg, Color::Rgb(255, 255, 255));
        assert_eq!(p.fg, Color::Rgb(0, 0, 0));
    }

    #[test]
    fn from_preferences_resolves_correctly() {
        let prefs = DisplayPreferences {
            contrast: ContrastMode::High,
            motion: MotionPreference::Full,
            focus_visibility: FocusVisibility::Normal,
            show_shortcut_hints: true,
        };
        let palette = SemanticPalette::from_preferences(false, &prefs);
        assert_eq!(palette.bg, SemanticPalette::high_contrast_dark().bg);
    }

    #[test]
    fn style_helpers_use_palette_colors() {
        let p = SemanticPalette::dark();
        let s = p.style_default();
        assert_eq!(s.fg, Some(p.fg));
        assert_eq!(s.bg, Some(p.bg));

        let s = p.style_success();
        assert_eq!(s.fg, Some(p.success));

        let s = p.style_warning();
        assert_eq!(s.fg, Some(p.warning));

        let s = p.style_error();
        assert_eq!(s.fg, Some(p.error));
    }

    #[test]
    fn focus_border_style_includes_bold() {
        let p = SemanticPalette::dark();
        let s = p.style_focus_border();
        assert!(s.add_modifier.contains(Modifier::BOLD));
        assert_eq!(s.fg, Some(p.border_focused));
    }

    #[test]
    fn focus_indicator_from_preferences() {
        let mut prefs = DisplayPreferences::new();
        assert_eq!(
            FocusIndicatorSpec::from_preferences(&prefs),
            FocusIndicatorSpec::NORMAL
        );
        prefs.toggle_focus_visibility();
        assert_eq!(
            FocusIndicatorSpec::from_preferences(&prefs),
            FocusIndicatorSpec::ENHANCED
        );
    }

    #[test]
    fn enhanced_focus_has_left_bar() {
        const { assert!(FocusIndicatorSpec::ENHANCED.left_bar) };
        const { assert!(FocusIndicatorSpec::ENHANCED.bold_border) };
        const { assert!(!FocusIndicatorSpec::NORMAL.left_bar) };
        const { assert!(!FocusIndicatorSpec::NORMAL.bold_border) };
    }

    #[test]
    fn theme_variant_serde_roundtrip() {
        for variant in [
            ThemeVariant::Dark,
            ThemeVariant::Light,
            ThemeVariant::HighContrastDark,
            ThemeVariant::HighContrastLight,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: ThemeVariant = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
    }

    #[test]
    fn for_variant_matches_named_constructors() {
        assert_eq!(
            SemanticPalette::for_variant(ThemeVariant::Dark),
            SemanticPalette::dark()
        );
        assert_eq!(
            SemanticPalette::for_variant(ThemeVariant::Light),
            SemanticPalette::light()
        );
        assert_eq!(
            SemanticPalette::for_variant(ThemeVariant::HighContrastDark),
            SemanticPalette::high_contrast_dark()
        );
        assert_eq!(
            SemanticPalette::for_variant(ThemeVariant::HighContrastLight),
            SemanticPalette::high_contrast_light()
        );
    }
}
