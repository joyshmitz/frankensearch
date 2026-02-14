//! Screen categories for ops TUI navigation.
//!
//! Screens are grouped into categories for tab-based navigation.
//! Each category maps to a section of the status bar / breadcrumbs.

use serde::{Deserialize, Serialize};

/// Category grouping for ops TUI screens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreenCategory {
    /// Fleet-wide overview and instance discovery.
    Fleet,
    /// Live search stream and query analytics.
    Search,
    /// Index status, embedding progress, staleness.
    Index,
    /// CPU, memory, I/O resource monitoring.
    Resource,
    /// Historical analytics and explainability.
    Analytics,
    /// Configuration and settings.
    Settings,
}

impl ScreenCategory {
    /// All categories in display order.
    pub const ALL: &'static [Self] = &[
        Self::Fleet,
        Self::Search,
        Self::Index,
        Self::Resource,
        Self::Analytics,
        Self::Settings,
    ];

    /// Short label for the status bar.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Fleet => "Fleet",
            Self::Search => "Search",
            Self::Index => "Index",
            Self::Resource => "Resources",
            Self::Analytics => "Analytics",
            Self::Settings => "Settings",
        }
    }

    /// Icon hint (single char) for compact display.
    #[must_use]
    pub const fn icon(self) -> char {
        match self {
            Self::Fleet => 'F',
            Self::Search => 'S',
            Self::Index => 'I',
            Self::Resource => 'R',
            Self::Analytics => 'A',
            Self::Settings => 'C',
        }
    }
}

impl std::fmt::Display for ScreenCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_categories_covered() {
        assert_eq!(ScreenCategory::ALL.len(), 6);
    }

    #[test]
    fn category_display() {
        assert_eq!(ScreenCategory::Fleet.to_string(), "Fleet");
        assert_eq!(ScreenCategory::Analytics.to_string(), "Analytics");
    }

    #[test]
    fn category_serde_roundtrip() {
        for cat in ScreenCategory::ALL {
            let json = serde_json::to_string(cat).unwrap();
            let decoded: ScreenCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, *cat);
        }
    }

    #[test]
    fn category_labels_nonempty() {
        for cat in ScreenCategory::ALL {
            assert!(!cat.label().is_empty());
        }
    }
}
