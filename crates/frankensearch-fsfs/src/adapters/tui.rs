use crate::config::{Density, FsfsConfig, TuiTheme};

/// TUI-facing settings derived from resolved fsfs config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiAdapterSettings {
    pub theme: TuiTheme,
    pub density: Density,
    pub show_explanations: bool,
}

impl From<&FsfsConfig> for TuiAdapterSettings {
    fn from(config: &FsfsConfig) -> Self {
        Self {
            theme: config.tui.theme,
            density: config.tui.density,
            show_explanations: config.tui.show_explanations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TuiAdapterSettings;
    use crate::config::{Density, FsfsConfig, TuiTheme};

    #[test]
    fn converts_from_resolved_config() {
        let mut config = FsfsConfig::default();
        config.tui.theme = TuiTheme::Light;
        config.tui.density = Density::Compact;
        config.tui.show_explanations = false;

        let settings = TuiAdapterSettings::from(&config);
        assert_eq!(settings.theme, TuiTheme::Light);
        assert_eq!(settings.density, Density::Compact);
        assert!(!settings.show_explanations);
    }
}
