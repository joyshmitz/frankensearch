//! Unified input model: keymap, bindings, mouse support.
//!
//! Provides a configurable keymap that maps terminal events to semantic
//! [`KeyAction`] values. Product crates extend the action set; the shell
//! handles navigation-level actions (quit, tab switch, palette toggle).

use std::collections::HashMap;

use ftui_core::event::{KeyCode, Modifiers, MouseEventKind};
use serde::{Deserialize, Serialize};

// ─── Input Event Abstraction ────────────────────────────────────────────────

/// High-level input event consumed by screens and the shell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputEvent {
    /// A key press with modifiers.
    Key(KeyCode, Modifiers),
    /// A mouse event at a position.
    Mouse(MouseEventKind, u16, u16),
    /// Terminal resize.
    Resize(u16, u16),
    /// A resolved semantic action (after keymap lookup).
    Action(KeyAction),
}

// ─── Semantic Key Actions ───────────────────────────────────────────────────

/// Semantic action resolved from key bindings.
///
/// Shell-level actions are handled by the app shell. Screen-level actions
/// are forwarded to the active screen. Product crates can define custom
/// actions using the `Custom` variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyAction {
    // ── Shell-level ─────────────────────────────────────────────────
    /// Quit the application.
    Quit,
    /// Toggle the command palette.
    TogglePalette,
    /// Navigate to the next screen (tab).
    NextScreen,
    /// Navigate to the previous screen (shift-tab).
    PrevScreen,
    /// Toggle help overlay.
    ToggleHelp,
    /// Cycle to the next theme preset.
    CycleTheme,
    /// Dismiss current overlay / cancel.
    Dismiss,

    // ── Navigation ──────────────────────────────────────────────────
    /// Move focus up.
    Up,
    /// Move focus down.
    Down,
    /// Move focus left.
    Left,
    /// Move focus right.
    Right,
    /// Page up.
    PageUp,
    /// Page down.
    PageDown,
    /// Go to first item.
    Home,
    /// Go to last item.
    End,

    // ── Interaction ─────────────────────────────────────────────────
    /// Confirm / select / enter.
    Confirm,
    /// Delete / backspace.
    Delete,
    /// Copy to clipboard.
    Copy,

    // ── Product-specific ────────────────────────────────────────────
    /// Custom action defined by product crates.
    Custom(String),
}

// ─── Key Binding ────────────────────────────────────────────────────────────

/// A key binding maps a key+modifier combination to a semantic action.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyBinding {
    /// The key code.
    pub key: String,
    /// Modifier keys (ctrl, alt, shift).
    pub modifiers: Vec<String>,
    /// The action this binding triggers.
    pub action: KeyAction,
}

// ─── Keymap ─────────────────────────────────────────────────────────────────

/// Configurable keymap that resolves key events to semantic actions.
pub struct Keymap {
    bindings: HashMap<(KeyCode, Modifiers), KeyAction>,
}

impl Keymap {
    /// Create a keymap with the default bindings.
    #[must_use]
    pub fn default_bindings() -> Self {
        let mut bindings = HashMap::new();

        // Quit
        bindings.insert((KeyCode::Char('q'), Modifiers::NONE), KeyAction::Quit);
        bindings.insert((KeyCode::Char('c'), Modifiers::CTRL), KeyAction::Quit);

        // Command palette
        bindings.insert(
            (KeyCode::Char('p'), Modifiers::CTRL),
            KeyAction::TogglePalette,
        );
        bindings.insert(
            (KeyCode::Char(':'), Modifiers::NONE),
            KeyAction::TogglePalette,
        );

        // Navigation
        bindings.insert((KeyCode::Tab, Modifiers::NONE), KeyAction::NextScreen);
        bindings.insert((KeyCode::BackTab, Modifiers::SHIFT), KeyAction::PrevScreen);

        // Help
        bindings.insert((KeyCode::Char('?'), Modifiers::NONE), KeyAction::ToggleHelp);
        bindings.insert((KeyCode::F(1), Modifiers::NONE), KeyAction::ToggleHelp);

        // Dismiss
        bindings.insert((KeyCode::Escape, Modifiers::NONE), KeyAction::Dismiss);

        // Movement
        bindings.insert((KeyCode::Up, Modifiers::NONE), KeyAction::Up);
        bindings.insert((KeyCode::Down, Modifiers::NONE), KeyAction::Down);
        bindings.insert((KeyCode::Left, Modifiers::NONE), KeyAction::Left);
        bindings.insert((KeyCode::Right, Modifiers::NONE), KeyAction::Right);
        bindings.insert((KeyCode::Char('k'), Modifiers::NONE), KeyAction::Up);
        bindings.insert((KeyCode::Char('j'), Modifiers::NONE), KeyAction::Down);
        bindings.insert((KeyCode::Char('h'), Modifiers::NONE), KeyAction::Left);
        bindings.insert((KeyCode::Char('l'), Modifiers::NONE), KeyAction::Right);

        // Page navigation
        bindings.insert((KeyCode::PageUp, Modifiers::NONE), KeyAction::PageUp);
        bindings.insert((KeyCode::PageDown, Modifiers::NONE), KeyAction::PageDown);
        bindings.insert((KeyCode::Home, Modifiers::NONE), KeyAction::Home);
        bindings.insert((KeyCode::End, Modifiers::NONE), KeyAction::End);

        // Theme cycling
        bindings.insert((KeyCode::Char('t'), Modifiers::CTRL), KeyAction::CycleTheme);

        // Interaction
        bindings.insert((KeyCode::Enter, Modifiers::NONE), KeyAction::Confirm);
        bindings.insert((KeyCode::Backspace, Modifiers::NONE), KeyAction::Delete);
        bindings.insert((KeyCode::Char('y'), Modifiers::CTRL), KeyAction::Copy);

        Self { bindings }
    }

    /// Resolve a key event to a semantic action.
    #[must_use]
    pub fn resolve(&self, key: KeyCode, modifiers: Modifiers) -> Option<&KeyAction> {
        self.bindings.get(&(key, modifiers))
    }

    /// Add or override a binding.
    pub fn bind(&mut self, key: KeyCode, modifiers: Modifiers, action: KeyAction) {
        self.bindings.insert((key, modifiers), action);
    }

    /// Remove a binding.
    pub fn unbind(&mut self, key: KeyCode, modifiers: Modifiers) {
        self.bindings.remove(&(key, modifiers));
    }

    /// Number of active bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether the keymap is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

impl Default for Keymap {
    fn default() -> Self {
        Self::default_bindings()
    }
}

#[cfg(test)]
mod tests {
    use ftui_core::event::{KeyCode, Modifiers};

    use super::*;

    #[test]
    fn default_keymap_has_bindings() {
        let keymap = Keymap::default_bindings();
        assert!(!keymap.is_empty());
        assert!(keymap.len() > 15);
    }

    #[test]
    fn resolve_quit_q() {
        let keymap = Keymap::default_bindings();
        let action = keymap.resolve(KeyCode::Char('q'), Modifiers::NONE);
        assert_eq!(action, Some(&KeyAction::Quit));
    }

    #[test]
    fn resolve_quit_ctrl_c() {
        let keymap = Keymap::default_bindings();
        let action = keymap.resolve(KeyCode::Char('c'), Modifiers::CTRL);
        assert_eq!(action, Some(&KeyAction::Quit));
    }

    #[test]
    fn resolve_palette_ctrl_p() {
        let keymap = Keymap::default_bindings();
        let action = keymap.resolve(KeyCode::Char('p'), Modifiers::CTRL);
        assert_eq!(action, Some(&KeyAction::TogglePalette));
    }

    #[test]
    fn resolve_vim_movement() {
        let keymap = Keymap::default_bindings();
        assert_eq!(
            keymap.resolve(KeyCode::Char('j'), Modifiers::NONE),
            Some(&KeyAction::Down)
        );
        assert_eq!(
            keymap.resolve(KeyCode::Char('k'), Modifiers::NONE),
            Some(&KeyAction::Up)
        );
    }

    #[test]
    fn resolve_unknown_returns_none() {
        let keymap = Keymap::default_bindings();
        assert!(
            keymap
                .resolve(KeyCode::Char('z'), Modifiers::NONE)
                .is_none()
        );
    }

    #[test]
    fn custom_binding() {
        let mut keymap = Keymap::default_bindings();
        keymap.bind(
            KeyCode::Char('s'),
            Modifiers::CTRL,
            KeyAction::Custom("save".to_string()),
        );
        let action = keymap.resolve(KeyCode::Char('s'), Modifiers::CTRL);
        assert_eq!(action, Some(&KeyAction::Custom("save".to_string())));
    }

    #[test]
    fn unbind_removes_binding() {
        let mut keymap = Keymap::default_bindings();
        assert!(
            keymap
                .resolve(KeyCode::Char('q'), Modifiers::NONE)
                .is_some()
        );
        keymap.unbind(KeyCode::Char('q'), Modifiers::NONE);
        assert!(
            keymap
                .resolve(KeyCode::Char('q'), Modifiers::NONE)
                .is_none()
        );
    }

    #[test]
    fn key_action_serde_roundtrip() {
        for action in [
            KeyAction::Quit,
            KeyAction::TogglePalette,
            KeyAction::CycleTheme,
            KeyAction::Up,
            KeyAction::Custom("test".to_string()),
        ] {
            let json = serde_json::to_string(&action).unwrap();
            let decoded: KeyAction = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, action);
        }
    }
}
