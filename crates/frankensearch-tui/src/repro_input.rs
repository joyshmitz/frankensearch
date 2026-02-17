
#[cfg(test)]
mod tests {
    use crate::shell::{AppShell, ShellConfig};
    use crate::input::InputEvent;
    use crate::palette::PaletteState;
    use ftui_core::event::{KeyCode, Modifiers};

    #[test]
    fn repro_super_modifier_leaks_into_palette() {
        let mut shell = AppShell::new(ShellConfig::default());
        
        // Open palette
        let open = InputEvent::Key(
            KeyCode::Char('p'),
            Modifiers::CTRL,
        );
        shell.handle_input(&open);
        assert_eq!(shell.palette.state(), &PaletteState::Open);

        // Simulate Cmd+A (Super + 'a')
        let cmd_a = InputEvent::Key(
            KeyCode::Char('a'),
            Modifiers::SUPER,
        );
        shell.handle_input(&cmd_a);

        // We expect this NOT to type 'a' into the palette query.
        // Current behavior (suspected): query becomes "a" because SUPER is not filtered.
        assert_eq!(shell.palette.query(), "", "Super+A should not insert text");
    }
}
