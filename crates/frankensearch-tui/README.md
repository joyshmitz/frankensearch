# frankensearch-tui

Shared TUI framework for frankensearch products.

## Overview

This crate provides reusable terminal UI primitives shared by both the `fsfs` deluxe TUI and the `frankensearch-ops` observability TUI. It ensures consistent UX, keyboard shortcuts, theming, and accessibility across all frankensearch TUI products. It sits between the product crates and the low-level FrankenTUI (`ftui-*`) rendering library.

```text
+-------------------------------------------+
| Product crates (fsfs, ops)                |
|   product-specific screens + data sources |
+-------------------------------------------+
| frankensearch-tui (this crate)            |
|   Screen, AppShell, Keymap, Theme, ...    |
+-------------------------------------------+
| FrankenTUI (ftui-*)                       |
+-------------------------------------------+
```

## Key Types

### Screen System

- `Screen` - trait that product screens implement for rendering and input handling
- `ScreenId` - typed screen identifier
- `ScreenRegistry` - registry of available screens for navigation
- `ScreenContext` - context passed to screens during rendering

### Application Shell

- `AppShell` - top-level app shell managing navigation, overlays, input dispatch, and frame timing
- `ShellConfig` - shell configuration (title, initial screen, etc.)
- `StatusLine` - status bar content model

### Input and Keybindings

- `InputEvent` - unified input event (key, mouse, resize)
- `KeyAction` / `KeyBinding` - key action definitions and bindings
- `Keymap` - configurable keymap with serialization support

### Command Palette

- `CommandPalette` - fuzzy-searchable command palette overlay
- `Action` / `ActionCategory` - palette action definitions and groupings
- `PaletteState` - current palette filter and selection state

### Overlay System

- `OverlayManager` - manages modal overlays (help, alerts, confirmations)
- `OverlayKind` / `OverlayRequest` - overlay type and request model

### Theming

- `Theme` / `ThemePreset` - theme definitions with dark/light presets
- `ColorScheme` - semantic color assignments

### Accessibility

- `FocusManager` / `FocusDirection` - keyboard focus traversal
- `SemanticRole` - semantic annotations for screen regions

### Frame Budget

- `FrameBudget` / `FrameMetrics` - frame time budget enforcement and jank detection
- `FramePipelineTimer` - per-phase frame timing
- `CachedLayout` / `CachedTabState` - layout caching for render optimization

### Replay and Determinism

- `ReplayRecorder` / `ReplayPlayer` - input recording and deterministic replay
- `Clock` / `WallClock` / `TickClock` - clock abstraction for deterministic testing
- `DeterministicSeed` - seeded RNG for reproducible behavior

### Evidence

- `EvidenceSink` / `EvidenceEnvelope` - JSONL evidence emission with redaction
- `EvidenceEvent` / `EvidenceSeverity` - structured evidence events

### Terminal

- `TerminalState` / `TerminalMode` - terminal mode detection and state
- `TerminalEvent` - terminal lifecycle events (resize, reconnect)

## Usage

```rust
use frankensearch_tui::{Screen, ScreenId, ScreenRegistry, AppShell, ShellConfig};

// Implement the Screen trait for your product view
// struct MyScreen { ... }
// impl Screen for MyScreen { ... }

// Register screens and run the app shell
// let mut registry = ScreenRegistry::new();
// registry.register(ScreenId::new("main"), Box::new(MyScreen::new()));
// let shell = AppShell::new(ShellConfig::default(), registry);
```

## Dependency Graph Position

```
frankensearch-core
  ^
  |
frankensearch-tui
  ^
  |-- frankensearch-fsfs
  |-- frankensearch-ops
```

## License

MIT
