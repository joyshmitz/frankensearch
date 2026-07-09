//! Keymap resolution benchmark.
//!
//! LEGACY ORIGINAL stored the default keymap entirely in `HashMap<(KeyCode,
//! Modifiers), KeyAction>` and paid a hash probe for every key event, including
//! normal text-input misses. The production candidate dispatches the unmodified
//! default map through a static action table and falls back to the `HashMap` after
//! any custom bind/unbind.
//!
//! Run with:
//! ```bash
//! AGENT_NAME=SearchCod \
//! CARGO_TARGET_DIR=/data/projects/frankensearch/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-tui --profile release --bench keymap_resolve
//! ```

use std::{collections::HashMap, hint::black_box};

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_tui::{KeyAction, Keymap};
use ftui_core::event::{KeyCode, Modifiers};

fn legacy_default_bindings() -> HashMap<(KeyCode, Modifiers), KeyAction> {
    let mut bindings = HashMap::new();

    bindings.insert((KeyCode::Char('q'), Modifiers::NONE), KeyAction::Quit);
    bindings.insert((KeyCode::Char('c'), Modifiers::CTRL), KeyAction::Quit);
    bindings.insert(
        (KeyCode::Char('p'), Modifiers::CTRL),
        KeyAction::TogglePalette,
    );
    bindings.insert(
        (KeyCode::Char(':'), Modifiers::NONE),
        KeyAction::TogglePalette,
    );
    bindings.insert((KeyCode::Tab, Modifiers::NONE), KeyAction::NextScreen);
    bindings.insert((KeyCode::BackTab, Modifiers::SHIFT), KeyAction::PrevScreen);
    bindings.insert((KeyCode::Char('?'), Modifiers::NONE), KeyAction::ToggleHelp);
    bindings.insert((KeyCode::F(1), Modifiers::NONE), KeyAction::ToggleHelp);
    bindings.insert((KeyCode::Escape, Modifiers::NONE), KeyAction::Dismiss);
    bindings.insert((KeyCode::Up, Modifiers::NONE), KeyAction::Up);
    bindings.insert((KeyCode::Down, Modifiers::NONE), KeyAction::Down);
    bindings.insert((KeyCode::Left, Modifiers::NONE), KeyAction::Left);
    bindings.insert((KeyCode::Right, Modifiers::NONE), KeyAction::Right);
    bindings.insert((KeyCode::Char('k'), Modifiers::NONE), KeyAction::Up);
    bindings.insert((KeyCode::Char('j'), Modifiers::NONE), KeyAction::Down);
    bindings.insert((KeyCode::Char('h'), Modifiers::NONE), KeyAction::Left);
    bindings.insert((KeyCode::Char('l'), Modifiers::NONE), KeyAction::Right);
    bindings.insert((KeyCode::PageUp, Modifiers::NONE), KeyAction::PageUp);
    bindings.insert((KeyCode::PageDown, Modifiers::NONE), KeyAction::PageDown);
    bindings.insert((KeyCode::Home, Modifiers::NONE), KeyAction::Home);
    bindings.insert((KeyCode::End, Modifiers::NONE), KeyAction::End);
    bindings.insert((KeyCode::Char('t'), Modifiers::CTRL), KeyAction::CycleTheme);
    bindings.insert((KeyCode::Enter, Modifiers::NONE), KeyAction::Confirm);
    bindings.insert((KeyCode::Backspace, Modifiers::NONE), KeyAction::Delete);
    bindings.insert((KeyCode::Char('y'), Modifiers::CTRL), KeyAction::Copy);

    bindings
}

fn legacy_resolve(
    bindings: &HashMap<(KeyCode, Modifiers), KeyAction>,
    key: KeyCode,
    modifiers: Modifiers,
) -> Option<&KeyAction> {
    bindings.get(&(key, modifiers))
}

fn key_workload() -> [(KeyCode, Modifiers); 40] {
    [
        (KeyCode::Char('j'), Modifiers::NONE),
        (KeyCode::Char('k'), Modifiers::NONE),
        (KeyCode::Down, Modifiers::NONE),
        (KeyCode::Up, Modifiers::NONE),
        (KeyCode::Enter, Modifiers::NONE),
        (KeyCode::Escape, Modifiers::NONE),
        (KeyCode::Char('p'), Modifiers::CTRL),
        (KeyCode::Char(':'), Modifiers::NONE),
        (KeyCode::Char('q'), Modifiers::NONE),
        (KeyCode::Char('c'), Modifiers::CTRL),
        (KeyCode::Tab, Modifiers::NONE),
        (KeyCode::BackTab, Modifiers::SHIFT),
        (KeyCode::Char('?'), Modifiers::NONE),
        (KeyCode::F(1), Modifiers::NONE),
        (KeyCode::Left, Modifiers::NONE),
        (KeyCode::Right, Modifiers::NONE),
        (KeyCode::PageUp, Modifiers::NONE),
        (KeyCode::PageDown, Modifiers::NONE),
        (KeyCode::Home, Modifiers::NONE),
        (KeyCode::End, Modifiers::NONE),
        (KeyCode::Char('t'), Modifiers::CTRL),
        (KeyCode::Backspace, Modifiers::NONE),
        (KeyCode::Char('y'), Modifiers::CTRL),
        (KeyCode::Char('a'), Modifiers::NONE),
        (KeyCode::Char('e'), Modifiers::NONE),
        (KeyCode::Char('s'), Modifiers::NONE),
        (KeyCode::Char('r'), Modifiers::NONE),
        (KeyCode::Char('u'), Modifiers::NONE),
        (KeyCode::Char('n'), Modifiers::NONE),
        (KeyCode::Char('x'), Modifiers::NONE),
        (KeyCode::Char('q'), Modifiers::CTRL),
        (KeyCode::Char('j'), Modifiers::CTRL),
        (KeyCode::Char('p'), Modifiers::NONE),
        (KeyCode::Char('c'), Modifiers::NONE),
        (KeyCode::Delete, Modifiers::NONE),
        (KeyCode::Insert, Modifiers::NONE),
        (KeyCode::Null, Modifiers::CTRL),
        (KeyCode::F(9), Modifiers::NONE),
        (KeyCode::MediaPlayPause, Modifiers::NONE),
        (KeyCode::Char('z'), Modifiers::ALT),
    ]
}

fn action_score(action: Option<&KeyAction>) -> u64 {
    match action {
        Some(KeyAction::Quit) => 1,
        Some(KeyAction::TogglePalette) => 2,
        Some(KeyAction::NextScreen) => 3,
        Some(KeyAction::PrevScreen) => 4,
        Some(KeyAction::ToggleHelp) => 5,
        Some(KeyAction::CycleTheme) => 6,
        Some(KeyAction::Dismiss) => 7,
        Some(KeyAction::Up) => 8,
        Some(KeyAction::Down) => 9,
        Some(KeyAction::Left) => 10,
        Some(KeyAction::Right) => 11,
        Some(KeyAction::PageUp) => 12,
        Some(KeyAction::PageDown) => 13,
        Some(KeyAction::Home) => 14,
        Some(KeyAction::End) => 15,
        Some(KeyAction::Confirm) => 16,
        Some(KeyAction::Delete) => 17,
        Some(KeyAction::Copy) => 18,
        Some(KeyAction::Custom(_)) => 19,
        None => 0,
    }
}

fn bench_keymap_resolve(c: &mut Criterion) {
    let legacy = legacy_default_bindings();
    let keymap = Keymap::default_bindings();
    let workload = key_workload();

    for &(key, modifiers) in &workload {
        assert_eq!(
            legacy_resolve(&legacy, key, modifiers),
            keymap.resolve(key, modifiers)
        );
    }

    let mut group = c.benchmark_group("tui_keymap_resolve");
    group.bench_function("legacy_hashmap_ORIG", |b| {
        b.iter(|| {
            let mut score = 0_u64;
            for &(key, modifiers) in &workload {
                score = score.saturating_add(action_score(legacy_resolve(
                    black_box(&legacy),
                    black_box(key),
                    black_box(modifiers),
                )));
            }
            black_box(score)
        });
    });
    group.bench_function("static_dispatch", |b| {
        b.iter(|| {
            let mut score = 0_u64;
            for &(key, modifiers) in &workload {
                score = score.saturating_add(action_score(
                    keymap.resolve(black_box(key), black_box(modifiers)),
                ));
            }
            black_box(score)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_keymap_resolve);
criterion_main!(benches);
