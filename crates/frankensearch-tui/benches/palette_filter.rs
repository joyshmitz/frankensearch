use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use frankensearch_tui::{Action, ActionCategory, CommandPalette};

fn make_actions(count: usize) -> Vec<Action> {
    (0..count)
        .map(|index| {
            let category = match index % 4 {
                0 => ActionCategory::Navigation,
                1 => ActionCategory::Search,
                2 => ActionCategory::Settings,
                _ => ActionCategory::Debug,
            };
            let label = match index % 8 {
                0 => format!("Open vector lane {index}"),
                1 => format!("Inspect index shard {index}"),
                2 => format!("Toggle theme variant {index}"),
                3 => format!("Show structured log stream {index}"),
                4 => format!("Replay evidence frame {index}"),
                5 => format!("Focus result panel {index}"),
                6 => format!("Rebuild ranking surface {index}"),
                _ => format!("Export telemetry bundle {index}"),
            };
            let description = match index % 6 {
                0 => Some(format!(
                    "Vector search command for shard {index} with structured diagnostics"
                )),
                1 => Some(format!(
                    "Navigation action that opens cached result pane {index}"
                )),
                2 => Some(format!(
                    "Settings command for theme and accessibility mode {index}"
                )),
                _ => None,
            };

            let mut action = Action::new(format!("action.{index:04}"), label, category);
            if let Some(description) = description {
                action = action.with_description(description);
            }
            action
        })
        .collect()
}

fn make_palette(actions: &[Action], query: &str) -> CommandPalette {
    let mut palette = CommandPalette::new();
    for action in actions {
        palette.register(action.clone());
    }
    palette.open();
    for ch in query.chars() {
        palette.push_char(ch);
    }
    palette
}

fn legacy_filtered<'a>(actions: &'a [Action], query: &str) -> Vec<&'a Action> {
    if query.is_empty() {
        return actions.iter().collect();
    }

    let query_lower = query.to_lowercase();
    actions
        .iter()
        .filter(|action| {
            action.label.to_lowercase().contains(&query_lower)
                || action.id.to_lowercase().contains(&query_lower)
                || action
                    .description
                    .as_ref()
                    .is_some_and(|description| description.to_lowercase().contains(&query_lower))
        })
        .collect()
}

fn bench_palette_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("palette_filter");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(450));
    group.sample_size(20);

    for action_count in [128_usize, 1_024, 4_096] {
        let actions = make_actions(action_count);
        let query = "vector";
        let palette = make_palette(&actions, query);
        assert_eq!(
            legacy_filtered(&actions, query).len(),
            palette.filtered().len()
        );

        group.throughput(Throughput::Elements(
            u64::try_from(action_count).unwrap_or(u64::MAX),
        ));
        group.bench_with_input(
            BenchmarkId::new("legacy_ORIG", action_count),
            &actions,
            |bench, actions| {
                bench.iter(|| {
                    let filtered = legacy_filtered(black_box(actions), black_box(query));
                    black_box(filtered.len())
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cached_index", action_count),
            &palette,
            |bench, palette| {
                bench.iter(|| {
                    let filtered = palette.filtered();
                    black_box(filtered.len())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_palette_filter);
criterion_main!(benches);
