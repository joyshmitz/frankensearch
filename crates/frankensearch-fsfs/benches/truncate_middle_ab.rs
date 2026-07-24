//! Within-process paired A/B for `truncate_middle` (fsfs runtime path/name display).
//!
//! The old form materialized the ENTIRE input into a `Vec<char>` just to slice a
//! `max_chars`-sized prefix+suffix (O(text) alloc + fill, called at ~21 render sites).
//! The new form walks at most ~`max_chars` chars (bounded overflow check + `char_indices`
//! for the prefix + `char_indices().rev()` for the suffix) with no `Vec<char>` alloc, so
//! it is `O(max_chars)` regardless of input length. Both arms must produce byte-identical
//! strings. Run both in one process (immune to the `RCH_WORKER` soft-pin) with an A/A null.

use std::hint::black_box;
use std::time::Instant;

/// ORIG: collect the whole string into a Vec<char>, then slice.
fn truncate_middle_orig(text: &str, max_chars: usize) -> String {
    if max_chars < 5 {
        return text.chars().take(max_chars).collect();
    }
    let chars = text.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return text.to_owned();
    }
    let left = (max_chars - 3) / 2;
    let right = max_chars - 3 - left;
    let prefix = chars.iter().take(left).collect::<String>();
    let suffix = chars
        .iter()
        .skip(chars.len().saturating_sub(right))
        .collect::<String>();
    format!("{prefix}...{suffix}")
}

/// NEW: bounded walks, no Vec<char> alloc.
fn truncate_middle_new(text: &str, max_chars: usize) -> String {
    if max_chars < 5 {
        return text.chars().take(max_chars).collect();
    }
    if text.chars().nth(max_chars).is_none() {
        return text.to_owned();
    }
    let left = (max_chars - 3) / 2;
    let right = max_chars - 3 - left;
    let left_end = text.char_indices().nth(left).map_or(text.len(), |(i, _)| i);
    let suffix_start = text
        .char_indices()
        .rev()
        .nth(right - 1)
        .map_or(0, |(i, _)| i);
    format!("{}...{}", &text[..left_end], &text[suffix_start..])
}

fn time_many(iters: usize, text: &str, max_chars: usize, f: fn(&str, usize) -> String) -> f64 {
    let start = Instant::now();
    let mut acc = 0usize;
    for _ in 0..iters {
        acc = acc.wrapping_add(black_box(f(black_box(text), max_chars)).len());
    }
    black_box(acc);
    start.elapsed().as_secs_f64() / iters as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
fn p5(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 20]
}

fn main() {
    // Byte-identity across short/exact/long, ASCII + Unicode, and various max_chars.
    let ascii_path = "/data/projects/frankensearch/crates/frankensearch-fsfs/src/runtime.rs";
    let unicode = "café/naïve/Straße/日本語/ファイル/emoji-😀-mix/παράδειγμα/директория/end";
    let cases: &[(&str, usize)] = &[
        ("", 80),
        ("short", 80),
        ("abcd", 3),
        ("exactly-five", 12),
        (ascii_path, 40),
        (ascii_path, 80),
        (ascii_path, 5),
        (unicode, 20),
        (unicode, 40),
        (unicode, 7),
        ("aaaaaaaaaa", 6),
    ];
    for (text, mc) in cases {
        let a = truncate_middle_orig(text, *mc);
        let b = truncate_middle_new(text, *mc);
        assert_eq!(a, b, "MISMATCH text={text:?} max_chars={mc}");
    }
    // A long input (thousands of chars) truncated small — the win case.
    let long: String = "x/very-long-path-segment".repeat(300); // ~7200 chars
    assert_eq!(
        truncate_middle_orig(&long, 80),
        truncate_middle_new(&long, 80),
        "MISMATCH long"
    );
    println!(
        "[sanity] arms byte-identical across {} cases + long input",
        cases.len()
    );

    let iters = 4000usize;
    let rounds = 50usize;
    // Two regimes: a typical path (len ~72, mc 60) and a long value (len ~7200, mc 80).
    for (label, text, mc) in [
        ("path72", ascii_path.to_string(), 60usize),
        ("long7200", long, 80usize),
    ] {
        let mut orig = Vec::new();
        let mut new = Vec::new();
        let mut na = Vec::new();
        let mut nb = Vec::new();
        for _ in 0..rounds {
            orig.push(time_many(iters, &text, mc, truncate_middle_orig));
            new.push(time_many(iters, &text, mc, truncate_middle_new));
            na.push(time_many(iters, &text, mc, truncate_middle_new));
            nb.push(time_many(iters, &text, mc, truncate_middle_new));
        }
        let m_orig = median(orig);
        let m_new = median(new);
        let null_p5 = p5(na.iter().zip(&nb).map(|(a, b)| b / a).collect());
        let ratio = m_new / m_orig;
        println!(
            "[{label:>8}] orig {:>9.1} ns  new {:>9.1} ns  ratio {ratio:.4}  null_p5 {null_p5:.4}  -> {}",
            m_orig * 1e9,
            m_new * 1e9,
            if ratio < null_p5 {
                "CANDIDATE_FASTER"
            } else {
                "INCONCLUSIVE"
            }
        );
    }
}
