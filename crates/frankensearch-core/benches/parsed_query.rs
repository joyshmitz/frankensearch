//! Negation-query parse benchmark.
//!
//! `ParsedQuery::parse` runs per search query (the searcher parses for `-term` /
//! `NOT "phrase"` negations). The committed parser always materialized a
//! `Vec<char>` and re-collected each word with `chars[a..b].iter().collect()`.
//! Most queries contain no negation-syntax chars (`-`, `"`, `\`), so the new fast
//! path returns the whitespace-normalized input directly (split + `push_str`),
//! skipping the char machinery. This bench is the head-to-head on a plain query
//! (`old` = char-based parse, `new` = fast path).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-core --bench parsed_query
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::parsed_query::ParsedQuery;

/// Prior char-based positive extraction (the plain-query path of the full parser).
fn parse_old(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut parts: Vec<String> = Vec::new();
    while i < len {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        while i < len && !chars[i].is_whitespace() {
            i += 1;
        }
        parts.push(chars[start..i].iter().collect());
    }
    parts.join(" ")
}

/// New fast path: whitespace-normalize the input directly, no `Vec<char>`.
fn parse_new(raw: &str) -> String {
    let mut positive = String::with_capacity(raw.len());
    for word in raw.split_whitespace() {
        if !positive.is_empty() {
            positive.push(' ');
        }
        positive.push_str(word);
    }
    positive
}

/// Prior full parser for negation-capable queries. The key baseline cost is
/// `matches_not_keyword_old`, which allocated a 3-char `String` at each tested
/// token boundary before comparing it with `NOT`.
fn parse_negated_old(raw: &str) -> ParsedQuery {
    let mut positive_parts = Vec::new();
    let mut negative_terms = Vec::new();
    let mut negative_phrases = Vec::new();

    let chars: Vec<char> = raw.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }

        if i + 3 < len && matches_not_keyword_old(&chars, i) {
            let after_not = i + 3;
            let mut j = after_not;
            while j < len && chars[j].is_whitespace() {
                j += 1;
            }
            if j < len && chars[j] == '"' {
                j += 1;
                let start = j;
                while j < len && chars[j] != '"' {
                    j += 1;
                }
                let phrase: String = chars[start..j].iter().collect();
                if !phrase.is_empty() {
                    negative_phrases.push(phrase);
                }
                if j < len {
                    j += 1;
                }
                i = j;
                continue;
            }
        }

        if chars[i] == '\\' && i + 1 < len && chars[i + 1] == '-' {
            i += 1;
            let start = i;
            while i < len && !chars[i].is_whitespace() {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            positive_parts.push(word);
            continue;
        }

        if chars[i] == '-'
            && (i == 0 || chars[i - 1].is_whitespace())
            && i + 1 < len
            && chars[i + 1] != '-'
            && !chars[i + 1].is_whitespace()
        {
            i += 1;
            let start = i;
            while i < len && !chars[i].is_whitespace() {
                i += 1;
            }
            let term: String = chars[start..i].iter().collect();
            if !term.is_empty() {
                negative_terms.push(term);
            }
            continue;
        }

        let start = i;
        while i < len && !chars[i].is_whitespace() {
            i += 1;
        }
        let word: String = chars[start..i].iter().collect();
        positive_parts.push(word);
    }

    ParsedQuery {
        positive: positive_parts.join(" "),
        negative_terms,
        negative_phrases,
    }
}

fn matches_not_keyword_old(chars: &[char], i: usize) -> bool {
    let len = chars.len();
    if i + 3 > len {
        return false;
    }
    let word: String = chars[i..i + 3].iter().collect();
    if !word.eq_ignore_ascii_case("NOT") {
        return false;
    }
    if i > 0 && !chars[i - 1].is_whitespace() {
        return false;
    }
    i + 3 >= len || chars[i + 3].is_whitespace()
}

fn bench_parsed_query(c: &mut Criterion) {
    // A plain multi-word query (the common case — no negation syntax).
    let query = "how does the hybrid search ranking actually work in practice here";
    let negated =
        r#"rust async search engine -tokio ranking NOT "deep learning" safe mode NOT "slow path""#;

    debug_assert_eq!(parse_old(query), parse_new(query));
    debug_assert_eq!(parse_negated_old(negated), ParsedQuery::parse(negated));

    let mut g = c.benchmark_group("parsed_query");
    g.bench_with_input("old", query, |b, q| {
        b.iter(|| black_box(parse_old(black_box(q))));
    });
    g.bench_with_input("new", query, |b, q| {
        b.iter(|| black_box(parse_new(black_box(q))));
    });
    g.bench_with_input("not_phrase_old", negated, |b, q| {
        b.iter(|| black_box(parse_negated_old(black_box(q))));
    });
    g.bench_with_input("not_phrase_new", negated, |b, q| {
        b.iter(|| black_box(ParsedQuery::parse(black_box(q))));
    });
    g.finish();
}

criterion_group!(benches, bench_parsed_query);
criterion_main!(benches);
