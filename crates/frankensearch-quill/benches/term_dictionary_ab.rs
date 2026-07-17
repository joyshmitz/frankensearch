//! Same-binary TERMDICT-versus-sorted-`Vec<String>` honesty benchmark.
//!
//! Deterministic field-major rows, section spans, exact probes, and prefix
//! probes are built before timing. Exact results and every ordered scan are
//! asserted equal before Criterion starts. Timed arms reuse exact-lookup
//! scratch, consume every scan result, and compare the opened durable
//! dictionary with a binary-searched sorted vector rather than a hash table.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 RCH_WORKER=<pinned-worker> \
//!   rch exec -- env CARGO_TARGET_DIR=/tmp/frankensearch-quill-termdict \
//!   cargo bench -p frankensearch-quill --features bench-internals \
//!   --bench term_dictionary_ab
//! ```

use std::cmp::Ordering;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use frankensearch_quill::DEFAULT_SCHEMA;
use frankensearch_quill::grimoire::{
    ByteSpan, EncodedTermDictionary, TermCursor, TermDictionary, TermDictionaryError, TermInput,
    TermMatch, TermMetadata, TermScratch, TermSectionLengths,
};

const FIELD_ORDS: [u16; 3] = [0, 1, 2];
const MODULES_PER_FIELD: usize = 96;
const SYMBOLS_PER_MODULE: usize = 64;
const EXACT_QUERY_COUNT: usize = 4_096;
const PREFIXES_PER_FIELD: usize = 16;
const DIGEST_SEED: u64 = 0xcbf2_9ce4_8422_2325;
const DIGEST_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct BaselineTerm {
    term_ord: u32,
    field_ord: u16,
    term: String,
    metadata: TermMetadata,
}

#[derive(Debug)]
struct Query {
    field_ord: u16,
    term: String,
}

#[derive(Debug)]
struct Fixture {
    encoded: EncodedTermDictionary,
    sections: TermSectionLengths,
    baseline: Vec<BaselineTerm>,
    exact_queries: Vec<Query>,
    prefix_queries: Vec<Query>,
}

#[derive(Debug, Eq, PartialEq)]
struct ObservedTerm {
    term_ord: u32,
    field_ord: u16,
    term: Vec<u8>,
    metadata: TermMetadata,
}

fn fail(stage: &str, error: &TermDictionaryError) -> ! {
    eprintln!("term-dictionary-ab {stage} failed: {error}");
    std::process::exit(2);
}

fn require<T>(stage: &str, result: Result<T, TermDictionaryError>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => fail(stage, &error),
    }
}

fn as_u32(stage: &str, value: usize) -> u32 {
    match u32::try_from(value) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("term-dictionary-ab {stage} cannot represent {value} as u32: {error}");
            std::process::exit(2);
        }
    }
}

fn as_u64(stage: &str, value: usize) -> u64 {
    match u64::try_from(value) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("term-dictionary-ab {stage} cannot represent {value} as u64: {error}");
            std::process::exit(2);
        }
    }
}

fn checked_advance(stage: &str, offset: &mut u64, len: u64) -> ByteSpan {
    let span = ByteSpan::new(*offset, len);
    *offset = offset.checked_add(len).unwrap_or_else(|| {
        eprintln!("term-dictionary-ab {stage} offset overflow");
        std::process::exit(2);
    });
    span
}

fn build_fixture() -> Fixture {
    let capacity = FIELD_ORDS
        .len()
        .saturating_mul(MODULES_PER_FIELD)
        .saturating_mul(SYMBOLS_PER_MODULE);
    let mut baseline = Vec::with_capacity(capacity);
    let mut postings_offset = 0_u64;
    let mut positions_offset = 0_u64;
    let mut blockmax_offset = 0_u64;

    for field_ord in FIELD_ORDS {
        for module in 0..MODULES_PER_FIELD {
            for symbol in 0..SYMBOLS_PER_MODULE {
                let ordinal = baseline.len();
                let ordinal_u64 = as_u64("term ordinal", ordinal);
                let postings =
                    checked_advance("POSTINGS", &mut postings_offset, 5 + ordinal_u64 % 11);
                let blockmax =
                    checked_advance("BLOCKMAX", &mut blockmax_offset, 3 + ordinal_u64 % 5);
                let positions = if field_ord == 0 {
                    None
                } else {
                    Some(checked_advance(
                        "POSITIONS",
                        &mut positions_offset,
                        4 + ordinal_u64 % 7,
                    ))
                };
                let doc_freq = as_u32("doc frequency", ordinal % 4_096 + 1);
                let metadata = positions.map_or_else(
                    || TermMetadata::without_positions(doc_freq, postings, blockmax),
                    |positions| {
                        TermMetadata::with_positions(doc_freq, postings, positions, blockmax)
                    },
                );
                baseline.push(BaselineTerm {
                    term_ord: as_u32("term ordinal", ordinal),
                    field_ord,
                    term: format!(
                        "crate::{field_ord:02}::module::{module:04}::symbol::{symbol:04}"
                    ),
                    metadata,
                });
            }
        }
    }

    let sections = TermSectionLengths {
        postings: postings_offset,
        positions: Some(positions_offset),
        blockmax: blockmax_offset,
    };
    let inputs: Vec<_> = baseline
        .iter()
        .map(|row| TermInput::new(row.field_ord, row.term.as_bytes(), row.metadata))
        .collect();
    let encoded = require(
        "encode",
        EncodedTermDictionary::encode_sorted(DEFAULT_SCHEMA, sections, &inputs),
    );

    let mut exact_queries = Vec::with_capacity(EXACT_QUERY_COUNT);
    for index in 0..EXACT_QUERY_COUNT / 2 {
        let field_ord = FIELD_ORDS[index % FIELD_ORDS.len()];
        let module = index.wrapping_mul(37) % MODULES_PER_FIELD;
        let symbol = index.wrapping_mul(17) % SYMBOLS_PER_MODULE;
        exact_queries.push(Query {
            field_ord,
            term: format!("crate::{field_ord:02}::module::{module:04}::symbol::{symbol:04}"),
        });
        exact_queries.push(Query {
            field_ord,
            term: format!("crate::{field_ord:02}::module::{module:04}::symbol::9999"),
        });
    }

    let mut prefix_queries = Vec::with_capacity(FIELD_ORDS.len() * PREFIXES_PER_FIELD);
    for field_ord in FIELD_ORDS {
        for prefix_index in 0..PREFIXES_PER_FIELD {
            let module = prefix_index.wrapping_mul(5) % MODULES_PER_FIELD;
            prefix_queries.push(Query {
                field_ord,
                term: format!("crate::{field_ord:02}::module::{module:04}::"),
            });
        }
    }

    Fixture {
        encoded,
        sections,
        baseline,
        exact_queries,
        prefix_queries,
    }
}

fn compare_row(row: &BaselineTerm, field_ord: u16, term: &[u8]) -> Ordering {
    row.field_ord
        .cmp(&field_ord)
        .then_with(|| row.term.as_bytes().cmp(term))
}

fn baseline_lookup(baseline: &[BaselineTerm], field_ord: u16, term: &[u8]) -> Option<TermMatch> {
    baseline
        .binary_search_by(|row| compare_row(row, field_ord, term))
        .ok()
        .map(|index| TermMatch {
            term_ord: baseline[index].term_ord,
            metadata: baseline[index].metadata,
        })
}

fn dictionary_rows(mut cursor: TermCursor<'_, '_>) -> Vec<ObservedTerm> {
    let mut rows = Vec::new();
    while let Some(current) = cursor.current() {
        rows.push(ObservedTerm {
            term_ord: current.term_ord,
            field_ord: current.field_ord,
            term: current.term.to_vec(),
            metadata: current.metadata,
        });
        require("cursor advance during parity", cursor.next());
    }
    rows
}

fn baseline_rows(baseline: &[BaselineTerm]) -> Vec<ObservedTerm> {
    baseline
        .iter()
        .map(|row| ObservedTerm {
            term_ord: row.term_ord,
            field_ord: row.field_ord,
            term: row.term.as_bytes().to_vec(),
            metadata: row.metadata,
        })
        .collect()
}

fn baseline_prefix_rows(baseline: &[BaselineTerm], query: &Query) -> Vec<ObservedTerm> {
    let prefix = query.term.as_bytes();
    let start =
        baseline.partition_point(|row| compare_row(row, query.field_ord, prefix) == Ordering::Less);
    baseline[start..]
        .iter()
        .take_while(|row| {
            row.field_ord == query.field_ord && row.term.as_bytes().starts_with(prefix)
        })
        .map(|row| ObservedTerm {
            term_ord: row.term_ord,
            field_ord: row.field_ord,
            term: row.term.as_bytes().to_vec(),
            metadata: row.metadata,
        })
        .collect()
}

fn assert_parity(
    dictionary: &TermDictionary<'_>,
    baseline: &[BaselineTerm],
    exact_queries: &[Query],
    prefix_queries: &[Query],
) {
    let mut scratch = TermScratch::new();
    for query in exact_queries {
        let encoded = require(
            "exact lookup parity",
            dictionary.lookup_with_scratch(query.field_ord, query.term.as_bytes(), &mut scratch),
        );
        let sorted = baseline_lookup(baseline, query.field_ord, query.term.as_bytes());
        assert_eq!(encoded, sorted, "exact lookup parity for {query:?}");
    }

    let encoded_full = dictionary_rows(require("full cursor parity", dictionary.cursor()));
    let sorted_full = baseline_rows(baseline);
    assert_eq!(encoded_full, sorted_full, "full ordered scan parity");

    for query in prefix_queries {
        let encoded = dictionary_rows(require(
            "prefix cursor parity",
            dictionary.prefix_cursor(query.field_ord, query.term.as_bytes()),
        ));
        let sorted = baseline_prefix_rows(baseline, query);
        assert_eq!(encoded, sorted, "prefix scan parity for {query:?}");
    }
}

fn mix(digest: u64, value: u64) -> u64 {
    digest.wrapping_mul(DIGEST_PRIME) ^ value
}

fn digest_metadata(mut digest: u64, metadata: TermMetadata) -> u64 {
    digest = mix(digest, u64::from(metadata.doc_freq));
    digest = mix(digest, metadata.postings.offset);
    digest = mix(digest, metadata.postings.len);
    if let Some(positions) = metadata.positions {
        digest = mix(digest, positions.offset);
        digest = mix(digest, positions.len);
    }
    digest = mix(digest, metadata.blockmax.offset);
    mix(digest, metadata.blockmax.len)
}

fn digest_term(
    mut digest: u64,
    term_ord: u32,
    field_ord: u16,
    term: &[u8],
    metadata: TermMetadata,
) -> u64 {
    digest = mix(digest, u64::from(term_ord));
    digest = mix(digest, u64::from(field_ord));
    for byte in term {
        digest = mix(digest, u64::from(*byte));
    }
    digest_metadata(digest, metadata)
}

fn digest_match(digest: u64, found: Option<TermMatch>) -> u64 {
    found.map_or_else(
        || mix(digest, u64::MAX),
        |found| digest_metadata(mix(digest, u64::from(found.term_ord)), found.metadata),
    )
}

fn dictionary_scan_digest(mut cursor: TermCursor<'_, '_>) -> (usize, u64) {
    let mut count = 0_usize;
    let mut digest = DIGEST_SEED;
    while let Some(current) = cursor.current() {
        digest = digest_term(
            digest,
            current.term_ord,
            current.field_ord,
            current.term,
            current.metadata,
        );
        count = count.saturating_add(1);
        require("timed cursor advance", cursor.next());
    }
    (count, digest)
}

fn baseline_scan_digest<'a>(rows: impl Iterator<Item = &'a BaselineTerm>) -> (usize, u64) {
    let mut count = 0_usize;
    let mut digest = DIGEST_SEED;
    for row in rows {
        digest = digest_term(
            digest,
            row.term_ord,
            row.field_ord,
            row.term.as_bytes(),
            row.metadata,
        );
        count = count.saturating_add(1);
    }
    (count, digest)
}

fn baseline_prefix_digest(baseline: &[BaselineTerm], query: &Query) -> (usize, u64) {
    let prefix = query.term.as_bytes();
    let start =
        baseline.partition_point(|row| compare_row(row, query.field_ord, prefix) == Ordering::Less);
    baseline_scan_digest(baseline[start..].iter().take_while(|row| {
        row.field_ord == query.field_ord && row.term.as_bytes().starts_with(prefix)
    }))
}

// Criterion groups intentionally own their reporting state across all registrations.
#[allow(clippy::significant_drop_tightening)]
fn bench(c: &mut Criterion) {
    let fixture = build_fixture();
    let dictionary = require(
        "open",
        fixture.encoded.dictionary(DEFAULT_SCHEMA, fixture.sections),
    );
    assert_parity(
        &dictionary,
        &fixture.baseline,
        &fixture.exact_queries,
        &fixture.prefix_queries,
    );

    let mut exact = c.benchmark_group("grimoire_exact_lookup");
    exact.throughput(Throughput::Elements(as_u64(
        "exact query throughput",
        fixture.exact_queries.len(),
    )));
    let mut scratch = TermScratch::new();
    exact.bench_with_input(
        BenchmarkId::new("termdict", fixture.baseline.len()),
        &fixture.exact_queries,
        |bencher, queries| {
            bencher.iter(|| {
                let mut digest = DIGEST_SEED;
                for query in black_box(queries.as_slice()) {
                    let found = require(
                        "timed exact lookup",
                        dictionary.lookup_with_scratch(
                            query.field_ord,
                            black_box(query.term.as_bytes()),
                            &mut scratch,
                        ),
                    );
                    digest = digest_match(digest, found);
                }
                black_box(digest)
            });
        },
    );
    exact.bench_with_input(
        BenchmarkId::new("sorted_vec", fixture.baseline.len()),
        &fixture.exact_queries,
        |bencher, queries| {
            bencher.iter(|| {
                let mut digest = DIGEST_SEED;
                for query in black_box(queries.as_slice()) {
                    let found = baseline_lookup(
                        &fixture.baseline,
                        query.field_ord,
                        black_box(query.term.as_bytes()),
                    );
                    digest = digest_match(digest, found);
                }
                black_box(digest)
            });
        },
    );
    exact.finish();

    let prefix_elements = fixture
        .prefix_queries
        .iter()
        .map(|query| baseline_prefix_digest(&fixture.baseline, query).0)
        .sum::<usize>();
    let mut prefix = c.benchmark_group("grimoire_prefix_scan");
    prefix.throughput(Throughput::Elements(as_u64(
        "prefix scan throughput",
        prefix_elements,
    )));
    prefix.bench_with_input(
        BenchmarkId::new("termdict", fixture.baseline.len()),
        &fixture.prefix_queries,
        |bencher, queries| {
            bencher.iter(|| {
                let mut digest = DIGEST_SEED;
                for query in black_box(queries.as_slice()) {
                    let cursor = require(
                        "timed prefix cursor",
                        dictionary.prefix_cursor(query.field_ord, black_box(query.term.as_bytes())),
                    );
                    let (count, scan_digest) = dictionary_scan_digest(cursor);
                    digest = mix(digest, as_u64("prefix result count", count));
                    digest = mix(digest, scan_digest);
                }
                black_box(digest)
            });
        },
    );
    prefix.bench_with_input(
        BenchmarkId::new("sorted_vec", fixture.baseline.len()),
        &fixture.prefix_queries,
        |bencher, queries| {
            bencher.iter(|| {
                let mut digest = DIGEST_SEED;
                for query in black_box(queries.as_slice()) {
                    let (count, scan_digest) =
                        baseline_prefix_digest(&fixture.baseline, black_box(query));
                    digest = mix(digest, as_u64("prefix result count", count));
                    digest = mix(digest, scan_digest);
                }
                black_box(digest)
            });
        },
    );
    prefix.finish();

    let mut full = c.benchmark_group("grimoire_full_scan");
    full.throughput(Throughput::Elements(as_u64(
        "full scan throughput",
        fixture.baseline.len(),
    )));
    full.bench_function(
        BenchmarkId::new("termdict", fixture.baseline.len()),
        |bencher| {
            bencher.iter(|| {
                let cursor = require("timed full cursor", dictionary.cursor());
                black_box(dictionary_scan_digest(cursor))
            });
        },
    );
    full.bench_function(
        BenchmarkId::new("sorted_vec", fixture.baseline.len()),
        |bencher| {
            bencher.iter(|| {
                black_box(baseline_scan_digest(
                    black_box(fixture.baseline.as_slice()).iter(),
                ))
            });
        },
    );
    full.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
