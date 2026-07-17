//! Quill tracing convention.
//!
//! Every engine span uses target [`TARGET`] and a stable
//! `frankensearch::quill::<subsystem>::<phase>` name. Later implementations
//! attach only applicable fields from this common vocabulary: `phase`,
//! `schema_id`, `generation`, `segment_id`, `shard_id`, `doc_count`,
//! `query_len`, `result_count`, and `duration_us`. High-cardinality document
//! IDs, terms, query text, and source content are never span fields.

/// Stable tracing target for every Quill event and span.
pub const TARGET: &str = "frankensearch.quill";
/// Scribe batch ingest.
pub const SCRIBE_INGEST: &str = "frankensearch::quill::scribe::ingest";
/// Grimoire dictionary lookup or scan.
pub const GRIMOIRE_LOOKUP: &str = "frankensearch::quill::grimoire::lookup";
/// Quiver posting/position decode.
pub const QUIVER_DECODE: &str = "frankensearch::quill::quiver::decode";
/// Argus parse, plan, and query execution.
pub const ARGUS_QUERY: &str = "frankensearch::quill::argus::query";
/// Keeper index open/recovery.
pub const KEEPER_OPEN: &str = "frankensearch::quill::keeper::open";
/// Keeper delta seal.
pub const KEEPER_SEAL: &str = "frankensearch::quill::keeper::seal";
/// Keeper durable commit/publish.
pub const KEEPER_COMMIT: &str = "frankensearch::quill::keeper::commit";
/// Keeper tombstone compaction.
pub const KEEPER_COMPACT: &str = "frankensearch::quill::keeper::compact";

/// All stable span names, used by policy tests and logging adapters.
pub const ALL_SPAN_NAMES: &[&str] = &[
    SCRIBE_INGEST,
    GRIMOIRE_LOOKUP,
    QUIVER_DECODE,
    ARGUS_QUERY,
    KEEPER_OPEN,
    KEEPER_SEAL,
    KEEPER_COMMIT,
    KEEPER_COMPACT,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_names_use_the_stable_namespace() {
        assert_eq!(TARGET, "frankensearch.quill");
        assert!(
            ALL_SPAN_NAMES
                .iter()
                .all(|name| name.starts_with("frankensearch::quill::"))
        );
    }
}
