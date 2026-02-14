# frankensearch Shared Test Fixtures

This directory contains deterministic fixture data for workspace and cross-crate tests.

## Files

- `corpus.json`: Document corpus used by indexing/search tests.
- `relevance.json`: 20-query ground-truth mapping (`query` -> expected top-10 doc IDs).
- `queries.json`: Extended query set with `query_class` annotations (25 queries).
- `edge_cases.json`: Canonicalization and query edge-case inputs.

## Corpus Layout

`corpus.json` uses:

- `doc_id`
- `title`
- `content`
- `created_at`
- `doc_type`
- `metadata.word_count`
- `metadata.reading_level`
- `metadata.language`

Current corpus composition:

- Core set: 100 documents across 5 clusters (`rust`, `ml`, `sysadmin`, `cooking`, `mixed`), 20 each.
- Supplemental set: 6 additional machine-wide style documents required by extended query fixtures:
  - `adversarial`: 3
  - `code`: 2
  - `config`: 1

Total documents: 106.

## Ground Truth Notes

- `relevance.json` is the baseline 20-query fixture from bead `bd-3un.38`.
- `queries.json` is an extended parallel fixture used by newer test scenarios.
- All IDs referenced by both files are present in `corpus.json`.

## Hash Embedder Caveat

Relevance judgments are most meaningful for semantic models (for example, Model2Vec and MiniLM). For hash-based embeddings, these fixtures primarily validate pipeline correctness and deterministic behavior.

## Maintenance

- Keep fixture IDs stable.
- Prefer additive changes and avoid renaming existing IDs unless tests are migrated in the same commit.
- If a query file references new IDs, add matching corpus documents in the same change.
