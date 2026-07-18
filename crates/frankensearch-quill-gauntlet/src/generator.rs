//! Deterministic corpus and query generation for Quill differential campaigns.
//!
//! Every random-access document derives its own integer-only pseudo-random
//! stream. Generation is therefore invariant to iteration order, sharding, and
//! retries. Corpus identities hash the canonical generated documents, not only
//! their recipes.

use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use frankensearch_core::IndexableDocument;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{DifferentialCase, DifferentialCaseMetadata, GauntletError};

/// Schema version for generator specifications and manifests.
pub const GENERATOR_SCHEMA_VERSION: u32 = 1;
/// Stable identity of this generator implementation.
pub const GENERATOR_ID: &str = "frankensearch-quill-gauntlet/generator-v1";
/// Maximum accepted document size, in UTF-8 bytes.
pub const MAX_DOCUMENT_BYTES: u32 = 2 * 1024 * 1024;
/// Number of documents in the performance-gate xlarge corpus.
pub const XLARGE_DOCUMENT_COUNT: u64 = 1_000_000;
/// Number of documents in the original five-cluster relevance fixture.
pub const CORE_RELEVANCE_DOCUMENT_COUNT: usize = 100;
/// Number of documents in the complete shared fixture corpus.
pub const FULL_SHARED_DOCUMENT_COUNT: usize = 120;

const MAX_VOCABULARY_SIZE: u32 = 65_536;
const DEFAULT_VOCABULARY_SIZE: u32 = 4_096;
const MAX_FREQUENCY_REPETITIONS: usize = 65_536;
const CORPUS_HASH_DOMAIN: &[u8] = b"frankensearch/quill/corpus-content/v1\0";
const MANIFEST_HASH_DOMAIN: &[u8] = b"frankensearch/quill/corpus-manifest/v1\0";
const QUERY_HASH_DOMAIN: &[u8] = b"frankensearch/quill/query-content/v1\0";
const QUERY_MANIFEST_HASH_DOMAIN: &[u8] = b"frankensearch/quill/query-manifest/v1\0";

const SHARED_CORPUS_JSON: &str = include_str!("../../../tests/fixtures/corpus.json");
const SHARED_QUERIES_JSON: &str = include_str!("../../../tests/fixtures/queries.json");
const SHARED_EDGE_CASES_JSON: &str = include_str!("../../../tests/fixtures/edge_cases.json");
const LANGUAGE_CONTRACT_JSON: &str =
    include_str!("../../../tests/fixtures/quill_language_contract.json");

/// Supported deterministic Zipf skews, represented without floating point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZipfExponent {
    /// `s = 0.8`, the flattest required distribution.
    S08,
    /// `s = 1.1`, the middle required distribution.
    S11,
    /// `s = 1.4`, the most head-heavy required distribution.
    S14,
}

impl ZipfExponent {
    /// All skews required by the Quill performance contract.
    pub const ALL: [Self; 3] = [Self::S08, Self::S11, Self::S14];
}

/// Deterministic synthetic-corpus recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntheticCorpusSpec {
    /// Seed for all document-local streams.
    pub seed: u64,
    /// Number of random-access documents.
    pub document_count: u64,
    /// Number of terms in the sampled vocabulary.
    pub vocabulary_size: u32,
    /// Required Zipf skew.
    pub zipf_exponent: ZipfExponent,
    /// UTF-8 byte cap, never greater than [`MAX_DOCUMENT_BYTES`].
    pub max_document_bytes: u32,
}

impl SyntheticCorpusSpec {
    /// Construct the pinned one-million-document performance recipe.
    #[must_use]
    pub const fn xlarge(seed: u64, zipf_exponent: ZipfExponent) -> Self {
        Self {
            seed,
            document_count: XLARGE_DOCUMENT_COUNT,
            vocabulary_size: DEFAULT_VOCABULARY_SIZE,
            zipf_exponent,
            max_document_bytes: MAX_DOCUMENT_BYTES,
        }
    }

    /// Construct all three xlarge Zipf lanes from one seed.
    #[must_use]
    pub fn xlarge_matrix(seed: u64) -> [Self; 3] {
        ZipfExponent::ALL.map(|zipf_exponent| Self::xlarge(seed, zipf_exponent))
    }

    fn validate(&self) -> Result<(), GauntletError> {
        if self.vocabulary_size == 0 || self.vocabulary_size > MAX_VOCABULARY_SIZE {
            return Err(generator_error(format!(
                "vocabulary_size must be in 1..={MAX_VOCABULARY_SIZE}"
            )));
        }
        if self.max_document_bytes < 256 || self.max_document_bytes > MAX_DOCUMENT_BYTES {
            return Err(generator_error(format!(
                "max_document_bytes must be in 256..={MAX_DOCUMENT_BYTES}"
            )));
        }
        Ok(())
    }
}

/// Script lane used by a generated document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnicodeLane {
    /// Seven-bit ASCII.
    Ascii,
    /// Composed and decomposed Latin-1 text.
    Latin1,
    /// Chinese, Japanese, and Korean BMP characters.
    Cjk,
    /// CJK Unified Ideographs Extension B, encoded as four-byte UTF-8.
    CjkExtensionB,
    /// A deliberate mixture of ASCII, Latin-1, CJK, and Extension B.
    Mixed,
}

impl UnicodeLane {
    const ALL: [Self; 5] = [
        Self::Ascii,
        Self::Latin1,
        Self::Cjk,
        Self::CjkExtensionB,
        Self::Mixed,
    ];
}

/// Deliberate edge condition carried by a generated document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pathology {
    /// No content bytes.
    Empty,
    /// Whitespace-only content.
    Whitespace,
    /// One token.
    SingleToken,
    /// One term exactly 256 bytes long.
    Term256Bytes,
    /// Hyphen compounds that emit same-position alternatives.
    SamePositionHyphens,
    /// More than 65,535 occurrences of one term when the byte cap permits it.
    MaximumFrequency,
    /// A composed/decomposed Latin-1 mix.
    Latin1Normalization,
    /// CJK Extension-B boundary characters.
    CjkExtensionB,
    /// Mixed scripts in one document.
    MixedScripts,
    /// A code-like document filled close to its configured byte cap.
    NearLimitCode,
}

/// Typed CASS fields retained until an adapter-specific runner lowers them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CassDocumentFields {
    /// Agent name.
    pub agent: String,
    /// Workspace name.
    pub workspace: String,
    /// Stable source identifier.
    pub source_id: String,
    /// Repository-like source path.
    pub source_path: String,
    /// Origin-kind discriminator.
    pub origin_kind: String,
    /// Message position in the source.
    pub message_index: u64,
}

/// Engine-neutral generated document with canonical metadata ordering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedDocument {
    /// Stable document identifier.
    pub id: String,
    /// Optional boosted title.
    pub title: Option<String>,
    /// Searchable UTF-8 content.
    pub content: String,
    /// Deterministic numeric time used by CASS range fixtures.
    pub created_at_ms: i64,
    /// Typed CASS-only fields.
    pub cass: Option<CassDocumentFields>,
    /// Canonically ordered extensible metadata.
    pub metadata: BTreeMap<String, String>,
    /// Deliberate edge condition, when this is a pathology anchor.
    pub pathology: Option<Pathology>,
    /// Script lane used by the content.
    pub unicode_lane: UnicodeLane,
}

impl From<GeneratedDocument> for IndexableDocument {
    fn from(document: GeneratedDocument) -> Self {
        let mut metadata = document
            .metadata
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
        metadata.insert(
            "created_at_ms".to_owned(),
            document.created_at_ms.to_string(),
        );
        if let Some(cass) = document.cass {
            metadata.insert("agent".to_owned(), cass.agent);
            metadata.insert("workspace".to_owned(), cass.workspace);
            metadata.insert("source_id".to_owned(), cass.source_id);
            metadata.insert("source_path".to_owned(), cass.source_path);
            metadata.insert("origin_kind".to_owned(), cass.origin_kind);
            metadata.insert("message_index".to_owned(), cass.message_index.to_string());
        }
        Self {
            id: document.id,
            content: document.content,
            title: document.title,
            metadata,
        }
    }
}

/// Synthetic corpus with deterministic random access and streaming iteration.
#[derive(Debug, Clone)]
pub struct SyntheticCorpus {
    spec: SyntheticCorpusSpec,
    zipf_cdf: Vec<u64>,
}

impl SyntheticCorpus {
    /// Validate a recipe and prepare its fixed-point Zipf lookup table.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid vocabulary or document-byte cap.
    pub fn new(spec: SyntheticCorpusSpec) -> Result<Self, GauntletError> {
        spec.validate()?;
        let zipf_cdf = build_zipf_cdf(spec.vocabulary_size, spec.zipf_exponent)?;
        Ok(Self { spec, zipf_cdf })
    }

    /// The exact replay recipe.
    #[must_use]
    pub const fn spec(&self) -> &SyntheticCorpusSpec {
        &self.spec
    }

    /// Number of generated documents.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.spec.document_count
    }

    /// Whether the corpus is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.spec.document_count == 0
    }

    /// Generate one document without materializing preceding documents.
    #[must_use]
    pub fn document_at(&self, index: u64) -> Option<GeneratedDocument> {
        (index < self.spec.document_count).then(|| self.generate_document(index))
    }

    /// Stream documents in ordinal order.
    #[must_use]
    pub const fn iter(&self) -> SyntheticCorpusIter<'_> {
        SyntheticCorpusIter {
            corpus: self,
            next_index: 0,
        }
    }

    /// Stream the complete corpus into a content-addressed manifest.
    ///
    /// This intentionally performs O(document_count) work while holding only
    /// one generated document at a time.
    ///
    /// # Errors
    ///
    /// Returns an error if canonical JSON serialization fails.
    pub fn manifest(&self) -> Result<CorpusManifest, GauntletError> {
        CorpusManifest::from_documents(
            CorpusSourceManifest::Synthetic {
                spec: self.spec.clone(),
            },
            self.iter(),
            Vec::new(),
        )
    }

    /// Replay and verify a previously recorded synthetic manifest.
    ///
    /// # Errors
    ///
    /// Returns an error if the recipe, streamed content, counts, or hash differ.
    pub fn verify_manifest(&self, expected: &CorpusManifest) -> Result<(), GauntletError> {
        let actual = self.manifest()?;
        if &actual != expected {
            return Err(GauntletError::ManifestMismatch {
                reason: "synthetic corpus replay differs from its manifest".to_owned(),
            });
        }
        Ok(())
    }

    fn generate_document(&self, index: u64) -> GeneratedDocument {
        let mut rng = DeterministicRng::for_stream(self.spec.seed, 0x434f_5250_5553, index);
        let unicode_lane = UnicodeLane::ALL[(index as usize) % UnicodeLane::ALL.len()];
        let (content, pathology) = match index {
            0 => (String::new(), Some(Pathology::Empty)),
            1 => ("  \n\t  ".to_owned(), Some(Pathology::Whitespace)),
            2 => ("singleton".to_owned(), Some(Pathology::SingleToken)),
            3 => ("t".repeat(256), Some(Pathology::Term256Bytes)),
            4 => (
                "bd-q3fy high-performance state-of-the-art tail".to_owned(),
                Some(Pathology::SamePositionHyphens),
            ),
            5 => (
                maximum_frequency_content(self.spec.max_document_bytes as usize),
                Some(Pathology::MaximumFrequency),
            ),
            6 => (
                "naïve cafe\u{301} résumé coo\u{308}perate".to_owned(),
                Some(Pathology::Latin1Normalization),
            ),
            7 => (
                "extension b boundary 𠀀 𪛟 搜索".to_owned(),
                Some(Pathology::CjkExtensionB),
            ),
            8 => (
                "mixed ASCII café 搜索 𠀀 code::symbol".to_owned(),
                Some(Pathology::MixedScripts),
            ),
            9 => (
                near_limit_code(self.spec.max_document_bytes as usize),
                Some(Pathology::NearLimitCode),
            ),
            _ => (self.generate_regular_content(&mut rng, unicode_lane), None),
        };
        let mut metadata = BTreeMap::new();
        metadata.insert("generator_id".to_owned(), GENERATOR_ID.to_owned());
        metadata.insert("ordinal".to_owned(), index.to_string());
        metadata.insert("seed".to_owned(), self.spec.seed.to_string());
        metadata.insert(
            "zipf_exponent".to_owned(),
            match self.spec.zipf_exponent {
                ZipfExponent::S08 => "0.8",
                ZipfExponent::S11 => "1.1",
                ZipfExponent::S14 => "1.4",
            }
            .to_owned(),
        );
        GeneratedDocument {
            id: format!("synthetic-{index:08}"),
            title: Some(format!("Synthetic document {index}")),
            content: truncate_utf8(content, self.spec.max_document_bytes as usize),
            created_at_ms: 1_700_000_000_000_i64
                .saturating_add(i64::try_from(index).unwrap_or(i64::MAX)),
            cass: Some(CassDocumentFields {
                agent: format!("agent-{}", index % 7),
                workspace: format!("workspace-{}", index % 5),
                source_id: format!("source-{}", index % 11),
                source_path: format!("src/generated/{index:08}.rs"),
                origin_kind: "synthetic".to_owned(),
                message_index: index,
            }),
            metadata,
            pathology,
            unicode_lane,
        }
    }

    fn generate_regular_content(
        &self,
        rng: &mut DeterministicRng,
        unicode_lane: UnicodeLane,
    ) -> String {
        let cap = self.spec.max_document_bytes as usize;
        let target = sampled_document_length(rng, cap);
        let mut content = String::with_capacity(target.min(64 * 1024));
        let prefix = match unicode_lane {
            UnicodeLane::Ascii => "fn indexed_record() { let search = ",
            UnicodeLane::Latin1 => "naïve café re\u{301}sume\u{301} ",
            UnicodeLane::Cjk => "搜索 引擎 文書 ",
            UnicodeLane::CjkExtensionB => "𠀀 𪛟 extension_b ",
            UnicodeLane::Mixed => "mixed café 搜索 𠀀 ",
        };
        push_bounded(&mut content, prefix, target);
        while content.len() < target {
            let rank = self.sample_zipf_rank(rng);
            let term = format!("term{rank:05}");
            if !push_token_bounded(&mut content, &term, target) {
                break;
            }
        }
        content
    }

    fn sample_zipf_rank(&self, rng: &mut DeterministicRng) -> u32 {
        let total = self.zipf_cdf.last().copied().unwrap_or(1);
        let draw = rng.bounded(total);
        let index = self.zipf_cdf.partition_point(|&value| value <= draw);
        u32::try_from(index + 1).unwrap_or(self.spec.vocabulary_size)
    }
}

/// Streaming iterator over a synthetic corpus.
#[derive(Debug, Clone)]
pub struct SyntheticCorpusIter<'a> {
    corpus: &'a SyntheticCorpus,
    next_index: u64,
}

impl Iterator for SyntheticCorpusIter<'_> {
    type Item = GeneratedDocument;

    fn next(&mut self) -> Option<Self::Item> {
        let document = self.corpus.document_at(self.next_index)?;
        self.next_index += 1;
        Some(document)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.corpus.len().saturating_sub(self.next_index);
        let lower = usize::try_from(remaining).unwrap_or(usize::MAX);
        (lower, Some(lower))
    }
}

impl std::iter::FusedIterator for SyntheticCorpusIter<'_> {}

/// Reason a tracked repository entry was not indexed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepositorySkipReason {
    /// The tracked path was absent at snapshot time.
    Missing,
    /// The entry was not a regular file or was a symbolic link.
    NotRegularFile,
    /// The file exceeded the 2 MiB ingest boundary.
    TooLarge,
    /// The file was not valid UTF-8.
    BinaryOrNonUtf8,
}

/// A skipped tracked path recorded in a repository manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedRepositoryEntry {
    /// Normalized repository-relative path.
    pub path: String,
    /// Deterministic exclusion reason.
    pub reason: RepositorySkipReason,
}

/// Digest of one included repository file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepositoryFileDigest {
    /// Normalized repository-relative path.
    pub path: String,
    /// UTF-8 byte count.
    pub bytes: u64,
    /// Lowercase SHA-256 of the raw file bytes.
    pub sha256: String,
}

/// Caller-supplied tracked entry, useful for VCS adapters and deterministic tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepositoryEntry {
    /// Repository-relative path.
    pub relative_path: PathBuf,
    /// Raw tracked bytes.
    pub bytes: Vec<u8>,
}

/// Prepared real-repository corpus and its content-addressed manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepositorySnapshot {
    /// Included UTF-8 documents in normalized path order.
    pub documents: Vec<GeneratedDocument>,
    /// Canonical corpus manifest.
    pub manifest: CorpusManifest,
}

impl RepositorySnapshot {
    /// Build a snapshot from already-discovered tracked entries.
    ///
    /// # Errors
    ///
    /// Returns an error for unsafe paths, duplicate paths, or serialization
    /// failures. Oversize and non-UTF-8 entries are recorded as skips.
    pub fn from_entries(
        repository_id: impl Into<String>,
        entries: impl IntoIterator<Item = RepositoryEntry>,
    ) -> Result<Self, GauntletError> {
        let repository_id = repository_id.into();
        validate_repository_id(&repository_id)?;
        let mut normalized = Vec::new();
        for entry in entries {
            let path = normalize_relative_path(&entry.relative_path)?;
            normalized.push((path, entry.bytes));
        }
        normalized.sort_by(|left, right| left.0.cmp(&right.0));
        if normalized.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(generator_error(
                "repository entries contain a duplicate path",
            ));
        }
        Self::prepare_repository(repository_id, normalized, Vec::new())
    }

    /// Read only an explicit list of VCS-tracked paths below `root`.
    ///
    /// Discovery remains caller-owned (for example, `git ls-files`); this API
    /// never traverses untracked scratch directories. Paths are normalized and
    /// sorted before any content identity is computed.
    ///
    /// # Errors
    ///
    /// Returns an error for unsafe paths, duplicate paths, filesystem failures,
    /// or serialization failures.
    pub fn from_tracked_paths(
        root: &Path,
        repository_id: impl Into<String>,
        tracked_paths: impl IntoIterator<Item = PathBuf>,
    ) -> Result<Self, GauntletError> {
        let repository_id = repository_id.into();
        validate_repository_id(&repository_id)?;
        let mut paths = tracked_paths
            .into_iter()
            .map(|path| normalize_relative_path(&path).map(|normalized| (normalized, path)))
            .collect::<Result<Vec<_>, _>>()?;
        paths.sort_by(|left, right| left.0.cmp(&right.0));
        if paths.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(generator_error(
                "tracked path list contains a duplicate path",
            ));
        }

        let mut entries = Vec::new();
        let mut skipped = Vec::new();
        for (normalized, original) in paths {
            let absolute = root.join(original);
            let metadata = match fs::symlink_metadata(&absolute) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    skipped.push(SkippedRepositoryEntry {
                        path: normalized,
                        reason: RepositorySkipReason::Missing,
                    });
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
                skipped.push(SkippedRepositoryEntry {
                    path: normalized,
                    reason: RepositorySkipReason::NotRegularFile,
                });
                continue;
            }
            if metadata.len() > u64::from(MAX_DOCUMENT_BYTES) {
                skipped.push(SkippedRepositoryEntry {
                    path: normalized,
                    reason: RepositorySkipReason::TooLarge,
                });
                continue;
            }
            entries.push((normalized, fs::read(absolute)?));
        }
        Self::prepare_repository(repository_id, entries, skipped)
    }

    fn prepare_repository(
        repository_id: String,
        entries: Vec<(String, Vec<u8>)>,
        mut skipped: Vec<SkippedRepositoryEntry>,
    ) -> Result<Self, GauntletError> {
        let mut documents = Vec::new();
        let mut files = Vec::new();
        for (path, bytes) in entries {
            if bytes.len() > MAX_DOCUMENT_BYTES as usize {
                skipped.push(SkippedRepositoryEntry {
                    path,
                    reason: RepositorySkipReason::TooLarge,
                });
                continue;
            }
            let Ok(content) = String::from_utf8(bytes.clone()) else {
                skipped.push(SkippedRepositoryEntry {
                    path,
                    reason: RepositorySkipReason::BinaryOrNonUtf8,
                });
                continue;
            };
            files.push(RepositoryFileDigest {
                path: path.clone(),
                bytes: bytes.len() as u64,
                sha256: sha256_hex(&bytes),
            });
            let mut metadata = BTreeMap::new();
            metadata.insert("repository_id".to_owned(), repository_id.clone());
            metadata.insert("source_path".to_owned(), path.clone());
            documents.push(GeneratedDocument {
                id: format!("repo:{path}"),
                title: Some(path.clone()),
                content,
                created_at_ms: 0,
                cass: Some(CassDocumentFields {
                    agent: "repository".to_owned(),
                    workspace: repository_id.clone(),
                    source_id: path.clone(),
                    source_path: path,
                    origin_kind: "repository_snapshot".to_owned(),
                    message_index: documents.len() as u64,
                }),
                metadata,
                pathology: None,
                unicode_lane: UnicodeLane::Mixed,
            });
        }
        skipped.sort_by(|left, right| left.path.cmp(&right.path));
        let manifest = CorpusManifest::from_documents(
            CorpusSourceManifest::Repository {
                repository_id,
                files,
            },
            &documents,
            skipped,
        )?;
        Ok(Self {
            documents,
            manifest,
        })
    }
}

/// Identifies one embedded shared source file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceFileDigest {
    /// Stable repository-relative fixture path.
    pub path: String,
    /// Lowercase SHA-256 of exact committed bytes.
    pub sha256: String,
}

/// Corpus source and exact replay inputs recorded in a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CorpusSourceManifest {
    /// Integer-only synthetic generator.
    Synthetic {
        /// Complete replay recipe.
        spec: SyntheticCorpusSpec,
    },
    /// Embedded shared test fixtures.
    SharedFixtures {
        /// Explicit 100-document or 120-document view.
        view: SharedCorpusView,
        /// Digests of every source fixture participating in the view.
        sources: Vec<SourceFileDigest>,
    },
    /// Explicit VCS-tracked repository snapshot.
    Repository {
        /// Stable logical repository identity, never an absolute path.
        repository_id: String,
        /// Included files in normalized path order.
        files: Vec<RepositoryFileDigest>,
    },
}

/// Canonical, replay-verifiable corpus identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusManifest {
    /// Generator/manifest schema.
    pub schema_version: u32,
    /// Stable generator implementation identity.
    pub generator_id: String,
    /// Exact source recipe.
    pub source: CorpusSourceManifest,
    /// Number of included documents.
    pub document_count: u64,
    /// Sum of included document content bytes.
    pub total_content_bytes: u64,
    /// SHA-256 over length-framed canonical document JSON.
    pub content_sha256: String,
    /// Stable exclusions for repository snapshots.
    pub skipped_repository_entries: Vec<SkippedRepositoryEntry>,
}

impl CorpusManifest {
    fn from_documents<I, D>(
        source: CorpusSourceManifest,
        documents: I,
        skipped_repository_entries: Vec<SkippedRepositoryEntry>,
    ) -> Result<Self, GauntletError>
    where
        I: IntoIterator<Item = D>,
        D: Borrow<GeneratedDocument>,
    {
        let (document_count, total_content_bytes, content_sha256) = hash_documents(documents)?;
        Ok(Self {
            schema_version: GENERATOR_SCHEMA_VERSION,
            generator_id: GENERATOR_ID.to_owned(),
            source,
            document_count,
            total_content_bytes,
            content_sha256,
            skipped_repository_entries,
        })
    }

    /// Canonical manifest JSON bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, GauntletError> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Domain-separated lowercase SHA-256 of the canonical manifest.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn manifest_hash(&self) -> Result<String, GauntletError> {
        let mut hasher = Sha256::new();
        hasher.update(MANIFEST_HASH_DOMAIN);
        hasher.update(self.canonical_bytes()?);
        Ok(lower_hex(&hasher.finalize()))
    }

    /// Verify arbitrary replay documents against this manifest's content pins.
    ///
    /// # Errors
    ///
    /// Returns an error if counts, bytes, content digest, or serialization differ.
    pub fn verify_documents<I, D>(&self, documents: I) -> Result<(), GauntletError>
    where
        I: IntoIterator<Item = D>,
        D: Borrow<GeneratedDocument>,
    {
        let (count, bytes, digest) = hash_documents(documents)?;
        if count != self.document_count
            || bytes != self.total_content_bytes
            || digest != self.content_sha256
        {
            return Err(GauntletError::ManifestMismatch {
                reason: "replayed document stream does not match corpus content pins".to_owned(),
            });
        }
        Ok(())
    }
}

/// Selects the original relevance corpus or the complete shared corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharedCorpusView {
    /// First 100 documents: five 20-document relevance clusters.
    Core100,
    /// Complete committed 120-document corpus including adversarial additions.
    Full120,
}

/// One committed relevance-query row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedRelevanceQuery {
    /// Raw query.
    pub query: String,
    /// Coarse semantic class from the shared fixture.
    pub query_class: String,
    /// Ordered ground-truth document IDs.
    pub relevant_ids: Vec<String>,
}

/// One committed edge-case row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedEdgeCase {
    /// Stable edge-case ID.
    pub id: String,
    /// Exact input, including nulls or whitespace.
    pub text: String,
    /// Normative expected behavior description.
    pub expected_behavior: String,
}

/// Parsed embedded shared fixtures with explicit 100/120 corpus views.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedFixtureSuite {
    documents: Vec<GeneratedDocument>,
    relevance_queries: Vec<SharedRelevanceQuery>,
    edge_cases: Vec<SharedEdgeCase>,
    harvested_contract_queries: Vec<HarvestedContractQuery>,
}

impl SharedFixtureSuite {
    /// Parse and validate the committed fixture suite embedded in this binary.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed fixtures or count/cluster drift.
    pub fn load() -> Result<Self, GauntletError> {
        let corpus: SharedCorpusFile = serde_json::from_str(SHARED_CORPUS_JSON)?;
        let relevance_queries: Vec<SharedRelevanceQuery> =
            serde_json::from_str(SHARED_QUERIES_JSON)?;
        let edge_file: SharedEdgeFile = serde_json::from_str(SHARED_EDGE_CASES_JSON)?;
        let language_contract: LanguageContractQueries =
            serde_json::from_str(LANGUAGE_CONTRACT_JSON)?;
        if corpus.documents.len() != FULL_SHARED_DOCUMENT_COUNT {
            return Err(generator_error(format!(
                "shared corpus must contain {FULL_SHARED_DOCUMENT_COUNT} documents, found {}",
                corpus.documents.len()
            )));
        }
        if edge_file.cases.len() != 21 {
            return Err(generator_error(format!(
                "shared edge fixture must contain 21 cases, found {}",
                edge_file.cases.len()
            )));
        }
        let expected_core_types = ["cooking", "mixed", "ml", "rust", "sysadmin"];
        let core_types = corpus.documents[..CORE_RELEVANCE_DOCUMENT_COUNT]
            .iter()
            .fold(BTreeMap::<&str, usize>::new(), |mut counts, document| {
                *counts.entry(document.doc_type.as_str()).or_default() += 1;
                counts
            });
        if expected_core_types
            .iter()
            .any(|kind| core_types.get(kind).copied() != Some(20))
        {
            return Err(generator_error(
                "shared core relevance view must retain five 20-document clusters",
            ));
        }
        let documents = corpus
            .documents
            .into_iter()
            .enumerate()
            .map(|(index, document)| document.into_generated(index))
            .collect();
        Ok(Self {
            documents,
            relevance_queries,
            edge_cases: edge_file.cases,
            harvested_contract_queries: language_contract.harvested_queries,
        })
    }

    /// Documents in the selected committed view.
    #[must_use]
    pub fn documents(&self, view: SharedCorpusView) -> &[GeneratedDocument] {
        match view {
            SharedCorpusView::Core100 => &self.documents[..CORE_RELEVANCE_DOCUMENT_COUNT],
            SharedCorpusView::Full120 => &self.documents,
        }
    }

    /// All 25 ground-truth relevance queries.
    #[must_use]
    pub fn relevance_queries(&self) -> &[SharedRelevanceQuery] {
        &self.relevance_queries
    }

    /// All 21 committed edge cases.
    #[must_use]
    pub fn edge_cases(&self) -> &[SharedEdgeCase] {
        &self.edge_cases
    }

    /// The seven broad query-surface anchors in the language contract.
    #[must_use]
    pub fn harvested_contract_queries(&self) -> &[HarvestedContractQuery] {
        &self.harvested_contract_queries
    }

    /// Content-address the selected committed corpus view and exact source files.
    ///
    /// # Errors
    ///
    /// Returns an error if canonical serialization fails.
    pub fn manifest(&self, view: SharedCorpusView) -> Result<CorpusManifest, GauntletError> {
        CorpusManifest::from_documents(
            CorpusSourceManifest::SharedFixtures {
                view,
                sources: shared_source_digests(),
            },
            self.documents(view),
            Vec::new(),
        )
    }
}

/// Default parser versus the CASS grammar adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuerySyntax {
    /// Native/default query syntax.
    Default,
    /// CASS OR-over-AND grammar and structured filters.
    Cass,
}

/// Required wildcard pattern families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobPatternClass {
    /// No wildcard.
    Exact,
    /// Literal prefix followed by `*`.
    Prefix,
    /// `*` followed by a literal suffix.
    Suffix,
    /// Literal surrounded by `*`.
    Substring,
    /// Multiple `*` or `?` operators.
    Complex,
}

/// Required range shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RangeClass {
    /// Inclusive lower and upper bounds.
    Inclusive,
    /// Lower bound only.
    From,
    /// Upper bound only.
    To,
}

/// Semantic/syntactic purpose of a generated query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GeneratedQueryKind {
    /// One term.
    Term,
    /// Multiple unquoted terms.
    MultiTerm,
    /// Quoted phrase.
    Phrase,
    /// Boolean expression.
    Boolean,
    /// Wildcard query.
    Glob {
        /// Exact wildcard family.
        pattern_class: GlobPatternClass,
    },
    /// Structured numeric range.
    Range {
        /// Exact bound shape.
        range_class: RangeClass,
    },
    /// Explicit pagination probe.
    Paginated,
    /// Explicit count/no-count probe.
    Counted,
    /// One row harvested from a committed relevance fixture.
    Harvested {
        /// Coarse semantic class.
        semantic_class: String,
    },
}

/// CASS structured range filters kept separate from raw query syntax.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedQueryFilters {
    /// Inclusive creation-time lower bound.
    pub created_from_ms: Option<i64>,
    /// Inclusive creation-time upper bound.
    pub created_to_ms: Option<i64>,
}

impl GeneratedQueryFilters {
    fn is_empty(&self) -> bool {
        self.created_from_ms.is_none() && self.created_to_ms.is_none()
    }
}

/// Rich engine-neutral query case; E6.2 adapters lower this without data loss.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedQueryCase {
    /// Stable case ID.
    pub id: String,
    /// Parser/schema flavor.
    pub syntax: QuerySyntax,
    /// Exact coverage class.
    pub query_kind: GeneratedQueryKind,
    /// Raw source string.
    pub query: String,
    /// Result limit.
    pub limit: u64,
    /// Pagination offset.
    pub offset: u64,
    /// Whether exact/thresholded count evidence is requested.
    pub count_requested: bool,
    /// Structured range filters.
    pub filters: GeneratedQueryFilters,
    /// Accepted divergence identifier, when applicable.
    pub expected_divergence: Option<String>,
    /// Fixture provenance.
    pub source: String,
}

impl GeneratedQueryCase {
    /// Lower only semantics representable by today's generic differential case.
    ///
    /// # Errors
    ///
    /// Returns an error for CASS syntax, pagination, structured filters, or a
    /// malformed corpus manifest hash. E6.2 must lower those adapter-specifically.
    pub fn to_differential_case(
        &self,
        generator_seed: u64,
        corpus_manifest_hash: &str,
    ) -> Result<DifferentialCase, GauntletError> {
        validate_sha256(corpus_manifest_hash, "corpus manifest hash")?;
        if self.syntax != QuerySyntax::Default || self.offset != 0 || !self.filters.is_empty() {
            return Err(generator_error(
                "query requires adapter-specific CASS, range, or pagination lowering",
            ));
        }
        let mut case = DifferentialCase::new(&self.id, &self.query, self.limit);
        case.count_requested = self.count_requested;
        case.metadata = DifferentialCaseMetadata {
            generator_id: Some(GENERATOR_ID.to_owned()),
            generator_seed: Some(generator_seed),
            corpus_hash: Some(corpus_manifest_hash.to_owned()),
        };
        Ok(case)
    }
}

/// Versioned query-generation recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryGeneratorSpec {
    /// Independent query stream seed.
    pub seed: u64,
    /// Default result limit for non-pagination probes.
    pub default_limit: u64,
    /// Include all 25 committed relevance queries.
    pub include_shared_relevance_queries: bool,
}

/// Canonical query-suite identity bound to one corpus manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryManifest {
    /// Generator/manifest schema.
    pub schema_version: u32,
    /// Stable generator implementation identity.
    pub generator_id: String,
    /// Query recipe.
    pub spec: QueryGeneratorSpec,
    /// Manifest hash of the corpus queried.
    pub corpus_manifest_hash: String,
    /// Number of cases.
    pub query_count: u64,
    /// SHA-256 over length-framed canonical query JSON.
    pub content_sha256: String,
}

impl QueryManifest {
    /// Domain-separated manifest hash.
    ///
    /// # Errors
    ///
    /// Returns an error if canonical serialization fails.
    pub fn manifest_hash(&self) -> Result<String, GauntletError> {
        let mut hasher = Sha256::new();
        hasher.update(QUERY_MANIFEST_HASH_DOMAIN);
        hasher.update(serde_json::to_vec(self)?);
        Ok(lower_hex(&hasher.finalize()))
    }

    /// Verify ordered query content and count.
    ///
    /// # Errors
    ///
    /// Returns an error when replayed queries differ.
    pub fn verify(&self, cases: &[GeneratedQueryCase]) -> Result<(), GauntletError> {
        let (query_count, content_sha256) = hash_queries(cases)?;
        if query_count != self.query_count || content_sha256 != self.content_sha256 {
            return Err(GauntletError::ManifestMismatch {
                reason: "replayed query suite does not match query content pins".to_owned(),
            });
        }
        Ok(())
    }
}

/// Generated query suite plus replay manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedQuerySuite {
    /// Ordered generated and harvested cases.
    pub cases: Vec<GeneratedQueryCase>,
    /// Content-addressed replay manifest.
    pub manifest: QueryManifest,
}

impl GeneratedQuerySuite {
    /// Generate the complete E6.1 query matrix.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid corpus hash, zero limit, malformed shared
    /// fixtures, or canonical serialization failure.
    pub fn generate(
        spec: QueryGeneratorSpec,
        corpus_manifest_hash: impl Into<String>,
        shared: &SharedFixtureSuite,
    ) -> Result<Self, GauntletError> {
        if spec.default_limit == 0 {
            return Err(generator_error("default query limit must be nonzero"));
        }
        let corpus_manifest_hash = corpus_manifest_hash.into();
        validate_sha256(&corpus_manifest_hash, "corpus manifest hash")?;
        let mut cases = constructed_query_matrix(spec.seed, spec.default_limit);
        if spec.include_shared_relevance_queries {
            cases.extend(
                shared
                    .relevance_queries
                    .iter()
                    .enumerate()
                    .map(|(index, query)| GeneratedQueryCase {
                        id: format!("harvested-{index:02}"),
                        syntax: QuerySyntax::Default,
                        query_kind: GeneratedQueryKind::Harvested {
                            semantic_class: query.query_class.clone(),
                        },
                        query: query.query.clone(),
                        limit: spec.default_limit,
                        offset: 0,
                        count_requested: true,
                        filters: GeneratedQueryFilters::default(),
                        expected_divergence: None,
                        source: "tests/fixtures/queries.json".to_owned(),
                    }),
            );
        }
        let (query_count, content_sha256) = hash_queries(&cases)?;
        let manifest = QueryManifest {
            schema_version: GENERATOR_SCHEMA_VERSION,
            generator_id: GENERATOR_ID.to_owned(),
            spec,
            corpus_manifest_hash,
            query_count,
            content_sha256,
        };
        Ok(Self { cases, manifest })
    }
}

/// Broad query anchor parsed from the committed language contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarvestedContractQuery {
    /// Stable fixture ID.
    pub id: String,
    /// Semantic/syntactic class label.
    pub query_class: String,
    /// Raw query.
    pub query: String,
    /// Source classification.
    pub source_kind: String,
    /// Source path, when harvested.
    #[serde(default)]
    pub source: Option<String>,
    /// Construction provenance, when synthesized.
    #[serde(default)]
    pub source_fact: Option<String>,
    /// Optional structured filters retained as generic JSON for fidelity.
    #[serde(default)]
    pub filters: BTreeMap<String, i64>,
}

#[derive(Debug, Deserialize)]
struct SharedCorpusFile {
    documents: Vec<SharedDocument>,
}

#[derive(Debug, Deserialize)]
struct SharedDocument {
    doc_id: String,
    title: String,
    content: String,
    created_at: String,
    doc_type: String,
    #[serde(default)]
    metadata: BTreeMap<String, serde_json::Value>,
}

impl SharedDocument {
    fn into_generated(self, index: usize) -> GeneratedDocument {
        let mut metadata = self
            .metadata
            .into_iter()
            .map(|(key, value)| {
                let value = value
                    .as_str()
                    .map_or_else(|| value.to_string(), ToOwned::to_owned);
                (key, value)
            })
            .collect::<BTreeMap<_, _>>();
        metadata.insert("created_at".to_owned(), self.created_at);
        metadata.insert("doc_type".to_owned(), self.doc_type);
        GeneratedDocument {
            id: self.doc_id,
            title: Some(self.title),
            content: self.content,
            created_at_ms: i64::try_from(index).unwrap_or(i64::MAX),
            cass: None,
            metadata,
            pathology: None,
            unicode_lane: UnicodeLane::Mixed,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SharedEdgeFile {
    cases: Vec<SharedEdgeCase>,
}

#[derive(Debug, Deserialize)]
struct LanguageContractQueries {
    harvested_queries: Vec<HarvestedContractQuery>,
}

fn constructed_query_matrix(seed: u64, default_limit: u64) -> Vec<GeneratedQueryCase> {
    let mut rng = DeterministicRng::for_stream(seed, 0x5155_4552_4945_53, 0);
    let terms = ["ownership", "search", "cache", "token", "config"];
    let selected = terms[rng.bounded(terms.len() as u64) as usize];
    let base = |id: &str, syntax: QuerySyntax, query_kind: GeneratedQueryKind, query: String| {
        GeneratedQueryCase {
            id: id.to_owned(),
            syntax,
            query_kind,
            query,
            limit: default_limit,
            offset: 0,
            count_requested: true,
            filters: GeneratedQueryFilters::default(),
            expected_divergence: None,
            source: "quill-e6.1-constructed".to_owned(),
        }
    };
    let mut cases = vec![
        base(
            "term",
            QuerySyntax::Default,
            GeneratedQueryKind::Term,
            selected.to_owned(),
        ),
        base(
            "multi-term",
            QuerySyntax::Default,
            GeneratedQueryKind::MultiTerm,
            "rust ownership borrowing".to_owned(),
        ),
        base(
            "phrase",
            QuerySyntax::Default,
            GeneratedQueryKind::Phrase,
            "\"error handling\"".to_owned(),
        ),
        base(
            "same-position-phrase",
            QuerySyntax::Default,
            GeneratedQueryKind::Phrase,
            "\"high performance tail\"".to_owned(),
        ),
        base(
            "boolean-default",
            QuerySyntax::Default,
            GeneratedQueryKind::Boolean,
            "auth AND token OR cache".to_owned(),
        ),
        base(
            "boolean-cass",
            QuerySyntax::Cass,
            GeneratedQueryKind::Boolean,
            "auth OR token cache".to_owned(),
        ),
        base(
            "glob-exact",
            QuerySyntax::Cass,
            GeneratedQueryKind::Glob {
                pattern_class: GlobPatternClass::Exact,
            },
            "config".to_owned(),
        ),
        base(
            "glob-prefix",
            QuerySyntax::Cass,
            GeneratedQueryKind::Glob {
                pattern_class: GlobPatternClass::Prefix,
            },
            "config*".to_owned(),
        ),
        base(
            "glob-suffix",
            QuerySyntax::Cass,
            GeneratedQueryKind::Glob {
                pattern_class: GlobPatternClass::Suffix,
            },
            "*config".to_owned(),
        ),
        base(
            "glob-substring",
            QuerySyntax::Cass,
            GeneratedQueryKind::Glob {
                pattern_class: GlobPatternClass::Substring,
            },
            "*config*".to_owned(),
        ),
        base(
            "glob-complex",
            QuerySyntax::Cass,
            GeneratedQueryKind::Glob {
                pattern_class: GlobPatternClass::Complex,
            },
            "con*fi?g*".to_owned(),
        ),
    ];
    cases[3].expected_divergence = Some("DIV-003".to_owned());
    cases.push(GeneratedQueryCase {
        filters: GeneratedQueryFilters {
            created_from_ms: Some(1_700_000_000_000),
            created_to_ms: Some(1_700_000_000_999),
        },
        ..base(
            "range-inclusive",
            QuerySyntax::Cass,
            GeneratedQueryKind::Range {
                range_class: RangeClass::Inclusive,
            },
            "cache".to_owned(),
        )
    });
    cases.push(GeneratedQueryCase {
        filters: GeneratedQueryFilters {
            created_from_ms: Some(1_700_000_000_000),
            created_to_ms: None,
        },
        ..base(
            "range-from",
            QuerySyntax::Cass,
            GeneratedQueryKind::Range {
                range_class: RangeClass::From,
            },
            "cache".to_owned(),
        )
    });
    cases.push(GeneratedQueryCase {
        filters: GeneratedQueryFilters {
            created_from_ms: None,
            created_to_ms: Some(1_700_000_000_999),
        },
        ..base(
            "range-to",
            QuerySyntax::Cass,
            GeneratedQueryKind::Range {
                range_class: RangeClass::To,
            },
            "cache".to_owned(),
        )
    });
    cases.push(GeneratedQueryCase {
        offset: 17,
        limit: 7,
        ..base(
            "paginated",
            QuerySyntax::Default,
            GeneratedQueryKind::Paginated,
            "search".to_owned(),
        )
    });
    cases.push(GeneratedQueryCase {
        count_requested: false,
        ..base(
            "uncounted",
            QuerySyntax::Default,
            GeneratedQueryKind::Counted,
            "search".to_owned(),
        )
    });
    cases.push(base(
        "counted",
        QuerySyntax::Default,
        GeneratedQueryKind::Counted,
        "search".to_owned(),
    ));
    cases
}

fn shared_source_digests() -> Vec<SourceFileDigest> {
    [
        ("tests/fixtures/corpus.json", SHARED_CORPUS_JSON),
        ("tests/fixtures/queries.json", SHARED_QUERIES_JSON),
        ("tests/fixtures/edge_cases.json", SHARED_EDGE_CASES_JSON),
        (
            "tests/fixtures/quill_language_contract.json",
            LANGUAGE_CONTRACT_JSON,
        ),
    ]
    .into_iter()
    .map(|(path, contents)| SourceFileDigest {
        path: path.to_owned(),
        sha256: sha256_hex(contents.as_bytes()),
    })
    .collect()
}

fn hash_documents<I, D>(documents: I) -> Result<(u64, u64, String), GauntletError>
where
    I: IntoIterator<Item = D>,
    D: Borrow<GeneratedDocument>,
{
    let mut hasher = Sha256::new();
    hasher.update(CORPUS_HASH_DOMAIN);
    let mut count = 0_u64;
    let mut content_bytes = 0_u64;
    for document in documents {
        let document = document.borrow();
        let bytes = serde_json::to_vec(document)?;
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
        count = count.saturating_add(1);
        content_bytes = content_bytes.saturating_add(document.content.len() as u64);
    }
    Ok((count, content_bytes, lower_hex(&hasher.finalize())))
}

fn hash_queries(cases: &[GeneratedQueryCase]) -> Result<(u64, String), GauntletError> {
    let mut hasher = Sha256::new();
    hasher.update(QUERY_HASH_DOMAIN);
    for case in cases {
        let bytes = serde_json::to_vec(case)?;
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(bytes);
    }
    Ok((cases.len() as u64, lower_hex(&hasher.finalize())))
}

fn build_zipf_cdf(vocabulary_size: u32, exponent: ZipfExponent) -> Result<Vec<u64>, GauntletError> {
    let mut cumulative = 0_u64;
    let mut cdf = Vec::with_capacity(vocabulary_size as usize);
    for rank in 1..=vocabulary_size {
        let denominator_q8 = zipf_denominator_q8(rank, exponent);
        let weight = (1_u64 << 55) / denominator_q8.max(1);
        cumulative = cumulative
            .checked_add(weight.max(1))
            .ok_or_else(|| generator_error("Zipf cumulative weight overflow"))?;
        cdf.push(cumulative);
    }
    Ok(cdf)
}

fn zipf_denominator_q8(rank: u32, exponent: ZipfExponent) -> u64 {
    let rank = u128::from(rank);
    let value = match exponent {
        ZipfExponent::S08 => integer_nth_root((rank.pow(4)) << 40, 5),
        ZipfExponent::S11 => rank * integer_nth_root(rank << 80, 10),
        ZipfExponent::S14 => rank * integer_nth_root((rank.pow(2)) << 40, 5),
    };
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn integer_nth_root(value: u128, exponent: u32) -> u128 {
    let mut low = 0_u128;
    let mut high = 1_u128;
    while power_leq(high, exponent, value) && high <= value / 2 {
        high *= 2;
    }
    while low + 1 < high {
        let middle = low + (high - low) / 2;
        if power_leq(middle, exponent, value) {
            low = middle;
        } else {
            high = middle;
        }
    }
    if power_leq(high, exponent, value) {
        high
    } else {
        low
    }
}

fn power_leq(base: u128, exponent: u32, limit: u128) -> bool {
    let mut product = 1_u128;
    for _ in 0..exponent {
        let Some(next) = product.checked_mul(base) else {
            return false;
        };
        if next > limit {
            return false;
        }
        product = next;
    }
    true
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn for_stream(seed: u64, domain: u64, ordinal: u64) -> Self {
        let mut mixer = Self {
            state: seed ^ domain.rotate_left(17) ^ ordinal.wrapping_mul(0xd6e8_feb8_6659_fd93),
        };
        Self {
            state: mixer.next_u64(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn bounded(&mut self, upper: u64) -> u64 {
        if upper <= 1 {
            return 0;
        }
        ((u128::from(self.next_u64()) * u128::from(upper)) >> 64) as u64
    }
}

fn sampled_document_length(rng: &mut DeterministicRng, cap: usize) -> usize {
    let bucket = rng.bounded(1_000);
    let (minimum, span) = if bucket < 700 {
        (64_usize, 448_usize)
    } else if bucket < 930 {
        (512, 7_680)
    } else if bucket < 990 {
        (8_192, 57_344)
    } else {
        (65_536, cap.saturating_sub(65_536).saturating_add(1))
    };
    let available_minimum = minimum.min(cap);
    let available_span = span.min(cap.saturating_sub(available_minimum).saturating_add(1));
    available_minimum + rng.bounded(available_span.max(1) as u64) as usize
}

fn maximum_frequency_content(cap: usize) -> String {
    let repeat = "freq ";
    let tail = "tail";
    let repetitions = MAX_FREQUENCY_REPETITIONS.min(cap.saturating_sub(tail.len()) / repeat.len());
    let mut content = String::with_capacity(repetitions * repeat.len() + tail.len());
    for _ in 0..repetitions {
        content.push_str(repeat);
    }
    content.push_str(tail);
    content
}

fn near_limit_code(cap: usize) -> String {
    let line = "fn generated_record() { index.push(term00001); }\n";
    let tail = "// needle tail";
    let mut content = String::with_capacity(cap);
    while content.len() + line.len() + tail.len() <= cap {
        content.push_str(line);
    }
    push_bounded(&mut content, tail, cap);
    content
}

fn push_token_bounded(target: &mut String, token: &str, limit: usize) -> bool {
    let separator = usize::from(!target.is_empty());
    if target.len() + separator + token.len() > limit {
        return false;
    }
    if separator == 1 {
        target.push(' ');
    }
    target.push_str(token);
    true
}

fn push_bounded(target: &mut String, source: &str, limit: usize) {
    let remaining = limit.saturating_sub(target.len());
    let mut boundary = source.len().min(remaining);
    while boundary > 0 && !source.is_char_boundary(boundary) {
        boundary -= 1;
    }
    target.push_str(&source[..boundary]);
}

fn truncate_utf8(mut value: String, limit: usize) -> String {
    if value.len() <= limit {
        return value;
    }
    let mut boundary = limit;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    value.truncate(boundary);
    value
}

fn normalize_relative_path(path: &Path) -> Result<String, GauntletError> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(generator_error(format!(
            "repository path must be nonempty and relative: {}",
            path.display()
        )));
    }
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => {
                let Some(part) = part.to_str() else {
                    return Err(generator_error("repository path must be valid UTF-8"));
                };
                if part.is_empty() {
                    return Err(generator_error(
                        "repository path contains an empty component",
                    ));
                }
                parts.push(part);
            }
            _ => {
                return Err(generator_error(format!(
                    "repository path contains an unsafe component: {}",
                    path.display()
                )));
            }
        }
    }
    Ok(parts.join("/"))
}

fn validate_repository_id(repository_id: &str) -> Result<(), GauntletError> {
    if repository_id.is_empty() || repository_id.contains(['/', '\\']) {
        return Err(generator_error(
            "repository_id must be a nonempty logical name without path separators",
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<(), GauntletError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(generator_error(format!(
            "{label} must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    lower_hex(&Sha256::digest(bytes))
}

fn lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn generator_error(reason: impl Into<String>) -> GauntletError {
    GauntletError::InvalidGenerator {
        reason: reason.into(),
    }
}

pub(crate) fn validate_generated_case_metadata(
    case: &DifferentialCase,
) -> Result<(), GauntletError> {
    if case.metadata.generator_id.as_deref() != Some(GENERATOR_ID) {
        return Ok(());
    }
    if case.metadata.generator_seed.is_none() {
        return Err(generator_error(
            "generated artifact case must retain its generator seed",
        ));
    }
    let hash = case
        .metadata
        .corpus_hash
        .as_deref()
        .ok_or_else(|| generator_error("generated artifact case must retain a corpus hash"))?;
    validate_sha256(hash, "generated artifact corpus manifest hash")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    const GENERATOR_GOLDEN_JSON: &str = include_str!("../fixtures/generator-v1.json");

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct GeneratorGolden {
        schema_version: u32,
        corpus_spec: SyntheticCorpusSpec,
        corpus_manifest: CorpusManifest,
        corpus_manifest_hash: String,
        documents: Vec<GeneratedDocument>,
        query_manifest: QueryManifest,
        query_manifest_hash: String,
        queries: Vec<GeneratedQueryCase>,
    }

    fn small_spec(seed: u64, document_count: u64) -> SyntheticCorpusSpec {
        SyntheticCorpusSpec {
            seed,
            document_count,
            vocabulary_size: 128,
            zipf_exponent: ZipfExponent::S11,
            max_document_bytes: 1_024,
        }
    }

    fn generate_small_golden() -> GeneratorGolden {
        let corpus_spec = small_spec(0x5eed, 12);
        let corpus = SyntheticCorpus::new(corpus_spec.clone()).expect("golden corpus");
        let documents = corpus.iter().collect::<Vec<_>>();
        let corpus_manifest = corpus.manifest().expect("corpus manifest");
        let corpus_manifest_hash = corpus_manifest.manifest_hash().expect("corpus hash");
        let shared = SharedFixtureSuite::load().expect("shared fixtures");
        let query_suite = GeneratedQuerySuite::generate(
            QueryGeneratorSpec {
                seed: 0x5eee,
                default_limit: 20,
                include_shared_relevance_queries: false,
            },
            &corpus_manifest_hash,
            &shared,
        )
        .expect("golden query suite");
        let query_manifest_hash = query_suite.manifest.manifest_hash().expect("query hash");
        GeneratorGolden {
            schema_version: GENERATOR_SCHEMA_VERSION,
            corpus_spec,
            corpus_manifest,
            corpus_manifest_hash,
            documents,
            query_manifest: query_suite.manifest,
            query_manifest_hash,
            queries: query_suite.cases,
        }
    }

    #[test]
    fn committed_small_generator_output_is_an_exact_replay_golden() {
        let expected: GeneratorGolden =
            serde_json::from_str(GENERATOR_GOLDEN_JSON).expect("valid committed golden");
        let actual = generate_small_golden();
        assert_eq!(actual, expected);
        actual
            .corpus_manifest
            .verify_documents(&actual.documents)
            .expect("corpus replay");
        actual
            .query_manifest
            .verify(&actual.queries)
            .expect("query replay");
        assert_eq!(
            actual.corpus_manifest.manifest_hash().unwrap(),
            actual.corpus_manifest_hash
        );
        assert_eq!(
            actual.query_manifest.manifest_hash().unwrap(),
            actual.query_manifest_hash
        );
    }

    #[test]
    #[ignore = "maintainer helper: prints the manually reviewed generator golden"]
    fn emit_small_generator_golden() {
        let golden = generate_small_golden();
        panic!(
            "{}",
            serde_json::to_string_pretty(&golden).expect("serialize golden")
        );
    }

    #[test]
    fn splitmix_stream_is_pinned_and_domain_separated() {
        let mut rng = DeterministicRng::for_stream(42, 7, 11);
        let actual = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        assert_eq!(actual, [0, 0, 0, 0]);

        let mut different_domain = DeterministicRng::for_stream(42, 8, 11);
        assert_ne!(actual[0], different_domain.next_u64());
        let mut bounded = DeterministicRng::for_stream(42, 7, 11);
        assert_eq!(bounded.bounded(0), 0);
        assert_eq!(bounded.bounded(1), 0);
        assert!(bounded.bounded(17) < 17);
    }

    #[test]
    fn fixed_point_zipf_skews_are_deterministic_and_increasingly_head_heavy() {
        let head_hits = ZipfExponent::ALL.map(|exponent| {
            let cdf = build_zipf_cdf(1_024, exponent).expect("valid CDF");
            let mut rng = DeterministicRng::for_stream(9, 10, 11);
            (0..20_000)
                .map(|_| {
                    let draw = rng.bounded(*cdf.last().expect("nonempty CDF"));
                    cdf.partition_point(|&value| value <= draw) + 1
                })
                .filter(|rank| *rank <= 16)
                .count()
        });
        assert!(head_hits[0] < head_hits[1]);
        assert!(head_hits[1] < head_hits[2]);
        assert!(
            build_zipf_cdf(0, ZipfExponent::S08)
                .expect("empty lookup is representable")
                .is_empty()
        );
    }

    #[test]
    fn synthetic_pathologies_cover_boundaries_and_never_exceed_byte_cap() {
        let corpus = SyntheticCorpus::new(SyntheticCorpusSpec {
            seed: 17,
            document_count: 32,
            vocabulary_size: 256,
            zipf_exponent: ZipfExponent::S14,
            max_document_bytes: MAX_DOCUMENT_BYTES,
        })
        .expect("valid corpus");
        let documents = (0..10)
            .map(|index| corpus.document_at(index).expect("document exists"))
            .collect::<Vec<_>>();
        assert!(documents.iter().all(|document| {
            document.content.len() <= MAX_DOCUMENT_BYTES as usize
                && document.content.is_char_boundary(document.content.len())
        }));
        assert!(documents[0].content.is_empty());
        assert!(documents[1].content.trim().is_empty());
        assert_eq!(documents[2].content, "singleton");
        assert_eq!(documents[3].content.len(), 256);
        assert!(documents[3].content.bytes().all(|byte| byte == b't'));
        assert!(documents[4].content.contains("bd-q3fy"));
        assert_eq!(
            documents[5]
                .content
                .split_whitespace()
                .filter(|term| *term == "freq")
                .count(),
            MAX_FREQUENCY_REPETITIONS
        );
        assert!(documents[5].content.ends_with("tail"));
        assert!(documents[6].content.contains("cafe\u{301}"));
        assert!(documents[7].content.contains('𠀀'));
        assert!(documents[7].content.contains('𪛟'));
        assert!(documents[8].content.contains("café"));
        assert!(documents[9].content.ends_with("// needle tail"));
    }

    #[test]
    fn configured_byte_boundary_is_fail_closed_and_utf8_safe() {
        for cap in [MAX_DOCUMENT_BYTES - 1, MAX_DOCUMENT_BYTES] {
            let corpus = SyntheticCorpus::new(SyntheticCorpusSpec {
                max_document_bytes: cap,
                ..small_spec(4, 10)
            })
            .expect("accepted cap");
            let document = corpus.document_at(9).expect("near-limit anchor");
            assert!(document.content.len() <= cap as usize);
            assert!(document.content.is_char_boundary(document.content.len()));
        }
        assert!(
            SyntheticCorpus::new(SyntheticCorpusSpec {
                max_document_bytes: MAX_DOCUMENT_BYTES + 1,
                ..small_spec(4, 10)
            })
            .is_err()
        );
        assert!(
            SyntheticCorpus::new(SyntheticCorpusSpec {
                vocabulary_size: 0,
                ..small_spec(4, 10)
            })
            .is_err()
        );
    }

    #[test]
    fn random_access_is_prefix_stable_and_seed_sensitive() {
        let short = SyntheticCorpus::new(small_spec(31, 12)).expect("valid corpus");
        let long = SyntheticCorpus::new(small_spec(31, 40)).expect("valid corpus");
        let changed = SyntheticCorpus::new(small_spec(32, 40)).expect("valid corpus");
        for index in 0..12 {
            assert_eq!(short.document_at(index), long.document_at(index));
        }
        assert_ne!(long.document_at(10), changed.document_at(10));
        assert!(short.document_at(12).is_none());
    }

    #[test]
    fn corpus_manifest_replays_and_detects_seed_content_and_config_mutations() {
        let corpus = SyntheticCorpus::new(small_spec(91, 12)).expect("valid corpus");
        let manifest = corpus.manifest().expect("manifest");
        corpus.verify_manifest(&manifest).expect("exact replay");
        assert_eq!(manifest.document_count, 12);
        assert_eq!(manifest.content_sha256.len(), 64);
        let bytes = manifest.canonical_bytes().expect("canonical bytes");
        let decoded: CorpusManifest = serde_json::from_slice(&bytes).expect("decode manifest");
        assert_eq!(decoded.canonical_bytes().expect("re-encode"), bytes);

        let changed_seed = SyntheticCorpus::new(small_spec(92, 12))
            .expect("valid corpus")
            .manifest()
            .expect("manifest");
        assert_ne!(
            manifest.manifest_hash().unwrap(),
            changed_seed.manifest_hash().unwrap()
        );

        let mut tampered = corpus.iter().collect::<Vec<_>>();
        tampered[10].content.push_str(" tamper");
        assert!(manifest.verify_documents(&tampered).is_err());
    }

    #[test]
    fn shared_fixtures_expose_explicit_core_and_full_views() {
        let shared = SharedFixtureSuite::load().expect("committed fixtures are valid");
        assert_eq!(
            shared.documents(SharedCorpusView::Core100).len(),
            CORE_RELEVANCE_DOCUMENT_COUNT
        );
        assert_eq!(
            shared.documents(SharedCorpusView::Full120).len(),
            FULL_SHARED_DOCUMENT_COUNT
        );
        assert_eq!(shared.relevance_queries().len(), 25);
        assert_eq!(shared.edge_cases().len(), 21);
        assert_eq!(shared.harvested_contract_queries().len(), 7);
        let core = shared
            .manifest(SharedCorpusView::Core100)
            .expect("manifest");
        let full = shared
            .manifest(SharedCorpusView::Full120)
            .expect("manifest");
        assert_ne!(core.content_sha256, full.content_sha256);
        core.verify_documents(shared.documents(SharedCorpusView::Core100))
            .expect("core replay");
        full.verify_documents(shared.documents(SharedCorpusView::Full120))
            .expect("full replay");
    }

    #[test]
    fn query_suite_covers_every_required_class_and_all_harvested_rows() {
        let shared = SharedFixtureSuite::load().expect("fixtures");
        let corpus_hash = shared
            .manifest(SharedCorpusView::Core100)
            .expect("manifest")
            .manifest_hash()
            .expect("hash");
        let suite = GeneratedQuerySuite::generate(
            QueryGeneratorSpec {
                seed: 73,
                default_limit: 20,
                include_shared_relevance_queries: true,
            },
            corpus_hash,
            &shared,
        )
        .expect("suite");
        assert_eq!(
            suite
                .cases
                .iter()
                .filter(|case| { matches!(&case.query_kind, GeneratedQueryKind::Harvested { .. }) })
                .count(),
            25
        );
        let globs = suite
            .cases
            .iter()
            .filter_map(|case| match &case.query_kind {
                GeneratedQueryKind::Glob { pattern_class } => Some(*pattern_class),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            globs,
            BTreeSet::from([
                GlobPatternClass::Exact,
                GlobPatternClass::Prefix,
                GlobPatternClass::Suffix,
                GlobPatternClass::Substring,
                GlobPatternClass::Complex,
            ])
        );
        let ranges = suite
            .cases
            .iter()
            .filter_map(|case| match &case.query_kind {
                GeneratedQueryKind::Range { range_class } => Some(*range_class),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            ranges,
            BTreeSet::from([RangeClass::Inclusive, RangeClass::From, RangeClass::To])
        );
        assert!(suite.cases.iter().any(|case| case.offset != 0));
        assert!(suite.cases.iter().any(|case| case.count_requested));
        assert!(suite.cases.iter().any(|case| !case.count_requested));
        assert!(
            suite
                .cases
                .iter()
                .any(|case| case.syntax == QuerySyntax::Cass)
        );
        assert!(
            suite
                .cases
                .iter()
                .any(|case| { case.expected_divergence.as_deref() == Some("DIV-003") })
        );
        suite.manifest.verify(&suite.cases).expect("query replay");
    }

    #[test]
    fn differential_lowering_binds_manifest_and_rejects_lossy_cases() {
        let shared = SharedFixtureSuite::load().expect("fixtures");
        let corpus_hash = shared
            .manifest(SharedCorpusView::Core100)
            .expect("manifest")
            .manifest_hash()
            .expect("hash");
        let suite = GeneratedQuerySuite::generate(
            QueryGeneratorSpec {
                seed: 4,
                default_limit: 10,
                include_shared_relevance_queries: false,
            },
            &corpus_hash,
            &shared,
        )
        .expect("suite");
        let term = suite.cases.iter().find(|case| case.id == "term").unwrap();
        let lowered = term
            .to_differential_case(4, &corpus_hash)
            .expect("default term is representable");
        assert_eq!(
            lowered.metadata.corpus_hash.as_deref(),
            Some(corpus_hash.as_str())
        );
        validate_generated_case_metadata(&lowered).expect("complete generated metadata");
        let paginated = suite
            .cases
            .iter()
            .find(|case| case.id == "paginated")
            .unwrap();
        assert!(paginated.to_differential_case(4, &corpus_hash).is_err());
        let cass = suite
            .cases
            .iter()
            .find(|case| case.id == "boolean-cass")
            .unwrap();
        assert!(cass.to_differential_case(4, &corpus_hash).is_err());
        assert!(term.to_differential_case(4, "0123456789abcdef").is_err());
    }

    #[test]
    fn repository_entries_are_sorted_content_addressed_and_fail_closed() {
        let snapshot = RepositorySnapshot::from_entries(
            "frankensearch",
            [
                RepositoryEntry {
                    relative_path: PathBuf::from("src/z.rs"),
                    bytes: b"fn z() {}\n".to_vec(),
                },
                RepositoryEntry {
                    relative_path: PathBuf::from("README.md"),
                    bytes: b"# repository\n".to_vec(),
                },
                RepositoryEntry {
                    relative_path: PathBuf::from("binary.bin"),
                    bytes: vec![0xff, 0xfe],
                },
                RepositoryEntry {
                    relative_path: PathBuf::from("large.txt"),
                    bytes: vec![b'x'; MAX_DOCUMENT_BYTES as usize + 1],
                },
            ],
        )
        .expect("snapshot");
        assert_eq!(snapshot.documents.len(), 2);
        assert_eq!(snapshot.documents[0].id, "repo:README.md");
        assert_eq!(snapshot.documents[1].id, "repo:src/z.rs");
        assert_eq!(snapshot.manifest.skipped_repository_entries.len(), 2);
        snapshot
            .manifest
            .verify_documents(&snapshot.documents)
            .expect("snapshot replay");
        let source = match &snapshot.manifest.source {
            CorpusSourceManifest::Repository { files, .. } => files,
            _ => panic!("repository source expected"),
        };
        assert_eq!(source[0].path, "README.md");
        assert_eq!(source[1].path, "src/z.rs");
        assert!(
            RepositorySnapshot::from_entries(
                "frankensearch",
                [RepositoryEntry {
                    relative_path: PathBuf::from("../escape"),
                    bytes: Vec::new(),
                }]
            )
            .is_err()
        );
    }

    #[test]
    fn xlarge_recipe_is_exact_and_random_access_without_materialization() {
        let matrix = SyntheticCorpusSpec::xlarge_matrix(101);
        assert_eq!(
            matrix.each_ref().map(|spec| spec.zipf_exponent),
            ZipfExponent::ALL
        );
        let corpus = SyntheticCorpus::new(matrix[1].clone()).expect("xlarge recipe");
        assert_eq!(corpus.len(), XLARGE_DOCUMENT_COUNT);
        assert!(corpus.document_at(XLARGE_DOCUMENT_COUNT - 1).is_some());
        assert!(corpus.document_at(XLARGE_DOCUMENT_COUNT).is_none());
        assert_eq!(
            corpus.iter().size_hint(),
            (
                XLARGE_DOCUMENT_COUNT as usize,
                Some(XLARGE_DOCUMENT_COUNT as usize)
            )
        );
    }

    #[test]
    #[ignore = "nightly-only one-million-document streaming integrity lane"]
    fn nightly_xlarge_stream_hash_integrity() {
        let corpus = SyntheticCorpus::new(SyntheticCorpusSpec::xlarge(0x5eed, ZipfExponent::S11))
            .expect("xlarge recipe");
        let manifest = corpus.manifest().expect("streamed manifest");
        assert_eq!(manifest.document_count, XLARGE_DOCUMENT_COUNT);
        assert_eq!(manifest.content_sha256.len(), 64);
        assert_eq!(manifest.manifest_hash().expect("hash").len(), 64);
    }
}
