use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::GauntletError;

pub const SCORE_EPSILON: f32 = 0.0001;

/// Native secondary ordering evidence retained for every ranked hit.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeTieKey {
    /// Quill's globally assigned document ID.
    QuillDocId { doc_id: u32 },
    /// Tantivy's full segment-qualified document address.
    TantivyDocAddress { segment_ord: u32, doc_id: u32 },
}

/// One engine-native ranked hit. Scores are stored as raw bits in artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankedHit {
    pub doc_id: String,
    pub score_bits: u32,
    pub native_tie_key: NativeTieKey,
}

impl RankedHit {
    /// Recover the score without changing its bit pattern.
    #[must_use]
    pub fn score(&self) -> f32 {
        f32::from_bits(self.score_bits)
    }
}

/// Complete observable output for one differential query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineObservation {
    /// Native top-k order. The comparator never sorts this vector by external ID.
    pub hits: Vec<RankedHit>,
    /// Full oracle cutoff tie group when top-k cuts an exact-score group.
    pub cutoff_tie_group: Vec<RankedHit>,
    /// Whether `cutoff_tie_group` is proven complete rather than fetch-limited.
    pub cutoff_tie_complete: bool,
    /// Snippets keyed by external document ID.
    pub snippets: BTreeMap<String, String>,
    /// Exact match count, or an explicit marker that the case did not request it.
    pub match_count: CountState,
    /// Exact live-document count.
    pub doc_count: u64,
}

/// Query-count observation. Missing evidence is never treated as zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CountState {
    NotRequested,
    Value(u64),
}

/// Reviewed reason that permits `ScoreEpsilon` instead of `RankExact`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreEpsilonReason {
    OracleSegmentGeometry,
    PlatformLibm,
}

/// Comparator configuration encoded without JSON floating-point ambiguity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparatorConfig {
    pub score_epsilon_bits: u32,
    pub score_epsilon_reason: Option<ScoreEpsilonReason>,
}

impl Default for ComparatorConfig {
    fn default() -> Self {
        Self {
            score_epsilon_bits: 0.0001_f32.to_bits(),
            score_epsilon_reason: None,
        }
    }
}

impl ComparatorConfig {
    /// Construct a comparator configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::InvalidComparatorConfig`] unless
    /// `score_epsilon` is the contract-pinned [`SCORE_EPSILON`].
    pub fn new(score_epsilon: f32) -> Result<Self, GauntletError> {
        if score_epsilon.to_bits() != SCORE_EPSILON.to_bits() {
            return Err(GauntletError::InvalidComparatorConfig {
                reason: format!("score epsilon must be the contract-pinned {SCORE_EPSILON}"),
            });
        }
        Ok(Self {
            score_epsilon_bits: score_epsilon.to_bits(),
            score_epsilon_reason: None,
        })
    }

    /// Permit `ScoreEpsilon` with a closed, artifact-visible reason.
    #[must_use]
    pub const fn with_score_epsilon_reason(mut self, reason: ScoreEpsilonReason) -> Self {
        self.score_epsilon_reason = Some(reason);
        self
    }

    #[must_use]
    pub fn score_epsilon(self) -> f32 {
        f32::from_bits(self.score_epsilon_bits)
    }

    pub(crate) fn validate_contract(self) -> Result<(), GauntletError> {
        if self.score_epsilon_bits == SCORE_EPSILON.to_bits() {
            Ok(())
        } else {
            Err(GauntletError::InvalidComparatorConfig {
                reason: format!("score epsilon must be the contract-pinned {SCORE_EPSILON}"),
            })
        }
    }
}

/// Rank-level outcome before snippet and count checks are folded in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankClass {
    RankExact,
    TieOrder,
    ScoreEpsilon,
    RankMismatch,
}

/// Overall comparison posture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonStatus {
    Exact,
    Classified,
    Failed,
}

/// Stable taxonomy used by differential artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceClass {
    TieOrder,
    ScoreEpsilon,
    RankMismatch,
    SnippetMismatch,
    CountMismatch,
    DocumentCountMismatch,
}

impl DivergenceClass {
    const fn is_failure(self) -> bool {
        matches!(
            self,
            Self::RankMismatch
                | Self::SnippetMismatch
                | Self::CountMismatch
                | Self::DocumentCountMismatch
        )
    }
}

/// First-class, pointer-addressable comparison difference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Divergence {
    pub class: DivergenceClass,
    /// RFC 6901 JSON pointer into the containing `ArtifactObject`.
    pub pointer: String,
    pub oracle: String,
    pub subject: String,
}

/// Pure comparator output, including the native-order evidence it evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub status: ComparisonStatus,
    pub rank_class: RankClass,
    pub score_epsilon_reason: Option<ScoreEpsilonReason>,
    pub divergences: Vec<Divergence>,
    pub first_divergence: Option<String>,
    pub subject: EngineObservation,
    pub oracle: EngineObservation,
}

/// Compare two observations while preserving each engine's native order.
///
/// Exact-score tie reorders are classified only after proving that the change
/// remains inside one oracle score group. A cutoff substitution additionally
/// requires a complete expanded oracle tie group. Insufficient evidence fails
/// closed as [`RankClass::RankMismatch`].
///
/// # Errors
///
/// Returns an error for duplicate document IDs, non-finite scores, invalid
/// score ordering, or invalid comparator configuration.
pub fn compare_observations(
    subject: EngineObservation,
    oracle: EngineObservation,
    config: ComparatorConfig,
) -> Result<ComparisonReport, GauntletError> {
    config.validate_contract()?;
    let epsilon = config.score_epsilon();

    validate_observation("subject", &subject)?;
    validate_observation("oracle", &oracle)?;

    let (rank_class, rank_divergence) = classify_rank(
        &subject,
        &oracle,
        epsilon,
        config.score_epsilon_reason.is_some(),
    );
    let mut divergences = Vec::new();
    if let Some(divergence) = rank_divergence {
        divergences.push(divergence);
    }
    compare_snippets(&subject, &oracle, &mut divergences);
    if subject.match_count != oracle.match_count {
        divergences.push(Divergence {
            class: DivergenceClass::CountMismatch,
            pointer: "/comparison/subject/match_count".to_owned(),
            oracle: describe_count(oracle.match_count),
            subject: describe_count(subject.match_count),
        });
    }
    if subject.doc_count != oracle.doc_count {
        divergences.push(Divergence {
            class: DivergenceClass::DocumentCountMismatch,
            pointer: "/comparison/subject/doc_count".to_owned(),
            oracle: oracle.doc_count.to_string(),
            subject: subject.doc_count.to_string(),
        });
    }

    let status = if divergences.is_empty() {
        ComparisonStatus::Exact
    } else if divergences.iter().any(|item| item.class.is_failure()) {
        ComparisonStatus::Failed
    } else {
        ComparisonStatus::Classified
    };
    let first_divergence = divergences.first().map(|item| item.pointer.clone());
    let score_epsilon_reason = if rank_class == RankClass::ScoreEpsilon {
        config.score_epsilon_reason
    } else {
        None
    };

    Ok(ComparisonReport {
        status,
        rank_class,
        score_epsilon_reason,
        divergences,
        first_divergence,
        subject,
        oracle,
    })
}

fn validate_observation(label: &str, observation: &EngineObservation) -> Result<(), GauntletError> {
    validate_hit_slice(label, "hits", &observation.hits, true)?;
    validate_hit_slice(
        label,
        "cutoff_tie_group",
        &observation.cutoff_tie_group,
        false,
    )?;
    validate_cross_evidence_identity(label, &observation.hits, &observation.cutoff_tie_group)?;
    let hit_key = observation.hits.first().map(|hit| &hit.native_tie_key);
    let cutoff_key = observation
        .cutoff_tie_group
        .first()
        .map(|hit| &hit.native_tie_key);
    if hit_key
        .zip(cutoff_key)
        .is_some_and(|(left, right)| !same_tie_key_family(left, right))
    {
        return Err(GauntletError::InvalidObservation {
            reason: format!("{label} mixes native tie-key families across evidence"),
        });
    }
    let hit_ids = observation
        .hits
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect::<BTreeSet<_>>();
    if observation
        .snippets
        .keys()
        .any(|doc_id| !hit_ids.contains(doc_id.as_str()))
    {
        return Err(GauntletError::InvalidObservation {
            reason: format!("{label}.snippets contains a document outside top-k hits"),
        });
    }
    Ok(())
}

fn validate_hit_slice(
    label: &str,
    field: &str,
    hits: &[RankedHit],
    require_score_order: bool,
) -> Result<(), GauntletError> {
    let mut ids = BTreeSet::new();
    let mut native_keys = BTreeSet::new();
    let mut previous_score: Option<f32> = None;
    let mut previous_tie_key: Option<&NativeTieKey> = None;
    let mut tie_key_family: Option<&NativeTieKey> = None;
    for hit in hits {
        if hit.doc_id.is_empty() {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} contains an empty document ID"),
            });
        }
        if !ids.insert(hit.doc_id.as_str()) {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} repeats document ID {}", hit.doc_id),
            });
        }
        if !native_keys.insert(&hit.native_tie_key) {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} repeats a native tie key"),
            });
        }
        let score = hit.score();
        if !score.is_finite() {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} has a non-finite score for {}", hit.doc_id),
            });
        }
        if tie_key_family.is_some_and(|key| !same_tie_key_family(key, &hit.native_tie_key)) {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} mixes native tie-key families"),
            });
        }
        if require_score_order
            && let Some(previous) = previous_score
            && previous.total_cmp(&score).is_lt()
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} is not ordered by descending score"),
            });
        }
        if previous_score.is_some_and(|previous| {
            previous.total_cmp(&score).is_eq()
                && previous_tie_key.is_some_and(|key| key >= &hit.native_tie_key)
        }) {
            return Err(GauntletError::InvalidObservation {
                reason: format!(
                    "{label}.{field} is not ordered by ascending native key inside an exact-score tie"
                ),
            });
        }
        tie_key_family.get_or_insert(&hit.native_tie_key);
        previous_score = Some(score);
        previous_tie_key = Some(&hit.native_tie_key);
    }
    Ok(())
}

fn validate_cross_evidence_identity(
    label: &str,
    hits: &[RankedHit],
    cutoff_tie_group: &[RankedHit],
) -> Result<(), GauntletError> {
    let hits_by_doc = hits
        .iter()
        .map(|hit| (hit.doc_id.as_str(), (hit.score_bits, &hit.native_tie_key)))
        .collect::<BTreeMap<_, _>>();
    let hits_by_native_key = hits
        .iter()
        .map(|hit| (hit.native_tie_key.clone(), hit.doc_id.as_str()))
        .collect::<BTreeMap<_, _>>();

    for cutoff_hit in cutoff_tie_group {
        if let Some((score_bits, native_tie_key)) = hits_by_doc.get(cutoff_hit.doc_id.as_str())
            && (!scores_exact(*score_bits, cutoff_hit.score_bits)
                || *native_tie_key != &cutoff_hit.native_tie_key)
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!(
                    "{label} gives document {} inconsistent top-k and cutoff evidence",
                    cutoff_hit.doc_id
                ),
            });
        }
        if let Some(hit_doc_id) = hits_by_native_key.get(&cutoff_hit.native_tie_key)
            && *hit_doc_id != cutoff_hit.doc_id
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!(
                    "{label} assigns one native tie key to multiple document IDs across evidence"
                ),
            });
        }
    }
    Ok(())
}

fn classify_rank(
    subject: &EngineObservation,
    oracle: &EngineObservation,
    epsilon: f32,
    score_epsilon_allowed: bool,
) -> (RankClass, Option<Divergence>) {
    if sequence_is_exact(&subject.hits, &oracle.hits) {
        return (RankClass::RankExact, None);
    }

    let pointer = first_rank_pointer(&subject.hits, &oracle.hits);
    if is_exact_tie_reorder(&subject.hits, &oracle.hits)
        || is_proven_cutoff_tie_substitution(subject, oracle)
    {
        return (
            RankClass::TieOrder,
            Some(rank_divergence(
                DivergenceClass::TieOrder,
                pointer,
                subject,
                oracle,
            )),
        );
    }

    if score_epsilon_allowed && is_score_epsilon_equivalent(&subject.hits, &oracle.hits, epsilon) {
        return (
            RankClass::ScoreEpsilon,
            Some(rank_divergence(
                DivergenceClass::ScoreEpsilon,
                pointer,
                subject,
                oracle,
            )),
        );
    }

    (
        RankClass::RankMismatch,
        Some(rank_divergence(
            DivergenceClass::RankMismatch,
            pointer,
            subject,
            oracle,
        )),
    )
}

fn same_tie_key_family(left: &NativeTieKey, right: &NativeTieKey) -> bool {
    matches!(
        (left, right),
        (
            NativeTieKey::QuillDocId { .. },
            NativeTieKey::QuillDocId { .. }
        ) | (
            NativeTieKey::TantivyDocAddress { .. },
            NativeTieKey::TantivyDocAddress { .. }
        )
    )
}

fn sequence_is_exact(subject: &[RankedHit], oracle: &[RankedHit]) -> bool {
    subject.len() == oracle.len()
        && subject.iter().zip(oracle).all(|(subject_hit, oracle_hit)| {
            subject_hit.doc_id == oracle_hit.doc_id
                && scores_exact(subject_hit.score_bits, oracle_hit.score_bits)
        })
}

fn is_exact_tie_reorder(subject: &[RankedHit], oracle: &[RankedHit]) -> bool {
    if subject.len() != oracle.len() || subject.is_empty() {
        return false;
    }
    let Some(oracle_map) = score_map(oracle) else {
        return false;
    };
    let Some(subject_map) = score_map(subject) else {
        return false;
    };
    if oracle_map != subject_map {
        return false;
    }

    let groups = exact_group_map(oracle);
    groups_are_nondecreasing(subject, &groups)
}

fn is_proven_cutoff_tie_substitution(
    subject: &EngineObservation,
    oracle: &EngineObservation,
) -> bool {
    if subject.hits.len() != oracle.hits.len()
        || subject.hits.is_empty()
        || !oracle.cutoff_tie_complete
        || oracle.cutoff_tie_group.is_empty()
    {
        return false;
    }

    let cutoff_bits = oracle.hits.last().map(|hit| hit.score_bits);
    let Some(cutoff_bits) = cutoff_bits else {
        return false;
    };
    let group_start = oracle
        .hits
        .iter()
        .position(|hit| scores_exact(hit.score_bits, cutoff_bits))
        .unwrap_or(oracle.hits.len());

    if !sequence_is_exact(&subject.hits[..group_start], &oracle.hits[..group_start]) {
        return false;
    }

    let expansion = oracle
        .cutoff_tie_group
        .iter()
        .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
        .collect::<BTreeMap<_, _>>();
    if expansion.len() != oracle.cutoff_tie_group.len()
        || expansion
            .values()
            .any(|score_bits| !scores_exact(*score_bits, cutoff_bits))
    {
        return false;
    }

    subject.hits[group_start..]
        .iter()
        .chain(&oracle.hits[group_start..])
        .all(|hit| {
            scores_exact(hit.score_bits, cutoff_bits)
                && expansion
                    .get(hit.doc_id.as_str())
                    .is_some_and(|score_bits| scores_exact(*score_bits, cutoff_bits))
        })
}

fn is_score_epsilon_equivalent(subject: &[RankedHit], oracle: &[RankedHit], epsilon: f32) -> bool {
    if subject.len() != oracle.len() || subject.is_empty() {
        return false;
    }
    let Some(oracle_scores) = score_map(oracle) else {
        return false;
    };
    let subject_ids = subject
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect::<BTreeSet<_>>();
    if subject_ids.len() != subject.len()
        || subject_ids != oracle_scores.keys().copied().collect::<BTreeSet<_>>()
    {
        return false;
    }
    if subject.iter().any(|hit| {
        oracle_scores
            .get(hit.doc_id.as_str())
            .is_none_or(|oracle_bits| {
                !within_relative_epsilon(hit.score(), f32::from_bits(*oracle_bits), epsilon)
            })
    }) {
        return false;
    }

    let groups = epsilon_group_map(oracle, epsilon);
    groups_are_nondecreasing(subject, &groups)
}

fn score_map(hits: &[RankedHit]) -> Option<BTreeMap<&str, u32>> {
    let map = hits
        .iter()
        .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
        .collect::<BTreeMap<_, _>>();
    (map.len() == hits.len()).then_some(map)
}

fn exact_group_map(hits: &[RankedHit]) -> BTreeMap<&str, usize> {
    group_map(hits, scores_exact)
}

fn epsilon_group_map(hits: &[RankedHit], epsilon: f32) -> BTreeMap<&str, usize> {
    group_map(hits, |left, right| {
        within_relative_epsilon(f32::from_bits(left), f32::from_bits(right), epsilon)
    })
}

fn group_map(hits: &[RankedHit], adjacent: impl Fn(u32, u32) -> bool) -> BTreeMap<&str, usize> {
    let mut groups = BTreeMap::new();
    let mut group = 0usize;
    let mut previous = None;
    for hit in hits {
        if previous.is_some_and(|score| !adjacent(score, hit.score_bits)) {
            group = group.saturating_add(1);
        }
        groups.insert(hit.doc_id.as_str(), group);
        previous = Some(hit.score_bits);
    }
    groups
}

fn groups_are_nondecreasing(hits: &[RankedHit], groups: &BTreeMap<&str, usize>) -> bool {
    let mut previous = None;
    for hit in hits {
        let Some(group) = groups.get(hit.doc_id.as_str()).copied() else {
            return false;
        };
        if previous.is_some_and(|prior| group < prior) {
            return false;
        }
        previous = Some(group);
    }
    true
}

fn scores_exact(left: u32, right: u32) -> bool {
    f32::from_bits(left)
        .total_cmp(&f32::from_bits(right))
        .is_eq()
}

fn within_relative_epsilon(left: f32, right: f32, epsilon: f32) -> bool {
    let left = f64::from(left);
    let right = f64::from(right);
    let denominator = left.abs().max(right.abs()).max(1e-12);
    (left - right).abs() / denominator <= f64::from(epsilon)
}

fn first_rank_pointer(subject: &[RankedHit], oracle: &[RankedHit]) -> String {
    let index = subject
        .iter()
        .zip(oracle)
        .position(|(subject_hit, oracle_hit)| {
            subject_hit.doc_id != oracle_hit.doc_id
                || !scores_exact(subject_hit.score_bits, oracle_hit.score_bits)
        });
    index.map_or_else(
        || "/comparison/subject/hits".to_owned(),
        |index| format!("/comparison/subject/hits/{index}"),
    )
}

fn rank_divergence(
    class: DivergenceClass,
    pointer: String,
    subject: &EngineObservation,
    oracle: &EngineObservation,
) -> Divergence {
    let index = pointer
        .rsplit('/')
        .next()
        .and_then(|value| value.parse::<usize>().ok());
    Divergence {
        class,
        pointer,
        oracle: index
            .and_then(|index| oracle.hits.get(index))
            .map_or_else(|| oracle.hits.len().to_string(), describe_hit),
        subject: index
            .and_then(|index| subject.hits.get(index))
            .map_or_else(|| subject.hits.len().to_string(), describe_hit),
    }
}

fn describe_hit(hit: &RankedHit) -> String {
    format!("{}@{:08x}", hit.doc_id, hit.score_bits)
}

fn describe_count(count: CountState) -> String {
    match count {
        CountState::NotRequested => "not_requested".to_owned(),
        CountState::Value(value) => value.to_string(),
    }
}

fn compare_snippets(
    subject: &EngineObservation,
    oracle: &EngineObservation,
    divergences: &mut Vec<Divergence>,
) {
    let subject_ids = subject
        .hits
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect::<BTreeSet<_>>();
    let oracle_ids = oracle
        .hits
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect::<BTreeSet<_>>();
    let ids = subject_ids.intersection(&oracle_ids);
    for doc_id in ids {
        let subject_snippet = subject.snippets.get(*doc_id);
        let oracle_snippet = oracle.snippets.get(*doc_id);
        if subject_snippet != oracle_snippet {
            divergences.push(Divergence {
                class: DivergenceClass::SnippetMismatch,
                pointer: format!(
                    "/comparison/{}/snippets/{}",
                    if subject_snippet.is_some() {
                        "subject"
                    } else {
                        "oracle"
                    },
                    escape_json_pointer_token(doc_id)
                ),
                oracle: oracle_snippet
                    .cloned()
                    .unwrap_or_else(|| "<missing>".to_owned()),
                subject: subject_snippet
                    .cloned()
                    .unwrap_or_else(|| "<missing>".to_owned()),
            });
            break;
        }
    }
}

fn escape_json_pointer_token(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quill_hit(doc_id: &str, score: f32, native_doc_id: u32) -> RankedHit {
        RankedHit {
            doc_id: doc_id.to_owned(),
            score_bits: score.to_bits(),
            native_tie_key: NativeTieKey::QuillDocId {
                doc_id: native_doc_id,
            },
        }
    }

    fn tantivy_hit(doc_id: &str, score: f32, doc_id_in_segment: u32) -> RankedHit {
        RankedHit {
            doc_id: doc_id.to_owned(),
            score_bits: score.to_bits(),
            native_tie_key: NativeTieKey::TantivyDocAddress {
                segment_ord: 3,
                doc_id: doc_id_in_segment,
            },
        }
    }

    fn observation(hits: Vec<RankedHit>) -> EngineObservation {
        EngineObservation {
            match_count: CountState::Value(u64::try_from(hits.len()).unwrap_or(u64::MAX)),
            doc_count: 9,
            hits,
            cutoff_tie_group: Vec::new(),
            cutoff_tie_complete: true,
            snippets: BTreeMap::new(),
        }
    }

    #[test]
    fn exact_score_tie_order_is_classified_without_rewriting_native_order() {
        let subject = observation(vec![quill_hit("b", 4.0, 1), quill_hit("a", 4.0, 2)]);
        let oracle = observation(vec![tantivy_hit("a", 4.0, 8), tantivy_hit("b", 4.0, 9)]);

        let report =
            compare_observations(subject.clone(), oracle.clone(), ComparatorConfig::default())
                .expect("tie comparison");

        assert_eq!(report.status, ComparisonStatus::Classified);
        assert_eq!(report.rank_class, RankClass::TieOrder);
        assert_eq!(report.subject.hits, subject.hits);
        assert_eq!(report.oracle.hits, oracle.hits);
        assert_eq!(
            report.first_divergence.as_deref(),
            Some("/comparison/subject/hits/0")
        );
    }

    #[test]
    fn top_k_tie_substitution_requires_complete_expansion() {
        let subject = observation(vec![quill_hit("a", 5.0, 1), quill_hit("c", 4.0, 3)]);
        let mut oracle = observation(vec![tantivy_hit("a", 5.0, 1), tantivy_hit("b", 4.0, 2)]);
        oracle.match_count = CountState::Value(3);
        oracle.cutoff_tie_group = vec![tantivy_hit("b", 4.0, 2), tantivy_hit("c", 4.0, 3)];

        let classified = compare_observations(
            EngineObservation {
                match_count: CountState::Value(3),
                ..subject.clone()
            },
            oracle.clone(),
            ComparatorConfig::default(),
        )
        .expect("complete tie evidence");
        assert_eq!(classified.rank_class, RankClass::TieOrder);

        oracle.cutoff_tie_complete = false;
        let failed = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("incomplete tie evidence fails closed");
        assert_eq!(failed.rank_class, RankClass::RankMismatch);
        assert_eq!(failed.status, ComparisonStatus::Failed);
    }

    #[test]
    fn epsilon_reorders_only_inside_oracle_connected_components() {
        let subject = observation(vec![
            quill_hit("b", 10.0004, 2),
            quill_hit("a", 9.9996, 1),
            quill_hit("c", 8.0, 3),
        ]);
        let oracle = observation(vec![
            tantivy_hit("a", 10.0, 1),
            tantivy_hit("b", 9.9999, 2),
            tantivy_hit("c", 8.0, 3),
        ]);
        let report = compare_observations(
            subject,
            oracle,
            ComparatorConfig::default()
                .with_score_epsilon_reason(ScoreEpsilonReason::OracleSegmentGeometry),
        )
        .expect("epsilon comparison");
        assert_eq!(report.rank_class, RankClass::ScoreEpsilon);
        assert_eq!(
            report.score_epsilon_reason,
            Some(ScoreEpsilonReason::OracleSegmentGeometry)
        );
        assert_eq!(report.status, ComparisonStatus::Classified);
    }

    #[test]
    fn epsilon_above_contract_or_without_reason_is_rejected() {
        assert!(ComparatorConfig::new(0.001).is_err());
        let subject = observation(vec![quill_hit("a", 1.000_05, 1)]);
        let oracle = observation(vec![tantivy_hit("a", 1.0, 1)]);
        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("unreviewed epsilon fails as a rank mismatch");
        assert_eq!(report.rank_class, RankClass::RankMismatch);
        assert_eq!(report.status, ComparisonStatus::Failed);
        assert_eq!(report.score_epsilon_reason, None);
    }

    #[test]
    fn rank_exact_rejects_out_of_order_native_tie_keys() {
        let subject = observation(vec![quill_hit("a", 1.0, 2), quill_hit("b", 1.0, 1)]);
        let oracle = observation(vec![tantivy_hit("a", 1.0, 1), tantivy_hit("b", 1.0, 2)]);
        assert!(compare_observations(subject, oracle, ComparatorConfig::default()).is_err());
    }

    #[test]
    fn duplicate_native_identity_across_scores_is_rejected() {
        let subject = observation(vec![quill_hit("a", 2.0, 7), quill_hit("b", 1.0, 7)]);
        let oracle = observation(vec![tantivy_hit("a", 2.0, 1), tantivy_hit("b", 1.0, 2)]);

        assert!(matches!(
            compare_observations(subject, oracle, ComparatorConfig::default()),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn top_k_and_cutoff_identity_must_be_consistent() {
        let oracle = observation(vec![tantivy_hit("a", 2.0, 1)]);

        let mut changed_score = observation(vec![quill_hit("a", 2.0, 7)]);
        changed_score.cutoff_tie_group = vec![quill_hit("a", 1.0, 7)];
        assert!(matches!(
            compare_observations(changed_score, oracle.clone(), ComparatorConfig::default()),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut changed_doc = observation(vec![quill_hit("a", 2.0, 7)]);
        changed_doc.cutoff_tie_group = vec![quill_hit("b", 2.0, 7)];
        assert!(matches!(
            compare_observations(changed_doc, oracle, ComparatorConfig::default()),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn cutoff_tie_substitution_does_not_compare_unaligned_snippets() {
        let mut subject = observation(vec![quill_hit("a", 5.0, 1), quill_hit("c", 4.0, 3)]);
        let mut oracle = observation(vec![tantivy_hit("a", 5.0, 1), tantivy_hit("b", 4.0, 2)]);
        subject.match_count = CountState::Value(3);
        oracle.match_count = CountState::Value(3);
        oracle.cutoff_tie_group = vec![tantivy_hit("b", 4.0, 2), tantivy_hit("c", 4.0, 3)];
        subject.snippets.insert("c".to_owned(), "c body".to_owned());
        oracle.snippets.insert("b".to_owned(), "b body".to_owned());

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("complete cutoff tie evidence");
        assert_eq!(report.rank_class, RankClass::TieOrder);
        assert_eq!(report.status, ComparisonStatus::Classified);
    }

    #[test]
    fn snippet_and_counts_have_stable_json_pointers() {
        let mut subject = observation(vec![quill_hit("a/b~c", 1.0, 1)]);
        let mut oracle = observation(vec![tantivy_hit("a/b~c", 1.0, 1)]);
        subject
            .snippets
            .insert("a/b~c".to_owned(), "left".to_owned());
        oracle
            .snippets
            .insert("a/b~c".to_owned(), "right".to_owned());
        subject.match_count = CountState::Value(2);
        subject.doc_count = 8;

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("surface comparison");
        let pointers = report
            .divergences
            .iter()
            .map(|item| item.pointer.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            pointers,
            vec![
                "/comparison/subject/snippets/a~1b~0c",
                "/comparison/subject/match_count",
                "/comparison/subject/doc_count"
            ]
        );
        let projection = serde_json::json!({ "comparison": &report });
        assert!(
            report
                .divergences
                .iter()
                .all(|divergence| projection.pointer(&divergence.pointer).is_some())
        );
        assert_eq!(report.status, ComparisonStatus::Failed);
    }
}
