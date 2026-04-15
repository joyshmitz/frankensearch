use serde::{Deserialize, Serialize};
use serde_json::Number;
use std::path::Path;

pub const FILE_CLASSIFICATION_CONTRACT_KIND: &str = "fsfs_file_classification_contract_definition";
pub const FILE_CLASSIFICATION_DECISION_KIND: &str = "fsfs_file_classification_decision";
pub const FILE_CLASSIFICATION_CORRUPT_EVENT_KIND: &str = "fsfs_file_classification_corrupt_event";
pub const FILE_CLASSIFICATION_SCHEMA_VERSION: u32 = 1;

const BOM_NONE: &str = "none";
const BOM_UTF8: &str = "utf8";
const BOM_UTF16_LE: &str = "utf16le";
const BOM_UTF16_BE: &str = "utf16be";
const ENCODING_NONE: &str = "none";
const ENCODING_UTF8: &str = "utf-8";
const ENCODING_UTF16_LE: &str = "utf-16le";
const ENCODING_UTF16_BE: &str = "utf-16be";
const ENCODING_UNKNOWN_8BIT: &str = "unknown-8bit";
const NORMALIZATION_NONE: &str = "none";
const FSFS_TEXT_UTF8_BOM: &str = "FSFS_TEXT_UTF8_BOM";
const FSFS_TEXT_UTF8_HIGH_CONFIDENCE: &str = "FSFS_TEXT_UTF8_HIGH_CONFIDENCE";
const FSFS_TEXT_UTF16_REQUIRES_TRANSCODE: &str = "FSFS_TEXT_UTF16_REQUIRES_TRANSCODE";
const FSFS_TEXT_HEURISTIC_LOSSY_DECODE: &str = "FSFS_TEXT_HEURISTIC_LOSSY_DECODE";
const FSFS_TEXT_HEURISTIC_SKIP: &str = "FSFS_TEXT_HEURISTIC_SKIP";
const FSFS_TEXT_HEURISTIC_QUARANTINE: &str = "FSFS_TEXT_HEURISTIC_QUARANTINE";
const FSFS_BINARY_NULL_BYTE_DETECTED: &str = "FSFS_BINARY_NULL_BYTE_DETECTED";
const FSFS_BINARY_HEURISTIC_THRESHOLD: &str = "FSFS_BINARY_HEURISTIC_THRESHOLD";
const FSFS_ARCHIVE_EXTENSION_BLOCKED: &str = "FSFS_ARCHIVE_EXTENSION_BLOCKED";
const FSFS_PARTIAL_TRUNCATED_PREFIX_ONLY: &str = "FSFS_PARTIAL_TRUNCATED_PREFIX_ONLY";
const FSFS_PARTIAL_ENCODING_REQUIRES_TRANSCODE: &str = "FSFS_PARTIAL_ENCODING_REQUIRES_TRANSCODE";
const FSFS_PARTIAL_HEURISTIC_LOSSY_DECODE: &str = "FSFS_PARTIAL_HEURISTIC_LOSSY_DECODE";
const FSFS_PARTIAL_HEURISTIC_SKIP: &str = "FSFS_PARTIAL_HEURISTIC_SKIP";
const FSFS_PARTIAL_HEURISTIC_QUARANTINE: &str = "FSFS_PARTIAL_HEURISTIC_QUARANTINE";
const FSFS_CORRUPT_CHECKSUM_MISMATCH: &str = "FSFS_CORRUPT_CHECKSUM_MISMATCH";
const FSFS_CORRUPT_DECODE_ERROR: &str = "FSFS_CORRUPT_DECODE_ERROR";
const FSFS_CORRUPT_IO_SHORT_READ: &str = "FSFS_CORRUPT_IO_SHORT_READ";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SniffHeuristics {
    pub max_probe_bytes: u32,
    pub binary_byte_threshold_pct: Number,
    pub high_bit_ratio_threshold_pct: Number,
    pub null_byte_hard_binary: bool,
}

impl SniffHeuristics {
    #[must_use]
    pub fn binary_byte_threshold_pct(&self) -> f64 {
        self.binary_byte_threshold_pct.as_f64().unwrap_or(30.0)
    }

    #[must_use]
    pub fn high_bit_ratio_threshold_pct(&self) -> f64 {
        self.high_bit_ratio_threshold_pct.as_f64().unwrap_or(60.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EncodingDetector {
    Bom,
    Utf8Validation,
    IcuHeuristic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationPolicy {
    Utf8Nfc,
    Utf8NfcLossy,
    RejectNonText,
}

impl NormalizationPolicy {
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Utf8Nfc => "utf8_nfc",
            Self::Utf8NfcLossy => "utf8_nfc_lossy",
            Self::RejectNonText => NORMALIZATION_NONE,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UnknownEncodingAction {
    Skip,
    LossyDecode,
    Quarantine,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncodingPolicy {
    pub primary_detectors: Vec<EncodingDetector>,
    pub normalization: NormalizationPolicy,
    pub unknown_encoding_action: UnknownEncodingAction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TruncatedAction {
    Quarantine,
    Skip,
    IndexPartialWithFlag,
}

impl TruncatedAction {
    #[must_use]
    pub const fn ingest_action(&self) -> IngestAction {
        match self {
            Self::Quarantine => IngestAction::Quarantine,
            Self::Skip => IngestAction::Skip,
            Self::IndexPartialWithFlag => IngestAction::IndexPartialWithFlag,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChecksumMismatchAction {
    Quarantine,
    Skip,
}

impl ChecksumMismatchAction {
    #[must_use]
    pub const fn ingest_action(&self) -> IngestAction {
        match self {
            Self::Quarantine => IngestAction::Quarantine,
            Self::Skip => IngestAction::Skip,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorruptPartialPolicy {
    pub truncated_action: TruncatedAction,
    pub checksum_mismatch_action: ChecksumMismatchAction,
    pub max_recovery_prefix_bytes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfidenceSignals {
    pub min_confidence_for_text: f64,
    pub min_confidence_for_lossy_decode: f64,
    pub emit_required_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileClassificationContractDefinition {
    pub kind: String,
    pub v: u32,
    pub sniff_heuristics: SniffHeuristics,
    pub encoding_policy: EncodingPolicy,
    pub corrupt_partial_policy: CorruptPartialPolicy,
    pub confidence_signals: ConfidenceSignals,
}

impl Default for FileClassificationContractDefinition {
    fn default() -> Self {
        Self {
            kind: FILE_CLASSIFICATION_CONTRACT_KIND.to_string(),
            v: FILE_CLASSIFICATION_SCHEMA_VERSION,
            sniff_heuristics: SniffHeuristics {
                max_probe_bytes: 8_192,
                binary_byte_threshold_pct: Number::from(30),
                high_bit_ratio_threshold_pct: Number::from(60),
                null_byte_hard_binary: true,
            },
            encoding_policy: EncodingPolicy {
                primary_detectors: vec![
                    EncodingDetector::Bom,
                    EncodingDetector::Utf8Validation,
                    EncodingDetector::IcuHeuristic,
                ],
                normalization: NormalizationPolicy::Utf8Nfc,
                unknown_encoding_action: UnknownEncodingAction::Quarantine,
            },
            corrupt_partial_policy: CorruptPartialPolicy {
                truncated_action: TruncatedAction::IndexPartialWithFlag,
                checksum_mismatch_action: ChecksumMismatchAction::Quarantine,
                max_recovery_prefix_bytes: 16_384,
            },
            confidence_signals: ConfidenceSignals {
                min_confidence_for_text: 0.8,
                min_confidence_for_lossy_decode: 0.9,
                emit_required_fields: vec![
                    "classification_confidence".to_string(),
                    "encoding_confidence".to_string(),
                    "reason_code".to_string(),
                    "utility_penalty".to_string(),
                    "skip_candidate".to_string(),
                    "requires_manual_review".to_string(),
                ],
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SniffFeatures {
    pub null_bytes: u32,
    pub non_printable_ratio: f64,
    pub high_bit_ratio: f64,
    pub bom: String,
}

impl SniffFeatures {
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self {
                null_bytes: 0,
                non_printable_ratio: 0.0,
                high_bit_ratio: 0.0,
                bom: BOM_NONE.to_string(),
            };
        }

        let mut null_bytes = 0_u32;
        let mut non_printable = 0_u32;
        let mut high_bit = 0_u32;

        for &byte in bytes {
            if byte == 0 {
                null_bytes = null_bytes.saturating_add(1);
            }
            if is_non_printable(byte) {
                non_printable = non_printable.saturating_add(1);
            }
            if byte >= 0x80 {
                high_bit = high_bit.saturating_add(1);
            }
        }

        let sample_len = bytes.len() as f64;
        Self {
            null_bytes,
            non_printable_ratio: f64::from(non_printable) / sample_len,
            high_bit_ratio: f64::from(high_bit) / sample_len,
            bom: detect_bom(bytes).to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DetectedType {
    Text,
    Binary,
    Archive,
    Partial,
    Corrupt,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IngestAction {
    Index,
    Skip,
    Quarantine,
    IndexPartialWithFlag,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DownstreamSignals {
    pub utility_penalty: f64,
    pub skip_candidate: bool,
    pub requires_manual_review: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileClassificationDecision {
    pub kind: String,
    pub v: u32,
    pub path: String,
    pub size_bytes: u64,
    pub probe_bytes: u64,
    pub sniff_features: SniffFeatures,
    pub detected_type: DetectedType,
    pub detected_encoding: String,
    pub normalization_applied: String,
    pub ingest_action: IngestAction,
    pub classification_confidence: f64,
    pub encoding_confidence: f64,
    pub reason_code: String,
    pub downstream_signals: DownstreamSignals,
}

impl FileClassificationDecision {
    #[must_use]
    pub fn satisfies_contract(&self) -> bool {
        let confidence_ok = (0.0..=1.0).contains(&self.classification_confidence)
            && (0.0..=1.0).contains(&self.encoding_confidence);
        let reason_code_ok = self.reason_code.starts_with("FSFS_")
            && self
                .reason_code
                .chars()
                .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_');
        let encoding_is_none = self.detected_encoding == ENCODING_NONE;
        let normalization_is_none = self.normalization_applied == NORMALIZATION_NONE;
        let encoding_contract_ok = match self.detected_type {
            DetectedType::Binary | DetectedType::Archive | DetectedType::Corrupt => {
                encoding_is_none && normalization_is_none
            }
            DetectedType::Text | DetectedType::Partial => !self.detected_encoding.is_empty(),
        };
        let ingest_contract_ok = match self.detected_type {
            DetectedType::Archive | DetectedType::Corrupt => {
                matches!(
                    self.ingest_action,
                    IngestAction::Skip | IngestAction::Quarantine
                )
            }
            DetectedType::Binary | DetectedType::Text | DetectedType::Partial => true,
        };

        confidence_ok && reason_code_ok && encoding_contract_ok && ingest_contract_ok
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    Truncated,
    ChecksumMismatch,
    DecodeError,
    IoShortRead,
}

impl ErrorClass {
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            Self::Truncated => FSFS_PARTIAL_TRUNCATED_PREFIX_ONLY,
            Self::ChecksumMismatch => FSFS_CORRUPT_CHECKSUM_MISMATCH,
            Self::DecodeError => FSFS_CORRUPT_DECODE_ERROR,
            Self::IoShortRead => FSFS_CORRUPT_IO_SHORT_READ,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileClassificationCorruptEvent {
    pub kind: String,
    pub v: u32,
    pub path: String,
    pub error_class: ErrorClass,
    pub bytes_recovered: u64,
    pub ingest_action: IngestAction,
    pub reason_code: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrityState {
    Clean,
    Truncated,
    ChecksumMismatch,
    DecodeError,
    IoShortRead,
}

impl IntegrityState {
    #[must_use]
    pub const fn error_class(self) -> Option<ErrorClass> {
        match self {
            Self::Clean => None,
            Self::Truncated => Some(ErrorClass::Truncated),
            Self::ChecksumMismatch => Some(ErrorClass::ChecksumMismatch),
            Self::DecodeError => Some(ErrorClass::DecodeError),
            Self::IoShortRead => Some(ErrorClass::IoShortRead),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileClassificationInput<'a> {
    pub path: &'a str,
    pub size_bytes: u64,
    pub probe: &'a [u8],
    pub integrity: IntegrityState,
}

impl<'a> FileClassificationInput<'a> {
    #[must_use]
    pub fn from_bytes(path: &'a str, bytes: &'a [u8]) -> Self {
        Self {
            path,
            size_bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            probe: bytes,
            integrity: IntegrityState::Clean,
        }
    }

    #[must_use]
    pub fn with_integrity(path: &'a str, bytes: &'a [u8], integrity: IntegrityState) -> Self {
        Self {
            integrity,
            ..Self::from_bytes(path, bytes)
        }
    }
}

impl FileClassificationContractDefinition {
    #[must_use]
    pub fn classify(&self, input: FileClassificationInput<'_>) -> FileClassificationDecision {
        let capped_probe = self.cap_probe(input.probe);
        let sniff_features = SniffFeatures::from_bytes(capped_probe);
        let probe_len = u64::try_from(capped_probe.len()).unwrap_or(u64::MAX);

        if let Some(error_class) = input.integrity.error_class() {
            if !matches!(error_class, ErrorClass::Truncated) {
                return self.corrupt_decision(input, probe_len, sniff_features, error_class);
            }
        }

        if is_archive_path(input.path) {
            return FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features,
                detected_type: DetectedType::Archive,
                detected_encoding: ENCODING_NONE.to_string(),
                normalization_applied: NORMALIZATION_NONE.to_string(),
                ingest_action: IngestAction::Skip,
                classification_confidence: 0.99,
                encoding_confidence: 0.0,
                reason_code: FSFS_ARCHIVE_EXTENSION_BLOCKED.to_string(),
                downstream_signals: downstream(0.95, true, false),
            };
        }

        let utf8_valid = std::str::from_utf8(capped_probe).is_ok();
        if self.is_binary(&sniff_features, utf8_valid) {
            return FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features: sniff_features.clone(),
                detected_type: DetectedType::Binary,
                detected_encoding: ENCODING_NONE.to_string(),
                normalization_applied: NORMALIZATION_NONE.to_string(),
                ingest_action: IngestAction::Skip,
                classification_confidence: if sniff_features.null_bytes > 0 {
                    0.99
                } else {
                    0.9
                },
                encoding_confidence: 0.0,
                reason_code: if sniff_features.null_bytes > 0 {
                    FSFS_BINARY_NULL_BYTE_DETECTED.to_string()
                } else {
                    FSFS_BINARY_HEURISTIC_THRESHOLD.to_string()
                },
                downstream_signals: downstream(0.9, true, false),
            };
        }

        let encoding = detect_encoding(capped_probe, &sniff_features, utf8_valid);
        if matches!(input.integrity, IntegrityState::Truncated) {
            return self.partial_decision(input, probe_len, sniff_features, encoding);
        }

        self.text_decision(input, probe_len, sniff_features, encoding)
    }

    #[must_use]
    pub fn classify_bytes(&self, path: &str, bytes: &[u8]) -> FileClassificationDecision {
        self.classify(FileClassificationInput::from_bytes(path, bytes))
    }

    #[must_use]
    pub fn classify_with_integrity(
        &self,
        path: &str,
        bytes: &[u8],
        integrity: IntegrityState,
    ) -> FileClassificationDecision {
        self.classify(FileClassificationInput::with_integrity(
            path, bytes, integrity,
        ))
    }

    #[must_use]
    pub fn build_corrupt_event(
        &self,
        path: impl Into<String>,
        error_class: ErrorClass,
        bytes_recovered: u64,
    ) -> FileClassificationCorruptEvent {
        let ingest_action = match error_class {
            ErrorClass::Truncated => self.corrupt_partial_policy.truncated_action.ingest_action(),
            ErrorClass::ChecksumMismatch => self
                .corrupt_partial_policy
                .checksum_mismatch_action
                .ingest_action(),
            ErrorClass::DecodeError | ErrorClass::IoShortRead => IngestAction::Quarantine,
        };

        FileClassificationCorruptEvent {
            kind: FILE_CLASSIFICATION_CORRUPT_EVENT_KIND.to_string(),
            v: FILE_CLASSIFICATION_SCHEMA_VERSION,
            path: path.into(),
            error_class,
            bytes_recovered,
            ingest_action,
            reason_code: error_class.reason_code().to_string(),
        }
    }

    #[must_use]
    fn cap_probe<'a>(&self, probe: &'a [u8]) -> &'a [u8] {
        let max_probe =
            usize::try_from(self.sniff_heuristics.max_probe_bytes).unwrap_or(usize::MAX);
        &probe[..probe.len().min(max_probe)]
    }

    #[must_use]
    fn is_binary(&self, sniff_features: &SniffFeatures, utf8_valid: bool) -> bool {
        if self.sniff_heuristics.null_byte_hard_binary && sniff_features.null_bytes > 0 {
            return true;
        }

        let non_printable_pct = sniff_features.non_printable_ratio * 100.0;
        if non_printable_pct >= self.sniff_heuristics.binary_byte_threshold_pct() {
            return true;
        }

        let high_bit_pct = sniff_features.high_bit_ratio * 100.0;
        !utf8_valid && high_bit_pct >= self.sniff_heuristics.high_bit_ratio_threshold_pct()
    }

    #[must_use]
    fn corrupt_decision(
        &self,
        input: FileClassificationInput<'_>,
        probe_len: u64,
        sniff_features: SniffFeatures,
        error_class: ErrorClass,
    ) -> FileClassificationDecision {
        let ingest_action = match error_class {
            ErrorClass::ChecksumMismatch => self
                .corrupt_partial_policy
                .checksum_mismatch_action
                .ingest_action(),
            ErrorClass::DecodeError | ErrorClass::IoShortRead => IngestAction::Quarantine,
            ErrorClass::Truncated => self.corrupt_partial_policy.truncated_action.ingest_action(),
        };

        FileClassificationDecision {
            kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
            v: FILE_CLASSIFICATION_SCHEMA_VERSION,
            path: input.path.to_string(),
            size_bytes: input.size_bytes,
            probe_bytes: probe_len,
            sniff_features,
            detected_type: DetectedType::Corrupt,
            detected_encoding: ENCODING_NONE.to_string(),
            normalization_applied: NORMALIZATION_NONE.to_string(),
            ingest_action,
            classification_confidence: 0.95,
            encoding_confidence: 0.0,
            reason_code: error_class.reason_code().to_string(),
            downstream_signals: downstream(1.0, true, true),
        }
    }

    #[must_use]
    fn partial_decision(
        &self,
        input: FileClassificationInput<'_>,
        probe_len: u64,
        sniff_features: SniffFeatures,
        encoding: EncodingAssessment,
    ) -> FileClassificationDecision {
        match encoding {
            EncodingAssessment::Utf8 { bom } => FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features,
                detected_type: DetectedType::Partial,
                detected_encoding: ENCODING_UTF8.to_string(),
                normalization_applied: self.encoding_policy.normalization.label().to_string(),
                ingest_action: self.corrupt_partial_policy.truncated_action.ingest_action(),
                classification_confidence: clamp_unit(
                    self.confidence_signals.min_confidence_for_text.max(0.82),
                ),
                encoding_confidence: if bom == BOM_UTF8 { 1.0 } else { 0.98 },
                reason_code: FSFS_PARTIAL_TRUNCATED_PREFIX_ONLY.to_string(),
                downstream_signals: downstream(
                    if matches!(
                        self.corrupt_partial_policy.truncated_action,
                        TruncatedAction::IndexPartialWithFlag
                    ) {
                        0.35
                    } else {
                        0.8
                    },
                    true,
                    true,
                ),
            },
            EncodingAssessment::Utf16 { label } => FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features,
                detected_type: DetectedType::Partial,
                detected_encoding: label.to_string(),
                normalization_applied: NORMALIZATION_NONE.to_string(),
                ingest_action: IngestAction::Quarantine,
                classification_confidence: 0.9,
                encoding_confidence: 0.99,
                reason_code: FSFS_PARTIAL_ENCODING_REQUIRES_TRANSCODE.to_string(),
                downstream_signals: downstream(0.8, true, true),
            },
            EncodingAssessment::Heuristic8Bit { confidence } => {
                let lossy_allowed =
                    confidence >= self.confidence_signals.min_confidence_for_lossy_decode;
                let (ingest_action, normalization_applied, reason_code) =
                    match self.encoding_policy.unknown_encoding_action {
                        UnknownEncodingAction::LossyDecode if lossy_allowed => (
                            self.corrupt_partial_policy.truncated_action.ingest_action(),
                            NormalizationPolicy::Utf8NfcLossy.label().to_string(),
                            FSFS_PARTIAL_HEURISTIC_LOSSY_DECODE.to_string(),
                        ),
                        UnknownEncodingAction::LossyDecode | UnknownEncodingAction::Quarantine => (
                            IngestAction::Quarantine,
                            NORMALIZATION_NONE.to_string(),
                            FSFS_PARTIAL_HEURISTIC_QUARANTINE.to_string(),
                        ),
                        UnknownEncodingAction::Skip => (
                            IngestAction::Skip,
                            NORMALIZATION_NONE.to_string(),
                            FSFS_PARTIAL_HEURISTIC_SKIP.to_string(),
                        ),
                    };

                FileClassificationDecision {
                    kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                    v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                    path: input.path.to_string(),
                    size_bytes: input.size_bytes,
                    probe_bytes: probe_len,
                    sniff_features,
                    detected_type: DetectedType::Partial,
                    detected_encoding: ENCODING_UNKNOWN_8BIT.to_string(),
                    normalization_applied,
                    ingest_action,
                    classification_confidence: clamp_unit(confidence.max(0.7)),
                    encoding_confidence: confidence,
                    reason_code,
                    downstream_signals: downstream(0.65, true, true),
                }
            }
        }
    }

    #[must_use]
    fn text_decision(
        &self,
        input: FileClassificationInput<'_>,
        probe_len: u64,
        sniff_features: SniffFeatures,
        encoding: EncodingAssessment,
    ) -> FileClassificationDecision {
        match encoding {
            EncodingAssessment::Utf8 { bom } => FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features,
                detected_type: DetectedType::Text,
                detected_encoding: ENCODING_UTF8.to_string(),
                normalization_applied: self.encoding_policy.normalization.label().to_string(),
                ingest_action: IngestAction::Index,
                classification_confidence: clamp_unit(
                    self.confidence_signals.min_confidence_for_text.max(0.95),
                ),
                encoding_confidence: if bom == BOM_UTF8 { 1.0 } else { 0.98 },
                reason_code: if bom == BOM_UTF8 {
                    FSFS_TEXT_UTF8_BOM.to_string()
                } else {
                    FSFS_TEXT_UTF8_HIGH_CONFIDENCE.to_string()
                },
                downstream_signals: downstream(0.02, false, false),
            },
            EncodingAssessment::Utf16 { label } => FileClassificationDecision {
                kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                path: input.path.to_string(),
                size_bytes: input.size_bytes,
                probe_bytes: probe_len,
                sniff_features,
                detected_type: DetectedType::Text,
                detected_encoding: label.to_string(),
                normalization_applied: NORMALIZATION_NONE.to_string(),
                ingest_action: IngestAction::Quarantine,
                classification_confidence: 0.92,
                encoding_confidence: 0.99,
                reason_code: FSFS_TEXT_UTF16_REQUIRES_TRANSCODE.to_string(),
                downstream_signals: downstream(0.7, true, true),
            },
            EncodingAssessment::Heuristic8Bit { confidence } => {
                let lossy_allowed =
                    confidence >= self.confidence_signals.min_confidence_for_lossy_decode;
                let (ingest_action, normalization_applied, reason_code, utility_penalty) =
                    match self.encoding_policy.unknown_encoding_action {
                        UnknownEncodingAction::LossyDecode if lossy_allowed => (
                            IngestAction::Index,
                            NormalizationPolicy::Utf8NfcLossy.label().to_string(),
                            FSFS_TEXT_HEURISTIC_LOSSY_DECODE.to_string(),
                            0.25,
                        ),
                        UnknownEncodingAction::LossyDecode | UnknownEncodingAction::Quarantine => (
                            IngestAction::Quarantine,
                            NORMALIZATION_NONE.to_string(),
                            FSFS_TEXT_HEURISTIC_QUARANTINE.to_string(),
                            0.75,
                        ),
                        UnknownEncodingAction::Skip => (
                            IngestAction::Skip,
                            NORMALIZATION_NONE.to_string(),
                            FSFS_TEXT_HEURISTIC_SKIP.to_string(),
                            0.85,
                        ),
                    };

                FileClassificationDecision {
                    kind: FILE_CLASSIFICATION_DECISION_KIND.to_string(),
                    v: FILE_CLASSIFICATION_SCHEMA_VERSION,
                    path: input.path.to_string(),
                    size_bytes: input.size_bytes,
                    probe_bytes: probe_len,
                    sniff_features,
                    detected_type: DetectedType::Text,
                    detected_encoding: ENCODING_UNKNOWN_8BIT.to_string(),
                    normalization_applied,
                    ingest_action: ingest_action.clone(),
                    classification_confidence: clamp_unit(confidence.max(0.8)),
                    encoding_confidence: confidence,
                    reason_code,
                    downstream_signals: downstream(
                        utility_penalty,
                        !matches!(ingest_action, IngestAction::Index),
                        !matches!(ingest_action, IngestAction::Index),
                    ),
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EncodingAssessment {
    Utf8 { bom: &'static str },
    Utf16 { label: &'static str },
    Heuristic8Bit { confidence: f64 },
}

#[must_use]
fn is_non_printable(byte: u8) -> bool {
    !matches!(byte, b'\t' | b'\n' | b'\r') && (byte < 0x20 || byte == 0x7f)
}

#[must_use]
fn detect_bom(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
        BOM_UTF8
    } else if bytes.starts_with(&[0xff, 0xfe]) {
        BOM_UTF16_LE
    } else if bytes.starts_with(&[0xfe, 0xff]) {
        BOM_UTF16_BE
    } else {
        BOM_NONE
    }
}

#[must_use]
fn detect_encoding(
    _bytes: &[u8],
    sniff_features: &SniffFeatures,
    utf8_valid: bool,
) -> EncodingAssessment {
    match sniff_features.bom.as_str() {
        BOM_UTF16_LE => EncodingAssessment::Utf16 {
            label: ENCODING_UTF16_LE,
        },
        BOM_UTF16_BE => EncodingAssessment::Utf16 {
            label: ENCODING_UTF16_BE,
        },
        BOM_UTF8 if utf8_valid => EncodingAssessment::Utf8 { bom: BOM_UTF8 },
        _ if utf8_valid => EncodingAssessment::Utf8 { bom: BOM_NONE },
        _ => {
            let confidence = clamp_unit(
                1.0 - (sniff_features.non_printable_ratio * 0.35)
                    - (sniff_features.high_bit_ratio * 0.1),
            );
            EncodingAssessment::Heuristic8Bit { confidence }
        }
    }
}

#[must_use]
fn downstream(
    utility_penalty: f64,
    skip_candidate: bool,
    requires_manual_review: bool,
) -> DownstreamSignals {
    DownstreamSignals {
        utility_penalty,
        skip_candidate,
        requires_manual_review,
    }
}

#[must_use]
fn clamp_unit(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

#[must_use]
fn is_archive_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    if [
        ".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst", ".tgz", ".tbz2", ".txz",
    ]
    .iter()
    .any(|suffix| lower.ends_with(suffix))
    {
        return true;
    }

    Path::new(&lower)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension,
                "7z" | "apk"
                    | "bz2"
                    | "crate"
                    | "dmg"
                    | "ear"
                    | "gz"
                    | "iso"
                    | "jar"
                    | "pkg"
                    | "rar"
                    | "tar"
                    | "war"
                    | "whl"
                    | "xz"
                    | "zip"
                    | "zst"
            )
        })
}

#[cfg(test)]
mod tests {
    use super::{
        ChecksumMismatchAction, DetectedType, ErrorClass, FILE_CLASSIFICATION_CONTRACT_KIND,
        FILE_CLASSIFICATION_CORRUPT_EVENT_KIND, FILE_CLASSIFICATION_DECISION_KIND,
        FILE_CLASSIFICATION_SCHEMA_VERSION, FileClassificationContractDefinition, IngestAction,
        IntegrityState, NormalizationPolicy, TruncatedAction,
    };

    #[test]
    fn default_contract_matches_documented_defaults() {
        let contract = FileClassificationContractDefinition::default();
        assert_eq!(contract.kind, FILE_CLASSIFICATION_CONTRACT_KIND);
        assert_eq!(contract.v, FILE_CLASSIFICATION_SCHEMA_VERSION);
        assert_eq!(contract.sniff_heuristics.max_probe_bytes, 8_192);
        assert_eq!(
            contract.sniff_heuristics.binary_byte_threshold_pct.as_u64(),
            Some(30)
        );
        assert_eq!(
            contract
                .sniff_heuristics
                .high_bit_ratio_threshold_pct
                .as_u64(),
            Some(60)
        );
        assert!(contract.sniff_heuristics.null_byte_hard_binary);
        assert_eq!(
            contract.encoding_policy.normalization,
            NormalizationPolicy::Utf8Nfc
        );
        assert_eq!(
            contract.corrupt_partial_policy.truncated_action,
            TruncatedAction::IndexPartialWithFlag
        );
        assert_eq!(
            contract.corrupt_partial_policy.checksum_mismatch_action,
            ChecksumMismatchAction::Quarantine
        );
    }

    #[test]
    fn classify_utf8_text_indexes_normally() {
        let contract = FileClassificationContractDefinition::default();
        let decision = contract.classify_bytes(
            "/workspace/src/lib.rs",
            b"pub fn score() {\n    println!(\"hello\");\n}\n",
        );

        assert_eq!(decision.kind, FILE_CLASSIFICATION_DECISION_KIND);
        assert_eq!(decision.detected_type, DetectedType::Text);
        assert_eq!(decision.detected_encoding, "utf-8");
        assert_eq!(decision.normalization_applied, "utf8_nfc");
        assert_eq!(decision.ingest_action, IngestAction::Index);
        assert_eq!(decision.reason_code, "FSFS_TEXT_UTF8_HIGH_CONFIDENCE");
        assert!(decision.satisfies_contract());
    }

    #[test]
    fn classify_binary_with_null_bytes_skips() {
        let contract = FileClassificationContractDefinition::default();
        let decision = contract.classify_bytes(
            "/workspace/tmp/blob.bin",
            &[0, 17, 2, 144, 0, 31, 7, 0, 255],
        );

        assert_eq!(decision.detected_type, DetectedType::Binary);
        assert_eq!(decision.detected_encoding, "none");
        assert_eq!(decision.normalization_applied, "none");
        assert_eq!(decision.ingest_action, IngestAction::Skip);
        assert_eq!(decision.reason_code, "FSFS_BINARY_NULL_BYTE_DETECTED");
        assert!(decision.satisfies_contract());
    }

    #[test]
    fn classify_archive_path_skips_before_text_decode() {
        let contract = FileClassificationContractDefinition::default();
        let decision = contract.classify_bytes(
            "/workspace/build/cache.tar.gz",
            b"this would otherwise be valid utf8",
        );

        assert_eq!(decision.detected_type, DetectedType::Archive);
        assert_eq!(decision.ingest_action, IngestAction::Skip);
        assert_eq!(decision.detected_encoding, "none");
        assert_eq!(decision.reason_code, "FSFS_ARCHIVE_EXTENSION_BLOCKED");
        assert!(decision.satisfies_contract());
    }

    #[test]
    fn classify_truncated_utf8_text_uses_partial_policy() {
        let contract = FileClassificationContractDefinition::default();
        let decision = contract.classify_with_integrity(
            "/workspace/logs/partial.md",
            b"# heading\nbody that was cut off",
            IntegrityState::Truncated,
        );

        assert_eq!(decision.detected_type, DetectedType::Partial);
        assert_eq!(decision.detected_encoding, "utf-8");
        assert_eq!(decision.ingest_action, IngestAction::IndexPartialWithFlag);
        assert_eq!(decision.reason_code, "FSFS_PARTIAL_TRUNCATED_PREFIX_ONLY");
        assert!(decision.downstream_signals.requires_manual_review);
        assert!(decision.satisfies_contract());
    }

    #[test]
    fn classify_checksum_mismatch_quarantines_corrupt_input() {
        let contract = FileClassificationContractDefinition::default();
        let decision = contract.classify_with_integrity(
            "/workspace/bad/corrupt.txt",
            b"definitely not trustworthy",
            IntegrityState::ChecksumMismatch,
        );

        assert_eq!(decision.detected_type, DetectedType::Corrupt);
        assert_eq!(decision.ingest_action, IngestAction::Quarantine);
        assert_eq!(decision.detected_encoding, "none");
        assert_eq!(decision.reason_code, "FSFS_CORRUPT_CHECKSUM_MISMATCH");
        assert!(decision.satisfies_contract());
    }

    #[test]
    fn classify_unknown_encoding_obeys_quarantine_policy() {
        let contract = FileClassificationContractDefinition::default();
        let bytes = &[0x93, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x94];
        let decision = contract.classify_bytes("/workspace/notes/cp1252.txt", bytes);

        assert_eq!(decision.detected_type, DetectedType::Text);
        assert_eq!(decision.detected_encoding, "unknown-8bit");
        assert_eq!(decision.ingest_action, IngestAction::Quarantine);
        assert_eq!(decision.reason_code, "FSFS_TEXT_HEURISTIC_QUARANTINE");
        assert!(decision.satisfies_contract());
    }

    #[test]
    fn build_corrupt_event_respects_partial_and_corrupt_actions() {
        let contract = FileClassificationContractDefinition::default();

        let truncated = contract.build_corrupt_event(
            "/workspace/tmp/partial.log",
            ErrorClass::Truncated,
            4_096,
        );
        assert_eq!(truncated.kind, FILE_CLASSIFICATION_CORRUPT_EVENT_KIND);
        assert_eq!(truncated.ingest_action, IngestAction::IndexPartialWithFlag);
        assert_eq!(truncated.reason_code, "FSFS_PARTIAL_TRUNCATED_PREFIX_ONLY");

        let decode_error =
            contract.build_corrupt_event("/workspace/tmp/bad.log", ErrorClass::DecodeError, 0);
        assert_eq!(decode_error.ingest_action, IngestAction::Quarantine);
        assert_eq!(decode_error.reason_code, "FSFS_CORRUPT_DECODE_ERROR");
    }
}
