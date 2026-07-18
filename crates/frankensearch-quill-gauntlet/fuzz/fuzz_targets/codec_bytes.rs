#![no_main]

use frankensearch_quill::grimoire::{TermDictionary, TermDictionaryLimits, TermSectionLengths};
use frankensearch_quill::quiver::{
    BlockMaxConcatList, DocLenLimits, DocLenSection, EncodedIdMapSection, EncodedPostingList,
    IdHashLimits, IdHashSection, IdMapEntryInput, IdMapLimits, IdMapSection, NumericLimits,
    NumericSection, PositionList, PositionListLimits, Posting, PostingList, PostingListLimits,
    StatsLimits, StatsSection,
};
use frankensearch_quill::{CASS_SEMANTIC_SCHEMA, DEFAULT_SCHEMA, SegmentReader};
use libfuzzer_sys::fuzz_target;

const MAX_FUZZ_BYTES: usize = 4_096;
const TEXT_FIELD_ORDS: [u16; 3] = [0, 1, 2];

fuzz_target!(|input: &[u8]| {
    if input.len() > MAX_FUZZ_BYTES {
        return;
    }
    let bytes = input;
    let byte_limit = u64::try_from(MAX_FUZZ_BYTES).expect("fuzz byte limit fits u64");
    let span = u64::from(bytes.first().copied().unwrap_or(0) % 65);
    let expected_doc_freq = u32::from(bytes.get(1).copied().unwrap_or(0) % 129);

    if let Ok(reader) = SegmentReader::from_owned(bytes.to_vec(), DEFAULT_SCHEMA) {
        let _ = reader.verify();
    }
    let _ = TermDictionary::parse_with_limits(
        bytes,
        DEFAULT_SCHEMA,
        TermSectionLengths {
            postings: byte_limit,
            positions: Some(byte_limit),
            blockmax: byte_limit,
        },
        TermDictionaryLimits {
            max_bytes: MAX_FUZZ_BYTES,
            max_blocks: 64,
            max_terms: 256,
            max_restarts: 256,
        },
    );
    let _ = PostingList::parse_with_limits(
        bytes,
        expected_doc_freq,
        PostingListLimits {
            max_blocks: 64,
            max_postings: 512,
        },
    );

    let fixed_postings = [Posting::new(1, 1), Posting::new(130, 3)];
    let encoded_postings =
        EncodedPostingList::encode(&fixed_postings).expect("fixed postings must encode");
    let postings = encoded_postings
        .posting_list()
        .expect("fixed postings must reopen");
    let _ = PositionList::parse_with_limits(
        bytes,
        &postings,
        PositionListLimits {
            max_bytes: MAX_FUZZ_BYTES,
            max_blocks: 64,
            max_positions: 512,
        },
    );
    let _ = BlockMaxConcatList::parse(bytes, &postings);

    let _ = DocLenSection::parse_with_limits(
        bytes,
        0,
        span,
        &TEXT_FIELD_ORDS,
        DocLenLimits {
            max_fields: TEXT_FIELD_ORDS.len(),
            max_docid_span: 64,
            max_section_bytes: byte_limit,
        },
    );
    let id_map_limits = IdMapLimits {
        max_docid_span: 64,
        max_section_bytes: byte_limit,
        max_blob_bytes: byte_limit,
        max_document_id_bytes: byte_limit,
    };
    let _ = IdMapSection::parse_with_limits(bytes, 0, span, id_map_limits);

    let fixed_id_rows = [
        Some(IdMapEntryInput::new("alpha", 1)),
        None,
        Some(IdMapEntryInput::new("omega", 2)),
    ];
    let encoded_fixed_id_map =
        EncodedIdMapSection::encode(0, 3, &fixed_id_rows).expect("fixed IDMAP must encode");
    let fixed_id_map = encoded_fixed_id_map
        .section()
        .expect("fixed IDMAP must reopen");
    let _ = IdHashSection::parse_with_limits(
        bytes,
        fixed_id_map,
        IdHashLimits {
            max_entries: 64,
            max_capacity: 128,
            max_section_bytes: byte_limit,
            max_probe_steps: 512,
        },
    );
    let _ = NumericSection::parse_with_limits(
        bytes,
        CASS_SEMANTIC_SCHEMA,
        0,
        span,
        NumericLimits {
            max_fields: 16,
            max_entries_per_field: 256,
            max_total_entries: 512,
            max_docid_span: 64,
            max_section_bytes: byte_limit,
        },
    );
    let _ = StatsSection::parse_with_limits(
        bytes,
        &TEXT_FIELD_ORDS,
        u32::from(bytes.get(2).copied().unwrap_or(0)),
        StatsLimits {
            max_fields: TEXT_FIELD_ORDS.len(),
            max_section_bytes: byte_limit,
        },
    );
});
