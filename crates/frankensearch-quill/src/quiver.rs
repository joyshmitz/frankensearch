//! Quiver postings and positions.
//!
//! FSLX posting lists are streams of self-delimiting FOR, bitmap, and partial
//! VINT blocks. Full blocks contain 128 postings; a fresh seal emits at most one
//! final partial block, while Q1 concat-merge may preserve partial blocks at
//! interior seams. Every full block stores an absolute first docid, so ordinary
//! merge copies POSTINGS bytes without rebasing or seam rewrites.

mod bitpack {
    use thiserror::Error;
    #[cfg(any(test, feature = "bench-internals"))]
    use wide::u32x8;

    // The validated reader deliberately uses the scalar reservoir below. A
    // same-binary A/B rejected a single wide-for-all production policy on
    // 2026-07-17; the portable kernel remains here for differential fixtures
    // and for a future independently measured per-width dispatcher.

    /// Typed failures for the canonical FSLX bitpacked-value stream.
    #[derive(Clone, Debug, Error, Eq, PartialEq)]
    pub enum BitpackError {
        /// FSLX values use at most all 32 bits of a `u32`.
        #[error("bit width {width} is outside the supported range 0..=32")]
        InvalidWidth {
            /// Rejected width.
            width: u8,
        },
        /// Computing the canonical byte length overflowed `usize`.
        #[error("packed length overflow for {count} values at width {width}")]
        LengthOverflow {
            /// Number of encoded values.
            count: usize,
            /// Bits per value.
            width: u8,
        },
        /// A payload was not exactly the canonical length for its shape.
        #[error("packed payload length mismatch: expected {expected} bytes, got {actual}")]
        LengthMismatch {
            /// Canonical payload length.
            expected: usize,
            /// Supplied payload length.
            actual: usize,
        },
        /// An encoder input cannot be represented at the requested width.
        #[error("value {value} at index {index} does not fit in {width} bits")]
        ValueOutOfRange {
            /// Value ordinal.
            index: usize,
            /// Rejected value.
            value: u32,
            /// Requested width.
            width: u8,
        },
        /// Canonical LSB-first packing requires unused high bits to be zero.
        #[error(
            "packed payload has non-zero padding in final byte {byte:#04x}; only {used_bits} bits are used"
        )]
        NonCanonicalPadding {
            /// Final payload byte.
            byte: u8,
            /// Low bits belonging to encoded values.
            used_bits: usize,
        },
    }

    fn validate_width(width: u8) -> Result<(), BitpackError> {
        if width <= 32 {
            Ok(())
        } else {
            Err(BitpackError::InvalidWidth { width })
        }
    }

    pub(super) fn packed_byte_len(count: usize, width: u8) -> Result<usize, BitpackError> {
        validate_width(width)?;
        count
            .checked_mul(usize::from(width))
            .map(|bits| bits.div_ceil(u8::BITS as usize))
            .ok_or(BitpackError::LengthOverflow { count, width })
    }

    fn value_mask(width: u8) -> u32 {
        debug_assert!((1..=32).contains(&width));
        u32::MAX >> (u32::BITS - u32::from(width))
    }

    fn validate_payload(input: &[u8], width: u8, count: usize) -> Result<(), BitpackError> {
        let expected = packed_byte_len(count, width)?;
        if input.len() != expected {
            return Err(BitpackError::LengthMismatch {
                expected,
                actual: input.len(),
            });
        }

        let used_bits = count
            .checked_mul(usize::from(width))
            .ok_or(BitpackError::LengthOverflow { count, width })?
            % (u8::BITS as usize);
        if used_bits != 0 {
            let final_byte = input.last().copied().ok_or(BitpackError::LengthMismatch {
                expected,
                actual: input.len(),
            })?;
            let allowed = u8::MAX >> ((u8::BITS as usize) - used_bits);
            if final_byte & !allowed != 0 {
                return Err(BitpackError::NonCanonicalPadding {
                    byte: final_byte,
                    used_bits,
                });
            }
        }
        Ok(())
    }

    /// Canonically pack `u32` values LSB-first into a byte stream.
    #[allow(clippy::cast_possible_truncation)]
    pub fn pack_lsb(values: &[u32], width: u8) -> Result<Vec<u8>, BitpackError> {
        validate_width(width)?;
        let output_len = packed_byte_len(values.len(), width)?;
        if width == 0 {
            if let Some((index, &value)) = values.iter().enumerate().find(|(_, value)| **value != 0)
            {
                return Err(BitpackError::ValueOutOfRange {
                    index,
                    value,
                    width,
                });
            }
            return Ok(Vec::new());
        }

        let mask = value_mask(width);
        let width = usize::from(width);
        let mut output = Vec::with_capacity(output_len);
        let mut reservoir = 0_u64;
        let mut available = 0_usize;
        for (index, &value) in values.iter().enumerate() {
            if value > mask {
                return Err(BitpackError::ValueOutOfRange {
                    index,
                    value,
                    width: width as u8,
                });
            }
            reservoir |= u64::from(value) << available;
            available += width;
            while available >= u8::BITS as usize {
                output.push((reservoir & u64::from(u8::MAX)) as u8);
                reservoir >>= u8::BITS;
                available -= u8::BITS as usize;
            }
        }
        if available != 0 {
            output.push((reservoir & u64::from(u8::MAX)) as u8);
        }
        debug_assert_eq!(output.len(), output_len);
        Ok(output)
    }

    /// Decode the canonical stream with the scalar reference reservoir.
    #[allow(clippy::cast_possible_truncation)]
    pub fn unpack_scalar_into(
        input: &[u8],
        width: u8,
        output: &mut [u32],
    ) -> Result<(), BitpackError> {
        validate_payload(input, width, output.len())?;
        if width == 0 {
            output.fill(0);
            return Ok(());
        }

        let width = usize::from(width);
        let mask = u64::from(value_mask(width as u8));
        let mut bytes = input.iter().copied();
        let mut reservoir = 0_u64;
        let mut available = 0_usize;
        for value in output {
            while available < width {
                // `validate_payload` proved the exact source length.
                let byte = bytes.next().ok_or(BitpackError::LengthMismatch {
                    expected: input.len(),
                    actual: input.len(),
                })?;
                reservoir |= u64::from(byte) << available;
                available += u8::BITS as usize;
            }
            *value = (reservoir & mask) as u32;
            reservoir >>= width;
            available -= width;
        }
        Ok(())
    }

    /// Unpack one byte-aligned group of eight fixed-width values.
    ///
    /// Eight `WIDTH`-bit values occupy exactly `WIDTH` bytes. We gather the
    /// possibly crossing little-endian words per lane, then perform the actual
    /// variable shifts, merge, and mask in one `wide::u32x8` kernel.
    #[cfg(any(test, feature = "bench-internals"))]
    #[allow(clippy::cast_possible_truncation)]
    fn unpack_eight<const WIDTH: usize>(input: &[u8]) -> [u32; 8] {
        debug_assert!((1..=32).contains(&WIDTH));
        debug_assert_eq!(input.len(), WIDTH);

        let mut words = [0_u32; 9];
        for (word, chunk) in words.iter_mut().zip(input.chunks(4)) {
            let mut little_endian = [0_u8; 4];
            little_endian[..chunk.len()].copy_from_slice(chunk);
            *word = u32::from_le_bytes(little_endian);
        }

        let mut low_words = [0_u32; 8];
        let mut high_words = [0_u32; 8];
        let mut right_shifts = [0_u32; 8];
        let mut left_shifts = [0_u32; 8];
        for lane in 0..8 {
            let start = lane * WIDTH;
            let word_index = start / (u32::BITS as usize);
            let right_shift = start % (u32::BITS as usize);
            low_words[lane] = words[word_index];
            right_shifts[lane] = right_shift as u32;
            if right_shift + WIDTH > u32::BITS as usize {
                high_words[lane] = words[word_index + 1];
                left_shifts[lane] = (u32::BITS as usize - right_shift) as u32;
            }
        }

        let mask = u32::MAX >> (u32::BITS as usize - WIDTH);
        (((u32x8::from(low_words) >> u32x8::from(right_shifts))
            | (u32x8::from(high_words) << u32x8::from(left_shifts)))
            & u32x8::splat(mask))
        .to_array()
    }

    #[cfg(any(test, feature = "bench-internals"))]
    macro_rules! dispatch_eight {
        ($width:expr, $input:expr) => {
            match $width {
                1 => Ok(unpack_eight::<1>($input)),
                2 => Ok(unpack_eight::<2>($input)),
                3 => Ok(unpack_eight::<3>($input)),
                4 => Ok(unpack_eight::<4>($input)),
                5 => Ok(unpack_eight::<5>($input)),
                6 => Ok(unpack_eight::<6>($input)),
                7 => Ok(unpack_eight::<7>($input)),
                8 => Ok(unpack_eight::<8>($input)),
                9 => Ok(unpack_eight::<9>($input)),
                10 => Ok(unpack_eight::<10>($input)),
                11 => Ok(unpack_eight::<11>($input)),
                12 => Ok(unpack_eight::<12>($input)),
                13 => Ok(unpack_eight::<13>($input)),
                14 => Ok(unpack_eight::<14>($input)),
                15 => Ok(unpack_eight::<15>($input)),
                16 => Ok(unpack_eight::<16>($input)),
                17 => Ok(unpack_eight::<17>($input)),
                18 => Ok(unpack_eight::<18>($input)),
                19 => Ok(unpack_eight::<19>($input)),
                20 => Ok(unpack_eight::<20>($input)),
                21 => Ok(unpack_eight::<21>($input)),
                22 => Ok(unpack_eight::<22>($input)),
                23 => Ok(unpack_eight::<23>($input)),
                24 => Ok(unpack_eight::<24>($input)),
                25 => Ok(unpack_eight::<25>($input)),
                26 => Ok(unpack_eight::<26>($input)),
                27 => Ok(unpack_eight::<27>($input)),
                28 => Ok(unpack_eight::<28>($input)),
                29 => Ok(unpack_eight::<29>($input)),
                30 => Ok(unpack_eight::<30>($input)),
                31 => Ok(unpack_eight::<31>($input)),
                32 => Ok(unpack_eight::<32>($input)),
                width => Err(BitpackError::InvalidWidth { width }),
            }
        };
    }

    /// Decode the canonical stream eight lanes at a time with `wide::u32x8`.
    #[cfg(any(test, feature = "bench-internals"))]
    pub fn unpack_wide_into(
        input: &[u8],
        width: u8,
        output: &mut [u32],
    ) -> Result<(), BitpackError> {
        validate_payload(input, width, output.len())?;
        if width == 0 {
            output.fill(0);
            return Ok(());
        }

        let width_bytes = usize::from(width);
        let output_len = output.len();
        let groups = output_len / 8;
        for group in 0..groups {
            let source_start = group * width_bytes;
            let source = input.get(source_start..source_start + width_bytes).ok_or(
                BitpackError::LengthMismatch {
                    expected: input.len(),
                    actual: input.len(),
                },
            )?;
            let decoded = dispatch_eight!(width, source)?;
            let destination_start = group * 8;
            let destination = output
                .get_mut(destination_start..destination_start + 8)
                .ok_or(BitpackError::LengthMismatch {
                    expected: output_len,
                    actual: output_len,
                })?;
            destination.copy_from_slice(&decoded);
        }

        let decoded_values = groups * 8;
        if decoded_values != output_len {
            let source_start = groups * width_bytes;
            let source = input
                .get(source_start..)
                .ok_or(BitpackError::LengthMismatch {
                    expected: input.len(),
                    actual: input.len(),
                })?;
            let destination =
                output
                    .get_mut(decoded_values..)
                    .ok_or(BitpackError::LengthMismatch {
                        expected: output_len,
                        actual: output_len,
                    })?;
            unpack_scalar_into(source, width, destination)?;
        }
        Ok(())
    }
}

use std::ops::Range;

use thiserror::Error;

use crate::scribe::{ColumnarAccumulator, TokenAnalyzer};

pub use bitpack::BitpackError;

/// Number of postings in every full FOR or bitmap block.
pub const POSTINGS_PER_BLOCK: usize = 128;
/// Default cap on metadata entries retained for one term's posting stream.
///
/// The default covers more block fragments than an ordinary segment should
/// need while bounding hostile one-posting fragmentation. Callers opening
/// deliberately larger trusted streams can opt into higher caps with
/// [`PostingList::parse_with_limits`].
pub const DEFAULT_MAX_POSTING_BLOCKS: usize = 1 << 18;
/// Default cap on postings eagerly validated for one term.
pub const DEFAULT_MAX_POSTINGS_PER_TERM: usize = 1 << 22;

/// Explicit resource budgets for validating one term's posting stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PostingListLimits {
    /// Maximum validated self-delimiting blocks retained for cursor skips.
    pub max_blocks: usize,
    /// Maximum postings eagerly validated across those blocks.
    pub max_postings: usize,
}

impl Default for PostingListLimits {
    fn default() -> Self {
        Self {
            max_blocks: DEFAULT_MAX_POSTING_BLOCKS,
            max_postings: DEFAULT_MAX_POSTINGS_PER_TERM,
        }
    }
}

const POSTINGS_PER_BLOCK_U8: u8 = 128;
const BLOCK_HEADER_LEN: usize = 4;
const FOR_KIND: u8 = 0;
const BITMAP_KIND: u8 = 1;
const VINT_KIND: u8 = 2;
const FREQ_ALL_ONE: u8 = 0;
const FREQ_BITPACKED: u8 = 1;

/// One document occurrence in a term's posting list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Posting {
    /// Absolute global document id.
    pub doc_id: u32,
    /// Term frequency in this document. Always at least one.
    pub freq: u32,
}

impl Posting {
    /// Construct a posting. Validation occurs when a list is encoded.
    #[must_use]
    pub const fn new(doc_id: u32, freq: u32) -> Self {
        Self { doc_id, freq }
    }
}

/// Durable FSLX posting-block codec.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PostingBlockKind {
    /// Frame-of-reference block: absolute first docid plus bitpacked deltas.
    FrameOfReference,
    /// Dense 512-bit docid bitmap plus aligned frequencies.
    Bitmap,
    /// Partial block encoded as canonical u32 LEB128 pairs.
    Vint,
}

impl PostingBlockKind {
    const fn code(self) -> u8 {
        match self {
            Self::FrameOfReference => FOR_KIND,
            Self::Bitmap => BITMAP_KIND,
            Self::Vint => VINT_KIND,
        }
    }

    fn from_code(code: u8, offset: usize) -> Result<Self, PostingCodecError> {
        match code {
            FOR_KIND => Ok(Self::FrameOfReference),
            BITMAP_KIND => Ok(Self::Bitmap),
            VINT_KIND => Ok(Self::Vint),
            kind => Err(PostingCodecError::InvalidBlockKind { offset, kind }),
        }
    }
}

/// Typed failures from encoding or validating an FSLX posting stream.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PostingCodecError {
    /// A slice cannot be represented by the u32 TERMDICT `doc_freq` field.
    #[error("posting count {count} exceeds the u32 doc_freq range")]
    TooManyPostings {
        /// Rejected posting count.
        count: usize,
    },
    /// Frequencies are strictly positive.
    #[error("posting {index} for docid {doc_id} has zero frequency")]
    ZeroFrequency {
        /// Posting ordinal.
        index: usize,
        /// Posting docid.
        doc_id: u32,
    },
    /// Encoder input must be strictly ordered by absolute docid.
    #[error(
        "posting docids are not strictly ascending at index {index}: previous {previous}, current {doc_id}"
    )]
    NonAscendingInput {
        /// Posting ordinal.
        index: usize,
        /// Previous docid.
        previous: u32,
        /// Current docid.
        doc_id: u32,
    },
    /// A block payload cannot fit in its u16 length field.
    #[error("posting block at offset {block_offset} has {length} payload bytes, exceeding u16")]
    PayloadTooLarge {
        /// Block start in the term stream.
        block_offset: usize,
        /// Computed payload length.
        length: usize,
    },
    /// Durable bytes ended before a declared field or payload did.
    #[error(
        "truncated posting stream at offset {offset}: need {needed} bytes, only {remaining} remain"
    )]
    Truncated {
        /// Absolute byte offset in the term stream.
        offset: usize,
        /// Required byte count.
        needed: usize,
        /// Available byte count.
        remaining: usize,
    },
    /// The common header named an unknown codec.
    #[error("unknown posting block kind {kind} at offset {offset}")]
    InvalidBlockKind {
        /// Common-header offset.
        offset: usize,
        /// Rejected kind byte.
        kind: u8,
    },
    /// A codec/count pairing violates the FSLX grammar.
    #[error("posting block kind {kind} at offset {offset} has invalid count {count}")]
    InvalidBlockCount {
        /// Common-header offset.
        offset: usize,
        /// Raw kind byte.
        kind: u8,
        /// Raw posting count.
        count: u8,
    },
    /// Frequency metadata named an unknown representation.
    #[error("posting block at offset {block_offset} has invalid frequency kind {kind}")]
    InvalidFrequencyKind {
        /// Common-header offset.
        block_offset: usize,
        /// Rejected frequency kind.
        kind: u8,
    },
    /// Bitpacked frequencies were used for an all-one run.
    #[error("posting block at offset {block_offset} bitpacks an all-one frequency run")]
    NonCanonicalFrequencyEncoding {
        /// Common-header offset.
        block_offset: usize,
    },
    /// A bitmap's inclusive docid span is outside 128..=511.
    #[error("bitmap block at offset {block_offset} has invalid inclusive span {span}")]
    InvalidBitmapSpan {
        /// Common-header offset.
        block_offset: usize,
        /// Rejected span.
        span: u16,
    },
    /// A bitmap violates its boundary, cardinality, or padding invariant.
    #[error("bitmap block at offset {block_offset} violates {invariant}")]
    InvalidBitmap {
        /// Common-header offset.
        block_offset: usize,
        /// Stable invariant name.
        invariant: &'static str,
    },
    /// A bitpacked field did not use its unique minimal width.
    #[error(
        "posting block at offset {block_offset} uses non-canonical {field} width {encoded}; required {required}"
    )]
    NonCanonicalWidth {
        /// Common-header offset.
        block_offset: usize,
        /// Width-bearing field.
        field: &'static str,
        /// Encoded width.
        encoded: u8,
        /// Minimal required width.
        required: u8,
    },
    /// An unsigned LEB128 value used more bytes than its shortest form.
    #[error("non-canonical u32 vint at offset {offset}")]
    NonCanonicalVint {
        /// Vint start offset.
        offset: usize,
    },
    /// An unsigned LEB128 value exceeds u32 or five bytes.
    #[error("u32 vint overflow at offset {offset}")]
    VintOverflow {
        /// Vint start offset.
        offset: usize,
    },
    /// Reconstructing a durable value overflowed its declared integer domain.
    #[error("arithmetic overflow for {field} in posting block at offset {block_offset}")]
    ArithmeticOverflow {
        /// Common-header offset.
        block_offset: usize,
        /// Value being reconstructed.
        field: &'static str,
    },
    /// Decoded docids were not strictly ordered, including across blocks.
    #[error(
        "decoded posting order violation at ordinal {posting_ordinal}: previous {previous}, current {doc_id}"
    )]
    NonAscendingDecoded {
        /// Posting ordinal in the term stream.
        posting_ordinal: u32,
        /// Previous decoded docid.
        previous: u32,
        /// Current decoded docid.
        doc_id: u32,
    },
    /// A declared payload contained bytes beyond its canonical fields.
    #[error("posting block at offset {block_offset} has {remaining} trailing payload bytes")]
    TrailingPayload {
        /// Common-header offset.
        block_offset: usize,
        /// Unconsumed bytes.
        remaining: usize,
    },
    /// TERMDICT `doc_freq` disagrees with the sum of self-delimiting blocks.
    #[error("posting doc_freq mismatch: expected {expected}, decoded {actual}")]
    DocFrequencyMismatch {
        /// TERMDICT value.
        expected: u32,
        /// Sum of block counts.
        actual: u32,
    },
    /// Decoded blocks already exceed TERMDICT `doc_freq`; validation stops
    /// before processing any later bytes.
    #[error("posting doc_freq exceeded: expected {expected}, decoded at least {actual}")]
    DocFrequencyExceeded {
        /// TERMDICT value.
        expected: u32,
        /// Sum including the first block that exceeds the declaration.
        actual: u32,
    },
    /// A caller-selected block metadata budget was exhausted before any later
    /// bytes were inspected.
    #[error("posting block budget {limit} exhausted after {validated} validated blocks")]
    BlockBudgetExhausted {
        /// Configured metadata-entry cap.
        limit: usize,
        /// Blocks validated before validation stopped.
        validated: usize,
    },
    /// TERMDICT `doc_freq` exceeded a caller-selected eager validation budget.
    #[error("posting validation limit {limit} exceeded by declared doc_freq {actual}")]
    PostingLimitExceeded {
        /// Configured posting-work cap.
        limit: usize,
        /// Declared posting count.
        actual: usize,
    },
    /// Reserving validated block metadata failed without panicking.
    #[error("unable to reserve metadata for posting block {block_index}")]
    MetadataAllocation {
        /// Zero-based block index that could not be reserved.
        block_index: usize,
    },
    /// A bounded collection request exceeded its explicit posting budget.
    #[error("posting materialization limit {limit} exceeded by {actual} postings")]
    MaterializationLimitExceeded {
        /// Caller-selected posting cap.
        limit: usize,
        /// Validated posting count.
        actual: usize,
    },
    /// Reserving the bounded materialized posting vector failed.
    #[error("unable to reserve space for {count} decoded postings")]
    MaterializationAllocation {
        /// Validated posting count requested.
        count: usize,
    },
    /// A canonical bitpacked substream failed validation.
    #[error("invalid bitpacked payload in posting block at offset {block_offset}: {source}")]
    Bitpack {
        /// Common-header offset.
        block_offset: usize,
        /// Bitpacking failure.
        #[source]
        source: BitpackError,
    },
}

/// Validated location and bounds of one self-delimiting posting block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PostingBlockMeta {
    /// Durable codec.
    pub kind: PostingBlockKind,
    /// Number of postings in this block.
    pub posting_count: u8,
    /// Common-header offset relative to the term's `postings_offset`.
    pub byte_offset: usize,
    /// Common header plus payload length.
    pub byte_len: usize,
    /// First absolute docid in this block.
    pub first_doc: u32,
    /// Last absolute docid in this block.
    pub last_doc: u32,
    /// Posting ordinal of this block's first entry.
    pub base_posting_ordinal: u32,
    /// Canonical conservative maximum-frequency code derived during validation.
    pub max_frequency_code: u8,
}

impl PostingBlockMeta {
    /// Exact raw-byte range copied by Q1 concat-merge.
    #[must_use]
    pub fn byte_range(&self) -> Option<Range<usize>> {
        self.byte_offset
            .checked_add(self.byte_len)
            .map(|end| self.byte_offset..end)
    }
}

/// Owned bytes produced by the deterministic fresh-seal writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedPostingList {
    bytes: Vec<u8>,
    doc_freq: u32,
    block_count: usize,
}

impl EncodedPostingList {
    const fn validation_limits(posting_count: usize, block_count: usize) -> PostingListLimits {
        PostingListLimits {
            max_blocks: block_count,
            max_postings: posting_count,
        }
    }

    /// Encode a strictly docid-ascending posting slice.
    ///
    /// Full blocks use BITMAP iff their inclusive span is below 512 and FOR
    /// otherwise. The final 1..=127 postings use a partial VINT block.
    ///
    /// # Errors
    ///
    /// Returns a typed error for duplicate/descending docids, zero frequency,
    /// an unrepresentable count, or any internal payload-length overflow.
    pub fn encode(postings: &[Posting]) -> Result<Self, PostingCodecError> {
        let doc_freq =
            u32::try_from(postings.len()).map_err(|_| PostingCodecError::TooManyPostings {
                count: postings.len(),
            })?;
        validate_encoder_input(postings)?;

        let mut bytes = Vec::new();
        for block in postings.chunks(POSTINGS_PER_BLOCK) {
            encode_posting_block(block, &mut bytes)?;
        }

        let expected_block_count = postings.len().div_ceil(POSTINGS_PER_BLOCK);
        Self::finish_encoding(bytes, doc_freq, postings.len(), expected_block_count)
    }

    /// Encode POSTINGS and its aligned BLOCKMAX stream in one fresh-seal pass.
    ///
    /// Bound components are derived directly from the source postings before
    /// the newly written POSTINGS bytes undergo their normal self-validation;
    /// fresh seal therefore does not decode the output a second time merely to
    /// compute BLOCKMAX. Compaction may instead call [`EncodedBlockMax::encode`]
    /// against an already validated posting view.
    ///
    /// # Errors
    ///
    /// Returns a typed posting-codec error, a missing DOCLEN value, checked
    /// offset overflow, or a fallible BLOCKMAX allocation failure.
    pub fn encode_with_block_max<F>(
        postings: &[Posting],
        mut fieldnorm_for_doc: F,
    ) -> Result<(Self, EncodedBlockMax), BlockMaxError>
    where
        F: FnMut(u32) -> Option<u8>,
    {
        let doc_freq =
            u32::try_from(postings.len()).map_err(|_| PostingCodecError::TooManyPostings {
                count: postings.len(),
            })?;
        validate_encoder_input(postings)?;

        let entry_count = postings.len().div_ceil(POSTINGS_PER_BLOCK);
        let mut posting_bytes = Vec::new();
        let mut block_max_bytes = allocate_block_max_bytes(entry_count, "fresh-seal bytes")?;
        for (block_index, block) in postings.chunks(POSTINGS_PER_BLOCK).enumerate() {
            let block_offset = u64::try_from(posting_bytes.len()).map_err(|_| {
                BlockMaxError::ArithmeticOverflow {
                    field: "fresh-seal posting block offset",
                }
            })?;
            let entry = derive_fresh_block_max_entry(
                block,
                block_index,
                block_offset,
                &mut fieldnorm_for_doc,
            )?;
            encode_posting_block(block, &mut posting_bytes)?;
            append_block_max_entry(entry, &mut block_max_bytes);
        }

        let postings = Self::finish_encoding(posting_bytes, doc_freq, postings.len(), entry_count)?;
        let block_max = EncodedBlockMax {
            bytes: block_max_bytes,
            entry_count,
        };
        Ok((postings, block_max))
    }

    fn finish_encoding(
        bytes: Vec<u8>,
        doc_freq: u32,
        posting_count: usize,
        expected_block_count: usize,
    ) -> Result<Self, PostingCodecError> {
        let parsed = PostingList::parse_with_limits(
            &bytes,
            doc_freq,
            Self::validation_limits(posting_count, expected_block_count),
        )?;
        let block_count = parsed.block_count();
        Ok(Self {
            bytes,
            doc_freq,
            block_count,
        })
    }

    /// Borrow the exact durable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return the exact durable bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of encoded postings.
    #[must_use]
    pub const fn doc_freq(&self) -> u32 {
        self.doc_freq
    }

    /// Number of self-delimiting blocks.
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    /// Re-open the owned bytes through the validating reader.
    ///
    /// # Errors
    ///
    /// Returns a typed error if an internal invariant was violated.
    pub fn posting_list(&self) -> Result<PostingList<'_>, PostingCodecError> {
        let posting_count =
            usize::try_from(self.doc_freq).map_err(|_| PostingCodecError::ArithmeticOverflow {
                block_offset: 0,
                field: "owned doc_freq",
            })?;
        PostingList::parse_with_limits(
            &self.bytes,
            self.doc_freq,
            Self::validation_limits(posting_count, self.block_count),
        )
    }
}

/// Borrowed, fully validated view of one term's POSTINGS byte range.
#[derive(Clone, Debug)]
pub struct PostingList<'a> {
    bytes: &'a [u8],
    doc_freq: u32,
    blocks: Vec<PostingBlockMeta>,
}

impl<'a> PostingList<'a> {
    /// Validate a complete TERMDICT-referenced posting range.
    ///
    /// Validation is eager for block grammar, canonical widths/vints, decoded
    /// ordering, arithmetic, and the expected `doc_freq`. Cursors may therefore
    /// skip by validated block bounds without trusting unbounded durable counts.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed or non-canonical durable bytes.
    pub fn parse(bytes: &'a [u8], expected_doc_freq: u32) -> Result<Self, PostingCodecError> {
        Self::parse_with_limits(bytes, expected_doc_freq, PostingListLimits::default())
    }

    /// Validate a complete posting range with explicit block and posting caps.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed bytes, a `doc_freq` mismatch, block
    /// budget exhaustion, or fallible metadata reservation failure.
    pub fn parse_with_limits(
        bytes: &'a [u8],
        expected_doc_freq: u32,
        limits: PostingListLimits,
    ) -> Result<Self, PostingCodecError> {
        let expected_postings = usize::try_from(expected_doc_freq).map_err(|_| {
            PostingCodecError::PostingLimitExceeded {
                limit: limits.max_postings,
                actual: usize::MAX,
            }
        })?;
        if expected_postings > limits.max_postings {
            return Err(PostingCodecError::PostingLimitExceeded {
                limit: limits.max_postings,
                actual: expected_postings,
            });
        }
        let mut offset = 0_usize;
        let mut actual_doc_freq = 0_u32;
        let mut previous_last = None;
        let mut blocks = Vec::new();

        while offset < bytes.len() {
            if blocks.len() >= limits.max_blocks {
                return Err(PostingCodecError::BlockBudgetExhausted {
                    limit: limits.max_blocks,
                    validated: blocks.len(),
                });
            }
            let base_posting_ordinal = usize::try_from(actual_doc_freq).map_err(|_| {
                PostingCodecError::ArithmeticOverflow {
                    block_offset: offset,
                    field: "posting ordinal",
                }
            })?;
            let decoded = decode_block_at(bytes, offset, base_posting_ordinal)?;
            if let Some(previous) = previous_last {
                if decoded.first_doc <= previous {
                    return Err(PostingCodecError::NonAscendingDecoded {
                        posting_ordinal: actual_doc_freq,
                        previous,
                        doc_id: decoded.first_doc,
                    });
                }
            }
            let count = u32::from(decoded.posting_count);
            let next_doc_freq = actual_doc_freq.checked_add(count).ok_or(
                PostingCodecError::ArithmeticOverflow {
                    block_offset: offset,
                    field: "doc_freq sum",
                },
            )?;
            if next_doc_freq > expected_doc_freq {
                return Err(PostingCodecError::DocFrequencyExceeded {
                    expected: expected_doc_freq,
                    actual: next_doc_freq,
                });
            }
            blocks
                .try_reserve(1)
                .map_err(|_| PostingCodecError::MetadataAllocation {
                    block_index: blocks.len(),
                })?;
            blocks.push(PostingBlockMeta {
                kind: decoded.kind,
                posting_count: decoded.posting_count,
                byte_offset: offset,
                byte_len: decoded.byte_len,
                first_doc: decoded.first_doc,
                last_doc: decoded.last_doc,
                base_posting_ordinal: actual_doc_freq,
                max_frequency_code: canonical_block_max_frequency_code(&decoded),
            });
            previous_last = Some(decoded.last_doc);
            actual_doc_freq = next_doc_freq;
            offset = offset.checked_add(decoded.byte_len).ok_or(
                PostingCodecError::ArithmeticOverflow {
                    block_offset: offset,
                    field: "block byte offset",
                },
            )?;
        }

        if actual_doc_freq != expected_doc_freq {
            return Err(PostingCodecError::DocFrequencyMismatch {
                expected: expected_doc_freq,
                actual: actual_doc_freq,
            });
        }
        Ok(Self {
            bytes,
            doc_freq: actual_doc_freq,
            blocks,
        })
    }

    /// Exact borrowed durable bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Validated document frequency.
    #[must_use]
    pub const fn doc_freq(&self) -> u32 {
        self.doc_freq
    }

    /// Number of validated blocks.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Validated block index in durable order.
    #[must_use]
    pub fn blocks(&self) -> &[PostingBlockMeta] {
        &self.blocks
    }

    /// Exact raw bytes for one validated block.
    #[must_use]
    pub fn block_bytes(&self, block_index: usize) -> Option<&'a [u8]> {
        let block = self.blocks.get(block_index)?;
        self.bytes.get(block.byte_range()?)
    }

    /// Create a cursor positioned on the first posting, or exhausted if empty.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the already-validated immutable bytes fail to
    /// decode, which indicates an internal reader invariant regression.
    pub fn cursor(&self) -> Result<PostingCursor<'_>, PostingCodecError> {
        PostingCursor::new(self.bytes, &self.blocks)
    }

    /// Test-only convenience; production callers must select an explicit cap.
    ///
    /// # Errors
    ///
    /// Returns a typed error if allocation or the validated cursor invariant
    /// fails while materializing the list.
    #[cfg(test)]
    pub fn decode_all(&self) -> Result<Vec<Posting>, PostingCodecError> {
        self.decode_all_bounded(self.blocks.len().saturating_mul(POSTINGS_PER_BLOCK))
    }

    /// Decode all postings under a caller-selected materialization cap.
    ///
    /// # Errors
    ///
    /// Returns a typed error if `max_postings` is exceeded, allocation fails,
    /// or cursor decoding fails.
    pub fn decode_all_bounded(
        &self,
        max_postings: usize,
    ) -> Result<Vec<Posting>, PostingCodecError> {
        let posting_count = usize::try_from(self.doc_freq).map_err(|_| {
            PostingCodecError::MaterializationLimitExceeded {
                limit: max_postings,
                actual: usize::MAX,
            }
        })?;
        if posting_count > max_postings {
            return Err(PostingCodecError::MaterializationLimitExceeded {
                limit: max_postings,
                actual: posting_count,
            });
        }
        let mut output = Vec::new();
        output.try_reserve_exact(posting_count).map_err(|_| {
            PostingCodecError::MaterializationAllocation {
                count: posting_count,
            }
        })?;
        let mut cursor = self.cursor()?;
        while let Some(posting) = cursor.current() {
            output.push(posting);
            cursor.next()?;
        }
        Ok(output)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CursorState {
    Positioned { block: usize, within: usize },
    Exhausted,
}

/// Lazy posting cursor with block-level `advance` skipping.
pub struct PostingCursor<'a> {
    bytes: &'a [u8],
    blocks: &'a [PostingBlockMeta],
    state: CursorState,
    decoded_docs: [u32; POSTINGS_PER_BLOCK],
    decoded_freqs: [u32; POSTINGS_PER_BLOCK],
    decoded_count: usize,
}

impl<'a> PostingCursor<'a> {
    fn new(bytes: &'a [u8], blocks: &'a [PostingBlockMeta]) -> Result<Self, PostingCodecError> {
        let mut cursor = Self {
            bytes,
            blocks,
            state: CursorState::Exhausted,
            decoded_docs: [0; POSTINGS_PER_BLOCK],
            decoded_freqs: [0; POSTINGS_PER_BLOCK],
            decoded_count: 0,
        };
        if !blocks.is_empty() {
            cursor.load_block(0)?;
            cursor.state = CursorState::Positioned {
                block: 0,
                within: 0,
            };
        }
        Ok(cursor)
    }

    fn load_block(&mut self, block_index: usize) -> Result<(), PostingCodecError> {
        let block = self
            .blocks
            .get(block_index)
            .ok_or(PostingCodecError::ArithmeticOverflow {
                block_offset: self.bytes.len(),
                field: "cursor block index",
            })?;
        let base_posting_ordinal = usize::try_from(block.base_posting_ordinal).map_err(|_| {
            PostingCodecError::ArithmeticOverflow {
                block_offset: block.byte_offset,
                field: "posting ordinal",
            }
        })?;
        let decoded = decode_block_at(self.bytes, block.byte_offset, base_posting_ordinal)?;
        if decoded.byte_len != block.byte_len
            || decoded.kind != block.kind
            || decoded.posting_count != block.posting_count
        {
            return Err(PostingCodecError::ArithmeticOverflow {
                block_offset: block.byte_offset,
                field: "validated block metadata",
            });
        }
        self.decoded_docs = decoded.docs;
        self.decoded_freqs = decoded.freqs;
        self.decoded_count = usize::from(decoded.posting_count);
        Ok(())
    }

    /// Current posting, including a valid `u32::MAX` docid when present.
    #[must_use]
    pub fn current(&self) -> Option<Posting> {
        let CursorState::Positioned { within, .. } = self.state else {
            return None;
        };
        Some(Posting {
            doc_id: *self.decoded_docs.get(within)?,
            freq: *self.decoded_freqs.get(within)?,
        })
    }

    /// Current absolute docid.
    #[must_use]
    pub fn doc(&self) -> Option<u32> {
        self.current().map(|posting| posting.doc_id)
    }

    /// Current term frequency.
    #[must_use]
    pub fn freq(&self) -> Option<u32> {
        self.current().map(|posting| posting.freq)
    }

    /// Current zero-based posting ordinal for later POSITIONS alignment.
    #[must_use]
    pub fn posting_ordinal(&self) -> Option<u32> {
        let CursorState::Positioned { block, within } = self.state else {
            return None;
        };
        let base = self.blocks.get(block)?.base_posting_ordinal;
        base.checked_add(u32::try_from(within).ok()?)
    }

    /// Move strictly forward by one posting. Exhaustion is fused.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the next validated block cannot be decoded.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<Posting>, PostingCodecError> {
        let CursorState::Positioned { block, within } = self.state else {
            return Ok(None);
        };
        if within + 1 < self.decoded_count {
            self.state = CursorState::Positioned {
                block,
                within: within + 1,
            };
            return Ok(self.current());
        }

        let next_block = block + 1;
        if next_block >= self.blocks.len() {
            self.state = CursorState::Exhausted;
            self.decoded_count = 0;
            return Ok(None);
        }
        self.load_block(next_block)?;
        self.state = CursorState::Positioned {
            block: next_block,
            within: 0,
        };
        Ok(self.current())
    }

    /// Land on the first posting whose docid is at least `target`.
    ///
    /// If the current docid already satisfies the target, the cursor does not
    /// move. Exhaustion is fused and `u32::MAX` remains a normal searchable id.
    ///
    /// # Errors
    ///
    /// Returns a typed error if a selected validated block cannot be decoded.
    pub fn advance(&mut self, target: u32) -> Result<Option<Posting>, PostingCodecError> {
        let CursorState::Positioned {
            block: current_block,
            within: current_within,
        } = self.state
        else {
            return Ok(None);
        };
        if self.decoded_docs[current_within] >= target {
            return Ok(self.current());
        }

        let current_tail = &self.decoded_docs[current_within + 1..self.decoded_count];
        let within_tail = current_tail.partition_point(|doc_id| *doc_id < target);
        if within_tail < current_tail.len() {
            self.state = CursorState::Positioned {
                block: current_block,
                within: current_within + 1 + within_tail,
            };
            return Ok(self.current());
        }

        let later = &self.blocks[current_block + 1..];
        let relative_block = later.partition_point(|block| block.last_doc < target);
        if relative_block == later.len() {
            self.state = CursorState::Exhausted;
            self.decoded_count = 0;
            return Ok(None);
        }
        let block = current_block + 1 + relative_block;
        self.load_block(block)?;
        let within =
            self.decoded_docs[..self.decoded_count].partition_point(|doc_id| *doc_id < target);
        if within == self.decoded_count {
            return Err(PostingCodecError::ArithmeticOverflow {
                block_offset: self.blocks[block].byte_offset,
                field: "validated block last_doc",
            });
        }
        self.state = CursorState::Positioned { block, within };
        Ok(self.current())
    }
}

/// Typed failures from encoding or validating an FSLX BLOCKMAX term stream.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum BlockMaxError {
    /// The referenced POSTINGS block could not be decoded consistently.
    #[error("posting validation failed while processing BLOCKMAX: {0}")]
    Posting(#[from] PostingCodecError),
    /// Durable bytes ended before one complete BLOCKMAX entry was available.
    #[error(
        "truncated BLOCKMAX stream at offset {offset}: need {needed} bytes, only {remaining} remain"
    )]
    Truncated {
        /// Byte offset relative to the term's BLOCKMAX span.
        offset: usize,
        /// Required byte count.
        needed: usize,
        /// Available byte count.
        remaining: usize,
    },
    /// A u64 LEB128 offset used more bytes than its shortest encoding.
    #[error("non-canonical BLOCKMAX offset vint at byte {offset}")]
    NonCanonicalVint {
        /// Vint start relative to the term's BLOCKMAX span.
        offset: usize,
    },
    /// A u64 LEB128 offset exceeded ten bytes or the u64 domain.
    #[error("BLOCKMAX offset vint overflow at byte {offset}")]
    VintOverflow {
        /// Vint start relative to the term's BLOCKMAX span.
        offset: usize,
    },
    /// A real posting block cannot have maximum frequency zero.
    #[error("BLOCKMAX entry {block_index} uses zero as its maximum-frequency code")]
    ZeroMaximumFrequency {
        /// Zero-based posting-block index.
        block_index: usize,
    },
    /// The redundant first-doc skip key disagreed with POSTINGS.
    #[error("BLOCKMAX entry {block_index} first doc mismatch: expected {expected}, got {actual}")]
    FirstDocMismatch {
        /// Zero-based posting-block index.
        block_index: usize,
        /// POSTINGS block's validated first docid.
        expected: u32,
        /// Durable BLOCKMAX value.
        actual: u32,
    },
    /// The redundant posting-byte skip offset disagreed with POSTINGS.
    #[error(
        "BLOCKMAX entry {block_index} posting offset mismatch: expected {expected}, got {actual}"
    )]
    BlockOffsetMismatch {
        /// Zero-based posting-block index.
        block_index: usize,
        /// POSTINGS block's validated term-relative byte offset.
        expected: u64,
        /// Durable BLOCKMAX value.
        actual: u64,
    },
    /// A frequency code was not the unique conservative encoding of the block.
    #[error(
        "BLOCKMAX entry {block_index} frequency code mismatch: expected {expected}, got {actual}"
    )]
    MaximumFrequencyMismatch {
        /// Zero-based posting-block index.
        block_index: usize,
        /// Canonical code derived from POSTINGS.
        expected: u8,
        /// Durable BLOCKMAX value.
        actual: u8,
    },
    /// A posting docid had no corresponding DOCLEN value.
    #[error("BLOCKMAX entry {block_index} has no fieldnorm for docid {doc_id}")]
    MissingFieldnorm {
        /// Zero-based posting-block index.
        block_index: usize,
        /// Present posting docid missing from DOCLEN.
        doc_id: u32,
    },
    /// The stored componentwise minimum fieldnorm disagreed with DOCLEN.
    #[error(
        "BLOCKMAX entry {block_index} minimum fieldnorm mismatch: expected {expected}, got {actual}"
    )]
    MinimumFieldnormMismatch {
        /// Zero-based posting-block index.
        block_index: usize,
        /// Minimum stored fieldnorm ID over the block's present docs.
        expected: u8,
        /// Durable BLOCKMAX value.
        actual: u8,
    },
    /// Bytes remained after the exact one-entry-per-posting-block grammar.
    #[error("BLOCKMAX stream has {remaining} trailing bytes at offset {offset}")]
    TrailingBytes {
        /// First unconsumed byte.
        offset: usize,
        /// Unconsumed byte count.
        remaining: usize,
    },
    /// Q1 concat inputs were not in strictly increasing docid order.
    #[error(
        "BLOCKMAX concat part {part_index} starts at docid {first_doc}, not after {previous_last_doc}"
    )]
    NonAscendingConcat {
        /// Source part containing the invalid first block.
        part_index: usize,
        /// Last docid in the prior non-empty source part.
        previous_last_doc: u32,
        /// First docid in this source part.
        first_doc: u32,
    },
    /// Checked size, offset, or posting-ordinal arithmetic overflowed.
    #[error("arithmetic overflow while computing BLOCKMAX {field}")]
    ArithmeticOverflow {
        /// Value being computed.
        field: &'static str,
    },
    /// A bounded metadata allocation failed instead of panicking.
    #[error("unable to reserve BLOCKMAX {resource} for {count} entries")]
    Allocation {
        /// Collection being reserved.
        resource: &'static str,
        /// Requested entry count.
        count: usize,
    },
}

/// One canonical BLOCKMAX record, aligned one-for-one with a POSTINGS block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockMaxEntry {
    /// First absolute docid in the posting block.
    first_doc: u32,
    /// Byte offset from this term's `postings_offset`.
    block_offset: u64,
    /// Conservative maximum-frequency code.
    max_frequency_code: u8,
    /// Minimum stored fieldnorm ID among docs in the block.
    min_fieldnorm_id: u8,
}

impl BlockMaxEntry {
    /// First absolute docid in the posting block.
    #[must_use]
    pub const fn first_doc(self) -> u32 {
        self.first_doc
    }

    /// Byte offset from this term's `postings_offset`.
    #[must_use]
    pub const fn block_offset(self) -> u64 {
        self.block_offset
    }

    /// Conservative maximum-frequency wire code.
    #[must_use]
    pub const fn max_frequency_code(self) -> u8 {
        self.max_frequency_code
    }

    /// Exact minimum stored fieldnorm ID among docs in the block.
    #[must_use]
    pub const fn min_fieldnorm_id(self) -> u8 {
        self.min_fieldnorm_id
    }

    /// Decode the conservative maximum frequency represented by this entry.
    #[must_use]
    pub const fn max_frequency(self) -> u32 {
        crate::contract::block_max_frequency_from_code(self.max_frequency_code)
    }

    /// Decode the minimum quantized document length represented by this entry.
    #[must_use]
    pub fn min_fieldnorm(self) -> u32 {
        crate::contract::id_to_fieldnorm(self.min_fieldnorm_id)
    }

    /// Compute the conservative live-snapshot BM25 tf-factor.
    #[must_use]
    pub fn tf_factor_upper_bound(self, live_avgdl: f32) -> Option<f32> {
        crate::contract::block_max_tf_factor(
            self.max_frequency_code,
            self.min_fieldnorm_id,
            live_avgdl,
        )
    }

    /// Apply the full non-negative `idf * (1 + k1) * boost` live term weight.
    #[must_use]
    pub fn score_upper_bound(self, live_avgdl: f32, nonnegative_weight: f32) -> Option<f32> {
        crate::contract::block_max_score(
            self.max_frequency_code,
            self.min_fieldnorm_id,
            live_avgdl,
            nonnegative_weight,
        )
    }
}

/// Owned canonical bytes for one term's BLOCKMAX range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedBlockMax {
    bytes: Vec<u8>,
    entry_count: usize,
}

impl EncodedBlockMax {
    /// Recompute one bound record per validated posting block.
    ///
    /// `fieldnorm_for_doc` must return the stored DOCLEN byte for every
    /// POSTINGS-present docid. Every `u8`, including zero, is valid; `None`
    /// means the column is missing or the docid is out of range.
    ///
    /// # Errors
    ///
    /// Returns a typed error for a missing fieldnorm, inconsistent validated
    /// POSTINGS bytes, checked arithmetic overflow, or allocation failure.
    pub fn encode<F>(
        postings: &PostingList<'_>,
        mut fieldnorm_for_doc: F,
    ) -> Result<Self, BlockMaxError>
    where
        F: FnMut(u32) -> Option<u8>,
    {
        let entry_count = postings.block_count();
        let mut bytes = allocate_block_max_bytes(entry_count, "fresh-seal bytes")?;
        for block_index in 0..postings.block_count() {
            let entry = derive_block_max_entry(postings, block_index, &mut fieldnorm_for_doc)?;
            append_block_max_entry(entry, &mut bytes);
        }
        Ok(Self { bytes, entry_count })
    }

    /// Re-emit Q1 concat inputs without recomputing either bound component.
    ///
    /// Only `block_offset` changes: each source offset is increased by the sum
    /// of prior source term POSTINGS byte lengths. No seam entry is added.
    ///
    /// # Errors
    ///
    /// Returns a typed error for out-of-order source docids, offset/size
    /// overflow, or allocation failure.
    pub fn concatenate(parts: &[&BlockMaxConcatList<'_>]) -> Result<Self, BlockMaxError> {
        let entry_count = parts.iter().try_fold(0_usize, |count, part| {
            count
                .checked_add(part.entry_count())
                .ok_or(BlockMaxError::ArithmeticOverflow {
                    field: "concat entry count",
                })
        })?;
        let mut bytes = allocate_block_max_bytes(entry_count, "concat bytes")?;

        let mut posting_base = 0_u64;
        let mut previous_last_doc = None;
        for (part_index, part) in parts.iter().enumerate() {
            if let (Some(previous), Some(first)) = (previous_last_doc, part.entries.first()) {
                if first.first_doc <= previous {
                    return Err(BlockMaxError::NonAscendingConcat {
                        part_index,
                        previous_last_doc: previous,
                        first_doc: first.first_doc,
                    });
                }
            }
            for entry in &part.entries {
                append_block_max_entry(
                    BlockMaxEntry {
                        first_doc: entry.first_doc,
                        block_offset: posting_base.checked_add(entry.block_offset).ok_or(
                            BlockMaxError::ArithmeticOverflow {
                                field: "rebased block offset",
                            },
                        )?,
                        max_frequency_code: entry.max_frequency_code,
                        min_fieldnorm_id: entry.min_fieldnorm_id,
                    },
                    &mut bytes,
                );
            }
            posting_base = posting_base.checked_add(part.posting_bytes_len).ok_or(
                BlockMaxError::ArithmeticOverflow {
                    field: "concat posting-byte prefix",
                },
            )?;
            if let Some(last_doc) = part.last_doc {
                previous_last_doc = Some(last_doc);
            }
        }

        Ok(Self { bytes, entry_count })
    }

    /// Borrow the exact durable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return the exact durable bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of one-per-posting-block records.
    #[must_use]
    pub const fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Re-open bytes as a merge-only view that cannot calculate score bounds.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the wire or POSTINGS cross-section invariants
    /// fail. The opaque fieldnorm byte is intentionally not trusted for score
    /// pruning until [`Self::block_max_list`] performs DOCLEN validation.
    pub fn concat_list<'a>(
        &'a self,
        postings: &PostingList<'_>,
    ) -> Result<BlockMaxConcatList<'a>, BlockMaxError> {
        BlockMaxConcatList::parse(&self.bytes, postings)
    }

    /// Re-open these owned bytes against the referenced POSTINGS term stream.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the wire or cross-section invariants fail.
    pub fn block_max_list<'a>(
        &'a self,
        postings: &PostingList<'_>,
        fieldnorms: DocLenField<'_>,
    ) -> Result<BlockMaxList<'a>, BlockMaxError> {
        BlockMaxList::parse(&self.bytes, postings, fieldnorms)
    }
}

/// Merge-only BLOCKMAX view with no score-bound API.
///
/// This validates canonical wire shape plus redundant POSTINGS keys and keeps
/// `min_fieldnorm_id` opaque. It is sufficient for Q1 byte-preserving concat;
/// only [`BlockMaxList`] can expose entries and calculate pruning bounds after
/// validation against DOCLEN.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockMaxConcatList<'a> {
    bytes: &'a [u8],
    entries: Vec<BlockMaxEntry>,
    posting_bytes_len: u64,
    last_doc: Option<u32>,
}

impl<'a> BlockMaxConcatList<'a> {
    /// Validate the merge-safe wire and POSTINGS invariants.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed/non-canonical bytes, redundant-key
    /// mismatch, a non-canonical frequency code, or allocation failure.
    pub fn parse(bytes: &'a [u8], postings: &PostingList<'_>) -> Result<Self, BlockMaxError> {
        let block_count = postings.block_count();
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(block_count)
            .map_err(|_| BlockMaxError::Allocation {
                resource: "entries",
                count: block_count,
            })?;

        let mut reader = BlockMaxByteReader::new(bytes);
        for (block_index, posting_block) in postings.blocks().iter().enumerate() {
            let first_doc = reader.read_u32()?;
            let block_offset = reader.read_vint()?;
            let max_frequency_code = reader.read_u8()?;
            let min_fieldnorm_id = reader.read_u8()?;
            if max_frequency_code == 0 {
                return Err(BlockMaxError::ZeroMaximumFrequency { block_index });
            }
            if first_doc != posting_block.first_doc {
                return Err(BlockMaxError::FirstDocMismatch {
                    block_index,
                    expected: posting_block.first_doc,
                    actual: first_doc,
                });
            }
            let expected_offset = u64::try_from(posting_block.byte_offset).map_err(|_| {
                BlockMaxError::ArithmeticOverflow {
                    field: "posting block offset",
                }
            })?;
            if block_offset != expected_offset {
                return Err(BlockMaxError::BlockOffsetMismatch {
                    block_index,
                    expected: expected_offset,
                    actual: block_offset,
                });
            }
            let required_code = posting_block.max_frequency_code;
            if max_frequency_code != required_code {
                return Err(BlockMaxError::MaximumFrequencyMismatch {
                    block_index,
                    expected: required_code,
                    actual: max_frequency_code,
                });
            }
            entries.push(BlockMaxEntry {
                first_doc,
                block_offset,
                max_frequency_code,
                min_fieldnorm_id,
            });
        }
        reader.finish()?;

        let posting_bytes_len = u64::try_from(postings.as_bytes().len()).map_err(|_| {
            BlockMaxError::ArithmeticOverflow {
                field: "posting term byte length",
            }
        })?;
        let last_doc = postings.blocks().last().map(|block| block.last_doc);
        Ok(Self {
            bytes,
            entries,
            posting_bytes_len,
            last_doc,
        })
    }

    /// Exact borrowed durable bytes copied by Q1 concat.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Number of structurally validated merge records.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

/// Borrowed, validated view of one term's BLOCKMAX range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockMaxList<'a> {
    bytes: &'a [u8],
    entries: Vec<BlockMaxEntry>,
    block_last_docs: Vec<u32>,
    posting_bytes_len: u64,
}

impl<'a> BlockMaxList<'a> {
    /// Validate exact wire grammar and cross-check POSTINGS plus DOCLEN bounds.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed/non-canonical bytes, an under- or
    /// over-stated frequency/fieldnorm bound, a cross-section mismatch, or
    /// allocation failure. No score-capable view exists before both bound
    /// components have been validated.
    pub fn parse(
        bytes: &'a [u8],
        postings: &PostingList<'_>,
        fieldnorms: DocLenField<'_>,
    ) -> Result<Self, BlockMaxError> {
        Self::parse_with_fieldnorms(bytes, postings, |doc_id| {
            fieldnorms.fieldnorm_id(u64::from(doc_id))
        })
    }

    fn parse_with_fieldnorms<F>(
        bytes: &'a [u8],
        postings: &PostingList<'_>,
        mut fieldnorm_for_doc: F,
    ) -> Result<Self, BlockMaxError>
    where
        F: FnMut(u32) -> Option<u8>,
    {
        let structural = BlockMaxConcatList::parse(bytes, postings)?;
        let block_count = structural.entry_count();
        let mut block_last_docs = Vec::new();
        block_last_docs
            .try_reserve_exact(block_count)
            .map_err(|_| BlockMaxError::Allocation {
                resource: "last-doc skip keys",
                count: block_count,
            })?;

        for (block_index, (entry, posting_block)) in
            structural.entries.iter().zip(postings.blocks()).enumerate()
        {
            let decoded = decode_block_for_block_max(postings, block_index)?;
            let required_minimum =
                minimum_block_fieldnorm(&decoded, block_index, &mut fieldnorm_for_doc)?;
            if entry.min_fieldnorm_id != required_minimum {
                return Err(BlockMaxError::MinimumFieldnormMismatch {
                    block_index,
                    expected: required_minimum,
                    actual: entry.min_fieldnorm_id,
                });
            }
            block_last_docs.push(posting_block.last_doc);
        }
        Ok(Self {
            bytes,
            entries: structural.entries,
            block_last_docs,
            posting_bytes_len: structural.posting_bytes_len,
        })
    }

    /// Exact borrowed durable bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Validated one-per-posting-block entries in durable order.
    #[must_use]
    pub fn entries(&self) -> &[BlockMaxEntry] {
        &self.entries
    }

    /// Number of validated BLOCKMAX records.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Referenced term POSTINGS byte length used by concat rebasing.
    #[must_use]
    pub const fn posting_bytes_len(&self) -> u64 {
        self.posting_bytes_len
    }

    /// Create a metadata-only block cursor positioned on the first entry.
    ///
    /// Validation is intentionally eager once per immutable term view; callers
    /// should retain this view and create cheap cursors from it per scorer.
    #[must_use]
    pub fn cursor(&self) -> BlockMaxCursor<'_> {
        BlockMaxCursor {
            entries: &self.entries,
            block_last_docs: &self.block_last_docs,
            block_index: (!self.entries.is_empty()).then_some(0),
        }
    }
}

/// Metadata-only cursor that performs no POSTINGS decoding while it moves.
pub struct BlockMaxCursor<'a> {
    entries: &'a [BlockMaxEntry],
    block_last_docs: &'a [u32],
    block_index: Option<usize>,
}

impl BlockMaxCursor<'_> {
    /// Current validated BLOCKMAX entry.
    #[must_use]
    pub fn current(&self) -> Option<BlockMaxEntry> {
        self.block_index
            .and_then(|index| self.entries.get(index).copied())
    }

    /// Current zero-based posting-block index.
    #[must_use]
    pub const fn block_index(&self) -> Option<usize> {
        self.block_index
    }

    /// Validated inclusive last docid of the current posting block.
    #[must_use]
    pub fn last_doc(&self) -> Option<u32> {
        self.block_index
            .and_then(|index| self.block_last_docs.get(index).copied())
    }

    /// Advance by one posting block. Exhaustion is fused.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<BlockMaxEntry> {
        let current = self.block_index?;
        let next = current + 1;
        if next >= self.entries.len() {
            self.block_index = None;
            return None;
        }
        self.block_index = Some(next);
        self.current()
    }

    /// Land on the first posting block whose validated last doc reaches target.
    ///
    /// This never rewinds. It skips complete blocks without decoding their
    /// postings and returns `None` when no later block can contain a matching
    /// docid.
    pub fn advance(&mut self, target: u32) -> Option<BlockMaxEntry> {
        let current = self.block_index?;
        if self.block_last_docs.get(current).copied()? >= target {
            return self.current();
        }
        let later = self.block_last_docs.get(current + 1..)?;
        let relative = later.partition_point(|last_doc| *last_doc < target);
        if relative == later.len() {
            self.block_index = None;
            return None;
        }
        self.block_index = Some(current + 1 + relative);
        self.current()
    }
}

fn encode_posting_block(block: &[Posting], output: &mut Vec<u8>) -> Result<(), PostingCodecError> {
    if block.len() == POSTINGS_PER_BLOCK {
        let first = block.first().map_or(0, |posting| posting.doc_id);
        let last = block.last().map_or(first, |posting| posting.doc_id);
        let span = u64::from(last) - u64::from(first) + 1;
        if span < 512 {
            encode_bitmap_block(block, output)
        } else {
            encode_for_block(block, output)
        }
    } else {
        encode_vint_block(block, output)
    }
}

fn derive_fresh_block_max_entry<F>(
    postings: &[Posting],
    block_index: usize,
    block_offset: u64,
    fieldnorm_for_doc: &mut F,
) -> Result<BlockMaxEntry, BlockMaxError>
where
    F: FnMut(u32) -> Option<u8>,
{
    let first_doc = postings
        .first()
        .ok_or(BlockMaxError::ArithmeticOverflow {
            field: "empty fresh-seal posting block",
        })?
        .doc_id;
    let maximum_frequency = postings
        .iter()
        .map(|posting| posting.freq)
        .max()
        .unwrap_or(0);
    let mut min_fieldnorm_id = u8::MAX;
    for posting in postings {
        let fieldnorm_id =
            fieldnorm_for_doc(posting.doc_id).ok_or(BlockMaxError::MissingFieldnorm {
                block_index,
                doc_id: posting.doc_id,
            })?;
        min_fieldnorm_id = min_fieldnorm_id.min(fieldnorm_id);
    }
    Ok(BlockMaxEntry {
        first_doc,
        block_offset,
        max_frequency_code: crate::contract::block_max_frequency_to_code(maximum_frequency),
        min_fieldnorm_id,
    })
}

fn derive_block_max_entry<F>(
    postings: &PostingList<'_>,
    block_index: usize,
    fieldnorm_for_doc: &mut F,
) -> Result<BlockMaxEntry, BlockMaxError>
where
    F: FnMut(u32) -> Option<u8>,
{
    let block = postings
        .blocks()
        .get(block_index)
        .ok_or(BlockMaxError::ArithmeticOverflow {
            field: "posting block index",
        })?;
    let decoded = decode_block_for_block_max(postings, block_index)?;
    let min_fieldnorm_id = minimum_block_fieldnorm(&decoded, block_index, fieldnorm_for_doc)?;
    Ok(BlockMaxEntry {
        first_doc: block.first_doc,
        block_offset: u64::try_from(block.byte_offset).map_err(|_| {
            BlockMaxError::ArithmeticOverflow {
                field: "posting block offset",
            }
        })?,
        max_frequency_code: canonical_block_max_frequency_code(&decoded),
        min_fieldnorm_id,
    })
}

fn decode_block_for_block_max(
    postings: &PostingList<'_>,
    block_index: usize,
) -> Result<DecodedBlock, BlockMaxError> {
    let block = postings
        .blocks()
        .get(block_index)
        .ok_or(BlockMaxError::ArithmeticOverflow {
            field: "posting block index",
        })?;
    let base_posting_ordinal = usize::try_from(block.base_posting_ordinal).map_err(|_| {
        BlockMaxError::ArithmeticOverflow {
            field: "posting ordinal",
        }
    })?;
    Ok(decode_block_at(
        postings.as_bytes(),
        block.byte_offset,
        base_posting_ordinal,
    )?)
}

fn canonical_block_max_frequency_code(decoded: &DecodedBlock) -> u8 {
    let count = usize::from(decoded.posting_count);
    let maximum = decoded.freqs[..count].iter().copied().max().unwrap_or(0);
    crate::contract::block_max_frequency_to_code(maximum)
}

fn minimum_block_fieldnorm<F>(
    decoded: &DecodedBlock,
    block_index: usize,
    fieldnorm_for_doc: &mut F,
) -> Result<u8, BlockMaxError>
where
    F: FnMut(u32) -> Option<u8>,
{
    let mut minimum = u8::MAX;
    for &doc_id in &decoded.docs[..usize::from(decoded.posting_count)] {
        let fieldnorm_id = fieldnorm_for_doc(doc_id).ok_or(BlockMaxError::MissingFieldnorm {
            block_index,
            doc_id,
        })?;
        minimum = minimum.min(fieldnorm_id);
    }
    Ok(minimum)
}

fn allocate_block_max_bytes(
    entry_count: usize,
    resource: &'static str,
) -> Result<Vec<u8>, BlockMaxError> {
    const MAX_ENTRY_BYTES: usize = 4 + 10 + 1 + 1;
    let capacity =
        entry_count
            .checked_mul(MAX_ENTRY_BYTES)
            .ok_or(BlockMaxError::ArithmeticOverflow {
                field: "encoded byte capacity",
            })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| BlockMaxError::Allocation {
            resource,
            count: entry_count,
        })?;
    Ok(bytes)
}

fn append_block_max_entry(entry: BlockMaxEntry, bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(&entry.first_doc.to_le_bytes());
    write_vint64(entry.block_offset, bytes);
    bytes.push(entry.max_frequency_code);
    bytes.push(entry.min_fieldnorm_id);
}

fn vint64_length(mut value: u64) -> usize {
    let mut length = 1;
    while value >= 0x80 {
        value >>= 7;
        length += 1;
    }
    length
}

#[allow(clippy::cast_possible_truncation)]
fn write_vint64(mut value: u64, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

struct BlockMaxByteReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> BlockMaxByteReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], BlockMaxError> {
        let offset = self.position;
        let remaining = self.bytes.len().saturating_sub(offset);
        let end = offset
            .checked_add(length)
            .ok_or(BlockMaxError::ArithmeticOverflow {
                field: "reader position",
            })?;
        let bytes = self
            .bytes
            .get(offset..end)
            .ok_or(BlockMaxError::Truncated {
                offset,
                needed: length,
                remaining,
            })?;
        self.position = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, BlockMaxError> {
        let offset = self.position;
        self.take(1)?
            .first()
            .copied()
            .ok_or(BlockMaxError::Truncated {
                offset,
                needed: 1,
                remaining: 0,
            })
    }

    fn read_u32(&mut self) -> Result<u32, BlockMaxError> {
        let bytes = self.take(4)?;
        let mut little_endian = [0_u8; 4];
        little_endian.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(little_endian))
    }

    fn read_vint(&mut self) -> Result<u64, BlockMaxError> {
        let start = self.position;
        let mut value = 0_u64;
        for byte_index in 0..10 {
            let byte = self.read_u8()?;
            if byte_index == 9 && byte & 0xfe != 0 {
                return Err(BlockMaxError::VintOverflow { offset: start });
            }
            value |= u64::from(byte & 0x7f) << (byte_index * 7);
            if byte & 0x80 == 0 {
                if vint64_length(value) != byte_index + 1 {
                    return Err(BlockMaxError::NonCanonicalVint { offset: start });
                }
                return Ok(value);
            }
        }
        Err(BlockMaxError::VintOverflow { offset: start })
    }

    fn finish(self) -> Result<(), BlockMaxError> {
        let remaining = self.bytes.len().saturating_sub(self.position);
        if remaining == 0 {
            Ok(())
        } else {
            Err(BlockMaxError::TrailingBytes {
                offset: self.position,
                remaining,
            })
        }
    }
}

#[derive(Clone, Copy)]
enum FrequencyEncoding {
    AllOne,
    Bitpacked(u8),
}

struct DecodedBlock {
    kind: PostingBlockKind,
    posting_count: u8,
    byte_len: usize,
    first_doc: u32,
    last_doc: u32,
    docs: [u32; POSTINGS_PER_BLOCK],
    freqs: [u32; POSTINGS_PER_BLOCK],
}

struct PayloadReader<'a> {
    bytes: &'a [u8],
    position: usize,
    base_offset: usize,
    block_offset: usize,
}

impl<'a> PayloadReader<'a> {
    const fn new(bytes: &'a [u8], base_offset: usize, block_offset: usize) -> Self {
        Self {
            bytes,
            position: 0,
            base_offset,
            block_offset,
        }
    }

    fn absolute_position(&self) -> usize {
        self.base_offset + self.position
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], PostingCodecError> {
        let offset = self.absolute_position();
        let remaining = self.bytes.len().saturating_sub(self.position);
        let end =
            self.position
                .checked_add(length)
                .ok_or(PostingCodecError::ArithmeticOverflow {
                    block_offset: self.block_offset,
                    field: "payload cursor",
                })?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or(PostingCodecError::Truncated {
                offset,
                needed: length,
                remaining,
            })?;
        self.position = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, PostingCodecError> {
        let offset = self.absolute_position();
        self.take(1)?
            .first()
            .copied()
            .ok_or(PostingCodecError::Truncated {
                offset,
                needed: 1,
                remaining: 0,
            })
    }

    fn read_u16(&mut self) -> Result<u16, PostingCodecError> {
        let bytes = self.take(2)?;
        let mut little_endian = [0_u8; 2];
        little_endian.copy_from_slice(bytes);
        Ok(u16::from_le_bytes(little_endian))
    }

    fn read_u32(&mut self) -> Result<u32, PostingCodecError> {
        let bytes = self.take(4)?;
        let mut little_endian = [0_u8; 4];
        little_endian.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(little_endian))
    }

    fn read_vint(&mut self) -> Result<u32, PostingCodecError> {
        let start = self.absolute_position();
        let mut value = 0_u32;
        for byte_index in 0..5 {
            let byte = self.read_u8()?;
            if byte_index == 4 && byte & 0xf0 != 0 {
                return Err(PostingCodecError::VintOverflow { offset: start });
            }
            value |= u32::from(byte & 0x7f) << (byte_index * 7);
            if byte & 0x80 == 0 {
                if vint_length(value) != byte_index + 1 {
                    return Err(PostingCodecError::NonCanonicalVint { offset: start });
                }
                return Ok(value);
            }
        }
        Err(PostingCodecError::VintOverflow { offset: start })
    }

    fn finish(self) -> Result<(), PostingCodecError> {
        let remaining = self.bytes.len().saturating_sub(self.position);
        if remaining == 0 {
            Ok(())
        } else {
            Err(PostingCodecError::TrailingPayload {
                block_offset: self.block_offset,
                remaining,
            })
        }
    }
}

fn validate_encoder_input(postings: &[Posting]) -> Result<(), PostingCodecError> {
    let mut previous = None;
    for (index, posting) in postings.iter().enumerate() {
        if posting.freq == 0 {
            return Err(PostingCodecError::ZeroFrequency {
                index,
                doc_id: posting.doc_id,
            });
        }
        if let Some(previous_doc) = previous {
            if posting.doc_id <= previous_doc {
                return Err(PostingCodecError::NonAscendingInput {
                    index,
                    previous: previous_doc,
                    doc_id: posting.doc_id,
                });
            }
        }
        previous = Some(posting.doc_id);
    }
    Ok(())
}

#[allow(clippy::cast_possible_truncation)]
fn bit_width(value: u32) -> u8 {
    (u32::BITS - value.leading_zeros()) as u8
}

fn map_bitpack<T>(
    block_offset: usize,
    result: Result<T, BitpackError>,
) -> Result<T, PostingCodecError> {
    result.map_err(|source| PostingCodecError::Bitpack {
        block_offset,
        source,
    })
}

fn encode_frequency_parts(
    postings: &[Posting],
    block_offset: usize,
) -> Result<(Vec<u8>, Vec<u8>), PostingCodecError> {
    let stored: Vec<u32> = postings.iter().map(|posting| posting.freq - 1).collect();
    let width = stored.iter().copied().map(bit_width).max().unwrap_or(0);
    if width == 0 {
        return Ok((vec![FREQ_ALL_ONE], Vec::new()));
    }
    let packed = map_bitpack(block_offset, bitpack::pack_lsb(&stored, width))?;
    Ok((vec![FREQ_BITPACKED, width], packed))
}

fn append_block(
    output: &mut Vec<u8>,
    kind: PostingBlockKind,
    posting_count: u8,
    payload: &[u8],
) -> Result<(), PostingCodecError> {
    let block_offset = output.len();
    let payload_len =
        u16::try_from(payload.len()).map_err(|_| PostingCodecError::PayloadTooLarge {
            block_offset,
            length: payload.len(),
        })?;
    output.push(kind.code());
    output.push(posting_count);
    output.extend_from_slice(&payload_len.to_le_bytes());
    output.extend_from_slice(payload);
    Ok(())
}

fn encode_for_block(postings: &[Posting], output: &mut Vec<u8>) -> Result<(), PostingCodecError> {
    let block_offset = output.len();
    let first_doc = postings.first().map_or(0, |posting| posting.doc_id);
    let deltas: Vec<u32> = postings
        .windows(2)
        .map(|pair| pair[1].doc_id - pair[0].doc_id - 1)
        .collect();
    let doc_width = deltas.iter().copied().map(bit_width).max().unwrap_or(0);
    let packed_docs = map_bitpack(block_offset, bitpack::pack_lsb(&deltas, doc_width))?;
    let (freq_meta, packed_freqs) = encode_frequency_parts(postings, block_offset)?;

    let mut payload =
        Vec::with_capacity(4 + 1 + freq_meta.len() + packed_docs.len() + packed_freqs.len());
    payload.extend_from_slice(&first_doc.to_le_bytes());
    payload.push(doc_width);
    payload.extend_from_slice(&freq_meta);
    payload.extend_from_slice(&packed_docs);
    payload.extend_from_slice(&packed_freqs);
    append_block(
        output,
        PostingBlockKind::FrameOfReference,
        POSTINGS_PER_BLOCK_U8,
        &payload,
    )
}

fn encode_bitmap_block(
    postings: &[Posting],
    output: &mut Vec<u8>,
) -> Result<(), PostingCodecError> {
    let block_offset = output.len();
    let first_doc = postings.first().map_or(0, |posting| posting.doc_id);
    let last_doc = postings.last().map_or(first_doc, |posting| posting.doc_id);
    let span_u64 = u64::from(last_doc) - u64::from(first_doc) + 1;
    let span = u16::try_from(span_u64).map_err(|_| PostingCodecError::InvalidBitmapSpan {
        block_offset,
        span: u16::MAX,
    })?;
    if !(128..512).contains(&span) {
        return Err(PostingCodecError::InvalidBitmapSpan { block_offset, span });
    }

    let mut bitmap = [0_u8; 64];
    for posting in postings {
        let relative = usize::try_from(posting.doc_id - first_doc).map_err(|_| {
            PostingCodecError::ArithmeticOverflow {
                block_offset,
                field: "bitmap relative docid",
            }
        })?;
        let byte = bitmap
            .get_mut(relative / 8)
            .ok_or(PostingCodecError::InvalidBitmap {
                block_offset,
                invariant: "docid inside 512-bit container",
            })?;
        *byte |= 1_u8 << (relative % 8);
    }
    let (freq_meta, packed_freqs) = encode_frequency_parts(postings, block_offset)?;
    let mut payload =
        Vec::with_capacity(4 + 2 + bitmap.len() + freq_meta.len() + packed_freqs.len());
    payload.extend_from_slice(&first_doc.to_le_bytes());
    payload.extend_from_slice(&span.to_le_bytes());
    payload.extend_from_slice(&bitmap);
    payload.extend_from_slice(&freq_meta);
    payload.extend_from_slice(&packed_freqs);
    append_block(
        output,
        PostingBlockKind::Bitmap,
        POSTINGS_PER_BLOCK_U8,
        &payload,
    )
}

fn encode_vint_block(postings: &[Posting], output: &mut Vec<u8>) -> Result<(), PostingCodecError> {
    if postings.is_empty() {
        return Ok(());
    }
    let mut payload = Vec::new();
    let mut previous_doc = 0_u32;
    for (index, posting) in postings.iter().enumerate() {
        let stored_doc = if index == 0 {
            posting.doc_id
        } else {
            posting.doc_id - previous_doc - 1
        };
        write_vint(stored_doc, &mut payload);
        write_vint(posting.freq, &mut payload);
        previous_doc = posting.doc_id;
    }
    let posting_count =
        u8::try_from(postings.len()).map_err(|_| PostingCodecError::InvalidBlockCount {
            offset: output.len(),
            kind: VINT_KIND,
            count: u8::MAX,
        })?;
    append_block(output, PostingBlockKind::Vint, posting_count, &payload)
}

fn vint_length(mut value: u32) -> usize {
    let mut length = 1;
    while value >= 0x80 {
        value >>= 7;
        length += 1;
    }
    length
}

#[allow(clippy::cast_possible_truncation)]
fn write_vint(mut value: u32, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn read_frequency_meta(
    reader: &mut PayloadReader<'_>,
) -> Result<FrequencyEncoding, PostingCodecError> {
    match reader.read_u8()? {
        FREQ_ALL_ONE => Ok(FrequencyEncoding::AllOne),
        FREQ_BITPACKED => Ok(FrequencyEncoding::Bitpacked(reader.read_u8()?)),
        kind => Err(PostingCodecError::InvalidFrequencyKind {
            block_offset: reader.block_offset,
            kind,
        }),
    }
}

fn decode_frequencies(
    reader: &mut PayloadReader<'_>,
    encoding: FrequencyEncoding,
    count: usize,
    output: &mut [u32; POSTINGS_PER_BLOCK],
) -> Result<(), PostingCodecError> {
    match encoding {
        FrequencyEncoding::AllOne => output[..count].fill(1),
        FrequencyEncoding::Bitpacked(width) => {
            let packed_len =
                map_bitpack(reader.block_offset, bitpack::packed_byte_len(count, width))?;
            let bytes = reader.take(packed_len)?;
            let mut stored = [0_u32; POSTINGS_PER_BLOCK];
            map_bitpack(
                reader.block_offset,
                bitpack::unpack_scalar_into(bytes, width, &mut stored[..count]),
            )?;
            let required = stored[..count]
                .iter()
                .copied()
                .map(bit_width)
                .max()
                .unwrap_or(0);
            if required == 0 {
                return Err(PostingCodecError::NonCanonicalFrequencyEncoding {
                    block_offset: reader.block_offset,
                });
            }
            if width != required {
                return Err(PostingCodecError::NonCanonicalWidth {
                    block_offset: reader.block_offset,
                    field: "frequency",
                    encoded: width,
                    required,
                });
            }
            for (destination, stored_freq) in output[..count]
                .iter_mut()
                .zip(stored[..count].iter().copied())
            {
                *destination =
                    stored_freq
                        .checked_add(1)
                        .ok_or(PostingCodecError::ArithmeticOverflow {
                            block_offset: reader.block_offset,
                            field: "frequency",
                        })?;
            }
        }
    }
    Ok(())
}

fn decode_block_at(
    bytes: &[u8],
    offset: usize,
    base_posting_ordinal: usize,
) -> Result<DecodedBlock, PostingCodecError> {
    let term_tail = bytes.get(offset..).ok_or(PostingCodecError::Truncated {
        offset,
        needed: BLOCK_HEADER_LEN,
        remaining: 0,
    })?;
    let mut header = PayloadReader::new(term_tail, offset, offset);
    let kind_code = header.read_u8()?;
    let posting_count = header.read_u8()?;
    let payload_len = usize::from(header.read_u16()?);
    let kind = PostingBlockKind::from_code(kind_code, offset)?;
    let valid_count = match kind {
        PostingBlockKind::FrameOfReference | PostingBlockKind::Bitmap => {
            usize::from(posting_count) == POSTINGS_PER_BLOCK
        }
        PostingBlockKind::Vint => (1..POSTINGS_PER_BLOCK).contains(&usize::from(posting_count)),
    };
    if !valid_count {
        return Err(PostingCodecError::InvalidBlockCount {
            offset,
            kind: kind_code,
            count: posting_count,
        });
    }
    let payload_start =
        offset
            .checked_add(BLOCK_HEADER_LEN)
            .ok_or(PostingCodecError::ArithmeticOverflow {
                block_offset: offset,
                field: "payload start",
            })?;
    let payload = header.take(payload_len)?;
    let byte_len =
        BLOCK_HEADER_LEN
            .checked_add(payload_len)
            .ok_or(PostingCodecError::ArithmeticOverflow {
                block_offset: offset,
                field: "block byte length",
            })?;

    let mut decoded = match kind {
        PostingBlockKind::FrameOfReference => decode_for_payload(payload, payload_start, offset)?,
        PostingBlockKind::Bitmap => decode_bitmap_payload(payload, payload_start, offset)?,
        PostingBlockKind::Vint => decode_vint_payload(
            payload,
            payload_start,
            offset,
            usize::from(posting_count),
            base_posting_ordinal,
        )?,
    };
    decoded.kind = kind;
    decoded.posting_count = posting_count;
    decoded.byte_len = byte_len;
    Ok(decoded)
}

fn decode_for_payload(
    payload: &[u8],
    payload_start: usize,
    block_offset: usize,
) -> Result<DecodedBlock, PostingCodecError> {
    let mut reader = PayloadReader::new(payload, payload_start, block_offset);
    let first_doc = reader.read_u32()?;
    let doc_width = reader.read_u8()?;
    let freq_encoding = read_frequency_meta(&mut reader)?;
    let packed_doc_len = map_bitpack(
        block_offset,
        bitpack::packed_byte_len(POSTINGS_PER_BLOCK - 1, doc_width),
    )?;
    let packed_docs = reader.take(packed_doc_len)?;
    let mut stored_deltas = [0_u32; POSTINGS_PER_BLOCK - 1];
    map_bitpack(
        block_offset,
        bitpack::unpack_scalar_into(packed_docs, doc_width, &mut stored_deltas),
    )?;
    let required_width = stored_deltas
        .iter()
        .copied()
        .map(bit_width)
        .max()
        .unwrap_or(0);
    if doc_width != required_width {
        return Err(PostingCodecError::NonCanonicalWidth {
            block_offset,
            field: "doc delta",
            encoded: doc_width,
            required: required_width,
        });
    }

    let mut docs = [0_u32; POSTINGS_PER_BLOCK];
    docs[0] = first_doc;
    for index in 1..POSTINGS_PER_BLOCK {
        let step = stored_deltas[index - 1].checked_add(1).ok_or(
            PostingCodecError::ArithmeticOverflow {
                block_offset,
                field: "doc delta",
            },
        )?;
        docs[index] =
            docs[index - 1]
                .checked_add(step)
                .ok_or(PostingCodecError::ArithmeticOverflow {
                    block_offset,
                    field: "docid",
                })?;
    }
    let mut freqs = [0_u32; POSTINGS_PER_BLOCK];
    decode_frequencies(&mut reader, freq_encoding, POSTINGS_PER_BLOCK, &mut freqs)?;
    reader.finish()?;
    Ok(DecodedBlock {
        kind: PostingBlockKind::FrameOfReference,
        posting_count: POSTINGS_PER_BLOCK_U8,
        byte_len: 0,
        first_doc,
        last_doc: docs[POSTINGS_PER_BLOCK - 1],
        docs,
        freqs,
    })
}

fn decode_bitmap_payload(
    payload: &[u8],
    payload_start: usize,
    block_offset: usize,
) -> Result<DecodedBlock, PostingCodecError> {
    let mut reader = PayloadReader::new(payload, payload_start, block_offset);
    let first_doc = reader.read_u32()?;
    let span = reader.read_u16()?;
    if !(128..512).contains(&span) {
        return Err(PostingCodecError::InvalidBitmapSpan { block_offset, span });
    }
    let bitmap = reader.take(64)?;
    let last_relative = usize::from(span) - 1;
    let first_is_set = bitmap.first().is_some_and(|byte| byte & 1 != 0);
    let last_is_set = bitmap
        .get(last_relative / 8)
        .is_some_and(|byte| byte & (1_u8 << (last_relative % 8)) != 0);
    if !first_is_set || !last_is_set {
        return Err(PostingCodecError::InvalidBitmap {
            block_offset,
            invariant: "first and last span bits set",
        });
    }

    let mut docs = [0_u32; POSTINGS_PER_BLOCK];
    let mut cardinality = 0_usize;
    for (byte_index, &byte) in bitmap.iter().enumerate() {
        for bit_index in 0..8 {
            if byte & (1_u8 << bit_index) == 0 {
                continue;
            }
            let relative = byte_index * 8 + bit_index;
            if relative >= usize::from(span) {
                return Err(PostingCodecError::InvalidBitmap {
                    block_offset,
                    invariant: "zero bits beyond inclusive span",
                });
            }
            let destination =
                docs.get_mut(cardinality)
                    .ok_or(PostingCodecError::InvalidBitmap {
                        block_offset,
                        invariant: "exactly 128 set bits",
                    })?;
            *destination = first_doc
                .checked_add(u32::try_from(relative).map_err(|_| {
                    PostingCodecError::ArithmeticOverflow {
                        block_offset,
                        field: "bitmap docid",
                    }
                })?)
                .ok_or(PostingCodecError::ArithmeticOverflow {
                    block_offset,
                    field: "bitmap docid",
                })?;
            cardinality += 1;
        }
    }
    if cardinality != POSTINGS_PER_BLOCK {
        return Err(PostingCodecError::InvalidBitmap {
            block_offset,
            invariant: "exactly 128 set bits",
        });
    }

    let freq_encoding = read_frequency_meta(&mut reader)?;
    let mut freqs = [0_u32; POSTINGS_PER_BLOCK];
    decode_frequencies(&mut reader, freq_encoding, POSTINGS_PER_BLOCK, &mut freqs)?;
    reader.finish()?;
    Ok(DecodedBlock {
        kind: PostingBlockKind::Bitmap,
        posting_count: POSTINGS_PER_BLOCK_U8,
        byte_len: 0,
        first_doc,
        last_doc: docs[POSTINGS_PER_BLOCK - 1],
        docs,
        freqs,
    })
}

fn decode_vint_payload(
    payload: &[u8],
    payload_start: usize,
    block_offset: usize,
    count: usize,
    base_posting_ordinal: usize,
) -> Result<DecodedBlock, PostingCodecError> {
    let mut reader = PayloadReader::new(payload, payload_start, block_offset);
    let mut docs = [0_u32; POSTINGS_PER_BLOCK];
    let mut freqs = [0_u32; POSTINGS_PER_BLOCK];
    for index in 0..count {
        let stored_doc = reader.read_vint()?;
        let doc_id = if index == 0 {
            stored_doc
        } else {
            let step = stored_doc
                .checked_add(1)
                .ok_or(PostingCodecError::ArithmeticOverflow {
                    block_offset,
                    field: "vint doc delta",
                })?;
            docs[index - 1]
                .checked_add(step)
                .ok_or(PostingCodecError::ArithmeticOverflow {
                    block_offset,
                    field: "vint docid",
                })?
        };
        let freq = reader.read_vint()?;
        if freq == 0 {
            let posting_ordinal = base_posting_ordinal.checked_add(index).ok_or(
                PostingCodecError::ArithmeticOverflow {
                    block_offset,
                    field: "posting ordinal",
                },
            )?;
            return Err(PostingCodecError::ZeroFrequency {
                index: posting_ordinal,
                doc_id,
            });
        }
        docs[index] = doc_id;
        freqs[index] = freq;
    }
    reader.finish()?;
    Ok(DecodedBlock {
        kind: PostingBlockKind::Vint,
        posting_count: u8::try_from(count).unwrap_or(u8::MAX),
        byte_len: 0,
        first_doc: docs[0],
        last_doc: docs[count - 1],
        docs,
        freqs,
    })
}

/// Fresh-seal payload target for one FSLX POSITIONS block.
///
/// Complete document runs are never split. A single run wider than the target
/// remains a legal oversized singleton, and Q1 concatenation preserves source
/// block seams without repacking them.
pub const POSITION_BLOCK_TARGET_BYTES: usize = 4_096;
/// Default maximum bytes accepted for one term's complete POSITIONS span.
pub const DEFAULT_MAX_POSITION_BYTES_PER_TERM: usize = 128 * 1024 * 1024;
/// Default maximum skip-directory entries retained for one positional term.
pub const DEFAULT_MAX_POSITION_BLOCKS: usize = 1 << 18;
/// Default maximum frequency-derived position values validated for one term.
pub const DEFAULT_MAX_POSITIONS_PER_TERM: u64 = 1 << 24;

const POSITION_DIRECTORY_HEADER_LEN: usize = 4;

/// Explicit resource ceilings for one TERMDICT-referenced POSITIONS span.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PositionListLimits {
    /// Maximum complete term-span bytes, including the block directory.
    pub max_bytes: usize,
    /// Maximum retained block-directory entries.
    pub max_blocks: usize,
    /// Maximum sum of POSTINGS frequencies validated for the term.
    pub max_positions: u64,
}

impl Default for PositionListLimits {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_MAX_POSITION_BYTES_PER_TERM,
            max_blocks: DEFAULT_MAX_POSITION_BLOCKS,
            max_positions: DEFAULT_MAX_POSITIONS_PER_TERM,
        }
    }
}

/// Typed failures from encoding or validating an FSLX POSITIONS term stream.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PositionCodecError {
    /// The corresponding POSTINGS list was invalid or could not be decoded.
    #[error(transparent)]
    Posting(#[from] PostingCodecError),
    /// Flat position values did not match the sum of posting frequencies.
    #[error("position count mismatch: POSTINGS frequencies require {expected}, got {actual}")]
    PositionCountMismatch {
        /// Frequency-derived position count.
        expected: u64,
        /// Supplied flat position count.
        actual: usize,
    },
    /// Encoder input positions decreased within one document run.
    #[error(
        "positions decrease for posting {posting_ordinal} at run index {position_index}: {previous} then {position}"
    )]
    NonAscendingPosition {
        /// Zero-based posting ordinal.
        posting_ordinal: u32,
        /// Zero-based position index within the document run.
        position_index: usize,
        /// Previous absolute position.
        previous: u32,
        /// Rejected absolute position.
        position: u32,
    },
    /// A complete term span exceeded the caller-selected byte ceiling.
    #[error("position byte limit {limit} exceeded by {actual} bytes")]
    ByteLimitExceeded {
        /// Configured byte ceiling.
        limit: usize,
        /// Required or supplied bytes.
        actual: usize,
    },
    /// Frequency-derived work exceeded the caller-selected position ceiling.
    #[error("position validation limit {limit} exceeded by {actual} values")]
    PositionLimitExceeded {
        /// Configured value ceiling.
        limit: u64,
        /// Frequency-derived values.
        actual: u64,
    },
    /// A durable or newly produced directory exceeded its block ceiling.
    #[error("position block limit {limit} exceeded by {actual} blocks")]
    BlockLimitExceeded {
        /// Configured block ceiling.
        limit: usize,
        /// Declared or produced blocks.
        actual: usize,
    },
    /// The fixed block count cannot describe the referenced posting list.
    #[error("position block count {block_count} is invalid for doc_freq {doc_freq}")]
    InvalidBlockCount {
        /// Declared directory row count.
        block_count: usize,
        /// Validated posting count.
        doc_freq: u32,
    },
    /// The first directory row was not the canonical `(0, 0)` pair.
    #[error(
        "first position block must start at posting 0 and payload offset 0, got ({first_posting_ordinal}, {block_offset})"
    )]
    InvalidDirectoryStart {
        /// First durable posting ordinal.
        first_posting_ordinal: u32,
        /// First blocks-area-relative byte offset.
        block_offset: u64,
    },
    /// Directory posting ordinals were not strictly increasing.
    #[error(
        "position block {block_index} posting ordinal is not ascending: {previous} then {current}"
    )]
    NonAscendingBlockOrdinal {
        /// Rejected block index.
        block_index: usize,
        /// Previous first posting ordinal.
        previous: u32,
        /// Rejected first posting ordinal.
        current: u32,
    },
    /// Directory payload offsets were not strictly increasing.
    #[error(
        "position block {block_index} payload offset is not ascending: {previous} then {current}"
    )]
    NonAscendingBlockOffset {
        /// Rejected block index.
        block_index: usize,
        /// Previous blocks-area-relative byte offset.
        previous: u64,
        /// Rejected blocks-area-relative byte offset.
        current: u64,
    },
    /// A block began outside the corresponding POSTINGS ordinal domain.
    #[error(
        "position block {block_index} starts at posting {first_posting_ordinal}, outside doc_freq {doc_freq}"
    )]
    BlockOrdinalOutOfRange {
        /// Rejected block index.
        block_index: usize,
        /// Rejected posting ordinal.
        first_posting_ordinal: u32,
        /// Validated posting count.
        doc_freq: u32,
    },
    /// A block offset did not point inside the blocks area.
    #[error(
        "position block {block_index} offset {block_offset} is outside payload length {payload_len}"
    )]
    BlockOffsetOutOfRange {
        /// Rejected block index.
        block_index: usize,
        /// Rejected blocks-area-relative byte offset.
        block_offset: u64,
        /// Complete blocks-area byte length.
        payload_len: usize,
    },
    /// A directory row described no postings or no payload bytes.
    #[error("position block {block_index} is empty")]
    EmptyBlock {
        /// Rejected block index.
        block_index: usize,
    },
    /// Only one complete document run may exceed the 4096-byte target.
    #[error(
        "position block {block_index} has {posting_count} documents in oversized {byte_len}-byte payload"
    )]
    OversizedMultiDocumentBlock {
        /// Rejected block index.
        block_index: usize,
        /// Frequency-derived document runs in the block.
        posting_count: u32,
        /// Block payload bytes.
        byte_len: usize,
    },
    /// Durable bytes ended before a fixed field or vint completed.
    #[error(
        "truncated position stream at offset {offset}: need {needed} bytes, only {remaining} remain"
    )]
    Truncated {
        /// Absolute byte offset in the term span.
        offset: usize,
        /// Minimum additional bytes needed.
        needed: usize,
        /// Available bytes in the bounded range.
        remaining: usize,
    },
    /// A vint used more bytes than its shortest representation.
    #[error("non-canonical {domain} position vint at offset {offset}")]
    NonCanonicalVint {
        /// Absolute vint start offset.
        offset: usize,
        /// Stable integer-domain name.
        domain: &'static str,
    },
    /// A vint exceeded its declared integer domain.
    #[error("{domain} position vint overflow at offset {offset}")]
    VintOverflow {
        /// Absolute vint start offset.
        offset: usize,
        /// Stable integer-domain name.
        domain: &'static str,
    },
    /// Reconstructing a delta-encoded absolute position overflowed u32.
    #[error(
        "position overflow for posting {posting_ordinal}: previous {previous} plus delta {delta}"
    )]
    PositionOverflow {
        /// Zero-based posting ordinal.
        posting_ordinal: u32,
        /// Previous absolute position.
        previous: u32,
        /// Rejected delta.
        delta: u32,
    },
    /// Frequency-derived runs did not exhaust one bounded block exactly.
    #[error("position block {block_index} has {remaining} trailing bytes")]
    TrailingBlockBytes {
        /// Rejected block index.
        block_index: usize,
        /// Unconsumed bytes.
        remaining: usize,
    },
    /// Concatenation inputs were not ordered by disjoint absolute docids.
    #[error("position concat docids are not ascending: {previous} then {current}")]
    NonAscendingConcat {
        /// Last docid in the preceding non-empty input.
        previous: u32,
        /// First docid in the rejected input.
        current: u32,
    },
    /// A requested posting ordinal was outside the paired list.
    #[error("position posting ordinal {posting_ordinal} is outside doc_freq {doc_freq}")]
    PostingOrdinalOutOfRange {
        /// Rejected zero-based posting ordinal.
        posting_ordinal: u32,
        /// Validated posting count.
        doc_freq: u32,
    },
    /// Checked offset, count, or length arithmetic overflowed.
    #[error("position arithmetic overflow for {field}")]
    ArithmeticOverflow {
        /// Stable value name.
        field: &'static str,
    },
    /// A bounded allocation request failed.
    #[error("unable to reserve {count} entries for {resource}")]
    Allocation {
        /// Stable allocation purpose.
        resource: &'static str,
        /// Requested entries or bytes.
        count: usize,
    },
    /// Already validated paired metadata disagreed during cursor use.
    #[error("position cursor invariant failed for {field}")]
    CursorInvariant {
        /// Stable invariant name.
        field: &'static str,
    },
}

/// Validated location and posting interval of one POSITIONS payload block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositionBlockMeta {
    base_posting_ordinal: u32,
    posting_count: u32,
    first_doc: u32,
    last_doc: u32,
    byte_offset: usize,
    byte_len: usize,
}

impl PositionBlockMeta {
    /// First zero-based posting ordinal represented by the block.
    #[must_use]
    pub const fn base_posting_ordinal(&self) -> u32 {
        self.base_posting_ordinal
    }

    /// Number of complete document runs represented by the block.
    #[must_use]
    pub const fn posting_count(&self) -> u32 {
        self.posting_count
    }

    /// Absolute first docid, derived from the paired POSTINGS stream.
    #[must_use]
    pub const fn first_doc(&self) -> u32 {
        self.first_doc
    }

    /// Absolute last docid, derived from the paired POSTINGS stream.
    #[must_use]
    pub const fn last_doc(&self) -> u32 {
        self.last_doc
    }

    /// Absolute byte offset inside the complete positional term span.
    #[must_use]
    pub const fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Exact payload byte length.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.byte_len
    }

    fn byte_range(&self) -> Option<Range<usize>> {
        self.byte_offset
            .checked_add(self.byte_len)
            .map(|end| self.byte_offset..end)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RawPositionBlock {
    first_posting_ordinal: u32,
    block_offset: u64,
}

/// Owned canonical bytes for one term's POSITIONS stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedPositionList {
    bytes: Vec<u8>,
    doc_freq: u32,
    total_positions: u64,
    block_count: usize,
}

impl EncodedPositionList {
    /// Encode flat absolute positions aligned by each posting's `freq`.
    ///
    /// Fresh blocks greedily contain complete document runs up to the 4096-byte
    /// payload target. Positions reset per posting and may repeat.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid postings, count misalignment,
    /// decreasing positions, arithmetic overflow, or resource exhaustion.
    pub fn encode(postings: &[Posting], positions: &[u32]) -> Result<Self, PositionCodecError> {
        Self::encode_with_limits(postings, positions, PositionListLimits::default())
    }

    /// Encode with explicit byte, block, and position ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`], plus a selected
    /// resource ceiling when exceeded.
    pub fn encode_with_limits(
        postings: &[Posting],
        positions: &[u32],
        limits: PositionListLimits,
    ) -> Result<Self, PositionCodecError> {
        validate_encoder_input(postings)?;
        let doc_freq =
            u32::try_from(postings.len()).map_err(|_| PostingCodecError::TooManyPostings {
                count: postings.len(),
            })?;
        let total_positions = expected_input_position_count(postings, limits.max_positions)?;
        if total_positions != u64::try_from(positions.len()).unwrap_or(u64::MAX) {
            return Err(PositionCodecError::PositionCountMismatch {
                expected: total_positions,
                actual: positions.len(),
            });
        }

        let mut payload = Vec::new();
        let mut directory = Vec::new();
        let mut position_index = 0_usize;
        for (posting_index, posting) in postings.iter().enumerate() {
            let count = usize::try_from(posting.freq).map_err(|_| {
                PositionCodecError::ArithmeticOverflow {
                    field: "posting frequency",
                }
            })?;
            let end = position_index.checked_add(count).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "flat position range",
                },
            )?;
            let run = positions.get(position_index..end).ok_or(
                PositionCodecError::PositionCountMismatch {
                    expected: total_positions,
                    actual: positions.len(),
                },
            )?;
            let posting_ordinal = u32::try_from(posting_index).map_err(|_| {
                PositionCodecError::ArithmeticOverflow {
                    field: "posting ordinal",
                }
            })?;
            let run_len = validated_position_run_len(posting_ordinal, run)?;
            let next_payload_len = payload.len().checked_add(run_len).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "position payload length",
                },
            )?;
            if next_payload_len > limits.max_bytes {
                return Err(PositionCodecError::ByteLimitExceeded {
                    limit: limits.max_bytes,
                    actual: next_payload_len,
                });
            }

            let start_new_block = directory.last().is_none_or(|_| {
                let current_start = usize::try_from(
                    directory
                        .last()
                        .map_or(0, |block: &RawPositionBlock| block.block_offset),
                )
                .unwrap_or(usize::MAX);
                payload
                    .len()
                    .checked_sub(current_start)
                    .and_then(|current| current.checked_add(run_len))
                    .is_none_or(|combined| combined > POSITION_BLOCK_TARGET_BYTES)
            });
            if start_new_block {
                if directory.len() >= limits.max_blocks {
                    return Err(PositionCodecError::BlockLimitExceeded {
                        limit: limits.max_blocks,
                        actual: directory.len().saturating_add(1),
                    });
                }
                directory
                    .try_reserve(1)
                    .map_err(|_| PositionCodecError::Allocation {
                        resource: "position block directory",
                        count: directory.len().saturating_add(1),
                    })?;
                directory.push(RawPositionBlock {
                    first_posting_ordinal: posting_ordinal,
                    block_offset: u64::try_from(payload.len()).map_err(|_| {
                        PositionCodecError::ArithmeticOverflow {
                            field: "position payload offset",
                        }
                    })?,
                });
            }
            payload
                .try_reserve(run_len)
                .map_err(|_| PositionCodecError::Allocation {
                    resource: "position payload bytes",
                    count: next_payload_len,
                })?;
            write_position_run(run, &mut payload);
            position_index = end;
        }

        Self::finish_encoding(&directory, &payload, doc_freq, total_positions, limits)
    }

    /// Concatenate validated source streams without rewriting payload blocks.
    ///
    /// Only directory ordinals and offsets are rebased. Underfilled source
    /// seams remain visible, so direct and staged Q1 merges create no fragments.
    ///
    /// # Errors
    ///
    /// Returns a typed error for reversed/overlapping source docids, checked
    /// count overflow, or resource exhaustion.
    pub fn concatenate(parts: &[&PositionList<'_>]) -> Result<Self, PositionCodecError> {
        Self::concatenate_with_limits(parts, PositionListLimits::default())
    }

    /// Concatenate with explicit byte, block, and position ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::concatenate`], plus a selected
    /// resource ceiling when exceeded.
    pub fn concatenate_with_limits(
        parts: &[&PositionList<'_>],
        limits: PositionListLimits,
    ) -> Result<Self, PositionCodecError> {
        let mut directory = Vec::new();
        let mut payload_len = 0_usize;
        let mut doc_freq = 0_u32;
        let mut total_positions = 0_u64;
        let mut previous_last_doc = None;

        for part in parts {
            let first_doc = part.posting_blocks.first().map(|block| block.first_doc);
            let last_doc = part.posting_blocks.last().map(|block| block.last_doc);
            if let (Some(previous), Some(current)) = (previous_last_doc, first_doc)
                && current <= previous
            {
                return Err(PositionCodecError::NonAscendingConcat { previous, current });
            }
            if last_doc.is_some() {
                previous_last_doc = last_doc;
            }

            total_positions = total_positions.checked_add(part.total_positions).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "concatenated position count",
                },
            )?;
            if total_positions > limits.max_positions {
                return Err(PositionCodecError::PositionLimitExceeded {
                    limit: limits.max_positions,
                    actual: total_positions,
                });
            }
            for block in &part.blocks {
                if directory.len() >= limits.max_blocks {
                    return Err(PositionCodecError::BlockLimitExceeded {
                        limit: limits.max_blocks,
                        actual: directory.len().saturating_add(1),
                    });
                }
                directory
                    .try_reserve(1)
                    .map_err(|_| PositionCodecError::Allocation {
                        resource: "concatenated position directory",
                        count: directory.len().saturating_add(1),
                    })?;
                directory.push(RawPositionBlock {
                    first_posting_ordinal: doc_freq.checked_add(block.base_posting_ordinal).ok_or(
                        PositionCodecError::ArithmeticOverflow {
                            field: "concatenated posting ordinal",
                        },
                    )?,
                    block_offset: u64::try_from(payload_len).map_err(|_| {
                        PositionCodecError::ArithmeticOverflow {
                            field: "concatenated payload offset",
                        }
                    })?,
                });
                payload_len = payload_len.checked_add(block.byte_len).ok_or(
                    PositionCodecError::ArithmeticOverflow {
                        field: "concatenated payload length",
                    },
                )?;
            }
            doc_freq = doc_freq.checked_add(part.doc_freq).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "concatenated doc_freq",
                },
            )?;
        }

        let directory_len = position_directory_len(&directory)?;
        let complete_len = directory_len.checked_add(payload_len).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "concatenated term length",
            },
        )?;
        if complete_len > limits.max_bytes {
            return Err(PositionCodecError::ByteLimitExceeded {
                limit: limits.max_bytes,
                actual: complete_len,
            });
        }
        let block_count =
            u32::try_from(directory.len()).map_err(|_| PositionCodecError::BlockLimitExceeded {
                limit: usize::try_from(u32::MAX).unwrap_or(usize::MAX),
                actual: directory.len(),
            })?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(complete_len)
            .map_err(|_| PositionCodecError::Allocation {
                resource: "concatenated position bytes",
                count: complete_len,
            })?;
        bytes.extend_from_slice(&block_count.to_le_bytes());
        for block in &directory {
            write_vint(block.first_posting_ordinal, &mut bytes);
            write_vint64(block.block_offset, &mut bytes);
        }
        for part in parts {
            for block in &part.blocks {
                let range = block
                    .byte_range()
                    .ok_or(PositionCodecError::ArithmeticOverflow {
                        field: "source position block range",
                    })?;
                let payload = part
                    .bytes
                    .get(range)
                    .ok_or(PositionCodecError::CursorInvariant {
                        field: "validated source position block",
                    })?;
                bytes.extend_from_slice(payload);
            }
        }
        debug_assert_eq!(bytes.len(), complete_len);
        Ok(Self {
            bytes,
            doc_freq,
            total_positions,
            block_count: directory.len(),
        })
    }

    fn finish_encoding(
        directory: &[RawPositionBlock],
        payload: &[u8],
        doc_freq: u32,
        total_positions: u64,
        limits: PositionListLimits,
    ) -> Result<Self, PositionCodecError> {
        let directory_len = position_directory_len(directory)?;
        let complete_len = directory_len.checked_add(payload.len()).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "position term length",
            },
        )?;
        if complete_len > limits.max_bytes {
            return Err(PositionCodecError::ByteLimitExceeded {
                limit: limits.max_bytes,
                actual: complete_len,
            });
        }
        let block_count =
            u32::try_from(directory.len()).map_err(|_| PositionCodecError::BlockLimitExceeded {
                limit: usize::try_from(u32::MAX).unwrap_or(usize::MAX),
                actual: directory.len(),
            })?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(complete_len)
            .map_err(|_| PositionCodecError::Allocation {
                resource: "position term bytes",
                count: complete_len,
            })?;
        bytes.extend_from_slice(&block_count.to_le_bytes());
        for block in directory {
            write_vint(block.first_posting_ordinal, &mut bytes);
            write_vint64(block.block_offset, &mut bytes);
        }
        bytes.extend_from_slice(payload);
        debug_assert_eq!(bytes.len(), complete_len);
        Ok(Self {
            bytes,
            doc_freq,
            total_positions,
            block_count: directory.len(),
        })
    }

    /// Exact canonical term bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return exact canonical term bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of aligned postings.
    #[must_use]
    pub const fn doc_freq(&self) -> u32 {
        self.doc_freq
    }

    /// Frequency-derived number of encoded positions.
    #[must_use]
    pub const fn total_positions(&self) -> u64 {
        self.total_positions
    }

    /// Number of preserved payload blocks.
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    /// Re-open owned bytes against the corresponding validated POSTINGS list.
    ///
    /// # Errors
    ///
    /// Returns a typed error if an internal encoder invariant regressed.
    pub fn position_list<'a>(
        &'a self,
        postings: &'a PostingList<'_>,
    ) -> Result<PositionList<'a>, PositionCodecError> {
        PositionList::parse_with_limits(
            &self.bytes,
            postings,
            PositionListLimits {
                max_bytes: self.bytes.len(),
                max_blocks: self.block_count,
                max_positions: self.total_positions,
            },
        )
    }
}

/// Borrowed, fully validated POSITIONS span paired with its POSTINGS list.
#[derive(Clone, Debug)]
pub struct PositionList<'a> {
    bytes: &'a [u8],
    posting_bytes: &'a [u8],
    posting_blocks: &'a [PostingBlockMeta],
    doc_freq: u32,
    total_positions: u64,
    blocks: Vec<PositionBlockMeta>,
}

impl<'a> PositionList<'a> {
    /// Validate a complete positional term span against POSTINGS frequencies.
    ///
    /// # Errors
    ///
    /// Returns a typed error for malformed/corrupt durable bytes, frequency
    /// misalignment, integer overflow, or resource exhaustion.
    pub fn parse(
        bytes: &'a [u8],
        postings: &'a PostingList<'_>,
    ) -> Result<Self, PositionCodecError> {
        Self::parse_with_limits(bytes, postings, PositionListLimits::default())
    }

    /// Validate with explicit byte, block, and position ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::parse`], plus a selected resource
    /// ceiling when exceeded.
    pub fn parse_with_limits(
        bytes: &'a [u8],
        postings: &'a PostingList<'_>,
        limits: PositionListLimits,
    ) -> Result<Self, PositionCodecError> {
        if bytes.len() > limits.max_bytes {
            return Err(PositionCodecError::ByteLimitExceeded {
                limit: limits.max_bytes,
                actual: bytes.len(),
            });
        }
        let total_positions = expected_position_count(postings, limits.max_positions)?;
        let count_bytes =
            bytes
                .get(..POSITION_DIRECTORY_HEADER_LEN)
                .ok_or(PositionCodecError::Truncated {
                    offset: 0,
                    needed: POSITION_DIRECTORY_HEADER_LEN,
                    remaining: bytes.len(),
                })?;
        let block_count = usize::try_from(u32::from_le_bytes([
            count_bytes[0],
            count_bytes[1],
            count_bytes[2],
            count_bytes[3],
        ]))
        .map_err(|_| PositionCodecError::ArithmeticOverflow {
            field: "position block count",
        })?;
        if block_count > limits.max_blocks {
            return Err(PositionCodecError::BlockLimitExceeded {
                limit: limits.max_blocks,
                actual: block_count,
            });
        }
        if (postings.doc_freq == 0 && block_count != 0)
            || (postings.doc_freq != 0
                && (block_count == 0
                    || u64::try_from(block_count).unwrap_or(u64::MAX)
                        > u64::from(postings.doc_freq)))
        {
            return Err(PositionCodecError::InvalidBlockCount {
                block_count,
                doc_freq: postings.doc_freq,
            });
        }
        if block_count == 0 {
            if bytes.len() != POSITION_DIRECTORY_HEADER_LEN {
                return Err(PositionCodecError::TrailingBlockBytes {
                    block_index: 0,
                    remaining: bytes.len() - POSITION_DIRECTORY_HEADER_LEN,
                });
            }
            return Ok(Self {
                bytes,
                posting_bytes: postings.bytes,
                posting_blocks: &postings.blocks,
                doc_freq: 0,
                total_positions: 0,
                blocks: Vec::new(),
            });
        }

        let mut reader =
            PositionByteReader::new(bytes, POSITION_DIRECTORY_HEADER_LEN, bytes.len())?;
        let mut raw_blocks: Vec<RawPositionBlock> = Vec::new();
        raw_blocks
            .try_reserve_exact(block_count)
            .map_err(|_| PositionCodecError::Allocation {
                resource: "position block directory",
                count: block_count,
            })?;
        for block_index in 0..block_count {
            let first_posting_ordinal = reader.read_u32_vint()?;
            let block_offset = reader.read_u64_vint()?;
            if block_index == 0 && (first_posting_ordinal != 0 || block_offset != 0) {
                return Err(PositionCodecError::InvalidDirectoryStart {
                    first_posting_ordinal,
                    block_offset,
                });
            }
            if let Some(previous) = raw_blocks.last() {
                if first_posting_ordinal <= previous.first_posting_ordinal {
                    return Err(PositionCodecError::NonAscendingBlockOrdinal {
                        block_index,
                        previous: previous.first_posting_ordinal,
                        current: first_posting_ordinal,
                    });
                }
                if block_offset <= previous.block_offset {
                    return Err(PositionCodecError::NonAscendingBlockOffset {
                        block_index,
                        previous: previous.block_offset,
                        current: block_offset,
                    });
                }
            }
            if first_posting_ordinal >= postings.doc_freq {
                return Err(PositionCodecError::BlockOrdinalOutOfRange {
                    block_index,
                    first_posting_ordinal,
                    doc_freq: postings.doc_freq,
                });
            }
            raw_blocks.push(RawPositionBlock {
                first_posting_ordinal,
                block_offset,
            });
        }
        let blocks_area_start = reader.position();
        let payload_len = bytes.len().checked_sub(blocks_area_start).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "position blocks-area length",
            },
        )?;

        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(block_count)
            .map_err(|_| PositionCodecError::Allocation {
                resource: "validated position metadata",
                count: block_count,
            })?;
        let mut postings_cursor = postings.cursor()?;
        let mut decoded_positions = 0_u64;
        for (block_index, raw) in raw_blocks.iter().copied().enumerate() {
            let next_ordinal = raw_blocks
                .get(block_index + 1)
                .map_or(postings.doc_freq, |next| next.first_posting_ordinal);
            let posting_count = next_ordinal.checked_sub(raw.first_posting_ordinal).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "position block posting count",
                },
            )?;
            if posting_count == 0 {
                return Err(PositionCodecError::EmptyBlock { block_index });
            }
            let relative_start = usize::try_from(raw.block_offset).map_err(|_| {
                PositionCodecError::BlockOffsetOutOfRange {
                    block_index,
                    block_offset: raw.block_offset,
                    payload_len,
                }
            })?;
            let relative_end_u64 = raw_blocks.get(block_index + 1).map_or_else(
                || u64::try_from(payload_len).unwrap_or(u64::MAX),
                |next| next.block_offset,
            );
            let relative_end = usize::try_from(relative_end_u64).map_err(|_| {
                PositionCodecError::BlockOffsetOutOfRange {
                    block_index,
                    block_offset: relative_end_u64,
                    payload_len,
                }
            })?;
            if relative_start >= payload_len || relative_end > payload_len {
                return Err(PositionCodecError::BlockOffsetOutOfRange {
                    block_index,
                    block_offset: if relative_start >= payload_len {
                        raw.block_offset
                    } else {
                        relative_end_u64
                    },
                    payload_len,
                });
            }
            if relative_start >= relative_end {
                return Err(PositionCodecError::EmptyBlock { block_index });
            }
            let byte_offset = blocks_area_start.checked_add(relative_start).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "position block start",
                },
            )?;
            let byte_end = blocks_area_start.checked_add(relative_end).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "position block end",
                },
            )?;
            let byte_len = byte_end.checked_sub(byte_offset).ok_or(
                PositionCodecError::ArithmeticOverflow {
                    field: "position block length",
                },
            )?;
            if byte_len > POSITION_BLOCK_TARGET_BYTES && posting_count != 1 {
                return Err(PositionCodecError::OversizedMultiDocumentBlock {
                    block_index,
                    posting_count,
                    byte_len,
                });
            }

            let actual_ordinal =
                postings_cursor
                    .posting_ordinal()
                    .ok_or(PositionCodecError::CursorInvariant {
                        field: "position block first posting",
                    })?;
            if actual_ordinal != raw.first_posting_ordinal {
                return Err(PositionCodecError::CursorInvariant {
                    field: "position directory posting ordinal",
                });
            }
            let first_doc = postings_cursor
                .doc()
                .ok_or(PositionCodecError::CursorInvariant {
                    field: "position block first doc",
                })?;
            let mut last_doc = first_doc;
            let mut block_reader = PositionByteReader::new(bytes, byte_offset, byte_end)?;
            for expected_ordinal in raw.first_posting_ordinal..next_ordinal {
                if postings_cursor.posting_ordinal() != Some(expected_ordinal) {
                    return Err(PositionCodecError::CursorInvariant {
                        field: "position/posting ordinal alignment",
                    });
                }
                let freq = postings_cursor
                    .freq()
                    .ok_or(PositionCodecError::CursorInvariant {
                        field: "position/posting frequency alignment",
                    })?;
                consume_position_run(&mut block_reader, expected_ordinal, freq)?;
                decoded_positions = decoded_positions.checked_add(u64::from(freq)).ok_or(
                    PositionCodecError::ArithmeticOverflow {
                        field: "decoded position count",
                    },
                )?;
                if decoded_positions > limits.max_positions {
                    return Err(PositionCodecError::PositionLimitExceeded {
                        limit: limits.max_positions,
                        actual: decoded_positions,
                    });
                }
                last_doc = postings_cursor
                    .doc()
                    .ok_or(PositionCodecError::CursorInvariant {
                        field: "position block last doc",
                    })?;
                postings_cursor.next()?;
            }
            if !block_reader.is_empty() {
                return Err(PositionCodecError::TrailingBlockBytes {
                    block_index,
                    remaining: block_reader.remaining(),
                });
            }
            blocks.push(PositionBlockMeta {
                base_posting_ordinal: raw.first_posting_ordinal,
                posting_count,
                first_doc,
                last_doc,
                byte_offset,
                byte_len,
            });
        }
        if postings_cursor.current().is_some() || decoded_positions != total_positions {
            return Err(PositionCodecError::CursorInvariant {
                field: "complete position/posting coverage",
            });
        }

        Ok(Self {
            bytes,
            posting_bytes: postings.bytes,
            posting_blocks: &postings.blocks,
            doc_freq: postings.doc_freq,
            total_positions,
            blocks,
        })
    }

    /// Exact borrowed durable term bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Validated aligned posting count.
    #[must_use]
    pub const fn doc_freq(&self) -> u32 {
        self.doc_freq
    }

    /// Frequency-derived position count.
    #[must_use]
    pub const fn total_positions(&self) -> u64 {
        self.total_positions
    }

    /// Number of validated payload blocks.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Validated payload block directory.
    #[must_use]
    pub fn blocks(&self) -> &[PositionBlockMeta] {
        &self.blocks
    }

    /// Exact payload bytes for one validated block.
    #[must_use]
    pub fn block_bytes(&self, block_index: usize) -> Option<&'a [u8]> {
        let range = self.blocks.get(block_index)?.byte_range()?;
        self.bytes.get(range)
    }

    /// Stream one document's absolute positions by zero-based posting ordinal.
    ///
    /// Lookup binary-searches the bounded block directory, then scans at most
    /// that block's complete frequency-derived runs. It allocates no per-doc
    /// metadata or output buffer.
    ///
    /// # Errors
    ///
    /// Returns a typed out-of-range error or an internal validated-cursor
    /// invariant failure.
    pub fn positions_for_ordinal(
        &self,
        posting_ordinal: u32,
    ) -> Result<PositionIter<'_>, PositionCodecError> {
        positions_for_ordinal(
            self.bytes,
            &self.blocks,
            self.posting_bytes,
            self.posting_blocks,
            self.doc_freq,
            posting_ordinal,
        )
    }

    /// Create an allocation-free posting cursor layered with position access.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the paired, already validated POSTINGS bytes no
    /// longer satisfy their cursor invariant.
    pub fn cursor(&self) -> Result<PositionCursor<'_>, PositionCodecError> {
        let postings = PostingCursor::new(self.posting_bytes, self.posting_blocks)?;
        let (position_block_index, position_reader) = if postings.current().is_some() {
            let first_block = self
                .blocks
                .first()
                .ok_or(PositionCodecError::CursorInvariant {
                    field: "initial position block",
                })?;
            (
                Some(0),
                PositionByteReader::from_block(self.bytes, first_block)?,
            )
        } else {
            (
                None,
                PositionByteReader::new(self.bytes, self.bytes.len(), self.bytes.len())?,
            )
        };
        Ok(PositionCursor {
            position_bytes: self.bytes,
            position_blocks: &self.blocks,
            position_block_index,
            position_reader,
            postings,
        })
    }
}

/// Streaming decoder for one frequency-bounded document position run.
pub struct PositionIter<'a> {
    reader: PositionByteReader<'a>,
    posting_ordinal: u32,
    remaining: u32,
    previous: Option<u32>,
}

impl Iterator for PositionIter<'_> {
    type Item = Result<u32, PositionCodecError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let encoded = match self.reader.read_u32_vint() {
            Ok(value) => value,
            Err(error) => {
                self.remaining = 0;
                return Some(Err(error));
            }
        };
        let position = if let Some(previous) = self.previous {
            match previous.checked_add(encoded) {
                Some(position) => position,
                None => {
                    self.remaining = 0;
                    return Some(Err(PositionCodecError::PositionOverflow {
                        posting_ordinal: self.posting_ordinal,
                        previous,
                        delta: encoded,
                    }));
                }
            }
        } else {
            encoded
        };
        self.previous = Some(position);
        self.remaining -= 1;
        Some(Ok(position))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = usize::try_from(self.remaining).unwrap_or(usize::MAX);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PositionIter<'_> {}

/// Posting cursor paired with allocation-free access to the current positions.
pub struct PositionCursor<'a> {
    position_bytes: &'a [u8],
    position_blocks: &'a [PositionBlockMeta],
    position_block_index: Option<usize>,
    position_reader: PositionByteReader<'a>,
    postings: PostingCursor<'a>,
}

impl PositionCursor<'_> {
    /// Current posting, including a valid `u32::MAX` docid when present.
    #[must_use]
    pub fn current(&self) -> Option<Posting> {
        self.postings.current()
    }

    /// Current absolute docid.
    #[must_use]
    pub fn doc(&self) -> Option<u32> {
        self.postings.doc()
    }

    /// Current term frequency.
    #[must_use]
    pub fn freq(&self) -> Option<u32> {
        self.postings.freq()
    }

    /// Current zero-based posting ordinal.
    #[must_use]
    pub fn posting_ordinal(&self) -> Option<u32> {
        self.postings.posting_ordinal()
    }

    /// Stream the current posting's absolute positions without allocation.
    ///
    /// # Errors
    ///
    /// Returns an internal invariant failure if the paired validated streams
    /// disagree.
    pub fn positions(&self) -> Result<Option<PositionIter<'_>>, PositionCodecError> {
        let Some(posting_ordinal) = self.postings.posting_ordinal() else {
            return Ok(None);
        };
        let freq = self
            .postings
            .freq()
            .ok_or(PositionCodecError::CursorInvariant {
                field: "current position frequency",
            })?;
        if self.position_block_index.is_none() {
            return Err(PositionCodecError::CursorInvariant {
                field: "current position block",
            });
        }
        Ok(Some(PositionIter {
            reader: self.position_reader.clone(),
            posting_ordinal,
            remaining: freq,
            previous: None,
        }))
    }

    /// Move strictly forward by one posting. Exhaustion is fused.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the paired POSTINGS cursor cannot decode its
    /// next validated block.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<Posting>, PositionCodecError> {
        self.advance_one()
    }

    /// Land on the first posting whose docid is at least `target`.
    ///
    /// # Errors
    ///
    /// Returns a typed error if the paired POSTINGS cursor cannot decode the
    /// selected validated block.
    pub fn advance(&mut self, target: u32) -> Result<Option<Posting>, PositionCodecError> {
        let Some(current) = self.postings.current() else {
            return Ok(None);
        };
        if current.doc_id >= target {
            return Ok(Some(current));
        }

        let destination = self
            .position_blocks
            .partition_point(|block| block.last_doc < target);
        let Some(destination_block) = self.position_blocks.get(destination) else {
            let landed = self.postings.advance(target)?;
            if landed.is_some() {
                return Err(PositionCodecError::CursorInvariant {
                    field: "position advance exhaustion",
                });
            }
            self.position_block_index = None;
            self.position_reader = PositionByteReader::new(
                self.position_bytes,
                self.position_bytes.len(),
                self.position_bytes.len(),
            )?;
            return Ok(None);
        };

        let current_block =
            self.position_block_index
                .ok_or(PositionCodecError::CursorInvariant {
                    field: "position advance current block",
                })?;
        if destination > current_block {
            let landed = self.postings.advance(destination_block.first_doc)?;
            if landed.map(|posting| posting.doc_id) != Some(destination_block.first_doc)
                || self.postings.posting_ordinal() != Some(destination_block.base_posting_ordinal)
            {
                return Err(PositionCodecError::CursorInvariant {
                    field: "position advance block seek",
                });
            }
            self.position_block_index = Some(destination);
            self.position_reader =
                PositionByteReader::from_block(self.position_bytes, destination_block)?;
        } else if destination < current_block {
            return Err(PositionCodecError::CursorInvariant {
                field: "position advance moved backward",
            });
        }

        while self.postings.doc().is_some_and(|doc| doc < target) {
            self.advance_one()?;
        }
        Ok(self.postings.current())
    }

    fn advance_one(&mut self) -> Result<Option<Posting>, PositionCodecError> {
        let Some(posting_ordinal) = self.postings.posting_ordinal() else {
            return Ok(None);
        };
        let freq = self
            .postings
            .freq()
            .ok_or(PositionCodecError::CursorInvariant {
                field: "position cursor frequency",
            })?;
        consume_position_run(&mut self.position_reader, posting_ordinal, freq)?;
        let next = self.postings.next()?;
        let Some(next_posting) = next else {
            if !self.position_reader.is_empty() {
                return Err(PositionCodecError::CursorInvariant {
                    field: "final position block exhaustion",
                });
            }
            self.position_block_index = None;
            return Ok(None);
        };

        let block_index = self
            .position_block_index
            .ok_or(PositionCodecError::CursorInvariant {
                field: "position cursor block index",
            })?;
        let block =
            self.position_blocks
                .get(block_index)
                .ok_or(PositionCodecError::CursorInvariant {
                    field: "position cursor block metadata",
                })?;
        let block_end_ordinal = block
            .base_posting_ordinal
            .checked_add(block.posting_count)
            .ok_or(PositionCodecError::ArithmeticOverflow {
                field: "position cursor block end ordinal",
            })?;
        let next_ordinal =
            self.postings
                .posting_ordinal()
                .ok_or(PositionCodecError::CursorInvariant {
                    field: "position cursor next ordinal",
                })?;
        if next_ordinal == block_end_ordinal {
            if !self.position_reader.is_empty() {
                return Err(PositionCodecError::CursorInvariant {
                    field: "position block exhaustion",
                });
            }
            let next_block_index =
                block_index
                    .checked_add(1)
                    .ok_or(PositionCodecError::ArithmeticOverflow {
                        field: "next position block index",
                    })?;
            let next_block = self.position_blocks.get(next_block_index).ok_or(
                PositionCodecError::CursorInvariant {
                    field: "next position block metadata",
                },
            )?;
            if next_block.base_posting_ordinal != next_ordinal
                || next_block.first_doc != next_posting.doc_id
            {
                return Err(PositionCodecError::CursorInvariant {
                    field: "next position block alignment",
                });
            }
            self.position_block_index = Some(next_block_index);
            self.position_reader = PositionByteReader::from_block(self.position_bytes, next_block)?;
        } else if next_ordinal > block_end_ordinal {
            return Err(PositionCodecError::CursorInvariant {
                field: "position cursor skipped block seam",
            });
        }
        Ok(Some(next_posting))
    }
}

#[derive(Clone, Debug)]
struct PositionByteReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
    end: usize,
}

impl<'a> PositionByteReader<'a> {
    fn new(bytes: &'a [u8], start: usize, end: usize) -> Result<Self, PositionCodecError> {
        if start > end || end > bytes.len() {
            return Err(PositionCodecError::Truncated {
                offset: start.min(bytes.len()),
                needed: end.saturating_sub(start),
                remaining: bytes.len().saturating_sub(start.min(bytes.len())),
            });
        }
        Ok(Self {
            bytes,
            cursor: start,
            end,
        })
    }

    fn from_block(bytes: &'a [u8], block: &PositionBlockMeta) -> Result<Self, PositionCodecError> {
        let end = block.byte_offset.checked_add(block.byte_len).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "position block reader end",
            },
        )?;
        Self::new(bytes, block.byte_offset, end)
    }

    const fn position(&self) -> usize {
        self.cursor
    }

    const fn remaining(&self) -> usize {
        self.end - self.cursor
    }

    const fn is_empty(&self) -> bool {
        self.cursor == self.end
    }

    fn read_byte(&mut self) -> Result<u8, PositionCodecError> {
        let byte =
            self.bytes
                .get(self.cursor)
                .copied()
                .ok_or_else(|| PositionCodecError::Truncated {
                    offset: self.cursor,
                    needed: 1,
                    remaining: self.remaining(),
                })?;
        if self.cursor >= self.end {
            return Err(PositionCodecError::Truncated {
                offset: self.cursor,
                needed: 1,
                remaining: 0,
            });
        }
        self.cursor += 1;
        Ok(byte)
    }

    fn read_u32_vint(&mut self) -> Result<u32, PositionCodecError> {
        let start = self.cursor;
        let mut value = 0_u32;
        for byte_index in 0..5 {
            let byte = self.read_byte()?;
            let payload = byte & 0x7f;
            if byte_index == 4 && payload > 0x0f {
                return Err(PositionCodecError::VintOverflow {
                    offset: start,
                    domain: "u32",
                });
            }
            value |= u32::from(payload) << (byte_index * 7);
            if byte & 0x80 == 0 {
                if vint_length(value) != byte_index + 1 {
                    return Err(PositionCodecError::NonCanonicalVint {
                        offset: start,
                        domain: "u32",
                    });
                }
                return Ok(value);
            }
        }
        Err(PositionCodecError::VintOverflow {
            offset: start,
            domain: "u32",
        })
    }

    fn read_u64_vint(&mut self) -> Result<u64, PositionCodecError> {
        let start = self.cursor;
        let mut value = 0_u64;
        for byte_index in 0..10 {
            let byte = self.read_byte()?;
            let payload = byte & 0x7f;
            if byte_index == 9 && payload > 1 {
                return Err(PositionCodecError::VintOverflow {
                    offset: start,
                    domain: "u64",
                });
            }
            value |= u64::from(payload) << (byte_index * 7);
            if byte & 0x80 == 0 {
                if vint64_length(value) != byte_index + 1 {
                    return Err(PositionCodecError::NonCanonicalVint {
                        offset: start,
                        domain: "u64",
                    });
                }
                return Ok(value);
            }
        }
        Err(PositionCodecError::VintOverflow {
            offset: start,
            domain: "u64",
        })
    }
}

fn expected_position_count(
    postings: &PostingList<'_>,
    limit: u64,
) -> Result<u64, PositionCodecError> {
    let mut total = 0_u64;
    let mut cursor = postings.cursor()?;
    while let Some(posting) = cursor.current() {
        total = total.checked_add(u64::from(posting.freq)).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "frequency-derived position count",
            },
        )?;
        if total > limit {
            return Err(PositionCodecError::PositionLimitExceeded {
                limit,
                actual: total,
            });
        }
        cursor.next()?;
    }
    Ok(total)
}

fn expected_input_position_count(
    postings: &[Posting],
    limit: u64,
) -> Result<u64, PositionCodecError> {
    let mut total = 0_u64;
    for posting in postings {
        total = total.checked_add(u64::from(posting.freq)).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "frequency-derived position count",
            },
        )?;
        if total > limit {
            return Err(PositionCodecError::PositionLimitExceeded {
                limit,
                actual: total,
            });
        }
    }
    Ok(total)
}

fn validated_position_run_len(
    posting_ordinal: u32,
    positions: &[u32],
) -> Result<usize, PositionCodecError> {
    let Some((&first, rest)) = positions.split_first() else {
        return Ok(0);
    };
    let mut length = vint_length(first);
    let mut previous = first;
    for (index, &position) in rest.iter().enumerate() {
        if position < previous {
            return Err(PositionCodecError::NonAscendingPosition {
                posting_ordinal,
                position_index: index + 1,
                previous,
                position,
            });
        }
        length = length.checked_add(vint_length(position - previous)).ok_or(
            PositionCodecError::ArithmeticOverflow {
                field: "encoded position run length",
            },
        )?;
        previous = position;
    }
    Ok(length)
}

fn write_position_run(positions: &[u32], output: &mut Vec<u8>) {
    let Some((&first, rest)) = positions.split_first() else {
        return;
    };
    write_vint(first, output);
    let mut previous = first;
    for &position in rest {
        write_vint(position - previous, output);
        previous = position;
    }
}

fn position_directory_len(directory: &[RawPositionBlock]) -> Result<usize, PositionCodecError> {
    let mut length = POSITION_DIRECTORY_HEADER_LEN;
    for block in directory {
        length = length
            .checked_add(vint_length(block.first_posting_ordinal))
            .and_then(|value| value.checked_add(vint64_length(block.block_offset)))
            .ok_or(PositionCodecError::ArithmeticOverflow {
                field: "position directory length",
            })?;
    }
    Ok(length)
}

fn consume_position_run(
    reader: &mut PositionByteReader<'_>,
    posting_ordinal: u32,
    freq: u32,
) -> Result<(), PositionCodecError> {
    let mut previous: Option<u32> = None;
    for _ in 0..freq {
        let encoded = reader.read_u32_vint()?;
        let position = if let Some(previous) = previous {
            previous
                .checked_add(encoded)
                .ok_or(PositionCodecError::PositionOverflow {
                    posting_ordinal,
                    previous,
                    delta: encoded,
                })?
        } else {
            encoded
        };
        previous = Some(position);
    }
    Ok(())
}

fn positions_for_ordinal<'a>(
    position_bytes: &'a [u8],
    position_blocks: &[PositionBlockMeta],
    posting_bytes: &[u8],
    posting_blocks: &[PostingBlockMeta],
    doc_freq: u32,
    posting_ordinal: u32,
) -> Result<PositionIter<'a>, PositionCodecError> {
    if posting_ordinal >= doc_freq {
        return Err(PositionCodecError::PostingOrdinalOutOfRange {
            posting_ordinal,
            doc_freq,
        });
    }
    let after =
        position_blocks.partition_point(|block| block.base_posting_ordinal <= posting_ordinal);
    let block_index = after
        .checked_sub(1)
        .ok_or(PositionCodecError::CursorInvariant {
            field: "position block lookup",
        })?;
    let block = position_blocks
        .get(block_index)
        .ok_or(PositionCodecError::CursorInvariant {
            field: "position block index",
        })?;
    let block_end_ordinal = block
        .base_posting_ordinal
        .checked_add(block.posting_count)
        .ok_or(PositionCodecError::ArithmeticOverflow {
            field: "position block end ordinal",
        })?;
    if posting_ordinal >= block_end_ordinal {
        return Err(PositionCodecError::CursorInvariant {
            field: "position block ordinal range",
        });
    }
    let block_end = block.byte_offset.checked_add(block.byte_len).ok_or(
        PositionCodecError::ArithmeticOverflow {
            field: "position block end",
        },
    )?;
    let mut reader = PositionByteReader::new(position_bytes, block.byte_offset, block_end)?;
    let mut postings = PostingCursor::new(posting_bytes, posting_blocks)?;
    let first = postings
        .advance(block.first_doc)?
        .ok_or(PositionCodecError::CursorInvariant {
            field: "position block first posting seek",
        })?;
    if first.doc_id != block.first_doc
        || postings.posting_ordinal() != Some(block.base_posting_ordinal)
    {
        return Err(PositionCodecError::CursorInvariant {
            field: "position block first posting metadata",
        });
    }

    loop {
        let current_ordinal =
            postings
                .posting_ordinal()
                .ok_or(PositionCodecError::CursorInvariant {
                    field: "position ordinal scan",
                })?;
        let freq = postings.freq().ok_or(PositionCodecError::CursorInvariant {
            field: "position frequency scan",
        })?;
        if current_ordinal == posting_ordinal {
            return Ok(PositionIter {
                reader,
                posting_ordinal,
                remaining: freq,
                previous: None,
            });
        }
        consume_position_run(&mut reader, current_ordinal, freq)?;
        if current_ordinal > posting_ordinal || postings.next()?.is_none() {
            return Err(PositionCodecError::CursorInvariant {
                field: "position ordinal reachability",
            });
        }
    }
}

/// Byte alignment of every per-field FSLX DOCLEN column.
pub const DOCLEN_ALIGNMENT: usize = 64;
/// Canonical byte written for a global-docid hole in a DOCLEN column.
///
/// This is only a canonical fill value. It is also the valid fieldnorm ID for
/// an empty present field, so presence must come from IDMAP or a posting cursor.
pub const DOCLEN_HOLE_FIELDNORM_ID: u8 = 0;
/// Packed size of one `{ field_ord: u16, offset: u32 }` directory entry.
pub const DOCLEN_DIRECTORY_ENTRY_LEN: usize = 6;
/// Packed size of one `{ field_ord: u16, total_tokens: u64, doc_count: u32 }` row.
pub const STATS_ENTRY_LEN: usize = 14;

/// Explicit resource ceilings for an FSLX DOCLEN section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DocLenLimits {
    /// Maximum schema-derived Text/Keyword field count.
    pub max_fields: usize,
    /// Maximum half-open global docid span represented by each column.
    pub max_docid_span: u64,
    /// Maximum complete section size, including directory and padding.
    pub max_section_bytes: u64,
}

impl Default for DocLenLimits {
    fn default() -> Self {
        Self {
            max_fields: 65_536,
            max_docid_span: u64::from(u32::MAX),
            max_section_bytes: u64::from(u32::MAX),
        }
    }
}

/// One raw document-length column supplied to the deterministic DOCLEN writer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DocLenFieldInput<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Raw token counts by `global_docid - docid_lo`; `None` denotes a hole.
    pub document_lengths: &'a [Option<u32>],
}

impl<'a> DocLenFieldInput<'a> {
    /// Construct one schema-ordered input column.
    #[must_use]
    pub const fn new(field_ord: u16, document_lengths: &'a [Option<u32>]) -> Self {
        Self {
            field_ord,
            document_lengths,
        }
    }
}

/// Typed failures from encoding or validating an FSLX DOCLEN section.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DocLenCodecError {
    /// A segment's half-open docid range was reversed.
    #[error("invalid DOCLEN docid range [{docid_lo}, {docid_hi})")]
    InvalidDocIdRange {
        /// Inclusive lower bound.
        docid_lo: u64,
        /// Exclusive upper bound.
        docid_hi: u64,
    },
    /// A caller or durable declaration exceeded an explicit resource ceiling.
    #[error("DOCLEN {resource} {actual} exceeds limit {limit}")]
    ResourceLimit {
        /// Bounded resource name.
        resource: &'static str,
        /// Rejected amount.
        actual: u64,
        /// Configured ceiling.
        limit: u64,
    },
    /// Expected schema field ordinals must be strictly ascending.
    #[error(
        "DOCLEN field ordinals are not strictly ascending at index {index}: previous {previous}, current {current}"
    )]
    NonAscendingFields {
        /// Rejected field index.
        index: usize,
        /// Previous ordinal.
        previous: u16,
        /// Current ordinal.
        current: u16,
    },
    /// Input or durable field identity disagreed with the schema-derived set.
    #[error("DOCLEN field {index} mismatch: expected {expected:?}, got {actual:?}")]
    UnexpectedField {
        /// Field position in schema order.
        index: usize,
        /// Expected ordinal, if the schema has one at this position.
        expected: Option<u16>,
        /// Supplied or decoded ordinal, if present.
        actual: Option<u16>,
    },
    /// Every field column must cover the complete global-docid span.
    #[error("DOCLEN field {field_ord} has {actual} entries, expected {expected}")]
    ColumnLengthMismatch {
        /// Schema field ordinal.
        field_ord: u16,
        /// Required span.
        expected: usize,
        /// Supplied entries.
        actual: usize,
    },
    /// Checked layout arithmetic overflowed.
    #[error("DOCLEN layout arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Layout component being computed.
        field: &'static str,
    },
    /// A canonical payload-relative offset did not fit the durable u32 field.
    #[error("DOCLEN offset {offset} for field {field_ord} does not fit u32")]
    OffsetUnrepresentable {
        /// Schema field ordinal.
        field_ord: u16,
        /// Computed offset.
        offset: usize,
    },
    /// A durable directory offset did not equal the one canonical layout.
    #[error("DOCLEN field {field_ord} uses non-canonical offset {encoded}; expected {expected}")]
    NonCanonicalOffset {
        /// Schema field ordinal.
        field_ord: u16,
        /// Durable payload-relative offset.
        encoded: u32,
        /// Required payload-relative offset.
        expected: u32,
    },
    /// Alignment padding is required to be all zeroes.
    #[error("DOCLEN has non-zero alignment padding at byte {offset}")]
    NonZeroPadding {
        /// First rejected padding byte.
        offset: usize,
    },
    /// Durable bytes ended before the canonical layout did.
    #[error("truncated DOCLEN section: expected at least {expected} bytes, got {actual}")]
    Truncated {
        /// Minimum length required by the validated prefix.
        expected: usize,
        /// Available length.
        actual: usize,
    },
    /// Canonical sections end exactly after their final field column.
    #[error("DOCLEN section has trailing bytes: expected {expected}, got {actual}")]
    TrailingBytes {
        /// Canonical complete size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },
    /// Fallible allocation failed without panicking.
    #[error("unable to reserve {bytes} bytes for DOCLEN {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested amount.
        bytes: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DocLenFieldMeta {
    field_ord: u16,
    range: Range<usize>,
}

/// Owned canonical bytes produced by the fresh-seal DOCLEN writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedDocLenSection {
    bytes: Vec<u8>,
    docid_lo: u64,
    docid_hi: u64,
    field_count: usize,
}

impl EncodedDocLenSection {
    /// Encode raw token counts through the pinned Tantivy fieldnorm table.
    ///
    /// `expected_field_ords` is the schema-derived, strictly ascending set of
    /// Text/Keyword fields. Every input column must match it exactly.
    ///
    /// # Errors
    ///
    /// Returns a typed error for invalid ranges, field-set drift, column-length
    /// mismatch, resource-limit exhaustion, layout overflow, or allocation
    /// failure.
    pub fn encode(
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
        fields: &[DocLenFieldInput<'_>],
    ) -> Result<Self, DocLenCodecError> {
        Self::encode_with_limits(
            docid_lo,
            docid_hi,
            expected_field_ords,
            fields,
            DocLenLimits::default(),
        )
    }

    /// Encode with caller-selected validation and allocation ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`], plus explicit
    /// resource-limit failures selected by `limits`.
    pub fn encode_with_limits(
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
        fields: &[DocLenFieldInput<'_>],
        limits: DocLenLimits,
    ) -> Result<Self, DocLenCodecError> {
        validate_doclen_field_set(expected_field_ords, fields, limits.max_fields)?;
        let span = checked_doclen_span(docid_lo, docid_hi, limits)?;
        for field in fields {
            if field.document_lengths.len() != span {
                return Err(DocLenCodecError::ColumnLengthMismatch {
                    field_ord: field.field_ord,
                    expected: span,
                    actual: field.document_lengths.len(),
                });
            }
        }

        let (offsets, total_len) = doclen_layout(expected_field_ords, span, limits)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_len)
            .map_err(|_| DocLenCodecError::Allocation {
                resource: "section bytes",
                bytes: total_len,
            })?;
        for (&field_ord, &offset) in expected_field_ords.iter().zip(&offsets) {
            bytes.extend_from_slice(&field_ord.to_le_bytes());
            let offset = u32::try_from(offset)
                .map_err(|_| DocLenCodecError::OffsetUnrepresentable { field_ord, offset })?;
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        for (field, &offset) in fields.iter().zip(&offsets) {
            bytes.resize(offset, 0);
            bytes.extend(field.document_lengths.iter().map(|length| {
                length.map_or(DOCLEN_HOLE_FIELDNORM_ID, crate::contract::fieldnorm_to_id)
            }));
        }
        debug_assert_eq!(bytes.len(), total_len);
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            field_count: fields.len(),
        })
    }

    /// Borrow the exact canonical durable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return its durable bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of schema-indexed text columns in this section.
    #[must_use]
    pub const fn field_count(&self) -> usize {
        self.field_count
    }

    /// Re-open the owned bytes through the validating reader.
    ///
    /// # Errors
    ///
    /// Returns an error if an internal invariant was violated.
    pub fn section(
        &self,
        expected_field_ords: &[u16],
    ) -> Result<DocLenSection<'_>, DocLenCodecError> {
        DocLenSection::parse(
            &self.bytes,
            self.docid_lo,
            self.docid_hi,
            expected_field_ords,
        )
    }
}

/// Borrowed view of one validated DOCLEN field column.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DocLenField<'a> {
    field_ord: u16,
    docid_lo: u64,
    fieldnorm_ids: &'a [u8],
}

impl<'a> DocLenField<'a> {
    /// Stable schema field ordinal.
    #[must_use]
    pub const fn field_ord(self) -> u16 {
        self.field_ord
    }

    /// Borrow all fieldnorm IDs in global-docid order.
    #[must_use]
    pub const fn fieldnorm_ids(self) -> &'a [u8] {
        self.fieldnorm_ids
    }

    /// Fetch one fieldnorm ID with one checked subtraction and slice lookup.
    ///
    /// `None` means the docid is outside this segment. The byte cannot identify
    /// holes: callers score only docids proven present by IDMAP/posting state.
    #[must_use]
    pub fn fieldnorm_id(self, global_docid: u64) -> Option<u8> {
        let ordinal = global_docid.checked_sub(self.docid_lo)?;
        let ordinal = usize::try_from(ordinal).ok()?;
        self.fieldnorm_ids.get(ordinal).copied()
    }

    /// Decode one in-range fieldnorm ID through the pinned Tantivy table.
    #[must_use]
    pub fn decoded_fieldnorm(self, global_docid: u64) -> Option<u32> {
        self.fieldnorm_id(global_docid)
            .map(crate::contract::id_to_fieldnorm)
    }
}

/// Borrowed, fully validated FSLX DOCLEN section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DocLenSection<'a> {
    bytes: &'a [u8],
    docid_lo: u64,
    docid_hi: u64,
    fields: Vec<DocLenFieldMeta>,
}

impl<'a> DocLenSection<'a> {
    /// Validate a DOCLEN payload against its segment range and schema field set.
    ///
    /// # Errors
    ///
    /// Rejects field drift, noncanonical offsets or padding, truncation,
    /// trailing bytes, arithmetic overflow, resource abuse, and allocation
    /// failure.
    pub fn parse(
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
    ) -> Result<Self, DocLenCodecError> {
        Self::parse_with_limits(
            bytes,
            docid_lo,
            docid_hi,
            expected_field_ords,
            DocLenLimits::default(),
        )
    }

    /// Validate with caller-selected resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::parse`], including a
    /// resource-limit error when `limits` is exceeded.
    pub fn parse_with_limits(
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
        limits: DocLenLimits,
    ) -> Result<Self, DocLenCodecError> {
        validate_doclen_expected_fields(expected_field_ords, limits.max_fields)?;
        let span = checked_doclen_span(docid_lo, docid_hi, limits)?;
        let section_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if section_len > limits.max_section_bytes {
            return Err(DocLenCodecError::ResourceLimit {
                resource: "section bytes",
                actual: section_len,
                limit: limits.max_section_bytes,
            });
        }
        let directory_len = expected_field_ords
            .len()
            .checked_mul(DOCLEN_DIRECTORY_ENTRY_LEN)
            .ok_or(DocLenCodecError::ArithmeticOverflow {
                field: "directory length",
            })?;
        if bytes.len() < directory_len {
            return Err(DocLenCodecError::Truncated {
                expected: directory_len,
                actual: bytes.len(),
            });
        }

        let (canonical_offsets, canonical_len) = doclen_layout(expected_field_ords, span, limits)?;
        let mut fields = Vec::new();
        fields
            .try_reserve_exact(expected_field_ords.len())
            .map_err(|_| DocLenCodecError::Allocation {
                resource: "field metadata",
                bytes: expected_field_ords
                    .len()
                    .saturating_mul(std::mem::size_of::<DocLenFieldMeta>()),
            })?;
        let mut previous_end = directory_len;
        for (index, (&expected_field, &canonical_offset)) in expected_field_ords
            .iter()
            .zip(&canonical_offsets)
            .enumerate()
        {
            let entry = index * DOCLEN_DIRECTORY_ENTRY_LEN;
            let actual_field = u16::from_le_bytes([bytes[entry], bytes[entry + 1]]);
            if actual_field != expected_field {
                return Err(DocLenCodecError::UnexpectedField {
                    index,
                    expected: Some(expected_field),
                    actual: Some(actual_field),
                });
            }
            let encoded_offset = u32::from_le_bytes([
                bytes[entry + 2],
                bytes[entry + 3],
                bytes[entry + 4],
                bytes[entry + 5],
            ]);
            let expected_offset = u32::try_from(canonical_offset).map_err(|_| {
                DocLenCodecError::OffsetUnrepresentable {
                    field_ord: expected_field,
                    offset: canonical_offset,
                }
            })?;
            if encoded_offset != expected_offset {
                return Err(DocLenCodecError::NonCanonicalOffset {
                    field_ord: expected_field,
                    encoded: encoded_offset,
                    expected: expected_offset,
                });
            }
            if bytes.len() < canonical_offset {
                return Err(DocLenCodecError::Truncated {
                    expected: canonical_offset,
                    actual: bytes.len(),
                });
            }
            if let Some(relative) = bytes[previous_end..canonical_offset]
                .iter()
                .position(|&byte| byte != 0)
            {
                return Err(DocLenCodecError::NonZeroPadding {
                    offset: previous_end + relative,
                });
            }
            let end =
                canonical_offset
                    .checked_add(span)
                    .ok_or(DocLenCodecError::ArithmeticOverflow {
                        field: "column end",
                    })?;
            if bytes.len() < end {
                return Err(DocLenCodecError::Truncated {
                    expected: end,
                    actual: bytes.len(),
                });
            }
            fields.push(DocLenFieldMeta {
                field_ord: expected_field,
                range: canonical_offset..end,
            });
            previous_end = end;
        }
        if bytes.len() > canonical_len {
            return Err(DocLenCodecError::TrailingBytes {
                expected: canonical_len,
                actual: bytes.len(),
            });
        }
        if bytes.len() < canonical_len {
            return Err(DocLenCodecError::Truncated {
                expected: canonical_len,
                actual: bytes.len(),
            });
        }
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            fields,
        })
    }

    /// Exact durable bytes after canonical validation.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Inclusive lower global docid bound.
    #[must_use]
    pub const fn docid_lo(&self) -> u64 {
        self.docid_lo
    }

    /// Exclusive upper global docid bound.
    #[must_use]
    pub const fn docid_hi(&self) -> u64 {
        self.docid_hi
    }

    /// Number of validated schema fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Bind one field once before the scoring loop.
    #[must_use]
    pub fn field(&self, field_ord: u16) -> Option<DocLenField<'a>> {
        let index = self
            .fields
            .binary_search_by_key(&field_ord, |field| field.field_ord)
            .ok()?;
        let field = self.fields.get(index)?;
        Some(DocLenField {
            field_ord,
            docid_lo: self.docid_lo,
            fieldnorm_ids: self.bytes.get(field.range.clone())?,
        })
    }

    /// Iterate borrowed field views in schema order.
    #[must_use]
    pub fn fields(&self) -> impl ExactSizeIterator<Item = DocLenField<'a>> + '_ {
        self.fields.iter().map(|field| DocLenField {
            field_ord: field.field_ord,
            docid_lo: self.docid_lo,
            fieldnorm_ids: &self.bytes[field.range.clone()],
        })
    }
}

fn validate_doclen_expected_fields(
    expected_field_ords: &[u16],
    max_fields: usize,
) -> Result<(), DocLenCodecError> {
    if expected_field_ords.len() > max_fields {
        return Err(DocLenCodecError::ResourceLimit {
            resource: "field count",
            actual: u64::try_from(expected_field_ords.len()).unwrap_or(u64::MAX),
            limit: u64::try_from(max_fields).unwrap_or(u64::MAX),
        });
    }
    for (index, pair) in expected_field_ords.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(DocLenCodecError::NonAscendingFields {
                index: index + 1,
                previous: pair[0],
                current: pair[1],
            });
        }
    }
    Ok(())
}

fn validate_doclen_field_set(
    expected_field_ords: &[u16],
    fields: &[DocLenFieldInput<'_>],
    max_fields: usize,
) -> Result<(), DocLenCodecError> {
    validate_doclen_expected_fields(expected_field_ords, max_fields)?;
    let compared = expected_field_ords.len().max(fields.len());
    for index in 0..compared {
        let expected = expected_field_ords.get(index).copied();
        let actual = fields.get(index).map(|field| field.field_ord);
        if expected != actual {
            return Err(DocLenCodecError::UnexpectedField {
                index,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn checked_doclen_span(
    docid_lo: u64,
    docid_hi: u64,
    limits: DocLenLimits,
) -> Result<usize, DocLenCodecError> {
    let span = docid_hi
        .checked_sub(docid_lo)
        .ok_or(DocLenCodecError::InvalidDocIdRange { docid_lo, docid_hi })?;
    if span > limits.max_docid_span {
        return Err(DocLenCodecError::ResourceLimit {
            resource: "docid span",
            actual: span,
            limit: limits.max_docid_span,
        });
    }
    usize::try_from(span).map_err(|_| DocLenCodecError::ResourceLimit {
        resource: "host docid span",
        actual: span,
        limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
    })
}

fn doclen_layout(
    field_ords: &[u16],
    span: usize,
    limits: DocLenLimits,
) -> Result<(Vec<usize>, usize), DocLenCodecError> {
    let directory_len = field_ords
        .len()
        .checked_mul(DOCLEN_DIRECTORY_ENTRY_LEN)
        .ok_or(DocLenCodecError::ArithmeticOverflow {
            field: "directory length",
        })?;
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(field_ords.len())
        .map_err(|_| DocLenCodecError::Allocation {
            resource: "directory offsets",
            bytes: field_ords
                .len()
                .saturating_mul(std::mem::size_of::<usize>()),
        })?;
    let mut cursor = directory_len;
    for &field_ord in field_ords {
        let offset = align_doclen(cursor).ok_or(DocLenCodecError::ArithmeticOverflow {
            field: "aligned column offset",
        })?;
        if u32::try_from(offset).is_err() {
            return Err(DocLenCodecError::OffsetUnrepresentable { field_ord, offset });
        }
        offsets.push(offset);
        cursor = offset
            .checked_add(span)
            .ok_or(DocLenCodecError::ArithmeticOverflow {
                field: "column end",
            })?;
    }
    let section_len = u64::try_from(cursor).unwrap_or(u64::MAX);
    if section_len > limits.max_section_bytes {
        return Err(DocLenCodecError::ResourceLimit {
            resource: "section bytes",
            actual: section_len,
            limit: limits.max_section_bytes,
        });
    }
    Ok((offsets, cursor))
}

fn align_doclen(value: usize) -> Option<usize> {
    value
        .checked_add(DOCLEN_ALIGNMENT - 1)
        .map(|sum| sum & !(DOCLEN_ALIGNMENT - 1))
}

/// Packed size of one `{ field_ord: u16, field_offset: u32 }` STOREDMETA
/// directory entry.
pub const STORED_META_DIRECTORY_ENTRY_LEN: usize = 6;

/// Explicit resource ceilings for an FSLX STOREDMETA section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoredMetaLimits {
    /// Maximum schema-derived stored field count.
    pub max_fields: usize,
    /// Maximum half-open global docid span represented by each field.
    pub max_docid_span: u64,
    /// Maximum complete section size.
    pub max_section_bytes: u64,
    /// Maximum concatenated byte blob for one stored field.
    pub max_field_blob_bytes: u64,
    /// Maximum single stored value size.
    pub max_value_bytes: u64,
}

impl Default for StoredMetaLimits {
    fn default() -> Self {
        Self {
            max_fields: 65_536,
            max_docid_span: u64::from(u32::MAX),
            max_section_bytes: u64::from(u32::MAX),
            max_field_blob_bytes: u64::from(u32::MAX),
            max_value_bytes: u64::from(u32::MAX),
        }
    }
}

/// One schema-ordered opaque stored-field column supplied to the writer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoredMetaFieldInput<'a> {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Values indexed by `global_docid - docid_lo`; `None` is an absent field
    /// or a segment-range hole, while `Some(&[])` is a present empty value.
    pub values: &'a [Option<&'a [u8]>],
}

impl<'a> StoredMetaFieldInput<'a> {
    /// Construct one schema-ordered input column.
    #[must_use]
    pub const fn new(field_ord: u16, values: &'a [Option<&'a [u8]>]) -> Self {
        Self { field_ord, values }
    }
}

/// Typed failures from STOREDMETA encoding, validation, or concatenation.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum StoredMetaCodecError {
    /// A segment's half-open docid range was reversed.
    #[error("invalid STOREDMETA docid range [{docid_lo}, {docid_hi})")]
    InvalidDocIdRange {
        /// Inclusive lower bound.
        docid_lo: u64,
        /// Exclusive upper bound.
        docid_hi: u64,
    },
    /// A caller or durable declaration exceeded an explicit resource ceiling.
    #[error("STOREDMETA {resource} {actual} exceeds limit {limit}")]
    ResourceLimit {
        /// Bounded resource name.
        resource: &'static str,
        /// Rejected amount.
        actual: u64,
        /// Configured ceiling.
        limit: u64,
    },
    /// Expected schema field ordinals must be strictly ascending.
    #[error(
        "STOREDMETA field ordinals are not strictly ascending at index {index}: previous {previous}, current {current}"
    )]
    NonAscendingFields {
        /// Rejected field index.
        index: usize,
        /// Previous ordinal.
        previous: u16,
        /// Current ordinal.
        current: u16,
    },
    /// Input or durable field identity disagreed with the schema-derived set.
    #[error("STOREDMETA field {index} mismatch: expected {expected:?}, got {actual:?}")]
    UnexpectedField {
        /// Field position in schema order.
        index: usize,
        /// Expected ordinal, if any.
        expected: Option<u16>,
        /// Supplied or decoded ordinal, if present.
        actual: Option<u16>,
    },
    /// Every stored field column must cover the complete segment span.
    #[error("STOREDMETA field {field_ord} has {actual} entries, expected {expected}")]
    ColumnLengthMismatch {
        /// Schema field ordinal.
        field_ord: u16,
        /// Required span.
        expected: usize,
        /// Supplied entries.
        actual: usize,
    },
    /// Sparse accumulator documents must map into the exact segment range.
    #[error(
        "STOREDMETA source document {index} maps to docid {docid}, outside [{docid_lo}, {docid_hi})"
    )]
    SourceDocumentOutOfRange {
        /// Completed-document index in the accumulator.
        index: usize,
        /// Rebased global document ID.
        docid: u64,
        /// Inclusive segment lower bound.
        docid_lo: u64,
        /// Exclusive segment upper bound.
        docid_hi: u64,
    },
    /// Rebased accumulator documents must remain strictly ascending.
    #[error(
        "STOREDMETA source documents are not strictly ascending at index {index}: previous {previous}, current {current}"
    )]
    NonAscendingSourceDocuments {
        /// Rejected completed-document index.
        index: usize,
        /// Previous rebased global document ID.
        previous: u64,
        /// Current rebased global document ID.
        current: u64,
    },
    /// A private Scribe column invariant was violated before sealing.
    #[error("invalid STOREDMETA source column {field_ord} at value {index}: {detail}")]
    InvalidSourceColumn {
        /// Stable schema field ordinal.
        field_ord: u16,
        /// Rejected completed-document index.
        index: usize,
        /// Violated invariant.
        detail: &'static str,
    },
    /// Checked layout arithmetic overflowed.
    #[error("STOREDMETA arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Layout component being computed.
        field: &'static str,
    },
    /// A section-relative field offset did not fit its durable u32 slot.
    #[error("STOREDMETA offset {offset} for field {field_ord} does not fit u32")]
    OffsetUnrepresentable {
        /// Schema field ordinal.
        field_ord: u16,
        /// Computed offset.
        offset: usize,
    },
    /// A durable field offset did not equal the canonical packed layout.
    #[error(
        "STOREDMETA field {field_ord} uses non-canonical offset {encoded}; expected {expected}"
    )]
    NonCanonicalFieldOffset {
        /// Schema field ordinal.
        field_ord: u16,
        /// Durable section-relative offset.
        encoded: u32,
        /// Required section-relative offset.
        expected: u32,
    },
    /// Unused high bits in the final presence byte must be zero.
    #[error("STOREDMETA field {field_ord} has non-zero unused presence bits in byte {byte:#04x}")]
    NonCanonicalPresencePadding {
        /// Schema field ordinal.
        field_ord: u16,
        /// Rejected final presence byte.
        byte: u8,
    },
    /// Every offset table begins at zero.
    #[error("STOREDMETA field {field_ord} first offset is {actual}, expected 0")]
    NonZeroFirstOffset {
        /// Schema field ordinal.
        field_ord: u16,
        /// Rejected first offset.
        actual: u32,
    },
    /// Offsets must be monotone.
    #[error("STOREDMETA field {field_ord} offsets descend at value {index}: {start} to {end}")]
    DescendingOffsets {
        /// Schema field ordinal.
        field_ord: u16,
        /// Rejected value ordinal.
        index: usize,
        /// Start offset.
        start: u32,
        /// End offset.
        end: u32,
    },
    /// An offset pointed beyond the field blob declared by the next directory
    /// entry or section end.
    #[error(
        "STOREDMETA field {field_ord} offset {offset} at value {index} exceeds blob length {blob_len}"
    )]
    OffsetOutOfBounds {
        /// Schema field ordinal.
        field_ord: u16,
        /// Rejected value ordinal.
        index: usize,
        /// Rejected offset.
        offset: u32,
        /// Exact available blob length.
        blob_len: usize,
    },
    /// An absent value or hole cannot own blob bytes.
    #[error("STOREDMETA field {field_ord} absent value {index} spans bytes {start}..{end}")]
    AbsentValueHasBytes {
        /// Schema field ordinal.
        field_ord: u16,
        /// Rejected value ordinal.
        index: usize,
        /// Start offset.
        start: u32,
        /// End offset.
        end: u32,
    },
    /// The terminal offset must name the exact end of the field payload.
    #[error(
        "STOREDMETA field {field_ord} terminal offset is {terminal}, but blob length is {blob_len}"
    )]
    TerminalOffsetMismatch {
        /// Schema field ordinal.
        field_ord: u16,
        /// Encoded terminal offset.
        terminal: u32,
        /// Exact bytes available before the next field or section end.
        blob_len: usize,
    },
    /// Durable bytes ended before a canonical component did.
    #[error("truncated STOREDMETA section: expected at least {expected} bytes, got {actual}")]
    Truncated {
        /// Minimum required length.
        expected: usize,
        /// Available length.
        actual: usize,
    },
    /// Canonical sections end exactly after their final field blob.
    #[error("STOREDMETA section has trailing bytes: expected {expected}, got {actual}")]
    TrailingBytes {
        /// Canonical complete size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },
    /// Concatenation requires at least one validated source section.
    #[error("cannot concatenate an empty STOREDMETA section list")]
    EmptyConcat,
    /// Concat inputs must be ordered and non-overlapping.
    #[error(
        "STOREDMETA concat section {index} begins at {current_lo}, before prior end {previous_hi}"
    )]
    ConcatRangeOrder {
        /// Rejected source index.
        index: usize,
        /// Prior source's exclusive high bound.
        previous_hi: u64,
        /// Current source's inclusive low bound.
        current_lo: u64,
    },
    /// Fallible allocation failed without panicking.
    #[error("unable to reserve {bytes} bytes for STOREDMETA {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested amount.
        bytes: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StoredMetaFieldMeta {
    field_ord: u16,
    presence: Range<usize>,
    offsets: Range<usize>,
    blob: Range<usize>,
}

/// Owned canonical bytes produced by the fresh-seal STOREDMETA writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedStoredMetaSection {
    bytes: Vec<u8>,
    docid_lo: u64,
    docid_hi: u64,
    field_count: usize,
    blob_bytes: u64,
}

impl EncodedStoredMetaSection {
    /// Encode schema-derived opaque stored columns.
    ///
    /// # Errors
    ///
    /// Rejects invalid ranges, field-set drift, column-length mismatch,
    /// individual or aggregate blob overflow, resource abuse, and allocation
    /// failure.
    pub fn encode(
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
        fields: &[StoredMetaFieldInput<'_>],
    ) -> Result<Self, StoredMetaCodecError> {
        Self::encode_with_limits(
            docid_lo,
            docid_hi,
            expected_field_ords,
            fields,
            StoredMetaLimits::default(),
        )
    }

    /// Encode with caller-selected validation and allocation ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`], including explicit
    /// resource-limit failures selected by `limits`.
    pub fn encode_with_limits(
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
        fields: &[StoredMetaFieldInput<'_>],
        limits: StoredMetaLimits,
    ) -> Result<Self, StoredMetaCodecError> {
        validate_stored_meta_field_set(expected_field_ords, fields, limits.max_fields)?;
        let span = checked_stored_meta_span(docid_lo, docid_hi, limits)?;
        let mut blob_lengths = Vec::new();
        blob_lengths.try_reserve_exact(fields.len()).map_err(|_| {
            StoredMetaCodecError::Allocation {
                resource: "field blob lengths",
                bytes: fields.len().saturating_mul(std::mem::size_of::<usize>()),
            }
        })?;
        let mut total_blob_bytes = 0_u64;
        for field in fields {
            if field.values.len() != span {
                return Err(StoredMetaCodecError::ColumnLengthMismatch {
                    field_ord: field.field_ord,
                    expected: span,
                    actual: field.values.len(),
                });
            }
            let mut blob_len = 0_usize;
            for value in field.values.iter().flatten() {
                let value_len = u64::try_from(value.len()).unwrap_or(u64::MAX);
                if value_len > limits.max_value_bytes {
                    return Err(StoredMetaCodecError::ResourceLimit {
                        resource: "value bytes",
                        actual: value_len,
                        limit: limits.max_value_bytes,
                    });
                }
                blob_len = blob_len.checked_add(value.len()).ok_or(
                    StoredMetaCodecError::ArithmeticOverflow {
                        field: "field blob length",
                    },
                )?;
            }
            let blob_len_u64 = u64::try_from(blob_len).unwrap_or(u64::MAX);
            if blob_len_u64 > limits.max_field_blob_bytes {
                return Err(StoredMetaCodecError::ResourceLimit {
                    resource: "field blob bytes",
                    actual: blob_len_u64,
                    limit: limits.max_field_blob_bytes,
                });
            }
            if u32::try_from(blob_len).is_err() {
                return Err(StoredMetaCodecError::ResourceLimit {
                    resource: "durable field blob bytes",
                    actual: blob_len_u64,
                    limit: u64::from(u32::MAX),
                });
            }
            total_blob_bytes = total_blob_bytes.checked_add(blob_len_u64).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "total blob bytes",
                },
            )?;
            blob_lengths.push(blob_len);
        }

        let (field_offsets, total_len) =
            stored_meta_layout(expected_field_ords, span, &blob_lengths, limits)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_len)
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "section bytes",
                bytes: total_len,
            })?;
        for (&field_ord, &offset) in expected_field_ords.iter().zip(&field_offsets) {
            bytes.extend_from_slice(&field_ord.to_le_bytes());
            let offset = u32::try_from(offset)
                .map_err(|_| StoredMetaCodecError::OffsetUnrepresentable { field_ord, offset })?;
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        let presence_len = stored_meta_presence_len(span)?;
        for ((field, &field_offset), &blob_len) in
            fields.iter().zip(&field_offsets).zip(&blob_lengths)
        {
            debug_assert_eq!(bytes.len(), field_offset);
            let presence_start = bytes.len();
            bytes.resize(
                presence_start.checked_add(presence_len).ok_or(
                    StoredMetaCodecError::ArithmeticOverflow {
                        field: "presence end",
                    },
                )?,
                0,
            );
            for (index, value) in field.values.iter().enumerate() {
                if value.is_some() {
                    bytes[presence_start + index / 8] |= 1 << (index % 8);
                }
            }
            let mut current_offset = 0_u32;
            bytes.extend_from_slice(&current_offset.to_le_bytes());
            for value in field.values {
                if let Some(value) = value {
                    current_offset = current_offset
                        .checked_add(u32::try_from(value.len()).map_err(|_| {
                            StoredMetaCodecError::ResourceLimit {
                                resource: "value bytes",
                                actual: u64::try_from(value.len()).unwrap_or(u64::MAX),
                                limit: u64::from(u32::MAX),
                            }
                        })?)
                        .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                            field: "value offset",
                        })?;
                }
                bytes.extend_from_slice(&current_offset.to_le_bytes());
            }
            debug_assert_eq!(usize::try_from(current_offset).ok(), Some(blob_len));
            for value in field.values.iter().flatten() {
                bytes.extend_from_slice(value);
            }
            tracing::debug!(
                field_ord = field.field_ord,
                blob_bytes = blob_len,
                "encoded Quill STOREDMETA field blob"
            );
        }
        debug_assert_eq!(bytes.len(), total_len);
        tracing::debug!(
            field_count = fields.len(),
            blob_bytes = total_blob_bytes,
            section_bytes = total_len,
            "encoded Quill STOREDMETA section"
        );
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            field_count: fields.len(),
            blob_bytes: total_blob_bytes,
        })
    }

    /// Seal the stored columns of one Scribe accumulator directly into the
    /// segment's positional global-docid span.
    ///
    /// `lease_docid_base` is the global docid corresponding to Scribe's local
    /// ordinal zero. Sparse local ordinals become zero-presence holes without
    /// materializing a span-sized value matrix. Field ordinals come directly
    /// from the accumulator's validated schema-derived stored columns.
    ///
    /// # Errors
    ///
    /// Rejects a range/rebase mismatch, malformed source-column invariant,
    /// durable offset overflow, resource abuse, or allocation failure.
    pub fn encode_accumulator<A: TokenAnalyzer>(
        docid_lo: u64,
        docid_hi: u64,
        lease_docid_base: u64,
        accumulator: &ColumnarAccumulator<A>,
    ) -> Result<Self, StoredMetaCodecError> {
        Self::encode_accumulator_with_limits(
            docid_lo,
            docid_hi,
            lease_docid_base,
            accumulator,
            StoredMetaLimits::default(),
        )
    }

    /// Seal Scribe columns with caller-selected validation and allocation
    /// ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode_accumulator`].
    pub fn encode_accumulator_with_limits<A: TokenAnalyzer>(
        docid_lo: u64,
        docid_hi: u64,
        lease_docid_base: u64,
        accumulator: &ColumnarAccumulator<A>,
        limits: StoredMetaLimits,
    ) -> Result<Self, StoredMetaCodecError> {
        let span = checked_stored_meta_span(docid_lo, docid_hi, limits)?;
        let document_ords = accumulator.document_ords();
        let stored_fields = accumulator.stored_fields();

        let mut previous_docid = None;
        for (index, &document_ord) in document_ords.iter().enumerate() {
            let docid = lease_docid_base
                .checked_add(u64::from(document_ord))
                .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                    field: "source document rebase",
                })?;
            if docid < docid_lo || docid >= docid_hi {
                return Err(StoredMetaCodecError::SourceDocumentOutOfRange {
                    index,
                    docid,
                    docid_lo,
                    docid_hi,
                });
            }
            if let Some(previous) = previous_docid
                && docid <= previous
            {
                return Err(StoredMetaCodecError::NonAscendingSourceDocuments {
                    index,
                    previous,
                    current: docid,
                });
            }
            previous_docid = Some(docid);
        }

        let mut expected_field_ords = Vec::new();
        expected_field_ords
            .try_reserve_exact(stored_fields.len())
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "accumulator field ordinals",
                bytes: stored_fields
                    .len()
                    .saturating_mul(std::mem::size_of::<u16>()),
            })?;
        expected_field_ords.extend(stored_fields.iter().map(|field| field.field_ord()));
        validate_stored_meta_expected_fields(&expected_field_ords, limits.max_fields)?;

        let expected_offset_count =
            document_ords
                .len()
                .checked_add(1)
                .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                    field: "source offset count",
                })?;
        let mut blob_lengths = Vec::new();
        blob_lengths
            .try_reserve_exact(stored_fields.len())
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "accumulator field blob lengths",
                bytes: stored_fields
                    .len()
                    .saturating_mul(std::mem::size_of::<usize>()),
            })?;
        let mut total_blob_bytes = 0_u64;
        for field in stored_fields {
            if field.document_count() != document_ords.len() {
                return Err(StoredMetaCodecError::ColumnLengthMismatch {
                    field_ord: field.field_ord(),
                    expected: document_ords.len(),
                    actual: field.document_count(),
                });
            }
            if field.presence().len() != document_ords.len() {
                return Err(StoredMetaCodecError::InvalidSourceColumn {
                    field_ord: field.field_ord(),
                    index: field.presence().len(),
                    detail: "presence length does not match completed documents",
                });
            }
            if field.offsets().len() != expected_offset_count {
                return Err(StoredMetaCodecError::InvalidSourceColumn {
                    field_ord: field.field_ord(),
                    index: field.offsets().len(),
                    detail: "offset count is not completed documents plus one",
                });
            }
            let first = field.offsets().first().copied().ok_or_else(|| {
                StoredMetaCodecError::InvalidSourceColumn {
                    field_ord: field.field_ord(),
                    index: 0,
                    detail: "offset table is empty",
                }
            })?;
            if first != 0 {
                return Err(StoredMetaCodecError::NonZeroFirstOffset {
                    field_ord: field.field_ord(),
                    actual: first,
                });
            }
            for index in 0..document_ords.len() {
                let presence = field.presence()[index];
                if presence > 1 {
                    return Err(StoredMetaCodecError::InvalidSourceColumn {
                        field_ord: field.field_ord(),
                        index,
                        detail: "presence byte is not canonical zero or one",
                    });
                }
                let start = field.offsets()[index];
                let end = field.offsets()[index + 1];
                if end < start {
                    return Err(StoredMetaCodecError::DescendingOffsets {
                        field_ord: field.field_ord(),
                        index,
                        start,
                        end,
                    });
                }
                if usize::try_from(end).map_or(true, |offset| offset > field.blob().len()) {
                    return Err(StoredMetaCodecError::OffsetOutOfBounds {
                        field_ord: field.field_ord(),
                        index,
                        offset: end,
                        blob_len: field.blob().len(),
                    });
                }
                if presence == 0 && start != end {
                    return Err(StoredMetaCodecError::AbsentValueHasBytes {
                        field_ord: field.field_ord(),
                        index,
                        start,
                        end,
                    });
                }
                let value_len = u64::from(end - start);
                if value_len > limits.max_value_bytes {
                    return Err(StoredMetaCodecError::ResourceLimit {
                        resource: "value bytes",
                        actual: value_len,
                        limit: limits.max_value_bytes,
                    });
                }
            }
            let terminal = field.offsets().last().copied().ok_or_else(|| {
                StoredMetaCodecError::InvalidSourceColumn {
                    field_ord: field.field_ord(),
                    index: 0,
                    detail: "offset table is empty",
                }
            })?;
            if usize::try_from(terminal).ok() != Some(field.blob().len()) {
                return Err(StoredMetaCodecError::TerminalOffsetMismatch {
                    field_ord: field.field_ord(),
                    terminal,
                    blob_len: field.blob().len(),
                });
            }
            let blob_len = field.blob().len();
            let blob_len_u64 = u64::try_from(blob_len).unwrap_or(u64::MAX);
            if blob_len_u64 > limits.max_field_blob_bytes {
                return Err(StoredMetaCodecError::ResourceLimit {
                    resource: "field blob bytes",
                    actual: blob_len_u64,
                    limit: limits.max_field_blob_bytes,
                });
            }
            total_blob_bytes = total_blob_bytes.checked_add(blob_len_u64).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "total blob bytes",
                },
            )?;
            blob_lengths.push(blob_len);
        }

        let (field_offsets, total_len) =
            stored_meta_layout(&expected_field_ords, span, &blob_lengths, limits)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_len)
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "section bytes",
                bytes: total_len,
            })?;
        for (&field_ord, &offset) in expected_field_ords.iter().zip(&field_offsets) {
            bytes.extend_from_slice(&field_ord.to_le_bytes());
            let offset = u32::try_from(offset)
                .map_err(|_| StoredMetaCodecError::OffsetUnrepresentable { field_ord, offset })?;
            bytes.extend_from_slice(&offset.to_le_bytes());
        }

        let presence_len = stored_meta_presence_len(span)?;
        for ((field, &field_offset), &blob_len) in
            stored_fields.iter().zip(&field_offsets).zip(&blob_lengths)
        {
            debug_assert_eq!(bytes.len(), field_offset);
            let presence_start = bytes.len();
            let presence_end = presence_start.checked_add(presence_len).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "presence end",
                },
            )?;
            bytes.resize(presence_end, 0);
            let mut current_offset = 0_u32;
            bytes.extend_from_slice(&current_offset.to_le_bytes());
            let mut source_index = 0_usize;
            for ordinal in 0..span {
                let global_docid = docid_lo
                    .checked_add(u64::try_from(ordinal).map_err(|_| {
                        StoredMetaCodecError::ArithmeticOverflow {
                            field: "global document ordinal",
                        }
                    })?)
                    .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                        field: "global document id",
                    })?;
                let next_source_docid = document_ords
                    .get(source_index)
                    .map(|&document_ord| {
                        lease_docid_base.checked_add(u64::from(document_ord)).ok_or(
                            StoredMetaCodecError::ArithmeticOverflow {
                                field: "source document rebase",
                            },
                        )
                    })
                    .transpose()?;
                if next_source_docid == Some(global_docid) {
                    if field.presence()[source_index] == 1 {
                        bytes[presence_start + ordinal / 8] |= 1 << (ordinal % 8);
                    }
                    current_offset = field.offsets()[source_index + 1];
                    source_index += 1;
                }
                bytes.extend_from_slice(&current_offset.to_le_bytes());
            }
            debug_assert_eq!(source_index, document_ords.len());
            debug_assert_eq!(usize::try_from(current_offset).ok(), Some(blob_len));
            bytes.extend_from_slice(field.blob());
            tracing::debug!(
                field_ord = field.field_ord(),
                blob_bytes = blob_len,
                "encoded Quill STOREDMETA accumulator field blob"
            );
        }
        debug_assert_eq!(bytes.len(), total_len);
        tracing::debug!(
            field_count = stored_fields.len(),
            blob_bytes = total_blob_bytes,
            section_bytes = total_len,
            "encoded Quill STOREDMETA section from Scribe accumulator"
        );
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            field_count: stored_fields.len(),
            blob_bytes: total_blob_bytes,
        })
    }

    /// Concatenate ordered non-overlapping STOREDMETA sections.
    ///
    /// Inter-segment docid gaps become holes. Source values remain opaque; the
    /// output only rebases offsets and concatenates their exact bytes.
    ///
    /// # Errors
    ///
    /// Rejects empty input, field drift, range overlap/reversal, resource
    /// abuse, arithmetic overflow, or allocation failure.
    pub fn concatenate(
        sections: &[StoredMetaSection<'_>],
        expected_field_ords: &[u16],
    ) -> Result<Self, StoredMetaCodecError> {
        Self::concatenate_with_limits(sections, expected_field_ords, StoredMetaLimits::default())
    }

    /// Concatenate with caller-selected validation and allocation ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::concatenate`].
    pub fn concatenate_with_limits(
        sections: &[StoredMetaSection<'_>],
        expected_field_ords: &[u16],
        limits: StoredMetaLimits,
    ) -> Result<Self, StoredMetaCodecError> {
        let Some(first) = sections.first() else {
            return Err(StoredMetaCodecError::EmptyConcat);
        };
        validate_stored_meta_expected_fields(expected_field_ords, limits.max_fields)?;
        for (section_index, section) in sections.iter().enumerate() {
            let compared = expected_field_ords.len().max(section.fields.len());
            for field_index in 0..compared {
                let expected = expected_field_ords.get(field_index).copied();
                let actual = section.fields.get(field_index).map(|field| field.field_ord);
                if expected != actual {
                    return Err(StoredMetaCodecError::UnexpectedField {
                        index: field_index,
                        expected,
                        actual,
                    });
                }
            }
            if section_index != 0 {
                let previous_hi = sections[section_index - 1].docid_hi;
                if section.docid_lo < previous_hi {
                    return Err(StoredMetaCodecError::ConcatRangeOrder {
                        index: section_index,
                        previous_hi,
                        current_lo: section.docid_lo,
                    });
                }
            }
        }
        let docid_lo = first.docid_lo;
        let docid_hi = sections
            .last()
            .map(|section| section.docid_hi)
            .ok_or(StoredMetaCodecError::EmptyConcat)?;
        let span = checked_stored_meta_span(docid_lo, docid_hi, limits)?;

        // Preflight the exact packed output before allocating it. Concat does
        // not materialize values: it validates offset deltas under the target
        // limits, then copies each opaque source blob exactly once.
        let mut blob_lengths = Vec::new();
        blob_lengths
            .try_reserve_exact(expected_field_ords.len())
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "concat field blob lengths",
                bytes: expected_field_ords
                    .len()
                    .saturating_mul(std::mem::size_of::<usize>()),
            })?;
        let mut total_blob_bytes = 0_u64;
        for (field_index, &field_ord) in expected_field_ords.iter().enumerate() {
            let mut blob_len = 0_usize;
            for section in sections {
                let field =
                    section
                        .field_at(field_index)
                        .ok_or(StoredMetaCodecError::UnexpectedField {
                            index: field_index,
                            expected: Some(field_ord),
                            actual: None,
                        })?;
                let source_span = usize::try_from(section.span()).map_err(|_| {
                    StoredMetaCodecError::ResourceLimit {
                        resource: "source docid span",
                        actual: section.span(),
                        limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
                    }
                })?;
                for ordinal in 0..source_span {
                    let start = stored_meta_offset(field.offsets, ordinal).ok_or_else(|| {
                        StoredMetaCodecError::Truncated {
                            expected: (ordinal + 1).saturating_mul(std::mem::size_of::<u32>()),
                            actual: field.offsets.len(),
                        }
                    })?;
                    let end = stored_meta_offset(field.offsets, ordinal + 1).ok_or_else(|| {
                        StoredMetaCodecError::Truncated {
                            expected: (ordinal + 2).saturating_mul(std::mem::size_of::<u32>()),
                            actual: field.offsets.len(),
                        }
                    })?;
                    let value_len = u64::from(end - start);
                    if value_len > limits.max_value_bytes {
                        return Err(StoredMetaCodecError::ResourceLimit {
                            resource: "value bytes",
                            actual: value_len,
                            limit: limits.max_value_bytes,
                        });
                    }
                }
                blob_len = blob_len.checked_add(field.blob.len()).ok_or(
                    StoredMetaCodecError::ArithmeticOverflow {
                        field: "concatenated field blob length",
                    },
                )?;
            }
            let blob_len_u64 = u64::try_from(blob_len).unwrap_or(u64::MAX);
            if blob_len_u64 > limits.max_field_blob_bytes {
                return Err(StoredMetaCodecError::ResourceLimit {
                    resource: "field blob bytes",
                    actual: blob_len_u64,
                    limit: limits.max_field_blob_bytes,
                });
            }
            if u32::try_from(blob_len).is_err() {
                return Err(StoredMetaCodecError::ResourceLimit {
                    resource: "durable field blob bytes",
                    actual: blob_len_u64,
                    limit: u64::from(u32::MAX),
                });
            }
            total_blob_bytes = total_blob_bytes.checked_add(blob_len_u64).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "total blob bytes",
                },
            )?;
            blob_lengths.push(blob_len);
        }

        let (field_offsets, total_len) =
            stored_meta_layout(expected_field_ords, span, &blob_lengths, limits)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_len)
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "section bytes",
                bytes: total_len,
            })?;
        for (&field_ord, &offset) in expected_field_ords.iter().zip(&field_offsets) {
            bytes.extend_from_slice(&field_ord.to_le_bytes());
            let offset = u32::try_from(offset)
                .map_err(|_| StoredMetaCodecError::OffsetUnrepresentable { field_ord, offset })?;
            bytes.extend_from_slice(&offset.to_le_bytes());
        }

        let presence_len = stored_meta_presence_len(span)?;
        for (field_index, ((&field_ord, &field_offset), &blob_len)) in expected_field_ords
            .iter()
            .zip(&field_offsets)
            .zip(&blob_lengths)
            .enumerate()
        {
            debug_assert_eq!(bytes.len(), field_offset);
            let presence_start = bytes.len();
            let presence_end = presence_start.checked_add(presence_len).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "presence end",
                },
            )?;
            bytes.resize(presence_end, 0);
            let mut current_offset = 0_u32;
            bytes.extend_from_slice(&current_offset.to_le_bytes());
            let mut output_ordinal = 0_usize;
            let mut cursor = docid_lo;
            for section in sections {
                let gap = section.docid_lo.checked_sub(cursor).ok_or(
                    StoredMetaCodecError::ConcatRangeOrder {
                        index: 0,
                        previous_hi: cursor,
                        current_lo: section.docid_lo,
                    },
                )?;
                let gap =
                    usize::try_from(gap).map_err(|_| StoredMetaCodecError::ResourceLimit {
                        resource: "concat gap",
                        actual: gap,
                        limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
                    })?;
                output_ordinal = output_ordinal.checked_add(gap).ok_or(
                    StoredMetaCodecError::ArithmeticOverflow {
                        field: "concat gap end",
                    },
                )?;
                for _ in 0..gap {
                    bytes.extend_from_slice(&current_offset.to_le_bytes());
                }

                let field =
                    section
                        .field_at(field_index)
                        .ok_or(StoredMetaCodecError::UnexpectedField {
                            index: field_index,
                            expected: Some(field_ord),
                            actual: None,
                        })?;
                let source_span = usize::try_from(section.span()).map_err(|_| {
                    StoredMetaCodecError::ResourceLimit {
                        resource: "source docid span",
                        actual: section.span(),
                        limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
                    }
                })?;
                for ordinal in 0..source_span {
                    if stored_meta_presence_bit(field.presence, ordinal).unwrap_or(false) {
                        bytes[presence_start + output_ordinal / 8] |= 1 << (output_ordinal % 8);
                    }
                    let start = stored_meta_offset(field.offsets, ordinal).ok_or_else(|| {
                        StoredMetaCodecError::Truncated {
                            expected: (ordinal + 1).saturating_mul(std::mem::size_of::<u32>()),
                            actual: field.offsets.len(),
                        }
                    })?;
                    let end = stored_meta_offset(field.offsets, ordinal + 1).ok_or_else(|| {
                        StoredMetaCodecError::Truncated {
                            expected: (ordinal + 2).saturating_mul(std::mem::size_of::<u32>()),
                            actual: field.offsets.len(),
                        }
                    })?;
                    current_offset = current_offset.checked_add(end - start).ok_or(
                        StoredMetaCodecError::ArithmeticOverflow {
                            field: "concatenated value offset",
                        },
                    )?;
                    bytes.extend_from_slice(&current_offset.to_le_bytes());
                    output_ordinal += 1;
                }
                cursor = section.docid_hi;
            }
            debug_assert_eq!(output_ordinal, span);
            debug_assert_eq!(usize::try_from(current_offset).ok(), Some(blob_len));
            for section in sections {
                let field =
                    section
                        .field_at(field_index)
                        .ok_or(StoredMetaCodecError::UnexpectedField {
                            index: field_index,
                            expected: Some(field_ord),
                            actual: None,
                        })?;
                bytes.extend_from_slice(field.blob);
            }
            tracing::debug!(
                field_ord,
                blob_bytes = blob_len,
                source_segments = sections.len(),
                "concatenated Quill STOREDMETA field blob"
            );
        }
        debug_assert_eq!(bytes.len(), total_len);
        tracing::debug!(
            field_count = expected_field_ords.len(),
            blob_bytes = total_blob_bytes,
            section_bytes = total_len,
            source_segments = sections.len(),
            "concatenated Quill STOREDMETA section"
        );
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            field_count: expected_field_ords.len(),
            blob_bytes: total_blob_bytes,
        })
    }

    /// Borrow the exact canonical durable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return its durable bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of schema-stored fields.
    #[must_use]
    pub const fn field_count(&self) -> usize {
        self.field_count
    }

    /// Sum of opaque field blob bytes, excluding directories and offsets.
    #[must_use]
    pub const fn blob_bytes(&self) -> u64 {
        self.blob_bytes
    }

    /// Re-open the owned bytes through the validating zero-copy reader.
    ///
    /// # Errors
    ///
    /// Returns an error if an internal invariant was violated.
    pub fn section(
        &self,
        expected_field_ords: &[u16],
    ) -> Result<StoredMetaSection<'_>, StoredMetaCodecError> {
        StoredMetaSection::parse(
            &self.bytes,
            self.docid_lo,
            self.docid_hi,
            expected_field_ords,
        )
    }
}

/// Borrowed zero-copy view of one validated STOREDMETA field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoredMetaField<'a> {
    field_ord: u16,
    docid_lo: u64,
    presence: &'a [u8],
    offsets: &'a [u8],
    blob: &'a [u8],
}

impl<'a> StoredMetaField<'a> {
    /// Stable schema field ordinal.
    #[must_use]
    pub const fn field_ord(self) -> u16 {
        self.field_ord
    }

    /// Exact concatenated opaque blob for diagnostics and concat proofs.
    #[must_use]
    pub const fn blob(self) -> &'a [u8] {
        self.blob
    }

    /// Whether one in-range document carries this field.
    #[must_use]
    pub fn is_present(self, global_docid: u64) -> bool {
        let Some(ordinal) = global_docid.checked_sub(self.docid_lo) else {
            return false;
        };
        let Ok(ordinal) = usize::try_from(ordinal) else {
            return false;
        };
        stored_meta_presence_bit(self.presence, ordinal).unwrap_or(false)
    }

    /// Borrow one opaque value with two little-endian offset reads.
    ///
    /// `None` means the docid is out of range or the value is absent.
    /// `Some(&[])` is a present empty value.
    #[must_use]
    pub fn get(self, global_docid: u64) -> Option<&'a [u8]> {
        let ordinal = global_docid.checked_sub(self.docid_lo)?;
        let ordinal = usize::try_from(ordinal).ok()?;
        if !stored_meta_presence_bit(self.presence, ordinal)? {
            return None;
        }
        let start = usize::try_from(stored_meta_offset(self.offsets, ordinal)?).ok()?;
        let end = usize::try_from(stored_meta_offset(self.offsets, ordinal + 1)?).ok()?;
        self.blob.get(start..end)
    }
}

/// Borrowed, fully validated FSLX STOREDMETA section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StoredMetaSection<'a> {
    bytes: &'a [u8],
    docid_lo: u64,
    docid_hi: u64,
    fields: Vec<StoredMetaFieldMeta>,
}

impl<'a> StoredMetaSection<'a> {
    /// Validate a STOREDMETA payload against its segment range and stored-field
    /// schema set.
    ///
    /// # Errors
    ///
    /// Rejects field drift, noncanonical directory offsets or presence bits,
    /// malformed value offsets, absent values owning bytes, truncation,
    /// trailing bytes, resource abuse, and allocation failure.
    pub fn parse(
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
    ) -> Result<Self, StoredMetaCodecError> {
        Self::parse_with_limits(
            bytes,
            docid_lo,
            docid_hi,
            expected_field_ords,
            StoredMetaLimits::default(),
        )
    }

    /// Validate with caller-selected resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::parse`], including explicit
    /// limit failures selected by `limits`.
    pub fn parse_with_limits(
        bytes: &'a [u8],
        docid_lo: u64,
        docid_hi: u64,
        expected_field_ords: &[u16],
        limits: StoredMetaLimits,
    ) -> Result<Self, StoredMetaCodecError> {
        validate_stored_meta_expected_fields(expected_field_ords, limits.max_fields)?;
        let span = checked_stored_meta_span(docid_lo, docid_hi, limits)?;
        let section_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if section_len > limits.max_section_bytes {
            return Err(StoredMetaCodecError::ResourceLimit {
                resource: "section bytes",
                actual: section_len,
                limit: limits.max_section_bytes,
            });
        }
        let directory_len = expected_field_ords
            .len()
            .checked_mul(STORED_META_DIRECTORY_ENTRY_LEN)
            .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                field: "directory length",
            })?;
        if bytes.len() < directory_len {
            return Err(StoredMetaCodecError::Truncated {
                expected: directory_len,
                actual: bytes.len(),
            });
        }
        if expected_field_ords.is_empty() {
            if bytes.is_empty() {
                return Ok(Self {
                    bytes,
                    docid_lo,
                    docid_hi,
                    fields: Vec::new(),
                });
            }
            return Err(StoredMetaCodecError::TrailingBytes {
                expected: 0,
                actual: bytes.len(),
            });
        }

        let presence_len = stored_meta_presence_len(span)?;
        let offset_count = span
            .checked_add(1)
            .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                field: "offset count",
            })?;
        let offsets_len = offset_count.checked_mul(std::mem::size_of::<u32>()).ok_or(
            StoredMetaCodecError::ArithmeticOverflow {
                field: "offset table length",
            },
        )?;
        let mut fields = Vec::new();
        fields
            .try_reserve_exact(expected_field_ords.len())
            .map_err(|_| StoredMetaCodecError::Allocation {
                resource: "field metadata",
                bytes: expected_field_ords
                    .len()
                    .saturating_mul(std::mem::size_of::<StoredMetaFieldMeta>()),
            })?;
        let mut canonical_field_offset = directory_len;
        for (index, &expected_field) in expected_field_ords.iter().enumerate() {
            let entry = index * STORED_META_DIRECTORY_ENTRY_LEN;
            let actual_field = u16::from_le_bytes([bytes[entry], bytes[entry + 1]]);
            if actual_field != expected_field {
                return Err(StoredMetaCodecError::UnexpectedField {
                    index,
                    expected: Some(expected_field),
                    actual: Some(actual_field),
                });
            }
            let encoded_offset = u32::from_le_bytes([
                bytes[entry + 2],
                bytes[entry + 3],
                bytes[entry + 4],
                bytes[entry + 5],
            ]);
            let expected_offset = u32::try_from(canonical_field_offset).map_err(|_| {
                StoredMetaCodecError::OffsetUnrepresentable {
                    field_ord: expected_field,
                    offset: canonical_field_offset,
                }
            })?;
            if encoded_offset != expected_offset {
                return Err(StoredMetaCodecError::NonCanonicalFieldOffset {
                    field_ord: expected_field,
                    encoded: encoded_offset,
                    expected: expected_offset,
                });
            }
            let next_field_offset = if index + 1 == expected_field_ords.len() {
                bytes.len()
            } else {
                let next_entry = entry + STORED_META_DIRECTORY_ENTRY_LEN;
                usize::try_from(u32::from_le_bytes([
                    bytes[next_entry + 2],
                    bytes[next_entry + 3],
                    bytes[next_entry + 4],
                    bytes[next_entry + 5],
                ]))
                .map_err(|_| StoredMetaCodecError::ResourceLimit {
                    resource: "host field offset",
                    actual: u64::from(u32::MAX),
                    limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
                })?
            };
            if next_field_offset > bytes.len() {
                return Err(StoredMetaCodecError::Truncated {
                    expected: next_field_offset,
                    actual: bytes.len(),
                });
            }
            let presence_end = canonical_field_offset.checked_add(presence_len).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "presence end",
                },
            )?;
            let offsets_end = presence_end.checked_add(offsets_len).ok_or(
                StoredMetaCodecError::ArithmeticOverflow {
                    field: "offset table end",
                },
            )?;
            if next_field_offset < offsets_end || bytes.len() < offsets_end {
                return Err(StoredMetaCodecError::Truncated {
                    expected: offsets_end,
                    actual: next_field_offset.min(bytes.len()),
                });
            }
            let presence = canonical_field_offset..presence_end;
            let offsets = presence_end..offsets_end;
            let blob = offsets_end..next_field_offset;
            if span % 8 != 0 {
                let used = span % 8;
                let allowed = u8::MAX >> (8 - used);
                let final_byte = bytes[presence.end - 1];
                if final_byte & !allowed != 0 {
                    return Err(StoredMetaCodecError::NonCanonicalPresencePadding {
                        field_ord: expected_field,
                        byte: final_byte,
                    });
                }
            }
            let blob_len = blob.len();
            let blob_len_u64 = u64::try_from(blob_len).unwrap_or(u64::MAX);
            if blob_len_u64 > limits.max_field_blob_bytes {
                return Err(StoredMetaCodecError::ResourceLimit {
                    resource: "field blob bytes",
                    actual: blob_len_u64,
                    limit: limits.max_field_blob_bytes,
                });
            }
            let offset_bytes = &bytes[offsets.clone()];
            let first_offset =
                stored_meta_offset(offset_bytes, 0).ok_or(StoredMetaCodecError::Truncated {
                    expected: offsets.start + std::mem::size_of::<u32>(),
                    actual: bytes.len(),
                })?;
            if first_offset != 0 {
                return Err(StoredMetaCodecError::NonZeroFirstOffset {
                    field_ord: expected_field,
                    actual: first_offset,
                });
            }
            for value_index in 0..span {
                let start = stored_meta_offset(offset_bytes, value_index).ok_or(
                    StoredMetaCodecError::Truncated {
                        expected: offsets.start + (value_index + 1) * 4,
                        actual: bytes.len(),
                    },
                )?;
                let end = stored_meta_offset(offset_bytes, value_index + 1).ok_or(
                    StoredMetaCodecError::Truncated {
                        expected: offsets.start + (value_index + 2) * 4,
                        actual: bytes.len(),
                    },
                )?;
                if end < start {
                    return Err(StoredMetaCodecError::DescendingOffsets {
                        field_ord: expected_field,
                        index: value_index,
                        start,
                        end,
                    });
                }
                if usize::try_from(end).map_or(true, |offset| offset > blob_len) {
                    return Err(StoredMetaCodecError::OffsetOutOfBounds {
                        field_ord: expected_field,
                        index: value_index,
                        offset: end,
                        blob_len,
                    });
                }
                let value_len = u64::from(end - start);
                if value_len > limits.max_value_bytes {
                    return Err(StoredMetaCodecError::ResourceLimit {
                        resource: "value bytes",
                        actual: value_len,
                        limit: limits.max_value_bytes,
                    });
                }
                let is_present = stored_meta_presence_bit(&bytes[presence.clone()], value_index)
                    .unwrap_or(false);
                if !is_present && start != end {
                    return Err(StoredMetaCodecError::AbsentValueHasBytes {
                        field_ord: expected_field,
                        index: value_index,
                        start,
                        end,
                    });
                }
            }
            let terminal =
                stored_meta_offset(offset_bytes, span).ok_or(StoredMetaCodecError::Truncated {
                    expected: offsets.end,
                    actual: bytes.len(),
                })?;
            if usize::try_from(terminal).ok() != Some(blob_len) {
                if index + 1 == expected_field_ords.len()
                    && usize::try_from(terminal).is_ok_and(|terminal| terminal < blob_len)
                {
                    return Err(StoredMetaCodecError::TrailingBytes {
                        expected: offsets_end + usize::try_from(terminal).unwrap_or(0),
                        actual: bytes.len(),
                    });
                }
                return Err(StoredMetaCodecError::TerminalOffsetMismatch {
                    field_ord: expected_field,
                    terminal,
                    blob_len,
                });
            }
            fields.push(StoredMetaFieldMeta {
                field_ord: expected_field,
                presence,
                offsets,
                blob,
            });
            canonical_field_offset = next_field_offset;
        }
        Ok(Self {
            bytes,
            docid_lo,
            docid_hi,
            fields,
        })
    }

    /// Exact durable bytes after canonical validation.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Inclusive lower global docid bound.
    #[must_use]
    pub const fn docid_lo(&self) -> u64 {
        self.docid_lo
    }

    /// Exclusive upper global docid bound.
    #[must_use]
    pub const fn docid_hi(&self) -> u64 {
        self.docid_hi
    }

    /// Half-open global docid span.
    #[must_use]
    pub const fn span(&self) -> u64 {
        self.docid_hi - self.docid_lo
    }

    /// Number of validated stored fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Bind one stored field for repeated zero-copy lookups.
    #[must_use]
    pub fn field(&self, field_ord: u16) -> Option<StoredMetaField<'a>> {
        let index = self
            .fields
            .binary_search_by_key(&field_ord, |field| field.field_ord)
            .ok()?;
        self.field_at(index)
    }

    /// Iterate borrowed field views in schema order.
    #[must_use]
    pub fn fields(&self) -> impl ExactSizeIterator<Item = StoredMetaField<'a>> + '_ {
        self.fields.iter().map(|field| StoredMetaField {
            field_ord: field.field_ord,
            docid_lo: self.docid_lo,
            presence: &self.bytes[field.presence.clone()],
            offsets: &self.bytes[field.offsets.clone()],
            blob: &self.bytes[field.blob.clone()],
        })
    }

    fn field_at(&self, index: usize) -> Option<StoredMetaField<'a>> {
        let field = self.fields.get(index)?;
        Some(StoredMetaField {
            field_ord: field.field_ord,
            docid_lo: self.docid_lo,
            presence: self.bytes.get(field.presence.clone())?,
            offsets: self.bytes.get(field.offsets.clone())?,
            blob: self.bytes.get(field.blob.clone())?,
        })
    }
}

fn validate_stored_meta_expected_fields(
    expected_field_ords: &[u16],
    max_fields: usize,
) -> Result<(), StoredMetaCodecError> {
    if expected_field_ords.len() > max_fields {
        return Err(StoredMetaCodecError::ResourceLimit {
            resource: "field count",
            actual: u64::try_from(expected_field_ords.len()).unwrap_or(u64::MAX),
            limit: u64::try_from(max_fields).unwrap_or(u64::MAX),
        });
    }
    for (index, pair) in expected_field_ords.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(StoredMetaCodecError::NonAscendingFields {
                index: index + 1,
                previous: pair[0],
                current: pair[1],
            });
        }
    }
    Ok(())
}

fn validate_stored_meta_field_set(
    expected_field_ords: &[u16],
    fields: &[StoredMetaFieldInput<'_>],
    max_fields: usize,
) -> Result<(), StoredMetaCodecError> {
    validate_stored_meta_expected_fields(expected_field_ords, max_fields)?;
    let compared = expected_field_ords.len().max(fields.len());
    for index in 0..compared {
        let expected = expected_field_ords.get(index).copied();
        let actual = fields.get(index).map(|field| field.field_ord);
        if expected != actual {
            return Err(StoredMetaCodecError::UnexpectedField {
                index,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn checked_stored_meta_span(
    docid_lo: u64,
    docid_hi: u64,
    limits: StoredMetaLimits,
) -> Result<usize, StoredMetaCodecError> {
    let span = docid_hi
        .checked_sub(docid_lo)
        .ok_or(StoredMetaCodecError::InvalidDocIdRange { docid_lo, docid_hi })?;
    if span > limits.max_docid_span {
        return Err(StoredMetaCodecError::ResourceLimit {
            resource: "docid span",
            actual: span,
            limit: limits.max_docid_span,
        });
    }
    usize::try_from(span).map_err(|_| StoredMetaCodecError::ResourceLimit {
        resource: "host docid span",
        actual: span,
        limit: u64::try_from(usize::MAX).unwrap_or(u64::MAX),
    })
}

fn stored_meta_presence_len(span: usize) -> Result<usize, StoredMetaCodecError> {
    span.checked_add(7)
        .map(|bits| bits / 8)
        .ok_or(StoredMetaCodecError::ArithmeticOverflow {
            field: "presence bitmap length",
        })
}

fn stored_meta_layout(
    field_ords: &[u16],
    span: usize,
    blob_lengths: &[usize],
    limits: StoredMetaLimits,
) -> Result<(Vec<usize>, usize), StoredMetaCodecError> {
    let directory_len = field_ords
        .len()
        .checked_mul(STORED_META_DIRECTORY_ENTRY_LEN)
        .ok_or(StoredMetaCodecError::ArithmeticOverflow {
            field: "directory length",
        })?;
    let presence_len = stored_meta_presence_len(span)?;
    let offsets_len = span
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
        .ok_or(StoredMetaCodecError::ArithmeticOverflow {
            field: "offset table length",
        })?;
    let field_prefix_len =
        presence_len
            .checked_add(offsets_len)
            .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                field: "field prefix length",
            })?;
    let mut field_offsets = Vec::new();
    field_offsets
        .try_reserve_exact(field_ords.len())
        .map_err(|_| StoredMetaCodecError::Allocation {
            resource: "directory offsets",
            bytes: field_ords
                .len()
                .saturating_mul(std::mem::size_of::<usize>()),
        })?;
    let mut cursor = directory_len;
    for (index, &field_ord) in field_ords.iter().enumerate() {
        if u32::try_from(cursor).is_err() {
            return Err(StoredMetaCodecError::OffsetUnrepresentable {
                field_ord,
                offset: cursor,
            });
        }
        field_offsets.push(cursor);
        let blob_len =
            *blob_lengths
                .get(index)
                .ok_or(StoredMetaCodecError::ArithmeticOverflow {
                    field: "field blob layout",
                })?;
        cursor = cursor
            .checked_add(field_prefix_len)
            .and_then(|end| end.checked_add(blob_len))
            .ok_or(StoredMetaCodecError::ArithmeticOverflow { field: "field end" })?;
    }
    let section_len = u64::try_from(cursor).unwrap_or(u64::MAX);
    if section_len > limits.max_section_bytes {
        return Err(StoredMetaCodecError::ResourceLimit {
            resource: "section bytes",
            actual: section_len,
            limit: limits.max_section_bytes,
        });
    }
    Ok((field_offsets, cursor))
}

fn stored_meta_presence_bit(presence: &[u8], ordinal: usize) -> Option<bool> {
    let byte = *presence.get(ordinal / 8)?;
    Some(byte & (1 << (ordinal % 8)) != 0)
}

fn stored_meta_offset(offsets: &[u8], index: usize) -> Option<u32> {
    let start = index.checked_mul(std::mem::size_of::<u32>())?;
    let end = start.checked_add(std::mem::size_of::<u32>())?;
    let bytes = offsets.get(start..end)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// One at-seal FSLX STATS row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FieldStats {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// BM25 token numerator captured at seal time.
    ///
    /// Fresh seals store the exact raw token count. A compaction replacement
    /// may instead sum decoded fieldnorms for the retained documents.
    pub total_tokens: u64,
    /// Segment-wide at-seal document count (Tantivy `max_doc` denominator).
    pub doc_count: u32,
}

impl FieldStats {
    /// Construct one schema-ordered at-seal statistics row.
    #[must_use]
    pub const fn new(field_ord: u16, total_tokens: u64, doc_count: u32) -> Self {
        Self {
            field_ord,
            total_tokens,
            doc_count,
        }
    }

    /// Raw average field length used by Tantivy BM25.
    ///
    /// This is deliberately not an average of decoded fieldnorm buckets.
    #[must_use]
    pub fn average_field_length(self) -> Option<f32> {
        (self.doc_count != 0).then(|| self.total_tokens as f32 / self.doc_count as f32)
    }
}

/// Explicit resource ceilings for an FSLX STATS section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StatsLimits {
    /// Maximum schema-derived Text/Keyword field count.
    pub max_fields: usize,
    /// Maximum complete packed section size.
    pub max_section_bytes: u64,
}

impl Default for StatsLimits {
    fn default() -> Self {
        Self {
            max_fields: 65_536,
            max_section_bytes: u64::from(u32::MAX),
        }
    }
}

/// Typed failures from STATS encoding, validation, or snapshot aggregation.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum StatsCodecError {
    /// A caller or durable declaration exceeded an explicit resource ceiling.
    #[error("STATS {resource} {actual} exceeds limit {limit}")]
    ResourceLimit {
        /// Bounded resource name.
        resource: &'static str,
        /// Rejected amount.
        actual: u64,
        /// Configured ceiling.
        limit: u64,
    },
    /// Schema field ordinals must be strictly ascending.
    #[error(
        "STATS field ordinals are not strictly ascending at index {index}: previous {previous}, current {current}"
    )]
    NonAscendingFields {
        /// Rejected field index.
        index: usize,
        /// Previous ordinal.
        previous: u16,
        /// Current ordinal.
        current: u16,
    },
    /// A row disagreed with the schema-derived field set.
    #[error("STATS field {index} mismatch: expected {expected:?}, got {actual:?}")]
    UnexpectedField {
        /// Field position in schema order.
        index: usize,
        /// Expected ordinal, if any.
        expected: Option<u16>,
        /// Supplied or decoded ordinal, if any.
        actual: Option<u16>,
    },
    /// Every indexed field uses the segment-wide at-seal denominator.
    #[error(
        "STATS field {field_ord} has doc_count {actual}, expected segment doc_count {expected}"
    )]
    DocCountMismatch {
        /// Schema field ordinal.
        field_ord: u16,
        /// Required segment at-seal document count.
        expected: u32,
        /// Supplied count.
        actual: u32,
    },
    /// A zero-document segment cannot contain indexed tokens.
    #[error("STATS field {field_ord} has {total_tokens} tokens with zero documents")]
    TokensWithoutDocuments {
        /// Schema field ordinal.
        field_ord: u16,
        /// Impossible raw token count.
        total_tokens: u64,
    },
    /// Packed STATS rows have one exact schema-derived length.
    #[error("STATS byte length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Required packed length.
        expected: usize,
        /// Supplied bytes.
        actual: usize,
    },
    /// Checked layout arithmetic overflowed.
    #[error("STATS arithmetic overflow while computing {field}")]
    ArithmeticOverflow {
        /// Layout or aggregate component.
        field: &'static str,
    },
    /// Snapshot segments did not carry the same schema-derived field count.
    #[error("STATS aggregate segment {segment_index} has {actual} fields, expected {expected}")]
    AggregateFieldCountMismatch {
        /// Zero-based segment ordinal in the aggregation input.
        segment_index: usize,
        /// Required row count.
        expected: usize,
        /// Supplied row count.
        actual: usize,
    },
    /// Snapshot segments did not carry the same field at one schema position.
    #[error(
        "STATS aggregate segment {segment_index} field {field_index} is {actual}, expected {expected}"
    )]
    AggregateFieldMismatch {
        /// Zero-based segment ordinal.
        segment_index: usize,
        /// Schema field position.
        field_index: usize,
        /// Required ordinal.
        expected: u16,
        /// Supplied ordinal.
        actual: u16,
    },
    /// A checked raw-token or document-count sum exceeded u64.
    #[error("STATS aggregate overflow for field {field_ord} {counter}")]
    AggregateOverflow {
        /// Schema field ordinal.
        field_ord: u16,
        /// Counter being summed.
        counter: &'static str,
    },
    /// Fallible allocation failed without panicking.
    #[error("unable to reserve {bytes} bytes for STATS {resource}")]
    Allocation {
        /// Allocation purpose.
        resource: &'static str,
        /// Requested amount.
        bytes: usize,
    },
}

/// Owned canonical bytes produced by the fresh-seal STATS writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedStatsSection {
    bytes: Vec<u8>,
    segment_doc_count: u32,
    field_count: usize,
}

impl EncodedStatsSection {
    /// Encode the schema-derived set of at-seal BM25 counters.
    ///
    /// # Errors
    ///
    /// Rejects missing/extra fields, a per-field denominator that differs from
    /// `segment_doc_count`, impossible zero-document token counts, resource
    /// abuse, arithmetic overflow, and allocation failure.
    pub fn encode(
        expected_field_ords: &[u16],
        rows: &[FieldStats],
        segment_doc_count: u32,
    ) -> Result<Self, StatsCodecError> {
        Self::encode_with_limits(
            expected_field_ords,
            rows,
            segment_doc_count,
            StatsLimits::default(),
        )
    }

    /// Encode with caller-selected resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::encode`].
    pub fn encode_with_limits(
        expected_field_ords: &[u16],
        rows: &[FieldStats],
        segment_doc_count: u32,
        limits: StatsLimits,
    ) -> Result<Self, StatsCodecError> {
        validate_stats_rows(expected_field_ords, rows, segment_doc_count, limits)?;
        let section_len = stats_section_len(expected_field_ords.len(), limits)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(section_len)
            .map_err(|_| StatsCodecError::Allocation {
                resource: "section bytes",
                bytes: section_len,
            })?;
        for row in rows {
            bytes.extend_from_slice(&row.field_ord.to_le_bytes());
            bytes.extend_from_slice(&row.total_tokens.to_le_bytes());
            bytes.extend_from_slice(&row.doc_count.to_le_bytes());
        }
        Ok(Self {
            bytes,
            segment_doc_count,
            field_count: rows.len(),
        })
    }

    /// Borrow the exact canonical durable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the wrapper and return its durable bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Number of schema-indexed text fields in this section.
    #[must_use]
    pub const fn field_count(&self) -> usize {
        self.field_count
    }

    /// Re-open the owned bytes through the validating reader.
    ///
    /// # Errors
    ///
    /// Returns an error if an internal invariant was violated.
    pub fn section(&self, expected_field_ords: &[u16]) -> Result<StatsSection, StatsCodecError> {
        StatsSection::parse(&self.bytes, expected_field_ords, self.segment_doc_count)
    }
}

/// Owned, fully validated FSLX STATS rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StatsSection {
    rows: Vec<FieldStats>,
    segment_doc_count: u32,
}

impl StatsSection {
    /// Validate a packed STATS payload against its schema and segment header.
    ///
    /// # Errors
    ///
    /// Rejects wrong byte length, missing/extra/reordered fields, denominator
    /// drift, impossible zero-document tokens, resource abuse, and allocation
    /// failure.
    pub fn parse(
        bytes: &[u8],
        expected_field_ords: &[u16],
        segment_doc_count: u32,
    ) -> Result<Self, StatsCodecError> {
        Self::parse_with_limits(
            bytes,
            expected_field_ords,
            segment_doc_count,
            StatsLimits::default(),
        )
    }

    /// Validate with caller-selected resource ceilings.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::parse`].
    pub fn parse_with_limits(
        bytes: &[u8],
        expected_field_ords: &[u16],
        segment_doc_count: u32,
        limits: StatsLimits,
    ) -> Result<Self, StatsCodecError> {
        validate_stats_expected_fields(expected_field_ords, limits.max_fields)?;
        let expected_len = stats_section_len(expected_field_ords.len(), limits)?;
        if bytes.len() != expected_len {
            return Err(StatsCodecError::LengthMismatch {
                expected: expected_len,
                actual: bytes.len(),
            });
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(expected_field_ords.len())
            .map_err(|_| StatsCodecError::Allocation {
                resource: "decoded rows",
                bytes: expected_field_ords
                    .len()
                    .saturating_mul(std::mem::size_of::<FieldStats>()),
            })?;
        for (index, &expected_field) in expected_field_ords.iter().enumerate() {
            let offset = index * STATS_ENTRY_LEN;
            let field_ord = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            if field_ord != expected_field {
                return Err(StatsCodecError::UnexpectedField {
                    index,
                    expected: Some(expected_field),
                    actual: Some(field_ord),
                });
            }
            let total_tokens = u64::from_le_bytes([
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
                bytes[offset + 8],
                bytes[offset + 9],
            ]);
            let doc_count = u32::from_le_bytes([
                bytes[offset + 10],
                bytes[offset + 11],
                bytes[offset + 12],
                bytes[offset + 13],
            ]);
            rows.push(FieldStats::new(field_ord, total_tokens, doc_count));
        }
        validate_stats_rows(expected_field_ords, &rows, segment_doc_count, limits)?;
        Ok(Self {
            rows,
            segment_doc_count,
        })
    }

    /// Schema-ordered at-seal counters.
    #[must_use]
    pub fn rows(&self) -> &[FieldStats] {
        &self.rows
    }

    /// Segment-wide at-seal document count used by every row.
    #[must_use]
    pub const fn segment_doc_count(&self) -> u32 {
        self.segment_doc_count
    }

    /// Resolve one field's at-seal counters.
    #[must_use]
    pub fn field(&self, field_ord: u16) -> Option<FieldStats> {
        self.rows
            .binary_search_by_key(&field_ord, |row| row.field_ord)
            .ok()
            .and_then(|index| self.rows.get(index))
            .copied()
    }
}

/// Snapshot-level checked sum for one schema field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotFieldStats {
    /// Stable schema field ordinal.
    pub field_ord: u16,
    /// Checked sum of the live segments' at-seal token numerators.
    pub total_tokens: u64,
    /// Checked sum of at-seal segment document counts.
    pub doc_count: u64,
}

impl SnapshotFieldStats {
    /// Raw Tantivy-compatible average field length.
    ///
    /// Returns `None` for an empty snapshot rather than producing NaN.
    #[must_use]
    pub fn average_field_length(self) -> Option<f32> {
        (self.doc_count != 0).then(|| self.total_tokens as f32 / self.doc_count as f32)
    }
}

/// Checked-sum identical validated STATS sections across live segments.
///
/// Tombstones do not alter these at-seal counters. Compaction removes the old
/// segment row and supplies a replacement whose token numerator may be
/// re-derived from decoded fieldnorms for retained documents.
///
/// # Errors
///
/// Returns a typed error for field-set drift, counter overflow, or allocation
/// failure.
pub fn aggregate_field_stats<'a>(
    segments: impl IntoIterator<Item = &'a StatsSection>,
) -> Result<Vec<SnapshotFieldStats>, StatsCodecError> {
    let mut segments = segments.into_iter();
    let Some(first) = segments.next() else {
        return Ok(Vec::new());
    };
    let first_rows = first.rows();
    let mut aggregate = Vec::new();
    aggregate
        .try_reserve_exact(first_rows.len())
        .map_err(|_| StatsCodecError::Allocation {
            resource: "aggregate rows",
            bytes: first_rows
                .len()
                .saturating_mul(std::mem::size_of::<SnapshotFieldStats>()),
        })?;
    for row in first_rows {
        aggregate.push(SnapshotFieldStats {
            field_ord: row.field_ord,
            total_tokens: row.total_tokens,
            doc_count: u64::from(row.doc_count),
        });
    }
    for (relative_segment_index, segment) in segments.enumerate() {
        let segment_index = relative_segment_index + 1;
        let rows = segment.rows();
        if rows.len() != aggregate.len() {
            return Err(StatsCodecError::AggregateFieldCountMismatch {
                segment_index,
                expected: aggregate.len(),
                actual: rows.len(),
            });
        }
        for (field_index, (total, row)) in aggregate.iter_mut().zip(rows).enumerate() {
            if row.field_ord != total.field_ord {
                return Err(StatsCodecError::AggregateFieldMismatch {
                    segment_index,
                    field_index,
                    expected: total.field_ord,
                    actual: row.field_ord,
                });
            }
            total.total_tokens = total.total_tokens.checked_add(row.total_tokens).ok_or(
                StatsCodecError::AggregateOverflow {
                    field_ord: row.field_ord,
                    counter: "total_tokens",
                },
            )?;
            total.doc_count = total
                .doc_count
                .checked_add(u64::from(row.doc_count))
                .ok_or(StatsCodecError::AggregateOverflow {
                    field_ord: row.field_ord,
                    counter: "doc_count",
                })?;
        }
    }
    Ok(aggregate)
}

fn validate_stats_expected_fields(
    expected_field_ords: &[u16],
    max_fields: usize,
) -> Result<(), StatsCodecError> {
    if expected_field_ords.len() > max_fields {
        return Err(StatsCodecError::ResourceLimit {
            resource: "field count",
            actual: u64::try_from(expected_field_ords.len()).unwrap_or(u64::MAX),
            limit: u64::try_from(max_fields).unwrap_or(u64::MAX),
        });
    }
    for (index, pair) in expected_field_ords.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(StatsCodecError::NonAscendingFields {
                index: index + 1,
                previous: pair[0],
                current: pair[1],
            });
        }
    }
    Ok(())
}

fn validate_stats_rows(
    expected_field_ords: &[u16],
    rows: &[FieldStats],
    segment_doc_count: u32,
    limits: StatsLimits,
) -> Result<(), StatsCodecError> {
    validate_stats_expected_fields(expected_field_ords, limits.max_fields)?;
    let compared = expected_field_ords.len().max(rows.len());
    for index in 0..compared {
        let expected = expected_field_ords.get(index).copied();
        let actual = rows.get(index).map(|row| row.field_ord);
        if expected != actual {
            return Err(StatsCodecError::UnexpectedField {
                index,
                expected,
                actual,
            });
        }
    }
    for row in rows {
        if row.doc_count != segment_doc_count {
            return Err(StatsCodecError::DocCountMismatch {
                field_ord: row.field_ord,
                expected: segment_doc_count,
                actual: row.doc_count,
            });
        }
        if segment_doc_count == 0 && row.total_tokens != 0 {
            return Err(StatsCodecError::TokensWithoutDocuments {
                field_ord: row.field_ord,
                total_tokens: row.total_tokens,
            });
        }
    }
    Ok(())
}

fn stats_section_len(field_count: usize, limits: StatsLimits) -> Result<usize, StatsCodecError> {
    if field_count > limits.max_fields {
        return Err(StatsCodecError::ResourceLimit {
            resource: "field count",
            actual: u64::try_from(field_count).unwrap_or(u64::MAX),
            limit: u64::try_from(limits.max_fields).unwrap_or(u64::MAX),
        });
    }
    let section_len =
        field_count
            .checked_mul(STATS_ENTRY_LEN)
            .ok_or(StatsCodecError::ArithmeticOverflow {
                field: "section length",
            })?;
    let section_len_u64 = u64::try_from(section_len).unwrap_or(u64::MAX);
    if section_len_u64 > limits.max_section_bytes {
        return Err(StatsCodecError::ResourceLimit {
            resource: "section bytes",
            actual: section_len_u64,
            limit: limits.max_section_bytes,
        });
    }
    Ok(section_len)
}

/// Feature-gated scalar/SIMD oracle surface for same-binary benches and the
/// Quill gauntlet's internal differential suite.
#[cfg(feature = "bench-internals")]
pub mod differential {
    pub use super::bitpack::BitpackError;

    /// Stable fixture identifier reserved for gauntlet registration.
    pub const FIXTURE_ID: &str = "quiver-postings-bitpack-scalar-wide-v1";
    /// Normative source exercised by this differential.
    pub const SPEC_SECTION: &str = "FSLX v1 section 5.2 LSB-first bitpacked payloads";

    /// Pack values with the canonical scalar encoder.
    ///
    /// # Errors
    ///
    /// Returns a typed error for widths above 32, length overflow, or a value
    /// that cannot be represented at `width`.
    pub fn pack_values(values: &[u32], width: u8) -> Result<Vec<u8>, BitpackError> {
        super::bitpack::pack_lsb(values, width)
    }

    /// Decode into `output` with the scalar reference implementation.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid width, non-canonical padding, or a
    /// payload whose length does not exactly match `(output.len(), width)`.
    pub fn unpack_scalar_into(
        input: &[u8],
        width: u8,
        output: &mut [u32],
    ) -> Result<(), BitpackError> {
        super::bitpack::unpack_scalar_into(input, width, output)
    }

    /// Decode into `output` with the portable `wide::u32x8` implementation.
    ///
    /// # Errors
    ///
    /// Returns a typed error for an invalid width, non-canonical padding, or a
    /// payload whose length does not exactly match `(output.len(), width)`.
    pub fn unpack_wide_into(
        input: &[u8],
        width: u8,
        output: &mut [u32],
    ) -> Result<(), BitpackError> {
        super::bitpack::unpack_wide_into(input, width, output)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::bitpack::{BitpackError, pack_lsb, unpack_scalar_into, unpack_wide_into};
    use super::*;

    type TestResult = Result<(), Box<dyn Error>>;

    #[allow(clippy::cast_possible_truncation)]
    fn random_u32(state: &mut u64) -> u32 {
        let mut value = *state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        *state = value;
        (value.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
    }

    fn values(width: u8, count: usize, seed: u64) -> Vec<u32> {
        if width == 0 {
            return vec![0; count];
        }
        let mask = u32::MAX >> (u32::BITS - u32::from(width));
        let mut state = seed;
        let mut output: Vec<u32> = (0..count).map(|_| random_u32(&mut state) & mask).collect();
        if let Some(last) = output.last_mut() {
            *last = mask;
        }
        output
    }

    #[allow(clippy::cast_possible_truncation)]
    fn dense_postings(count: usize, start: u32) -> Vec<Posting> {
        (0..count)
            .map(|index| Posting::new(start + index as u32, (index % 7 + 1) as u32))
            .collect()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn sparse_postings(count: usize, start: u32) -> Vec<Posting> {
        (0..count)
            .map(|index| Posting::new(start + index as u32 * 11, (index % 7 + 1) as u32))
            .collect()
    }

    fn postings_with_span(span: u32) -> Vec<Posting> {
        let mut postings: Vec<Posting> = (0_u32..127)
            .map(|index| Posting::new(1_000 + index, index % 5 + 1))
            .collect();
        postings.push(Posting::new(1_000 + span - 1, 9));
        postings
    }

    #[allow(clippy::cast_possible_truncation, clippy::unnecessary_wraps)]
    fn fixture_fieldnorm(doc_id: u32) -> Option<u8> {
        Some((doc_id % 251) as u8)
    }

    fn raw_block(kind: u8, count: u8, payload: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
        let payload_len = u16::try_from(payload.len())?;
        let mut bytes = vec![kind, count];
        bytes.extend_from_slice(&payload_len.to_le_bytes());
        bytes.extend_from_slice(payload);
        Ok(bytes)
    }

    fn forced_for(postings: &[Posting]) -> Result<Vec<u8>, PostingCodecError> {
        let mut bytes = Vec::new();
        encode_for_block(postings, &mut bytes)?;
        Ok(bytes)
    }

    fn raw_positions(directory: &[(u32, u64)], payload: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
        let block_count = u32::try_from(directory.len())?;
        let mut bytes = block_count.to_le_bytes().to_vec();
        for &(first_posting_ordinal, block_offset) in directory {
            write_vint(first_posting_ordinal, &mut bytes);
            write_vint64(block_offset, &mut bytes);
        }
        bytes.extend_from_slice(payload);
        Ok(bytes)
    }

    fn collect_positions(
        positions: &PositionList<'_>,
        posting_ordinal: u32,
    ) -> Result<Vec<u32>, PositionCodecError> {
        positions.positions_for_ordinal(posting_ordinal)?.collect()
    }

    fn zero_positions(postings: &[Posting]) -> Result<Vec<u32>, PositionCodecError> {
        let count = postings.iter().try_fold(0_usize, |total, posting| {
            let freq = usize::try_from(posting.freq).map_err(|_| {
                PositionCodecError::ArithmeticOverflow {
                    field: "test position count",
                }
            })?;
            total
                .checked_add(freq)
                .ok_or(PositionCodecError::ArithmeticOverflow {
                    field: "test position count",
                })
        })?;
        Ok(vec![0; count])
    }

    #[test]
    fn independent_bitpack_known_answers() -> TestResult {
        assert_eq!(pack_lsb(&[1, 2, 5], 3)?, [0x51, 0x01]);
        assert_eq!(pack_lsb(&[1, 2, 3], 2)?, [0x39]);
        assert_eq!(
            pack_lsb(&[0x7856_3412, 0xdead_beef], 32)?,
            [0x12, 0x34, 0x56, 0x78, 0xef, 0xbe, 0xad, 0xde]
        );

        for (packed, width, expected) in [
            (&[0x51, 0x01][..], 3, &[1, 2, 5][..]),
            (&[0x39][..], 2, &[1, 2, 3][..]),
        ] {
            let mut scalar = vec![0; expected.len()];
            let mut wide = vec![0; expected.len()];
            unpack_scalar_into(packed, width, &mut scalar)?;
            unpack_wide_into(packed, width, &mut wide)?;
            assert_eq!(scalar, expected);
            assert_eq!(wide, expected);
        }
        Ok(())
    }

    #[test]
    fn scalar_and_wide_match_every_width_and_shape() -> TestResult {
        let mut counts: Vec<usize> = (0..=16).collect();
        counts.extend_from_slice(&[120, 127, 128]);
        for width in 0..=32 {
            for &count in &counts {
                let expected = values(width, count, 0x0123_4567_89ab_cdef ^ count as u64);
                let packed = pack_lsb(&expected, width)?;
                let mut scalar = vec![u32::MAX; count];
                let mut wide = vec![u32::MAX; count];
                unpack_scalar_into(&packed, width, &mut scalar)?;
                unpack_wide_into(&packed, width, &mut wide)?;
                assert_eq!(scalar, expected, "scalar width={width} count={count}");
                assert_eq!(wide, expected, "wide width={width} count={count}");
            }
        }
        Ok(())
    }

    #[test]
    fn wide_decode_is_independent_of_slice_alignment() -> TestResult {
        for offset in 0..=31 {
            for width in 1..=32 {
                let expected = values(width, 127, 0xa5a5_5a5a_dead_beef ^ offset as u64);
                let packed = pack_lsb(&expected, width)?;
                let mut storage = vec![0_u8; offset];
                storage.extend_from_slice(&packed);
                storage.extend_from_slice(&[0_u8; 7]);
                let input = &storage[offset..offset + packed.len()];
                let mut actual = vec![0_u32; expected.len()];
                unpack_wide_into(input, width, &mut actual)?;
                assert_eq!(actual, expected, "offset={offset} width={width}");
            }
        }
        Ok(())
    }

    #[test]
    fn bitpack_rejects_truncation_overlong_payloads_bad_widths_and_padding() -> TestResult {
        for width in 1..=32 {
            let expected = values(width, 127, 0xfeed_face_cafe_babe);
            let packed = pack_lsb(&expected, width)?;
            for cut in 0..packed.len() {
                let mut scalar = vec![0_u32; expected.len()];
                let mut wide = vec![0_u32; expected.len()];
                assert!(matches!(
                    unpack_scalar_into(&packed[..cut], width, &mut scalar),
                    Err(BitpackError::LengthMismatch { .. })
                ));
                assert!(matches!(
                    unpack_wide_into(&packed[..cut], width, &mut wide),
                    Err(BitpackError::LengthMismatch { .. })
                ));
            }
        }

        let mut output = [0_u32; 1];
        assert!(matches!(
            unpack_scalar_into(&[], 33, &mut output),
            Err(BitpackError::InvalidWidth { width: 33 })
        ));
        assert!(matches!(
            unpack_wide_into(&[], 33, &mut output),
            Err(BitpackError::InvalidWidth { width: 33 })
        ));
        assert!(matches!(
            unpack_scalar_into(&[0], 0, &mut output),
            Err(BitpackError::LengthMismatch { .. })
        ));
        assert!(matches!(
            pack_lsb(&[1], 0),
            Err(BitpackError::ValueOutOfRange { .. })
        ));
        assert!(matches!(
            pack_lsb(&[8], 3),
            Err(BitpackError::ValueOutOfRange { .. })
        ));

        for (width, count) in [(1, 1), (3, 3), (5, 7), (7, 9)] {
            let expected = values(width, count, 0x55aa_aa55);
            let mut noncanonical = pack_lsb(&expected, width)?;
            let used_bits = usize::from(width) * count % 8;
            assert_ne!(used_bits, 0);
            if let Some(last) = noncanonical.last_mut() {
                *last |= 1_u8 << used_bits;
            }
            let mut scalar = vec![0_u32; count];
            let mut wide = vec![0_u32; count];
            assert!(matches!(
                unpack_scalar_into(&noncanonical, width, &mut scalar),
                Err(BitpackError::NonCanonicalPadding { .. })
            ));
            assert!(matches!(
                unpack_wide_into(&noncanonical, width, &mut wide),
                Err(BitpackError::NonCanonicalPadding { .. })
            ));
        }

        let mut overlong = pack_lsb(&[5], 3)?;
        overlong.push(0);
        assert!(matches!(
            unpack_wide_into(&overlong, 3, &mut output),
            Err(BitpackError::LengthMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn posting_roundtrip_covers_empty_partial_full_and_multiblock_shapes() -> TestResult {
        for count in [0, 1, 2, 127, 128, 129, 255, 256, 257, 400] {
            let expected = sparse_postings(count, 100);
            let encoded = EncodedPostingList::encode(&expected)?;
            let repeated = EncodedPostingList::encode(&expected)?;
            assert_eq!(encoded, repeated, "determinism count={count}");
            let parsed = encoded.posting_list()?;
            assert_eq!(parsed.doc_freq(), u32::try_from(count)?);
            assert_eq!(parsed.decode_all()?, expected, "roundtrip count={count}");
            assert_eq!(
                parsed.block_count(),
                count.div_ceil(POSTINGS_PER_BLOCK),
                "block count={count}"
            );
        }
        Ok(())
    }

    #[test]
    fn owned_lists_reopen_with_shape_derived_limits() -> TestResult {
        let posting_count = DEFAULT_MAX_POSTINGS_PER_TERM + 1;
        let block_count = posting_count.div_ceil(POSTINGS_PER_BLOCK);
        let limits = EncodedPostingList::validation_limits(posting_count, block_count);
        assert_eq!(limits.max_postings, posting_count);
        assert_eq!(limits.max_blocks, block_count);

        // Avoid allocating a multi-million-posting fixture: an empty synthetic
        // owned wrapper must get past the hostile-input cap and fail on its
        // actual byte/count mismatch instead.
        let owned = EncodedPostingList {
            bytes: Vec::new(),
            doc_freq: u32::try_from(posting_count)?,
            block_count,
        };
        let expected_doc_freq = u32::try_from(posting_count)?;
        assert!(matches!(
            owned.posting_list(),
            Err(PostingCodecError::DocFrequencyMismatch {
                expected,
                actual: 0
            }) if expected == expected_doc_freq
        ));
        Ok(())
    }

    #[test]
    fn randomized_posting_roundtrip_and_cursor_match_linear_oracle() -> TestResult {
        let boundary_counts = [0, 1, 2, 126, 127, 128, 129, 255, 256, 257, 511, 512];
        for case in 0_u64..48 {
            let mut state = 0x6a09_e667_f3bc_c909 ^ case;
            let count = if case < u64::try_from(boundary_counts.len())? {
                boundary_counts[usize::try_from(case)?]
            } else {
                usize::try_from(random_u32(&mut state) % 513)?
            };
            let mut doc_id = random_u32(&mut state) % 100;
            let mut expected = Vec::with_capacity(count);
            for ordinal in 0..count {
                if ordinal != 0 {
                    let random = random_u32(&mut state);
                    let step = if ordinal % 37 == 0 {
                        512 + random % 4_096
                    } else {
                        1 + random % 7
                    };
                    doc_id = doc_id.checked_add(step).ok_or("fixture docid overflow")?;
                }
                let random = random_u32(&mut state);
                let freq = match ordinal % 41 {
                    0 => u32::MAX,
                    1 => 256,
                    _ => 1 + random % 31,
                };
                expected.push(Posting::new(doc_id, freq));
            }

            let encoded = EncodedPostingList::encode(&expected)?;
            assert_eq!(encoded, EncodedPostingList::encode(&expected)?);
            let list = encoded.posting_list()?;
            assert_eq!(list.decode_all_bounded(count)?, expected, "case={case}");

            let mut cursor = list.cursor()?;
            for (ordinal, posting) in expected.iter().copied().enumerate() {
                assert_eq!(cursor.current(), Some(posting), "case={case}");
                assert_eq!(cursor.posting_ordinal(), Some(u32::try_from(ordinal)?));
                cursor.next()?;
            }
            assert_eq!(cursor.current(), None);

            for _ in 0..8 {
                let target = random_u32(&mut state) % doc_id.saturating_add(2);
                let oracle = expected
                    .iter()
                    .copied()
                    .find(|posting| posting.doc_id >= target);
                assert_eq!(list.cursor()?.advance(target)?, oracle, "case={case}");
            }
        }
        Ok(())
    }

    #[test]
    fn writer_selects_bitmap_at_511_and_for_at_512() -> TestResult {
        let bitmap_input = postings_with_span(511);
        let bitmap = EncodedPostingList::encode(&bitmap_input)?;
        let bitmap_list = bitmap.posting_list()?;
        assert_eq!(bitmap_list.blocks()[0].kind, PostingBlockKind::Bitmap);
        assert_eq!(bitmap_list.decode_all()?, bitmap_input);

        let for_input = postings_with_span(512);
        let frame = EncodedPostingList::encode(&for_input)?;
        let frame_list = frame.posting_list()?;
        assert_eq!(
            frame_list.blocks()[0].kind,
            PostingBlockKind::FrameOfReference
        );
        assert_eq!(frame_list.decode_all()?, for_input);

        let mut invalid_bitmap = bitmap.as_bytes().to_vec();
        invalid_bitmap[4 + 4 + 2 + 63] |= 0x80;
        assert!(matches!(
            PostingList::parse(&invalid_bitmap, u32::from(POSTINGS_PER_BLOCK_U8)),
            Err(PostingCodecError::InvalidBitmap { .. })
        ));
        Ok(())
    }

    #[test]
    fn reader_accepts_width_zero_for_and_width_32_extremes() -> TestResult {
        let consecutive: Vec<Posting> = dense_postings(POSTINGS_PER_BLOCK, 7)
            .into_iter()
            .map(|posting| Posting::new(posting.doc_id, 1))
            .collect();
        let width_zero = forced_for(&consecutive)?;
        assert_eq!(
            width_zero,
            [FOR_KIND, POSTINGS_PER_BLOCK_U8, 6, 0, 7, 0, 0, 0, 0, 0]
        );
        let parsed = PostingList::parse(&width_zero, u32::from(POSTINGS_PER_BLOCK_U8))?;
        assert_eq!(parsed.decode_all()?, consecutive);

        let mut extremes = Vec::with_capacity(POSTINGS_PER_BLOCK);
        extremes.push(Posting::new(0, 1));
        extremes.push(Posting::new(0x8000_0001, u32::MAX));
        for doc_id in 0x8000_0002..=0x8000_007f {
            extremes.push(Posting::new(doc_id, 1));
        }
        assert_eq!(extremes.len(), POSTINGS_PER_BLOCK);
        let encoded = EncodedPostingList::encode(&extremes)?;
        assert_eq!(encoded.as_bytes()[8], 32, "doc width");
        assert_eq!(encoded.as_bytes()[9], FREQ_BITPACKED);
        assert_eq!(encoded.as_bytes()[10], 32, "frequency width");
        assert_eq!(encoded.posting_list()?.decode_all()?, extremes);
        Ok(())
    }

    #[test]
    fn vint_wire_goldens_and_canonicality() -> TestResult {
        assert_eq!(
            EncodedPostingList::encode(&[Posting::new(0, 127)])?.as_bytes(),
            [VINT_KIND, 1, 2, 0, 0, 127]
        );
        assert_eq!(
            EncodedPostingList::encode(&[Posting::new(128, 1)])?.as_bytes(),
            [VINT_KIND, 1, 3, 0, 0x80, 0x01, 0x01]
        );
        assert_eq!(
            EncodedPostingList::encode(&[Posting::new(u32::MAX, 1)])?.as_bytes(),
            [VINT_KIND, 1, 6, 0, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x01]
        );

        let overlong = raw_block(VINT_KIND, 1, &[0x80, 0x00, 0x01])?;
        assert!(matches!(
            PostingList::parse(&overlong, 1),
            Err(PostingCodecError::NonCanonicalVint { .. })
        ));
        let overflow = raw_block(VINT_KIND, 1, &[0xff, 0xff, 0xff, 0xff, 0x10, 0x01])?;
        assert!(matches!(
            PostingList::parse(&overflow, 1),
            Err(PostingCodecError::VintOverflow { .. })
        ));
        let zero_frequency = raw_block(VINT_KIND, 1, &[0x01, 0x00])?;
        assert!(matches!(
            PostingList::parse(&zero_frequency, 1),
            Err(PostingCodecError::ZeroFrequency { .. })
        ));

        let first_block = forced_for(&dense_postings(POSTINGS_PER_BLOCK, 7))?;
        let mut interior_zero_frequency = first_block;
        interior_zero_frequency.extend_from_slice(&raw_block(VINT_KIND, 1, &[0xc8, 0x01, 0x00])?);
        assert!(matches!(
            PostingList::parse(&interior_zero_frequency, 129),
            Err(PostingCodecError::ZeroFrequency { index: 128, .. })
        ));
        Ok(())
    }

    #[test]
    fn encoder_rejects_zero_frequency_duplicates_and_descending_docids() {
        assert!(matches!(
            EncodedPostingList::encode(&[Posting::new(1, 0)]),
            Err(PostingCodecError::ZeroFrequency { .. })
        ));
        assert!(matches!(
            EncodedPostingList::encode(&[Posting::new(1, 1), Posting::new(1, 2)]),
            Err(PostingCodecError::NonAscendingInput { .. })
        ));
        assert!(matches!(
            EncodedPostingList::encode(&[Posting::new(2, 1), Posting::new(1, 1)]),
            Err(PostingCodecError::NonAscendingInput { .. })
        ));
    }

    #[test]
    fn parser_rejects_header_payload_width_and_docfreq_corruption() -> TestResult {
        let expected = sparse_postings(400, 100);
        let encoded = EncodedPostingList::encode(&expected)?;
        for cut in 0..encoded.as_bytes().len() {
            assert!(
                PostingList::parse(&encoded.as_bytes()[..cut], encoded.doc_freq()).is_err(),
                "cut={cut}"
            );
        }

        let mut invalid_kind = encoded.as_bytes().to_vec();
        invalid_kind[0] = 9;
        assert!(matches!(
            PostingList::parse(&invalid_kind, encoded.doc_freq()),
            Err(PostingCodecError::InvalidBlockKind { .. })
        ));
        let mut invalid_count = encoded.as_bytes().to_vec();
        invalid_count[1] = 0;
        assert!(matches!(
            PostingList::parse(&invalid_count, encoded.doc_freq()),
            Err(PostingCodecError::InvalidBlockCount { .. })
        ));
        let mut declared_too_long = encoded.as_bytes().to_vec();
        let declared = u16::from_le_bytes([declared_too_long[2], declared_too_long[3]]);
        let longer = declared.saturating_add(1).to_le_bytes();
        declared_too_long[2..4].copy_from_slice(&longer);
        assert!(PostingList::parse(&declared_too_long, encoded.doc_freq()).is_err());
        assert!(matches!(
            PostingList::parse(encoded.as_bytes(), encoded.doc_freq() - 1),
            Err(PostingCodecError::DocFrequencyExceeded { .. })
        ));
        assert!(matches!(
            PostingList::parse(encoded.as_bytes(), encoded.doc_freq() + 1),
            Err(PostingCodecError::DocFrequencyMismatch { .. })
        ));

        let mut nonminimal_doc_payload = vec![0_u8; 4 + 1 + 1 + 16];
        nonminimal_doc_payload[..4].copy_from_slice(&7_u32.to_le_bytes());
        nonminimal_doc_payload[4] = 1;
        nonminimal_doc_payload[5] = FREQ_ALL_ONE;
        let nonminimal_doc = raw_block(FOR_KIND, POSTINGS_PER_BLOCK_U8, &nonminimal_doc_payload)?;
        assert!(matches!(
            PostingList::parse(&nonminimal_doc, u32::from(POSTINGS_PER_BLOCK_U8)),
            Err(PostingCodecError::NonCanonicalWidth { .. })
        ));

        let mut noncanonical_freq_payload = Vec::new();
        noncanonical_freq_payload.extend_from_slice(&7_u32.to_le_bytes());
        noncanonical_freq_payload.extend_from_slice(&[0, FREQ_BITPACKED, 0]);
        let noncanonical_freq =
            raw_block(FOR_KIND, POSTINGS_PER_BLOCK_U8, &noncanonical_freq_payload)?;
        assert!(matches!(
            PostingList::parse(&noncanonical_freq, u32::from(POSTINGS_PER_BLOCK_U8)),
            Err(PostingCodecError::NonCanonicalFrequencyEncoding { .. })
        ));
        Ok(())
    }

    #[test]
    fn parser_and_materializer_enforce_resource_budgets_early() -> TestResult {
        let first = forced_for(&dense_postings(POSTINGS_PER_BLOCK, 7))?;
        let mut malformed_tail = first.clone();
        malformed_tail.extend_from_slice(&[9, 0, 0, 0]);
        assert!(matches!(
            PostingList::parse(&malformed_tail, 0),
            Err(PostingCodecError::DocFrequencyExceeded {
                expected: 0,
                actual: 128
            })
        ));

        let second = forced_for(&dense_postings(POSTINGS_PER_BLOCK, 1_000))?;
        let first_len = first.len();
        let mut two_blocks = first;
        two_blocks.extend_from_slice(&second);
        let one_block_limit = PostingListLimits {
            max_blocks: 1,
            max_postings: 256,
        };
        assert!(matches!(
            PostingList::parse_with_limits(&two_blocks, 256, one_block_limit),
            Err(PostingCodecError::BlockBudgetExhausted {
                limit: 1,
                validated: 1
            })
        ));
        assert!(
            PostingList::parse_with_limits(
                &two_blocks[..first_len],
                128,
                PostingListLimits {
                    max_blocks: 1,
                    max_postings: 128,
                }
            )
            .is_ok()
        );
        let mut cap_plus_tail = two_blocks[..first_len].to_vec();
        cap_plus_tail.push(0xff);
        assert!(matches!(
            PostingList::parse_with_limits(&cap_plus_tail, 128, one_block_limit),
            Err(PostingCodecError::BlockBudgetExhausted {
                limit: 1,
                validated: 1
            })
        ));
        assert!(matches!(
            PostingList::parse(&[], u32::try_from(DEFAULT_MAX_POSTINGS_PER_TERM + 1)?),
            Err(PostingCodecError::PostingLimitExceeded { .. })
        ));
        let list = PostingList::parse(&two_blocks, 256)?;
        assert!(list.block_bytes(usize::MAX).is_none());
        assert!(matches!(
            list.decode_all_bounded(255),
            Err(PostingCodecError::MaterializationLimitExceeded {
                limit: 255,
                actual: 256
            })
        ));
        Ok(())
    }

    #[test]
    fn structured_arithmetic_and_cross_block_corruption_is_rejected() -> TestResult {
        let mut for_doc_overflow_payload = u32::MAX.to_le_bytes().to_vec();
        for_doc_overflow_payload.extend_from_slice(&[0, FREQ_ALL_ONE]);
        let for_doc_overflow =
            raw_block(FOR_KIND, POSTINGS_PER_BLOCK_U8, &for_doc_overflow_payload)?;
        assert!(matches!(
            PostingList::parse(&for_doc_overflow, 128),
            Err(PostingCodecError::ArithmeticOverflow { field: "docid", .. })
        ));

        let mut bitmap_overflow_payload = (u32::MAX - 100).to_le_bytes().to_vec();
        bitmap_overflow_payload.extend_from_slice(&128_u16.to_le_bytes());
        bitmap_overflow_payload.extend_from_slice(&[u8::MAX; 16]);
        bitmap_overflow_payload.extend_from_slice(&[0_u8; 48]);
        bitmap_overflow_payload.push(FREQ_ALL_ONE);
        let bitmap_overflow =
            raw_block(BITMAP_KIND, POSTINGS_PER_BLOCK_U8, &bitmap_overflow_payload)?;
        assert!(matches!(
            PostingList::parse(&bitmap_overflow, 128),
            Err(PostingCodecError::ArithmeticOverflow {
                field: "bitmap docid",
                ..
            })
        ));

        let vint_doc_overflow = raw_block(VINT_KIND, 2, &[0xff, 0xff, 0xff, 0xff, 0x0f, 1, 0, 1])?;
        assert!(matches!(
            PostingList::parse(&vint_doc_overflow, 2),
            Err(PostingCodecError::ArithmeticOverflow {
                field: "vint docid",
                ..
            })
        ));

        let mut stored_freqs = [0_u32; POSTINGS_PER_BLOCK];
        stored_freqs[0] = u32::MAX;
        let mut freq_overflow_payload = 0_u32.to_le_bytes().to_vec();
        freq_overflow_payload.extend_from_slice(&[0, FREQ_BITPACKED, 32]);
        freq_overflow_payload.extend_from_slice(&pack_lsb(&stored_freqs, 32)?);
        let freq_overflow = raw_block(FOR_KIND, POSTINGS_PER_BLOCK_U8, &freq_overflow_payload)?;
        assert!(matches!(
            PostingList::parse(&freq_overflow, 128),
            Err(PostingCodecError::ArithmeticOverflow {
                field: "frequency",
                ..
            })
        ));

        let one = raw_block(VINT_KIND, 1, &[1, 1])?;
        let mut equal_seam = one.clone();
        equal_seam.extend_from_slice(&one);
        assert!(matches!(
            PostingList::parse(&equal_seam, 2),
            Err(PostingCodecError::NonAscendingDecoded {
                posting_ordinal: 1,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn cursor_starts_on_first_advances_over_holes_and_fuses_at_u32_max() -> TestResult {
        let expected = vec![
            Posting::new(0, 1),
            Posting::new(2, 2),
            Posting::new(100, 3),
            Posting::new(u32::MAX, 4),
        ];
        let encoded = EncodedPostingList::encode(&expected)?;
        let list = encoded.posting_list()?;
        let mut cursor = list.cursor()?;
        assert_eq!(cursor.current(), Some(expected[0]));
        assert_eq!(cursor.posting_ordinal(), Some(0));
        assert_eq!(cursor.advance(0)?, Some(expected[0]));
        assert_eq!(cursor.advance(1)?, Some(expected[1]));
        assert_eq!(cursor.advance(2)?, Some(expected[1]));
        assert_eq!(cursor.advance(3)?, Some(expected[2]));
        assert_eq!(cursor.advance(u32::MAX)?, Some(expected[3]));
        assert_eq!(cursor.doc(), Some(u32::MAX));
        assert_eq!(cursor.freq(), Some(4));
        assert_eq!(cursor.next()?, None);
        assert_eq!(cursor.next()?, None);
        assert_eq!(cursor.advance(0)?, None);
        Ok(())
    }

    #[test]
    fn advance_matches_linear_scan_across_block_boundaries() -> TestResult {
        let expected = sparse_postings(400, 100);
        let encoded = EncodedPostingList::encode(&expected)?;
        let list = encoded.posting_list()?;
        let last_doc = expected.last().map_or(0, |posting| posting.doc_id);
        for target in (0..=last_doc.saturating_add(1)).step_by(13) {
            let linear = expected
                .iter()
                .copied()
                .find(|posting| posting.doc_id >= target);
            let mut cursor = list.cursor()?;
            assert_eq!(cursor.advance(target)?, linear, "target={target}");
            if let Some(posting) = linear {
                let expected_ordinal = expected
                    .iter()
                    .position(|candidate| *candidate == posting)
                    .and_then(|ordinal| u32::try_from(ordinal).ok());
                assert_eq!(cursor.posting_ordinal(), expected_ordinal);
            }
        }
        Ok(())
    }

    #[test]
    fn q1_df_100_plus_300_preserves_blocks_and_decoded_behavior() -> TestResult {
        let left_postings = sparse_postings(100, 100);
        let right_postings = sparse_postings(300, 10_000);
        let left = EncodedPostingList::encode(&left_postings)?;
        let right = EncodedPostingList::encode(&right_postings)?;
        let left_list = left.posting_list()?;
        let right_list = right.posting_list()?;

        let mut concatenated = left.as_bytes().to_vec();
        concatenated.extend_from_slice(right.as_bytes());
        let merged = PostingList::parse(&concatenated, 400)?;
        assert_eq!(
            merged
                .blocks()
                .iter()
                .map(|block| block.posting_count)
                .collect::<Vec<_>>(),
            [100, 128, 128, 44]
        );

        let mut expected_raw =
            Vec::with_capacity(left_list.block_count() + right_list.block_count());
        for index in 0..left_list.block_count() {
            expected_raw.push(
                left_list
                    .block_bytes(index)
                    .ok_or("validated left block bytes")?
                    .to_vec(),
            );
        }
        for index in 0..right_list.block_count() {
            expected_raw.push(
                right_list
                    .block_bytes(index)
                    .ok_or("validated right block bytes")?
                    .to_vec(),
            );
        }
        let mut merged_raw = Vec::with_capacity(merged.block_count());
        for index in 0..merged.block_count() {
            merged_raw.push(
                merged
                    .block_bytes(index)
                    .ok_or("validated merged block bytes")?
                    .to_vec(),
            );
        }
        assert_eq!(merged_raw, expected_raw);

        let mut expected = left_postings.clone();
        expected.extend_from_slice(&right_postings);
        assert_eq!(merged.decode_all()?, expected);
        let monolithic = EncodedPostingList::encode(&expected)?;
        assert_eq!(monolithic.posting_list()?.decode_all()?, expected);
        assert_ne!(concatenated, monolithic.as_bytes());

        let seam_doc = right_postings[0].doc_id;
        let mut cursor = merged.cursor()?;
        assert_eq!(cursor.advance(seam_doc - 1)?, Some(right_postings[0]));
        assert_eq!(cursor.advance(seam_doc)?, Some(right_postings[0]));
        assert_eq!(cursor.next()?, Some(right_postings[1]));

        let mut reversed = right.as_bytes().to_vec();
        reversed.extend_from_slice(left.as_bytes());
        assert!(matches!(
            PostingList::parse(&reversed, 400),
            Err(PostingCodecError::NonAscendingDecoded { .. })
        ));
        Ok(())
    }

    #[test]
    fn every_partial_size_is_legal_at_an_interior_seam() -> TestResult {
        let right = sparse_postings(3, 100_000);
        let right_encoded = EncodedPostingList::encode(&right)?;
        for left_count in 1..POSTINGS_PER_BLOCK {
            let left = sparse_postings(left_count, 10);
            let left_encoded = EncodedPostingList::encode(&left)?;
            let mut bytes = left_encoded.as_bytes().to_vec();
            bytes.extend_from_slice(right_encoded.as_bytes());
            let list = PostingList::parse(&bytes, u32::try_from(left_count + right.len())?)?;
            let mut expected = left;
            expected.extend_from_slice(&right);
            assert_eq!(list.decode_all()?, expected, "left_count={left_count}");
            assert_eq!(list.blocks()[0].kind, PostingBlockKind::Vint);
            assert_eq!(list.blocks()[1].kind, PostingBlockKind::Vint);
        }
        Ok(())
    }

    #[test]
    fn positions_wire_golden_and_full_u32_vint_domain() -> TestResult {
        let postings = [Posting::new(10, 3), Posting::new(20, 2)];
        let posting_bytes = EncodedPostingList::encode(&postings)?;
        let posting_list = posting_bytes.posting_list()?;
        let encoded = EncodedPositionList::encode(&postings, &[0, 0, 7, 130, 130])?;
        assert_eq!(
            encoded.as_bytes(),
            [1, 0, 0, 0, 0, 0, 0, 0, 7, 0x82, 0x01, 0]
        );
        let positions = encoded.position_list(&posting_list)?;
        assert_eq!(collect_positions(&positions, 0)?, [0, 0, 7]);
        assert_eq!(collect_positions(&positions, 1)?, [130, 130]);

        let boundary_values = [0, 127, 128, 16_383, 16_384, 65_535, 65_536, u32::MAX];
        let boundary_postings: Vec<Posting> = boundary_values
            .iter()
            .enumerate()
            .map(|(index, _)| Ok(Posting::new(100 + u32::try_from(index)?, 1)))
            .collect::<Result<_, std::num::TryFromIntError>>()?;
        let boundary_posting_bytes = EncodedPostingList::encode(&boundary_postings)?;
        let boundary_posting_list = boundary_posting_bytes.posting_list()?;
        let boundary_encoded = EncodedPositionList::encode(&boundary_postings, &boundary_values)?;
        let boundary_list = boundary_encoded.position_list(&boundary_posting_list)?;
        assert_eq!(
            boundary_list.block_bytes(0),
            Some(
                &[
                    0x00, 0x7f, 0x80, 0x01, 0xff, 0x7f, 0x80, 0x80, 0x01, 0xff, 0xff, 0x03, 0x80,
                    0x80, 0x04, 0xff, 0xff, 0xff, 0xff, 0x0f,
                ][..]
            )
        );
        for (ordinal, expected) in boundary_values.iter().copied().enumerate() {
            assert_eq!(
                collect_positions(&boundary_list, u32::try_from(ordinal)?)?,
                [expected]
            );
        }
        Ok(())
    }

    #[test]
    fn positions_empty_count_alignment_and_resource_limits() -> TestResult {
        let empty_postings = EncodedPostingList::encode(&[])?;
        let empty_posting_list = empty_postings.posting_list()?;
        let empty = EncodedPositionList::encode(&[], &[])?;
        assert_eq!(empty.as_bytes(), [0, 0, 0, 0]);
        let empty_list = empty.position_list(&empty_posting_list)?;
        assert_eq!(empty_list.doc_freq(), 0);
        assert_eq!(empty_list.total_positions(), 0);
        assert_eq!(empty_list.block_count(), 0);
        assert!(empty_list.cursor()?.positions()?.is_none());

        let postings = [Posting::new(1, 2)];
        assert!(matches!(
            EncodedPositionList::encode(&postings, &[7]),
            Err(PositionCodecError::PositionCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
        assert!(matches!(
            EncodedPositionList::encode(&postings, &[7, 8, 9]),
            Err(PositionCodecError::PositionCountMismatch {
                expected: 2,
                actual: 3
            })
        ));
        assert!(matches!(
            EncodedPositionList::encode(&postings, &[8, 7]),
            Err(PositionCodecError::NonAscendingPosition { .. })
        ));
        assert!(matches!(
            EncodedPositionList::encode_with_limits(
                &postings,
                &[7, 8],
                PositionListLimits {
                    max_bytes: usize::MAX,
                    max_blocks: usize::MAX,
                    max_positions: 1,
                }
            ),
            Err(PositionCodecError::PositionLimitExceeded {
                limit: 1,
                actual: 2
            })
        ));
        assert!(matches!(
            EncodedPositionList::encode_with_limits(
                &postings,
                &[7, 8],
                PositionListLimits {
                    max_bytes: 5,
                    max_blocks: 1,
                    max_positions: 2,
                }
            ),
            Err(PositionCodecError::ByteLimitExceeded { limit: 5, .. })
        ));
        assert!(matches!(
            EncodedPositionList::encode_with_limits(
                &postings,
                &[7, 8],
                PositionListLimits {
                    max_bytes: usize::MAX,
                    max_blocks: 0,
                    max_positions: 2,
                }
            ),
            Err(PositionCodecError::BlockLimitExceeded {
                limit: 0,
                actual: 1
            })
        ));

        let posting_bytes = EncodedPostingList::encode(&postings)?;
        let posting_list = posting_bytes.posting_list()?;
        let encoded = EncodedPositionList::encode(&postings, &[7, 8])?;
        assert!(matches!(
            PositionList::parse_with_limits(
                encoded.as_bytes(),
                &posting_list,
                PositionListLimits {
                    max_bytes: encoded.as_bytes().len() - 1,
                    max_blocks: 1,
                    max_positions: 2,
                }
            ),
            Err(PositionCodecError::ByteLimitExceeded { .. })
        ));
        assert!(matches!(
            PositionList::parse_with_limits(
                encoded.as_bytes(),
                &posting_list,
                PositionListLimits {
                    max_bytes: encoded.as_bytes().len(),
                    max_blocks: 0,
                    max_positions: 2,
                }
            ),
            Err(PositionCodecError::BlockLimitExceeded {
                limit: 0,
                actual: 1
            })
        ));
        assert!(matches!(
            PositionList::parse_with_limits(
                encoded.as_bytes(),
                &posting_list,
                PositionListLimits {
                    max_bytes: encoded.as_bytes().len(),
                    max_blocks: 1,
                    max_positions: 1,
                }
            ),
            Err(PositionCodecError::PositionLimitExceeded {
                limit: 1,
                actual: 2
            })
        ));
        Ok(())
    }

    #[test]
    fn positions_fresh_blocks_cover_exact_fit_and_oversized_singleton() -> TestResult {
        let exact_postings = [Posting::new(1, 4_096), Posting::new(2, 1)];
        let exact_posting_bytes = EncodedPostingList::encode(&exact_postings)?;
        let exact_posting_list = exact_posting_bytes.posting_list()?;
        let exact_values = zero_positions(&exact_postings)?;
        let exact = EncodedPositionList::encode(&exact_postings, &exact_values)?;
        let exact_list = exact.position_list(&exact_posting_list)?;
        assert_eq!(exact_list.block_count(), 2);
        assert_eq!(exact_list.blocks()[0].byte_len(), 4_096);
        assert_eq!(exact_list.blocks()[0].posting_count(), 1);
        assert_eq!(exact_list.blocks()[1].byte_len(), 1);
        assert_eq!(exact_list.blocks()[1].base_posting_ordinal(), 1);

        let oversized_postings = [Posting::new(10, 4_097), Posting::new(20, 1)];
        let oversized_posting_bytes = EncodedPostingList::encode(&oversized_postings)?;
        let oversized_posting_list = oversized_posting_bytes.posting_list()?;
        let oversized_values = zero_positions(&oversized_postings)?;
        let oversized = EncodedPositionList::encode(&oversized_postings, &oversized_values)?;
        let oversized_list = oversized.position_list(&oversized_posting_list)?;
        assert_eq!(oversized_list.block_count(), 2);
        assert_eq!(oversized_list.blocks()[0].byte_len(), 4_097);
        assert_eq!(oversized_list.blocks()[0].posting_count(), 1);
        assert_eq!(collect_positions(&oversized_list, 0)?.len(), 4_097);

        let invalid_postings = [Posting::new(100, 3_000), Posting::new(200, 3_000)];
        let invalid_posting_bytes = EncodedPostingList::encode(&invalid_postings)?;
        let invalid_posting_list = invalid_posting_bytes.posting_list()?;
        let oversized_multi = raw_positions(&[(0, 0)], &vec![0; 6_000])?;
        assert!(matches!(
            PositionList::parse(&oversized_multi, &invalid_posting_list),
            Err(PositionCodecError::OversizedMultiDocumentBlock {
                block_index: 0,
                posting_count: 2,
                byte_len: 6_000
            })
        ));
        Ok(())
    }

    #[test]
    fn randomized_positions_roundtrip_duplicates_and_large_u32_values() -> TestResult {
        for case in 0_u64..32 {
            let mut state = 0x243f_6a88_85a3_08d3 ^ case;
            let count = if case == 0 {
                0
            } else {
                1 + usize::try_from(random_u32(&mut state) % 96)?
            };
            let mut postings = Vec::with_capacity(count);
            let mut expected_runs = Vec::with_capacity(count);
            let mut flat = Vec::new();
            let mut doc_id = random_u32(&mut state) % 100;
            for ordinal in 0..count {
                if ordinal != 0 {
                    doc_id = doc_id
                        .checked_add(1 + random_u32(&mut state) % 9)
                        .ok_or("random position fixture docid overflow")?;
                }
                let freq = 1 + random_u32(&mut state) % 8;
                postings.push(Posting::new(doc_id, freq));
                let mut run = Vec::with_capacity(usize::try_from(freq)?);
                let mut position = if ordinal % 5 == 0 {
                    65_536 + random_u32(&mut state) % 100_000
                } else {
                    random_u32(&mut state) % 10_000
                };
                for index in 0..freq {
                    if index != 0 {
                        position = position
                            .checked_add(random_u32(&mut state) % 7)
                            .ok_or("random position fixture overflow")?;
                    }
                    run.push(position);
                    flat.push(position);
                }
                expected_runs.push(run);
            }

            let posting_bytes = EncodedPostingList::encode(&postings)?;
            let posting_list = posting_bytes.posting_list()?;
            let encoded = EncodedPositionList::encode(&postings, &flat)?;
            assert_eq!(encoded, EncodedPositionList::encode(&postings, &flat)?);
            let list = encoded.position_list(&posting_list)?;
            assert_eq!(list.doc_freq(), u32::try_from(count)?);
            assert_eq!(list.total_positions(), u64::try_from(flat.len())?);
            for (ordinal, expected) in expected_runs.iter().enumerate() {
                assert_eq!(
                    collect_positions(&list, u32::try_from(ordinal)?)?,
                    *expected,
                    "case={case} ordinal={ordinal}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn position_cursor_reuses_state_across_posting_and_position_blocks() -> TestResult {
        let mut postings = Vec::with_capacity(270);
        let mut expected_runs = Vec::with_capacity(270);
        let mut flat = Vec::new();
        for ordinal in 0_u32..270 {
            postings.push(Posting::new(10 + ordinal * 3, 24));
            let base = 70_000 + ordinal * 100;
            let run: Vec<u32> = (0_u32..24).map(|index| base + index / 2).collect();
            flat.extend_from_slice(&run);
            expected_runs.push(run);
        }
        let posting_bytes = EncodedPostingList::encode(&postings)?;
        let posting_list = posting_bytes.posting_list()?;
        assert_eq!(posting_list.blocks()[1].base_posting_ordinal, 128);
        let encoded = EncodedPositionList::encode(&postings, &flat)?;
        let positions = encoded.position_list(&posting_list)?;
        assert!(positions.block_count() >= 2);
        assert_ne!(positions.blocks()[1].base_posting_ordinal(), 128);

        let mut cursor = positions.cursor()?;
        assert_eq!(cursor.posting_ordinal(), Some(0));
        assert_eq!(
            cursor
                .positions()?
                .ok_or("position cursor start")?
                .collect::<Result<Vec<_>, _>>()?,
            expected_runs[0]
        );
        assert_eq!(cursor.advance(postings[127].doc_id)?, Some(postings[127]));
        assert_eq!(cursor.posting_ordinal(), Some(127));
        assert_eq!(
            cursor
                .positions()?
                .ok_or("position cursor 127")?
                .collect::<Result<Vec<_>, _>>()?,
            expected_runs[127]
        );
        assert_eq!(cursor.next()?, Some(postings[128]));
        assert_eq!(cursor.posting_ordinal(), Some(128));
        assert_eq!(
            cursor
                .positions()?
                .ok_or("position cursor 128")?
                .collect::<Result<Vec<_>, _>>()?,
            expected_runs[128]
        );

        let position_seam = usize::try_from(positions.blocks()[1].base_posting_ordinal())?;
        let before_seam = position_seam
            .checked_sub(1)
            .ok_or("position seam at zero")?;
        assert_eq!(
            cursor.advance(postings[before_seam].doc_id)?,
            Some(postings[before_seam])
        );
        assert_eq!(cursor.next()?, Some(postings[position_seam]));
        assert_eq!(
            cursor
                .positions()?
                .ok_or("sequential position seam")?
                .collect::<Result<Vec<_>, _>>()?,
            expected_runs[position_seam]
        );
        let after_seam = position_seam + 7;
        assert_eq!(
            cursor.advance(postings[after_seam].doc_id)?,
            Some(postings[after_seam])
        );
        assert_eq!(
            cursor
                .positions()?
                .ok_or("position seam in-block advance")?
                .collect::<Result<Vec<_>, _>>()?,
            expected_runs[after_seam]
        );
        assert_eq!(cursor.advance(u32::MAX)?, None);
        assert_eq!(cursor.next()?, None);
        assert!(cursor.positions()?.is_none());

        let maximum_postings = [Posting::new(0, 2), Posting::new(u32::MAX, 1)];
        let maximum_posting_bytes = EncodedPostingList::encode(&maximum_postings)?;
        let maximum_posting_list = maximum_posting_bytes.posting_list()?;
        let maximum_encoded = EncodedPositionList::encode(&maximum_postings, &[1, 1, u32::MAX])?;
        let maximum_positions = maximum_encoded.position_list(&maximum_posting_list)?;
        let mut maximum_cursor = maximum_positions.cursor()?;
        assert_eq!(maximum_cursor.advance(u32::MAX)?, Some(maximum_postings[1]));
        assert_eq!(
            maximum_cursor
                .positions()?
                .ok_or("u32 max position cursor")?
                .collect::<Result<Vec<_>, _>>()?,
            [u32::MAX]
        );
        assert_eq!(maximum_cursor.next()?, None);
        Ok(())
    }

    #[test]
    fn positions_q1_concat_preserves_payloads_and_is_associative() -> TestResult {
        let left_postings = sparse_postings(100, 100);
        let right_postings = sparse_postings(300, 100_000);
        let left_posting_bytes = EncodedPostingList::encode(&left_postings)?;
        let right_posting_bytes = EncodedPostingList::encode(&right_postings)?;
        let left_posting_list = left_posting_bytes.posting_list()?;
        let right_posting_list = right_posting_bytes.posting_list()?;
        let left_values = zero_positions(&left_postings)?;
        let right_values = zero_positions(&right_postings)?;
        let left_encoded = EncodedPositionList::encode(&left_postings, &left_values)?;
        let right_encoded = EncodedPositionList::encode(&right_postings, &right_values)?;
        let left = left_encoded.position_list(&left_posting_list)?;
        let right = right_encoded.position_list(&right_posting_list)?;

        let merged_encoded = EncodedPositionList::concatenate(&[&left, &right])?;
        let expected_blocks = left.block_count() + right.block_count();
        assert!(matches!(
            EncodedPositionList::concatenate_with_limits(
                &[&left, &right],
                PositionListLimits {
                    max_bytes: usize::MAX,
                    max_blocks: left.block_count(),
                    max_positions: u64::MAX,
                }
            ),
            Err(PositionCodecError::BlockLimitExceeded { actual, .. })
                if actual == left.block_count() + 1
        ));
        assert!(matches!(
            EncodedPositionList::concatenate_with_limits(
                &[&left, &right],
                PositionListLimits {
                    max_bytes: usize::MAX,
                    max_blocks: expected_blocks,
                    max_positions: merged_encoded.total_positions() - 1,
                }
            ),
            Err(PositionCodecError::PositionLimitExceeded { .. })
        ));
        assert!(matches!(
            EncodedPositionList::concatenate_with_limits(
                &[&left, &right],
                PositionListLimits {
                    max_bytes: merged_encoded.as_bytes().len() - 1,
                    max_blocks: expected_blocks,
                    max_positions: merged_encoded.total_positions(),
                }
            ),
            Err(PositionCodecError::ByteLimitExceeded { .. })
        ));
        let mut merged_posting_bytes = left_posting_bytes.as_bytes().to_vec();
        merged_posting_bytes.extend_from_slice(right_posting_bytes.as_bytes());
        let merged_postings = PostingList::parse(&merged_posting_bytes, 400)?;
        let merged = merged_encoded.position_list(&merged_postings)?;
        assert_eq!(merged.doc_freq(), 400);
        assert_eq!(
            merged.block_count(),
            left.block_count() + right.block_count()
        );
        let mut expected_payloads = Vec::new();
        for source in [&left, &right] {
            for block_index in 0..source.block_count() {
                expected_payloads.push(
                    source
                        .block_bytes(block_index)
                        .ok_or("source position block")?
                        .to_vec(),
                );
            }
        }
        let mut actual_payloads = Vec::new();
        for block_index in 0..merged.block_count() {
            actual_payloads.push(
                merged
                    .block_bytes(block_index)
                    .ok_or("merged position block")?
                    .to_vec(),
            );
        }
        assert_eq!(actual_payloads, expected_payloads);
        assert_eq!(
            merged.blocks()[left.block_count()].base_posting_ordinal(),
            100
        );
        assert_eq!(
            collect_positions(&merged, 100)?.len(),
            usize::try_from(right_postings[0].freq)?
        );

        let middle_postings = sparse_postings(120, 100_000);
        let tail_postings = sparse_postings(180, 200_000);
        let middle_posting_bytes = EncodedPostingList::encode(&middle_postings)?;
        let tail_posting_bytes = EncodedPostingList::encode(&tail_postings)?;
        let middle_posting_list = middle_posting_bytes.posting_list()?;
        let tail_posting_list = tail_posting_bytes.posting_list()?;
        let middle_encoded =
            EncodedPositionList::encode(&middle_postings, &zero_positions(&middle_postings)?)?;
        let tail_encoded =
            EncodedPositionList::encode(&tail_postings, &zero_positions(&tail_postings)?)?;
        let middle = middle_encoded.position_list(&middle_posting_list)?;
        let tail = tail_encoded.position_list(&tail_posting_list)?;

        let direct = EncodedPositionList::concatenate(&[&left, &middle, &tail])?;
        let first_pair = EncodedPositionList::concatenate(&[&left, &middle])?;
        let mut first_pair_posting_bytes = left_posting_bytes.as_bytes().to_vec();
        first_pair_posting_bytes.extend_from_slice(middle_posting_bytes.as_bytes());
        let first_pair_postings = PostingList::parse(&first_pair_posting_bytes, 220)?;
        let first_pair_list = first_pair.position_list(&first_pair_postings)?;
        let staged = EncodedPositionList::concatenate(&[&first_pair_list, &tail])?;
        assert_eq!(direct, staged);
        assert!(matches!(
            EncodedPositionList::concatenate(&[&tail, &left]),
            Err(PositionCodecError::NonAscendingConcat { .. })
        ));
        Ok(())
    }

    #[test]
    fn positions_parser_rejects_structured_corruption_and_never_panics() -> TestResult {
        let one_posting_bytes = EncodedPostingList::encode(&[Posting::new(10, 1)])?;
        let one_posting = one_posting_bytes.posting_list()?;
        let valid = EncodedPositionList::encode(&[Posting::new(10, 1)], &[0])?;
        for cut in 0..valid.as_bytes().len() {
            assert!(
                PositionList::parse(&valid.as_bytes()[..cut], &one_posting).is_err(),
                "cut={cut}"
            );
        }
        assert!(matches!(
            PositionList::parse(&[0, 0, 0, 0], &one_posting),
            Err(PositionCodecError::InvalidBlockCount { .. })
        ));
        assert!(matches!(
            PositionList::parse(&raw_positions(&[(1, 0)], &[0])?, &one_posting),
            Err(PositionCodecError::InvalidDirectoryStart { .. })
        ));
        assert!(matches!(
            PositionList::parse(&[1, 0, 0, 0, 0x80, 0, 0, 0], &one_posting),
            Err(PositionCodecError::NonCanonicalVint { domain: "u32", .. })
        ));
        assert!(matches!(
            PositionList::parse(&[1, 0, 0, 0, 0, 0x80, 0, 0], &one_posting),
            Err(PositionCodecError::NonCanonicalVint { domain: "u64", .. })
        ));
        assert!(matches!(
            PositionList::parse(&[1, 0, 0, 0, 0, 0, 0x80, 0], &one_posting),
            Err(PositionCodecError::NonCanonicalVint { domain: "u32", .. })
        ));
        assert!(matches!(
            PositionList::parse(
                &[1, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0x10],
                &one_posting
            ),
            Err(PositionCodecError::VintOverflow { domain: "u32", .. })
        ));

        let mut offset_overflow = vec![1, 0, 0, 0, 0];
        offset_overflow.extend_from_slice(&[0xff; 9]);
        offset_overflow.extend_from_slice(&[0x02, 0]);
        assert!(matches!(
            PositionList::parse(&offset_overflow, &one_posting),
            Err(PositionCodecError::VintOverflow { domain: "u64", .. })
        ));

        let two_postings = [Posting::new(10, 1), Posting::new(20, 1)];
        let two_posting_bytes = EncodedPostingList::encode(&two_postings)?;
        let two_posting_list = two_posting_bytes.posting_list()?;
        assert!(matches!(
            PositionList::parse(
                &raw_positions(&[(0, 0), (0, 1)], &[0, 0])?,
                &two_posting_list
            ),
            Err(PositionCodecError::NonAscendingBlockOrdinal { .. })
        ));
        assert!(matches!(
            PositionList::parse(
                &raw_positions(&[(0, 0), (1, 0)], &[0, 0])?,
                &two_posting_list
            ),
            Err(PositionCodecError::NonAscendingBlockOffset { .. })
        ));
        assert!(matches!(
            PositionList::parse(
                &raw_positions(&[(0, 0), (2, 1)], &[0, 0])?,
                &two_posting_list
            ),
            Err(PositionCodecError::BlockOrdinalOutOfRange {
                block_index: 1,
                first_posting_ordinal: 2,
                doc_freq: 2
            })
        ));
        assert!(matches!(
            PositionList::parse(
                &raw_positions(&[(0, 0), (1, 3)], &[0, 0])?,
                &two_posting_list
            ),
            Err(PositionCodecError::BlockOffsetOutOfRange { .. })
        ));
        assert!(matches!(
            PositionList::parse(
                &raw_positions(&[(0, 0), (1, 1)], &[0x80, 0x01, 0])?,
                &two_posting_list
            ),
            Err(PositionCodecError::Truncated { .. })
        ));

        let overflow_postings = [Posting::new(10, 2)];
        let overflow_posting_bytes = EncodedPostingList::encode(&overflow_postings)?;
        let overflow_posting_list = overflow_posting_bytes.posting_list()?;
        assert!(matches!(
            PositionList::parse(
                &raw_positions(&[(0, 0)], &[0xff, 0xff, 0xff, 0xff, 0x0f, 1])?,
                &overflow_posting_list
            ),
            Err(PositionCodecError::PositionOverflow { .. })
        ));
        assert!(matches!(
            PositionList::parse(&raw_positions(&[(0, 0)], &[0, 0])?, &one_posting),
            Err(PositionCodecError::TrailingBlockBytes {
                block_index: 0,
                remaining: 1
            })
        ));

        let mut state = 0x1319_8a2e_0370_7344;
        for len in 0..64 {
            let bytes: Vec<u8> = (0..len)
                .map(|_| random_u32(&mut state).to_le_bytes()[0])
                .collect();
            let _ = PositionList::parse(&bytes, &two_posting_list);
        }
        Ok(())
    }

    #[test]
    fn blockmax_fresh_seal_roundtrips_all_posting_block_shapes() -> TestResult {
        for count in [0, 1, 127, 128, 129, 255, 256, 400] {
            let mut postings = sparse_postings(count, 100);
            for (ordinal, posting) in postings.iter_mut().enumerate() {
                posting.freq = match ordinal % 7 {
                    0 => 1,
                    1 => 254,
                    2 => 255,
                    3 => 256,
                    4 => 65_535,
                    5 => u32::MAX,
                    _ => 17,
                };
            }
            let ordinary_postings = EncodedPostingList::encode(&postings)?;
            let (encoded_postings, encoded) =
                EncodedPostingList::encode_with_block_max(&postings, fixture_fieldnorm)?;
            assert_eq!(
                encoded_postings, ordinary_postings,
                "posting parity count={count}"
            );
            let posting_list = encoded_postings.posting_list()?;
            let recomputed = EncodedBlockMax::encode(&posting_list, fixture_fieldnorm)?;
            assert_eq!(encoded, recomputed, "integrated parity count={count}");
            let (repeated_postings, repeated_bounds) =
                EncodedPostingList::encode_with_block_max(&postings, fixture_fieldnorm)?;
            assert_eq!(
                encoded_postings, repeated_postings,
                "posting determinism count={count}"
            );
            assert_eq!(encoded, repeated_bounds, "bound determinism count={count}");
            let parsed = BlockMaxList::parse_with_fieldnorms(
                encoded.as_bytes(),
                &posting_list,
                fixture_fieldnorm,
            )?;
            assert_eq!(parsed.entry_count(), posting_list.block_count());
            assert_eq!(encoded.entry_count(), posting_list.block_count());
            assert_eq!(parsed.as_bytes(), encoded.as_bytes());
            for (block_index, (entry, block)) in parsed
                .entries()
                .iter()
                .zip(posting_list.blocks())
                .enumerate()
            {
                assert_eq!(entry.first_doc(), block.first_doc, "count={count}");
                assert_eq!(
                    entry.block_offset(),
                    u64::try_from(block.byte_offset)?,
                    "count={count}"
                );
                let decoded = decode_block_for_block_max(&posting_list, block_index)?;
                assert_eq!(
                    entry.max_frequency_code(),
                    canonical_block_max_frequency_code(&decoded),
                    "count={count}"
                );
                assert_eq!(
                    entry.min_fieldnorm(),
                    crate::contract::id_to_fieldnorm(entry.min_fieldnorm_id())
                );
            }
            assert_eq!(parsed.cursor().current(), parsed.entries().first().copied());
        }

        let (_, singleton) =
            EncodedPostingList::encode_with_block_max(&[Posting::new(128, 256)], |_| Some(0))?;
        assert_eq!(singleton.as_bytes(), [0x80, 0, 0, 0, 0, 0xff, 0]);
        Ok(())
    }

    #[test]
    fn blockmax_u64_vints_are_canonical_through_the_full_domain() -> TestResult {
        for value in [
            0,
            1,
            127,
            128,
            16_383,
            16_384,
            u64::from(u32::MAX),
            u64::MAX,
        ] {
            let mut bytes = Vec::new();
            write_vint64(value, &mut bytes);
            assert_eq!(bytes.len(), vint64_length(value));
            let mut reader = BlockMaxByteReader::new(&bytes);
            assert_eq!(reader.read_vint()?, value);
            reader.finish()?;
        }
        assert_eq!(
            {
                let mut bytes = Vec::new();
                write_vint64(u64::MAX, &mut bytes);
                bytes
            },
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]
        );
        Ok(())
    }

    #[test]
    fn blockmax_parser_rejects_wire_and_cross_section_corruption() -> TestResult {
        let encoded_postings = EncodedPostingList::encode(&[Posting::new(128, 7)])?;
        let posting_list = encoded_postings.posting_list()?;
        let encoded = EncodedBlockMax::encode(&posting_list, |_| Some(4))?;
        assert_eq!(encoded.as_bytes(), [0x80, 0, 0, 0, 0, 7, 4]);

        for cut in 0..encoded.as_bytes().len() {
            assert!(
                BlockMaxList::parse_with_fieldnorms(
                    &encoded.as_bytes()[..cut],
                    &posting_list,
                    |_| Some(4),
                )
                .is_err(),
                "cut={cut}"
            );
        }

        let mut first_doc = encoded.as_bytes().to_vec();
        first_doc[0] ^= 1;
        assert!(matches!(
            BlockMaxList::parse_with_fieldnorms(&first_doc, &posting_list, |_| Some(4)),
            Err(BlockMaxError::FirstDocMismatch { .. })
        ));

        let mut block_offset = encoded.as_bytes().to_vec();
        block_offset[4] = 1;
        assert!(matches!(
            BlockMaxList::parse_with_fieldnorms(&block_offset, &posting_list, |_| Some(4)),
            Err(BlockMaxError::BlockOffsetMismatch { .. })
        ));

        let mut zero_frequency = encoded.as_bytes().to_vec();
        zero_frequency[5] = 0;
        assert!(matches!(
            BlockMaxList::parse_with_fieldnorms(&zero_frequency, &posting_list, |_| Some(4)),
            Err(BlockMaxError::ZeroMaximumFrequency { .. })
        ));

        let mut wrong_frequency = encoded.as_bytes().to_vec();
        wrong_frequency[5] = 6;
        assert!(matches!(
            BlockMaxList::parse_with_fieldnorms(&wrong_frequency, &posting_list, |_| Some(4)),
            Err(BlockMaxError::MaximumFrequencyMismatch { .. })
        ));

        let mut wrong_fieldnorm = encoded.as_bytes().to_vec();
        wrong_fieldnorm[6] = u8::MAX;
        let merge_only = BlockMaxConcatList::parse(&wrong_fieldnorm, &posting_list)?;
        assert_eq!(merge_only.entry_count(), 1);
        assert!(matches!(
            BlockMaxList::parse_with_fieldnorms(&wrong_fieldnorm, &posting_list, |_| Some(4)),
            Err(BlockMaxError::MinimumFieldnormMismatch { .. })
        ));
        assert!(matches!(
            BlockMaxList::parse_with_fieldnorms(encoded.as_bytes(), &posting_list, |_| None),
            Err(BlockMaxError::MissingFieldnorm { doc_id: 128, .. })
        ));

        let mut trailing = encoded.as_bytes().to_vec();
        trailing.push(0);
        assert!(matches!(
            BlockMaxList::parse_with_fieldnorms(&trailing, &posting_list, |_| Some(4)),
            Err(BlockMaxError::TrailingBytes { remaining: 1, .. })
        ));

        let mut noncanonical = 128_u32.to_le_bytes().to_vec();
        noncanonical.extend_from_slice(&[0x80, 0x00, 7, 4]);
        assert!(matches!(
            BlockMaxList::parse_with_fieldnorms(&noncanonical, &posting_list, |_| Some(4)),
            Err(BlockMaxError::NonCanonicalVint { .. })
        ));

        let mut overflow = 128_u32.to_le_bytes().to_vec();
        overflow.extend_from_slice(&[0x80; 9]);
        overflow.extend_from_slice(&[0x02, 7, 4]);
        assert!(matches!(
            BlockMaxList::parse_with_fieldnorms(&overflow, &posting_list, |_| Some(4)),
            Err(BlockMaxError::VintOverflow { .. })
        ));
        Ok(())
    }

    #[test]
    fn blockmax_public_score_view_requires_validated_doclen() -> TestResult {
        let postings = [Posting::new(128, 7), Posting::new(130, 3)];
        let document_lengths = [Some(3), None, Some(41)];
        let field_inputs = [DocLenFieldInput::new(7, &document_lengths)];
        let doclen = EncodedDocLenSection::encode(128, 131, &[7], &field_inputs)?;
        let section = doclen.section(&[7])?;
        let fieldnorms = section.field(7).ok_or("validated field")?;

        let (encoded_postings, encoded_bounds) =
            EncodedPostingList::encode_with_block_max(&postings, |doc_id| {
                fieldnorms.fieldnorm_id(u64::from(doc_id))
            })?;
        let posting_list = encoded_postings.posting_list()?;
        let validated = encoded_bounds.block_max_list(&posting_list, fieldnorms)?;
        assert_eq!(validated.entry_count(), 1);

        let mut understated = encoded_bounds.as_bytes().to_vec();
        let minimum = understated.last_mut().ok_or("fieldnorm byte")?;
        *minimum = u8::MAX;
        assert!(matches!(
            BlockMaxList::parse(&understated, &posting_list, fieldnorms),
            Err(BlockMaxError::MinimumFieldnormMismatch { .. })
        ));

        let maximum_length = [Some(u32::MAX)];
        let maximum_input = [DocLenFieldInput::new(7, &maximum_length)];
        let maximum_doclen = EncodedDocLenSection::encode(200, 201, &[7], &maximum_input)?;
        let maximum_section = maximum_doclen.section(&[7])?;
        let maximum_field = maximum_section.field(7).ok_or("maximum fieldnorm")?;
        let (maximum_postings, maximum_bounds) =
            EncodedPostingList::encode_with_block_max(&[Posting::new(200, 1)], |doc_id| {
                maximum_field.fieldnorm_id(u64::from(doc_id))
            })?;
        let maximum_list = maximum_postings.posting_list()?;
        let maximum_validated = maximum_bounds.block_max_list(&maximum_list, maximum_field)?;
        assert_eq!(maximum_validated.entries()[0].min_fieldnorm_id(), u8::MAX);
        Ok(())
    }

    #[test]
    fn blockmax_cursor_advance_matches_validated_posting_blocks() -> TestResult {
        let postings = sparse_postings(400, 100);
        let encoded_postings = EncodedPostingList::encode(&postings)?;
        let posting_list = encoded_postings.posting_list()?;
        let encoded = EncodedBlockMax::encode(&posting_list, fixture_fieldnorm)?;
        let list = BlockMaxList::parse_with_fieldnorms(
            encoded.as_bytes(),
            &posting_list,
            fixture_fieldnorm,
        )?;
        let last_doc = postings.last().ok_or("non-empty postings")?.doc_id;
        let targets = [
            0,
            postings[0].doc_id,
            postings[127].doc_id,
            postings[127].doc_id + 1,
            postings[128].doc_id,
            last_doc,
            last_doc + 1,
            u32::MAX,
        ];
        for target in targets {
            let expected_index = posting_list
                .blocks()
                .iter()
                .position(|block| block.last_doc >= target);
            let mut cursor = list.cursor();
            let actual = cursor.advance(target);
            assert_eq!(cursor.block_index(), expected_index, "target={target}");
            assert_eq!(
                cursor.last_doc(),
                expected_index.map(|index| posting_list.blocks()[index].last_doc),
                "target={target}"
            );
            assert_eq!(
                actual,
                expected_index.and_then(|index| list.entries().get(index).copied()),
                "target={target}"
            );
        }

        let mut cursor = list.cursor();
        for block_index in 0..list.entry_count() {
            assert_eq!(cursor.block_index(), Some(block_index));
            if block_index + 1 == list.entry_count() {
                assert_eq!(cursor.next(), None);
                assert_eq!(cursor.next(), None);
            } else {
                assert_eq!(cursor.next(), list.entries().get(block_index + 1).copied());
            }
        }

        let mut no_rewind = list.cursor();
        let late_target = postings[200].doc_id;
        let late_entry = no_rewind.advance(late_target);
        let late_index = no_rewind.block_index();
        assert!(late_index.is_some());
        assert_eq!(no_rewind.advance(0), late_entry);
        assert_eq!(no_rewind.block_index(), late_index);

        let max_postings = [Posting::new(u32::MAX - 1, 3), Posting::new(u32::MAX, 7)];
        let max_encoded = EncodedPostingList::encode(&max_postings)?;
        let max_posting_list = max_encoded.posting_list()?;
        let max_bounds = EncodedBlockMax::encode(&max_posting_list, fixture_fieldnorm)?;
        let max_list = BlockMaxList::parse_with_fieldnorms(
            max_bounds.as_bytes(),
            &max_posting_list,
            fixture_fieldnorm,
        )?;
        let mut max_cursor = max_list.cursor();
        assert_eq!(
            max_cursor.advance(u32::MAX),
            max_list.entries().first().copied()
        );
        assert_eq!(max_cursor.last_doc(), Some(u32::MAX));
        assert_eq!(max_cursor.next(), None);
        assert_eq!(max_cursor.advance(u32::MAX), None);
        Ok(())
    }

    #[test]
    fn blockmax_covers_bitmap_posting_blocks() -> TestResult {
        let postings = postings_with_span(511);
        let ordinary_postings = EncodedPostingList::encode(&postings)?;
        let (encoded_postings, encoded) =
            EncodedPostingList::encode_with_block_max(&postings, fixture_fieldnorm)?;
        assert_eq!(encoded_postings, ordinary_postings);
        let posting_list = encoded_postings.posting_list()?;
        assert_eq!(posting_list.blocks()[0].kind, PostingBlockKind::Bitmap);

        assert_eq!(
            encoded,
            EncodedBlockMax::encode(&posting_list, fixture_fieldnorm)?
        );
        let parsed = BlockMaxList::parse_with_fieldnorms(
            encoded.as_bytes(),
            &posting_list,
            fixture_fieldnorm,
        )?;
        let entry = parsed
            .entries()
            .first()
            .copied()
            .ok_or("one bitmap bound")?;
        assert_eq!(parsed.entry_count(), 1);
        assert_eq!(entry.first_doc(), postings[0].doc_id);
        assert_eq!(entry.block_offset(), 0);
        assert_eq!(entry.max_frequency(), 9);
        assert_eq!(
            entry.min_fieldnorm_id(),
            postings
                .iter()
                .filter_map(|posting| fixture_fieldnorm(posting.doc_id))
                .min()
                .ok_or("bitmap fieldnorm minimum")?
        );
        Ok(())
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn blockmax_bounds_dominate_every_score_across_live_avgdl_regimes() -> TestResult {
        let mut postings = dense_postings(400, 100);
        for (ordinal, posting) in postings.iter_mut().enumerate() {
            posting.freq = match ordinal / POSTINGS_PER_BLOCK {
                0 => u32::try_from(ordinal % 37 + 1)?,
                1 => u32::try_from(ordinal % 254 + 1)?,
                2 => match ordinal % 11 {
                    0 => u32::MAX,
                    1 => 256,
                    2 => 255,
                    _ => u32::try_from(ordinal % 31 + 1)?,
                },
                _ => u32::try_from(ordinal % 17 + 1)?,
            };
        }
        let encoded_postings = EncodedPostingList::encode(&postings)?;
        let posting_list = encoded_postings.posting_list()?;
        let encoded = EncodedBlockMax::encode(&posting_list, fixture_fieldnorm)?;
        let blockmax = BlockMaxList::parse_with_fieldnorms(
            encoded.as_bytes(),
            &posting_list,
            fixture_fieldnorm,
        )?;
        let snapshot_pairs = blockmax.entries().to_vec();

        for average_fieldnorm in [0.25_f32, 1.0, 17.5, 1_000.0, 1_000_000.0] {
            for doc_count in [400_u64, 1_000, 100_000] {
                let idf = crate::contract::idf(400, doc_count);
                for boost in [0.25_f32, 1.0, 4.0] {
                    let weight = idf * (1.0 + crate::contract::BM25_K1) * boost;
                    for (block_index, entry) in blockmax.entries().iter().copied().enumerate() {
                        let bound = entry
                            .score_upper_bound(average_fieldnorm, weight)
                            .ok_or("valid positive block bound")?;
                        let decoded = decode_block_for_block_max(&posting_list, block_index)?;
                        for within in 0..usize::from(decoded.posting_count) {
                            let frequency = decoded.freqs[within] as f32;
                            let fieldnorm_id = fixture_fieldnorm(decoded.docs[within])
                                .ok_or("fixture fieldnorm")?;
                            let norm = crate::contract::cached_tf_component(
                                crate::contract::id_to_fieldnorm(fieldnorm_id),
                                average_fieldnorm,
                            );
                            let score = weight * (frequency / (frequency + norm));
                            assert!(
                                bound >= score,
                                "block={block_index} within={within} avgdl={average_fieldnorm} doc_count={doc_count} boost={boost} bound={bound} score={score}"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(blockmax.entries(), snapshot_pairs);
        Ok(())
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn randomized_blockmax_bounds_dominate_seeded_oracle() -> TestResult {
        let boundary_counts = [1, 127, 128, 129, 255, 256, 400];
        for case in 0_u64..32 {
            let seed = 0xd1b5_4a32_d192_ed03 ^ case;
            let mut state = seed;
            let count = boundary_counts[usize::try_from(case)? % boundary_counts.len()];
            let mut doc_id = random_u32(&mut state) % 100;
            let mut postings = Vec::with_capacity(count);
            for ordinal in 0..count {
                if ordinal != 0 {
                    let random = random_u32(&mut state);
                    let step = if case.is_multiple_of(2) {
                        1 + random % 3
                    } else {
                        5 + random % 12
                    };
                    doc_id = doc_id.checked_add(step).ok_or("seeded docid overflow")?;
                }
                let random = random_u32(&mut state);
                let frequency = if case.is_multiple_of(4) {
                    1 + random % 254
                } else {
                    match ordinal % 43 {
                        0 => 255,
                        1 => 256,
                        2 => u32::MAX,
                        _ => 1 + random % 254,
                    }
                };
                postings.push(Posting::new(doc_id, frequency));
            }
            let fieldnorm_for_doc = |doc_id: u32| {
                u8::try_from((u64::from(doc_id).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ seed) & 0xff)
                    .ok()
            };
            let (encoded_postings, encoded_bounds) =
                EncodedPostingList::encode_with_block_max(&postings, fieldnorm_for_doc)?;
            let posting_list = encoded_postings.posting_list()?;
            let bounds = BlockMaxList::parse_with_fieldnorms(
                encoded_bounds.as_bytes(),
                &posting_list,
                fieldnorm_for_doc,
            )?;

            for live_avgdl in [0.25_f32, 1.0, 17.5, 1_000.0, 1_000_000.0] {
                for weight in [0.0_f32, 0.25, 3.25] {
                    for (block_index, entry) in bounds.entries().iter().copied().enumerate() {
                        let upper = entry
                            .score_upper_bound(live_avgdl, weight)
                            .ok_or("valid seeded bound")?;
                        let decoded = decode_block_for_block_max(&posting_list, block_index)?;
                        for within in 0..usize::from(decoded.posting_count) {
                            let frequency = decoded.freqs[within] as f32;
                            let fieldnorm_id = fieldnorm_for_doc(decoded.docs[within])
                                .ok_or("seeded fieldnorm")?;
                            let norm = crate::contract::cached_tf_component(
                                crate::contract::id_to_fieldnorm(fieldnorm_id),
                                live_avgdl,
                            );
                            let score = weight * (frequency / (frequency + norm));
                            assert!(
                                upper >= score,
                                "seed={seed:#x} case={case} block={block_index} within={within} avgdl={live_avgdl} weight={weight} upper={upper} score={score}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn blockmax_q1_concat_preserves_bounds_and_rebases_only_offsets() -> TestResult {
        let left_postings = sparse_postings(100, 100);
        let right_postings = sparse_postings(300, 10_000);
        let left_encoded = EncodedPostingList::encode(&left_postings)?;
        let right_encoded = EncodedPostingList::encode(&right_postings)?;
        let left_list = left_encoded.posting_list()?;
        let right_list = right_encoded.posting_list()?;
        let left_bounds = EncodedBlockMax::encode(&left_list, fixture_fieldnorm)?;
        let right_bounds = EncodedBlockMax::encode(&right_list, fixture_fieldnorm)?;
        let left_blockmax = BlockMaxList::parse_with_fieldnorms(
            left_bounds.as_bytes(),
            &left_list,
            fixture_fieldnorm,
        )?;
        let right_blockmax = BlockMaxList::parse_with_fieldnorms(
            right_bounds.as_bytes(),
            &right_list,
            fixture_fieldnorm,
        )?;
        let left_concat = left_bounds.concat_list(&left_list)?;
        let right_concat = right_bounds.concat_list(&right_list)?;

        let merged_bounds = EncodedBlockMax::concatenate(&[&left_concat, &right_concat])?;
        let mut merged_posting_bytes = left_encoded.as_bytes().to_vec();
        merged_posting_bytes.extend_from_slice(right_encoded.as_bytes());
        let merged_postings = PostingList::parse(&merged_posting_bytes, 400)?;
        let merged = BlockMaxList::parse_with_fieldnorms(
            merged_bounds.as_bytes(),
            &merged_postings,
            fixture_fieldnorm,
        )?;
        assert_eq!(merged.entry_count(), 4);
        assert_eq!(
            merged_postings
                .blocks()
                .iter()
                .map(|block| block.posting_count)
                .collect::<Vec<_>>(),
            [100, 128, 128, 44]
        );

        let left_prefix = u64::try_from(left_encoded.as_bytes().len())?;
        let mut expected_entries = left_blockmax.entries().to_vec();
        expected_entries.extend(right_blockmax.entries().iter().map(|entry| BlockMaxEntry {
            first_doc: entry.first_doc,
            block_offset: left_prefix + entry.block_offset,
            max_frequency_code: entry.max_frequency_code,
            min_fieldnorm_id: entry.min_fieldnorm_id,
        }));
        assert_eq!(merged.entries(), expected_entries);

        for left_count in 1..POSTINGS_PER_BLOCK {
            let left = sparse_postings(left_count, 10);
            let right = sparse_postings(3, 100_000);
            let left_bytes = EncodedPostingList::encode(&left)?;
            let right_bytes = EncodedPostingList::encode(&right)?;
            let left_posting_list = left_bytes.posting_list()?;
            let right_posting_list = right_bytes.posting_list()?;
            let left_bound_bytes = EncodedBlockMax::encode(&left_posting_list, fixture_fieldnorm)?;
            let right_bound_bytes =
                EncodedBlockMax::encode(&right_posting_list, fixture_fieldnorm)?;
            let left_bound_list = left_bound_bytes.concat_list(&left_posting_list)?;
            let right_bound_list = right_bound_bytes.concat_list(&right_posting_list)?;
            let bounds = EncodedBlockMax::concatenate(&[&left_bound_list, &right_bound_list])?;
            let mut postings_bytes = left_bytes.as_bytes().to_vec();
            postings_bytes.extend_from_slice(right_bytes.as_bytes());
            let postings_list =
                PostingList::parse(&postings_bytes, u32::try_from(left_count + right.len())?)?;
            let parsed = BlockMaxList::parse_with_fieldnorms(
                bounds.as_bytes(),
                &postings_list,
                fixture_fieldnorm,
            )?;
            assert_eq!(parsed.entry_count(), 2, "left_count={left_count}");
            assert_eq!(parsed.entries()[0].block_offset, 0);
            assert_eq!(
                parsed.entries()[1].block_offset,
                u64::try_from(left_bytes.as_bytes().len())?,
                "left_count={left_count}"
            );
        }
        Ok(())
    }

    #[test]
    fn blockmax_concat_rebasing_is_associative_and_checks_q1_order() -> TestResult {
        let a_postings = EncodedPostingList::encode(&sparse_postings(100, 100))?;
        let b_postings = EncodedPostingList::encode(&sparse_postings(300, 10_000))?;
        let c_postings = EncodedPostingList::encode(&sparse_postings(17, 100_000))?;
        let a_list = a_postings.posting_list()?;
        let b_list = b_postings.posting_list()?;
        let c_list = c_postings.posting_list()?;
        let a_bounds = EncodedBlockMax::encode(&a_list, fixture_fieldnorm)?;
        let b_bounds = EncodedBlockMax::encode(&b_list, fixture_fieldnorm)?;
        let c_bounds = EncodedBlockMax::encode(&c_list, fixture_fieldnorm)?;
        let a_blockmax = a_bounds.concat_list(&a_list)?;
        let b_blockmax = b_bounds.concat_list(&b_list)?;
        let c_blockmax = c_bounds.concat_list(&c_list)?;

        let direct = EncodedBlockMax::concatenate(&[&a_blockmax, &b_blockmax, &c_blockmax])?;
        let ab = EncodedBlockMax::concatenate(&[&a_blockmax, &b_blockmax])?;
        let mut ab_posting_bytes = a_postings.as_bytes().to_vec();
        ab_posting_bytes.extend_from_slice(b_postings.as_bytes());
        let ab_posting_list = PostingList::parse(&ab_posting_bytes, 400)?;
        let first_pair_blockmax = ab.concat_list(&ab_posting_list)?;
        let staged = EncodedBlockMax::concatenate(&[&first_pair_blockmax, &c_blockmax])?;
        assert_eq!(staged, direct);

        assert!(matches!(
            EncodedBlockMax::concatenate(&[&c_blockmax, &a_blockmax]),
            Err(BlockMaxError::NonAscendingConcat { .. })
        ));
        Ok(())
    }

    #[test]
    fn for_and_bitmap_bitpacked_payloads_match_scalar_reference() -> TestResult {
        let frame_input = sparse_postings(POSTINGS_PER_BLOCK, 100);
        let frame = EncodedPostingList::encode(&frame_input)?;
        let frame_bytes = frame.as_bytes();
        let doc_width = frame_bytes[8];
        let freq_width = frame_bytes[10];
        let doc_len = bitpack::packed_byte_len(POSTINGS_PER_BLOCK - 1, doc_width)?;
        let doc_payload = &frame_bytes[11..11 + doc_len];
        let mut scalar_docs = [0_u32; POSTINGS_PER_BLOCK - 1];
        let mut wide_docs = [0_u32; POSTINGS_PER_BLOCK - 1];
        unpack_scalar_into(doc_payload, doc_width, &mut scalar_docs)?;
        unpack_wide_into(doc_payload, doc_width, &mut wide_docs)?;
        assert_eq!(scalar_docs, wide_docs);
        let freq_payload = &frame_bytes[11 + doc_len..];
        let mut scalar_freqs = [0_u32; POSTINGS_PER_BLOCK];
        let mut wide_freqs = [0_u32; POSTINGS_PER_BLOCK];
        unpack_scalar_into(freq_payload, freq_width, &mut scalar_freqs)?;
        unpack_wide_into(freq_payload, freq_width, &mut wide_freqs)?;
        assert_eq!(scalar_freqs, wide_freqs);

        let bitmap_input = dense_postings(POSTINGS_PER_BLOCK, 1_000);
        let bitmap = EncodedPostingList::encode(&bitmap_input)?;
        let bitmap_bytes = bitmap.as_bytes();
        assert_eq!(bitmap_bytes[0], BITMAP_KIND);
        let bitmap_freq_width = bitmap_bytes[4 + 4 + 2 + 64 + 1];
        let bitmap_freq_payload = &bitmap_bytes[4 + 4 + 2 + 64 + 2..];
        unpack_scalar_into(bitmap_freq_payload, bitmap_freq_width, &mut scalar_freqs)?;
        unpack_wide_into(bitmap_freq_payload, bitmap_freq_width, &mut wide_freqs)?;
        assert_eq!(scalar_freqs, wide_freqs);
        Ok(())
    }

    #[test]
    fn doclen_roundtrip_quantizes_aligns_and_keeps_presence_external() -> TestResult {
        let expected_fields = [1_u16, 2];
        let content = [Some(0), Some(41), None, Some(2_013_265_944), Some(u32::MAX)];
        let title = [Some(1), Some(42), Some(0), Some(65), Some(100)];
        let inputs = [
            DocLenFieldInput::new(1, &content),
            DocLenFieldInput::new(2, &title),
        ];
        let encoded = EncodedDocLenSection::encode(10, 15, &expected_fields, &inputs)?;
        assert_eq!(encoded.field_count(), 2);
        assert_eq!(encoded.as_bytes().len(), 133);
        assert_eq!(&encoded.as_bytes()[..6], &[1, 0, 64, 0, 0, 0]);
        assert_eq!(&encoded.as_bytes()[6..12], &[2, 0, 128, 0, 0, 0]);
        assert!(encoded.as_bytes()[12..64].iter().all(|&byte| byte == 0));
        assert!(encoded.as_bytes()[69..128].iter().all(|&byte| byte == 0));

        let section = encoded.section(&expected_fields)?;
        assert_eq!(section.docid_lo(), 10);
        assert_eq!(section.docid_hi(), 15);
        assert_eq!(section.field_count(), 2);
        assert_eq!(
            section
                .fields()
                .map(DocLenField::field_ord)
                .collect::<Vec<_>>(),
            expected_fields
        );
        let content = section.field(1).ok_or("content DOCLEN field")?;
        assert_eq!(content.fieldnorm_ids(), &[0, 40, 0, 255, 255]);
        assert_eq!(content.fieldnorm_id(9), None);
        assert_eq!(content.fieldnorm_id(10), Some(0));
        assert_eq!(content.fieldnorm_id(12), Some(DOCLEN_HOLE_FIELDNORM_ID));
        assert_eq!(content.fieldnorm_id(13), Some(255));
        assert_eq!(content.fieldnorm_id(15), None);
        assert_eq!(content.decoded_fieldnorm(11), Some(40));
        assert_eq!(content.decoded_fieldnorm(13), Some(2_013_265_944));
        assert_eq!(content.fieldnorm_id(12), content.fieldnorm_id(10));
        assert_ne!(content.fieldnorm_id(12), content.fieldnorm_id(13));

        let empty = EncodedDocLenSection::encode(7, 7, &[], &[])?;
        assert!(empty.as_bytes().is_empty());
        assert_eq!(empty.section(&[])?.field_count(), 0);
        Ok(())
    }

    #[test]
    fn doclen_rejects_bad_ranges_fields_lengths_and_layout() -> TestResult {
        let lengths = [Some(1), None, Some(3)];
        let input = [DocLenFieldInput::new(1, &lengths)];
        assert!(matches!(
            EncodedDocLenSection::encode(4, 3, &[1], &input),
            Err(DocLenCodecError::InvalidDocIdRange { .. })
        ));
        assert!(matches!(
            EncodedDocLenSection::encode(0, 3, &[1, 1], &input),
            Err(DocLenCodecError::NonAscendingFields { .. })
        ));
        assert!(matches!(
            EncodedDocLenSection::encode(0, 3, &[2], &input),
            Err(DocLenCodecError::UnexpectedField { .. })
        ));
        assert!(matches!(
            EncodedDocLenSection::encode(0, 4, &[1], &input),
            Err(DocLenCodecError::ColumnLengthMismatch { .. })
        ));
        assert!(matches!(
            EncodedDocLenSection::encode_with_limits(
                0,
                3,
                &[1],
                &input,
                DocLenLimits {
                    max_fields: 1,
                    max_docid_span: 2,
                    max_section_bytes: 1_000,
                }
            ),
            Err(DocLenCodecError::ResourceLimit {
                resource: "docid span",
                ..
            })
        ));
        assert!(matches!(
            EncodedDocLenSection::encode_with_limits(
                0,
                3,
                &[1],
                &input,
                DocLenLimits {
                    max_fields: 0,
                    max_docid_span: 3,
                    max_section_bytes: 67,
                }
            ),
            Err(DocLenCodecError::ResourceLimit {
                resource: "field count",
                ..
            })
        ));
        assert!(matches!(
            EncodedDocLenSection::encode_with_limits(
                0,
                3,
                &[1],
                &input,
                DocLenLimits {
                    max_fields: 1,
                    max_docid_span: 3,
                    max_section_bytes: 66,
                }
            ),
            Err(DocLenCodecError::ResourceLimit {
                resource: "section bytes",
                actual: 67,
                limit: 66,
            })
        ));

        let encoded = EncodedDocLenSection::encode(0, 3, &[1], &input)?;
        for cut in 0..encoded.as_bytes().len() {
            assert!(
                DocLenSection::parse(&encoded.as_bytes()[..cut], 0, 3, &[1]).is_err(),
                "cut={cut}"
            );
        }
        let mut offset = encoded.as_bytes().to_vec();
        offset[2..6].copy_from_slice(&65_u32.to_le_bytes());
        assert!(matches!(
            DocLenSection::parse(&offset, 0, 3, &[1]),
            Err(DocLenCodecError::NonCanonicalOffset { .. })
        ));
        let mut padding = encoded.as_bytes().to_vec();
        padding[6] = 1;
        assert!(matches!(
            DocLenSection::parse(&padding, 0, 3, &[1]),
            Err(DocLenCodecError::NonZeroPadding { offset: 6 })
        ));
        let mut trailing = encoded.as_bytes().to_vec();
        trailing.push(0);
        assert!(matches!(
            DocLenSection::parse(&trailing, 0, 3, &[1]),
            Err(DocLenCodecError::TrailingBytes { .. })
        ));
        assert!(matches!(
            DocLenSection::parse(encoded.as_bytes(), 0, 3, &[2]),
            Err(DocLenCodecError::UnexpectedField { .. })
        ));
        Ok(())
    }

    #[test]
    fn stored_meta_roundtrip_distinguishes_absent_empty_and_opaque_bytes() -> TestResult {
        let expected_fields = [2_u16, 7];
        let large = vec![b'x'; 70_000];
        let non_utf8 = [0xff, 0x00, 0x80];
        let metadata = [
            None,
            Some(b"".as_slice()),
            Some(non_utf8.as_slice()),
            Some(large.as_slice()),
            None,
        ];
        let content = [
            Some(b"alpha".as_slice()),
            None,
            Some(b"".as_slice()),
            None,
            Some(b"omega".as_slice()),
        ];
        let inputs = [
            StoredMetaFieldInput::new(2, &metadata),
            StoredMetaFieldInput::new(7, &content),
        ];
        let encoded = EncodedStoredMetaSection::encode(100, 105, &expected_fields, &inputs)?;
        assert_eq!(encoded.field_count(), 2);
        assert_eq!(encoded.blob_bytes(), 70_013);
        assert_eq!(&encoded.as_bytes()[..2], &2_u16.to_le_bytes());
        assert_eq!(&encoded.as_bytes()[6..8], &7_u16.to_le_bytes());
        assert_eq!(
            u32::from_le_bytes(encoded.as_bytes()[2..6].try_into()?),
            u32::try_from(2 * STORED_META_DIRECTORY_ENTRY_LEN)?
        );

        let section = encoded.section(&expected_fields)?;
        assert_eq!(section.docid_lo(), 100);
        assert_eq!(section.docid_hi(), 105);
        assert_eq!(section.span(), 5);
        assert_eq!(
            section
                .fields()
                .map(StoredMetaField::field_ord)
                .collect::<Vec<_>>(),
            expected_fields
        );
        let metadata = section.field(2).ok_or("metadata field")?;
        assert!(!metadata.is_present(99));
        assert!(!metadata.is_present(100));
        assert!(metadata.is_present(101));
        assert_eq!(metadata.get(100), None);
        assert_eq!(metadata.get(101), Some(b"".as_slice()));
        assert_eq!(metadata.get(102), Some(non_utf8.as_slice()));
        assert_eq!(metadata.get(103), Some(large.as_slice()));
        assert_eq!(metadata.get(104), None);
        assert_eq!(metadata.get(105), None);
        let content = section.field(7).ok_or("content field")?;
        assert_eq!(content.get(100), Some(b"alpha".as_slice()));
        assert_eq!(content.get(102), Some(b"".as_slice()));
        assert_eq!(content.get(104), Some(b"omega".as_slice()));

        let empty = EncodedStoredMetaSection::encode(7, 7, &[], &[])?;
        assert!(empty.as_bytes().is_empty());
        assert_eq!(empty.section(&[])?.field_count(), 0);
        Ok(())
    }

    #[test]
    fn stored_meta_wire_is_exactly_packed_little_endian() -> TestResult {
        let expected_fields = [2_u16, 7];
        let field_two = [None, Some(b"".as_slice()), Some([0xff, 0x00].as_slice())];
        let field_seven = [Some(b"A".as_slice()), None, Some(b"BC".as_slice())];
        let encoded = EncodedStoredMetaSection::encode(
            40,
            43,
            &expected_fields,
            &[
                StoredMetaFieldInput::new(2, &field_two),
                StoredMetaFieldInput::new(7, &field_seven),
            ],
        )?;
        let golden = [
            0x02, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x07, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x06, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
            0x00, 0xff, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x00, 0x00, b'A', b'B', b'C',
        ];
        assert_eq!(encoded.as_bytes(), golden);

        let section = StoredMetaSection::parse(&golden, 40, 43, &expected_fields)?;
        let two = section.field(2).ok_or("field two")?;
        assert_eq!(two.get(40), None);
        assert_eq!(two.get(41), Some(b"".as_slice()));
        assert_eq!(two.get(42), Some([0xff, 0x00].as_slice()));
        let seven = section.field(7).ok_or("field seven")?;
        assert_eq!(seven.get(40), Some(b"A".as_slice()));
        assert_eq!(seven.get(41), None);
        assert_eq!(seven.get(42), Some(b"BC".as_slice()));

        let mut later_directory_offset = golden;
        later_directory_offset[8..12].copy_from_slice(&32_u32.to_le_bytes());
        assert!(matches!(
            StoredMetaSection::parse(&later_directory_offset, 40, 43, &expected_fields),
            Err(StoredMetaCodecError::TerminalOffsetMismatch { field_ord: 2, .. })
        ));
        assert!(matches!(
            StoredMetaSection::parse(&golden, 40, 43, &[2]),
            Err(StoredMetaCodecError::NonCanonicalFieldOffset { field_ord: 2, .. })
        ));
        Ok(())
    }

    #[test]
    fn stored_meta_seals_sparse_scribe_columns_without_value_staging() -> TestResult {
        use crate::schema::DEFAULT_SCHEMA;
        use crate::scribe::{IndexedFieldValue, StoredFieldValue};

        let mut accumulator = ColumnarAccumulator::new(DEFAULT_SCHEMA)?;
        accumulator.add_document_with_stored(
            1,
            &[
                IndexedFieldValue::new(0, "doc-a"),
                IndexedFieldValue::new(1, ""),
            ],
            &[StoredFieldValue::new(3, b"")],
        )?;
        let opaque = [0xff, 0x00, 0x80];
        accumulator.add_document_with_stored(
            3,
            &[IndexedFieldValue::new(0, "doc-b")],
            &[StoredFieldValue::new(3, &opaque)],
        )?;

        let encoded = EncodedStoredMetaSection::encode_accumulator(100, 105, 100, &accumulator)?;
        let expected_fields = [0_u16, 1, 2, 3, 4];
        let section = encoded.section(&expected_fields)?;
        assert_eq!(section.span(), 5);
        let ids = section.field(0).ok_or("stored id")?;
        assert_eq!(ids.get(100), None);
        assert_eq!(ids.get(101), Some(b"doc-a".as_slice()));
        assert_eq!(ids.get(102), None);
        assert_eq!(ids.get(103), Some(b"doc-b".as_slice()));
        assert_eq!(ids.get(104), None);
        let content = section.field(1).ok_or("stored content")?;
        assert_eq!(content.get(101), Some(b"".as_slice()));
        assert_eq!(content.get(103), None);
        let metadata = section.field(3).ok_or("stored metadata")?;
        assert_eq!(metadata.get(101), Some(b"".as_slice()));
        assert_eq!(metadata.get(103), Some(opaque.as_slice()));
        assert_eq!(metadata.blob(), opaque);

        assert!(matches!(
            EncodedStoredMetaSection::encode_accumulator(102, 105, 100, &accumulator),
            Err(StoredMetaCodecError::SourceDocumentOutOfRange {
                index: 0,
                docid: 101,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn stored_meta_rejects_noncanonical_bytes_and_resource_abuse() -> TestResult {
        let expected_fields = [3_u16];
        let values = [None, Some(b"x".as_slice()), Some(b"".as_slice())];
        let input = [StoredMetaFieldInput::new(3, &values)];
        assert!(matches!(
            EncodedStoredMetaSection::encode(4, 3, &expected_fields, &input),
            Err(StoredMetaCodecError::InvalidDocIdRange { .. })
        ));
        assert!(matches!(
            EncodedStoredMetaSection::encode(0, 3, &[3, 3], &input),
            Err(StoredMetaCodecError::NonAscendingFields { .. })
        ));
        assert!(matches!(
            EncodedStoredMetaSection::encode(0, 3, &[4], &input),
            Err(StoredMetaCodecError::UnexpectedField { .. })
        ));
        assert!(matches!(
            EncodedStoredMetaSection::encode(0, 4, &expected_fields, &input),
            Err(StoredMetaCodecError::ColumnLengthMismatch { .. })
        ));
        assert!(matches!(
            EncodedStoredMetaSection::encode_with_limits(
                0,
                3,
                &expected_fields,
                &input,
                StoredMetaLimits {
                    max_fields: 1,
                    max_docid_span: 3,
                    max_section_bytes: 1_000,
                    max_field_blob_bytes: 1_000,
                    max_value_bytes: 0,
                }
            ),
            Err(StoredMetaCodecError::ResourceLimit {
                resource: "value bytes",
                actual: 1,
                limit: 0,
            })
        ));

        let encoded = EncodedStoredMetaSection::encode(0, 3, &expected_fields, &input)?;
        assert!(matches!(
            EncodedStoredMetaSection::encode_with_limits(
                0,
                3,
                &expected_fields,
                &input,
                StoredMetaLimits {
                    max_docid_span: 2,
                    ..StoredMetaLimits::default()
                },
            ),
            Err(StoredMetaCodecError::ResourceLimit {
                resource: "docid span",
                actual: 3,
                limit: 2,
            })
        ));
        assert!(matches!(
            EncodedStoredMetaSection::encode_with_limits(
                0,
                3,
                &expected_fields,
                &input,
                StoredMetaLimits {
                    max_field_blob_bytes: 0,
                    ..StoredMetaLimits::default()
                },
            ),
            Err(StoredMetaCodecError::ResourceLimit {
                resource: "field blob bytes",
                actual: 1,
                limit: 0,
            })
        ));
        assert!(matches!(
            EncodedStoredMetaSection::encode_with_limits(
                0,
                3,
                &expected_fields,
                &input,
                StoredMetaLimits {
                    max_section_bytes: 23,
                    ..StoredMetaLimits::default()
                },
            ),
            Err(StoredMetaCodecError::ResourceLimit {
                resource: "section bytes",
                actual: 24,
                limit: 23,
            })
        ));

        for limits in [
            StoredMetaLimits {
                max_fields: 0,
                ..StoredMetaLimits::default()
            },
            StoredMetaLimits {
                max_docid_span: 2,
                ..StoredMetaLimits::default()
            },
            StoredMetaLimits {
                max_section_bytes: 23,
                ..StoredMetaLimits::default()
            },
            StoredMetaLimits {
                max_field_blob_bytes: 0,
                ..StoredMetaLimits::default()
            },
            StoredMetaLimits {
                max_value_bytes: 0,
                ..StoredMetaLimits::default()
            },
        ] {
            assert!(
                StoredMetaSection::parse_with_limits(
                    encoded.as_bytes(),
                    0,
                    3,
                    &expected_fields,
                    limits,
                )
                .is_err(),
                "parse limit {limits:?} must be enforced"
            );
        }

        let present_empty = [Some(b"".as_slice())];
        let empty_encoded = EncodedStoredMetaSection::encode_with_limits(
            9,
            10,
            &expected_fields,
            &[StoredMetaFieldInput::new(3, &present_empty)],
            StoredMetaLimits {
                max_value_bytes: 0,
                ..StoredMetaLimits::default()
            },
        )?;
        assert_eq!(
            empty_encoded
                .section(&expected_fields)?
                .field(3)
                .ok_or("empty field")?
                .get(9),
            Some(b"".as_slice())
        );

        for cut in 0..encoded.as_bytes().len() {
            assert!(
                StoredMetaSection::parse(&encoded.as_bytes()[..cut], 0, 3, &expected_fields)
                    .is_err(),
                "cut={cut}"
            );
        }
        let mut bad_field_offset = encoded.as_bytes().to_vec();
        bad_field_offset[2..6].copy_from_slice(&7_u32.to_le_bytes());
        assert!(matches!(
            StoredMetaSection::parse(&bad_field_offset, 0, 3, &expected_fields),
            Err(StoredMetaCodecError::NonCanonicalFieldOffset { .. })
        ));
        let presence_start = STORED_META_DIRECTORY_ENTRY_LEN;
        let offsets_start = presence_start + 1;
        let mut bad_presence_tail = encoded.as_bytes().to_vec();
        bad_presence_tail[presence_start] |= 0x80;
        assert!(matches!(
            StoredMetaSection::parse(&bad_presence_tail, 0, 3, &expected_fields),
            Err(StoredMetaCodecError::NonCanonicalPresencePadding { .. })
        ));
        let mut bad_first_offset = encoded.as_bytes().to_vec();
        bad_first_offset[offsets_start..offsets_start + 4].copy_from_slice(&1_u32.to_le_bytes());
        assert!(matches!(
            StoredMetaSection::parse(&bad_first_offset, 0, 3, &expected_fields),
            Err(StoredMetaCodecError::NonZeroFirstOffset { .. })
        ));
        let mut descending = encoded.as_bytes().to_vec();
        descending[presence_start] |= 1;
        descending[offsets_start + 4..offsets_start + 8].copy_from_slice(&1_u32.to_le_bytes());
        descending[offsets_start + 8..offsets_start + 12].copy_from_slice(&0_u32.to_le_bytes());
        assert!(matches!(
            StoredMetaSection::parse(&descending, 0, 3, &expected_fields),
            Err(StoredMetaCodecError::DescendingOffsets { .. })
        ));
        let mut absent_with_bytes = encoded.as_bytes().to_vec();
        absent_with_bytes[presence_start] &= !(1 << 1);
        assert!(matches!(
            StoredMetaSection::parse(&absent_with_bytes, 0, 3, &expected_fields),
            Err(StoredMetaCodecError::AbsentValueHasBytes { index: 1, .. })
        ));
        let mut trailing = encoded.as_bytes().to_vec();
        trailing.push(0);
        assert!(matches!(
            StoredMetaSection::parse(&trailing, 0, 3, &expected_fields),
            Err(StoredMetaCodecError::TrailingBytes { .. })
        ));
        Ok(())
    }

    #[test]
    fn stored_meta_concat_preserves_holes_blobs_and_schedule_equivalence() -> TestResult {
        let expected_fields = [3_u16];
        let first_values = [Some(b"a".as_slice()), None, Some(b"".as_slice())];
        let second_values = [Some(b"bc".as_slice()), Some(b"d".as_slice())];
        let third_values = [None, Some(b"ef".as_slice())];
        let first = EncodedStoredMetaSection::encode(
            10,
            13,
            &expected_fields,
            &[StoredMetaFieldInput::new(3, &first_values)],
        )?;
        let second = EncodedStoredMetaSection::encode(
            15,
            17,
            &expected_fields,
            &[StoredMetaFieldInput::new(3, &second_values)],
        )?;
        let third = EncodedStoredMetaSection::encode(
            17,
            19,
            &expected_fields,
            &[StoredMetaFieldInput::new(3, &third_values)],
        )?;
        let first_section = first.section(&expected_fields)?;
        let second_section = second.section(&expected_fields)?;
        let third_section = third.section(&expected_fields)?;
        let direct = EncodedStoredMetaSection::concatenate(
            &[
                first_section.clone(),
                second_section.clone(),
                third_section.clone(),
            ],
            &expected_fields,
        )?;

        let monolithic_values = [
            Some(b"a".as_slice()),
            None,
            Some(b"".as_slice()),
            None,
            None,
            Some(b"bc".as_slice()),
            Some(b"d".as_slice()),
            None,
            Some(b"ef".as_slice()),
        ];
        let monolithic = EncodedStoredMetaSection::encode(
            10,
            19,
            &expected_fields,
            &[StoredMetaFieldInput::new(3, &monolithic_values)],
        )?;
        assert_eq!(direct.as_bytes(), monolithic.as_bytes());
        let direct_section = direct.section(&expected_fields)?;
        let merged_field = direct_section.field(3).ok_or("merged stored field")?;
        assert_eq!(merged_field.blob(), b"abcdef");
        assert_eq!(merged_field.get(12), Some(b"".as_slice()));
        assert_eq!(merged_field.get(13), None);
        assert_eq!(merged_field.get(14), None);

        let staged_prefix = EncodedStoredMetaSection::concatenate(
            &[first_section.clone(), second_section.clone()],
            &expected_fields,
        )?;
        let staged_prefix_section = staged_prefix.section(&expected_fields)?;
        let staged = EncodedStoredMetaSection::concatenate(
            &[staged_prefix_section, third_section.clone()],
            &expected_fields,
        )?;
        assert_eq!(staged.as_bytes(), direct.as_bytes());

        let staged_suffix = EncodedStoredMetaSection::concatenate(
            &[second_section.clone(), third_section.clone()],
            &expected_fields,
        )?;
        let staged_suffix_section = staged_suffix.section(&expected_fields)?;
        let right_associated = EncodedStoredMetaSection::concatenate(
            &[first_section.clone(), staged_suffix_section],
            &expected_fields,
        )?;
        assert_eq!(right_associated.as_bytes(), direct.as_bytes());

        assert!(matches!(
            EncodedStoredMetaSection::concatenate(
                &[second_section.clone(), first_section.clone()],
                &expected_fields,
            ),
            Err(StoredMetaCodecError::ConcatRangeOrder { .. })
        ));
        let overlapping = EncodedStoredMetaSection::encode(
            12,
            14,
            &expected_fields,
            &[StoredMetaFieldInput::new(
                3,
                &[Some(b"x".as_slice()), Some(b"y".as_slice())],
            )],
        )?;
        assert!(matches!(
            EncodedStoredMetaSection::concatenate(
                &[first_section, overlapping.section(&expected_fields)?],
                &expected_fields,
            ),
            Err(StoredMetaCodecError::ConcatRangeOrder { .. })
        ));
        assert!(matches!(
            EncodedStoredMetaSection::concatenate(&[], &expected_fields),
            Err(StoredMetaCodecError::EmptyConcat)
        ));
        Ok(())
    }

    #[test]
    fn stored_meta_concat_rebases_multiple_fields_without_decoding_blobs() -> TestResult {
        let expected_fields = [3_u16, 8];
        let left_three = [Some(b"a".as_slice()), None];
        let left_eight = [Some(b"".as_slice()), Some(b"x".as_slice())];
        let right_three = [Some(b"b".as_slice()), Some(b"".as_slice())];
        let right_eight = [None, Some([0xff, 0x00].as_slice())];
        let left = EncodedStoredMetaSection::encode(
            0,
            2,
            &expected_fields,
            &[
                StoredMetaFieldInput::new(3, &left_three),
                StoredMetaFieldInput::new(8, &left_eight),
            ],
        )?;
        let right = EncodedStoredMetaSection::encode(
            3,
            5,
            &expected_fields,
            &[
                StoredMetaFieldInput::new(3, &right_three),
                StoredMetaFieldInput::new(8, &right_eight),
            ],
        )?;
        let left_section = left.section(&expected_fields)?;
        let right_section = right.section(&expected_fields)?;
        let merged = EncodedStoredMetaSection::concatenate(
            &[left_section.clone(), right_section.clone()],
            &expected_fields,
        )?;

        let monolithic_three = [
            Some(b"a".as_slice()),
            None,
            None,
            Some(b"b".as_slice()),
            Some(b"".as_slice()),
        ];
        let monolithic_eight = [
            Some(b"".as_slice()),
            Some(b"x".as_slice()),
            None,
            None,
            Some([0xff, 0x00].as_slice()),
        ];
        let monolithic = EncodedStoredMetaSection::encode(
            0,
            5,
            &expected_fields,
            &[
                StoredMetaFieldInput::new(3, &monolithic_three),
                StoredMetaFieldInput::new(8, &monolithic_eight),
            ],
        )?;
        assert_eq!(merged.as_bytes(), monolithic.as_bytes());
        let merged_section = merged.section(&expected_fields)?;
        assert_eq!(merged_section.field(3).ok_or("field three")?.blob(), b"ab");
        assert_eq!(
            merged_section.field(8).ok_or("field eight")?.blob(),
            [b'x', 0xff, 0x00]
        );
        assert!(matches!(
            EncodedStoredMetaSection::concatenate_with_limits(
                &[left_section, right_section],
                &expected_fields,
                StoredMetaLimits {
                    max_value_bytes: 0,
                    ..StoredMetaLimits::default()
                },
            ),
            Err(StoredMetaCodecError::ResourceLimit {
                resource: "value bytes",
                actual: 1,
                limit: 0,
            })
        ));
        Ok(())
    }

    #[test]
    fn stats_wire_is_packed_little_endian_and_uses_raw_average() -> TestResult {
        let expected_fields = [1_u16, 2];
        let rows = [FieldStats::new(1, 21, 3), FieldStats::new(2, 3, 3)];
        let encoded = EncodedStatsSection::encode(&expected_fields, &rows, 3)?;
        assert_eq!(encoded.field_count(), 2);
        assert_eq!(encoded.as_bytes().len(), 2 * STATS_ENTRY_LEN);
        assert_eq!(
            encoded.as_bytes(),
            &[
                1, 0, 21, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
                0,
            ]
        );
        let section = encoded.section(&expected_fields)?;
        assert_eq!(section.rows(), rows);
        assert_eq!(section.segment_doc_count(), 3);
        let content = section.field(1).ok_or("content STATS row")?;
        assert_eq!(
            content.average_field_length().map(f32::to_bits),
            Some(7.0_f32.to_bits())
        );

        let raw = FieldStats::new(1, 148, 3)
            .average_field_length()
            .ok_or("non-empty raw average")?;
        let decoded = [41_u32, 42, 65]
            .into_iter()
            .map(crate::contract::fieldnorm_to_id)
            .map(crate::contract::id_to_fieldnorm)
            .sum::<u32>() as f32
            / 3.0;
        assert_eq!(raw.to_bits(), (148.0_f32 / 3.0).to_bits());
        assert_ne!(raw.to_bits(), decoded.to_bits());
        Ok(())
    }

    #[test]
    fn stats_rejects_field_denominator_and_byte_corruption() -> TestResult {
        let rows = [FieldStats::new(1, 5, 2), FieldStats::new(2, 0, 2)];
        assert!(matches!(
            EncodedStatsSection::encode(&[1, 1], &rows, 2),
            Err(StatsCodecError::NonAscendingFields { .. })
        ));
        assert!(matches!(
            EncodedStatsSection::encode(&[1, 3], &rows, 2),
            Err(StatsCodecError::UnexpectedField { .. })
        ));
        let wrong_count = [FieldStats::new(1, 5, 1)];
        assert!(matches!(
            EncodedStatsSection::encode(&[1], &wrong_count, 2),
            Err(StatsCodecError::DocCountMismatch { .. })
        ));
        let mixed_denominators = [FieldStats::new(1, 5, 2), FieldStats::new(2, 0, 1)];
        assert!(matches!(
            EncodedStatsSection::encode(&[1, 2], &mixed_denominators, 2),
            Err(StatsCodecError::DocCountMismatch { .. })
        ));
        let tokens_without_docs = [FieldStats::new(1, 1, 0)];
        assert!(matches!(
            EncodedStatsSection::encode(&[1], &tokens_without_docs, 0),
            Err(StatsCodecError::TokensWithoutDocuments { .. })
        ));
        let one_row = [FieldStats::new(1, 5, 2)];
        assert!(matches!(
            EncodedStatsSection::encode_with_limits(
                &[1],
                &one_row,
                2,
                StatsLimits {
                    max_fields: 0,
                    max_section_bytes: 14,
                }
            ),
            Err(StatsCodecError::ResourceLimit {
                resource: "field count",
                ..
            })
        ));
        assert!(matches!(
            EncodedStatsSection::encode_with_limits(
                &[1],
                &one_row,
                2,
                StatsLimits {
                    max_fields: 1,
                    max_section_bytes: 13,
                }
            ),
            Err(StatsCodecError::ResourceLimit {
                resource: "section bytes",
                actual: 14,
                limit: 13,
            })
        ));

        let encoded = EncodedStatsSection::encode(&[1, 2], &rows, 2)?;
        for cut in 0..encoded.as_bytes().len() {
            assert!(matches!(
                StatsSection::parse(&encoded.as_bytes()[..cut], &[1, 2], 2),
                Err(StatsCodecError::LengthMismatch { .. })
            ));
        }
        let mut trailing = encoded.as_bytes().to_vec();
        trailing.push(0);
        assert!(matches!(
            StatsSection::parse(&trailing, &[1, 2], 2),
            Err(StatsCodecError::LengthMismatch { .. })
        ));
        let mut wrong_field = encoded.as_bytes().to_vec();
        wrong_field[0..2].copy_from_slice(&9_u16.to_le_bytes());
        assert!(matches!(
            StatsSection::parse(&wrong_field, &[1, 2], 2),
            Err(StatsCodecError::UnexpectedField { .. })
        ));
        let mut wrong_doc_count = encoded.as_bytes().to_vec();
        wrong_doc_count[10..14].copy_from_slice(&1_u32.to_le_bytes());
        assert!(matches!(
            StatsSection::parse(&wrong_doc_count, &[1, 2], 2),
            Err(StatsCodecError::DocCountMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn stats_aggregation_checked_sums_identical_field_sets() -> TestResult {
        let first_rows = [FieldStats::new(1, 21, 3), FieldStats::new(2, 3, 3)];
        let second_rows = [FieldStats::new(1, 9, 2), FieldStats::new(2, 0, 2)];
        let first = EncodedStatsSection::encode(&[1, 2], &first_rows, 3)?.section(&[1, 2])?;
        let second = EncodedStatsSection::encode(&[1, 2], &second_rows, 2)?.section(&[1, 2])?;
        let aggregate = aggregate_field_stats([&first, &second])?;
        assert_eq!(
            aggregate,
            [
                SnapshotFieldStats {
                    field_ord: 1,
                    total_tokens: 30,
                    doc_count: 5,
                },
                SnapshotFieldStats {
                    field_ord: 2,
                    total_tokens: 3,
                    doc_count: 5,
                },
            ]
        );
        assert_eq!(
            aggregate[0].average_field_length().map(f32::to_bits),
            Some(6.0_f32.to_bits())
        );
        assert!(aggregate_field_stats(std::iter::empty::<&StatsSection>())?.is_empty());
        assert_eq!(
            SnapshotFieldStats {
                field_ord: 1,
                total_tokens: 0,
                doc_count: 0,
            }
            .average_field_length(),
            None
        );

        let short_rows = [FieldStats::new(1, 9, 2)];
        let short = EncodedStatsSection::encode(&[1], &short_rows, 2)?.section(&[1])?;
        assert!(matches!(
            aggregate_field_stats([&first, &short]),
            Err(StatsCodecError::AggregateFieldCountMismatch { .. })
        ));
        let wrong_field_rows = [FieldStats::new(1, 9, 2), FieldStats::new(3, 0, 2)];
        let wrong_field =
            EncodedStatsSection::encode(&[1, 3], &wrong_field_rows, 2)?.section(&[1, 3])?;
        assert!(matches!(
            aggregate_field_stats([&first, &wrong_field]),
            Err(StatsCodecError::AggregateFieldMismatch { .. })
        ));
        let maximum_rows = [FieldStats::new(1, u64::MAX, 1)];
        let one_rows = [FieldStats::new(1, 1, 1)];
        let maximum = EncodedStatsSection::encode(&[1], &maximum_rows, 1)?.section(&[1])?;
        let one = EncodedStatsSection::encode(&[1], &one_rows, 1)?.section(&[1])?;
        assert!(matches!(
            aggregate_field_stats([&maximum, &one]),
            Err(StatsCodecError::AggregateOverflow {
                counter: "total_tokens",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn compaction_replaces_stats_and_canonically_fills_removed_doc() -> TestResult {
        let original_lengths = [Some(41), Some(42), Some(65)];
        let replacement_lengths = [Some(41), None, Some(65)];
        let original_input = [DocLenFieldInput::new(1, &original_lengths)];
        let replacement_input = [DocLenFieldInput::new(1, &replacement_lengths)];
        let original_doclen = EncodedDocLenSection::encode(0, 3, &[1], &original_input)?;
        let replacement_doclen = EncodedDocLenSection::encode(0, 3, &[1], &replacement_input)?;
        let original_ids = original_doclen
            .section(&[1])?
            .field(1)
            .ok_or("original DOCLEN field")?
            .fieldnorm_ids()
            .to_vec();
        let replacement_ids = replacement_doclen
            .section(&[1])?
            .field(1)
            .ok_or("replacement DOCLEN field")?
            .fieldnorm_ids()
            .to_vec();
        assert_eq!(replacement_ids[0], original_ids[0]);
        assert_eq!(replacement_ids[1], DOCLEN_HOLE_FIELDNORM_ID);
        assert_eq!(replacement_ids[2], original_ids[2]);

        let retained_decoded_tokens = [original_ids[0], original_ids[2]]
            .into_iter()
            .map(crate::contract::id_to_fieldnorm)
            .map(u64::from)
            .sum::<u64>();
        let replacement_rows = [FieldStats::new(1, retained_decoded_tokens, 2)];
        let replacement_stats =
            EncodedStatsSection::encode(&[1], &replacement_rows, 2)?.section(&[1])?;
        let aggregate = aggregate_field_stats([&replacement_stats])?;
        assert_eq!(aggregate[0].total_tokens, retained_decoded_tokens);
        assert_eq!(aggregate[0].doc_count, 2);
        assert_ne!(aggregate[0].total_tokens, 148);
        assert_eq!(
            aggregate[0].average_field_length().map(f32::to_bits),
            Some((retained_decoded_tokens as f32 / 2.0).to_bits())
        );
        Ok(())
    }

    #[test]
    fn arbitrary_doclen_and_stats_bytes_never_panic() {
        let mut state = 0x6d8f_3a51_c0de_f00d;
        for case in 0..1_000 {
            let length = usize::try_from(random_u32(&mut state) % 257).unwrap_or(0);
            let bytes: Vec<u8> = (0..length)
                .map(|_| random_u32(&mut state).to_le_bytes()[0])
                .collect();
            let span = u64::from(random_u32(&mut state) % 32);
            let doclen =
                std::panic::catch_unwind(|| DocLenSection::parse(&bytes, 100, 100 + span, &[1, 2]));
            assert!(
                doclen.is_ok(),
                "DOCLEN parser panic case={case} len={length}"
            );
            let stats_parse = std::panic::catch_unwind(|| StatsSection::parse(&bytes, &[1, 2], 3));
            assert!(
                stats_parse.is_ok(),
                "STATS parser panic case={case} len={length}"
            );
        }
    }

    #[test]
    fn arbitrary_bytes_never_panic_parser_or_cursor() {
        let mut state = 0x91e1_0da5_c79e_7b1d;
        for case in 0..1_000 {
            let length = usize::try_from(random_u32(&mut state) % 257).unwrap_or(0);
            let bytes: Vec<u8> = (0..length)
                .map(|_| random_u32(&mut state).to_le_bytes()[0])
                .collect();
            let expected = random_u32(&mut state) % 300;
            let parsed = std::panic::catch_unwind(|| PostingList::parse(&bytes, expected));
            assert!(parsed.is_ok(), "parser panic case={case} len={length}");
            if let Ok(Ok(list)) = parsed {
                let decoded = std::panic::catch_unwind(|| list.decode_all());
                assert!(decoded.is_ok(), "cursor panic case={case} len={length}");
            }
        }
    }
}
