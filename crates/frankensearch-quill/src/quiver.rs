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
            if block.len() == POSTINGS_PER_BLOCK {
                let first = block.first().map_or(0, |posting| posting.doc_id);
                let last = block.last().map_or(first, |posting| posting.doc_id);
                let span = u64::from(last) - u64::from(first) + 1;
                if span < 512 {
                    encode_bitmap_block(block, &mut bytes)?;
                } else {
                    encode_for_block(block, &mut bytes)?;
                }
            } else {
                encode_vint_block(block, &mut bytes)?;
            }
        }

        let expected_block_count = postings.len().div_ceil(POSTINGS_PER_BLOCK);
        let parsed = PostingList::parse_with_limits(
            &bytes,
            doc_freq,
            Self::validation_limits(postings.len(), expected_block_count),
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
