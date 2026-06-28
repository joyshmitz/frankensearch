//! Portable SIMD dot-product helpers for vector search.
#![allow(clippy::inline_always)]

use frankensearch_core::{SearchError, SearchResult};
use half::f16;
use wide::{f32x8, i8x16, i16x8, i16x16, i32x8, u16x8, u32x8};

// ── SIMD load/decode helpers ────────────────────────────────────────────────
//
// The f32 byte-decoding dot-product kernel below uses four independent f32x8
// accumulators per 32 elements. A single accumulator serializes on SIMD add
// latency; four accumulators expose enough instruction-level parallelism to run
// the loop at throughput instead.

/// Load 8 consecutive `f32` lanes from a fixed 32-element array at a const offset.
#[inline(always)]
fn load8_f32<const OFF: usize>(a: &[f32; 32]) -> f32x8 {
    f32x8::from([
        a[OFF],
        a[OFF + 1],
        a[OFF + 2],
        a[OFF + 3],
        a[OFF + 4],
        a[OFF + 5],
        a[OFF + 6],
        a[OFF + 7],
    ])
}

/// Decode 8 little-endian `f32` values from a 32-byte block to `f32x8`.
#[inline(always)]
fn decode8_f32(b: &[u8; 32]) -> f32x8 {
    f32x8::from([
        f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        f32::from_le_bytes([b[8], b[9], b[10], b[11]]),
        f32::from_le_bytes([b[12], b[13], b[14], b[15]]),
        f32::from_le_bytes([b[16], b[17], b[18], b[19]]),
        f32::from_le_bytes([b[20], b[21], b[22], b[23]]),
        f32::from_le_bytes([b[24], b[25], b[26], b[27]]),
        f32::from_le_bytes([b[28], b[29], b[30], b[31]]),
    ])
}

// ── Branchless SIMD f16 → f32 widen ─────────────────────────────────────────
//
// The f16 dot-product paths are decode-bound: per-element scalar `f16::to_f32()`
// dominates (see docs/NEGATIVE_EVIDENCE.md). This widens 8 f16 lanes at once via
// the Giesen "magic multiply" trick (a denormal-as-float multiply renormalizes
// both normals and subnormals in one shot), with a pure-integer inf/nan fixup.
//
// It is **bit-exact** to `f16::to_f32()` for every finite (incl. subnormal) and
// zero input, and maps inf→inf / nan→nan with the sign preserved (the nan
// payload's quiet bit may differ, which only affects the bits of a NaN score —
// never produced by real, finite, L2-normalized embeddings). Exhaustively
// verified over all 65 536 bit patterns by `simd_f16_widen_is_bit_exact`.
//
// Because the decoded f32 values are bit-identical for finite data and the
// accumulation order is unchanged, every dot-product score is bit-identical to
// the prior scalar-decode kernel on real corpora — no determinism/golden risk.

/// 2^112 — the Giesen magic factor that rebiases the f16 exponent.
const F16_WIDEN_MAGIC: f32 = f32::from_bits(0x7780_0000);

/// Widen 8 f16 bit-patterns (held in the low 16 bits of each `u32x8` lane) to f32.
#[inline(always)]
fn widen8_f16_lanes(h: u32x8) -> f32x8 {
    let sign = (h & u32x8::splat(0x0000_8000)) << 16_u32;
    let exp_mant = (h & u32x8::splat(0x0000_7fff)) << 13_u32;
    let scaled = bytemuck::cast::<u32x8, f32x8>(exp_mant) * f32x8::splat(F16_WIDEN_MAGIC);
    let scaled_bits = bytemuck::cast::<f32x8, u32x8>(scaled);

    // inf/nan: f16 exponent field == 0x7c00. Detect via carry out of bit 15
    // ((he + 0x0400) sets bit 15 iff he == 0x7c00), spread to a full lane mask,
    // and OR the f32 exponent up to all-ones (0xff << 23).
    let he = h & u32x8::splat(0x0000_7c00);
    let carry = (he + u32x8::splat(0x0000_0400)) & u32x8::splat(0x0000_8000);
    let infnan_mask = (carry >> 15_u32) * u32x8::splat(0xff << 23);

    bytemuck::cast::<u32x8, f32x8>((scaled_bits | infnan_mask) | sign)
}

/// Decode 8 little-endian f16 values from a 16-byte block to `f32x8` (SIMD widen).
///
/// On little-endian targets (x86/ARM — the only ones the FSVI LE format targets),
/// the 16 bytes already *are* 8 little-endian `u16` lanes, so they load straight
/// into a `u16x8` and zero-extend to `u32x8` (one SIMD load + widen) — avoiding the
/// 8 scalar `from_le_bytes` reads and the stack round-trip of building a `[u32; 8]`.
#[cfg(target_endian = "little")]
#[inline(always)]
fn widen8_f16_bytes(b: &[u8; 16]) -> f32x8 {
    let lanes = bytemuck::cast::<[u8; 16], u16x8>(*b);
    widen8_f16_lanes(u32x8::from(lanes))
}

/// Big-endian fallback: decode each `u16` explicitly as little-endian.
#[cfg(target_endian = "big")]
#[inline(always)]
fn widen8_f16_bytes(b: &[u8; 16]) -> f32x8 {
    let lanes: [u32; 8] = [
        u32::from(u16::from_le_bytes([b[0], b[1]])),
        u32::from(u16::from_le_bytes([b[2], b[3]])),
        u32::from(u16::from_le_bytes([b[4], b[5]])),
        u32::from(u16::from_le_bytes([b[6], b[7]])),
        u32::from(u16::from_le_bytes([b[8], b[9]])),
        u32::from(u16::from_le_bytes([b[10], b[11]])),
        u32::from(u16::from_le_bytes([b[12], b[13]])),
        u32::from(u16::from_le_bytes([b[14], b[15]])),
    ];
    widen8_f16_lanes(bytemuck::cast::<[u32; 8], u32x8>(lanes))
}

/// Widen 8 consecutive `f16` values (from a fixed array) to `f32x8` (SIMD widen).
#[inline(always)]
fn widen8_f16_slice(s: &[f16; 8]) -> f32x8 {
    let lanes: [u32; 8] = [
        u32::from(s[0].to_bits()),
        u32::from(s[1].to_bits()),
        u32::from(s[2].to_bits()),
        u32::from(s[3].to_bits()),
        u32::from(s[4].to_bits()),
        u32::from(s[5].to_bits()),
        u32::from(s[6].to_bits()),
        u32::from(s[7].to_bits()),
    ];
    widen8_f16_lanes(bytemuck::cast::<[u32; 8], u32x8>(lanes))
}

/// Dot product between two f32 vectors.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when slice lengths differ.
pub fn dot_product_f32_f32(a: &[f32], b: &[f32]) -> SearchResult<f32> {
    ensure_same_len(a.len(), b.len())?;
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present; lengths checked equal above.
            #[allow(unsafe_code)]
            return Ok(unsafe { dot_product_f32_f32_avx2(a, b) });
        }
    }
    Ok(dot_product_f32_f32_generic(a, b))
}

/// Hand-written AVX2 f32·f32 dot — 256-bit `vmulps`/`vaddps` mirroring the `wide`
/// kernel's 4 accumulators, `(acc0+acc1)+(acc2+acc3)` reduction (routed through
/// `f32x8::reduce_add`), 8-chunk tail, and separate-mul+add scalar tail →
/// **bit-identical** (`avx2_f32slicedot_matches_generic`). f32 has no decode win,
/// so the gain is purely the 256-bit width over the `wide` SSE2-default path.
///
/// # Safety
/// Caller must ensure `avx2` is available (the dispatch in [`dot_product_f32_f32`]
/// guarantees it) and `a.len() == b.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[must_use]
unsafe fn dot_product_f32_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let n = a.len().min(b.len());
    let groups = n / 32;
    let chunks = n / 8;
    let mut acc_arr = [0.0_f32; 8];
    // SAFETY: avx2 by contract; every load is group/chunk-bounded by `o+8 ≤ n`.
    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for g in 0..groups {
            let o = g * 32;
            acc0 = _mm256_add_ps(
                acc0,
                _mm256_mul_ps(_mm256_loadu_ps(ap.add(o)), _mm256_loadu_ps(bp.add(o))),
            );
            acc1 = _mm256_add_ps(
                acc1,
                _mm256_mul_ps(_mm256_loadu_ps(ap.add(o + 8)), _mm256_loadu_ps(bp.add(o + 8))),
            );
            acc2 = _mm256_add_ps(
                acc2,
                _mm256_mul_ps(_mm256_loadu_ps(ap.add(o + 16)), _mm256_loadu_ps(bp.add(o + 16))),
            );
            acc3 = _mm256_add_ps(
                acc3,
                _mm256_mul_ps(_mm256_loadu_ps(ap.add(o + 24)), _mm256_loadu_ps(bp.add(o + 24))),
            );
        }
        let mut sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut c = groups * 4;
        while c < chunks {
            let o = c * 8;
            sum = _mm256_add_ps(
                sum,
                _mm256_mul_ps(_mm256_loadu_ps(ap.add(o)), _mm256_loadu_ps(bp.add(o))),
            );
            c += 1;
        }
        _mm256_storeu_ps(acc_arr.as_mut_ptr(), sum);
    }
    let mut result = f32x8::from(acc_arr).reduce_add();
    for i in (chunks * 8)..n {
        result += a[i] * b[i];
    }
    result
}

/// Dot product between an f16 stored vector and an f32 query vector.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when slice lengths differ.
pub fn dot_product_f16_f32(stored: &[f16], query: &[f32]) -> SearchResult<f32> {
    ensure_same_len(stored.len(), query.len())?;
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present; lengths checked equal above.
            #[allow(unsafe_code)]
            return Ok(unsafe { dot_product_f16_f32_avx2(stored, query) });
        }
    }
    Ok(dot_product_f16_f32_generic(stored, query))
}

/// Hand-written AVX2+F16C f16-slice·f32 dot — same `vcvtph2ps` hardware decode +
/// `reduce`-through-`wide` trick as [`dot_product_f16_bytes_f32`], so it is
/// **bit-identical** to the portable kernel (the slice tail is separate mul+add,
/// matched here; proven by `avx2_f16slicedot_matches_generic`).
///
/// # Safety
/// Caller must ensure `avx2` + `f16c` are available (the dispatch in
/// [`dot_product_f16_f32`] guarantees this) and `stored.len() == query.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[allow(unsafe_code)]
#[must_use]
unsafe fn dot_product_f16_f32_avx2(stored: &[f16], query: &[f32]) -> f32 {
    use core::arch::x86_64::{
        __m128i, _mm256_add_ps, _mm256_cvtph_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps,
        _mm256_storeu_ps, _mm_loadu_si128,
    };
    let n = stored.len().min(query.len());
    let chunks = n / 8;
    let mut arr = [0.0_f32; 8];
    // SAFETY: avx2+f16c by contract; loads are `c < chunks`-bounded; `&[f16]` is
    // contiguous little-endian f16 (2 bytes/elem), so a 16-byte load is 8 f16.
    unsafe {
        let mut sum = _mm256_setzero_ps();
        for c in 0..chunks {
            let f16bits = _mm_loadu_si128(stored.as_ptr().add(c * 8).cast::<__m128i>());
            let s = _mm256_cvtph_ps(f16bits);
            let q = _mm256_loadu_ps(query.as_ptr().add(c * 8));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(s, q));
        }
        _mm256_storeu_ps(arr.as_mut_ptr(), sum);
    }
    let mut result = f32x8::from(arr).reduce_add();
    for index in (chunks * 8)..n {
        result += stored[index].to_f32() * query[index];
    }
    result
}

/// Portable (`wide`-SIMD) f16-slice·f32 dot — the AVX2+F16C-dispatch fallback and
/// the path on non-x86_64 / pre-AVX2 hosts. Exposed (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn dot_product_f16_f32_generic(stored: &[f16], query: &[f32]) -> f32 {
    let mut sum = f32x8::splat(0.0);
    let mut stored_chunks = stored.chunks_exact(8);
    let mut query_chunks = query.chunks_exact(8);

    for (stored_chunk, query_chunk) in stored_chunks.by_ref().zip(query_chunks.by_ref()) {
        let s: &[f16; 8] = stored_chunk
            .try_into()
            .expect("chunks_exact(8) yields 8 elements");
        let q: &[f32; 8] = query_chunk
            .try_into()
            .expect("chunks_exact(8) yields 8 elements");
        sum += widen8_f16_slice(s) * f32x8::from(*q);
    }

    let mut result = sum.reduce_add();
    for (s, q) in stored_chunks
        .remainder()
        .iter()
        .zip(query_chunks.remainder())
    {
        result += s.to_f32() * q;
    }
    result
}

/// Cosine similarity helper for f16 stored vectors.
///
/// Assumes both vectors are already L2-normalized and therefore returns the
/// raw dot product value.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when slice lengths differ.
pub fn cosine_similarity_f16(stored: &[f16], query: &[f32]) -> SearchResult<f32> {
    dot_product_f16_f32(stored, query)
}

/// Dot product between f16 bytes and an f32 query vector.
///
/// Avoids intermediate allocation by decoding f16s on the fly.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when `stored_bytes.len()` is not
/// exactly `query.len() * 2`.
pub fn dot_product_f16_bytes_f32(stored_bytes: &[u8], query: &[f32]) -> SearchResult<f32> {
    let dim = query.len();
    if stored_bytes.len() != dim * 2 {
        return Err(SearchError::DimensionMismatch {
            expected: dim,
            found: stored_bytes.len() / 2,
        });
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present by the runtime check above;
            // `stored_bytes.len() == dim*2` checked above.
            #[allow(unsafe_code)]
            return Ok(unsafe { dot_product_f16_bytes_f32_avx2(stored_bytes, query) });
        }
    }
    Ok(dot_product_f16_bytes_f32_generic(stored_bytes, query))
}

/// Hand-written AVX2+F16C f16·f32 dot. `vcvtph2ps` (`_mm256_cvtph_ps`) decodes 8
/// f16 in one instruction — the portable `wide` decode (`widen8_f16_bytes`) is
/// software, and this kernel is decode-bound — then the **same** separate-mul+add
/// f32 accumulation, the **same** `wide::f32x8::reduce_add` final reduction (the
/// 256-bit accumulator is round-tripped through `f32x8`), and the **same** scalar
/// `mul_add` tail as the generic kernel. f16→f32 is exact (f32 has more mantissa
/// bits), so the products / accumulation / reduction are byte-for-byte the generic
/// path → **bit-identical** (proven by `avx2_f16dot_matches_generic`).
///
/// # Safety
/// Caller must ensure `avx2` + `f16c` are available (the dispatch in
/// [`dot_product_f16_bytes_f32`] guarantees this) and `stored_bytes.len() == query.len()*2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[allow(unsafe_code)]
#[must_use]
unsafe fn dot_product_f16_bytes_f32_avx2(stored_bytes: &[u8], query: &[f32]) -> f32 {
    use core::arch::x86_64::{
        __m128i, _mm256_add_ps, _mm256_cvtph_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps,
        _mm256_storeu_ps, _mm_loadu_si128,
    };
    let dim = query.len();
    let chunks = dim / 8;
    let mut arr = [0.0_f32; 8];
    // SAFETY: avx2+f16c by contract; every load is `chunk_index < chunks`-bounded.
    unsafe {
        let mut sum = _mm256_setzero_ps();
        for chunk_index in 0..chunks {
            let f16bits =
                _mm_loadu_si128(stored_bytes.as_ptr().add(chunk_index * 16).cast::<__m128i>());
            let stored = _mm256_cvtph_ps(f16bits);
            let q = _mm256_loadu_ps(query.as_ptr().add(chunk_index * 8));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(stored, q));
        }
        _mm256_storeu_ps(arr.as_mut_ptr(), sum);
    }
    // Final reduce + scalar tail are byte-for-byte the generic path.
    let mut result = f32x8::from(arr).reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 2..];
        let val = f16::from_le_bytes([b[0], b[1]]).to_f32();
        result = val.mul_add(query[index], result);
    }
    result
}

/// Portable (`wide`-SIMD) f16-bytes·f32 dot — the AVX2+F16C-dispatch fallback and
/// the path on non-x86_64 / pre-AVX2 hosts. Exposed (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn dot_product_f16_bytes_f32_generic(stored_bytes: &[u8], query: &[f32]) -> f32 {
    let dim = query.len();
    let chunks = dim / 8;
    let mut sum = f32x8::splat(0.0);

    for chunk_index in 0..chunks {
        let byte_offset = chunk_index * 16;
        let query_offset = chunk_index * 8;

        let block: &[u8; 16] = stored_bytes[byte_offset..byte_offset + 16]
            .try_into()
            .expect("16-byte f16 block");
        let stored_chunk = widen8_f16_bytes(block);

        let q: &[f32; 8] = query[query_offset..query_offset + 8]
            .try_into()
            .expect("8-element query block");

        sum += stored_chunk * f32x8::from(*q);
    }

    let mut result = sum.reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 2..];
        let val = f16::from_le_bytes([b[0], b[1]]).to_f32();
        result = val.mul_add(query[index], result);
    }

    result
}

/// Dot product between f32 bytes and an f32 query vector.
///
/// Avoids intermediate allocation by decoding f32s on the fly.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when `stored_bytes.len()` is not
/// exactly `query.len() * 4`.
pub fn dot_product_f32_bytes_f32(stored_bytes: &[u8], query: &[f32]) -> SearchResult<f32> {
    let dim = query.len();
    if stored_bytes.len() != dim * 4 {
        return Err(SearchError::DimensionMismatch {
            expected: dim,
            found: stored_bytes.len() / 4,
        });
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present; `stored_bytes.len() == dim*4` checked.
            #[allow(unsafe_code)]
            return Ok(unsafe { dot_product_f32_bytes_f32_avx2(stored_bytes, query) });
        }
    }
    Ok(dot_product_f32_bytes_f32_generic(stored_bytes, query))
}

/// Hand-written AVX2 f32-bytes·f32 dot — 256-bit `vmulps`/`vaddps` with the SAME
/// 4-accumulator grouping, `(acc0+acc1)+(acc2+acc3)` reduction, `wide` 8-chunk
/// tail (routed through `f32x8::reduce_add`), and `mul_add` scalar tail as the
/// portable kernel. f32 LE bytes are native on x86, so an unaligned `loadu_ps` of
/// the stored bytes yields the same lanes as `decode8_f32` → **bit-identical**
/// (proven by `avx2_f32dot_matches_generic`). f32 has no decode win, so the gain
/// is purely the 256-bit width over the `wide` SSE2-default path.
///
/// # Safety
/// Caller must ensure `avx2` is available (the dispatch in
/// [`dot_product_f32_bytes_f32`] guarantees it) and `stored_bytes.len() == query.len()*4`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[must_use]
unsafe fn dot_product_f32_bytes_f32_avx2(stored_bytes: &[u8], query: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let dim = query.len();
    let groups = dim / 32;
    let chunks = dim / 8;
    let mut acc_arr = [0.0_f32; 8];
    // SAFETY: avx2 by contract; `stored_bytes.len() == dim*4`, so a `*const f32`
    // load of 8 floats at element offset o (o+8 ≤ dim) is in-bounds; `loadu_ps`
    // is unaligned. Every offset below is group/chunk-bounded by `o+8 ≤ dim`.
    unsafe {
        let sptr = stored_bytes.as_ptr().cast::<f32>();
        let qptr = query.as_ptr();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        for g in 0..groups {
            let o = g * 32;
            acc0 = _mm256_add_ps(
                acc0,
                _mm256_mul_ps(_mm256_loadu_ps(sptr.add(o)), _mm256_loadu_ps(qptr.add(o))),
            );
            acc1 = _mm256_add_ps(
                acc1,
                _mm256_mul_ps(_mm256_loadu_ps(sptr.add(o + 8)), _mm256_loadu_ps(qptr.add(o + 8))),
            );
            acc2 = _mm256_add_ps(
                acc2,
                _mm256_mul_ps(_mm256_loadu_ps(sptr.add(o + 16)), _mm256_loadu_ps(qptr.add(o + 16))),
            );
            acc3 = _mm256_add_ps(
                acc3,
                _mm256_mul_ps(_mm256_loadu_ps(sptr.add(o + 24)), _mm256_loadu_ps(qptr.add(o + 24))),
            );
        }
        let mut sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut c = groups * 4;
        while c < chunks {
            let o = c * 8;
            sum = _mm256_add_ps(
                sum,
                _mm256_mul_ps(_mm256_loadu_ps(sptr.add(o)), _mm256_loadu_ps(qptr.add(o))),
            );
            c += 1;
        }
        _mm256_storeu_ps(acc_arr.as_mut_ptr(), sum);
    }
    let mut result = f32x8::from(acc_arr).reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 4..];
        let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        result = val.mul_add(query[index], result);
    }
    result
}

/// Portable (`wide`-SIMD) f32-bytes·f32 dot — the AVX2-dispatch fallback and the
/// path on non-x86_64 / pre-AVX2 hosts. Exposed (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn dot_product_f32_bytes_f32_generic(stored_bytes: &[u8], query: &[f32]) -> f32 {
    let dim = query.len();
    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);

    // Main loop: 32 elements (128 stored bytes) per iteration, 4 accumulators.
    let groups = dim / 32;
    for g in 0..groups {
        let bo = g * 128;
        let qo = g * 32;
        let block: &[u8; 128] = stored_bytes[bo..bo + 128]
            .try_into()
            .expect("128-byte f32 block");
        let q: &[f32; 32] = query[qo..qo + 32]
            .try_into()
            .expect("32-element query block");
        acc0 +=
            decode8_f32(block[0..32].try_into().expect("32-byte sub-block")) * load8_f32::<0>(q);
        acc1 +=
            decode8_f32(block[32..64].try_into().expect("32-byte sub-block")) * load8_f32::<8>(q);
        acc2 +=
            decode8_f32(block[64..96].try_into().expect("32-byte sub-block")) * load8_f32::<16>(q);
        acc3 +=
            decode8_f32(block[96..128].try_into().expect("32-byte sub-block")) * load8_f32::<24>(q);
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);

    // Tail of full 8-element chunks not covered by the 32-wide main loop.
    let chunks = dim / 8;
    for chunk_index in (groups * 4)..chunks {
        let bo = chunk_index * 32;
        let qo = chunk_index * 8;
        let block: &[u8; 32] = stored_bytes[bo..bo + 32].try_into().expect("32-byte block");
        let q: &[f32; 8] = query[qo..qo + 8].try_into().expect("8-element block");
        sum += decode8_f32(block) * f32x8::from(*q);
    }

    let mut result = sum.reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 4..];
        let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        result = val.mul_add(query[index], result);
    }

    result
}

/// Symmetric int8 dot product (both operands int8-quantized) returning the raw
/// `i32` inner product `Σ stored[i] * query[i]`.
///
/// This is the candidate **pass-1 kernel** for an int8 ADC two-pass scan (`bd-b5wl`):
/// quantized vectors are 1 byte/elem (half the bandwidth of f16) and the multiply
/// accumulates in integer lanes. `i16::mul_widen` keeps every product in full i32
/// precision, so the only overflow bound is the i32 accumulator (a 512-dim dot of
/// ±127 values peaks at ~8.3M, far below `i32::MAX`) — exact for any realistic dim.
///
/// Lengths are assumed equal (caller-guaranteed in the scan); a short tail is
/// handled scalar. Returns the raw integer dot; the caller applies the dequant
/// scale.
///
/// Dispatches to a hand-written AVX2 kernel at runtime when the CPU supports it:
/// the portable `wide` path only reaches AVX2 with a global `+avx2` build, which
/// the published binary can't assume (it would `SIGILL` on older hosts). Because
/// the dot is integer and integer addition is associative, the 256-bit `madd`
/// reduction is **bit-identical** to the generic kernel (asserted by
/// `avx2_dot_matches_generic`).
#[must_use]
pub fn dot_i8_i8(stored: &[i8], query: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe { dot_i8_i8_avx2(stored, query) };
        }
    }
    dot_i8_i8_generic(stored, query)
}

/// Hand-written AVX2 i8·i8 dot: 256-bit `vpmaddwd` over sign-extended i16 lanes,
/// two accumulators, horizontal-summed; scalar tail.
///
/// # Safety
/// The caller must ensure the `avx2` target feature is available; the runtime
/// dispatch in [`dot_i8_i8`] guarantees this.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[must_use]
unsafe fn dot_i8_i8_avx2(stored: &[i8], query: &[i8]) -> i32 {
    use core::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_setzero_si256,
        _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
    };
    let n = stored.len().min(query.len());
    // SAFETY: avx2 is guaranteed by the caller / `dot_i8_i8` dispatch; every load
    // is `i + 32 <= n`-bounded; the scalar tail covers `n % 32`.
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut i = 0_usize;
        while i + 32 <= n {
            let s = _mm256_loadu_si256(stored.as_ptr().add(i).cast::<__m256i>());
            let q = _mm256_loadu_si256(query.as_ptr().add(i).cast::<__m256i>());
            // Sign-extend low/high 16 i8 lanes to i16, then pairwise multiply-add.
            let s_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s));
            let q_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
            let s_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(s));
            let q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(q));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(s_lo, q_lo));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(s_hi, q_hi));
            i += 32;
        }
        let acc = _mm256_add_epi32(acc0, acc1);
        // Horizontal sum of the 8 i32 lanes.
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256::<1>(acc),
        );
        let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0b01>(sum64));
        let mut result = _mm_cvtsi128_si32(sum32);
        while i < n {
            result += i32::from(stored[i]) * i32::from(query[i]);
            i += 1;
        }
        result
    }
}

/// Portable (`wide`-SIMD) i8·i8 dot — the AVX2-dispatch fallback and the path on
/// non-x86_64 / pre-AVX2 hosts. Exposed (doc-hidden) so the `dot_product` bench
/// can A/B it against the AVX2 kernel.
#[doc(hidden)]
#[must_use]
pub fn dot_i8_i8_generic(stored: &[i8], query: &[i8]) -> i32 {
    // Four independent `i32x8` accumulators break the single-accumulator integer-add
    // dependency chain. The i8→i16 decode is a cheap sign-extend (unlike the
    // decode-bound f16 dot, where extra accumulators regress — see
    // docs/NEGATIVE_EVIDENCE.md), so this kernel is sum-chain-bound and the extra ILP
    // measured ~6% (dim256) to ~16% (dim384) faster. Integer sum is associative, so
    // the result is bit-identical to the prior single-accumulator kernel.
    #[inline(always)]
    fn w8<const O: usize>(a: &[i8; 32]) -> i16x8 {
        i16x8::from([
            i16::from(a[O]),
            i16::from(a[O + 1]),
            i16::from(a[O + 2]),
            i16::from(a[O + 3]),
            i16::from(a[O + 4]),
            i16::from(a[O + 5]),
            i16::from(a[O + 6]),
            i16::from(a[O + 7]),
        ])
    }
    let mut acc0 = i32x8::splat(0);
    let mut acc1 = i32x8::splat(0);
    let mut acc2 = i32x8::splat(0);
    let mut acc3 = i32x8::splat(0);

    let mut s32 = stored.chunks_exact(32);
    let mut q32 = query.chunks_exact(32);
    for (sc, qc) in s32.by_ref().zip(q32.by_ref()) {
        let s: &[i8; 32] = sc.try_into().expect("chunks_exact(32)");
        let q: &[i8; 32] = qc.try_into().expect("chunks_exact(32)");
        acc0 += w8::<0>(s).mul_widen(w8::<0>(q));
        acc1 += w8::<8>(s).mul_widen(w8::<8>(q));
        acc2 += w8::<16>(s).mul_widen(w8::<16>(q));
        acc3 += w8::<24>(s).mul_widen(w8::<24>(q));
    }
    let mut sum = (acc0 + acc1) + (acc2 + acc3);

    // Tail: remaining full 8-chunks, then a scalar remainder.
    let mut s8 = s32.remainder().chunks_exact(8);
    let mut q8 = q32.remainder().chunks_exact(8);
    for (sc, qc) in s8.by_ref().zip(q8.by_ref()) {
        let s: &[i8; 8] = sc.try_into().expect("chunks_exact(8)");
        let q: &[i8; 8] = qc.try_into().expect("chunks_exact(8)");
        sum += i16x8::from(s.map(i16::from)).mul_widen(i16x8::from(q.map(i16::from)));
    }
    let mut result = sum.reduce_add();
    for (s, q) in s8.remainder().iter().zip(q8.remainder()) {
        result += i32::from(*s) * i32::from(*q);
    }
    result
}

/// Sign-extend the low nibble of a packed byte (4-bit two's complement → i8).
#[inline(always)]
fn nibble_lo(b: u8) -> i32 {
    i32::from((((b & 0x0F) ^ 0x08) as i8) - 8)
}

/// Sign-extend the high nibble of a packed byte (4-bit two's complement → i8).
#[inline(always)]
fn nibble_hi(b: u8) -> i32 {
    i32::from((((b >> 4) ^ 0x08) as i8) - 8)
}

/// A query pre-unpacked into per-16-byte-chunk sign-extended low/high nibble lanes
/// (+ a scalar tail), so the query nibbles are decoded **once** rather than for
/// every stored vector in a scan. See [`prepare_4bit_query`] / [`dot_4bit_prepared`].
pub struct PreparedQuery4bit {
    low: Vec<i16x16>,
    high: Vec<i16x16>,
    tail: Vec<(i32, i32)>,
}

/// Pre-unpack a packed 4-bit query (the loop-invariant operand of a scan). For each
/// 16-byte chunk, store the sign-extended low/high nibble lanes (`i16x16`); the
/// remainder bytes go to a scalar `(low, high)` tail.
pub fn prepare_4bit_query(query: &[u8]) -> PreparedQuery4bit {
    let mut chunks = query.chunks_exact(16);
    let mut low = Vec::with_capacity(query.len() / 16);
    let mut high = Vec::with_capacity(query.len() / 16);
    for qc in chunks.by_ref() {
        let qa: [u8; 16] = qc.try_into().expect("chunks_exact(16)");
        let q = i16x16::from_i8x16(i8x16::from(qa.map(|b| b as i8)));
        low.push((q << 12_i32) >> 12_i32);
        high.push((q << 8_i32) >> 12_i32);
    }
    let tail = chunks
        .remainder()
        .iter()
        .map(|&b| (nibble_lo(b), nibble_hi(b)))
        .collect();
    PreparedQuery4bit { low, high, tail }
}

/// Dot product of a packed 4-bit `stored` vector against a [`PreparedQuery4bit`].
/// The stored nibbles are decoded per call; the query was decoded once. Result is
/// exact (per-dim products ≤ 49). Identical to `dot_packed_4bit(stored, query)`.
///
/// SIMD: load 16 packed bytes → `i16x16`, extract nibbles via arithmetic
/// `(x<<12)>>12` / `(x<<8)>>12`, multiply by the prepared query lanes, and
/// accumulate vertically (flushing before any `i16` lane can overflow).
pub fn dot_4bit_prepared(stored: &[u8], query: &PreparedQuery4bit) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe { dot_4bit_prepared_avx2(stored, query) };
        }
    }
    dot_4bit_prepared_generic(stored, query)
}

/// Hand-written AVX2 4-bit prepared dot: load 16 packed bytes, sign-extend to
/// i16, arithmetic-shift out the low/high nibbles, 256-bit `vpmullw` against the
/// prepared query lanes, accumulate in i16 (flushing every 16 chunks before any
/// lane can overflow), and reduce via `vpmaddwd`. Bit-identical to the portable
/// `wide` kernel (integer, in-range — proven by `avx2_dot4bit_matches_generic`).
///
/// # Safety
/// Caller must ensure the `avx2` target feature is available (the runtime
/// dispatch in [`dot_4bit_prepared`] guarantees this).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[must_use]
unsafe fn dot_4bit_prepared_avx2(stored: &[u8], query: &PreparedQuery4bit) -> i32 {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi16, _mm256_castsi256_si128, _mm256_cvtepi8_epi16,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_mullo_epi16,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_slli_epi16, _mm256_srai_epi16,
        _mm_add_epi32, _mm_cvtsi128_si32, _mm_loadu_si128, _mm_shuffle_epi32, _mm_unpackhi_epi64,
    };
    let chunks = stored.len() / 16;
    let n = chunks.min(query.low.len());
    let mut sum = 0_i32;
    // SAFETY: avx2 guaranteed by the caller; every load is bounded by `i < n`
    // (`n ≤ stored.len()/16`); the scalar tail covers the remainder.
    unsafe {
        // Reduce 16 i16 lanes → i32 (`vpmaddwd` by ones, then hsum the 8 i32).
        let hsum16 = |acc: __m256i| -> i32 {
            let m = _mm256_madd_epi16(acc, _mm256_set1_epi16(1));
            let p128 =
                _mm_add_epi32(_mm256_castsi256_si128(m), _mm256_extracti128_si256::<1>(m));
            let p64 = _mm_add_epi32(p128, _mm_unpackhi_epi64(p128, p128));
            let p32 = _mm_add_epi32(p64, _mm_shuffle_epi32::<0b01>(p64));
            _mm_cvtsi128_si32(p32)
        };
        let mut acc = _mm256_setzero_si256();
        let mut pending = 0_usize;
        for i in 0..n {
            let sbytes = _mm_loadu_si128(stored.as_ptr().add(i * 16).cast::<__m128i>());
            let s = _mm256_cvtepi8_epi16(sbytes);
            let s_low = _mm256_srai_epi16::<12>(_mm256_slli_epi16::<12>(s));
            let s_high = _mm256_srai_epi16::<12>(_mm256_slli_epi16::<8>(s));
            // The prepared query lanes are `i16x16` (32 bytes, lane order) — load
            // their bytes directly as `__m256i`.
            let q_low = _mm256_loadu_si256(core::ptr::from_ref(&query.low[i]).cast::<__m256i>());
            let q_high = _mm256_loadu_si256(core::ptr::from_ref(&query.high[i]).cast::<__m256i>());
            let prod = _mm256_add_epi16(
                _mm256_mullo_epi16(s_low, q_low),
                _mm256_mullo_epi16(s_high, q_high),
            );
            acc = _mm256_add_epi16(acc, prod);
            pending += 1;
            if pending == 16 {
                sum += hsum16(acc);
                acc = _mm256_setzero_si256();
                pending = 0;
            }
        }
        sum += hsum16(acc);
    }
    for (sb, &(qlo, qhi)) in stored[chunks * 16..].iter().zip(&query.tail) {
        sum += nibble_lo(*sb) * qlo + nibble_hi(*sb) * qhi;
    }
    sum
}

/// Portable (`wide`-SIMD) 4-bit prepared dot — the AVX2-dispatch fallback and the
/// path on non-x86_64 / pre-AVX2 hosts. Exposed (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn dot_4bit_prepared_generic(stored: &[u8], query: &PreparedQuery4bit) -> i32 {
    let mut sum = 0_i32;
    let mut acc = i16x16::splat(0);
    let mut pending = 0_usize;
    let mut s16 = stored.chunks_exact(16);
    for (sc, (q_low, q_high)) in s16.by_ref().zip(query.low.iter().zip(&query.high)) {
        let sa: [u8; 16] = sc.try_into().expect("chunks_exact(16)");
        let s = i16x16::from_i8x16(i8x16::from(sa.map(|b| b as i8)));
        let s_low = (s << 12_i32) >> 12_i32;
        let s_high = (s << 8_i32) >> 12_i32;
        acc += s_low * *q_low + s_high * *q_high;
        pending += 1;
        if pending == 16 {
            sum += i32::from(acc.reduce_add());
            acc = i16x16::splat(0);
            pending = 0;
        }
    }
    sum += i32::from(acc.reduce_add());
    for (sb, &(qlo, qhi)) in s16.remainder().iter().zip(&query.tail) {
        sum += nibble_lo(*sb) * qlo + nibble_hi(*sb) * qhi;
    }
    sum
}

/// Dot product of two vectors stored as packed signed 4-bit nibbles: 2 dims per
/// byte, low nibble = even dim, high nibble = odd dim, each a 4-bit two's
/// complement in `[-7, 7]`. `stored` and `query` must have equal packed length.
/// Result is exact (per-dim products ≤ 49). For a scan over many stored vectors,
/// prefer [`prepare_4bit_query`] + [`dot_4bit_prepared`] to decode the query once.
pub fn dot_packed_4bit(stored: &[u8], query: &[u8]) -> i32 {
    dot_4bit_prepared(stored, &prepare_4bit_query(query))
}

/// Portable (`wide`-SIMD) f32·f32 dot — the AVX2-dispatch fallback and the path
/// on non-x86_64 / pre-AVX2 hosts. Exposed (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn dot_product_f32_f32_generic(a: &[f32], b: &[f32]) -> f32 {
    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);

    let mut a32 = a.chunks_exact(32);
    let mut b32 = b.chunks_exact(32);
    for (a_chunk, b_chunk) in a32.by_ref().zip(b32.by_ref()) {
        let a_block: &[f32; 32] = a_chunk.try_into().expect("chunks_exact(32)");
        let b_block: &[f32; 32] = b_chunk.try_into().expect("chunks_exact(32)");
        acc0 += load8_f32::<0>(a_block) * load8_f32::<0>(b_block);
        acc1 += load8_f32::<8>(a_block) * load8_f32::<8>(b_block);
        acc2 += load8_f32::<16>(a_block) * load8_f32::<16>(b_block);
        acc3 += load8_f32::<24>(a_block) * load8_f32::<24>(b_block);
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);
    let mut a8 = a32.remainder().chunks_exact(8);
    let mut b8 = b32.remainder().chunks_exact(8);

    for (a_chunk, b_chunk) in a8.by_ref().zip(b8.by_ref()) {
        let a_block: &[f32; 8] = a_chunk.try_into().expect("chunks_exact(8)");
        let b_block: &[f32; 8] = b_chunk.try_into().expect("chunks_exact(8)");
        sum += f32x8::from(*a_block) * f32x8::from(*b_block);
    }

    let mut result = sum.reduce_add();
    for (x, y) in a8.remainder().iter().zip(b8.remainder()) {
        result += x * y;
    }
    result
}

const fn ensure_same_len(expected: usize, found: usize) -> SearchResult<()> {
    if expected != found {
        return Err(SearchError::DimensionMismatch { expected, found });
    }
    Ok(())
}

/// Quantize an f16 slab to int8 with one corpus-wide max-abs scale (the lazy int8
/// ADC slab build). Runtime-dispatches to an AVX2+F16C kernel when available: the
/// build is **decode-bound** (`f16::to_f32` is software, run twice — once for
/// max-abs, once to quantize) and `round` is a per-element scalar op, both of which
/// `vcvtph2ps` + vector round crush. Falls back to the portable scalar kernel.
#[must_use]
pub fn quantize_f16_slab_to_i8(vectors_f16: &[f16]) -> Vec<i8> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe { quantize_f16_slab_to_i8_avx2(vectors_f16) };
        }
    }
    quantize_f16_slab_to_i8_generic(vectors_f16)
}

/// Hand-written AVX2+F16C int8 slab quantizer. Bit-identical to the scalar kernel:
/// `max` is exact/associative so the vector max-abs equals the scalar fold; the
/// round is `f32::round` (half-away-from-zero) emulated as `trunc(v + copysign(0.5,
/// v))` (exact for `|v| ≤ 127`); the clamp + `as i8` cast are unchanged. Proven by
/// `avx2_quantize_i8_matches_generic`.
///
/// # Safety
/// Caller must ensure `avx2` + `f16c` are available (the dispatch in
/// [`quantize_f16_slab_to_i8`] guarantees this).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[allow(unsafe_code)]
#[must_use]
unsafe fn quantize_f16_slab_to_i8_avx2(vectors_f16: &[f16]) -> Vec<i8> {
    use core::arch::x86_64::{
        _MM_FROUND_NO_EXC, _MM_FROUND_TO_ZERO, _mm256_add_ps, _mm256_and_ps, _mm256_cvtph_ps,
        _mm256_cvttps_epi32, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_or_ps,
        _mm256_round_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_storeu_si256,
        _mm_loadu_si128,
    };
    let n = vectors_f16.len();
    let chunks = n / 8;
    // `half::f16` is `repr(transparent)` over `u16`, so the slab is `n*2` LE bytes.
    // SAFETY: `vectors_f16` has `n` f16 = `n*2` bytes, contiguous.
    #[allow(unsafe_code)]
    let bytes = unsafe { std::slice::from_raw_parts(vectors_f16.as_ptr().cast::<u8>(), n * 2) };

    // Pass 1: corpus-wide max-abs (F16C decode + clear sign bit + vector max).
    let mut max_abs = 0.0_f32;
    // SAFETY: avx2+f16c by contract; loads are `c < chunks`-bounded.
    unsafe {
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7fff_ffff));
        let mut vmax = _mm256_setzero_ps();
        for c in 0..chunks {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(bytes.as_ptr().add(c * 16).cast()));
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(x, abs_mask));
        }
        let mut arr = [0.0_f32; 8];
        _mm256_storeu_ps(arr.as_mut_ptr(), vmax);
        for &v in &arr {
            max_abs = max_abs.max(v);
        }
    }
    for &x in &vectors_f16[chunks * 8..] {
        max_abs = max_abs.max(x.to_f32().abs());
    }
    if max_abs <= 0.0 {
        return vec![0_i8; n];
    }
    let scale = 127.0 / max_abs;

    // Pass 2: quantize (decode → ×scale → round-half-away → clamp → i8).
    let mut out = Vec::with_capacity(n);
    // SAFETY: avx2+f16c by contract; loads are `c < chunks`-bounded.
    unsafe {
        let vscale = _mm256_set1_ps(scale);
        let vhalf = _mm256_set1_ps(0.5);
        let vmaxc = _mm256_set1_ps(127.0);
        let vminc = _mm256_set1_ps(-127.0);
        let vsign = _mm256_set1_ps(-0.0);
        let mut tmp = [0_i32; 8];
        for c in 0..chunks {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(bytes.as_ptr().add(c * 16).cast()));
            let v = _mm256_mul_ps(x, vscale);
            // copysign(0.5, v): OR the 0.5 magnitude with v's sign bit.
            let half_signed = _mm256_or_ps(vhalf, _mm256_and_ps(v, vsign));
            let rounded = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(
                _mm256_add_ps(v, half_signed),
            );
            let clamped = _mm256_min_ps(_mm256_max_ps(rounded, vminc), vmaxc);
            let i = _mm256_cvttps_epi32(clamped);
            _mm256_storeu_si256(tmp.as_mut_ptr().cast(), i);
            for &t in &tmp {
                #[allow(clippy::cast_possible_truncation)]
                out.push(t as i8);
            }
        }
    }
    for &x in &vectors_f16[chunks * 8..] {
        #[allow(clippy::cast_possible_truncation)]
        out.push((x.to_f32() * scale).round().clamp(-127.0, 127.0) as i8);
    }
    out
}

/// Portable scalar int8 slab quantizer — the AVX2+F16C-dispatch fallback. Exposed
/// (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn quantize_f16_slab_to_i8_generic(vectors_f16: &[f16]) -> Vec<i8> {
    let max_abs = vectors_f16
        .iter()
        .map(|x| x.to_f32().abs())
        .fold(0.0_f32, f32::max);
    if max_abs <= 0.0 {
        return vec![0_i8; vectors_f16.len()];
    }
    let scale = 127.0 / max_abs;
    vectors_f16
        .iter()
        .map(|x| {
            #[allow(clippy::cast_possible_truncation)]
            let q = (x.to_f32() * scale).round().clamp(-127.0, 127.0) as i8;
            q
        })
        .collect()
}

/// Quantize one f32 to a signed 4-bit nibble (`[-7, 7]`, 4-bit two's complement in
/// the low 4 bits) given a scale — the per-element op inside the 4-bit slab build.
#[inline(always)]
fn nibble_of_4bit(value: f32, scale: f32) -> u8 {
    #[allow(clippy::cast_possible_truncation)]
    let q = (value * scale).round().clamp(-7.0, 7.0) as i8;
    (q as u8) & 0x0F
}

/// Pack a contiguous f16 slab (`count·dim`) into signed 4-bit nibbles (2 dims/byte,
/// `dim.div_ceil(2)` bytes/vector) with one corpus-wide max-abs scale — the lazy
/// 4-bit ADC slab build (the **wired-default** two-pass pass-1 storage). Like the
/// int8 slab, it is decode-bound; runtime-dispatches to AVX2+F16C when available.
#[must_use]
pub fn pack_f16_slab_to_4bit(vectors_f16: &[f16], dim: usize) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
            // SAFETY: avx2 + f16c verified present by the runtime check above.
            #[allow(unsafe_code)]
            return unsafe { pack_f16_slab_to_4bit_avx2(vectors_f16, dim) };
        }
    }
    pack_f16_slab_to_4bit_generic(vectors_f16, dim)
}

/// Hand-written AVX2+F16C 4-bit slab packer. `vcvtph2ps` decodes 8 f16/instruction
/// for both the max-abs pass and the quantize pass; the nibble values come from the
/// SAME `×scale` → round-half-away → clamp(-7,7) → `cvttps_epi32` pipeline as the
/// int8 kernel, then the (cheap) nibble packing stays scalar. Bit-identical to the
/// scalar kernel (`avx2_pack_4bit_matches_generic`).
///
/// # Safety
/// Caller must ensure `avx2` + `f16c` are available (the dispatch in
/// [`pack_f16_slab_to_4bit`] guarantees this).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[allow(unsafe_code)]
#[must_use]
unsafe fn pack_f16_slab_to_4bit_avx2(vectors_f16: &[f16], dim: usize) -> Vec<u8> {
    use core::arch::x86_64::{
        _MM_FROUND_NO_EXC, _MM_FROUND_TO_ZERO, _mm256_add_ps, _mm256_and_ps, _mm256_cvtph_ps,
        _mm256_cvttps_epi32, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_or_ps,
        _mm256_round_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_storeu_si256,
        _mm_loadu_si128,
    };
    if dim == 0 {
        return Vec::new();
    }
    let n = vectors_f16.len();
    let total_chunks = n / 8;
    // SAFETY: `half::f16` is `repr(transparent)` over `u16`; `n` f16 = `n*2` bytes.
    #[allow(unsafe_code)]
    let bytes = unsafe { std::slice::from_raw_parts(vectors_f16.as_ptr().cast::<u8>(), n * 2) };

    // Pass 1: corpus-wide max-abs (F16C decode + clear sign + vector max).
    let mut max_abs = 0.0_f32;
    // SAFETY: avx2+f16c by contract; loads are `c < total_chunks`-bounded.
    unsafe {
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7fff_ffff));
        let mut vmax = _mm256_setzero_ps();
        for c in 0..total_chunks {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(bytes.as_ptr().add(c * 16).cast()));
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(x, abs_mask));
        }
        let mut arr = [0.0_f32; 8];
        _mm256_storeu_ps(arr.as_mut_ptr(), vmax);
        for &v in &arr {
            max_abs = max_abs.max(v);
        }
    }
    for &x in &vectors_f16[total_chunks * 8..] {
        max_abs = max_abs.max(x.to_f32().abs());
    }
    let scale = if max_abs > 1e-9 { 7.0 / max_abs } else { 0.0 };

    // Pass 2: per-vector quantize → pack 2 dims/byte.
    let count = n / dim;
    let bytes_per_vector = dim.div_ceil(2);
    let mut slab = vec![0_u8; count * bytes_per_vector];
    // SAFETY: avx2+f16c by contract; within-vector loads are `d+8 ≤ dim`-bounded.
    unsafe {
        let vscale = _mm256_set1_ps(scale);
        let vhalf = _mm256_set1_ps(0.5);
        let vmaxc = _mm256_set1_ps(7.0);
        let vminc = _mm256_set1_ps(-7.0);
        let vsign = _mm256_set1_ps(-0.0);
        let mut tmp = [0_i32; 8];
        for v in 0..count {
            let base = v * dim;
            let out = v * bytes_per_vector;
            let mut d = 0;
            while d + 8 <= dim {
                let x =
                    _mm256_cvtph_ps(_mm_loadu_si128(bytes.as_ptr().add((base + d) * 2).cast()));
                let vv = _mm256_mul_ps(x, vscale);
                let half_signed = _mm256_or_ps(vhalf, _mm256_and_ps(vv, vsign));
                let rounded = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(
                    _mm256_add_ps(vv, half_signed),
                );
                let clamped = _mm256_min_ps(_mm256_max_ps(rounded, vminc), vmaxc);
                _mm256_storeu_si256(tmp.as_mut_ptr().cast(), _mm256_cvttps_epi32(clamped));
                // 8 dims (d even) → 4 fully-determined bytes: direct write (no RMW).
                for m in 0..4 {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let lo = (tmp[2 * m] as u8) & 0x0F;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let hi = (tmp[2 * m + 1] as u8) & 0x0F;
                    slab[out + d / 2 + m] = lo | (hi << 4);
                }
                d += 8;
            }
            // Scalar tail for the final < 8 dims (handles odd dim / last partial byte).
            while d < dim {
                let nib = nibble_of_4bit(vectors_f16[base + d].to_f32(), scale);
                if d % 2 == 0 {
                    slab[out + d / 2] |= nib;
                } else {
                    slab[out + d / 2] |= nib << 4;
                }
                d += 1;
            }
        }
    }
    slab
}

/// Portable scalar 4-bit slab packer — the AVX2+F16C-dispatch fallback. Exposed
/// (doc-hidden) for the bench A/B.
#[doc(hidden)]
#[must_use]
pub fn pack_f16_slab_to_4bit_generic(vectors_f16: &[f16], dim: usize) -> Vec<u8> {
    if dim == 0 {
        return Vec::new();
    }
    let max_abs = vectors_f16
        .iter()
        .map(|x| x.to_f32().abs())
        .fold(0.0_f32, f32::max);
    let scale = if max_abs > 1e-9 { 7.0 / max_abs } else { 0.0 };
    let count = vectors_f16.len() / dim;
    let bytes_per_vector = dim.div_ceil(2);
    let mut slab = vec![0_u8; count * bytes_per_vector];
    for v in 0..count {
        let base = v * dim;
        let out = v * bytes_per_vector;
        for d in 0..dim {
            let nib = nibble_of_4bit(vectors_f16[base + d].to_f32(), scale);
            if d % 2 == 0 {
                slab[out + d / 2] |= nib;
            } else {
                slab[out + d / 2] |= nib << 4;
            }
        }
    }
    slab
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The runtime AVX2 i8·i8 dot must be bit-identical to the portable `wide`
    /// kernel across every dim shape (full vectors, sub-32 tails, odd lengths) —
    /// integer addition is associative, so the 256-bit reduction equals the
    /// scalar/`wide` sum exactly. Skips when the host lacks AVX2 (dispatch falls
    /// back to the generic kernel there).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_dot_matches_generic() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next_i8 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_possible_truncation)]
            ((state >> 24) as i8)
        };
        for &dim in &[0_usize, 1, 7, 8, 31, 32, 33, 63, 64, 65, 100, 256, 384, 511, 512] {
            let s: Vec<i8> = (0..dim).map(|_| next_i8()).collect();
            let q: Vec<i8> = (0..dim).map(|_| next_i8()).collect();
            let generic = dot_i8_i8_generic(&s, &q);
            // SAFETY: avx2 verified present above.
            #[allow(unsafe_code)]
            let avx2 = unsafe { dot_i8_i8_avx2(&s, &q) };
            assert_eq!(generic, avx2, "dim={dim}");
        }
    }

    /// The runtime AVX2 4-bit prepared dot must be bit-identical to the portable
    /// `wide` kernel across packed-length shapes (full 16-byte chunks, sub-chunk
    /// tails) — integer/in-range, so the 256-bit accumulation equals the `wide`
    /// sum exactly. Skips when the host lacks AVX2.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_dot4bit_matches_generic() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0xdead_beef_cafe_1234_u64;
        let mut next_u8 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_possible_truncation)]
            ((state >> 24) as u8)
        };
        for &len in &[0_usize, 1, 7, 15, 16, 17, 31, 32, 33, 100, 192, 256] {
            let q: Vec<u8> = (0..len).map(|_| next_u8()).collect();
            let s: Vec<u8> = (0..len).map(|_| next_u8()).collect();
            let prepared = prepare_4bit_query(&q);
            let generic = dot_4bit_prepared_generic(&s, &prepared);
            // SAFETY: avx2 verified present above.
            #[allow(unsafe_code)]
            let avx2 = unsafe { dot_4bit_prepared_avx2(&s, &prepared) };
            assert_eq!(generic, avx2, "len={len}");
        }
    }

    /// The runtime AVX2+F16C f16·f32 dot must be **bit-identical** to the portable
    /// `wide` kernel for finite inputs (f16→f32 is exact; the accumulation and
    /// `reduce_add` reductions are byte-for-byte the same). Embeddings are always
    /// finite, so the test uses finite f16 (NaN payloads are the only legitimate
    /// divergence and never occur in real vectors). Skips without AVX2+F16C.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_f16dot_matches_generic() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")) {
            return;
        }
        let mut state = 0x1357_9bdf_2468_ace0_u64;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
        };
        for &dim in &[1_usize, 7, 8, 9, 16, 17, 31, 64, 100, 256, 384, 512] {
            let q: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
            let mut bytes = Vec::with_capacity(dim * 2);
            for _ in 0..dim {
                bytes.extend_from_slice(&f16::from_f32(next_f32()).to_le_bytes());
            }
            let generic = dot_product_f16_bytes_f32_generic(&bytes, &q);
            // SAFETY: avx2+f16c verified present above.
            #[allow(unsafe_code)]
            let avx2 = unsafe { dot_product_f16_bytes_f32_avx2(&bytes, &q) };
            assert_eq!(generic.to_bits(), avx2.to_bits(), "dim={dim}");
        }
    }

    /// The runtime AVX2+F16C f16-slice dot must be bit-identical to the portable
    /// `wide` kernel for finite inputs (same exact decode + reduce-through-`wide`).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_f16slicedot_matches_generic() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")) {
            return;
        }
        let mut state = 0x2468_ace0_1357_9bdf_u64;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
        };
        for &dim in &[1_usize, 7, 8, 9, 16, 17, 31, 64, 100, 256, 384, 512] {
            let q: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
            let s: Vec<f16> = (0..dim).map(|_| f16::from_f32(next_f32())).collect();
            let generic = dot_product_f16_f32_generic(&s, &q);
            // SAFETY: avx2+f16c verified present above.
            #[allow(unsafe_code)]
            let avx2 = unsafe { dot_product_f16_f32_avx2(&s, &q) };
            assert_eq!(generic.to_bits(), avx2.to_bits(), "dim={dim}");
        }
    }

    /// The runtime AVX2 f32-bytes dot must be bit-identical to the portable `wide`
    /// kernel (same 4-accumulator grouping + reduce-through-`wide` + `mul_add`
    /// tail; f32 LE bytes are native on x86). Skips without AVX2.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_f32dot_matches_generic() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
        };
        for &dim in &[1_usize, 7, 8, 9, 16, 31, 32, 33, 64, 100, 256, 384, 512] {
            let q: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
            let mut bytes = Vec::with_capacity(dim * 4);
            for _ in 0..dim {
                bytes.extend_from_slice(&next_f32().to_le_bytes());
            }
            let generic = dot_product_f32_bytes_f32_generic(&bytes, &q);
            // SAFETY: avx2 verified present above.
            #[allow(unsafe_code)]
            let avx2 = unsafe { dot_product_f32_bytes_f32_avx2(&bytes, &q) };
            assert_eq!(generic.to_bits(), avx2.to_bits(), "dim={dim}");
        }
    }

    /// The runtime AVX2 f32-slice dot must be bit-identical to the portable `wide`
    /// kernel (same 4-accumulator grouping + reduce-through-`wide` + separate
    /// mul+add tail). Skips without AVX2.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_f32slicedot_matches_generic() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x7c6d_5e4f_3a2b_1908_u64;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
        };
        for &dim in &[1_usize, 7, 8, 9, 16, 31, 32, 33, 64, 100, 256, 384, 512] {
            let a: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
            let b: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
            let generic = dot_product_f32_f32_generic(&a, &b);
            // SAFETY: avx2 verified present above.
            #[allow(unsafe_code)]
            let avx2 = unsafe { dot_product_f32_f32_avx2(&a, &b) };
            assert_eq!(generic.to_bits(), avx2.to_bits(), "dim={dim}");
        }
    }

    /// The runtime AVX2+F16C int8 slab quantizer must be byte-for-byte identical to
    /// the scalar kernel for finite inputs (`max` is exact, round-half-away is
    /// emulated exactly, clamp/cast unchanged). Skips without AVX2+F16C.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_quantize_i8_matches_generic() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")) {
            return;
        }
        let mut state = 0x51ed_270b_9c4d_a3f8_u64;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
        };
        // Various lengths incl. sub-8 tails + a zero-vector edge (max_abs == 0).
        for &n in &[0_usize, 1, 7, 8, 9, 16, 31, 100, 384, 769] {
            let v: Vec<f16> = (0..n).map(|_| f16::from_f32(next_f32() * 3.0)).collect();
            let generic = quantize_f16_slab_to_i8_generic(&v);
            // SAFETY: avx2+f16c verified present above.
            #[allow(unsafe_code)]
            let avx2 = unsafe { quantize_f16_slab_to_i8_avx2(&v) };
            assert_eq!(generic, avx2, "n={n}");
        }
        let zeros = vec![f16::from_f32(0.0); 40];
        // SAFETY: avx2+f16c verified present above.
        #[allow(unsafe_code)]
        let avx2_zero = unsafe { quantize_f16_slab_to_i8_avx2(&zeros) };
        assert_eq!(quantize_f16_slab_to_i8_generic(&zeros), avx2_zero);
    }

    /// The runtime AVX2+F16C 4-bit slab packer must be byte-for-byte identical to
    /// the scalar kernel across dim shapes (full 8-chunks, sub-8 tails incl. odd
    /// dim) and multiple vectors. Skips without AVX2+F16C.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_pack_4bit_matches_generic() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")) {
            return;
        }
        let mut state = 0xa1b2_c3d4_e5f6_0789_u64;
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            (((state >> 40) as f32 / (1_u64 << 23) as f32) - 1.0)
        };
        for &dim in &[1_usize, 2, 7, 8, 9, 15, 16, 17, 33, 64, 100, 384] {
            for &count in &[1_usize, 3] {
                let v: Vec<f16> = (0..dim * count)
                    .map(|_| f16::from_f32(next_f32() * 2.5))
                    .collect();
                let generic = pack_f16_slab_to_4bit_generic(&v, dim);
                // SAFETY: avx2+f16c verified present above.
                #[allow(unsafe_code)]
                let avx2 = unsafe { pack_f16_slab_to_4bit_avx2(&v, dim) };
                assert_eq!(generic, avx2, "dim={dim} count={count}");
            }
        }
    }

    /// Exhaustive proof that the SIMD f16->f32 widen matches the scalar reference
    /// (`f16::to_f32`) for **every** one of the 65 536 f16 bit patterns. Finite,
    /// zero and subnormal inputs must be bit-identical; inf/nan must map to
    /// inf/nan with matching sign (the nan payload's quiet bit may legitimately
    /// differ between the two decode methods).
    #[test]
    fn simd_f16_widen_is_bit_exact() {
        for base in (0u32..=0xFFFF).step_by(8) {
            let lanes = [
                base,
                base + 1,
                base + 2,
                base + 3,
                base + 4,
                base + 5,
                base + 6,
                base + 7,
            ];
            let widened = widen8_f16_lanes(bytemuck::cast::<[u32; 8], u32x8>(lanes));
            let out = bytemuck::cast::<f32x8, [f32; 8]>(widened);
            for (lane, &simd) in lanes.iter().zip(out.iter()) {
                let bits = u16::try_from(*lane).expect("<= 0xFFFF");
                let scalar = f16::from_bits(bits).to_f32();
                if scalar.is_nan() {
                    assert!(simd.is_nan(), "bits={bits:#06x}: scalar nan, simd={simd}");
                    assert_eq!(
                        scalar.is_sign_negative(),
                        simd.is_sign_negative(),
                        "bits={bits:#06x}: nan sign mismatch"
                    );
                } else {
                    assert_eq!(
                        scalar.to_bits(),
                        simd.to_bits(),
                        "bits={bits:#06x}: scalar={scalar} simd={simd}"
                    );
                }
            }
        }
    }

    /// The 16-byte SIMD load path must decode to exactly the same lanes as the
    /// scalar reference for representative patterns (sign, subnormal, zero, large,
    /// inf, nan), proving the little-endian byte reinterpretation is correct.
    #[test]
    fn simd_f16_bytes_load_matches_scalar() {
        let samples: [f16; 8] = [
            f16::from_f32(0.0),
            f16::from_f32(-0.0),
            f16::from_f32(1.5),
            f16::from_f32(-2.25),
            f16::from_bits(0x0001), // smallest positive subnormal
            f16::from_f32(65504.0), // f16::MAX
            f16::INFINITY,
            f16::NAN,
        ];
        let mut bytes = [0u8; 16];
        for (i, v) in samples.iter().enumerate() {
            bytes[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
        let out = bytemuck::cast::<f32x8, [f32; 8]>(widen8_f16_bytes(&bytes));
        for (v, &simd) in samples.iter().zip(out.iter()) {
            let scalar = v.to_f32();
            if scalar.is_nan() {
                assert!(simd.is_nan(), "expected nan for {v:?}");
            } else {
                assert_eq!(scalar.to_bits(), simd.to_bits(), "mismatch for {v:?}");
            }
        }
    }

    #[test]
    fn dot_i8_i8_matches_scalar() {
        fn scalar(a: &[i8], b: &[i8]) -> i32 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| i32::from(x) * i32::from(y))
                .sum()
        }
        // Lengths that exercise the 8-wide loop, a partial tail, and extremes.
        let to_i8 = |i: usize, m: usize| -> i8 {
            i8::try_from(i32::try_from(i * m % 255).expect("< 255") - 127).expect("in range")
        };
        for len in [0_usize, 1, 7, 8, 9, 16, 17, 256, 384, 385] {
            let a: Vec<i8> = (0..len).map(|i| to_i8(i, 37)).collect();
            let b: Vec<i8> = (0..len).map(|i| to_i8(i, 53)).collect();
            assert_eq!(dot_i8_i8(&a, &b), scalar(&a, &b), "len={len}");
        }
        // Worst-case magnitude: all -128 * -128 over 512 dims must not overflow i32.
        let a = vec![i8::MIN; 512];
        assert_eq!(dot_i8_i8(&a, &a), 512 * 128 * 128);
    }

    #[test]
    fn dot_packed_4bit_matches_scalar() {
        fn lo(b: u8) -> i32 {
            i32::from((((b & 0x0F) ^ 0x08) as i8) - 8)
        }
        fn hi(b: u8) -> i32 {
            i32::from((((b >> 4) ^ 0x08) as i8) - 8)
        }
        fn scalar(s: &[u8], q: &[u8]) -> i32 {
            s.iter()
                .zip(q)
                .map(|(&a, &b)| lo(a) * lo(b) + hi(a) * hi(b))
                .sum()
        }
        // Lengths exercising the 16-wide loop, a partial tail, and extremes.
        for len in [0_usize, 1, 5, 15, 16, 17, 32, 33, 192, 193] {
            let s: Vec<u8> = (0..len).map(|i| ((i * 37 + 11) % 256) as u8).collect();
            let q: Vec<u8> = (0..len).map(|i| ((i * 53 + 7) % 256) as u8).collect();
            assert_eq!(dot_packed_4bit(&s, &q), scalar(&s, &q), "len={len}");
            let prepared = prepare_4bit_query(&q);
            assert_eq!(
                dot_4bit_prepared(&s, &prepared),
                scalar(&s, &q),
                "prepared len={len}"
            );
        }
        // All nibbles = -7 (0x99 byte): each dim contributes 49.
        let a = vec![0x99_u8; 16];
        assert_eq!(dot_packed_4bit(&a, &a), 32 * 49);
        let prepared = prepare_4bit_query(&a);
        assert_eq!(dot_4bit_prepared(&a, &prepared), 32 * 49);
    }

    /// End-to-end quality gate for the int8 ADC two-pass (`bd-b5wl`): does an int8
    /// pass-1 (top `k*mult` candidates) + exact f16 rescore recover the true f16
    /// top-k? Self-contained over random L2-normalized vectors. Prints the measured
    /// recall@10 (run with `-- --nocapture`) and asserts a conservative floor.
    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn int8_two_pass_recall_at_10() {
        fn xorshift(s: &mut u64) -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            ((*s >> 40) as f32 / (1_u64 << 24) as f32).mul_add(2.0, -1.0)
        }
        fn normalize(v: &mut [f32]) {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 1e-9 {
                for x in v.iter_mut() {
                    *x /= n;
                }
            }
        }
        fn quant_i8(x: f32) -> i8 {
            let s = (x * 127.0).round();
            if s >= 127.0 {
                127
            } else if s <= -127.0 {
                -127
            } else {
                s as i8
            }
        }

        let (dim, n, k, queries) = (128_usize, 3000_usize, 10_usize, 25_usize);
        // recall@k for a given mult == fraction of the exact top-k that land in the
        // int8 top-(k*mult) candidate set (pass-2 rescores exactly, so any candidate
        // that is truly top-k is recovered). Monotonic in mult; sweep to find the
        // smallest candidate budget that still holds recall (smaller => less select).
        let mults = [2_usize, 3, 5, 10, 20];
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut vecs_f16: Vec<Vec<f16>> = Vec::with_capacity(n);
        let mut vecs_i8: Vec<Vec<i8>> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| xorshift(&mut state)).collect();
            normalize(&mut v);
            vecs_f16.push(v.iter().map(|&x| f16::from_f32(x)).collect());
            vecs_i8.push(v.iter().map(|&x| quant_i8(x)).collect());
        }

        let mut recall_sums = vec![0.0_f64; mults.len()];
        for _ in 0..queries {
            let mut q: Vec<f32> = (0..dim).map(|_| xorshift(&mut state)).collect();
            normalize(&mut q);
            let qi8: Vec<i8> = q.iter().map(|&x| quant_i8(x)).collect();

            let mut exact: Vec<(f32, usize)> = vecs_f16
                .iter()
                .enumerate()
                .map(|(i, fv)| (dot_product_f16_f32(fv, &q).expect("dot"), i))
                .collect();
            exact.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let exact_set: std::collections::HashSet<usize> =
                exact[..k].iter().map(|&(_, i)| i).collect();

            let mut p1: Vec<(i32, usize)> = vecs_i8
                .iter()
                .enumerate()
                .map(|(i, iv)| (dot_i8_i8(iv, &qi8), i))
                .collect();
            p1.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

            for (mi, &mult) in mults.iter().enumerate() {
                let cand = (k * mult).min(p1.len());
                let cand_set: std::collections::HashSet<usize> =
                    p1[..cand].iter().map(|&(_, i)| i).collect();
                let hit = exact_set.iter().filter(|i| cand_set.contains(i)).count();
                recall_sums[mi] += hit as f64 / k as f64;
            }
        }
        for (mi, &mult) in mults.iter().enumerate() {
            let avg = recall_sums[mi] / queries as f64;
            println!("int8 two-pass recall@{k} mult={mult} (n={n}, dim={dim}): {avg:.4}");
        }
        // Gate on the largest mult; smaller mults are reported for tuning.
        let avg_max = recall_sums[mults.len() - 1] / queries as f64;
        assert!(
            avg_max >= 0.80,
            "int8 two-pass recall@{k} too low: {avg_max:.4}"
        );
    }

    /// Viability probe for a **binary-quantization** first pass (Meilisearch-style):
    /// pack `sign(x_i)` to bits, rank by Hamming agreement (`popcnt` — fast even on
    /// SSE2, 1/16 the bytes of f16), then exact f16 rescore. Reports recall@10 vs the
    /// candidate budget so we know whether binary ADC is worth building (it is much
    /// coarser than int8, so the question is how big a `mult` it needs).
    #[test]
    fn binary_quant_recall_at_10() {
        fn xorshift(s: &mut u64) -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            ((*s >> 40) as f32 / (1_u64 << 24) as f32).mul_add(2.0, -1.0)
        }
        fn normalize(v: &mut [f32]) {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 1e-9 {
                for x in v.iter_mut() {
                    *x /= n;
                }
            }
        }
        fn pack_bits(v: &[f32]) -> Vec<u64> {
            let mut bits = vec![0_u64; v.len().div_ceil(64)];
            for (i, &x) in v.iter().enumerate() {
                if x >= 0.0 {
                    bits[i / 64] |= 1_u64 << (i % 64);
                }
            }
            bits
        }
        // Lower hamming distance == more sign agreement == more similar.
        fn hamming(a: &[u64], b: &[u64]) -> u32 {
            a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
        }

        let (dim, n, k, queries) = (128_usize, 3000_usize, 10_usize, 25_usize);
        let mults = [5_usize, 10, 20, 50, 100];
        let mut state = 0x2468_ace0_1357_9bdf_u64;
        let mut vecs_f16: Vec<Vec<f16>> = Vec::with_capacity(n);
        let mut vecs_bits: Vec<Vec<u64>> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| xorshift(&mut state)).collect();
            normalize(&mut v);
            vecs_bits.push(pack_bits(&v));
            vecs_f16.push(v.iter().map(|&x| f16::from_f32(x)).collect());
        }

        let mut recall_sums = vec![0.0_f64; mults.len()];
        for _ in 0..queries {
            let mut q: Vec<f32> = (0..dim).map(|_| xorshift(&mut state)).collect();
            normalize(&mut q);
            let qbits = pack_bits(&q);

            let mut exact: Vec<(f32, usize)> = vecs_f16
                .iter()
                .enumerate()
                .map(|(i, fv)| (dot_product_f16_f32(fv, &q).expect("dot"), i))
                .collect();
            exact.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let exact_set: std::collections::HashSet<usize> =
                exact[..k].iter().map(|&(_, i)| i).collect();

            // Rank by ascending hamming (descending agreement), index tie-break.
            let mut ranked: Vec<(u32, usize)> = vecs_bits
                .iter()
                .enumerate()
                .map(|(i, bv)| (hamming(bv, &qbits), i))
                .collect();
            ranked.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

            for (mi, &mult) in mults.iter().enumerate() {
                let cand = (k * mult).min(ranked.len());
                let cand_set: std::collections::HashSet<usize> =
                    ranked[..cand].iter().map(|&(_, i)| i).collect();
                let hit = exact_set.iter().filter(|i| cand_set.contains(i)).count();
                recall_sums[mi] += hit as f64 / k as f64;
            }
        }
        for (mi, &mult) in mults.iter().enumerate() {
            let avg = recall_sums[mi] / queries as f64;
            println!("binary-quant recall@{k} mult={mult} (n={n}, dim={dim}): {avg:.4}");
        }
        // No hard gate — this is a viability probe; just keep it from silently
        // returning 0 (which would signal a logic error).
        assert!(recall_sums[mults.len() - 1] / queries as f64 > 0.0);
    }

    fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn scalar_dot_f16(stored: &[f16], query: &[f32]) -> f32 {
        stored.iter().zip(query).map(|(x, y)| x.to_f32() * y).sum()
    }

    fn normalize(vec: &[f32]) -> Vec<f32> {
        let norm = vec.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm < f32::EPSILON {
            return vec.to_vec();
        }
        vec.iter().map(|value| value / norm).collect()
    }

    #[test]
    fn simd_matches_scalar_f32() {
        let a = vec![
            0.4, -0.1, 0.6, 0.2, -0.3, 0.8, 0.7, -0.5, 0.9, -0.6, 0.11, 0.25, 0.41, -0.72, 0.55,
            0.31,
        ];
        let b = vec![
            -0.8, 0.7, 0.6, -0.2, 0.3, 0.9, -0.4, 0.1, 0.12, 0.21, -0.14, 0.75, -0.22, 0.35, 0.66,
            -0.19,
        ];
        let simd = dot_product_f32_f32(&a, &b).expect("dot product");
        let scalar = scalar_dot_f32(&a, &b);
        assert!((simd - scalar).abs() < 1e-6, "simd={simd}, scalar={scalar}");
    }

    #[test]
    fn simd_matches_scalar_f16() {
        let query = vec![
            0.4, -0.1, 0.6, 0.2, -0.3, 0.8, 0.7, -0.5, 0.9, -0.6, 0.11, 0.25, 0.41, -0.72, 0.55,
            0.31,
        ];
        let stored = vec![
            f16::from_f32(-0.8),
            f16::from_f32(0.7),
            f16::from_f32(0.6),
            f16::from_f32(-0.2),
            f16::from_f32(0.3),
            f16::from_f32(0.9),
            f16::from_f32(-0.4),
            f16::from_f32(0.1),
            f16::from_f32(0.12),
            f16::from_f32(0.21),
            f16::from_f32(-0.14),
            f16::from_f32(0.75),
            f16::from_f32(-0.22),
            f16::from_f32(0.35),
            f16::from_f32(0.66),
            f16::from_f32(-0.19),
        ];
        let simd = dot_product_f16_f32(&stored, &query).expect("dot product");
        let scalar = scalar_dot_f16(&stored, &query);
        assert!((simd - scalar).abs() < 1e-6, "simd={simd}, scalar={scalar}");
    }

    #[test]
    fn remainder_elements_are_handled() {
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let b = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let simd = dot_product_f32_f32(&a, &b).expect("dot product");
        let scalar = scalar_dot_f32(&a, &b);
        assert!((simd - scalar).abs() < 1e-6, "simd={simd}, scalar={scalar}");
    }

    #[test]
    fn zero_vector_dot_product_is_zero() {
        let stored = vec![f16::from_f32(0.0); 16];
        let query = vec![1.0; 16];
        let result = dot_product_f16_f32(&stored, &query).expect("dot product");
        assert!(result.abs() < f32::EPSILON);
    }

    #[test]
    fn nan_input_propagates_nan() {
        let mut a = vec![1.0; 16];
        a[3] = f32::NAN;
        let b = vec![1.0; 16];
        let result = dot_product_f32_f32(&a, &b).expect("dot product");
        assert!(result.is_nan());
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let a = vec![1.0; 8];
        let b = vec![1.0; 7];
        let err = dot_product_f32_f32(&a, &b).expect_err("must fail");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 8,
                found: 7
            }
        ));
    }

    #[test]
    fn f16_precision_error_is_bounded_for_unit_vectors() {
        let pattern = [
            0.11_f32, -0.07, 0.19, 0.02, -0.13, 0.23, 0.31, -0.17, 0.05, -0.29, 0.37, 0.41,
        ];
        let mut stored_full = Vec::with_capacity(384);
        let mut query = Vec::with_capacity(384);
        for index in 0..384 {
            let value = pattern[index % pattern.len()];
            let other = pattern[(index + 3) % pattern.len()];
            stored_full.push(value);
            query.push(other);
        }
        let stored_full = normalize(&stored_full);
        let query = normalize(&query);
        let stored_f16: Vec<f16> = stored_full.iter().copied().map(f16::from_f32).collect();

        let f32_dot = scalar_dot_f32(&stored_full, &query);
        let f16_dot = dot_product_f16_f32(&stored_f16, &query).expect("dot product");
        assert!(
            (f32_dot - f16_dot).abs() < 0.01,
            "f32_dot={f32_dot}, f16_dot={f16_dot}"
        );
    }

    // ─── bd-1l4g tests begin ───

    #[test]
    fn cosine_similarity_f16_matches_dot_product() {
        let stored: Vec<f16> = (0_u16..16)
            .map(|i| f16::from_f32(f32::from(i) * 0.1))
            .collect();
        let query: Vec<f32> = (0_u16..16).map(|i| f32::from(i) * 0.2).collect();

        let cosine = cosine_similarity_f16(&stored, &query).expect("cosine");
        let dot = dot_product_f16_f32(&stored, &query).expect("dot");
        assert!(
            (cosine - dot).abs() < f32::EPSILON,
            "cosine_similarity_f16 should delegate to dot_product_f16_f32"
        );
    }

    #[test]
    fn cosine_similarity_f16_dimension_mismatch() {
        let stored = vec![f16::from_f32(1.0); 8];
        let query = vec![1.0_f32; 9];
        let err = cosine_similarity_f16(&stored, &query).expect_err("must fail");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 8,
                found: 9
            }
        ));
    }

    #[test]
    fn dot_product_f16_f32_dimension_mismatch() {
        let stored = vec![f16::from_f32(1.0); 4];
        let query = vec![1.0_f32; 5];
        let err = dot_product_f16_f32(&stored, &query).expect_err("must fail");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 4,
                found: 5
            }
        ));
    }

    #[test]
    fn empty_vectors_dot_product_f32() {
        let result = dot_product_f32_f32(&[], &[]).expect("dot product");
        assert!(result.abs() < f32::EPSILON);
    }

    #[test]
    fn empty_vectors_dot_product_f16() {
        let result = dot_product_f16_f32(&[], &[]).expect("dot product");
        assert!(result.abs() < f32::EPSILON);
    }

    #[test]
    fn exactly_eight_elements_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let simd = dot_product_f32_f32(&a, &b).expect("dot product");
        let scalar = scalar_dot_f32(&a, &b);
        assert!(
            (simd - scalar).abs() < 1e-6,
            "exactly 8 elements (one full SIMD chunk, no remainder)"
        );
    }

    #[test]
    fn single_element_dot_product() {
        let a = vec![3.0_f32];
        let b = vec![4.0_f32];
        let result = dot_product_f32_f32(&a, &b).expect("dot product");
        assert!((result - 12.0).abs() < f32::EPSILON);
    }

    #[test]
    fn self_dot_product_is_norm_squared() {
        let v = vec![3.0_f32, 4.0];
        let result = dot_product_f32_f32(&v, &v).expect("dot product");
        assert!((result - 25.0).abs() < f32::EPSILON); // 3^2 + 4^2 = 25
    }

    #[test]
    fn f16_nan_propagates() {
        let stored = vec![
            f16::from_f32(1.0),
            f16::NAN,
            f16::from_f32(1.0),
            f16::from_f32(1.0),
        ];
        let query = vec![1.0_f32; 4];
        let result = dot_product_f16_f32(&stored, &query).expect("dot product");
        assert!(result.is_nan());
    }

    #[test]
    fn large_256d_matches_scalar_f32() {
        let a: Vec<f32> = (0_u16..256).map(|i| (f32::from(i) * 0.01).sin()).collect();
        let b: Vec<f32> = (0_u16..256).map(|i| (f32::from(i) * 0.02).cos()).collect();
        let simd = dot_product_f32_f32(&a, &b).expect("dot product");
        let scalar = scalar_dot_f32(&a, &b);
        assert!(
            (simd - scalar).abs() < 1e-4,
            "256d: simd={simd}, scalar={scalar}"
        );
    }

    // ─── bd-1l4g tests end ───
}
