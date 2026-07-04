//! Small runtime-dispatched SIMD helpers for hot element-wise float loops.
//!
//! The workspace builds without a global `+avx2`, so LLVM auto-vectorizes these
//! loops only to the SSE2 baseline (16-byte ops). These helpers runtime-detect
//! AVX2 and use 32-byte ops. Only **element-wise** kernels live here (each output
//! lane depends on one input lane) — they are bit-identical under SIMD because
//! there is no cross-lane reduction to reorder.

/// In-place `vec[d] *= factor` — the scale half of an L2 normalize.
///
/// Runtime-dispatches to an AVX2 kernel (`vmulps` on 8 f32, 32-byte loads/stores)
/// when available; the portable scalar loop (LLVM auto-vectorizes to SSE2) is the
/// fallback. **Bit-identical**: each `vec[d] *= factor` is an independent IEEE f32
/// multiply, identical whether done 1/4/8-wide.
#[inline]
pub fn scale_f32_in_place(vec: &mut [f32], factor: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present by the runtime check above.
            #[allow(unsafe_code)]
            unsafe {
                scale_f32_in_place_avx2(vec, factor);
            }
            return;
        }
    }
    for x in vec.iter_mut() {
        *x *= factor;
    }
}

/// Hand-written AVX2 `vec[d] *= factor` (8 f32 / instruction).
///
/// # Safety
/// Caller must ensure `avx2` is available (the dispatch in [`scale_f32_in_place`]
/// guarantees it).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
fn scale_f32_in_place_avx2(vec: &mut [f32], factor: f32) {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps};
    let n = vec.len();
    let chunks = n / 8;
    // SAFETY: avx2 by contract; every load/store is `c < chunks`-bounded.
    unsafe {
        let f = _mm256_set1_ps(factor);
        for c in 0..chunks {
            let v = _mm256_loadu_ps(vec.as_ptr().add(c * 8));
            _mm256_storeu_ps(vec.as_mut_ptr().add(c * 8), _mm256_mul_ps(v, f));
        }
    }
    for x in vec.iter_mut().skip(chunks * 8) {
        *x *= factor;
    }
}

#[cfg(test)]
mod tests {
    use super::scale_f32_in_place;

    #[test]
    fn avx2_scale_matches_scalar() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            ((state >> 40) as f32 / (1_u64 << 23) as f32 - 1.0)
        };
        for &dim in &[0_usize, 1, 7, 8, 9, 16, 31, 128, 256, 257, 384] {
            let base: Vec<f32> = (0..dim).map(|_| next()).collect();
            let factor = 0.317_f32;
            let mut simd = base.clone();
            scale_f32_in_place(&mut simd, factor);
            let scalar: Vec<f32> = base.iter().map(|x| x * factor).collect();
            let sb: Vec<u32> = simd.iter().map(|x| x.to_bits()).collect();
            let cb: Vec<u32> = scalar.iter().map(|x| x.to_bits()).collect();
            assert_eq!(sb, cb, "dim={dim}");
        }
    }
}
