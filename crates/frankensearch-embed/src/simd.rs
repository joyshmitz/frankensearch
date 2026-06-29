//! Small runtime-dispatched SIMD helpers for the embedders.
//!
//! The workspace builds without a global `+avx2`, so LLVM auto-vectorizes the hot
//! element-wise loops only to the SSE2 baseline (16-byte ops). These helpers
//! runtime-detect AVX2 and use 32-byte ops, which roughly doubles the per-cycle
//! load bandwidth on the memory-bound accumulate.

/// Element-wise `sum[d] += row[d]` — the model2vec mean-pool inner loop.
///
/// Runtime-dispatches to an AVX2 kernel (32-byte `vmovups`/`vaddps`) when
/// available; the portable scalar loop (which LLVM auto-vectorizes to SSE2) is the
/// fallback. **Bit-identical** to the scalar path: each `sum[d] += row[d]` is an
/// independent element-wise add (no cross-lane reduction), so SIMD only changes how
/// many dims are added per instruction, never the per-dim arithmetic.
#[inline]
pub fn accumulate_f32_into(sum: &mut [f32], row: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present by the runtime check above.
            #[allow(unsafe_code)]
            unsafe {
                accumulate_f32_into_avx2(sum, row);
            }
            return;
        }
    }
    for (s, r) in sum.iter_mut().zip(row.iter()) {
        *s += *r;
    }
}

/// Hand-written AVX2 `sum[d] += row[d]` (8 f32 / instruction).
///
/// # Safety
/// Caller must ensure `avx2` is available (the dispatch in [`accumulate_f32_into`]
/// guarantees it).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
fn accumulate_f32_into_avx2(sum: &mut [f32], row: &[f32]) {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    let n = sum.len().min(row.len());
    let chunks = n / 8;
    // SAFETY: avx2 by contract; every load/store is `c < chunks`-bounded (`c*8+8 ≤ n`).
    unsafe {
        for c in 0..chunks {
            let s = _mm256_loadu_ps(sum.as_ptr().add(c * 8));
            let r = _mm256_loadu_ps(row.as_ptr().add(c * 8));
            _mm256_storeu_ps(sum.as_mut_ptr().add(c * 8), _mm256_add_ps(s, r));
        }
    }
    for i in (chunks * 8)..n {
        sum[i] += row[i];
    }
}

#[cfg(test)]
mod tests {
    use super::accumulate_f32_into;

    #[test]
    fn avx2_accumulate_matches_scalar() {
        // The AVX2 path must be byte-for-byte identical to the scalar fallback.
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            ((state >> 40) as f32 / (1_u64 << 23) as f32 - 1.0)
        };
        for &dim in &[0_usize, 1, 7, 8, 9, 16, 31, 128, 256, 257, 384] {
            let row: Vec<f32> = (0..dim).map(|_| next()).collect();
            // Accumulate several rows so values build up beyond a single add.
            let mut simd = vec![0.0_f32; dim];
            let mut scalar = vec![0.0_f32; dim];
            for _ in 0..5 {
                accumulate_f32_into(&mut simd, &row);
                for (s, r) in scalar.iter_mut().zip(row.iter()) {
                    *s += *r;
                }
            }
            let sb: Vec<u32> = simd.iter().map(|x| x.to_bits()).collect();
            let cb: Vec<u32> = scalar.iter().map(|x| x.to_bits()).collect();
            assert_eq!(sb, cb, "dim={dim}");
        }
    }
}
