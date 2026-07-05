//! CLS-only final-layer attention A/B for the native reranker.
//!
//! The shipped final encoder layer only needs the `[CLS]` row, but the original
//! path still repacked K/V into head-major buffers and launched two tiny BMMs
//! with `m = 1`. This bench keeps that path as `bmm_repack` and compares the
//! direct rank-1 kernel used by `native.rs`.
//!
//! Run:
//! ```text
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-rerank --features native \
//!     --profile release --bench cls_attention_ab -- cls_attention
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ft_core::{DType, Device, TensorMeta};
use wide::f32x8;

const H: usize = 384;
const NH: usize = 12;
const HD: usize = H / NH;
const STRIDE: usize = 3 * H;
const SCALE: f32 = 0.176_776_69; // 1 / sqrt(32)

#[derive(Default)]
struct Scratch {
    q_hm: Vec<f32>,
    kt: Vec<f32>,
    v_hm: Vec<f32>,
    scores: Vec<f32>,
    ctx_hm: Vec<f32>,
}

impl Scratch {
    fn ensure(&mut self, s_len: usize) {
        let hm = s_len * H;
        let sc = NH * s_len * s_len;
        if self.q_hm.len() < hm {
            self.q_hm.resize(hm, 0.0);
        }
        if self.kt.len() < hm {
            self.kt.resize(hm, 0.0);
        }
        if self.v_hm.len() < hm {
            self.v_hm.resize(hm, 0.0);
        }
        if self.scores.len() < sc {
            self.scores.resize(sc, 0.0);
        }
        if self.ctx_hm.len() < hm {
            self.ctx_hm.resize(hm, 0.0);
        }
    }
}

#[inline]
fn f32x8_from_slice(slice: &[f32]) -> f32x8 {
    let mut buf = [0.0f32; 8];
    buf.copy_from_slice(&slice[..8]);
    f32x8::new(buf)
}

#[inline]
fn dot_hd(q: &[f32], k: &[f32]) -> f32 {
    (f32x8_from_slice(&q[0..]) * f32x8_from_slice(&k[0..])
        + f32x8_from_slice(&q[8..]) * f32x8_from_slice(&k[8..])
        + f32x8_from_slice(&q[16..]) * f32x8_from_slice(&k[16..])
        + f32x8_from_slice(&q[24..]) * f32x8_from_slice(&k[24..]))
    .reduce_add()
}

fn softmax_row_fused(row: &mut [f32], scale: f32) {
    let mut max_raw = f32::NEG_INFINITY;
    for &x in row.iter() {
        max_raw = max_raw.max(x);
    }
    let max_v = f32x8::splat(max_raw);
    let scale_v = f32x8::splat(scale);
    let mut sum_v = f32x8::splat(0.0);
    let mut i = 0;
    while i + 8 <= row.len() {
        let e = ((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v).exp();
        row[i..i + 8].copy_from_slice(&e.to_array());
        sum_v += e;
        i += 8;
    }
    let mut sum = sum_v.reduce_add();
    while i < row.len() {
        let e = ((row[i] - max_raw) * scale).exp();
        row[i] = e;
        sum += e;
        i += 1;
    }
    let inv = 1.0 / sum;
    for x in row {
        *x *= inv;
    }
}

fn softmax_rows(data: &mut [f32], rows: usize, n: usize, scale: f32) {
    debug_assert_eq!(data.len(), rows * n);
    for row in data.chunks_exact_mut(n) {
        softmax_row_fused(row, scale);
    }
}

fn cls_attention_bmm_repack(scratch: &mut Scratch, qkv: &[f32], s_len: usize, out: &mut [f32]) {
    scratch.ensure(s_len);
    let hm = s_len * H;
    let sc = NH * s_len;
    let q_cls = &mut scratch.q_hm[..H];
    let kt = &mut scratch.kt[..hm];
    let v_hm = &mut scratch.v_hm[..hm];
    let scores = &mut scratch.scores[..sc];
    let ctx_hm = &mut scratch.ctx_hm[..H];

    for head in 0..NH {
        q_cls[head * HD..head * HD + HD].copy_from_slice(&qkv[head * HD..head * HD + HD]);
    }
    for j in 0..s_len {
        let base = j * STRIDE;
        for head in 0..NH {
            let hmj = (head * s_len + j) * HD;
            v_hm[hmj..hmj + HD]
                .copy_from_slice(&qkv[base + 2 * H + head * HD..base + 2 * H + head * HD + HD]);
        }
    }
    for head in 0..NH {
        for d in 0..HD {
            let col = H + head * HD + d;
            let row = &mut kt[head * HD * s_len + d * s_len..head * HD * s_len + d * s_len + s_len];
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = qkv[j * STRIDE + col];
            }
        }
    }
    let qm = TensorMeta::from_shape(vec![NH, 1, HD], DType::F32, Device::Cpu);
    let km = TensorMeta::from_shape(vec![NH, HD, s_len], DType::F32, Device::Cpu);
    ft_api::bmm_tensor_contiguous_f32_into(q_cls, kt, &qm, &km, scores).expect("cls qk bmm shape");
    softmax_rows(scores, NH, s_len, SCALE);
    let sm = TensorMeta::from_shape(vec![NH, 1, s_len], DType::F32, Device::Cpu);
    let vm = TensorMeta::from_shape(vec![NH, s_len, HD], DType::F32, Device::Cpu);
    ft_api::bmm_tensor_contiguous_f32_into(scores, v_hm, &sm, &vm, ctx_hm)
        .expect("cls value bmm shape");
    out.copy_from_slice(ctx_hm);
}

fn weighted_value_sum_hd(qkv: &[f32], probs: &[f32], head: usize, out: &mut [f32]) {
    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);
    for (j, &prob) in probs.iter().enumerate() {
        let p = f32x8::splat(prob);
        let base = j * STRIDE + 2 * H + head * HD;
        acc0 += p * f32x8_from_slice(&qkv[base..base + 8]);
        acc1 += p * f32x8_from_slice(&qkv[base + 8..base + 16]);
        acc2 += p * f32x8_from_slice(&qkv[base + 16..base + 24]);
        acc3 += p * f32x8_from_slice(&qkv[base + 24..base + 32]);
    }
    out[0..8].copy_from_slice(&acc0.to_array());
    out[8..16].copy_from_slice(&acc1.to_array());
    out[16..24].copy_from_slice(&acc2.to_array());
    out[24..32].copy_from_slice(&acc3.to_array());
}

fn cls_attention_direct(scratch: &mut Scratch, qkv: &[f32], s_len: usize, out: &mut [f32]) {
    scratch.ensure(s_len);
    let sc = NH * s_len;
    for head in 0..NH {
        let q = &qkv[head * HD..head * HD + HD];
        let row = &mut scratch.scores[head * s_len..(head + 1) * s_len];
        for (j, slot) in row.iter_mut().enumerate() {
            let k_base = j * STRIDE + H + head * HD;
            *slot = dot_hd(q, &qkv[k_base..k_base + HD]);
        }
    }
    let scores = &mut scratch.scores[..sc];
    softmax_rows(scores, NH, s_len, SCALE);
    for head in 0..NH {
        let row = &scores[head * s_len..(head + 1) * s_len];
        weighted_value_sum_hd(qkv, row, head, &mut out[head * HD..head * HD + HD]);
    }
}

fn qkv_fixture(s_len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(s_len * STRIDE);
    for i in 0..s_len * STRIDE {
        let v = ((i.wrapping_mul(31) % 257) as f32 - 128.0) * 0.0025;
        out.push(v);
    }
    out
}

fn assert_close(a: &[f32], b: &[f32]) {
    let max_delta = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta <= 1.0e-4,
        "direct CLS attention diverged from BMM reference by {max_delta}"
    );
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("cls_attention");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(500));

    for &s_len in &[64usize, 128, 256, 512] {
        let qkv = qkv_fixture(s_len);
        let mut bmm_scratch = Scratch::default();
        let mut direct_scratch = Scratch::default();
        let mut expected = vec![0.0f32; H];
        let mut actual = vec![0.0f32; H];
        cls_attention_bmm_repack(&mut bmm_scratch, &qkv, s_len, &mut expected);
        cls_attention_direct(&mut direct_scratch, &qkv, s_len, &mut actual);
        assert_close(&expected, &actual);

        group.bench_with_input(
            BenchmarkId::new("bmm_repack_orig", s_len),
            &qkv,
            |b, qkv| {
                let mut scratch = Scratch::default();
                let mut out = vec![0.0f32; H];
                b.iter(|| {
                    cls_attention_bmm_repack(
                        black_box(&mut scratch),
                        black_box(qkv),
                        black_box(s_len),
                        black_box(&mut out),
                    );
                    black_box(&out);
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("direct_rank1", s_len), &qkv, |b, qkv| {
            let mut scratch = Scratch::default();
            let mut out = vec![0.0f32; H];
            b.iter(|| {
                cls_attention_direct(
                    black_box(&mut scratch),
                    black_box(qkv),
                    black_box(s_len),
                    black_box(&mut out),
                );
                black_box(&out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
