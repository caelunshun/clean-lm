use cleandl::{
    context::Context,
    operators::attention::{
        AttentionConfig, AttentionExecutor, AttentionForwardOutput, CuSeqlenTensors,
    },
    tensor::Tensor,
};
use float_cmp::{ApproxEq, F32Margin};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};
use skytens_core::{
    data_type::{ElementType, FloatType, SIntType},
    shape::Shape,
};

const SEQLENS: [usize; 4] = [37, 128, 201, 411];

const Q_HEADS: usize = 64;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;

const REL_TOLERANCE: f32 = 2e-2;
const ABS_TOLERANCE: f32 = 4e-3;

#[test]
fn attention() -> anyhow::Result<()> {
    let cx = Context::new(
        0,
        DeviceMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        false,
    )?;
    let stream = cx.cuda().create_stream()?;

    let num_sequences = SEQLENS.len();
    let max_seqlen = SEQLENS.iter().copied().max().unwrap();
    let total_tokens: usize = SEQLENS.iter().sum();

    let executor = AttentionExecutor::get(
        AttentionConfig {
            is_causal: true,
            max_q_seqlen: max_seqlen,
            max_kv_seqlen: max_seqlen,
            max_num_sequences: num_sequences,
            q_heads: Q_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
        },
        &cx,
    )?;

    let bf16_type = ElementType::Float(FloatType::Bf16);
    let mut q = Tensor::new_zeroed(
        Shape::new([total_tokens, Q_HEADS, HEAD_DIM]),
        bf16_type,
        &cx,
    )?;
    let mut k = Tensor::new_zeroed(
        Shape::new([total_tokens, KV_HEADS, HEAD_DIM]),
        bf16_type,
        &cx,
    )?;
    let mut v = Tensor::new_zeroed(
        Shape::new([total_tokens, KV_HEADS, HEAD_DIM]),
        bf16_type,
        &cx,
    )?;
    let mut cu_seqlen_q_times_hd_times_qh = Tensor::new_zeroed(
        Shape::new([num_sequences + 1]),
        ElementType::SInt(SIntType::S64),
        &cx,
    )?;
    let mut cu_seqlen_kv_times_hd_times_kvh = Tensor::new_zeroed(
        Shape::new([num_sequences + 1]),
        ElementType::SInt(SIntType::S64),
        &cx,
    )?;
    let mut seqlen_q = Tensor::new_zeroed(
        Shape::new([num_sequences]),
        ElementType::SInt(SIntType::S32),
        &cx,
    )?;
    let mut seqlen_kv = Tensor::new_zeroed(
        Shape::new([num_sequences]),
        ElementType::SInt(SIntType::S32),
        &cx,
    )?;

    let mut rng = StdRng::seed_from_u64(555);
    let mut sample = |count: usize| -> Vec<bf16> {
        (0..count)
            .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal)))
            .collect()
    };
    let q_host = sample(total_tokens * Q_HEADS * HEAD_DIM);
    let k_host = sample(total_tokens * KV_HEADS * HEAD_DIM);
    let v_host = sample(total_tokens * KV_HEADS * HEAD_DIM);

    // Cumulative sequence offsets, in tokens.
    let mut cu_seqlens_host: Vec<i64> = Vec::with_capacity(num_sequences + 1);
    cu_seqlens_host.push(0);
    for len in SEQLENS {
        cu_seqlens_host.push(cu_seqlens_host.last().unwrap() + len as i64);
    }
    let seqlens_host: Vec<i32> = SEQLENS.iter().map(|&len| len as i32).collect();

    for (tensor, host) in [(&mut q, &q_host), (&mut k, &k_host), (&mut v, &v_host)] {
        cx.upload_to_device(host, tensor.storage_mut(), &stream)?;
    }

    let scaled = |multiplier: usize| -> Vec<i64> {
        cu_seqlens_host
            .iter()
            .map(|&offset| offset * multiplier as i64)
            .collect()
    };
    for (tensor, host) in [
        (
            &mut cu_seqlen_q_times_hd_times_qh,
            scaled(HEAD_DIM * Q_HEADS),
        ),
        (
            &mut cu_seqlen_kv_times_hd_times_kvh,
            scaled(HEAD_DIM * KV_HEADS),
        ),
    ] {
        cx.upload_to_device(&host, tensor.storage_mut(), &stream)?;
    }
    for tensor in [&mut seqlen_q, &mut seqlen_kv] {
        cx.upload_to_device(&seqlens_host, tensor.storage_mut(), &stream)?;
    }

    let AttentionForwardOutput { out, .. } = executor.forward_attn(
        &q.view(),
        &k.view(),
        &v.view(),
        &CuSeqlenTensors {
            cu_seqlen_q_times_hd_times_qh: cu_seqlen_q_times_hd_times_qh.view(),
            cu_seqlen_kv_times_hd_times_kvh: cu_seqlen_kv_times_hd_times_kvh.view(),
        },
        &seqlen_q.view(),
        &seqlen_kv.view(),
        &stream,
    )?;

    // Read the result back to the host.
    let out_staging = cx.download_from_device(out.storage(), &stream)?;
    stream.block_on()?;
    let out_host: &[bf16] = bytemuck::cast_slice(out_staging.view()?);

    let expected = reference_attention(&q_host, &k_host, &v_host, &cu_seqlens_host);

    let mut max_error = 0.0f32;
    for token_idx in 0..total_tokens {
        for head in 0..Q_HEADS {
            for d in 0..HEAD_DIM {
                let idx = (token_idx * Q_HEADS + head) * HEAD_DIM + d;
                let expected = expected[idx];
                assert!(out_host[idx].is_finite());
                let actual = out_host[idx].to_f32();

                max_error = max_error.max((actual - expected).abs());

                let margin = F32Margin {
                    epsilon: ABS_TOLERANCE.max(REL_TOLERANCE * expected.abs()),
                    ulps: 0,
                };
                anyhow::ensure!(
                    actual.approx_eq(expected, margin),
                    "mismatch at token {token_idx}, head {head}, dim {d}: \
                     expected {expected}, got {actual}"
                );
            }
        }
    }

    println!("results match the reference (max absolute error: {max_error})");

    Ok(())
}

fn reference_attention(q: &[bf16], k: &[bf16], v: &[bf16], cu_seqlens: &[i64]) -> Vec<f32> {
    let total_tokens = *cu_seqlens.last().unwrap() as usize;
    let mut out = vec![0.0f32; total_tokens * Q_HEADS * HEAD_DIM];

    let scale = (HEAD_DIM as f32).sqrt().recip();
    let group_size = Q_HEADS / KV_HEADS;

    for seq in 0..cu_seqlens.len() - 1 {
        let base = cu_seqlens[seq] as usize;
        let seqlen = cu_seqlens[seq + 1] as usize - base;

        for head in 0..Q_HEADS {
            let kv_head = head / group_size;

            for i in 0..seqlen {
                let q_offset = ((base + i) * Q_HEADS + head) * HEAD_DIM;

                let mut scores = vec![0.0f32; i + 1];
                for (j, score) in scores.iter_mut().enumerate() {
                    let k_offset = ((base + j) * KV_HEADS + kv_head) * HEAD_DIM;
                    let dot: f32 = (0..HEAD_DIM)
                        .map(|d| q[q_offset + d].to_f32() * k[k_offset + d].to_f32())
                        .sum();
                    *score = dot * scale;
                }

                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                    sum_exp += *score;
                }

                let out_offset = q_offset;
                for (j, weight) in scores.iter().enumerate() {
                    let weight = weight / sum_exp;
                    let v_offset = ((base + j) * KV_HEADS + kv_head) * HEAD_DIM;
                    for d in 0..HEAD_DIM {
                        out[out_offset + d] += weight * v[v_offset + d].to_f32();
                    }
                }
            }
        }
    }

    out
}
