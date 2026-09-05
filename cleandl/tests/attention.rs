use cleandl::{
    bulk_copy::BulkCopyEngine,
    context::{Context, CudaStream},
    operators::attention::{AttentionConfig, AttentionExecutor},
    tensor::TensorView,
};
use float_cmp::{ApproxEq, F32Margin};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
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
    let cx = Context::new(0)?;

    let num_sequences = SEQLENS.len();
    let max_seqlen = SEQLENS.iter().copied().max().unwrap();
    let total_tokens: usize = SEQLENS.iter().sum();

    let executor = AttentionExecutor::get(
        AttentionConfig {
            max_q_seqlen: max_seqlen,
            max_kv_seqlen: max_seqlen,
            max_num_sequences: num_sequences,
            q_heads: Q_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
        },
        &cx,
    )?;

    let q = TensorView::new_zeroed(
        Shape::new([total_tokens, Q_HEADS, HEAD_DIM]),
        ElementType::Float(FloatType::Bf16),
        &cx,
    )?;
    let k = TensorView::new_zeroed(
        Shape::new([total_tokens, KV_HEADS, HEAD_DIM]),
        ElementType::Float(FloatType::Bf16),
        &cx,
    )?;
    let v = TensorView::new_zeroed(
        Shape::new([total_tokens, KV_HEADS, HEAD_DIM]),
        ElementType::Float(FloatType::Bf16),
        &cx,
    )?;
    let out = TensorView::new_zeroed(
        Shape::new([total_tokens, Q_HEADS, HEAD_DIM]),
        ElementType::Float(FloatType::Bf16),
        &cx,
    )?;
    let out_stats = TensorView::new_zeroed(
        Shape::new([total_tokens, Q_HEADS, 1]),
        ElementType::Float(FloatType::F32),
        &cx,
    )?;

    let cu_seqlen_q = TensorView::new_zeroed(
        Shape::new([num_sequences + 1]),
        ElementType::SInt(SIntType::S64),
        &cx,
    )?;
    let cu_seqlen_kv = TensorView::new_zeroed(
        Shape::new([num_sequences + 1]),
        ElementType::SInt(SIntType::S64),
        &cx,
    )?;
    let seqlen_q = TensorView::new_zeroed(
        Shape::new([num_sequences]),
        ElementType::SInt(SIntType::S32),
        &cx,
    )?;
    let seqlen_kv = TensorView::new_zeroed(
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

    let mut copy_engine = BulkCopyEngine::new(&cx)?;
    let mut batch = copy_engine.start_batch()?;
    for (tensor, host) in [(&q, &q_host), (&k, &k_host), (&v, &v_host)] {
        batch.copy_to_device(tensor, |dest| {
            dest.copy_from_slice(bytemuck::cast_slice(host));
        })?;
    }
    for tensor in [&cu_seqlen_q, &cu_seqlen_kv] {
        batch.copy_to_device(tensor, |dest| {
            dest.copy_from_slice(bytemuck::cast_slice(&cu_seqlens_host));
        })?;
    }
    for tensor in [&seqlen_q, &seqlen_kv] {
        batch.copy_to_device(tensor, |dest| {
            dest.copy_from_slice(bytemuck::cast_slice(&seqlens_host));
        })?;
    }
    batch.commit()?;

    executor.forward_attn(
        &q,
        &k,
        &v,
        &cu_seqlen_q,
        &cu_seqlen_kv,
        &seqlen_q,
        &seqlen_kv,
        &out,
        &out_stats,
        &CudaStream::new(&cx)?,
    )?;

    cx.cuda().synchronize()?;

    // Read the result back to the host.
    let mut batch = copy_engine.start_batch()?;
    let token = batch.copy_to_host(&out)?;
    batch.commit()?;
    let out_host: &[bf16] = bytemuck::cast_slice(copy_engine.retrieve(token)?);

    let expected = reference_attention(&q_host, &k_host, &v_host, &cu_seqlens_host);

    let mut max_error = 0.0f32;
    for token_idx in 0..total_tokens {
        for head in 0..Q_HEADS {
            for d in 0..HEAD_DIM {
                let idx = (token_idx * Q_HEADS + head) * HEAD_DIM + d;
                let expected = expected[idx];
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
