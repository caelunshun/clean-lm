use cleandl::{
    context::Context, operators::adamw::AdamWExecutor, par_random::par_vec_from_distr,
    tensor::Tensor,
};
use float_cmp::{ApproxEq, F32Margin};
use half::bf16;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

const N: usize = 8192;

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.95;
const WEIGHT_DECAY: f32 = 0.1;
const MAX_GRADIENT_NORM: f32 = 1.0;
const LR: f32 = 1e-3;
const TRAIN_STEP: u32 = 3;

const EPS: f32 = 1e-6;

const REL_TOLERANCE: f32 = 1e-3;
const ABS_TOLERANCE: f32 = 1e-4;
const REPLICA_REL_TOLERANCE: f32 = 8e-3;

#[test]
fn adamw_weight_update() -> anyhow::Result<()> {
    let cx = Context::new(
        0,
        DeviceMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
            max_total_bytes: None,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        false,
    )?;
    let stream = cx.cuda().create_stream()?;
    let adamw = AdamWExecutor::get(&cx)?;

    let shape = Shape::new([N]);
    let f32_type = ElementType::Float(FloatType::F32);
    let bf16_type = ElementType::Float(FloatType::Bf16);
    let mut gradient = Tensor::new_zeroed(shape, f32_type, &cx)?;
    let mut weight = Tensor::new_zeroed(shape, f32_type, &cx)?;
    let mut weight_bf16_replica = Tensor::new_zeroed(shape, bf16_type, &cx)?;
    let mut adam_m = Tensor::new_zeroed(shape, f32_type, &cx)?;
    let mut adam_v = Tensor::new_zeroed(shape, f32_type, &cx)?;
    let mut gradient_l2_global_sum = Tensor::new_zeroed(Shape::new([]), f32_type, &cx)?;

    let mut rng = StdRng::seed_from_u64(0xADA3);
    let mut sample = |count: usize, scale: f32| -> Vec<f32> {
        par_vec_from_distr(StandardNormal.map(|x: f32| x * scale), count, &mut rng)
    };

    let gradient_host = sample(N, 1.0);
    let weight_host = sample(N, 0.1);
    let adam_m_host = sample(N, 0.01);
    let adam_v_host: Vec<f32> = sample(N, 0.01).into_par_iter().map(f32::abs).collect();

    cx.upload_to_device(&gradient_host, gradient.storage_mut(), &stream)?;
    for (tensor, host) in [
        (&mut weight, &weight_host),
        (&mut adam_m, &adam_m_host),
        (&mut adam_v, &adam_v_host),
    ] {
        cx.upload_to_device(host, tensor.storage_mut(), &stream)?;
    }
    cx.upload_to_device(&[0.0f32], gradient_l2_global_sum.storage_mut(), &stream)?;

    adamw.gradient_l2_norm_reduction(
        &gradient.view(),
        &mut gradient_l2_global_sum.view_mut(),
        &stream,
    )?;
    adamw.adamw_weight_update(
        &gradient.view(),
        &gradient_l2_global_sum.view(),
        &mut weight.view_mut(),
        &mut weight_bf16_replica.view_mut(),
        &mut adam_m.view_mut(),
        &mut adam_v.view_mut(),
        BETA1,
        BETA2,
        WEIGHT_DECAY,
        MAX_GRADIENT_NORM,
        LR,
        TRAIN_STEP,
        &stream,
    )?;

    let weight_staging = cx.download_from_device(weight.storage(), &stream)?;
    let replica_staging = cx.download_from_device(weight_bf16_replica.storage(), &stream)?;
    let m_staging = cx.download_from_device(adam_m.storage(), &stream)?;
    let v_staging = cx.download_from_device(adam_v.storage(), &stream)?;
    stream.block_on()?;

    let weight_out: &[f32] = bytemuck::cast_slice(weight_staging.view()?);
    let replica_out: &[bf16] = bytemuck::cast_slice(replica_staging.view()?);
    let m_out: &[f32] = bytemuck::cast_slice(m_staging.view()?);
    let v_out: &[f32] = bytemuck::cast_slice(v_staging.view()?);

    let expected = reference_adamw(&gradient_host, &weight_host, &adam_m_host, &adam_v_host);

    let check =
        |name: &str, i: usize, actual: f32, expected: f32, rel: f32| -> anyhow::Result<()> {
            let margin = F32Margin {
                epsilon: ABS_TOLERANCE.max(rel * expected.abs()),
                ulps: 0,
            };
            anyhow::ensure!(
                actual.approx_eq(expected, margin),
                "{name} mismatch at index {i}: expected {expected}, got {actual}"
            );
            Ok(())
        };

    let max_error = (0..N)
        .into_par_iter()
        .try_fold(
            || 0.0f32,
            |max_error, i| -> anyhow::Result<f32> {
                assert!(weight_out[i].is_finite());

                check(
                    "weight",
                    i,
                    weight_out[i],
                    expected.weight[i],
                    REL_TOLERANCE,
                )?;
                check(
                    "bf16 replica",
                    i,
                    replica_out[i].to_f32(),
                    expected.weight[i],
                    REPLICA_REL_TOLERANCE,
                )?;
                check("adam_m", i, m_out[i], expected.m[i], REL_TOLERANCE)?;
                check("adam_v", i, v_out[i], expected.v[i], REL_TOLERANCE)?;

                Ok(max_error.max((weight_out[i] - expected.weight[i]).abs()))
            },
        )
        .try_reduce(|| 0.0, |a, b| Ok(a.max(b)))?;

    println!("results match the reference (max absolute weight error: {max_error})");

    Ok(())
}

struct Reference {
    weight: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

fn reference_adamw(gradient: &[f32], weight: &[f32], m: &[f32], v: &[f32]) -> Reference {
    let sum_squares: f64 = gradient
        .par_iter()
        .map(|g| {
            let g = *g as f64;
            g * g
        })
        .sum();
    let gradient_norm = (sum_squares as f32).sqrt();
    let clip_scale = 1.0f32.min(MAX_GRADIENT_NORM / (gradient_norm + EPS));

    let bias_correction1 = 1.0 - BETA1.powf(TRAIN_STEP as f32);
    let bias_correction2 = 1.0 - BETA2.powf(TRAIN_STEP as f32);

    let (weight, (m, v)) = (0..weight.len())
        .into_par_iter()
        .map(|i| {
            let g = gradient[i] * clip_scale;

            let m_new = BETA1 * m[i] + (1.0 - BETA1) * g;
            let v_new = BETA2 * v[i] + (1.0 - BETA2) * (g * g);

            let m_hat = m_new / bias_correction1;
            let v_hat = v_new / bias_correction2;

            let w = weight[i] - LR * (m_hat / (v_hat.sqrt() + EPS) + WEIGHT_DECAY * weight[i]);

            (w, (m_new, v_new))
        })
        .unzip();

    Reference { weight, m, v }
}
