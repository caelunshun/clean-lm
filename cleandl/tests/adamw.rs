use cleandl::{context::Context, operators::adamw::AdamWExecutor, tensor::Tensor};
use float_cmp::{ApproxEq, F32Margin};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
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

const REL_TOLERANCE: f32 = 1e-4;
const ABS_TOLERANCE: f32 = 1e-6;
const REPLICA_REL_TOLERANCE: f32 = 8e-3;

#[test]
fn adamw_weight_update() -> anyhow::Result<()> {
    let cx = Context::new(
        0,
        DeviceMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
    )?;
    let stream = cx.cuda().create_stream()?;
    let adamw = AdamWExecutor::get(&cx)?;

    let shape = Shape::new([N]);
    let bf16_type = ElementType::Float(FloatType::Bf16);
    let f32_type = ElementType::Float(FloatType::F32);
    let mut gradient = Tensor::new_zeroed(shape, bf16_type, &stream, &cx)?;
    let mut weight = Tensor::new_zeroed(shape, f32_type, &stream, &cx)?;
    let mut weight_bf16_replica = Tensor::new_zeroed(shape, bf16_type, &stream, &cx)?;
    let mut adam_m = Tensor::new_zeroed(shape, f32_type, &stream, &cx)?;
    let mut adam_v = Tensor::new_zeroed(shape, f32_type, &stream, &cx)?;
    let mut gradient_l2_global_sum = Tensor::new_zeroed(Shape::new([]), f32_type, &stream, &cx)?;

    let mut rng = StdRng::seed_from_u64(0xADA3);
    let mut sample = |count: usize, scale: f32| -> Vec<f32> {
        (0..count)
            .map(|_| rng.sample::<f32, _>(StandardNormal) * scale)
            .collect()
    };

    let gradient_host: Vec<bf16> = sample(N, 1.0).into_iter().map(bf16::from_f32).collect();
    let weight_host = sample(N, 0.1);
    let adam_m_host = sample(N, 0.01);
    let adam_v_host: Vec<f32> = sample(N, 0.01).into_iter().map(f32::abs).collect();

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

    let mut max_error = 0.0f32;
    for i in 0..N {
        assert!(weight_out[i].is_finite());
        max_error = max_error.max((weight_out[i] - expected.weight[i]).abs());

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
    }

    println!("results match the reference (max absolute weight error: {max_error})");

    Ok(())
}

struct Reference {
    weight: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

fn reference_adamw(gradient: &[bf16], weight: &[f32], m: &[f32], v: &[f32]) -> Reference {
    let sum_squares: f64 = gradient
        .iter()
        .map(|g| {
            let g = g.to_f32() as f64;
            g * g
        })
        .sum();
    let gradient_norm = (sum_squares as f32).sqrt();
    let clip_scale = 1.0f32.min(MAX_GRADIENT_NORM / (gradient_norm + EPS));

    let bias_correction1 = 1.0 - BETA1.powf(TRAIN_STEP as f32);
    let bias_correction2 = 1.0 - BETA2.powf(TRAIN_STEP as f32);

    let mut out = Reference {
        weight: Vec::with_capacity(weight.len()),
        m: Vec::with_capacity(weight.len()),
        v: Vec::with_capacity(weight.len()),
    };

    for i in 0..weight.len() {
        let g = gradient[i].to_f32() * clip_scale;

        let m_new = BETA1 * m[i] + (1.0 - BETA1) * g;
        let v_new = BETA2 * v[i] + (1.0 - BETA2) * (g * g);

        let m_hat = m_new / bias_correction1;
        let v_hat = v_new / bias_correction2;

        let w = weight[i] - LR * (m_hat / (v_hat.sqrt() + EPS) + WEIGHT_DECAY * weight[i]);

        out.weight.push(w);
        out.m.push(m_new);
        out.v.push(v_new);
    }

    out
}
