use cleandl::{
    bulk_copy::BulkCopyEngine,
    context::{Context, CudaStream},
    operators::adamw::AdamWExecutor,
    tensor::TensorView,
};
use float_cmp::{ApproxEq, F32Margin};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
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
    let cx = Context::new(0)?;
    let adamw = AdamWExecutor::get(&cx)?;

    let shape = Shape::new([N]);
    let gradient = TensorView::new_zeroed(shape, ElementType::Float(FloatType::Bf16), &cx)?;
    let weight = TensorView::new_zeroed(shape, ElementType::Float(FloatType::F32), &cx)?;
    let weight_bf16_replica =
        TensorView::new_zeroed(shape, ElementType::Float(FloatType::Bf16), &cx)?;
    let adam_m = TensorView::new_zeroed(shape, ElementType::Float(FloatType::F32), &cx)?;
    let adam_v = TensorView::new_zeroed(shape, ElementType::Float(FloatType::F32), &cx)?;
    let gradient_l2_global_sum =
        TensorView::new_zeroed(Shape::new([]), ElementType::Float(FloatType::F32), &cx)?;

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

    let mut copy_engine = BulkCopyEngine::new(&cx)?;
    let mut batch = copy_engine.start_batch()?;
    batch.copy_to_device(&gradient, |dest| {
        dest.copy_from_slice(bytemuck::cast_slice(&gradient_host));
    })?;
    for (tensor, host) in [
        (&weight, &weight_host),
        (&adam_m, &adam_m_host),
        (&adam_v, &adam_v_host),
    ] {
        batch.copy_to_device(tensor, |dest| {
            dest.copy_from_slice(bytemuck::cast_slice(host));
        })?;
    }
    batch.copy_to_device(&gradient_l2_global_sum, |dest| {
        dest.copy_from_slice(bytemuck::cast_slice(&[0.0f32]));
    })?;
    batch.commit()?;

    let stream = CudaStream::new(&cx)?;
    adamw.gradient_l2_norm_reduction(&gradient, &gradient_l2_global_sum, &stream)?;
    adamw.adamw_weight_update(
        &gradient,
        &gradient_l2_global_sum,
        &weight,
        &weight_bf16_replica,
        &adam_m,
        &adam_v,
        BETA1,
        BETA2,
        WEIGHT_DECAY,
        MAX_GRADIENT_NORM,
        LR,
        TRAIN_STEP,
        &stream,
    )?;
    cx.cuda().synchronize()?;

    let mut batch = copy_engine.start_batch()?;
    let weight_token = batch.copy_to_host(&weight)?;
    let replica_token = batch.copy_to_host(&weight_bf16_replica)?;
    let m_token = batch.copy_to_host(&adam_m)?;
    let v_token = batch.copy_to_host(&adam_v)?;
    batch.commit()?;

    let weight_out: &[f32] = bytemuck::cast_slice(copy_engine.retrieve(weight_token)?);
    let replica_out: &[bf16] = bytemuck::cast_slice(copy_engine.retrieve(replica_token)?);
    let m_out: &[f32] = bytemuck::cast_slice(copy_engine.retrieve(m_token)?);
    let v_out: &[f32] = bytemuck::cast_slice(copy_engine.retrieve(v_token)?);

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
