use cleandl::{
    bulk_copy::BulkCopyEngine,
    context::{Context, CudaStream},
    operators::swiglu_ffn::{SwiGluConfig, SwiGluFfnExecutor},
    tensor::TensorView,
};
use faer::Mat;
use float_cmp::{ApproxEq, F32Margin};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
use skytens_core::data_type::{ElementType, FloatType};

const BATCH_SIZE: usize = 16384;
const IN_FEATURES: usize = 4096;
const INTERMEDIATE_FEATURES: usize = 8192;
const OUT_FEATURES: usize = 2048;

const REL_TOLERANCE: f32 = 2e-2;
const ABS_TOLERANCE: f32 = 1e-3;

#[test]
fn swiglu_ffn_forward_gate_proj() -> anyhow::Result<()> {
    let cx = &Context::new(0)?;

    let executor = SwiGluFfnExecutor::get(
        SwiGluConfig {
            batch_size: BATCH_SIZE,
            in_features: IN_FEATURES,
            intermediate_features: INTERMEDIATE_FEATURES,
            out_features: OUT_FEATURES,
        },
        cx,
    )?;

    let bf16_type = ElementType::Float(FloatType::Bf16);
    let data = TensorView::new_zeroed([BATCH_SIZE, IN_FEATURES], bf16_type, cx)?;
    let up = TensorView::new_zeroed([BATCH_SIZE, INTERMEDIATE_FEATURES], bf16_type, cx)?;
    let weight = TensorView::new_zeroed([IN_FEATURES, INTERMEDIATE_FEATURES], bf16_type, cx)?;
    let out_gate = TensorView::new_zeroed([BATCH_SIZE, INTERMEDIATE_FEATURES], bf16_type, cx)?;
    let out_activation =
        TensorView::new_zeroed([BATCH_SIZE, INTERMEDIATE_FEATURES], bf16_type, cx)?;

    let mut rng = StdRng::seed_from_u64(555);
    let scale = (IN_FEATURES as f32).sqrt().recip();
    let data_host: Vec<bf16> = (0..BATCH_SIZE * IN_FEATURES)
        .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal) * scale))
        .collect();
    let weight_host: Vec<bf16> = (0..IN_FEATURES * INTERMEDIATE_FEATURES)
        .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal)))
        .collect();
    let up_host: Vec<bf16> = (0..BATCH_SIZE * INTERMEDIATE_FEATURES)
        .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal)))
        .collect();

    let mut copy_engine = BulkCopyEngine::new(cx)?;
    let mut batch = copy_engine.start_batch()?;
    batch.copy_to_device(&data, |dest| {
        dest.copy_from_slice(bytemuck::cast_slice(&data_host));
    })?;
    batch.copy_to_device(&weight, |dest| {
        dest.copy_from_slice(bytemuck::cast_slice(&weight_host));
    })?;
    batch.copy_to_device(&up, |dest| {
        dest.copy_from_slice(bytemuck::cast_slice(&up_host));
    })?;
    batch.commit()?;

    let stream = CudaStream::new(&cx)?;
    copy_engine.wait_for_last_batch_on(&stream)?;

    executor.forward_gate_proj(&data, &up, &weight, &out_gate, &out_activation, &stream)?;

    cx.cuda().synchronize()?;

    // Read the results back to the host.
    let mut batch = copy_engine.start_batch()?;
    let gate_token = batch.copy_to_host(&out_gate, None)?;
    let activation_token = batch.copy_to_host(&out_activation, None)?;
    batch.commit()?;
    let gate_host: Vec<bf16> = bytemuck::cast_slice(copy_engine.retrieve(gate_token)?).to_vec();
    let activation_host: &[bf16] = bytemuck::cast_slice(copy_engine.retrieve(activation_token)?);

    let data_ref = Mat::from_fn(BATCH_SIZE, IN_FEATURES, |i, j| {
        data_host[i * IN_FEATURES + j].to_f32()
    });
    let weight_ref = Mat::from_fn(IN_FEATURES, INTERMEDIATE_FEATURES, |i, j| {
        weight_host[i * INTERMEDIATE_FEATURES + j].to_f32()
    });
    let gate_ref = data_ref * weight_ref;

    let mut max_gate_error = 0.0f32;
    let mut max_activation_error = 0.0f32;
    for i in 0..BATCH_SIZE {
        for j in 0..INTERMEDIATE_FEATURES {
            let idx = i * INTERMEDIATE_FEATURES + j;

            let expected_gate = gate_ref[(i, j)];
            let actual_gate = gate_host[idx].to_f32();
            max_gate_error = max_gate_error.max((actual_gate - expected_gate).abs());
            check(expected_gate, actual_gate, "gate", i, j)?;

            let expected_activation = swish(expected_gate) * up_host[idx].to_f32();
            let actual_activation = activation_host[idx].to_f32();
            max_activation_error =
                max_activation_error.max((actual_activation - expected_activation).abs());
            check(expected_activation, actual_activation, "activation", i, j)?;
        }
    }

    println!(
        "results match the reference (max absolute error: gate {max_gate_error}, \
         activation {max_activation_error})"
    );

    Ok(())
}

fn check(expected: f32, actual: f32, name: &str, i: usize, j: usize) -> anyhow::Result<()> {
    let margin = F32Margin {
        epsilon: ABS_TOLERANCE.max(REL_TOLERANCE * expected.abs()),
        ulps: 0,
    };
    anyhow::ensure!(
        actual.approx_eq(expected, margin),
        "{name} mismatch at ({i}, {j}): expected {expected}, got {actual}"
    );
    Ok(())
}

/// Swish (SiLU) activation: `x * sigmoid(x)`.
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
