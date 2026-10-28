use cleandl::{
    context::Context,
    operators::swiglu_ffn::{GateProjForwardOutput, SwiGluConfig, SwiGluFfnExecutor},
    par_random::par_vec_from_distr,
    tensor::Tensor,
};
use faer::MatRef;
use float_cmp::{ApproxEq, F32Margin};
use half::bf16;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};
use skytens_core::data_type::{ElementType, FloatType};

const BATCH_SIZE: usize = 16384;
const IN_FEATURES: usize = 4096;
const INTERMEDIATE_FEATURES: usize = 8192;
const OUT_FEATURES: usize = 2048;

const REL_TOLERANCE: f32 = 2e-2;
const ABS_TOLERANCE: f32 = 1e-3;

#[test]
fn swiglu_ffn_forward_gate_proj() -> anyhow::Result<()> {
    let cx = &Context::new(
        0,
        DeviceMemoryPoolOptions::default(),
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        false,
    )?;
    let stream = cx.cuda().create_stream()?;

    let executor = SwiGluFfnExecutor::get(
        SwiGluConfig {
            estimated_batch_size: BATCH_SIZE / 2,
            in_features: IN_FEATURES,
            intermediate_features: INTERMEDIATE_FEATURES,
            out_features: OUT_FEATURES,
        },
        cx,
    )?;

    let bf16_type = ElementType::Float(FloatType::Bf16);
    let mut data = Tensor::new_zeroed([BATCH_SIZE, IN_FEATURES], bf16_type, &stream, cx)?;
    let mut up = Tensor::new_zeroed([BATCH_SIZE, INTERMEDIATE_FEATURES], bf16_type, &stream, cx)?;
    let mut weight =
        Tensor::new_zeroed([IN_FEATURES, INTERMEDIATE_FEATURES], bf16_type, &stream, cx)?;

    let mut rng = StdRng::seed_from_u64(555);
    let scale = (IN_FEATURES as f32).sqrt().recip();
    let data_host = par_vec_from_distr(
        StandardNormal.map(|x: f32| bf16::from_f32(x * scale)),
        BATCH_SIZE * IN_FEATURES,
        &mut rng,
    );
    let weight_host = par_vec_from_distr(
        StandardNormal.map(bf16::from_f32),
        IN_FEATURES * INTERMEDIATE_FEATURES,
        &mut rng,
    );
    let up_host = par_vec_from_distr(
        StandardNormal.map(bf16::from_f32),
        BATCH_SIZE * INTERMEDIATE_FEATURES,
        &mut rng,
    );

    cx.upload_to_device(&data_host, data.storage_mut(), &stream)?;
    cx.upload_to_device(&weight_host, weight.storage_mut(), &stream)?;
    cx.upload_to_device(&up_host, up.storage_mut(), &stream)?;

    let GateProjForwardOutput {
        gate: out_gate,
        activation: out_activation,
    } = executor.forward_gate_proj(&data.view(), &up.view(), &weight.view(), &stream)?;

    // Read the results back to the host.
    let gate_staging = cx.download_from_device(out_gate.storage(), &stream)?;
    let activation_staging = cx.download_from_device(out_activation.storage(), &stream)?;
    stream.block_on()?;

    let gate_host: &[bf16] = bytemuck::cast_slice(gate_staging.view()?);
    let activation_host: &[bf16] = bytemuck::cast_slice(activation_staging.view()?);

    let data_f32 = to_f32(&data_host);
    let weight_f32 = to_f32(&weight_host);
    let data_ref = MatRef::from_row_major_slice(&data_f32, BATCH_SIZE, IN_FEATURES);
    let weight_ref = MatRef::from_row_major_slice(&weight_f32, IN_FEATURES, INTERMEDIATE_FEATURES);
    let gate_ref = data_ref * weight_ref;

    let (max_gate_error, max_activation_error) = (0..BATCH_SIZE * INTERMEDIATE_FEATURES)
        .into_par_iter()
        .try_fold(
            || (0.0f32, 0.0f32),
            |(max_gate_error, max_activation_error), idx| -> anyhow::Result<(f32, f32)> {
                let (i, j) = (idx / INTERMEDIATE_FEATURES, idx % INTERMEDIATE_FEATURES);

                let expected_gate = gate_ref[(i, j)];
                let actual_gate = gate_host[idx].to_f32();
                assert!(actual_gate.is_finite());
                check(expected_gate, actual_gate, "gate", i, j)?;

                let expected_activation = swish(expected_gate) * up_host[idx].to_f32();
                let actual_activation = activation_host[idx].to_f32();
                assert!(actual_activation.is_finite());
                check(expected_activation, actual_activation, "activation", i, j)?;

                Ok((
                    max_gate_error.max((actual_gate - expected_gate).abs()),
                    max_activation_error.max((actual_activation - expected_activation).abs()),
                ))
            },
        )
        .try_reduce(
            || (0.0, 0.0),
            |(a_gate, a_act), (b_gate, b_act)| Ok((a_gate.max(b_gate), a_act.max(b_act))),
        )?;

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

fn to_f32(data: &[bf16]) -> Vec<f32> {
    data.par_iter().map(|x| x.to_f32()).collect()
}

/// Swish (SiLU) activation: `x * sigmoid(x)`.
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
