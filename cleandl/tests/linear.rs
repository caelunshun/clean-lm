// --- LLM-generated --- //

use cleandl::{
    context::Context,
    operators::linear::{LinearConfig, LinearExecutor},
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

const BATCH_SIZE: usize = 4096;
const IN_FEATURES: usize = 1024;
const OUT_FEATURES: usize = 768;

const REL_TOLERANCE: f32 = 2e-2;
const ABS_TOLERANCE: f32 = 1e-3;

fn make_context() -> anyhow::Result<Context> {
    Context::new(
        0,
        DeviceMemoryPoolOptions::default(),
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        false,
    )
}

#[test]
fn linear_backward_wgrad_accumulate() -> anyhow::Result<()> {
    let cx = &make_context()?;
    let stream = cx.cuda().create_stream()?;

    let executor = LinearExecutor::get(
        LinearConfig {
            estimated_batch_size: BATCH_SIZE / 2,
            in_features: IN_FEATURES,
            out_features: OUT_FEATURES,
            forward_fused_residual_add: false,
        },
        cx,
    )?;

    let bf16_type = ElementType::Float(FloatType::Bf16);
    let f32_type = ElementType::Float(FloatType::F32);
    let mut data = Tensor::new_zeroed([BATCH_SIZE, IN_FEATURES], bf16_type, cx)?;
    let mut dy = Tensor::new_zeroed([BATCH_SIZE, OUT_FEATURES], bf16_type, cx)?;
    let mut gradient = Tensor::new_zeroed([IN_FEATURES, OUT_FEATURES], f32_type, cx)?;

    let mut rng = StdRng::seed_from_u64(777);
    let scale = (BATCH_SIZE as f32).sqrt().recip();
    let data_host = par_vec_from_distr(
        StandardNormal.map(bf16::from_f32),
        BATCH_SIZE * IN_FEATURES,
        &mut rng,
    );
    let dy_host = par_vec_from_distr(
        StandardNormal.map(|x: f32| bf16::from_f32(x * scale)),
        BATCH_SIZE * OUT_FEATURES,
        &mut rng,
    );
    let gradient_host: Vec<f32> = par_vec_from_distr(
        StandardNormal.map(|x: f32| x),
        IN_FEATURES * OUT_FEATURES,
        &mut rng,
    );

    cx.upload_to_device(&data_host, data.storage_mut(), &stream)?;
    cx.upload_to_device(&dy_host, dy.storage_mut(), &stream)?;
    cx.upload_to_device(&gradient_host, gradient.storage_mut(), &stream)?;

    // Accumulate twice to check that existing gradient contents are preserved.
    for _ in 0..2 {
        executor.backward_wgrad_accumulate(
            &data.view(),
            &dy.view(),
            &mut gradient.view_mut(),
            &stream,
        )?;
    }

    let gradient_staging = cx.download_from_device(gradient.storage(), &stream)?;
    stream.block_on()?;
    let gradient_out: &[f32] = bytemuck::cast_slice(gradient_staging.view()?);

    let data_f32 = to_f32(&data_host);
    let dy_f32 = to_f32(&dy_host);
    let data_ref = MatRef::from_row_major_slice(&data_f32, BATCH_SIZE, IN_FEATURES);
    let dy_ref = MatRef::from_row_major_slice(&dy_f32, BATCH_SIZE, OUT_FEATURES);
    let dw_ref = data_ref.transpose() * dy_ref;

    (0..IN_FEATURES * OUT_FEATURES)
        .into_par_iter()
        .try_for_each(|idx| {
            let (i, j) = (idx / OUT_FEATURES, idx % OUT_FEATURES);
            let expected = gradient_host[idx] + 2.0 * dw_ref[(i, j)];
            check(expected, gradient_out[idx], "gradient", i, j)
        })?;

    Ok(())
}

#[test]
fn linear_forward_with_residual() -> anyhow::Result<()> {
    let cx = &make_context()?;
    let stream = cx.cuda().create_stream()?;

    let executor = LinearExecutor::get(
        LinearConfig {
            estimated_batch_size: BATCH_SIZE / 2,
            in_features: IN_FEATURES,
            out_features: OUT_FEATURES,
            forward_fused_residual_add: true,
        },
        cx,
    )?;

    let bf16_type = ElementType::Float(FloatType::Bf16);
    let mut data = Tensor::new_zeroed([BATCH_SIZE, IN_FEATURES], bf16_type, cx)?;
    let mut weight = Tensor::new_zeroed([IN_FEATURES, OUT_FEATURES], bf16_type, cx)?;
    let mut residual = Tensor::new_zeroed([BATCH_SIZE, OUT_FEATURES], bf16_type, cx)?;

    let mut rng = StdRng::seed_from_u64(778);
    let scale = (IN_FEATURES as f32).sqrt().recip();
    let data_host = par_vec_from_distr(
        StandardNormal.map(|x: f32| bf16::from_f32(x * scale)),
        BATCH_SIZE * IN_FEATURES,
        &mut rng,
    );
    let weight_host = par_vec_from_distr(
        StandardNormal.map(bf16::from_f32),
        IN_FEATURES * OUT_FEATURES,
        &mut rng,
    );
    let residual_host = par_vec_from_distr(
        StandardNormal.map(bf16::from_f32),
        BATCH_SIZE * OUT_FEATURES,
        &mut rng,
    );

    cx.upload_to_device(&data_host, data.storage_mut(), &stream)?;
    cx.upload_to_device(&weight_host, weight.storage_mut(), &stream)?;
    cx.upload_to_device(&residual_host, residual.storage_mut(), &stream)?;

    let out =
        executor.forward_with_residual(&data.view(), &weight.view(), &residual.view(), &stream)?;

    let out_staging = cx.download_from_device(out.storage(), &stream)?;
    stream.block_on()?;
    let out_host: &[bf16] = bytemuck::cast_slice(out_staging.view()?);

    let data_f32 = to_f32(&data_host);
    let weight_f32 = to_f32(&weight_host);
    let data_ref = MatRef::from_row_major_slice(&data_f32, BATCH_SIZE, IN_FEATURES);
    let weight_ref = MatRef::from_row_major_slice(&weight_f32, IN_FEATURES, OUT_FEATURES);
    let y_ref = data_ref * weight_ref;

    (0..BATCH_SIZE * OUT_FEATURES)
        .into_par_iter()
        .try_for_each(|idx| {
            let (i, j) = (idx / OUT_FEATURES, idx % OUT_FEATURES);
            let expected = y_ref[(i, j)] + residual_host[idx].to_f32();
            check(expected, out_host[idx].to_f32(), "output", i, j)
        })?;

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
