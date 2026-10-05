use cleandl::{
    context::Context,
    operators::norm::{NormExecutor, RmsNormForwardOutput},
    par_random::par_vec_from_distr,
    tensor::Tensor,
};
use half::bf16;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions, stream::Stream,
};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

const BATCH_SIZE: usize = 32769;
const INNER_DIMS: &[usize] = &[2048, 1536];
const EPSILON: f32 = 1e-6;

const R_INV_REL_TOLERANCE: f32 = 1e-3;
const Y_REL_TOLERANCE: f32 = 2e-2;
const DX_REL_TOLERANCE: f32 = 2e-2;
const DX_ABS_TOLERANCE: f32 = 1e-2;
const DGAMMA_REL_TOLERANCE: f32 = 1e-3;

#[test]
fn rms_prenorm() -> anyhow::Result<()> {
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
    let norm = NormExecutor::get(&cx)?;

    for &inner_dim in INNER_DIMS {
        for with_residuals in [true, false] {
            run_case(&cx, &stream, &norm, inner_dim, with_residuals)?;
        }
    }

    Ok(())
}

fn run_case(
    cx: &Context,
    stream: &Stream,
    norm: &NormExecutor,
    inner_dim: usize,
    with_residuals: bool,
) -> anyhow::Result<()> {
    let mut rng = StdRng::seed_from_u64(0x9e11);
    let mut normal = |mean: f32, scale: f32, len: usize| -> Vec<bf16> {
        par_vec_from_distr(
            StandardNormal.map(|x: f32| bf16::from_f32(mean + x * scale)),
            len,
            &mut rng,
        )
    };

    let prev_layer_output_host = normal(0.0, 1.0, BATCH_SIZE * inner_dim);
    let dy_host = normal(0.0, 1.0, BATCH_SIZE * inner_dim);
    let residual_gradient_stream_host = normal(0.0, 0.5, BATCH_SIZE * inner_dim);
    let gamma_host = normal(1.0, 0.1, inner_dim);

    let matrix_shape = Shape::new([BATCH_SIZE, inner_dim]);
    let bf16_type = ElementType::Float(FloatType::Bf16);
    let f32_type = ElementType::Float(FloatType::F32);

    let upload = |data: &[bf16], shape: Shape| -> anyhow::Result<Tensor> {
        let mut tensor = Tensor::new_zeroed(shape, bf16_type, cx)?;
        cx.upload_to_device(data, tensor.storage_mut(), stream)?;
        Ok(tensor)
    };

    let prev_layer_output = upload(&prev_layer_output_host, matrix_shape)?;
    let dy = upload(&dy_host, matrix_shape)?;
    let residual_gradient_stream = upload(&residual_gradient_stream_host, matrix_shape)?;
    let gamma = upload(&gamma_host, Shape::new([inner_dim]))?;

    let mut dgamma = Tensor::new_zeroed(Shape::new([inner_dim]), f32_type, cx)?;

    let residual_gradient_stream_view = with_residuals.then(|| residual_gradient_stream.view());

    let RmsNormForwardOutput { y, r_inv } =
        norm.rmsnorm_forward(&prev_layer_output.view(), &gamma.view(), stream)?;

    // The kernel accumulates dgamma atomically.
    stream.fill(&mut dgamma.view_mut().storage_mut(), 0u8)?;

    let dx = norm.rmsnorm_backward(
        &prev_layer_output.view(),
        &dy.view(),
        residual_gradient_stream_view.as_ref(),
        &gamma.view(),
        &r_inv.view(),
        &mut dgamma.view_mut(),
        stream,
    )?;

    let y_staging = cx.download_from_device(y.storage(), stream)?;
    let r_inv_staging = cx.download_from_device(r_inv.storage(), stream)?;
    let dx_staging = cx.download_from_device(dx.storage(), stream)?;
    let dgamma_staging = cx.download_from_device(dgamma.storage(), stream)?;
    stream.block_on()?;

    let y_out: &[bf16] = bytemuck::cast_slice(y_staging.view()?);
    let r_inv_out: &[f32] = bytemuck::cast_slice(r_inv_staging.view()?);
    let dx_out: &[bf16] = bytemuck::cast_slice(dx_staging.view()?);
    let dgamma_out: &[f32] = bytemuck::cast_slice(dgamma_staging.view()?);

    let expected = reference(
        &prev_layer_output_host,
        &dy_host,
        with_residuals.then_some(residual_gradient_stream_host.as_slice()),
        &gamma_host,
        inner_dim,
    );

    let check = |name: &str, i: usize, actual: f32, expected: f32, tolerance: f32| {
        anyhow::ensure!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{name} mismatch at index {i} (inner_dim = {inner_dim}, \
             with_residuals = {with_residuals}): \
             expected {expected}, got {actual}"
        );
        Ok(())
    };

    (0..BATCH_SIZE).into_par_iter().try_for_each(|row| {
        check(
            "r_inv",
            row,
            r_inv_out[row],
            expected.r_inv[row],
            R_INV_REL_TOLERANCE * expected.r_inv[row].abs(),
        )
    })?;

    let (max_y_error, max_dx_error) = (0..BATCH_SIZE * inner_dim)
        .into_par_iter()
        .try_fold(
            || (0.0f32, 0.0f32),
            |(max_y_error, max_dx_error), i| -> anyhow::Result<(f32, f32)> {
                let y_actual = y_out[i].to_f32();
                assert!(y_actual.is_finite());
                check(
                    "y",
                    i,
                    y_actual,
                    expected.y[i],
                    Y_REL_TOLERANCE * expected.y[i].abs(),
                )?;

                let dx_actual = dx_out[i].to_f32();
                check(
                    "dx",
                    i,
                    dx_actual,
                    expected.dx[i],
                    DX_ABS_TOLERANCE.max(DX_REL_TOLERANCE * expected.dx[i].abs()),
                )?;

                Ok((
                    max_y_error.max((y_actual - expected.y[i]).abs()),
                    max_dx_error.max((dx_actual - expected.dx[i]).abs()),
                ))
            },
        )
        .try_reduce(
            || (0.0, 0.0),
            |(a_y, a_dx), (b_y, b_dx)| Ok((a_y.max(b_y), a_dx.max(b_dx))),
        )?;

    let max_dgamma_error = (0..inner_dim)
        .into_par_iter()
        .try_fold(
            || 0.0f32,
            |max_dgamma_error, i| -> anyhow::Result<f32> {
                assert!(dgamma_out[i].is_finite());
                // dgamma accumulates over the whole batch, so tolerate error relative
                // to the magnitude of the summed terms rather than of the result.
                check(
                    "dgamma",
                    i,
                    dgamma_out[i],
                    expected.dgamma[i],
                    DGAMMA_REL_TOLERANCE * expected.dgamma_abs_sum[i],
                )?;
                Ok(max_dgamma_error.max((dgamma_out[i] - expected.dgamma[i]).abs()))
            },
        )
        .try_reduce(|| 0.0, |a, b| Ok(a.max(b)))?;

    println!(
        "results match the reference (inner_dim: {inner_dim}, with_residuals: {with_residuals}, \
         max absolute errors: \
         y {max_y_error}, dx {max_dx_error}, dgamma {max_dgamma_error})"
    );

    Ok(())
}

struct Reference {
    r_inv: Vec<f32>,
    y: Vec<f32>,
    dx: Vec<f32>,
    dgamma: Vec<f32>,
    dgamma_abs_sum: Vec<f32>,
}

fn reference(
    prev_layer_output: &[bf16],
    dy: &[bf16],
    residual_gradient_stream: Option<&[bf16]>,
    gamma: &[bf16],
    inner_dim: usize,
) -> Reference {
    let gamma: Vec<f32> = gamma.iter().map(|x| x.to_f32()).collect();

    let r_inv: Vec<f32> = prev_layer_output
        .par_chunks(inner_dim)
        .map(|x| {
            let mean_square = x
                .iter()
                .map(|v| (v.to_f32() * v.to_f32()) as f64)
                .sum::<f64>()
                / inner_dim as f64;
            (mean_square as f32 + EPSILON).sqrt().recip()
        })
        .collect();

    let mut y = vec![0.0; BATCH_SIZE * inner_dim];
    let mut dx = vec![0.0; BATCH_SIZE * inner_dim];

    let (dgamma, dgamma_abs_sum) = y
        .par_chunks_mut(inner_dim)
        .zip(dx.par_chunks_mut(inner_dim))
        .enumerate()
        .fold(
            || (vec![0.0f32; inner_dim], vec![0.0f32; inner_dim]),
            |(mut dgamma, mut dgamma_abs_sum), (row, (y, dx))| {
                let range = row * inner_dim..(row + 1) * inner_dim;
                let x: Vec<f32> = prev_layer_output[range]
                    .iter()
                    .map(|v| v.to_f32())
                    .collect();
                let r_inv = r_inv[row];

                // Forward pass.
                for (col, &v) in x.iter().enumerate() {
                    y[col] = v * r_inv * gamma[col];
                }

                // Backward pass.
                let x_hat: Vec<f32> = x.iter().map(|&v| v * r_inv).collect();
                let g: Vec<f32> = (0..inner_dim)
                    .map(|col| dy[row * inner_dim + col].to_f32() * gamma[col])
                    .collect();
                let c = (0..inner_dim)
                    .map(|col| (g[col] * x_hat[col]) as f64)
                    .sum::<f64>() as f32
                    / inner_dim as f32;

                for col in 0..inner_dim {
                    let i = row * inner_dim + col;
                    dx[col] = r_inv * (g[col] - x_hat[col] * c)
                        + residual_gradient_stream.map_or(0.0, |residual| residual[i].to_f32());

                    let dgamma_term = dy[i].to_f32() * x_hat[col];
                    dgamma[col] += dgamma_term;
                    dgamma_abs_sum[col] += dgamma_term.abs();
                }

                (dgamma, dgamma_abs_sum)
            },
        )
        .reduce(
            || (vec![0.0f32; inner_dim], vec![0.0f32; inner_dim]),
            |(mut dgamma, mut dgamma_abs_sum), (other_dgamma, other_abs_sum)| {
                for col in 0..inner_dim {
                    dgamma[col] += other_dgamma[col];
                    dgamma_abs_sum[col] += other_abs_sum[col];
                }
                (dgamma, dgamma_abs_sum)
            },
        );

    Reference {
        r_inv,
        y,
        dx,
        dgamma,
        dgamma_abs_sum,
    }
}
