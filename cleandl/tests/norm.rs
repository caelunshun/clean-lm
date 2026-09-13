use cleandl::{context::Context, operators::norm::NormExecutor, tensor::Tensor};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions, stream::Stream,
};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

const BATCH_SIZE: usize = 517;
const INNER_DIM: usize = 2048;
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
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
    )?;
    let stream = cx.cuda().create_stream()?;
    let norm = NormExecutor::get(&cx)?;

    for with_residuals in [true, false] {
        run_case(&cx, &stream, &norm, with_residuals)?;
    }

    Ok(())
}

fn run_case(
    cx: &Context,
    stream: &Stream,
    norm: &NormExecutor,
    with_residuals: bool,
) -> anyhow::Result<()> {
    let mut rng = StdRng::seed_from_u64(0x9e11);
    let mut normal = |scale: f32| -> Vec<bf16> {
        (0..BATCH_SIZE * INNER_DIM)
            .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal) * scale))
            .collect()
    };

    let prev_layer_output_host = normal(1.0);
    let dy_host = normal(1.0);
    let residual_gradient_stream_host = normal(0.5);
    let gamma_host: Vec<bf16> = (0..INNER_DIM)
        .map(|_| bf16::from_f32(1.0 + rng.sample::<f32, _>(StandardNormal) * 0.1))
        .collect();

    let matrix_shape = Shape::new([BATCH_SIZE, INNER_DIM]);
    let bf16_type = ElementType::Float(FloatType::Bf16);
    let f32_type = ElementType::Float(FloatType::F32);

    let upload = |data: &[bf16], shape: Shape| -> anyhow::Result<Tensor> {
        let mut tensor = Tensor::new_zeroed(shape, bf16_type, stream, cx)?;
        cx.upload_to_device(data, tensor.storage_mut(), stream)?;
        Ok(tensor)
    };

    let prev_layer_output = upload(&prev_layer_output_host, matrix_shape)?;
    let dy = upload(&dy_host, matrix_shape)?;
    let residual_gradient_stream = upload(&residual_gradient_stream_host, matrix_shape)?;
    let gamma = upload(&gamma_host, Shape::new([INNER_DIM]))?;

    let mut y = Tensor::new_zeroed(matrix_shape, bf16_type, stream, cx)?;
    let mut r_inv = Tensor::new_zeroed(Shape::new([BATCH_SIZE]), f32_type, stream, cx)?;
    let mut dx = Tensor::new_zeroed(matrix_shape, bf16_type, stream, cx)?;
    let mut dgamma = Tensor::new_zeroed(Shape::new([INNER_DIM]), f32_type, stream, cx)?;

    let residual_gradient_stream_view = with_residuals.then(|| residual_gradient_stream.view());

    norm.rmsnorm_forward(
        &prev_layer_output.view(),
        &gamma.view(),
        &mut y.view_mut(),
        &mut r_inv.view_mut(),
        stream,
    )?;

    // The kernel accumulates dgamma atomically.
    stream.fill(&mut dgamma.view_mut().storage_mut(), 0u8)?;

    norm.rmsnorm_backward(
        &prev_layer_output.view(),
        &dy.view(),
        residual_gradient_stream_view.as_ref(),
        &gamma.view(),
        &r_inv.view(),
        &mut dx.view_mut(),
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
    );

    let check = |name: &str, i: usize, actual: f32, expected: f32, tolerance: f32| {
        anyhow::ensure!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{name} mismatch at index {i} (with_residuals = {with_residuals}): \
             expected {expected}, got {actual}"
        );
        Ok(())
    };

    let mut max_y_error = 0.0f32;
    let mut max_dx_error = 0.0f32;

    for row in 0..BATCH_SIZE {
        check(
            "r_inv",
            row,
            r_inv_out[row],
            expected.r_inv[row],
            R_INV_REL_TOLERANCE * expected.r_inv[row].abs(),
        )?;
    }

    for i in 0..BATCH_SIZE * INNER_DIM {
        let y_actual = y_out[i].to_f32();
        assert!(y_actual.is_finite());
        max_y_error = max_y_error.max((y_actual - expected.y[i]).abs());
        check(
            "y",
            i,
            y_actual,
            expected.y[i],
            Y_REL_TOLERANCE * expected.y[i].abs(),
        )?;

        let dx_actual = dx_out[i].to_f32();
        max_dx_error = max_dx_error.max((dx_actual - expected.dx[i]).abs());
        check(
            "dx",
            i,
            dx_actual,
            expected.dx[i],
            DX_ABS_TOLERANCE.max(DX_REL_TOLERANCE * expected.dx[i].abs()),
        )?;
    }

    let mut max_dgamma_error = 0.0f32;
    for i in 0..INNER_DIM {
        assert!(dgamma_out[i].is_finite());
        max_dgamma_error = max_dgamma_error.max((dgamma_out[i] - expected.dgamma[i]).abs());
        // dgamma accumulates over the whole batch, so tolerate error relative
        // to the magnitude of the summed terms rather than of the result.
        check(
            "dgamma",
            i,
            dgamma_out[i],
            expected.dgamma[i],
            DGAMMA_REL_TOLERANCE * expected.dgamma_abs_sum[i],
        )?;
    }

    println!(
        "results match the reference (with_residuals: {with_residuals}, max absolute errors: \
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
) -> Reference {
    let mut out = Reference {
        r_inv: Vec::with_capacity(BATCH_SIZE),
        y: vec![0.0; BATCH_SIZE * INNER_DIM],
        dx: vec![0.0; BATCH_SIZE * INNER_DIM],
        dgamma: vec![0.0; INNER_DIM],
        dgamma_abs_sum: vec![0.0; INNER_DIM],
    };

    let gamma: Vec<f32> = gamma.iter().map(|x| x.to_f32()).collect();

    for row in 0..BATCH_SIZE {
        let range = row * INNER_DIM..(row + 1) * INNER_DIM;
        let x: Vec<f32> = (range.clone())
            .map(|i| prev_layer_output[i].to_f32())
            .collect();

        let mean_square = x.iter().map(|&v| (v * v) as f64).sum::<f64>() / INNER_DIM as f64;
        let r_inv = (mean_square as f32 + EPSILON).sqrt().recip();
        out.r_inv.push(r_inv);

        // Forward pass.
        for (col, &v) in x.iter().enumerate() {
            out.y[row * INNER_DIM + col] = v * r_inv * gamma[col];
        }

        // Backward pass.
        let x_hat: Vec<f32> = x.iter().map(|&v| v * r_inv).collect();
        let g: Vec<f32> = (0..INNER_DIM)
            .map(|col| dy[row * INNER_DIM + col].to_f32() * gamma[col])
            .collect();
        let c = (0..INNER_DIM)
            .map(|col| (g[col] * x_hat[col]) as f64)
            .sum::<f64>() as f32
            / INNER_DIM as f32;

        for col in 0..INNER_DIM {
            let i = row * INNER_DIM + col;
            out.dx[i] = r_inv * (g[col] - x_hat[col] * c)
                + residual_gradient_stream.map_or(0.0, |residual| residual[i].to_f32());

            let dgamma_term = dy[i].to_f32() * x_hat[col];
            out.dgamma[col] += dgamma_term;
            out.dgamma_abs_sum[col] += dgamma_term.abs();
        }
    }

    out
}
