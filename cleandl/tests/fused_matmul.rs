use cleandl::{
    context::Context,
    cudnn::CudnnGraphBuilder,
    par_random::par_vec_from_distr,
    tensor::{Tensor, VirtTensorView},
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
use skycuda_cudnn::{DataType, MatmulAttributes, PointwiseAttributes, PointwiseMode};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

const M_BUILT: usize = 8192;
const M_ACTUAL: usize = 16384;
const N: usize = 8192;
const K: usize = 8192;

const REL_TOLERANCE: f32 = 2e-2;
const ABS_TOLERANCE: f32 = 1e-3;

#[test]
fn fused_matmul() -> anyhow::Result<()> {
    let cx = Context::new(
        0,
        DeviceMemoryPoolOptions::default(),
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        false,
    )?;
    let stream = cx.cuda().create_stream()?;

    let bf16_type = ElementType::Float(FloatType::Bf16);
    let mut a = Tensor::new_zeroed(Shape::new([M_ACTUAL, K]), bf16_type, &stream, &cx)?;
    let mut b = Tensor::new_zeroed(Shape::new([K, N]), bf16_type, &stream, &cx)?;
    let mut c = Tensor::new_zeroed(Shape::new([M_ACTUAL, N]), bf16_type, &stream, &cx)?;

    let mut rng = StdRng::seed_from_u64(555);
    let scale = (K as f32).sqrt().recip();
    let a_host = par_vec_from_distr(
        StandardNormal.map(|x: f32| bf16::from_f32(x * scale)),
        M_ACTUAL * K,
        &mut rng,
    );
    let b_host = par_vec_from_distr(StandardNormal.map(bf16::from_f32), K * N, &mut rng);

    cx.upload_to_device(&a_host, a.storage_mut(), &stream)?;
    cx.upload_to_device(&b_host, b.storage_mut(), &stream)?;

    let a_layout = VirtTensorView::new([1, M_BUILT, K], bf16_type);
    let b_layout = VirtTensorView::new([1, K, N], bf16_type);
    let c_layout = VirtTensorView::new([1, M_BUILT, N], bf16_type);

    let graph = CudnnGraphBuilder::new(&cx)?;
    let a_tensor = graph.bind_tensor(&a_layout);
    let b_tensor = graph.bind_tensor(&b_layout);
    let product = graph.graph().matmul(
        a_tensor,
        b_tensor,
        &MatmulAttributes::new().compute_data_type(DataType::Float),
    )?;
    graph.graph().set_override_shape_enabled(true);
    let activation = graph.graph().pointwise_unary(
        product,
        &PointwiseAttributes::new(PointwiseMode::SwishFwd).compute_data_type(DataType::Float),
    )?;
    let c_id = graph.mark_output(activation, &c_layout);
    let (a_id, b_id) = (a_tensor.uid(), b_tensor.uid());
    let graph = graph.build()?;

    graph.execute(
        &[
            (a_id, &a.reshaped([1, M_ACTUAL, K])),
            (b_id, &b.reshaped([1, K, N])),
        ],
        &mut [(c_id, c.reshaped_mut([1, M_ACTUAL, N]))],
        &stream,
        cx.device_memory_pool(),
    )?;

    // Read the result back to the host.
    let c_staging = cx.download_from_device(c.storage(), &stream)?;
    stream.block_on()?;
    let c_host: &[bf16] = bytemuck::cast_slice(c_staging.view()?);

    let a_f32 = to_f32(&a_host);
    let b_f32 = to_f32(&b_host);
    let a_ref = MatRef::from_row_major_slice(&a_f32, M_ACTUAL, K);
    let b_ref = MatRef::from_row_major_slice(&b_f32, K, N);
    let c_ref = a_ref * b_ref;

    let max_error = (0..M_ACTUAL * N)
        .into_par_iter()
        .try_fold(
            || 0.0f32,
            |max_error, idx| -> anyhow::Result<f32> {
                let (i, j) = (idx / N, idx % N);
                let expected = swish(c_ref[(i, j)]);
                let actual = c_host[idx].to_f32();
                assert!(actual.is_finite());

                let margin = F32Margin {
                    epsilon: ABS_TOLERANCE.max(REL_TOLERANCE * expected.abs()),
                    ulps: 0,
                };
                anyhow::ensure!(
                    actual.approx_eq(expected, margin),
                    "mismatch at ({i}, {j}): expected {expected}, got {actual}"
                );

                Ok(max_error.max((actual - expected).abs()))
            },
        )
        .try_reduce(|| 0.0, |a, b| Ok(a.max(b)))?;

    println!("results match the reference (max absolute error: {max_error})");

    Ok(())
}

fn to_f32(data: &[bf16]) -> Vec<f32> {
    data.par_iter().map(|x| x.to_f32()).collect()
}

/// Swish (SiLU) activation: `x * sigmoid(x)`.
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
