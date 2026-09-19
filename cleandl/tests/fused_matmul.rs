use cleandl::{
    context::Context,
    cudnn::CudnnGraphBuilder,
    tensor::{Tensor, VirtTensorView},
};
use faer::Mat;
use float_cmp::{ApproxEq, F32Margin};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
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
        DeviceMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        false,
    )?;
    let stream = cx.cuda().create_stream()?;

    let bf16_type = ElementType::Float(FloatType::Bf16);
    let mut a = Tensor::new_zeroed(Shape::new([M_ACTUAL, K]), bf16_type, &cx)?;
    let mut b = Tensor::new_zeroed(Shape::new([K, N]), bf16_type, &cx)?;
    let mut c = Tensor::new_zeroed(Shape::new([M_ACTUAL, N]), bf16_type, &cx)?;

    let mut rng = StdRng::seed_from_u64(555);
    let scale = (K as f32).sqrt().recip();
    let a_host: Vec<bf16> = (0..M_ACTUAL * K)
        .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal) * scale))
        .collect();
    let b_host: Vec<bf16> = (0..K * N)
        .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal)))
        .collect();

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
    )?;

    // Read the result back to the host.
    let c_staging = cx.download_from_device(c.storage(), &stream)?;
    stream.block_on()?;
    let c_host: &[bf16] = bytemuck::cast_slice(c_staging.view()?);

    let a_ref = Mat::from_fn(M_ACTUAL, K, |i, j| a_host[i * K + j].to_f32());
    let b_ref = Mat::from_fn(K, N, |i, j| b_host[i * N + j].to_f32());
    let c_ref = a_ref * b_ref;

    let mut max_error = 0.0f32;
    for i in 0..M_ACTUAL {
        for j in 0..N {
            let expected = swish(c_ref[(i, j)]);
            let actual = c_host[i * N + j].to_f32();
            assert!(actual.is_finite());

            max_error = max_error.max((actual - expected).abs());

            let margin = F32Margin {
                epsilon: ABS_TOLERANCE.max(REL_TOLERANCE * expected.abs()),
                ulps: 0,
            };
            anyhow::ensure!(
                actual.approx_eq(expected, margin),
                "mismatch at ({i}, {j}): expected {expected}, got {actual}"
            );
        }
    }

    println!("results match the reference (max absolute error: {max_error})");

    Ok(())
}

/// Swish (SiLU) activation: `x * sigmoid(x)`.
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
