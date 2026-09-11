use cleandl::{
    bulk_copy::BulkCopyEngine,
    context::{Context, CudaStream},
    cudnn::CudnnGraphBuilder,
    tensor::TensorView,
};
use faer::Mat;
use float_cmp::{ApproxEq, F32Margin};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    data_type::DataType,
    frontend::operation::{MatmulConfig, PointwiseOperation},
    math::NanPropagation,
};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

const M: usize = 8192;
const N: usize = 8192;
const K: usize = 8192;

const REL_TOLERANCE: f32 = 2e-2;
const ABS_TOLERANCE: f32 = 1e-3;

#[test]
fn fused_matmul() -> anyhow::Result<()> {
    let cx = Context::new(0)?;

    let a = TensorView::new_zeroed(Shape::new([M, K]), ElementType::Float(FloatType::Bf16), &cx)?;
    let b = TensorView::new_zeroed(Shape::new([K, N]), ElementType::Float(FloatType::Bf16), &cx)?;
    let c = TensorView::new_zeroed(Shape::new([M, N]), ElementType::Float(FloatType::Bf16), &cx)?;

    let mut rng = StdRng::seed_from_u64(555);
    let scale = (K as f32).sqrt().recip();
    let a_host: Vec<bf16> = (0..M * K)
        .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal) * scale))
        .collect();
    let b_host: Vec<bf16> = (0..K * N)
        .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal)))
        .collect();

    let mut copy_engine = BulkCopyEngine::new(&cx)?;
    let mut batch = copy_engine.start_batch()?;
    batch.copy_to_device(&a, |dest| {
        dest.copy_from_slice(bytemuck::cast_slice(&a_host));
    })?;
    batch.copy_to_device(&b, |dest| {
        dest.copy_from_slice(bytemuck::cast_slice(&b_host));
    })?;
    batch.commit()?;

    let mut graph = CudnnGraphBuilder::new(&cx);
    let a_id = graph.bind_tensor(a.virt());
    let b_id = graph.bind_tensor(b.virt());
    let c_id = graph.bind_tensor(c.virt());
    let temp_id =
        graph
            .graph_mut()
            .matmul_infer_with_config(a_id, b_id, MatmulConfig::new(DataType::F32))?;
    graph.graph_mut().mark_as_virtual(temp_id)?;
    graph.graph_mut().pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::SwishFwd,
        input: temp_id,
        output: c_id,
        compute_type: DataType::F32,
        alpha1: 1.0,
        nan_propagation: NanPropagation::Propagate,
        axis: None,
    });

    let graph = graph.build()?;

    let stream = CudaStream::new(&cx)?;
    copy_engine.wait_for_last_batch_on(&stream)?;

    graph.execute([(a_id, &a), (b_id, &b), (c_id, &c)], &stream)?;

    cx.cuda().synchronize()?;

    // Read the result back to the host.
    let mut batch = copy_engine.start_batch()?;
    let token = batch.copy_to_host(&c, None)?;
    batch.commit()?;
    let c_host: &[bf16] = bytemuck::cast_slice(copy_engine.retrieve(token)?);

    let a_ref = Mat::from_fn(M, K, |i, j| a_host[i * K + j].to_f32());
    let b_ref = Mat::from_fn(K, N, |i, j| b_host[i * N + j].to_f32());
    let c_ref = a_ref * b_ref;

    let mut max_error = 0.0f32;
    for i in 0..M {
        for j in 0..N {
            let expected = swish(c_ref[(i, j)]);
            let actual = c_host[i * N + j].to_f32();

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
