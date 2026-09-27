// --- LLM-generated --- //

use cleandl::{
    context::Context,
    operators::{
        matmul::MatrixFormat,
        sky_matmul::{
            IdentityEpilogue, SkyMatmulConfig, SkyMatmulExecutor, SwiGluGateTrainFwdEpilogue,
            SwiGluGateTrainFwdOutputs,
        },
    },
    tensor::Tensor,
};
use faer::Mat;
use float_cmp::{ApproxEq, F32Margin};
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

const TUNE_M: usize = 4096;
const TUNE_N: usize = 2048;
const TUNE_K: usize = 2048;

const REL_TOLERANCE: f32 = 2e-2;
const ABS_TOLERANCE: f32 = 1e-3;

fn new_context() -> anyhow::Result<Context> {
    Context::new(
        0,
        DeviceMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        false,
    )
}

/// Test shapes: the tuned shape, plus shapes with partial CGA tiles and a K extent
/// that is not a multiple of the K tile.
const CASES: [(usize, usize, usize); 3] =
    [(TUNE_M, TUNE_N, TUNE_K), (3000, 1320, 1000), (200, 264, 72)];

#[test]
fn sky_matmul_identity() -> anyhow::Result<()> {
    let cx = new_context()?;
    let stream = cx.cuda().create_stream()?;

    let executor = SkyMatmulExecutor::get(
        SkyMatmulConfig {
            tune_for_m: TUNE_M as u32,
            tune_for_n: TUNE_N as u32,
            tune_for_k: TUNE_K as u32,
            format_a: MatrixFormat::RowMajor,
            format_b: MatrixFormat::ColumnMajor,
            epilogue: IdentityEpilogue,
        },
        &cx,
    )?;
    println!("using native backend: {}", executor.is_native());

    for (m, n, k) in CASES {
        let inputs = Inputs::new_random(&cx, &stream, m, n, k)?;
        let c = executor.execute(&inputs.a.view(), &inputs.b_t.transposed(0, 1), (), &stream)?;
        assert_eq!(c.shape(), Shape::new([m, n]));
        assert!(c.is_packed_row_major());

        let c_host = download(&cx, &stream, &c)?;
        let c_ref = inputs.reference_product();

        let max_error = check_matrix(&c_host, (m, n, k), |i, j| c_ref[(i, j)], "c")?;
        println!("{m}x{n}x{k}: results match the reference (max absolute error: {max_error})");
    }

    Ok(())
}

#[test]
fn sky_matmul_swiglu_gate_train_fwd() -> anyhow::Result<()> {
    let cx = new_context()?;
    let stream = cx.cuda().create_stream()?;

    let executor = SkyMatmulExecutor::get(
        SkyMatmulConfig {
            tune_for_m: TUNE_M as u32,
            tune_for_n: TUNE_N as u32,
            tune_for_k: TUNE_K as u32,
            format_a: MatrixFormat::RowMajor,
            format_b: MatrixFormat::ColumnMajor,
            epilogue: SwiGluGateTrainFwdEpilogue,
        },
        &cx,
    )?;
    println!("using native backend: {}", executor.is_native());

    for (m, n, k) in CASES {
        let inputs = Inputs::new_random(&cx, &stream, m, n, k)?;

        let mut rng = StdRng::seed_from_u64(0xf00d + (m * n * k) as u64);
        let x_host: Vec<bf16> = (0..m * n)
            .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal)))
            .collect();
        let mut x = Tensor::new_zeroed(Shape::new([m, n]), BF16, &cx)?;
        cx.upload_to_device(&x_host, x.storage_mut(), &stream)?;

        let SwiGluGateTrainFwdOutputs { gate, activation } = executor.execute(
            &inputs.a.view(),
            &inputs.b_t.transposed(0, 1),
            &x.view(),
            &stream,
        )?;
        for out in [&gate, &activation] {
            assert_eq!(out.shape(), Shape::new([m, n]));
            assert!(out.is_packed_row_major());
        }

        let gate_host = download(&cx, &stream, &gate)?;
        let activation_host = download(&cx, &stream, &activation)?;
        let gate_ref = inputs.reference_product();

        let max_gate_error = check_matrix(&gate_host, (m, n, k), |i, j| gate_ref[(i, j)], "gate")?;
        let max_activation_error = check_matrix(
            &activation_host,
            (m, n, k),
            |i, j| swish(gate_ref[(i, j)]) * x_host[i * n + j].to_f32(),
            "activation",
        )?;
        println!(
            "{m}x{n}x{k}: results match the reference (max absolute error: \
             gate {max_gate_error}, activation {max_activation_error})"
        );
    }

    Ok(())
}

const BF16: ElementType = ElementType::Float(FloatType::Bf16);

/// Random matmul inputs: A is row-major `[m, k]`, B is column-major `[k, n]`,
/// i.e. stored as row-major `[n, k]`.
struct Inputs {
    m: usize,
    n: usize,
    k: usize,
    a_host: Vec<bf16>,
    b_host: Vec<bf16>,
    a: Tensor,
    b_t: Tensor,
}

impl Inputs {
    fn new_random(
        cx: &Context,
        stream: &Stream,
        m: usize,
        n: usize,
        k: usize,
    ) -> anyhow::Result<Self> {
        let mut rng = StdRng::seed_from_u64(0x5c1 + (m * n * k) as u64);
        let scale = (k as f32).sqrt().recip();
        let a_host: Vec<bf16> = (0..m * k)
            .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal) * scale))
            .collect();
        let b_host: Vec<bf16> = (0..n * k)
            .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal)))
            .collect();

        let mut a = Tensor::new_zeroed(Shape::new([m, k]), BF16, cx)?;
        let mut b_t = Tensor::new_zeroed(Shape::new([n, k]), BF16, cx)?;
        cx.upload_to_device(&a_host, a.storage_mut(), stream)?;
        cx.upload_to_device(&b_host, b_t.storage_mut(), stream)?;

        Ok(Self {
            m,
            n,
            k,
            a_host,
            b_host,
            a,
            b_t,
        })
    }

    fn reference_product(&self) -> Mat<f32> {
        let (m, n, k) = (self.m, self.n, self.k);
        let a_ref = Mat::from_fn(m, k, |i, j| self.a_host[i * k + j].to_f32());
        let b_ref = Mat::from_fn(k, n, |i, j| self.b_host[j * k + i].to_f32());
        a_ref * b_ref
    }
}

fn download(cx: &Context, stream: &Stream, tensor: &Tensor) -> anyhow::Result<Vec<bf16>> {
    let staging = cx.download_from_device(tensor.storage(), stream)?;
    stream.block_on()?;
    Ok(bytemuck::cast_slice(staging.view()?).to_vec())
}

/// Checks a row-major `[m, n]` result against a reference, returning the max absolute error.
fn check_matrix(
    actual: &[bf16],
    (m, n, k): (usize, usize, usize),
    expected: impl Fn(usize, usize) -> f32,
    name: &str,
) -> anyhow::Result<f32> {
    let mut max_error = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let expected = expected(i, j);
            let actual = actual[i * n + j].to_f32();
            assert!(actual.is_finite());

            max_error = max_error.max((actual - expected).abs());

            let margin = F32Margin {
                epsilon: ABS_TOLERANCE.max(REL_TOLERANCE * expected.abs()),
                ulps: 0,
            };
            anyhow::ensure!(
                actual.approx_eq(expected, margin),
                "{m}x{n}x{k}: {name} mismatch at ({i}, {j}): expected {expected}, got {actual}"
            );
        }
    }
    Ok(max_error)
}

/// Swish (SiLU) activation: `x * sigmoid(x)`.
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
