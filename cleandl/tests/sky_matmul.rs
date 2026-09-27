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
    par_random::par_vec_from_distr,
    tensor::{Tensor, TensorView},
};
use faer::{Mat, MatRef};
use float_cmp::{ApproxEq, F32Margin};
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

/// All supported (A format, B format) combinations.
const FORMATS: [(MatrixFormat, MatrixFormat); 4] = [
    (MatrixFormat::RowMajor, MatrixFormat::RowMajor),
    (MatrixFormat::RowMajor, MatrixFormat::ColumnMajor),
    (MatrixFormat::ColumnMajor, MatrixFormat::RowMajor),
    (MatrixFormat::ColumnMajor, MatrixFormat::ColumnMajor),
];

#[test]
fn sky_matmul_identity() -> anyhow::Result<()> {
    let cx = new_context()?;
    let stream = cx.cuda().create_stream()?;

    for (format_a, format_b) in FORMATS {
        let executor = SkyMatmulExecutor::get(
            SkyMatmulConfig {
                tune_for_m: TUNE_M as u32,
                tune_for_n: TUNE_N as u32,
                tune_for_k: TUNE_K as u32,
                format_a,
                format_b,
                epilogue: IdentityEpilogue,
            },
            &cx,
        )?;
        println!(
            "A {format_a:?}, B {format_b:?}: using native backend: {}",
            executor.is_native()
        );

        for (m, n, k) in CASES {
            let inputs = Inputs::new_random(&cx, &stream, m, n, k, format_a, format_b)?;
            let c = executor.execute(&inputs.a_view(), &inputs.b_view(), (), &stream)?;
            assert_eq!(c.shape(), Shape::new([m, n]));
            assert!(c.is_packed_row_major());

            let c_host = download(&cx, &stream, &c)?;
            let c_ref = inputs.reference_product();

            let max_error = check_matrix(&c_host, &inputs, |i, j| c_ref[(i, j)], "c")?;
            println!(
                "A {format_a:?}, B {format_b:?}, {m}x{n}x{k}: results match the reference \
                 (max absolute error: {max_error})"
            );
        }
    }

    Ok(())
}

#[test]
fn sky_matmul_swiglu_gate_train_fwd() -> anyhow::Result<()> {
    let cx = new_context()?;
    let stream = cx.cuda().create_stream()?;

    for (format_a, format_b) in FORMATS {
        let executor = SkyMatmulExecutor::get(
            SkyMatmulConfig {
                tune_for_m: TUNE_M as u32,
                tune_for_n: TUNE_N as u32,
                tune_for_k: TUNE_K as u32,
                format_a,
                format_b,
                epilogue: SwiGluGateTrainFwdEpilogue,
            },
            &cx,
        )?;
        println!(
            "A {format_a:?}, B {format_b:?}: using native backend: {}",
            executor.is_native()
        );

        for (m, n, k) in CASES {
            let inputs = Inputs::new_random(&cx, &stream, m, n, k, format_a, format_b)?;

            let mut rng = StdRng::seed_from_u64(0xf00d + (m * n * k) as u64);
            let x_host = par_vec_from_distr(StandardNormal.map(bf16::from_f32), m * n, &mut rng);
            let mut x = Tensor::new_zeroed(Shape::new([m, n]), BF16, &cx)?;
            cx.upload_to_device(&x_host, x.storage_mut(), &stream)?;

            let SwiGluGateTrainFwdOutputs { gate, activation } =
                executor.execute(&inputs.a_view(), &inputs.b_view(), &x.view(), &stream)?;
            for out in [&gate, &activation] {
                assert_eq!(out.shape(), Shape::new([m, n]));
                assert!(out.is_packed_row_major());
            }

            let gate_host = download(&cx, &stream, &gate)?;
            let activation_host = download(&cx, &stream, &activation)?;
            let gate_ref = inputs.reference_product();

            let max_gate_error =
                check_matrix(&gate_host, &inputs, |i, j| gate_ref[(i, j)], "gate")?;
            let max_activation_error = check_matrix(
                &activation_host,
                &inputs,
                |i, j| swish(gate_ref[(i, j)]) * x_host[i * n + j].to_f32(),
                "activation",
            )?;
            println!(
                "A {format_a:?}, B {format_b:?}, {m}x{n}x{k}: results match the reference \
                 (max absolute error: gate {max_gate_error}, activation {max_activation_error})"
            );
        }
    }

    Ok(())
}

const BF16: ElementType = ElementType::Float(FloatType::Bf16);

/// Random matmul inputs: A is `[m, k]` and B is `[k, n]`, each stored in the given format.
struct Inputs {
    m: usize,
    n: usize,
    k: usize,
    format_a: MatrixFormat,
    format_b: MatrixFormat,
    a_host: Vec<bf16>,
    b_host: Vec<bf16>,
    /// Stored as row-major `[m, k]` if A is row-major, else as row-major `[k, m]`.
    a: Tensor,
    /// Stored as row-major `[k, n]` if B is row-major, else as row-major `[n, k]`.
    b: Tensor,
}

impl Inputs {
    fn new_random(
        cx: &Context,
        stream: &Stream,
        m: usize,
        n: usize,
        k: usize,
        format_a: MatrixFormat,
        format_b: MatrixFormat,
    ) -> anyhow::Result<Self> {
        let mut rng = StdRng::seed_from_u64(0x5c1 + (m * n * k) as u64);
        let scale = (k as f32).sqrt().recip();
        let a_host = par_vec_from_distr(
            StandardNormal.map(|x: f32| bf16::from_f32(x * scale)),
            m * k,
            &mut rng,
        );
        let b_host = par_vec_from_distr(StandardNormal.map(bf16::from_f32), n * k, &mut rng);

        let a_shape = match format_a {
            MatrixFormat::RowMajor => [m, k],
            MatrixFormat::ColumnMajor => [k, m],
        };
        let b_shape = match format_b {
            MatrixFormat::RowMajor => [k, n],
            MatrixFormat::ColumnMajor => [n, k],
        };
        let mut a = Tensor::new_zeroed(Shape::new(a_shape), BF16, cx)?;
        let mut b = Tensor::new_zeroed(Shape::new(b_shape), BF16, cx)?;
        cx.upload_to_device(&a_host, a.storage_mut(), stream)?;
        cx.upload_to_device(&b_host, b.storage_mut(), stream)?;

        Ok(Self {
            m,
            n,
            k,
            format_a,
            format_b,
            a_host,
            b_host,
            a,
            b,
        })
    }

    /// A as a `[m, k]` view.
    fn a_view(&self) -> TensorView<'_> {
        match self.format_a {
            MatrixFormat::RowMajor => self.a.view(),
            MatrixFormat::ColumnMajor => self.a.transposed(0, 1),
        }
    }

    /// B as a `[k, n]` view.
    fn b_view(&self) -> TensorView<'_> {
        match self.format_b {
            MatrixFormat::RowMajor => self.b.view(),
            MatrixFormat::ColumnMajor => self.b.transposed(0, 1),
        }
    }

    fn reference_product(&self) -> Mat<f32> {
        let (m, n, k) = (self.m, self.n, self.k);
        let a_f32: Vec<f32> = self.a_host.par_iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = self.b_host.par_iter().map(|x| x.to_f32()).collect();
        let a_ref = match self.format_a {
            MatrixFormat::RowMajor => MatRef::from_row_major_slice(&a_f32, m, k),
            MatrixFormat::ColumnMajor => MatRef::from_column_major_slice(&a_f32, m, k),
        };
        let b_ref = match self.format_b {
            MatrixFormat::RowMajor => MatRef::from_row_major_slice(&b_f32, k, n),
            MatrixFormat::ColumnMajor => MatRef::from_column_major_slice(&b_f32, k, n),
        };
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
    inputs: &Inputs,
    expected: impl Fn(usize, usize) -> f32 + Sync,
    name: &str,
) -> anyhow::Result<f32> {
    let &Inputs {
        m,
        n,
        k,
        format_a,
        format_b,
        ..
    } = inputs;
    (0..m * n)
        .into_par_iter()
        .try_fold(
            || 0.0f32,
            |max_error, idx| -> anyhow::Result<f32> {
                let (i, j) = (idx / n, idx % n);
                let expected = expected(i, j);
                let actual = actual[idx].to_f32();
                assert!(actual.is_finite());

                let margin = F32Margin {
                    epsilon: ABS_TOLERANCE.max(REL_TOLERANCE * expected.abs()),
                    ulps: 0,
                };
                anyhow::ensure!(
                    actual.approx_eq(expected, margin),
                    "A {format_a:?}, B {format_b:?}, {m}x{n}x{k}: {name} mismatch at ({i}, {j}): expected {expected}, got {actual}"
                );

                Ok(max_error.max((actual - expected).abs()))
            },
        )
        .try_reduce(|| 0.0, |a, b| Ok(a.max(b)))
}

/// Swish (SiLU) activation: `x * sigmoid(x)`.
fn swish(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
