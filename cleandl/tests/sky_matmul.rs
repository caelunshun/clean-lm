// --- LLM-generated --- //

use cleandl::{
    context::Context,
    operators::{
        matmul::MatrixFormat,
        sky_matmul::{IdentityEpilogue, SkyMatmulConfig, SkyMatmulExecutor},
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

#[test]
fn sky_matmul_identity() -> anyhow::Result<()> {
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

    // The tuned shape, plus shapes with partial CGA tiles and a K extent
    // that is not a multiple of the K tile.
    for (m, n, k) in [(TUNE_M, TUNE_N, TUNE_K), (3000, 1320, 1000), (200, 264, 72)] {
        run_case(&cx, &stream, &executor, m, n, k)?;
    }

    Ok(())
}

fn run_case(
    cx: &Context,
    stream: &Stream,
    executor: &SkyMatmulExecutor<IdentityEpilogue>,
    m: usize,
    n: usize,
    k: usize,
) -> anyhow::Result<()> {
    let mut rng = StdRng::seed_from_u64(0x5c1 + (m * n * k) as u64);
    let scale = (k as f32).sqrt().recip();
    let a_host: Vec<bf16> = (0..m * k)
        .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal) * scale))
        .collect();
    // B is column-major, i.e. stored as row-major [n, k].
    let b_host: Vec<bf16> = (0..n * k)
        .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal)))
        .collect();

    let bf16_type = ElementType::Float(FloatType::Bf16);
    let mut a = Tensor::new_zeroed(Shape::new([m, k]), bf16_type, cx)?;
    let mut b_t = Tensor::new_zeroed(Shape::new([n, k]), bf16_type, cx)?;
    cx.upload_to_device(&a_host, a.storage_mut(), stream)?;
    cx.upload_to_device(&b_host, b_t.storage_mut(), stream)?;

    let c = executor.execute(&a.view(), &b_t.transposed(0, 1), (), stream)?;
    assert_eq!(c.shape(), Shape::new([m, n]));
    assert!(c.is_packed_row_major());

    let c_staging = cx.download_from_device(c.storage(), stream)?;
    stream.block_on()?;
    let c_host: &[bf16] = bytemuck::cast_slice(c_staging.view()?);

    let a_ref = Mat::from_fn(m, k, |i, j| a_host[i * k + j].to_f32());
    let b_ref = Mat::from_fn(k, n, |i, j| b_host[j * k + i].to_f32());
    let c_ref = a_ref * b_ref;

    let mut max_error = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let expected = c_ref[(i, j)];
            let actual = c_host[i * n + j].to_f32();
            assert!(actual.is_finite());

            max_error = max_error.max((actual - expected).abs());

            let margin = F32Margin {
                epsilon: ABS_TOLERANCE.max(REL_TOLERANCE * expected.abs()),
                ulps: 0,
            };
            anyhow::ensure!(
                actual.approx_eq(expected, margin),
                "{m}x{n}x{k}: mismatch at ({i}, {j}): expected {expected}, got {actual}"
            );
        }
    }

    println!("{m}x{n}x{k}: results match the reference (max absolute error: {max_error})");

    Ok(())
}
