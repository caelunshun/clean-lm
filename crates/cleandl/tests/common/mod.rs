// --- LLM-generated --- //
#![allow(dead_code)]

use bytemuck::Pod;
use cleandl::{
    context::Context,
    mx_tensor::{MxDim, MxFormat, MxTensor},
    par_random::par_fill_from_distr,
    tensor::{Tensor, TensorElement},
};
use float8::F8E4M3;
use half::bf16;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};
use skytens_core::shape::Shape;

pub fn test_context() -> Context {
    Context::new(
        0,
        DeviceMemoryPoolOptions::default(),
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        false,
    )
    .expect("failed to create CUDA context")
}

pub fn seeded_rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

pub fn random_vec<T: Pod + Send + Sync>(
    distr: impl Distribution<T> + Send + Sync,
    len: usize,
    rng: &mut impl Rng,
) -> Vec<T> {
    let mut data = vec![T::zeroed(); len];
    par_fill_from_distr(distr, &mut data, rng);
    data
}

pub fn random_bf16_vec(
    distr: impl Distribution<f32> + Send + Sync,
    len: usize,
    rng: &mut impl Rng,
) -> Vec<bf16> {
    random_vec(distr.map(bf16::from_f32), len, rng)
}

/// Samples a host vector from `distr` and uploads it as a packed row-major tensor.
pub fn random_tensor<T: TensorElement>(
    distr: impl Distribution<T> + Send + Sync,
    shape: impl Into<Shape>,
    rng: &mut impl Rng,
    cx: &Context,
) -> anyhow::Result<(Tensor, Vec<T>)> {
    let shape = shape.into();
    let host = random_vec(distr, shape.element_count(), rng);
    let tensor = Tensor::from_slice(&host, shape, cx.default_compute_stream(), cx)?;
    Ok((tensor, host))
}

pub fn random_bf16_tensor(
    distr: impl Distribution<f32> + Send + Sync,
    shape: impl Into<Shape>,
    rng: &mut impl Rng,
    cx: &Context,
) -> anyhow::Result<(Tensor, Vec<bf16>)> {
    random_tensor(distr.map(bf16::from_f32), shape, rng, cx)
}

/// Downloads the raw bytes addressed by the tensor's view, starting at its base pointer.
pub fn download_bytes(tensor: &Tensor, cx: &Context) -> anyhow::Result<Vec<u8>> {
    let host = cx.download_from_device(tensor.storage(), cx.default_compute_stream())?;
    Ok(host.view()?[..tensor.virt().required_bytes()].to_vec())
}

pub fn download<T: Pod + Send>(tensor: &Tensor, cx: &Context) -> anyhow::Result<Vec<T>> {
    anyhow::ensure!(
        tensor.is_packed_row_major(),
        "download requires a packed row-major tensor"
    );
    Ok(download_bytes(tensor, cx)?
        .par_chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect())
}

pub fn download_bf16_as_f32(tensor: &Tensor, cx: &Context) -> anyhow::Result<Vec<f32>> {
    Ok(bf16_to_f32(&download::<bf16>(tensor, cx)?))
}

pub fn bf16_to_f32(values: &[bf16]) -> Vec<f32> {
    values.par_iter().map(|v| v.to_f32()).collect()
}

/// Index of the scale factor for block `(row, block_col)` within the 128x4
/// tiled, partition-swizzled scale factor layout.
pub fn mx_scale_factor_index(row: usize, block_col: usize, num_block_cols: usize) -> usize {
    ((row / 128) * num_block_cols.div_ceil(4) + block_col / 4) * 512
        + (row % 32) * 16
        + ((row % 128) / 32) * 4
        + block_col % 4
}

pub fn decode_ue8m0(bits: u8) -> f32 {
    if bits == 0xFF {
        f32::NAN
    } else {
        (bits as f32 - 127.0).exp2()
    }
}

/// Host copy of a dequantized 2D MXFP8 tensor.
pub struct DequantizedMx {
    pub rows: usize,
    pub cols: usize,
    /// Dequantized values, row-major `[rows, cols]`.
    pub values: Vec<f32>,
    /// Scale factor applied to each element, row-major `[rows, cols]`.
    pub element_scales: Vec<f32>,
}

pub fn dequantize_mxfp8(tensor: &MxTensor, cx: &Context) -> anyhow::Result<DequantizedMx> {
    anyhow::ensure!(tensor.format() == MxFormat::Mxfp8);
    let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
    let block_size = tensor.format().block_size();

    let values = download_bytes(tensor.values(), cx)?;
    let scale_factors = download_bytes(tensor.scale_factors(), cx)?;

    let mut out = DequantizedMx {
        rows,
        cols,
        values: vec![0.0; rows * cols],
        element_scales: vec![0.0; rows * cols],
    };
    out.values
        .par_chunks_exact_mut(cols)
        .zip(out.element_scales.par_chunks_exact_mut(cols))
        .enumerate()
        .for_each(|(r, (values_row, scales_row))| {
            for c in 0..cols {
                // Colwise tensors are stored transposed, with blocks running down columns.
                let (value_idx, sf_idx) = match tensor.mx_dim() {
                    MxDim::Rowwise => (
                        r * cols + c,
                        mx_scale_factor_index(r, c / block_size, cols.div_ceil(block_size)),
                    ),
                    MxDim::Colwise => (
                        c * rows + r,
                        mx_scale_factor_index(c, r / block_size, rows.div_ceil(block_size)),
                    ),
                };
                let scale = decode_ue8m0(scale_factors[sf_idx]);
                values_row[c] = F8E4M3::from_bits(values[value_idx]).to_f32() * scale;
                scales_row[c] = scale;
            }
        });
    Ok(out)
}

/// Number of worst out-of-tolerance elements reported by [`assert_all_close`].
const NUM_REPORTED_FAILURES: usize = 8;

/// Failure statistics, tracking only the worst few failures to bound memory.
#[derive(Default)]
struct FailureSummary {
    count: usize,
    max_abs_err: f32,
    /// `(index, actual, expected, abs_err)`
    worst: Vec<(usize, f32, f32, f32)>,
}

impl FailureSummary {
    fn truncate_worst(&mut self) {
        self.worst.sort_by(|x, y| y.3.total_cmp(&x.3));
        self.worst.truncate(NUM_REPORTED_FAILURES);
    }

    fn merge(mut self, mut other: Self) -> Self {
        self.count += other.count;
        self.max_abs_err = self.max_abs_err.max(other.max_abs_err);
        self.worst.append(&mut other.worst);
        self.truncate_worst();
        self
    }
}

/// Asserts `|actual[i] - expected[i]| <= tolerance(i, expected[i])` for all `i`,
/// reporting the worst offenders on failure. NaNs always fail.
#[track_caller]
pub fn assert_all_close(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: impl Fn(usize, f32) -> f32 + Sync,
) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");

    let summary = actual
        .par_iter()
        .zip(expected)
        .enumerate()
        .fold(FailureSummary::default, |mut summary, (i, (&a, &e))| {
            let err = (a - e).abs();
            summary.max_abs_err = summary.max_abs_err.max(err);
            if err.is_nan() || err > tolerance(i, e) {
                summary.count += 1;
                summary.worst.push((i, a, e, err));
                if summary.worst.len() > 2 * NUM_REPORTED_FAILURES {
                    summary.truncate_worst();
                }
            }
            summary
        })
        .reduce(FailureSummary::default, FailureSummary::merge);

    if summary.count > 0 {
        let worst = summary
            .worst
            .iter()
            .map(|(i, a, e, err)| format!("  [{i}] actual={a} expected={e} err={err}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "{name}: {}/{} elements out of tolerance (max abs err {}). Worst:\n{worst}",
            summary.count,
            actual.len(),
            summary.max_abs_err
        );
    }
}

pub fn abs_rel_tolerance(atol: f32, rtol: f32) -> impl Fn(usize, f32) -> f32 + Sync {
    move |_, expected| atol + rtol * expected.abs()
}
