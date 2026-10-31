// --- LLM-generated --- //
mod common;

use cleandl::{
    context::Context,
    executors::{
        Executor,
        norm_fwd::{Args, NormFwdExecutor, Output},
    },
    tensor::Tensor,
};
use common::{
    abs_rel_tolerance, assert_all_close, bf16_to_f32, dequantize_mxfp8, download,
    download_bf16_as_f32, random_bf16_tensor, random_bf16_vec, random_vec, seeded_rng,
    test_context,
};
use half::bf16;
use kernel_generator::kernels::norm::{Config, NormType, OutputTypes};
use rand_distr::{Distribution, Normal, Uniform};
use rayon::prelude::*;
use skytens_core::shape::Shape;

/// Input shapes to test; the norm is taken over the last dimension.
/// Row sizes must be multiples of 32 to support MXFP8 outputs.
const SHAPES: &[&[usize]] = &[
    &[1, 128],
    &[7, 3, 256],
    &[33, 1056],
    &[200, 2048],
    &[129, 4096],
    &[64, 8192],
    &[131072, 2048],
];

/// (has_scale, has_bias) combinations to test.
const AFFINE_VARIANTS: &[(bool, bool)] = &[(false, false), (true, false), (true, true)];

const EPS: f32 = 1e-6;

struct Reference {
    y: Vec<bf16>,
    r_inv: Vec<f32>,
}

fn reference_rms_norm(
    x: &[bf16],
    row_size: usize,
    scale: Option<&[bf16]>,
    bias: Option<&[bf16]>,
) -> Reference {
    let mut y = vec![bf16::ZERO; x.len()];
    let mut r_inv = vec![0.0f32; x.len() / row_size];
    x.par_chunks_exact(row_size)
        .zip(y.par_chunks_exact_mut(row_size))
        .zip(&mut r_inv)
        .for_each(|((row, y_row), row_r_inv)| {
            let sum_sq: f64 = row.iter().map(|v| (v.to_f32() as f64).powi(2)).sum();
            *row_r_inv = (1.0 / (sum_sq / row_size as f64 + EPS as f64).sqrt()) as f32;
            for (c, (v, out)) in row.iter().zip(y_row).enumerate() {
                let mut val = v.to_f32() * *row_r_inv;
                if let Some(scale) = scale {
                    val *= scale[c].to_f32();
                }
                if let Some(bias) = bias {
                    val += bias[c].to_f32();
                }
                *out = bf16::from_f32(val);
            }
        });
    Reference { y, r_inv }
}

struct Case {
    shape: Shape,
    has_scale: bool,
    has_bias: bool,
}

impl Case {
    fn row_size(&self) -> usize {
        self.shape[self.shape.num_dims() - 1]
    }

    fn num_rows(&self) -> usize {
        self.shape.element_count() / self.row_size()
    }

    fn label(&self) -> String {
        format!(
            "shape={:?} scale={} bias={}",
            self.shape.dims_slice(),
            self.has_scale,
            self.has_bias
        )
    }
}

fn for_each_case(mut f: impl FnMut(&Case) -> anyhow::Result<()>) {
    for shape in SHAPES {
        for &(has_scale, has_bias) in AFFINE_VARIANTS {
            let case = Case {
                shape: Shape::new(shape.iter().copied()),
                has_scale,
                has_bias,
            };
            f(&case).unwrap_or_else(|e| panic!("{}: {e:?}", case.label()));
        }
    }
}

/// Runs the executor on random inputs, returning its output alongside the CPU reference.
fn run_case(
    case: &Case,
    output_types: OutputTypes,
    cx: &Context,
) -> anyhow::Result<(Output, Reference)> {
    let mut rng = seeded_rng(case.shape.element_count() as u64);
    let row_size = case.row_size();

    // Vary row magnitudes so that per-row and per-block scaling is exercised.
    let row_magnitudes = random_vec(
        Uniform::new(-6.0f32, 6.0)?.map(f32::exp2),
        case.num_rows(),
        &mut rng,
    );
    let mut x_host = random_bf16_vec(Normal::new(0.0, 1.0)?, case.shape.element_count(), &mut rng);
    x_host
        .par_chunks_exact_mut(row_size)
        .zip(&row_magnitudes)
        .for_each(|(row, &magnitude)| {
            row.iter_mut()
                .for_each(|v| *v = bf16::from_f32(v.to_f32() * magnitude));
        });
    let x = Tensor::from_slice(&x_host, case.shape, cx.default_compute_stream(), cx)?;
    let scale = case
        .has_scale
        .then(|| random_bf16_tensor(Normal::new(1.0, 0.25)?, [row_size], &mut rng, cx))
        .transpose()?;
    let bias = case
        .has_bias
        .then(|| random_bf16_tensor(Normal::new(0.0, 0.5)?, [row_size], &mut rng, cx))
        .transpose()?;

    let executor = NormFwdExecutor::get(
        Config {
            norm_type: NormType::RmsNorm,
            has_scale: case.has_scale,
            has_bias: case.has_bias,
            output_types,
            max_row_size: row_size,
        },
        cx,
    )?;
    let output = executor.execute(
        Args {
            x: x.view(),
            scale: scale.as_ref().map(|(t, _)| t.view()),
            bias: bias.as_ref().map(|(t, _)| t.view()),
            write_r_inv: true,
        },
        cx.default_compute_stream(),
        cx,
    )?;

    let reference = reference_rms_norm(
        &x_host,
        row_size,
        scale.as_ref().map(|(_, h)| h.as_slice()),
        bias.as_ref().map(|(_, h)| h.as_slice()),
    );
    Ok((output, reference))
}

fn check_r_inv(
    case: &Case,
    output: &Output,
    reference: &Reference,
    cx: &Context,
) -> anyhow::Result<()> {
    let r_inv = output.r_inv.as_ref().expect("r_inv requested");
    assert_eq!(r_inv.shape(), case.shape.remove(case.shape.num_dims() - 1));
    assert_all_close(
        &format!("{} r_inv", case.label()),
        &download::<f32>(r_inv, cx)?,
        &reference.r_inv,
        abs_rel_tolerance(0.0, 1e-3),
    );
    Ok(())
}

#[test]
fn rms_norm_bf16_matches_reference() {
    let cx = test_context();
    for_each_case(|case| {
        let (output, reference) = run_case(case, OutputTypes::Bf16, &cx)?;
        let y = output.y_bf16.as_ref().expect("bf16 output");
        assert_eq!(y.shape(), case.shape);

        let expected = bf16_to_f32(&reference.y);
        // Allow one bf16 ulp of rounding difference.
        assert_all_close(
            &format!("{} y", case.label()),
            &download_bf16_as_f32(y, &cx)?,
            &expected,
            abs_rel_tolerance(1e-4, 2f32.powi(-7)),
        );
        check_r_inv(case, &output, &reference, &cx)
    });
}

#[test]
fn rms_norm_mxfp8_rowwise_matches_reference() {
    let cx = test_context();
    for_each_case(|case| {
        let (output, reference) = run_case(case, OutputTypes::Mxfp8Rowwise, &cx)?;
        assert!(output.y_bf16.is_none() && output.y_mxfp8_colwise.is_none());
        let y = output
            .y_mxfp8_rowwise
            .as_ref()
            .expect("mxfp8 rowwise output");
        assert_eq!(y.shape(), Shape::from([case.num_rows(), case.row_size()]));

        let dequantized = dequantize_mxfp8(y, &cx)?;
        let expected = bf16_to_f32(&reference.y);

        // E4M3 has 3 mantissa bits (relative rounding error <= 2^-4), plus one bf16 ulp of
        // difference in the unquantized value; subnormal steps are 2^-9 times the block scale.
        assert_all_close(
            &format!("{} y (dequantized)", case.label()),
            &dequantized.values,
            &expected,
            |i, e| {
                e.abs() * (2f32.powi(-4) + 2f32.powi(-7))
                    + dequantized.element_scales[i] * 2f32.powi(-9)
            },
        );

        // Each block's scale must avoid saturation (scale >= amax / 448) without
        // wasting range (scale < 2 * amax / 448). Allow slack for bf16 rounding.
        let block_size = 32;
        let bad_block = expected
            .par_chunks_exact(block_size)
            .enumerate()
            .find_map_first(|(block_idx, block)| {
                let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let scale = dequantized.element_scales[block_idx * block_size];
                let ideal = amax / 448.0;
                (amax != 0.0 && !(scale >= ideal * 0.98 && scale < ideal * 2.0 * 1.02)).then(|| {
                    format!(
                        "block {block_idx} has scale {scale}, but amax {amax} implies [{ideal}, {})",
                        ideal * 2.0
                    )
                })
            });
        if let Some(message) = bad_block {
            panic!("{}: {message}", case.label());
        }

        check_r_inv(case, &output, &reference, &cx)
    });
}
