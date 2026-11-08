mod common;

use anyhow::Context as _;
use cleandl::{
    context::Context,
    execgraph::{
        BuildOptions, ExecGraph, GraphBuilder, GraphOutput, Input, Operator, SymTensor,
        ops::norm::{NormFwd, NormFwdOpts, NormOutput, NormType},
        sym::Monomial,
        tune::Autotune,
    },
    mx_tensor::{MxDim, MxFormat},
    tensor::Tensor,
};
use common::{
    abs_rel_tolerance, assert_all_close, dequantize_mx, download, download_bf16_as_f32,
    random_bf16_tensor, random_tensor, seeded_rng, test_context,
};
use rand_distr::{Normal, Uniform};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

const F32: ElementType = ElementType::Float(FloatType::F32);
const BF16: ElementType = ElementType::Float(FloatType::Bf16);

#[derive(Clone, Copy, Debug)]
struct Case {
    rows: usize,
    cols: usize,
    norm_type: NormType,
    x_ty: ElementType,
    scale_ty: Option<ElementType>,
    bias_ty: Option<ElementType>,
    output: NormOutput,
    save_stats: bool,
    eps: f32,
    dynamic_rows: bool,
    dynamic_cols: bool,
    force_fallback: bool,
    /// Whether the specialized kernel must be used, if the device supports it.
    expect_specialized: bool,
}

impl Case {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            norm_type: NormType::RmsNorm,
            x_ty: BF16,
            scale_ty: None,
            bias_ty: None,
            output: NormOutput::Plain(BF16),
            save_stats: false,
            eps: 1e-6,
            dynamic_rows: false,
            dynamic_cols: false,
            force_fallback: false,
            expect_specialized: false,
        }
    }
}

fn supports_specialized(cx: &Context) -> bool {
    cx.cuda().compute_capability_major() >= 10
}

fn supports_fp4(cx: &Context) -> bool {
    cx.cuda().compute_capability_major() >= 10
}

/// Host values of a random tensor of type `ty`, rounded to it.
fn random_input(
    ty: ElementType,
    shape: &[usize],
    distr: impl rand_distr::Distribution<f32> + Send + Sync + Copy,
    seed: u64,
    cx: &Context,
) -> anyhow::Result<(Tensor, Vec<f32>)> {
    let mut rng = seeded_rng(seed);
    Ok(if ty == BF16 {
        let (t, h) = random_bf16_tensor(distr, Shape::new(shape.iter().copied()), &mut rng, cx)?;
        (t, h.iter().map(|v| v.to_f32()).collect())
    } else {
        random_tensor(distr, Shape::new(shape.iter().copied()), &mut rng, cx)?
    })
}

struct Reference {
    y: Vec<f32>,
    mean: Vec<f32>,
    r_inv: Vec<f32>,
}

fn reference(case: &Case, x: &[f32], scale: Option<&[f32]>, bias: Option<&[f32]>) -> Reference {
    let (rows, cols) = (case.rows, case.cols);
    let mut out = Reference {
        y: vec![0.0; rows * cols],
        mean: vec![0.0; rows],
        r_inv: vec![0.0; rows],
    };
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = match case.norm_type {
            NormType::RmsNorm => 0.0,
            NormType::LayerNorm => row.iter().map(|&v| f64::from(v)).sum::<f64>() / cols as f64,
        };
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / cols as f64;
        let r_inv = 1.0 / (var + f64::from(case.eps)).sqrt();
        out.mean[r] = mean as f32;
        out.r_inv[r] = r_inv as f32;
        for c in 0..cols {
            let mut y = (f64::from(row[c]) - mean) * r_inv;
            if let Some(s) = scale {
                y *= f64::from(s[c]);
            }
            if let Some(b) = bias {
                y += f64::from(b[c]);
            }
            out.y[r * cols + c] = y as f32;
        }
    }
    out
}

/// Checks a block-scaled output against unquantized reference values, allowing for
/// the quantization error of the format.
fn check_mx(name: &str, out: &GraphOutput, y: &[f32], case: &Case, dim: MxDim, cx: &Context) {
    let GraphOutput::Mx(t) = out else {
        panic!("{name}: expected a block-scaled output")
    };
    assert_eq!(t.mx_dim(), dim, "{name}");
    let got = dequantize_mx(t, cx).unwrap();
    assert_eq!((got.rows, got.cols), (case.rows, case.cols), "{name}");
    let (rel, abs) = match t.format() {
        // 3 mantissa bits, plus rounding the input to bf16, and the subnormal step.
        MxFormat::Mxfp8 => (0.07, 2f32.powi(-9)),
        MxFormat::Nvfp4 => (0.34, 0.5),
    };
    assert_all_close(name, &got.values, y, |i, e| {
        rel * e.abs() + abs * got.element_scales[i] + 1e-30
    });
}

fn run(case: Case, cx: &Context) -> anyhow::Result<()> {
    run_with(case, cx, Autotune::Off)
}

fn run_with(case: Case, cx: &Context, autotune: Autotune) -> anyhow::Result<()> {
    run_case(case, cx, autotune).with_context(|| format!("{case:?}"))
}

fn run_case(case: Case, cx: &Context, autotune: Autotune) -> anyhow::Result<()> {
    let name = format!("{case:?}");
    let (rows, cols) = (case.rows, case.cols);
    let gb = GraphBuilder::new();
    let row_dim: Monomial = if case.dynamic_rows {
        gb.dynamic_dim("rows", rows, 0).into()
    } else {
        rows.into()
    };
    let col_dim: Monomial = if case.dynamic_cols {
        gb.dynamic_dim("cols", cols, 0).into()
    } else {
        cols.into()
    };
    let xs = gb.input("x", case.x_ty, vec![row_dim, col_dim.clone()])?;
    let vector = |name: &str, ty: Option<ElementType>| -> anyhow::Result<Option<SymTensor<'_>>> {
        ty.map(|ty| gb.input(name, ty, vec![col_dim.clone()]))
            .transpose()
    };
    let scale_s = vector("scale", case.scale_ty)?;
    let bias_s = vector("bias", case.bias_ty)?;
    let outs = xs.norm_fwd(NormFwdOpts {
        norm_type: case.norm_type,
        scale: scale_s,
        bias: bias_s,
        eps: case.eps,
        output: case.output,
        save_stats: case.save_stats,
        force_fallback: case.force_fallback,
    })?;
    for t in [Some(outs.y), outs.y_colwise, outs.mean, outs.r_inv]
        .into_iter()
        .flatten()
    {
        gb.output(t)?;
    }

    let op = NormFwd {
        x: xs.id(),
        scale: scale_s.map(SymTensor::id),
        bias: bias_s.map(SymTensor::id),
        norm_type: case.norm_type,
        eps: case.eps,
        output: case.output,
        save_stats: case.save_stats,
        force_fallback: case.force_fallback,
    };
    let graph = gb.finish()?;
    let specialized = op.specialized_config(cx, graph.info()).is_some();
    if case.expect_specialized {
        assert_eq!(
            specialized,
            supports_specialized(cx),
            "{name}: specialized kernel applicability"
        );
    }
    if case.force_fallback {
        assert!(!specialized);
    }
    assert_eq!(op.name(), "norm_fwd");

    let (x, xh) = random_input(case.x_ty, &[rows, cols], Normal::new(0.5, 2.0)?, 1, cx)?;
    let scale = case
        .scale_ty
        .map(|ty| random_input(ty, &[cols], Uniform::new(0.5, 1.5)?, 2, cx))
        .transpose()?;
    let bias = case
        .bias_ty
        .map(|ty| random_input(ty, &[cols], Normal::new(0.0, 1.0)?, 3, cx))
        .transpose()?;
    let expected = reference(
        &case,
        &xh,
        scale.as_ref().map(|s| &s.1[..]),
        bias.as_ref().map(|b| &b.1[..]),
    );

    let exec = ExecGraph::build(
        graph,
        cx,
        BuildOptions {
            autotune,
            ..Default::default()
        },
    )?;
    let mut inputs: Vec<Input<'_>> = [
        Some(&x),
        scale.as_ref().map(|s| &s.0),
        bias.as_ref().map(|b| &b.0),
    ]
    .into_iter()
    .flatten()
    .map(Input::from)
    .collect();
    let outputs = exec.execute_default(&mut inputs, cx.default_compute_stream())?;
    cx.default_compute_stream().block_on()?;
    let mut outputs = outputs.into_iter();

    let y = outputs.next().unwrap();
    match case.output {
        NormOutput::Plain(ty) => {
            let y = y.into_plain();
            let (got, tol) = if ty == BF16 {
                (
                    download_bf16_as_f32(&y, cx)?,
                    abs_rel_tolerance(1e-5, 2f32.powi(-7)),
                )
            } else {
                (download::<f32>(&y, cx)?, abs_rel_tolerance(1e-5, 1e-4))
            };
            assert_all_close(&format!("{name} y"), &got, &expected.y, tol);
        }
        NormOutput::Mxfp8Rowwise | NormOutput::Nvfp4Rowwise => check_mx(
            &format!("{name} y"),
            &y,
            &expected.y,
            &case,
            MxDim::Rowwise,
            cx,
        ),
        NormOutput::Mxfp8RowwiseAndColwise => {
            check_mx(
                &format!("{name} y"),
                &y,
                &expected.y,
                &case,
                MxDim::Rowwise,
                cx,
            );
            let y_col = outputs.next().unwrap();
            check_mx(
                &format!("{name} y_col"),
                &y_col,
                &expected.y,
                &case,
                MxDim::Colwise,
                cx,
            );
        }
    }
    if case.save_stats {
        if case.norm_type == NormType::LayerNorm {
            let mean = download::<f32>(&outputs.next().unwrap().into_plain(), cx)?;
            assert_all_close(
                &format!("{name} mean"),
                &mean,
                &expected.mean,
                abs_rel_tolerance(1e-5, 1e-4),
            );
        }
        let r_inv = download::<f32>(&outputs.next().unwrap().into_plain(), cx)?;
        assert_all_close(
            &format!("{name} r_inv"),
            &r_inv,
            &expected.r_inv,
            abs_rel_tolerance(1e-6, 1e-4),
        );
    }
    assert!(outputs.next().is_none());
    Ok(())
}

/// Configurations the specialized kernel supports.
fn specialized_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (rows, cols) in [
        (1usize, 128),
        (37, 4096),
        (300, 2048),
        (128, 8192),
        (100, 7168),
        (5, 1024),
    ] {
        for output in [
            NormOutput::Plain(BF16),
            NormOutput::Mxfp8Rowwise,
            NormOutput::Mxfp8RowwiseAndColwise,
            NormOutput::Nvfp4Rowwise,
        ] {
            if output == NormOutput::Mxfp8RowwiseAndColwise && !rows.is_multiple_of(32) {
                continue;
            }
            for (scale, bias, save_stats) in [
                (None, None, false),
                (Some(BF16), None, true),
                (Some(BF16), Some(BF16), true),
            ] {
                cases.push(Case {
                    output,
                    scale_ty: scale,
                    bias_ty: bias,
                    save_stats,
                    // Too large for the shared memory of sm_120 (but not of sm_100).
                    expect_specialized: !(output == NormOutput::Mxfp8RowwiseAndColwise
                        && cols == 8192),
                    ..Case::new(rows, cols)
                });
            }
        }
    }
    // Colwise outputs need the row count to be provably a multiple of 32.
    cases.extend([
        Case {
            output: NormOutput::Mxfp8RowwiseAndColwise,
            ..Case::new(96, 512)
        },
        Case {
            dynamic_rows: true,
            eps: 1e-2,
            scale_ty: Some(BF16),
            save_stats: true,
            expect_specialized: true,
            ..Case::new(333, 1536)
        },
        Case {
            output: NormOutput::Plain(BF16),
            expect_specialized: true,
            ..Case::new(64, 136)
        },
    ]);
    cases
}

#[test]
fn specialized_configurations() -> anyhow::Result<()> {
    let cx = test_context();
    for case in specialized_cases() {
        if case.output == NormOutput::Nvfp4Rowwise && !supports_fp4(&cx) {
            continue;
        }
        run(case, &cx)?;
    }
    Ok(())
}

#[test]
fn fallback_matches_on_specialized_configurations() -> anyhow::Result<()> {
    let cx = test_context();
    for case in specialized_cases() {
        if case.output == NormOutput::Nvfp4Rowwise && !supports_fp4(&cx) {
            continue;
        }
        run(
            Case {
                force_fallback: true,
                expect_specialized: false,
                ..case
            },
            &cx,
        )?;
    }
    Ok(())
}

#[test]
fn fallback_configurations() -> anyhow::Result<()> {
    let cx = test_context();
    let mut cases = Vec::new();
    for norm_type in [NormType::RmsNorm, NormType::LayerNorm] {
        for (x_ty, out_ty) in [(F32, F32), (F32, BF16), (BF16, F32), (BF16, BF16)] {
            for (scale, bias) in [
                (None, None),
                (Some(F32), Some(BF16)),
                (Some(BF16), Some(F32)),
            ] {
                cases.push(Case {
                    norm_type,
                    x_ty,
                    output: NormOutput::Plain(out_ty),
                    scale_ty: scale,
                    bias_ty: bias,
                    save_stats: true,
                    ..Case::new(77, 1000)
                });
            }
        }
        // Long rows, streamed in chunks.
        cases.push(Case {
            norm_type,
            x_ty: F32,
            output: NormOutput::Plain(F32),
            scale_ty: Some(F32),
            save_stats: true,
            ..Case::new(9, 20000)
        });
        cases.push(Case {
            norm_type,
            output: NormOutput::Mxfp8RowwiseAndColwise,
            bias_ty: Some(BF16),
            ..Case::new(64, 12288)
        });
        // Dynamic shapes.
        cases.push(Case {
            norm_type,
            dynamic_rows: true,
            dynamic_cols: true,
            scale_ty: Some(BF16),
            save_stats: true,
            eps: 1e-3,
            ..Case::new(50, 3000)
        });
        for output in [
            NormOutput::Mxfp8Rowwise,
            NormOutput::Mxfp8RowwiseAndColwise,
            NormOutput::Nvfp4Rowwise,
        ] {
            if output == NormOutput::Nvfp4Rowwise && !supports_fp4(&cx) {
                continue;
            }
            cases.push(Case {
                norm_type,
                x_ty: F32,
                output,
                scale_ty: Some(F32),
                save_stats: true,
                ..Case::new(160, 96)
            });
        }
    }
    // Not handled by the specialized kernel.
    cases.push(Case {
        output: NormOutput::Plain(BF16),
        ..Case::new(31, 8200)
    });
    cases.push(Case {
        output: NormOutput::Mxfp8Rowwise,
        ..Case::new(40, 16384)
    });
    for case in cases {
        run(case, &cx)?;
    }
    Ok(())
}

#[test]
fn eligibility() -> anyhow::Result<()> {
    let cx = test_context();
    let check = |expect: bool, f: &dyn Fn(&GraphBuilder) -> anyhow::Result<NormFwd>| {
        let gb = GraphBuilder::new();
        let op = f(&gb).unwrap();
        let graph = gb.finish().unwrap();
        assert_eq!(
            op.specialized_config(&cx, graph.info()).is_some(),
            expect && supports_specialized(&cx),
            "{op:?}"
        );
    };
    let op = |x: SymTensor<'_>, opts: NormFwdOpts<'_>| NormFwd {
        x: x.id(),
        scale: opts.scale.map(SymTensor::id),
        bias: opts.bias.map(SymTensor::id),
        norm_type: opts.norm_type,
        eps: opts.eps,
        output: opts.output,
        save_stats: opts.save_stats,
        force_fallback: opts.force_fallback,
    };
    check(true, &|gb| {
        let x = gb.input("x", BF16, [64, 4096])?;
        Ok(op(x, NormFwdOpts::default()))
    });
    check(false, &|gb| {
        let x = gb.input("x", F32, [64, 4096])?;
        Ok(op(x, NormFwdOpts::default()))
    });
    check(false, &|gb| {
        let x = gb.input("x", BF16, [64, 4096])?;
        Ok(op(
            x,
            NormFwdOpts {
                output: NormOutput::Plain(F32),
                ..Default::default()
            },
        ))
    });
    check(false, &|gb| {
        let x = gb.input("x", BF16, [64, 4096])?;
        Ok(op(
            x,
            NormFwdOpts {
                norm_type: NormType::LayerNorm,
                ..Default::default()
            },
        ))
    });
    check(false, &|gb| {
        let x = gb.input("x", BF16, [64, 8192 + 64])?;
        Ok(op(x, NormFwdOpts::default()))
    });
    check(false, &|gb| {
        let x = gb.input("x", BF16, [64, 4100])?;
        Ok(op(x, NormFwdOpts::default()))
    });
    check(false, &|gb| {
        let cols = gb.dynamic_dim("cols", 4096, 8);
        let x = gb.input("x", BF16, vec![Monomial::fixed(64), cols.into()])?;
        Ok(op(x, NormFwdOpts::default()))
    });
    check(false, &|gb| {
        let x = gb.input("x", BF16, [64, 8192])?.sliced(1, 0, 4096)?;
        Ok(op(x, NormFwdOpts::default()))
    });
    check(false, &|gb| {
        let x = gb.input("x", BF16, [64, 4096])?;
        let scale = gb.input("scale", F32, [4096])?;
        Ok(op(
            x,
            NormFwdOpts {
                scale: Some(scale),
                ..Default::default()
            },
        ))
    });
    check(false, &|gb| {
        let x = gb.input("x", BF16, [64, 4096])?;
        Ok(op(
            x,
            NormFwdOpts {
                force_fallback: true,
                ..Default::default()
            },
        ))
    });
    Ok(())
}

#[test]
fn strided_input() -> anyhow::Result<()> {
    let cx = test_context();
    let (rows, cols) = (48, 384);
    let gb = GraphBuilder::new();
    let xs = gb
        .input("x", BF16, [rows, 2 * cols])?
        .sliced(1, cols, cols)?;
    let outs = xs.norm_fwd(NormFwdOpts {
        norm_type: NormType::LayerNorm,
        save_stats: true,
        ..Default::default()
    })?;
    gb.output(outs.y)?;
    let exec = ExecGraph::build(gb.finish()?, &cx, BuildOptions::default())?;

    let mut rng = seeded_rng(7);
    let (x, xh) = random_bf16_tensor(Normal::new(0.0, 1.0)?, [rows, 2 * cols], &mut rng, &cx)?;
    let xh: Vec<f32> = xh
        .chunks(2 * cols)
        .flat_map(|row| row[cols..].iter().map(|v| v.to_f32()))
        .collect();
    let case = Case {
        norm_type: NormType::LayerNorm,
        ..Case::new(rows, cols)
    };
    let expected = reference(&case, &xh, None, None);
    let out = exec.execute_default(&mut [Input::from(&x)], cx.default_compute_stream())?;
    let got = download_bf16_as_f32(&out.into_iter().next().unwrap().into_plain(), &cx)?;
    assert_all_close(
        "strided",
        &got,
        &expected.y,
        abs_rel_tolerance(1e-5, 2f32.powi(-7)),
    );
    Ok(())
}

#[test]
fn autotuned_fallback() -> anyhow::Result<()> {
    let cx = test_context();
    for _ in 0..2 {
        run_with(
            Case {
                norm_type: NormType::LayerNorm,
                output: NormOutput::Mxfp8RowwiseAndColwise,
                scale_ty: Some(BF16),
                save_stats: true,
                ..Case::new(256, 3072)
            },
            &cx,
            Autotune::Full,
        )?;
    }
    Ok(())
}

#[test]
fn validation_errors() -> anyhow::Result<()> {
    let err = |f: &dyn Fn(&GraphBuilder) -> anyhow::Result<()>| {
        let gb = GraphBuilder::new();
        format!("{:#}", f(&gb).unwrap_err())
    };
    let e = err(&|gb| {
        let x = gb.input("x", ElementType::Float(FloatType::F16), [4, 64])?;
        x.norm_fwd(Default::default()).map(|_| ())
    });
    assert!(e.contains("bf16 or f32"), "{e}");
    let e = err(&|gb| {
        let x = gb.input("x", BF16, [2, 4, 64])?;
        x.norm_fwd(Default::default()).map(|_| ())
    });
    assert!(e.contains("2D"), "{e}");
    let e = err(&|gb| {
        let x = gb.input("x", BF16, [4, 64])?;
        let scale = gb.input("scale", BF16, [32])?;
        x.norm_fwd(NormFwdOpts {
            scale: Some(scale),
            ..Default::default()
        })
        .map(|_| ())
    });
    assert!(e.contains("scale must have shape"), "{e}");
    let e = err(&|gb| {
        let x = gb.input("x", BF16, [4, 64])?;
        x.norm_fwd(NormFwdOpts {
            output: NormOutput::Plain(ElementType::Float(FloatType::F8E4M3)),
            ..Default::default()
        })
        .map(|_| ())
    });
    assert!(e.contains("plain output"), "{e}");
    let e = err(&|gb| {
        let x = gb.input("x", BF16, [4, 48])?;
        x.norm_fwd(NormFwdOpts {
            output: NormOutput::Mxfp8Rowwise,
            ..Default::default()
        })
        .map(|_| ())
    });
    assert!(e.contains("block size"), "{e}");
    let e = err(&|gb| {
        let x = gb.input("x", BF16, [48, 64])?;
        x.norm_fwd(NormFwdOpts {
            output: NormOutput::Mxfp8RowwiseAndColwise,
            ..Default::default()
        })
        .map(|_| ())
    });
    assert!(e.contains("block size"), "{e}");
    let e = err(&|gb| {
        let x = gb.input("x", BF16, [4, 64])?;
        x.norm_fwd(NormFwdOpts {
            eps: f32::NAN,
            ..Default::default()
        })
        .map(|_| ())
    });
    assert!(e.contains("eps"), "{e}");
    Ok(())
}
