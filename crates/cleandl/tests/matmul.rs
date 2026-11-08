mod common;

use cleandl::{
    context::Context,
    execgraph::{
        BuildOptions, ExecGraph, GraphBuilder, Input, SymTensor,
        ops::matmul::{MatmulOpts, MatmulOutput},
        sym::{Monomial, Polynomial, SymLayout},
        tune::Autotune,
    },
    mx_tensor::{MxDim, MxFormat, MxTensor},
    tensor::Tensor,
};
use common::{
    assert_all_close, bf16_to_f32, dequantize_mx, download, download_bf16_as_f32,
    mx_scale_factor_index, random_bf16_tensor, random_tensor, seeded_rng, test_context,
};
use float8::F8E4M3;
use parking_lot::Mutex;
use rand_distr::Uniform;
use rayon::prelude::*;
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};
use std::sync::OnceLock;

const BF16: ElementType = ElementType::Float(FloatType::Bf16);
const F32: ElementType = ElementType::Float(FloatType::F32);

/// Row-major host matrices `[batch, rows, cols]`.
#[derive(Clone)]
struct HostMat {
    batch: usize,
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl HostMat {
    fn at(&self, b: usize, r: usize, c: usize) -> f32 {
        self.data[(b * self.rows + r) * self.cols + c]
    }

    fn transposed(&self) -> HostMat {
        let mut data = Vec::with_capacity(self.data.len());
        for b in 0..self.batch {
            for c in 0..self.cols {
                for r in 0..self.rows {
                    data.push(self.at(b, r, c));
                }
            }
        }
        HostMat {
            batch: self.batch,
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    /// Columns `start..start + len` of a 2D matrix.
    fn cols_slice(&self, start: usize, len: usize) -> HostMat {
        assert_eq!(self.batch, 1);
        let data = (0..self.rows)
            .flat_map(|r| (start..start + len).map(move |c| (r, c)))
            .map(|(r, c)| self.at(0, r, c))
            .collect();
        HostMat {
            batch: 1,
            rows: self.rows,
            cols: len,
            data,
        }
    }

    fn rows_slice(&self, start: usize, len: usize) -> HostMat {
        assert_eq!(self.batch, 1);
        HostMat {
            batch: 1,
            rows: len,
            cols: self.cols,
            data: self.data[start * self.cols..(start + len) * self.cols].to_vec(),
        }
    }
}

/// `alpha * A·B + beta * C` in f64, plus `|alpha| * |A|·|B| + |beta * C|` for error bounds.
fn reference(
    a: &HostMat,
    b: &HostMat,
    c: Option<&HostMat>,
    alpha: f32,
    beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(a.batch, b.batch);
    assert_eq!(a.cols, b.rows);
    let (m, k, n) = (a.rows, a.cols, b.cols);
    (0..a.batch * m * n)
        .into_par_iter()
        .map(|i| {
            let (bi, r, col) = (i / (m * n), (i / n) % m, i % n);
            let (mut sum, mut abs) = (0f64, 0f64);
            for kk in 0..k {
                let p = a.at(bi, r, kk) as f64 * b.at(bi, kk, col) as f64;
                sum += p;
                abs += p.abs();
            }
            let (mut value, mut bound) = (alpha as f64 * sum, alpha.abs() as f64 * abs);
            if let Some(c) = c {
                let cv = beta as f64 * c.at(bi, r, col) as f64;
                value += cv;
                bound += cv.abs();
            }
            (value as f32, bound as f32)
        })
        .unzip()
}

fn random_bf16(
    batch: usize,
    rows: usize,
    cols: usize,
    seed: u64,
    cx: &Context,
) -> anyhow::Result<(Tensor, HostMat)> {
    let shape = if batch == 1 {
        Shape::new([rows, cols])
    } else {
        Shape::new([batch, rows, cols])
    };
    let (tensor, host) = random_bf16_tensor(
        Uniform::new(-1.0f32, 1.0)?,
        shape,
        &mut seeded_rng(seed),
        cx,
    )?;
    let data = bf16_to_f32(&host);
    Ok((
        tensor,
        HostMat {
            batch,
            rows,
            cols,
            data,
        },
    ))
}

fn random_f32(
    rows: usize,
    cols: usize,
    seed: u64,
    cx: &Context,
) -> anyhow::Result<(Tensor, HostMat)> {
    let (tensor, data) = random_tensor(
        Uniform::new(-1.0f32, 1.0)?,
        [rows, cols],
        &mut seeded_rng(seed),
        cx,
    )?;
    Ok((
        tensor,
        HostMat {
            batch: 1,
            rows,
            cols,
            data,
        },
    ))
}

fn build(gb: GraphBuilder, cx: &Context, autotune: Autotune) -> anyhow::Result<ExecGraph> {
    ExecGraph::build(
        gb.finish()?,
        cx,
        BuildOptions {
            autotune,
            ..Default::default()
        },
    )
}

fn run(graph: &ExecGraph, inputs: &[&Tensor], cx: &Context) -> anyhow::Result<Tensor> {
    let mut inputs: Vec<Input<'_>> = inputs.iter().map(|&t| t.into()).collect();
    let mut outputs = graph.execute_default(&mut inputs, cx.default_compute_stream())?;
    Ok(outputs.remove(0).into_plain())
}

fn download_f32(tensor: &Tensor, cx: &Context) -> anyhow::Result<Vec<f32>> {
    match tensor.element_type() {
        BF16 => download_bf16_as_f32(tensor, cx),
        F32 => download::<f32>(tensor, cx),
        ty => panic!("unexpected output type {ty:?}"),
    }
}

/// Bound for an output of type `out` computed from bf16 (or exactly representable) inputs.
fn tolerance(out: ElementType, bounds: &[f32]) -> impl Fn(usize, f32) -> f32 + Sync + '_ {
    let rounding = if out == BF16 { 2f32.powi(-8) } else { 0.0 };
    move |i, expected| rounding * expected.abs() + 1e-5 * bounds[i] + 1e-6
}

fn matmul_opts<'g>(out: ElementType) -> MatmulOpts<'g> {
    MatmulOpts {
        out: MatmulOutput::Plain(out),
        ..Default::default()
    }
}

#[test]
fn bf16_matmul() -> anyhow::Result<()> {
    let cx = test_context();
    for (i, &(m, k, n)) in [(64, 64, 64), (129, 264, 72), (1, 8, 8), (300, 520, 136)]
        .iter()
        .enumerate()
    {
        let (a, ha) = random_bf16(1, m, k, 2 * i as u64, &cx)?;
        let (b, hb) = random_bf16(1, k, n, 2 * i as u64 + 1, &cx)?;
        let (expected, bounds) = reference(&ha, &hb, None, 1.0, 0.0);
        for out in [BF16, F32] {
            let gb = GraphBuilder::new();
            let ta = gb.input("a", BF16, [m, k])?;
            let tb = gb.input("b", BF16, [k, n])?;
            let d = ta.matmul(tb, matmul_opts(out))?;
            assert_eq!(d.element_type(), Some(out));
            gb.output(d)?;
            let graph = build(gb, &cx, Autotune::Off)?;
            let d = run(&graph, &[&a, &b], &cx)?;
            assert_eq!(d.shape().dims_slice(), [m, n]);
            assert_all_close(
                &format!("bf16 {m}x{k}x{n} -> {out:?}"),
                &download_f32(&d, &cx)?,
                &expected,
                tolerance(out, &bounds),
            );
        }
    }
    Ok(())
}

#[test]
fn f32_tf32_matmul() -> anyhow::Result<()> {
    let cx = test_context();
    let (m, k, n) = (96, 200, 136);
    let (a, ha) = random_f32(m, k, 10, &cx)?;
    let (b, hb) = random_f32(k, n, 11, &cx)?;
    let (expected, bounds) = reference(&ha, &hb, None, 1.0, 0.0);
    let gb = GraphBuilder::new();
    let d = gb.input("a", F32, [m, k])?.matmul(
        gb.input("b", F32, [k, n])?,
        MatmulOpts {
            out: MatmulOutput::Plain(F32),
            ..Default::default()
        },
    )?;
    gb.output(d)?;
    let graph = build(gb, &cx, Autotune::Off)?;
    let d = run(&graph, &[&a, &b], &cx)?;
    // TF32 keeps 10 mantissa bits of each input.
    assert_all_close("f32 tf32", &download_f32(&d, &cx)?, &expected, |i, _| {
        2e-3 * bounds[i] + 1e-5
    });
    Ok(())
}

#[test]
fn transposed_operands() -> anyhow::Result<()> {
    let cx = test_context();
    let (m, k, n) = (72, 136, 88);
    let (a_t, ha_t) = random_bf16(1, k, m, 20, &cx)?;
    let (b_t, hb_t) = random_bf16(1, n, k, 21, &cx)?;
    let (a, ha) = random_bf16(1, m, k, 22, &cx)?;
    let (b, hb) = random_bf16(1, k, n, 23, &cx)?;
    for (transpose_a, transpose_b) in [(true, false), (false, true), (true, true)] {
        let gb = GraphBuilder::new();
        let ta = if transpose_a {
            gb.input("a", BF16, [k, m])?.transposed(0, 1)?
        } else {
            gb.input("a", BF16, [m, k])?
        };
        let tb = if transpose_b {
            gb.input("b", BF16, [n, k])?.transposed(0, 1)?
        } else {
            gb.input("b", BF16, [k, n])?
        };
        gb.output(ta.matmul(tb, matmul_opts(F32))?)?;
        let graph = build(gb, &cx, Autotune::Off)?;

        let (a_in, a_host) = if transpose_a {
            (&a_t, ha_t.transposed())
        } else {
            (&a, ha.clone())
        };
        let (b_in, b_host) = if transpose_b {
            (&b_t, hb_t.transposed())
        } else {
            (&b, hb.clone())
        };
        let (expected, bounds) = reference(&a_host, &b_host, None, 1.0, 0.0);
        let d = run(&graph, &[a_in, b_in], &cx)?;
        assert_all_close(
            &format!("transposed a={transpose_a} b={transpose_b}"),
            &download_f32(&d, &cx)?,
            &expected,
            tolerance(F32, &bounds),
        );
    }
    Ok(())
}

#[test]
fn sliced_operands() -> anyhow::Result<()> {
    let cx = test_context();
    let (m, k, n) = (40, 64, 48);
    // A is columns 8..8+k of a wider matrix, B rows 3..3+k of a taller one.
    let (a_full, ha_full) = random_bf16(1, m, k + 16, 30, &cx)?;
    let (b_full, hb_full) = random_bf16(1, k + 5, n, 31, &cx)?;
    let gb = GraphBuilder::new();
    let ta = gb.input("a", BF16, [m, k + 16])?.sliced(1, 8, k)?;
    let tb = gb.input("b", BF16, [k + 5, n])?.sliced(0, 3, k)?;
    gb.output(ta.matmul(tb, matmul_opts(BF16))?)?;
    let graph = build(gb, &cx, Autotune::Off)?;
    let (expected, bounds) = reference(
        &ha_full.cols_slice(8, k),
        &hb_full.rows_slice(3, k),
        None,
        1.0,
        0.0,
    );
    let d = run(&graph, &[&a_full, &b_full], &cx)?;
    assert_all_close(
        "sliced",
        &download_f32(&d, &cx)?,
        &expected,
        tolerance(BF16, &bounds),
    );
    Ok(())
}

fn expect_infer_error(
    f: impl FnOnce(&GraphBuilder) -> anyhow::Result<SymTensor<'_>>,
    needle: &str,
) {
    let gb = GraphBuilder::new();
    let err = f(&gb).expect_err("expected matmul to be rejected");
    let message = format!("{err:#}");
    assert!(
        message.contains(needle),
        "error {message:?} does not mention {needle:?}"
    );
}

#[test]
fn invalid_operands() {
    expect_infer_error(
        |gb| {
            gb.input("a", BF16, [32, 64])?
                .matmul(gb.input("b", BF16, [48, 32])?, MatmulOpts::default())
        },
        "contraction",
    );
    expect_infer_error(
        |gb| {
            gb.input("a", BF16, [2, 32, 64])?
                .matmul(gb.input("b", BF16, [3, 64, 32])?, MatmulOpts::default())
        },
        "batch",
    );
    expect_infer_error(
        |gb| {
            gb.input("a", BF16, [32, 64])?
                .matmul(gb.input("b", BF16, [2, 64, 32])?, MatmulOpts::default())
        },
        "rank",
    );
    expect_infer_error(
        |gb| {
            gb.input("a", ElementType::Float(FloatType::F16), [32, 64])?
                .matmul(gb.input("b", BF16, [64, 32])?, MatmulOpts::default())
        },
        "f32 or bf16",
    );
    // Neither inner dimension contiguous.
    expect_infer_error(
        |gb| {
            let layout = SymLayout::new(
                [32usize, 64].into_iter().map(Into::into).collect(),
                vec![128usize.into(), 2usize.into()],
                Polynomial::zero(),
            );
            gb.input_with_layout("a", BF16, layout)?
                .matmul(gb.input("b", BF16, [64, 32])?, MatmulOpts::default())
        },
        "contiguous",
    );
    expect_infer_error(
        |gb| {
            gb.input_mx("a", MxFormat::Mxfp8, MxDim::Rowwise, [128, 64])?
                .matmul(gb.input("b", BF16, [64, 32])?, MatmulOpts::default())
        },
        "mix",
    );
    expect_infer_error(
        |gb| {
            gb.input_mx("a", MxFormat::Mxfp8, MxDim::Colwise, [128, 64])?
                .matmul(
                    gb.input_mx("b", MxFormat::Mxfp8, MxDim::Colwise, [64, 128])?,
                    MatmulOpts::default(),
                )
        },
        "Rowwise",
    );
    expect_infer_error(
        |gb| {
            gb.input_mx("a", MxFormat::Mxfp8, MxDim::Rowwise, [128, 64])?
                .matmul(
                    gb.input_mx("b", MxFormat::Mxfp8, MxDim::Rowwise, [64, 128])?,
                    MatmulOpts::default(),
                )
        },
        "Colwise",
    );
    expect_infer_error(
        |gb| {
            gb.input("a", BF16, [32, 64])?.matmul(
                gb.input("b", BF16, [64, 32])?,
                MatmulOpts {
                    beta: 1.0,
                    ..Default::default()
                },
            )
        },
        "no C",
    );
    expect_infer_error(
        |gb| {
            gb.input("a", BF16, [32, 0])?
                .matmul(gb.input("b", BF16, [0, 32])?, MatmulOpts::default())
        },
        "must not be zero",
    );
    expect_infer_error(
        |gb| {
            gb.input("a", BF16, [32, 64])?.matmul(
                gb.input("b", BF16, [64, 32])?,
                matmul_opts(ElementType::Float(FloatType::F16)),
            )
        },
        "output must be f32 or bf16",
    );
    // Zero stride along M.
    expect_infer_error(
        |gb| {
            gb.input_with_layout("a", BF16, strided([32, 64], [0, 1]))?
                .matmul(gb.input("b", BF16, [64, 32])?, MatmulOpts::default())
        },
        "zero stride in dimension 0",
    );
    // Rows of 64 elements, 32 apart.
    expect_infer_error(
        |gb| {
            gb.input_with_layout("a", BF16, strided([32, 64], [32, 1]))?
                .matmul(gb.input("b", BF16, [64, 32])?, MatmulOpts::default())
        },
        "smaller than the 64 contiguous elements",
    );
}

fn strided<const N: usize>(shape: [usize; N], strides: [usize; N]) -> SymLayout {
    SymLayout::new(
        shape.into_iter().map(Into::into).collect(),
        strides.into_iter().map(Into::into).collect(),
        Polynomial::zero(),
    )
}

fn with_c<'g>(c: SymTensor<'g>) -> MatmulOpts<'g> {
    MatmulOpts {
        c: Some(c),
        beta: 1.0,
        ..Default::default()
    }
}

#[test]
fn invalid_c() {
    expect_infer_error(
        |gb| {
            let c = gb.input("c", F32, [32, 32])?;
            gb.input("a", BF16, [32, 64])?
                .matmul(gb.input("b", BF16, [64, 32])?, with_c(c))
        },
        "C must have the output element type",
    );
    expect_infer_error(
        |gb| {
            let c = gb.input("c", BF16, [32, 16])?;
            gb.input("a", BF16, [32, 64])?
                .matmul(gb.input("b", BF16, [64, 32])?, with_c(c))
        },
        "C has shape",
    );
    expect_infer_error(
        |gb| {
            let c = gb.input("c", BF16, [32, 32])?.transposed(0, 1)?;
            gb.input("a", BF16, [32, 64])?
                .matmul(gb.input("b", BF16, [64, 32])?, with_c(c))
        },
        "not a transposed view",
    );
    expect_infer_error(
        |gb| {
            let c = gb.input("c", BF16, [128, 64])?;
            gb.input_mx("a", MxFormat::Mxfp8, MxDim::Rowwise, [128, 64])?
                .matmul(
                    gb.input_mx("b", MxFormat::Mxfp8, MxDim::Colwise, [64, 64])?,
                    MatmulOpts {
                        out: MatmulOutput::BlockScaled {
                            format: MxFormat::Mxfp8,
                            dim: MxDim::Rowwise,
                        },
                        ..with_c(c)
                    },
                )
        },
        "only supported for plain outputs",
    );
}

fn mx_matmul(
    gb: &GraphBuilder,
    (fa, fb): (MxFormat, MxFormat),
    (m, k, n): (usize, usize, usize),
    out: MatmulOutput,
) -> anyhow::Result<SymTensor<'_>> {
    gb.input_mx("a", fa, MxDim::Rowwise, [m, k])?.matmul(
        gb.input_mx("b", fb, MxDim::Colwise, [k, n])?,
        MatmulOpts {
            out,
            ..Default::default()
        },
    )
}

#[test]
fn invalid_block_scaled_operands() {
    let nvfp4 = MxFormat::Nvfp4;
    let mxfp8 = MxFormat::Mxfp8;
    // NVFP4 needs K to be a multiple of 32 (16 bytes); 16 is a whole block but not aligned.
    let gb = GraphBuilder::new();
    mx_matmul(
        &gb,
        (nvfp4, nvfp4),
        (128, 32, 64),
        MatmulOutput::Plain(BF16),
    )
    .expect("K = 32 is valid");
    expect_infer_error(
        |gb| mx_matmul(gb, (nvfp4, nvfp4), (128, 16, 64), MatmulOutput::Plain(BF16)),
        "inputs need K = 16 to be a provable multiple of 32",
    );
    // K = 8 is not even a whole NVFP4 block.
    expect_infer_error(
        |gb| mx_matmul(gb, (nvfp4, nvfp4), (128, 8, 64), MatmulOutput::Plain(BF16)),
        "block size 16",
    );
    expect_infer_error(
        |gb| mx_matmul(gb, (mxfp8, nvfp4), (128, 64, 64), MatmulOutput::Plain(BF16)),
        "same format",
    );
    expect_infer_error(
        |gb| {
            mx_matmul(
                gb,
                (nvfp4, nvfp4),
                (128, 64, 48),
                MatmulOutput::BlockScaled {
                    format: nvfp4,
                    dim: MxDim::Rowwise,
                },
            )
        },
        "needs its block dimension to be a provable multiple of 32",
    );
    // Block-scaled outputs are 2D.
    expect_infer_error(
        |gb| {
            gb.input("a", BF16, [2, 128, 64])?.matmul(
                gb.input("b", BF16, [2, 64, 64])?,
                MatmulOpts {
                    out: MatmulOutput::BlockScaled {
                        format: mxfp8,
                        dim: MxDim::Rowwise,
                    },
                    ..Default::default()
                },
            )
        },
        "block-scaled matmul outputs must be 2D",
    );
}

/// Operands whose pointers or leading dimensions are not 16-byte aligned.
#[test]
fn unaligned_operands() -> anyhow::Result<()> {
    let cx = test_context();
    // A starts 3 elements into its rows; N = 65 and N = 1 give unaligned D rows.
    for (m, k, n) in [(37, 64, 65), (40, 72, 1), (16, 24, 8)] {
        let (a_full, ha_full) = random_bf16(1, m, k + 3, 110, &cx)?;
        // B is uploaded as [N, K] (the helper needs N > 1 for [K, N]) and transposed.
        let (b_t, hb_t) = random_bf16(1, n, k, 111, &cx)?;
        let hb = hb_t.transposed();
        let gb = GraphBuilder::new();
        let ta = gb.input("a", BF16, [m, k + 3])?.sliced(1, 3, k)?;
        let tb = gb.input("b", BF16, [n, k])?.transposed(0, 1)?;
        gb.output(ta.matmul(tb, matmul_opts(BF16))?)?;
        let graph = build(gb, &cx, Autotune::Off)?;
        let (expected, bounds) = reference(&ha_full.cols_slice(3, k), &hb, None, 1.0, 0.0);
        let d = run(&graph, &[&a_full, &b_t], &cx)?;
        assert_all_close(
            &format!("unaligned {m}x{k}x{n}"),
            &download_f32(&d, &cx)?,
            &expected,
            tolerance(BF16, &bounds),
        );
    }
    Ok(())
}

#[test]
fn batched_transposed_with_c() -> anyhow::Result<()> {
    let cx = test_context();
    let (batch, m, k, n) = (3, 40, 56, 24);
    let (alpha, beta) = (2.0, 0.5);
    let (a_t, ha_t) = random_bf16(batch, k, m, 120, &cx)?;
    let (b, hb) = random_bf16(batch, k, n, 121, &cx)?;
    let (c, hc) = random_bf16(batch, m, n, 122, &cx)?;
    let gb = GraphBuilder::new();
    let ta = gb.input("a", BF16, [batch, k, m])?.transposed(1, 2)?;
    let tb = gb.input("b", BF16, [batch, k, n])?;
    let tc = gb.input("c", BF16, [batch, m, n])?;
    gb.output(ta.matmul(
        tb,
        MatmulOpts {
            out: MatmulOutput::Plain(BF16),
            c: Some(tc),
            alpha,
            beta,
        },
    )?)?;
    let graph = build(gb, &cx, Autotune::Off)?;
    let (expected, bounds) = reference(&ha_t.transposed(), &hb, Some(&hc), alpha, beta);
    let d = run(&graph, &[&a_t, &b, &c], &cx)?;
    assert_eq!(d.shape().dims_slice(), [batch, m, n]);
    assert_all_close(
        "batched transposed with C",
        &download_f32(&d, &cx)?,
        &expected,
        tolerance(BF16, &bounds),
    );
    Ok(())
}

#[test]
fn batched() -> anyhow::Result<()> {
    let cx = test_context();
    let (batch, m, k, n) = (3, 50, 96, 40);
    let (a, ha) = random_bf16(batch, m, k, 40, &cx)?;
    let (b, hb) = random_bf16(batch, k, n, 41, &cx)?;
    let gb = GraphBuilder::new();
    let d = gb
        .input("a", BF16, [batch, m, k])?
        .matmul(gb.input("b", BF16, [batch, k, n])?, matmul_opts(F32))?;
    gb.output(d)?;
    let graph = build(gb, &cx, Autotune::Off)?;
    let (expected, bounds) = reference(&ha, &hb, None, 1.0, 0.0);
    let d = run(&graph, &[&a, &b], &cx)?;
    assert_eq!(d.shape().dims_slice(), [batch, m, n]);
    assert_all_close(
        "batched",
        &download_f32(&d, &cx)?,
        &expected,
        tolerance(F32, &bounds),
    );
    Ok(())
}

#[test]
fn accumulate_c() -> anyhow::Result<()> {
    let cx = test_context();
    let (m, k, n) = (64, 72, 56);
    let (alpha, beta) = (0.5, -2.0);
    let (a, ha) = random_bf16(1, m, k, 50, &cx)?;
    let (b, hb) = random_bf16(1, k, n, 51, &cx)?;
    for out in [BF16, F32] {
        let (c, hc) = match out {
            BF16 => random_bf16(1, m, n, 52, &cx)?,
            _ => random_f32(m, n, 52, &cx)?,
        };
        let gb = GraphBuilder::new();
        let ta = gb.input("a", BF16, [m, k])?;
        let tb = gb.input("b", BF16, [k, n])?;
        let tc = gb.input("c", out, [m, n])?;
        gb.output(ta.matmul(
            tb,
            MatmulOpts {
                out: MatmulOutput::Plain(out),
                c: Some(tc),
                alpha,
                beta,
            },
        )?)?;
        let graph = build(gb, &cx, Autotune::Off)?;
        let (expected, bounds) = reference(&ha, &hb, Some(&hc), alpha, beta);
        let d = run(&graph, &[&a, &b, &c], &cx)?;
        assert_all_close(
            &format!("accumulate {out:?}"),
            &download_f32(&d, &cx)?,
            &expected,
            tolerance(out, &bounds),
        );
    }
    Ok(())
}

#[test]
fn dynamic_m() -> anyhow::Result<()> {
    let cx = test_context();
    let (k, n) = (128, 64);
    let gb = GraphBuilder::new();
    let m = gb.dynamic_dim("m", 64, 0);
    let d = gb
        .input("a", BF16, [Monomial::from(m), Monomial::fixed(k)])?
        .matmul(gb.input("b", BF16, [k, n])?, matmul_opts(F32))?;
    gb.output(d)?;
    let graph = build(gb, &cx, Autotune::Off)?;
    let (b, hb) = random_bf16(1, k, n, 61, &cx)?;
    // The hint, two other sizes (the reshape path), then the first again (cached).
    for (i, m) in [64, 37, 1000, 37].into_iter().enumerate() {
        let (a, ha) = random_bf16(1, m, k, 62 + i as u64, &cx)?;
        let (expected, bounds) = reference(&ha, &hb, None, 1.0, 0.0);
        let d = run(&graph, &[&a, &b], &cx)?;
        assert_eq!(d.shape().dims_slice(), [m, n]);
        assert_all_close(
            &format!("dynamic m={m}"),
            &download_f32(&d, &cx)?,
            &expected,
            tolerance(F32, &bounds),
        );
    }
    Ok(())
}

/// Debug logs of this test binary.
fn captured_logs() -> &'static Mutex<String> {
    static LOGS: OnceLock<Mutex<String>> = OnceLock::new();
    LOGS.get_or_init(|| {
        struct Capture;
        impl std::io::Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                captured_logs()
                    .lock()
                    .push_str(&String::from_utf8_lossy(buf));
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(false)
            .with_writer(|| Capture)
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("failed to install subscriber");
        Mutex::new(String::new())
    })
}

/// The algorithm configs chosen for matmuls whose hint D' layout contains `needle`.
fn chosen_configs(needle: &str) -> Vec<String> {
    captured_logs()
        .lock()
        .lines()
        .filter(|line| line.contains("cleandl::matmul") && line.contains(needle))
        .filter_map(|line| line.split_once(": chose ").map(|(_, c)| c.to_owned()))
        .collect()
}

#[test]
fn autotune_then_cached() -> anyhow::Result<()> {
    captured_logs();
    let cx = test_context();
    // A shape no other test uses, so the tuning cache starts empty.
    let (m, k, n) = (264, 392, 200);
    let (a, ha) = random_bf16(1, m, k, 70, &cx)?;
    let (b, hb) = random_bf16(1, k, n, 71, &cx)?;
    let (expected, bounds) = reference(&ha, &hb, None, 1.0, 0.0);
    for autotune in [Autotune::Full, Autotune::CachedOnly] {
        let gb = GraphBuilder::new();
        let d = gb
            .input("a", BF16, [m, k])?
            .matmul(gb.input("b", BF16, [k, n])?, matmul_opts(BF16))?;
        gb.output(d)?;
        let graph = build(gb, &cx, autotune)?;
        let d = run(&graph, &[&a, &b], &cx)?;
        assert_all_close(
            &format!("autotune {autotune:?}"),
            &download_f32(&d, &cx)?,
            &expected,
            tolerance(BF16, &bounds),
        );
    }
    let chosen = chosen_configs(&format!("rows: {n}, cols: {m},"));
    assert_eq!(chosen.len(), 2, "expected one choice per build: {chosen:?}");
    assert_eq!(
        chosen[0], chosen[1],
        "CachedOnly must reuse the tuned algorithm"
    );
    Ok(())
}

#[test]
fn dynamic_k_zero_fails() -> anyhow::Result<()> {
    let cx = test_context();
    let (m, n) = (32, 16);
    let gb = GraphBuilder::new();
    let k = gb.dynamic_dim("k", 64, 0);
    let d = gb
        .input("a", BF16, [Monomial::fixed(m), Monomial::from(k)])?
        .matmul(
            gb.input("b", BF16, [Monomial::from(k), Monomial::fixed(n)])?,
            matmul_opts(F32),
        )?;
    gb.output(d)?;
    let graph = build(gb, &cx, Autotune::Off)?;
    let a = Tensor::new_zeroed([m, 0], BF16, cx.default_compute_stream(), &cx)?;
    let b = Tensor::new_zeroed([0, n], BF16, cx.default_compute_stream(), &cx)?;
    let err = run(&graph, &[&a, &b], &cx).expect_err("K = 0 should be rejected");
    assert!(format!("{err:#}").contains("K is zero"), "{err:#}");
    Ok(())
}

#[test]
fn unsupported_combination_fails_build() -> anyhow::Result<()> {
    let cx = test_context();
    let gb = GraphBuilder::new();
    let d = gb
        .input("a", BF16, [64, 64])?
        .matmul(gb.input("b", F32, [64, 64])?, matmul_opts(F32))?;
    gb.output(d)?;
    let err = build(gb, &cx, Autotune::Off).expect_err("bf16 x f32 should be unsupported");
    let message = format!("{err:#}");
    assert!(
        message.contains("does not support matmul Float(Bf16) x Float(F32)"),
        "error {message:?} should name the type combination"
    );
    Ok(())
}

// ---------------------------------------------------------------- block-scaled

fn supports_block_scaling(cx: &Context) -> bool {
    if cx.cuda().compute_capability_major() < 10 {
        eprintln!("skipping: cuBLASLt block scaling needs compute capability 10.0 or newer");
        return false;
    }
    true
}

fn encode_e2m1(x: f32) -> u8 {
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let code = (0..8)
        .min_by(|&i, &j| {
            (MAGNITUDES[i] - x.abs())
                .abs()
                .total_cmp(&(MAGNITUDES[j] - x.abs()).abs())
        })
        .unwrap() as u8;
    if x < 0.0 { code | 0x8 } else { code }
}

/// Quantizes a row-major `[rows, cols]` matrix on the host.
fn quantize_mx(
    m: &HostMat,
    format: MxFormat,
    dim: MxDim,
    cx: &Context,
) -> anyhow::Result<MxTensor> {
    let (rows, cols) = (m.rows, m.cols);
    let block = format.block_size();
    // Logical (outer, inner) with blocks along inner.
    let (outer, inner) = match dim {
        MxDim::Rowwise => (rows, cols),
        MxDim::Colwise => (cols, rows),
    };
    let get = |o: usize, i: usize| match dim {
        MxDim::Rowwise => m.at(0, o, i),
        MxDim::Colwise => m.at(0, i, o),
    };
    let blocks = inner / block;
    let sf_rows = outer.next_multiple_of(128);
    let sf_cols = blocks.next_multiple_of(4);
    let mut scales = vec![0u8; sf_rows * sf_cols];
    let mut values = vec![0u8; (outer * inner * format.values_type().bits_per_element()) / 8];
    for o in 0..outer {
        for bi in 0..blocks {
            let amax = (0..block)
                .map(|j| get(o, bi * block + j).abs())
                .fold(0f32, f32::max);
            let (scale_bits, scale) = match format {
                MxFormat::Mxfp8 => {
                    let exp = if amax == 0.0 {
                        -127
                    } else {
                        (amax / 448.0).log2().ceil() as i32
                    };
                    let exp = exp.clamp(-127, 127);
                    ((exp + 127) as u8, (exp as f32).exp2())
                }
                MxFormat::Nvfp4 => {
                    let s = F8E4M3::from_f32((amax / 6.0).max(2f32.powi(-9)));
                    (s.to_bits(), s.to_f32())
                }
            };
            scales[mx_scale_factor_index(o, bi, blocks)] = scale_bits;
            for j in 0..block {
                let idx = o * inner + bi * block + j;
                let x = get(o, bi * block + j) / scale;
                match format {
                    MxFormat::Mxfp8 => values[idx] = F8E4M3::from_f32(x).to_bits(),
                    MxFormat::Nvfp4 => values[idx / 2] |= encode_e2m1(x) << (4 * (idx % 2)),
                }
            }
        }
    }
    let stream = cx.default_compute_stream();
    let values = Tensor::from_bytes(&values, format.values_type(), [outer, inner], stream, cx)?;
    let values = match dim {
        MxDim::Rowwise => values,
        MxDim::Colwise => values.into_transposed(0, 1),
    };
    let scales = Tensor::from_bytes(
        &scales,
        format.scale_factors_type(),
        [sf_rows, sf_cols],
        stream,
        cx,
    )?;
    MxTensor::from_parts(values, scales, format, dim, [rows, cols])
}

/// Quantized random inputs, with their exact dequantized values.
fn mx_inputs(
    format: MxFormat,
    (m, k, n): (usize, usize, usize),
    seed: u64,
    cx: &Context,
) -> anyhow::Result<(MxTensor, HostMat, MxTensor, HostMat)> {
    let (_, ha) = random_bf16(1, m, k, seed, cx)?;
    let (_, hb) = random_bf16(1, k, n, seed + 1, cx)?;
    let a = quantize_mx(&ha, format, MxDim::Rowwise, cx)?;
    let b = quantize_mx(&hb, format, MxDim::Colwise, cx)?;
    let deq = |t: &MxTensor, rows, cols| -> anyhow::Result<HostMat> {
        Ok(HostMat {
            batch: 1,
            rows,
            cols,
            data: dequantize_mx(t, cx)?.values,
        })
    };
    let (da, db) = (deq(&a, m, k)?, deq(&b, k, n)?);
    Ok((a, da, b, db))
}

fn mx_graph(
    format: MxFormat,
    (m, k, n): (usize, usize, usize),
    out: MatmulOutput,
    cx: &Context,
) -> anyhow::Result<ExecGraph> {
    let gb = GraphBuilder::new();
    let a = gb.input_mx("a", format, MxDim::Rowwise, [m, k])?;
    let b = gb.input_mx("b", format, MxDim::Colwise, [k, n])?;
    gb.output(a.matmul(
        b,
        MatmulOpts {
            out,
            ..Default::default()
        },
    )?)?;
    build(gb, cx, Autotune::Off)
}

fn run_mx(
    graph: &ExecGraph,
    a: &MxTensor,
    b: &MxTensor,
    cx: &Context,
) -> anyhow::Result<cleandl::execgraph::GraphOutput> {
    let mut inputs: Vec<Input<'_>> = vec![a.into(), b.into()];
    Ok(graph
        .execute_default(&mut inputs, cx.default_compute_stream())?
        .remove(0))
}

#[test]
fn mx_graph_construction() -> anyhow::Result<()> {
    // Runs everywhere: block-scaled operands are accepted by inference.
    for (format, out) in [
        (MxFormat::Mxfp8, MatmulOutput::Plain(BF16)),
        (
            MxFormat::Mxfp8,
            MatmulOutput::BlockScaled {
                format: MxFormat::Mxfp8,
                dim: MxDim::Colwise,
            },
        ),
        (MxFormat::Nvfp4, MatmulOutput::Plain(F32)),
    ] {
        let gb = GraphBuilder::new();
        let a = gb.input_mx("a", format, MxDim::Rowwise, [128, 256])?;
        let b = gb.input_mx("b", format, MxDim::Colwise, [256, 64])?;
        let d = a.matmul(
            b,
            MatmulOpts {
                out,
                ..Default::default()
            },
        )?;
        assert_eq!(
            d.shape(),
            [128usize, 64].into_iter().map(Into::into).collect()
        );
    }
    Ok(())
}

#[test]
fn mxfp8_inputs() -> anyhow::Result<()> {
    let cx = test_context();
    if !supports_block_scaling(&cx) {
        return Ok(());
    }
    let dims = (192, 256, 160);
    let (a, ha, b, hb) = mx_inputs(MxFormat::Mxfp8, dims, 80, &cx)?;
    let (expected, bounds) = reference(&ha, &hb, None, 1.0, 0.0);
    for out in [BF16, F32] {
        let graph = mx_graph(MxFormat::Mxfp8, dims, MatmulOutput::Plain(out), &cx)?;
        let d = run_mx(&graph, &a, &b, &cx)?.into_plain();
        assert_all_close(
            &format!("mxfp8 -> {out:?}"),
            &download_f32(&d, &cx)?,
            &expected,
            tolerance(out, &bounds),
        );
    }
    Ok(())
}

#[test]
fn nvfp4_inputs() -> anyhow::Result<()> {
    let cx = test_context();
    if !supports_block_scaling(&cx) {
        return Ok(());
    }
    let dims = (128, 256, 64);
    let (a, ha, b, hb) = mx_inputs(MxFormat::Nvfp4, dims, 90, &cx)?;
    let (expected, bounds) = reference(&ha, &hb, None, 1.0, 0.0);
    let graph = mx_graph(MxFormat::Nvfp4, dims, MatmulOutput::Plain(BF16), &cx)?;
    let d = run_mx(&graph, &a, &b, &cx)?.into_plain();
    assert_all_close(
        "nvfp4 -> bf16",
        &download_f32(&d, &cx)?,
        &expected,
        tolerance(BF16, &bounds),
    );
    Ok(())
}

#[test]
fn block_scaled_outputs() -> anyhow::Result<()> {
    let cx = test_context();
    if !supports_block_scaling(&cx) {
        return Ok(());
    }
    let dims = (256, 128, 192);
    for (format, dim) in [
        (MxFormat::Mxfp8, MxDim::Rowwise),
        (MxFormat::Mxfp8, MxDim::Colwise),
        (MxFormat::Nvfp4, MxDim::Rowwise),
    ] {
        let (a, ha, b, hb) = mx_inputs(format, dims, 100, &cx)?;
        let (expected, bounds) = reference(&ha, &hb, None, 1.0, 0.0);
        let graph = mx_graph(format, dims, MatmulOutput::BlockScaled { format, dim }, &cx)?;
        let d = run_mx(&graph, &a, &b, &cx)?.into_mx();
        assert_eq!((d.format(), d.mx_dim()), (format, dim));
        let deq = dequantize_mx(&d, &cx)?;
        // Half an ulp of the value type, relative to the block scale, plus accumulation error.
        let (rel, unit) = match format {
            MxFormat::Mxfp8 => (2f32.powi(-4), 2f32.powi(-9)),
            MxFormat::Nvfp4 => (0.25, 0.5),
        };
        assert_all_close(
            &format!("{format:?} {dim:?} output"),
            &deq.values,
            &expected,
            |i, e| rel * e.abs() + unit * deq.element_scales[i] + 1e-4 * bounds[i],
        );
    }
    Ok(())
}
