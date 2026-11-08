mod common;

use cleandl::{
    context::Context,
    execgraph::{
        Bound, BuildOptions, ExecGraph, Graph, GraphBuilder, GraphOutput, Input, OpIo, SymTensor,
        SymTensorData, TensorRef, TensorRefMut, TensorTy,
        ops::pointwise::{Pointwise, PwBuilder, PwOutput, Reduce},
        sym::{DimValues, Monomial, Unifier},
        tune::Autotune,
    },
    mx_tensor::{MxDim, MxFormat, MxTensor},
    tensor::{Tensor, TensorView, VirtTensorView},
};
use common::{
    abs_rel_tolerance, assert_all_close, decode_e2m1, decode_ue8m0, dequantize_mx, download,
    download_bytes, mx_scale_factor_index, random_bf16_tensor, random_tensor, seeded_rng,
    test_context,
};
use float8::F8E4M3;
use half::bf16;
use rand_distr::{Normal, Uniform};
use skytens_core::data_type::{ElementType, FloatType, SIntType, UIntType};
use std::cell::RefCell;

const F32: ElementType = ElementType::Float(FloatType::F32);
const BF16: ElementType = ElementType::Float(FloatType::Bf16);
const E4M3: ElementType = ElementType::Float(FloatType::F8E4M3);

/// A graph of pointwise ops whose outputs are all graph outputs.
struct Case {
    gb: GraphBuilder,
    ops: RefCell<Vec<Pointwise>>,
}

impl Case {
    fn new() -> Self {
        Self {
            gb: GraphBuilder::new(),
            ops: RefCell::new(Vec::new()),
        }
    }

    fn pointwise(
        &self,
        f: impl FnOnce(&PwBuilder<'_>) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<SymTensor<'_>>> {
        let op = Pointwise::trace(&self.gb, f)?;
        self.ops.borrow_mut().push(op.clone());
        let outputs = self.gb.add_op(op)?;
        for &output in &outputs {
            self.gb.output(output)?;
        }
        Ok(outputs)
    }

    fn finish(self, cx: &Context) -> anyhow::Result<Built> {
        self.finish_with(cx, Autotune::Off)
    }

    fn finish_with(self, cx: &Context, autotune: Autotune) -> anyhow::Result<Built> {
        let mut ops = self.ops.into_inner();
        assert_eq!(ops.len(), 1, "cases have exactly one op");
        let options = BuildOptions {
            autotune,
            ..Default::default()
        };
        Ok(Built {
            exec: ExecGraph::build(self.gb.finish()?, cx, options)?,
            op: ops.pop().unwrap(),
        })
    }
}

struct Built {
    exec: ExecGraph,
    op: Pointwise,
}

impl Built {
    fn graph(&self) -> &Graph {
        self.exec.graph()
    }

    /// Runs the graph through the execution engine.
    fn run(&self, cx: &Context, inputs: &[TensorView<'_>]) -> anyhow::Result<Vec<GraphOutput>> {
        let mut inputs: Vec<Input<'_>> = inputs.iter().cloned().map(Input::from).collect();
        let outputs = self
            .exec
            .execute_default(&mut inputs, cx.default_compute_stream())?;
        cx.default_compute_stream().block_on()?;
        Ok(outputs)
    }

    /// Runs the op's executor directly on views of `inputs`, with freshly allocated outputs.
    fn run_direct(
        &self,
        cx: &Context,
        inputs: &[TensorView<'_>],
        tile: Option<&[u32]>,
    ) -> anyhow::Result<Vec<GraphOutput>> {
        let stream = cx.default_compute_stream();
        let info = self.graph().info();
        let dims = self.bind_dims(inputs)?;
        let executor = self.op.build_executor(cx, info, tile)?;

        let mut outputs = Vec::new();
        for (i, &id) in self.graph().outputs().iter().enumerate() {
            let shape = info.shape(id).eval(&dims)?;
            outputs.push(match info.ty(id) {
                TensorTy::Plain(ty) => GraphOutput::Plain(match &self.op.ir().outputs[i] {
                    PwOutput::Reduce { op, .. } => Tensor::from_slice(
                        &vec![op.identity(); shape.element_count()],
                        shape,
                        stream,
                        cx,
                    )?,
                    _ => Tensor::new_zeroed(shape, ty, stream, cx)?,
                }),
                TensorTy::BlockScaled { format, dim } => {
                    GraphOutput::Mx(MxTensor::new_zeroed(shape, format, dim, stream, cx)?)
                }
            });
        }

        let operands = self
            .op
            .inputs()
            .iter()
            .map(|&id| {
                Ok(Bound::Ro(TensorRef::Plain(
                    self.operand_view(id, inputs, &dims)?,
                )))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut io = OpIo {
            inputs: operands,
            outputs: outputs
                .iter_mut()
                .map(|o| match o {
                    GraphOutput::Plain(t) => TensorRefMut::Plain(t.view_mut()),
                    GraphOutput::Mx(t) => TensorRefMut::Mx(t.view_mut()),
                })
                .collect(),
            scratch: None,
            dims: &dims,
        };
        executor.execute(&mut io, stream, cx)?;
        drop(io);
        stream.block_on()?;
        Ok(outputs)
    }

    fn bind_dims(&self, inputs: &[TensorView<'_>]) -> anyhow::Result<DimValues> {
        let info = self.graph().info();
        let mut unifier = Unifier::new();
        for (input, tensor) in self.graph().inputs().iter().zip(inputs) {
            unifier.add_layout(
                &input.name,
                info.layout(input.tensor).unwrap(),
                &tensor.shape(),
                &tensor.strides(),
                0,
            )?;
        }
        let mut dims = DimValues::new();
        unifier.solve(&mut dims, info.dims())?;
        Ok(dims)
    }

    /// The view of a graph input addressed by symbolic tensor `id`.
    fn operand_view<'a>(
        &self,
        id: cleandl::execgraph::SymTensorId,
        inputs: &[TensorView<'a>],
        dims: &DimValues,
    ) -> anyhow::Result<TensorView<'a>> {
        let info = self.graph().info();
        let SymTensorData::Plain { buffer, layout, ty } = info.tensor(id) else {
            panic!("block-scaled operand")
        };
        let input = self
            .graph()
            .inputs()
            .iter()
            .position(|input| match info.tensor(input.tensor) {
                SymTensorData::Plain { buffer: b, .. } => b == buffer,
                _ => false,
            })
            .expect("operand is not a view of a graph input");
        let (shape, strides, offset) = layout.eval(dims)?;
        assert_eq!(offset, 0);
        let base = &inputs[input];
        Ok(TensorView::from_parts(
            VirtTensorView::with_strides(shape, strides, *ty),
            *base.storage(),
            base.cx(),
        ))
    }
}

trait AsPlain {
    fn as_plain(&self) -> &Tensor;
}

impl AsPlain for GraphOutput {
    fn as_plain(&self) -> &Tensor {
        match self {
            GraphOutput::Plain(t) => t,
            GraphOutput::Mx(_) => panic!("expected a plain output"),
        }
    }
}

fn download_f32(output: &GraphOutput, cx: &Context) -> anyhow::Result<Vec<f32>> {
    let GraphOutput::Plain(t) = output else {
        panic!("expected a plain output")
    };
    Ok(match t.element_type() {
        ElementType::Float(FloatType::F32) => download::<f32>(t, cx)?,
        ElementType::Float(FloatType::Bf16) => download::<bf16>(t, cx)?
            .into_iter()
            .map(|v| v.to_f32())
            .collect(),
        ElementType::Float(FloatType::F8E4M3) => download::<u8>(t, cx)?
            .into_iter()
            .map(|b| F8E4M3::from_bits(b).to_f32())
            .collect(),
        ty => panic!("unsupported output type {ty:?}"),
    })
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn bf16_round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

#[test]
fn elementwise_broadcasting() -> anyhow::Result<()> {
    let cx = test_context();
    let mut rng = seeded_rng(1);
    let (m, n) = (33, 1000);
    let (x, xh) = random_tensor(Normal::new(0.0f32, 1.0)?, [m, n], &mut rng, &cx)?;
    let (bias, bh) = random_tensor(Normal::new(0.0f32, 1.0)?, [n], &mut rng, &cx)?;
    let (col, ch) = random_tensor(Uniform::new(0.5f32, 2.0)?, [m], &mut rng, &cx)?;
    let (s, sh) = random_tensor(Uniform::new(0.5f32, 2.0)?, [1], &mut rng, &cx)?;

    let case = Case::new();
    let xs = case.gb.input("x", F32, [m, n])?;
    let biass = case.gb.input("bias", F32, [n])?;
    let cols = case.gb.input("col", F32, [m, 1])?;
    let ss = case.gb.input("s", F32, [1])?;
    case.pointwise(|p| {
        let x = p.input(xs)?;
        let y = (x + p.input(biass)?) * p.input(cols)? * 0.5 - 1.0;
        let z = (y.tanh() + y.sigmoid()) * p.input(ss)? + 2.0 * y.exp().min(10.0);
        p.output(z, F32)?;
        p.output(y.abs().sqrt().max(x), F32)
    })?;
    let built = case.finish(&cx)?;
    let out = built.run(
        &cx,
        &[x.view(), bias.view(), col.reshaped([m, 1]), s.view()],
    )?;

    let mut expected_z = vec![0.0; m * n];
    let mut expected_w = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let x = xh[i * n + j];
            let y = (x + bh[j]) * ch[i] * 0.5 - 1.0;
            expected_z[i * n + j] = (y.tanh() + sigmoid(y)) * sh[0] + 2.0 * y.exp().min(10.0);
            expected_w[i * n + j] = y.abs().sqrt().max(x);
        }
    }
    let tol = abs_rel_tolerance(1e-5, 1e-5);
    assert_all_close("z", &download_f32(&out[0], &cx)?, &expected_z, &tol);
    assert_all_close("w", &download_f32(&out[1], &cx)?, &expected_w, &tol);
    Ok(())
}

#[test]
fn mixed_dtypes_and_outputs() -> anyhow::Result<()> {
    let cx = test_context();
    let mut rng = seeded_rng(2);
    let (m, n) = (64, 256);
    let (a, ah) = random_bf16_tensor(Normal::new(0.0f32, 1.0)?, [m, n], &mut rng, &cx)?;
    let (b, bh) = random_tensor(Normal::new(0.0f32, 1.0)?, [m, n], &mut rng, &cx)?;

    let case = Case::new();
    let as_ = case.gb.input("a", BF16, [m, n])?;
    let bs = case.gb.input("b", F32, [m, n])?;
    case.pointwise(|p| {
        let (a, b) = (p.input(as_)?, p.input(bs)?);
        p.output(a * b, BF16)?;
        p.output(a + b, F32)?;
        // Large magnitudes saturate to the largest finite E4M3 value.
        p.output(a * b * 300.0, E4M3)
    })?;
    let out = case.finish(&cx)?.run(&cx, &[a.view(), b.view()])?;

    let prod: Vec<f32> = ah.iter().zip(&bh).map(|(a, b)| a.to_f32() * b).collect();
    let sum: Vec<f32> = ah.iter().zip(&bh).map(|(a, b)| a.to_f32() + b).collect();
    let fp8: Vec<f32> = prod
        .iter()
        .map(|p| F8E4M3::from_f32((p * 300.0).clamp(-448.0, 448.0)).to_f32())
        .collect();
    let bf16_prod: Vec<f32> = prod.iter().map(|&p| bf16_round(p)).collect();
    assert_all_close(
        "a*b bf16",
        &download_f32(&out[0], &cx)?,
        &bf16_prod,
        |_, _| 0.0,
    );
    assert_all_close("a+b", &download_f32(&out[1], &cx)?, &sum, |_, _| 0.0);
    let got = download_f32(&out[2], &cx)?;
    assert!(got.iter().all(|v| v.abs() <= 448.0));
    assert!(got.iter().any(|v| v.abs() == 448.0));
    assert_all_close("a*b fp8", &got, &fp8, |_, _| 0.0);
    Ok(())
}

#[test]
fn transposed_input_and_odd_sizes() -> anyhow::Result<()> {
    let cx = test_context();
    let mut rng = seeded_rng(3);
    let (b, m, n) = (7, 3, 257);
    let (x, xh) = random_tensor(Normal::new(0.0f32, 1.0)?, [b, m, n], &mut rng, &cx)?;
    let (y, yh) = random_tensor(Normal::new(0.0f32, 1.0)?, [n, m], &mut rng, &cx)?;
    let (z, zh) = random_tensor(Normal::new(0.0f32, 1.0)?, [m], &mut rng, &cx)?;

    let case = Case::new();
    let xs = case.gb.input("x", F32, [b, m, n])?;
    let ys = case.gb.input("y", F32, [n, m])?.transposed(0, 1)?;
    let zs = case.gb.input("z", F32, [m, 1])?;
    case.pointwise(|p| {
        let v = p.input(xs)? * p.input(ys)? - p.input(zs)?;
        p.output(v.fma(2.0, 1.0), F32)
    })?;
    let out = case
        .finish(&cx)?
        .run(&cx, &[x.view(), y.view(), z.reshaped([m, 1])])?;

    let mut expected = vec![0.0; b * m * n];
    for i in 0..b {
        for j in 0..m {
            for k in 0..n {
                let v = xh[(i * m + j) * n + k] * yh[k * m + j] - zh[j];
                expected[(i * m + j) * n + k] = v.mul_add(2.0, 1.0);
            }
        }
    }
    assert_all_close(
        "out",
        &download_f32(&out[0], &cx)?,
        &expected,
        abs_rel_tolerance(1e-6, 1e-6),
    );
    Ok(())
}

#[test]
fn index_masks() -> anyhow::Result<()> {
    let cx = test_context();
    let mut rng = seeded_rng(4);
    let (m, n) = (65, 130);
    let (x, xh) = random_tensor(Normal::new(0.0f32, 1.0)?, [m, n], &mut rng, &cx)?;

    let case = Case::new();
    let xs = case.gb.input("x", F32, [m, n])?;
    case.pointwise(|p| {
        let x = p.input(xs)?;
        let (i, j) = (p.index(0), p.index(1));
        let causal = p.select(j.le(i), x, f32::NEG_INFINITY);
        p.output(causal, F32)?;
        let even = (i % 2).eq(0) & !(j.lt(3));
        p.output(even.select(i * 1000 + j, -1).to_f32(), F32)
    })?;
    let out = case.finish(&cx)?.run(&cx, &[x.view()])?;

    let mut causal = vec![0.0; m * n];
    let mut pattern = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            causal[i * n + j] = if j <= i {
                xh[i * n + j]
            } else {
                f32::NEG_INFINITY
            };
            pattern[i * n + j] = if i % 2 == 0 && j >= 3 {
                (i * 1000 + j) as f32
            } else {
                -1.0
            };
        }
    }
    let got = download_f32(&out[0], &cx)?;
    for (i, (&g, &e)) in got.iter().zip(&causal).enumerate() {
        assert!(g == e, "causal[{i}]: {g} != {e}");
    }
    assert_all_close("pattern", &download_f32(&out[1], &cx)?, &pattern, |_, _| {
        0.0
    });
    Ok(())
}

/// Sums in f64 for references.
fn reduce_ref(shape: &[usize], data: &[f32], axes: &[usize], op: Reduce) -> Vec<f32> {
    let kept: Vec<usize> = (0..shape.len()).filter(|a| !axes.contains(a)).collect();
    let out_len: usize = kept.iter().map(|&a| shape[a]).product();
    let mut out = vec![
        match op {
            Reduce::Sum => 0.0f64,
            Reduce::Max => f64::NEG_INFINITY,
            Reduce::Min => f64::INFINITY,
        };
        out_len
    ];
    let mut coord = vec![0usize; shape.len()];
    for &v in data {
        let o = kept.iter().fold(0, |acc, &a| acc * shape[a] + coord[a]);
        let v = f64::from(v);
        out[o] = match op {
            Reduce::Sum => out[o] + v,
            Reduce::Max => out[o].max(v),
            Reduce::Min => out[o].min(v),
        };
        for a in (0..shape.len()).rev() {
            coord[a] += 1;
            if coord[a] < shape[a] {
                break;
            }
            coord[a] = 0;
        }
    }
    out.into_iter().map(|v| v as f32).collect()
}

#[test]
fn reductions() -> anyhow::Result<()> {
    let cx = test_context();
    let mut rng = seeded_rng(5);
    let shape = [7, 3, 257];
    let (x, xh) = random_tensor(Normal::new(0.0f32, 1.0)?, shape, &mut rng, &cx)?;

    let case = Case::new();
    let xs = case.gb.input("x", F32, shape)?;
    case.pointwise(|p| {
        let x = p.input(xs)?;
        p.output_reduce(x, Reduce::Sum, &[2])?;
        p.output_reduce(x, Reduce::Sum, &[0])?;
        p.output_reduce(x * x, Reduce::Sum, &[0, 1, 2])?;
        p.output_reduce(x, Reduce::Max, &[2])?;
        p.output_reduce(x, Reduce::Min, &[0, 2])?;
        p.output_reduce(-x.abs() - 1.0, Reduce::Max, &[1])
    })?;
    let out = case.finish(&cx)?.run(&cx, &[x.view()])?;

    type Check = (&'static [usize], Reduce, Box<dyn Fn(f32) -> f32>);
    let checks: [Check; 6] = [
        (&[2], Reduce::Sum, Box::new(|x| x)),
        (&[0], Reduce::Sum, Box::new(|x| x)),
        (&[0, 1, 2], Reduce::Sum, Box::new(|x| x * x)),
        (&[2], Reduce::Max, Box::new(|x| x)),
        (&[0, 2], Reduce::Min, Box::new(|x| x)),
        (&[1], Reduce::Max, Box::new(|x| -x.abs() - 1.0)),
    ];
    for (i, (axes, op, f)) in checks.iter().enumerate() {
        let data: Vec<f32> = xh.iter().map(|&x| f(x)).collect();
        let expected = reduce_ref(&shape, &data, axes, *op);
        assert_all_close(
            &format!("reduction {i} ({op:?} over {axes:?})"),
            &download_f32(&out[i], &cx)?,
            &expected,
            abs_rel_tolerance(1e-3, 1e-5),
        );
    }
    Ok(())
}

#[test]
fn reduction_of_transposed_input() -> anyhow::Result<()> {
    let cx = test_context();
    let mut rng = seeded_rng(6);
    let (m, n) = (300, 77);
    let (x, xh) = random_tensor(Normal::new(0.0f32, 1.0)?, [n, m], &mut rng, &cx)?;

    let case = Case::new();
    let xs = case.gb.input("x", F32, [n, m])?.transposed(0, 1)?;
    case.pointwise(|p| {
        let x = p.input(xs)?;
        p.output_reduce(x, Reduce::Sum, &[0])?;
        p.output_reduce(x, Reduce::Min, &[1])
    })?;
    let out = case.finish(&cx)?.run(&cx, &[x.view()])?;

    // x is [n, m]; the op sees its transpose [m, n].
    let sums = reduce_ref(&[n, m], &xh, &[1], Reduce::Sum);
    let mins = reduce_ref(&[n, m], &xh, &[0], Reduce::Min);
    let tol = abs_rel_tolerance(1e-4, 1e-5);
    assert_all_close("sums", &download_f32(&out[0], &cx)?, &sums, &tol);
    assert_all_close("mins", &download_f32(&out[1], &cx)?, &mins, &tol);
    Ok(())
}

/// Stores of partial tiles must not touch memory past the end of the output view.
#[test]
fn partial_tiles_are_masked() -> anyhow::Result<()> {
    let cx = test_context();
    let stream = cx.default_compute_stream();
    let mut rng = seeded_rng(7);
    let (m, n) = (33, 1000);
    // Transposed so the two axes don't collapse and both get partial tiles.
    let (x, xh) = random_tensor(Normal::new(0.0f32, 1.0)?, [n, m], &mut rng, &cx)?;

    let case = Case::new();
    let xs = case.gb.input("x", F32, [n, m])?.transposed(0, 1)?;
    case.pointwise(|p| {
        let x = p.input(xs)?;
        p.output(x + 1.0, F32)
    })?;
    let built = case.finish(&cx)?;
    let dims = built.bind_dims(&[x.view()])?;

    for tile in [[4u32, 256], [64, 64], [1, 4096]] {
        const SENTINEL: f32 = 12345.0;
        let extra = 5000;
        let mut buffer =
            Tensor::from_slice(&vec![SENTINEL; m * n + extra], [m * n + extra], stream, &cx)?;
        let executor = built
            .op
            .build_executor(&cx, built.graph().info(), Some(&tile))?;
        {
            let mut prefix = buffer.sliced_mut(0, ..m * n);
            let mut view = prefix.reshaped_mut([m, n]);
            let operand = built.operand_view(built.op.inputs()[0], &[x.view()], &dims)?;
            let mut io = OpIo {
                inputs: vec![Bound::Ro(TensorRef::Plain(operand))],
                outputs: vec![TensorRefMut::Plain(view.view_mut())],
                scratch: None,
                dims: &dims,
            };
            executor.execute(&mut io, stream, &cx)?;
        }
        let got = download::<f32>(&buffer, &cx)?;
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    got[i * n + j],
                    xh[j * m + i] + 1.0,
                    "tile {tile:?} at ({i}, {j})"
                );
            }
        }
        assert!(
            got[m * n..].iter().all(|&v| v == SENTINEL),
            "tile {tile:?}: elements past the end were written"
        );
    }
    Ok(())
}

#[test]
fn dynamic_dims() -> anyhow::Result<()> {
    let cx = test_context();
    let mut rng = seeded_rng(8);
    let n = 96;

    let case = Case::new();
    let mdim = case.gb.dynamic_dim("m", 64, 0);
    let shape = vec![Monomial::from(mdim), Monomial::from(n)];
    let xs = case.gb.input("x", F32, shape)?;
    let bs = case.gb.input("b", F32, [n])?;
    case.pointwise(|p| {
        let x = p.input(xs)?;
        p.output(x * p.input(bs)?, F32)?;
        p.output_reduce(x, Reduce::Sum, &[0])
    })?;
    let built = case.finish(&cx)?;

    let (b, bh) = random_tensor(Normal::new(0.0f32, 1.0)?, [n], &mut rng, &cx)?;
    for m in [37, 130, 1] {
        let (x, xh) = random_tensor(Normal::new(0.0f32, 1.0)?, [m, n], &mut rng, &cx)?;
        let out = built.run(&cx, &[x.view(), b.view()])?;
        let prod: Vec<f32> = xh.iter().enumerate().map(|(i, &x)| x * bh[i % n]).collect();
        assert_all_close(
            &format!("prod m={m}"),
            &download_f32(&out[0], &cx)?,
            &prod,
            |_, _| 0.0,
        );
        let sums = reduce_ref(&[m, n], &xh, &[0], Reduce::Sum);
        assert_all_close(
            &format!("sums m={m}"),
            &download_f32(&out[1], &cx)?,
            &sums,
            abs_rel_tolerance(1e-4, 1e-5),
        );
    }
    Ok(())
}

/// Host reference for block-scaled quantization of a row-major `[rows, cols]` matrix:
/// per element, the dequantized value and its scale.
fn quantize_ref(
    x: &[f32],
    rows: usize,
    cols: usize,
    format: MxFormat,
    dim: MxDim,
) -> (Vec<f32>, Vec<f32>) {
    let bs = format.block_size();
    let mut values = vec![0.0; rows * cols];
    let mut scales = vec![0.0; rows * cols];
    // Blocks as lists of element indices.
    let blocks: Vec<Vec<usize>> = match dim {
        MxDim::Rowwise => (0..rows)
            .flat_map(|r| {
                (0..cols / bs).map(move |b| (0..bs).map(|i| r * cols + b * bs + i).collect())
            })
            .collect(),
        MxDim::Colwise => (0..cols)
            .flat_map(|c| {
                (0..rows / bs).map(move |b| (0..bs).map(|i| (b * bs + i) * cols + c).collect())
            })
            .collect(),
    };
    for block in blocks {
        let amax = block.iter().map(|&i| x[i].abs()).fold(0.0f32, f32::max);
        let (scale, descale) = match format {
            MxFormat::Mxfp8 => {
                let bits = (amax / 448.0).to_bits() as i32;
                let (exp, mantissa) = ((bits >> 23) & 0xFF, bits & 0x7F_FFFF);
                let mut encoded = if exp == 0 {
                    i32::from(mantissa > 0x40_0000)
                } else {
                    (exp + i32::from(mantissa != 0)).min(254)
                };
                if exp == 255 && mantissa != 0 {
                    encoded = 255;
                }
                let descale_exp = 254 - encoded;
                let descale = if descale_exp >= 1 {
                    f32::from_bits((descale_exp as u32) << 23)
                } else if descale_exp == 0 {
                    f32::from_bits(0x0020_0000)
                } else {
                    f32::NAN
                };
                (decode_ue8m0(encoded as u8), descale)
            }
            MxFormat::Nvfp4 => {
                let scale = F8E4M3::from_f32((amax / 6.0).min(448.0)).to_f32();
                (scale, if scale == 0.0 { 0.0 } else { 1.0 / scale })
            }
        };
        for &i in &block {
            let v = x[i] * descale;
            let q = match format {
                MxFormat::Mxfp8 => F8E4M3::from_f32(v.clamp(-448.0, 448.0)).to_f32(),
                MxFormat::Nvfp4 => round_e2m1(v.clamp(-6.0, 6.0)),
            };
            values[i] = q * scale;
            scales[i] = scale;
        }
    }
    (values, scales)
}

/// Rounds to the nearest E2M1 value, ties to even.
fn round_e2m1(v: f32) -> f32 {
    let mut best = 0u8;
    for bits in 0..8u8 {
        let (d, best_d) = (
            (decode_e2m1(bits) - v.abs()).abs(),
            (decode_e2m1(best) - v.abs()).abs(),
        );
        if d < best_d || (d == best_d && bits % 2 == 0) {
            best = bits;
        }
    }
    decode_e2m1(best).copysign(v)
}

#[allow(clippy::too_many_arguments)]
fn check_mx(
    name: &str,
    out: &GraphOutput,
    x: &[f32],
    rows: usize,
    cols: usize,
    format: MxFormat,
    dim: MxDim,
    cx: &Context,
) -> anyhow::Result<()> {
    let GraphOutput::Mx(t) = out else {
        panic!("expected a block-scaled output")
    };
    assert_eq!((t.shape()[0], t.shape()[1]), (rows, cols));
    assert_eq!((t.format(), t.mx_dim()), (format, dim));
    let got = dequantize_mx(t, cx)?;
    let (values, scales) = quantize_ref(x, rows, cols, format, dim);
    assert_all_close(
        &format!("{name} scales"),
        &got.element_scales,
        &scales,
        |_, _| 0.0,
    );
    assert_all_close(&format!("{name} values"), &got.values, &values, |_, _| 0.0);

    // Scale factor padding is never written.
    let sf = download_bytes(t.scale_factors(), cx)?;
    let (sf_rows, blocks) = match dim {
        MxDim::Rowwise => (rows, cols / format.block_size()),
        MxDim::Colwise => (cols, rows / format.block_size()),
    };
    let mut valid = vec![false; sf.len()];
    for r in 0..sf_rows {
        for b in 0..blocks {
            valid[mx_scale_factor_index(r, b, blocks)] = true;
        }
    }
    assert!(
        sf.iter().zip(&valid).all(|(&s, &v)| v || s == 0),
        "{name}: scale factor padding was written"
    );
    Ok(())
}

/// Rows with very different magnitudes, plus some zero blocks.
fn mx_test_data(
    rows: usize,
    cols: usize,
    seed: u64,
    cx: &Context,
) -> anyhow::Result<(Tensor, Vec<f32>)> {
    let mut rng = seeded_rng(seed);
    let (_, mut host) = random_tensor(Normal::new(0.0f32, 1.0)?, [rows, cols], &mut rng, cx)?;
    for r in 0..rows {
        let scale = 2.0f32.powi((r % 41) as i32 - 20);
        for c in 0..cols {
            host[r * cols + c] *= scale;
            if (r / 7 + c / 40) % 11 == 0 {
                host[r * cols + c] = 0.0;
            }
        }
    }
    let tensor = Tensor::from_slice(&host, [rows, cols], cx.default_compute_stream(), cx)?;
    Ok((tensor, host))
}

#[test]
fn mxfp8_outputs() -> anyhow::Result<()> {
    let cx = test_context();
    for (rows, cols) in [(256, 512), (100, 96), (96, 100), (160, 64)] {
        let (x, xh) = mx_test_data(rows, cols, rows as u64, &cx)?;
        let case = Case::new();
        let xs = case.gb.input("x", F32, [rows, cols])?;
        let dims_ok: Vec<MxDim> = [MxDim::Rowwise, MxDim::Colwise]
            .into_iter()
            .filter(|d| [rows, cols][d.block_dimension_2d()] % 32 == 0)
            .collect();
        case.pointwise(|p| {
            let x = p.input(xs)?;
            for &dim in &dims_ok {
                p.output_mx(x * 1.5, MxFormat::Mxfp8, dim)?;
            }
            Ok(())
        })?;
        let out = case.finish(&cx)?.run(&cx, &[x.view()])?;
        let scaled: Vec<f32> = xh.iter().map(|v| v * 1.5).collect();
        for (o, &dim) in out.iter().zip(&dims_ok) {
            check_mx(
                &format!("[{rows}, {cols}] {dim:?}"),
                o,
                &scaled,
                rows,
                cols,
                MxFormat::Mxfp8,
                dim,
                &cx,
            )?;
        }
    }
    Ok(())
}

#[test]
fn mxfp8_output_of_3d_iteration_space() -> anyhow::Result<()> {
    let cx = test_context();
    let (b, m, n) = (3, 64, 96);
    let (x, xh) = mx_test_data(b * m, n, 9, &cx)?;
    let mut rng = seeded_rng(10);
    let (bias, bh) = random_tensor(Normal::new(0.0f32, 1.0)?, [m], &mut rng, &cx)?;

    let case = Case::new();
    let xs = case.gb.input("x", F32, [b * m, n])?.reshaped([b, m, n])?;
    let biass = case.gb.input("bias", F32, [m, 1])?;
    case.pointwise(|p| {
        let v = p.input(xs)? + p.input(biass)?;
        p.output_mx(v, MxFormat::Mxfp8, MxDim::Rowwise)?;
        p.output_mx(v, MxFormat::Mxfp8, MxDim::Colwise)
    })?;
    let out = case
        .finish(&cx)?
        .run(&cx, &[x.view(), bias.reshaped([m, 1])])?;
    let v: Vec<f32> = xh
        .iter()
        .enumerate()
        .map(|(i, x)| x + bh[(i / n) % m])
        .collect();
    check_mx(
        "rowwise",
        &out[0],
        &v,
        b * m,
        n,
        MxFormat::Mxfp8,
        MxDim::Rowwise,
        &cx,
    )?;
    check_mx(
        "colwise",
        &out[1],
        &v,
        b * m,
        n,
        MxFormat::Mxfp8,
        MxDim::Colwise,
        &cx,
    )?;
    Ok(())
}

fn supports_fp4(cx: &Context) -> bool {
    cx.cuda().compute_capability_major() >= 10
}

#[test]
fn nvfp4_outputs() -> anyhow::Result<()> {
    let cx = test_context();
    if !supports_fp4(&cx) {
        eprintln!("skipping nvfp4_outputs: f32 -> e2m1 conversion requires sm_100 or newer");
        return Ok(());
    }
    for (rows, cols) in [(256, 512), (100, 96), (96, 100)] {
        let (x, xh) = mx_test_data(rows, cols, rows as u64 + 1, &cx)?;
        let case = Case::new();
        let xs = case.gb.input("x", F32, [rows, cols])?;
        let dims_ok: Vec<MxDim> = [MxDim::Rowwise, MxDim::Colwise]
            .into_iter()
            .filter(|d| [rows, cols][d.block_dimension_2d()] % 16 == 0)
            .collect();
        case.pointwise(|p| {
            let x = p.input(xs)?;
            for &dim in &dims_ok {
                p.output_mx(x, MxFormat::Nvfp4, dim)?;
            }
            Ok(())
        })?;
        let out = case.finish(&cx)?.run(&cx, &[x.view()])?;
        for (o, &dim) in out.iter().zip(&dims_ok) {
            check_mx(
                &format!("[{rows}, {cols}] {dim:?}"),
                o,
                &xh,
                rows,
                cols,
                MxFormat::Nvfp4,
                dim,
                &cx,
            )?;
        }
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
        let a = gb.input("a", F32, [4, 8])?;
        let b = gb.input("b", F32, [5])?;
        gb.pointwise(|p| p.output(p.input(a)? + p.input(b)?, F32))?;
        Ok(())
    });
    assert!(e.contains("broadcast"), "{e}");

    let e = err(&|gb| {
        let a = gb.input("a", F32, [4, 100])?;
        gb.pointwise(|p| p.output_mx(p.input(a)?, MxFormat::Mxfp8, MxDim::Rowwise))?;
        Ok(())
    });
    assert!(e.contains("block size"), "{e}");

    let e = err(&|gb| {
        let a = gb.input("a", F32, [100, 64])?;
        gb.pointwise(|p| p.output_mx(p.input(a)?, MxFormat::Nvfp4, MxDim::Colwise))?;
        Ok(())
    });
    assert!(e.contains("block size"), "{e}");

    let e = err(&|gb| {
        let a = gb.input("a", F32, [4, 8])?;
        gb.pointwise(|p| p.output(p.input(a)? + p.index(2).to_f32(), F32))?;
        Ok(())
    });
    assert!(e.contains("axis 2 out of range"), "{e}");

    let e = err(&|gb| {
        let a = gb.input("a", F32, [4, 8])?;
        gb.pointwise(|p| p.output_reduce(p.input(a)?, Reduce::Sum, &[2]))?;
        Ok(())
    });
    assert!(e.contains("axis 2 out of range"), "{e}");

    let e = err(&|gb| {
        let a = gb.input("a", F32, [4, 8])?;
        gb.pointwise(|p| p.output(p.input(a)?.lt(0.0) * 2.0, F32))?;
        Ok(())
    });
    assert!(e.contains("bool"), "{e}");
    Ok(())
}

#[test]
fn integer_inputs_and_more_output_types() -> anyhow::Result<()> {
    let cx = test_context();
    let stream = cx.default_compute_stream();
    let mut rng = seeded_rng(11);
    let n = 777;
    let ih: Vec<i32> = (0..n as i32).map(|i| i * 37 % 1001 - 500).collect();
    let uh: Vec<u32> = (0..n as u32).map(|i| i * 3).collect();
    let i = Tensor::from_slice(&ih, [n], stream, &cx)?;
    let u = Tensor::from_slice(&uh, [n], stream, &cx)?;
    let (x, xh) = random_tensor(Normal::new(0.0f32, 100.0)?, [n], &mut rng, &cx)?;

    let case = Case::new();
    let is = case.gb.input("i", ElementType::SInt(SIntType::S32), [n])?;
    let us = case.gb.input("u", ElementType::UInt(UIntType::U32), [n])?;
    let xs = case.gb.input("x", F32, [n])?;
    case.pointwise(|p| {
        let (i, u, x) = (p.input(is)?, p.input(us)?, p.input(xs)?);
        p.output(i * 2 + u / 2 - i % 7, ElementType::SInt(SIntType::S32))?;
        p.output(x.to_i32(), ElementType::SInt(SIntType::S32))?;
        p.output(x * i * 0.01, ElementType::Float(FloatType::F16))?;
        p.output(x * 1000.0, ElementType::Float(FloatType::F8E5M2))
    })?;
    let out = case
        .finish(&cx)?
        .run(&cx, &[i.view(), u.view(), x.view()])?;
    let plain = |k: usize| out[k].as_plain();

    let ints = download::<i32>(plain(0), &cx)?;
    for k in 0..n {
        assert_eq!(
            ints[k],
            ih[k] * 2 + (uh[k] / 2) as i32 - ih[k] % 7,
            "ints[{k}]"
        );
    }
    let truncated = download::<i32>(plain(1), &cx)?;
    for k in 0..n {
        assert_eq!(truncated[k], xh[k] as i32, "truncated[{k}]");
    }
    let halves: Vec<f32> = download::<half::f16>(plain(2), &cx)?
        .into_iter()
        .map(|v| v.to_f32())
        .collect();
    let expected: Vec<f32> = (0..n)
        .map(|k| half::f16::from_f32(xh[k] * ih[k] as f32 * 0.01).to_f32())
        .collect();
    assert_all_close("f16", &halves, &expected, |_, _| 0.0);
    let e5m2: Vec<f32> = download::<u8>(plain(3), &cx)?
        .into_iter()
        .map(|b| float8::F8E5M2::from_bits(b).to_f32())
        .collect();
    let expected: Vec<f32> = xh
        .iter()
        .map(|x| float8::F8E5M2::from_f32((x * 1000.0).clamp(-57344.0, 57344.0)).to_f32())
        .collect();
    assert_all_close("e5m2", &e5m2, &expected, |_, _| 0.0);
    Ok(())
}

/// Every autotuning candidate computes the same results.
#[test]
fn all_tile_candidates_agree() -> anyhow::Result<()> {
    let cx = test_context();
    let (rows, cols) = (128, 96);
    let (x, xh) = mx_test_data(rows, cols, 12, &cx)?;
    let mut rng = seeded_rng(13);
    let (bias, bh) = random_tensor(Normal::new(0.0f32, 1.0)?, [cols], &mut rng, &cx)?;

    let case = Case::new();
    let xs = case.gb.input("x", F32, [rows, cols])?;
    let biass = case.gb.input("bias", F32, [cols])?;
    case.pointwise(|p| {
        let v = p.input(xs)? + p.input(biass)?;
        p.output(v, BF16)?;
        p.output_reduce(v, Reduce::Sum, &[1])?;
        p.output_reduce(v, Reduce::Max, &[0])?;
        p.output_mx(v, MxFormat::Mxfp8, MxDim::Rowwise)?;
        p.output_mx(v, MxFormat::Mxfp8, MxDim::Colwise)
    })?;
    let built = case.finish(&cx)?;
    let v: Vec<f32> = xh
        .iter()
        .enumerate()
        .map(|(i, x)| x + bh[i % cols])
        .collect();

    let candidates = built.op.tile_candidates(built.graph().info())?;
    assert!(candidates.len() > 1);
    for tile in &candidates {
        let out = built.run_direct(&cx, &[x.view(), bias.view()], Some(tile))?;
        let bf: Vec<f32> = v.iter().map(|&v| bf16_round(v)).collect();
        assert_all_close(
            &format!("{tile:?} bf16"),
            &download_f32(&out[0], &cx)?,
            &bf,
            |_, _| 0.0,
        );
        assert_all_close(
            &format!("{tile:?} sum"),
            &download_f32(&out[1], &cx)?,
            &reduce_ref(&[rows, cols], &v, &[1], Reduce::Sum),
            abs_rel_tolerance(1e-3, 1e-5),
        );
        assert_all_close(
            &format!("{tile:?} max"),
            &download_f32(&out[2], &cx)?,
            &reduce_ref(&[rows, cols], &v, &[0], Reduce::Max),
            |_, _| 0.0,
        );
        check_mx(
            &format!("{tile:?} rowwise"),
            &out[3],
            &v,
            rows,
            cols,
            MxFormat::Mxfp8,
            MxDim::Rowwise,
            &cx,
        )?;
        check_mx(
            &format!("{tile:?} colwise"),
            &out[4],
            &v,
            rows,
            cols,
            MxFormat::Mxfp8,
            MxDim::Colwise,
            &cx,
        )?;
    }
    Ok(())
}

#[test]
fn autotune_full_then_cached() -> anyhow::Result<()> {
    let cx = test_context();
    let mut rng = seeded_rng(14);
    // A shape no other test uses, so no tuning result is cached yet.
    let (m, n) = (123, 456);
    let (x, xh) = random_tensor(Normal::new(0.0f32, 1.0)?, [m, n], &mut rng, &cx)?;
    let (b, bh) = random_tensor(Normal::new(0.0f32, 1.0)?, [n], &mut rng, &cx)?;
    let expected: Vec<f32> = xh
        .iter()
        .enumerate()
        .map(|(i, &x)| bf16_round((x - bh[i % n]).tanh()))
        .collect();
    let sums = reduce_ref(&[m, n], &xh, &[1], Reduce::Sum);

    for autotune in [Autotune::Full, Autotune::CachedOnly] {
        let case = Case::new();
        let xs = case.gb.input("x", F32, [m, n])?;
        let bs = case.gb.input("b", F32, [n])?;
        case.pointwise(|p| {
            let x = p.input(xs)?;
            p.output((x - p.input(bs)?).tanh(), BF16)?;
            p.output_reduce(x, Reduce::Sum, &[1])
        })?;
        let built = case.finish_with(&cx, autotune)?;
        let out = built.run(&cx, &[x.view(), b.view()])?;
        assert_all_close(
            &format!("{autotune:?} tanh"),
            &download_f32(&out[0], &cx)?,
            &expected,
            abs_rel_tolerance(0.0, 1.0 / 128.0),
        );
        assert_all_close(
            &format!("{autotune:?} sums"),
            &download_f32(&out[1], &cx)?,
            &sums,
            abs_rel_tolerance(1e-4, 1e-5),
        );
    }
    Ok(())
}

#[test]
fn chained_ops() -> anyhow::Result<()> {
    let cx = test_context();
    let mut rng = seeded_rng(15);
    let (m, n) = (40, 300);
    let (x, xh) = random_tensor(Normal::new(0.0f32, 1.0)?, [m, n], &mut rng, &cx)?;

    let gb = GraphBuilder::new();
    let xs = gb.input("x", F32, [m, n])?;
    let [y] = gb.pointwise(|p| p.output(p.input(xs)? * 2.0, BF16))?[..] else {
        unreachable!()
    };
    let [z, sums] = gb.pointwise(|p| {
        let v = p.input(y)?;
        p.output(v + 1.0, F32)?;
        p.output_reduce(v, Reduce::Sum, &[0])
    })?[..] else {
        unreachable!()
    };
    gb.output(z)?;
    gb.output(sums)?;
    let exec = ExecGraph::build(gb.finish()?, &cx, BuildOptions::default())?;
    let out = exec.execute_default(&mut [x.view().into()], cx.default_compute_stream())?;

    let y: Vec<f32> = xh.iter().map(|&x| bf16_round(x * 2.0)).collect();
    let z: Vec<f32> = y.iter().map(|&y| y + 1.0).collect();
    assert_all_close("z", &download_f32(&out[0], &cx)?, &z, |_, _| 0.0);
    assert_all_close(
        "sums",
        &download_f32(&out[1], &cx)?,
        &reduce_ref(&[m, n], &y, &[0], Reduce::Sum),
        abs_rel_tolerance(1e-4, 1e-5),
    );
    Ok(())
}

/// The executor rejects bound views whose layout differs from the lowered one.
#[test]
fn rejects_mismatched_views() -> anyhow::Result<()> {
    let cx = test_context();
    let stream = cx.default_compute_stream();
    let (m, n) = (16, 64);
    let case = Case::new();
    let xs = case.gb.input("x", F32, [n, m])?.transposed(0, 1)?;
    case.pointwise(|p| p.output(p.input(xs)? + 1.0, F32))?;
    let built = case.finish(&cx)?;
    let executor = built.op.build_executor(&cx, built.graph().info(), None)?;

    // A packed [m, n] view instead of the transposed one.
    let x = Tensor::new_zeroed([m, n], F32, stream, &cx)?;
    let mut out = Tensor::new_zeroed([m, n], F32, stream, &cx)?;
    let dims = DimValues::new();
    let mut io = OpIo {
        inputs: vec![Bound::Ro(TensorRef::Plain(x.view()))],
        outputs: vec![TensorRefMut::Plain(out.view_mut())],
        scratch: None,
        dims: &dims,
    };
    let err = executor.execute(&mut io, stream, &cx).unwrap_err();
    assert!(format!("{err:#}").contains("expected shape"), "{err:#}");
    Ok(())
}
