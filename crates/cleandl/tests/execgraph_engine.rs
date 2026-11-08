mod common;

use cleandl::{
    context::Context,
    execgraph::{
        Access, BuildCx, BuildOptions, ExecGraph, Executor, GraphBuilder, GraphInfo, Init, Input,
        OpIo, Operand, Operator, OutputSpec, SymTensor, SymTensorData, SymTensorId, TensorTy,
        jit::get_or_compile_kernel,
        sym::{DimValues, Monomial, SizeExpr, SymLayout, SymShape},
        tune::Autotune,
    },
    mx_tensor::{MxDim, MxFormat, MxTensor},
    tensor::Tensor,
};
use common::{download, test_context};
use kernel_build_util::{
    ArgBundle, LoadedKernel, Param, PtrArg,
    tileir::{ArrayArg, KernelBuilder},
};
use serde::Serialize;
use skycuda::{
    device_memory_pool::{DeviceMemoryPool, DeviceMemoryPoolOptions},
    stream::Stream,
};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tileir_builder::{Array, ArrayOpts, KernelOpts, bid};

const F32: ElementType = ElementType::Float(FloatType::F32);
const BF16: ElementType = ElementType::Float(FloatType::Bf16);

// ---- AddScalar: `out = x + addend`, or `x += addend` in place, via a TileIR kernel ----

#[derive(Serialize, Clone, Copy, Debug)]
struct AddKey {
    tile: u32,
    addend_bits: u32,
    in_place: bool,
}

struct AddParams {
    x: Param<Array<f32>>,
    out: Option<Param<Array<f32>>>,
}

fn declare_add(kb: &mut KernelBuilder, in_place: bool) -> AddParams {
    AddParams {
        x: kb.array::<f32>("x", ArrayOpts::rank(2)),
        out: (!in_place).then(|| kb.array::<f32>("out", ArrayOpts::rank(2))),
    }
}

fn add_kernel(cx: &Context, key: AddKey) -> anyhow::Result<Arc<LoadedKernel>> {
    get_or_compile_kernel(cx, &key, "test_add_scalar", |module| {
        let mut kb = KernelBuilder::new();
        let p = declare_add(&mut kb, key.in_place);
        kb.define(module, "test_add_scalar", KernelOpts::new(), |args| {
            let index = [bid(0), bid(1)];
            let x = args
                .get_array(p.x)
                .load(index, [1, key.tile])
                .downcast::<f32>();
            let y = x + f32::from_bits(key.addend_bits);
            args.get_array(p.out.unwrap_or(p.x)).store(index, y);
        })
    })
}

struct AddScalar {
    x: SymTensorId,
    addend: f32,
    in_place: bool,
    /// Distinguishes tuning cache entries between tests.
    nonce: u64,
    executions: Arc<AtomicUsize>,
}

impl AddScalar {
    fn new(x: SymTensor<'_>, addend: f32) -> Self {
        Self {
            x: x.id(),
            addend,
            in_place: false,
            nonce: 0,
            executions: Default::default(),
        }
    }
}

fn add_scalar<'g>(x: SymTensor<'g>, addend: f32) -> anyhow::Result<SymTensor<'g>> {
    x.builder().add_op1(AddScalar::new(x, addend))
}

impl Operator for AddScalar {
    fn name(&self) -> &str {
        "add_scalar"
    }

    fn operands(&self) -> Vec<Operand> {
        vec![Operand {
            tensor: self.x,
            access: if self.in_place {
                Access::ReadWrite
            } else {
                Access::ReadOnly
            },
        }]
    }

    fn infer_outputs(&self, g: &GraphInfo) -> anyhow::Result<Vec<OutputSpec>> {
        anyhow::ensure!(g.ty(self.x) == TensorTy::Plain(F32), "expected f32");
        anyhow::ensure!(g.shape(self.x).rank() == 2, "expected rank 2");
        Ok(if self.in_place {
            vec![]
        } else {
            vec![OutputSpec::new(F32, g.shape(self.x).clone(), Init::Uninit)]
        })
    }

    fn build(&self, bcx: &BuildCx<'_>) -> anyhow::Result<Box<dyn Executor>> {
        let hint_shape = bcx.graph.shape(self.x).eval_hint(bcx.graph.dims());
        let key = bcx.tune_key(&(
            self.nonce,
            self.addend.to_bits(),
            self.in_place,
            hint_shape.dims_slice(),
        ))?;
        let (_, executor) = bcx.tune_executor(&key, vec![64u32, 32, 128], |&tile| {
            let key = AddKey {
                tile,
                addend_bits: self.addend.to_bits(),
                in_place: self.in_place,
            };
            Ok(AddExecutor {
                kernel: add_kernel(bcx.cx, key)?,
                key,
                executions: self.executions.clone(),
            })
        })?;
        Ok(Box::new(executor))
    }
}

struct AddExecutor {
    kernel: Arc<LoadedKernel>,
    key: AddKey,
    executions: Arc<AtomicUsize>,
}

fn dims_i64(dims: impl Iterator<Item = usize>) -> Vec<i64> {
    dims.map(|d| d as i64).collect()
}

impl Executor for AddExecutor {
    fn execute(&self, io: &mut OpIo<'_>, stream: &Stream, _cx: &Context) -> anyhow::Result<()> {
        self.executions.fetch_add(1, Ordering::Relaxed);
        let p = declare_add(&mut KernelBuilder::new(), self.key.in_place);
        let x_ref = io.inputs[0].as_ref();
        let (shape, x_strides) = (x_ref.plain().shape(), x_ref.plain().strides());
        drop(x_ref);
        let shape_i64 = dims_i64(shape.dims());
        let x_strides = dims_i64(x_strides.dims());
        let grid = (
            shape[0] as u32,
            shape[1].div_ceil(self.key.tile as usize) as u32,
            1,
        );
        if self.key.in_place {
            let x = io.inputs[0].as_mut().plain();
            let mut args = ArgBundle::new();
            args.set(
                p.x,
                ArrayArg::new(PtrArg::new_mut(x.storage_mut()), &shape_i64, &x_strides),
            );
            unsafe { self.kernel.launch(args, grid, stream)? };
        } else {
            let x_ref = io.input_ro(0);
            let x = x_ref.plain();
            let (_, outputs, _) = io.split();
            let out = outputs[0].plain();
            let out_strides = dims_i64(out.strides().dims());
            let mut args = ArgBundle::new();
            args.set(
                p.x,
                ArrayArg::new(PtrArg::new(x.storage()), &shape_i64, &x_strides),
            );
            args.set(
                p.out.unwrap(),
                ArrayArg::new(PtrArg::new_mut(out.storage_mut()), &shape_i64, &out_strides),
            );
            unsafe { self.kernel.launch(args, grid, stream)? };
        }
        Ok(())
    }
}

// ---- Dummy: arbitrary operands and outputs, optionally filling output 0 via scratch ----

#[derive(Default)]
struct Dummy {
    operands: Vec<Operand>,
    outputs: Vec<OutputSpec>,
    /// Fills output 0 with this bit pattern by way of scratch memory.
    fill_via_scratch: Option<u32>,
    fail_build: bool,
}

impl Operator for Dummy {
    fn name(&self) -> &str {
        "dummy"
    }

    fn operands(&self) -> Vec<Operand> {
        self.operands.clone()
    }

    fn infer_outputs(&self, _: &GraphInfo) -> anyhow::Result<Vec<OutputSpec>> {
        Ok(self.outputs.clone())
    }

    fn build(&self, bcx: &BuildCx<'_>) -> anyhow::Result<Box<dyn Executor>> {
        anyhow::ensure!(!self.fail_build, "requested failure");
        let scratch = self.fill_via_scratch.map(|bits| {
            let SymTensorData::Plain { buffer, .. } = bcx.graph.tensor(bcx.outputs[0]) else {
                panic!("expected a plain output")
            };
            (bits, bcx.graph.buffer(*buffer).bytes.clone())
        });
        Ok(Box::new(DummyExecutor { scratch }))
    }
}

struct DummyExecutor {
    scratch: Option<(u32, SizeExpr)>,
}

impl Executor for DummyExecutor {
    fn scratch_bytes(&self, dims: &DimValues) -> usize {
        self.scratch
            .as_ref()
            .map_or(0, |(_, bytes)| bytes.eval(dims).unwrap())
    }

    fn execute(&self, io: &mut OpIo<'_>, stream: &Stream, _cx: &Context) -> anyhow::Result<()> {
        if let Some((bits, bytes)) = &self.scratch {
            let len = bytes.eval(io.dims)?;
            let scratch = io.scratch.as_mut().unwrap();
            assert!(scratch.len() >= len);
            stream.fill(scratch, *bits)?;
            let out = io.outputs[0].plain();
            stream.copy(&*scratch, out.storage_mut())?;
        }
        Ok(())
    }
}

// ---- helpers ----

fn host_iota(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| i as f32 * scale).collect()
}

fn upload(host: &[f32], shape: impl Into<Shape>, cx: &Context) -> Tensor {
    Tensor::from_slice(host, shape, cx.default_compute_stream(), cx).unwrap()
}

fn off() -> BuildOptions {
    BuildOptions {
        autotune: Autotune::Off,
        pool: None,
    }
}

fn run(eg: &ExecGraph, inputs: &mut [Input<'_>], cx: &Context) -> anyhow::Result<Vec<Tensor>> {
    let outputs = eg.execute_default(inputs, cx.default_compute_stream())?;
    Ok(outputs.into_iter().map(|o| o.into_plain()).collect())
}

fn err_string<T: std::fmt::Debug>(r: anyhow::Result<T>) -> String {
    format!("{:#}", r.expect_err("expected an error"))
}

// ---- tests ----

#[test]
fn graphs_are_thread_safe() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ExecGraph>();
    assert_send_sync::<cleandl::execgraph::Graph>();
}

#[test]
fn fixed_shape_add() -> anyhow::Result<()> {
    let cx = test_context();
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [64usize, 200])?;
    let y = add_scalar(x, 1.5)?;
    gb.output(y)?;
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;
    assert_eq!(eg.inputs()[0].name, "x");
    assert_eq!(eg.num_outputs(), 1);

    let host = host_iota(64 * 200, 0.25);
    let xt = upload(&host, [64, 200], &cx);
    let out = run(&eg, &mut [(&xt).into()], &cx)?;
    assert_eq!(out[0].shape(), Shape::new([64, 200]));
    let expected: Vec<f32> = host.iter().map(|v| v + 1.5).collect();
    assert_eq!(download::<f32>(&out[0], &cx)?, expected);
    Ok(())
}

#[test]
fn dynamic_dims_unify_and_validate() -> anyhow::Result<()> {
    let cx = test_context();
    let gb = GraphBuilder::new();
    let b = gb.dynamic_dim("batch", 32, 0);
    let n = gb.dynamic_dim("n", 64, 3);
    let x = gb.input("x", F32, [Monomial::from(b), n.into()])?;
    let y = gb.input("y", F32, [Monomial::from(b), 16.into()])?;
    gb.output(add_scalar(x, 1.0)?)?;
    gb.output(add_scalar(y, 2.0)?)?;
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;

    let xh = host_iota(48 * 40, 1.0);
    let yh = host_iota(48 * 16, -1.0);
    let xt = upload(&xh, [48, 40], &cx);
    let yt = upload(&yh, [48, 16], &cx);
    let out = run(&eg, &mut [(&xt).into(), (&yt).into()], &cx)?;
    assert_eq!(out[0].shape(), Shape::new([48, 40]));
    assert_eq!(out[1].shape(), Shape::new([48, 16]));
    assert_eq!(
        download::<f32>(&out[0], &cx)?,
        xh.iter().map(|v| v + 1.0).collect::<Vec<_>>()
    );
    assert_eq!(
        download::<f32>(&out[1], &cx)?,
        yh.iter().map(|v| v + 2.0).collect::<Vec<_>>()
    );

    // `batch` disagrees between the inputs.
    let y_bad = upload(&host_iota(47 * 16, 1.0), [47, 16], &cx);
    let e = err_string(run(&eg, &mut [(&xt).into(), (&y_bad).into()], &cx));
    assert!(e.contains("shape[0]"), "{e}");
    // Fixed dim mismatch.
    let y_bad = upload(&host_iota(48 * 17, 1.0), [48, 17], &cx);
    let e = err_string(run(&eg, &mut [(&xt).into(), (&y_bad).into()], &cx));
    assert!(e.contains("y shape[1]"), "{e}");
    // `n` must be a multiple of 8.
    let x_bad = upload(&host_iota(48 * 36, 1.0), [48, 36], &cx);
    let e = err_string(run(&eg, &mut [(&x_bad).into(), (&yt).into()], &cx));
    assert!(e.contains("alignment"), "{e}");
    // Wrong rank, type and count.
    let x_bad = upload(&host_iota(48, 1.0), [48], &cx);
    let e = err_string(run(&eg, &mut [(&x_bad).into(), (&yt).into()], &cx));
    assert!(e.contains("x: expected rank 2"), "{e}");
    let x_bf16 = Tensor::new_zeroed([48, 40], BF16, cx.default_compute_stream(), &cx)?;
    let e = err_string(run(&eg, &mut [(&x_bf16).into(), (&yt).into()], &cx));
    assert!(e.contains("input x"), "{e}");
    let e = err_string(run(&eg, &mut [(&xt).into()], &cx));
    assert!(e.contains("2 inputs"), "{e}");
    Ok(())
}

#[test]
fn views_feed_ops() -> anyhow::Result<()> {
    let cx = test_context();
    let gb = GraphBuilder::new();
    let a = gb.input("a", F32, [8usize, 48])?;
    let b = gb.input("b", F32, [4usize, 6, 32])?;
    let c = gb.input("c", F32, [64usize, 32])?;
    gb.output(add_scalar(a.transposed(0, 1)?, 1.0)?)?;
    gb.output(add_scalar(b.reshaped([24usize, 32])?, 2.0)?)?;
    gb.output(add_scalar(c.sliced(0, 8, 16)?.sliced(1, 4, 20)?, 3.0)?)?;
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;

    let (ah, bh, ch) = (
        host_iota(8 * 48, 1.0),
        host_iota(4 * 6 * 32, 0.5),
        host_iota(64 * 32, 2.0),
    );
    let (at, bt, ct) = (
        upload(&ah, [8, 48], &cx),
        upload(&bh, [4, 6, 32], &cx),
        upload(&ch, [64, 32], &cx),
    );
    let out = run(&eg, &mut [(&at).into(), (&bt).into(), (&ct).into()], &cx)?;

    let mut expected = vec![0.0; 48 * 8];
    for i in 0..48 {
        for j in 0..8 {
            expected[i * 8 + j] = ah[j * 48 + i] + 1.0;
        }
    }
    assert_eq!(download::<f32>(&out[0], &cx)?, expected);
    assert_eq!(
        download::<f32>(&out[1], &cx)?,
        bh.iter().map(|v| v + 2.0).collect::<Vec<_>>()
    );
    let mut expected = Vec::new();
    for i in 8..24 {
        for j in 4..24 {
            expected.push(ch[i * 32 + j] + 3.0);
        }
    }
    assert_eq!(out[2].shape(), Shape::new([16, 20]));
    assert_eq!(download::<f32>(&out[2], &cx)?, expected);
    Ok(())
}

#[test]
fn input_with_symbolic_strides() -> anyhow::Result<()> {
    let cx = test_context();
    let gb = GraphBuilder::new();
    let rows = gb.dynamic_dim("rows", 16, 0);
    let ld = gb.dynamic_dim("ld", 64, 0);
    let layout = SymLayout::new(
        SymShape::new([Monomial::from(rows), 32.into()]),
        vec![ld.into(), 1.into()],
        Default::default(),
    );
    let x = gb.input_with_layout("x", F32, layout)?;
    gb.output(add_scalar(x, 1.0)?)?;
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;

    // A [10, 32] view with row stride 40.
    let host = host_iota(10 * 40, 1.0);
    let full = upload(&host, [10, 40], &cx);
    let view = full.sliced(1, 0..32);
    let out = run(&eg, &mut [view.into()], &cx)?;
    let expected: Vec<f32> = (0..10)
        .flat_map(|i| (0..32).map(move |j| (i * 40 + j) as f32 + 1.0))
        .collect();
    assert_eq!(download::<f32>(&out[0], &cx)?, expected);
    Ok(())
}

#[test]
fn read_write_mutable_input() -> anyhow::Result<()> {
    let cx = test_context();
    let gb = GraphBuilder::new();
    let x = gb.input_mut("x", F32, [16usize, 64])?;
    gb.add_op(AddScalar {
        in_place: true,
        ..AddScalar::new(x.transposed(0, 1)?, 4.0)
    })?;
    let y = add_scalar(x, 1.0)?;
    gb.output(y)?;
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;

    let host = host_iota(16 * 64, 1.0);
    let mut xt = upload(&host, [16, 64], &cx);
    // Mutable inputs must be passed as Rw.
    let e = err_string(run(&eg, &mut [(&xt).into()], &cx));
    assert!(e.contains("Input::Rw"), "{e}");
    let out = run(&eg, &mut [(&mut xt).into()], &cx)?;
    assert_eq!(
        download::<f32>(&xt, &cx)?,
        host.iter().map(|v| v + 4.0).collect::<Vec<_>>()
    );
    assert_eq!(
        download::<f32>(&out[0], &cx)?,
        host.iter().map(|v| v + 5.0).collect::<Vec<_>>()
    );

    // Immutable inputs may be given as Rw, but not written in place.
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [16usize, 64])?;
    let e = err_string(gb.add_op(AddScalar {
        in_place: true,
        ..AddScalar::new(x, 4.0)
    }));
    assert!(e.contains("graph input x is not mutable"), "{e}");
    gb.output(add_scalar(x, 1.0)?)?;
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;
    let out = run(&eg, &mut [(&mut xt).into()], &cx)?;
    assert_eq!(
        download::<f32>(&out[0], &cx)?,
        host.iter().map(|v| v + 5.0).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn chained_ops_free_intermediates() -> anyhow::Result<()> {
    let cx = test_context();
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [256usize, 256])?;
    let mut t = x;
    for _ in 0..4 {
        t = add_scalar(t, 1.0)?;
    }
    // Two outputs from one chain, plus a dead branch.
    gb.output(t)?;
    let u = add_scalar(t, 0.5)?;
    add_scalar(x, 7.0)?;
    gb.output(u)?;
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;

    let host = host_iota(256 * 256, 1.0);
    let xt = upload(&host, [256, 256], &cx);
    let pool = cx
        .cuda()
        .create_device_memory_pool(DeviceMemoryPoolOptions::default())?;
    let stream = cx.default_compute_stream();
    let out = eg.execute(&mut [(&xt).into()], stream, &pool)?;
    stream.block_on()?;
    let bytes = 256 * 256 * 4;
    // At most two buffers are live, plus the first output during the dead branch.
    assert_eq!(pool.peak_allocated_bytes(), 3 * bytes);
    assert_eq!(pool.current_allocated_bytes(), 2 * bytes);
    let out: Vec<Tensor> = out.into_iter().map(|o| o.into_plain()).collect();
    assert_eq!(
        download::<f32>(&out[0], &cx)?,
        host.iter().map(|v| v + 4.0).collect::<Vec<_>>()
    );
    assert_eq!(
        download::<f32>(&out[1], &cx)?,
        host.iter().map(|v| v + 4.5).collect::<Vec<_>>()
    );
    drop(out);
    assert_eq!(pool.current_allocated_bytes(), 0);
    Ok(())
}

#[test]
fn autotune_modes() -> anyhow::Result<()> {
    let cx = test_context();
    let nonce = rand::random();
    let build = |mode| -> anyhow::Result<(ExecGraph, Arc<AtomicUsize>)> {
        let gb = GraphBuilder::new();
        let b = gb.dynamic_dim("b", 256, 0);
        let x = gb.input("x", F32, [Monomial::from(b), 512.into()])?;
        let op = AddScalar {
            nonce,
            ..AddScalar::new(x, 3.0)
        };
        let executions = op.executions.clone();
        gb.output(gb.add_op1(op)?)?;
        let eg = ExecGraph::build(
            gb.finish()?,
            &cx,
            BuildOptions {
                autotune: mode,
                pool: None,
            },
        )?;
        Ok((eg, executions))
    };
    let check = |eg: &ExecGraph| -> anyhow::Result<()> {
        let host = host_iota(100 * 512, 1.0);
        let xt = upload(&host, [100, 512], &cx);
        let out = run(eg, &mut [(&xt).into()], &cx)?;
        assert_eq!(
            download::<f32>(&out[0], &cx)?,
            host.iter().map(|v| v + 3.0).collect::<Vec<_>>()
        );
        Ok(())
    };

    for mode in [Autotune::Off, Autotune::CachedOnly] {
        let (eg, executions) = build(mode)?;
        assert_eq!(executions.load(Ordering::Relaxed), 0, "{mode:?}");
        check(&eg)?;
    }
    let (eg, executions) = build(Autotune::Full)?;
    assert!(executions.load(Ordering::Relaxed) > 0);
    check(&eg)?;
    // Cached now.
    for mode in [Autotune::CachedOnly, Autotune::Full] {
        let (eg, executions) = build(mode)?;
        assert_eq!(executions.load(Ordering::Relaxed), 0, "{mode:?}");
        check(&eg)?;
    }
    Ok(())
}

#[test]
fn hint_io_binds_operands() -> anyhow::Result<()> {
    struct HintCheck {
        x: SymTensorId,
    }
    impl Operator for HintCheck {
        fn name(&self) -> &str {
            "hint_check"
        }
        fn operands(&self) -> Vec<Operand> {
            vec![Operand::read_write(self.x)]
        }
        fn infer_outputs(&self, g: &GraphInfo) -> anyhow::Result<Vec<OutputSpec>> {
            Ok(vec![OutputSpec::new(
                TensorTy::BlockScaled {
                    format: MxFormat::Mxfp8,
                    dim: MxDim::Rowwise,
                },
                g.shape(self.x).clone(),
                Init::Uninit,
            )])
        }
        fn build(&self, bcx: &BuildCx<'_>) -> anyhow::Result<Box<dyn Executor>> {
            bcx.hint_io(100, |io| {
                let x = io.input_mut(0).plain();
                assert_eq!(x.shape(), Shape::new([64, 32]));
                assert_eq!(x.strides().dims_slice(), [1, 64]);
                let out = io.output(0).mx();
                assert_eq!(out.shape(), Shape::new([64, 32]));
                assert!(io.scratch().len() >= 100);
                assert_eq!(io.dims.iter().count(), 1);
                Ok(())
            })?;
            Ok(Box::new(DummyExecutor { scratch: None }))
        }
    }
    let cx = test_context();
    let gb = GraphBuilder::new();
    let n = gb.dynamic_dim("n", 32, 5);
    let x = gb.input_mut("x", F32, [Monomial::from(n), 64.into()])?;
    let out = gb.add_op1(HintCheck {
        x: x.transposed(0, 1)?.id(),
    })?;
    gb.output(out)?;
    ExecGraph::build(gb.finish()?, &cx, off())?;
    Ok(())
}

#[test]
fn init_modes_scratch_and_mx() -> anyhow::Result<()> {
    let cx = test_context();
    let gb = GraphBuilder::new();
    let rows = gb.dynamic_dim("rows", 128, 0);
    let x = gb.input("x", F32, [Monomial::from(rows), 8.into()])?;
    let m = gb.input_mx(
        "m",
        MxFormat::Mxfp8,
        MxDim::Rowwise,
        [Monomial::from(rows), 64.into()],
    )?;
    let e = err_string(m.transposed(0, 1));
    assert!(e.contains("cannot transpose a block-scaled tensor"), "{e}");
    let e = err_string(m.reshaped([64usize * 128]));
    assert!(e.contains("cannot reshape a block-scaled tensor"), "{e}");
    let shape = SymShape::new([Monomial::from(rows)]);
    let outs = gb.add_op(Dummy {
        operands: vec![Operand::read(x.id()), Operand::read(m.id())],
        outputs: vec![
            OutputSpec::new(F32, shape.clone(), Init::Uninit),
            OutputSpec::new(F32, shape.clone(), Init::Zero),
            OutputSpec::new(F32, shape.clone(), Init::Fill32(2.5f32.to_bits())),
            OutputSpec::new(
                TensorTy::BlockScaled {
                    format: MxFormat::Mxfp8,
                    dim: MxDim::Colwise,
                },
                SymShape::new([64.into(), Monomial::from(rows)]),
                Init::Uninit,
            ),
        ],
        fill_via_scratch: Some(7.0f32.to_bits()),
        fail_build: false,
    })?;
    for &out in &outs {
        gb.output(out)?;
    }
    let e = err_string(gb.add_op(Dummy {
        outputs: vec![OutputSpec::new(BF16, shape.clone(), Init::Fill32(0))],
        ..Default::default()
    }));
    assert!(e.contains("32-bit"), "{e}");
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;

    let stream = cx.default_compute_stream();
    let xt = upload(&host_iota(40 * 8, 1.0), [40, 8], &cx);
    let mt = MxTensor::new_zeroed([40, 64], MxFormat::Mxfp8, MxDim::Rowwise, stream, &cx)?;
    let mut out = eg
        .execute_default(&mut [(&xt).into(), (&mt).into()], stream)?
        .into_iter();
    let o0 = out.next().unwrap().into_plain();
    let o1 = out.next().unwrap().into_plain();
    let o2 = out.next().unwrap().into_plain();
    let o3 = out.next().unwrap().into_mx();
    assert_eq!(download::<f32>(&o0, &cx)?, vec![7.0; 40]);
    assert_eq!(download::<f32>(&o1, &cx)?, vec![0.0; 40]);
    assert_eq!(download::<f32>(&o2, &cx)?, vec![2.5; 40]);
    assert_eq!(o3.shape(), Shape::new([64, 40]));
    assert_eq!(o3.mx_dim(), MxDim::Colwise);
    let scales = common::download_bytes(o3.scale_factors(), &cx)?;
    assert!(scales.iter().all(|&b| b == 0));

    // MX input with the wrong format or shape.
    let bad = MxTensor::new_zeroed([40, 64], MxFormat::Nvfp4, MxDim::Rowwise, stream, &cx)?;
    let e = err_string(eg.execute_default(&mut [(&xt).into(), (&bad).into()], stream));
    assert!(e.contains("input m: expected BlockScaled"), "{e}");
    let bad = MxTensor::new_zeroed([41, 64], MxFormat::Mxfp8, MxDim::Rowwise, stream, &cx)?;
    let e = err_string(eg.execute_default(&mut [(&xt).into(), (&bad).into()], stream));
    assert!(e.contains("m shape[0]"), "{e}");

    // Empty block-scaled tensors are rejected before any work is enqueued.
    let x0 = Tensor::new_zeroed([0, 8], F32, stream, &cx)?;
    let m0 = MxTensor::new_zeroed([0, 64], MxFormat::Mxfp8, MxDim::Rowwise, stream, &cx)?;
    let e = err_string(eg.execute_default(&mut [(&x0).into(), (&m0).into()], stream));
    assert!(e.contains("block-scaled tensors must not be empty"), "{e}");
    Ok(())
}

#[test]
fn graph_validation_errors() -> anyhow::Result<()> {
    let cx = test_context();

    // Output aliasing an input (directly or through a view).
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [4usize, 4])?;
    gb.output(x.transposed(0, 1)?)?;
    let e = err_string(gb.finish());
    assert!(e.contains("aliases graph input x"), "{e}");

    // Two outputs on one buffer.
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [4usize, 4])?;
    let y = add_scalar(x, 1.0)?;
    gb.output(y)?;
    gb.output(y.transposed(0, 1)?)?;
    let e = err_string(gb.finish());
    assert!(e.contains("shares its buffer"), "{e}");

    // Output at a nonzero offset.
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [4usize, 4])?;
    gb.output(add_scalar(x, 1.0)?.sliced(0, 1, 2)?)?;
    let e = err_string(gb.finish());
    assert!(e.contains("offset zero"), "{e}");

    // An op reading and writing the same buffer.
    let gb = GraphBuilder::new();
    let x = gb.input_mut("x", F32, [4usize, 4])?;
    gb.add_op(Dummy {
        operands: vec![
            Operand::read_write(x.sliced(0, 0, 2)?.id()),
            Operand::read(x.sliced(0, 2, 2)?.id()),
        ],
        ..Default::default()
    })?;
    let e = err_string(gb.finish());
    assert!(e.contains("reads and writes the same buffer"), "{e}");

    let gb = GraphBuilder::new();
    let x = gb.input_mut("x", F32, [4usize, 4])?;
    gb.add_op(Dummy {
        operands: vec![Operand::read_write(x.id()), Operand::read_write(x.id())],
        ..Default::default()
    })?;
    let e = err_string(gb.finish());
    assert!(e.contains("multiple operands"), "{e}");

    // Reshapes that are not views.
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [4usize, 6])?;
    let e = err_string(x.transposed(0, 1)?.reshaped([24usize]));
    assert!(
        e.contains("without copying") && e.contains("strides [1, 6]"),
        "{e}"
    );
    let e = err_string(x.reshaped([5usize, 5]));
    assert!(e.contains("cannot reshape tensor with layout"), "{e}");
    let e = err_string(x.transposed(0, 2));
    assert!(e.contains("out of range for rank 2"), "{e}");
    let e = err_string(x.sliced(1, 4, 3));
    assert!(e.contains("slice 4..7 out of range"), "{e}");

    // A dimension that can't be inferred from inputs.
    let gb = GraphBuilder::new();
    let k = gb.dynamic_dim("k", 4, 0);
    gb.add_op(Dummy {
        outputs: vec![OutputSpec::new(F32, SymShape::new([k.into()]), Init::Zero)],
        ..Default::default()
    })?;
    let e = err_string(gb.finish());
    assert!(e.contains("dimension k cannot be inferred"), "{e}");

    // Duplicate input names.
    let gb = GraphBuilder::new();
    gb.input("x", F32, [4usize])?;
    let e = err_string(gb.input("x", F32, [4usize]));
    assert!(e.contains("duplicate graph input name x"), "{e}");

    // Dimensions only appearing in products are not inferable.
    let gb = GraphBuilder::new();
    let b = gb.dynamic_dim("b", 4, 0);
    let s = gb.dynamic_dim("s", 4, 0);
    gb.input("x", F32, [Monomial::new(1, [b, s]), 16.into()])?;
    let e = err_string(gb.finish());
    assert!(e.contains("cannot be inferred"), "{e}");
    // ... unless the other factor is bound elsewhere.
    let gb = GraphBuilder::new();
    let b = gb.dynamic_dim("b", 4, 0);
    let s = gb.dynamic_dim("s", 4, 0);
    gb.input("x", F32, [Monomial::new(1, [b, s]), 16.into()])?;
    gb.input("y", F32, [Monomial::from(b)])?;
    gb.finish()?;

    // Build errors name the op.
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [4usize, 4])?;
    for _ in 0..2 {
        gb.add_op(Dummy {
            operands: vec![Operand::read(x.id())],
            fail_build: true,
            ..Default::default()
        })?;
    }
    let e = err_string(ExecGraph::build(gb.finish()?, &cx, off()));
    assert!(
        e.contains("2 ops failed") && e.contains("op 1 (dummy)") && e.contains("requested failure"),
        "{e}"
    );
    // A single build error keeps its chain.
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [4usize, 4])?;
    gb.add_op(Dummy {
        operands: vec![Operand::read(x.id())],
        fail_build: true,
        ..Default::default()
    })?;
    let e = ExecGraph::build(gb.finish()?, &cx, off()).unwrap_err();
    assert_eq!(e.to_string(), "failed to build op 0 (dummy)");
    assert_eq!(e.root_cause().to_string(), "requested failure");

    // add_op1 with the wrong output count records nothing.
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [4usize, 64])?;
    let e = err_string(gb.add_op1(Dummy::default()));
    assert!(e.contains("expected 1 output(s), got 0"), "{e}");
    gb.output(add_scalar(x, 1.0)?)?;
    let graph = gb.finish()?;
    assert_eq!(graph.num_ops(), 1);
    ExecGraph::build(graph, &cx, off())?;

    // A failed add_op leaves the builder usable.
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [4usize, 64])?;
    let e = err_string(gb.add_op(Dummy {
        outputs: vec![
            OutputSpec::new(F32, SymShape::new([4.into()]), Init::Zero),
            OutputSpec::new(BF16, SymShape::new([4.into()]), Init::Fill32(0)),
        ],
        ..Default::default()
    }));
    assert!(e.contains("dummy: output 1") && e.contains("32-bit"), "{e}");
    gb.output(add_scalar(x, 1.0)?)?;
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;
    let xt = upload(&host_iota(4 * 64, 1.0), [4, 64], &cx);
    let out = run(&eg, &mut [(&xt).into()], &cx)?;
    assert_eq!(
        download::<f32>(&out[0], &cx)?,
        host_iota(4 * 64, 1.0)
            .iter()
            .map(|v| v + 1.0)
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn execute_on_custom_pool_and_stream() -> anyhow::Result<()> {
    let cx = test_context();
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [32usize, 64])?;
    gb.output(add_scalar(x, -1.0)?)?;
    let pool: DeviceMemoryPool = cx
        .cuda()
        .create_device_memory_pool(DeviceMemoryPoolOptions::default())?;
    let eg = ExecGraph::build(
        gb.finish()?,
        &cx,
        BuildOptions {
            autotune: Autotune::Off,
            pool: Some(pool.clone()),
        },
    )?;
    let stream = cx.cuda().create_stream()?;
    let host = host_iota(32 * 64, 1.0);
    let xt = upload(&host, [32, 64], &cx);
    let out = eg.execute(&mut [(&xt).into()], &stream, &pool)?;
    stream.block_on()?;
    assert_eq!(
        download::<f32>(&out.into_iter().next().unwrap().into_plain(), &cx)?,
        host.iter().map(|v| v - 1.0).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn diamond_liveness() -> anyhow::Result<()> {
    let cx = test_context();
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [1usize])?;
    let spec = |n: usize| OutputSpec::new(F32, SymShape::new([n.into()]), Init::Zero);
    let dummy = |operands: Vec<SymTensor<'_>>, n: usize| Dummy {
        operands: operands
            .into_iter()
            .map(|t| Operand::read(t.id()))
            .collect(),
        outputs: vec![spec(n)],
        ..Default::default()
    };
    let (sa, sb, sc, sd) = (1usize << 16, 1 << 10, 1 << 12, 1 << 14);
    let a = gb.add_op1(dummy(vec![x], sa))?;
    let b = gb.add_op1(dummy(vec![a], sb))?;
    let c = gb.add_op1(dummy(vec![b], sc))?;
    // `a` must stay alive until this late consumer.
    let d = gb.add_op1(dummy(vec![a, c], sd))?;
    gb.output(d)?;
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;

    let pool = cx
        .cuda()
        .create_device_memory_pool(DeviceMemoryPoolOptions::default())?;
    let stream = cx.default_compute_stream();
    let xt = upload(&[0.0], [1], &cx);
    let out = eg.execute(&mut [(&xt).into()], stream, &pool)?;
    stream.block_on()?;
    let peak = 4 * (sa + sb + sc).max(sa + sc + sd);
    assert_eq!(pool.peak_allocated_bytes(), peak);
    assert_eq!(pool.current_allocated_bytes(), 4 * sd);
    drop(out);
    Ok(())
}

#[test]
fn stride_only_dims_and_atomic_validation() -> anyhow::Result<()> {
    let cx = test_context();
    let stream = cx.default_compute_stream();
    let gb = GraphBuilder::new();
    let rows = gb.dynamic_dim("rows", 16, 0);
    let ld = gb.dynamic_dim("ld", 64, 0);
    let layout = SymLayout::new(
        SymShape::new([Monomial::from(rows), 32.into()]),
        vec![ld.into(), 1.into()],
        Default::default(),
    );
    let x = gb.input_with_layout("x", F32, layout)?;
    gb.output(add_scalar(x, 1.0)?)?;
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;
    // With one row, `ld` has no equation; it doesn't affect addressing.
    let host = host_iota(32, 1.0);
    let xt = upload(&host, [1, 32], &cx);
    let out = run(&eg, &mut [(&xt).into()], &cx)?;
    assert_eq!(
        download::<f32>(&out[0], &cx)?,
        host.iter().map(|v| v + 1.0).collect::<Vec<_>>()
    );

    // Op 1's operand fails validation; op 0 must not have modified `x`.
    let gb = GraphBuilder::new();
    let rows = gb.dynamic_dim("rows", 128, 0);
    let x = gb.input_mut("x", F32, [16usize, 64])?;
    let m = gb.input_mx(
        "m",
        MxFormat::Mxfp8,
        MxDim::Rowwise,
        [Monomial::from(rows), 64.into()],
    )?;
    gb.add_op(AddScalar {
        in_place: true,
        ..AddScalar::new(x, 4.0)
    })?;
    gb.add_op(Dummy {
        operands: vec![Operand::read(m.id())],
        ..Default::default()
    })?;
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;
    let host = host_iota(16 * 64, 1.0);
    let mut xt = upload(&host, [16, 64], &cx);
    let m0 = MxTensor::new_zeroed([0, 64], MxFormat::Mxfp8, MxDim::Rowwise, stream, &cx)?;
    let e = err_string(eg.execute_default(&mut [(&mut xt).into(), (&m0).into()], stream));
    assert!(
        e.contains("op 1 (dummy)") && e.contains("must not be empty"),
        "{e}"
    );
    assert_eq!(download::<f32>(&xt, &cx)?, host);
    Ok(())
}

#[test]
fn misaligned_input_storage() -> anyhow::Result<()> {
    let cx = test_context();
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [4usize, 2])?;
    gb.output(add_scalar(x, 1.0)?)?;
    let eg = ExecGraph::build(gb.finish()?, &cx, off())?;
    let host = host_iota(6 * 2, 1.0);
    let full = upload(&host, [6, 2], &cx);
    // One row in is 8 bytes: misaligned.
    let e = err_string(run(&eg, &mut [full.sliced(0, 1..5).into()], &cx));
    assert!(
        e.contains("input x") && e.contains("16-byte aligned"),
        "{e}"
    );
    let out = run(&eg, &mut [full.sliced(0, 2..6).into()], &cx)?;
    assert_eq!(
        download::<f32>(&out[0], &cx)?,
        host[4..].iter().map(|v| v + 1.0).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn hint_io_rejects_gpu_lock_reentry() -> anyhow::Result<()> {
    struct Reenter {
        x: SymTensorId,
        nonce: u64,
        seen: Arc<parking_lot::Mutex<Vec<String>>>,
    }
    impl Operator for Reenter {
        fn name(&self) -> &str {
            "reenter"
        }
        fn operands(&self) -> Vec<Operand> {
            vec![Operand::read(self.x)]
        }
        fn infer_outputs(&self, _: &GraphInfo) -> anyhow::Result<Vec<OutputSpec>> {
            Ok(vec![])
        }
        fn build(&self, bcx: &BuildCx<'_>) -> anyhow::Result<Box<dyn Executor>> {
            let key = bcx.tune_key(&self.nonce)?;
            let _ = bcx.tuner.tune(
                &key,
                vec![1u32, 2],
                |&c| Ok(c),
                |_, _| {
                    if !self.seen.lock().is_empty() {
                        return Ok(());
                    }
                    let e = bcx.hint_io(0, |_| Ok(())).unwrap_err();
                    self.seen.lock().push(e.to_string());
                    let nested = bcx.tune_key(&(self.nonce, "nested"))?;
                    let e = bcx
                        .tuner
                        .tune(&nested, vec![1u32, 2], |&c| Ok(c), |_, _| Ok(()))
                        .unwrap_err();
                    self.seen.lock().push(e.to_string());
                    Ok(())
                },
            );
            // Outside the lock, hint_io works.
            bcx.hint_io(0, |io| {
                assert_eq!(io.input_ro(0).plain().shape(), Shape::new([4, 4]));
                Ok(())
            })?;
            Ok(Box::new(DummyExecutor { scratch: None }))
        }
    }
    let cx = test_context();
    let gb = GraphBuilder::new();
    let x = gb.input("x", F32, [4usize, 4])?;
    let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
    gb.add_op(Reenter {
        x: x.id(),
        nonce: rand::random(),
        seen: seen.clone(),
    })?;
    ExecGraph::build(
        gb.finish()?,
        &cx,
        BuildOptions {
            autotune: Autotune::Full,
            pool: None,
        },
    )?;
    let seen = seen.lock();
    assert!(!seen.is_empty());
    assert!(seen.iter().all(|e| e.contains("GPU lock")), "{seen:?}");
    assert!(seen.iter().any(|e| e.contains("nested tuning")), "{seen:?}");
    assert!(!cx.holds_gpu_lock());
    Ok(())
}
