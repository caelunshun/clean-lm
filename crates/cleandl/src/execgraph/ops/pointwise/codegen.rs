//! Tile IR code generation for lowered pointwise kernels.

use super::{
    ir::{BinaryOp, Node, PwIr, PwOutput, Reduce, UnaryOp, ValueTy},
    lower::{Facts, OutputFacts, OutputPlan, Plan},
    mx,
};
use crate::{
    execgraph::{
        Bound, TensorRef, TensorRefMut,
        sym::{DimValues, Monomial},
    },
    mx_tensor::{MxDim, MxFormat},
};
use anyhow::{bail, ensure};
use kernel_build_util::{
    ArgBundle, Param, PtrArg,
    tileir::{Args, ArrayArg, KernelBuilder, KernelSpec},
};
use serde::Serialize;
use skycuda::device_memory::{DeviceVisibleMemory, DeviceVisibleMemoryMut};
use skytens_core::data_type::{ElementType, FloatType, SIntType, UIntType};
use tileir_builder::{
    Array, ArrayOpts, AtomicOpts, AtomicRmwMode, Bool, DType, DynArray, DynTile, F4E2M1, F8E4M3,
    F8E5M2, KernelOpts, ModuleBuilder, Ptr, RoundingMode, Tile, bf16, bid, f16, iota, num_blocks,
    select,
};

pub const KERNEL_NAME: &str = "pointwise";

/// Identifies a compiled kernel.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct KernelKey<'a> {
    pub ir: &'a PwIr,
    pub facts: &'a Facts,
    pub tile: &'a [u32],
}

pub fn dtype_of(ty: ElementType) -> DType {
    match ty {
        ElementType::Float(f) => match f {
            FloatType::F64 => DType::F64,
            FloatType::F32 => DType::F32,
            FloatType::Bf16 => DType::BF16,
            FloatType::F16 => DType::F16,
            FloatType::F8E4M3 => DType::F8E4M3FN,
            FloatType::F8E5M2 => DType::F8E5M2,
            FloatType::UF8E8M0 => DType::F8E8M0FNU,
            FloatType::F4E2M1 => DType::F4E2M1FN,
        },
        ElementType::SInt(s) => DType::int(s.bits_per_element() as u8),
        ElementType::UInt(u) => DType::uint(u.bits_per_element() as u8),
        ElementType::Bool => DType::int(1),
    }
}

macro_rules! array_params {
    ($($variant:ident => $ty:ty, $et:pat),* $(,)?) => {
        /// An array parameter of a run-time element type.
        #[derive(Clone, Copy, Debug)]
        pub enum ArrayParam {
            $($variant(Param<Array<$ty>>),)*
        }

        impl ArrayParam {
            #[track_caller]
            fn declare(kb: &mut KernelBuilder, name: String, ty: ElementType, opts: ArrayOpts) -> Self {
                match ty {
                    $($et => Self::$variant(kb.array::<$ty>(name, opts)),)*
                    _ => panic!("unsupported array element type {ty:?}"),
                }
            }

            fn get(self, args: &Args) -> DynArray {
                match self {
                    $(Self::$variant(p) => args.get_array(p),)*
                }
            }

            fn bind<'a, M: DeviceVisibleMemory + 'a>(
                self,
                args: &mut ArgBundle<'a>,
                memory: &'a M,
                shape: &[i64],
                strides: &[i64],
            ) {
                match self {
                    $(Self::$variant(p) => {
                        args.set(p, ArrayArg::new(PtrArg::new(memory), shape, strides))
                    })*
                }
            }

            fn bind_mut<'a, M: DeviceVisibleMemoryMut + 'a>(
                self,
                args: &mut ArgBundle<'a>,
                memory: &'a mut M,
                shape: &[i64],
                strides: &[i64],
            ) {
                match self {
                    $(Self::$variant(p) => {
                        args.set(p, ArrayArg::new(PtrArg::new_mut(memory), shape, strides))
                    })*
                }
            }
        }
    };
}

array_params!(
    F64 => f64, ElementType::Float(FloatType::F64),
    F32 => f32, ElementType::Float(FloatType::F32),
    Bf16 => bf16, ElementType::Float(FloatType::Bf16),
    F16 => f16, ElementType::Float(FloatType::F16),
    F8E4M3 => F8E4M3, ElementType::Float(FloatType::F8E4M3),
    F8E5M2 => F8E5M2, ElementType::Float(FloatType::F8E5M2),
    F4E2M1 => F4E2M1, ElementType::Float(FloatType::F4E2M1),
    I8 => i8, ElementType::SInt(SIntType::S8),
    I16 => i16, ElementType::SInt(SIntType::S16),
    I32 => i32, ElementType::SInt(SIntType::S32),
    I64 => i64, ElementType::SInt(SIntType::S64),
    U8 => u8, ElementType::UInt(UIntType::U8),
    U16 => u16, ElementType::UInt(UIntType::U16),
    U32 => u32, ElementType::UInt(UIntType::U32),
    U64 => u64, ElementType::UInt(UIntType::U64),
);

#[derive(Clone, Copy, Debug)]
enum ReducePtr {
    F32(Param<Ptr<f32>>),
    /// Max and min reduce on the bits of the `f32` values.
    I32(Param<Ptr<i32>>),
}

#[derive(Clone, Copy, Debug)]
enum ScalePtr {
    /// UE8M0 scales, as bytes.
    Ue8m0(Param<Ptr<u8>>),
    E4m3(Param<Ptr<F8E4M3>>),
}

#[derive(Clone, Debug)]
enum OutputParams {
    Elementwise(ArrayParam),
    Reduce {
        ptr: ReducePtr,
        strides: Vec<Param<i64>>,
    },
    Mx {
        values: ArrayParam,
        scales: ScalePtr,
        row_strides: Vec<Param<i64>>,
        num_blocks: Param<i64>,
    },
}

/// Kernel parameters; declared identically when defining and when launching.
#[derive(Clone, Debug)]
pub struct Params {
    sizes: Vec<Param<i64>>,
    inputs: Vec<Option<ArrayParam>>,
    outputs: Vec<OutputParams>,
}

fn array_opts(rank: usize, unit_stride: &[bool]) -> ArrayOpts {
    let mut opts = ArrayOpts::rank(rank.max(1)).index_i64();
    if rank == 0 {
        opts = opts.static_dim(0, 1).static_stride(0, 1);
    }
    for (axis, &unit) in unit_stride.iter().enumerate() {
        if unit {
            opts = opts.static_stride(axis, 1);
        }
    }
    opts
}

impl Params {
    pub fn declare(kb: &mut KernelBuilder, facts: &Facts) -> Self {
        let sizes = (0..facts.rank)
            .map(|a| kb.param::<i64>(format!("size{a}")))
            .collect();
        let inputs = facts
            .inputs
            .iter()
            .enumerate()
            .map(|(k, input)| {
                input.live.then(|| {
                    ArrayParam::declare(
                        kb,
                        format!("in{k}"),
                        input.ty,
                        array_opts(input.axes.len(), &input.unit_stride),
                    )
                })
            })
            .collect();
        let outputs = facts
            .outputs
            .iter()
            .enumerate()
            .map(|(i, output)| match output {
                OutputFacts::Elementwise { ty, unit_stride } => {
                    OutputParams::Elementwise(ArrayParam::declare(
                        kb,
                        format!("out{i}"),
                        *ty,
                        array_opts(facts.rank, unit_stride),
                    ))
                }
                OutputFacts::Reduce { op, kept } => OutputParams::Reduce {
                    ptr: match op {
                        Reduce::Sum => ReducePtr::F32(kb.param(format!("out{i}"))),
                        Reduce::Max | Reduce::Min => ReducePtr::I32(kb.param(format!("out{i}"))),
                    },
                    strides: (0..kept.len())
                        .map(|j| kb.param(format!("out{i}_stride{j}")))
                        .collect(),
                },
                OutputFacts::Mx {
                    format,
                    unit_stride,
                    ..
                } => OutputParams::Mx {
                    values: ArrayParam::declare(
                        kb,
                        format!("out{i}_values"),
                        format.values_type(),
                        array_opts(facts.rank, unit_stride),
                    ),
                    scales: match format {
                        MxFormat::Mxfp8 => ScalePtr::Ue8m0(kb.param(format!("out{i}_scales"))),
                        MxFormat::Nvfp4 => ScalePtr::E4m3(kb.param(format!("out{i}_scales"))),
                    },
                    row_strides: (0..facts.rank - 1)
                        .map(|j| kb.param(format!("out{i}_row_stride{j}")))
                        .collect(),
                    num_blocks: kb.param(format!("out{i}_num_blocks")),
                },
            })
            .collect();
        Self {
            sizes,
            inputs,
            outputs,
        }
    }

    /// Binds the operands and outputs of one execution. `sizes` are the evaluated
    /// lowered axis sizes.
    #[allow(clippy::too_many_arguments)]
    pub fn bind<'a>(
        &self,
        args: &mut ArgBundle<'a>,
        facts: &Facts,
        plan: &Plan,
        sizes: &[i64],
        inputs: &'a [Bound<'_>],
        outputs: &'a mut [TensorRefMut<'_>],
        dims: &DimValues,
    ) -> anyhow::Result<()> {
        let eval = |strides: &[Monomial]| -> anyhow::Result<Vec<i64>> {
            strides
                .iter()
                .map(|s| Ok(i64::try_from(s.eval(dims)?)?))
                .collect()
        };
        for (&p, &size) in self.sizes.iter().zip(sizes) {
            args.set(p, size);
        }

        ensure!(
            inputs.len() == self.inputs.len() && outputs.len() == self.outputs.len(),
            "pointwise: operand or output count mismatch"
        );
        for (k, (input, param)) in inputs.iter().zip(&self.inputs).enumerate() {
            let Some(param) = param else { continue };
            let axes = &facts.inputs[k].axes;
            let (shape, strides) = if axes.is_empty() {
                (vec![1], vec![1])
            } else {
                (
                    axes.iter().map(|&a| sizes[a]).collect(),
                    eval(&plan.inputs[k])?,
                )
            };
            match input {
                Bound::Ro(TensorRef::Plain(view)) => {
                    param.bind(args, view.storage(), &shape, &strides)
                }
                Bound::Rw(TensorRefMut::Plain(view)) => {
                    param.bind(args, view.storage(), &shape, &strides)
                }
                _ => bail!("pointwise operand {k} must be a plain tensor"),
            }
        }

        for (i, ((output, params), plan)) in outputs
            .iter_mut()
            .zip(&self.outputs)
            .zip(&plan.outputs)
            .enumerate()
        {
            match (params, plan, output) {
                (
                    OutputParams::Elementwise(param),
                    OutputPlan::Elementwise { strides },
                    TensorRefMut::Plain(view),
                ) => param.bind_mut(args, view.storage_mut(), sizes, &eval(strides)?),
                (
                    OutputParams::Reduce {
                        ptr,
                        strides: stride_params,
                    },
                    OutputPlan::Reduce { strides },
                    TensorRefMut::Plain(view),
                ) => {
                    match *ptr {
                        ReducePtr::F32(p) => args.set(p, PtrArg::new_mut(view.storage_mut())),
                        ReducePtr::I32(p) => args.set(p, PtrArg::new_mut(view.storage_mut())),
                    }
                    for (&p, stride) in stride_params.iter().zip(eval(strides)?) {
                        args.set(p, stride);
                    }
                }
                (
                    OutputParams::Mx {
                        values,
                        scales,
                        row_strides: row_stride_params,
                        num_blocks: num_blocks_param,
                    },
                    OutputPlan::Mx {
                        values_strides,
                        row_strides,
                        block_dim,
                    },
                    TensorRefMut::Mx(mx),
                ) => {
                    let block_size = mx.format().block_size();
                    let (values_view, scales_view) = mx.values_and_scale_factors_mut();
                    values.bind_mut(
                        args,
                        values_view.storage_mut(),
                        sizes,
                        &eval(values_strides)?,
                    );
                    match *scales {
                        ScalePtr::Ue8m0(p) => {
                            args.set(p, PtrArg::new_mut(scales_view.storage_mut()))
                        }
                        ScalePtr::E4m3(p) => {
                            args.set(p, PtrArg::new_mut(scales_view.storage_mut()))
                        }
                    }
                    for (&p, stride) in row_stride_params.iter().zip(eval(row_strides)?) {
                        args.set(p, stride);
                    }
                    let num_blocks = block_dim.eval(dims)? / block_size;
                    args.set(*num_blocks_param, i64::try_from(num_blocks)?);
                }
                _ => bail!("pointwise output {i} has an unexpected tensor kind"),
            }
        }
        Ok(())
    }
}

/// A traced value.
#[derive(Clone, Copy)]
enum Val {
    F(Tile<f32>),
    I(Tile<i32>),
    B(Tile<bool>),
}

impl Val {
    fn f32(self) -> Tile<f32> {
        match self {
            Val::F(v) => v,
            Val::I(v) => v.astype(RoundingMode::NearestEven),
            Val::B(v) => select(v, 1.0f32, 0.0f32),
        }
    }

    fn i32(self) -> Tile<i32> {
        match self {
            Val::F(v) => v.astype(RoundingMode::Zero),
            Val::I(v) => v,
            Val::B(v) => select(v, 1i32, 0i32),
        }
    }

    fn bool(self) -> Bool {
        match self {
            Val::F(v) => v.ne(0.0f32),
            Val::I(v) => v.ne(0i32),
            Val::B(v) => v,
        }
    }

    fn cast(self, ty: ValueTy) -> Val {
        match ty {
            ValueTy::F32 => Val::F(self.f32()),
            ValueTy::I32 => Val::I(self.i32()),
            ValueTy::Bool => Val::B(self.bool()),
        }
    }

    fn reshape(self, shape: Vec<u32>) -> Val {
        match self {
            Val::F(v) => Val::F(v.reshape(shape)),
            Val::I(v) => Val::I(v.reshape(shape)),
            Val::B(v) => Val::B(v.reshape(shape)),
        }
    }
}

/// `tile` of shape `[n]` as axis `axis` of a rank-`rank` tile.
fn along<T: tileir_builder::Elem>(tile: Tile<T>, axis: usize, rank: usize) -> Tile<T> {
    let mut dims = vec![1u32; rank];
    dims[axis] = tile.shape().dims()[0];
    tile.reshape(dims)
}

fn and_all(masks: impl IntoIterator<Item = Bool>) -> Option<Bool> {
    masks.into_iter().reduce(|a, b| a & b)
}

fn load(tile: DynTile, ty: ElementType) -> Val {
    match ty {
        ElementType::Float(_) => Val::F(
            tile.astype(&DType::F32, RoundingMode::NearestEven)
                .downcast(),
        ),
        _ => Val::I(
            tile.astype(&DType::int(32), RoundingMode::NearestEven)
                .downcast(),
        ),
    }
}

fn store_value(v: Val, ty: ElementType) -> DynTile {
    let rounding = RoundingMode::NearestEven;
    match ty {
        ElementType::Float(f) => {
            let v = v.f32();
            let v = match f {
                FloatType::F8E4M3 => mx::clamp(v, mx::E4M3_MAX),
                FloatType::F8E5M2 => mx::clamp(v, mx::E5M2_MAX),
                _ => v,
            };
            v.erase().astype(&dtype_of(ty), rounding)
        }
        _ => v.i32().erase().astype(&dtype_of(ty), rounding),
    }
}

fn unary(op: UnaryOp, a: Val) -> Val {
    match (op, a) {
        (UnaryOp::Neg, Val::F(a)) => Val::F(-a),
        (UnaryOp::Neg, Val::I(a)) => Val::I(-a),
        (UnaryOp::Abs, Val::F(a)) => Val::F(a.abs()),
        (UnaryOp::Abs, Val::I(a)) => Val::I(a.abs()),
        (UnaryOp::Not, Val::B(a)) => Val::B(!a),
        (UnaryOp::Exp, Val::F(a)) => Val::F(a.exp()),
        (UnaryOp::Log, Val::F(a)) => Val::F(a.log()),
        (UnaryOp::Sqrt, Val::F(a)) => Val::F(a.sqrt()),
        (UnaryOp::Rsqrt, Val::F(a)) => Val::F(a.rsqrt()),
        (UnaryOp::Tanh, Val::F(a)) => Val::F(a.tanh()),
        (UnaryOp::Sigmoid, Val::F(a)) => Val::F(Tile::<f32>::scalar(1.0) / ((-a).exp() + 1.0f32)),
        (UnaryOp::Floor, Val::F(a)) => Val::F(a.floor()),
        _ => unreachable!("ill-typed unary op {op:?}"),
    }
}

fn binary(op: BinaryOp, a: Val, b: Val) -> Val {
    use BinaryOp::*;
    match (a, b) {
        (Val::F(a), Val::F(b)) => match op {
            Add => Val::F(a + b),
            Sub => Val::F(a - b),
            Mul => Val::F(a * b),
            Div => Val::F(a / b),
            Min => Val::F(a.minimum(b)),
            Max => Val::F(a.maximum(b)),
            Pow => Val::F(a.pow(b)),
            Lt => Val::B(a.lt(b)),
            Le => Val::B(a.le(b)),
            Gt => Val::B(a.gt(b)),
            Ge => Val::B(a.ge(b)),
            Eq => Val::B(a.eq(b)),
            Ne => Val::B(a.ne(b)),
            Rem | And | Or => unreachable!("ill-typed binary op {op:?}"),
        },
        (Val::I(a), Val::I(b)) => match op {
            Add => Val::I(a + b),
            Sub => Val::I(a - b),
            Mul => Val::I(a * b),
            Div => Val::I(a / b),
            Rem => Val::I(a % b),
            Min => Val::I(a.minimum(b)),
            Max => Val::I(a.maximum(b)),
            Lt => Val::B(a.lt(b)),
            Le => Val::B(a.le(b)),
            Gt => Val::B(a.gt(b)),
            Ge => Val::B(a.ge(b)),
            Eq => Val::B(a.eq(b)),
            Ne => Val::B(a.ne(b)),
            Pow | And | Or => unreachable!("ill-typed binary op {op:?}"),
        },
        (Val::B(a), Val::B(b)) => match op {
            And => Val::B(a & b),
            Or => Val::B(a | b),
            Eq => Val::B(a.eq(b)),
            Ne => Val::B(a.ne(b)),
            _ => unreachable!("ill-typed binary op {op:?}"),
        },
        _ => unreachable!("binary op {op:?} on values of different types"),
    }
}

/// Defines the kernel for `key` in `module`.
pub fn define(module: &mut ModuleBuilder, key: &KernelKey<'_>) -> KernelSpec {
    let mut kb = KernelBuilder::new();
    let params = Params::declare(&mut kb, key.facts);
    kb.define(module, KERNEL_NAME, KernelOpts::new(), |args| {
        body(args, &params, key)
    })
}

fn body(args: &Args, params: &Params, key: &KernelKey<'_>) {
    let KernelKey { ir, facts, tile } = *key;
    let rank = facts.rank;
    let full: Vec<u32> = tile.to_vec();

    let sizes: Vec<Tile<i64>> = params.sizes.iter().map(|&p| args.get(p)).collect();
    let num_tiles: Vec<Tile<i64>> = sizes
        .iter()
        .zip(tile)
        .map(|(&s, &t)| (s + (i64::from(t) - 1)) / i64::from(t))
        .collect();
    let mut linear = bid(0).astype::<i64>(RoundingMode::NearestEven)
        + bid(1).astype::<i64>(RoundingMode::NearestEven)
            * num_blocks(0).astype::<i64>(RoundingMode::NearestEven);
    let mut tile_index = vec![Tile::<i64>::scalar(0); rank];
    for a in (1..rank).rev() {
        tile_index[a] = linear % num_tiles[a];
        linear = linear / num_tiles[a];
    }
    tile_index[0] = linear;

    let offsets: Vec<Tile<i64>> = (0..rank)
        .map(|a| tile_index[a] * i64::from(tile[a]))
        .collect();
    // Per axis, the `[tile]` coordinates and whether they are in bounds.
    let coords: Vec<Tile<i64>> = (0..rank)
        .map(|a| iota::<i64>(tile[a]) + offsets[a])
        .collect();
    let in_bounds: Vec<Bool> = (0..rank).map(|a| coords[a].lt(sizes[a])).collect();
    let full_mask = || {
        and_all((0..rank).map(|a| along(in_bounds[a], a, rank)))
            .unwrap()
            .broadcast_to(full.clone())
    };

    let live = ir.live_nodes();
    let mut vals: Vec<Option<Val>> = vec![None; ir.nodes.len()];
    for (i, node) in ir.nodes.iter().enumerate() {
        if !live[i] {
            continue;
        }
        let v = |id: super::ir::NodeId| vals[id.index()].unwrap();
        let value = match *node {
            Node::Input(k) => {
                let input = &facts.inputs[k];
                let array = params.inputs[k].unwrap().get(args);
                let loaded = if input.axes.is_empty() {
                    array.load(vec![Tile::<i64>::scalar(0)], [1])
                } else {
                    let index: Vec<Tile<i64>> = input.axes.iter().map(|&a| tile_index[a]).collect();
                    let shape: Vec<u32> = input.axes.iter().map(|&a| tile[a]).collect();
                    array.load(index, shape)
                };
                let dims: Vec<u32> = (0..rank)
                    .map(|a| if input.axes.contains(&a) { tile[a] } else { 1 })
                    .collect();
                load(loaded, input.ty).reshape(dims)
            }
            Node::Index(axis) => {
                let lowered = facts
                    .index_axes
                    .iter()
                    .find(|(a, _)| *a == axis)
                    .and_then(|(_, l)| *l);
                match lowered {
                    Some(a) => Val::I(along(
                        coords[a].astype::<i32>(RoundingMode::NearestEven),
                        a,
                        rank,
                    )),
                    None => Val::I(Tile::scalar(0)),
                }
            }
            Node::ConstF32(bits) => Val::F(Tile::scalar(f32::from_bits(bits))),
            Node::ConstI32(x) => Val::I(Tile::scalar(x)),
            Node::ConstBool(x) => Val::B(Tile::scalar(x)),
            Node::Unary(op, a) => unary(op, v(a)),
            Node::Binary(op, a, b) => binary(op, v(a), v(b)),
            Node::Fma(a, b, c) => Val::F(v(a).f32().fma(v(b).f32(), v(c).f32())),
            Node::Select(c, a, b) => {
                let c = v(c).bool();
                match (v(a), v(b)) {
                    (Val::F(a), Val::F(b)) => Val::F(select(c, a, b)),
                    (Val::I(a), Val::I(b)) => Val::I(select(c, a, b)),
                    (Val::B(a), Val::B(b)) => Val::B(select(c, a, b)),
                    _ => unreachable!("select on values of different types"),
                }
            }
            Node::Cast(ty, a) => v(a).cast(ty),
        };
        vals[i] = Some(value);
    }

    for ((output, output_facts), output_params) in
        ir.outputs.iter().zip(&facts.outputs).zip(&params.outputs)
    {
        let value = vals[output.value().index()].unwrap();
        match (output, output_facts, output_params) {
            (
                PwOutput::Elementwise { ty, .. },
                OutputFacts::Elementwise { .. },
                OutputParams::Elementwise(array),
            ) => {
                let v = match value {
                    Val::F(v) => Val::F(v.broadcast_to(full.clone())),
                    Val::I(v) => Val::I(v.broadcast_to(full.clone())),
                    Val::B(v) => Val::B(v.broadcast_to(full.clone())),
                };
                array
                    .get(args)
                    .store(tile_index.clone(), store_value(v, *ty));
            }
            (
                PwOutput::Reduce { .. },
                OutputFacts::Reduce { op, kept },
                OutputParams::Reduce { ptr, strides },
            ) => {
                let v = tileir_builder::where_(
                    full_mask(),
                    value.f32().broadcast_to(full.clone()),
                    op.identity(),
                );
                let mut r = v;
                for a in (0..rank).rev().filter(|a| !kept.contains(a)) {
                    r = match op {
                        Reduce::Sum => r.sum(a),
                        Reduce::Max => r.max(a),
                        Reduce::Min => r.min(a),
                    };
                }
                let k = kept.len();
                let offset = (0..k)
                    .map(|j| along(coords[kept[j]], j, k) * args.get(strides[j]))
                    .reduce(|a, b| a + b)
                    .unwrap_or_else(|| Tile::scalar(0i64));
                let mask = and_all((0..k).map(|j| along(in_bounds[kept[j]], j, k)));
                let opts = AtomicOpts::default();
                match (op, *ptr) {
                    (Reduce::Sum, ReducePtr::F32(p)) => {
                        args.get(p).offset(offset).atomic_add(r, mask, opts);
                    }
                    (Reduce::Max | Reduce::Min, ReducePtr::I32(p)) => {
                        // For non-negative floats, the order of the bits as signed integers
                        // matches; for negative floats it is reversed as unsigned integers.
                        let bits = r.bitcast::<i32>();
                        let negative = bits.lt(0i32);
                        let (pos_mode, neg_mode) = match op {
                            Reduce::Max => (AtomicRmwMode::Max, AtomicRmwMode::UMin),
                            _ => (AtomicRmwMode::Min, AtomicRmwMode::UMax),
                        };
                        let and_mask = |m: Bool| Some(mask.map_or(m, |mask| mask & m));
                        let ptrs = args.get(p).offset(offset);
                        ptrs.atomic_rmw(pos_mode, bits, and_mask(!negative), opts);
                        ptrs.atomic_rmw(neg_mode, bits, and_mask(negative), opts);
                    }
                    _ => unreachable!(),
                }
            }
            (
                PwOutput::Mx { .. },
                OutputFacts::Mx {
                    format,
                    dim,
                    block_axis,
                    ..
                },
                OutputParams::Mx {
                    values,
                    scales,
                    row_strides,
                    num_blocks,
                },
            ) => {
                let b = *block_axis;
                let bs = format.block_size() as u32;
                let v = value.f32().broadcast_to(full.clone());
                let mut blocked = full.clone();
                blocked[b] /= bs;
                blocked.insert(b + 1, bs);
                let magnitude =
                    tileir_builder::where_(full_mask(), v.abs(), 0.0f32).reshape(blocked.clone());
                let amax = magnitude.max(b + 1);
                let scale_shape = amax.shape().dims().to_vec();
                let expand = |descale: Tile<f32>| {
                    descale
                        .expand_dims(b + 1)
                        .broadcast_to(blocked.clone())
                        .reshape(full.clone())
                };

                // Coordinates of each block (its first element) in the scale-shaped tile.
                let block_start = iota::<i64>(scale_shape[b]) * i64::from(bs) + offsets[b];
                let coord = |a: usize| {
                    if a == b {
                        along(block_start, a, rank)
                    } else {
                        along(coords[a], a, rank)
                    }
                };
                let flat_row = (0..rank - 1)
                    .map(|a| coord(a) * args.get(row_strides[a]))
                    .reduce(|x, y| x + y)
                    .unwrap_or_else(|| Tile::scalar(0i64));
                let (sf_row, sf_col) = match dim {
                    MxDim::Rowwise => (flat_row, coord(rank - 1) / i64::from(bs)),
                    MxDim::Colwise => (coord(rank - 1), flat_row / i64::from(bs)),
                };
                let index = mx::scale_factor_index(sf_row, sf_col, args.get(*num_blocks))
                    .broadcast_to(scale_shape.clone());
                let block_in_bounds = block_start.lt(sizes[b]);
                let scale_mask = and_all((0..rank).map(|a| {
                    if a == b {
                        along(block_in_bounds, a, rank)
                    } else {
                        along(in_bounds[a], a, rank)
                    }
                }))
                .unwrap()
                .broadcast_to(scale_shape.clone());

                let values = values.get(args);
                match (format, *scales) {
                    (MxFormat::Mxfp8, ScalePtr::Ue8m0(p)) => {
                        let (scale, descale) = mx::mxfp8_scale(amax);
                        let q = mx::clamp(v * expand(descale), mx::E4M3_MAX)
                            .astype::<F8E4M3>(RoundingMode::NearestEven);
                        values.store(tile_index.clone(), q.erase());
                        args.get(p).offset(index).store(scale, Some(scale_mask));
                    }
                    (MxFormat::Nvfp4, ScalePtr::E4m3(p)) => {
                        let (scale, descale) = mx::nvfp4_scale(amax);
                        let q = mx::clamp(v * expand(descale), mx::E2M1_MAX)
                            .astype::<F4E2M1>(RoundingMode::NearestEven);
                        values.store(tile_index.clone(), q.erase());
                        args.get(p).offset(index).store(scale, Some(scale_mask));
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!("output facts do not match the IR"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execgraph::{
        ops::pointwise::{
            ir::NodeId,
            lower::{OperandDesc, lower},
        },
        sym::{SymDims, SymLayout},
    };
    use tileir_builder::TileirasOptions;

    /// Compiles kernels with block-scaled outputs for sm_100, which this machine may
    /// not be able to run.
    #[test]
    fn block_scaled_kernels_compile_for_sm100() {
        if tileir::tools::Tileiras::locate().is_err() {
            eprintln!("skipping: tileiras not found");
            return;
        }
        let dims = SymDims::new();
        let f32_ty = ElementType::Float(FloatType::F32);
        let layout = SymLayout::contiguous_row_major(
            [64usize, 128].map(Into::into).into_iter().collect(),
            f32_ty,
            &dims,
        )
        .unwrap();
        for format in [MxFormat::Nvfp4, MxFormat::Mxfp8] {
            let ir = PwIr {
                nodes: vec![Node::Input(0)],
                outputs: [MxDim::Rowwise, MxDim::Colwise]
                    .map(|dim| PwOutput::Mx {
                        value: NodeId(0),
                        format,
                        dim,
                    })
                    .to_vec(),
            };
            let lowering = lower(
                &ir,
                &[OperandDesc {
                    layout: &layout,
                    ty: f32_ty,
                }],
                &dims,
            )
            .unwrap();
            let tile = [32, 64];
            let key = KernelKey {
                ir: &ir,
                facts: &lowering.facts,
                tile: &tile,
            };
            let mut module = ModuleBuilder::new("pointwise_test");
            define(&mut module, &key);
            let built = module.finish().unwrap();
            built
                .compile(&TileirasOptions::new().arch("sm_100"))
                .unwrap_or_else(|e| panic!("{format:?}: {e}"));
        }
    }
}
