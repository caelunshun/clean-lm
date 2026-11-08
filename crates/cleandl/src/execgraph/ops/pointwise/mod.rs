//! Fused elementwise kernels with broadcasting, reductions and block-scaled outputs,
//! JIT-compiled to Tile IR from a small dataflow IR.
//!
//! ```ignore
//! let [y, row_sums] = gb.pointwise(|p| {
//!     let x = p.input(x)?;
//!     let bias = p.input(bias)?;
//!     let y = (x + bias).max(0.0);
//!     p.output(y, ElementType::Float(FloatType::Bf16))?;
//!     p.output_reduce(y, Reduce::Sum, &[1])
//! })?[..] else { unreachable!() };
//! ```

pub(crate) mod codegen;
pub mod ir;
mod lower;
pub(crate) mod mx;

use crate::{
    context::Context,
    execgraph::{
        BuildCx, Executor, GraphBuilder, GraphInfo, OpIo, Operand, Operator, OutputSpec, SymTensor,
        SymTensorId,
    },
    mx_tensor::{MxDim, MxFormat},
};
use anyhow::{anyhow, bail, ensure};
use codegen::{KERNEL_NAME, KernelKey, Params};
pub use ir::{BinaryOp, Node, NodeId, PwIr, PwOutput, Reduce, UnaryOp, ValueTy};
use kernel_build_util::{ArgBundle, LoadedKernel, tileir::KernelBuilder};
use lower::{Facts, Lowering, OperandDesc, Plan};
use serde::Serialize;
use skycuda::stream::Stream;
use skytens_core::data_type::ElementType;
use std::{
    cell::RefCell,
    ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Not, Rem, Sub},
    sync::Arc,
};

/// A fused pointwise operator. Usually built with [`GraphBuilder::pointwise`].
#[derive(Clone, Debug)]
pub struct Pointwise {
    ir: PwIr,
    inputs: Vec<SymTensorId>,
}

impl Pointwise {
    /// `inputs[i]` is the operand read by [`Node::Input`]`(i)`.
    pub fn new(ir: PwIr, inputs: Vec<SymTensorId>) -> Self {
        Self { ir, inputs }
    }

    pub fn ir(&self) -> &PwIr {
        &self.ir
    }

    pub fn inputs(&self) -> &[SymTensorId] {
        &self.inputs
    }

    /// Records the operator defined by `f` without adding it to the graph.
    pub fn trace(
        gb: &GraphBuilder,
        f: impl FnOnce(&PwBuilder<'_>) -> anyhow::Result<()>,
    ) -> anyhow::Result<Self> {
        let builder = PwBuilder {
            gb,
            state: PwState::default(),
        };
        f(&builder)?;
        let inner = builder.state.0.into_inner();
        if let Some(error) = inner.error {
            return Err(error.context("pointwise"));
        }
        ensure!(!inner.ir.outputs.is_empty(), "pointwise: no outputs");
        Ok(Self::new(inner.ir, inner.inputs))
    }

    fn lower(&self, graph: &GraphInfo) -> anyhow::Result<Lowering> {
        let operands = self
            .inputs
            .iter()
            .enumerate()
            .map(|(k, &id)| {
                let layout = graph
                    .layout(id)
                    .ok_or_else(|| anyhow!("operand {k}: block-scaled inputs are not supported"))?;
                let ty = graph.ty(id).element_type().unwrap();
                Ok(OperandDesc { layout, ty })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        lower::lower(&self.ir, &operands, graph.dims())
    }

    fn prepare(
        &self,
        cx: &Context,
        lowering: &Lowering,
        tile: &[u32],
    ) -> anyhow::Result<PointwiseExecutor> {
        let key = KernelKey {
            ir: &self.ir,
            facts: &lowering.facts,
            tile,
        };
        let kernel = crate::execgraph::jit::get_or_compile_kernel(cx, &key, KERNEL_NAME, |m| {
            codegen::define(m, &key)
        })?;
        Ok(PointwiseExecutor {
            kernel,
            params: Params::declare(&mut KernelBuilder::new(), &lowering.facts),
            facts: lowering.facts.clone(),
            plan: lowering.plan.clone(),
            tile: tile.to_vec(),
        })
    }

    /// Builds an executor without tuning, using `tile` or the heuristic default.
    #[doc(hidden)]
    pub fn build_executor(
        &self,
        cx: &Context,
        graph: &GraphInfo,
        tile: Option<&[u32]>,
    ) -> anyhow::Result<Box<dyn Executor>> {
        let lowering = self.lower(graph)?;
        let default = candidates(&lowering, graph).swap_remove(0);
        let tile = tile.unwrap_or(&default);
        ensure!(
            tile.len() == lowering.facts.rank,
            "tile rank {} does not match the lowered rank {}",
            tile.len(),
            lowering.facts.rank
        );
        let min_tile = lowering.facts.min_tile();
        ensure!(
            tile.iter()
                .zip(&min_tile)
                .all(|(&t, &min)| t.is_power_of_two() && t >= min),
            "tile {tile:?} must consist of powers of two of at least {min_tile:?}"
        );
        ensure!(
            tile.iter()
                .try_fold(1u32, |a, &t| a.checked_mul(t))
                .is_some_and(|n| n <= 1 << 16),
            "tile {tile:?} has more than 65536 elements"
        );
        Ok(Box::new(self.prepare(cx, &lowering, tile)?))
    }

    /// The tile shapes autotuning chooses from, the default first.
    #[doc(hidden)]
    pub fn tile_candidates(&self, graph: &GraphInfo) -> anyhow::Result<Vec<Vec<u32>>> {
        Ok(candidates(&self.lower(graph)?, graph))
    }
}

fn candidates(lowering: &Lowering, graph: &GraphInfo) -> Vec<Vec<u32>> {
    let hints: Vec<usize> = lowering
        .plan
        .sizes
        .iter()
        .map(|s| s.eval_hint(graph.dims()))
        .collect();
    lower::tile_candidates(&hints, &lowering.facts.min_tile())
}

#[derive(Serialize)]
struct TunePayload<'a> {
    ir: &'a PwIr,
    facts: &'a Facts,
    hints: Vec<usize>,
}

impl Operator for Pointwise {
    fn name(&self) -> &str {
        "pointwise"
    }

    fn operands(&self) -> Vec<Operand> {
        self.inputs.iter().map(|&t| Operand::read(t)).collect()
    }

    fn infer_outputs(&self, graph: &GraphInfo) -> anyhow::Result<Vec<OutputSpec>> {
        Ok(self.lower(graph)?.output_specs)
    }

    fn build(&self, bcx: &BuildCx<'_>) -> anyhow::Result<Box<dyn Executor>> {
        let lowering = self.lower(bcx.graph)?;
        let payload = TunePayload {
            ir: &self.ir,
            facts: &lowering.facts,
            hints: lowering
                .plan
                .sizes
                .iter()
                .map(|s| s.eval_hint(bcx.graph.dims()))
                .collect(),
        };
        let key = bcx.tune_key(&payload)?;
        let (_, executor) =
            bcx.tune_executor(&key, candidates(&lowering, bcx.graph), |tile: &Vec<u32>| {
                self.prepare(bcx.cx, &lowering, tile)
            })?;
        Ok(Box::new(executor))
    }
}

struct PointwiseExecutor {
    kernel: Arc<LoadedKernel>,
    params: Params,
    facts: Facts,
    plan: Plan,
    tile: Vec<u32>,
}

/// Maximum grid dimension of a Tile IR launch.
const MAX_GRID_DIM: u64 = (1 << 24) - 1;

impl PointwiseExecutor {
    /// Splits the tiles over the grid's x (inner axes) and y (outer axes) dimensions.
    fn grid(&self, sizes: &[i64]) -> anyhow::Result<(u32, u32, u32)> {
        let tiles: Vec<u64> = sizes
            .iter()
            .zip(&self.tile)
            .map(|(&s, &t)| (s as u64).div_ceil(u64::from(t)))
            .collect();
        let product = |t: &[u64]| t.iter().try_fold(1u64, |a, &b| a.checked_mul(b));
        for split in 0..=tiles.len() {
            if let (Some(y), Some(x)) = (product(&tiles[..split]), product(&tiles[split..]))
                && x <= MAX_GRID_DIM
                && y <= MAX_GRID_DIM
            {
                return Ok((x as u32, y as u32, 1));
            }
        }
        bail!(
            "pointwise: iteration space {sizes:?} is too large for tile {:?}",
            self.tile
        )
    }
}

impl Executor for PointwiseExecutor {
    fn execute(&self, io: &mut OpIo<'_>, stream: &Stream, _cx: &Context) -> anyhow::Result<()> {
        let sizes = self
            .plan
            .sizes
            .iter()
            .map(|s| Ok(i64::try_from(s.eval(io.dims)?)?))
            .collect::<anyhow::Result<Vec<i64>>>()?;
        if sizes.contains(&0) {
            return Ok(());
        }
        let grid = self.grid(&sizes)?;
        let mut args = ArgBundle::new();
        self.params.bind(
            &mut args,
            &self.facts,
            &self.plan,
            &sizes,
            &io.inputs,
            &mut io.outputs,
            io.dims,
        )?;
        // Soundness: arguments match the declared parameters, and every access is
        // in bounds of the bound views.
        unsafe { self.kernel.launch(args, grid, stream)? };
        Ok(())
    }
}

impl GraphBuilder {
    /// Adds a fused pointwise operator defined by `f`, returning its outputs in
    /// declaration order.
    ///
    /// Operands are broadcast NumPy-style (right-aligned; each axis must have
    /// provably equal sizes or size one). Values are computed in `f32`, `i32` or
    /// `bool`; float inputs are loaded as `f32` and integer inputs as `i32`. Integer
    /// division by zero yields the dividend (`x / 0 == x`, `x % 0 == 0`).
    pub fn pointwise(
        &self,
        f: impl FnOnce(&PwBuilder<'_>) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<SymTensor<'_>>> {
        let op = Pointwise::trace(self, f)?;
        self.add_op(op)
    }
}

/// Builder state shared by the values of one pointwise operator.
#[derive(Default)]
pub struct PwState(RefCell<PwInner>);

#[derive(Default)]
struct PwInner {
    ir: PwIr,
    tys: Vec<ValueTy>,
    inputs: Vec<SymTensorId>,
    error: Option<anyhow::Error>,
}

impl PwState {
    fn push(&self, node: Node, ty: ValueTy) -> PwValue<'_> {
        let mut inner = self.0.borrow_mut();
        let id = NodeId(inner.ir.nodes.len() as u32);
        inner.ir.nodes.push(node);
        inner.tys.push(ty);
        PwValue {
            state: self,
            id,
            ty,
        }
    }

    /// Records the first error and returns a placeholder value of type `ty`.
    fn fail(&self, ty: ValueTy, error: anyhow::Error) -> PwValue<'_> {
        self.0.borrow_mut().error.get_or_insert(error);
        match ty {
            ValueTy::F32 => self.push(Node::ConstF32(0), ty),
            ValueTy::I32 => self.push(Node::ConstI32(0), ty),
            ValueTy::Bool => self.push(Node::ConstBool(false), ty),
        }
    }

    fn own<'p>(&'p self, v: PwValue<'p>) -> PwValue<'p> {
        if std::ptr::eq(self, v.state) {
            v
        } else {
            self.fail(
                v.ty,
                anyhow!("value used with a different pointwise builder"),
            )
        }
    }

    fn cast<'p>(&'p self, v: PwValue<'p>, ty: ValueTy) -> PwValue<'p> {
        let v = self.own(v);
        if v.ty == ty {
            v
        } else {
            self.push(Node::Cast(ty, v.id), ty)
        }
    }

    fn unary<'p>(&'p self, op: UnaryOp, a: PwValue<'p>) -> PwValue<'p> {
        let a = self.own(a);
        let a = match (op, a.ty) {
            (UnaryOp::Not, ValueTy::Bool) => a,
            (UnaryOp::Not, _) | (_, ValueTy::Bool) => {
                return self.fail(a.ty, anyhow!("{op:?} is not defined for {:?} values", a.ty));
            }
            (UnaryOp::Neg | UnaryOp::Abs, _) => a,
            _ => self.cast(a, ValueTy::F32),
        };
        self.push(Node::Unary(op, a.id), a.ty)
    }

    fn binary<'p>(&'p self, op: BinaryOp, a: PwValue<'p>, b: PwValue<'p>) -> PwValue<'p> {
        let (a, b) = (self.own(a), self.own(b));
        let ty = match op {
            BinaryOp::And | BinaryOp::Or => {
                if a.ty != ValueTy::Bool || b.ty != ValueTy::Bool {
                    return self.fail(
                        ValueTy::Bool,
                        anyhow!("{op:?} requires bool values, got {:?} and {:?}", a.ty, b.ty),
                    );
                }
                ValueTy::Bool
            }
            BinaryOp::Pow => ValueTy::F32,
            BinaryOp::Eq | BinaryOp::Ne if a.ty == b.ty => a.ty,
            _ => {
                if a.ty == ValueTy::Bool || b.ty == ValueTy::Bool {
                    let ty = if op.is_comparison() {
                        ValueTy::Bool
                    } else {
                        ValueTy::F32
                    };
                    return self.fail(ty, anyhow!("{op:?} is not defined for bool values"));
                }
                if a.ty == ValueTy::F32 || b.ty == ValueTy::F32 {
                    ValueTy::F32
                } else {
                    ValueTy::I32
                }
            }
        };
        if op == BinaryOp::Rem && ty != ValueTy::I32 {
            return self.fail(ty, anyhow!("remainder is only defined for i32 values"));
        }
        let (a, b) = (self.cast(a, ty), self.cast(b, ty));
        let result = if op.is_comparison() {
            ValueTy::Bool
        } else {
            ty
        };
        self.push(Node::Binary(op, a.id, b.id), result)
    }

    fn fma<'p>(&'p self, a: PwValue<'p>, b: PwValue<'p>, c: PwValue<'p>) -> PwValue<'p> {
        if [a.ty, b.ty, c.ty].contains(&ValueTy::Bool) {
            return self.fail(ValueTy::F32, anyhow!("fma is not defined for bool values"));
        }
        let [a, b, c] = [a, b, c].map(|v| self.cast(v, ValueTy::F32));
        self.push(Node::Fma(a.id, b.id, c.id), ValueTy::F32)
    }

    fn select<'p>(&'p self, c: PwValue<'p>, a: PwValue<'p>, b: PwValue<'p>) -> PwValue<'p> {
        let (c, a, b) = (self.own(c), self.own(a), self.own(b));
        let ty = if a.ty == b.ty {
            a.ty
        } else if a.ty == ValueTy::Bool || b.ty == ValueTy::Bool {
            return self.fail(
                a.ty,
                anyhow!("select between {:?} and {:?} values", a.ty, b.ty),
            );
        } else {
            ValueTy::F32
        };
        if c.ty != ValueTy::Bool {
            return self.fail(ty, anyhow!("select condition must be bool, got {:?}", c.ty));
        }
        let (a, b) = (self.cast(a, ty), self.cast(b, ty));
        self.push(Node::Select(c.id, a.id, b.id), ty)
    }
}

/// Defines a pointwise operator; see [`GraphBuilder::pointwise`].
pub struct PwBuilder<'g> {
    gb: &'g GraphBuilder,
    state: PwState,
}

impl PwBuilder<'_> {
    /// Reads an operand, broadcast over the iteration space. Float inputs are
    /// loaded as `f32` values, integer inputs as `i32`.
    pub fn input(&self, tensor: SymTensor<'_>) -> anyhow::Result<PwValue<'_>> {
        ensure!(
            std::ptr::eq(tensor.builder(), self.gb),
            "pointwise input belongs to a different graph"
        );
        let Some(elem) = tensor.element_type() else {
            bail!("pointwise inputs must be plain tensors, got a block-scaled tensor");
        };
        let ty = ValueTy::of_element(elem)?;
        let k = {
            let mut inner = self.state.0.borrow_mut();
            match inner.inputs.iter().position(|&t| t == tensor.id()) {
                Some(k) => k,
                None => {
                    inner.inputs.push(tensor.id());
                    inner.inputs.len() - 1
                }
            }
        };
        Ok(self.state.push(Node::Input(k), ty))
    }

    /// The `i32` coordinate along an axis of the iteration space.
    pub fn index(&self, axis: usize) -> PwValue<'_> {
        self.state.push(Node::Index(axis), ValueTy::I32)
    }

    pub fn constant(&self, value: f32) -> PwValue<'_> {
        self.state
            .push(Node::ConstF32(value.to_bits()), ValueTy::F32)
    }

    pub fn constant_i32(&self, value: i32) -> PwValue<'_> {
        self.state.push(Node::ConstI32(value), ValueTy::I32)
    }

    pub fn constant_bool(&self, value: bool) -> PwValue<'_> {
        self.state.push(Node::ConstBool(value), ValueTy::Bool)
    }

    /// `a` where `cond` holds, else `b`.
    pub fn select<'p>(
        &'p self,
        cond: impl IntoPwValue<'p>,
        a: impl IntoPwValue<'p>,
        b: impl IntoPwValue<'p>,
    ) -> PwValue<'p> {
        let s = &self.state;
        s.select(cond.into_pw(s), a.into_pw(s), b.into_pw(s))
    }

    fn own<'p>(&'p self, v: PwValue<'p>) -> anyhow::Result<PwValue<'p>> {
        ensure!(
            std::ptr::eq(&self.state, v.state),
            "value belongs to a different pointwise builder"
        );
        Ok(v)
    }

    fn push_output(&self, output: PwOutput) {
        self.state.0.borrow_mut().ir.outputs.push(output);
    }

    /// An output holding `value` at every point of the iteration space, converted
    /// to `ty`. Conversions to 8-bit floats saturate to the largest finite value.
    pub fn output(&self, value: PwValue<'_>, ty: ElementType) -> anyhow::Result<()> {
        let value = self.own(value)?;
        ir::check_output_type(ty)?;
        self.push_output(PwOutput::Elementwise {
            value: value.id,
            ty,
        });
        Ok(())
    }

    /// An `f32` output reducing `value` over the given iteration axes, which are
    /// dropped from its shape (a full reduction has shape `[1]`). A NaN among the
    /// reduced values makes the result NaN for every op. Sums are accumulated with
    /// atomics, so their rounding is nondeterministic, and subnormal partial sums of
    /// a tile may be flushed to zero.
    pub fn output_reduce(
        &self,
        value: PwValue<'_>,
        op: Reduce,
        axes: &[usize],
    ) -> anyhow::Result<()> {
        let value = self.own(value)?;
        let value = self.state.cast(value, ValueTy::F32);
        self.push_output(PwOutput::Reduce {
            value: value.id,
            op,
            axes: axes.to_vec(),
        });
        Ok(())
    }

    /// A block-scaled output of the iteration space viewed as 2D `[product of all
    /// but the last axis, last axis]`.
    pub fn output_mx(
        &self,
        value: PwValue<'_>,
        format: MxFormat,
        dim: MxDim,
    ) -> anyhow::Result<()> {
        let value = self.own(value)?;
        let value = self.state.cast(value, ValueTy::F32);
        self.push_output(PwOutput::Mx {
            value: value.id,
            format,
            dim,
        });
        Ok(())
    }
}

/// A value of a pointwise operator, valid while its [`PwBuilder`] is alive.
#[derive(Clone, Copy)]
pub struct PwValue<'p> {
    state: &'p PwState,
    id: NodeId,
    ty: ValueTy,
}

impl std::fmt::Debug for PwValue<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PwValue({}: {:?})", self.id.0, self.ty)
    }
}

/// Values and literals usable as operands of [`PwValue`] operations.
pub trait IntoPwValue<'p> {
    fn into_pw(self, state: &'p PwState) -> PwValue<'p>;
}

impl<'p> IntoPwValue<'p> for PwValue<'p> {
    fn into_pw(self, state: &'p PwState) -> PwValue<'p> {
        state.own(self)
    }
}

impl<'p> IntoPwValue<'p> for f32 {
    fn into_pw(self, state: &'p PwState) -> PwValue<'p> {
        state.push(Node::ConstF32(self.to_bits()), ValueTy::F32)
    }
}

impl<'p> IntoPwValue<'p> for i32 {
    fn into_pw(self, state: &'p PwState) -> PwValue<'p> {
        state.push(Node::ConstI32(self), ValueTy::I32)
    }
}

impl<'p> IntoPwValue<'p> for bool {
    fn into_pw(self, state: &'p PwState) -> PwValue<'p> {
        state.push(Node::ConstBool(self), ValueTy::Bool)
    }
}

macro_rules! unary_methods {
    ($($name:ident => $op:ident),* $(,)?) => {$(
        #[allow(clippy::should_implement_trait)]
        pub fn $name(self) -> Self {
            self.state.unary(UnaryOp::$op, self)
        }
    )*};
}

macro_rules! binary_methods {
    ($($name:ident => $op:ident),* $(,)?) => {$(
        pub fn $name(self, other: impl IntoPwValue<'p>) -> Self {
            self.state.binary(BinaryOp::$op, self, other.into_pw(self.state))
        }
    )*};
}

impl<'p> PwValue<'p> {
    pub fn ty(self) -> ValueTy {
        self.ty
    }

    unary_methods!(
        exp => Exp,
        log => Log,
        sqrt => Sqrt,
        rsqrt => Rsqrt,
        abs => Abs,
        tanh => Tanh,
        sigmoid => Sigmoid,
        neg => Neg,
        floor => Floor,
    );

    binary_methods!(
        min => Min,
        max => Max,
        pow => Pow,
        lt => Lt,
        le => Le,
        gt => Gt,
        ge => Ge,
        eq => Eq,
        ne => Ne,
    );

    /// `self * b + c`.
    pub fn fma(self, b: impl IntoPwValue<'p>, c: impl IntoPwValue<'p>) -> Self {
        let s = self.state;
        s.fma(self, b.into_pw(s), c.into_pw(s))
    }

    /// `a` where `self` holds, else `b`.
    pub fn select(self, a: impl IntoPwValue<'p>, b: impl IntoPwValue<'p>) -> Self {
        let s = self.state;
        s.select(self, a.into_pw(s), b.into_pw(s))
    }

    /// Converts like Rust's `as`; to `bool` means `!= 0`.
    pub fn cast(self, ty: ValueTy) -> Self {
        self.state.cast(self, ty)
    }

    pub fn to_f32(self) -> Self {
        self.cast(ValueTy::F32)
    }

    pub fn to_i32(self) -> Self {
        self.cast(ValueTy::I32)
    }

    pub fn to_bool(self) -> Self {
        self.cast(ValueTy::Bool)
    }
}

macro_rules! binary_operators {
    ($($tr:ident $method:ident $op:ident),* $(,)?) => {$(
        impl<'p, R: IntoPwValue<'p>> $tr<R> for PwValue<'p> {
            type Output = PwValue<'p>;
            fn $method(self, rhs: R) -> PwValue<'p> {
                self.state.binary(BinaryOp::$op, self, rhs.into_pw(self.state))
            }
        }
        impl<'p> $tr<PwValue<'p>> for f32 {
            type Output = PwValue<'p>;
            fn $method(self, rhs: PwValue<'p>) -> PwValue<'p> {
                rhs.state.binary(BinaryOp::$op, self.into_pw(rhs.state), rhs)
            }
        }
        impl<'p> $tr<PwValue<'p>> for i32 {
            type Output = PwValue<'p>;
            fn $method(self, rhs: PwValue<'p>) -> PwValue<'p> {
                rhs.state.binary(BinaryOp::$op, self.into_pw(rhs.state), rhs)
            }
        }
    )*};
}

binary_operators!(
    Add add Add,
    Sub sub Sub,
    Mul mul Mul,
    Div div Div,
    Rem rem Rem,
);

impl<'p, R: IntoPwValue<'p>> BitAnd<R> for PwValue<'p> {
    type Output = PwValue<'p>;
    fn bitand(self, rhs: R) -> PwValue<'p> {
        self.state
            .binary(BinaryOp::And, self, rhs.into_pw(self.state))
    }
}

impl<'p, R: IntoPwValue<'p>> BitOr<R> for PwValue<'p> {
    type Output = PwValue<'p>;
    fn bitor(self, rhs: R) -> PwValue<'p> {
        self.state
            .binary(BinaryOp::Or, self, rhs.into_pw(self.state))
    }
}

impl<'p> Not for PwValue<'p> {
    type Output = PwValue<'p>;
    fn not(self) -> PwValue<'p> {
        self.state.unary(UnaryOp::Not, self)
    }
}

impl<'p> Neg for PwValue<'p> {
    type Output = PwValue<'p>;
    fn neg(self) -> PwValue<'p> {
        self.state.unary(UnaryOp::Neg, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skytens_core::data_type::FloatType;

    const F32: ElementType = ElementType::Float(FloatType::F32);

    #[test]
    fn traces_with_promotion_and_dedup() {
        let gb = GraphBuilder::new();
        let x = gb.input("x", F32, [4, 8]).unwrap();
        let i = gb
            .input(
                "i",
                ElementType::SInt(skytens_core::data_type::SIntType::S32),
                [8],
            )
            .unwrap();
        let op = Pointwise::trace(&gb, |p| {
            let a = p.input(x)?;
            let b = p.input(x)?;
            let j = p.input(i)?;
            let y = (a * 2.0 + b) / j;
            let m = p.index(0).lt(2);
            p.output(p.select(m, y, -y), F32)?;
            p.output_reduce(j + 1, Reduce::Sum, &[0])
        })
        .unwrap();
        assert_eq!(op.inputs(), [x.id(), i.id()]);
        let tys = op.ir().check(&[ValueTy::F32, ValueTy::I32]).unwrap();
        assert_eq!(tys[op.ir().outputs[0].value().index()], ValueTy::F32);
        assert_eq!(tys[op.ir().outputs[1].value().index()], ValueTy::F32);
    }

    #[test]
    fn records_type_errors() {
        let gb = GraphBuilder::new();
        let x = gb.input("x", F32, [4]).unwrap();
        let err = Pointwise::trace(&gb, |p| {
            let a = p.input(x)?;
            let bad = a.gt(0.0) + 1.0;
            p.output(bad, F32)
        })
        .unwrap_err();
        assert!(format!("{err:#}").contains("bool"), "{err:#}");

        let err = Pointwise::trace(&gb, |p| {
            let a = p.input(x)?;
            p.output(a.exp(), ElementType::Bool)
        })
        .unwrap_err();
        assert!(format!("{err:#}").contains("not supported"), "{err:#}");
    }
}
