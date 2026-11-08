//! Tracing of symbolic execution graphs.

use super::{
    op::{Access, GraphInfo, Init, Operand, Operator, OutputSpec, TensorTy},
    sym::{Monomial, SizeExpr, SymDim, SymDimId, SymDims, SymLayout, SymShape},
};
use crate::mx_tensor::{MxDim, MxFormat};
use anyhow::{Context as _, anyhow, bail, ensure};
use skytens_core::{data_type::ElementType, shape::Shape};
use std::{cell::RefCell, collections::HashSet, fmt};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymTensorId(pub(crate) usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferId(pub(crate) usize);

impl BufferId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// A symbolic tensor: a view over (one or two) graph buffers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymTensorData {
    Plain {
        buffer: BufferId,
        layout: SymLayout,
        ty: ElementType,
    },
    /// Opaque 2D block-scaled tensor in the [`MxLayout`](crate::mx_tensor::MxLayout).
    BlockScaled {
        values: BufferId,
        scales: BufferId,
        format: MxFormat,
        dim: MxDim,
        shape: SymShape,
    },
}

impl SymTensorData {
    pub fn ty(&self) -> TensorTy {
        match self {
            SymTensorData::Plain { ty, .. } => TensorTy::Plain(*ty),
            SymTensorData::BlockScaled { format, dim, .. } => TensorTy::BlockScaled {
                format: *format,
                dim: *dim,
            },
        }
    }

    pub fn shape(&self) -> &SymShape {
        match self {
            SymTensorData::Plain { layout, .. } => &layout.shape,
            SymTensorData::BlockScaled { shape, .. } => shape,
        }
    }

    pub fn layout(&self) -> Option<&SymLayout> {
        match self {
            SymTensorData::Plain { layout, .. } => Some(layout),
            SymTensorData::BlockScaled { .. } => None,
        }
    }

    pub fn buffers(&self) -> impl Iterator<Item = BufferId> + use<> {
        let (a, b) = match self {
            SymTensorData::Plain { buffer, .. } => (*buffer, None),
            SymTensorData::BlockScaled { values, scales, .. } => (*values, Some(*scales)),
        };
        std::iter::once(a).chain(b)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BufferPart {
    Plain,
    Values,
    Scales,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BufferSource {
    /// Storage of a graph input.
    Input { input: usize, part: BufferPart },
    /// Allocated for an output of an op.
    Op { op: usize, init: Init },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferInfo {
    /// Element type the buffer is allocated for.
    pub ty: ElementType,
    pub bytes: SizeExpr,
    pub source: BufferSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InputInfo {
    pub name: String,
    pub ty: TensorTy,
    pub mutable: bool,
    pub tensor: SymTensorId,
}

pub(crate) struct OpRecord {
    pub op: Box<dyn Operator>,
    pub operands: Vec<Operand>,
    pub outputs: Vec<SymTensorId>,
}

impl fmt::Debug for OpRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpRecord")
            .field("op", &self.op.name())
            .field("operands", &self.operands)
            .field("outputs", &self.outputs)
            .finish()
    }
}

/// Shapes accepted by the graph builder.
pub trait IntoSymShape {
    fn into_sym_shape(self) -> SymShape;
}

impl IntoSymShape for SymShape {
    fn into_sym_shape(self) -> SymShape {
        self
    }
}

impl IntoSymShape for &SymShape {
    fn into_sym_shape(self) -> SymShape {
        self.clone()
    }
}

impl IntoSymShape for Shape {
    fn into_sym_shape(self) -> SymShape {
        self.into()
    }
}

impl<T: Into<Monomial>, const N: usize> IntoSymShape for [T; N] {
    fn into_sym_shape(self) -> SymShape {
        self.into_iter().map(Into::into).collect()
    }
}

impl<T: Into<Monomial>> IntoSymShape for Vec<T> {
    fn into_sym_shape(self) -> SymShape {
        self.into_iter().map(Into::into).collect()
    }
}

impl<T: Into<Monomial> + Clone> IntoSymShape for &[T] {
    fn into_sym_shape(self) -> SymShape {
        self.iter().cloned().map(Into::into).collect()
    }
}

#[derive(Debug, Default)]
struct BuilderState {
    info: GraphInfo,
    inputs: Vec<InputInfo>,
    ops: Vec<OpRecord>,
    outputs: Vec<SymTensorId>,
}

/// Traces a graph. Tensors are referred to by [`SymTensor`] handles borrowing the builder.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    state: RefCell<BuilderState>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a dimension known only at execution time. `hint` is used for autotuning;
    /// the size is guaranteed to be a multiple of `2^alignment_log2`.
    ///
    /// Panics if `hint` violates the alignment.
    pub fn dynamic_dim(
        &self,
        name: impl Into<String>,
        hint: usize,
        alignment_log2: u32,
    ) -> SymDimId {
        self.state.borrow_mut().info.dims.insert(SymDim {
            name: name.into(),
            hint,
            alignment_log2,
        })
    }

    pub fn dims(&self) -> SymDims {
        self.state.borrow().info.dims.clone()
    }

    /// A read-only input with a packed row-major layout.
    pub fn input(
        &self,
        name: impl Into<String>,
        ty: ElementType,
        shape: impl IntoSymShape,
    ) -> anyhow::Result<SymTensor<'_>> {
        self.plain_input(name.into(), ty, shape.into_sym_shape(), None, false)
    }

    /// Like [`Self::input`], but ops may modify it in place.
    pub fn input_mut(
        &self,
        name: impl Into<String>,
        ty: ElementType,
        shape: impl IntoSymShape,
    ) -> anyhow::Result<SymTensor<'_>> {
        self.plain_input(name.into(), ty, shape.into_sym_shape(), None, true)
    }

    /// A read-only input with arbitrary symbolic strides. The offset must be zero, since
    /// the input's view starts at the beginning of its storage slice.
    pub fn input_with_layout(
        &self,
        name: impl Into<String>,
        ty: ElementType,
        layout: SymLayout,
    ) -> anyhow::Result<SymTensor<'_>> {
        let shape = layout.shape.clone();
        self.plain_input(name.into(), ty, shape, Some(layout), false)
    }

    pub fn input_with_layout_mut(
        &self,
        name: impl Into<String>,
        ty: ElementType,
        layout: SymLayout,
    ) -> anyhow::Result<SymTensor<'_>> {
        let shape = layout.shape.clone();
        self.plain_input(name.into(), ty, shape, Some(layout), true)
    }

    /// A read-only 2D block-scaled input.
    pub fn input_mx(
        &self,
        name: impl Into<String>,
        format: MxFormat,
        dim: MxDim,
        shape: impl IntoSymShape,
    ) -> anyhow::Result<SymTensor<'_>> {
        self.mx_input(name.into(), format, dim, shape.into_sym_shape(), false)
    }

    pub fn input_mx_mut(
        &self,
        name: impl Into<String>,
        format: MxFormat,
        dim: MxDim,
        shape: impl IntoSymShape,
    ) -> anyhow::Result<SymTensor<'_>> {
        self.mx_input(name.into(), format, dim, shape.into_sym_shape(), true)
    }

    fn plain_input(
        &self,
        name: String,
        ty: ElementType,
        shape: SymShape,
        layout: Option<SymLayout>,
        mutable: bool,
    ) -> anyhow::Result<SymTensor<'_>> {
        let mut state = self.state.borrow_mut();
        state.check_input_name(&name)?;
        let layout = match layout {
            Some(layout) => {
                ensure!(
                    layout.offset.is_zero(),
                    "input {name}: layout offset must be zero, got {}",
                    layout.offset.display(&state.info.dims)
                );
                layout
            }
            None => SymLayout::contiguous_row_major(shape, ty, &state.info.dims)
                .with_context(|| format!("input {name}"))?,
        };
        let input = state.inputs.len();
        let buffer = state.add_buffer(BufferInfo {
            ty,
            bytes: layout.required_bytes(ty),
            source: BufferSource::Input {
                input,
                part: BufferPart::Plain,
            },
        });
        let tensor = state.add_tensor(SymTensorData::Plain { buffer, layout, ty });
        state.inputs.push(InputInfo {
            name,
            ty: TensorTy::Plain(ty),
            mutable,
            tensor,
        });
        Ok(self.handle(tensor))
    }

    fn mx_input(
        &self,
        name: String,
        format: MxFormat,
        dim: MxDim,
        shape: SymShape,
        mutable: bool,
    ) -> anyhow::Result<SymTensor<'_>> {
        let mut state = self.state.borrow_mut();
        state.check_input_name(&name)?;
        let input = state.inputs.len();
        let data = state
            .add_mx_buffers(format, dim, shape, |part| BufferSource::Input {
                input,
                part,
            })
            .with_context(|| format!("input {name}"))?;
        let tensor = state.add_tensor(data);
        state.inputs.push(InputInfo {
            name,
            ty: TensorTy::BlockScaled { format, dim },
            mutable,
            tensor,
        });
        Ok(self.handle(tensor))
    }

    /// Records an operator and returns handles to its outputs.
    pub fn add_op(&self, op: impl Operator) -> anyhow::Result<Vec<SymTensor<'_>>> {
        let mut state = self.state.borrow_mut();
        let op_index = state.ops.len();
        let name = op.name().to_owned();
        let operands = op.operands();
        for (i, operand) in operands.iter().enumerate() {
            ensure!(
                operand.tensor.0 < state.info.tensors.len(),
                "{name}: operand {i} refers to an unknown tensor"
            );
            if operand.access == Access::ReadWrite {
                for buffer in state.info.tensors[operand.tensor.0].buffers() {
                    if let BufferSource::Input { input, .. } = state.info.buffers[buffer.0].source {
                        let input = &state.inputs[input];
                        ensure!(
                            input.mutable,
                            "{name}: operand {i} is written in place, but graph input {} is not mutable",
                            input.name
                        );
                    }
                }
            }
        }

        let specs = op
            .infer_outputs(&state.info)
            .with_context(|| name.clone())?;
        let mut outputs = Vec::with_capacity(specs.len());
        for (i, spec) in specs.into_iter().enumerate() {
            let tensor = state
                .add_output(op_index, spec)
                .with_context(|| format!("{name}: output {i}"))?;
            outputs.push(tensor);
        }
        state.ops.push(OpRecord {
            op: Box::new(op),
            operands,
            outputs: outputs.clone(),
        });
        Ok(outputs.into_iter().map(|t| self.handle(t)).collect())
    }

    /// Like [`Self::add_op`] for operators with exactly one output.
    pub fn add_op1(&self, op: impl Operator) -> anyhow::Result<SymTensor<'_>> {
        let name = op.name().to_owned();
        let outputs = self.add_op(op)?;
        ensure!(
            outputs.len() == 1,
            "{name}: expected one output, got {}",
            outputs.len()
        );
        Ok(outputs[0])
    }

    /// Marks a tensor as a graph output, returned by [`ExecGraph::execute`](super::ExecGraph::execute)
    /// in the order of these calls.
    pub fn output(&self, tensor: SymTensor<'_>) -> anyhow::Result<()> {
        ensure!(
            std::ptr::eq(tensor.builder, self),
            "output tensor belongs to a different graph"
        );
        self.state.borrow_mut().outputs.push(tensor.id);
        Ok(())
    }

    /// Wraps a tensor id of this graph in a handle.
    #[track_caller]
    pub fn tensor(&self, id: SymTensorId) -> SymTensor<'_> {
        assert!(
            id.0 < self.state.borrow().info.tensors.len(),
            "unknown tensor {id:?}"
        );
        self.handle(id)
    }

    /// Validates the graph and returns its immutable description.
    pub fn finish(self) -> anyhow::Result<Graph> {
        let state = self.state.into_inner();
        let graph = Graph {
            info: state.info,
            inputs: state.inputs,
            ops: state.ops,
            outputs: state.outputs,
        };
        graph.validate()?;
        Ok(graph)
    }

    fn handle(&self, id: SymTensorId) -> SymTensor<'_> {
        SymTensor { builder: self, id }
    }

    fn with_tensor<R>(&self, id: SymTensorId, f: impl FnOnce(&SymTensorData, &SymDims) -> R) -> R {
        let state = self.state.borrow();
        f(&state.info.tensors[id.0], &state.info.dims)
    }

    fn add_view(
        &self,
        id: SymTensorId,
        what: &str,
        f: impl FnOnce(&SymLayout, ElementType, &SymDims) -> anyhow::Result<SymLayout>,
    ) -> anyhow::Result<SymTensor<'_>> {
        let mut state = self.state.borrow_mut();
        let (buffer, layout, ty) = match &state.info.tensors[id.0] {
            SymTensorData::Plain { buffer, layout, ty } => (*buffer, layout, *ty),
            SymTensorData::BlockScaled { .. } => {
                bail!("cannot {what} a block-scaled tensor")
            }
        };
        let layout = f(layout, ty, &state.info.dims)?;
        let tensor = state.add_tensor(SymTensorData::Plain { buffer, layout, ty });
        Ok(self.handle(tensor))
    }
}

impl BuilderState {
    fn check_input_name(&self, name: &str) -> anyhow::Result<()> {
        ensure!(
            self.inputs.iter().all(|input| input.name != name),
            "duplicate graph input name {name}"
        );
        Ok(())
    }

    fn add_buffer(&mut self, buffer: BufferInfo) -> BufferId {
        self.info.buffers.push(buffer);
        BufferId(self.info.buffers.len() - 1)
    }

    fn add_tensor(&mut self, tensor: SymTensorData) -> SymTensorId {
        self.info.tensors.push(tensor);
        SymTensorId(self.info.tensors.len() - 1)
    }

    fn add_mx_buffers(
        &mut self,
        format: MxFormat,
        dim: MxDim,
        shape: SymShape,
        source: impl Fn(BufferPart) -> BufferSource,
    ) -> anyhow::Result<SymTensorData> {
        let dims = &self.info.dims;
        ensure!(
            shape.rank() == 2,
            "block-scaled tensors must be 2D, got shape {}",
            shape.display(dims)
        );
        let block_size = format.block_size();
        let block_dim = &shape[dim.block_dimension_2d()];
        ensure!(
            block_dim.is_divisible_by_pow2(block_size.ilog2(), dims),
            "block dimension of shape {} is not provably a multiple of the {format:?} block size {block_size}",
            shape.display(dims)
        );
        let numel = shape
            .checked_numel()
            .ok_or_else(|| anyhow!("element count of shape {} overflows", shape.display(dims)))?;
        let values_ty = format.values_type();
        let values_bytes =
            (SizeExpr::Mono(numel) * SizeExpr::Const(values_ty.bits_per_element())).div_ceil(8);
        let scales_ty = format.scale_factors_type();
        let scales_bytes = SizeExpr::Mono(shape[dim.non_block_dimension_2d()].clone())
            .round_up_to(128)
            * SizeExpr::Mono(block_dim.clone())
                .div_ceil(block_size)
                .round_up_to(4)
            * SizeExpr::Const(scales_ty.bits_per_element().div_ceil(8));

        let values = self.add_buffer(BufferInfo {
            ty: values_ty,
            bytes: values_bytes,
            source: source(BufferPart::Values),
        });
        let scales = self.add_buffer(BufferInfo {
            ty: scales_ty,
            bytes: scales_bytes,
            source: source(BufferPart::Scales),
        });
        Ok(SymTensorData::BlockScaled {
            values,
            scales,
            format,
            dim,
            shape,
        })
    }

    fn add_output(&mut self, op: usize, spec: OutputSpec) -> anyhow::Result<SymTensorId> {
        let data = match spec.ty {
            TensorTy::Plain(ty) => {
                if let Init::Fill32(_) = spec.init {
                    ensure!(
                        ty.bits_per_element() == 32,
                        "Init::Fill32 requires a 32-bit element type, got {ty:?}"
                    );
                }
                let layout = SymLayout::contiguous_row_major(spec.shape, ty, &self.info.dims)?;
                let buffer = self.add_buffer(BufferInfo {
                    ty,
                    bytes: layout.required_bytes(ty),
                    source: BufferSource::Op {
                        op,
                        init: spec.init,
                    },
                });
                SymTensorData::Plain { buffer, layout, ty }
            }
            TensorTy::BlockScaled { format, dim } => {
                ensure!(
                    !matches!(spec.init, Init::Fill32(_)),
                    "Init::Fill32 is not supported for block-scaled outputs"
                );
                let scales_init = if spec.init == Init::Uninit
                    && !mx_may_have_padding(&spec.shape, format, dim, &self.info.dims)
                {
                    Init::Uninit
                } else {
                    Init::Zero
                };
                self.add_mx_buffers(format, dim, spec.shape, |part| BufferSource::Op {
                    op,
                    init: if part == BufferPart::Scales {
                        scales_init
                    } else {
                        spec.init
                    },
                })?
            }
        };
        Ok(self.add_tensor(data))
    }
}

/// Whether the scale factors of a block-scaled tensor may contain padding,
/// which must be zeroed.
fn mx_may_have_padding(shape: &SymShape, format: MxFormat, dim: MxDim, dims: &SymDims) -> bool {
    if shape.rank() != 2 {
        return true;
    }
    let rows = &shape[dim.non_block_dimension_2d()];
    let block_dim = &shape[dim.block_dimension_2d()];
    !(rows.is_divisible_by_pow2(7, dims)
        && block_dim.is_divisible_by_pow2((format.block_size() * 4).ilog2(), dims))
}

/// Handle to a tensor of a [`GraphBuilder`].
#[derive(Copy, Clone)]
pub struct SymTensor<'g> {
    builder: &'g GraphBuilder,
    id: SymTensorId,
}

impl<'g> SymTensor<'g> {
    pub fn id(self) -> SymTensorId {
        self.id
    }

    pub fn builder(self) -> &'g GraphBuilder {
        self.builder
    }

    pub fn data(self) -> SymTensorData {
        self.builder.with_tensor(self.id, |t, _| t.clone())
    }

    pub fn ty(self) -> TensorTy {
        self.builder.with_tensor(self.id, |t, _| t.ty())
    }

    /// `None` for block-scaled tensors.
    pub fn element_type(self) -> Option<ElementType> {
        self.ty().element_type()
    }

    pub fn shape(self) -> SymShape {
        self.builder.with_tensor(self.id, |t, _| t.shape().clone())
    }

    pub fn rank(self) -> usize {
        self.builder.with_tensor(self.id, |t, _| t.shape().rank())
    }

    /// `None` for block-scaled tensors.
    pub fn layout(self) -> Option<SymLayout> {
        self.builder
            .with_tensor(self.id, |t, _| t.layout().cloned())
    }

    /// A view with two dimensions swapped.
    pub fn transposed(self, dim_a: usize, dim_b: usize) -> anyhow::Result<SymTensor<'g>> {
        self.builder.add_view(self.id, "transpose", |layout, _, _| {
            ensure!(
                dim_a < layout.rank() && dim_b < layout.rank(),
                "transpose dimensions ({dim_a}, {dim_b}) out of range for rank {}",
                layout.rank()
            );
            Ok(layout.transposed(dim_a, dim_b))
        })
    }

    /// A view with a different shape; fails unless the reshape is provably a view
    /// for all values of the dynamic dimensions.
    pub fn reshaped(self, shape: impl IntoSymShape) -> anyhow::Result<SymTensor<'g>> {
        let shape = shape.into_sym_shape();
        self.builder
            .add_view(self.id, "reshape", |layout, _, dims| {
                layout.try_reshaped(&shape).ok_or_else(|| {
                    anyhow!(
                        "cannot reshape tensor with layout {} to shape {} without copying",
                        DisplayLayout(layout, dims),
                        shape.display(dims)
                    )
                })
            })
    }

    /// A view of `len` elements starting at `start` along a fixed-size dimension.
    pub fn sliced(self, dim: usize, start: usize, len: usize) -> anyhow::Result<SymTensor<'g>> {
        self.builder.add_view(self.id, "slice", |layout, ty, dims| {
            layout.sliced(dim, start, len, ty, dims)
        })
    }
}

impl fmt::Debug for SymTensor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.builder.with_tensor(self.id, |t, dims| match t {
            SymTensorData::Plain { layout, ty, .. } => write!(
                f,
                "SymTensor({:?}, {ty:?}, {})",
                self.id.0,
                DisplayLayout(layout, dims)
            ),
            SymTensorData::BlockScaled {
                format, dim, shape, ..
            } => write!(
                f,
                "SymTensor({:?}, {format:?} {dim:?}, shape {})",
                self.id.0,
                shape.display(dims)
            ),
        })
    }
}

pub(crate) struct DisplayLayout<'a>(pub &'a SymLayout, pub &'a SymDims);

impl fmt::Display for DisplayLayout<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let DisplayLayout(layout, dims) = self;
        let strides: Vec<String> = layout
            .strides
            .iter()
            .map(|s| s.display(dims).to_string())
            .collect();
        write!(
            f,
            "shape {} strides [{}] offset {}",
            layout.shape.display(dims),
            strides.join(", "),
            layout.offset.display(dims)
        )
    }
}

/// An immutable, validated graph description.
pub struct Graph {
    pub(crate) info: GraphInfo,
    pub(crate) inputs: Vec<InputInfo>,
    pub(crate) ops: Vec<OpRecord>,
    pub(crate) outputs: Vec<SymTensorId>,
}

impl Graph {
    pub fn info(&self) -> &GraphInfo {
        &self.info
    }

    pub fn inputs(&self) -> &[InputInfo] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[SymTensorId] {
        &self.outputs
    }

    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    fn validate(&self) -> anyhow::Result<()> {
        let dims = &self.info.dims;
        let mut output_buffers = HashSet::new();
        for (i, &output) in self.outputs.iter().enumerate() {
            let tensor = self.info.tensor(output);
            if let SymTensorData::Plain { layout, .. } = tensor {
                ensure!(
                    layout.offset.is_zero(),
                    "output {i} must be a view at offset zero of its buffer, got {}",
                    DisplayLayout(layout, dims)
                );
            }
            for buffer in tensor.buffers() {
                if let BufferSource::Input { input, .. } = self.info.buffer(buffer).source {
                    bail!("output {i} aliases graph input {}", self.inputs[input].name);
                }
                ensure!(
                    output_buffers.insert(buffer),
                    "output {i} shares its buffer with another output"
                );
            }
        }

        for (op_index, op) in self.ops.iter().enumerate() {
            let mut written = HashSet::new();
            for operand in &op.operands {
                if operand.access == Access::ReadWrite {
                    for buffer in self.info.tensor(operand.tensor).buffers() {
                        ensure!(
                            written.insert(buffer),
                            "op {op_index} ({}) writes the same buffer through multiple operands",
                            op.op.name()
                        );
                    }
                }
            }
            for operand in &op.operands {
                if operand.access == Access::ReadOnly {
                    for buffer in self.info.tensor(operand.tensor).buffers() {
                        ensure!(
                            !written.contains(&buffer),
                            "op {op_index} ({}) reads and writes the same buffer through different operands",
                            op.op.name()
                        );
                    }
                }
            }
        }

        // Every dimension must be inferable from the inputs.
        let mut inferable = HashSet::new();
        for input in &self.inputs {
            let tensor = self.info.tensor(input.tensor);
            inferable.extend(
                tensor
                    .shape()
                    .dims()
                    .iter()
                    .flat_map(|d| d.atoms().to_vec()),
            );
            if let Some(layout) = tensor.layout() {
                inferable.extend(layout.strides.iter().flat_map(|s| s.atoms().to_vec()));
            }
        }
        for tensor in &self.info.tensors {
            for d in tensor.shape().dims() {
                for atom in d.atoms() {
                    ensure!(
                        inferable.contains(atom),
                        "dimension {} cannot be inferred from the graph inputs",
                        dims.name(*atom)
                    );
                }
            }
        }
        Ok(())
    }
}

impl fmt::Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field("dims", &self.info.dims)
            .field("inputs", &self.inputs)
            .field("ops", &self.ops)
            .field("outputs", &self.outputs)
            .finish()
    }
}
