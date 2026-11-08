//! The interface between the execution graph engine and operator implementations.

use super::{
    graph::{BufferId, BufferInfo, SymTensorData, SymTensorId},
    sym::{DimValues, SymDims, SymLayout, SymShape},
    tune::{Autotune, TuneKey, Tuner},
};
use crate::{
    context::Context,
    mx_tensor::{MxDim, MxFormat, MxLayout, MxTensorView, MxTensorViewMut},
    tensor::{TensorView, TensorViewMut, VirtTensorView},
};
use anyhow::{Context as _, anyhow, bail, ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use skycuda::{
    device_memory::DeviceMemory,
    device_memory_pool::DeviceMemoryPool,
    memory_slice::{MemorySlice, MemorySliceMut},
    stream::Stream,
};
use skytens_core::data_type::ElementType;
use std::collections::HashMap;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    ReadOnly,
    /// The operator may modify the operand in place.
    ReadWrite,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Operand {
    pub tensor: SymTensorId,
    pub access: Access,
}

impl Operand {
    pub fn read(tensor: SymTensorId) -> Self {
        Self {
            tensor,
            access: Access::ReadOnly,
        }
    }

    pub fn read_write(tensor: SymTensorId) -> Self {
        Self {
            tensor,
            access: Access::ReadWrite,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorTy {
    Plain(ElementType),
    /// A 2D block-scaled tensor, stored as values plus scale factors as given by
    /// [`MxLayout`].
    BlockScaled {
        format: MxFormat,
        dim: MxDim,
    },
}

impl TensorTy {
    pub fn element_type(self) -> Option<ElementType> {
        match self {
            TensorTy::Plain(ty) => Some(ty),
            TensorTy::BlockScaled { .. } => None,
        }
    }
}

impl From<ElementType> for TensorTy {
    fn from(ty: ElementType) -> Self {
        Self::Plain(ty)
    }
}

/// Initial contents of an output buffer, established before the operator runs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum Init {
    /// The operator overwrites every element of the output.
    #[default]
    Uninit,
    Zero,
    /// Every element set to this bit pattern. Only for 32-bit plain element types.
    Fill32(u32),
}

/// An output of an operator. Outputs get fresh buffers with packed row-major
/// layouts (or the [`MxLayout`] for block-scaled outputs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OutputSpec {
    pub ty: TensorTy,
    pub shape: SymShape,
    pub init: Init,
}

impl OutputSpec {
    pub fn new(ty: impl Into<TensorTy>, shape: SymShape, init: Init) -> Self {
        Self {
            ty: ty.into(),
            shape,
            init,
        }
    }
}

/// A node of the graph, as recorded during tracing.
pub trait Operator: Send + Sync + 'static {
    /// Used in errors and profiling.
    fn name(&self) -> &str;

    fn operands(&self) -> Vec<Operand>;

    /// Validates the operand types and layouts and returns the outputs.
    fn infer_outputs(&self, graph: &GraphInfo) -> anyhow::Result<Vec<OutputSpec>>;

    /// Prepares the operator for execution, e.g. compiling and autotuning kernels.
    fn build(&self, bcx: &BuildCx<'_>) -> anyhow::Result<Box<dyn Executor>>;
}

/// A built operator.
pub trait Executor: Send + Sync + 'static {
    /// Bytes of scratch memory required for the given dimension values.
    fn scratch_bytes(&self, dims: &DimValues) -> usize {
        let _ = dims;
        0
    }

    /// Submits the operator on `stream`. Must not allocate device memory.
    fn execute(&self, io: &mut OpIo<'_>, stream: &Stream, cx: &Context) -> anyhow::Result<()>;
}

impl<E: Executor + ?Sized> Executor for Box<E> {
    fn scratch_bytes(&self, dims: &DimValues) -> usize {
        (**self).scratch_bytes(dims)
    }

    fn execute(&self, io: &mut OpIo<'_>, stream: &Stream, cx: &Context) -> anyhow::Result<()> {
        (**self).execute(io, stream, cx)
    }
}

/// Read access to the symbolic graph.
#[derive(Debug, Clone, Default)]
pub struct GraphInfo {
    pub(crate) dims: SymDims,
    pub(crate) tensors: Vec<SymTensorData>,
    pub(crate) buffers: Vec<BufferInfo>,
}

impl GraphInfo {
    pub fn dims(&self) -> &SymDims {
        &self.dims
    }

    #[track_caller]
    pub fn tensor(&self, id: SymTensorId) -> &SymTensorData {
        &self.tensors[id.0]
    }

    pub fn buffer(&self, id: BufferId) -> &BufferInfo {
        &self.buffers[id.0]
    }

    /// The layout of a plain tensor; `None` for block-scaled tensors.
    pub fn layout(&self, id: SymTensorId) -> Option<&SymLayout> {
        self.tensor(id).layout()
    }

    pub fn ty(&self, id: SymTensorId) -> TensorTy {
        self.tensor(id).ty()
    }

    pub fn shape(&self, id: SymTensorId) -> &SymShape {
        self.tensor(id).shape()
    }

    /// Every dimension bound to its hint.
    pub fn hint_values(&self) -> DimValues {
        let mut values = DimValues::new();
        for (id, dim) in self.dims.iter() {
            values.bind(id, dim.hint);
        }
        values
    }
}

/// Context for [`Operator::build`].
pub struct BuildCx<'a> {
    pub cx: &'a Context,
    pub graph: &'a GraphInfo,
    /// Tensors of [`Operator::operands`], in order.
    pub operands: &'a [SymTensorId],
    pub outputs: &'a [SymTensorId],
    pub tuner: &'a Tuner<'a>,
    pub autotune: Autotune,
    pub op_name: &'a str,
    pub(crate) operand_access: &'a [Access],
    pub(crate) pool: &'a DeviceMemoryPool,
}

impl BuildCx<'_> {
    /// Tuning key for this operator, named after it.
    pub fn tune_key(&self, payload: &impl Serialize) -> anyhow::Result<TuneKey> {
        TuneKey::new(self.op_name, payload, self.cx)
    }

    /// Runs `f` on freshly allocated, zeroed buffers shaped like this operator's
    /// operands and outputs with all dimensions at their hints, plus `scratch_bytes`
    /// of scratch memory. Work is allocated on the context's default compute stream.
    pub fn hint_io<R>(
        &self,
        scratch_bytes: usize,
        f: impl FnOnce(&mut OpIo<'_>) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        let mut buffers = HintBuffers::new(self)?;
        buffers.ensure_scratch(scratch_bytes, self)?;
        let mut io = buffers.io(scratch_bytes, self)?;
        f(&mut io)
    }

    /// Tunes over candidates whose prepared state is an [`Executor`], measuring each
    /// on hint-shaped buffers (allocated only if measurement is needed).
    pub fn tune_executor<C, E>(
        &self,
        key: &TuneKey,
        candidates: Vec<C>,
        prepare: impl Fn(&C) -> anyhow::Result<E> + Sync,
    ) -> anyhow::Result<(C, E)>
    where
        C: Serialize + DeserializeOwned + Clone + Send + Sync,
        E: Executor,
    {
        let mut buffers: Option<HintBuffers> = None;
        self.tuner
            .tune(key, candidates, prepare, |executor, stream| {
                let buffers = match &mut buffers {
                    Some(buffers) => buffers,
                    None => buffers.insert(HintBuffers::new(self)?),
                };
                let scratch_bytes = executor.scratch_bytes(&buffers.dims);
                buffers.ensure_scratch(scratch_bytes, self)?;
                let mut io = buffers.io(scratch_bytes, self)?;
                executor.execute(&mut io, stream, self.cx)
            })
    }
}

/// Bound operands, outputs and scratch memory of one operator execution.
pub struct OpIo<'a> {
    /// One per operand, in order. [`Access::ReadWrite`] operands are [`Bound::Rw`].
    pub inputs: Vec<Bound<'a>>,
    pub outputs: Vec<TensorRefMut<'a>>,
    /// At least [`Executor::scratch_bytes`] bytes, if that is nonzero.
    pub scratch: Option<MemorySliceMut<&'a mut DeviceMemory>>,
    pub dims: &'a DimValues,
}

impl<'a> OpIo<'a> {
    /// Read-only view of an operand, which may be read-write.
    #[track_caller]
    pub fn input(&self, i: usize) -> TensorRef<'_> {
        self.inputs[i].as_ref()
    }

    /// A [`Access::ReadWrite`] operand.
    #[track_caller]
    pub fn input_mut(&mut self, i: usize) -> &mut TensorRefMut<'a> {
        self.inputs[i].as_mut()
    }

    #[track_caller]
    pub fn output(&mut self, i: usize) -> &mut TensorRefMut<'a> {
        &mut self.outputs[i]
    }

    #[track_caller]
    pub fn scratch(&mut self) -> &mut MemorySliceMut<&'a mut DeviceMemory> {
        self.scratch
            .as_mut()
            .expect("operator has no scratch memory")
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum TensorRef<'a> {
    Plain(TensorView<'a>),
    Mx(MxTensorView<'a>),
}

impl<'a> TensorRef<'a> {
    #[track_caller]
    pub fn plain(&self) -> &TensorView<'a> {
        match self {
            TensorRef::Plain(t) => t,
            TensorRef::Mx(_) => panic!("expected a plain tensor, got a block-scaled tensor"),
        }
    }

    #[track_caller]
    pub fn mx(&self) -> &MxTensorView<'a> {
        match self {
            TensorRef::Mx(t) => t,
            TensorRef::Plain(_) => panic!("expected a block-scaled tensor, got a plain tensor"),
        }
    }

    #[track_caller]
    pub fn into_plain(self) -> TensorView<'a> {
        match self {
            TensorRef::Plain(t) => t,
            TensorRef::Mx(_) => panic!("expected a plain tensor, got a block-scaled tensor"),
        }
    }

    #[track_caller]
    pub fn into_mx(self) -> MxTensorView<'a> {
        match self {
            TensorRef::Mx(t) => t,
            TensorRef::Plain(_) => panic!("expected a block-scaled tensor, got a plain tensor"),
        }
    }

    pub fn ty(&self) -> TensorTy {
        match self {
            TensorRef::Plain(t) => TensorTy::Plain(t.element_type()),
            TensorRef::Mx(t) => TensorTy::BlockScaled {
                format: t.format(),
                dim: t.mx_dim(),
            },
        }
    }
}

impl<'a> From<TensorView<'a>> for TensorRef<'a> {
    fn from(t: TensorView<'a>) -> Self {
        Self::Plain(t)
    }
}

impl<'a> From<MxTensorView<'a>> for TensorRef<'a> {
    fn from(t: MxTensorView<'a>) -> Self {
        Self::Mx(t)
    }
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum TensorRefMut<'a> {
    Plain(TensorViewMut<'a>),
    Mx(MxTensorViewMut<'a>),
}

impl<'a> TensorRefMut<'a> {
    pub fn as_ref(&self) -> TensorRef<'_> {
        match self {
            TensorRefMut::Plain(t) => TensorRef::Plain(t.view()),
            TensorRefMut::Mx(t) => TensorRef::Mx(t.view()),
        }
    }

    #[track_caller]
    pub fn plain(&mut self) -> &mut TensorViewMut<'a> {
        match self {
            TensorRefMut::Plain(t) => t,
            TensorRefMut::Mx(_) => panic!("expected a plain tensor, got a block-scaled tensor"),
        }
    }

    #[track_caller]
    pub fn mx(&mut self) -> &mut MxTensorViewMut<'a> {
        match self {
            TensorRefMut::Mx(t) => t,
            TensorRefMut::Plain(_) => {
                panic!("expected a block-scaled tensor, got a plain tensor")
            }
        }
    }

    pub fn ty(&self) -> TensorTy {
        self.as_ref().ty()
    }
}

impl<'a> From<TensorViewMut<'a>> for TensorRefMut<'a> {
    fn from(t: TensorViewMut<'a>) -> Self {
        Self::Plain(t)
    }
}

impl<'a> From<MxTensorViewMut<'a>> for TensorRefMut<'a> {
    fn from(t: MxTensorViewMut<'a>) -> Self {
        Self::Mx(t)
    }
}

/// A bound operand.
#[derive(Debug)]
pub enum Bound<'a> {
    Ro(TensorRef<'a>),
    Rw(TensorRefMut<'a>),
}

impl<'a> Bound<'a> {
    pub fn as_ref(&self) -> TensorRef<'_> {
        match self {
            Bound::Ro(t) => t.clone(),
            Bound::Rw(t) => t.as_ref(),
        }
    }

    #[track_caller]
    #[allow(clippy::should_implement_trait)]
    pub fn as_mut(&mut self) -> &mut TensorRefMut<'a> {
        match self {
            Bound::Rw(t) => t,
            Bound::Ro(_) => panic!("operand is read-only"),
        }
    }
}

/// Device memory of one graph buffer, borrowed for one operator execution.
pub(crate) enum Mem<'a> {
    Ro(MemorySlice<&'a DeviceMemory>),
    Rw(MemorySliceMut<&'a mut DeviceMemory>),
}

pub(crate) type MemMap<'a> = HashMap<BufferId, Mem<'a>>;

/// Distinct buffers used by an op, and whether each is written.
pub(crate) fn op_buffer_access(
    graph: &GraphInfo,
    operands: &[SymTensorId],
    access: &[Access],
    outputs: &[SymTensorId],
) -> Vec<(BufferId, bool)> {
    let mut buffers: Vec<(BufferId, bool)> = Vec::new();
    let tensors = operands
        .iter()
        .zip(access)
        .map(|(&t, &a)| (t, a == Access::ReadWrite))
        .chain(outputs.iter().map(|&t| (t, true)));
    for (tensor, write) in tensors {
        for buffer in graph.tensor(tensor).buffers() {
            match buffers.iter_mut().find(|(b, _)| *b == buffer) {
                Some((_, w)) => *w |= write,
                None => buffers.push((buffer, write)),
            }
        }
    }
    buffers
}

/// Evaluates a plain tensor's layout, returning its view and byte offset into the buffer.
fn eval_plain(
    layout: &SymLayout,
    ty: ElementType,
    dims: &DimValues,
) -> anyhow::Result<(VirtTensorView, usize)> {
    let (shape, strides, offset) = layout.eval(dims)?;
    let virt = VirtTensorView::with_strides(shape, strides, ty);
    if shape.element_count() == 0 {
        return Ok((virt, 0));
    }
    let offset_bits = offset
        .checked_mul(ty.bits_per_element())
        .ok_or_else(|| anyhow!("tensor offset overflows"))?;
    ensure!(
        offset_bits.is_multiple_of(8),
        "tensor offset of {offset} {ty:?} elements is not byte-aligned"
    );
    Ok((virt, offset_bits / 8))
}

fn check_fits(virt: &VirtTensorView, offset: usize, len: usize) -> anyhow::Result<()> {
    let end = offset
        .checked_add(virt.required_bytes())
        .ok_or_else(|| anyhow!("tensor extent overflows"))?;
    ensure!(
        end <= len,
        "tensor with shape {:?} and strides {:?} at byte offset {offset} needs {end} bytes, but its buffer has {len}",
        virt.shape(),
        virt.strides()
    );
    Ok(())
}

fn view_ro<'a>(
    mem: MemorySlice<&'a DeviceMemory>,
    virt: VirtTensorView,
    offset: usize,
    cx: &Context,
) -> anyhow::Result<TensorView<'a>> {
    check_fits(&virt, offset, mem.len())?;
    let (start, end) = (mem.offset() + offset, mem.offset() + mem.len());
    Ok(TensorView::from_parts(
        virt,
        mem.into_inner().slice(start..end),
        cx,
    ))
}

fn view_rw<'a>(
    mem: MemorySliceMut<&'a mut DeviceMemory>,
    virt: VirtTensorView,
    offset: usize,
    cx: &Context,
) -> anyhow::Result<TensorViewMut<'a>> {
    check_fits(&virt, offset, mem.len())?;
    let (start, end) = (mem.offset() + offset, mem.offset() + mem.len());
    Ok(TensorViewMut::from_parts(
        virt,
        mem.into_inner().slice_mut(start..end),
        cx,
    ))
}

fn take_ro<'a>(mems: &MemMap<'a>, buffer: BufferId) -> MemorySlice<&'a DeviceMemory> {
    match &mems[&buffer] {
        Mem::Ro(mem) => *mem,
        Mem::Rw(_) => unreachable!("buffer {buffer:?} is both read and written by one op"),
    }
}

fn take_rw<'a>(mems: &mut MemMap<'a>, buffer: BufferId) -> MemorySliceMut<&'a mut DeviceMemory> {
    match mems.remove(&buffer) {
        Some(Mem::Rw(mem)) => mem,
        _ => unreachable!("buffer {buffer:?} is written through multiple operands"),
    }
}

pub(crate) fn bind_ro<'a>(
    tensor: &SymTensorData,
    mems: &MemMap<'a>,
    dims: &DimValues,
    cx: &Context,
) -> anyhow::Result<TensorRef<'a>> {
    Ok(match tensor {
        SymTensorData::Plain { buffer, layout, ty } => {
            let (virt, offset) = eval_plain(layout, *ty, dims)?;
            TensorRef::Plain(view_ro(take_ro(mems, *buffer), virt, offset, cx)?)
        }
        SymTensorData::BlockScaled {
            values,
            scales,
            format,
            dim,
            shape,
        } => {
            let shape = shape.eval(dims)?;
            let layout = MxLayout::new(shape, *format, *dim)?;
            TensorRef::Mx(MxTensorView::from_parts(
                view_ro(take_ro(mems, *values), layout.values, 0, cx)?,
                view_ro(take_ro(mems, *scales), layout.scale_factors, 0, cx)?,
                *format,
                *dim,
                shape,
            )?)
        }
    })
}

pub(crate) fn bind_rw<'a>(
    tensor: &SymTensorData,
    mems: &mut MemMap<'a>,
    dims: &DimValues,
    cx: &Context,
) -> anyhow::Result<TensorRefMut<'a>> {
    Ok(match tensor {
        SymTensorData::Plain { buffer, layout, ty } => {
            let (virt, offset) = eval_plain(layout, *ty, dims)?;
            TensorRefMut::Plain(view_rw(take_rw(mems, *buffer), virt, offset, cx)?)
        }
        SymTensorData::BlockScaled {
            values,
            scales,
            format,
            dim,
            shape,
        } => {
            let shape = shape.eval(dims)?;
            let layout = MxLayout::new(shape, *format, *dim)?;
            TensorRefMut::Mx(MxTensorViewMut::from_parts(
                view_rw(take_rw(mems, *values), layout.values, 0, cx)?,
                view_rw(take_rw(mems, *scales), layout.scale_factors, 0, cx)?,
                *format,
                *dim,
                shape,
            )?)
        }
    })
}

/// Binds an op's operands and outputs to the buffers in `mems`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bind_io<'a>(
    graph: &GraphInfo,
    operands: &[SymTensorId],
    access: &[Access],
    outputs: &[SymTensorId],
    mut mems: MemMap<'a>,
    scratch: Option<MemorySliceMut<&'a mut DeviceMemory>>,
    dims: &'a DimValues,
    cx: &Context,
) -> anyhow::Result<OpIo<'a>> {
    let mut inputs = Vec::with_capacity(operands.len());
    for (i, (&tensor, &access)) in operands.iter().zip(access).enumerate() {
        let tensor = graph.tensor(tensor);
        let bound = match access {
            Access::ReadOnly => bind_ro(tensor, &mems, dims, cx).map(Bound::Ro),
            Access::ReadWrite => bind_rw(tensor, &mut mems, dims, cx).map(Bound::Rw),
        };
        inputs.push(bound.with_context(|| format!("operand {i}"))?);
    }
    let outputs = outputs
        .iter()
        .enumerate()
        .map(|(i, &tensor)| {
            bind_rw(graph.tensor(tensor), &mut mems, dims, cx)
                .with_context(|| format!("output {i}"))
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(OpIo {
        inputs,
        outputs,
        scratch,
        dims,
    })
}

/// Zeroed, hint-sized buffers for one op, for autotuning.
struct HintBuffers {
    dims: DimValues,
    buffers: Vec<(BufferId, bool, DeviceMemory)>,
    scratch: Option<DeviceMemory>,
}

impl HintBuffers {
    fn new(bcx: &BuildCx<'_>) -> anyhow::Result<Self> {
        let dims = bcx.graph.hint_values();
        let stream = bcx.cx.default_compute_stream();
        let buffers = op_buffer_access(bcx.graph, bcx.operands, bcx.operand_access, bcx.outputs)
            .into_iter()
            .map(|(buffer, write)| {
                let bytes = bcx.graph.buffer(buffer).bytes.eval(&dims)?;
                Ok((buffer, write, bcx.pool.alloc_zeroed_on(bytes, stream)?))
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Self {
            dims,
            buffers,
            scratch: None,
        })
    }

    fn ensure_scratch(&mut self, bytes: usize, bcx: &BuildCx<'_>) -> anyhow::Result<()> {
        if bytes > 0 && self.scratch.as_ref().is_none_or(|s| s.len() < bytes) {
            self.scratch = Some(
                bcx.pool
                    .alloc_zeroed_on(bytes, bcx.cx.default_compute_stream())?,
            );
        }
        Ok(())
    }

    fn io<'a>(&'a mut self, scratch_bytes: usize, bcx: &BuildCx<'_>) -> anyhow::Result<OpIo<'a>> {
        let mems = self
            .buffers
            .iter_mut()
            .map(|(buffer, write, mem)| {
                let mem = if *write {
                    Mem::Rw(mem.slice_mut(..))
                } else {
                    Mem::Ro(mem.slice(..))
                };
                (*buffer, mem)
            })
            .collect();
        let scratch = match (&mut self.scratch, scratch_bytes) {
            (_, 0) => None,
            (Some(scratch), n) => Some(scratch.slice_mut(..n)),
            (None, _) => bail!("scratch memory was not allocated"),
        };
        bind_io(
            bcx.graph,
            bcx.operands,
            bcx.operand_access,
            bcx.outputs,
            mems,
            scratch,
            &self.dims,
            bcx.cx,
        )
    }
}
