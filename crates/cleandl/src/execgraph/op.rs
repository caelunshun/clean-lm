//! The interface between the execution graph engine and operator implementations.

use super::{
    bind::HintBuffers,
    graph::{BufferId, BufferInfo, SymTensorData, SymTensorId},
    sym::{DimValues, SymDims, SymLayout, SymShape},
    tune::{Autotune, TuneKey, Tuner},
};
use crate::{
    context::Context,
    mx_tensor::{MxDim, MxFormat, MxTensorView, MxTensorViewMut},
    tensor::{TensorView, TensorViewMut},
};
use anyhow::ensure;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use skycuda::{
    device_memory::DeviceMemory, device_memory_pool::DeviceMemoryPool,
    memory_slice::MemorySliceMut, stream::Stream,
};
use skytens_core::data_type::ElementType;

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
///
/// With [`Init::Uninit`], the buffer's contents are undefined: the operator must
/// write every element of the output before anything reads it (including itself).
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
///
/// Memory guarantees for operator authors: every buffer (graph-owned, graph input,
/// or hint buffer) starts at a 16-byte aligned device address, so alignment proofs
/// from [`SymLayout::offset_is_aligned`] and [`SymLayout::stride_is_aligned`] with
/// `align_bytes <= 16` hold for the actual addresses of bound views.
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
    ///
    /// Outputs declared [`Init::Uninit`] and the scratch memory have undefined
    /// contents: they must be written before they are read.
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
    /// of scratch memory.
    ///
    /// The buffers are allocated and zeroed on the context's default compute stream
    /// while holding [`Context::gpu_lock`], and freed under it, so this never overlaps
    /// another op's autotuning measurement. `f` runs without the lock and may call
    /// [`Tuner::tune`]. Errors if called while this thread holds the GPU lock (e.g.
    /// inside a `run` closure of [`Tuner::tune`]); use [`Self::tune_with_io`] there.
    /// Buffers live for the duration of `f`, so concurrent builds calling this hold
    /// several sets at once; prefer [`Self::tune_with_io`] for large operators.
    ///
    /// Since `f` runs unlocked, it must submit GPU work only through [`Tuner::tune`]
    /// (whose `run` closure executes under the lock); work submitted directly from
    /// `f` may overlap other ops' measurements.
    pub fn hint_io<R>(
        &self,
        scratch_bytes: usize,
        f: impl FnOnce(&mut OpIo<'_>) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        ensure!(
            !self.cx.holds_gpu_lock(),
            "{}: hint_io cannot be called while holding the GPU lock; use tune_with_io",
            self.op_name
        );
        let mut buffers = {
            let _gpu = self.cx.gpu_lock();
            let mut buffers = HintBuffers::new(self)?;
            buffers.ensure_scratch(scratch_bytes, self)?;
            self.cx.default_compute_stream().block_on()?;
            buffers
        };
        let result = buffers
            .io(scratch_bytes, self)
            .and_then(|mut io| f(&mut io));
        let _gpu = self.cx.gpu_lock();
        drop(buffers);
        result
    }

    /// [`Tuner::tune`], with each run given hint-shaped buffers like [`Self::hint_io`]
    /// and `scratch_bytes(prepared, hint_dims)` bytes of scratch.
    ///
    /// Buffers are allocated and zeroed lazily inside the tuner's locked measurement
    /// region (during a candidate's warmup run, which is synchronized before timing
    /// starts), so they exist only if measurement is needed, at most one tuning op
    /// holds them at a time, and no build work overlaps another measurement.
    pub fn tune_with_io<C, P>(
        &self,
        key: &TuneKey,
        candidates: Vec<C>,
        prepare: impl Fn(&C) -> anyhow::Result<P> + Sync,
        scratch_bytes: impl Fn(&P, &DimValues) -> usize,
        mut run: impl FnMut(&P, &mut OpIo<'_>, &Stream) -> anyhow::Result<()>,
    ) -> anyhow::Result<(C, P)>
    where
        C: Serialize + DeserializeOwned + Clone + Send + Sync,
        P: Send,
    {
        let mut buffers: Option<HintBuffers> = None;
        let result = self
            .tuner
            .tune(key, candidates, prepare, |prepared, stream| {
                let buffers = match &mut buffers {
                    Some(buffers) => buffers,
                    None => buffers.insert(HintBuffers::new(self)?),
                };
                let scratch_bytes = scratch_bytes(prepared, &buffers.dims);
                buffers.ensure_scratch(scratch_bytes, self)?;
                let mut io = buffers.io(scratch_bytes, self)?;
                run(prepared, &mut io, stream)
            });
        if buffers.is_some() {
            let _gpu = self.cx.gpu_lock();
            drop(buffers);
        }
        result
    }

    /// [`Self::tune_with_io`] for candidates whose prepared state is an [`Executor`].
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
        self.tune_with_io(
            key,
            candidates,
            prepare,
            |executor, dims| executor.scratch_bytes(dims),
            |executor, io, stream| executor.execute(io, stream, self.cx),
        )
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

    /// A [`Access::ReadOnly`] operand. Not tied to the borrow of `self`, so it can be
    /// held together with outputs and scratch.
    #[track_caller]
    pub fn input_ro(&self, i: usize) -> TensorRef<'a> {
        self.inputs[i].ro().clone()
    }

    /// Operands, outputs and scratch, borrowed together.
    #[allow(clippy::type_complexity)]
    pub fn split(
        &mut self,
    ) -> (
        &mut [Bound<'a>],
        &mut [TensorRefMut<'a>],
        Option<&mut MemorySliceMut<&'a mut DeviceMemory>>,
    ) {
        (&mut self.inputs, &mut self.outputs, self.scratch.as_mut())
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
    /// A [`Access::ReadOnly`] operand, with the full lifetime of the binding.
    #[track_caller]
    pub fn ro(&self) -> &TensorRef<'a> {
        match self {
            Bound::Ro(t) => t,
            Bound::Rw(_) => panic!("operand is read-write; use as_ref"),
        }
    }

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
