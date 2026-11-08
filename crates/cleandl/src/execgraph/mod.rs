mod graph;
pub mod jit;
mod op;
pub mod ops;
mod plan;
pub mod sym;
pub mod tune;

use crate::{
    context::Context,
    mx_tensor::{MxTensor, MxTensorView, MxTensorViewMut},
    tensor::{Tensor, TensorView, TensorViewMut},
};
pub use graph::{
    BufferId, BufferInfo, BufferPart, BufferSource, Graph, GraphBuilder, InputInfo, IntoSymShape,
    SymTensor, SymTensorData, SymTensorId,
};
pub use op::{
    Access, Bound, BuildCx, Executor, GraphInfo, Init, OpIo, Operand, Operator, OutputSpec,
    TensorRef, TensorRefMut, TensorTy,
};
use skycuda::{device_memory_pool::DeviceMemoryPool, stream::Stream};
use std::fmt;
use tune::Autotune;

#[derive(Debug, Clone, Default)]
pub struct BuildOptions {
    pub autotune: Autotune,
    /// Pool for build-time allocations such as autotuning buffers.
    /// Defaults to the context's pool.
    pub pool: Option<DeviceMemoryPool>,
}

/// A graph input bound at execution time. Mutable inputs must be given as [`Input::Rw`].
#[derive(Debug)]
pub enum Input<'a> {
    Ro(TensorRef<'a>),
    Rw(TensorRefMut<'a>),
}

impl<'a> From<TensorView<'a>> for Input<'a> {
    fn from(t: TensorView<'a>) -> Self {
        Self::Ro(TensorRef::Plain(t))
    }
}

impl<'a> From<TensorViewMut<'a>> for Input<'a> {
    fn from(t: TensorViewMut<'a>) -> Self {
        Self::Rw(TensorRefMut::Plain(t))
    }
}

impl<'a> From<MxTensorView<'a>> for Input<'a> {
    fn from(t: MxTensorView<'a>) -> Self {
        Self::Ro(TensorRef::Mx(t))
    }
}

impl<'a> From<MxTensorViewMut<'a>> for Input<'a> {
    fn from(t: MxTensorViewMut<'a>) -> Self {
        Self::Rw(TensorRefMut::Mx(t))
    }
}

impl<'a> From<&'a Tensor> for Input<'a> {
    fn from(t: &'a Tensor) -> Self {
        t.view().into()
    }
}

impl<'a> From<&'a mut Tensor> for Input<'a> {
    fn from(t: &'a mut Tensor) -> Self {
        t.view_mut().into()
    }
}

impl<'a> From<&'a MxTensor> for Input<'a> {
    fn from(t: &'a MxTensor) -> Self {
        t.view().into()
    }
}

impl<'a> From<&'a mut MxTensor> for Input<'a> {
    fn from(t: &'a mut MxTensor) -> Self {
        t.view_mut().into()
    }
}

/// A graph output, owning its storage.
#[derive(Debug)]
pub enum GraphOutput {
    Plain(Tensor),
    Mx(MxTensor),
}

impl GraphOutput {
    #[track_caller]
    pub fn into_plain(self) -> Tensor {
        match self {
            GraphOutput::Plain(t) => t,
            GraphOutput::Mx(_) => panic!("expected a plain output, got a block-scaled output"),
        }
    }

    #[track_caller]
    pub fn into_mx(self) -> MxTensor {
        match self {
            GraphOutput::Mx(t) => t,
            GraphOutput::Plain(_) => panic!("expected a block-scaled output, got a plain output"),
        }
    }
}

/// A graph with built executors, ready to run.
pub struct ExecGraph {
    graph: Graph,
    executors: Vec<Box<dyn Executor>>,
    plan: plan::Plan,
    cx: Context,
}

impl ExecGraph {
    /// Builds all operators in parallel.
    pub fn build(graph: Graph, cx: &Context, opts: BuildOptions) -> anyhow::Result<Self> {
        let pool = opts.pool.unwrap_or_else(|| cx.device_memory_pool().clone());
        let executors = plan::build_executors(&graph, cx, opts.autotune, &pool)?;
        let plan = plan::Plan::new(&graph);
        Ok(Self {
            graph,
            executors,
            plan,
            cx: cx.clone(),
        })
    }

    /// Runs the graph on `stream`, allocating intermediates and outputs from `pool`.
    /// `inputs` are in declaration order.
    pub fn execute(
        &self,
        inputs: &mut [Input<'_>],
        stream: &Stream,
        pool: &DeviceMemoryPool,
    ) -> anyhow::Result<Vec<GraphOutput>> {
        plan::execute(self, inputs, stream, pool)
    }

    /// [`Self::execute`] with the context's memory pool.
    pub fn execute_default(
        &self,
        inputs: &mut [Input<'_>],
        stream: &Stream,
    ) -> anyhow::Result<Vec<GraphOutput>> {
        self.execute(inputs, stream, self.cx.device_memory_pool())
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn inputs(&self) -> &[InputInfo] {
        self.graph.inputs()
    }

    pub fn num_outputs(&self) -> usize {
        self.graph.outputs().len()
    }
}

impl fmt::Debug for ExecGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecGraph")
            .field("graph", &self.graph)
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}
