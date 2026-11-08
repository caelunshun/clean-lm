//! Buffer liveness, operator building, and execution.

use super::{ExecGraph, GraphOutput, Input, graph::Graph, op::Executor, tune::Autotune};
use crate::context::Context;
use skycuda::{device_memory_pool::DeviceMemoryPool, stream::Stream};

#[derive(Debug)]
pub(crate) struct Plan {}

impl Plan {
    pub fn new(_graph: &Graph) -> Self {
        todo!()
    }
}

pub(crate) fn build_executors(
    _graph: &Graph,
    _cx: &Context,
    _autotune: Autotune,
    _pool: &DeviceMemoryPool,
) -> anyhow::Result<Vec<Box<dyn Executor>>> {
    todo!()
}

pub(crate) fn execute(
    _graph: &ExecGraph,
    _inputs: &mut [Input<'_>],
    _stream: &Stream,
    _pool: &DeviceMemoryPool,
) -> anyhow::Result<Vec<GraphOutput>> {
    todo!()
}
