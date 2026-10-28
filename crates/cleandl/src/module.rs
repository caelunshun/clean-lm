//! Skeleton for deep learning modules.
//!
//! A ModuleConfig specifies the shape and layout of input tensors, and other parameters
//! as needed by the module. A ModuleExecutor contains required pre-compiled kernels (e.g. cuDNN graphs)
//! and workspace allocations and can be reused across multiple layers with the same
//! parameters (so long as it is not in use multiple times in parallel). A Module
//! is an instance of the module which holds tensor data. By convention, Module execution
//! should not allocate new tensors; output tensors should be passed as arguments,
//! and scratch tensors reused in the Executor workspace.

use std::sync::Arc;

use singe_cuda::{context::Context, stream::Stream};

pub trait ModuleConfig {
    type Module: Module;

    fn make_executor(
        &self,
        cx: &Arc<Context>,
    ) -> anyhow::Result<<Self::Module as Module>::Executor>;
    fn make_module_zeroed(&self, cx: &Arc<Context>) -> anyhow::Result<Self::Module>;
}

pub trait Module: 'static {
    type Executor: 'static;
    type InferenceOperands<'a>;

    fn inference(
        &self,
        operands: Self::InferenceOperands<'_>,
        streams: &[Stream],
        executor: &mut Self::Executor,
    ) -> anyhow::Result<()>;
}
