use crate::error::Result;
use nvvm_builder::nvvm_sys::Arch;
use serde::{Serialize, de::DeserializeOwned};
use skycuda::{function::Function, stream::Stream};
use std::sync::Arc;

pub mod norm;

pub trait KernelGenerator: Sized {
    type Config: Eq + Serialize + DeserializeOwned;
    type ExecArgs<'a>;

    fn generate_kernel(&self, config: Self::Config, arch: Arch) -> Result<BuiltKernel<Self>>;

    /// # Safety
    /// Arguments may contain device pointers, whose layouts must match
    /// the kernel generator's expectations.
    unsafe fn execute_kernel<'a>(
        &self,
        kernel: &BuiltKernel<Self>,
        args: Self::ExecArgs<'a>,
        stream: &Stream,
    ) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct BuiltKernel<G: KernelGenerator> {
    config: G::Config,
    ptx: Arc<[u8]>,
    arch: Arch,
}

pub struct LoadedKernel<G: KernelGenerator> {
    built: BuiltKernel<G>,
    func: Function,
}
