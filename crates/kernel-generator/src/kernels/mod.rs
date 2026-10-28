use crate::error::Result;
use nvvm_builder::nvvm_sys::Arch;
use serde::{Serialize, de::DeserializeOwned};
use skycuda::stream::Stream;

pub mod norm;

pub trait KernelGenerator: Sized {
    type Config: Eq + Serialize + DeserializeOwned;
    type BuiltKernel: Serialize + DeserializeOwned + HasCubin;
    type ExecArgs<'a>;

    fn generate_kernel(&self, config: Self::Config, arch: Arch) -> Result<Self::BuiltKernel>;

    /// # Safety
    /// Arguments may contain device pointers, whose layouts must match
    /// the kernel generator's expectations.
    unsafe fn execute_kernel<'a>(
        &self,
        kernel: &Self::BuiltKernel,
        args: Self::ExecArgs<'a>,
        stream: &Stream,
    ) -> Result<()>;
}

pub trait HasCubin {
    fn cubin(&self) -> &[u8];
}
