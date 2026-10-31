//! Helper for building kernels with dynamic arguments at runtime.
//!
//! Kernels can be built either as NVVM->PTX (via `nvvm-builder`)
//! or TileIR (via `tileir-builder`). Kernels can also be serialized.

use derive_where::derive_where;
use serde::{Deserialize, Serialize};
use skycuda::{
    device::Device,
    device_memory::{
        DevicePtrGuard, DevicePtrGuardMut, DeviceVisibleMemory, DeviceVisibleMemoryMut,
    },
    function::Function,
    launch::{KernelArg, LaunchBuilder},
    stream::Stream,
};
use std::{collections::HashMap, fmt, marker::PhantomData, sync::Arc};

pub mod nvvm;
pub mod tileir;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] skycuda::error::Error),
    #[error(
        "kernel targets sm_{}{}, but the device has compute capability {}.{}",
        kernel.0, kernel.1, device.0, device.1
    )]
    ArchMismatch {
        kernel: (u32, u32),
        device: (u32, u32),
    },
    #[error("unrecognized cubin architecture `{0}`")]
    UnknownArch(String),
    #[error(transparent)]
    NvvmBuild(#[from] nvvm_builder::BuildError),
    #[error(transparent)]
    NvvmCompile(#[from] nvvm_builder::CompileError),
    #[error(transparent)]
    Ptxas(#[from] nvvm_builder::tools::ToolError),
    #[error(transparent)]
    TileirBuild(#[from] tileir_builder::BuildError),
    #[error(transparent)]
    TileirCompile(#[from] tileir_builder::CompileError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive_where(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Param<T> {
    index: usize,
    _marker: PhantomData<T>,
}

impl<T> Param<T> {
    fn new(index: usize) -> Self {
        Self {
            index,
            _marker: PhantomData,
        }
    }
}

/// How a logical parameter is expanded into launch arguments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParamAbi {
    /// A single launch argument.
    Scalar,
    /// A Tile IR array. a pointer, then each dynamic dimension, then
    /// each dynamic stride. `None` entries are dynamic.
    Array {
        shape: Vec<Option<i64>>,
        strides: Vec<Option<i64>>,
        index_i64: bool,
        alignment: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kernel {
    exact_target_sm_major: u32,
    exact_target_sm_minor: u32,
    cubin: Arc<[u8]>,
    function_name: String,
    params: Vec<ParamAbi>,
    dynamic_smem_bytes: usize,
    block_size: (u32, u32, u32),
    cluster_size: (u32, u32, u32),
}

impl Kernel {
    /// # Safety
    /// Kernel must be a valid cubin. If it was deserizalized
    /// then it must have come from a trusted source.
    pub unsafe fn load(&self, device: &Device) -> Result<LoadedKernel> {
        let device_cc = (
            device.compute_capability_major(),
            device.compute_capability_minor(),
        );
        let kernel_cc = (self.exact_target_sm_major, self.exact_target_sm_minor);
        if device_cc != kernel_cc {
            return Err(Error::ArchMismatch {
                kernel: kernel_cc,
                device: device_cc,
            });
        }

        let module = unsafe { device.load_module_cubin(&self.cubin)? };
        let loaded_function = module.get_function(&self.function_name)?;

        if self.dynamic_smem_bytes > 0 {
            loaded_function.set_max_dynamic_smem(self.dynamic_smem_bytes)?;
        }

        Ok(LoadedKernel {
            kernel: self.clone(),
            loaded_function,
        })
    }

    pub fn cubin(&self) -> &[u8] {
        &self.cubin
    }
}

/// Parses a cubin target such as `sm_103a` into `(10, 3)`.
fn parse_sm_arch(arch: &str) -> Result<(u32, u32)> {
    let digits = arch
        .strip_prefix("sm_")
        .map(|s| s.trim_end_matches(|c: char| c.is_ascii_alphabetic()))
        .and_then(|s| s.parse::<u32>().ok())
        .ok_or_else(|| Error::UnknownArch(arch.to_owned()))?;
    Ok((digits / 10, digits % 10))
}

#[derive(Debug, Clone)]
pub struct LoadedKernel {
    kernel: Kernel,
    loaded_function: Function,
}

impl LoadedKernel {
    /// # Safety
    /// See LaunchBuilder::launch.
    pub unsafe fn launch<'stream, 'args>(
        &'stream self,
        mut args: ArgBundle<'args>,
        grid_size: (u32, u32, u32),
        stream: &'stream Stream,
    ) -> Result<()>
    where
        'stream: 'args,
    {
        let mut launch_builder = stream.launch_builder(&self.loaded_function);

        launch_builder.block_size(
            self.kernel.block_size.0,
            self.kernel.block_size.1,
            self.kernel.block_size.2,
        )?;
        launch_builder.dynamic_smem(self.kernel.dynamic_smem_bytes)?;

        for (param_idx, abi) in self.kernel.params.iter().enumerate() {
            (args
                .add_param_to_builder
                .remove(&param_idx)
                .unwrap_or_else(|| panic!("missing parameter at index {param_idx}")))(
                &mut launch_builder,
                abi,
            )?;
        }

        unsafe {
            launch_builder.launch(grid_size.0, grid_size.1, grid_size.2)?;
        }

        Ok(())
    }
}

/// Lazily loads a kernel once per device, handing out references that
/// live as long as the cache.
#[derive(Default)]
pub struct LoadedKernelCache {
    loaded: boxcar::Vec<LoadedKernel>,
}

impl LoadedKernelCache {
    /// # Safety
    /// See [`Kernel::load`].
    pub unsafe fn get_or_load(&self, kernel: &Kernel, device: &Device) -> Result<&LoadedKernel> {
        if let Some((_, loaded)) = self
            .loaded
            .iter()
            .find(|(_, loaded)| loaded.loaded_function.device() == device)
        {
            return Ok(loaded);
        }
        let index = self.loaded.push(unsafe { kernel.load(device)? });
        Ok(&self.loaded[index])
    }
}

impl Clone for LoadedKernelCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl fmt::Debug for LoadedKernelCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoadedKernelCache")
            .field("num_loaded", &self.loaded.count())
            .finish()
    }
}

type AddParamFn<'a> =
    Box<dyn FnOnce(&mut LaunchBuilder<'a>, &ParamAbi) -> Result<(), skycuda::error::Error> + 'a>;

#[derive(Default)]
pub struct ArgBundle<'a> {
    add_param_to_builder: HashMap<usize, AddParamFn<'a>>,
}

impl<'a> ArgBundle<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set<T, U>(&mut self, param: Param<T>, value: U)
    where
        U: PassableAs<'a, T> + 'a,
    {
        self.add_param_to_builder.insert(
            param.index,
            Box::new(|launch_builder, abi| value.push(abi, launch_builder)),
        );
    }
}

/// # Safety
/// Must be valid to pass `Self` as an argument for a parameter of type `T`.
pub unsafe trait PassableAs<'a, T> {
    /// Adds the launch arguments for this value, as laid out by `abi`.
    fn push(
        self,
        abi: &ParamAbi,
        launch_builder: &mut LaunchBuilder<'a>,
    ) -> Result<(), skycuda::error::Error>;
}

/// Implements [`PassableAs::push`] for a value passed as one launch argument.
macro_rules! push_scalar {
    () => {
        fn push(
            self,
            abi: &$crate::ParamAbi,
            launch_builder: &mut ::skycuda::launch::LaunchBuilder<'a>,
        ) -> Result<(), ::skycuda::error::Error> {
            assert_eq!(
                abi,
                &$crate::ParamAbi::Scalar,
                "expected a scalar parameter"
            );
            launch_builder.arg(self)?;
            Ok(())
        }
    };
}
pub(crate) use push_scalar;

macro_rules! scalar_passable {
    ($ty:ty as $as:ty) => {
        unsafe impl<'a> PassableAs<'a, $as> for $ty {
            push_scalar!();
        }
    };
}

scalar_passable!(u8 as u8);
scalar_passable!(u16 as u16);
scalar_passable!(u32 as u32);
scalar_passable!(u64 as u64);

scalar_passable!(i8 as i8);
scalar_passable!(i16 as i16);
scalar_passable!(i32 as i32);
scalar_passable!(i64 as i64);

scalar_passable!(nvvm_builder::f16 as nvvm_builder::f16);
scalar_passable!(nvvm_builder::bf16 as nvvm_builder::bf16);
scalar_passable!(f32 as f32);
scalar_passable!(f64 as f64);

/// Keeps a device access alive until the launch is issued.
pub trait Held {}
impl<T> Held for T {}

type Access<'a> =
    Box<dyn FnOnce(&'a Stream) -> Result<(u64, Box<dyn Held + 'a>), skycuda::error::Error> + 'a>;

/// A type-erased device pointer to `T` elements, or null.
pub struct PtrArg<'a, T> {
    access: Option<Access<'a>>,
    _marker: PhantomData<fn() -> T>,
}

impl<'a, T> PtrArg<'a, T> {
    /// A pointer the kernel must not write through.
    pub fn new<M: DeviceVisibleMemory + 'a>(memory: &'a M) -> Self {
        Self {
            access: Some(Box::new(move |stream| {
                let guard = memory.access_on(stream)?;
                let ptr = unsafe { DevicePtrGuard::ptr(&guard) };
                Ok((ptr, Box::new(guard) as Box<dyn Held + 'a>))
            })),
            _marker: PhantomData,
        }
    }

    /// A pointer the kernel may write through.
    pub fn new_mut<M: DeviceVisibleMemoryMut + 'a>(memory: &'a mut M) -> Self {
        Self {
            access: Some(Box::new(move |stream| {
                let mut guard = memory.access_mut_on(stream)?;
                let ptr = unsafe { DevicePtrGuardMut::ptr(&mut guard) };
                Ok((ptr, Box::new(guard) as Box<dyn Held + 'a>))
            })),
            _marker: PhantomData,
        }
    }

    pub fn null() -> Self {
        Self {
            access: None,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> KernelArg<'a> for PtrArg<'a, T> {
    type Repr = u64;
    type Guard = Option<Box<dyn Held + 'a>>;

    unsafe fn into_arg(
        self,
        stream: &'a Stream,
    ) -> Result<(Self::Repr, Self::Guard), skycuda::error::Error> {
        match self.access {
            Some(access) => {
                let (ptr, guard) = access(stream)?;
                Ok((ptr, Some(guard)))
            }
            None => Ok((0, None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sm_arch() {
        assert_eq!(parse_sm_arch("sm_100a").unwrap(), (10, 0));
        assert_eq!(parse_sm_arch("sm_103f").unwrap(), (10, 3));
        assert_eq!(parse_sm_arch("sm_89").unwrap(), (8, 9));
        assert_eq!(parse_sm_arch("sm_120").unwrap(), (12, 0));
        assert!(parse_sm_arch("compute_100").is_err());
    }
}
