//! Helper for building kernels with dynamic arguments at runtime.
//!
//! Kernels can be built either as NVVM->PTX (via `nvvm-builder`)
//! or TileIR (via `tileir-builder`). Kernels can also be serialized.

use derive_where::derive_where;
use serde::{Deserialize, Serialize};
use skycuda::{
    device::Device,
    function::Function,
    launch::{KernelArg, LaunchBuilder},
    stream::Stream,
};
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

pub mod nvvm;
pub mod tileir;

#[derive_where(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Param<T> {
    index: usize,
    _marker: PhantomData<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kernel {
    exact_target_sm_major: u32,
    exact_target_sm_minor: u32,
    cubin: Arc<[u8]>,
    function_name: String,
    num_params: usize,
    dynamic_smem_bytes: usize,
    block_size: (u32, u32, u32),
    cluster_size: (u32, u32, u32),
}

impl Kernel {
    pub unsafe fn load(&self, device: &Device) -> Result<LoadedKernel, skycuda::error::Error> {
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
}

#[derive(Debug, Clone)]
pub struct LoadedKernel {
    kernel: Kernel,
    loaded_function: Function,
}

impl LoadedKernel {
    pub unsafe fn launch<'stream, 'args>(
        &'stream self,
        mut args: ArgBundle<'args>,
        grid_size: (u32, u32, u32),
        stream: &'stream Stream,
    ) -> Result<(), skycuda::error::Error>
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

        for param_idx in 0..self.kernel.num_params {
            (args
                .add_param_to_builder
                .remove(&param_idx)
                .unwrap_or_else(|| panic!("missing parameter at index {param_idx}")))(
                &mut launch_builder,
            )?;
        }

        unsafe {
            launch_builder.launch(grid_size.0, grid_size.1, grid_size.2)?;
        }

        Ok(())
    }
}

#[derive(Default)]
pub struct ArgBundle<'a> {
    add_param_to_builder: HashMap<
        usize,
        Box<dyn FnOnce(&mut LaunchBuilder<'a>) -> Result<(), skycuda::error::Error> + 'a>,
    >,
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
            Box::new(|launch_builder| launch_builder.arg(value).map(|_| ())),
        );
    }
}

pub unsafe trait PassableAs<'a, T>: KernelArg<'a> {}
