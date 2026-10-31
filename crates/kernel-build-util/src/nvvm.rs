use crate::{Kernel, Param, ParamAbi, PassableAs, PtrArg, Result, parse_sm_arch, push_scalar};
use nvvm_builder::{
    Cubin, DType, DynVal, Generic, Global, KernelOpts, ModuleBuilder, Ptr, PtxasOptions, Val,
    ValueType, nvvm_sys::CompileOptions, tools::ptxas,
};
use skycuda::device_memory::{DeviceVisibleMemory, DeviceVisibleMemoryMut};
use std::cell::Cell;

macro_rules! ptr_passable {
    ($space:ty) => {
        unsafe impl<'a, T, M> PassableAs<'a, Ptr<T, $space>> for &'a M
        where
            M: DeviceVisibleMemory,
        {
            push_scalar!();
        }
        unsafe impl<'a, T, M> PassableAs<'a, Ptr<T, $space>> for &'a mut M
        where
            M: DeviceVisibleMemoryMut,
        {
            push_scalar!();
        }
        unsafe impl<'a, T> PassableAs<'a, Ptr<T, $space>> for PtrArg<'a, T> {
            push_scalar!();
        }
    };
}

ptr_passable!(Generic);
ptr_passable!(Global);

/// Declares the parameters of a kernel built with `nvvm-builder`.
#[derive(Debug, Default)]
pub struct KernelBuilder {
    params: Vec<(String, DType)>,
}

impl KernelBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn param<T: ValueType<Shape = ()>>(&mut self, name: impl Into<String>) -> Param<T> {
        self.params.push((name.into(), T::dtype(())));
        Param::new(self.params.len() - 1)
    }

    /// Defines the kernel in `module`. `opts.req_threads` must be set,
    /// since it determines the launch block size.
    #[track_caller]
    pub fn define(
        self,
        module: &mut ModuleBuilder,
        name: &str,
        mut opts: KernelOpts,
        body: impl FnOnce(&Args),
    ) -> KernelSpec {
        let block_size = opts
            .req_threads
            .expect("KernelOpts::req_threads must be set to determine the block size")
            .0;
        let cluster_size = opts.cluster_dims.map_or([1, 1, 1], |dims| dims.0);

        let (names, dtypes): (Vec<_>, Vec<_>) = self.params.into_iter().unzip();
        if opts.arg_names.is_empty() {
            opts.arg_names = names;
        }

        let dynamic_smem_bytes = Cell::new(0);
        module.kernel_dyn(name, opts, &dtypes, |vals| {
            let args = Args {
                vals,
                dynamic_smem_bytes: Cell::new(0),
            };
            body(&args);
            dynamic_smem_bytes.set(args.dynamic_smem_bytes.get());
        });

        KernelSpec {
            function_name: name.to_owned(),
            num_params: dtypes.len(),
            dynamic_smem_bytes: dynamic_smem_bytes.get(),
            block_size: block_size.into(),
            cluster_size: cluster_size.into(),
        }
    }
}

/// Kernel parameters, bound while tracing the kernel body.
pub struct Args {
    vals: Vec<DynVal>,
    dynamic_smem_bytes: Cell<usize>,
}

impl Args {
    #[track_caller]
    pub fn get<T: ValueType>(&self, param: Param<T>) -> Val<T> {
        self.vals[param.index]
            .downcast()
            .expect("kernel parameter type mismatch")
    }

    /// Sets the dynamic shared memory size to launch the kernel with.
    pub fn set_dynamic_smem_bytes(&self, bytes: usize) {
        self.dynamic_smem_bytes.set(bytes);
    }
}

/// A kernel defined in a module, waiting for the module to be compiled.
#[derive(Debug, Clone)]
pub struct KernelSpec {
    function_name: String,
    num_params: usize,
    dynamic_smem_bytes: usize,
    block_size: (u32, u32, u32),
    cluster_size: (u32, u32, u32),
}

impl KernelSpec {
    pub fn finish(self, cubin: Cubin) -> Result<Kernel> {
        let (major, minor) = parse_sm_arch(&cubin.arch)?;
        Ok(Kernel {
            exact_target_sm_major: major,
            exact_target_sm_minor: minor,
            cubin: cubin.bytes.into(),
            function_name: self.function_name,
            params: vec![ParamAbi::Scalar; self.num_params],
            dynamic_smem_bytes: self.dynamic_smem_bytes,
            block_size: self.block_size,
            cluster_size: self.cluster_size,
        })
    }
}

/// Compiles a module to a cubin via NVVM and `ptxas`.
pub fn compile(
    module: ModuleBuilder,
    nvvm_options: &CompileOptions,
    ptxas_options: &PtxasOptions,
) -> Result<Cubin> {
    let built = module.finish()?;
    let compiled = built.compile(nvvm_options)?;
    let ptx = compiled
        .ptx()
        .expect("NVVM output is not PTX (was LTO enabled?)");
    Ok(ptxas(ptx, ptxas_options)?)
}
