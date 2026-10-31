// --- LLM-generated --- //
use crate::{
    Held, Kernel, Param, ParamAbi, PassableAs, PtrArg, Result, parse_sm_arch, push_scalar,
};
use skycuda::{
    device_memory::{DeviceVisibleMemory, DeviceVisibleMemoryMut},
    launch::{KernelArg, LaunchBuilder},
    stream::Stream,
};
use tileir::Dim;
use tileir_builder::{
    Array, ArrayOpts, Built, Cubin, DynArg, DynArray, DynParam, Elem, IndexType, KernelId,
    KernelOpts, ModuleBuilder, ParamKind, Ptr, Tile, TileirasOptions,
};

unsafe impl<'a, T: Elem, M> PassableAs<'a, Ptr<T>> for &'a M
where
    M: DeviceVisibleMemory,
{
    push_scalar!();
}
unsafe impl<'a, T: Elem, M> PassableAs<'a, Ptr<T>> for &'a mut M
where
    M: DeviceVisibleMemoryMut,
{
    push_scalar!();
}
unsafe impl<'a, T: Elem> PassableAs<'a, Ptr<T>> for PtrArg<'a, T> {
    push_scalar!();
}

/// A strided device array, for parameters declared with [`KernelBuilder::array`].
/// Shape and strides are in elements.
pub struct ArrayArg<'a, T> {
    ptr: PtrArg<'a, T>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

impl<'a, T> ArrayArg<'a, T> {
    #[track_caller]
    pub fn new(ptr: PtrArg<'a, T>, shape: &[i64], strides: &[i64]) -> Self {
        assert_eq!(
            shape.len(),
            strides.len(),
            "array shape/stride rank mismatch"
        );
        Self {
            ptr,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
        }
    }
}

/// A device pointer checked for alignment once its access is granted.
struct AlignedPtrArg<'a, T> {
    ptr: PtrArg<'a, T>,
    alignment: u64,
}

impl<'a, T> KernelArg<'a> for AlignedPtrArg<'a, T> {
    type Repr = u64;
    type Guard = Option<Box<dyn Held + 'a>>;

    unsafe fn into_arg(
        self,
        stream: &'a Stream,
    ) -> Result<(Self::Repr, Self::Guard), skycuda::error::Error> {
        let (ptr, guard) = unsafe { self.ptr.into_arg(stream)? };
        assert!(
            ptr % self.alignment == 0,
            "array pointer {ptr:#x} is not aligned to {} bytes",
            self.alignment
        );
        Ok((ptr, guard))
    }
}

unsafe impl<'a, T: Elem> PassableAs<'a, Array<T>> for ArrayArg<'a, T> {
    fn push(
        self,
        abi: &ParamAbi,
        launch_builder: &mut LaunchBuilder<'a>,
    ) -> Result<(), skycuda::error::Error> {
        let ParamAbi::Array {
            shape,
            strides,
            index_i64,
            alignment,
        } = abi
        else {
            panic!("expected an array parameter");
        };
        assert_eq!(self.shape.len(), shape.len(), "array rank mismatch");

        launch_builder.arg(AlignedPtrArg {
            ptr: self.ptr,
            alignment: (*alignment).max(u64::from(T::BITS).div_ceil(8)),
        })?;

        for (label, actual, expected) in [
            ("dimension", &self.shape, shape),
            ("stride", &self.strides, strides),
        ] {
            for (i, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
                match expected {
                    Some(expected) => {
                        assert_eq!(actual, *expected, "array static {label} {i} mismatch")
                    }
                    None if *index_i64 => {
                        launch_builder.arg(actual)?;
                    }
                    None => {
                        let actual = i32::try_from(actual).unwrap_or_else(|_| {
                            panic!("array {label} {i} = {actual} does not fit in i32")
                        });
                        launch_builder.arg(actual)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Declares the parameters of a kernel built with `tileir-builder`.
#[derive(Debug, Default)]
pub struct KernelBuilder {
    params: Vec<(String, DynParam)>,
}

impl KernelBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// A scalar parameter. Use `Ptr<T>` for raw pointers.
    pub fn param<T: Elem>(&mut self, name: impl Into<String>) -> Param<T> {
        self.params
            .push((name.into(), DynParam::scalar(T::dtype())));
        Param::new(self.params.len() - 1)
    }

    #[track_caller]
    pub fn array<T: Elem>(&mut self, name: impl Into<String>, opts: ArrayOpts) -> Param<Array<T>> {
        self.params
            .push((name.into(), DynParam::array(T::dtype(), opts)));
        Param::new(self.params.len() - 1)
    }

    /// Defines the kernel in `module`. Overrides any argument names in
    /// `opts` with the declared parameter names.
    #[track_caller]
    pub fn define(
        self,
        module: &mut ModuleBuilder,
        name: &str,
        opts: KernelOpts,
        body: impl FnOnce(&Args),
    ) -> KernelSpec {
        let (names, params): (Vec<_>, Vec<_>) = self.params.into_iter().unzip();
        let id = module.kernel_dyn(name, opts.arg_names(names), &params, |vals| {
            body(&Args { vals });
        });
        KernelSpec { id }
    }
}

/// Kernel parameters, bound while tracing the kernel body.
pub struct Args {
    vals: Vec<DynArg>,
}

impl Args {
    #[track_caller]
    pub fn get<T: Elem>(&self, param: Param<T>) -> Tile<T> {
        self.vals[param.index].tile().downcast()
    }

    #[track_caller]
    pub fn get_array<T: Elem>(&self, param: Param<Array<T>>) -> DynArray {
        self.vals[param.index].array().clone()
    }
}

/// A kernel defined in a module, waiting for the module to be compiled.
#[derive(Debug, Clone)]
pub struct KernelSpec {
    id: KernelId,
}

impl KernelSpec {
    /// `built` must be the module this kernel was defined in, and
    /// `cubin` its compiled image.
    pub fn finish(self, built: &Built, cubin: Cubin) -> Result<Kernel> {
        let (major, minor) = parse_sm_arch(&cubin.arch)?;
        let info = built.kernel(self.id);

        let static_dims = |dims: &[Dim]| {
            dims.iter()
                .map(|dim| match dim {
                    Dim::Static(n) => Some(*n),
                    Dim::Dynamic => None,
                })
                .collect()
        };
        let params = info
            .params
            .iter()
            .map(|param| match &param.kind {
                ParamKind::Scalar(_) => ParamAbi::Scalar,
                ParamKind::Array {
                    shape,
                    strides,
                    index_type,
                    alignment,
                    ..
                } => ParamAbi::Array {
                    shape: static_dims(shape),
                    strides: static_dims(strides),
                    index_i64: *index_type == IndexType::I64,
                    alignment: *alignment,
                },
            })
            .collect();

        Ok(Kernel {
            exact_target_sm_major: major,
            exact_target_sm_minor: minor,
            cubin: cubin.bytes.into(),
            function_name: info.name.clone(),
            params,
            dynamic_smem_bytes: 0,
            block_size: (1, 1, 1),
            cluster_size: (1, 1, 1),
        })
    }
}

/// Builds a module and compiles it with `tileiras`.
pub fn compile(module: ModuleBuilder, options: &TileirasOptions) -> Result<(Built, Cubin)> {
    let built = module.finish()?;
    let cubin = built.compile(options)?;
    Ok((built, cubin))
}
