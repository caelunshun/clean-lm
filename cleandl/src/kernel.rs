use crate::context::{Context, ResourceKey};
use anyhow::bail;
use serde::Serialize;
use skycuda::module::Module;

#[macro_export]
macro_rules! include_cubins {
    ($base_path:literal) => {
        $crate::include_cubins!(@build $base_path, Some(include_bytes!(concat!($base_path, ".sm89.cubin"))))
    };
    // For modules that only support sm_90 and newer.
    ($base_path:literal, sm90_plus) => {
        $crate::include_cubins!(@build $base_path, None)
    };
    (@build $base_path:literal, $sm89:expr) => {
        $crate::kernel::Cubins {
            sm89: $sm89,
            sm90a: include_bytes!(concat!($base_path, ".sm90a.cubin")),
            sm100a: include_bytes!(concat!($base_path, ".sm100a.cubin")),
            sm103a: include_bytes!(concat!($base_path, ".sm103a.cubin")),
            sm120a: include_bytes!(concat!($base_path, ".sm120a.cubin")),
            sm121a: include_bytes!(concat!($base_path, ".sm121a.cubin")),
        }
    };
}

#[derive(Debug, Copy, Clone, Serialize)]
pub struct Cubins<'a> {
    pub sm89: Option<&'a [u8]>,
    pub sm90a: &'a [u8],
    pub sm100a: &'a [u8],
    pub sm103a: &'a [u8],
    pub sm120a: &'a [u8],
    pub sm121a: &'a [u8],
}

impl ResourceKey for Cubins<'static> {
    type Resource = Module;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let cubin = match (
            cx.cuda().compute_capability_major(),
            cx.cuda().compute_capability_minor(),
        ) {
            (8, 9) => match self.sm89 {
                Some(cubin) => cubin,
                None => bail!("module does not support sm 8.9"),
            },
            (9, 0) => self.sm90a,
            (10, 0) => self.sm100a,
            (10, 3) => self.sm103a,
            (12, 0) => self.sm120a,
            (12, 1) => self.sm121a,
            (major, minor) => bail!("unsupported sm version {major}.{minor}"),
        };

        let module = unsafe { cx.cuda().load_module_cubin(cubin)? };
        Ok(module)
    }
}

#[derive(Debug, Copy, Clone, Serialize)]
pub struct RustKernels;

impl ResourceKey for RustKernels {
    type Resource = Module;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        Ok((*cx.get_or_create_resource(include_cubins!(
            "../rust-kernels/target/kernels",
            sm90_plus
        ))?)
        .clone())
    }
}

/// Whether `rust-kernels` can run on this device (requires sm_90+).
pub(crate) fn rust_kernels_supported(cx: &Context) -> bool {
    cx.cuda().compute_capability_major() >= 9
}

/// Host equivalent of the `has_227kb_smem` cfg alias in `rust-kernels/build.rs`.
pub(crate) fn rust_kernels_have_227kb_smem(cx: &Context) -> bool {
    (9..=11).contains(&cx.cuda().compute_capability_major())
}
