use anyhow::bail;
use serde::Serialize;
use skycuda::module::Module;

use crate::context::{Context, ResourceKey};

#[macro_export]
macro_rules! include_cubins {
    ($base_path:literal) => {
        $crate::kernel::Cubins {
            sm89: include_bytes!(concat!($base_path, ".sm89.cubin")),
            sm90a: include_bytes!(concat!($base_path, ".sm90a.cubin")),
            sm100a: include_bytes!(concat!($base_path, ".sm100a.cubin")),
            sm103a: include_bytes!(concat!($base_path, ".sm103a.cubin")),
            sm107a: include_bytes!(concat!($base_path, ".sm107a.cubin")),
            sm120a: include_bytes!(concat!($base_path, ".sm120a.cubin")),
            sm121a: include_bytes!(concat!($base_path, ".sm121a.cubin")),
        }
    };
}

#[derive(Debug, Copy, Clone, Serialize)]
pub struct Cubins<'a> {
    pub sm89: &'a [u8],
    pub sm90a: &'a [u8],
    pub sm100a: &'a [u8],
    pub sm103a: &'a [u8],
    pub sm107a: &'a [u8],
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
            (8, 9) => self.sm89,
            (9, 0) => self.sm90a,
            (10, 0) => self.sm100a,
            (10, 3) => self.sm103a,
            (10, 7) => self.sm107a,
            (12, 0) => self.sm120a,
            (12, 1) => self.sm121a,
            (major, minor) => bail!("unsupported sm version {major}.{minor}"),
        };

        let module = unsafe { cx.cuda().load_module_cubin(cubin)? };
        Ok(module)
    }
}
