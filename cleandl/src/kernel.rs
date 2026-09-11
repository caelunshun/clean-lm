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
            sm100f: include_bytes!(concat!($base_path, ".sm100f.cubin")),
            sm120f: include_bytes!(concat!($base_path, ".sm120f.cubin")),
        }
    };
}

#[derive(Debug, Copy, Clone, Serialize)]
pub struct Cubins<'a> {
    pub sm89: &'a [u8],
    pub sm90a: &'a [u8],
    pub sm100f: &'a [u8],
    pub sm120f: &'a [u8],
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
            (10, _) => self.sm100f,
            (12, _) => self.sm120f,
            (major, minor) => bail!("unsupported sm version {major}.{minor}"),
        };

        let module = unsafe { cx.cuda().load_module_cubin(cubin)? };
        Ok(module)
    }
}
