use serde::Serialize;
use singe_cuda::{
    jit::JitOptions,
    module::Module,
    nvrtc::{OutputKind, Program},
};

use crate::context::{Context, ResourceKey};

/// Resource cache key for a compiled cutile kernel via nvrtc.
#[derive(Debug, Clone, Serialize)]
pub struct CutileKernelKey {
    pub src: &'static str,
}

impl ResourceKey for CutileKernelKey {
    type Resource = Module;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let library = cx.cuda().load_nvrtc_library_with_options(
            &Program::from_source(self.src),
            OutputKind::Cubin,
            JitOptions::default()
                .with_optimization_level(3)
                .with_generate_line_info(true),
        )?;

        cx.cuda().bind()?;
        let module = library.module()?;
        Ok(module)
    }
}
