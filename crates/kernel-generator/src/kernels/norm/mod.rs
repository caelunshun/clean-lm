use kernel_build_util::nvvm;
use nvvm_builder::{
    Cubin, ModuleBuilder, PtxasOptions,
    nvvm_sys::{Arch, CompileOptions, OptLevel},
    tools::PtxasOptLevel,
};
use serde::{Deserialize, Serialize};

pub mod forward;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum OutputTypes {
    Bf16,
    Mxfp8Rowwise,
    Mxfp8RowwiseAndColwise,
    Nvfp4Rowwise,
}

pub const MAX_SUPPORTED_ROW_SIZE: usize = 8192;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(clap::Parser))]
pub struct Config {
    #[cfg_attr(feature = "clap", arg(long))]
    pub norm_type: NormType,
    #[cfg_attr(feature = "clap", arg(long))]
    pub has_scale: bool,
    #[cfg_attr(feature = "clap", arg(long))]
    pub has_bias: bool,
    #[cfg_attr(feature = "clap", arg(long))]
    pub output_types: OutputTypes,
    /// Maximum row size that the kernel needs to support.
    /// Using the kernel for larger rows than this has undefined behavior.
    /// Using the kernel for smaller rows will be correct, but may incur
    /// a performance penalty.
    ///
    /// Cannot exceed `MAX_SUPPORTED_ROW_SIZE`.
    #[cfg_attr(feature = "clap", arg(long))]
    pub max_row_size: usize,
}

fn compile_kernel_to_cubin(module: ModuleBuilder, arch: Arch) -> crate::error::Result<Cubin> {
    Ok(nvvm::compile(
        module,
        &CompileOptions::new(arch).opt_level(OptLevel::O3),
        &PtxasOptions::new().opt_level(PtxasOptLevel::O3),
    )?)
}
