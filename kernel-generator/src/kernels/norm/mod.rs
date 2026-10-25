use serde::{Deserialize, Serialize};

pub mod forward;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputTypes {
    Bf16,
    Mxfp8Rowwise,
    Mxfp8RowwiseAndColwise,
    Nvfp4Rowwise,
}

pub const MAX_SUPPORTED_ROW_SIZE: usize = 8192;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Config {
    pub norm_type: NormType,
    pub has_scale: bool,
    pub has_bias: bool,
    pub output_types: OutputTypes,
    /// Maximum row size that the kernel needs to support.
    /// Using the kernel for larger rows than this has undefined behavior.
    /// Using the kernel for smaller rows will be correct, but may incur
    /// a performance penalty.
    ///
    /// Cannot exceed `MAX_SUPPORTED_ROW_SIZE`.
    pub max_row_size: usize,
}
