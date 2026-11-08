use crate::kernels::matmul::epilogue::Epilogue;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    pub mma_type: MmaType,
    pub tuning_config: TuningConfig,
    pub epilogue: Epilogue,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningConfig {
    pub raster_order: RasterOrder,
    pub cga_swizzle_size: u32,
    /// On SM10x and SM11x: Must be a power of 2, and at least 2, because 2CTA mode
    /// is always used.
    /// On other architectures: Must be 1.
    pub cluster_size_m: u32,
    /// On SM10x and SM11x: Must be a power of 2.
    /// On other architectures: Must be 1.
    pub cluster_size_n: u32,
    /// Uses fallback clusters of size 2x1 to fill all SMs, via the preferred-cluster-size
    /// feature on Blackwell.
    pub use_fallback_clusters: bool,
    /// Must be nonzero if epilogue has load operations.
    pub epilogue_stages: u32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RasterOrder {
    AlongM,
    AlongN,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MmaType {
    Bf16 {
        a_layout: MatrixLayout,
        b_layout: MatrixLayout,
    },
    /// A must be row-major, and B column-major.
    Mxfp8,
    /// A must be row-major, and B column-major.
    Nvfp4,
}

impl MmaType {
    /// NB doesn't include scale factor bits.
    pub(super) fn bits_per_element(&self) -> usize {
        match self {
            MmaType::Bf16 { .. } => 16,
            MmaType::Mxfp8 => 8,
            MmaType::Nvfp4 => 4,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatrixLayout {
    RowMajor,
    ColMajor,
}
