#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MatrixLayout {
    RowMajor,
    ColMajor,
}
