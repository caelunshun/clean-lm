use nvvm_builder::{AddrSpace, Ptr, Scalar, Val};

#[derive(Debug, Copy, Clone)]
pub struct View1D<T: Scalar, S: AddrSpace> {
    /// Start of the view (e.g. a row of a tensor).
    pub base: Val<Ptr<T, S>>,
    /// Number of elements in the view.
    pub len: Val<u32>,
}

#[derive(Debug, Copy, Clone)]
pub struct View2D<T: Scalar, S: AddrSpace> {
    /// Start of the view.
    pub base: Val<Ptr<T, S>>,
    /// Number of elements in the view along the innermost dimension.
    pub inner_len: Val<u32>,
    /// Number of elements in the view along the outermost dimension.
    pub outer_len: Val<u32>,
    /// Stride, in elements, as the outermost dimension coordinate increases.
    pub outer_stride: Val<u32>,
}

#[derive(Debug, Copy, Clone)]
pub struct ViewPosWithinTensor2D {
    pub row_offset: Val<u32>,
    pub col_offset: Val<u32>,
    pub total_tensor_rows: Val<u32>,
    pub total_tensor_cols: Val<u32>,
}
