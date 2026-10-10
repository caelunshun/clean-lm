//! Kernels for applying block scaling.
//! These formats are only supported on >=SM100 (including consumer),
//! and all such archs have TMA, so we assume TMA is available.
//! These kernels only support 2D tensors and use
//! the 128x4 tiled block scale factor format prescribed by cuBLASlt and CUTLASS.
//! Kernels can emit both rowwise and columnwise block-scaled output, or
//! just one or the other.

