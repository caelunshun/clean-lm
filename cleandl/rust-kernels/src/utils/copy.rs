use crate::{elements_per_u32, elements_per_u128, elements_per_u256};
use bytemuck::Pod;
use core::simd::{u32x4, u32x8};

/// Copies with 32-byte alignment between two arrays.
///
/// Should only be used for copies between GMEM and registers.
/// SMEM does not support 256-bit operations, so using 256-bit loads
/// would lower to a strided access pattern with bank conflicts.
#[inline(always)]
pub(crate) unsafe fn copy_aligned32<T: Pod>(
    src: *const T,
    dst: *mut T,
    // Number of 32-byte blocks to copy.
    copy_blocks: usize,
    // Stride in T-elements between 32-byte blocks in `src` to copy.
    src_stride: usize,
    dst_stride: usize,
) {
    unsafe {
        #[rustc_unroll(full)]
        for i in 0..copy_blocks {
            let src = src
                .add(i * src_stride * elements_per_u256::<T>() as usize)
                .cast::<u32x8>();
            let dst = dst
                .add(i * dst_stride * elements_per_u256::<T>() as usize)
                .cast::<u32x8>();
            dst.write(src.read());
        }
    }
}

/// Copies with 16-byte alignment between two arrays.
#[inline(always)]
pub(crate) unsafe fn copy_aligned16<T: Pod>(
    src: *const T,
    dst: *mut T,
    // Number of 16-byte blocks to copy.
    copy_blocks: usize,
    // Stride in T-elements between 16-byte blocks in `src` to copy.
    src_stride: usize,
    dst_stride: usize,
) {
    unsafe {
        #[rustc_unroll(full)]
        for i in 0..copy_blocks {
            let src = src
                .add(i * src_stride * elements_per_u128::<T>() as usize)
                .cast::<u32x4>();
            let dst = dst
                .add(i * dst_stride * elements_per_u128::<T>() as usize)
                .cast::<u32x4>();
            dst.write(src.read());
        }
    }
}

/// Copies with 4-byte alignment between two arrays.
#[inline(always)]
pub(crate) unsafe fn copy_aligned4<T: Pod>(
    src: *const T,
    dst: *mut T,
    // Number of 4-byte blocks to copy.
    copy_blocks: usize,
    // Stride in T-elements between 4-byte blocks in `src` to copy.
    src_stride: usize,
    dst_stride: usize,
) {
    unsafe {
        #[rustc_unroll(full)]
        for i in 0..copy_blocks {
            let src = src
                .add(i * src_stride * elements_per_u32::<T>() as usize)
                .cast::<u32>();
            let dst = dst
                .add(i * dst_stride * elements_per_u32::<T>() as usize)
                .cast::<u32>();
            dst.write(src.read());
        }
    }
}
