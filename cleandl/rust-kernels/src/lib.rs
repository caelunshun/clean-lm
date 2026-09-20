#![no_std]
#![feature(min_adt_const_params, const_param_ty_trait)]
#![allow(incomplete_features)]

use core::{marker::ConstParamTy_, slice};

use cuda_device::uniform::block_dim_x;
use cuda_device_ext::{bf16, bf16x2};

pub mod norm;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PrefetchMode {
    /// SASS LDGSTS, >= SM80
    CpAsync,
    /// >= SM90
    CpAsyncBulk,
}
impl ConstParamTy_ for PrefetchMode {}

const WARP_SIZE: usize = 32;

fn num_warps() -> usize {
    block_dim_x().get() as usize / 32
}

unsafe fn as_aligned_slice<T>(s: &[T]) -> &[u128] {
    const {
        assert!(size_of::<u128>().is_multiple_of(size_of::<T>()));
    }
    unsafe {
        slice::from_raw_parts(
            s.as_ptr().cast(),
            s.len() / (size_of::<u128>() / size_of::<T>()),
        )
    }
}

unsafe fn as_aligned_slice_mut<T>(s: &mut [T]) -> &mut [u128] {
    const {
        assert!(size_of::<u128>().is_multiple_of(size_of::<T>()));
    }
    unsafe {
        slice::from_raw_parts_mut(
            s.as_mut_ptr().cast(),
            s.len() / (size_of::<u128>() / size_of::<T>()),
        )
    }
}

unsafe fn as_bf16x2_slice(s: &[bf16]) -> &[bf16x2] {
    unsafe { slice::from_raw_parts(s.as_ptr().cast(), s.len() / 2) }
}

unsafe fn as_bf16x2_slice_mut(s: &mut [bf16]) -> &mut [bf16x2] {
    unsafe { slice::from_raw_parts_mut(s.as_mut_ptr().cast(), s.len() / 2) }
}
