#![no_std]
#![feature(
    min_adt_const_params,
    const_param_ty_trait,
    impl_trait_in_assoc_type,
    pointer_is_aligned_to,
    trusted_len
)]
#![allow(incomplete_features, clippy::implied_bounds_in_impls)]

use core::{
    hint::{self, assert_unchecked},
    marker::ConstParamTy_,
};

use cuda_device::uniform::block_dim_x;

pub mod norm;
pub mod smem_ring;
pub mod view;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(unused)]
enum PrefetchMode {
    /// SASS LDGSTS, >= SM80
    CpAsync,
    /// >= SM90
    CpAsyncBulk,
}
impl ConstParamTy_ for PrefetchMode {}

const WARP_SIZE: usize = 32;

#[inline(always)]
fn num_warps() -> usize {
    let num_warps = block_dim_x().get() as usize / 32;
    unsafe {
        assert_unchecked(num_warps > 0);
    }
    num_warps
}

#[inline(always)]
fn lane_id() -> usize {
    let id = cuda_device::warp::lane_id() as usize;
    unsafe {
        hint::assert_unchecked(id < 32);
    }
    id
}
