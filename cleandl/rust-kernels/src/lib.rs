#![no_std]
#![feature(
    min_adt_const_params,
    const_param_ty_trait,
    impl_trait_in_assoc_type,
    pointer_is_aligned_to,
    loop_hints,
    trusted_len
)]
#![allow(incomplete_features, clippy::implied_bounds_in_impls)]

use core::{
    hint::{self, assert_unchecked},
    iter::TrustedLen,
    marker::ConstParamTy_,
};

use cuda_device::uniform::block_dim_x;
use cuda_macros::device;

pub mod norm;
pub mod smem_ring;
pub mod view;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

/// Takes zip of 2 iterators that should have the same length
/// (else behavior is unspecified but not UB). Then invokes the callback.
///
/// This method seems to optimize/unroll better than using `for (a, b) in a.into_iter().zip(b)`.
#[device]
fn zip2_for_each<A, B, AI, BI, F>(a: AI, b: BI, mut f: F)
where
    AI: IntoIterator<Item = A, IntoIter: ExactSizeIterator + TrustedLen>,
    BI: IntoIterator<Item = B, IntoIter: ExactSizeIterator + TrustedLen>,
    F: FnMut(A, B),
{
    let mut a = a.into_iter();
    let mut b = b.into_iter();

    assert_eq!(a.len(), b.len());
    let n = a.len();

    let mut i = 0;
    #[llvm_unroll]
    while i < n {
        let a_val = unsafe { a.next().unwrap_unchecked() };
        let b_val = unsafe { b.next().unwrap_unchecked() };

        f(a_val, b_val);
        i += 1;
    }
}
