#![no_std]
#![feature(
    stmt_expr_attributes,
    min_adt_const_params,
    min_generic_const_args,
    generic_const_args,
    generic_const_exprs
)]
#![allow(
    incomplete_features,
    clippy::too_many_arguments,
    // These unidiomatic expressions are needed
    // to get proper loop unrolling in same cases. 
    clippy::needless_range_loop,
    clippy::chunks_exact_to_as_chunks
)]

use core::hint::{self, assert_unchecked};

use bytemuck::Pod;
use cuda_device::ptx_asm;
use cuda_device::{
    async_copy,
    cooperative_groups::ThreadGroup,
    uniform::{block_dim_x, block_dim_y, block_dim_z},
};
use cuda_macros::device;

pub mod norm;
pub mod matmul;

const WARP_SIZE: u32 = 32;

#[inline(always)]
#[allow(unused)]
fn num_warps() -> u32 {
    let num_warps = block_dim_x().get() * block_dim_y().get() * block_dim_z().get() / WARP_SIZE;
    unsafe {
        assert_unchecked(num_warps > 0);
    }
    num_warps
}

#[inline(always)]
fn lane_id() -> u32 {
    let id = cuda_device::warp::lane_id();
    unsafe {
        hint::assert_unchecked(id < 32);
    }
    id
}

pub fn elements_per_u128<T>() -> u32 {
    assert!(size_of::<u128>().is_multiple_of(size_of::<T>()));
    (size_of::<u128>() / size_of::<T>()) as u32
}

/// Collective copy from GMEM into SMEM of a 1D array,
/// distributed among the group.
#[device]
pub(crate) unsafe fn collective_async_copy_g2s_128bit<T: Pod, G: ThreadGroup>(
    src_gmem: *const T,
    dst_smem: *mut T,
    len: u32,
    group: G,
    // Specified so that a compile-time known constant can be used
    // and avoid a division instruction.
    group_size: u32,
) {
    debug_assert_eq!(group.size(), group_size);
    debug_assert!(len.is_multiple_of(elements_per_u128::<T>() * group_size));

    let num_copies = len / (elements_per_u128::<T>() * group_size);
    #[llvm_unroll]
    for i in 0..num_copies {
        let offset = group.thread_rank() * elements_per_u128::<T>()
            + i * elements_per_u128::<T>() * group_size;
        unsafe {
            async_copy::cp_async_cg_16(
                dst_smem.add(offset as usize).cast(),
                src_gmem.add(offset as usize).cast(),
            );
        }
    }
}

/// Force using a global memory atomic instruction instead of CAS
/// loop or fallback path that needs to branch on shared vs. global address.
#[device]
pub(crate) unsafe fn red_add_global_f32(dst: *mut f32, val: f32) {
    unsafe {
        ptx_asm!(
            "{ .reg .b64 %%addr; cvta.to.global.u64 %%addr, %0; red.relaxed.gpu.global.add.f32 [%%addr], %1; }",
            in("l") dst as u64,
            in("f") val,
            clobber("memory"),
        );
    }
}
