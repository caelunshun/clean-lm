#![no_std]
#![feature(
    abi_ptx,
    nvptx_ext,
    loop_hints,
    asm_experimental_arch,
    stmt_expr_attributes,
    min_adt_const_params
)]
#![allow(
    clippy::too_many_arguments,
    clippy::missing_safety_doc,
    // These unidiomatic expressions are needed
    // to get proper loop unrolling in same cases.
    clippy::needless_range_loop,
    clippy::chunks_exact_to_as_chunks
)]

use bytemuck::Pod;
use core::{
    arch::asm,
    hint::{self, assert_unchecked},
};
use ptx_std::{
    cp_async, thread_id,
    warp_collectives::{LaneMask, shuffle_xor},
};

pub mod norm;
pub mod sky_matmul;

const WARP_SIZE: u32 = 32;

#[inline(always)]
#[allow(unused)]
fn num_warps() -> u32 {
    let num_warps = ptx_std::block_size() / WARP_SIZE;
    unsafe {
        assert_unchecked(num_warps > 0);
    }
    num_warps
}

/// Index of the executing thread within its (1D) block.
#[inline(always)]
fn thread_idx() -> u32 {
    thread_id().x
}

/// Index of the executing warp within its (1D) block.
#[inline(always)]
fn warp_id() -> u32 {
    thread_idx() / WARP_SIZE
}

#[inline(always)]
fn lane_id() -> u32 {
    let id = ptx_std::lane_id();
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
/// distributed among a group of threads.
#[inline(always)]
pub(crate) unsafe fn collective_async_copy_g2s_128bit<T: Pod>(
    src_gmem: *const T,
    dst_smem: *mut T,
    len: u32,
    thread_rank: u32,
    // Specified so that a compile-time known constant can be used
    // and avoid a division instruction.
    group_size: u32,
) {
    debug_assert!(len.is_multiple_of(elements_per_u128::<T>() * group_size));

    let num_copies = len / (elements_per_u128::<T>() * group_size);
    #[rustc_unroll(full)]
    for i in 0..num_copies {
        let offset =
            thread_rank * elements_per_u128::<T>() + i * elements_per_u128::<T>() * group_size;
        unsafe {
            cp_async::copy::<u128, { cp_async::CacheOperator::L2Only }>(
                dst_smem.add(offset as usize).cast(),
                src_gmem.add(offset as usize).cast(),
            );
        }
    }
}

/// `cp_async::wait_group` with a pending count that is only a constant
/// after monomorphization.
#[inline(always)]
pub(crate) fn cp_async_wait_group(pending: u32) {
    match pending {
        0 => cp_async::wait_group::<0>(),
        1 => cp_async::wait_group::<1>(),
        2 => cp_async::wait_group::<2>(),
        3 => cp_async::wait_group::<3>(),
        4 => cp_async::wait_group::<4>(),
        5 => cp_async::wait_group::<5>(),
        6 => cp_async::wait_group::<6>(),
        7 => cp_async::wait_group::<7>(),
        _ => unreachable!(),
    }
}

#[inline(always)]
pub(crate) fn warp_reduce_sum_f32(mut val: f32) -> f32 {
    #[rustc_unroll(full)]
    for i in (0..WARP_SIZE.ilog2()).rev() {
        val += unsafe { shuffle_xor(LaneMask::FULL, val, 1 << i) };
    }
    val
}


/// Creates an L2 cache policy where a fraction of the accessed lines are evicted first
/// (`createpolicy.fractional.L2::evict_first`).
#[inline(always)]
#[allow(unused)]
pub(crate) fn createpolicy_fractional_evict_first(fraction: f32) -> u64 {
    let policy: u64;
    unsafe {
        asm!(
            "createpolicy.fractional.L2::evict_first.b64 {policy}, {fraction};",
            policy = out(reg64) policy,
            fraction = in(reg32) fraction,
            options(pure, nomem, nostack),
        );
    }
    policy
}

/// Creates an L2 cache policy where a fraction of the accessed lines keep their eviction
/// priority (`createpolicy.fractional.L2::evict_unchanged`).
#[inline(always)]
#[allow(unused)]
pub(crate) fn createpolicy_fractional_evict_unchanged(fraction: f32) -> u64 {
    let policy: u64;
    unsafe {
        asm!(
            "createpolicy.fractional.L2::evict_unchanged.b64 {policy}, {fraction};",
            policy = out(reg64) policy,
            fraction = in(reg32) fraction,
            options(pure, nomem, nostack),
        );
    }
    policy
}
