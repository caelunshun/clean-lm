#![no_std]
#![feature(
    abi_ptx,
    nvptx_ext,
    loop_hints,
    stmt_expr_attributes,
    min_adt_const_params,
    gca_const_items,
    gca_min_const_items
)]
#![allow(
    incomplete_features,
    clippy::too_many_arguments,
    clippy::missing_safety_doc,
    // These unidiomatic expressions are needed
    // to get proper loop unrolling in same cases.
    clippy::needless_range_loop,
    clippy::chunks_exact_to_as_chunks,
)]

use bytemuck::Pod;
use ptx_std::{
    bf16x2, cp_async,
    warp_collectives::{Absolute, LaneMask, shuffle_xor},
};

#[cfg(not(target_feature = "sm_100"))]
compile_error!("rust-kernels requires sm_100 or newer");

pub mod norm;
pub mod rmsnorm;
pub mod sky_matmul;
mod utils;

const WARP_SIZE: u32 = 32;

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
pub(crate) unsafe fn warp_reduce_sum_f32(mut val: f32) -> f32 {
    #[rustc_unroll(full)]
    for i in (0..WARP_SIZE.ilog2()).rev() {
        val += unsafe { shuffle_xor(LaneMask::FULL, val, 1 << i) };
    }
    val
}

/// Warp-wide reduction of bf16x2 values with f32 accumulation.
#[inline(always)]
pub(crate) unsafe fn warp_reduce_sum_bf16x2_f32x2(val: bf16x2) -> [f32; 2] {
    // Micro-optimize by doing the first shuffle with bf16x2,
    // but the rest need f32 to avoid intermediate rounding.
    let peer_val = unsafe { shuffle_xor(LaneMask::FULL, val.to_bits(), 16) };
    let peer_val = bf16x2::from_bits(peer_val).to_f32x2();
    let val = val.to_f32x2();
    let mut sum = [val[0] + peer_val[0], val[1] + peer_val[1]];

    #[rustc_unroll(full)]
    for i in (0..WARP_SIZE.ilog2() - 1).rev() {
        sum[0] += unsafe { shuffle_xor(LaneMask::FULL, sum[0], 1 << i) };
        sum[1] += unsafe { shuffle_xor(LaneMask::FULL, sum[1], 1 << i) };
    }

    sum
}

#[inline(always)]
#[allow(unused_mut)]
pub(crate) unsafe fn warp_reduce_max_f32<const ABS: Absolute>(mut val: f32) -> f32 {
    #[cfg(target_feature = "sm_100f")]
    {
        use ptx_std::warp_collectives::reduce_f32;

        unsafe {
            reduce_f32::<
                {
                    use ptx_std::warp_collectives::ReduxOp;
                    ReduxOp::Max
                },
                ABS,
                {
                    use ptx_std::NanHandling;
                    NanHandling::Ignore
                },
            >(LaneMask::FULL, val)
        }
    }
    #[cfg(not(target_feature = "sm_100f"))]
    {
        if ABS == Absolute::Enabled {
            val = val.abs();
        }
        #[rustc_unroll(full)]
        for i in (0..WARP_SIZE.ilog2()).rev() {
            val = val.max(unsafe { shuffle_xor(LaneMask::FULL, val, 1 << i) });
        }
        val
    }
}

#[inline(always)]
#[allow(unused)]
pub(crate) unsafe fn warp_reduce_max_bf16x2<const ABS: Absolute>(mut val: bf16x2) -> bf16x2 {
    // #[cfg(target_feature = "sm_100f")]
    // {
    //     let [a, b] = val.to_f32x2();
    //     unsafe { bf16x2::from_f32x2(warp_reduce_max_f32::<ABS>(a), warp_reduce_max_f32::<ABS>(b)) }
    // }
    // #[cfg(not(target_feature = "sm_100f"))]
    {
        if ABS == Absolute::Enabled {
            val = val.abs();
        }
        #[rustc_unroll(full)]
        for i in (0..WARP_SIZE.ilog2()).rev() {
            val = val.max(unsafe { shuffle_xor(LaneMask::FULL, val, 1 << i) });
        }
        val
    }
}
