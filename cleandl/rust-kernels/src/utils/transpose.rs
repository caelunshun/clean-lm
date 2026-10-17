//! Various warp-collective register transposes.

use core::hint::select_unpredictable;
use ptx_std::{
    PermuteMode, U32Ext, lane_id,
    warp_collectives::{LaneMask, shuffle_xor},
};

/// Transposes a 32x32 matrix of 8-bit elements held in registers, where initially
/// lane `i` holds all elements of row `i` in order, and after the transpose
/// lane `i` holds all elements of column `i` in order. This is achieved
/// via butterfly shuffle instructions to avoid consuming shared memory capacity.
///
/// # Safety
/// All lanes in a warp must perform the same call.
#[inline(always)]
pub unsafe fn transpose_32x32_b8(arr: &mut [u8; 32]) {
    let mut vals = bytemuck::must_cast(*arr);
    transpose_32x32_generic::<8>(&mut vals, 8);
    *arr = bytemuck::must_cast(vals);
}

/// Transposes a 32x32 matrix of 16-bit elements held in registers, where initially
/// lane `i` holds all elements of row `i` in order, and after the transpose
/// lane `i` holds all elements of column `i` in order. This is achieved
/// via butterfly shuffle instructions to avoid consuming shared memory capacity.
///
/// # Safety
/// All lanes in a warp must perform the same call.
#[inline(always)]
pub unsafe fn transpose_32x32_b16(arr: &mut [u16; 32]) {
    let mut vals = bytemuck::must_cast(*arr);
    transpose_32x32_generic::<16>(&mut vals, 16);
    *arr = bytemuck::must_cast(vals);
}

/// Transposes a 32x32 matrix of 32-bit elements held in registers, where initially
/// lane `i` holds all elements of row `i` in order, and after the transpose
/// lane `i` holds all elements of column `i` in order. This is achieved
/// via butterfly shuffle instructions to avoid consuming shared memory capacity.
///
/// # Safety
/// All lanes in a warp must perform the same call.
#[inline(always)]
pub unsafe fn transpose_32x32_b32(arr: &mut [u32; 32]) {
    let mut vals = bytemuck::must_cast(*arr);
    transpose_32x32_generic::<32>(&mut vals, 32);
    *arr = bytemuck::must_cast(vals);
}

#[inline(always)]
fn transpose_32x32_generic<const N: usize>(arr: &mut [u32; N], w: u32) {
    let e = 32 / w;
    let el = e.ilog2();
    let r = w;

    #[rustc_unroll(full)]
    for t in 0..el {
        #[rustc_unroll(full)]
        for p in 0..r / 2 {
            let [x, y] = arr
                .get_disjoint_mut([2 * p as usize, 2 * p as usize + 1])
                .unwrap();

            if (w << t) >= 8 {
                swap_subword_bits_with_lane_bits_s_ge_8(x, y, t, w);
            } else {
                swap_subword_bits_with_lane_bits(x, y, t, w);
            }
        }
    }

    let h = lane_id() >> el;
    xor_permute(arr, h);
    #[rustc_unroll(full)]
    for k in 1..r {
        arr[k as usize] = unsafe { shuffle_xor(LaneMask::FULL, arr[k as usize], k << el) };
    }
    xor_permute(arr, h);
}

#[inline(always)]
fn xor_permute<const N: usize>(v: &mut [u32; N], h: u32) {
    #[rustc_unroll(full)]
    for t in 0..N.ilog2() {
        let c = ((h >> t) & 1) == 1;
        #[rustc_unroll(full)]
        for k in 0..N {
            if k & (1 << t) == 0 {
                let a = v[k];
                let b = v[k + (1 << t)];
                v[k] = select_unpredictable(c, b, a);
                v[k + (1 << t)] = select_unpredictable(c, a, b);
            }
        }
    }
}

#[inline(always)]
fn bitsel(m: u32, a: u32, b: u32) -> u32 {
    (a & m) | (b & !m)
}

#[inline(always)]
fn swap_subword_bits_with_lane_bits(x: &mut u32, y: &mut u32, t: u32, w: u32) {
    let s = w << t;
    let m = match s {
        4 => 0x0F0F0F0Fu32,
        8 => 0x00FF00FF,
        16 => 0x0000FFFF,
        _ => unreachable!(),
    };
    let side = ((lane_id() >> t) & 1) == 1;
    let sx = select_unpredictable(side, 0, s);
    let sy = select_unpredictable(side, s, 0);
    let k = select_unpredictable(side, !m, m);

    let send = bitsel(m, *x >> sx, *y << sy);
    let r = unsafe { shuffle_xor(LaneMask::FULL, send, 1 << t) };
    *x = bitsel(k, *x, r << sx);
    *y = bitsel(k, *y, r >> sy);
}

#[inline(always)]
fn swap_subword_bits_with_lane_bits_s_ge_8(x: &mut u32, y: &mut u32, t: u32, w: u32) {
    let s = w << t;
    assert!(s >= 8);
    let side = ((lane_id() >> t) & 1) == 1;

    let (sel_send, sel_x, sel_y) = match s {
        8 => (
            select_unpredictable(side, 0x6240, 0x7351),
            select_unpredictable(side, 0x3614, 0x6240),
            (select_unpredictable(side, 0x3715, 0x7250)),
        ),
        16 => (
            select_unpredictable(side, 0x5410, 0x7632),
            select_unpredictable(side, 0x3254, 0x5410),
            select_unpredictable(side, 0x3276, 0x7610),
        ),

        _ => unreachable!(),
    };

    let send = perm(*x, *y, sel_send);
    let r = unsafe { shuffle_xor(LaneMask::FULL, send, 1 << t) };
    *x = perm(*x, r, sel_x);
    *y = perm(*y, r, sel_y);
}

/// same as __byte_perm in cuda
#[inline(always)]
fn perm(x: u32, y: u32, s: u32) -> u32 {
    x.permute_bytes::<{ PermuteMode::Generic }>(y, s)
}
