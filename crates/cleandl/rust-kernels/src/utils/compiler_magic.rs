use core::mem::MaybeUninit;
use ptx_std::{
    warp_collectives::{LaneMask, shuffle},
    warp_id,
};

/// Attempts to prevent LLVM from applying LICM to the provided value,
/// by passing it through an empty inline assembly block labeled as having side-effects.
/// Unlike `core::hint::black_box`, this will not result in local memory usage from
/// pushing the value onto the stack.
#[inline(always)]
pub fn make_opaque<T: MakeOpaque>(val: T) -> T {
    val.make_opaque()
}

/// Returns a warp ID known by ptxas to be uniform across the warp.
/// See similar trick in CUTLASS (canonical_warp_idx_sync)
///
/// Warp must be fully converged upon calling this.
#[inline(always)]
pub unsafe fn uniform_warp_id() -> u32 {
    unsafe { shuffle(LaneMask::FULL, warp_id(), 0) }
}

#[inline(always)]
pub fn array_map_unrolled<const N: usize, T, U>(arr: [T; N], mut map: impl FnMut(T) -> U) -> [U; N]
where
    T: Copy,
    U: Copy,
{
    unsafe {
        let mut res = [MaybeUninit::<U>::uninit(); N];
        #[rustc_unroll(full)]
        for i in 0..N {
            res[i] = MaybeUninit::new(map(arr[i]));
        }
        MaybeUninit::array_assume_init(res)
    }
}

#[inline(always)]
pub fn array_zip_unrolled<const N: usize, T1, T2, U>(
    arr1: [T1; N],
    arr2: [T2; N],
    mut map: impl FnMut(T1, T2) -> U,
) -> [U; N]
where
    T1: Copy,
    T2: Copy,
    U: Copy,
{
    unsafe {
        let mut res = [MaybeUninit::<U>::uninit(); N];
        #[rustc_unroll(full)]
        for i in 0..N {
            res[i] = MaybeUninit::new(map(arr1[i], arr2[i]));
        }
        MaybeUninit::array_assume_init(res)
    }
}

pub trait MakeOpaque: sealed::MakeOpaqueSealed {}

impl<T> MakeOpaque for T where T: sealed::MakeOpaqueSealed {}

mod sealed {
    use crate::utils::compiler_magic::make_opaque;
    use core::arch::asm;
    use ptx_std::{bf16x2, bf16x4, bf16x8};

    pub trait MakeOpaqueSealed {
        fn make_opaque(self) -> Self;
    }

    impl MakeOpaqueSealed for bf16x2 {
        #[inline(always)]
        fn make_opaque(self) -> Self {
            let mut bits = self.to_bits();
            unsafe {
                asm!("/* {0} */", inout(reg32) bits, options(nomem, nostack));
            }
            Self::from_bits(bits)
        }
    }

    impl MakeOpaqueSealed for bf16x4 {
        #[inline(always)]
        fn make_opaque(self) -> Self {
            Self::from_x2s([make_opaque(self.to_x2s()[0]), make_opaque(self.to_x2s()[1])])
        }
    }

    impl MakeOpaqueSealed for bf16x8 {
        #[inline(always)]
        fn make_opaque(self) -> Self {
            Self::from_x2s([
                make_opaque(self.to_x2s()[0]),
                make_opaque(self.to_x2s()[1]),
                make_opaque(self.to_x2s()[2]),
                make_opaque(self.to_x2s()[3]),
            ])
        }
    }

    impl MakeOpaqueSealed for f32 {
        #[inline(always)]
        fn make_opaque(mut self) -> Self {
            unsafe {
                asm!("/* {0} */", inout(reg32) self, options(nomem, nostack));
            }
            self
        }
    }
}
