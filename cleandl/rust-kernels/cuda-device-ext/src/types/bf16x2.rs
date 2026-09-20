// --- LLM-generated --- //

//! Packed `bf16x2`.

use crate::types::bf16::bf16;
use cuda_device::bf16x2 as raw;
use cuda_device::convert;

/// Two bfloat16 values packed into the `u32` [`cuda_device::bf16x2`] expects,
/// low lane first.
///
/// This is the ALU packing for SIMD arithmetic, and the layout that gives wide
/// global loads: one 32-bit load carries two values where two scalar loads
/// would not. For over-aligned memory element types see
/// [`cuda_device::vector`]; for multi-register value groups see
/// [`cuda_device::cusimd::CuSimd`].
///
/// Every operator except [`Div`](core::ops::Div) is a single device
/// instruction.
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct bf16x2(u32);

impl bf16x2 {
    /// Both lanes positive zero.
    pub const ZERO: Self = Self(0);

    /// Wraps a raw packed bit pattern.
    #[must_use]
    #[inline(always)]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw packed bit pattern.
    #[must_use]
    #[inline(always)]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    /// Packs two scalars, `lo` into the low 16 bits.
    #[must_use]
    #[inline(always)]
    pub const fn new(lo: bf16, hi: bf16) -> Self {
        Self(convert::pack_bf16_pair(lo.to_bits(), hi.to_bits()))
    }

    /// Packs the same scalar into both lanes.
    #[must_use]
    #[inline(always)]
    pub const fn splat(value: bf16) -> Self {
        Self::new(value, value)
    }

    /// Returns the low lane.
    #[must_use]
    #[inline(always)]
    pub const fn lo(self) -> bf16 {
        bf16::from_bits(self.0 as u16)
    }

    /// Returns the high lane.
    #[must_use]
    #[inline(always)]
    pub const fn hi(self) -> bf16 {
        bf16::from_bits((self.0 >> 16) as u16)
    }

    /// Returns the pair with the low lane replaced.
    #[must_use]
    #[inline(always)]
    pub const fn with_lo(self, lo: bf16) -> Self {
        Self::new(lo, self.hi())
    }

    /// Returns the pair with the high lane replaced.
    #[must_use]
    #[inline(always)]
    pub const fn with_hi(self, hi: bf16) -> Self {
        Self::new(self.lo(), hi)
    }

    /// Widens both lanes to `f32`, low lane first. Exact, and host-safe: `bf16`
    /// is a prefix of `f32`, so this is a shift rather than a conversion.
    #[must_use]
    #[inline(always)]
    pub fn to_f32_pair(self) -> (f32, f32) {
        convert::cvt_f32x2_bf16x2(self.0)
    }

    /// Narrows and packs two `f32` values using round-to-nearest-even.
    #[must_use]
    #[inline(always)]
    pub fn from_f32_pair(lo: f32, hi: f32) -> Self {
        Self(convert::cvt_bf16x2_f32(lo, hi))
    }

    /// Narrows and packs two `f32` values, rounding toward zero.
    #[must_use]
    #[inline(always)]
    pub fn from_f32_pair_rz(lo: f32, hi: f32) -> Self {
        Self(convert::cvt_rz_bf16x2_f32(lo, hi))
    }

    /// Narrows and packs two `f32` values with round-to-nearest-even, clamping
    /// negatives to zero.
    #[must_use]
    #[inline(always)]
    pub fn from_f32_pair_relu(lo: f32, hi: f32) -> Self {
        Self(convert::cvt_rn_relu_bf16x2_f32(lo, hi))
    }

    /// Computes `self * b + c` lanewise as one fused operation.
    #[must_use]
    #[inline(always)]
    pub fn fma(self, b: Self, c: Self) -> Self {
        Self(raw::fma_bf16x2(self.0, b.0, c.0))
    }

    /// [`fma`](Self::fma) with the result clamped to zero from below.
    ///
    /// The `ftz` and `sat` forms [`crate::f16x2`] has do not exist for
    /// `bf16x2`: ptxas rejects `fma.rn.ftz.bf16x2` outright.
    #[must_use]
    #[inline(always)]
    pub fn fma_relu(self, b: Self, c: Self) -> Self {
        Self(raw::fma_relu_bf16x2(self.0, b.0, c.0))
    }

    forward_binary!(
        /// Returns the larger value in each lane.
        ///
        /// Requires `sm_80`. A `NaN` lane yields the other operand; use
        /// [`max_nan`](Self::max_nan) for `NaN` propagation.
        max, raw::max_bf16x2
    );
    forward_binary!(
        /// Lanewise max, propagating a canonical `NaN`. Requires `sm_80`.
        max_nan, raw::max_nan_bf16x2
    );
    forward_binary!(
        /// Lanewise max of magnitudes, sign taken from the XOR of the input
        /// signs. Requires `sm_86`.
        max_xorsign_abs, raw::max_xorsign_abs_bf16x2
    );
    forward_binary!(
        /// [`max_xorsign_abs`](Self::max_xorsign_abs) with `NaN` propagation.
        /// Requires `sm_86`.
        max_nan_xorsign_abs, raw::max_nan_xorsign_abs_bf16x2
    );
    forward_binary!(
        /// Returns the smaller value in each lane.
        ///
        /// Requires `sm_80`. A `NaN` lane yields the other operand; use
        /// [`min_nan`](Self::min_nan) for `NaN` propagation.
        min, raw::min_bf16x2
    );
    forward_binary!(
        /// Lanewise min, propagating a canonical `NaN`. Requires `sm_80`.
        min_nan, raw::min_nan_bf16x2
    );
    forward_binary!(
        /// Lanewise min of magnitudes, sign taken from the XOR of the input
        /// signs. Requires `sm_86`.
        min_xorsign_abs, raw::min_xorsign_abs_bf16x2
    );
    forward_binary!(
        /// [`min_xorsign_abs`](Self::min_xorsign_abs) with `NaN` propagation.
        /// Requires `sm_86`.
        min_nan_xorsign_abs, raw::min_nan_xorsign_abs_bf16x2
    );
}

impl_packed_common!(
    bf16x2,
    bf16,
    raw::add_bf16x2,
    raw::sub_bf16x2,
    raw::mul_bf16x2,
    raw::neg_bf16x2,
    raw::abs_bf16x2
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Arithmetic, `Neg` and `abs` are intrinsics and panic on the host, so
    /// these cover the packing side: lane order is the part that can silently
    /// invert.
    #[test]
    fn low_lane_occupies_the_low_bits() {
        let packed = bf16x2::new(bf16::ONE, bf16::from_f32(2.0));
        assert_eq!(packed.to_bits(), 0x4000_3F80);
        assert_eq!(packed.lo().to_bits(), bf16::ONE.to_bits());
        assert_eq!(packed.hi().to_f32(), 2.0);
        assert_eq!(packed.to_f32_pair(), (1.0, 2.0));
    }

    /// A negative lane must not bleed into the other half.
    #[test]
    fn sign_bits_stay_in_their_own_lane() {
        let packed = bf16x2::new(bf16::ONE, bf16::NEG_ONE);
        assert_eq!(packed.to_f32_pair(), (1.0, -1.0));
    }

    #[test]
    fn lane_replacement_leaves_the_other_lane_alone() {
        let packed = bf16x2::splat(bf16::ONE);
        assert_eq!(packed.to_f32_pair(), (1.0, 1.0));
        assert_eq!(
            packed.with_hi(bf16::from_f32(4.0)).to_f32_pair(),
            (1.0, 4.0)
        );
        assert_eq!(
            packed.with_lo(bf16::from_f32(4.0)).to_f32_pair(),
            (4.0, 1.0)
        );
    }

    #[test]
    fn equality_compares_lanes_as_floats() {
        assert_eq!(bf16x2::ZERO, bf16x2::splat(bf16::NEG_ZERO));
        assert!(bf16x2::splat(bf16::NAN) != bf16x2::splat(bf16::NAN));
    }

    #[test]
    fn transparent_over_a_single_u32() {
        assert_eq!(core::mem::size_of::<bf16x2>(), 4);
        assert_eq!(
            core::mem::align_of::<bf16x2>(),
            core::mem::align_of::<u32>()
        );
    }
}
