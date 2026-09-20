// --- LLM-generated --- //

//! Packed `f16x2`.

use crate::types::f16::f16;
use bytemuck::Pod;
use bytemuck::Zeroable;
use cuda_device::convert;
use cuda_device::f16x2 as raw;

/// Two IEEE binary16 values packed into the `u32` [`cuda_device::f16x2`]
/// expects, low lane first.
///
/// This is the ALU packing for SIMD arithmetic, and the layout that gives wide
/// global loads: reading weights as `u32`/`u64` and unpacking in registers
/// beats scalar half loads. For over-aligned memory element types see
/// [`cuda_device::vector`]; for multi-register value groups see
/// [`cuda_device::cusimd::CuSimd`].
///
/// Every operator except [`Div`](core::ops::Div) is a single device
/// instruction.
#[repr(transparent)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct f16x2(u32);

impl f16x2 {
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
    pub const fn new(lo: f16, hi: f16) -> Self {
        Self(convert::pack_f16_pair(lo.to_bits(), hi.to_bits()))
    }

    /// Packs the same scalar into both lanes.
    #[must_use]
    #[inline(always)]
    pub const fn splat(value: f16) -> Self {
        Self::new(value, value)
    }

    /// Returns the low lane.
    #[must_use]
    #[inline(always)]
    pub const fn lo(self) -> f16 {
        f16::from_bits(self.0 as u16)
    }

    /// Returns the high lane.
    #[must_use]
    #[inline(always)]
    pub const fn hi(self) -> f16 {
        f16::from_bits((self.0 >> 16) as u16)
    }

    /// Returns the pair with the low lane replaced.
    #[must_use]
    #[inline(always)]
    pub const fn with_lo(self, lo: f16) -> Self {
        Self::new(lo, self.hi())
    }

    /// Returns the pair with the high lane replaced.
    #[must_use]
    #[inline(always)]
    pub const fn with_hi(self, hi: f16) -> Self {
        Self::new(self.lo(), hi)
    }

    /// Widens both lanes to `f32`, low lane first. Exact, and host-safe.
    #[must_use]
    #[inline(always)]
    pub fn to_f32_pair(self) -> (f32, f32) {
        convert::cvt_f32x2_f16x2(self.0)
    }

    /// Narrows and packs two `f32` values using round-to-nearest-even.
    ///
    /// A finite `f32` above `65504.0` becomes infinity; `bf16x2` has no such
    /// cliff.
    #[must_use]
    #[inline(always)]
    pub fn from_f32_pair(lo: f32, hi: f32) -> Self {
        Self(convert::cvt_f16x2_f32(lo, hi))
    }

    /// Narrows and packs two `f32` values, rounding toward zero.
    #[must_use]
    #[inline(always)]
    pub fn from_f32_pair_rz(lo: f32, hi: f32) -> Self {
        Self(convert::cvt_rz_f16x2_f32(lo, hi))
    }

    /// Narrows and packs two `f32` values with round-to-nearest-even, clamping
    /// negatives to zero.
    #[must_use]
    #[inline(always)]
    pub fn from_f32_pair_relu(lo: f32, hi: f32) -> Self {
        Self(convert::cvt_rn_relu_f16x2_f32(lo, hi))
    }

    /// Computes `self * b + c` lanewise as one fused operation.
    #[must_use]
    #[inline(always)]
    pub fn fma(self, b: Self, c: Self) -> Self {
        Self(raw::fma_f16x2(self.0, b.0, c.0))
    }

    /// [`fma`](Self::fma) with subnormals flushed to zero.
    #[must_use]
    #[inline(always)]
    pub fn fma_ftz(self, b: Self, c: Self) -> Self {
        Self(raw::fma_ftz_f16x2(self.0, b.0, c.0))
    }

    /// [`fma`](Self::fma) with the result clamped to zero from below.
    #[must_use]
    #[inline(always)]
    pub fn fma_relu(self, b: Self, c: Self) -> Self {
        Self(raw::fma_relu_f16x2(self.0, b.0, c.0))
    }

    /// [`fma`](Self::fma) with the result saturated to `[0.0, 1.0]`.
    #[must_use]
    #[inline(always)]
    pub fn fma_sat(self, b: Self, c: Self) -> Self {
        Self(raw::fma_sat_f16x2(self.0, b.0, c.0))
    }

    /// [`fma_relu`](Self::fma_relu) with subnormals flushed to zero.
    #[must_use]
    #[inline(always)]
    pub fn fma_ftz_relu(self, b: Self, c: Self) -> Self {
        Self(raw::fma_ftz_relu_f16x2(self.0, b.0, c.0))
    }

    /// [`fma_sat`](Self::fma_sat) with subnormals flushed to zero.
    #[must_use]
    #[inline(always)]
    pub fn fma_ftz_sat(self, b: Self, c: Self) -> Self {
        Self(raw::fma_ftz_sat_f16x2(self.0, b.0, c.0))
    }

    forward_binary!(
        /// Returns the larger value in each lane.
        ///
        /// Requires `sm_80`. A `NaN` lane yields the other operand; use
        /// [`max_nan`](Self::max_nan) for `NaN` propagation.
        max, raw::max_f16x2
    );
    forward_binary!(
        /// [`max`](Self::max) with subnormals flushed to zero. Requires
        /// `sm_80`.
        max_ftz, raw::max_ftz_f16x2
    );
    forward_binary!(
        /// Lanewise max, propagating a canonical `NaN`. Requires `sm_80`.
        max_nan, raw::max_nan_f16x2
    );
    forward_binary!(
        /// [`max_nan`](Self::max_nan) with subnormals flushed to zero.
        /// Requires `sm_80`.
        max_ftz_nan, raw::max_ftz_nan_f16x2
    );
    forward_binary!(
        /// Lanewise max of magnitudes, sign taken from the XOR of the input
        /// signs. Requires `sm_86`.
        max_xorsign_abs, raw::max_xorsign_abs_f16x2
    );
    forward_binary!(
        /// [`max_xorsign_abs`](Self::max_xorsign_abs) with subnormals flushed
        /// to zero. Requires `sm_86`.
        max_ftz_xorsign_abs, raw::max_ftz_xorsign_abs_f16x2
    );
    forward_binary!(
        /// [`max_xorsign_abs`](Self::max_xorsign_abs) with `NaN` propagation.
        /// Requires `sm_86`.
        max_nan_xorsign_abs, raw::max_nan_xorsign_abs_f16x2
    );
    forward_binary!(
        /// [`max_nan_xorsign_abs`](Self::max_nan_xorsign_abs) with subnormals
        /// flushed to zero. Requires `sm_86`.
        max_ftz_nan_xorsign_abs, raw::max_ftz_nan_xorsign_abs_f16x2
    );
    forward_binary!(
        /// Returns the smaller value in each lane.
        ///
        /// Requires `sm_80`. A `NaN` lane yields the other operand; use
        /// [`min_nan`](Self::min_nan) for `NaN` propagation.
        min, raw::min_f16x2
    );
    forward_binary!(
        /// [`min`](Self::min) with subnormals flushed to zero. Requires
        /// `sm_80`.
        min_ftz, raw::min_ftz_f16x2
    );
    forward_binary!(
        /// Lanewise min, propagating a canonical `NaN`. Requires `sm_80`.
        min_nan, raw::min_nan_f16x2
    );
    forward_binary!(
        /// [`min_nan`](Self::min_nan) with subnormals flushed to zero.
        /// Requires `sm_80`.
        min_ftz_nan, raw::min_ftz_nan_f16x2
    );
    forward_binary!(
        /// Lanewise min of magnitudes, sign taken from the XOR of the input
        /// signs. Requires `sm_86`.
        min_xorsign_abs, raw::min_xorsign_abs_f16x2
    );
    forward_binary!(
        /// [`min_xorsign_abs`](Self::min_xorsign_abs) with subnormals flushed
        /// to zero. Requires `sm_86`.
        min_ftz_xorsign_abs, raw::min_ftz_xorsign_abs_f16x2
    );
    forward_binary!(
        /// [`min_xorsign_abs`](Self::min_xorsign_abs) with `NaN` propagation.
        /// Requires `sm_86`.
        min_nan_xorsign_abs, raw::min_nan_xorsign_abs_f16x2
    );
    forward_binary!(
        /// [`min_nan_xorsign_abs`](Self::min_nan_xorsign_abs) with subnormals
        /// flushed to zero. Requires `sm_86`.
        min_ftz_nan_xorsign_abs, raw::min_ftz_nan_xorsign_abs_f16x2
    );
}

impl_packed_common!(
    f16x2,
    f16,
    raw::add_f16x2,
    raw::sub_f16x2,
    raw::mul_f16x2,
    raw::neg_f16x2,
    raw::abs_f16x2
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_lane_occupies_the_low_bits() {
        let packed = f16x2::new(f16::ONE, f16::from_f32(2.0));
        assert_eq!(packed.to_bits(), 0x4000_3C00);
        assert_eq!(packed.lo().to_bits(), f16::ONE.to_bits());
        assert_eq!(packed.hi().to_f32(), 2.0);
        assert_eq!(packed.to_f32_pair(), (1.0, 2.0));
    }

    #[test]
    fn sign_bits_stay_in_their_own_lane() {
        let packed = f16x2::new(f16::ONE, f16::NEG_ONE);
        assert_eq!(packed.to_f32_pair(), (1.0, -1.0));
    }

    #[test]
    fn lane_replacement_leaves_the_other_lane_alone() {
        let packed = f16x2::splat(f16::ONE);
        assert_eq!(packed.to_f32_pair(), (1.0, 1.0));
        assert_eq!(packed.with_hi(f16::from_f32(4.0)).to_f32_pair(), (1.0, 4.0));
        assert_eq!(packed.with_lo(f16::from_f32(4.0)).to_f32_pair(), (4.0, 1.0));
    }

    #[test]
    fn equality_compares_lanes_as_floats() {
        assert_eq!(f16x2::ZERO, f16x2::splat(f16::NEG_ZERO));
        assert!(f16x2::splat(f16::NAN) != f16x2::splat(f16::NAN));
    }

    #[test]
    fn transparent_over_a_single_u32() {
        assert_eq!(core::mem::size_of::<f16x2>(), 4);
        assert_eq!(core::mem::align_of::<f16x2>(), core::mem::align_of::<u32>());
    }
}
