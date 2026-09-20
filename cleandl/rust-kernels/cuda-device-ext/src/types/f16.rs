// --- LLM-generated --- //

//! Scalar `f16`.

use crate::primitive::Half;
use cuda_device::f16 as raw;

/// One IEEE binary16 value, carried as the `u16` bit pattern
/// [`cuda_device::f16`] expects.
///
/// This shadows the built-in `f16` primitive, deliberately: it is the name PTX
/// and the rest of this tree use, and the wrapper is what matches the intrinsic
/// ABI. Conversions go through the primitive internally, so they lower to
/// `cvt.rn.f16.f32` and `cvt.f32.f16` rather than to hand-rolled bit
/// arithmetic.
///
/// Arithmetic widens to `f32` and rounds back; see the crate docs for why, and
/// prefer [`crate::f16x2`] when more than one value is in flight.
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct f16(u16);

impl f16 {
    /// Positive zero.
    pub const ZERO: Self = Self(0x0000);
    /// Negative zero.
    pub const NEG_ZERO: Self = Self(0x8000);
    /// 1.0.
    pub const ONE: Self = Self(0x3C00);
    /// -1.0.
    pub const NEG_ONE: Self = Self(0xBC00);
    /// The largest finite value, `65504.0`.
    pub const MAX: Self = Self(0x7BFF);
    /// The smallest finite value, `-65504.0`.
    pub const MIN: Self = Self(0xFBFF);
    /// The smallest positive normal value, approximately `6.10e-5`.
    pub const MIN_POSITIVE: Self = Self(0x0400);
    /// The difference between 1.0 and the next larger value, `2^-10`.
    pub const EPSILON: Self = Self(0x1400);
    /// Positive infinity.
    pub const INFINITY: Self = Self(0x7C00);
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self(0xFC00);
    /// The default quiet `NaN`.
    ///
    /// The `NaN` the `*_nan` min/max forms return is `0x7FFF`, not this value:
    /// PTX canonicalizes to an all-ones exponent and mantissa. See
    /// [`cuda_device::f16`].
    pub const NAN: Self = Self(0x7E00);

    /// Wraps a raw half bit pattern.
    #[must_use]
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw half bit pattern.
    #[must_use]
    #[inline(always)]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Narrows an `f32` using round-to-nearest-even.
    ///
    /// Unlike [`crate::bf16`], the exponent range shrinks too: a finite `f32`
    /// above `65504.0` becomes infinity.
    #[must_use]
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        Self((value as Half).to_bits())
    }

    /// Widens to `f32`. Always exact.
    #[must_use]
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        Half::from_bits(self.0) as f32
    }

    forward_binary!(
        /// Returns the larger of the two values.
        ///
        /// Requires `sm_80`. If either input is `NaN` the result is the other
        /// operand; use [`max_nan`](Self::max_nan) for `NaN` propagation.
        max, raw::max_f16
    );
    forward_binary!(
        /// [`max`](Self::max) with subnormal inputs and results flushed to
        /// zero. Requires `sm_80`.
        max_ftz, raw::max_ftz_f16
    );
    forward_binary!(
        /// Returns the larger value, or a canonical `NaN` if either input is
        /// `NaN`. Requires `sm_80`.
        max_nan, raw::max_nan_f16
    );
    forward_binary!(
        /// [`max_nan`](Self::max_nan) with subnormals flushed to zero.
        /// Requires `sm_80`.
        max_ftz_nan, raw::max_ftz_nan_f16
    );
    forward_binary!(
        /// Compares magnitudes and takes the sign from the XOR of the input
        /// signs. Requires `sm_86`.
        max_xorsign_abs, raw::max_xorsign_abs_f16
    );
    forward_binary!(
        /// [`max_xorsign_abs`](Self::max_xorsign_abs) with subnormals flushed
        /// to zero. Requires `sm_86`.
        max_ftz_xorsign_abs, raw::max_ftz_xorsign_abs_f16
    );
    forward_binary!(
        /// [`max_xorsign_abs`](Self::max_xorsign_abs) with `NaN` propagation.
        /// Requires `sm_86`.
        max_nan_xorsign_abs, raw::max_nan_xorsign_abs_f16
    );
    forward_binary!(
        /// [`max_nan_xorsign_abs`](Self::max_nan_xorsign_abs) with subnormals
        /// flushed to zero. Requires `sm_86`.
        max_ftz_nan_xorsign_abs, raw::max_ftz_nan_xorsign_abs_f16
    );
    forward_binary!(
        /// Returns the smaller of the two values.
        ///
        /// Requires `sm_80`. If either input is `NaN` the result is the other
        /// operand; use [`min_nan`](Self::min_nan) for `NaN` propagation.
        min, raw::min_f16
    );
    forward_binary!(
        /// [`min`](Self::min) with subnormal inputs and results flushed to
        /// zero. Requires `sm_80`.
        min_ftz, raw::min_ftz_f16
    );
    forward_binary!(
        /// Returns the smaller value, or a canonical `NaN` if either input is
        /// `NaN`. Requires `sm_80`.
        min_nan, raw::min_nan_f16
    );
    forward_binary!(
        /// [`min_nan`](Self::min_nan) with subnormals flushed to zero.
        /// Requires `sm_80`.
        min_ftz_nan, raw::min_ftz_nan_f16
    );
    forward_binary!(
        /// Compares magnitudes and takes the sign from the XOR of the input
        /// signs. Requires `sm_86`.
        min_xorsign_abs, raw::min_xorsign_abs_f16
    );
    forward_binary!(
        /// [`min_xorsign_abs`](Self::min_xorsign_abs) with subnormals flushed
        /// to zero. Requires `sm_86`.
        min_ftz_xorsign_abs, raw::min_ftz_xorsign_abs_f16
    );
    forward_binary!(
        /// [`min_xorsign_abs`](Self::min_xorsign_abs) with `NaN` propagation.
        /// Requires `sm_86`.
        min_nan_xorsign_abs, raw::min_nan_xorsign_abs_f16
    );
    forward_binary!(
        /// [`min_nan_xorsign_abs`](Self::min_nan_xorsign_abs) with subnormals
        /// flushed to zero. Requires `sm_86`.
        min_ftz_nan_xorsign_abs, raw::min_ftz_nan_xorsign_abs_f16
    );
}

impl_scalar_common!(f16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_round_trip_through_f32() {
        assert_eq!(f16::from_f32(1.0).to_bits(), f16::ONE.to_bits());
        assert_eq!(f16::from_f32(-1.0).to_bits(), f16::NEG_ONE.to_bits());
        assert_eq!(f16::ONE.to_f32(), 1.0);
        assert_eq!(f16::MAX.to_f32(), 65504.0);
    }

    /// The narrow exponent range is the difference that matters against
    /// [`crate::bf16`], which would keep this value finite.
    #[test]
    fn finite_f32_above_the_range_becomes_infinity() {
        assert_eq!(f16::from_f32(1.0e30).to_bits(), f16::INFINITY.to_bits());
    }

    #[test]
    fn arithmetic_widens_and_rounds_back() {
        assert_eq!((f16::ONE + f16::ONE).to_f32(), 2.0);
        assert_eq!((f16::from_f32(3.0) * f16::from_f32(4.0)).to_f32(), 12.0);
        assert_eq!((f16::from_f32(6.0) / f16::from_f32(2.0)).to_f32(), 3.0);
    }

    #[test]
    fn neg_and_abs_only_touch_the_sign_bit() {
        let value = f16::from_bits(0x1234);
        assert_eq!((-value).to_bits(), 0x9234);
        assert_eq!(f16::from_bits(0x9234).abs().to_bits(), 0x1234);
    }

    #[test]
    fn comparison_follows_float_semantics() {
        assert!(f16::NAN != f16::NAN);
        assert_eq!(f16::ZERO, f16::NEG_ZERO);
        assert!(f16::ONE > f16::ZERO);
    }

    #[test]
    fn classification_matches_the_constants() {
        assert!(f16::NAN.is_nan());
        assert!(f16::NEG_INFINITY.is_infinite());
        assert!(f16::MAX.is_finite());
        assert!(f16::NEG_ZERO.is_sign_negative());
    }

    #[test]
    fn transparent_over_a_single_u16() {
        assert_eq!(core::mem::size_of::<f16>(), 2);
        assert_eq!(core::mem::align_of::<f16>(), core::mem::align_of::<u16>());
    }
}
