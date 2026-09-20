// --- LLM-generated --- //

//! Scalar `bf16`.

use bytemuck::Pod;
use bytemuck::Zeroable;
use cuda_device::bf16 as raw;
use cuda_device::convert;

/// One bfloat16 value, carried as the `u16` bit pattern
/// [`cuda_device::bf16`] expects.
///
/// `bf16` is `f32` truncated to its top 16 bits: same 8-bit exponent, 7 bits of
/// mantissa instead of 23. Widening to `f32` is therefore exact and free, and
/// narrowing only rounds the mantissa — a `bf16` cannot overflow to infinity
/// converting from a finite `f32` the way [`crate::f16`] can.
///
/// Arithmetic widens to `f32` and rounds back.
#[repr(transparent)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct bf16(u16);

impl bf16 {
    /// Positive zero.
    pub const ZERO: Self = Self(0x0000);
    /// Negative zero.
    pub const NEG_ZERO: Self = Self(0x8000);
    /// 1.0.
    pub const ONE: Self = Self(0x3F80);
    /// -1.0.
    pub const NEG_ONE: Self = Self(0xBF80);
    /// The largest finite value, approximately `3.39e38`.
    pub const MAX: Self = Self(0x7F7F);
    /// The smallest finite value, approximately `-3.39e38`.
    pub const MIN: Self = Self(0xFF7F);
    /// The smallest positive normal value, approximately `1.18e-38`.
    pub const MIN_POSITIVE: Self = Self(0x0080);
    /// The difference between 1.0 and the next larger value, `2^-7`.
    pub const EPSILON: Self = Self(0x3C00);
    /// Positive infinity.
    pub const INFINITY: Self = Self(0x7F80);
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self(0xFF80);
    /// The default quiet `NaN`.
    ///
    /// The `NaN` the `*_nan` min/max forms return is `0x7FFF`, not this value:
    /// PTX canonicalizes to an all-ones exponent and mantissa. See
    /// [`cuda_device::bf16`].
    pub const NAN: Self = Self(0x7FC0);

    /// Wraps a raw bfloat16 bit pattern.
    #[must_use]
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bfloat16 bit pattern.
    #[must_use]
    #[inline(always)]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Narrows an `f32` using round-to-nearest-even.
    #[must_use]
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        Self(convert::f32_to_bf16_rne(value))
    }

    /// Narrows an `f32` by truncating its low 16 bits.
    ///
    /// Cheaper than [`from_f32`](Self::from_f32) — it is a shift rather than a
    /// rounding step — but an `f32` `NaN` whose payload lives only in the
    /// discarded bits becomes an infinity.
    #[must_use]
    #[inline(always)]
    pub fn from_f32_rz(value: f32) -> Self {
        Self(convert::f32_to_bf16(value))
    }

    /// Widens to `f32`. Exact: `bf16` is a prefix of `f32`.
    #[must_use]
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        convert::bf16_to_f32(self.0)
    }

    forward_binary!(
        /// Returns the larger of the two values.
        ///
        /// Requires `sm_80`. If either input is `NaN` the result is the other
        /// operand; use [`max_nan`](Self::max_nan) for `NaN` propagation.
        max, raw::max_bf16
    );
    forward_binary!(
        /// Returns the larger value, or a canonical `NaN` if either input is
        /// `NaN`. Requires `sm_80`.
        max_nan, raw::max_nan_bf16
    );
    forward_binary!(
        /// Compares magnitudes and takes the sign from the XOR of the input
        /// signs. Requires `sm_86`.
        max_xorsign_abs, raw::max_xorsign_abs_bf16
    );
    forward_binary!(
        /// [`max_xorsign_abs`](Self::max_xorsign_abs) with `NaN` propagation.
        /// Requires `sm_86`.
        max_nan_xorsign_abs, raw::max_nan_xorsign_abs_bf16
    );
    forward_binary!(
        /// Returns the smaller of the two values.
        ///
        /// Requires `sm_80`. If either input is `NaN` the result is the other
        /// operand; use [`min_nan`](Self::min_nan) for `NaN` propagation.
        min, raw::min_bf16
    );
    forward_binary!(
        /// Returns the smaller value, or a canonical `NaN` if either input is
        /// `NaN`. Requires `sm_80`.
        min_nan, raw::min_nan_bf16
    );
    forward_binary!(
        /// Compares magnitudes and takes the sign from the XOR of the input
        /// signs. Requires `sm_86`.
        min_xorsign_abs, raw::min_xorsign_abs_bf16
    );
    forward_binary!(
        /// [`min_xorsign_abs`](Self::min_xorsign_abs) with `NaN` propagation.
        /// Requires `sm_86`.
        min_nan_xorsign_abs, raw::min_nan_xorsign_abs_bf16
    );
}

impl_scalar_common!(bf16);

#[cfg(test)]
mod tests {
    use super::*;

    /// The intrinsic-backed methods are device-only, but everything reached
    /// through `f32` is ordinary Rust and holds on the host too.
    #[test]
    fn conversions_round_trip_through_f32() {
        assert_eq!(bf16::from_f32(1.0).to_bits(), bf16::ONE.to_bits());
        assert_eq!(bf16::from_f32(-1.0).to_bits(), bf16::NEG_ONE.to_bits());
        assert_eq!(bf16::ONE.to_f32(), 1.0);
        assert_eq!(
            bf16::from_f32(f32::INFINITY).to_bits(),
            bf16::INFINITY.to_bits()
        );
    }

    /// The truncating form drops the low 16 bits; the default form rounds, so
    /// a value just under the halfway point goes the other way.
    #[test]
    fn rz_and_rne_disagree_where_they_should() {
        let value = f32::from_bits(0x3F80_FFFF);
        assert_eq!(bf16::from_f32_rz(value).to_bits(), 0x3F80);
        assert_eq!(bf16::from_f32(value).to_bits(), 0x3F81);
    }

    #[test]
    fn arithmetic_widens_and_rounds_back() {
        assert_eq!((bf16::ONE + bf16::ONE).to_f32(), 2.0);
        assert_eq!((bf16::ONE - bf16::ONE).to_f32(), 0.0);
        assert_eq!((bf16::from_f32(3.0) * bf16::from_f32(4.0)).to_f32(), 12.0);
        assert_eq!((bf16::from_f32(6.0) / bf16::from_f32(2.0)).to_f32(), 3.0);
    }

    /// Sign handling stays in the bit domain, so it must not disturb anything
    /// else in the pattern.
    #[test]
    fn neg_and_abs_only_touch_the_sign_bit() {
        let value = bf16::from_bits(0x1234);
        assert_eq!((-value).to_bits(), 0x9234);
        assert_eq!((-(-value)).to_bits(), 0x1234);
        assert_eq!(bf16::from_bits(0x9234).abs().to_bits(), 0x1234);
    }

    /// IEEE comparison semantics, not bit equality: that is why `Eq` and `Hash`
    /// are absent.
    #[test]
    fn comparison_follows_float_semantics() {
        assert!(bf16::NAN != bf16::NAN);
        assert_eq!(bf16::ZERO, bf16::NEG_ZERO);
        assert_ne!(bf16::ZERO.to_bits(), bf16::NEG_ZERO.to_bits());
        assert!(bf16::ONE > bf16::ZERO);
        assert!(bf16::NEG_INFINITY < bf16::MIN);
    }

    #[test]
    fn classification_matches_the_constants() {
        assert!(bf16::NAN.is_nan());
        assert!(!bf16::INFINITY.is_nan());
        assert!(bf16::NEG_INFINITY.is_infinite());
        assert!(bf16::MAX.is_finite());
        assert!(bf16::NEG_ZERO.is_sign_negative());
        assert!(!bf16::ZERO.is_sign_negative());
    }

    #[test]
    fn transparent_over_a_single_u16() {
        assert_eq!(core::mem::size_of::<bf16>(), 2);
        assert_eq!(core::mem::align_of::<bf16>(), core::mem::align_of::<u16>());
    }
}
