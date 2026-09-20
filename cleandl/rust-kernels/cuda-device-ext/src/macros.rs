// --- LLM-generated --- //

//! Internal macros shared by the four wrapper types.

/// Implements the surface every 16-bit scalar shares.
///
/// Arithmetic widens to `f32` and rounds back, since PTX has no scalar 16-bit
/// ALU ops. `Neg` and `abs` stay in the bit domain: both formats put the sign
/// in the top bit, so flipping or clearing it is exact and cannot perturb a
/// `NaN` payload the way a round trip through `f32` can.
macro_rules! impl_scalar_common {
    ($ty:ident) => {
        impl $ty {
            /// Returns the value with the sign bit cleared.
            #[must_use]
            #[inline(always)]
            pub const fn abs(self) -> Self {
                Self(self.0 & 0x7FFF)
            }

            /// Returns `true` if the value is `NaN`.
            #[must_use]
            #[inline(always)]
            pub const fn is_nan(self) -> bool {
                (self.0 & 0x7FFF) > (Self::INFINITY.0 & 0x7FFF)
            }

            /// Returns `true` if the value is positive or negative infinity.
            #[must_use]
            #[inline(always)]
            pub const fn is_infinite(self) -> bool {
                (self.0 & 0x7FFF) == (Self::INFINITY.0 & 0x7FFF)
            }

            /// Returns `true` if the value is neither `NaN` nor infinite.
            #[must_use]
            #[inline(always)]
            pub const fn is_finite(self) -> bool {
                (self.0 & 0x7FFF) < (Self::INFINITY.0 & 0x7FFF)
            }

            /// Returns `true` if the sign bit is set, including for `-0.0` and
            /// negative `NaN`.
            #[must_use]
            #[inline(always)]
            pub const fn is_sign_negative(self) -> bool {
                (self.0 & 0x8000) != 0
            }
        }

        impl ::core::ops::Neg for $ty {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self {
                Self(self.0 ^ 0x8000)
            }
        }

        impl_scalar_binop!($ty, Add, add, AddAssign, add_assign, +);
        impl_scalar_binop!($ty, Sub, sub, SubAssign, sub_assign, -);
        impl_scalar_binop!($ty, Mul, mul, MulAssign, mul_assign, *);
        impl_scalar_binop!($ty, Div, div, DivAssign, div_assign, /);
        impl_scalar_binop!($ty, Rem, rem, RemAssign, rem_assign, %);

        impl PartialEq for $ty {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                self.to_f32() == other.to_f32()
            }
        }

        impl PartialOrd for $ty {
            #[inline(always)]
            fn partial_cmp(&self, other: &Self) -> Option<::core::cmp::Ordering> {
                self.to_f32().partial_cmp(&other.to_f32())
            }
        }

        impl From<$ty> for f32 {
            #[inline(always)]
            fn from(value: $ty) -> f32 {
                value.to_f32()
            }
        }

        impl From<f32> for $ty {
            #[inline(always)]
            fn from(value: f32) -> Self {
                Self::from_f32(value)
            }
        }

        impl ::core::fmt::Debug for $ty {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Debug::fmt(&self.to_f32(), f)
            }
        }

        impl ::core::fmt::Display for $ty {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Display::fmt(&self.to_f32(), f)
            }
        }
    };
}

/// One scalar operator, computed in `f32` and rounded back.
macro_rules! impl_scalar_binop {
    ($ty:ident, $trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt) => {
        impl ::core::ops::$trait for $ty {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                Self::from_f32(self.to_f32() $op rhs.to_f32())
            }
        }

        impl ::core::ops::$assign_trait for $ty {
            #[inline(always)]
            fn $assign_method(&mut self, rhs: Self) {
                *self = ::core::ops::$trait::$method(*self, rhs);
            }
        }
    };
}

/// Implements the surface both packed pairs share.
///
/// Unlike the scalars these forward to real SIMD instructions, so every
/// operator here is one device instruction rather than a widening round trip.
/// `Div` is the exception: PTX has no packed 16-bit divide, so it unpacks,
/// divides in `f32`, and repacks.
macro_rules! impl_packed_common {
    ($ty:ident, $scalar:ident, $add:path, $sub:path, $mul:path, $neg:path, $abs:path) => {
        impl $ty {
            /// Returns the pair with both lanes' sign bits cleared.
            #[must_use]
            #[inline(always)]
            pub fn abs(self) -> Self {
                Self($abs(self.0))
            }
        }

        impl ::core::ops::Neg for $ty {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self {
                Self($neg(self.0))
            }
        }

        impl_packed_binop!($ty, Add, add, AddAssign, add_assign, $add);
        impl_packed_binop!($ty, Sub, sub, SubAssign, sub_assign, $sub);
        impl_packed_binop!($ty, Mul, mul, MulAssign, mul_assign, $mul);

        impl ::core::ops::Div for $ty {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                let (a_lo, a_hi) = self.to_f32_pair();
                let (b_lo, b_hi) = rhs.to_f32_pair();
                Self::from_f32_pair(a_lo / b_lo, a_hi / b_hi)
            }
        }

        impl ::core::ops::DivAssign for $ty {
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {
                *self = ::core::ops::Div::div(*self, rhs);
            }
        }

        impl PartialEq for $ty {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                self.to_f32_pair() == other.to_f32_pair()
            }
        }

        impl From<($scalar, $scalar)> for $ty {
            #[inline(always)]
            fn from((lo, hi): ($scalar, $scalar)) -> Self {
                Self::new(lo, hi)
            }
        }

        impl From<$ty> for ($scalar, $scalar) {
            #[inline(always)]
            fn from(value: $ty) -> Self {
                (value.lo(), value.hi())
            }
        }

        impl ::core::fmt::Debug for $ty {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let (lo, hi) = self.to_f32_pair();
                f.debug_tuple(stringify!($ty))
                    .field(&lo)
                    .field(&hi)
                    .finish()
            }
        }
    };
}

/// One packed operator, forwarded to its SIMD intrinsic.
macro_rules! impl_packed_binop {
    ($ty:ident, $trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $intrinsic:path) => {
        impl ::core::ops::$trait for $ty {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                Self($intrinsic(self.0, rhs.0))
            }
        }

        impl ::core::ops::$assign_trait for $ty {
            #[inline(always)]
            fn $assign_method(&mut self, rhs: Self) {
                *self = ::core::ops::$trait::$method(*self, rhs);
            }
        }
    };
}

/// Wraps a two-operand intrinsic as an inherent method on a wrapper type.
///
/// The generated names encode their PTX modifiers (`ftz`, `NaN`,
/// `xorsign.abs`), and the methods keep them: each modifier is a distinct
/// instruction with distinct `NaN` and subnormal behaviour, so collapsing them
/// under one name would hide the choice.
macro_rules! forward_binary {
    ($(#[$meta:meta])* $name:ident, $intrinsic:path) => {
        $(#[$meta])*
        #[must_use]
        #[inline(always)]
        pub fn $name(self, other: Self) -> Self {
            Self($intrinsic(self.0, other.0))
        }
    };
}
