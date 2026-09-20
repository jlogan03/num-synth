use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// One number represented by a normalized, unevaluated sum of two `f32`s.
///
/// Ordinary values have roughly 48 significant bits, with the exponent range
/// of `f32`. Operations are not generally correctly rounded to the nearest
/// pair. Overflowing leading intermediates are terminal; residual underflow
/// reduces precision. Requires nearest-even arithmetic without reassociation
/// or flushing of required subnormal intermediates.
///
/// ```
/// use num_synth::Df32;
/// let origin = Df32::from_f32(10.0);
/// let offset = Df32::from_f32(1e-6);
/// assert_eq!((origin + offset) - origin, offset);
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Df32 {
    hi: f32,
    lo: f32,
}

// Zero wraps to u32::MAX after subtracting one. Nonzero finite magnitudes
// remain below the threshold; infinities and NaNs are at or above it.
#[inline]
const fn is_nonfinite_or_zero(value: f32) -> bool {
    let magnitude = value.to_bits() & 0x7fff_ffff;
    magnitude.wrapping_sub(1) >= 0x7f7f_ffff
}

// No magnitude precondition. Keep each operation individually rounded.
#[inline]
const fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

#[inline]
fn two_prod(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    (p, libm::fmaf(a, b, -p))
}

// Fixed two-component compensated sum, not a normalized public value.
// Each TwoSum is exact in its range; only accumulation of its errors rounds.
#[derive(Clone, Copy)]
struct Correction {
    hi: f32,
    lo: f32,
}

impl Correction {
    #[inline]
    fn add(self, x: f32) -> Self {
        let (hi, error) = two_sum(self.hi, x);
        Self {
            hi,
            lo: self.lo + error,
        }
    }
}

impl Df32 {
    pub const ZERO: Self = Self::from_f32(0.0);
    pub const NEG_ZERO: Self = Self::from_f32(-0.0);
    pub const ONE: Self = Self::from_f32(1.0);
    pub const NAN: Self = Self {
        hi: f32::NAN,
        lo: 0.0,
    };
    pub const INFINITY: Self = Self::from_f32(f32::INFINITY);
    pub const NEG_INFINITY: Self = Self::from_f32(f32::NEG_INFINITY);

    /// Stores the scalar directly, with a zero residual.
    #[inline]
    pub const fn from_f32(value: f32) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    /// Splits a binary64 input at the boundary. Not an exact f64 round-trip.
    /// Values whose leading conversion overflows become signed infinity.
    #[inline]
    pub const fn from_f64(value: f64) -> Self {
        let hi = value as f32;
        // Avoid inf-inf in the residual and preserve signed zero, including
        // finite inputs that underflow to f32 zero. Their residual also rounds
        // to zero, so splitting them cannot recover another component.
        if is_nonfinite_or_zero(hi) {
            return Self::from_f32(hi);
        }
        Self::from_parts(hi, (value - hi as f64) as f32)
    }

    /// Normalizes the sum of arbitrary components, without a magnitude
    /// precondition. Opposite infinities and NaNs yield NaN; its bits are unspecified.
    #[inline]
    pub const fn from_parts(hi: f32, lo: f32) -> Self {
        let sum = hi + lo;
        // TwoSum's error recovery would turn a valid infinity into NaN.
        if !sum.is_finite() {
            return Self::from_f32(sum);
        }
        let (_, error) = two_sum(hi, lo);
        Self {
            hi: sum,
            lo: if error == 0.0 { 0.0 } else { error },
        }
    }

    /// Returns the normalized components without collapsing the residual.
    #[inline]
    pub const fn to_parts(self) -> (f32, f32) {
        (self.hi, self.lo)
    }

    /// Rounds the stored value to f32, discarding the residual.
    #[inline]
    pub const fn to_f32(self) -> f32 {
        self.hi
    }

    /// Rounds the stored sum to f64. A large exponent gap can still lose bits.
    #[inline]
    pub fn to_f64(self) -> f64 {
        if self.hi == 0.0 {
            self.hi as f64 // Preserve negative zero.
        } else {
            self.hi as f64 + self.lo as f64
        }
    }

    #[inline]
    pub const fn is_finite(self) -> bool {
        self.hi.is_finite()
    }

    #[inline]
    pub const fn is_nan(self) -> bool {
        self.hi.is_nan()
    }

    #[inline]
    pub const fn is_infinite(self) -> bool {
        self.hi.is_infinite()
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.hi == 0.0
    }

    #[inline]
    pub const fn is_sign_negative(self) -> bool {
        self.hi.is_sign_negative()
    }

    /// Computes `self * b + c` without first rounding the product to Df32.
    ///
    /// Retains all four component products and their fused residuals, then
    /// sums the corrections with fixed-size compensation. Internal f32
    /// rounding remains: this is not an IEEE single-rounding guarantee for
    /// the exact pair-valued expression. A leading product overflow is
    /// terminal even when the addend could have canceled it.
    #[inline]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        let p = self.hi * b.hi;
        let s = p + c.hi;
        // This one result check also detects every nonfinite input. An
        // infinite addend needs scalar FMA semantics: a finite-input product
        // that overflowed to opposite infinity must not override the addend.
        if !s.is_finite() {
            return if !c.is_finite() {
                Self::from_f32(libm::fmaf(self.hi, b.hi, c.hi))
            } else {
                Self::from_f32(s)
            };
        }
        let (_, t) = two_sum(p, c.hi);
        let e = libm::fmaf(self.hi, b.hi, -p);
        let (x, xe) = two_prod(self.hi, b.lo);
        let (y, ye) = two_prod(self.lo, b.hi);
        let (z, ze) = two_prod(self.lo, b.lo);
        // Leading cancellation happens before any correction is discarded.
        // Eight fixed updates: no magnitude sorting or data-dependent length.
        let r = Correction { hi: e, lo: 0.0 }
            .add(t)
            .add(c.lo)
            .add(x)
            .add(xe)
            .add(y)
            .add(ye)
            .add(z)
            .add(ze);
        let (hi, lo) = two_sum(s, r.hi);
        // A residual carry can overflow even when the leading sum was finite.
        if !hi.is_finite() {
            return Self::from_f32(hi);
        }
        let result = Self::from_parts(hi, lo + r.lo);
        if result.is_zero() && c.is_zero() {
            // A nonzero product can underflow to signed zero.
            Self::from_f32(f32::from_bits(
                libm::fmaf(self.hi, b.hi, c.hi).to_bits() & (1 << 31),
            ))
        } else {
            result
        }
    }
}

impl Add for Df32 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let sum = self.hi + rhs.hi;
        // Preserve infinities/overflow before TwoSum computes inf-inf.
        if !sum.is_finite() {
            return Self::from_f32(sum);
        }
        if self.is_zero() && rhs.is_zero() {
            return Self::from_f32(sum);
        }
        let (_, e) = two_sum(self.hi, rhs.hi);
        let (t, f) = two_sum(self.lo, rhs.lo);
        let (hi, lo) = two_sum(sum, e + t);
        // Required for overflow caused by carrying the residual into `hi`.
        if !hi.is_finite() {
            return Self::from_f32(hi);
        }
        Self::from_parts(hi, lo + f)
    }
}

impl Sub for Df32 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Neg for Df32 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: if self.lo == 0.0 { 0.0 } else { -self.lo },
        }
    }
}

impl Mul for Df32 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let p = self.hi * rhs.hi;
        // Preserve infinite products; FMA product-error recovery needs finite p.
        if !p.is_finite() {
            return Self::from_f32(p);
        }
        let e = libm::fmaf(self.hi, rhs.hi, -p);
        let e = libm::fmaf(self.hi, rhs.lo, e);
        let e = libm::fmaf(self.lo, rhs.hi, e);
        let e = libm::fmaf(self.lo, rhs.lo, e);
        let result = Self::from_parts(p, e);
        if result.is_zero() {
            Self::from_f32(f32::from_bits(p.to_bits() & (1 << 31)))
        } else {
            result
        }
    }
}

impl Div for Df32 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        let q0 = self.hi / rhs.hi;
        // An infinite quotient cannot be refined; division by infinity must
        // return scalar signed zero rather than evaluate infinity * zero.
        // Finite zero numerators work through the ordinary correction path.
        if !q0.is_finite() || !rhs.is_finite() {
            return Self::from_f32(q0);
        }
        // Fixed one-step correction. The separate-operation remainder is
        // retained until a fused remainder demonstrates a cost/accuracy win.
        let r = self - rhs * Self::from_f32(q0);
        let q1 = (r.hi + r.lo) / rhs.hi;
        let result = Self::from_parts(q0, q1);
        if result.is_zero() {
            Self::from_f32(f32::from_bits(q0.to_bits() & (1 << 31)))
        } else {
            result
        }
    }
}

impl PartialEq for Df32 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.hi == rhs.hi && self.lo == rhs.lo
    }
}

impl PartialOrd for Df32 {
    #[inline]
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        match self.hi.partial_cmp(&rhs.hi)? {
            Ordering::Equal => self.lo.partial_cmp(&rhs.lo),
            other => Some(other),
        }
    }
}

macro_rules! assign {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for Df32 {
            #[inline]
            fn $method(&mut self, rhs: Self) { *self = *self $op rhs; }
        }
    };
}
assign!(AddAssign, add_assign, +);
assign!(SubAssign, sub_assign, -);
assign!(MulAssign, mul_assign, *);
assign!(DivAssign, div_assign, /);

impl From<f32> for Df32 {
    #[inline]
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}
impl From<f64> for Df32 {
    #[inline]
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}
impl From<Df32> for f32 {
    #[inline]
    fn from(value: Df32) -> Self {
        value.to_f32()
    }
}
impl From<Df32> for f64 {
    #[inline]
    fn from(value: Df32) -> Self {
        value.to_f64()
    }
}

impl num_traits::Zero for Df32 {
    fn zero() -> Self {
        Self::ZERO
    }
    fn is_zero(&self) -> bool {
        (*self).is_zero()
    }
}
impl num_traits::One for Df32 {
    fn one() -> Self {
        Self::ONE
    }
}
impl num_traits::MulAdd for Df32 {
    type Output = Self;
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::mul_add(self, a, b)
    }
}
impl num_traits::MulAddAssign for Df32 {
    fn mul_add_assign(&mut self, a: Self, b: Self) {
        *self = self.mul_add(a, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn nonfinite_or_zero_boundaries() {
        for magnitude in [
            0,
            1,
            0x007f_ffff,
            0x0080_0000,
            0x7f7f_fffe,
            0x7f7f_ffff,
            0x7f80_0000,
            0x7f80_0001,
            0x7fc0_0000,
            0x7fff_ffff,
        ] {
            for sign in [0, 1 << 31] {
                let x = f32::from_bits(sign | magnitude);
                assert_eq!(is_nonfinite_or_zero(x), !x.is_finite() || x == 0.0);
            }
        }
    }

    proptest! {
        #[test]
        fn transforms_reconstruct(a in -1e10f32..1e10, b in -1e10f32..1e10) {
            let (s, e) = two_sum(a, b);
            prop_assert_eq!(f64::from(s) + f64::from(e), f64::from(a) + f64::from(b));
            let (p, e) = two_prod(a, b);
            prop_assert_eq!(f64::from(p) + f64::from(e), f64::from(a) * f64::from(b));
        }
    }

    #[test]
    fn underflow_is_not_an_exact_product_transform() {
        let tiny = f32::from_bits(1);
        assert_eq!(two_prod(tiny, 0.5), (0.0, 0.0));
        assert!(f64::from(tiny) * 0.5 > 0.0);
    }
}
