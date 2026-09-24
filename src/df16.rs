use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// One number represented by a normalized, unevaluated sum of two `f16`s.
///
/// Ordinary values have roughly 22 significant bits, with the exponent range
/// of `f16`. Operations are not generally correctly rounded to the nearest
/// pair. Overflowing leading intermediates are terminal; residual underflow
/// reduces precision. Requires nearest-even arithmetic without reassociation
/// or flushing of required subnormal intermediates.
///
/// ```
/// #![feature(f16)]
/// use num_synth::Df16;
/// let origin = Df16::from_f16(10.0);
/// let offset = Df16::from_f16(1e-6);
/// assert_eq!((origin + offset) - origin, offset);
/// ```
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, Default)]
pub struct Df16([f16; 2]);

// Zero wraps to u16::MAX after subtracting one. Nonzero finite magnitudes
// remain below the threshold; infinities and NaNs are at or above it.
#[inline]
const fn is_nonfinite_or_zero(value: f16) -> bool {
    let magnitude = value.to_bits() & 0x7fff;
    magnitude.wrapping_sub(1) >= 0x7bff
}

// No magnitude precondition. Keep each operation individually rounded.
#[inline]
const fn two_sum(a: f16, b: f16) -> (f16, f16) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

// Requires |a| >= |b|. Each call site must establish this, including at ties.
#[inline]
const fn fast_two_sum(a: f16, b: f16) -> (f16, f16) {
    let s = a + b;
    (s, b - (s - a))
}

#[inline]
fn two_prod(a: f16, b: f16) -> (f16, f16) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

// Fixed two-component compensated sum, not a normalized public value.
// Each TwoSum is exact in its range; only accumulation of its errors rounds.
#[derive(Clone, Copy)]
struct Correction {
    hi: f16,
    lo: f16,
}

impl Correction {
    #[inline]
    fn add(self, x: f16) -> Self {
        let (hi, error) = two_sum(self.hi, x);
        Self {
            hi,
            lo: self.lo + error,
        }
    }
}

impl Df16 {
    pub const ZERO: Self = Self::from_f16(0.0);
    pub const NEG_ZERO: Self = Self::from_f16(-0.0);
    pub const ONE: Self = Self::from_f16(1.0);
    pub const NAN: Self = Self([f16::NAN, 0.0]);
    pub const INFINITY: Self = Self::from_f16(f16::INFINITY);
    pub const NEG_INFINITY: Self = Self::from_f16(f16::NEG_INFINITY);

    /// Stores the scalar directly, with a zero residual.
    #[inline]
    pub const fn from_f16(value: f16) -> Self {
        Self([value, 0.0])
    }

    /// Splits a binary64 input at the boundary. Not an exact f64 round-trip.
    /// Values whose leading conversion overflows become signed infinity.
    #[inline]
    pub const fn from_f64(value: f64) -> Self {
        let hi = value as f16;
        let lo = (value - hi as f64) as f16;
        Self::from_split(hi, lo)
    }

    /// Splits a binary32 input without discarding its residual first.
    /// Not an exact f32 round-trip; leading overflow becomes infinity.
    #[inline]
    pub const fn from_f32(value: f32) -> Self {
        let hi = value as f16;
        let lo = (value - hi as f32) as f16;
        Self::from_split(hi, lo)
    }

    // Shared by the two scalar splits, whose residual magnitude is bounded.
    #[inline]
    const fn from_split(hi: f16, lo: f16) -> Self {
        // For finite nonzero hi, rounding the residual can reach but not
        // exceed half the wider adjacent spacing of hi. Thus |lo| <= |hi|,
        // including subnormal hi (whose residual rounds to zero). FastTwoSum
        // still normalizes the tie cases introduced by residual rounding.
        let (sum, error) = fast_two_sum(hi, lo);
        // Select whole bit patterns so the bulk loop can vectorize. Evaluating
        // unused nonfinite residuals is harmless: FP exception flags/traps are
        // not part of the contract. Preserve hi for zero/NaN/infinity and clear
        // the residual on overflow or zero, without data-dependent branches.
        let split = !is_nonfinite_or_zero(hi);
        let high_mask = 0u16.wrapping_sub(split as u16);
        let low_mask = 0u16.wrapping_sub((split & sum.is_finite() & (error != 0.0)) as u16);
        Self([
            f16::from_bits((sum.to_bits() & high_mask) | (hi.to_bits() & !high_mask)),
            f16::from_bits(error.to_bits() & low_mask),
        ])
    }

    /// Normalizes the sum of arbitrary components, without a magnitude
    /// precondition. Opposite infinities and NaNs yield NaN; its bits are unspecified.
    #[inline]
    pub const fn from_parts(hi: f16, lo: f16) -> Self {
        let (sum, error) = two_sum(hi, lo);
        // Discard error recovery's inf-inf/NaN residual while preserving the
        // scalar sum. Masking also canonicalizes a negative-zero residual and
        // lets callers operating on slices keep the finite path vectorizable.
        let mask = 0u16.wrapping_sub((sum.is_finite() & (error != 0.0)) as u16);
        Self([sum, f16::from_bits(error.to_bits() & mask)])
    }

    // Preserve a nonfinite leading result instead of the NaN produced by its
    // error recovery. Integer selection keeps independent lanes vectorizable.
    #[inline]
    const fn with_finite_leading(self, leading: f16) -> Self {
        let mask = 0u16.wrapping_sub(leading.is_finite() as u16);
        Self([
            f16::from_bits((self.0[0].to_bits() & mask) | (leading.to_bits() & !mask)),
            f16::from_bits(self.0[1].to_bits() & mask),
        ])
    }

    /// Returns the normalized components without collapsing the residual.
    #[inline]
    pub const fn to_parts(self) -> (f16, f16) {
        (self.0[0], self.0[1])
    }

    /// Rounds the stored value to f16, discarding the residual.
    #[inline]
    pub const fn to_f16(self) -> f16 {
        self.0[0]
    }

    /// Rounds the stored sum to f32; a large exponent gap can lose bits.
    #[inline]
    pub const fn to_f32(self) -> f32 {
        if self.0[0] == 0.0 {
            self.0[0] as f32
        } else {
            self.0[0] as f32 + self.0[1] as f32
        }
    }

    /// Returns the exact finite stored sum in f64, preserving signed zero.
    #[inline]
    pub const fn to_f64(self) -> f64 {
        if self.0[0] == 0.0 {
            self.0[0] as f64 // Preserve negative zero.
        } else {
            self.0[0] as f64 + self.0[1] as f64
        }
    }

    #[inline]
    pub const fn is_finite(self) -> bool {
        self.0[0].is_finite()
    }

    #[inline]
    pub const fn is_nan(self) -> bool {
        self.0[0].is_nan()
    }

    #[inline]
    pub const fn is_infinite(self) -> bool {
        self.0[0].is_infinite()
    }

    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0[0] == 0.0
    }

    #[inline]
    pub const fn is_sign_negative(self) -> bool {
        self.0[0].is_sign_negative()
    }

    /// Computes `self * b + c` without first rounding the product to Df16.
    ///
    /// Retains all four component products and their fused residuals, then
    /// sums the corrections with fixed-size compensation. Internal f16
    /// rounding remains: this is not an IEEE single-rounding guarantee for
    /// the exact pair-valued expression. A leading product overflow is
    /// terminal even when the addend could have canceled it.
    #[inline]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        let p = self.0[0] * b.0[0];
        let s = p + c.0[0];
        // This one result check also detects every nonfinite input. An
        // infinite addend needs scalar FMA semantics: a finite-input product
        // that overflowed to opposite infinity must not override the addend.
        if !s.is_finite() {
            return if !c.is_finite() {
                Self::from_f16(self.0[0].mul_add(b.0[0], c.0[0]))
            } else {
                Self::from_f16(s)
            };
        }
        let (_, t) = two_sum(p, c.0[0]);
        let e = self.0[0].mul_add(b.0[0], -p);
        let (x, xe) = two_prod(self.0[0], b.0[1]);
        let (y, ye) = two_prod(self.0[1], b.0[0]);
        let (z, ze) = two_prod(self.0[1], b.0[1]);
        // Leading cancellation happens before any correction is discarded.
        // Eight fixed updates: no magnitude sorting or data-dependent length.
        let r = Correction { hi: e, lo: 0.0 }
            .add(t)
            .add(c.0[1])
            .add(x)
            .add(xe)
            .add(y)
            .add(ye)
            .add(z)
            .add(ze);
        let (hi, lo) = two_sum(s, r.hi);
        // A residual carry can overflow even when the leading sum was finite.
        if !hi.is_finite() {
            return Self::from_f16(hi);
        }
        let result = Self::from_parts(hi, lo + r.lo);
        if result.is_zero() && c.is_zero() {
            // A nonzero product can underflow to signed zero.
            Self::from_f16(f16::from_bits(
                self.0[0].mul_add(b.0[0], c.0[0]).to_bits() & (1 << 15),
            ))
        } else {
            result
        }
    }
}

impl Add for Df16 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let (sum, e) = two_sum(self.0[0], rhs.0[0]);
        let (t, f) = two_sum(self.0[1], rhs.0[1]);
        let (hi, lo) = two_sum(sum, e + t);
        let result = Self::from_parts(hi, lo + f)
            .with_finite_leading(hi)
            .with_finite_leading(sum);
        // Only -0 + -0 gives a negative-zero leading sum, and both residuals
        // must then be zero. Preserve that sign without a zero-operand branch.
        let negative_zero = ((sum.to_bits() == 0x8000) as u16) << 15;
        Self([
            f16::from_bits(result.0[0].to_bits() | negative_zero),
            result.0[1],
        ])
    }
}

impl Sub for Df16 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Neg for Df16 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], if self.0[1] == 0.0 { 0.0 } else { -self.0[1] }])
    }
}

impl Mul for Df16 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let p = self.0[0] * rhs.0[0];
        // Preserve infinite products; FMA product-error recovery needs finite p.
        if !p.is_finite() {
            return Self::from_f16(p);
        }
        let e = self.0[0].mul_add(rhs.0[0], -p);
        let e = self.0[0].mul_add(rhs.0[1], e);
        let e = self.0[1].mul_add(rhs.0[0], e);
        let e = self.0[1].mul_add(rhs.0[1], e);
        let result = Self::from_parts(p, e);
        if result.is_zero() {
            Self::from_f16(f16::from_bits(p.to_bits() & (1 << 15)))
        } else {
            result
        }
    }
}

impl Div for Df16 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        let q0 = self.0[0] / rhs.0[0];
        // An infinite quotient cannot be refined; division by infinity must
        // return scalar signed zero rather than evaluate infinity * zero.
        // Finite zero numerators work through the ordinary correction path.
        if !q0.is_finite() || !rhs.is_finite() {
            return Self::from_f16(q0);
        }
        // Correct the leading quotient directly. The first fused subtraction
        // keeps the small remainder without materializing a rounded product;
        // the second includes the denominator's low component. No pair-valued
        // multiplication/subtraction (or pair FMA) is needed for this remainder.
        let r = (-q0).mul_add(rhs.0[0], self.0[0]);
        let r = (-q0).mul_add(rhs.0[1], r + self.0[1]);
        let q1 = r / rhs.0[0];
        let result = Self::from_parts(q0, q1);
        if result.is_zero() {
            Self::from_f16(f16::from_bits(q0.to_bits() & (1 << 15)))
        } else {
            result
        }
    }
}

impl PartialEq for Df16 {
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.0[0] == rhs.0[0] && self.0[1] == rhs.0[1]
    }
}

impl PartialOrd for Df16 {
    #[inline]
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        match self.0[0].partial_cmp(&rhs.0[0])? {
            Ordering::Equal => self.0[1].partial_cmp(&rhs.0[1]),
            other => Some(other),
        }
    }
}

macro_rules! assign {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for Df16 {
            #[inline]
            fn $method(&mut self, rhs: Self) { *self = *self $op rhs; }
        }
    };
}
assign!(AddAssign, add_assign, +);
assign!(SubAssign, sub_assign, -);
assign!(MulAssign, mul_assign, *);
assign!(DivAssign, div_assign, /);

impl From<f16> for Df16 {
    #[inline]
    fn from(value: f16) -> Self {
        Self::from_f16(value)
    }
}
impl From<f32> for Df16 {
    #[inline]
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}
impl From<Df16> for f32 {
    #[inline]
    fn from(value: Df16) -> Self {
        value.to_f32()
    }
}
impl From<f64> for Df16 {
    #[inline]
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}
impl From<Df16> for f16 {
    #[inline]
    fn from(value: Df16) -> Self {
        value.to_f16()
    }
}
impl From<Df16> for f64 {
    #[inline]
    fn from(value: Df16) -> Self {
        value.to_f64()
    }
}

impl num_traits::Zero for Df16 {
    fn zero() -> Self {
        Self::ZERO
    }
    fn is_zero(&self) -> bool {
        (*self).is_zero()
    }
}
impl num_traits::One for Df16 {
    fn one() -> Self {
        Self::ONE
    }
}
impl num_traits::MulAdd for Df16 {
    type Output = Self;
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::mul_add(self, a, b)
    }
}
impl num_traits::MulAddAssign for Df16 {
    fn mul_add_assign(&mut self, a: Self, b: Self) {
        *self = self.mul_add(a, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn exhaustive_nonfinite_or_zero() {
        for bits in 0..=u16::MAX {
            let x = f16::from_bits(bits);
            assert_eq!(is_nonfinite_or_zero(x), !x.is_finite() || x == 0.0);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4096))]

        #[test]
        fn transforms_reconstruct(a in any::<u16>(), b in any::<u16>()) {
            let (a,b) = (f16::from_bits(a), f16::from_bits(b));
            if a.is_finite() && b.is_finite() {
                let (s,e) = two_sum(a,b);
                if s.is_finite() { prop_assert_eq!(s as f64 + e as f64, a as f64 + b as f64); }
                let (p,e) = two_prod(a,b);
                if p.is_finite() {
                    // f64 represents the product of two f16 scalars exactly.
                    let exact = a as f64 * b as f64;
                    let residual = exact - p as f64;
                    if (residual as f16) as f64 == residual {
                        prop_assert_eq!(p as f64 + e as f64, exact);
                    } else {
                        prop_assert!((p as f64 + e as f64 - exact).abs() <= 2f64.powi(-25));
                    }
                }
            }
        }
    }
}
