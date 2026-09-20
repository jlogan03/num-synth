use num_bigint::{BigInt, Sign};
use num_synth::Df32;
use num_traits::{Signed, ToPrimitive, Zero};

// All f32 values lie on this exact integer lattice, measured in 2^-149.
pub fn scalar(x: f32) -> BigInt {
    assert!(x.is_finite());
    let bits = x.to_bits();
    let exponent = (bits >> 23) & 255;
    let fraction = bits & 0x7fffff;
    let n = if exponent == 0 {
        BigInt::from(fraction)
    } else {
        BigInt::from(fraction | 0x800000) << (exponent - 1) as usize
    };
    if x.is_sign_negative() { -n } else { n }
}

pub fn value(x: Df32) -> BigInt {
    let (hi, lo) = x.to_parts();
    scalar(hi) + scalar(lo)
}

// Independent integer nearest-even rounding, including subnormal/normal
// transitions and overflow. Input and output both use the 2^-149 lattice.
pub fn round32(n: &BigInt) -> f32 {
    let sign = if n.sign() == Sign::Minus { 1 << 31 } else { 0 };
    let a = n.magnitude();
    if a.bits() <= 23 {
        return f32::from_bits(sign | a.to_u32().unwrap());
    }
    let shift = a.bits().saturating_sub(24) as usize;
    let mut q = a >> shift;
    if shift > 0 {
        let remainder = a - (&q << shift);
        let half = num_bigint::BigUint::from(1u8) << (shift - 1);
        if remainder > half || (remainder == half && q.bit(0)) {
            q += 1u8;
        }
    }
    let mut exponent = shift + 1;
    if q.bits() > 24 {
        q >>= 1;
        exponent += 1;
    }
    if exponent >= 255 {
        f32::from_bits(sign | 0x7f800000)
    } else {
        f32::from_bits(sign | (exponent as u32) << 23 | (q.to_u32().unwrap() & 0x7fffff))
    }
}

pub fn normalized(x: Df32) {
    let (hi, lo) = x.to_parts();
    if !hi.is_finite() {
        assert_eq!(lo.to_bits(), 0);
        return;
    }
    assert!(lo.is_finite());
    if lo == 0.0 {
        assert_eq!(lo.to_bits(), 0);
    }
    let n = value(x);
    if n.is_zero() {
        assert_eq!(hi, 0.0);
    } else {
        assert_eq!(hi.to_bits(), round32(&n).to_bits(), "{x:?}");
    }
}

pub fn bounded_error(got: &BigInt, expected: &BigInt, scale: &BigInt, bits: usize) {
    let error = (got - expected).abs();
    assert!(
        (&error << bits) <= scale.abs(),
        "error={} scale={} bits={bits}",
        error,
        scale
    );
}
