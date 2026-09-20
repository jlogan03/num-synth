#![allow(dead_code)]
use num_bigint::{BigInt, BigUint, Sign};
use num_synth::S64U8;
pub const INF: u64 = 0x7ff0_0000_0000_0000;

pub fn finite(negative: bool, exponent: i16, digits: [u8; 32]) -> S64U8 {
    S64U8::from_finite_parts(negative, exponent, digits).unwrap()
}
pub fn power(exponent: i16) -> S64U8 {
    let mut digits = [0; 32];
    digits[31] = 128;
    finite(false, exponent, digits)
}
pub fn dyadic(value: S64U8) -> (BigInt, i32) {
    let (negative, exponent, digits) = value.finite_parts().unwrap();
    (
        BigInt::from_bytes_le(if negative { Sign::Minus } else { Sign::Plus }, &digits),
        i32::from(exponent) - 255,
    )
}
pub fn assert_same(a: S64U8, b: S64U8) {
    assert_eq!(a.class(), b.class());
    assert_eq!(a.is_sign_negative(), b.is_sign_negative());
    assert_eq!(a.finite_parts(), b.finite_parts());
}
pub fn sum((a, ap): (BigInt, i32), (b, bp): (BigInt, i32)) -> (BigInt, i32) {
    let power = ap.min(bp);
    (
        (a << (ap - power) as usize) + (b << (bp - power) as usize),
        power,
    )
}

pub fn product((a, ap): (BigInt, i32), (b, bp): (BigInt, i32)) -> (BigInt, i32) {
    (a * b, ap + bp)
}

// Independent 256-bit oracle: arbitrary-precision quotient/remainder and an
// exact distance comparison, rather than the implementation's limb guard bits.
pub fn synthetic_reference(n: &BigInt, power: i32) -> S64U8 {
    let negative = n.sign() == Sign::Minus;
    let m = n.magnitude();
    if m == &BigUint::from(0_u8) {
        return S64U8::from_f64(0.0);
    }
    let mut exponent = power + m.bits() as i32 - 1;
    if exponent < i32::from(i16::MIN) {
        let unit = power.min(i32::from(i16::MIN) - 1);
        let exact = m << (power - unit) as usize;
        let half_min = BigUint::from(1_u8) << (i32::from(i16::MIN) - 1 - unit) as usize;
        return if exact > half_min {
            finite(negative, i16::MIN, {
                let mut d = [0; 32];
                d[31] = 128;
                d
            })
        } else {
            S64U8::from_bits(u64::from(negative) << 63)
        };
    }
    if exponent > i32::from(i16::MAX) {
        return S64U8::from_bits((u64::from(negative) << 63) | INF);
    }
    let grid = exponent - 255;
    let numerator = m << (power - grid).max(0) as usize;
    let denominator = BigUint::from(1_u8) << (grid - power).max(0) as usize;
    let mut q = &numerator / &denominator;
    let remainder = numerator % &denominator;
    let twice = &remainder + &remainder;
    if twice > denominator || (twice == denominator && q.bit(0)) {
        q += 1_u8;
    }
    if q == (BigUint::from(1_u8) << 256) {
        q /= 2_u8;
        exponent += 1;
    }
    if exponent > i32::from(i16::MAX) {
        S64U8::from_bits((u64::from(negative) << 63) | INF)
    } else {
        let mut digits = [0; 32];
        let bytes = q.to_bytes_le();
        digits[..bytes.len()].copy_from_slice(&bytes);
        finite(negative, exponent as i16, digits)
    }
}

// IEEE conversion oracle: search adjacent encodings on an exact integer lattice.
pub fn format_reference(n: &BigInt, power: i32, fraction_bits: u32, bias: i32) -> u64 {
    let infinity = ((2 * bias + 1) as u64) << fraction_bits;
    let sign_bit = if fraction_bits == 52 { 63 } else { 31 };
    let min_power = 1 - bias - fraction_bits as i32;
    let sign = u64::from(n.sign() == Sign::Minus) << sign_bit;
    let unit = power.min(min_power);
    let exact = n.magnitude() << (power - unit) as usize;
    let point = |bits: u64| {
        if bits == infinity {
            BigUint::from(1_u8) << (bias + 1 - unit) as usize
        } else {
            let field = (bits >> fraction_bits) as i32;
            let fraction = bits & ((1 << fraction_bits) - 1);
            let (m, p) = if field == 0 {
                (fraction, min_power)
            } else {
                (
                    fraction | (1 << fraction_bits),
                    field - bias - fraction_bits as i32,
                )
            };
            BigUint::from(m) << (p - unit) as usize
        }
    };
    if exact >= point(infinity) {
        return sign | infinity;
    }
    let (mut lo, mut hi) = (0, infinity);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if point(mid) <= exact {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let down = &exact - point(lo);
    let up = point(hi) - exact;
    sign | if down < up || (down == up && lo & 1 == 0) {
        lo
    } else {
        hi
    }
}
