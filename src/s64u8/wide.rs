use super::{Class, S64U8};
use core::cmp::Ordering;

// Full 512-bit product plus 16 bits for alignment and carry. Fixed size only.
const N: usize = 66;
const TOP: i32 = 8 * N as i32 - 2;
pub(super) struct Wide([u8; N]);

impl Wide {
    pub(super) fn from_digits(digits: [u8; 32]) -> Self {
        let mut limbs = [0; N];
        limbs[..32].copy_from_slice(&digits);
        Self(limbs)
    }

    pub(super) fn product(a: &[u8; 32], b: &[u8; 32]) -> Self {
        let mut coefficients = [0_u32; 63];
        for (i, &a) in a.iter().enumerate() {
            for (j, &b) in b.iter().enumerate() {
                coefficients[i + j] += u32::from(a) * u32::from(b);
            }
        }
        let mut limbs = [0; N];
        let mut carry = 0;
        for (limb, coefficient) in limbs.iter_mut().zip(coefficients) {
            let value = coefficient + carry;
            *limb = value as u8;
            carry = value >> 8;
        }
        debug_assert!(carry < 256);
        limbs[63] = carry as u8;
        Self(limbs)
    }

    fn bits(&self) -> i32 {
        let mut bits = 0;
        for (i, &limb) in self.0.iter().enumerate() {
            if limb != 0 {
                bits = 8 * i as i32 + (8 - limb.leading_zeros()) as i32;
            }
        }
        bits
    }
    fn limb(&self, index: i32) -> u8 {
        self.0.get(index as usize).copied().unwrap_or(0)
    }
    fn byte_at(&self, bit: i32) -> u8 {
        let index = bit.div_euclid(8);
        let shift = bit.rem_euclid(8);
        ((u16::from(self.limb(index)) >> shift) | (u16::from(self.limb(index + 1)) << (8 - shift)))
            as u8
    }
    fn bit(&self, bit: i32) -> bool {
        self.byte_at(bit) & 1 != 0
    }
    fn any_below(&self, cutoff: i32) -> bool {
        let mut discarded = 0;
        for (i, &limb) in self.0.iter().enumerate() {
            let count = (cutoff - 8 * i as i32).clamp(0, 8);
            discarded |= u16::from(limb) & ((1_u16 << count) - 1);
        }
        discarded != 0
    }
    fn align(self, shift: i32) -> Self {
        debug_assert!(shift < 0 || self.bits() + shift <= 8 * N as i32);
        let mut limbs = [0; N];
        for (i, limb) in limbs.iter_mut().enumerate() {
            *limb = self.byte_at(8 * i as i32 - shift);
        }
        if shift < 0 {
            limbs[0] |= u8::from(self.any_below(-shift));
        }
        Self(limbs)
    }
    fn cmp(&self, other: &Self) -> Ordering {
        let mut order = Ordering::Equal;
        for (&a, &b) in self.0.iter().zip(&other.0) {
            if a != b {
                order = a.cmp(&b);
            }
        }
        order
    }
    fn add(&mut self, other: &Self) {
        let mut carry = 0_u16;
        for (a, &b) in self.0.iter_mut().zip(&other.0) {
            let value = u16::from(*a) + u16::from(b) + carry;
            *a = value as u8;
            carry = value >> 8;
        }
        debug_assert_eq!(carry, 0);
    }
    fn subtract(&mut self, other: &Self) {
        let mut borrow = 0_i16;
        for (a, &b) in self.0.iter_mut().zip(&other.0) {
            let value = i16::from(*a) - i16::from(b) - borrow;
            *a = value as u8;
            borrow = i16::from(value < 0);
        }
        debug_assert_eq!(borrow, 0);
    }
    fn increment_needed(&self, shift: i32) -> bool {
        self.bit(shift - 1) && (self.any_below(shift - 1) || self.bit(shift))
    }
    pub(super) fn rounded_u64(&self, shift: i32) -> u64 {
        debug_assert!(self.bits() - shift <= 53);
        let mut result = 0_u64;
        for i in 0..8 {
            result |= u64::from(self.byte_at(shift + 8 * i)) << (8 * i);
        }
        result + u64::from(self.increment_needed(shift))
    }
    pub(super) fn finish(self, power: i32, negative: bool) -> S64U8 {
        let bits = self.bits();
        if bits == 0 {
            return S64U8::special(Class::Zero, negative);
        }
        let mut exponent = power + bits - 1;
        let mut digits = [0; 32];
        if exponent < i32::from(i16::MIN) {
            if exponent == i32::from(i16::MIN) - 1 && self.any_below(bits - 1) {
                digits[31] = 128;
                return S64U8 {
                    exponent: i16::MIN,
                    digits,
                    class: Class::Finite,
                    negative,
                };
            }
            return S64U8::special(Class::Zero, negative);
        }
        let shift = bits - 256;
        for (i, digit) in digits.iter_mut().enumerate() {
            *digit = self.byte_at(shift + 8 * i as i32);
        }
        let mut carry = u16::from(self.increment_needed(shift));
        for digit in &mut digits {
            let value = u16::from(*digit) + carry;
            *digit = value as u8;
            carry = value >> 8;
        }
        if carry != 0 {
            digits[31] = 128;
            exponent += 1;
        }
        if exponent > i32::from(i16::MAX) {
            return S64U8::special(Class::Infinite, negative);
        }
        debug_assert!(digits[31] >= 128);
        S64U8 {
            exponent: exponent as i16,
            digits,
            class: Class::Finite,
            negative,
        }
    }
}

// Inputs here are exact canonical significands or a fresh exact product. Any
// jammed alignment is consumed by this operation and rounded before returning.
pub(super) fn combine(mut a: Wide, ap: i32, mut an: bool, mut b: Wide, bp: i32, bn: bool) -> S64U8 {
    let ab = a.bits();
    let bb = b.bits();
    if ab == 0 {
        return b.finish(bp, if bb == 0 { an && bn } else { bn });
    }
    if bb == 0 {
        return a.finish(ap, an);
    }
    let power = (ap + ab - 1).max(bp + bb - 1) - TOP;
    a = a.align(ap - power);
    b = b.align(bp - power);
    if an == bn {
        a.add(&b);
    } else {
        match a.cmp(&b) {
            Ordering::Less => {
                b.subtract(&a);
                a = b;
                an = bn;
            }
            Ordering::Equal => return S64U8::special(Class::Zero, false),
            Ordering::Greater => a.subtract(&b),
        }
    }
    a.finish(power, an)
}
