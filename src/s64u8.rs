use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

mod wide;
use wide::Wide;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Class {
    Zero,
    Finite,
    Infinite,
    Nan,
}

/// The top bit of a finite nonzero significand must be set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidFiniteParts;

impl fmt::Display for InvalidFiniteParts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the most significant digit must be at least 128")
    }
}
impl core::error::Error for InvalidFiniteParts {}

/// Fixed-width synthetic floating point, using 32 unsigned radix-256 digits.
///
/// Digits are little endian and encode `2^255 <= S < 2^256`. The finite value
/// is `(-1)^sign * S * 2^(exponent - 255)`, with every `i16` exponent supported.
/// Arithmetic rounds to nearest-even at this working precision. Conversion to
/// f64 or f32 separately rounds the represented value to the destination format.
///
/// No native floating-point arithmetic, heap allocation, or variable-length
/// storage is used. Products and FMA use fixed 66-byte scratch integers.
/// Binary64 bit conversions preserve all NaN encodings; arithmetic quiets the
/// first NaN while preserving its sign and payload. Invalid operations return
/// a positive quiet NaN. There are no exception flags or internal subnormals.
/// See `design/S64U8.md` for the bounds and rounding contract.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct S64U8 {
    exponent: i16,
    digits: [u8; 32],
    class: Class,
    negative: bool,
}

impl S64U8 {
    pub const PRECISION: u32 = 256;
    pub const DIGIT_COUNT: usize = 32;
    pub const RADIX: u16 = 256;

    pub fn from_finite_parts(
        negative: bool,
        exponent: i16,
        digits: [u8; 32],
    ) -> Result<Self, InvalidFiniteParts> {
        if digits[31] < 128 {
            return Err(InvalidFiniteParts);
        }
        Ok(Self {
            exponent,
            digits,
            class: Class::Finite,
            negative,
        })
    }

    /// Return `(negative, exponent, little_endian_digits)` for finite nonzero values.
    pub fn finite_parts(self) -> Option<(bool, i16, [u8; 32])> {
        (self.class == Class::Finite).then_some((self.negative, self.exponent, self.digits))
    }

    pub fn class(self) -> Class {
        self.class
    }
    pub fn is_sign_negative(self) -> bool {
        self.negative
    }

    /// Expand binary64 exactly, preserving every encoding, including signaling NaNs.
    pub fn from_f64(value: f64) -> Self {
        Self::from_bits(value.to_bits())
    }

    /// Expand a binary64 encoding without floating-point arithmetic.
    pub fn from_bits(bits: u64) -> Self {
        Self::decode(bits, 52, 11, 1023)
    }

    /// Expand binary32 exactly. NaN fraction bits are aligned with binary64's
    /// fraction so binary32 NaNs round-trip with sign and signaling bit intact.
    pub fn from_f32(value: f32) -> Self {
        Self::from_f32_bits(value.to_bits())
    }
    pub fn from_f32_bits(bits: u32) -> Self {
        Self::decode(u64::from(bits), 23, 8, 127)
    }

    fn decode(bits: u64, fraction_bits: u32, exponent_bits: u32, bias: i32) -> Self {
        let negative = bits >> (fraction_bits + exponent_bits) != 0;
        let fraction = bits & ((1_u64 << fraction_bits) - 1);
        let field_mask = (1_u64 << exponent_bits) - 1;
        let field = (bits >> fraction_bits) & field_mask;
        if field == field_mask {
            if fraction == 0 {
                return Self::special(Class::Infinite, negative);
            }
            let payload = fraction << (52 - fraction_bits);
            let mut result = Self::special(Class::Nan, negative);
            result.digits[..8].copy_from_slice(&payload.to_le_bytes());
            return result;
        }
        if field == 0 && fraction == 0 {
            return Self::special(Class::Zero, negative);
        }
        let (mantissa, power) = if field == 0 {
            (fraction, 1 - bias - fraction_bits as i32)
        } else {
            (
                fraction | (1_u64 << fraction_bits),
                field as i32 - bias - fraction_bits as i32,
            )
        };
        let leading = 63 - mantissa.leading_zeros() as i32;
        let mut digits = [0; 32];
        // Integer bit extraction keeps expansion independent of host FP support.
        for bit in 0..53 {
            let destination = 255 - leading + bit;
            if destination < 256 {
                digits[destination as usize / 8] |=
                    (((mantissa >> bit) & 1) as u8) << (destination % 8);
            }
        }
        Self {
            exponent: (power + leading) as i16,
            digits,
            class: Class::Finite,
            negative,
        }
    }

    /// Round the represented working value directly to binary64, nearest-even.
    pub fn to_f64(self) -> f64 {
        f64::from_bits(self.to_bits())
    }
    pub fn to_bits(self) -> u64 {
        self.encode(52, 11, 1023)
    }

    /// Round directly to binary32; no intermediate binary64 conversion is used.
    pub fn to_f32(self) -> f32 {
        f32::from_bits(self.to_f32_bits())
    }
    pub fn to_f32_bits(self) -> u32 {
        self.encode(23, 8, 127) as u32
    }

    fn encode(self, fraction_bits: u32, exponent_bits: u32, bias: i32) -> u64 {
        let sign = u64::from(self.negative) << (fraction_bits + exponent_bits);
        let infinity = ((1_u64 << exponent_bits) - 1) << fraction_bits;
        let magnitude = match self.class {
            Class::Zero => 0,
            Class::Infinite => infinity,
            Class::Nan => {
                let payload = u64::from_le_bytes(self.digits[..8].try_into().unwrap());
                // Narrow the payload while retaining NaN classification even
                // when only discarded payload bits were set.
                infinity | (payload >> (52 - fraction_bits)).max(1)
            }
            Class::Finite => {
                let wide = Wide::from_digits(self.digits);
                let mut exponent = i32::from(self.exponent);
                if exponent > bias {
                    infinity
                } else if exponent < 1 - bias {
                    wide.rounded_u64(255 - exponent + 1 - bias - fraction_bits as i32)
                } else {
                    let mut significand = wide.rounded_u64(255 - fraction_bits as i32);
                    if significand == 1_u64 << (fraction_bits + 1) {
                        significand >>= 1;
                        exponent += 1;
                    }
                    if exponent > bias {
                        infinity
                    } else {
                        ((exponent + bias) as u64) << fraction_bits
                            | (significand & ((1_u64 << fraction_bits) - 1))
                    }
                }
            }
        };
        sign | magnitude
    }

    fn special(class: Class, negative: bool) -> Self {
        Self {
            exponent: 0,
            digits: [0; 32],
            class,
            negative,
        }
    }
    fn invalid() -> Self {
        Self::from_bits(0x7ff8_0000_0000_0000)
    }
    fn first_nan(values: &[Self]) -> Option<Self> {
        values
            .iter()
            .copied()
            .find(|x| x.class == Class::Nan)
            .map(|mut x| {
                x.digits[6] |= 8; // binary64 quiet bit 51
                x
            })
    }

    /// Fused `self * b + c`, rounded once to the 256-bit working format.
    /// Product overflow/underflow is assessed only after incorporating `c`.
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        if let Some(nan) = Self::first_nan(&[self, b, c]) {
            return nan;
        }
        if (self.class == Class::Infinite && b.class == Class::Zero)
            || (b.class == Class::Infinite && self.class == Class::Zero)
        {
            return Self::invalid();
        }
        if self.class == Class::Infinite || b.class == Class::Infinite {
            return Self::special(Class::Infinite, self.negative ^ b.negative) + c;
        }
        if c.class == Class::Infinite {
            return c;
        }
        let product = Wide::product(&self.digits, &b.digits);
        wide::combine(
            product,
            i32::from(self.exponent) + i32::from(b.exponent) - 510,
            self.negative ^ b.negative,
            Wide::from_digits(c.digits),
            i32::from(c.exponent) - 255,
            c.negative,
        )
    }
}

impl From<f64> for S64U8 {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}
impl From<f32> for S64U8 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}
impl From<S64U8> for f64 {
    fn from(value: S64U8) -> Self {
        value.to_f64()
    }
}
impl From<S64U8> for f32 {
    fn from(value: S64U8) -> Self {
        value.to_f32()
    }
}

impl Add for S64U8 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        if let Some(nan) = Self::first_nan(&[self, rhs]) {
            return nan;
        }
        if self.class == Class::Infinite || rhs.class == Class::Infinite {
            if self.class == rhs.class && self.negative != rhs.negative {
                return Self::invalid();
            }
            return if self.class == Class::Infinite {
                self
            } else {
                rhs
            };
        }
        wide::combine(
            Wide::from_digits(self.digits),
            i32::from(self.exponent) - 255,
            self.negative,
            Wide::from_digits(rhs.digits),
            i32::from(rhs.exponent) - 255,
            rhs.negative,
        )
    }
}
impl Sub for S64U8 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::first_nan(&[self, rhs]).unwrap_or_else(|| self + -rhs)
    }
}
impl Mul for S64U8 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        if let Some(nan) = Self::first_nan(&[self, rhs]) {
            return nan;
        }
        let negative = self.negative ^ rhs.negative;
        if self.class == Class::Infinite || rhs.class == Class::Infinite {
            if self.class == Class::Zero || rhs.class == Class::Zero {
                return Self::invalid();
            }
            return Self::special(Class::Infinite, negative);
        }
        Wide::product(&self.digits, &rhs.digits).finish(
            i32::from(self.exponent) + i32::from(rhs.exponent) - 510,
            negative,
        )
    }
}
impl Neg for S64U8 {
    type Output = Self;
    fn neg(mut self) -> Self {
        self.negative = !self.negative;
        self
    }
}
impl PartialEq for S64U8 {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}
impl PartialOrd for S64U8 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.class == Class::Nan || other.class == Class::Nan {
            return None;
        }
        if self.class == Class::Zero && other.class == Class::Zero {
            return Some(Ordering::Equal);
        }
        if self.negative != other.negative {
            return Some(other.negative.cmp(&self.negative));
        }
        let order = (self.class as u8)
            .cmp(&(other.class as u8))
            .then_with(|| self.exponent.cmp(&other.exponent))
            .then_with(|| self.digits.iter().rev().cmp(other.digits.iter().rev()));
        Some(if self.negative {
            order.reverse()
        } else {
            order
        })
    }
}
