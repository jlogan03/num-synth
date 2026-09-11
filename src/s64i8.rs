use core::fmt;

const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
const INFINITY: u64 = 0x7ff0_0000_0000_0000;
const MIN_SIGNIFICAND: u128 = 1_u128 << 67;
const MAX_SIGNIFICAND: u128 = 1_u128 << 68;

/// Classification of a canonical synthetic value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Class {
    Zero,
    Finite,
    Infinite,
    Nan,
}

/// A violation of the canonical finite representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvalidFiniteParts {
    /// Every digit must belong to `-64..=63`.
    DigitOutOfRange,
    /// The decoded integer significand must belong to `2^67..2^68`.
    NotNormalized,
}

impl fmt::Display for InvalidFiniteParts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DigitOutOfRange => "digits must be in -64..=63",
            Self::NotNormalized => "significand must be in 2^67..2^68",
        })
    }
}

impl core::error::Error for InvalidFiniteParts {}

/// A 16-byte synthetic binary float with 68 significant bits.
///
/// For finite nonzero values, the ten most-significant-first balanced radix-128
/// digits encode `S = sum(d[i] * 128^(9-i))`, with `2^67 <= S < 2^68`.
/// The value is `(-1)^sign * S * 2^(exponent - 67)`. Every `i16` exponent is
/// supported. Private fields enforce canonical values; this is not a wire format.
///
/// Conversions preserve both signed zeros and every binary64 NaN bit pattern,
/// including its signaling bit. No floating-point arithmetic or exception flags
/// are involved. Collapse rounds finite values to nearest, ties to even.
///
/// See `design/S64I8.md` for the representation and accumulator proofs.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct S64I8 {
    exponent: i16,
    digits: [i8; 10],
    class: Class,
    negative: bool,
    reserved: [u8; 2],
}

impl S64I8 {
    pub const PRECISION: u32 = 68;
    pub const DIGIT_COUNT: usize = 10;
    pub const RADIX: i32 = 128;

    /// Expand any binary64 value exactly, preserving its complete bit pattern.
    pub fn expand(value: f64) -> Self {
        Self::from_bits(value.to_bits())
    }

    /// Expand a binary64 encoding without interpreting it in floating hardware.
    pub fn from_bits(bits: u64) -> Self {
        let negative = bits >> 63 != 0;
        let field = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & FRACTION_MASK;
        let (class, exponent, significand) = match (field, fraction) {
            (0, 0) => (Class::Zero, 0, 0),
            (0x7ff, 0) => (Class::Infinite, 0, 0),
            (0x7ff, _) => (Class::Nan, 0, u128::from(fraction)),
            (0, _) => {
                let leading = 63 - fraction.leading_zeros();
                (
                    Class::Finite,
                    leading as i16 - 1074,
                    u128::from(fraction) << (67 - leading),
                )
            }
            _ => (
                Class::Finite,
                (field - 1023) as i16,
                u128::from((1_u64 << 52) | fraction) << 15,
            ),
        };
        Self {
            exponent,
            digits: encode_digits(significand),
            class,
            negative,
            reserved: [0; 2],
        }
    }

    /// Construct a finite nonzero value from canonical digits.
    ///
    /// Digits are most significant first. Invalid or redundant representations
    /// are rejected rather than silently normalized or rounded.
    pub fn from_finite_parts(
        negative: bool,
        exponent: i16,
        digits: [i8; 10],
    ) -> Result<Self, InvalidFiniteParts> {
        if digits.iter().any(|&d| !(-64..=63).contains(&d)) {
            return Err(InvalidFiniteParts::DigitOutOfRange);
        }
        let significand = decode_digits(&digits);
        if !(MIN_SIGNIFICAND as i128..MAX_SIGNIFICAND as i128).contains(&significand) {
            return Err(InvalidFiniteParts::NotNormalized);
        }
        Ok(Self {
            exponent,
            digits,
            class: Class::Finite,
            negative,
            reserved: [0; 2],
        })
    }

    /// Return `(negative, exponent, digits)` for finite nonzero values.
    pub fn finite_parts(self) -> Option<(bool, i16, [i8; 10])> {
        (self.class == Class::Finite).then_some((self.negative, self.exponent, self.digits))
    }

    pub fn class(self) -> Class {
        self.class
    }

    pub fn is_sign_negative(self) -> bool {
        self.negative
    }

    /// Round to binary64 using round-to-nearest, ties-to-even.
    pub fn collapse(self) -> f64 {
        f64::from_bits(self.to_bits())
    }

    /// Return the correctly rounded binary64 encoding.
    pub fn to_bits(self) -> u64 {
        let sign = u64::from(self.negative) << 63;
        let magnitude = match self.class {
            Class::Zero => 0,
            Class::Infinite => INFINITY,
            Class::Nan => INFINITY | decode_digits(&self.digits) as u64,
            Class::Finite => {
                let significand = decode_digits(&self.digits) as u128;
                let mut exponent = i32::from(self.exponent);
                if exponent > 1023 {
                    INFINITY
                } else if exponent < -1022 {
                    // Express directly in units of 2^-1074: avoid double rounding.
                    // Rounding may yield 2^52, which encodes the smallest normal.
                    round_shift_even(significand, (-exponent - 1007) as u32) as u64
                } else {
                    let mut rounded = round_shift_even(significand, 15) as u64;
                    if rounded == 1_u64 << 53 {
                        rounded >>= 1;
                        exponent += 1;
                    }
                    if exponent > 1023 {
                        INFINITY
                    } else {
                        ((exponent + 1023) as u64) << 52 | (rounded & FRACTION_MASK)
                    }
                }
            }
        };
        sign | magnitude
    }
}

impl From<f64> for S64I8 {
    fn from(value: f64) -> Self {
        Self::expand(value)
    }
}

impl From<S64I8> for f64 {
    fn from(value: S64I8) -> Self {
        value.collapse()
    }
}

fn encode_digits(mut significand: u128) -> [i8; 10] {
    let mut digits = [0; 10];
    for digit in digits.iter_mut().rev() {
        let remainder = (significand & 127) as i16;
        *digit = if remainder >= 64 {
            (remainder - 128) as i8
        } else {
            remainder as i8
        };
        significand = (significand + 64) >> 7;
    }
    debug_assert_eq!(significand, 0);
    digits
}

fn decode_digits(digits: &[i8; 10]) -> i128 {
    digits
        .iter()
        .fold(0, |value, &d| value * 128 + i128::from(d))
}

fn round_shift_even(value: u128, shift: u32) -> u128 {
    // All callers supply a 68-bit value and a positive shift. For larger
    // shifts it lies strictly below half of the least output unit.
    if shift > S64I8::PRECISION {
        return 0;
    }
    let quotient = value >> shift;
    let remainder = value & ((1_u128 << shift) - 1);
    let halfway = 1_u128 << (shift - 1);
    quotient + u128::from(remainder > halfway || (remainder == halfway && quotient & 1 != 0))
}
