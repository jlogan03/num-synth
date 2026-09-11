use num_bigint::BigUint;
use num_synth::{Class, InvalidFiniteParts, S64I8};
use proptest::prelude::*;

const LOW: u128 = 1_u128 << 67;
const HIGH: u128 = 1_u128 << 68;
const INF: u64 = 0x7ff0_0000_0000_0000;
const FRAC: u64 = (1_u64 << 52) - 1;

fn digits(significand: u128) -> [i8; 10] {
    let mut remaining = significand as i128;
    let mut result = [0; 10];
    for digit in result.iter_mut().rev() {
        let balanced = (remaining + 64).rem_euclid(128) - 64;
        *digit = balanced as i8;
        remaining = (remaining - balanced) / 128;
    }
    assert_eq!(remaining, 0);
    result
}

fn decoded(digits: [i8; 10]) -> i128 {
    digits
        .into_iter()
        .enumerate()
        .map(|(i, digit)| i128::from(digit) * 128_i128.pow((9 - i) as u32))
        .sum()
}

fn finite(negative: bool, exponent: i16, significand: u128) -> S64I8 {
    S64I8::from_finite_parts(negative, exponent, digits(significand)).unwrap()
}

// Independent exact oracle: search the ordered positive binary64 encodings,
// compare distances in big-integer units, and break ties by encoding parity.
// The virtual successor of MAX is 2^1024; choosing it means overflow to infinity.
// This uses neither the implementation's shifts/rounding nor native f64 math.
fn reference_bits(negative: bool, exponent: i16, significand: u128) -> u64 {
    let power = i32::from(exponent) - 67;
    let unit = power.min(-1074);
    let exact = BigUint::from(significand) << (power - unit) as usize;
    let point = |bits: u64| -> BigUint {
        if bits == INF {
            return BigUint::from(1_u8) << (1024 - unit) as usize;
        }
        let field = (bits >> 52) as i32;
        let (mantissa, power) = if field == 0 {
            (bits & FRAC, -1074)
        } else {
            ((bits & FRAC) | (1_u64 << 52), field - 1075)
        };
        BigUint::from(mantissa) << (power - unit) as usize
    };
    let mut lower = 0;
    let mut upper = INF;
    let magnitude = if exact >= point(INF) {
        INF
    } else {
        while lower + 1 < upper {
            let middle = lower + (upper - lower) / 2;
            if point(middle) <= exact {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        let down = &exact - point(lower);
        let up = point(upper) - &exact;
        if down < up || (down == up && lower & 1 == 0) {
            lower
        } else {
            upper
        }
    };
    (u64::from(negative) << 63) | magnitude
}

fn check_expansion(bits: u64) {
    let value = S64I8::from_bits(bits);
    assert_eq!(value.to_bits(), bits);
    assert_eq!(
        S64I8::expand(f64::from_bits(bits)).collapse().to_bits(),
        bits
    );
    assert_eq!(value.is_sign_negative(), bits >> 63 != 0);
    let field = (bits >> 52) & 0x7ff;
    let fraction = bits & FRAC;
    let expected_class = match (field, fraction) {
        (0, 0) => Class::Zero,
        (0x7ff, 0) => Class::Infinite,
        (0x7ff, _) => Class::Nan,
        _ => Class::Finite,
    };
    assert_eq!(value.class(), expected_class);
    if let Some((negative, exponent, ds)) = value.finite_parts() {
        assert!(ds.iter().all(|&d| (-64..=63).contains(&d)));
        assert!((16..=32).contains(&ds[0]));
        let significand = decoded(ds);
        assert!((LOW as i128..HIGH as i128).contains(&significand));
        let (mantissa, power) = if field == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1_u64 << 52), field as i32 - 1075)
        };
        let internal_power = i32::from(exponent) - 67;
        let unit = power.min(internal_power);
        assert_eq!(
            BigUint::from(mantissa) << (power - unit) as usize,
            BigUint::from(significand as u128) << (internal_power - unit) as usize,
        );
        let rebuilt = S64I8::from_finite_parts(negative, exponent, ds).unwrap();
        assert_eq!(rebuilt.finite_parts(), value.finite_parts());
        assert_eq!(rebuilt.to_bits(), bits);
    } else {
        assert_ne!(expected_class, Class::Finite);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn arbitrary_binary64_encodings_round_trip(bits in any::<u64>()) {
        check_expansion(bits);
    }

    #[test]
    fn arbitrary_nan_payloads_round_trip(
        negative in any::<bool>(), payload in 1_u64..=FRAC,
    ) {
        check_expansion((u64::from(negative) << 63) | INF | payload);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn arbitrary_canonical_values_match_exact_oracle(
        negative in any::<bool>(),
        exponent in prop_oneof![
            4 => -1076_i16..=1025,
            1 => any::<i16>(),
            2 => prop::sample::select(vec![i16::MIN, -1076, -1075, -1074, -1023, -1022, 1023, 1024, i16::MAX]),
        ],
        significand in LOW..HIGH,
    ) {
        let value = finite(negative, exponent, significand);
        prop_assert_eq!(decoded(value.finite_parts().unwrap().2), significand as i128);
        prop_assert_eq!(value.to_bits(), reference_bits(negative, exponent, significand));
    }

    #[test]
    fn canonical_digit_vectors_match_exact_oracle(
        negative in any::<bool>(), exponent in -1076_i16..=1024,
        top in 17_i8..=31, tail in prop::array::uniform9(-64_i8..=63),
    ) {
        let mut ds = [top; 10];
        ds[1..].copy_from_slice(&tail);
        let significand = decoded(ds) as u128;
        let value = S64I8::from_finite_parts(negative, exponent, ds).unwrap();
        prop_assert_eq!(value.finite_parts(), Some((negative, exponent, ds)));
        prop_assert_eq!(value.to_bits(), reference_bits(negative, exponent, significand));
    }

    #[test]
    fn midpoint_neighbors_match_exact_oracle(lower in 0_u64..INF, negative in any::<bool>()) {
        let field = (lower >> 52) as i32;
        let (mantissa, power) = if field == 0 {
            (lower & FRAC, -1074)
        } else {
            ((lower & FRAC) | (1_u64 << 52), field - 1075)
        };
        // Neighbor spacing is 2^power even at binade and overflow boundaries.
        let midpoint = 2 * u128::from(mantissa) + 1;
        let leading = 127 - midpoint.leading_zeros();
        let s = midpoint << (67 - leading);
        let e = (power - 1 + leading as i32) as i16;
        for offset in [-1_i128, 0, 1] {
            let mut adjusted = (s as i128 + offset) as u128;
            let mut exponent = e;
            if adjusted < LOW {
                adjusted *= 2;
                exponent -= 1;
            }
            let expected_magnitude = match offset {
                -1 => lower,
                0 if lower & 1 == 0 => lower,
                _ => lower + 1,
            };
            let expected = (u64::from(negative) << 63) | expected_magnitude;
            prop_assert_eq!(finite(negative, exponent, adjusted).to_bits(), expected);
            prop_assert_eq!(reference_bits(negative, exponent, adjusted), expected);
        }
    }
}

#[test]
fn every_binary64_exponent_and_fraction_boundaries() {
    for field in 0_u64..=0x7ff {
        for fraction in [0, 1, 2, 3, (1 << 51) - 1, 1 << 51, FRAC - 1, FRAC] {
            for sign in [0, 1_u64 << 63] {
                check_expansion(sign | (field << 52) | fraction);
            }
        }
    }
    for bit in 0..52 {
        for fraction in [(1_u64 << bit) - 1, 1 << bit, (1 << bit) + 1] {
            check_expansion(fraction);
            check_expansion((1_u64 << 63) | fraction);
        }
    }
}

#[test]
fn ties_sticky_bits_and_range_transitions() {
    let one = 0x3ff0_0000_0000_0000;
    let cases = [
        (0, LOW + (1 << 14) - 1, one),
        (0, LOW + (1 << 14), one),
        (0, LOW + (1 << 14) + 1, one + 1),
        (0, LOW + (3 << 14) - 1, one + 1),
        (0, LOW + (3 << 14), one + 2),
        (0, LOW + (3 << 14) + 1, one + 2),
        (0, HIGH - (1 << 14) - 1, one + FRAC),
        (0, HIGH - (1 << 14), one + (1 << 52)),
        (-1076, HIGH - 1, 0),
        (-1075, LOW, 0),
        (-1075, LOW + 1, 1),
        (-1074, LOW + LOW / 2 - 1, 1),
        (-1074, LOW + LOW / 2, 2),
        (-1023, HIGH - (1 << 15) - 1, FRAC),
        (-1023, HIGH - (1 << 15), 1 << 52),
        (-1023, HIGH - (1 << 15) + 1, 1 << 52),
        (1023, HIGH - (1 << 14) - 1, INF - 1),
        (1023, HIGH - (1 << 14), INF),
        (1023, HIGH - (1 << 14) + 1, INF),
        (1024, LOW, INF),
        (i16::MIN, LOW, 0),
        (i16::MAX, HIGH - 1, INF),
    ];
    for (exponent, significand, magnitude) in cases {
        for negative in [false, true] {
            let expected = (u64::from(negative) << 63) | magnitude;
            assert_eq!(finite(negative, exponent, significand).to_bits(), expected);
            assert_eq!(reference_bits(negative, exponent, significand), expected);
        }
    }
}

#[test]
fn finite_constructor_rejects_noncanonical_parts() {
    for bad in [LOW - 1, HIGH, 0] {
        assert!(matches!(
            S64I8::from_finite_parts(false, 0, digits(bad)),
            Err(InvalidFiniteParts::NotNormalized)
        ));
    }
    for invalid in [i8::MIN, -65, 64, i8::MAX] {
        for index in 0..10 {
            let mut ds = digits(LOW);
            ds[index] = invalid;
            assert!(matches!(
                S64I8::from_finite_parts(false, 0, ds),
                Err(InvalidFiniteParts::DigitOutOfRange)
            ));
        }
    }
    assert!(matches!(
        S64I8::from_finite_parts(false, 0, [-64; 10]),
        Err(InvalidFiniteParts::NotNormalized)
    ));
    for exponent in [i16::MIN, i16::MAX] {
        for significand in [LOW, HIGH - 1] {
            assert_eq!(
                decoded(
                    finite(false, exponent, significand)
                        .finite_parts()
                        .unwrap()
                        .2
                ),
                significand as i128
            );
        }
    }
}

#[test]
fn layout_and_from_conversions() {
    assert_eq!(core::mem::size_of::<S64I8>(), 16);
    assert_eq!(S64I8::PRECISION, 68);
    assert_eq!(S64I8::DIGIT_COUNT, 10);
    assert_eq!(S64I8::RADIX, 128);
    for bits in [0, 1 << 63, 1, FRAC, INF - 1, INF, INF | 1, u64::MAX] {
        let value: S64I8 = f64::from_bits(bits).into();
        let restored: f64 = value.into();
        assert_eq!(restored.to_bits(), bits);
    }
}
