mod common;
use common::*;
use num_synth::{Class, S64U8};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]
    #[test]
    fn binary64_roundtrip(bits in any::<u64>()) {
        let value=S64U8::from_bits(bits);
        prop_assert_eq!(value.to_bits(),bits);
        prop_assert_eq!(S64U8::from_f64(f64::from_bits(bits)).to_f64().to_bits(),bits);
        if value.class()==Class::Finite {
            let (n,p)=dyadic(value);
            prop_assert_eq!(format_reference(&n,p,52,1023),bits);
        }
    }
    #[test]
    fn binary32_roundtrip(bits in any::<u32>()) {
        let value=S64U8::from_f32_bits(bits);
        prop_assert_eq!(value.to_f32_bits(),bits);
        prop_assert_eq!(S64U8::from_f32(f32::from_bits(bits)).to_f32().to_bits(),bits);
        if value.class()==Class::Finite {
            let (n,p)=dyadic(value);
            prop_assert_eq!(format_reference(&n,p,23,127) as u32,bits);
        }
    }
}
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn ieee_midpoints_and_neighbors(
        raw in any::<u64>(), negative in any::<bool>(), binary32 in any::<bool>(),
    ) {
        let (fraction, bias, sign_bit) = if binary32 { (23,127,31) } else { (52,1023,63) };
        let infinity = ((2*bias+1) as u64) << fraction;
        let lower = raw % infinity;
        let field = (lower >> fraction) as i32;
        let f = lower & ((1_u64 << fraction)-1);
        let (mantissa, power) = if field == 0 { (f,1-bias-fraction) }
            else { (f | (1 << fraction),field-bias-fraction) };
        let midpoint = 2*mantissa+1;
        let leading = 63-midpoint.leading_zeros() as i32;
        let shift = 255-leading;
        let n = num_bigint::BigInt::from(midpoint) << shift as usize;
        for offset in [-1_i32,0,1] {
            let exact = &n + offset;
            let value = synthetic_reference(&exact, power-1-shift);
            let value = if negative { -value } else { value };
            let magnitude = if offset < 0 || (offset == 0 && lower & 1 == 0) { lower } else { lower+1 };
            let expected = (u64::from(negative) << sign_bit) | magnitude;
            if binary32 { prop_assert_eq!(u64::from(value.to_f32_bits()), expected); }
            else { prop_assert_eq!(value.to_bits(), expected); }
        }
    }

    #[test]
    fn conversions_match_exact_distance_oracle(negative in any::<bool>(), exponent in prop_oneof![-1100_i16..=1100, any::<i16>()], digits in any::<[u8;32]>()) {
        let mut ds=digits; ds[31]|=128;
        let value=finite(negative,exponent,ds);
        let (n,p)=dyadic(value);
        prop_assert_eq!(value.to_bits(),format_reference(&n,p,52,1023));
        prop_assert_eq!(value.to_f32_bits(),format_reference(&n,p,23,127) as u32);
    }
}
#[test]
fn every_ieee_exponent_and_boundaries() {
    for field in 0..=0x7ff_u64 {
        for fraction in [0, 1, 2, (1 << 51) - 1, 1 << 51, (1 << 52) - 1] {
            for sign in [0, 1 << 63] {
                let bits = sign | (field << 52) | fraction;
                assert_eq!(S64U8::from_bits(bits).to_bits(), bits);
            }
        }
    }
    for field in 0..=0xff_u32 {
        for fraction in [0, 1, 2, (1 << 22) - 1, 1 << 22, (1 << 23) - 1] {
            for sign in [0, 1 << 31] {
                let bits = sign | (field << 23) | fraction;
                assert_eq!(S64U8::from_f32_bits(bits).to_f32_bits(), bits);
            }
        }
    }
    for bit in 0..52 {
        for fraction in [(1_u64 << bit) - 1, 1 << bit, (1 << bit) + 1] {
            for sign in [0, 1 << 63] {
                let bits = sign | fraction;
                assert_eq!(S64U8::from_bits(bits).to_bits(), bits);
            }
        }
    }
    for bit in 0..23 {
        for fraction in [(1_u32 << bit) - 1, 1 << bit, (1 << bit) + 1] {
            for sign in [0, 1 << 31] {
                let bits = sign | fraction;
                assert_eq!(S64U8::from_f32_bits(bits).to_f32_bits(), bits);
            }
        }
    }
}
#[test]
fn standalone_rounding_ties_subnormals_and_payloads() {
    let one = power(0);
    // Direct binary32 rounding must not pass through binary64.
    let midpoint = one + power(-24);
    assert_eq!(
        (midpoint + power(-100)).to_f32_bits(),
        1.0_f32.to_bits() + 1
    );
    assert_eq!((midpoint + power(-100)).to_f64() as f32, 1.0_f32);
    for (fraction, min_sub, max_e) in [(52, -1074, 1023), (23, -149, 127)] {
        let midpoint = one + power(-(fraction + 1));
        for v in [
            midpoint,
            midpoint + power(-200),
            midpoint - power(-200),
            power(min_sub - 1),
            power(min_sub - 1) + power(min_sub - 200),
            power(max_e + 1) - power(max_e - fraction - 1),
        ] {
            let (n, p) = dyadic(v);
            assert_eq!(v.to_bits(), format_reference(&n, p, 52, 1023));
            assert_eq!(v.to_f32_bits(), format_reference(&n, p, 23, 127) as u32);
        }
    }
    assert!(S64U8::from_finite_parts(false, 0, [0; 32]).is_err());
    assert_eq!(S64U8::PRECISION, 256);
    assert_eq!(core::mem::size_of::<S64U8>(), 36);
    assert_eq!(
        S64U8::from_bits(0x7ff0_0000_0000_0001).to_f32_bits(),
        0x7f80_0001
    );
    let v: f64 = S64U8::from(1.25_f64).into();
    assert_eq!(v, 1.25);
    let v: f32 = S64U8::from(1.25_f32).into();
    assert_eq!(v, 1.25);
}
