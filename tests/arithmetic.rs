mod common;
use common::*;
use num_synth::{Class, S64U8};
use proptest::prelude::*;

fn operand() -> impl Strategy<Value = S64U8> {
    (
        any::<bool>(),
        prop_oneof![4 => -1100_i16..=1100, 1 => any::<i16>(),
        1 => prop::sample::select(vec![i16::MIN, i16::MIN + 1, i16::MAX - 1, i16::MAX])],
        any::<[u8; 32]>(),
    )
        .prop_map(|(s, e, mut d)| {
            d[31] |= 128;
            finite(s, e, d)
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]
    #[test]
    fn working_arithmetic_matches_256_bit_reference(a in operand(), b in operand(), c in operand()) {
        let multiply = product(dyadic(a), dyadic(b));
        for (actual, (n,p)) in [
            (a+b, sum(dyadic(a),dyadic(b))),
            (a-b, sum(dyadic(a),dyadic(-b))),
            (a*b, multiply.clone()),
            (a.mul_add(b,c), sum(multiply,dyadic(c)))
        ] { assert_same(actual, synthetic_reference(&n,p)); }
        let diff = sum(dyadic(a),dyadic(-b)).0;
        prop_assert_eq!(a.partial_cmp(&b), Some(diff.cmp(&0.into())));
    }
    #[test]
    fn alignment_and_cancellation(a in any::<[u8;32]>(), b in any::<[u8;32]>(), c in any::<[u8;32]>(),
        gap in prop_oneof![-540_i16..=540, prop::sample::select(vec![-529,-528,-527,-273,-272,-271,-17,-16,-15,15,16,17,271,272,273,527,528,529])],
        opposite in any::<bool>()) {
        let mut ad=a; ad[31]|=128;
        let mut bd=b; bd[31]|=128;
        let mut cd=c; cd[31]|=128;
        let a=finite(false,0,ad); let b=finite(false,0,bd); let c=finite(opposite,gap,cd);
        let p=product(dyadic(a),dyadic(b));
        let (n,e)=sum(p.clone(),dyadic(c));
        assert_same(a.mul_add(b,c),synthetic_reference(&n,e));
        let cancel=-synthetic_reference(&p.0,p.1);
        let (n,e)=sum(p,dyadic(cancel));
        assert_same(a.mul_add(b,cancel),synthetic_reference(&n,e));
    }
    #[test]
    fn arithmetic_chains_use_working_precision(a in operand(), b in operand(), c in operand()) {
        let (n,p)=sum(dyadic(a),dyadic(b));
        let first=synthetic_reference(&n,p);
        if first.class()==Class::Finite {
            let (n,p)=product(dyadic(first),dyadic(c));
            assert_same((a+b)*c,synthetic_reference(&n,p));
        }
    }
}

#[test]
fn retains_precision_beyond_old_68_bit_stage() {
    let one = power(0);
    let epsilon = power(-200);
    assert_same((one + epsilon) - one, epsilon);
    assert_same((one - epsilon) - one, -epsilon);
    // 256-bit resolution: exact tie rounds to even, then an above-tie rounds up.
    assert_same(one + power(-256), one);
    let ulp = power(-255);
    assert_same(one + (power(-256) + power(-300)), one + ulp);
    // Full 512-bit product survives leading cancellation in FMA.
    let a = one + ulp;
    let b = one - ulp;
    assert_same(a.mul_add(b, -one), -power(-510));
    assert_same(a * b - one, S64U8::from_f64(0.0));
}

#[test]
fn extremes_specials_and_midpoints() {
    let zero = S64U8::from_f64(0.0);
    let nz = -zero;
    let one = power(0);
    let min = power(i16::MIN);
    let half = power(-1);
    let two = power(1);
    let max = finite(false, i16::MAX, [255; 32]);
    assert_same(min * half, zero);
    assert_same(-min * half, nz);
    assert_same(min * (half + power(-256)), min);
    assert_eq!((max * two).class(), Class::Infinite);
    assert_same(max.mul_add(two, -max), max);
    assert_same(max + min, max);
    assert_same(min.mul_add(min, max), max);
    assert_same(max + power(i16::MAX - 256), S64U8::from_f64(f64::INFINITY));
    for a in [
        zero,
        nz,
        one,
        -one,
        S64U8::from_f64(f64::INFINITY),
        S64U8::from_f64(f64::NEG_INFINITY),
    ] {
        for b in [
            zero,
            nz,
            one,
            -one,
            S64U8::from_f64(f64::INFINITY),
            S64U8::from_f64(f64::NEG_INFINITY),
        ] {
            for (got, expected) in [
                (a + b, a.to_f64() + b.to_f64()),
                (a - b, a.to_f64() - b.to_f64()),
                (a * b, a.to_f64() * b.to_f64()),
            ] {
                if expected.is_nan() {
                    assert_eq!(got.class(), Class::Nan);
                } else {
                    assert_eq!(got.to_bits(), expected.to_bits());
                }
            }
            for c in [zero, nz, one, -one] {
                let expected = a.to_f64().mul_add(b.to_f64(), c.to_f64());
                let got = a.mul_add(b, c);
                if expected.is_nan() {
                    assert_eq!(got.class(), Class::Nan);
                } else {
                    assert_eq!(got.to_bits(), expected.to_bits());
                }
            }
        }
    }
    let snan = S64U8::from_bits(0xfff0_0000_0000_0123);
    for n in [snan + one, one - snan, snan * one, one.mul_add(one, snan)] {
        assert_eq!(n.to_bits(), 0xfff8_0000_0000_0123);
    }
    assert_eq!((-snan).to_bits(), 0x7ff0_0000_0000_0123);
    assert_ne!(snan, snan);
    assert_eq!(snan.partial_cmp(&one), None);
    // A remote sticky tail must pick the correct side of a working midpoint.
    let a = one + power(-255);
    let b = S64U8::from_f64(1.5);
    let tail = power(-1000);
    let (n, p) = sum(product(dyadic(a), dyadic(b)), dyadic(tail));
    assert_same(a.mul_add(b, tail), synthetic_reference(&n, p));
    let (n, p) = sum(product(dyadic(a), dyadic(b)), dyadic(-tail));
    assert_same(a.mul_add(b, -tail), synthetic_reference(&n, p));
}
