mod common;
use common::*;
use num_synth::Df32;
use num_traits::{One, Signed, Zero};
use proptest::prelude::*;

fn pair() -> impl Strategy<Value = Df32> {
    (
        any::<bool>(),
        -10i32..=10,
        any::<u32>(),
        any::<bool>(),
        0i32..=10,
        any::<u32>(),
    )
        .prop_map(|(sign, e, h, low_sign, gap, l)| {
            let hi =
                f32::from_bits((u32::from(sign) << 31) | ((e + 127) as u32) << 23 | (h & 0x7fffff));
            let lo = f32::from_bits(
                (u32::from(low_sign) << 31) | ((e + 127 - 25 - gap) as u32) << 23 | (l & 0x7fffff),
            );
            Df32::from_parts(hi, lo)
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn scalar_round_trips(bits in any::<u32>()) {
        let x = f32::from_bits(bits);
        let d = Df32::from_f32(x);
        normalized(d);
        prop_assert_eq!(d.to_f32().to_bits(), if x.is_nan() { 0x7fc00000 } else { bits });
    }

    #[test]
    fn arbitrary_components_normalize(a in any::<u32>(), b in any::<u32>()) {
        let a = f32::from_bits(a);
        let b = f32::from_bits(b);
        let d = Df32::from_parts(a, b);
        normalized(d);
        if a.is_finite() && b.is_finite() {
            let exact = scalar(a) + scalar(b);
            if d.is_finite() {
                prop_assert_eq!(value(d), exact);
            } else {
                prop_assert_eq!(d.to_f32().to_bits(), round32(&exact).to_bits());
            }
        }
    }

    #[test]
    fn f64_conversion_normalizes(bits in any::<u64>()) {
        let x = f64::from_bits(bits);
        let d = Df32::from_f64(x);
        normalized(d);
        if x == 0.0 { prop_assert_eq!(d.to_f64().to_bits(), bits); }
        if x.is_nan() { prop_assert!(d.is_nan()); }
        if x.is_finite() && x.abs() >= 1e-30 && x.abs() <= 1e30 {
            prop_assert!((d.to_f64()-x).abs() <= x.abs() * 2f64.powi(-47));
        }
    }

    #[test]
    fn arithmetic_matches_exact_references(a in pair(), b in pair(), c in pair()) {
        let av = value(a);
        let bv = value(b);
        let cv = value(c);
        let sum = a+b;
        let difference = a-b;
        let product = a*b;
        let fused = a.mul_add(b,c);
        let quotient = a/b;
        for x in [sum, difference, product, fused, quotient, -a] { normalized(x); }
        bounded_error(&value(sum), &(&av + &bv), &(av.abs()+bv.abs()), 46);
        bounded_error(&value(difference), &(&av - &bv), &(av.abs()+bv.abs()), 46);
        let exact_product = &av * &bv;
        bounded_error(&(value(product)<<149), &exact_product, &exact_product, 46);
        let exact_fused = &exact_product + (&cv <<149);
        bounded_error(&(value(fused)<<149), &exact_fused, &(exact_product.abs()+(cv.abs()<<149)), 46);
        // Compare q*b with a on the product lattice: no approximate division oracle.
        bounded_error(&(value(quotient)*&bv), &(&av<<149), &(av.abs()<<149), 45);
        prop_assert_eq!(a.partial_cmp(&b), Some(av.cmp(&bv)));
        prop_assert_eq!(a == b, av == bv);
    }

    #[test]
    fn fused_cancellation_retains_extra_precision(a in pair(), b in pair()) {
        let c = -(a*b);
        let exact = value(a)*value(b) + (value(c)<<149);
        let fused = a.mul_add(b,c);
        normalized(fused);
        // Stronger cancellation check: absolute error scales with corrections,
        // not the canceled leading product. This is deliberately not a universal
        // relative-error assertion on an arbitrarily tiny result.
        bounded_error(&(value(fused)<<149), &exact, &(value(a)*value(b)), 66);
    }

    #[test]
    fn full_range_pairs_remain_normalized(bits in prop::array::uniform6(any::<u32>())) {
        let a = Df32::from_parts(f32::from_bits(bits[0]), f32::from_bits(bits[1]));
        let b = Df32::from_parts(f32::from_bits(bits[2]), f32::from_bits(bits[3]));
        let c = Df32::from_parts(f32::from_bits(bits[4]), f32::from_bits(bits[5]));
        for x in [a+b,a-b,a*b,a/b,a.mul_add(b,c)] { normalized(x); }
        if a.is_finite() && b.is_finite() {
            prop_assert_eq!(a.partial_cmp(&b), Some(value(a).cmp(&value(b))));
        }
    }

    #[test]
    fn full_exponent_outputs_are_canonical(a in any::<u32>(), b in any::<u32>(), c in any::<u32>()) {
        let a = Df32::from_f32(f32::from_bits(a));
        let b = Df32::from_f32(f32::from_bits(b));
        let c = Df32::from_f32(f32::from_bits(c));
        for x in [a+b, a-b, a*b, a/b, a.mul_add(b,c)] { normalized(x); }
    }
}

#[test]
fn representation_and_traits() {
    assert_eq!(core::mem::size_of::<Df32>(), 8);
    assert_eq!(core::mem::align_of::<Df32>(), core::mem::align_of::<f32>());
    assert_eq!(Df32::default(), Df32::zero());
    assert_eq!(Df32::one(), Df32::ONE);
    let mut x = Df32::from(3.0f32);
    x += Df32::ONE;
    x *= Df32::from(2.0f32);
    x -= Df32::from(2.0f32);
    x /= Df32::from(3.0f32);
    assert_eq!(f32::from(x), 2.0);
    assert_eq!(f64::from(x), 2.0);
    num_traits::MulAddAssign::mul_add_assign(&mut x, Df32::from(3.0f32), Df32::ONE);
    assert_eq!(x.to_f32(), 7.0);
    assert_eq!(
        num_traits::MulAdd::mul_add(x, Df32::ONE, Df32::ONE).to_f32(),
        8.0
    );
}

#[test]
fn ties_boundaries_and_arbitrary_order() {
    let half = 2f32.powi(-24);
    let even = Df32::from_parts(1.0, half);
    let odd_hi = f32::from_bits(1.0f32.to_bits() + 1);
    let odd = Df32::from_parts(odd_hi, half);
    assert_eq!(even.to_parts(), (1.0, half));
    assert_eq!(
        odd.to_parts(),
        (f32::from_bits(1.0f32.to_bits() + 2), -half)
    );
    assert_eq!(Df32::from_parts(half, 1.0), even);
    assert_eq!(Df32::from_parts(1.0, -1.0), Df32::ZERO);
    let tiny = f32::from_bits(1);
    let below_normal = f32::from_bits(0x007fffff);
    assert_eq!(
        Df32::from_parts(below_normal, tiny).to_parts(),
        (f32::MIN_POSITIVE, 0.0)
    );
    assert_eq!(
        Df32::from_parts(f32::MIN_POSITIVE, -tiny).to_parts(),
        (below_normal, 0.0)
    );
    assert!(Df32::from_parts(f32::MAX, 2f32.powi(103)).is_infinite());
    let wide = Df32::from_parts(1.0, tiny);
    assert_eq!(wide.to_parts(), (1.0, tiny));
    assert_eq!(wide.to_f64(), 1.0);
    assert!(wide > Df32::ONE);
    for x in [even, odd, wide, Df32::from_parts(1.0, -half)] {
        normalized(x);
    }
}

#[test]
fn fused_product_cancellation() {
    let a = Df32::from_parts(1.0, 2f32.powi(-24));
    let b = Df32::from_parts(1.0, -2f32.powi(-24));
    assert_eq!(a.mul_add(b, -Df32::ONE), Df32::from_f32(-2f32.powi(-48)));
    let c = -(a * a);
    assert_eq!(a * a + c, Df32::ZERO);
    assert_eq!(a.mul_add(a, c), Df32::from_f32(2f32.powi(-48)));
    // A far-separated residual remains meaningful after leading cancellation.
    let a = Df32::from_parts(1.0, 2f32.powi(-80));
    assert_eq!(
        a.mul_add(Df32::ONE, -Df32::ONE),
        Df32::from_f32(2f32.powi(-80))
    );
}

#[test]
fn small_features_and_transform_chains() {
    let offset = Df32::from_f32(1e-10);
    let base = Df32::from_f32(10.0);
    assert_eq!((base + offset) - base, offset);

    for origin in [-12.5, -10.0, -0.01, 0.0, 0.01, 10.0, 12.5] {
        for delta in [-1e-5, -1e-6, -1e-7, 1e-7, 1e-6, 1e-5] {
            let x = Df32::from_f64(origin);
            let y = x + Df32::from_f64(delta);
            for scale in [0.125, 0.7, 1.0, 3.25, 10.0] {
                let s = Df32::from_f64(scale);
                let t = Df32::from_f64(9.123456789);
                let inverse = |v: Df32| (v.mul_add(s, t) - t) / s;
                let result = (inverse(y) - inverse(x)).to_f64();
                assert!(
                    (result - delta).abs() < 1e-9,
                    "origin={origin} delta={delta} scale={scale} got={result}"
                );
            }
        }
    }
}

#[test]
fn special_values_and_signed_zeros() {
    let samples = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let same_scalar = |x: Df32, y: f32| {
        normalized(x);
        if y.is_nan() {
            assert!(x.is_nan());
        } else {
            assert_eq!(x.to_f32().to_bits(), y.to_bits());
        }
    };
    for a in samples {
        for b in samples {
            let x = Df32::from_f32(a);
            let y = Df32::from_f32(b);
            same_scalar(x + y, a + b);
            same_scalar(x - y, a - b);
            same_scalar(x * y, a * b);
            same_scalar(x / y, a / b);
            assert_eq!(x.partial_cmp(&y), a.partial_cmp(&b));
            for c in samples {
                same_scalar(x.mul_add(y, Df32::from_f32(c)), a.mul_add(b, c));
            }
        }
    }
    assert_eq!((-Df32::ZERO).to_f32().to_bits(), (-0.0f32).to_bits());
    assert_eq!(Df32::from_f64(-0.0).to_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(
        Df32::from_f32(f32::from_bits(0xff800001))
            .to_parts()
            .0
            .to_bits(),
        0x7fc00000
    );
    assert_eq!((-Df32::NAN).to_parts().0.to_bits(), 0x7fc00000);
    let max = Df32::from_f32(f32::MAX);
    assert!((max * Df32::from_f32(2.0)).is_infinite());
    assert!(max.mul_add(Df32::from_f32(2.0), -max).is_infinite());
    let tiny = Df32::from_f32(-f32::from_bits(1));
    same_scalar(tiny * Df32::from_f32(0.25), -0.0);
    same_scalar(tiny / Df32::from_f32(4.0), -0.0);
}

#[test]
fn residual_carry_can_overflow() {
    let quarter_ulp = 2f32.powi(102);
    let a = Df32::from_parts(f32::MAX, quarter_ulp);
    let b = Df32::from_f32(quarter_ulp);
    assert_eq!(a + b, Df32::INFINITY);
    assert_eq!(a.mul_add(Df32::ONE, b), Df32::INFINITY);
    assert_eq!(-a - b, Df32::NEG_INFINITY);
}

#[test]
fn deterministic_exponent_boundaries() {
    for e in 0u32..255 {
        for fraction in [0, 1, 0x3fffff, 0x7ffffe, 0x7fffff] {
            for sign in [0, 1 << 31] {
                let hi = f32::from_bits(sign | e << 23 | fraction);
                let x = Df32::from_f32(hi);
                assert_eq!(x.to_f32().to_bits(), hi.to_bits());
                for tail in [f32::from_bits(1), -f32::from_bits(1), hi * 2f32.powi(-24)] {
                    let pair = Df32::from_parts(hi, tail);
                    normalized(pair);
                    if pair.is_finite() {
                        assert_eq!(value(pair), scalar(hi) + scalar(tail));
                        normalized(pair + pair);
                        normalized(pair * Df32::from_f32(0.5));
                        normalized(pair / Df32::from_f32(0.5));
                        normalized(pair.mul_add(Df32::ONE, -pair));
                    }
                }
            }
        }
    }
}

#[test]
fn fused_and_separate_division_remainders_meet_accuracy_target() {
    for a in [0.125, 0.99999999999, 1.00000000001, 10.000001, 1234.56789] {
        for b in [0.7, 1.00000001, 3.25, 9.87654321] {
            let a = Df32::from_f64(a);
            let b = Df32::from_f64(b);
            let q0 = Df32::from_f32(a.to_f32() / b.to_f32());
            let r = (-b).mul_add(q0, a);
            let (hi, lo) = r.to_parts();
            let fused = Df32::from_parts(q0.to_f32(), (hi + lo) / b.to_f32());
            normalized(fused);
            for quotient in [a / b, fused] {
                bounded_error(
                    &(value(quotient) * value(b)),
                    &(value(a) << 149),
                    &(value(a).abs() << 149),
                    45,
                );
            }
        }
    }
}
