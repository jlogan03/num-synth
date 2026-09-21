mod common;
use common::*;
use num_synth::Df32;
use num_traits::{One, Signed, Zero};
use proptest::prelude::*;

fn pair() -> impl Strategy<Value = Df32> {
    (
        any::<bool>(),
        -40i32..=40,
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
        prop_assert_eq!(d.to_f32().to_bits(), bits);
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
    fn multiplication_matches_full_twosum_normalization(bits in prop::array::uniform4(any::<u32>())) {
        let a = Df32::from_parts(f32::from_bits(bits[0]),f32::from_bits(bits[1]));
        let b = Df32::from_parts(f32::from_bits(bits[2]),f32::from_bits(bits[3]));
        let (ah,al)=a.to_parts();
        let (bh,bl)=b.to_parts();
        let p = ah*bh;
        let reference = if !p.is_finite() {
            Df32::from_f32(p)
        } else {
            let e = libm::fmaf(ah,bh,-p);
            let e = libm::fmaf(ah,bl,e);
            let e = libm::fmaf(al,bh,e);
            let e = libm::fmaf(al,bl,e);
            let result = Df32::from_parts(p,e);
            if result.is_zero() { Df32::from_f32(f32::from_bits(p.to_bits() & (1<<31))) } else { result }
        };
        let got = a*b;
        if reference.is_nan() {
            prop_assert!(got.is_nan());
        } else {
            prop_assert_eq!(got.to_parts().0.to_bits(),reference.to_parts().0.to_bits());
            prop_assert_eq!(got.to_parts().1.to_bits(),reference.to_parts().1.to_bits());
        }
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
    fn full_exponent_outputs_remain_normalized(a in any::<u32>(), b in any::<u32>(), c in any::<u32>()) {
        let a = Df32::from_f32(f32::from_bits(a));
        let b = Df32::from_f32(f32::from_bits(b));
        let c = Df32::from_f32(f32::from_bits(c));
        for x in [a+b, a-b, a*b, a/b, a.mul_add(b,c)] { normalized(x); }
    }
}

#[test]
fn f64_split_zero_and_nonfinite_boundaries() {
    const UNDERFLOW: Df32 = Df32::from_f64(-1e-100);
    assert_eq!(UNDERFLOW.to_parts().0.to_bits(), (-0.0f32).to_bits());

    let halfway = f64::from(f32::from_bits(1)) * 0.5;
    for magnitude in [
        0.0,
        f64::from_bits(1),
        f64::from_bits(halfway.to_bits() - 1),
        halfway,
        f64::from_bits(halfway.to_bits() + 1),
        f64::from(f32::from_bits(1)),
        f64::from(f32::MAX),
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ] {
        for sign in [1.0, -1.0] {
            let value = sign * magnitude;
            let result = Df32::from_f64(value);
            normalized(result);
            let (hi, lo) = result.to_parts();
            if value.is_nan() {
                assert!(hi.is_nan());
            } else {
                assert_eq!(hi.to_bits(), (value as f32).to_bits());
            }
            assert_eq!(lo.to_bits(), 0);
        }
    }
}

#[test]
fn representation_and_traits() {
    assert_eq!(core::mem::size_of::<Df32>(), 8);
    assert_eq!(core::mem::align_of::<Df32>(), 8);
    assert_eq!(core::mem::size_of::<[Df32; 3]>(), 24);
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
    assert!(Df32::from_f32(f32::from_bits(0xff800001)).is_nan());
    assert!((-Df32::NAN).is_nan());
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
        for fraction in [0, 1, 15, 16, 17, 0x3fffff, 0x7ffffe, 0x7fffff] {
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

#[test]
fn exceptional_results_survive_error_recovery() {
    let finite = Df32::from_parts(3.0, 1e-9);
    let max = Df32::from_f32(f32::MAX);
    let two = Df32::from_f32(2.0);
    assert_eq!(Df32::INFINITY + finite, Df32::INFINITY);
    assert_eq!(Df32::NEG_INFINITY * finite, Df32::NEG_INFINITY);
    assert_eq!(finite / Df32::INFINITY, Df32::ZERO);
    assert_eq!(
        (finite / Df32::NEG_INFINITY).to_f32().to_bits(),
        (-0.0f32).to_bits()
    );
    // Finite operands have a mathematically finite product even if the leading
    // f32 product overflows: the infinite addend still controls the FMA result.
    assert_eq!(max.mul_add(two, Df32::NEG_INFINITY), Df32::NEG_INFINITY);
    assert_eq!((-max).mul_add(two, Df32::INFINITY), Df32::INFINITY);
    assert!(Df32::INFINITY.mul_add(two, Df32::NEG_INFINITY).is_nan());
    assert!(Df32::ZERO.mul_add(Df32::INFINITY, finite).is_nan());
    // Removing the zero-operand shortcuts must retain the whole addend.
    for zero in [Df32::ZERO, Df32::NEG_ZERO] {
        assert_eq!(zero.mul_add(finite, finite), finite);
        assert_eq!(finite.mul_add(zero, finite), finite);
    }
}

// The previous conversion sequence is an independent full-TwoSum reference
// for the optimized split. This targets the rare midpoint cases random f64
// inputs almost never exercise, especially when residual rounding changes hi.
#[test]
fn f64_split_matches_full_twosum_at_midpoints() {
    for exponent in 0u32..255 {
        for fraction in [0, 1, 2, 0x3fffff, 0x7ffffd, 0x7ffffe, 0x7fffff] {
            let bits = (exponent << 23) | fraction;
            let lower = f32::from_bits(bits) as f64;
            let upper = if bits == f32::MAX.to_bits() {
                2f64.powi(128)
            } else {
                f32::from_bits(bits + 1) as f64
            };
            let midpoint = (lower + upper) * 0.5;
            for near in [
                midpoint.to_bits() - 1,
                midpoint.to_bits(),
                midpoint.to_bits() + 1,
            ] {
                for sign in [0, 1u64 << 63] {
                    let x = f64::from_bits(near | sign);
                    let hi = x as f32;
                    let reference = if !hi.is_finite() || x == 0.0 {
                        Df32::from_f32(hi)
                    } else {
                        Df32::from_parts(hi, (x - hi as f64) as f32)
                    };
                    let got = Df32::from_f64(x);
                    assert_eq!(
                        got.to_parts().0.to_bits(),
                        reference.to_parts().0.to_bits(),
                        "input={x:e}"
                    );
                    assert_eq!(
                        got.to_parts().1.to_bits(),
                        reference.to_parts().1.to_bits(),
                        "input={x:e}"
                    );
                    normalized(got);
                }
            }
        }
    }
}

#[test]
fn direct_division_remainder_extreme_scales() {
    for exponent in [-120, -80, -40, 0, 40, 80, 120] {
        for sign in [-1.0, 1.0] {
            let a = Df32::from_parts(sign * 2f32.powi(exponent), sign * 2f32.powi(exponent - 25));
            for b in [0.7, 1.00000001, 3.25, 9.87654321] {
                let b = Df32::from_f64(b);
                let quotient = a / b;
                normalized(quotient);
                let error = (value(quotient) * value(b) - (value(a) << 149usize)).abs();
                // Ordinary relative target plus two output subnormal quanta.
                let allowance = (value(a).abs() << 104) + (value(b).abs() << 1);
                assert!(error <= allowance, "a={a:?} b={b:?} q={quotient:?}");
            }
        }
    }
}
