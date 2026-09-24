#![cfg(feature = "half")]
#![feature(f16)]

use num_synth::Df16;
use num_traits::{One, Zero};
use proptest::prelude::*;

// Exact binary16 lattice in units of 2^-24. Even pair products fit i128.
fn scalar(x: f16) -> i128 {
    assert!(x.is_finite());
    let bits = x.to_bits();
    let exp = (bits >> 10) & 31;
    let frac = (bits & 1023) as i128;
    let n = if exp == 0 {
        frac
    } else {
        (frac | 1024) << (exp - 1)
    };
    if x.is_sign_negative() { -n } else { n }
}
fn value(x: Df16) -> i128 {
    let (h, l) = x.to_parts();
    scalar(h) + scalar(l)
}

// Integer nearest-even rounding, independent of the primitive cast/TwoSum.
fn round16(n: i128) -> u16 {
    let sign = if n < 0 { 0x8000 } else { 0 };
    let a = n.unsigned_abs();
    if a < 1024 {
        return sign | a as u16;
    }
    let shift = (128 - a.leading_zeros()).saturating_sub(11);
    let mut q = a >> shift;
    if shift > 0 {
        let rem = a - (q << shift);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && q & 1 != 0) {
            q += 1;
        }
    }
    let mut exp = shift + 1;
    if q == 2048 {
        q >>= 1;
        exp += 1;
    }
    sign | if exp >= 31 {
        0x7c00
    } else {
        (exp as u16) << 10 | (q as u16 & 1023)
    }
}
fn normalized(x: Df16) {
    let (h, l) = x.to_parts();
    if !h.is_finite() {
        assert_eq!(l.to_bits(), 0);
        return;
    }
    assert!(l.is_finite());
    if l == 0.0 {
        assert_eq!(l.to_bits(), 0);
    }
    if value(x) == 0 {
        assert_eq!(h, 0.0);
    } else {
        assert_eq!(h.to_bits(), round16(value(x)), "{x:?}");
    }
}
fn same(x: Df16, y: Df16) {
    normalized(x);
    if y.is_nan() {
        assert!(x.is_nan());
    } else {
        assert_eq!(x.to_parts().0.to_bits(), y.to_parts().0.to_bits());
        assert_eq!(x.to_parts().1.to_bits(), y.to_parts().1.to_bits());
    }
}
fn pair() -> impl Strategy<Value = Df16> {
    (any::<bool>(), -6i32..=6, 0u16..1024, -1024i32..=1024).prop_map(|(neg, e, frac, tail)| {
        let hi = f16::from_bits(((neg as u16) << 15) | ((e + 15) as u16) << 10 | frac);
        let lo = (tail as f64 * 2f64.powi(e - 22)) as f16;
        Df16::from_parts(hi, lo)
    })
}

#[test]
fn exhaustive_scalar_patterns() {
    for bits in 0..=u16::MAX {
        let x = f16::from_bits(bits);
        let d = Df16::from_f16(x);
        normalized(d);
        assert_eq!(d.to_f16().to_bits(), bits);
        same(Df16::from_f32(x as f32), d);
        same(Df16::from_f64(x as f64), d);
        assert_eq!(d.is_finite(), x.is_finite());
        assert_eq!(d.is_nan(), x.is_nan());
        assert_eq!(d.is_infinite(), x.is_infinite());
        assert_eq!(d.is_sign_negative(), x.is_sign_negative());
        assert_eq!(d.is_zero(), x == 0.0);
        if !x.is_nan() {
            assert_eq!(d.to_f32().to_bits(), (x as f32).to_bits());
            assert_eq!(d.to_f64().to_bits(), (x as f64).to_bits());
        }
    }
}

#[test]
fn all_midpoints_and_neighbors() {
    for bits in 0..0x7c00u16 {
        let lower = f16::from_bits(bits) as f64;
        let upper = if bits == 0x7bff {
            65536.0
        } else {
            f16::from_bits(bits + 1) as f64
        };
        let mid = (lower + upper) * 0.5;
        for sign in [1.0, -1.0] {
            for x in [mid.next_down(), mid, mid.next_up()] {
                let x = x * sign;
                let h = x as f16;
                let reference = if h == 0.0 || !h.is_finite() {
                    Df16::from_f16(h)
                } else {
                    Df16::from_parts(h, (x - h as f64) as f16)
                };
                same(Df16::from_f64(x), reference);
            }
            let mid = mid as f32;
            for x in [mid.next_down(), mid, mid.next_up()] {
                let x = x * sign as f32;
                let h = x as f16;
                let reference = if h == 0.0 || !h.is_finite() {
                    Df16::from_f16(h)
                } else {
                    Df16::from_parts(h, (x - h as f32) as f16)
                };
                same(Df16::from_f32(x), reference);
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn components_and_full_range_arithmetic(bits in prop::array::uniform6(any::<u16>())) {
        let pairs: [Df16; 3] = core::array::from_fn(|i| {
            let p = [bits[2*i], bits[2*i+1]];
            let (a,b) = (f16::from_bits(p[0]), f16::from_bits(p[1]));
            let d = Df16::from_parts(a,b);
            normalized(d);
            if a.is_finite() && b.is_finite() {
                if d.is_finite() { assert_eq!(value(d), scalar(a)+scalar(b)); }
                else { assert_eq!(d.to_f16().to_bits(), round16(scalar(a)+scalar(b))); }
            }
            d
        });
        let (a,b,c) = (pairs[0],pairs[1],pairs[2]);
        for x in [a+b,a-b,a*b,a/b,a.mul_add(b,c),-a] { normalized(x); }
        if a.is_finite() && b.is_finite() {
            prop_assert_eq!(a.partial_cmp(&b), Some(value(a).cmp(&value(b))));
            prop_assert_eq!(a == b, value(a) == value(b));
            prop_assert_eq!(a.to_f64(), value(a) as f64 * 2f64.powi(-24));
        }
    }

    #[test]
    fn wider_conversions_normalize(a in any::<u32>(), b in any::<u64>()) {
        for (d,x) in [(Df16::from_f32(f32::from_bits(a)),f32::from_bits(a) as f64),
                      (Df16::from_f64(f64::from_bits(b)),f64::from_bits(b))] {
            normalized(d);
            if x.is_nan() { prop_assert!(d.is_nan()); }
            else if x == 0.0 { prop_assert_eq!(d.to_f64().to_bits(), x.to_bits()); }
            else if x.abs() < 65000.0 {
                prop_assert!((d.to_f64()-x).abs() <= x.abs()*2f64.powi(-21)+2f64.powi(-25));
            }
        }
    }

    #[test]
    fn arithmetic_error_on_exact_lattice(a in pair(), b in pair(), c in pair()) {
        let (av,bv,cv) = (value(a),value(b),value(c));
        for (d,exact) in [(a+b,av+bv),(a-b,av-bv)] {
            normalized(d);
            prop_assert!((value(d)-exact).abs() <= ((av.abs()+bv.abs())>>19)+2);
        }
        let product = av*bv;
        for (d,exact,scale) in [(a*b,product,product.abs()),
            (a.mul_add(b,c),product+(cv<<24),product.abs()+(cv.abs()<<24))] {
            normalized(d);
            prop_assert!(((value(d)<<24)-exact).abs() <= (scale>>19)+(16<<24),
                "a={a:?} b={b:?} c={c:?} got={d:?}");
        }
        let q = a/b;
        normalized(q);
        prop_assert!((value(q)*bv-(av<<24)).abs() <= ((av.abs()<<24)>>18)+4*((1<<24)+bv.abs()),
            "a={a:?} b={b:?} q={q:?}");
    }
}

#[test]
fn representation_traits_and_conversions() {
    assert_eq!(core::mem::size_of::<Df16>(), 4);
    assert_eq!(core::mem::align_of::<Df16>(), 4);
    assert_eq!(core::mem::size_of::<[Df16; 3]>(), 12);
    assert_eq!(Df16::default(), Df16::zero());
    assert_eq!(Df16::one(), Df16::ONE);
    const SPLIT: Df16 = Df16::from_f64(1.0001);
    const SPLIT32: Df16 = Df16::from_f32(1.0001);
    normalized(SPLIT);
    normalized(SPLIT32);
    let mut x = Df16::from(3.0f16);
    x += Df16::ONE;
    x *= Df16::from(2.0f32);
    x -= Df16::from(2.0f64);
    x /= Df16::from(3.0f16);
    assert_eq!(f16::from(x), 2.0);
    assert_eq!(f32::from(x), 2.0);
    assert_eq!(f64::from(x), 2.0);
    num_traits::MulAddAssign::mul_add_assign(&mut x, Df16::from_f16(3.0), Df16::ONE);
    assert_eq!(
        num_traits::MulAdd::mul_add(x, Df16::ONE, Df16::ONE).to_f16(),
        8.0
    );
    let wide = Df16::from_parts(128.0, f16::from_bits(1));
    assert_eq!(wide.to_f32(), 128.0);
    assert!(wide.to_f64() > 128.0);
    // A direct f64 split preserves detail lost by passing through f32 first.
    assert_ne!(
        Df16::from_f64(128.0 + 2f64.powi(-24)),
        Df16::from_f32(128.0)
    );
}

#[test]
fn cancellation_and_underflow_limits() {
    let a = Df16::from_parts(128.0, 0.03125);
    let c = -(a * a);
    assert_eq!(a * a + c, Df16::ZERO);
    assert_eq!(a.mul_add(a, c).to_f64(), 2f64.powi(-10));
    let tiny = Df16::from_f16(f16::from_bits(1));
    assert_eq!((Df16::ONE + tiny) - Df16::ONE, tiny);
    assert_eq!(tiny * Df16::from_f16(0.5), Df16::ZERO);
    // Normal leading product, but its exact error is below one half quantum.
    let a = Df16::from_f16(f16::from_bits(0x2001));
    assert_ne!((a * a).to_f64(), a.to_f64() * a.to_f64());
    let max = Df16::from_parts(f16::MAX, 8.0);
    assert_eq!(max + Df16::from_f16(8.0), Df16::INFINITY);
    assert_eq!(max.mul_add(Df16::ONE, Df16::from_f16(8.0)), Df16::INFINITY);
}

#[test]
fn special_values_and_signed_zeros() {
    let samples = [
        0.0f16,
        -0.0,
        1.0,
        -1.0,
        f16::INFINITY,
        f16::NEG_INFINITY,
        f16::NAN,
    ];
    for a in samples {
        for b in samples {
            let (x, y) = (Df16::from_f16(a), Df16::from_f16(b));
            for (d, expected) in [
                (x + y, a + b),
                (x - y, a - b),
                (x * y, a * b),
                (x / y, a / b),
            ] {
                same(d, Df16::from_f16(expected));
            }
            assert_eq!(x.partial_cmp(&y), a.partial_cmp(&b));
            for c in samples {
                same(
                    x.mul_add(y, Df16::from_f16(c)),
                    Df16::from_f16(a.mul_add(b, c)),
                );
            }
        }
    }
    same(-Df16::ZERO, Df16::NEG_ZERO);
    let max = Df16::from_f16(f16::MAX);
    let two = Df16::from_f16(2.0);
    assert_eq!(max.mul_add(two, -max), Df16::INFINITY);
    assert_eq!(max.mul_add(two, Df16::NEG_INFINITY), Df16::NEG_INFINITY);
    let tiny = Df16::from_f16(-f16::from_bits(1));
    same(tiny * Df16::from_f16(0.25), Df16::NEG_ZERO);
    same(tiny / Df16::from_f16(4.0), Df16::NEG_ZERO);
    same(Df16::from_f64(-1e-100), Df16::NEG_ZERO);
}
