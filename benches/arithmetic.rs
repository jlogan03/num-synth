#![cfg_attr(feature = "half", feature(f16))]

#[cfg(feature = "half")]
use num_synth::Df16;
use num_synth::Df32;
use std::{hint::black_box, time::Instant};

const ITERATIONS: usize = 200_000;
const SAMPLES: usize = 7;

// Report the median of repeated runs. Every type gets the same loop and
// black_box placement; input conversions happen outside the timed region.
fn measure<T: Copy>(initial: T, mut op: impl FnMut(T) -> T) -> f64 {
    let mut state = black_box(initial);
    for _ in 0..2_000 {
        state = black_box(op(black_box(state)));
    }
    black_box(state);

    let mut samples = [0.0; SAMPLES];
    for sample in &mut samples {
        state = black_box(initial);
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            state = black_box(op(black_box(state)));
        }
        *sample = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        black_box(state);
    }
    samples.sort_unstable_by(f64::total_cmp);
    samples[SAMPLES / 2]
}

// Expand the same workload for each concrete type, using its own operators
// and inherent mul_add. No trait dispatch or conversions in the timed loops.
macro_rules! suite {
    ($ty:ty, $convert:expr) => {{
        let from: fn(f64) -> $ty = $convert;
        let one = from(1.000123456789);
        // Exact reciprocal powers of two keep every chain finite and normal
        // even in f16. Black-box coefficients prevent constant folding.
        let a = black_box(from(2.0));
        let b = black_box(from(0.5));
        let c = black_box(from(0.125));
        let d = black_box(from(-0.25));
        let scale = black_box(from(2.0));
        let inverse = black_box(from(0.5));
        let translation = black_box(from(1.25));
        [
            ("add chain", measure(one, |x| (x + c) + (-c))),
            ("sub chain", measure(one, |x| (x - c) - (-c))),
            ("mul chain", measure(one, |x| (x * a) * b)),
            ("div chain", measure(one, |x| (x / a) / b)),
            (
                "mul_add chain",
                measure(one, |x| x.mul_add(b, c).mul_add(a, d)),
            ),
            (
                "separate mul/add chain",
                measure(one, |x| (x * b + c) * a + d),
            ),
            (
                "independent mul_add x4",
                measure([one, from(1.25), from(1.5), from(1.75)], |x| {
                    x.map(|v| v.mul_add(b, c).mul_add(a, d))
                }),
            ),
            (
                "position round trip",
                measure(from(10.000001), |x| {
                    (x.mul_add(scale, translation) - translation) * inverse
                }),
            ),
        ]
    }};
}

fn main() {
    let single = suite!(f32, |x| x as f32);
    let double = suite!(f64, |x| x);
    let paired = suite!(Df32, Df32::from_f64);

    println!("Median of {SAMPLES} samples, {ITERATIONS} iterations each; ns/iteration.");
    println!("Each chain iteration has two operations (two mul/add pairs for separate mul/add).");
    println!("The x4 row has eight FMAs; position round trip has one FMA, subtraction, multiply.");
    println!(
        "Bounded round trips, including loop/black_box overhead; coefficients are black-boxed."
    );
    #[cfg(feature = "half")]
    let half = suite!(f16, |x| x as f16);
    #[cfg(feature = "half")]
    let half_pair = suite!(Df16, Df16::from_f64);
    print!(
        "{:28} {:>10} {:>10} {:>10}",
        "workload", "f32", "f64", "Df32"
    );
    #[cfg(feature = "half")]
    print!(" {:>10} {:>10}", "f16", "Df16");
    println!();
    for i in 0..single.len() {
        print!(
            "{:28} {:10.2} {:10.2} {:10.2}",
            single[i].0, single[i].1, double[i].1, paired[i].1
        );
        #[cfg(feature = "half")]
        print!(" {:10.2} {:10.2}", half[i].1, half_pair[i].1);
        println!();
    }

    // Df32-specific algorithm comparison: the provisional quotient and its
    // correction are f32, unlike native f64 division. Keep this separate from
    // the matched arithmetic workloads above.
    let a = black_box(Df32::from_f64(1.00000100000001));
    let separate = measure(Df32::ONE, |x| {
        let q = Df32::from_f32(x.to_f32() / a.to_f32());
        let r = x - a * q;
        q + Df32::from_f32(r.to_f32() / a.to_f32())
    });
    let fused = measure(Df32::ONE, |x| {
        let q = Df32::from_f32(x.to_f32() / a.to_f32());
        let r = (-a).mul_add(q, x);
        q + Df32::from_f32(r.to_f32() / a.to_f32())
    });
    println!("\nDf32 division remainder: separate {separate:.2}, fused {fused:.2} ns/iteration");
}
