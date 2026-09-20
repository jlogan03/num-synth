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
        let one = from(1.0);
        // These remain distinct from 1 even after rounding to f32 and keep
        // repeated multiplication/division finite and normal for the full run.
        let a = black_box(from(1.00000100000001));
        let b = black_box(from(0.99999900000001));
        let c = black_box(from(1e-7));
        let scale = black_box(from(3.25));
        let inverse = black_box(from(1.0 / 3.25));
        let translation = black_box(from(1.23456789));
        [
            ("add chain", measure(one, |x| x + c)),
            ("sub chain", measure(one, |x| x - c)),
            ("mul chain", measure(one, |x| x * a)),
            ("div chain", measure(one, |x| x / a)),
            ("mul_add chain", measure(one, |x| x.mul_add(b, c))),
            ("separate mul/add chain", measure(one, |x| x * b + c)),
            (
                "independent mul_add x4",
                measure([one, from(1.25), from(1.5), from(1.75)], |x| {
                    x.map(|v| v.mul_add(b, c))
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
    println!("The x4 row measures four values per iteration. Includes loop/black_box overhead.");
    println!(
        "{:28} {:>10} {:>10} {:>10}",
        "workload", "f32", "f64", "Df32"
    );
    for (((name, f32_ns), (_, f64_ns)), (_, df32_ns)) in single.into_iter().zip(double).zip(paired)
    {
        println!("{name:28} {f32_ns:10.2} {f64_ns:10.2} {df32_ns:10.2}");
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
