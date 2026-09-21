use num_synth::Df32;
use std::{hint::black_box, time::Instant};

const VALUES: usize = 10_000;
const BATCHES: usize = 200;
const SAMPLES: usize = 7;

#[inline(never)]
fn measure<T: Copy>(a: &[T], b: &[T], c: &[T], output: &mut [T], op: impl Fn(T, T, T) -> T) -> f64 {
    let mut run = || {
        for (((&a, &b), &c), slot) in black_box(a)
            .iter()
            .zip(black_box(b))
            .zip(black_box(c))
            .zip(black_box(&mut *output))
        {
            *slot = op(a, b, c);
        }
        black_box(&*output);
    };
    for _ in 0..16 {
        run();
    }
    let mut samples = [0.0; SAMPLES];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..BATCHES {
            run();
        }
        *sample = start.elapsed().as_nanos() as f64 / (BATCHES * VALUES) as f64;
    }
    samples.sort_unstable_by(f64::total_cmp);
    samples[SAMPLES / 2]
}

macro_rules! suite {
    ($ty:ty, $convert:expr) => {{
        let from: fn(f64) -> $ty = $convert;
        let a: Vec<_> = (0..VALUES)
            .map(|i| from((i as f64 - 5_000.0) / 997.0))
            .collect();
        let b: Vec<_> = (0..VALUES)
            .map(|i| from(0.75 + i as f64 / 13_037.0))
            .collect();
        let c: Vec<_> = (0..VALUES)
            .map(|i| from((i as f64 - 3_000.0) / 7_001.0))
            .collect();
        let mut out = vec![from(0.0); VALUES];
        [
            ("add", measure(&a, &b, &c, &mut out, |a, b, _| a + b)),
            ("sub", measure(&a, &b, &c, &mut out, |a, b, _| a - b)),
            ("mul", measure(&a, &b, &c, &mut out, |a, b, _| a * b)),
            ("div", measure(&a, &b, &c, &mut out, |a, b, _| a / b)),
            (
                "mul_add",
                measure(&a, &b, &c, &mut out, |a, b, c| a.mul_add(b, c)),
            ),
        ]
    }};
}

fn main() {
    let single = suite!(f32, |x| x as f32);
    let double = suite!(f64, |x| x);
    let paired = suite!(Df32, Df32::from_f64);
    println!("Bulk arithmetic: {VALUES} values, preallocated output; warm reused buffers.");
    println!(
        "Median ns/value of {SAMPLES} samples, {BATCHES} batches each; includes reads/writes."
    );
    println!(
        "{:16} {:>10} {:>10} {:>10}",
        "operation", "f32", "f64", "Df32"
    );
    for (((name, f32_ns), (_, f64_ns)), (_, df32_ns)) in single.into_iter().zip(double).zip(paired)
    {
        println!("{name:16} {f32_ns:10.3} {f64_ns:10.3} {df32_ns:10.3}");
    }
}
