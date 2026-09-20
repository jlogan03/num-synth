use num_synth::Df32;
use std::{hint::black_box, time::Instant};

const VALUES: usize = 10_000;
const BATCHES: usize = 1_000;
const SAMPLES: usize = 7;

fn convert_into<T>(input: &[f64], output: &mut [T], convert: &impl Fn(f64) -> T) {
    for (&value, slot) in input.iter().zip(output) {
        *slot = convert(value);
    }
}

// All buffers are allocated by the caller. Black-box whole slices at batch
// boundaries, allowing the compiler to optimize/vectorize the inner loop.
fn measure<T>(input: &[f64], output: &mut [T], convert: impl Fn(f64) -> T) -> f64 {
    assert_eq!(input.len(), output.len());
    for _ in 0..32 {
        convert_into(black_box(input), black_box(&mut *output), &convert);
        black_box(&*output);
    }

    let mut samples = [0.0; SAMPLES];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..BATCHES {
            convert_into(black_box(input), black_box(&mut *output), &convert);
            // Keep each batch's stores observable, not just the last batch.
            black_box(&*output);
        }
        *sample = start.elapsed().as_nanos() as f64 / (BATCHES * input.len()) as f64;
    }
    samples.sort_unstable_by(f64::total_cmp);
    samples[SAMPLES / 2]
}

fn main() {
    // Deterministic finite inputs, with both signs and detail below f32
    // precision. Data generation and every allocation are outside timing.
    let mut state = 0x1234_5678_9abc_def0u64;
    let input: Vec<f64> = (0..VALUES)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            (2.0 * unit - 1.0) * 1e6
        })
        .collect();
    let mut single = vec![0.0f32; VALUES];
    let mut paired = vec![Df32::ZERO; VALUES];
    let mut copied = vec![0.0f64; VALUES];

    let single_ns = measure(&input, &mut single, |x| x as f32);
    let paired_ns = measure(&input, &mut paired, Df32::from_f64);
    let copied_ns = measure(&input, &mut copied, |x| x);

    println!("Bulk conversion: {VALUES} finite f64 values in [-1e6, +1e6].");
    println!("Preallocated slices, reused after warmup; includes reads and writes.");
    println!(
        "Median of {SAMPLES} samples of {BATCHES} batches; excludes allocation and generation."
    );
    println!(
        "{:20} {:>12} {:>12} {:>16}",
        "conversion", "ns/value", "us/batch", "Mvalues/second"
    );
    for (name, ns) in [
        ("f64 -> f32", single_ns),
        ("f64 -> Df32", paired_ns),
        ("f64 -> f64 (copy)", copied_ns),
    ] {
        println!(
            "{name:20} {ns:12.3} {:12.3} {:16.2}",
            ns * VALUES as f64 / 1e3,
            1e3 / ns
        );
    }
}
