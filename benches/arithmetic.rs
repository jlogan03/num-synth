use num_synth::Df32;
use std::{hint::black_box, time::Instant};

fn bench<T: Copy>(name: &str, mut state: T, mut op: impl FnMut(T) -> T) {
    const N: usize = 200_000;
    let start = Instant::now();
    for _ in 0..N {
        state = black_box(op(black_box(state)));
    }
    println!(
        "{name:30} {:8.2} ns/iteration",
        start.elapsed().as_nanos() as f64 / N as f64
    );
    black_box(state);
}

fn main() {
    let a = Df32::from_f64(1.00000001);
    let b = Df32::from_f64(0.99999999);
    let c = Df32::from_f32(1e-7);
    bench("Df32 add chain", Df32::ONE, |x| x + c);
    bench("Df32 mul chain", Df32::ONE, |x| x * a);
    bench("Df32 div chain", Df32::ONE, |x| x / a);
    bench("Df32 mul_add chain", Df32::ONE, |x| x.mul_add(b, c));
    bench("Df32 separate mul/add chain", Df32::ONE, |x| x * b + c);
    bench("f32 mul_add chain", 1f32, |x| {
        x.mul_add(black_box(b.to_f32()), black_box(c.to_f32()))
    });
    bench("f64 mul_add chain", 1f64, |x| {
        x.mul_add(black_box(b.to_f64()), black_box(c.to_f64()))
    });
    bench("Df32 independent mul_add x4", [Df32::ONE; 4], |x| {
        x.map(|v| v.mul_add(b, c))
    });
    let inverse = Df32::from_f64(1.0 / 3.25);
    bench(
        "Df32 position round trip",
        Df32::from_parts(10.0, 1e-6),
        |x| (x.mul_add(Df32::from_f32(3.25), a) - a) * inverse,
    );
    bench("Df32 separate div remainder", Df32::ONE, |x| {
        let q = Df32::from_f32(x.to_f32() / a.to_f32());
        let r = x - a * q;
        q + Df32::from_f32(r.to_f32() / a.to_f32())
    });
    bench("Df32 fused div remainder", Df32::ONE, |x| {
        let q = Df32::from_f32(x.to_f32() / a.to_f32());
        let r = (-a).mul_add(q, x);
        q + Df32::from_f32(r.to_f32() / a.to_f32())
    });
}
