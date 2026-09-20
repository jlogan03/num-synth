# num-synth

Fixed-width synthetic floating point for devices without native f64 arithmetic.

`S64U8` keeps a 256-bit significand in `[u8; 32]`, with an `i16` exponent,
sign, and classification. Arithmetic uses byte digits, `u16`/`u32` accumulators,
and fixed scratch arrays. The library is unconditionally `no_std` and uses only
`core`, with no production dependencies, `alloc` dependency, or global allocator
requirement. This is the runtime contract for embedded and GPU builds;
host-side tests use separate development dependencies.

```rust
use num_synth::S64U8;

let a = S64U8::from_f64(1.25);
let b = S64U8::from_f32(0.5);
let working = (a + b) * a;
let result = a.mul_add(b, working);
let output64 = result.to_f64();
let output32 = result.to_f32();
```

Operators round to the working format's 256-bit capacity. Conversion is a
separate action and rounds the stored value directly to f64 or f32. There is
no intermediate 68-bit stage or arithmetic `_f64` API. The 32-byte significand
plus metadata occupies 36 bytes; multiplication and FMA use fixed 66-byte
scratch integers to hold products and alignment information.

Implemented: exact f64/f32 expansion, nearest-even conversion, addition,
subtraction, multiplication, fused multiply-add, negation, and comparisons.
Division, square root, and GPU kernels remain future work. Throughput and
register pressure have not yet been benchmarked.

See [the S64U8 design](design/S64U8.md) for range/precision proofs, accumulator
bounds, rounding, and special-value behavior. This replaces the S64I8 prototype.
Run `cargo check --lib` to check the runtime library and `cargo test` for
arbitrary-precision reference checks.
