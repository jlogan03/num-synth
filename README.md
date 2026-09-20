# num-synth

Synthetic floating-point arithmetic for embedded and GPU workloads.

`Df32` represents one number as the sum of two `f32` values:
a leading value and a small residual. Its purpose is to retain fine detail
during computation, such as micron-scale relative positions inside a
10-meter structure, while using native single-precision arithmetic.

The design prioritizes fixed storage, predictable operation sequences, and
consistent numerical behavior. It targets roughly doubled single-precision
accuracy in ordinary working ranges, without promising binary64 precision
or range everywhere. The runtime must remain `no_std` and allocation-free.

See [the Df32 design](design/Df32.md) for the representation, implemented
arithmetic, precision limits, and validation plan.

```rust
use num_synth::Df32;

let origin = Df32::from_f32(10.0);
let offset = Df32::from_f32(1e-6);
let position = origin + offset;
assert_eq!(position - origin, offset);

let scale = Df32::from_f64(3.25);
let transformed = position.mul_add(scale, Df32::from_f32(2.0));
let (hi, lo) = transformed.to_parts(); // Keep both components for transport.
let output = transformed.to_f64();    // Explicit scalar conversion.
```

Implemented: normalization, scalar conversions, comparisons, addition,
subtraction, multiplication, division, and combined multiply-add. `mul_add`
retains product residuals through cancellation; it does not promise correctly
rounded evaluation of every exact pair-valued expression.

The runtime uses `num-traits` and `libm`, with no `std` or `alloc` requirement.
`Zero`, `One`, `MulAdd`, and `MulAddAssign` are implemented. Scalar FMA uses
libm's hardware paths or software fallback; the latter can be expensive on
devices without f64. GPU primitive integration and execution testing remain
future work. No GPU throughput claim follows from no-std compatibility.

Run `cargo test` for exact-reference properties and directed tests, and
`cargo bench --bench arithmetic` for CPU measurements. Embedded and WebAssembly
compilation can be checked with `cargo check --lib --target thumbv7em-none-eabihf`
and `cargo check --lib --target wasm32-unknown-unknown`.
