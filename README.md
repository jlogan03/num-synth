# num-synth
Synthetic float types using smaller hardware width to emulate larger hardware width.

`S64I8` is a CPU reference representation with ten balanced radix-128 `i8`
digits, 68 significant bits, and an explicit `i16` exponent in 16 bytes.
It currently supports exact binary64 expansion and round-to-nearest-even
collapse, preserving signed zeros and all NaN bit patterns. Arithmetic is
not implemented yet.

```rust
use num_synth::S64I8;

let input = 1.25_f64;
let expanded = S64I8::expand(input);
assert_eq!(expanded.collapse().to_bits(), input.to_bits());
```

See [the S64I8 design](design/S64I8.md) for canonical invariants, range and
precision proofs, accumulator bounds, rounding requirements, and test coverage.

Run `cargo test` to check conversions against an independent exact integer
oracle and exercise boundary and property tests.
