//! Fixed-width synthetic floating point using small-width integer arithmetic.
//!
//! [`S64U8`] has 256-bit working precision in 32 unsigned bytes. Arithmetic
//! stays in this format; explicit conversions round to binary64 or binary32.
//!
//! The library uses only `core`: it does not link `std` or `alloc` and requires
//! no global allocator. All runtime storage is fixed-size. Host-side tests use
//! separate development dependencies for arbitrary-precision reference checks.

#![no_std]

mod s64u8;
pub use s64u8::{Class, InvalidFiniteParts, S64U8};
