//! Synthetic floating-point arithmetic for embedded and GPU workloads.
//!
//! [`Df32`] represents one number as a leading `f32` and a small residual.
//! Arithmetic retains that residual; conversion back to a scalar is explicit.
//! See `design/Df32.md` for algorithms, accuracy limits, and backend requirements.
//!
//! The library is unconditionally `no_std` and does not use `alloc`.
//! The opt-in `half` feature enables nightly Rust's `f16` support for the
//! Df16 format described in `design/Df16.md`.

#![no_std]
#![cfg_attr(feature = "half", feature(f16))]

mod df32;
pub use df32::Df32;

#[cfg(feature = "half")]
mod df16;
#[cfg(feature = "half")]
pub use df16::Df16;
