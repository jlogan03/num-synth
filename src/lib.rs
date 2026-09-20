//! Synthetic floating-point arithmetic for embedded and GPU workloads.
//!
//! [`Df32`] represents one number as a leading `f32` and a small residual.
//! Arithmetic retains that residual; conversion back to a scalar is explicit.
//! See `design/Df32.md` for algorithms, accuracy limits, and backend requirements.
//!
//! The library is unconditionally `no_std` and does not use `alloc`.

#![no_std]

mod df32;
pub use df32::Df32;
