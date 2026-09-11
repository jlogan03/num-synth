//! Synthetic floating-point representations using narrow integer digits.
//!
//! [`S64I8`] provides exact binary64 expansion and correctly rounded collapse.
//! Arithmetic is deferred until its rounding contract is established.

mod s64i8;

pub use s64i8::{Class, InvalidFiniteParts, S64I8};
