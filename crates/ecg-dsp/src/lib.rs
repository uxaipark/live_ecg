//! Streaming DSP primitives for the live ECG engine.
//!
//! Everything here is causal, fixed-latency and allocation-free after
//! construction: the same code runs in the server's per-channel worker and on
//! the patch-side host (Raspberry Pi 5, phone, low-end PC).

mod biquad;
mod window;

pub use biquad::{butterworth_qs, Biquad, Cascade};
pub use window::{Derivative5, Ewma, MovingAverage, MovingExtrema, MovingPower, Ring};

/// Convert a duration in milliseconds to a sample count at `fs`, at least 1.
#[inline]
pub fn ms_to_samples(fs: f64, ms: f64) -> usize {
    ((fs * ms / 1000.0).round() as usize).max(1)
}
