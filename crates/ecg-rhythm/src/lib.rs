//! Rhythm analysis: RR intervals and atrial fibrillation.
//!
//! Consumes the detector's beats and the quality monitor's verdict; never raw
//! samples. Streaming and allocation-free like the rest of the engine.

pub mod af;
pub mod episode;
pub mod rr;

pub use af::{AfConfig, AfDetector, AfFeatures, AfWeights, AfWindow};
pub use episode::{Episode, EpisodeConfig, EpisodeTracker};
pub use rr::{RrConfig, RrSample, RrStream};
