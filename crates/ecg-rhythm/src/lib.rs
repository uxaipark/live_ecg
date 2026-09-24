//! Rhythm analysis: RR intervals and atrial fibrillation.
//!
//! Consumes the detector's beats and the quality monitor's verdict; never raw
//! samples. Streaming and allocation-free like the rest of the engine.

pub mod af;
pub mod episode;
pub mod episodes;
pub mod rr;
pub mod sv_run;
pub mod vf;

pub use af::{AfConfig, AfDetector, AfFeatures, AfWeights, AfWindow};
pub use episode::{Episode, EpisodeConfig, EpisodeTracker};
pub use episodes::{Beat, Condition, RhythmBank, RhythmConfig, RhythmEpisode};
pub use rr::{RrConfig, RrSample, RrStream};
pub use sv_run::{SvRun, SvRunConfig, SvRunDetector};
pub use vf::{VfConfig, VfDetector, VfFeatures, VfWeights, VfWindow};
