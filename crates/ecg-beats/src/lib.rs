//! Beat classification: a shared feature layer under a bank of independent
//! binary detectors.
//!
//! Consumes the detector's beats, the analysis signal and the quality verdict.
//! Streaming and allocation-free after construction, like the rest of the
//! engine. See [`detectors`] for why the bank is shaped this way.

pub mod detectors;
pub mod features;
pub mod gbdt;
pub mod template;
pub mod trees_generated;

pub use detectors::{BeatBank, BeatClass, BeatVerdict, BinaryDetector, LinearBinary, Model};
pub use features::{BeatAnalyzer, BeatConfig, BeatFeatures, BeatObservation, NF};
pub use gbdt::{GbdtModel, Node};
pub use template::{BeatVector, Template, TemplateConfig};
