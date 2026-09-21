//! A bank of independent binary beat detectors.
//!
//! # Why a bank and not one multi-class model
//!
//! The conditions this engine has to report do not share an evidence base or a
//! cost function. Atrial fibrillation is decided from timing alone; a
//! ventricular beat from morphology; a pause from an interval; ventricular
//! tachycardia from all three plus duration. Folding them into one classifier
//! forces a single shared representation onto questions that are not the same
//! question.
//!
//! The decisive reason is the operating point. A missed ventricular run can
//! kill; a false premature atrial beat is a nuisance that costs a reviewer ten
//! seconds. Those call for thresholds on opposite sides, and a single softmax
//! with one `argmax` cannot express two different cost asymmetries at once.
//! Separate detectors each carry their own threshold, chosen against their own
//! clinical cost, and each can be revalidated or replaced without disturbing the
//! others.
//!
//! What is *not* separate is the feature layer. Morphology and interval context
//! are computed once per beat and shared, so the bank costs one feature
//! extraction plus a dot product per detector.
//!
//! # Where exclusivity is real
//!
//! One beat has one AAMI label, and the standard the field reports against
//! (ANSI/AAMI EC57) wants a confusion matrix over them. So the bank publishes
//! both: the per-detector scores, which is what downstream episode logic should
//! consume, and an arbitrated label for EC57 scoring. The arbitration rule is
//! explicit and lives in [`BeatBank::classify`] rather than being implied by an
//! `argmax` no one can inspect.

use crate::features::{BeatFeatures, BeatObservation, NF};
use crate::gbdt::GbdtModel;

/// A linear binary model over the shared feature vector.
#[derive(Debug, Clone, Copy)]
pub struct LinearBinary {
    pub bias: f32,
    pub w: [f32; NF],
}

impl LinearBinary {
    pub const ZERO: LinearBinary = LinearBinary {
        bias: 0e+00,
        w: [0.0; NF],
    };

    #[inline]
    pub fn probability(&self, f: &BeatFeatures) -> f32 {
        let v = f.vector();
        let mut z = self.bias;
        for (wi, vi) in self.w.iter().zip(v.iter()) {
            z += wi * vi;
        }
        1.0 / (1.0 + (-z).exp())
    }
}

/// What a detector scores with. Both forms are kept so the choice can be
/// justified by measurement rather than by preference.
#[derive(Debug, Clone, Copy)]
pub enum Model {
    Linear(LinearBinary),
    Gbdt(GbdtModel),
}

impl Model {
    #[inline]
    pub fn probability(&self, f: &BeatFeatures) -> f32 {
        match self {
            Model::Linear(m) => m.probability(f),
            Model::Gbdt(m) => m.probability(&f.vector()),
        }
    }
}

/// One condition, one score, one operating point.
#[derive(Debug, Clone, Copy)]
pub struct BinaryDetector {
    pub name: &'static str,
    pub model: Model,
    /// Score at or above which the condition is reported.
    pub threshold: f32,
}

impl BinaryDetector {
    #[inline]
    pub fn score(&self, f: &BeatFeatures) -> f32 {
        self.model.probability(f)
    }

    #[inline]
    pub fn fires(&self, f: &BeatFeatures) -> bool {
        self.score(f) >= self.threshold
    }
}

/// AAMI beat classes, plus an explicit "not judged".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeatClass {
    /// Normal or bundle-branch block beat.
    N,
    /// Supraventricular ectopic.
    S,
    /// Ventricular ectopic.
    V,
    /// Not classified: poor signal, or no dominant beat established yet.
    /// Reported rather than guessed - a wrong label is worse than an absent one.
    Unknown,
}

#[derive(Debug, Clone, Copy)]
pub struct BeatVerdict {
    pub sample: u64,
    pub class: BeatClass,
    /// Raw detector scores, published so episode logic can weigh ambiguity
    /// instead of inheriting a decision already collapsed to a label.
    pub p_ventricular: f32,
    pub p_supraventricular: f32,
    pub features: BeatFeatures,
}

#[derive(Debug, Clone, Copy)]
pub struct BeatBank {
    pub ventricular: BinaryDetector,
    pub supraventricular: BinaryDetector,
}

impl Default for BeatBank {
    fn default() -> Self {
        BeatBank {
            ventricular: BinaryDetector {
                name: "ventricular",
                model: weights::ventricular(),
                threshold: weights::VENTRICULAR_THRESHOLD,
            },
            supraventricular: BinaryDetector {
                name: "supraventricular",
                model: weights::supraventricular(),
                threshold: weights::SUPRAVENTRICULAR_THRESHOLD,
            },
        }
    }
}

impl BeatBank {
    /// Run every detector and arbitrate to a single AAMI label.
    ///
    /// Arbitration when both fire: the detector whose score clears its own
    /// threshold by the larger relative margin wins, so the comparison is
    /// between two calibrated confidences rather than between two raw
    /// probabilities that were never on the same scale. A ventricular beat wins
    /// an exact tie, because missing one costs more than mislabelling a
    /// supraventricular beat as ventricular.
    pub fn classify(&self, obs: &BeatObservation) -> BeatVerdict {
        let f = &obs.features;
        let pv = self.ventricular.score(f);
        let ps = self.supraventricular.score(f);

        let class = if !obs.quality_ok || !obs.template_ready {
            BeatClass::Unknown
        } else {
            let v_fires = pv >= self.ventricular.threshold;
            let s_fires = ps >= self.supraventricular.threshold;
            match (v_fires, s_fires) {
                (true, true) => {
                    let mv = margin(pv, self.ventricular.threshold);
                    let ms = margin(ps, self.supraventricular.threshold);
                    if ms > mv {
                        BeatClass::S
                    } else {
                        BeatClass::V
                    }
                }
                (true, false) => BeatClass::V,
                (false, true) => BeatClass::S,
                (false, false) => BeatClass::N,
            }
        };

        BeatVerdict {
            sample: obs.sample,
            class,
            p_ventricular: pv,
            p_supraventricular: ps,
            features: *f,
        }
    }
}

/// How far a score clears its threshold, as a fraction of the room available.
#[inline]
fn margin(p: f32, threshold: f32) -> f32 {
    let room = (1.0 - threshold).max(1e-6);
    (p - threshold) / room
}

/// Fitted models. Replaced by `ecg-eval fit-beats` on the TRAIN zone.
pub mod weights {
    use super::{LinearBinary, Model};
    use crate::gbdt::GbdtModel;

    /// The tree ensembles, when fitted; otherwise the linear fallback.
    pub fn ventricular() -> Model {
        if trees::VENTRICULAR.is_empty() {
            Model::Linear(VENTRICULAR)
        } else {
            Model::Gbdt(trees::VENTRICULAR)
        }
    }

    pub fn supraventricular() -> Model {
        if trees::SUPRAVENTRICULAR.is_empty() {
            Model::Linear(SUPRAVENTRICULAR)
        } else {
            Model::Gbdt(trees::SUPRAVENTRICULAR)
        }
    }

    /// Generated by `ecg-eval fit-beats --gbdt --emit`.
    pub use crate::trees_generated as trees;

    #[allow(unused_imports)]
    use GbdtModel as _;

    pub const VENTRICULAR: LinearBinary = LinearBinary {
        bias: 9.976564,
        w: [
            -1.463106, -2.047411, -0.445538, 2.25274, 0.822954, -4.383169, -7.06313, 3.406498,
            -3.039153, -0.768126,
        ],
    };
    /// Set below the supraventricular threshold on purpose. A missed ventricular
    /// beat costs more than a mislabelled one, and this is the knob that lets
    /// that asymmetry be expressed at all - a single multi-class `argmax` has
    /// nowhere to put it. On this corpus the trees are confident enough that the
    /// curve is flat between 0.5 and 0.97 (sensitivity 87.9% to 87.1%,
    /// precision 59.9% to 66.1%), so the choice is cheap here; on a noisier
    /// deployment it will not be.
    pub const VENTRICULAR_THRESHOLD: f32 = 0.85;

    pub const SUPRAVENTRICULAR: LinearBinary = LinearBinary {
        bias: 7.580941,
        w: [
            0.548726, 1.715293, -0.169099, 1.067046, 0.85265, -0.59709, -6.825176, 1.963676,
            -4.627464, -2.497577,
        ],
    };
    /// Higher than the ventricular threshold: a false premature atrial beat is a
    /// nuisance a reviewer pays for, and supraventricular ectopy is the class
    /// with the weaker evidence base in a single lead.
    pub const SUPRAVENTRICULAR_THRESHOLD: f32 = 0.93;
}
