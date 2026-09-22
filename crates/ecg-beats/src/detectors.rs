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
    /// Fusion of a conducted and a ventricular beat.
    ///
    /// Its own class because its morphology is genuinely intermediate: forcing
    /// it into N or V asks the model to draw a line through the middle of a
    /// continuum, and EC57 scores it separately for the same reason.
    F,
    /// Not classified: poor signal, or no dominant beat established yet.
    /// Reported rather than guessed - a wrong label is worse than an absent one.
    Unknown,
}

/// What the rhythm around a beat is, when the classification depends on it.
///
/// A beat's morphology is its own evidence, but prematurity is not: it is
/// measured against the rhythm the beat sits in, and there are rhythms in which
/// it is not defined. This is the only channel through which the bank learns
/// anything outside the beat, and it is one field rather than a rhythm handle
/// so that what classification is allowed to depend on stays enumerable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BeatContext {
    /// The atria are fibrillating, as judged from intervals ending before this
    /// beat.
    ///
    /// In fibrillation there is no sinus rhythm for a beat to be early *to* and
    /// no P wave to be absent, so the two strongest pieces of supraventricular
    /// evidence are not weak here - they are undefined, and a model fitted
    /// where they meant something goes on reading them as if they still did.
    pub fibrillating: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct BeatVerdict {
    pub sample: u64,
    pub class: BeatClass,
    /// Raw detector scores, published so episode logic can weigh ambiguity
    /// instead of inheriting a decision already collapsed to a label.
    pub p_ventricular: f32,
    pub p_supraventricular: f32,
    pub p_fusion: f32,
    /// The rhythm context this verdict was decided in, kept so a consumer -
    /// and an evaluation - can tell a class that was not reported from a class
    /// that was not asked for.
    pub context: BeatContext,
    /// The morphology this beat was assigned to, or zero before assignment.
    ///
    /// Set by the pipeline after classification, so a consumer can go from a
    /// cluster a reviewer has judged to the beats that belong to it - which is
    /// the whole point of clustering them.
    pub cluster: u32,
    pub features: BeatFeatures,
}

#[derive(Debug, Clone, Copy)]
pub struct BeatBank {
    pub ventricular: BinaryDetector,
    pub supraventricular: BinaryDetector,
    pub fusion: BinaryDetector,
    /// What the supraventricular detector must clear while the atria are
    /// fibrillating. A second operating point rather than a second model,
    /// because the model is not what changed - the meaning of its inputs is.
    ///
    /// At or above 1.0 the class is not reported in fibrillation at all, which
    /// is the answer if the evidence is not merely weaker there but absent.
    /// Equal to the ordinary threshold, this whole path is inert. It is a knob
    /// with both of those as interior points so the choice can be swept.
    pub supraventricular_in_af: f32,
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
            fusion: BinaryDetector {
                name: "fusion",
                model: weights::fusion(),
                threshold: weights::FUSION_THRESHOLD,
            },
            supraventricular_in_af: weights::SUPRAVENTRICULAR_IN_AF,
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
        self.classify_in(obs, BeatContext::default())
    }

    /// The same, told what rhythm the beat sits in.
    pub fn classify_in(&self, obs: &BeatObservation, context: BeatContext) -> BeatVerdict {
        let f = &obs.features;
        let pv = self.ventricular.score(f);
        let ps = self.supraventricular.score(f);
        let pf = self.fusion.score(f);

        let class = if !obs.quality_ok || !obs.template_ready {
            BeatClass::Unknown
        } else {
            // Each detector that fires, ranked by how far it clears its own
            // threshold as a fraction of the room above it - two calibrated
            // confidences, rather than two raw probabilities that were never on
            // the same scale. Order breaks exact ties, and it is deliberate:
            // ventricular first, because missing one costs more than
            // mislabelling either of the others.
            let candidates = [
                (BeatClass::V, pv, self.ventricular.threshold),
                (BeatClass::F, pf, self.fusion.threshold),
                (BeatClass::S, ps, self.supraventricular.threshold),
            ];
            let mut best: Option<(BeatClass, f32)> = None;
            for (class, p, threshold) in candidates {
                if p < threshold {
                    continue;
                }
                let m = margin(p, threshold);
                if best.map(|(_, bm)| m > bm).unwrap_or(true) {
                    best = Some((class, m));
                }
            }
            // Arbitration runs first and the context is applied to its winner,
            // rather than the class being dropped from the ballot.
            //
            // The difference is what happens to a beat the supraventricular
            // detector had won: dropping the class hands that beat to whichever
            // detector came second, and the label it then carries was produced
            // by the absence of a rival rather than by any evidence for itself.
            // Measured, that is not a quibble - it cost two points of
            // ventricular precision on sealed MIT-BIH for no gain in
            // ventricular sensitivity, because the beats arriving were
            // aberrantly conducted ones, not ventricular ones. A suppression
            // withdraws a claim; it must not create one.
            // A bar at or above 1.0 means "not reportable here", and is
            // tested as such: a saturated probability compares equal to 1.0 and
            // would otherwise walk straight through a bar of 1.0.
            let bar = self.supraventricular_in_af;
            let suppressed = |c: BeatClass| match c {
                BeatClass::S => context.fibrillating && (bar >= 1.0 || ps < bar),
                _ => false,
            };
            match best.map(|(c, _)| c) {
                Some(c) if suppressed(c) => BeatClass::N,
                Some(c) => c,
                None => BeatClass::N,
            }
        };

        BeatVerdict {
            sample: obs.sample,
            class,
            p_ventricular: pv,
            p_supraventricular: ps,
            p_fusion: pf,
            context,
            cluster: 0,
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

    pub fn fusion() -> Model {
        if trees::FUSION.is_empty() {
            Model::Linear(FUSION)
        } else {
            Model::Gbdt(trees::FUSION)
        }
    }

    /// Generated by `ecg-eval fit-beats --gbdt --emit`.
    pub use crate::trees_generated as trees;

    #[allow(unused_imports)]
    use GbdtModel as _;

    pub const VENTRICULAR: LinearBinary = LinearBinary {
        bias: 10.123707,
        w: [
            -2.29885, -1.702003, -0.443716, 3.371255, 0.611305, -4.925861, -6.764345, 3.42227,
            -3.254282, -0.883989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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
        bias: 8.363663,
        w: [
            0.384515, 1.327991, 0.53218, -2.13749, -0.943042, 3.662803, -7.247439, 2.455288,
            -4.927748, -2.455951, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    };
    /// Higher than the ventricular threshold: a false premature atrial beat is a
    /// nuisance a reviewer pays for, and supraventricular ectopy is the class
    /// with the weaker evidence base in a single lead.
    pub const SUPRAVENTRICULAR_THRESHOLD: f32 = 0.93;

    /// What the supraventricular detector must clear while the atria are
    /// fibrillating: nothing clears it, so the class is not reported there.
    ///
    /// Not a tuning choice. A premature atrial beat is premature *to* a sinus
    /// rhythm, and in fibrillation there is none - every conducted beat arrives
    /// at an irregular time, so "early" has no referent. The measurement agrees
    /// and is blunt about it: on the training half of MIT-BIH, 893 beats inside
    /// sustained fibrillation were called supraventricular and 15 of them were,
    /// which is 98.3 % wrong.
    ///
    /// The intermediate values were swept and do not work. Raising the bar to
    /// 0.999 recovers 1.2 points of precision, because the model is not
    /// marginally wrong here but confidently wrong: it goes on reading
    /// prematurity as evidence in the one rhythm where prematurity is not
    /// defined, and it reads it with conviction. A threshold cannot fix a
    /// confident error; only declining the question can.
    ///
    /// The cost is real and falls on the supraventricular corpus, which loses
    /// 5.4 points of sensitivity where MIT-BIH loses 1.9 and gains 16.0 of
    /// precision. See `PHASE-11.md`.
    pub const SUPRAVENTRICULAR_IN_AF: f32 = 1.0;

    /// No linear fallback was fitted for fusion; the class is rare enough that
    /// a linear model on it is not worth shipping. With no trees the detector
    /// scores every beat at one half and never fires, which is the right
    /// behaviour for a model that does not exist.
    pub const FUSION: LinearBinary = LinearBinary::ZERO;
    /// Higher than either of the others. A fusion beat is a ventricular beat
    /// that a conducted one arrived in the middle of, so the cost of calling
    /// one when it is really ventricular is the cost of a missed ventricular
    /// beat - and this class has two orders of magnitude less training data
    /// than the others.
    pub const FUSION_THRESHOLD: f32 = 0.95;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::BeatFeatures;
    use crate::template::{BeatVector, TEMPLATE_LEN};

    /// A detector that reports `p` for every beat.
    fn constant(name: &'static str, p: f32) -> BinaryDetector {
        // A linear model with no weights returns sigmoid(bias), so the bias is
        // the inverse sigmoid of the score wanted.
        let z = (p / (1.0 - p)).ln();
        BinaryDetector {
            name,
            model: Model::Linear(LinearBinary {
                bias: z,
                w: [0.0; NF],
            }),
            threshold: 0.5,
        }
    }

    fn observation() -> BeatObservation {
        BeatObservation {
            sample: 0,
            features: BeatFeatures::default(),
            vector: BeatVector {
                v: [0.0; TEMPLATE_LEN],
                scale: 1.0,
            },
            quality_ok: true,
            template_ready: true,
        }
    }

    /// Withdrawing the supraventricular claim must not hand the beat to the
    /// runner-up. The label the beat would then carry was produced by the
    /// absence of a rival rather than by evidence for itself.
    #[test]
    fn a_withdrawn_claim_does_not_become_another_one() {
        let bank = BeatBank {
            // Both fire; the supraventricular one clears its threshold by the
            // larger margin, so it wins arbitration outside fibrillation.
            ventricular: constant("ventricular", 0.60),
            supraventricular: constant("supraventricular", 0.99),
            fusion: constant("fusion", 0.0),
            supraventricular_in_af: 1.0,
        };
        let obs = observation();

        assert_eq!(bank.classify(&obs).class, BeatClass::S);
        assert_eq!(
            bank.classify_in(&obs, BeatContext { fibrillating: true })
                .class,
            BeatClass::N,
            "the beat was handed to the detector that came second"
        );
    }

    /// A ventricular beat is still ventricular in fibrillation. The rhythm
    /// invalidates prematurity, which is atrial evidence; it says nothing about
    /// the morphology a ventricular beat is identified by.
    #[test]
    fn suppression_reaches_only_the_class_it_names() {
        let bank = BeatBank {
            ventricular: constant("ventricular", 0.99),
            supraventricular: constant("supraventricular", 0.60),
            fusion: constant("fusion", 0.0),
            supraventricular_in_af: 1.0,
        };
        let obs = observation();
        for fibrillating in [false, true] {
            assert_eq!(
                bank.classify_in(&obs, BeatContext { fibrillating }).class,
                BeatClass::V
            );
        }
    }

    /// A bar of exactly 1.0 must suppress. A tree ensemble confident enough
    /// saturates the probability to 1.0 in f32, and `p >= bar` would then let
    /// the class through the one bar that means "never".
    #[test]
    fn a_saturated_score_does_not_walk_through_the_bar() {
        let bank = BeatBank {
            ventricular: constant("ventricular", 0.0),
            supraventricular: constant("supraventricular", 1.0),
            fusion: constant("fusion", 0.0),
            supraventricular_in_af: 1.0,
        };
        let obs = observation();
        assert_eq!(bank.classify(&obs).class, BeatClass::S);
        assert_eq!(
            bank.classify_in(&obs, BeatContext { fibrillating: true })
                .class,
            BeatClass::N
        );
    }
}
