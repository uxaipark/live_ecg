//! Two atrial templates: the hypothesis, and why it is not in the engine.
//!
//! This lives in the evaluation crate rather than in `ecg-beats` because it was
//! measured and refuted. It is kept so the refutation can be re-run rather than
//! re-derived.
//!
//! # The hypothesis
//!
//! A single running template of the patient's P wave is a **worse than chance**
//! predictor of the beat's class on the patch corpus - AUC 0.398, with
//! supraventricular beats fitting it at a median of 0.419 against normal beats'
//! 0.286. The reason looked obvious. Ninety per cent of the supraventricular
//! beats there sit in runs of eight or more and the long runs are hundreds of
//! beats, so a template with a fifty-beat memory *becomes* the ectopic P wave a
//! few seconds into a run, and from then on the beats it should flag are the
//! ones that fit it best. Give the atrium what the ventricle already has - a
//! dominant shape and a rival, kept apart - and ask which of the two a beat
//! looks like.
//!
//! # What happened
//!
//! Nothing. Swept on the development zone, the margin between the two shapes
//! never rises above chance:
//!
//! | admission bar | beats with no second shape | AUC |
//! |---|---|---|
//! | 0.60 | 10,430,831 | 0.467 |
//! | 0.40 | 8,937,080 | 0.515 |
//! | 0.25 | 7,491,340 | 0.522 |
//! | 0.10 | 5,718,158 | **0.526** |
//!
//! Loosening the bar buys coverage and no information. At the loosest setting
//! the margin reads a median of -0.575 on normal beats, -0.290 on
//! supraventricular ones and -0.498 on ventricular ones, with a tenth-to-third-
//! quartile spread from -1.6 to +1.4 - very nearly the whole range the statistic
//! can take. The second cluster is not filling with a second focus; it is
//! filling with noise, which is why normal beats sit in it as readily as
//! ectopic ones.
//!
//! # Why
//!
//! A P wave on this data is about a tenth of a QRS, so a tenth of a millivolt
//! once the gain is recovered, and the shape difference between two atrial foci
//! is smaller than that again. Consecutive P waves from the *same* focus match
//! at a median of 0.758 on the development half and 0.436 on the sealed one:
//! the measurement is not stable enough to cluster. Per-beat atrial identity on
//! a single-lead patch is below the noise floor, and no arrangement of
//! templates over it recovers what was not measured.

use ecg_beats::BeatVector;

#[derive(Debug, Clone, Copy)]
pub struct AtrialTemplateConfig {
    /// Correlation at which a P wave joins a cluster rather than starting one.
    pub admit_ncc: f32,
    /// Adaptation rate within a cluster.
    pub alpha: f32,
    /// Beats a cluster needs before it is worth comparing against. Below this
    /// the margin is withheld rather than guessed.
    pub bootstrap_beats: u32,
    /// How many times more beats the rival needs before it takes the dominant
    /// slot. Above one so that the two do not trade places over noise.
    pub takeover_ratio: f32,
}

impl Default for AtrialTemplateConfig {
    fn default() -> Self {
        AtrialTemplateConfig {
            admit_ncc: 0.6,
            alpha: 0.02,
            bootstrap_beats: 16,
            takeover_ratio: 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Cluster {
    vector: BeatVector,
    accepted: u64,
}

impl Cluster {
    fn blend(&mut self, p: &BeatVector, alpha: f32) {
        for (x, y) in self.vector.v.iter_mut().zip(p.v.iter()) {
            *x += alpha * (*y - *x);
        }
        let norm = self
            .vector
            .v
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-6);
        for x in self.vector.v.iter_mut() {
            *x /= norm;
        }
        self.accepted = self.accepted.saturating_add(1);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AtrialTemplate {
    dominant: Option<Cluster>,
    rival: Option<Cluster>,
}

impl AtrialTemplate {
    pub fn new() -> Self {
        AtrialTemplate::default()
    }

    /// How much more like the rival than the dominant this P wave is.
    ///
    /// Positive means the atrium looks like the recording's second shape.
    /// `None` until both shapes have enough beats to be worth comparing, which
    /// is most of the time in a patient who has only one.
    pub fn rival_margin(&self, p: &BeatVector, cfg: &AtrialTemplateConfig) -> Option<f32> {
        let d = self
            .dominant
            .filter(|c| c.accepted >= cfg.bootstrap_beats as u64)?;
        let r = self
            .rival
            .filter(|c| c.accepted >= cfg.bootstrap_beats as u64)?;
        Some(r.vector.ncc(p) - d.vector.ncc(p))
    }

    /// Similarity to the dominant shape alone.
    pub fn dominant_ncc(&self, p: &BeatVector) -> Option<f32> {
        self.dominant.map(|c| c.vector.ncc(p))
    }

    pub fn update(&mut self, p: &BeatVector, quality_ok: bool, cfg: &AtrialTemplateConfig) {
        if !quality_ok {
            return;
        }
        let sim = |c: &Option<Cluster>| c.map(|c| c.vector.ncc(p)).unwrap_or(f32::NEG_INFINITY);
        let (sd, sr) = (sim(&self.dominant), sim(&self.rival));

        if self.dominant.is_none() {
            self.dominant = Some(Cluster {
                vector: *p,
                accepted: 1,
            });
            return;
        }
        // The better match wins, and only if it is a match at all. A wave that
        // resembles neither starts the rival over rather than dragging one of
        // them towards it - the whole point is that the two stay apart.
        if sd >= sr && sd >= cfg.admit_ncc {
            if let Some(c) = self.dominant.as_mut() {
                c.blend(p, cfg.alpha);
            }
        } else if sr >= cfg.admit_ncc {
            if let Some(c) = self.rival.as_mut() {
                c.blend(p, cfg.alpha);
            }
        } else {
            self.rival = Some(Cluster {
                vector: *p,
                accepted: 1,
            });
        }

        // Sinus is whichever shape has more beats across the whole recording,
        // not across the last minute; inside a run the ectopic P is the local
        // majority and that is exactly the capture this exists to avoid.
        if let (Some(d), Some(r)) = (self.dominant, self.rival) {
            if (r.accepted as f32) > d.accepted as f32 * cfg.takeover_ratio {
                self.dominant = Some(r);
                self.rival = Some(d);
            }
        }
    }

    pub fn reset(&mut self) {
        self.dominant = None;
        self.rival = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ecg_beats::template::TEMPLATE_LEN;

    fn wave(seed: f32) -> BeatVector {
        let mut v = [0.0f32; TEMPLATE_LEN];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f32 * 0.1) + seed).sin();
        }
        BeatVector { v, scale: 1.0 }
    }

    /// The failure this exists to prevent: a long run of one shape must not
    /// take the slot that the rest of the recording occupies.
    #[test]
    fn a_long_run_does_not_capture_the_dominant_shape() {
        let cfg = AtrialTemplateConfig::default();
        let mut t = AtrialTemplate::new();
        let (sinus, ectopic) = (wave(0.0), wave(3.1));
        for _ in 0..600 {
            t.update(&sinus, true, &cfg);
        }
        for _ in 0..200 {
            t.update(&ectopic, true, &cfg);
        }
        assert!(
            t.rival_margin(&ectopic, &cfg).unwrap() > 0.0,
            "the ectopic wave does not read as the rival"
        );
        assert!(
            t.rival_margin(&sinus, &cfg).unwrap() < 0.0,
            "the sinus wave stopped reading as the dominant"
        );
    }

    /// And the case that is not a run: when the second shape really is most of
    /// the recording, it takes the slot.
    #[test]
    fn the_shape_that_holds_the_recording_takes_the_dominant_slot() {
        let cfg = AtrialTemplateConfig::default();
        let mut t = AtrialTemplate::new();
        let (first, second) = (wave(0.0), wave(3.1));
        for _ in 0..50 {
            t.update(&first, true, &cfg);
        }
        for _ in 0..500 {
            t.update(&second, true, &cfg);
        }
        assert!(t.rival_margin(&second, &cfg).unwrap() < 0.0);
    }

    #[test]
    fn nothing_is_claimed_before_both_shapes_exist() {
        let cfg = AtrialTemplateConfig::default();
        let mut t = AtrialTemplate::new();
        let sinus = wave(0.0);
        for _ in 0..100 {
            t.update(&sinus, true, &cfg);
        }
        assert!(
            t.rival_margin(&sinus, &cfg).is_none(),
            "a patient with one P wave was given a margin against a second"
        );
    }
}
