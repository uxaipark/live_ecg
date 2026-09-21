//! Running template of the patient's dominant beat.
//!
//! Almost every morphology judgement a single lead can make is relative: there
//! is no absolute shape that means "ventricular", only a shape that differs from
//! *this patient's* conducted beat. The template is that reference, learned
//! online and kept away from the beats it is meant to judge.
//!
//! Beats are resampled onto a fixed-length grid spanning a fixed duration, so a
//! template learned at 128 Hz and one learned at 360 Hz are the same object and
//! every feature derived from it is sampling-rate independent.

use ecg_dsp::Ring;

/// Samples in the resampled beat vector.
pub const TEMPLATE_LEN: usize = 48;
/// Window around the R peak, milliseconds.
pub const WINDOW_BEFORE_MS: f64 = 120.0;
pub const WINDOW_AFTER_MS: f64 = 200.0;

#[derive(Debug, Clone, Copy)]
pub struct BeatVector {
    pub v: [f32; TEMPLATE_LEN],
    /// Peak-to-peak of the window before normalisation, mV.
    pub scale: f32,
}

impl BeatVector {
    /// Zero-mean, unit-norm form; `scale` keeps the amplitude that was divided out.
    fn normalise(mut raw: [f32; TEMPLATE_LEN]) -> BeatVector {
        let mean = raw.iter().sum::<f32>() / TEMPLATE_LEN as f32;
        for x in raw.iter_mut() {
            *x -= mean;
        }
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = norm.max(1e-6);
        for x in raw.iter_mut() {
            *x /= scale;
        }
        BeatVector { v: raw, scale }
    }

    /// Normalised cross-correlation with another beat, in [-1, 1].
    #[inline]
    pub fn ncc(&self, other: &BeatVector) -> f32 {
        self.v
            .iter()
            .zip(other.v.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            .clamp(-1.0, 1.0)
    }
}

/// Extract the beat window from a signal ring and resample it to fixed length.
pub fn extract(ring: &Ring, r_sample: u64, now: u64, fs: f64, floor: u64) -> Option<BeatVector> {
    let before = (WINDOW_BEFORE_MS * fs / 1000.0) as u64;
    let after = (WINDOW_AFTER_MS * fs / 1000.0) as u64;
    extract_span(
        ring,
        r_sample.checked_sub(before)?,
        r_sample + after,
        now,
        floor,
    )
}

/// Resample an arbitrary span onto the same fixed grid.
///
/// Resampling rather than taking a fixed number of samples is what makes a
/// window whose length depends on the heart rate comparable with one taken at
/// another rate: the P wave before a beat at 50 per minute and the same wave at
/// 110 occupy different numbers of samples and the same fraction of the grid.
pub fn extract_span(ring: &Ring, start: u64, end: u64, now: u64, floor: u64) -> Option<BeatVector> {
    if end <= start + 2 || start < floor || end >= now {
        return None;
    }
    let span = (end - start) as f32;
    let mut raw = [0.0f32; TEMPLATE_LEN];
    for (k, slot) in raw.iter_mut().enumerate() {
        // Linear interpolation onto the fixed grid.
        let pos = start as f32 + span * k as f32 / (TEMPLATE_LEN - 1) as f32;
        let i = pos.floor() as u64;
        let frac = pos - i as f32;
        *slot = ring.at(i) * (1.0 - frac) + ring.at(i + 1) * frac;
    }
    Some(BeatVector::normalise(raw))
}

#[derive(Debug, Clone, Copy)]
pub struct TemplateConfig {
    /// A beat must match this well to be folded into the template.
    pub admit_ncc: f32,
    /// Update rate once established.
    pub alpha: f32,
    /// Beats accepted before the template is considered established.
    pub bootstrap_beats: u32,
    /// Consecutive beats that may fail to match before the template re-anchors.
    ///
    /// Chosen on held-out training records: classification accuracy is flat from
    /// 12 to 75, while coverage rises to about 50 and then plateaus, so the
    /// value is set where the most beats get classified at no cost in accuracy.
    ///
    /// Without this the template is an absorbing state. It admits only beats
    /// that already resemble it, which is what stops ectopy from polluting it -
    /// and also means that if it anchors on a bad first beat, nothing can ever
    /// match and it is stuck for the rest of the recording. Measured on the
    /// Long-Term AF corpus, that left 0.0% of beats classified across 1,960
    /// hours: every beat came back `Unknown`, and every morphology-dependent
    /// rhythm detector went silent with it.
    pub reanchor_after: u32,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        TemplateConfig {
            admit_ncc: 0.90,
            alpha: 0.05,
            bootstrap_beats: 8,
            reanchor_after: 50,
        }
    }
}

/// The dominant morphology, plus the amplitude and width scale that go with it.
#[derive(Debug, Clone)]
pub struct Template {
    cfg: TemplateConfig,
    vector: Option<BeatVector>,
    accepted: u32,
    /// Consecutive beats that did not match.
    missed: u32,
    /// Running medians of the accepted beats' amplitude, area and width.
    pub amplitude: f32,
    pub area: f32,
    pub width: f32,
    pub slope: f32,
}

impl Template {
    pub fn new(cfg: TemplateConfig) -> Self {
        Template {
            cfg,
            vector: None,
            accepted: 0,
            missed: 0,
            amplitude: 0.0,
            area: 0.0,
            width: 0.0,
            slope: 0.0,
        }
    }

    pub fn vector(&self) -> Option<&BeatVector> {
        self.vector.as_ref()
    }

    pub fn established(&self) -> bool {
        self.accepted >= self.cfg.bootstrap_beats
    }

    /// Similarity of `b` to the dominant beat. `None` until a template exists.
    pub fn similarity(&self, b: &BeatVector) -> Option<f32> {
        self.vector.as_ref().map(|t| t.ncc(b))
    }

    /// Fold a beat in, but only if it already looks like the dominant one.
    ///
    /// The gate is what keeps this useful: a template that learns from every
    /// beat drifts toward whatever is most frequent, and in a patient with
    /// frequent ectopy that is partly the ectopy itself - after which the
    /// feature that is supposed to flag ectopic beats no longer can.
    pub fn update(
        &mut self,
        b: &BeatVector,
        amplitude: f32,
        area: f32,
        width: f32,
        slope: f32,
        quality_ok: bool,
    ) {
        if !quality_ok {
            return;
        }
        let admit = match self.similarity(b) {
            None => true,
            Some(ncc) => ncc >= self.cfg.admit_ncc,
        };
        if !admit {
            self.missed = self.missed.saturating_add(1);
            if self.missed < self.cfg.reanchor_after {
                return;
            }
            // Nothing has matched for long enough that the template is more
            // likely wrong than the beats are. Start again from the current one.
            self.vector = None;
            self.accepted = 0;
        }
        self.missed = 0;
        match self.vector.as_mut() {
            None => {
                self.vector = Some(*b);
                self.amplitude = amplitude;
                self.area = area;
                self.width = width;
                self.slope = slope;
            }
            Some(t) => {
                let a = self.cfg.alpha;
                for (x, y) in t.v.iter_mut().zip(b.v.iter()) {
                    *x += a * (*y - *x);
                }
                // Renormalise so the template stays a unit vector.
                let norm = t.v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                for x in t.v.iter_mut() {
                    *x /= norm;
                }
                self.amplitude += a * (amplitude - self.amplitude);
                self.area += a * (area - self.area);
                self.width += a * (width - self.width);
                self.slope += a * (slope - self.slope);
            }
        }
        self.accepted = self.accepted.saturating_add(1);
    }

    pub fn reset(&mut self) {
        self.vector = None;
        self.accepted = 0;
        self.missed = 0;
        self.amplitude = 0.0;
        self.area = 0.0;
        self.width = 0.0;
        self.slope = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(shift: usize) -> BeatVector {
        let mut raw = [0.0f32; TEMPLATE_LEN];
        raw[shift % TEMPLATE_LEN] = 1.0;
        raw[(shift + 1) % TEMPLATE_LEN] = -1.0;
        BeatVector::normalise(raw)
    }

    /// A template anchored on an unrepresentative beat must be able to recover.
    /// It admits only beats that already resemble it, so without a re-anchor it
    /// stays wrong for the rest of the recording and every beat after it is
    /// returned unclassified.
    #[test]
    fn a_template_anchored_on_a_bad_beat_recovers() {
        // Explicit threshold: this tests the mechanism, not the tuned default.
        let cfg = TemplateConfig {
            reanchor_after: 12,
            ..TemplateConfig::default()
        };
        let mut t = Template::new(cfg);
        t.update(&vector(0), 1.0, 1.0, 100.0, 1.0, true); // anchors here
        assert!(t.vector().is_some());

        // A long run of a completely different, self-consistent morphology.
        let good = vector(20);
        for _ in 0..40 {
            t.update(&good, 1.0, 1.0, 100.0, 1.0, true);
        }
        assert!(t.established(), "template never re-anchored");
        assert!(
            t.similarity(&good).unwrap() > 0.9,
            "re-anchored template does not match the dominant beat"
        );
    }

    /// The re-anchor must not undo what the gate is for: isolated odd beats
    /// still have to be kept out.
    #[test]
    fn isolated_ectopy_does_not_reanchor_the_template() {
        let cfg = TemplateConfig {
            reanchor_after: 12,
            ..TemplateConfig::default()
        };
        let mut t = Template::new(cfg);
        let normal = vector(0);
        let ectopic = vector(20);
        for _ in 0..20 {
            t.update(&normal, 1.0, 1.0, 100.0, 1.0, true);
        }
        for i in 0..60 {
            // One ectopic beat every fourth beat, as in bigeminy or trigeminy.
            t.update(
                if i % 4 == 0 { &ectopic } else { &normal },
                1.0,
                1.0,
                100.0,
                1.0,
                true,
            );
        }
        assert!(
            t.similarity(&normal).unwrap() > 0.9,
            "template drifted onto the ectopic morphology"
        );
    }
}

/// A running template of a waveform's *shape* alone.
///
/// The same self-selecting gate as [`Template`], without the amplitude, area
/// and width a QRS complex carries and a P wave does not. Used for the atrial
/// segment, where the question is not how big the deflection is - that measure
/// failed, separating conducted beats from ventricular ones at an AUC of 0.48 -
/// but whether it has this patient's own atrial shape.
#[derive(Debug, Clone)]
pub struct ShapeTemplate {
    cfg: TemplateConfig,
    vector: Option<BeatVector>,
    accepted: u32,
    missed: u32,
}

impl ShapeTemplate {
    pub fn new(cfg: TemplateConfig) -> Self {
        ShapeTemplate {
            cfg,
            vector: None,
            accepted: 0,
            missed: 0,
        }
    }

    pub fn established(&self) -> bool {
        self.accepted >= self.cfg.bootstrap_beats
    }

    pub fn similarity(&self, b: &BeatVector) -> Option<f32> {
        self.vector.as_ref().map(|t| t.ncc(b))
    }

    pub fn update(&mut self, b: &BeatVector, quality_ok: bool) {
        if !quality_ok {
            return;
        }
        let admit = match self.similarity(b) {
            None => true,
            Some(ncc) => ncc >= self.cfg.admit_ncc,
        };
        if !admit {
            self.missed = self.missed.saturating_add(1);
            if self.missed < self.cfg.reanchor_after {
                return;
            }
            // Same absorbing state, same answer: a template nothing has matched
            // for long enough is more likely wrong than the beats are.
            self.vector = None;
            self.accepted = 0;
        }
        self.missed = 0;
        match self.vector.as_mut() {
            None => self.vector = Some(*b),
            Some(t) => {
                let a = self.cfg.alpha;
                for (x, y) in t.v.iter_mut().zip(b.v.iter()) {
                    *x += a * (*y - *x);
                }
                let norm = t.v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                for x in t.v.iter_mut() {
                    *x /= norm;
                }
            }
        }
        self.accepted = self.accepted.saturating_add(1);
    }

    pub fn reset(&mut self) {
        self.vector = None;
        self.accepted = 0;
        self.missed = 0;
    }
}
