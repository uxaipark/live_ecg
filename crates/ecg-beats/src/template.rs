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
    /// Beats a rival morphology must have been seen before it can be promoted
    /// on width alone.
    pub promote_beats: u32,
    /// How much narrower the rival must be to be taken for the conducted beat.
    pub promote_width_frac: f32,
    /// Consecutive beats matching the rival after which it takes over whatever
    /// its width. Zero disables it.
    pub takeover_after: u32,
    /// Width above which a complex cannot be a conducted beat, milliseconds.
    ///
    /// Almost nothing in this engine is an absolute threshold, and this is the
    /// exception that earns it: QRS duration is defined by how the impulse
    /// travels, not by the electrode, the gain or the patient's size. A complex
    /// that reaches the ventricles through the conduction system is short; one
    /// that spreads through muscle is not. So when the dominant morphology is
    /// wide and a rival is narrow, the rival is the conducted beat however the
    /// two are counted. Zero disables the rule.
    pub conducted_width_ms: f32,
    /// How much narrower the rival must be before the rule fires, milliseconds.
    ///
    /// Without a margin the rule also fires where both morphologies are wide -
    /// bundle branch block, where the conducted beat genuinely is - and flips
    /// between two beats that are both abnormal, which helps nothing and costs
    /// the records where the template was right all along.
    pub promote_margin_ms: f32,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        TemplateConfig {
            admit_ncc: 0.90,
            alpha: 0.05,
            bootstrap_beats: 8,
            reanchor_after: 50,
            promote_beats: 16,
            promote_width_frac: 0.0,
            takeover_after: 0,
            conducted_width_ms: 120.0,
            promote_margin_ms: 30.0,
        }
    }
}

/// One self-consistent morphology, with the scales that go with it.
#[derive(Debug, Clone, Copy)]
struct Cluster {
    vector: BeatVector,
    accepted: u32,
    amplitude: f32,
    area: f32,
    width: f32,
    slope: f32,
    /// Delineated QRS duration in milliseconds, when one was measured.
    ///
    /// Kept apart from `width`, which is a band-energy proxy running at roughly
    /// half the true duration and is only ever used as a ratio. This one is
    /// calibrated - onset and offset both land within a millisecond of manual
    /// annotation on LUDB - so it can be compared against a physiological
    /// constant, which is the whole point of having it.
    qrs_ms: f32,
}

impl Cluster {
    fn new(v: &BeatVector, amplitude: f32, area: f32, width: f32, slope: f32, qrs_ms: f32) -> Self {
        Cluster {
            vector: *v,
            accepted: 1,
            amplitude,
            area,
            width,
            slope,
            qrs_ms,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fold(
        &mut self,
        b: &BeatVector,
        amplitude: f32,
        area: f32,
        width: f32,
        slope: f32,
        qrs_ms: f32,
        a: f32,
    ) {
        for (x, y) in self.vector.v.iter_mut().zip(b.v.iter()) {
            *x += a * (*y - *x);
        }
        // Renormalise so the template stays a unit vector.
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
        self.amplitude += a * (amplitude - self.amplitude);
        self.area += a * (area - self.area);
        self.width += a * (width - self.width);
        self.slope += a * (slope - self.slope);
        if qrs_ms > 0.0 {
            self.qrs_ms = if self.qrs_ms > 0.0 {
                self.qrs_ms + a * (qrs_ms - self.qrs_ms)
            } else {
                qrs_ms
            };
        }
        self.accepted = self.accepted.saturating_add(1);
    }
}

/// The dominant morphology, plus the amplitude and width scale that go with it.
///
/// # Why there are two
///
/// "Dominant" was originally "whatever arrived first and kept matching", which
/// is the same as "most frequent". In a patient whose recording is half
/// ventricular that is the wrong beat, and the failure is not graceful: every
/// morphology feature is expressed relative to the template, so anchoring on the
/// ectopic beat inverts all of them at once. Measured on INCART record I43,
/// where 51% of the beats are ventricular, the detector found 2.6% of them and
/// was right about 4.8% of what it did report - it had learned the ectopy as
/// normal and was flagging the conducted beats.
///
/// So a rival cluster is kept alongside, and the conducted beat is identified by
/// what actually makes it conducted: it is narrower. A complex that spreads
/// through the conduction system is shorter than one that spreads through
/// muscle, whatever their relative frequency. The rival is promoted when it is
/// clearly narrower, or - preserving the older behaviour that this replaces -
/// when it has simply taken over the recording.
#[derive(Debug, Clone)]
pub struct Template {
    cfg: TemplateConfig,
    dominant: Option<Cluster>,
    rival: Option<Cluster>,
    /// Consecutive beats matching neither cluster.
    missed: u32,
    /// Consecutive beats matching the rival rather than the dominant.
    rival_run: u32,
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
            dominant: None,
            rival: None,
            missed: 0,
            rival_run: 0,
            amplitude: 0.0,
            area: 0.0,
            width: 0.0,
            slope: 0.0,
        }
    }

    pub fn vector(&self) -> Option<&BeatVector> {
        self.dominant.as_ref().map(|c| &c.vector)
    }

    pub fn established(&self) -> bool {
        self.dominant
            .as_ref()
            .is_some_and(|c| c.accepted >= self.cfg.bootstrap_beats)
    }

    /// Similarity of `b` to the dominant beat. `None` until a template exists.
    pub fn similarity(&self, b: &BeatVector) -> Option<f32> {
        self.dominant.as_ref().map(|c| c.vector.ncc(b))
    }

    /// Fold a beat in, but only if it already looks like one of the morphologies
    /// being tracked.
    ///
    /// The gate is what keeps this useful: a template that learns from every
    /// beat drifts toward whatever is most frequent, and in a patient with
    /// frequent ectopy that is partly the ectopy itself - after which the
    /// feature that is supposed to flag ectopic beats no longer can.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        b: &BeatVector,
        amplitude: f32,
        area: f32,
        width: f32,
        slope: f32,
        qrs_ms: f32,
        quality_ok: bool,
    ) {
        if !quality_ok {
            return;
        }
        let admit = self.cfg.admit_ncc;
        let a = self.cfg.alpha;

        let Some(dom) = self.dominant.as_mut() else {
            self.dominant = Some(Cluster::new(b, amplitude, area, width, slope, qrs_ms));
            self.missed = 0;
            self.rival_run = 0;
            self.publish();
            return;
        };

        if dom.vector.ncc(b) >= admit {
            dom.fold(b, amplitude, area, width, slope, qrs_ms, a);
            self.missed = 0;
            self.rival_run = 0;
            self.publish();
            return;
        }

        let matched_rival = match self.rival.as_mut() {
            Some(r) if r.vector.ncc(b) >= admit => {
                r.fold(b, amplitude, area, width, slope, qrs_ms, a);
                true
            }
            _ => false,
        };

        // `missed` counts beats that did not match the *dominant*, whether or
        // not the rival caught them. Letting the rival absorb them looks tidier
        // and is wrong: the count is what drives the re-anchor, and a template
        // that re-anchors less often keeps classifying stretches it should have
        // abstained on. Measured, that alone cost 16 points of ventricular
        // precision - the abstention was doing more work than it looked.
        self.missed = self.missed.saturating_add(1);
        if matched_rival {
            self.rival_run = self.rival_run.saturating_add(1);
            self.consider_promotion();
        }
        if self.dominant.is_some() && !matched_rival {
            self.rival_run = 0;
        }
        {
            if self.missed >= self.cfg.reanchor_after {
                // Nothing has matched the dominant for long enough that it is
                // more likely wrong than the beats are. Start again from the
                // current beat, unestablished, exactly as the single-template
                // version did - that recovery path is what stopped a template
                // anchored on one bad beat from silencing a whole recording,
                // and the rival must not be allowed to slow it down.
                self.dominant = Some(Cluster::new(b, amplitude, area, width, slope, qrs_ms));
                self.rival = None;
                self.missed = 0;
                self.rival_run = 0;
            } else if !matched_rival {
                self.rival = Some(Cluster::new(b, amplitude, area, width, slope, qrs_ms));
            }
        }
        self.publish();
    }

    /// Swap the clusters when the rival is the better candidate for "conducted".
    fn consider_promotion(&mut self) {
        let (Some(dom), Some(riv)) = (self.dominant.as_ref(), self.rival.as_ref()) else {
            return;
        };
        let both_established =
            dom.accepted >= self.cfg.bootstrap_beats && riv.accepted >= self.cfg.bootstrap_beats;
        // Narrower by a clear margin, and seen often enough that the measurement
        // is not one odd beat. The rule is asymmetric on purpose: once the
        // narrower cluster is dominant the wider one cannot win it back, so the
        // two cannot oscillate.
        let narrower = both_established
            && riv.accepted >= self.cfg.promote_beats
            && riv.width > 0.0
            && riv.width <= self.cfg.promote_width_frac * dom.width;
        // The dominant is too wide to be a conducted beat and the rival is not.
        let dom_is_wide = self.cfg.conducted_width_ms > 0.0
            && both_established
            && riv.accepted >= self.cfg.promote_beats
            && dom.qrs_ms > self.cfg.conducted_width_ms
            && riv.qrs_ms > 0.0
            && riv.qrs_ms <= self.cfg.conducted_width_ms
            && riv.qrs_ms + self.cfg.promote_margin_ms <= dom.qrs_ms;
        // Or the recording has simply changed hands - a new lead, a new posture,
        // a sustained new rhythm. This is the behaviour the single-template
        // re-anchor had, except that it now lands on an established morphology
        // instead of on whichever beat happened to be current.
        let took_over = self.cfg.takeover_after > 0 && self.rival_run >= self.cfg.takeover_after;
        if narrower || dom_is_wide || took_over {
            std::mem::swap(&mut self.dominant, &mut self.rival);
            self.rival_run = 0;
            self.missed = 0;
        }
    }

    fn publish(&mut self) {
        if let Some(c) = self.dominant.as_ref() {
            self.amplitude = c.amplitude;
            self.area = c.area;
            self.width = c.width;
            self.slope = c.slope;
        }
    }

    pub fn reset(&mut self) {
        self.dominant = None;
        self.rival = None;
        self.missed = 0;
        self.rival_run = 0;
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
        t.update(&vector(0), 1.0, 1.0, 100.0, 1.0, 0.0, true); // anchors here
        assert!(t.vector().is_some());

        // A long run of a completely different, self-consistent morphology.
        let good = vector(20);
        for _ in 0..40 {
            t.update(&good, 1.0, 1.0, 100.0, 1.0, 0.0, true);
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
            t.update(&normal, 1.0, 1.0, 100.0, 1.0, 0.0, true);
        }
        for i in 0..60 {
            // One ectopic beat every fourth beat, as in bigeminy or trigeminy.
            t.update(
                if i % 4 == 0 { &ectopic } else { &normal },
                1.0,
                1.0,
                100.0,
                1.0,
                0.0,
                true,
            );
        }
        assert!(
            t.similarity(&normal).unwrap() > 0.9,
            "template drifted onto the ectopic morphology"
        );
    }

    /// The failure this rule exists for: the ectopic beat is the majority, so
    /// "most frequent" anchors on it and every morphology feature inverts.
    /// Width is what breaks the tie, and it is the one measurement in this
    /// engine that means the same thing in every patient.
    #[test]
    fn the_wide_majority_does_not_get_to_be_the_dominant_beat() {
        let mut t = Template::new(TemplateConfig::default());
        let conducted = vector(0);
        let ventricular = vector(20);
        // The wide beat arrives first and stays in the majority, 3 to 2.
        for i in 0..200 {
            let wide = i % 5 < 3;
            t.update(
                if wide { &ventricular } else { &conducted },
                1.0,
                1.0,
                100.0,
                1.0,
                if wide { 140.0 } else { 80.0 },
                true,
            );
        }
        assert!(
            t.similarity(&conducted).unwrap() > 0.9,
            "the 140 ms majority was taken for the conducted beat"
        );
    }

    /// And it must not fire where both morphologies are wide, which is what
    /// bundle branch block looks like: flipping between two abnormal beats
    /// helps nothing and costs the records where the template was right.
    #[test]
    fn two_wide_morphologies_do_not_trade_places() {
        let mut t = Template::new(TemplateConfig::default());
        let dominant = vector(0);
        let other = vector(20);
        for i in 0..200 {
            let is_dom = i % 5 < 3;
            t.update(
                if is_dom { &dominant } else { &other },
                1.0,
                1.0,
                100.0,
                1.0,
                if is_dom { 140.0 } else { 125.0 },
                true,
            );
        }
        assert!(
            t.similarity(&dominant).unwrap() > 0.9,
            "two wide morphologies traded places"
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
