//! Per-beat feature extraction: the shared layer under every beat detector.
//!
//! Computed once per beat and handed to all of them. Each feature is
//! dimensionless, and each is expressed relative to *this patient's* dominant
//! beat rather than to an absolute scale — the same reasoning as the quality
//! monitor in Phase 1, and for the same reason: a threshold in millivolts or
//! milliseconds does not survive a change of electrode, patient or lead.
//!
//! # Latency
//!
//! A beat cannot be judged until the *next* beat has arrived, because the
//! interval that follows it is what separates a premature supraventricular beat
//! (which resets the sinus node, so the following pause is short) from a
//! ventricular one (which does not, so the pause is compensatory). The
//! classifier therefore runs one beat behind the detector. That is inherent to
//! the evidence, not an implementation choice.

use crate::template::{self, BeatVector, ShapeTemplate, Template, TemplateConfig};
use ecg_dsp::{ms_to_samples, Ring};
use ecg_qrs::QrsEvent;

/// Number of features in the shared vector.
pub const NF: usize = 16;

/// Confidence at which a P wave counts as half-present. A ratio of peak to
/// noise floor is unbounded above and the evidence it carries is not, so it is
/// squashed rather than fed in raw.
const P_CONFIDENCE_HALF: f32 = 4.0;

#[derive(Debug, Clone, Copy, Default)]
pub struct BeatFeatures {
    /// Correlation with the dominant beat. The single strongest morphology cue.
    pub ncc_template: f32,
    /// Correlation with the previous beat: distinguishes an isolated odd beat
    /// from a sustained change of morphology.
    pub ncc_prev: f32,
    /// QRS duration over the dominant beat's. Ventricular beats are wide because
    /// they spread through muscle instead of the conduction system.
    pub width_rel: f32,
    pub amp_rel: f32,
    pub area_rel: f32,
    pub slope_rel: f32,
    /// Interval before this beat, over the running median. Prematurity.
    pub rr_prev_rel: f32,
    /// Interval after it.
    pub rr_post_rel: f32,
    /// Half the sum of the two, over the median. Near 1 when the pause is fully
    /// compensatory, which is the classic signature of a ventricular beat;
    /// below 1 when the sinus node was reset, which is the supraventricular one.
    pub rr_sum_rel: f32,
    pub rr_ratio: f32,
    /// How firmly a P wave stands above the baseline before this beat, squashed
    /// into [0, 1). Zero means none was found - which is itself the evidence: a
    /// supraventricular ectopic beat's P wave is early enough to be buried in
    /// the preceding T wave, and a junctional one has none at all.
    pub p_present: f32,
    /// PR interval over this patient's running median. An atrial impulse that
    /// starts somewhere other than the sinus node reaches the ventricles by a
    /// different path, and takes a different time to do it.
    pub p_pr_rel: f32,
    /// P amplitude over the running median, unsigned.
    pub p_amp_rel: f32,
    /// Sign of the P wave against this patient's own dominant sign: +1 for the
    /// usual polarity, -1 for an inverted one, 0 for no P wave. Inversion means
    /// the atria were depolarised from below, which a sinus beat cannot do.
    pub p_polarity: f32,
    /// How well the atrial segment before this beat matches the running atrial
    /// template. The presence question asked in the form that can be answered:
    /// not "is there a deflection here" but "is it this patient's P wave".
    pub p_ncc: f32,
    /// The same match, against this patient's own typical match rather than
    /// against 1.0: how much worse the atrial segment fits than it usually
    /// does. A raw correlation is not comparable between patients - a clean P
    /// wave sits at 0.95 and a small one buried in noise at 0.6, and a
    /// threshold that means "no atrial activity" for the first means "perfectly
    /// normal" for the second. Everything else in this vector is relative for
    /// exactly this reason.
    pub p_ncc_rel: f32,
}

impl BeatFeatures {
    #[inline]
    pub fn vector(&self) -> [f32; NF] {
        [
            self.ncc_template,
            self.ncc_prev,
            self.width_rel,
            self.amp_rel,
            self.area_rel,
            self.slope_rel,
            self.rr_prev_rel,
            self.rr_post_rel,
            self.rr_sum_rel,
            self.rr_ratio,
            self.p_present,
            self.p_pr_rel,
            self.p_amp_rel,
            self.p_polarity,
            self.p_ncc,
            self.p_ncc_rel,
        ]
    }

    pub const NAMES: [&'static str; NF] = [
        "ncc_template",
        "ncc_prev",
        "width_rel",
        "amp_rel",
        "area_rel",
        "slope_rel",
        "rr_prev_rel",
        "rr_post_rel",
        "rr_sum_rel",
        "rr_ratio",
        "p_present",
        "p_pr_rel",
        "p_amp_rel",
        "p_polarity",
        "p_ncc",
        "p_ncc_rel",
    ];
}

/// Atrial evidence for one beat. Its defaults are the neutral values used when
/// there is no delineation to read: "nothing known", not "nothing there".
#[derive(Debug, Clone, Copy)]
struct Atrial {
    present: f32,
    pr_rel: f32,
    amp_rel: f32,
    polarity: f32,
    p_ncc: f32,
    p_ncc_rel: f32,
}

impl Default for Atrial {
    fn default() -> Self {
        Atrial {
            present: 0.0,
            pr_rel: 1.0,
            amp_rel: 1.0,
            polarity: 0.0,
            p_ncc: 0.0,
            p_ncc_rel: 1.0,
        }
    }
}

impl BeatFeatures {
    /// The features the fitted detectors are trained on.
    ///
    /// Two of the sixteen are left out, and the reason is worth keeping: both
    /// were measured, and both made the supraventricular detector worse.
    ///
    /// `p_present` scores the largest deflection in the atrial window against
    /// that window's noise floor. The search returns the largest deflection
    /// whatever the window contains, so on 205,000 beats it separated
    /// conducted beats from ventricular ones at an AUC of 0.48 - nothing.
    ///
    /// `p_ncc` is the raw correlation with the atrial template, and it is the
    /// strongest feature in the table by per-record AUC (0.088 for S). It still
    /// costs 0.033 of sealed AUC, because a correlation is not comparable
    /// between patients: 0.75 means "no atrial activity" for a patient whose P
    /// wave is clean and "entirely normal" for one whose P wave is small. It
    /// survives here as the quantity `p_ncc_rel` is measured against, which is
    /// the same evidence expressed the way the rest of this vector is.
    pub const MODEL_FEATURES: [&'static str; 14] = [
        "ncc_template",
        "ncc_prev",
        "width_rel",
        "amp_rel",
        "area_rel",
        "slope_rel",
        "rr_prev_rel",
        "rr_post_rel",
        "rr_sum_rel",
        "rr_ratio",
        "p_pr_rel",
        "p_amp_rel",
        "p_polarity",
        "p_ncc_rel",
    ];
}

/// A beat, its features, and whether they can be believed.
#[derive(Debug, Clone, Copy)]
pub struct BeatObservation {
    pub sample: u64,
    pub features: BeatFeatures,
    /// Signal quality was acceptable across this beat and both its intervals.
    pub quality_ok: bool,
    /// The dominant-beat template had been established when this beat was judged.
    pub template_ready: bool,
}

/// A beat waiting for the interval that follows it.
///
/// Only its time and context are held. The morphology is measured later, at
/// finalisation: the beat's window extends past the R peak, so at the moment the
/// beat arrives those samples have not been seen yet.
#[derive(Debug, Clone, Copy)]
struct Pending {
    sample: u64,
    rr_prev: f32,
    quality_ok: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct BeatConfig {
    pub fs: f64,
    pub template: TemplateConfig,
    /// Template for the atrial segment. A looser admission threshold than the
    /// QRS template's: a P wave is a tenth the amplitude of a complex and sits
    /// on whatever the T wave left behind, so a gate tight enough for a QRS
    /// never admits a second beat.
    pub atrial_template: TemplateConfig,
    /// Fraction of the window's peak QRS-band magnitude that bounds the complex.
    pub width_frac: f32,
    /// Half-width of the search for the QRS boundaries.
    pub width_search_ms: f64,
    /// Half-width of the search for the band-signal peak that anchors the
    /// boundary walk, and of the window amplitude and slope are taken over.
    pub anchor_search_ms: f64,
}

impl BeatConfig {
    pub fn new(fs: f64) -> Self {
        BeatConfig {
            fs,
            template: TemplateConfig::default(),
            atrial_template: TemplateConfig {
                admit_ncc: 0.70,
                ..TemplateConfig::default()
            },
            width_frac: 0.15,
            width_search_ms: 150.0,
            anchor_search_ms: 70.0,
        }
    }
}

/// Running median of the last eight usable values: this patient's own
/// reference, which is the only scale any of these features is expressed on.
#[derive(Debug, Clone, Copy)]
struct Median8 {
    buf: [f32; 8],
    n: usize,
    idx: usize,
}

impl Median8 {
    fn new(default: f32) -> Self {
        Median8 {
            buf: [default; 8],
            n: 0,
            idx: 0,
        }
    }
    fn push(&mut self, v: f32) {
        self.buf[self.idx] = v;
        self.idx = (self.idx + 1) & 7;
        self.n = (self.n + 1).min(8);
    }
    fn median(&self) -> f32 {
        if self.n == 0 {
            return self.buf[0];
        }
        let mut t = [0.0f32; 8];
        t[..self.n].copy_from_slice(&self.buf[..self.n]);
        t[..self.n].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        t[self.n / 2]
    }
}

pub struct BeatAnalyzer {
    cfg: BeatConfig,
    clean: Ring,
    qrs: Ring,
    n: u64,
    floor: u64,
    template: Template,
    prev_vector: Option<BeatVector>,
    pending: Option<Pending>,
    last_sample: Option<u64>,
    rr_ref: Median8,
    pr_ref: Median8,
    p_amp_ref: Median8,
    /// Running sign of the P wave: the patient's own dominant polarity.
    p_sign: f32,
    p_template: ShapeTemplate,
    p_ncc_ref: Median8,
    /// Quality across the current interval.
    clean_interval: bool,
    width_search: usize,
    anchor_search: usize,
}

impl BeatAnalyzer {
    pub fn new(cfg: BeatConfig) -> Self {
        // Long enough to hold a beat window plus the wait for the following beat.
        let cap = ms_to_samples(cfg.fs, 4000.0);
        BeatAnalyzer {
            clean: Ring::with_capacity(cap),
            qrs: Ring::with_capacity(cap),
            n: 0,
            floor: 0,
            template: Template::new(cfg.template),
            prev_vector: None,
            pending: None,
            last_sample: None,
            rr_ref: Median8::new(800.0),
            pr_ref: Median8::new(160.0),
            p_amp_ref: Median8::new(0.1),
            p_sign: 0.0,
            p_template: ShapeTemplate::new(cfg.atrial_template),
            p_ncc_ref: Median8::new(0.8),
            clean_interval: true,
            width_search: ms_to_samples(cfg.fs, cfg.width_search_ms),
            anchor_search: ms_to_samples(cfg.fs, cfg.anchor_search_ms),
            cfg,
        }
    }

    pub fn config(&self) -> &BeatConfig {
        &self.cfg
    }

    pub fn template(&self) -> &Template {
        &self.template
    }

    #[inline]
    pub fn push_sample(&mut self, clean: f32, qrs: f32, quality_ok: bool) {
        self.clean.push(clean);
        self.qrs.push(qrs);
        self.n += 1;
        if self.n > self.clean.capacity() as u64 {
            self.floor = self.n - self.clean.capacity() as u64;
        }
        self.clean_interval &= quality_ok;
    }

    /// Register a beat. Returns the *previous* beat's observation, now that the
    /// interval following it is known.
    pub fn push_beat(
        &mut self,
        ev: &QrsEvent,
        wave: Option<&crate::delineate::Delineation>,
    ) -> Option<BeatObservation> {
        let fs = self.cfg.fs as f32;
        let prev_sample = self.last_sample.replace(ev.sample);
        let quality = std::mem::replace(&mut self.clean_interval, true);

        let rr_prev = prev_sample
            .map(|p| (ev.sample.saturating_sub(p)) as f32 * 1000.0 / fs)
            .unwrap_or(f32::NAN);

        // Finalise the beat that is waiting, using this beat's interval as its
        // RR-post.
        // `wave` describes the beat being finalised, not the one arriving:
        // delineation lags by one beat for the same reason classification does.
        let out = self
            .pending
            .take()
            .and_then(|p| self.finalise(p, rr_prev, wave.filter(|d| d.r == p.sample)));

        self.pending = Some(Pending {
            sample: ev.sample,
            rr_prev,
            quality_ok: quality,
        });
        if rr_prev.is_finite() && (200.0..3000.0).contains(&rr_prev) && quality {
            self.rr_ref.push(rr_prev);
        }
        out
    }

    /// Atrial evidence for one beat, against this patient's own running
    /// references.
    ///
    /// The references are updated only from beats whose signal was clean and
    /// whose P wave was actually found, so a run of ectopy or a noisy stretch
    /// cannot redefine what normal looks like. That is the same guard the
    /// template carries, and for the same reason.
    fn atrial(&mut self, wave: Option<&crate::delineate::Delineation>, quality_ok: bool) -> Atrial {
        let Some(d) = wave else {
            return Atrial::default();
        };
        // The template is fed before it is read, exactly as the QRS one is: the
        // gate admits only segments that already match, so a beat cannot lift
        // its own score, and the first beat of a record has nothing to match.
        let (p_ncc, p_ncc_rel) = match d.atrial.as_ref() {
            Some(v) => {
                let ncc = self.p_template.similarity(v).unwrap_or(0.0);
                let ready = self.p_template.established();
                self.p_template.update(v, quality_ok);
                if !ready {
                    (0.0, 1.0)
                } else {
                    let typical = self.p_ncc_ref.median();
                    let rel = ((1.0 - ncc).max(0.0) / (1.0 - typical).max(0.02)).min(20.0);
                    if quality_ok {
                        self.p_ncc_ref.push(ncc);
                    }
                    (ncc, rel)
                }
            }
            None => (0.0, 1.0),
        };
        // A soft presence score rather than the raw ratio: the difference
        // between a P wave four times the noise floor and one forty times it is
        // not four times as much evidence that the atria fired.
        let conf = d.p_confidence.max(0.0);
        let present = conf / (conf + P_CONFIDENCE_HALF);
        if d.p.is_none() {
            return Atrial {
                present,
                p_ncc,
                p_ncc_rel,
                ..Atrial::default()
            };
        }

        let amp = d.p_amplitude;
        let polarity = if self.p_sign == 0.0 {
            0.0
        } else {
            (amp.signum() * self.p_sign).clamp(-1.0, 1.0)
        };
        let amp_rel = {
            let r = self.p_amp_ref.median();
            if r > 1e-6 {
                amp.abs() / r
            } else {
                1.0
            }
        };
        let pr_rel = match d.pr_ms {
            Some(pr) if pr.is_finite() => {
                let r = self.pr_ref.median();
                if r > 1e-6 {
                    pr / r
                } else {
                    1.0
                }
            }
            _ => 1.0,
        };

        if quality_ok {
            self.p_amp_ref.push(amp.abs());
            if let Some(pr) = d.pr_ms.filter(|v| (40.0..400.0).contains(v)) {
                self.pr_ref.push(pr);
            }
            // The dominant sign moves slowly, so one inverted P wave shifts it
            // a little and a sustained change of rhythm eventually flips it.
            self.p_sign = (self.p_sign + 0.1 * (amp.signum() - self.p_sign)).clamp(-1.0, 1.0);
        }
        Atrial {
            present,
            pr_rel,
            amp_rel,
            polarity,
            p_ncc,
            p_ncc_rel,
        }
    }

    fn finalise(
        &mut self,
        p: Pending,
        rr_post: f32,
        wave: Option<&crate::delineate::Delineation>,
    ) -> Option<BeatObservation> {
        let vector = template::extract(&self.clean, p.sample, self.n, self.cfg.fs, self.floor)?;
        let (amplitude, area, width, slope) = self.measure(p.sample);
        let ref_rr = self.rr_ref.median().max(1.0);
        let ready = self.template.established();
        let ncc_template = self.template.similarity(&vector).unwrap_or(0.0);
        let ncc_prev = self.prev_vector.map(|v| v.ncc(&vector)).unwrap_or(0.0);

        let rel = |x: f32, r: f32| if r > 1e-6 { x / r } else { 1.0 };
        let a = self.atrial(wave, p.quality_ok);
        let features = BeatFeatures {
            ncc_template,
            ncc_prev,
            width_rel: rel(width, self.template.width),
            amp_rel: rel(amplitude, self.template.amplitude),
            area_rel: rel(area, self.template.area),
            slope_rel: rel(slope, self.template.slope),
            rr_prev_rel: if p.rr_prev.is_finite() {
                p.rr_prev / ref_rr
            } else {
                1.0
            },
            rr_post_rel: if rr_post.is_finite() {
                rr_post / ref_rr
            } else {
                1.0
            },
            rr_sum_rel: if p.rr_prev.is_finite() && rr_post.is_finite() {
                0.5 * (p.rr_prev + rr_post) / ref_rr
            } else {
                1.0
            },
            p_present: a.present,
            p_pr_rel: a.pr_rel,
            p_amp_rel: a.amp_rel,
            p_polarity: a.polarity,
            p_ncc: a.p_ncc,
            p_ncc_rel: a.p_ncc_rel,
            rr_ratio: if p.rr_prev.is_finite() && rr_post > 1e-3 {
                p.rr_prev / rr_post
            } else {
                1.0
            },
        };

        // The template learns only after the beat has been described, so a beat
        // never contributes to the reference it is measured against.
        self.template
            .update(&vector, amplitude, area, width, slope, p.quality_ok);
        self.prev_vector = Some(vector);

        Some(BeatObservation {
            sample: p.sample,
            features,
            quality_ok: p.quality_ok,
            template_ready: ready,
        })
    }

    /// Amplitude, area, width and peak slope of the complex at `r`.
    ///
    /// The QRS-band tap is band-passed, so it lags the fiducial - which is placed
    /// on the analysis signal and already delay-compensated - by that filter's
    /// group delay. Walking outward from `r` therefore starts at a point where
    /// the band signal may not have risen yet, and the complex collapses to zero
    /// width. Measured on MIT-BIH that happened to a quarter of all beats, which
    /// is why `width_rel` and `amp_rel` read 0.000 at the first and second
    /// quartile of every class. The boundary search is anchored on the band's own
    /// local peak instead, which is correct whatever the group delay.
    fn measure(&self, r: u64) -> (f32, f32, f32, f32) {
        let half = self.width_search as u64;
        let lo = r.saturating_sub(half).max(self.floor);
        let hi = (r + half).min(self.n.saturating_sub(1));
        if hi <= lo {
            return (0.0, 0.0, 0.0, 0.0);
        }

        // Anchor: the band-signal peak nearest the fiducial.
        let anchor_lo = r.saturating_sub(self.anchor_search as u64).max(lo);
        let anchor_hi = (r + self.anchor_search as u64).min(hi);
        let mut anchor = anchor_lo;
        let mut anchor_v = -1.0f32;
        for i in anchor_lo..=anchor_hi {
            let v = self.qrs.at(i).abs();
            if v > anchor_v {
                anchor_v = v;
                anchor = i;
            }
        }
        if anchor_v <= 0.0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let bound = anchor_v * self.cfg.width_frac;

        // Walk outward from the anchor to the first sample below the boundary,
        // rather than thresholding the whole window: a neighbouring T wave or
        // the next beat would otherwise extend the measured complex.
        let mut start = anchor;
        while start > lo && self.qrs.at(start - 1).abs() > bound {
            start -= 1;
        }
        let mut end = anchor;
        while end < hi && self.qrs.at(end + 1).abs() > bound {
            end += 1;
        }

        let mut area = 0.0f32;
        for i in start..=end {
            area += self.qrs.at(i).abs();
        }

        // Amplitude and slope come from the analysis signal over a fixed window,
        // so they do not inherit the width measurement's boundaries.
        let amp_lo = r.saturating_sub(self.anchor_search as u64).max(self.floor);
        let amp_hi = (r + self.anchor_search as u64).min(self.n.saturating_sub(1));
        let (mut mn, mut mx, mut slope) = (f32::MAX, f32::MIN, 0.0f32);
        let mut prev = self.clean.at(amp_lo);
        for i in amp_lo..=amp_hi {
            let c = self.clean.at(i);
            mn = mn.min(c);
            mx = mx.max(c);
            slope = slope.max((c - prev).abs());
            prev = c;
        }

        let dt = 1000.0 / self.cfg.fs as f32;
        (
            mx - mn,
            area * dt,
            (end - start) as f32 * dt,
            slope * self.cfg.fs as f32,
        )
    }

    /// Samples were lost. The signal rings and the pending beat are stale, but
    /// the template and the interval reference still describe this patient, so
    /// they are kept: a dropped packet is not a new patient.
    /// `unobserved` is how many samples passed without being seen.
    pub fn on_gap(&mut self, unobserved: u64) {
        // The counter advances through the gap so beat positions stay true; the
        // floor rises so a window cannot straddle it.
        self.n += unobserved;
        self.floor = self.n;
        self.pending = None;
        self.prev_vector = None;
        self.last_sample = None;
        self.clean_interval = true;
    }

    pub fn reset(&mut self) {
        self.clean.reset();
        self.qrs.reset();
        self.n = 0;
        self.floor = 0;
        self.template.reset();
        self.prev_vector = None;
        self.pending = None;
        self.last_sample = None;
        self.rr_ref = Median8::new(800.0);
        self.pr_ref = Median8::new(160.0);
        self.p_amp_ref = Median8::new(0.1);
        self.p_sign = 0.0;
        self.p_template.reset();
        self.p_ncc_ref = Median8::new(0.8);
        self.clean_interval = true;
    }
}
