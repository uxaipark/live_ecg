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

use crate::template::{self, BeatVector, Template, TemplateConfig};
use ecg_dsp::{ms_to_samples, Ring};
use ecg_qrs::QrsEvent;

/// Number of features in the shared vector.
pub const NF: usize = 10;

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
            width_frac: 0.15,
            width_search_ms: 150.0,
            anchor_search_ms: 70.0,
        }
    }
}

/// Running median of the last eight usable intervals.
#[derive(Debug, Clone, Copy)]
struct RrReference {
    buf: [f32; 8],
    n: usize,
    idx: usize,
}

impl RrReference {
    fn new(default_ms: f32) -> Self {
        RrReference {
            buf: [default_ms; 8],
            n: 0,
            idx: 0,
        }
    }
    fn push(&mut self, rr: f32) {
        self.buf[self.idx] = rr;
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
    rr_ref: RrReference,
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
            rr_ref: RrReference::new(cfg.fs as f32 * 0.0 + 800.0),
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
    pub fn push_beat(&mut self, ev: &QrsEvent) -> Option<BeatObservation> {
        let fs = self.cfg.fs as f32;
        let prev_sample = self.last_sample.replace(ev.sample);
        let quality = std::mem::replace(&mut self.clean_interval, true);

        let rr_prev = prev_sample
            .map(|p| (ev.sample.saturating_sub(p)) as f32 * 1000.0 / fs)
            .unwrap_or(f32::NAN);

        // Finalise the beat that is waiting, using this beat's interval as its
        // RR-post.
        let out = self.pending.take().and_then(|p| self.finalise(p, rr_prev));

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

    fn finalise(&mut self, p: Pending, rr_post: f32) -> Option<BeatObservation> {
        let vector = template::extract(&self.clean, p.sample, self.n, self.cfg.fs, self.floor)?;
        let (amplitude, area, width, slope) = self.measure(p.sample);
        let ref_rr = self.rr_ref.median().max(1.0);
        let ready = self.template.established();
        let ncc_template = self.template.similarity(&vector).unwrap_or(0.0);
        let ncc_prev = self.prev_vector.map(|v| v.ncc(&vector)).unwrap_or(0.0);

        let rel = |x: f32, r: f32| if r > 1e-6 { x / r } else { 1.0 };
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
    pub fn on_gap(&mut self) {
        self.clean.reset();
        self.qrs.reset();
        self.n = 0;
        self.floor = 0;
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
        self.rr_ref = RrReference::new(800.0);
        self.clean_interval = true;
    }
}
