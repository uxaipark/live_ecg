//! Real-time signal-quality and noise detection.
//!
//! The question is narrow and operational: **is the next second of this channel
//! worth believing?** The output is a gate that keeps artefact out of the
//! detector's adaptive state and out of the episode logic, plus the flags a
//! reviewer needs to see why a stretch was dropped.
//!
//! # Why the features are scale-invariant
//!
//! An earlier version normalised each band's power by total in-band power. That
//! measured the opposite of what it claimed: added muscle and motion artefact
//! land *inside* the analysis band, so the denominator grows with the noise and
//! the ratio falls. Measured against the MIT noise-stress protocol it scored an
//! AUC of 0.31 - reliably backwards.
//!
//! Every feature here is invariant to the signal's amplitude instead, either
//! because it is a shape statistic (kurtosis, skewness), a bounded fraction of
//! total power, or a ratio against the channel's own slow history. That also
//! means the thresholds survive an uncalibrated recording, which a millivolt
//! threshold does not.
//!
//! # Cost
//!
//! Features are recomputed once per hop (default 250 ms) over the trailing
//! window rather than updated every sample. The work per sample is
//! `window / hop` multiply-accumulates - about four - and the moments are exact
//! rather than accumulated across hours.

use ecg_dsp::{ms_to_samples, MovingExtrema, Ring};

pub mod flags {
    pub const SATURATION: u8 = 1 << 0;
    pub const FLATLINE: u8 = 1 << 1;
    pub const BASELINE: u8 = 1 << 2;
    pub const EMG: u8 = 1 << 3;
    pub const LOW_AMP: u8 = 1 << 4;
    pub const UNSTABLE: u8 = 1 << 5;
    pub const LEAD_OFF: u8 = 1 << 6;

    pub fn names(f: u8) -> Vec<&'static str> {
        let mut v = Vec::new();
        for (bit, name) in [
            (SATURATION, "SATURATION"),
            (FLATLINE, "FLATLINE"),
            (BASELINE, "BASELINE"),
            (EMG, "EMG"),
            (LOW_AMP, "LOW_AMP"),
            (UNSTABLE, "UNSTABLE"),
            (LEAD_OFF, "LEAD_OFF"),
        ] {
            if f & bit != 0 {
                v.push(name);
            }
        }
        v
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Quality {
    /// Usable for every downstream task, morphology included.
    Good,
    /// Rhythm-usable: detections are trustworthy, morphology is not.
    Acceptable,
    /// Nothing downstream should believe this stretch.
    Unusable,
}

/// Raw features, exposed so each one's discriminative power can be measured
/// separately rather than assumed.
#[derive(Debug, Clone, Copy, Default)]
pub struct QualityFeatures {
    /// Excess kurtosis of the analysis band. A clean ECG is dominated by sparse,
    /// tall QRS complexes and is strongly leptokurtic; muscle noise is close to
    /// Gaussian and pulls this toward zero.
    pub kurtosis: f32,
    pub skewness: f32,
    /// Share of total power above the analysis band: `hf / (hf + clean)`.
    pub hf_ratio: f32,
    /// Share of total power below the analysis band: `base / (base + clean)`.
    pub base_ratio: f32,
    /// Share of in-band power inside the QRS sub-band.
    pub qrs_ratio: f32,
    /// `kurtosis` and `qrs_ratio` against this channel's own slow medians.
    ///
    /// Both have a legitimate absolute range that depends on the rhythm, not on
    /// the signal quality: bigeminy and paced rhythms fill the trace with wide,
    /// frequent complexes, so a clean recording of them is far less leptokurtic
    /// than clean sinus. Judged absolutely, whole healthy records were condemned
    /// - 90% of seconds on the bigeminal MIT-BIH Supraventricular record 801.
    pub kurtosis_rel: f32,
    pub qrs_ratio_rel: f32,
    /// Peak-to-peak of the analysis band over the window, mV.
    pub p2p: f32,
    /// Window peak-to-peak against this channel's own slow median. Catches both
    /// electrode dropout and motion swings without knowing the channel's gain.
    pub p2p_rel: f32,
    pub sat_frac: f32,
    pub flat_frac: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct QualityConfig {
    pub fs: f64,
    /// Window over which features are measured.
    pub window_ms: f64,
    /// How often features are recomputed.
    pub hop_ms: f64,
    /// Span of the slow amplitude reference.
    pub reference_ms: f64,
    /// Excess kurtosis below which the window looks like noise rather than ECG.
    pub kurtosis_warn: f32,
    pub kurtosis_bad: f32,
    pub hf_warn: f32,
    pub hf_bad: f32,
    pub base_warn: f32,
    pub base_bad: f32,
    /// Share of in-band power inside the QRS sub-band. Noise spreads power out
    /// of that sub-band, so a low value means the window stopped looking like ECG.
    pub qrs_ratio_warn: f32,
    pub qrs_ratio_bad: f32,
    /// Window amplitude relative to the channel's own slow median.
    pub p2p_rel_low: f32,
    pub p2p_rel_high: f32,
    /// Relative amplitude below which the lead is treated as off. Unlike
    /// `p2p_rel_low` this is a veto, not soft evidence: an electrode that has
    /// stopped making contact is a hard failure, and averaging it against two
    /// healthy-looking shape statistics hides it. Measured on the MIT-BIH Normal
    /// Sinus record 16272, whose amplitude collapses twentyfold five hours in,
    /// that averaging reported 6.6% of the record unusable while 74% of its
    /// beats were undetectable.
    pub p2p_rel_off: f32,
    pub sat_frac_bad: f32,
    pub flat_frac_bad: f32,
    /// Slope magnitude below which a sample counts as flat, as a fraction of the
    /// window's own peak-to-peak.
    pub flat_eps_rel: f32,
    /// Floors for the relative forms of `kurtosis` and `qrs_ratio`, as a
    /// fraction of this channel's own median.
    pub rel_bad: f32,
    pub rel_warn: f32,
    /// Relative weights of the three soft indicators in the score.
    pub w_kurtosis: f32,
    pub w_qrs_ratio: f32,
    pub w_amplitude: f32,
    /// Score at or below which a window is unusable.
    pub score_bad: f32,
    /// Score at or below which a window is degraded.
    pub score_warn: f32,
}

impl QualityConfig {
    pub fn new(fs: f64) -> Self {
        QualityConfig {
            fs,
            window_ms: 2000.0,
            hop_ms: 250.0,
            reference_ms: 300_000.0,
            // Calibrated from the measured distributions on clean signal
            // (`ecg-eval qfeat`), not chosen by eye: each `warn` sits at roughly
            // the first percentile of the clean population, so healthy signal is
            // not condemned, and each `bad` sits where the noisy population lives.
            kurtosis_warn: 2.5,
            kurtosis_bad: 0.5,
            hf_warn: 0.10,
            hf_bad: 0.35,
            base_warn: 0.50,
            base_bad: 0.85,
            qrs_ratio_warn: 0.22,
            qrs_ratio_bad: 0.08,
            p2p_rel_low: 0.30,
            p2p_rel_high: 2.0,
            p2p_rel_off: 0.15,
            sat_frac_bad: 0.02,
            flat_frac_bad: 0.90,
            flat_eps_rel: 1e-3,
            rel_bad: 0.20,
            rel_warn: 0.60,
            w_kurtosis: 1.0,
            w_qrs_ratio: 1.0,
            w_amplitude: 1.0,
            // Chosen from the measured trade-off curve (`ecg-eval qrs` false-alarm
            // rate on clean corpora against `ecg-eval quality` gated precision on
            // the noise-stress corpus). Between 0.15 and 0.55 the false-alarm rate
            // moves by 0.7 points while gated precision at -6 dB moves by 8, so
            // the operating point sits high.
            score_bad: 0.50,
            score_warn: 0.75,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QualitySample {
    pub flags: u8,
    /// Continuous 0..1; 1 is pristine.
    pub score: f32,
    pub features: QualityFeatures,
}

impl QualitySample {
    pub fn level(&self, cfg: &QualityConfig) -> Quality {
        if self.flags & (flags::SATURATION | flags::FLATLINE) != 0 || self.score <= cfg.score_bad {
            Quality::Unusable
        } else if self.score <= cfg.score_warn {
            Quality::Acceptable
        } else {
            Quality::Good
        }
    }
}

/// Median of a long, coarsely-sampled history; the slow amplitude reference.
///
/// The median is computed only when a new value is admitted, and the result is
/// cached. Sorting the whole history on every hop, as an earlier version did,
/// cost more than the rest of the engine combined - about 190 operations per
/// input sample for a quantity that describes five minutes of signal.
#[derive(Debug, Clone)]
struct SlowMedian {
    buf: Vec<f32>,
    scratch: Vec<f32>,
    n: usize,
    idx: usize,
    cached: f32,
}

impl SlowMedian {
    fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        SlowMedian {
            buf: vec![0.0; cap],
            scratch: Vec::with_capacity(cap),
            n: 0,
            idx: 0,
            cached: 0.0,
        }
    }

    fn push(&mut self, v: f32) {
        let cap = self.buf.len();
        self.buf[self.idx] = v;
        self.idx = (self.idx + 1) % cap;
        self.n = (self.n + 1).min(cap);
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.buf[..self.n]);
        self.scratch
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.cached = self.scratch[self.n / 2];
    }

    #[inline]
    fn median(&self) -> f32 {
        self.cached
    }
}

/// Trailing-window sums, maintained incrementally.
///
/// Recomputing these by sweeping the window on every hop costs `window / hop`
/// passes per input sample - roughly two hundred operations at the default
/// settings, which dominated the whole per-channel budget. Adding the arriving
/// sample and subtracting the departing one is O(1) instead.
#[derive(Debug, Clone, Copy, Default)]
struct Sums {
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    b1: f64,
    b2: f64,
    h2: f64,
    q2: f64,
    sat: f64,
    flat: f64,
}

impl Sums {
    #[inline(always)]
    fn add(&mut self, t: &Taps, sign: f64) {
        let c = t.clean;
        let c2 = c * c;
        self.c1 += sign * c;
        self.c2 += sign * c2;
        self.c3 += sign * c2 * c;
        self.c4 += sign * c2 * c2;
        self.b1 += sign * t.base;
        self.b2 += sign * t.base * t.base;
        self.h2 += sign * t.hf * t.hf;
        self.q2 += sign * t.qrs * t.qrs;
        self.sat += sign * t.sat;
        self.flat += sign * t.flat;
    }
}

/// One sample's worth of every tap the monitor consumes.
#[derive(Debug, Clone, Copy, Default)]
struct Taps {
    clean: f64,
    base: f64,
    hf: f64,
    qrs: f64,
    sat: f64,
    flat: f64,
}

pub struct QualityMonitor {
    cfg: QualityConfig,
    clean: Ring,
    base: Ring,
    hf: Ring,
    qrs: Ring,
    sat: Ring,
    flat: Ring,
    sums: Sums,
    window: usize,
    hop: usize,
    since_hop: usize,
    hops_since_refresh: u32,
    hops_since_slow: u32,
    n: u64,
    prev_clean: f32,
    flat_eps: f32,
    pp: MovingExtrema,
    /// Slow per-channel references. All three are admitted by the same
    /// predicate, which depends only on features with a rhythm-independent
    /// absolute meaning, so a reference can never be defined by the quantity it
    /// is meant to normalise.
    slow_p2p: SlowMedian,
    slow_kurt: SlowMedian,
    slow_qrs: SlowMedian,
    last: QualitySample,
}

/// Hops between exact recomputations of the running sums. Incremental
/// add-and-subtract drifts in the last bits over hours; a sweep every few
/// seconds costs nothing measurable and keeps the state exact.
const REFRESH_HOPS: u32 = 64;

/// Hops between admissions to the slow amplitude reference. It describes five
/// minutes of signal, so sampling it four times a second buys nothing.
const SLOW_HOPS: u32 = 4;

impl QualityMonitor {
    pub fn new(cfg: QualityConfig) -> Self {
        let window = ms_to_samples(cfg.fs, cfg.window_ms);
        let hop = ms_to_samples(cfg.fs, cfg.hop_ms);
        let slow_cap =
            ((cfg.reference_ms / (cfg.hop_ms * SLOW_HOPS as f64)).round() as usize).max(4);
        QualityMonitor {
            clean: Ring::with_capacity(window),
            base: Ring::with_capacity(window),
            hf: Ring::with_capacity(window),
            qrs: Ring::with_capacity(window),
            sat: Ring::with_capacity(window),
            flat: Ring::with_capacity(window),
            sums: Sums::default(),
            window,
            hop,
            since_hop: 0,
            hops_since_refresh: 0,
            hops_since_slow: SLOW_HOPS,
            n: 0,
            prev_clean: 0.0,
            flat_eps: 1e-7,
            pp: MovingExtrema::new(window),
            slow_p2p: SlowMedian::new(slow_cap),
            slow_kurt: SlowMedian::new(slow_cap),
            slow_qrs: SlowMedian::new(slow_cap),
            last: QualitySample {
                flags: 0,
                score: 1.0,
                features: QualityFeatures::default(),
            },
            cfg,
        }
    }

    pub fn config(&self) -> &QualityConfig {
        &self.cfg
    }

    pub fn window_samples(&self) -> usize {
        self.window
    }

    #[inline]
    pub fn process(
        &mut self,
        _raw: f32,
        clean: f32,
        baseline: f32,
        hf: f32,
        qrs: f32,
        saturated: bool,
    ) -> QualitySample {
        let flat = if (clean - self.prev_clean).abs() < self.flat_eps {
            1.0
        } else {
            0.0
        };
        self.prev_clean = clean;
        let t = Taps {
            clean: clean as f64,
            base: baseline as f64,
            hf: hf as f64,
            qrs: qrs as f64,
            sat: if saturated { 1.0 } else { 0.0 },
            flat,
        };

        if self.n >= self.window as u64 {
            let old = self.tap_at(self.n - self.window as u64);
            self.sums.add(&old, -1.0);
        }
        self.sums.add(&t, 1.0);

        self.clean.push(clean);
        self.base.push(baseline);
        self.hf.push(hf);
        self.qrs.push(qrs);
        self.sat.push(t.sat as f32);
        self.flat.push(flat as f32);
        let (lo, hi) = self.pp.process(clean);
        self.n += 1;
        self.since_hop += 1;

        if self.since_hop >= self.hop {
            self.since_hop = 0;
            // Nothing is judged until the window is genuinely full. A partially
            // filled window has a small peak-to-peak, and admitting that as the
            // channel's amplitude reference made every later second look ten
            // times too large - the monitor condemned whole clean records on its
            // own startup transient.
            if self.n >= self.window as u64 {
                self.hops_since_refresh += 1;
                if self.hops_since_refresh >= REFRESH_HOPS {
                    self.hops_since_refresh = 0;
                    self.exact_refresh();
                }
                self.last = self.derive(hi - lo);
            }
        }
        self.last
    }

    #[inline]
    fn tap_at(&self, i: u64) -> Taps {
        Taps {
            clean: self.clean.at(i) as f64,
            base: self.base.at(i) as f64,
            hf: self.hf.at(i) as f64,
            qrs: self.qrs.at(i) as f64,
            sat: self.sat.at(i) as f64,
            flat: self.flat.at(i) as f64,
        }
    }

    fn exact_refresh(&mut self) {
        let span = self.window.min(self.n as usize);
        let from = self.n - span as u64;
        let mut s = Sums::default();
        for i in from..self.n {
            let t = self.tap_at(i);
            s.add(&t, 1.0);
        }
        self.sums = s;
    }

    fn derive(&mut self, p2p: f32) -> QualitySample {
        let span = self.window.min(self.n as usize).max(1);
        let inv = 1.0 / span as f64;
        let s = self.sums;

        let mean = s.c1 * inv;
        let m2 = s.c2 * inv;
        let m3 = s.c3 * inv;
        let m4 = s.c4 * inv;
        // Central moments from raw moments. That conversion is safe here and only
        // here: `clean` has been high-passed at ~0.5 Hz, so its window mean is
        // orders of magnitude below its standard deviation and there is nothing
        // to cancel. The baseline tap, which does carry a standing offset, gets
        // its own mean-corrected variance below.
        let var = (m2 - mean * mean).max(0.0);
        let mu3 = m3 - 3.0 * mean * m2 + 2.0 * mean * mean * mean;
        let mu4 = m4 - 4.0 * mean * m3 + 6.0 * mean * mean * m2 - 3.0 * mean.powi(4);
        let sd = var.sqrt();
        let (kurtosis, skewness) = if sd > 1e-12 {
            (mu4 / (var * var) - 3.0, mu3 / (sd * sd * sd))
        } else {
            (0.0, 0.0)
        };

        let p_clean = m2;
        // The baseline tap is the only one still carrying the record's DC offset,
        // and raw power there is dominated by it: measured on the noise-stress
        // corpus, `p_base / (p_base + p_clean)` sat at 0.996 on pristine signal,
        // which condemned every window. Wander is a *varying* quantity, so the
        // baseline tap is scored by its variance.
        let base_mean = s.b1 * inv;
        let p_base = (s.b2 * inv - base_mean * base_mean).max(0.0);
        let p_hf = s.h2 * inv;
        let p_qrs = s.q2 * inv;

        let eps_p = 1e-12;
        let mut f = QualityFeatures {
            kurtosis: kurtosis as f32,
            skewness: skewness as f32,
            hf_ratio: (p_hf / (p_hf + p_clean + eps_p)) as f32,
            base_ratio: (p_base / (p_base + p_clean + eps_p)) as f32,
            qrs_ratio: (p_qrs / (p_clean + eps_p)) as f32,
            p2p,
            p2p_rel: 1.0,
            kurtosis_rel: 1.0,
            qrs_ratio_rel: 1.0,
            sat_frac: (s.sat * inv) as f32,
            flat_frac: (s.flat * inv) as f32,
        };

        let cfgv = self.cfg;
        let reference = self.slow_p2p.median();
        f.p2p_rel = if reference > 1e-9 {
            p2p / reference
        } else {
            1.0
        };
        let kurt_ref = self.slow_kurt.median().max(0.5);
        let qrs_ref = self.slow_qrs.median().max(1e-3);
        f.kurtosis_rel = f.kurtosis / kurt_ref;
        f.qrs_ratio_rel = f.qrs_ratio / qrs_ref;
        if self.slow_p2p.n <= 1 {
            // With a single reference sample there is nothing to compare against.
            f.p2p_rel = 1.0;
            f.kurtosis_rel = 1.0;
            f.qrs_ratio_rel = 1.0;
        }

        // Admission depends only on out-of-band power and rail contact. Those
        // mean the same thing whatever the rhythm, so a long artefact burst
        // cannot redefine what "normal" means for this channel, and no reference
        // is gated on the quantity it normalises.
        self.hops_since_slow += 1;
        // A dead lead must not be allowed to become this channel's idea of
        // normal. Without the amplitude condition the reference re-anchors to the
        // collapsed level within one reference span and the fault stops being
        // reported - the electrode failure in MIT-BIH Normal Sinus 16272 lasts
        // twenty hours and was visible for only the first few minutes of it.
        let admissible = f.hf_ratio < cfgv.hf_warn
            && f.base_ratio < cfgv.base_warn
            && f.sat_frac <= 0.0
            && f.p2p_rel > cfgv.p2p_rel_off;
        if (admissible && self.hops_since_slow >= SLOW_HOPS) || self.slow_p2p.n == 0 {
            self.hops_since_slow = 0;
            self.slow_p2p.push(p2p);
            self.slow_kurt.push(f.kurtosis);
            self.slow_qrs.push(f.qrs_ratio);
        }
        // Flatness is judged against the channel's own *slow* amplitude, never
        // the current window's. Scaling the threshold by the current
        // peak-to-peak made a noisy window look flat, because a larger excursion
        // raised the bar for what counted as motion - the feature measured
        // backwards (AUC 0.23).
        self.flat_eps =
            ((if reference > 1e-9 { reference } else { p2p }) * self.cfg.flat_eps_rel).max(1e-7);

        let cfg = &self.cfg;
        let mut flg = 0u8;
        if f.sat_frac > cfg.sat_frac_bad {
            flg |= flags::SATURATION;
        }
        if f.flat_frac > cfg.flat_frac_bad {
            flg |= flags::FLATLINE;
        }
        if f.hf_ratio > cfg.hf_warn {
            flg |= flags::EMG;
        }
        if f.base_ratio > cfg.base_warn {
            flg |= flags::BASELINE;
        }
        if f.kurtosis < cfg.kurtosis_warn && f.kurtosis_rel < cfg.rel_warn {
            flg |= flags::UNSTABLE;
        }
        if f.p2p_rel < cfg.p2p_rel_low || f.p2p_rel > cfg.p2p_rel_high {
            flg |= flags::LOW_AMP;
        }
        if f.p2p_rel < cfg.p2p_rel_off {
            flg |= flags::LEAD_OFF;
        }

        QualitySample {
            flags: flg,
            score: score(&f, cfg),
            features: f,
        }
    }

    pub fn reset(&mut self) {
        self.clean.reset();
        self.base.reset();
        self.hf.reset();
        self.qrs.reset();
        self.sat.reset();
        self.flat.reset();
        self.sums = Sums::default();
        self.pp.reset();
        let cap = self.slow_p2p.buf.len();
        self.slow_p2p = SlowMedian::new(cap);
        self.slow_kurt = SlowMedian::new(cap);
        self.slow_qrs = SlowMedian::new(cap);
        self.since_hop = 0;
        self.hops_since_refresh = 0;
        self.hops_since_slow = SLOW_HOPS;
        self.n = 0;
        self.prev_clean = 0.0;
        self.flat_eps = 1e-7;
        self.last = QualitySample {
            flags: 0,
            score: 1.0,
            features: QualityFeatures::default(),
        };
    }
}

/// Combine the features into one 0..1 score.
///
/// Two kinds of evidence, combined differently on purpose.
///
/// **Vetoes** multiply. Rail contact, a flatlined lead and extreme out-of-band
/// power each make a window unusable on their own, and nothing else can redeem
/// it.
///
/// **Soft evidence** is averaged. Kurtosis, QRS-band share and relative
/// amplitude each have a legitimate low range that depends on the rhythm rather
/// than on quality, so any one of them can be low on a perfectly clean
/// recording. Multiplying them, as an earlier version did, let a single term
/// veto the other two: measured on MIT-BIH Supraventricular record 801 - clean,
/// bigeminal, and less leptokurtic than sinus for that reason alone - 79% of
/// seconds were condemned. Averaging requires agreement before a window is
/// thrown away, and real artefact degrades all three at once.
///
/// Only features whose orientation has been **measured** contribute. Against the
/// MIT noise-stress protocol, kurtosis, QRS-band share and relative amplitude
/// each separate noisy seconds from clean ones; `hf_ratio` and `base_ratio` did
/// not, because that corpus's added noise is electrode motion, which lives
/// inside the analysis band. Those two stay as vetoes at extreme values and as
/// flags, so strong mains or wander still condemns a window, but they cannot
/// drag the score on noise they do not describe.
pub fn score(f: &QualityFeatures, cfg: &QualityConfig) -> f32 {
    // Shape features pass on either test: absolutely spiky enough to be ECG, or
    // normal for this channel.
    let s_kurt = ramp(f.kurtosis, cfg.kurtosis_bad, cfg.kurtosis_warn).max(ramp(
        f.kurtosis_rel,
        cfg.rel_bad,
        cfg.rel_warn,
    ));
    let s_qrs = ramp(f.qrs_ratio, cfg.qrs_ratio_bad, cfg.qrs_ratio_warn).max(ramp(
        f.qrs_ratio_rel,
        cfg.rel_bad,
        cfg.rel_warn,
    ));
    let s_amp = ramp(f.p2p_rel, cfg.p2p_rel_low * 0.5, cfg.p2p_rel_low)
        * (1.0 - ramp(f.p2p_rel, cfg.p2p_rel_high, cfg.p2p_rel_high * 2.0));

    let soft = cfg.w_kurtosis * s_kurt + cfg.w_qrs_ratio * s_qrs + cfg.w_amplitude * s_amp;
    let soft = soft / (cfg.w_kurtosis + cfg.w_qrs_ratio + cfg.w_amplitude).max(1e-6);

    // Every veto is a *threshold*, not a linear penalty. Subtracting a fraction
    // directly turns a property of healthy signal into a proportional demerit:
    // a clean trace spends most of each beat on the isoelectric line, so its
    // flat fraction is high by nature - measured against human annotation, the
    // median is 0.69 for full-quality seconds against 0.39 for degraded ones.
    // Multiplying by `1 - flat_frac` therefore scored the best signal *worst*,
    // and inverted the whole score (AUC 0.23 for degraded-versus-clean).
    let veto = (1.0 - ramp(f.hf_ratio, cfg.hf_bad, 1.0))
        * (1.0 - ramp(f.base_ratio, cfg.base_bad, 1.0))
        * (1.0 - ramp(f.sat_frac, cfg.sat_frac_bad * 0.5, cfg.sat_frac_bad))
        * (1.0 - ramp(f.flat_frac, cfg.flat_frac_bad * 0.8, cfg.flat_frac_bad))
        * ramp(f.p2p_rel, cfg.p2p_rel_off * 0.5, cfg.p2p_rel_off);

    (soft * veto).clamp(0.0, 1.0)
}

/// 0 at or below `lo`, 1 at or above `hi`, linear between.
#[inline]
fn ramp(x: f32, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        return if x >= hi { 1.0 } else { 0.0 };
    }
    ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
}
