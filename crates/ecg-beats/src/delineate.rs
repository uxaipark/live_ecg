//! PQRST delineation: the boundaries of each wave, and whether a P wave is
//! there at all.
//!
//! # Why this exists
//!
//! Three separate limitations in this engine turned out to be the same missing
//! piece of evidence:
//!
//! * **Supraventricular beats** are defined by an early *atrial* depolarisation.
//!   Without seeing the P wave the detector has only prematurity to go on, and
//!   prematurity alone reaches an AUC of 0.72 on MIT-BIH.
//! * **The atrial-fibrillation false-alarm tail** is marked respiratory sinus
//!   arrhythmia — a rhythm that is irregular *and has P waves*, which is exactly
//!   what fibrillation is not.
//! * **The quality monitor cannot separate** "every wave readable" from "QRS
//!   reliable only", because that distinction is about P and T waves and it
//!   measures neither.
//!
//! So the output that matters downstream is less the millisecond boundaries
//! than **whether a P wave precedes this beat, and by how much**.
//!
//! # Method
//!
//! Boundaries are found by searching the band each wave actually lives in:
//! the QRS complex in 5–20 Hz, the P and T waves in 0.5–10 Hz, where the complex
//! is attenuated and they are not. Onsets and offsets are the points where the
//! wave's own energy falls below a fraction of its peak, walked outward from the
//! peak rather than thresholded across the window — a neighbouring wave would
//! otherwise extend the one being measured.
//!
//! Search windows are proportions of the surrounding RR intervals, never fixed
//! durations: at 40 beats per minute a T wave ends where at 150 the next P wave
//! has already begun.
//!
//! Delineation lags the detector by one beat, for the same reason classification
//! does — the interval *after* a beat bounds where its T wave can be.

use ecg_dsp::{ms_to_samples, MovingAverage, Ring};

/// One wave's extent. All positions are on the input time base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wave {
    pub onset: u64,
    pub peak: u64,
    pub offset: u64,
}

impl Wave {
    pub fn duration_samples(&self) -> u64 {
        self.offset.saturating_sub(self.onset)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Delineation {
    /// The R peak this describes.
    pub r: u64,
    pub qrs: Wave,
    /// The P wave preceding this beat, when one was found.
    pub p: Option<Wave>,
    /// Amplitude of that P wave over the window's noise floor. The number the
    /// downstream detectors actually consume: "is there atrial activity here".
    pub p_confidence: f32,
    pub t: Option<Wave>,
    /// PR interval in milliseconds, from P onset to QRS onset.
    pub pr_ms: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct DelineateConfig {
    pub fs: f64,
    /// Fraction of the QRS envelope's peak that bounds the complex.
    pub qrs_boundary_frac: f32,
    /// Fraction of a P or T wave's peak magnitude that bounds it.
    pub pt_boundary_frac: f32,
    /// Width of the moving average that turns the QRS band into an envelope.
    ///
    /// The band-passed complex is a damped oscillation with several zero
    /// crossings inside it, so walking outward on |x| until it falls below a
    /// fraction of the peak stops at the first crossing - about 30 ms from the
    /// R peak, roughly a third of a real QRS. The envelope has no interior
    /// zeros, so the walk ends where the complex does.
    pub qrs_env_ms: f64,
    /// Half-width of the search for the QRS-band peak that anchors the complex.
    pub qrs_anchor_ms: f64,
    /// Largest QRS half-width considered.
    pub qrs_search_ms: f64,
    /// Search windows for the P and T waves, as offsets from the R peak.
    ///
    /// Anchored on the fiducial rather than on the QRS boundaries: the fiducial
    /// is the most reliable mark the engine produces, and hanging the P and T
    /// searches off a boundary makes every error in that boundary an error in
    /// two more waves. A few milliseconds of QRS-offset movement was worth
    /// twenty points of T-wave detection.
    ///
    /// The near edge is whichever is later of a fixed guard and a share of the
    /// interval, so the complex is cleared at every heart rate.
    pub t_guard_ms: f64,
    pub t_guard_frac: f32,
    pub t_window_frac: f32,
    pub t_window_max_ms: f64,
    pub p_guard_ms: f64,
    pub p_guard_frac: f32,
    pub p_window_frac: f32,
    pub p_window_max_ms: f64,
    /// A P wave must stand this far above the window's noise floor.
    pub p_min_confidence: f32,
    /// Window before QRS onset from which the isoelectric level is taken.
    ///
    /// The T wave's own search window is mostly T wave, so its median is not
    /// baseline - it sits part-way up the wave, and a tangent extrapolated to it
    /// stops short. The PQ segment is the level the wave actually returns to,
    /// and is what a reader measures against.
    pub iso_from_ms: f64,
    pub iso_to_ms: f64,
    /// Stop a boundary walk where the wave stops falling, not only where it
    /// falls below the threshold.
    ///
    /// A P wave sitting on the tail of the preceding T never returns to the
    /// threshold on its left-hand side, so the walk runs back through the
    /// valley between them and into the T wave. The valley is the boundary.
    pub valley_stop: bool,
    /// Place the T offset where the tangent at the wave's steepest descent
    /// meets the baseline, rather than where its amplitude crosses a fraction
    /// of the peak. The T wave ends by flattening out, so a fixed fraction of
    /// an amplitude that varies fivefold between subjects cuts it short.
    pub t_tangent: bool,
    /// Frequencies at which each tap's group delay is evaluated - the centre of
    /// each wave's own energy, and so the delay that wave actually incurs.
    ///
    /// They are not interchangeable and they are not guesses: each was chosen
    /// as the value that puts the *median* error of that wave's peak at zero on
    /// the development half of LUDB. Fitting them against a peak rather than a
    /// boundary keeps this a timing question - a boundary also carries whatever
    /// the threshold criterion does, and absorbing that into a delay would make
    /// the delay wrong for everything else the tap is used for.
    pub qrs_ref_hz: f64,
    pub p_ref_hz: f64,
    pub t_ref_hz: f64,
}

impl DelineateConfig {
    pub fn new(fs: f64) -> Self {
        DelineateConfig {
            fs,
            qrs_boundary_frac: 0.15,
            pt_boundary_frac: 0.15,
            qrs_env_ms: 24.0,
            qrs_anchor_ms: 70.0,
            qrs_search_ms: 120.0,
            t_guard_ms: 140.0,
            t_guard_frac: 0.15,
            t_window_frac: 0.60,
            t_window_max_ms: 520.0,
            p_guard_ms: 90.0,
            p_guard_frac: 0.10,
            p_window_frac: 0.45,
            p_window_max_ms: 360.0,
            p_min_confidence: 2.5,
            iso_from_ms: 60.0,
            iso_to_ms: 10.0,
            valley_stop: true,
            t_tangent: true,
            qrs_ref_hz: 10.0,
            p_ref_hz: 10.0,
            t_ref_hz: 6.0,
        }
    }
}

/// Which tap and which delay a search runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tap {
    P,
    T,
}

pub struct Delineator {
    cfg: DelineateConfig,
    /// QRS envelope and the P/T band, kept long enough to look back a whole beat.
    qrs_env: Ring,
    env: MovingAverage,
    pt_band: Ring,
    n: u64,
    floor: u64,
    /// Delay of each tap relative to the input time base, in samples. A mark at
    /// input time `t` sits at ring index `t + delay`; nothing else in this file
    /// is allowed to touch a ring index directly.
    qrs_delay: u64,
    p_delay: u64,
    t_delay: u64,
    /// Offset of the previous beat's T wave, so a P search cannot reach into it.
    prev_t_offset: Option<u64>,
}

impl Delineator {
    pub fn new(cfg: DelineateConfig) -> Self {
        let cap = ms_to_samples(cfg.fs, 4000.0);
        Delineator {
            qrs_env: Ring::with_capacity(cap),
            env: MovingAverage::new(ms_to_samples(cfg.fs, cfg.qrs_env_ms).max(1)),
            pt_band: Ring::with_capacity(cap),
            n: 0,
            floor: 0,
            qrs_delay: 0,
            p_delay: 0,
            t_delay: 0,
            prev_t_offset: None,
            cfg,
        }
    }

    pub fn config(&self) -> &DelineateConfig {
        &self.cfg
    }

    /// Tell the delineator how far each tap lags the input, in samples.
    ///
    /// `qrs` is the delay of the QRS band *before* the envelope; the averager's
    /// own half-window is added here, because it is this object's own doing.
    pub fn set_delays(&mut self, qrs: f64, p: f64, t: f64) {
        let env_half = (self.env.len().saturating_sub(1)) as f64 / 2.0;
        self.qrs_delay = (qrs + env_half).round().max(0.0) as u64;
        self.p_delay = p.round().max(0.0) as u64;
        self.t_delay = t.round().max(0.0) as u64;
    }

    #[inline]
    pub fn push_sample(&mut self, qrs: f32, pt: f32) {
        self.qrs_env.push(self.env.process(qrs.abs()));
        self.pt_band.push(pt);
        self.n += 1;
        if self.n > self.qrs_env.capacity() as u64 {
            self.floor = self.n - self.qrs_env.capacity() as u64;
        }
    }

    /// Delineate the beat at `r`, given the intervals either side of it.
    ///
    /// `rr_prev` and `rr_post` are in samples; either may be `None` at the edges
    /// of a stream, in which case the corresponding search is skipped rather
    /// than run over a guessed window.
    pub fn delineate(
        &mut self,
        r: u64,
        rr_prev: Option<u64>,
        rr_post: Option<u64>,
    ) -> Option<Delineation> {
        let qrs = self.qrs_bounds(r)?;

        let iso = self.isoelectric(qrs.onset, Tap::T);
        let t = rr_post.and_then(|rr| self.find_t(r, rr, iso));
        let (p, p_confidence) = match rr_prev {
            Some(rr) => self.find_p(r, rr),
            None => (None, 0.0),
        };
        self.prev_t_offset = t.map(|w| w.offset);

        let to_ms = 1000.0 / self.cfg.fs as f32;
        let pr_ms = p.map(|w| (qrs.onset.saturating_sub(w.onset)) as f32 * to_ms);

        Some(Delineation {
            r,
            qrs,
            p,
            p_confidence,
            t,
            pr_ms,
        })
    }

    /// QRS envelope at input time `t`, if that time is still in the ring.
    #[inline]
    fn qrs_at(&self, t: u64) -> f32 {
        self.qrs_env.at(t + self.qrs_delay)
    }

    #[inline]
    fn pt_at(&self, t: u64, tap: Tap) -> f32 {
        self.pt_band.at(t + self.delay_of(tap))
    }

    #[inline]
    fn delay_of(&self, tap: Tap) -> u64 {
        match tap {
            Tap::P => self.p_delay,
            Tap::T => self.t_delay,
        }
    }

    /// Input-time range readable on a tap with delay `d`.
    fn span(&self, d: u64) -> Option<(u64, u64)> {
        let lo = self.floor.saturating_sub(d);
        let hi = (self.n.saturating_sub(1)).checked_sub(d)?;
        if hi <= lo {
            None
        } else {
            Some((lo, hi))
        }
    }

    /// QRS onset and offset, anchored on the envelope's own peak.
    fn qrs_bounds(&self, r: u64) -> Option<Wave> {
        let (v_lo, v_hi) = self.span(self.qrs_delay)?;
        if r < v_lo || r > v_hi {
            return None;
        }
        let anchor_half = ms_to_samples(self.cfg.fs, self.cfg.qrs_anchor_ms) as u64;
        let search = ms_to_samples(self.cfg.fs, self.cfg.qrs_search_ms) as u64;
        let lo = r.saturating_sub(search).max(v_lo);
        let hi = (r + search).min(v_hi);
        if hi <= lo {
            return None;
        }
        let (a_lo, a_hi) = (
            r.saturating_sub(anchor_half).max(lo),
            (r + anchor_half).min(hi),
        );
        let mut anchor = a_lo;
        let mut peak = 0.0f32;
        for i in a_lo..=a_hi {
            let v = self.qrs_at(i);
            if v > peak {
                peak = v;
                anchor = i;
            }
        }
        if peak <= 0.0 {
            return None;
        }
        let bound = peak * self.cfg.qrs_boundary_frac;
        let mut onset = anchor;
        while onset > lo && self.qrs_at(onset - 1) > bound {
            onset -= 1;
        }
        let mut offset = anchor;
        while offset < hi && self.qrs_at(offset + 1) > bound {
            offset += 1;
        }
        Some(Wave {
            onset,
            peak: r,
            offset,
        })
    }

    /// The T wave after `r`, within a share of the following interval.
    fn find_t(&self, r: u64, rr_post: u64, iso: Option<f32>) -> Option<Wave> {
        let guard = (ms_to_samples(self.cfg.fs, self.cfg.t_guard_ms) as u64)
            .max((rr_post as f32 * self.cfg.t_guard_frac) as u64);
        let span = ((rr_post as f32 * self.cfg.t_window_frac) as u64)
            .min(ms_to_samples(self.cfg.fs, self.cfg.t_window_max_ms) as u64);
        if span <= guard {
            return None;
        }
        let (_, v_hi) = self.span(self.t_delay)?;
        let lo = r + guard;
        let hi = (r + span).min(v_hi);
        self.wave_in(lo, hi, Tap::T, iso)
    }

    /// The P wave before `r`, within a share of the preceding interval and
    /// never reaching back into the previous beat's T wave.
    fn find_p(&self, r: u64, rr_prev: u64) -> (Option<Wave>, f32) {
        let Some((v_lo, _)) = self.span(self.p_delay) else {
            return (None, 0.0);
        };
        let guard = (ms_to_samples(self.cfg.fs, self.cfg.p_guard_ms) as u64)
            .max((rr_prev as f32 * self.cfg.p_guard_frac) as u64);
        let span = ((rr_prev as f32 * self.cfg.p_window_frac) as u64)
            .min(ms_to_samples(self.cfg.fs, self.cfg.p_window_max_ms) as u64);
        let hi = r.saturating_sub(guard);
        let mut lo = hi.saturating_sub(span).max(v_lo);
        if let Some(t_off) = self.prev_t_offset {
            lo = lo.max(t_off);
        }
        if hi <= lo + 2 {
            return (None, 0.0);
        }
        let Some(wave) = self.wave_in(lo, hi, Tap::P, None) else {
            return (None, 0.0);
        };
        // A P wave is present when its deflection stands clear of what the same
        // stretch of signal does elsewhere: a robust z-score of the peak against
        // the window's own median and median absolute deviation, both of which
        // describe baseline wherever the wave is not.
        let base = self.baseline(lo, hi, Tap::P);
        let amplitude = (self.pt_at(wave.peak, Tap::P) - base).abs();
        let floor = self.spread(lo, hi, Tap::P, base).max(1e-6);
        let confidence = amplitude / floor;
        if confidence < self.cfg.p_min_confidence {
            (None, confidence)
        } else {
            (Some(wave), confidence)
        }
    }

    /// Largest deflection in `[lo, hi]` on the P/T band, with its boundaries.
    /// All four numbers are input times, not ring indices.
    ///
    /// Everything is measured against the window's own median, not against
    /// zero. The tap is a low-pass, so whatever baseline the segment sits on
    /// comes through it intact; judging amplitude by |x| would measure the
    /// offset far more than the wave.
    fn wave_in(&self, lo: u64, hi: u64, tap: Tap, iso: Option<f32>) -> Option<Wave> {
        let (v_lo, v_hi) = self.span(self.delay_of(tap))?;
        if hi <= lo + 2 || hi > v_hi || lo < v_lo {
            return None;
        }
        let base = iso.unwrap_or_else(|| self.baseline(lo, hi, tap));
        let mut peak = lo;
        let mut best = 0.0f32;
        for i in lo..=hi {
            let v = (self.pt_at(i, tap) - base).abs();
            if v > best {
                best = v;
                peak = i;
            }
        }
        if best <= 0.0 {
            return None;
        }
        let bound = best * self.cfg.pt_boundary_frac;
        let onset = self.walk(peak, lo, bound, -1, tap, base);
        let mut offset = self.walk(peak, hi, bound, 1, tap, base);
        if tap == Tap::T && self.cfg.t_tangent {
            offset = self.t_offset_tangent(peak, hi, offset, base);
        }
        Some(Wave {
            onset,
            peak,
            offset,
        })
    }

    /// Walk out from `peak` towards `limit` until the wave has ended.
    ///
    /// It has ended where its excursion from `base` drops below `bound`, or -
    /// when `valley_stop` is on - where it stops falling, whichever comes first.
    #[allow(clippy::too_many_arguments)]
    fn walk(&self, peak: u64, limit: u64, bound: f32, step: i64, tap: Tap, base: f32) -> u64 {
        let mut i = peak;
        let mut prev = (self.pt_at(i, tap) - base).abs();
        loop {
            let next = if step < 0 {
                if i <= limit {
                    return i;
                }
                i - 1
            } else {
                if i >= limit {
                    return i;
                }
                i + 1
            };
            let v = (self.pt_at(next, tap) - base).abs();
            if v <= bound {
                return next;
            }
            // A rise of more than a tenth of the bound: the far side of a valley,
            // not the noise on the way down. The margin keeps a flat tail from
            // being cut at the first upward sample.
            if self.cfg.valley_stop && v > prev + bound * 0.1 {
                return i;
            }
            prev = prev.min(v);
            i = next;
        }
    }

    /// T offset by the tangent construction: the steepest point of the descent,
    /// extrapolated along its own slope back to the baseline.
    fn t_offset_tangent(&self, peak: u64, hi: u64, fallback: u64, base: f32) -> u64 {
        if fallback <= peak + 1 || hi <= peak + 2 {
            return fallback;
        }
        // Search a little past the amplitude-based offset: the steepest descent
        // is inside the wave, but the tangent has to be free to land outside it.
        let end = (fallback + (fallback - peak) / 2).min(hi);
        let sign = (self.pt_at(peak, Tap::T) - base).signum();
        let mut best_i = peak + 1;
        let mut best_slope = 0.0f32;
        for i in (peak + 1)..end {
            // Towards the baseline is a fall in the wave's own direction.
            let slope = sign * (self.pt_at(i, Tap::T) - self.pt_at(i + 1, Tap::T));
            if slope > best_slope {
                best_slope = slope;
                best_i = i;
            }
        }
        if best_slope <= 0.0 {
            return fallback;
        }
        let height = sign * (self.pt_at(best_i, Tap::T) - base);
        if height <= 0.0 {
            return fallback;
        }
        let ahead = (height / best_slope).round().max(0.0) as u64;
        (best_i + ahead).min(hi)
    }

    /// Level of the PQ segment on `tap`: the isoelectric line this beat
    /// returns to. `None` when that stretch has left the ring.
    fn isoelectric(&self, qrs_on: u64, tap: Tap) -> Option<f32> {
        let (v_lo, v_hi) = self.span(self.delay_of(tap))?;
        let hi = qrs_on.saturating_sub(ms_to_samples(self.cfg.fs, self.cfg.iso_to_ms) as u64);
        let lo = qrs_on.saturating_sub(ms_to_samples(self.cfg.fs, self.cfg.iso_from_ms) as u64);
        if lo < v_lo || hi > v_hi || hi <= lo + 1 {
            return None;
        }
        Some(self.baseline(lo, hi, tap))
    }

    /// Median level of a window: the baseline the segment sits on.
    fn baseline(&self, lo: u64, hi: u64, tap: Tap) -> f32 {
        let mut v: Vec<f32> = (lo..=hi).map(|i| self.pt_at(i, tap)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    }

    /// Median absolute deviation from `base` - the window's noise floor.
    fn spread(&self, lo: u64, hi: u64, tap: Tap, base: f32) -> f32 {
        let mut v: Vec<f32> = (lo..=hi)
            .map(|i| (self.pt_at(i, tap) - base).abs())
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    }

    pub fn on_gap(&mut self, unobserved: u64) {
        self.n += unobserved;
        self.floor = self.n;
        self.prev_t_offset = None;
    }

    pub fn reset(&mut self) {
        self.qrs_env.reset();
        self.env.reset();
        self.pt_band.reset();
        self.n = 0;
        self.floor = 0;
        self.prev_t_offset = None;
    }
}
