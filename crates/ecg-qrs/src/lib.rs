//! Streaming QRS detector.
//!
//! Pan-Tompkins topology (band-pass -> derivative -> square -> moving-window
//! integration -> adaptive threshold) with the three refinements that decide
//! whether a detector survives real wearable data:
//!
//! 1. **T-wave discrimination** by slope ratio, so tall T waves after a short RR
//!    are not counted as beats.
//! 2. **Search-back** over the retained integration trace when a beat is overdue,
//!    which recovers low-amplitude beats the running threshold has climbed past.
//! 3. **Fiducial refinement** on the input waveform, because the integration peak
//!    sits roughly half an integration window after the R wave and RR-interval
//!    work downstream (AF, ectopy) cannot absorb that bias.
//!
//! The detector is causal and allocation-free once constructed. Detection latency
//! is reported by [`QrsDetector::latency_samples`].

use ecg_dsp::{ms_to_samples, Cascade, Derivative5, MovingAverage, Ring};

/// Threshold shrink applied each time a search-back finds nothing. Recovering
/// from a large amplitude drop must take a few beats, not a few minutes.
const SEARCHBACK_DECAY: f32 = 0.90;

/// The noise estimate may never reach this fraction of the signal estimate.
///
/// Without the ceiling the detector has an absorbing state. `npk` rises only
/// when a candidate is rejected, and a candidate exists only when the threshold
/// is crossed; once `npk` climbs to or above `spk` the threshold sits above
/// every real beat, no candidate is ever produced, and nothing can bring `npk`
/// back down. Observed on MIT-BIH Normal Sinus record 16272, where the detector
/// fell silent on visibly clean signal - 74% of the record's beats - with `npk`
/// pinned four times above the true beat energy.
const NPK_CEILING: f32 = 0.5;

/// Physiological bounds on an RR interval, in seconds. Intervals outside this
/// range are detection artefacts; letting them into the running average drags
/// the search-back deadline out to tens of seconds and the detector stops
/// looking for the beats it just missed.
const RR_MIN_S: f32 = 0.2;
const RR_MAX_S: f32 = 3.0;

/// A candidate inside the T-wave window is rejected only when it is gentle
/// *and* small: below this share of the recent median QRS slope ...
const TWAVE_SLOPE_RATIO: f32 = 0.5;

/// ... and below this share of the recent median QRS energy.
///
/// The slope test alone cannot be used. A ventricular beat is wide, so its
/// upstroke is genuinely gentler than a normal beat's, and a slope-only rule
/// discards it - measured at 53% of ventricular beats lost on MIT-BIH 208.
/// Width is what makes it gentle, and width is also what makes its integrated
/// area large, so the two tests together separate a wide QRS from a T wave in
/// the way neither does alone.
const TWAVE_ENERGY_RATIO: f32 = 0.5;

#[derive(Debug, Clone, Copy)]
pub struct QrsConfig {
    pub fs: f64,
    /// QRS band. Low edge trades baseline rejection against wide (ventricular)
    /// complexes; high edge trades narrow-QRS sensitivity against EMG pickup.
    pub bp_lo: f64,
    pub bp_hi: f64,
    pub bp_order: usize,
    /// Moving-window integration width; should approximate the widest QRS.
    pub integ_ms: f64,
    /// Absolute blanking after a detection.
    pub refractory_ms: f64,
    /// Below this RR a candidate must pass the slope test to count as a beat.
    pub twave_ms: f64,
    /// Fraction of the signal/noise gap at which the threshold sits.
    pub thr_frac: f32,
    /// Quantile of the recent accepted-beat energies used as the signal-peak
    /// estimate. Beat energies are heterogeneous by nature - a wide ventricular
    /// beat integrates to a fraction of a narrow sinus beat's energy because the
    /// stage squares slope - so a central estimate puts the bar above the very
    /// beats that are hardest to catch. A low quantile drops the bar exactly when
    /// the recent beats disagree and leaves it alone when they do not.
    pub spk_quantile: f32,
    /// A candidate is closed once the integration trace falls to this fraction of
    /// its own peak, even while still above threshold. Without it two beats whose
    /// integration humps overlap - couplets, bigeminy, fast rates - merge into one
    /// detection, which is the dominant miss mode on ventricular records.
    pub peak_drop: f32,
    /// Search-back fires after this multiple of the recent RR average.
    pub searchback_factor: f32,
    /// Half-width of the fiducial search around the expected R position.
    pub refine_halfwidth_ms: f64,
    /// Length of the initial threshold-learning window.
    pub learn_ms: f64,
    /// Residual trim applied to reported R positions, in milliseconds. The bulk
    /// of the correction comes from [`QrsDetector::set_input_delay`]; this is
    /// only for what measurement shows is left over.
    pub fiducial_bias_ms: f64,
}

impl QrsConfig {
    pub fn new(fs: f64) -> Self {
        QrsConfig {
            fs,
            bp_lo: 5.0,
            bp_hi: 20.0,
            bp_order: 2,
            integ_ms: 120.0,
            refractory_ms: 200.0,
            twave_ms: 360.0,
            thr_frac: 0.15,
            spk_quantile: 0.5,
            peak_drop: 0.5,
            searchback_factor: 1.66,
            refine_halfwidth_ms: 40.0,
            learn_ms: 2000.0,
            fiducial_bias_ms: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QrsEvent {
    /// Sample index of the R peak in the detector's input stream.
    pub sample: u64,
    /// Peak amplitude of the input waveform at the fiducial (signed, mV).
    pub amplitude: f32,
    /// Integration-trace peak height that triggered the detection.
    pub energy: f32,
    /// `energy / threshold` at detection; < 1 marks a search-back recovery.
    pub margin: f32,
    /// True when the beat came from search-back rather than a threshold crossing.
    pub recovered: bool,
}

/// Last-N history with a median accessor.
///
/// The running signal-peak estimate is a median, not an exponential mean: one
/// artefact spike or one tall ventricular beat moves a mean far enough to hide
/// the next several normal beats under the raised threshold, and that shows up
/// directly as missed beats on the ventricular records.
#[derive(Debug, Clone, Copy)]
struct History<const N: usize> {
    buf: [f32; N],
    n: usize,
    idx: usize,
}

impl<const N: usize> History<N> {
    fn new() -> Self {
        History {
            buf: [0.0; N],
            n: 0,
            idx: 0,
        }
    }

    fn push(&mut self, v: f32) {
        self.buf[self.idx] = v;
        self.idx = (self.idx + 1) % N;
        self.n = (self.n + 1).min(N);
    }

    fn median(&self) -> f32 {
        self.quantile(0.5)
    }

    /// `q` = 0 is the minimum, 0.5 the median.
    fn quantile(&self, q: f32) -> f32 {
        if self.n == 0 {
            return 0.0;
        }
        let mut t = [0.0f32; N];
        t[..self.n].copy_from_slice(&self.buf[..self.n]);
        t[..self.n].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let i = (q * (self.n - 1) as f32).round() as usize;
        t[i.min(self.n - 1)]
    }

    fn is_empty(&self) -> bool {
        self.n == 0
    }

    fn clear(&mut self) {
        self.n = 0;
        self.idx = 0;
        self.buf = [0.0; N];
    }
}

/// Rolling RR statistics driving the adaptive windows.
#[derive(Debug, Clone)]
struct RrTracker {
    recent: [f32; 8],
    stable: [f32; 8],
    n_recent: usize,
    n_stable: usize,
    idx_recent: usize,
    idx_stable: usize,
    min_rr: f32,
    max_rr: f32,
    pub avg_recent: f32,
    pub avg_stable: f32,
}

impl RrTracker {
    fn new(default_rr: f32, fs: f32) -> Self {
        RrTracker {
            recent: [default_rr; 8],
            stable: [default_rr; 8],
            n_recent: 0,
            n_stable: 0,
            idx_recent: 0,
            idx_stable: 0,
            min_rr: RR_MIN_S * fs,
            max_rr: RR_MAX_S * fs,
            avg_recent: default_rr,
            avg_stable: default_rr,
        }
    }

    fn push(&mut self, rr: f32) {
        // An interval outside the physiological range is a missed or spurious
        // detection, not a heartbeat. Averaging it in pushes the search-back
        // deadline past the point of usefulness.
        if rr < self.min_rr || rr > self.max_rr {
            return;
        }
        self.recent[self.idx_recent] = rr;
        self.idx_recent = (self.idx_recent + 1) & 7;
        self.n_recent = (self.n_recent + 1).min(8);
        self.avg_recent = mean(&self.recent[..self.n_recent]);

        // The "stable" average deliberately ignores RRs outside a normal-sinus
        // band, so a run of ectopy cannot drag the miss-limit with it.
        if rr >= 0.92 * self.avg_stable && rr <= 1.16 * self.avg_stable {
            self.stable[self.idx_stable] = rr;
            self.idx_stable = (self.idx_stable + 1) & 7;
            self.n_stable = (self.n_stable + 1).min(8);
            self.avg_stable = mean(&self.stable[..self.n_stable]);
        }
    }
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f32>() / xs.len() as f32
}

/// A snapshot of the adaptive state. Diagnostic only.
#[derive(Debug, Clone, Copy)]
pub struct DetectorState {
    pub threshold: f32,
    pub spk: f32,
    pub npk: f32,
    pub spk_scale: f32,
    pub rr_stable: f32,
    pub in_peak: bool,
    pub samples_since_qrs: u64,
}

pub struct QrsDetector {
    cfg: QrsConfig,
    bp: Cascade,
    deriv: Derivative5,
    integ: MovingAverage,

    x_ring: Ring,     // detector input, for fiducial refinement
    int_ring: Ring,   // integration trace, for search-back
    slope_ring: Ring, // |derivative|, for T-wave discrimination

    n: u64,
    integ_len: usize,
    refractory: usize,
    twave: usize,
    refine_hw: usize,
    learn_len: usize,
    max_candidate: usize,

    // adaptive threshold state
    spk_hist: History<8>,
    spk_scale: f32,
    npk: f32,
    learn_max: f32,
    learn_sum: f64,
    learning: bool,

    // crossing state
    in_peak: bool,
    peak_val: f32,
    peak_idx: u64,

    /// Delay of the input stream relative to the true signal, in samples.
    input_delay: f64,
    bias_samples: f64,

    last_qrs: Option<u64>,
    slope_hist: History<8>,
    rr: RrTracker,
    /// Newest sample index guaranteed to still be in `x_ring`.
    ring_floor: u64,
    /// Most recent quality verdict, which decides whether the threshold is
    /// allowed to come back down. See [`QrsDetector::threshold`].
    quality_ok: bool,
}

impl QrsDetector {
    pub fn new(cfg: QrsConfig) -> Self {
        let fs = cfg.fs;
        let integ_len = ms_to_samples(fs, cfg.integ_ms);
        let refractory = ms_to_samples(fs, cfg.refractory_ms);
        let twave = ms_to_samples(fs, cfg.twave_ms);
        let refine_hw = ms_to_samples(fs, cfg.refine_halfwidth_ms);
        let learn_len = ms_to_samples(fs, cfg.learn_ms);
        // Search-back must be able to reach the whole miss interval plus a margin.
        let ring_len = ms_to_samples(fs, 4000.0);

        QrsDetector {
            cfg,
            bp: Cascade::band_pass(fs, cfg.bp_lo, cfg.bp_hi.min(fs * 0.45), cfg.bp_order),
            deriv: Derivative5::new(fs),
            integ: MovingAverage::new(integ_len),
            x_ring: Ring::with_capacity(ring_len),
            int_ring: Ring::with_capacity(ring_len),
            slope_ring: Ring::with_capacity(ring_len),
            n: 0,
            integ_len,
            refractory,
            twave,
            refine_hw,
            learn_len,
            max_candidate: integ_len + refractory,
            spk_hist: History::new(),
            spk_scale: 1.0,
            npk: 0.0,
            learn_max: 0.0,
            learn_sum: 0.0,
            learning: true,
            in_peak: false,
            peak_val: 0.0,
            peak_idx: 0,
            input_delay: 0.0,
            bias_samples: cfg.fiducial_bias_ms * fs / 1000.0,
            last_qrs: None,
            slope_hist: History::new(),
            rr: RrTracker::new(fs as f32 * 0.8, fs as f32),
            ring_floor: 0,
            quality_ok: true,
        }
    }

    pub fn config(&self) -> &QrsConfig {
        &self.cfg
    }

    /// Tell the detector how far its input already lags the raw signal, so
    /// reported R positions land on the original time base.
    pub fn set_input_delay(&mut self, samples: f64) {
        self.input_delay = samples;
    }

    /// Adaptive state, for tracing why the detector is or is not firing.
    pub fn state(&self) -> DetectorState {
        DetectorState {
            threshold: self.threshold(),
            spk: self.spk(),
            npk: self.npk,
            spk_scale: self.spk_scale,
            rr_stable: self.rr.avg_stable,
            in_peak: self.in_peak,
            samples_since_qrs: self.last_qrs.map(|l| self.n.saturating_sub(l)).unwrap_or(0),
        }
    }

    /// Worst-case delay from the R wave to the event being emitted.
    pub fn latency_samples(&self) -> usize {
        self.integ_len + self.refine_hw
    }

    #[inline]
    fn spk(&self) -> f32 {
        self.spk_hist.quantile(self.cfg.spk_quantile) * self.spk_scale
    }

    /// The detection threshold.
    ///
    /// The noise-estimate ceiling applies **only while the quality monitor
    /// believes the signal**, and that asymmetry is the point:
    ///
    /// * On signal we believe, the detector must never be able to go deaf. `npk`
    ///   rises only when a candidate is rejected, and a candidate exists only
    ///   when the threshold is crossed, so once `npk` reaches `spk` nothing can
    ///   ever bring it back - an absorbing state that silenced 74% of MIT-BIH
    ///   Normal Sinus record 16272 on visibly clean signal.
    /// * On signal we do not believe, staying deaf is the correct behaviour. A
    ///   high `npk` through a noise burst or a dead lead is what stops the
    ///   detector inventing beats; forcing it down unconditionally cost 9 points
    ///   of precision across the Normal Sinus corpus.
    ///
    /// The quality monitor is what separates the two cases, which is most of why
    /// it runs ahead of the detector rather than beside it.
    #[inline]
    fn threshold(&self) -> f32 {
        let spk = self.spk();
        let npk = if self.quality_ok {
            self.npk.min(spk * NPK_CEILING)
        } else {
            self.npk
        };
        if spk > npk {
            npk + self.cfg.thr_frac * (spk - npk)
        } else {
            npk
        }
    }

    /// Feed one sample. `quality_ok` gates threshold adaptation: during a noise
    /// burst the detector keeps running but refuses to learn from it, so the
    /// thresholds that come out the far side are still the ones that fit the
    /// patient rather than the artefact.
    pub fn process(&mut self, x: f32, quality_ok: bool, out: &mut Vec<QrsEvent>) {
        let bp = self.bp.process(x);
        self.step(x, bp, quality_ok, out);
    }

    /// Same, but with the QRS band already filtered by the caller. The pipeline
    /// uses this so one band-pass serves both the detector and the quality
    /// monitor instead of each running its own copy.
    pub fn process_prefiltered(
        &mut self,
        x: f32,
        bp: f32,
        quality_ok: bool,
        out: &mut Vec<QrsEvent>,
    ) {
        self.step(x, bp, quality_ok, out);
    }

    #[inline]
    fn step(&mut self, x: f32, bp: f32, quality_ok: bool, out: &mut Vec<QrsEvent>) {
        self.quality_ok = quality_ok;
        let d = self.deriv.process(bp);
        let y = self.integ.process(d * d);

        self.x_ring.push(x);
        self.int_ring.push(y);
        self.slope_ring.push(d.abs());
        let n = self.n;
        self.n += 1;
        if self.n > self.x_ring.capacity() as u64 {
            self.ring_floor = self.n - self.x_ring.capacity() as u64;
        }

        if self.learning {
            self.learn_max = self.learn_max.max(y);
            self.learn_sum += y as f64;
            if self.n >= self.learn_len as u64 {
                let spk = (0.25 * self.learn_max).max(1e-12);
                self.npk = (0.5 * (self.learn_sum / self.n as f64) as f32).min(0.5 * spk);
                self.spk_hist.push(spk);
                self.spk_scale = 1.0;
                self.learning = false;
            }
            return;
        }

        let thr = self.threshold();

        if !self.in_peak {
            if y > thr {
                self.in_peak = true;
                self.peak_val = y;
                self.peak_idx = n;
            }
        } else if y > self.peak_val {
            self.peak_val = y;
            self.peak_idx = n;
        } else if y < thr
            || y < self.peak_val * self.cfg.peak_drop
            || n - self.peak_idx > self.max_candidate as u64
        {
            // The third condition is a liveness guarantee, not a tuning knob. An
            // integration trace that hovers between the threshold and the
            // peak-drop bound satisfies neither exit test, and because
            // search-back stands down while a candidate is open, the detector
            // would stop emitting entirely. Measured on the MIT-BIH Normal Sinus
            // record 16272 that silenced whole minutes at a time, 74% of the
            // record's beats. A candidate cannot outlive the widest QRS plus a
            // refractory period, so it is closed at that point regardless.
            self.in_peak = false;
            let (idx, val) = (self.peak_idx, self.peak_val);
            self.judge(idx, val, thr, quality_ok, false, out);
        }

        self.search_back(n, quality_ok, out);
    }

    /// Re-examine the retained integration trace when a beat is overdue.
    fn search_back(&mut self, n: u64, quality_ok: bool, out: &mut Vec<QrsEvent>) {
        let Some(last) = self.last_qrs else { return };
        let miss = (self.rr.avg_stable * self.cfg.searchback_factor) as u64;
        if n <= last + miss.max(self.refractory as u64) {
            return;
        }
        // A crossing in progress will resolve on its own; do not pre-empt it.
        if self.in_peak {
            return;
        }

        let from = (last + self.refractory as u64).max(self.ring_floor);
        let to = n.saturating_sub(self.integ_len as u64 / 2);
        if to <= from {
            // Nothing left to rescan: let the threshold relax so the next real
            // beat can clear it instead of waiting for a search-back that the
            // window can never satisfy.
            self.relax();
            self.last_qrs = Some(n.saturating_sub(miss / 2));
            return;
        }

        let half_thr = self.threshold() * 0.5;
        let (idx, val) = self.int_ring.argmax(from, to);
        if val > half_thr {
            self.judge(idx, val, half_thr, quality_ok, true, out);
        } else {
            self.relax();
            self.last_qrs = Some(n.saturating_sub(miss / 2));
        }
    }

    /// A beat is overdue and nothing was found: lower the signal estimate so the
    /// next real beat can clear the bar.
    ///
    /// Only `spk` moves. `npk` is pinned below it by [`NPK_CEILING`], so the
    /// threshold falls with `spk` anyway, and decaying `npk` as well removes the
    /// only thing stopping the threshold from chasing the background down during
    /// a genuine pause - precision on the Normal Sinus corpus fell from 99.6% to
    /// 90.5% when it did.
    #[inline]
    fn relax(&mut self) {
        // Only chase a beat we have reason to think is there.
        if self.quality_ok {
            self.spk_scale *= SEARCHBACK_DECAY;
        }
    }

    fn judge(
        &mut self,
        idx: u64,
        val: f32,
        thr: f32,
        quality_ok: bool,
        recovered: bool,
        out: &mut Vec<QrsEvent>,
    ) {
        if let Some(last) = self.last_qrs {
            if idx <= last + self.refractory as u64 {
                return; // inside the absolute blanking window
            }
            let rr = (idx - last) as f32;
            if rr < self.twave as f32 && self.looks_like_twave(idx, val) {
                // Rejected as a T wave: it is still evidence about the noise floor.
                if quality_ok {
                    self.npk = 0.125 * val + 0.875 * self.npk;
                }
                return;
            }
            self.rr.push(rr);
        }

        let (sample, amplitude) = self.refine(idx);
        let slope = self.max_slope(idx);

        if quality_ok {
            self.spk_hist.push(val);
            self.slope_hist.push(slope);
            // A beat cleared the bar, so the decay that was hunting for it is done.
            self.spk_scale = 1.0;
        }
        self.last_qrs = Some(idx);

        out.push(QrsEvent {
            sample,
            amplitude,
            energy: val,
            margin: if thr > 0.0 { val / thr } else { f32::INFINITY },
            recovered,
        });
    }

    /// Is this early candidate the previous beat's T wave rather than a beat?
    ///
    /// Both tests must agree. Compared against medians of recent beats rather
    /// than the single previous one, so one ventricular beat does not disqualify
    /// the beat after it.
    fn looks_like_twave(&self, idx: u64, energy: f32) -> bool {
        if self.slope_hist.is_empty() || self.spk_hist.is_empty() {
            return false;
        }
        let slope_ref = self.slope_hist.median();
        let energy_ref = self.spk_hist.median();
        if slope_ref <= 0.0 || energy_ref <= 0.0 {
            return false;
        }
        let gentle = self.max_slope(idx) < TWAVE_SLOPE_RATIO * slope_ref;
        let small = energy < TWAVE_ENERGY_RATIO * energy_ref;
        gentle && small
    }

    fn max_slope(&self, idx: u64) -> f32 {
        let from = idx
            .saturating_sub(self.integ_len as u64)
            .max(self.ring_floor);
        let to = (idx + 1).min(self.n);
        if to <= from {
            return 0.0;
        }
        self.slope_ring.argmax(from, to).1
    }

    /// Place the fiducial on the input waveform. The integration peak trails the
    /// R wave by about half an integration window; the search spans that offset
    /// plus the refinement half-width on either side.
    fn refine(&self, idx: u64) -> (u64, f32) {
        let half = self.integ_len as u64 / 2;
        let lo = idx
            .saturating_sub(self.integ_len as u64 + self.refine_hw as u64)
            .max(self.ring_floor);
        let hi = (idx.saturating_sub(half) + self.refine_hw as u64 + 1).min(self.n);
        if hi <= lo {
            return (idx, 0.0);
        }
        let (peak, _) = self.x_ring.argmax_abs(lo, hi);
        let amp = self.x_ring.at(peak);
        let shift = (self.input_delay + self.bias_samples).round() as i64;
        let corrected = (peak as i64 - shift).max(0) as u64;
        (corrected, amp)
    }

    /// Samples were lost. The retained traces no longer describe a continuous
    /// stretch, so search-back must not reach across the gap - it would find a
    /// "beat" made of two unrelated halves. The adaptive thresholds are kept:
    /// they describe the patient, and the patient did not change.
    /// `unobserved` is how many samples passed without being seen.
    pub fn on_gap(&mut self, unobserved: u64) {
        // The sample counter is the time base, so it advances *through* the gap
        // rather than restarting. Resetting it would make every later R position
        // wrong by the length of the gap, and nothing downstream would notice.
        // History is invalidated by raising the floor instead.
        self.n += unobserved;
        self.ring_floor = self.n;
        self.bp.reset();
        self.deriv.reset();
        self.integ.reset();
        self.in_peak = false;
        self.last_qrs = None;
    }

    pub fn reset(&mut self) {
        self.bp.reset();
        self.deriv.reset();
        self.integ.reset();
        self.x_ring.reset();
        self.int_ring.reset();
        self.slope_ring.reset();
        self.n = 0;
        self.spk_hist.clear();
        self.spk_scale = 1.0;
        self.npk = 0.0;
        self.learn_max = 0.0;
        self.learn_sum = 0.0;
        self.learning = true;
        self.in_peak = false;
        self.last_qrs = None;
        self.slope_hist.clear();
        self.ring_floor = 0;
        self.rr = RrTracker::new(self.cfg.fs as f32 * 0.8, self.cfg.fs as f32);
    }
}
