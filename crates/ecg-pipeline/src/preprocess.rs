//! Front-end filter bank.
//!
//! One pass over the sample produces five co-registered outputs:
//!
//! * `baseline` — everything below `hp_hz` (respiration, electrode motion).
//! * `clean`    — the analysis band, mains-notched. Morphology uses this.
//! * `hf`       — everything above `hf_hz` (EMG, contact chatter, switching noise).
//! * `qrs`      — the 5-20 Hz sub-band the detector and the QRS envelope run on.
//! * `pt`       — the low-pass tap the P and T waves are measured on.
//!
//! Noise detection needs the LF and HF components anyway, so they are split out
//! here instead of being filtered a second time downstream. Every tap is
//! produced by subtraction and cascade from one chain, never by re-running the
//! input, so they stay sample-aligned by construction.
//!
//! Aligned is not the same as simultaneous. Each tap lags the input by its own
//! group delay, and a downstream stage that marks a position on one tap and
//! reports it on the input time base has to take that delay back out. The
//! delays are computed from the coefficients actually in use rather than tuned,
//! so they stay right across sample rates and corner changes.

use ecg_dsp::{Biquad, Cascade};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mains {
    Off,
    Fixed50,
    Fixed60,
    /// Pick 50 or 60 Hz from the signal itself, then lock the choice.
    Auto,
}

#[derive(Debug, Clone, Copy)]
pub struct PreprocessConfig {
    pub fs: f64,
    /// Baseline-wander corner. Above ~0.67 Hz the ST segment starts to distort,
    /// which matters for morphology even though it helps detection.
    pub hp_hz: f64,
    pub hp_order: usize,
    /// Analysis-band upper corner.
    pub lp_hz: f64,
    pub lp_order: usize,
    pub mains: Mains,
    pub notch_bw_hz: f64,
    /// Corner of the high-frequency residual tap used for EMG/noise features.
    pub hf_hz: f64,
    /// QRS sub-band tap. The detector consumes this directly instead of running
    /// its own copy of the same filter, and the quality monitor uses its power.
    pub qrs_lo: f64,
    pub qrs_hi: f64,
    pub qrs_order: usize,
    /// Low-frequency tap for P and T waves.
    ///
    /// They are slow, low-amplitude deflections, and the analysis band's upper
    /// corner leaves enough QRS energy in to swamp them. Delineation needs a
    /// band where the complex is attenuated and the waves either side of it are
    /// not.
    ///
    /// `pt_lo` of zero means low-pass only, which is the default and the right
    /// answer: this tap is derived from `clean`, which has already been
    /// high-passed at `hp_hz`. A second high-pass at the same corner buys no
    /// further baseline rejection and costs an enormous, strongly
    /// frequency-dependent delay - 155 ms at 1 Hz against 24 ms at 10 Hz - on
    /// exactly the two waves whose positions this tap exists to measure.
    pub pt_lo: f64,
    pub pt_hi: f64,
    pub pt_order: usize,
    /// Excursion (mV) beyond which a sample counts as rail contact.
    ///
    /// Measured on the DC-removed signal, never on the raw sample. Electrode
    /// half-cell potential puts a standing offset of tens of millivolts on the
    /// input - far larger than the ECG itself - so an absolute test against the
    /// raw value reports permanent saturation on perfectly good signal.
    pub saturation_mv: f32,
}

impl PreprocessConfig {
    pub fn new(fs: f64) -> Self {
        PreprocessConfig {
            fs,
            hp_hz: 0.5,
            hp_order: 2,
            lp_hz: 40.0,
            lp_order: 4,
            mains: Mains::Auto,
            notch_bw_hz: 2.0,
            hf_hz: 40.0,
            qrs_lo: 5.0,
            qrs_hi: 20.0,
            qrs_order: 2,
            pt_lo: 0.0,
            pt_hi: 10.0,
            pt_order: 2,
            saturation_mv: 5.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Bands {
    pub raw: f32,
    pub clean: f32,
    pub baseline: f32,
    pub hf: f32,
    /// QRS sub-band, derived from `clean` so the two stay sample-aligned.
    pub qrs: f32,
    /// P and T band, likewise.
    pub pt: f32,
    pub saturated: bool,
}

/// Goertzel single-bin power probe over a fixed block.
#[derive(Debug, Clone, Copy)]
struct BinPower {
    coeff: f64,
    s1: f64,
    s2: f64,
    n: usize,
    window: usize,
    power: f64,
}

impl BinPower {
    fn new(fs: f64, f: f64, window: usize) -> Self {
        let w = 2.0 * std::f64::consts::PI * f / fs;
        BinPower {
            coeff: 2.0 * w.cos(),
            s1: 0.0,
            s2: 0.0,
            n: 0,
            window,
            power: 0.0,
        }
    }

    #[inline]
    fn push(&mut self, x: f32) -> bool {
        let s0 = x as f64 + self.coeff * self.s1 - self.s2;
        self.s2 = self.s1;
        self.s1 = s0;
        self.n += 1;
        if self.n >= self.window {
            self.power = self.s1 * self.s1 + self.s2 * self.s2 - self.coeff * self.s1 * self.s2;
            self.power /= (self.window * self.window) as f64;
            self.s1 = 0.0;
            self.s2 = 0.0;
            self.n = 0;
            true
        } else {
            false
        }
    }
}

/// One candidate mains frequency, probed against its own spectral neighbourhood.
///
/// Comparing 50 Hz against 60 Hz is not enough. Below about 150 Hz sampling only
/// one of the two lines is representable at all, the comparison degenerates, and
/// the surviving candidate always "wins" - so a notch was applied to every
/// 128 Hz recording whether or not it carried any mains interference. Notching
/// is not free: at 250 Hz it put a 12 ms lag on the reported R positions.
///
/// Interference from a power line is a narrow tone, so the test that actually
/// identifies it is the line against the background either side of it.
#[derive(Debug, Clone, Copy)]
struct MainsCandidate {
    hz: f64,
    line: BinPower,
    lo: BinPower,
    hi: BinPower,
    score: f64,
}

impl MainsCandidate {
    fn new(fs: f64, hz: f64, window: usize) -> Option<Self> {
        // The probe needs headroom below Nyquist, and so do its two sidebands.
        if hz + MAINS_SIDEBAND_HZ >= fs * 0.45 {
            return None;
        }
        Some(MainsCandidate {
            hz,
            line: BinPower::new(fs, hz, window),
            lo: BinPower::new(fs, hz - MAINS_SIDEBAND_HZ, window),
            hi: BinPower::new(fs, hz + MAINS_SIDEBAND_HZ, window),
            score: 0.0,
        })
    }

    fn push(&mut self, x: f32) -> bool {
        self.lo.push(x);
        self.hi.push(x);
        let ready = self.line.push(x);
        if ready {
            let background = 0.5 * (self.lo.power + self.hi.power);
            self.score = self.line.power / background.max(1e-15);
        }
        ready
    }
}

/// Offset of the background probes either side of a candidate line, in Hz.
const MAINS_SIDEBAND_HZ: f64 = 7.0;

/// A line must stand this far above its own neighbourhood to be notched.
const MAINS_MIN_SNR: f64 = 6.0;

pub struct Preprocessor {
    cfg: PreprocessConfig,
    hp: Cascade,
    lp: Cascade,
    notch: Cascade,
    hf_hp: Cascade,
    qrs_bp: Cascade,
    pt_bp: Cascade,
    candidates: Vec<MainsCandidate>,
    mains_locked: bool,
    mains_hz: Option<f64>,
    primed: bool,
}

impl Preprocessor {
    pub fn new(cfg: PreprocessConfig) -> Self {
        let fs = cfg.fs;
        let hp = Cascade::butter_high_pass(fs, cfg.hp_hz, cfg.hp_order);
        let lp = Cascade::butter_low_pass(fs, cfg.lp_hz.min(fs * 0.45), cfg.lp_order);
        let hf_hp = Cascade::butter_high_pass(fs, cfg.hf_hz.min(fs * 0.45), 2);
        let qrs_bp = Cascade::band_pass(fs, cfg.qrs_lo, cfg.qrs_hi.min(fs * 0.45), cfg.qrs_order);
        let pt_bp = if cfg.pt_lo > 0.0 {
            Cascade::band_pass(fs, cfg.pt_lo, cfg.pt_hi.min(fs * 0.45), cfg.pt_order)
        } else {
            Cascade::butter_low_pass(fs, cfg.pt_hi.min(fs * 0.45), cfg.pt_order)
        };

        let win = (fs * 2.0) as usize;
        let candidates: Vec<MainsCandidate> = match cfg.mains {
            Mains::Auto => [50.0, 60.0]
                .into_iter()
                .filter_map(|f| MainsCandidate::new(fs, f, win))
                .collect(),
            _ => Vec::new(),
        };

        let mut pre = Preprocessor {
            cfg,
            hp,
            lp,
            notch: Cascade::default(),
            hf_hp,
            qrs_bp,
            pt_bp,
            candidates,
            mains_locked: false,
            mains_hz: None,
            primed: false,
        };

        match cfg.mains {
            Mains::Fixed50 => pre.set_mains(50.0),
            Mains::Fixed60 => pre.set_mains(60.0),
            Mains::Off => pre.mains_locked = true,
            Mains::Auto => {
                if pre.candidates.is_empty() {
                    pre.mains_locked = true; // no line is representable at this rate
                }
            }
        }
        pre
    }

    fn set_mains(&mut self, f: f64) {
        let mut c = Cascade::default();
        c.push(Biquad::notch(self.cfg.fs, f, self.cfg.notch_bw_hz));
        // A second section at the first harmonic; mains pickup is rarely a pure tone.
        if 2.0 * f < self.cfg.fs * 0.45 {
            c.push(Biquad::notch(
                self.cfg.fs,
                2.0 * f,
                self.cfg.notch_bw_hz * 2.0,
            ));
        }
        self.notch = c;
        self.mains_hz = Some(f);
        self.mains_locked = true;
    }

    pub fn mains_hz(&self) -> Option<f64> {
        self.mains_hz
    }

    /// Group delay of the P/T tap at `f` Hz, relative to `clean`, in samples.
    ///
    /// Delineation marks positions on this tap and reports them on the input
    /// time base, so its own delay has to come back out - the same argument as
    /// for the fiducial, and the same reason it is computed rather than tuned.
    pub fn pt_group_delay_samples(&self, f: f64) -> f64 {
        self.pt_bp.group_delay(self.cfg.fs, f)
    }

    /// Group delay of the QRS tap at `f` Hz, relative to `clean`, in samples.
    pub fn qrs_group_delay_samples(&self, f: f64) -> f64 {
        self.qrs_bp.group_delay(self.cfg.fs, f)
    }

    /// Group delay of the `clean` tap at `f` Hz, in samples.
    ///
    /// The QRS fiducial is placed on `clean`, so this is exactly the amount by
    /// which a reported R position would otherwise lag the true one. It is
    /// computed from the coefficients actually in use rather than tuned, so it
    /// stays correct across sample rates, corner changes and mains decisions.
    pub fn group_delay_samples(&self, f: f64) -> f64 {
        let fs = self.cfg.fs;
        self.hp.group_delay(fs, f) + self.notch.group_delay(fs, f) + self.lp.group_delay(fs, f)
    }

    /// Frequency at which the delay is evaluated: the centre of QRS energy.
    pub const FIDUCIAL_REF_HZ: f64 = 12.0;

    pub fn config(&self) -> &PreprocessConfig {
        &self.cfg
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> Bands {
        if !self.primed {
            // Start the recursions at the incoming DC level so the first seconds
            // are usable instead of being dominated by the settling transient.
            self.hp.prime(x);
            self.lp.prime(0.0);
            self.hf_hp.prime(x);
            self.qrs_bp.prime(0.0);
            self.pt_bp.prime(0.0);
            self.primed = true;
        }

        if !self.mains_locked {
            self.probe_mains(x);
        }

        let hp = self.hp.process(x);
        let baseline = x - hp;
        let notched = if self.notch.is_empty() {
            hp
        } else {
            self.notch.process(hp)
        };
        let clean = self.lp.process(notched);
        let qrs = self.qrs_bp.process(clean);
        let pt = self.pt_bp.process(clean);
        let hf = self.hf_hp.process(x);
        let saturated = hp.abs() >= self.cfg.saturation_mv;

        Bands {
            raw: x,
            clean,
            baseline,
            hf,
            qrs,
            pt,
            saturated,
        }
    }

    fn probe_mains(&mut self, x: f32) {
        let mut ready = false;
        for c in self.candidates.iter_mut() {
            ready |= c.push(x);
        }
        if !ready {
            return;
        }
        // Take the strongest line, and notch only if it genuinely stands above
        // its own neighbourhood. Leaving the notch out costs a little mains
        // rejection; putting it in where there is no line costs group delay on
        // every reported R position and removes signal that is really there.
        let best = self
            .candidates
            .iter()
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied();
        match best {
            Some(c) if c.score >= MAINS_MIN_SNR => self.set_mains(c.hz),
            _ => self.mains_locked = true,
        }
    }

    pub fn reset(&mut self) {
        self.hp.reset();
        self.lp.reset();
        self.notch.reset();
        self.hf_hp.reset();
        self.qrs_bp.reset();
        self.pt_bp.reset();
        self.primed = false;
    }
}
