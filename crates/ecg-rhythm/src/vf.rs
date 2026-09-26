//! Ventricular fibrillation and flutter detection.
//!
//! # Why this cannot reuse anything above it
//!
//! Every stage of this engine downstream of QRS detection assumes there are
//! beats. In fibrillation there are none: the trace is a continuous, roughly
//! sinusoidal oscillation with no isoelectric line, no P wave and no QRS to
//! find. Feeding it to a beat detector produces detections, and everything
//! built on those detections is then describing an artefact.
//!
//! So this detector reads the signal directly and never asks what the beats
//! were. It runs beside the beat path rather than after it, which also means it
//! keeps working in exactly the situation where the beat path has quietly
//! stopped being meaningful.
//!
//! # The features
//!
//! Four of the five are classical and were chosen because they answer the
//! question from different directions - comparative studies of VF algorithms
//! consistently find that no single one of them is enough.
//!
//! * **Threshold-crossing sample count.** The fraction of a normalised window
//!   above a fifth of its own peak. Normal rhythm spends most of its time on the
//!   isoelectric line, so this is low; fibrillation has no isoelectric line at
//!   all.
//! * **VF filter leakage.** The signal is compared against itself delayed by
//!   half its own estimated period. A pure sinusoid cancels, so leakage falls to
//!   zero as the trace becomes sinusoidal, and stays high for the sharp
//!   asymmetric shape of a QRS complex.
//! * **Peak-to-mean ratio.** A normal complex is a narrow spike on a quiet
//!   baseline, so its peak is many times its mean magnitude. A sinusoid's is not.
//! * **Excess kurtosis.** The same statistic the quality monitor uses, for the
//!   same reason and with the opposite sign of interest.
//! * **Relative amplitude.** What separates fibrillation from asystole and from
//!   a dead lead, which are also beatless and would otherwise look alike.
//!
//! # What this detector is, and is not, good enough for
//!
//! Measured on a third of the fibrillation corpora held out from fitting, the
//! score reaches an AUC of 0.89 against a negative class that is itself mostly
//! malignant - ventricular tachycardia, asystole, noise. At the shipped
//! operating point that is 86% sensitivity at 28% precision.
//!
//! **That is not good enough to raise a fibrillation alarm on**, and it is
//! reported rather than dressed up. What it is good enough for is the job the
//! rest of the engine actually needs: flagging that the beat path's assumptions
//! have failed. Over-calling "do not trust the beats here" costs a suppressed
//! conclusion; over-calling "this patient is in fibrillation" costs something
//! else entirely. [`VfDetector::in_vf`] is meant to be read in the first sense.
//!
//! A tree ensemble was fitted over the same features and measured: AUC 0.876,
//! not better. The limit is the feature set, not the model.

use ecg_beats::gbdt::GbdtModel;
use ecg_dsp::{ms_to_samples, Ring};

/// Number of features in the model input vector.
pub const NF: usize = 12;

#[derive(Debug, Clone, Copy)]
pub struct VfConfig {
    pub fs: f64,
    /// Window the features are measured over. Long enough that a single wide
    /// complex cannot fill it, short enough to alarm in time.
    pub window_s: f32,
    /// How often the window is re-measured.
    pub hop_s: f32,
    /// Fraction of the window peak that counts as "not baseline".
    pub tcsc_threshold: f32,
    /// Cosine taper at each end of the window, seconds.
    pub taper_s: f32,
    pub model: VfModel,
    /// Chosen against what this detector can honestly be used for (see the note
    /// on [`VfDetector`]): high enough that normal rhythm is left alone -
    /// specificity 100.00% over 270 hours of it - and low enough to catch most
    /// fibrillation.
    ///
    /// The normal-sinus constraint stopped binding once the phase-space
    /// feature was added: specificity there is 100% at every threshold from
    /// 0.65 up. So the rule that picked this value is the same one, applied to
    /// what still varies - the most sensitive threshold whose alarm rate on
    /// normal rhythm stays inside the budget - and above 0.70 sensitivity falls
    /// away for very little specificity.
    ///
    /// Moved from 0.70 to 0.80 when the detector was first scored as an
    /// alarm - onsets found against false alarms per day - rather than per
    /// second. Chosen on the training zones: fibrillation onsets found stay at
    /// 38 of 38 on VFDB and go from 43 to 41 of 44 on CUDB, while false alarms
    /// on 1,947 hours of the Long-Term AF corpus fall from 0.18 to 0.02 a day
    /// and on MIT-BIH's training records from 6.5 to none. At 0.85 CUDB drops
    /// to 38. Noise remains the weak side: the noise-stress records, at
    /// signal-to-noise ratios down to -6 dB, still raise 60 a day.
    pub enter_prob: f32,
    pub exit_prob: f32,
    /// Fibrillation is immediately actionable, so the confirmation window is
    /// short - unlike atrial fibrillation, where thirty seconds is the clinical
    /// threshold and a late alarm costs nothing.
    pub min_episode_s: f32,
    pub bridge_s: f32,
    /// Bar for *withholding* the beat-derived analysis, as opposed to reporting
    /// fibrillation. It was set higher than `enter_prob`, and the asymmetry is
    /// measured; since the alarm's bar moved to 0.80 it is the lower of the two,
    /// and what keeps it the stricter decision is its duration - ten seconds
    /// against four.
    ///
    /// Reporting a false episode costs a reviewer's attention. Suppressing on a
    /// false episode deletes true findings: at the reporting bar, false
    /// positives on 1.7% of an AFDB record cost 95% of that record's atrial
    /// fibrillation windows and took corpus sensitivity from 86% to 34%. The two
    /// decisions do not deserve the same evidence.
    pub suppress_prob: f32,
    pub suppress_min_s: f32,
    /// Delay used to build the phase-space plot, milliseconds. Half a second,
    /// which is the value the method was published with: long enough that a
    /// beat and its own echo are at different points, short enough to stay
    /// inside the window.
    pub psr_tau_ms: f64,
}

impl VfConfig {
    pub fn new(fs: f64) -> Self {
        VfConfig {
            fs,
            window_s: 4.0,
            hop_s: 1.0,
            psr_tau_ms: 500.0,
            tcsc_threshold: 0.2,
            taper_s: 0.25,
            model: VfModel::default(),
            enter_prob: 0.8,
            exit_prob: 0.55,
            min_episode_s: 4.0,
            bridge_s: 4.0,
            suppress_prob: 0.75,
            suppress_min_s: 10.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VfFeatures {
    /// Share of the window above a fifth of its own peak, after tapering.
    pub tcsc: f32,
    /// Residual after cancelling the signal against itself at half its period.
    pub leakage: f32,
    /// Window peak over mean magnitude.
    pub peak_to_mean: f32,
    pub kurtosis: f32,
    /// Dominant frequency implied by the period estimate, Hz.
    pub dominant_hz: f32,
    /// Window amplitude over this channel's own slow reference.
    pub amplitude_rel: f32,
    /// Share of a 40 by 40 grid that the trajectory (x(t), x(t+tau)) visits.
    ///
    /// A rhythm with beats spends almost all of its time near the baseline and
    /// crosses the rest of the plane briefly, so it occupies a thin figure.
    /// Fibrillation wanders, and fills it. The one feature here that is about
    /// the *shape of the trajectory* rather than about amplitude or period,
    /// which is why it fails differently from the other five.
    pub psr_density: f32,
    /// The spectrum's own answer to the same question, and the reason for
    /// asking it: every feature above can be satisfied by a large slow artefact,
    /// and on the noise-stress records they are - sixty false alarms a day.
    /// Fibrillation is a narrow-band oscillation between about 3 and 8 Hz;
    /// electrode motion is broad and low, muscle is broad and high, and a
    /// rhythm with beats spreads its energy over the harmonics of its rate.
    ///
    /// Share of the 0.5-30 Hz power within half the dominant frequency either
    /// side of it, the dominant frequency taken between 1 and 10 Hz (the SPEC
    /// algorithm's concentration).
    pub spec_conc: f32,
    /// Share of the 0.5-30 Hz power between 2.5 and 7.5 Hz.
    pub vf_band_frac: f32,
    /// Share of the 0.5-30 Hz power above 12 Hz.
    pub hf_frac: f32,
    /// Entropy of the normalised 0.5-30 Hz spectrum, over its maximum.
    pub spec_entropy: f32,
    /// Variation of the window's amplitude envelope: the coefficient of
    /// variation of the peak magnitude in each quarter-second block.
    /// Fibrillation waxes and wanes slowly; artefact arrives in bursts.
    pub env_cv: f32,
}

impl VfFeatures {
    #[inline]
    pub fn vector(&self) -> [f32; NF] {
        [
            self.tcsc,
            self.leakage,
            self.peak_to_mean,
            self.kurtosis,
            self.dominant_hz,
            self.amplitude_rel,
            self.psr_density,
            self.spec_conc,
            self.vf_band_frac,
            self.hf_frac,
            self.spec_entropy,
            self.env_cv,
        ]
    }

    pub const NAMES: [&'static str; NF] = [
        "tcsc",
        "leakage",
        "peak_to_mean",
        "kurtosis",
        "dominant_hz",
        "amplitude_rel",
        "psr_density",
        "spec_conc",
        "vf_band_frac",
        "hf_frac",
        "spec_entropy",
        "env_cv",
    ];
}

#[derive(Debug, Clone, Copy)]
pub struct VfWeights {
    pub bias: f32,
    pub w: [f32; NF],
}

impl VfWeights {
    /// Fitted by `ecg-eval fit-vf` on all of the fibrillation corpora's
    /// training records (63,809 windows, 9.9 % fibrillation; `PHASE-11.md` §15), with the
    /// five spectral and envelope features added to the seven before them.
    ///
    /// Standardised influence, largest first: `psr_density` +1.15,
    /// `spec_conc` +0.40, `peak_to_mean` +0.34, `kurtosis` -0.28, `hf_frac`
    /// -0.28. Against the seven-feature fit on the same records, at the same
    /// alarm bar, onsets found are unchanged (VFDB 38 of 38, CUDB 41 of 44)
    /// and false alarms on records no fit has seen fall: the noise-stress
    /// records from 60 a day to 12, MIT-BIH's training records from 2.2 to
    /// none, the Long-Term AF corpus from 0.02 to 0.01.
    pub const BASELINE: VfWeights = VfWeights {
        bias: -2.945878,
        w: [
            -0.915637, -1.250130, 0.128096, -0.055942, -0.039030, 0.118203, 15.188091, 1.850344,
            0.692025, -2.925844, -1.585018, 0.701483,
        ],
    };

    /// The seven-feature fit that preceded the spectral features, kept as a
    /// selectable stage (`vf.linear@1`).
    pub const LINEAR7: VfWeights = VfWeights {
        bias: -0.490744,
        w: [
            -0.902319, -3.525691, 0.142076, -0.063149, -0.146428, 0.162449, 16.190325, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ],
    };

    #[inline]
    pub fn probability(&self, f: &VfFeatures) -> f32 {
        let v = f.vector();
        let mut z = self.bias;
        for (wi, vi) in self.w.iter().zip(v.iter()) {
            z += wi * vi;
        }
        1.0 / (1.0 + (-z).exp())
    }
}

/// What the detector scores with. Both forms are kept so the choice stays a
/// measured one, as it is for the beat detectors.
#[derive(Debug, Clone, Copy)]
pub enum VfModel {
    Linear(VfWeights),
    Gbdt(GbdtModel),
}

impl Default for VfModel {
    fn default() -> Self {
        if trees::VENTRICULAR_FIBRILLATION.is_empty() {
            VfModel::Linear(VfWeights::BASELINE)
        } else {
            VfModel::Gbdt(trees::VENTRICULAR_FIBRILLATION)
        }
    }
}

impl VfModel {
    #[inline]
    pub fn probability(&self, f: &VfFeatures) -> f32 {
        match self {
            VfModel::Linear(m) => m.probability(f),
            VfModel::Gbdt(m) => m.probability(&f.vector()),
        }
    }
}

/// Generated by `ecg-eval fit-vf --gbdt`.
pub mod trees {
    use ecg_beats::gbdt::GbdtModel;
    #[allow(unused_imports)]
    use ecg_beats::gbdt::Node;
    include!("vf_trees_generated.rs");
}

#[derive(Debug, Clone, Copy)]
pub struct VfWindow {
    /// Sample at which the window ends.
    pub sample: u64,
    pub features: VfFeatures,
    pub probability: f32,
    pub in_vf: bool,
}

pub struct VfDetector {
    cfg: VfConfig,
    ring: Ring,
    window: usize,
    hop: usize,
    /// Delay of the phase-space plot, samples.
    psr_tau: usize,
    taper: usize,
    since_hop: usize,
    n: u64,
    /// First sample after the most recent discontinuity. The window may not
    /// reach behind it.
    valid_from: u64,
    in_vf: bool,
    /// Slow amplitude reference, learned only from windows that look like a
    /// rhythm with beats - otherwise a long fibrillation would redefine normal.
    slow_amplitude: f32,
    slow_primed: bool,
    scratch: Vec<f32>,
    /// FFT workspace and tables, sized once: the next power of two above the
    /// window.
    fft: RealFft,
    /// The Hann taper, computed once rather than as a thousand cosines per
    /// window.
    hann: Vec<f32>,
}

impl VfDetector {
    pub fn new(cfg: VfConfig) -> Self {
        let window = ms_to_samples(cfg.fs, cfg.window_s as f64 * 1000.0);
        VfDetector {
            ring: Ring::with_capacity(window),
            window,
            hop: ms_to_samples(cfg.fs, cfg.hop_s as f64 * 1000.0),
            psr_tau: ms_to_samples(cfg.fs, cfg.psr_tau_ms).max(1),
            taper: ms_to_samples(cfg.fs, cfg.taper_s as f64 * 1000.0),
            since_hop: 0,
            n: 0,
            valid_from: 0,
            in_vf: false,
            slow_amplitude: 0.0,
            slow_primed: false,
            scratch: vec![0.0; window],
            fft: RealFft::new(window.next_power_of_two()),
            hann: hann(window),
            cfg,
        }
    }

    pub fn config(&self) -> &VfConfig {
        &self.cfg
    }

    pub fn in_vf(&self) -> bool {
        self.in_vf
    }

    /// Feed one sample of the analysis-band signal.
    pub fn process(&mut self, clean: f32) -> Option<VfWindow> {
        self.ring.push(clean);
        self.n += 1;
        self.since_hop += 1;
        if self.since_hop < self.hop || self.n - self.valid_from < self.window as u64 {
            return None;
        }
        self.since_hop = 0;

        let from = self.n - self.window as u64;
        for (k, slot) in self.scratch.iter_mut().enumerate() {
            *slot = self.ring.at(from + k as u64);
        }
        let features = self.measure();
        let probability = self.cfg.model.probability(&features);
        self.in_vf = if self.in_vf {
            probability >= self.cfg.exit_prob
        } else {
            probability >= self.cfg.enter_prob
        };
        Some(VfWindow {
            sample: self.n,
            features,
            probability,
            in_vf: self.in_vf,
        })
    }

    fn measure(&mut self) -> VfFeatures {
        let w = self.window;
        let x = &mut self.scratch[..w];

        // Remove the window mean before anything else; every feature below is
        // about shape, and a residual offset would move all of them.
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / w as f64;
        for v in x.iter_mut() {
            *v -= mean as f32;
        }

        let peak = x.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        let p2p = {
            let lo = x.iter().cloned().fold(f32::MAX, f32::min);
            let hi = x.iter().cloned().fold(f32::MIN, f32::max);
            hi - lo
        };
        if peak <= 1e-9 {
            return VfFeatures {
                amplitude_rel: 0.0,
                ..VfFeatures::default()
            };
        }

        // Moments, for kurtosis.
        let (mut m2, mut m4) = (0.0f64, 0.0f64);
        for &v in x.iter() {
            let d = v as f64;
            m2 += d * d;
            m4 += d * d * d * d;
        }
        m2 /= w as f64;
        m4 /= w as f64;
        let kurtosis = if m2 > 1e-18 {
            (m4 / (m2 * m2) - 3.0) as f32
        } else {
            0.0
        };

        let mean_abs = x.iter().map(|v| v.abs() as f64).sum::<f64>() / w as f64;
        let peak_to_mean = if mean_abs > 1e-9 {
            peak / mean_abs as f32
        } else {
            0.0
        };

        // Period estimate: for a sinusoid, `pi * sum|x| / sum|dx|` is half the
        // period in samples. It needs no spectrum and degrades gracefully on
        // signals that are not sinusoidal, which is the case it has to reject.
        let mut d_abs = 0.0f64;
        for i in 1..w {
            d_abs += (x[i] - x[i - 1]).abs() as f64;
        }
        let half_period = if d_abs > 1e-9 {
            (std::f64::consts::PI * mean_abs * w as f64 / d_abs).round() as usize
        } else {
            0
        };
        let half_period = half_period.clamp(1, w / 2);
        let dominant_hz = self.cfg.fs as f32 / (2.0 * half_period as f32);

        // VF filter leakage: cancel the signal against itself half a period
        // later. A sinusoid cancels; a train of sharp complexes does not.
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in half_period..w {
            num += (x[i] + x[i - half_period]).abs() as f64;
            den += (x[i].abs() + x[i - half_period].abs()) as f64;
        }
        let leakage = if den > 1e-9 { (num / den) as f32 } else { 1.0 };

        // Threshold-crossing sample count, over a tapered window so a complex
        // straddling an edge does not set the threshold for the whole window.
        let taper = self.taper.min(w / 4).max(1);
        let mut above = 0usize;
        let bar = peak * self.cfg.tcsc_threshold;
        for (i, &v) in x.iter().enumerate() {
            let weight = if i < taper {
                0.5 * (1.0 - (std::f32::consts::PI * (taper - i) as f32 / taper as f32).cos())
            } else if i >= w - taper {
                0.5 * (1.0 - (std::f32::consts::PI * (i - (w - taper)) as f32 / taper as f32).cos())
            } else {
                1.0
            };
            if v.abs() * weight >= bar {
                above += 1;
            }
        }
        let tcsc = above as f32 / w as f32;

        // The slow reference learns only from windows that still look like a
        // rhythm with beats. Otherwise several minutes of fibrillation would
        // become this channel's idea of normal amplitude, and the feature that
        // separates fibrillation from asystole would stop working.
        let amplitude_rel = if self.slow_primed && self.slow_amplitude > 1e-9 {
            p2p / self.slow_amplitude
        } else {
            1.0
        };
        if peak_to_mean > 3.0 && kurtosis > 2.0 {
            if self.slow_primed {
                self.slow_amplitude += 0.05 * (p2p - self.slow_amplitude);
            } else {
                self.slow_amplitude = p2p;
                self.slow_primed = true;
            }
        }

        let psr_density = phase_space_density(x, self.psr_tau, peak);
        let env_cv = envelope_cv(x, (self.cfg.fs * 0.25) as usize);
        let spectrum = spectral(x, &self.hann, &mut self.fft, self.cfg.fs as f32);

        VfFeatures {
            tcsc,
            leakage,
            peak_to_mean,
            kurtosis,
            dominant_hz,
            amplitude_rel,
            psr_density,
            spec_conc: spectrum[0],
            vf_band_frac: spectrum[1],
            hf_frac: spectrum[2],
            spec_entropy: spectrum[3],
            env_cv,
        }
    }

    /// Samples were lost; the window would otherwise splice two unrelated
    /// stretches into one apparent waveform.
    /// `unobserved` is how many samples passed without being seen.
    pub fn on_gap(&mut self, unobserved: u64) {
        self.n += unobserved;
        self.valid_from = self.n;
        self.since_hop = 0;
    }

    pub fn reset(&mut self) {
        self.ring.reset();
        self.n = 0;
        self.valid_from = 0;
        self.since_hop = 0;
        self.in_vf = false;
        self.slow_amplitude = 0.0;
        self.slow_primed = false;
    }
}

/// Share of a 40 by 40 grid visited by the trajectory `(x(t), x(t + tau))`.
///
/// The window is scaled by its own peak, so this says nothing about amplitude -
/// which is the point. Every other feature in this vector is a statement about
/// how big or how fast the signal is, and all of them can be satisfied by a
/// large slow artefact.
fn phase_space_density(x: &[f32], tau: usize, peak: f32) -> f32 {
    const G: usize = 40;
    if x.len() <= tau || peak <= 1e-9 {
        return 0.0;
    }
    let mut cells = [0u64; (G * G).div_ceil(64)];
    let scale = 0.5 * (G - 1) as f32 / peak;
    let mid = 0.5 * (G - 1) as f32;
    for i in 0..(x.len() - tau) {
        let a = (x[i] * scale + mid).clamp(0.0, (G - 1) as f32) as usize;
        let b = (x[i + tau] * scale + mid).clamp(0.0, (G - 1) as f32) as usize;
        let k = a * G + b;
        cells[k / 64] |= 1 << (k % 64);
    }
    let visited: u32 = cells.iter().map(|w| w.count_ones()).sum();
    visited as f32 / (G * G) as f32
}

/// Coefficient of variation of the peak magnitude in consecutive blocks.
fn envelope_cv(x: &[f32], block: usize) -> f32 {
    let block = block.max(1);
    let (mut n, mut sum, mut sq) = (0usize, 0.0f64, 0.0f64);
    for chunk in x.chunks(block) {
        if chunk.len() < block / 2 {
            continue;
        }
        let p = chunk.iter().fold(0.0f32, |a, b| a.max(b.abs())) as f64;
        n += 1;
        sum += p;
        sq += p * p;
    }
    if n < 2 || sum <= 1e-12 {
        return 0.0;
    }
    let mean = sum / n as f64;
    let var = (sq / n as f64 - mean * mean).max(0.0);
    (var.sqrt() / mean) as f32
}

/// `[concentration, 2.5-7.5 Hz share, >12 Hz share, normalised entropy]` of
/// the window's power spectrum over 0.5-30 Hz, Hann-tapered and zero-padded
/// into the workspace.
fn hann(w: usize) -> Vec<f32> {
    (0..w)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (w - 1).max(1) as f32).cos())
        .collect()
}

fn spectral(x: &[f32], taper: &[f32], fft: &mut RealFft, fs: f32) -> [f32; 4] {
    let m = fft.n;
    let df = fs / m as f32;
    let bin = |hz: f32| ((hz / df).round() as usize).min(m / 2);
    let (lo, hi) = (bin(0.5).max(1), bin(30.0));
    if hi <= lo {
        return [0.0; 4];
    }
    // Only the bins up to 30 Hz are ever read, so only those are formed.
    let spectrum = fft.power(x, taper, hi);
    let power = |k: usize| spectrum[k];
    let mut total = 0.0f64;
    let (mut band, mut high) = (0.0f64, 0.0f64);
    let (b_lo, b_hi, h_lo) = (bin(2.5), bin(7.5), bin(12.0));
    let (d_lo, d_hi) = (bin(1.0), bin(10.0));
    let (mut peak_k, mut peak_p) = (d_lo, -1.0f32);
    for k in lo..=hi {
        let p = power(k);
        total += p as f64;
        if (b_lo..=b_hi).contains(&k) {
            band += p as f64;
        }
        if k >= h_lo {
            high += p as f64;
        }
        if (d_lo..=d_hi).contains(&k) && p > peak_p {
            peak_p = p;
            peak_k = k;
        }
    }
    if total <= 1e-18 {
        return [0.0; 4];
    }
    let f0 = peak_k as f32 * df;
    let (c_lo, c_hi) = (bin(0.5 * f0).max(lo), bin(1.5 * f0).min(hi));
    let mut conc = 0.0f64;
    let mut entropy = 0.0f64;
    for k in lo..=hi {
        let p = power(k) as f64;
        if (c_lo..=c_hi).contains(&k) {
            conc += p;
        }
        let q = p / total;
        if q > 0.0 {
            entropy -= q * q.ln();
        }
    }
    let max_entropy = ((hi - lo + 1) as f64).ln().max(1e-9);
    [
        (conc / total) as f32,
        (band / total) as f32,
        (high / total) as f32,
        (entropy / max_entropy) as f32,
    ]
}

/// Power spectrum of a real window, by a half-length complex FFT.
///
/// A real signal of `n` samples is packed as `n / 2` complex ones (even
/// samples real, odd imaginary), transformed, and unpacked into the first
/// half of the spectrum - half the work of a full complex transform, and only
/// the bins asked for are unpacked. Every table is built once, so a window
/// costs no allocation and no trigonometry.
pub(crate) struct RealFft {
    n: usize,
    half: usize,
    zr: Vec<f32>,
    zi: Vec<f32>,
    power: Vec<f32>,
    /// `half`-point twiddles, `e^{-2 pi i j / half}`.
    tw_r: Vec<f32>,
    tw_i: Vec<f32>,
    /// Unpacking twiddles, `e^{-2 pi i k / n}`.
    un_r: Vec<f32>,
    un_i: Vec<f32>,
    rev: Vec<u32>,
}

impl RealFft {
    pub(crate) fn new(n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 4);
        let half = n / 2;
        let bits = half.trailing_zeros();
        let rev = (0..half as u32)
            .map(|i| {
                if bits == 0 {
                    0
                } else {
                    i.reverse_bits() >> (32 - bits)
                }
            })
            .collect();
        let tw = |j: usize, len: usize| {
            let a = -2.0 * std::f64::consts::PI * j as f64 / len as f64;
            (a.cos() as f32, a.sin() as f32)
        };
        let (tw_r, tw_i) = (0..half / 2).map(|j| tw(j, half)).unzip();
        let (un_r, un_i) = (0..=half).map(|k| tw(k, n)).unzip();
        RealFft {
            n,
            half,
            zr: vec![0.0; half],
            zi: vec![0.0; half],
            power: vec![0.0; half + 1],
            tw_r,
            tw_i,
            un_r,
            un_i,
            rev,
        }
    }

    /// `|X[k]|^2` for `k` in `0..=up_to`, of `x` tapered by `taper` and
    /// zero-padded to `n`.
    pub(crate) fn power(&mut self, x: &[f32], taper: &[f32], up_to: usize) -> &[f32] {
        let (h, w) = (self.half, x.len());
        let at = |i: usize| if i < w { x[i] * taper[i] } else { 0.0 };
        for j in 0..h {
            let r = self.rev[j] as usize;
            self.zr[r] = at(2 * j);
            self.zi[r] = at(2 * j + 1);
        }
        let mut len = 2;
        while len <= h {
            let step = h / len;
            for start in (0..h).step_by(len) {
                for k in 0..len / 2 {
                    let (cr, ci) = (self.tw_r[k * step], self.tw_i[k * step]);
                    let (a, b) = (start + k, start + k + len / 2);
                    let tr = self.zr[b] * cr - self.zi[b] * ci;
                    let ti = self.zr[b] * ci + self.zi[b] * cr;
                    self.zr[b] = self.zr[a] - tr;
                    self.zi[b] = self.zi[a] - ti;
                    self.zr[a] += tr;
                    self.zi[a] += ti;
                }
            }
            len <<= 1;
        }
        let up_to = up_to.min(h);
        for k in 0..=up_to {
            let (ar, ai) = (self.zr[k % h], self.zi[k % h]);
            let m = (h - k) % h;
            let (br, bi) = (self.zr[m], self.zi[m]);
            // Even and odd halves of the original sequence's spectrum.
            let (er, ei) = (0.5 * (ar + br), 0.5 * (ai - bi));
            let (or, oi) = (0.5 * (ai + bi), -0.5 * (ar - br));
            let (c, s) = (self.un_r[k], self.un_i[k]);
            let xr = er + c * or - s * oi;
            let xi = ei + c * oi + s * or;
            self.power[k] = xr * xr + xi * xi;
        }
        &self.power[..=up_to]
    }
}

#[cfg(test)]
mod spectral_tests {
    use super::*;

    /// In-place iterative radix-2 FFT. `re.len()` must be a power of two.
    fn fft(re: &mut [f32], im: &mut [f32]) {
        let n = re.len();
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let ang = -2.0 * std::f64::consts::PI / len as f64;
            let (wr, wi) = (ang.cos() as f32, ang.sin() as f32);
            let mut i = 0;
            while i < n {
                let (mut cr, mut ci) = (1.0f32, 0.0f32);
                for k in 0..len / 2 {
                    let (a, b) = (i + k, i + k + len / 2);
                    let tr = re[b] * cr - im[b] * ci;
                    let ti = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                    let nr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = nr;
                }
                i += len;
            }
            len <<= 1;
        }
    }

    /// The half-length transform gives the full complex transform's power.
    #[test]
    fn the_real_transform_matches_the_complex_one() {
        let n = 1024;
        let x: Vec<f32> = (0..1000)
            .map(|i| ((i * 7919) % 1013) as f32 / 1013.0 - 0.5 + (i as f32 * 0.13).sin())
            .collect();
        let taper = hann(1000);
        let (mut re, mut im) = (vec![0.0f32; n], vec![0.0f32; n]);
        for i in 0..1000 {
            re[i] = x[i] * taper[i];
        }
        fft(&mut re, &mut im);
        let mut r = RealFft::new(n);
        let p = r.power(&x, &taper, n / 2).to_vec();
        let peak = p.iter().cloned().fold(0.0f32, f32::max);
        for k in 0..=n / 2 {
            let full = re[k] * re[k] + im[k] * im[k];
            assert!(
                (full - p[k]).abs() <= 1e-4 * peak,
                "bin {k}: {full} against {}",
                p[k]
            );
        }
    }

    /// A 5 Hz sinusoid is fibrillation's spectrum at its simplest: all of it in
    /// the band, none of it high, and concentrated round its own frequency.
    #[test]
    fn a_sinusoid_is_concentrated_where_it_is() {
        let fs = 250.0f32;
        let x: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * 5.0 * i as f32 / fs).sin())
            .collect();
        let mut f = RealFft::new(1024);
        let [conc, band, high, entropy] = spectral(&x, &hann(1000), &mut f, fs);
        assert!(conc > 0.95, "concentration {conc}");
        assert!(band > 0.95, "band {band}");
        assert!(high < 0.01, "high {high}");
        assert!(entropy < 0.3, "entropy {entropy}");
    }
}
