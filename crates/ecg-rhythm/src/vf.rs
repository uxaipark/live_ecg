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
pub const NF: usize = 6;

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
    /// specificity 99.95% over 270 hours of it - and low enough to catch most
    /// fibrillation.
    pub enter_prob: f32,
    pub exit_prob: f32,
    /// Fibrillation is immediately actionable, so the confirmation window is
    /// short - unlike atrial fibrillation, where thirty seconds is the clinical
    /// threshold and a late alarm costs nothing.
    pub min_episode_s: f32,
    pub bridge_s: f32,
}

impl VfConfig {
    pub fn new(fs: f64) -> Self {
        VfConfig {
            fs,
            window_s: 4.0,
            hop_s: 1.0,
            tcsc_threshold: 0.2,
            taper_s: 0.25,
            model: VfModel::default(),
            enter_prob: 0.6,
            exit_prob: 0.45,
            min_episode_s: 4.0,
            bridge_s: 4.0,
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
        ]
    }

    pub const NAMES: [&'static str; NF] = [
        "tcsc",
        "leakage",
        "peak_to_mean",
        "kurtosis",
        "dominant_hz",
        "amplitude_rel",
    ];
}

#[derive(Debug, Clone, Copy)]
pub struct VfWeights {
    pub bias: f32,
    pub w: [f32; NF],
}

impl VfWeights {
    /// Fitted by `ecg-eval fit-vf` on two thirds of the fibrillation corpora
    /// (43,070 windows, 10.4% fibrillation), validated on the remaining third.
    ///
    /// Standardised influence, largest first: `leakage` -0.63, `kurtosis` -0.54,
    /// `tcsc` +0.45. The three carry it between them, which is the finding the
    /// VF literature keeps reporting - the shape measures agree with each other
    /// often enough to be redundant and disagree often enough to be needed.
    pub const BASELINE: VfWeights = VfWeights {
        bias: 1.899484,
        w: [2.23774, -3.941355, 0.016516, -0.117416, 0.003304, -0.218599],
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
    taper: usize,
    since_hop: usize,
    n: u64,
    in_vf: bool,
    /// Slow amplitude reference, learned only from windows that look like a
    /// rhythm with beats - otherwise a long fibrillation would redefine normal.
    slow_amplitude: f32,
    slow_primed: bool,
    scratch: Vec<f32>,
}

impl VfDetector {
    pub fn new(cfg: VfConfig) -> Self {
        let window = ms_to_samples(cfg.fs, cfg.window_s as f64 * 1000.0);
        VfDetector {
            ring: Ring::with_capacity(window),
            window,
            hop: ms_to_samples(cfg.fs, cfg.hop_s as f64 * 1000.0),
            taper: ms_to_samples(cfg.fs, cfg.taper_s as f64 * 1000.0),
            since_hop: 0,
            n: 0,
            in_vf: false,
            slow_amplitude: 0.0,
            slow_primed: false,
            scratch: vec![0.0; window],
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
        if self.since_hop < self.hop || self.n < self.window as u64 {
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

        VfFeatures {
            tcsc,
            leakage,
            peak_to_mean,
            kurtosis,
            dominant_hz,
            amplitude_rel,
        }
    }

    /// Samples were lost; the window would otherwise splice two unrelated
    /// stretches into one apparent waveform.
    pub fn on_gap(&mut self) {
        self.ring.reset();
        self.n = 0;
        self.since_hop = 0;
    }

    pub fn reset(&mut self) {
        self.ring.reset();
        self.n = 0;
        self.since_hop = 0;
        self.in_vf = false;
        self.slow_amplitude = 0.0;
        self.slow_primed = false;
    }
}
