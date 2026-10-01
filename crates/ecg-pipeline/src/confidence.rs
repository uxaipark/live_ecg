//! Calibrated confidence: how far to trust a detection, and a beat's class.
//!
//! The detectors' own scores rank well and are not probabilities. A QRS
//! detection's margin over its threshold is a ratio; the beat detectors were
//! fitted with balanced classes and per-record weights, so a score of 0.9 does
//! not mean nine in ten. The models here map those scores, and a little of
//! their context, to probabilities that are right on average - fitted at the
//! classes' natural prevalence on records the detectors were not fitted on,
//! and checked for calibration on sealed records (`ecg-eval conf-eval`).
//!
//! A calibration belongs to one implementation. It is attached to a stage by
//! name, and a stage without one - another implementation, or one built from
//! a tuned configuration - reports NaN rather than a number that means
//! nothing.

/// Number of inputs to the detection model.
pub const NQ: usize = 9;

/// What is known about a detection when it is made.
#[derive(Debug, Clone, Copy)]
pub struct QrsFeatures {
    /// Integration peak over the threshold it crossed; below one for a beat
    /// recovered by search-back.
    pub margin: f32,
    pub recovered: bool,
    /// Signal quality score at the detection, 0..1.
    pub quality: f32,
    /// The quality stage called the sample unusable.
    pub unusable: bool,
    /// Interval since the previous detection over this channel's running
    /// interval; 1.0 when there is none yet.
    pub rr_rel: f32,
}

impl QrsFeatures {
    pub fn vector(&self) -> [f32; NQ] {
        let m = self.margin.clamp(0.05, 50.0).ln();
        let r = self.rr_rel.clamp(0.1, 4.0).ln();
        let q = self.quality.clamp(0.0, 1.0);
        [
            m,
            m * m,
            self.recovered as u8 as f32,
            q,
            self.unusable as u8 as f32,
            r,
            r * r,
            q * q,
            m * q,
        ]
    }
}

/// A logistic model over a fixed vector.
#[derive(Debug, Clone, Copy)]
pub struct Logistic<const N: usize> {
    pub bias: f32,
    pub w: [f32; N],
}

impl<const N: usize> Logistic<N> {
    pub fn probability(&self, x: &[f32; N]) -> f32 {
        let mut z = self.bias;
        for i in 0..N {
            z += self.w[i] * x[i];
        }
        1.0 / (1.0 + (-z).exp())
    }
}

/// Number of inputs to the class model.
pub const NB: usize = 6;

/// The beat model's inputs: the three detectors' scores as log-odds, and the
/// rhythm context they were read in.
pub fn beat_vector(pv: f32, ps: f32, pf: f32, fibrillating: bool) -> [f32; NB] {
    let logit = |p: f32| {
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln()
    };
    let f = fibrillating as u8 as f32;
    [
        logit(pv),
        logit(ps),
        logit(pf),
        f,
        f * logit(ps),
        logit(pv) * logit(ps).max(0.0) / 10.0,
    ]
}

/// A softmax over N, S, V and F.
#[derive(Debug, Clone, Copy)]
pub struct Softmax<const N: usize> {
    /// Per class: bias, then weights.
    pub w: [[f32; N]; 4],
    pub bias: [f32; 4],
}

impl<const N: usize> Softmax<N> {
    pub fn probabilities(&self, x: &[f32; N]) -> [f32; 4] {
        let mut z = [0.0f32; 4];
        for c in 0..4 {
            z[c] = self.bias[c];
            for i in 0..N {
                z[c] += self.w[c][i] * x[i];
            }
        }
        let m = z.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0.0;
        for v in z.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        z.map(|v| v / sum)
    }
}

/// Fitted by `ecg-eval fit-qrs-conf` on the training zones of MIT-BIH, the
/// supraventricular, noise-stress, AF and long-term ST corpora: 1.22 million
/// detections, 3.96 % of them with no reference beat. The Long-Term AF corpus
/// is left out of the fit: its annotators skipped stretches they could not
/// read, so a real beat there counts as spurious and the model learned each
/// record's habits instead. On the quarter held out it is calibrated to
/// 1.37 % (expected calibration error), and a little cautious at the top: 97 %
/// predicted where 98 % were real.
pub const QRS_PT1: Option<Logistic<NQ>> = Some(Logistic {
    bias: 0.167814,
    w: [
        0.947158, -0.382409, 0.344810, -0.389818, 0.473048, 0.369443, -1.812427, 1.568032, 1.520297,
    ],
});

/// Fitted by `ecg-eval fit-beat-conf` for `beats.clinical@4`: out-of-fold
/// scores for every training record of MIT-BIH and the supraventricular
/// corpus - each quarter scored by a bank refitted without it - together with
/// the training zones of the Long-Term AF and Long-Term ST corpora, which no
/// detector was fitted on; 174 records, each weighing the same. Class N is
/// the reference. See `PHASE-11.md` §18 for why it is a mix: a calibration
/// fitted on the long-term records alone was calibrated there and said 32 % for
/// a MIT-BIH ventricular call that was right 72 % of the time.
pub const BEATS_CLINICAL4: Option<Softmax<NB>> = Some(Softmax {
    bias: [0.000000, -2.935872, -1.603803, -6.148267],
    w: [
        [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        [
            0.084314, 0.150091, -0.080899, -2.383425, -0.067497, -0.040144,
        ],
        [
            0.283742, 0.084875, 0.016681, -1.562722, -0.051106, -0.087180,
        ],
        [
            0.177240, -0.085609, -0.002183, -1.293278, -0.062153, -0.053667,
        ],
    ],
});

/// Fitted by `ecg-eval fit-beat-conf` for `beats.patch@3`, on the patch
/// corpus's development zone - analyst truth, outside the device's noise -
/// against the engine's own verdict, before a supraventricular run relabels
/// it: 15.2 million beats. On the quarter of recordings held out it is
/// calibrated to 3.85 %, and the error is one thing: an N call reads 97 % and
/// is right 93 % of the time, because the supraventricular beats it misses sit
/// in runs that no single beat's scores can show, and recordings differ
/// tenfold in how many they have. The run is reported on its own
/// (`ECG_EV_SV_RUN`).
pub const BEATS_PATCH3: Option<Softmax<NB>> = Some(Softmax {
    bias: [0.000000, -1.966072, -2.816300, -17.430817],
    w: [
        [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        [-0.014956, 0.094335, 0.000300, -0.555909, 0.011780, 0.100310],
        [
            1.176300, 0.059857, -0.008372, -1.012921, -0.017458, -0.705576,
        ],
        [0.467019, 0.170600, 0.246595, -0.117794, 0.059661, -0.103558],
    ],
});
