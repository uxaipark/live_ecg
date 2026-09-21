//! Atrial fibrillation detection from RR irregularity.
//!
//! AF is the one common arrhythmia that a single lead can identify from timing
//! alone: conduction through the AV node becomes effectively random, so the RR
//! series loses its structure. Everything here measures that loss in a different
//! way, because no single statistic separates AF from the two things that
//! imitate it.
//!
//! # What imitates AF, and how each feature answers it
//!
//! * **Frequent ectopy.** A run of premature beats makes RR irregular too, but
//!   the irregularity is *sparse and structured* - a few large excursions
//!   against an otherwise regular series. AF is *diffuse*: every interval is
//!   irregular by a similar amount. `rmssd_over_mad` is the direct test, since a
//!   root-mean-square is dragged by outliers and a median is not.
//! * **Sinus arrhythmia and rate change.** Respiratory variation is smooth and
//!   correlated between neighbouring intervals; AF is not. `tpr` and the entropy
//!   terms answer that, and every amplitude feature is normalised by the median
//!   interval so a rate change alone does not move them.
//!
//! Features are computed over a sliding window of usable intervals and are all
//! dimensionless, so they do not depend on heart rate or sampling rate.

use crate::rr::RrSample;

/// Maximum window length the detector will consider.
pub const MAX_WINDOW: usize = 128;

/// Number of features in the model input vector.
pub const NF: usize = 11;

#[derive(Debug, Clone, Copy)]
pub struct AfConfig {
    /// Number of usable RR intervals per decision.
    pub window_beats: usize,
    /// The window must not span more than this, or the intervals are too sparse
    /// to describe one rhythm - long gaps mean the detector was blind, not that
    /// the rhythm was stable across them.
    pub max_span_s: f32,
    /// Sample-entropy tolerance: the larger of an absolute floor and a fraction
    /// of the window's median interval.
    ///
    /// A fixed tolerance makes the feature rate-dependent, and that is not a
    /// subtlety - it is the single largest source of false alarms on 24-hour
    /// recordings. At 110 bpm, 30 ms is 5.5% of the interval; at 54 bpm it is
    /// 2.7%, so at night almost no template matches, sample entropy rises, and
    /// ordinary sleeping bradycardia with deep respiratory variation reads as
    /// fibrillation. Scaling the tolerance with the interval keeps the question
    /// the same one at every heart rate.
    pub sampen_r_ms: f32,
    pub sampen_r_frac: f32,
    /// Logistic model.
    pub weights: AfWeights,
    /// Probability to enter and to leave the AF state. The gap is deliberate:
    /// a single borderline window should not open or close an episode.
    pub enter_prob: f32,
    pub exit_prob: f32,
    /// Episodes shorter than this are not reported. Thirty seconds is the
    /// clinical threshold for AF.
    pub min_episode_s: f32,
}

impl Default for AfConfig {
    fn default() -> Self {
        // Chosen on DEV by an explicit criterion: the lowest false-alarm rate on
        // AF-free normal-sinus Holters that still finds at least 41 of the 43
        // reference episodes. Not by maximising accuracy - on a corpus that is
        // 8% AF, and on a patch that is far less, false alarms are what decide
        // whether anyone can use the output.
        //
        // This setting gives 0.50 false alarms per 24 h of normal sinus signal
        // (worst subject 3.1). Lowering `enter_prob` to 0.92 raises
        // duration-weighted sensitivity from 82% to 89% and finds 42 of 43
        // episodes, at 1.8 false alarms per 24 h - a reasonable choice for a
        // screening deployment where a reviewer sees every alarm.
        AfConfig {
            window_beats: 20,
            max_span_s: 90.0,
            sampen_r_ms: 20.0,
            sampen_r_frac: 0.05,
            weights: AfWeights::BASELINE,
            enter_prob: 0.95,
            exit_prob: 0.75,
            min_episode_s: 30.0,
        }
    }
}

/// Features of one window. Every one is dimensionless.
#[derive(Debug, Clone, Copy, Default)]
pub struct AfFeatures {
    /// Root mean square of successive differences, over the median interval.
    pub rmssd_norm: f32,
    /// Median absolute successive difference, over the median interval. Unlike
    /// `rmssd_norm` this ignores a handful of large excursions.
    pub mad_norm: f32,
    /// `rmssd_norm / mad_norm`. Near 1.3 when every interval is irregular by a
    /// similar amount; far above that when a few ectopic beats carry all of it.
    pub rmssd_over_mad: f32,
    /// Share of successive differences exceeding 5% of the median interval.
    pub pnn_norm: f32,
    /// Normalised Shannon entropy of the interval histogram.
    pub shannon_rr: f32,
    /// Normalised Shannon entropy of the successive-difference histogram.
    pub shannon_drr: f32,
    /// Coefficient of sample entropy (Lake & Moorman), density-corrected so it
    /// stays comparable across heart rates.
    pub cosen: f32,
    /// Turning-point ratio. An independent random series gives 2/3; a smoothly
    /// varying one gives less.
    pub tpr: f32,
    /// Poincare short-term dispersion over the median interval.
    pub sd1_norm: f32,
    /// Poincare axis ratio.
    pub sd1_sd2: f32,
    /// Interquartile range over the median interval.
    pub iqr_norm: f32,
    /// Heart rate implied by the median interval, beats per minute.
    pub hr: f32,
    /// Lag-1 autocorrelation of the successive differences.
    ///
    /// An ectopic beat produces a short interval immediately followed by a long
    /// one, so its differences alternate in sign and this goes strongly
    /// negative. Fibrillation has no such structure and it sits near zero. This
    /// is the feature that separates frequent ectopy from AF, which the
    /// amplitude-based measures cannot do - both make the series irregular.
    pub drr_acf1: f32,
    /// Lag-1 autocorrelation of the interval series itself.
    ///
    /// Respiratory sinus arrhythmia is a *smooth* oscillation: each interval is
    /// close to the one before it, so this sits high. Fibrillation is
    /// uncorrelated beat to beat and it collapses toward zero. Amplitude
    /// measures cannot tell the two apart - marked sinus arrhythmia in a healthy
    /// young subject is genuinely as variable as AF - and it is the single
    /// largest source of false alarms on normal-sinus Holters.
    pub rr_acf1: f32,
    /// Share of intervals within 5% of the median.
    ///
    /// Sinus rhythm interrupted by ectopy keeps a dominant interval: most beats
    /// are still on time. In fibrillation there is no dominant interval at all,
    /// so this collapses. It answers the same question as `drr_acf1` from the
    /// other side.
    pub frac_near_median: f32,
}

impl AfFeatures {
    /// Model input vector, in the order [`AfWeights`] expects.
    #[inline]
    pub fn vector(&self) -> [f32; NF] {
        [
            self.rmssd_norm,
            self.mad_norm,
            self.rmssd_over_mad,
            self.pnn_norm,
            self.shannon_drr,
            self.cosen,
            self.tpr,
            self.sd1_sd2,
            self.drr_acf1,
            self.frac_near_median,
            self.rr_acf1,
        ]
    }

    pub const NAMES: [&'static str; NF] = [
        "rmssd_norm",
        "mad_norm",
        "rmssd_over_mad",
        "pnn_norm",
        "shannon_drr",
        "cosen",
        "tpr",
        "sd1_sd2",
        "drr_acf1",
        "frac_near_median",
        "rr_acf1",
    ];
}

/// Logistic model over [`AfFeatures::vector`].
#[derive(Debug, Clone, Copy)]
pub struct AfWeights {
    pub bias: f32,
    pub w: [f32; NF],
}

impl AfWeights {
    /// Fitted by `ecg-eval fit-af` on the TRAIN zone of AFDB and LTAFDB
    /// (100 records, 604k windows, 58% atrial fibrillation), using the reference
    /// beat annotations so the rhythm model is not fitted around this project's
    /// own detector errors. Eight coefficients, no per-feature statistics needed
    /// at runtime: the standardisation is folded in.
    ///
    /// Standardised influence, largest first: `cosen` +2.91, `pnn_norm` +2.11,
    /// `mad_norm` -1.17, `shannon_drr` +0.97. Sample entropy carries the most,
    /// which is what the literature that introduced it for this purpose reports.
    /// Fitted by `ecg-eval fit-af` on the TRAIN zone of AFDB, LTAFDB and the
    /// Normal Sinus Rhythm Database (106 records, 640k windows, 55% AF), on the
    /// corpora's own beat annotations so the rhythm model is not fitted around
    /// this project's detector errors.
    ///
    /// The normal-sinus records are in the fit on purpose. Trained on AFDB and
    /// LTAFDB alone the negative class is almost entirely AF-adjacent rhythm,
    /// which is not what the model will meet: a patch spends its life on sinus
    /// rhythm with occasional ectopy and respiratory variation. That model raised
    /// 17 false alarms per 24 h of normal sinus signal. Making the negative class
    /// representative is what fixed it, not a larger model.
    ///
    /// The L2 penalty is 0.015, which is high for eleven features and deliberate.
    /// Several of these statistics measure nearly the same thing - `rmssd_norm`,
    /// `mad_norm` and `sd1_norm` are all dispersion of successive differences -
    /// and a weakly penalised fit answers that with large cancelling
    /// coefficients that do not survive a new subject. Raising the penalty cut
    /// the worst subject's false-alarm rate from 29 per 24 h to 3.1 while
    /// costing one reference episode out of 43.
    pub const BASELINE: AfWeights = AfWeights {
        bias: -3.843427,
        w: [
            1.440547, -3.119119, 0.000017, 3.095825, 5.586383, 1.052929, 0.895601, -0.000005,
            -0.065110, -2.656145, 0.348843,
        ],
    };

    #[inline]
    pub fn probability(&self, f: &AfFeatures) -> f32 {
        let v = f.vector();
        let mut z = self.bias;
        for (wi, vi) in self.w.iter().zip(v.iter()) {
            z += wi * vi;
        }
        1.0 / (1.0 + (-z).exp())
    }
}

/// One window's verdict.
#[derive(Debug, Clone, Copy)]
pub struct AfWindow {
    /// Sample index of the beat closing the window.
    pub sample: u64,
    /// Sample index of the beat opening it.
    pub start_sample: u64,
    pub features: AfFeatures,
    pub probability: f32,
    /// State after hysteresis.
    pub in_af: bool,
}

pub struct AfDetector {
    cfg: AfConfig,
    fs: f64,
    rr: [f32; MAX_WINDOW],
    pos: [u64; MAX_WINDOW],
    n: usize,
    idx: usize,
    in_af: bool,
    // scratch, so a decision allocates nothing
    buf: [f32; MAX_WINDOW],
    drr: [f32; MAX_WINDOW],
    sorted: [f32; MAX_WINDOW],
}

impl AfDetector {
    pub fn new(fs: f64, cfg: AfConfig) -> Self {
        assert!(cfg.window_beats >= 8 && cfg.window_beats <= MAX_WINDOW);
        AfDetector {
            cfg,
            fs,
            rr: [0.0; MAX_WINDOW],
            pos: [0; MAX_WINDOW],
            n: 0,
            idx: 0,
            in_af: false,
            buf: [0.0; MAX_WINDOW],
            drr: [0.0; MAX_WINDOW],
            sorted: [0.0; MAX_WINDOW],
        }
    }

    pub fn config(&self) -> &AfConfig {
        &self.cfg
    }

    /// Feed one interval. Unusable intervals are dropped rather than repaired:
    /// see the note on [`crate::rr`].
    pub fn push(&mut self, s: &RrSample) -> Option<AfWindow> {
        if !s.usable() {
            return None;
        }
        let w = self.cfg.window_beats;
        self.rr[self.idx] = s.rr_ms;
        self.pos[self.idx] = s.sample;
        self.idx = (self.idx + 1) % w;
        self.n = (self.n + 1).min(w);
        if self.n < w {
            return None;
        }

        // Copy the window into contiguous order, oldest first.
        for k in 0..w {
            let i = (self.idx + k) % w;
            self.buf[k] = self.rr[i];
        }
        let start_sample = self.pos[self.idx % w];
        let span_s = (s.sample.saturating_sub(start_sample)) as f32 / self.fs as f32;
        if span_s > self.cfg.max_span_s {
            // The window reaches across a blind stretch; it does not describe one
            // rhythm, so it gets no vote.
            return None;
        }

        let features = self.features(w);
        let probability = self.cfg.weights.probability(&features);
        self.in_af = if self.in_af {
            probability >= self.cfg.exit_prob
        } else {
            probability >= self.cfg.enter_prob
        };

        Some(AfWindow {
            sample: s.sample,
            start_sample,
            features,
            probability,
            in_af: self.in_af,
        })
    }

    fn features(&mut self, w: usize) -> AfFeatures {
        let rr = &self.buf[..w];

        self.sorted[..w].copy_from_slice(rr);
        let sorted = &mut self.sorted[..w];
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[w / 2].max(1.0);
        let iqr = sorted[(w * 3) / 4] - sorted[w / 4];

        let m = w - 1;
        for i in 0..m {
            self.drr[i] = rr[i + 1] - rr[i];
        }
        let drr = &self.drr[..m];

        let rmssd = (drr.iter().map(|d| (d * d) as f64).sum::<f64>() / m as f64).sqrt() as f32;
        let mut abs_d = [0.0f32; MAX_WINDOW];
        for i in 0..m {
            abs_d[i] = drr[i].abs();
        }
        abs_d[..m].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = abs_d[m / 2];
        let pnn = abs_d[..m].iter().filter(|&&d| d > 0.05 * median).count() as f32 / m as f32;

        // Poincare axes. SD1 is the short-term (beat-to-beat) dispersion, SD2 the
        // long-term; the ratio is close to 1 when successive intervals carry no
        // information about each other.
        let var_d = variance(drr);
        let var_rr = variance(rr);
        let sd1 = (var_d * 0.5).max(0.0).sqrt();
        let sd2 = (2.0 * var_rr - 0.5 * var_d).max(0.0).sqrt();

        let mut turns = 0usize;
        for i in 1..m {
            if (drr[i] > 0.0) != (drr[i - 1] > 0.0) {
                turns += 1;
            }
        }
        let tpr = if m > 1 {
            turns as f32 / (m - 1) as f32
        } else {
            0.0
        };

        let drr_mean = drr.iter().map(|&v| v as f64).sum::<f64>() / m as f64;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..m {
            let d = drr[i] as f64 - drr_mean;
            den += d * d;
            if i + 1 < m {
                num += d * (drr[i + 1] as f64 - drr_mean);
            }
        }
        let drr_acf1 = if den > 1e-9 { (num / den) as f32 } else { 0.0 };

        let near = rr
            .iter()
            .filter(|&&v| (v - median).abs() <= 0.05 * median)
            .count();
        let frac_near_median = near as f32 / w as f32;

        let rr_mean = rr.iter().map(|&v| v as f64).sum::<f64>() / w as f64;
        let mut rnum = 0.0f64;
        let mut rden = 0.0f64;
        for i in 0..w {
            let d = rr[i] as f64 - rr_mean;
            rden += d * d;
            if i + 1 < w {
                rnum += d * (rr[i + 1] as f64 - rr_mean);
            }
        }
        let rr_acf1 = if rden > 1e-9 {
            (rnum / rden) as f32
        } else {
            0.0
        };

        AfFeatures {
            rmssd_norm: rmssd / median,
            mad_norm: mad / median,
            rmssd_over_mad: rmssd / mad.max(1e-3),
            pnn_norm: pnn,
            shannon_rr: shannon(rr, 16),
            shannon_drr: shannon(drr, 16),
            cosen: cosen(
                rr,
                self.cfg.sampen_r_ms.max(self.cfg.sampen_r_frac * median),
                median,
            ),
            tpr,
            sd1_norm: sd1 / median,
            sd1_sd2: sd1 / sd2.max(1e-3),
            iqr_norm: iqr / median,
            hr: 60_000.0 / median,
            drr_acf1,
            frac_near_median,
            rr_acf1,
        }
    }

    pub fn in_af(&self) -> bool {
        self.in_af
    }

    pub fn reset(&mut self) {
        self.n = 0;
        self.idx = 0;
        self.in_af = false;
    }
}

fn variance(x: &[f32]) -> f32 {
    if x.len() < 2 {
        return 0.0;
    }
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
    (x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / x.len() as f64) as f32
}

/// Normalised Shannon entropy of a fixed-bin histogram.
///
/// Trimmed at both ends before binning: one extreme interval would otherwise set
/// the bin width for the whole window and collapse everything else into a single
/// bin, which reads as *perfect regularity* exactly when a beat was missed.
fn shannon(x: &[f32], bins: usize) -> f32 {
    let n = x.len();
    if n < 4 {
        return 0.0;
    }
    let mut s = [0.0f32; MAX_WINDOW];
    s[..n].copy_from_slice(x);
    s[..n].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let trim = (n / 16).max(1);
    let (lo, hi) = (s[trim], s[n - 1 - trim]);
    let width = (hi - lo) / bins as f32;
    // A window whose trimmed range collapsed carries no information to bin.
    if width.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return 0.0;
    }
    let mut counts = [0u32; 64];
    let bins = bins.min(64);
    for &v in x {
        let b = (((v - lo) / width) as isize).clamp(0, bins as isize - 1) as usize;
        counts[b] += 1;
    }
    let total = n as f32;
    let mut h = 0.0f32;
    for &c in counts[..bins].iter() {
        if c > 0 {
            let p = c as f32 / total;
            h -= p * p.ln();
        }
    }
    h / (bins as f32).ln()
}

/// Coefficient of sample entropy (Lake & Moorman, 2011).
///
/// Sample entropy with `m = 1` measures how often a pair of intervals that match
/// each other is still matched one beat later. The density correction
/// `ln(2r) - ln(mean)` is what makes it usable on short windows across different
/// heart rates, which is why this and not plain sample entropy.
fn cosen(rr: &[f32], r_ms: f32, median: f32) -> f32 {
    let n = rr.len();
    if n < 4 {
        return 0.0;
    }
    let r = r_ms.max(1.0);
    let (mut b, mut a) = (0u32, 0u32);
    for i in 0..n - 1 {
        for j in 0..n - 1 {
            if i == j {
                continue;
            }
            if (rr[i] - rr[j]).abs() <= r {
                b += 1;
                if (rr[i + 1] - rr[j + 1]).abs() <= r {
                    a += 1;
                }
            }
        }
    }
    if b == 0 {
        // No template matched: maximally irregular for this window length. Report
        // the largest entropy the window can express rather than an infinity.
        return ((n - 1) as f32).ln() + (2.0 * r).ln() - median.ln();
    }
    let sampen = if a == 0 {
        ((b as f32) * (n - 1) as f32).ln()
    } else {
        -((a as f32) / (b as f32)).ln()
    };
    sampen + (2.0 * r).ln() - median.ln()
}
