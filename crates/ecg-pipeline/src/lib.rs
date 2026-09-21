//! Per-channel real-time pipeline: filter bank -> quality gate -> QRS detection.
//!
//! One `ChannelPipeline` owns all state for one patch. It is `Send`, holds no
//! global state and allocates only at construction and when the caller's event
//! buffer grows, so a server scales by giving each worker thread a slice of the
//! channels rather than by sharing anything.

mod preprocess;

pub use preprocess::{Bands, Mains, PreprocessConfig, Preprocessor};

use ecg_beats::{BeatAnalyzer, BeatBank, BeatConfig, BeatVerdict};
use ecg_qrs::{QrsConfig, QrsDetector, QrsEvent};
use ecg_quality::{Quality, QualityConfig, QualityMonitor, QualitySample};
use ecg_rhythm::{AfConfig, AfDetector, AfWindow, RrConfig, RrSample, RrStream};

#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    pub fs: f64,
    pub preprocess: PreprocessConfig,
    pub quality: QualityConfig,
    pub qrs: QrsConfig,
    pub rr: RrConfig,
    pub af: AfConfig,
    pub beats: BeatConfig,
    pub bank: BeatBank,
    /// Detections inside an unusable stretch are suppressed rather than emitted.
    /// Off by default: dropping beats hides asystole, so the decision belongs to
    /// the caller. Threshold adaptation is gated regardless.
    pub suppress_unusable: bool,
}

impl PipelineConfig {
    pub fn new(fs: f64) -> Self {
        PipelineConfig {
            fs,
            preprocess: PreprocessConfig::new(fs),
            quality: QualityConfig::new(fs),
            qrs: QrsConfig::new(fs),
            rr: RrConfig::new(fs),
            af: AfConfig::default(),
            beats: BeatConfig::new(fs),
            bank: BeatBank::default(),
            suppress_unusable: false,
        }
    }
}

/// What one block of samples produced.
#[derive(Debug, Default)]
pub struct ChannelOutput {
    pub beats: Vec<QrsEvent>,
    /// One entry per beat that closed an interval.
    pub intervals: Vec<RrSample>,
    /// One entry per completed AF decision window.
    pub af: Vec<AfWindow>,
    /// One entry per classified beat. Lags `beats` by one beat: the interval
    /// following a beat is part of the evidence for what it was.
    pub classes: Vec<BeatVerdict>,
    /// Quality at the end of the block.
    pub quality: Option<QualitySample>,
    /// Samples spent in each quality level during the block.
    pub good_samples: u64,
    pub acceptable_samples: u64,
    pub unusable_samples: u64,
}

impl ChannelOutput {
    pub fn clear(&mut self) {
        self.beats.clear();
        self.intervals.clear();
        self.af.clear();
        self.classes.clear();
        self.quality = None;
        self.good_samples = 0;
        self.acceptable_samples = 0;
        self.unusable_samples = 0;
    }
}

pub struct ChannelPipeline {
    cfg: PipelineConfig,
    pre: Preprocessor,
    qual: QualityMonitor,
    qrs: QrsDetector,
    rr: RrStream,
    af: AfDetector,
    beats: BeatAnalyzer,
    bank: BeatBank,
    n: u64,
    scratch: Vec<QrsEvent>,
}

impl ChannelPipeline {
    pub fn new(mut cfg: PipelineConfig) -> Self {
        // One band-pass serves the detector and the quality monitor.
        cfg.preprocess.qrs_lo = cfg.qrs.bp_lo;
        cfg.preprocess.qrs_hi = cfg.qrs.bp_hi;
        cfg.preprocess.qrs_order = cfg.qrs.bp_order;
        let pre = Preprocessor::new(cfg.preprocess);
        let mut qrs = QrsDetector::new(cfg.qrs);
        // The detector marks fiducials on `clean`; hand it that tap's group delay
        // so R positions come back on the input time base.
        qrs.set_input_delay(pre.group_delay_samples(Preprocessor::FIDUCIAL_REF_HZ));
        ChannelPipeline {
            pre,
            qual: QualityMonitor::new(cfg.quality),
            qrs,
            rr: RrStream::new(cfg.rr),
            af: AfDetector::new(cfg.fs, cfg.af),
            beats: BeatAnalyzer::new(cfg.beats),
            bank: cfg.bank,
            cfg,
            n: 0,
            scratch: Vec::with_capacity(16),
        }
    }

    /// Group delay of the analysis tap at the QRS reference frequency, in samples.
    pub fn front_end_delay_samples(&self) -> f64 {
        self.pre.group_delay_samples(Preprocessor::FIDUCIAL_REF_HZ)
    }

    pub fn config(&self) -> &PipelineConfig {
        &self.cfg
    }

    pub fn samples_processed(&self) -> u64 {
        self.n
    }

    /// End-to-end delay from an R wave to its event, in samples.
    pub fn latency_samples(&self) -> usize {
        self.qrs.latency_samples()
    }

    /// Feed a block of samples (mV). Results are appended to `out`.
    pub fn push(&mut self, samples: &[f32], out: &mut ChannelOutput) {
        for &x in samples {
            let b = self.pre.process(x);
            let q = self
                .qual
                .process(b.raw, b.clean, b.baseline, b.hf, b.qrs, b.saturated);
            let level = q.level(&self.cfg.quality);

            match level {
                Quality::Good => out.good_samples += 1,
                Quality::Acceptable => out.acceptable_samples += 1,
                Quality::Unusable => out.unusable_samples += 1,
            }

            // The detector keeps running through bad signal but does not learn
            // from it; suppression of the resulting beats is a separate choice.
            let learn_ok = level != Quality::Unusable;
            self.scratch.clear();
            self.qrs
                .process_prefiltered(b.clean, b.qrs, learn_ok, &mut self.scratch);

            // An interval inherits the worst quality seen anywhere across its
            // span, not just at its closing beat: a noise burst in the middle is
            // what corrupts it.
            self.rr.observe_quality(learn_ok);
            self.beats.push_sample(b.clean, b.qrs, learn_ok);
            if !(self.cfg.suppress_unusable && level == Quality::Unusable) {
                out.beats.extend_from_slice(&self.scratch);
            }
            for i in 0..self.scratch.len() {
                let ev = self.scratch[i];
                if let Some(interval) = self.rr.push(&ev) {
                    out.intervals.push(interval);
                    if let Some(w) = self.af.push(&interval) {
                        out.af.push(w);
                    }
                }
                if let Some(obs) = self.beats.push_beat(&ev) {
                    out.classes.push(self.bank.classify(&obs));
                }
            }

            out.quality = Some(q);
            self.n += 1;
        }
    }

    /// Detector adaptive state. Diagnostic only.
    pub fn detector_state(&self) -> ecg_qrs::DetectorState {
        self.qrs.state()
    }

    pub fn mains_hz(&self) -> Option<f64> {
        self.pre.mains_hz()
    }

    /// True while the AF detector believes the rhythm is fibrillating.
    pub fn in_af(&self) -> bool {
        self.af.in_af()
    }

    pub fn reset(&mut self) {
        self.beats.reset();
        self.pre.reset();
        self.qual.reset();
        self.qrs.reset();
        self.rr.reset();
        self.af.reset();
        self.n = 0;
    }
}
