//! Per-channel real-time pipeline: filter bank -> quality gate -> QRS detection.
//!
//! One `ChannelPipeline` owns all state for one patch. It is `Send`, holds no
//! global state and allocates only at construction and when the caller's event
//! buffer grows, so a server scales by giving each worker thread a slice of the
//! channels rather than by sharing anything.

mod preprocess;

pub use preprocess::{Bands, Mains, PreprocessConfig, Preprocessor};

use ecg_beats::{
    BeatAnalyzer, BeatBank, BeatClass, BeatConfig, BeatVerdict, DelineateConfig, Delineation,
    Delineator,
};
use ecg_qrs::{QrsConfig, QrsDetector, QrsEvent};
use ecg_quality::{Quality, QualityConfig, QualityMonitor, QualitySample};
use ecg_rhythm::{
    AfConfig, AfDetector, AfWindow, Beat, EpisodeConfig, EpisodeTracker, RhythmBank, RhythmConfig,
    RhythmEpisode, RrConfig, RrSample, RrStream, VfConfig, VfDetector, VfWindow,
};

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
    pub rhythm: RhythmConfig,
    pub delineate: DelineateConfig,
    pub vf: VfConfig,
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
            rhythm: RhythmConfig::new(fs),
            delineate: DelineateConfig::new(fs),
            vf: VfConfig::new(fs),
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
    /// One entry per delineated beat. Lags `classes` by one further beat: the
    /// T wave's extent is bounded by the interval that follows it.
    pub waves: Vec<Delineation>,
    /// Rhythm episodes that ended during this block.
    pub episodes: Vec<RhythmEpisode>,
    /// One entry per completed ventricular-fibrillation decision window.
    pub vf: Vec<VfWindow>,
    /// Fibrillation episodes that ended during this block, as (start, end).
    pub vf_episodes: Vec<(u64, u64)>,
    /// Samples during which beat-derived analysis was withheld because the
    /// fibrillation detector was raised. Counted so the silence is visible: a
    /// consumer seeing no beats must be able to tell "nothing happened" from
    /// "we stopped believing the beats".
    pub suppressed_samples: u64,
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
        self.waves.clear();
        self.episodes.clear();
        self.vf.clear();
        self.vf_episodes.clear();
        self.suppressed_samples = 0;
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
    rhythm: RhythmBank,
    delineator: Delineator,
    /// The three most recent R positions, so the middle one can be delineated
    /// with a real interval on each side rather than a guessed one.
    recent_beats: [Option<u64>; 3],
    vf: VfDetector,
    vf_tracker: EpisodeTracker,
    /// A second tracker at a higher bar, for withholding rather than reporting.
    vf_suppress: EpisodeTracker,
    /// Whether beat-derived analysis is currently withheld.
    suppressing: bool,
    /// The interval whose closing beat has not been classified yet.
    ///
    /// The classifier runs one beat behind the detector, so an interval's
    /// ectopy status is known one step after the interval itself. Holding it
    /// here costs the rhythm path one more beat of latency and is what lets the
    /// AF detector see a series with ectopy already taken out.
    pending_interval: Option<RrSample>,
    /// Class of the beat that opens `pending_interval`.
    prev_ventricular: bool,
    prev_supraventricular: bool,
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
        // The delineator marks P and T positions on the low-frequency tap, so it
        // needs that tap's delay to report them on the input time base. Six hertz
        // is where P and T energy sits.
        let mut delineator = Delineator::new(cfg.delineate);
        // Each wave is measured on the tap where it is visible, and every tap
        // lags the input. The lag is evaluated at the centre of that wave's own
        // energy, because a two-pole 0.5-10 Hz band delays 1 Hz by 155 ms and
        // 10 Hz by 24 ms - one figure for both P and T would be wrong for both.
        let d = cfg.delineate;
        delineator.set_delays(
            pre.group_delay_samples(d.qrs_ref_hz) + pre.qrs_group_delay_samples(d.qrs_ref_hz),
            pre.group_delay_samples(d.p_ref_hz) + pre.pt_group_delay_samples(d.p_ref_hz),
            pre.group_delay_samples(d.t_ref_hz) + pre.pt_group_delay_samples(d.t_ref_hz),
        );
        ChannelPipeline {
            pre,
            qual: QualityMonitor::new(cfg.quality),
            qrs,
            rr: RrStream::new(cfg.rr),
            af: AfDetector::new(cfg.fs, cfg.af),
            beats: BeatAnalyzer::new(cfg.beats),
            bank: cfg.bank,
            rhythm: RhythmBank::new(cfg.rhythm),
            delineator,
            recent_beats: [None; 3],
            vf: VfDetector::new(cfg.vf),
            suppressing: false,
            vf_tracker: EpisodeTracker::new(
                cfg.fs,
                EpisodeConfig {
                    bridge_s: cfg.vf.bridge_s,
                    min_episode_s: cfg.vf.min_episode_s,
                },
            ),
            vf_suppress: EpisodeTracker::new(
                cfg.fs,
                EpisodeConfig {
                    bridge_s: cfg.vf.bridge_s,
                    min_episode_s: cfg.vf.suppress_min_s,
                },
            ),
            pending_interval: None,
            prev_ventricular: false,
            prev_supraventricular: false,
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
            // Rail contact only - deliberately *not* the low-amplitude flag.
            //
            // A detached lead and an asystole both read as a flat, low-amplitude
            // trace, and from one lead without electrode impedance there is no
            // clean way to tell them apart. Suppressing asystole whenever the
            // amplitude collapses means suppressing it exactly when it happens:
            // measured on the sick-sinus record 232, which has fourteen genuine
            // asystolic pauses and near-perfect beat detection, not one was
            // reported.
            //
            // So the tie is broken by consequence rather than by likelihood. A
            // false asystole alarm is reviewed and dismissed; a missed one is
            // not. The quality flags travel alongside the episode, so a consumer
            // can see that LEAD_OFF was also raised and weigh it.
            let lead_ok = q.flags & ecg_quality::flags::SATURATION == 0;
            self.rr.observe_lead(lead_ok);
            self.beats.push_sample(b.clean, b.qrs, learn_ok);
            self.delineator.push_sample(b.qrs, b.pt);

            // Fibrillation detection runs beside the beat path, not after it.
            // In fibrillation there are no beats, so everything downstream of
            // QRS detection is describing an artefact - this is the one stage
            // that still means something there.
            if let Some(w) = self.vf.process(b.clean) {
                if let Some(e) = self.vf_tracker.update(w.sample, w.in_vf) {
                    out.vf_episodes.push((e.start, e.end));
                }
                self.vf_suppress
                    .update(w.sample, w.probability >= self.cfg.vf.suppress_prob);
                out.vf.push(w);
            }

            // Suppression uses the **confirmed** episode, not the faster
            // per-window state, and the first attempt had that backwards.
            //
            // The argument for the looser signal was that suppression is cheap:
            // it costs a suspended conclusion where an alarm costs a clinician's
            // attention. The measurement says otherwise. Suppression deletes
            // true findings, so it is the more consequential decision, and the
            // per-window state flickers far too readily to make it: driving
            // suppression from it took atrial fibrillation sensitivity from
            // 86.0% to 28.7% and asystole from 100% to 14%, because the flicker
            // landed on exactly the rhythms worth reporting.
            //
            // The confirmed episode fires on none of AFDB, MIT-BIH or the Normal
            // Sinus corpus - 100% specificity on all three - which is the bar a
            // decision this destructive needs. The cost is latency: the window
            // plus the confirmation, so roughly eight seconds of artefact still
            // gets out before the silence starts.
            let suspected = self.vf_suppress.confirmed();
            if suspected != self.suppressing {
                if !suspected {
                    // Leaving fibrillation. The interval history now spans a
                    // stretch of nothing, and splicing across it would present
                    // the whole episode as one enormous interval - which, at the
                    // right length, is reported as asystole. Same hazard as a
                    // lost packet, same answer.
                    self.invalidate_beat_history();
                }
                self.suppressing = suspected;
            }
            if self.suppressing {
                // The raw detections still go out - a reviewer needs to see what
                // the detector did - but nothing is built on them. In
                // fibrillation there are no beats, so an RR interval, a beat
                // class and every rhythm episode derived from them describe an
                // artefact. Measured on the fibrillation corpus before this
                // existed: 982 ventricular runs, 707 ventricular tachycardias,
                // 303 pauses and 60 asystoles across 12.8 hours, none of them
                // real.
                //
                // This check must come *after* the fibrillation stage has seen
                // the sample. Skipping the stage while suppressing starves the
                // detector that decides when to stop, and suppression becomes
                // permanent: one four-second episode silenced the remaining ten
                // hours of an AFDB record and took corpus sensitivity from 86%
                // to 34%.
                out.suppressed_samples += 1;
                out.beats.extend_from_slice(&self.scratch);
                // The analyser still consumes the sample. Skipping it would stop
                // its clock while the detector's kept running, and every beat
                // after the episode would be measured against the wrong window.
                self.beats.push_sample(b.clean, b.qrs, learn_ok);
                self.delineator.push_sample(b.qrs, b.pt);
                out.quality = Some(q);
                self.n += 1;
                continue;
            }
            if !(self.cfg.suppress_unusable && level == Quality::Unusable) {
                out.beats.extend_from_slice(&self.scratch);
            }
            for i in 0..self.scratch.len() {
                let ev = self.scratch[i];
                let interval = self.rr.push(&ev);
                // This verdict belongs to the beat *before* `ev`, which is the
                // beat that closes the interval currently held.
                self.recent_beats = [self.recent_beats[1], self.recent_beats[2], Some(ev.sample)];
                let mut wave = None;
                if let (Some(a), Some(m), Some(c)) = (
                    self.recent_beats[0],
                    self.recent_beats[1],
                    self.recent_beats[2],
                ) {
                    if let Some(d) = self.delineator.delineate(m, Some(m - a), Some(c - m)) {
                        out.waves.push(d);
                        wave = Some(d);
                    }
                }
                // The delineation describes the same beat the analyser is about
                // to finalise - both lag the detector by one, for the same
                // reason - so the atrial evidence arrives with the morphology
                // rather than a beat too late to be used.
                if let Some(obs) = self.beats.push_beat(&ev, wave.as_ref()) {
                    let verdict = self.bank.classify(&obs);
                    out.classes.push(verdict);
                    let v = verdict.class == BeatClass::V;
                    let sv = verdict.class == BeatClass::S;
                    if let Some(mut held) = self.pending_interval.take() {
                        held.ventricular = self.prev_ventricular || v;
                        held.supraventricular = self.prev_supraventricular || sv;
                        out.intervals.push(held);
                        if let Some(w) = self.af.push(&held) {
                            out.af.push(w);
                        }
                        let beat = match verdict.class {
                            BeatClass::N => Beat::Normal,
                            BeatClass::S => Beat::Supraventricular,
                            BeatClass::V => Beat::Ventricular,
                            BeatClass::Unknown => Beat::Unknown,
                        };
                        self.rhythm.push(&held, beat, &mut out.episodes);
                    }
                    self.prev_ventricular = v;
                    self.prev_supraventricular = sv;
                }
                if interval.is_some() {
                    self.pending_interval = interval;
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

    /// Discard the beat-derived history without touching the clock.
    ///
    /// Shares its reasoning with [`Self::mark_gap`]: whenever the stream of
    /// beats has a hole in it - lost packets, or a stretch where the beats were
    /// not real - the state that spans the hole has to go, and the state that
    /// describes the patient stays.
    fn invalidate_beat_history(&mut self) {
        self.rr.reset();
        self.af.on_gap();
        // Zero unobserved samples: the signal *was* seen, its beats just were
        // not real. Only the derived history is discarded.
        self.beats.on_gap(0);
        self.delineator.on_gap(0);
        self.recent_beats = [None; 3];
        self.rhythm.on_gap();
        self.pending_interval = None;
        self.prev_ventricular = false;
        self.prev_supraventricular = false;
    }

    /// True while beat-derived analysis is being withheld.
    pub fn suppressing(&self) -> bool {
        self.suppressing
    }

    /// Declare that `samples` were lost before the next block.
    ///
    /// A gap is not silence and it is not continuous signal. Splicing the two
    /// sides together fabricates a waveform: the detector's search-back would
    /// reach across it, the RR stream would emit one enormous interval, and the
    /// episode layer would report a pause that never happened. A lost packet
    /// must not become a clinical finding.
    ///
    /// Windowed state is therefore discarded and adaptive state is kept - the
    /// thresholds, the beat template and the slow references all still describe
    /// this patient, and a dropped packet is not a new patient.
    pub fn mark_gap(&mut self, samples: u64) {
        self.pre.reset();
        self.qual.on_gap(samples);
        self.qrs.on_gap(samples);
        self.rr.reset();
        self.af.on_gap();
        self.beats.on_gap(samples);
        self.delineator.on_gap(samples);
        self.recent_beats = [None; 3];
        self.rhythm.on_gap();
        self.vf.on_gap(samples);
        self.pending_interval = None;
        self.prev_ventricular = false;
        self.prev_supraventricular = false;
        self.n += samples;
    }

    /// Close any rhythm episode still open at the end of a stream.
    pub fn finish(&mut self, out: &mut ChannelOutput) {
        self.rhythm.finish(&mut out.episodes);
        if let Some(e) = self.vf_tracker.finish() {
            out.vf_episodes.push((e.start, e.end));
        }
    }

    /// True while the fibrillation detector believes the rhythm is fibrillating.
    pub fn in_vf(&self) -> bool {
        self.vf.in_vf()
    }

    /// Diagnostic: AF windows too fragmented to judge, and windows seen.
    pub fn af_fragmentation(&self) -> (u64, u64) {
        self.af.fragmentation()
    }

    /// True while the AF detector believes the rhythm is fibrillating.
    pub fn in_af(&self) -> bool {
        self.af.in_af()
    }

    pub fn reset(&mut self) {
        self.beats.reset();
        self.delineator.reset();
        self.recent_beats = [None; 3];
        self.pre.reset();
        self.qual.reset();
        self.qrs.reset();
        self.rr.reset();
        self.af.reset();
        self.rhythm.reset();
        self.vf.reset();
        self.vf_tracker.reset();
        self.vf_suppress.reset();
        self.suppressing = false;
        self.pending_interval = None;
        self.prev_ventricular = false;
        self.prev_supraventricular = false;
        self.n = 0;
    }
}
