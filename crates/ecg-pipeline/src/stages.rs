//! The pipeline's replaceable stages.
//!
//! The whole engine can be replaced as one file behind the standard interface
//! (`ecg-ffi`). This is the finer grain: the five decision stages - QRS
//! detection, beat classification, atrial fibrillation, ventricular
//! fibrillation and supraventricular runs - each sit behind a small trait, and
//! the pipeline calls them only through it. A stage can then be replaced
//! inside a running engine, either by choosing another implementation the
//! engine carries (by name, from Rust or across the C interface) or, from
//! Rust, by handing the pipeline an implementation of one's own.
//!
//! # Names
//!
//! Every implementation the engine carries has a name of the form
//! `kind.variant@version`, listed by [`available`]. The version moves whenever
//! the implementation's output would, so a name identifies behaviour, not code:
//! a finding can be traced to the stage that produced it, and two engines that
//! report the same name for a stage made the same decisions there.
//!
//! The previous version of a stage is kept beside the current one where that
//! is cheap, so a deployment can compare the two on its own traffic, or fall
//! back, without a new engine.
//!
//! # What the traits promise
//!
//! The same input gives the same output; outputs come back in time order on
//! the input's time base; nothing allocates per sample. `stage_contract` in the
//! tests holds every registered implementation to that.

use ecg_beats::{BeatBank, BeatContext, BeatObservation, BeatVerdict};
use ecg_qrs::{DetectorState, QrsConfig, QrsDetector, QrsEvent};
use ecg_rhythm::{
    AfConfig, AfDetector, AfWindow, RrSample, SvRun, SvRunConfig, SvRunDetector, VfConfig,
    VfDetector, VfWeights, VfWindow,
};

/// QRS detection, sample by sample, on the front end's taps.
pub trait QrsStage: Send {
    fn id(&self) -> &'static str;
    /// One sample of the analysis tap and the QRS band; detections, if any,
    /// appended to `out` on the input time base.
    fn process(&mut self, clean: f32, band: f32, learn_ok: bool, out: &mut Vec<QrsEvent>);
    /// End-to-end delay from an R wave to its detection, in samples.
    fn latency_samples(&self) -> usize;
    /// The pipeline tells the stage how far its analysis tap lags the input,
    /// so detections can be reported on the input time base.
    fn set_input_delay(&mut self, _samples: f64) {}
    /// Adaptive state, for diagnostics; `None` for a stage that has none to
    /// show.
    fn state(&self) -> Option<DetectorState> {
        None
    }
    fn on_gap(&mut self, unobserved: u64);
    fn reset(&mut self);
}

/// Beat classification: one beat's features, in its rhythm context, to a
/// verdict.
pub trait BeatStage: Send {
    fn id(&self) -> &'static str;
    fn classify(&mut self, obs: &BeatObservation, context: BeatContext) -> BeatVerdict;
}

/// Atrial fibrillation, from the interval stream.
pub trait AfStage: Send {
    fn id(&self) -> &'static str;
    fn push(&mut self, interval: &RrSample) -> Option<AfWindow>;
    fn in_af(&self) -> bool;
    /// Whether fibrillation has been held long enough to condition other
    /// decisions on, as of sample `now`.
    fn sustained(&self, now: u64) -> bool;
    /// Windows too fragmented to judge, and windows seen. Diagnostic.
    fn fragmentation(&self) -> (u64, u64) {
        (0, 0)
    }
    fn on_gap(&mut self);
    fn reset(&mut self);
}

/// Ventricular fibrillation, from the analysis tap directly.
pub trait VfStage: Send {
    fn id(&self) -> &'static str;
    fn process(&mut self, clean: f32) -> Option<VfWindow>;
    fn in_vf(&self) -> bool;
    fn on_gap(&mut self, unobserved: u64);
    fn reset(&mut self);
}

/// Runs of supraventricular rhythm, from the interval stream.
pub trait SvRunStage: Send {
    fn id(&self) -> &'static str;
    fn push(
        &mut self,
        rr_ms: f32,
        sample: u64,
        usable: bool,
        fibrillating: bool,
        called: bool,
    ) -> Option<SvRun>;
    fn finish(&mut self) -> Option<SvRun>;
}

// ---- the engine's own implementations -----------------------------------

pub struct Qrs {
    id: &'static str,
    inner: QrsDetector,
}

impl Qrs {
    pub fn new(id: &'static str, cfg: QrsConfig) -> Self {
        Qrs {
            id,
            inner: QrsDetector::new(cfg),
        }
    }
}

impl QrsStage for Qrs {
    fn id(&self) -> &'static str {
        self.id
    }
    #[inline]
    fn process(&mut self, clean: f32, band: f32, learn_ok: bool, out: &mut Vec<QrsEvent>) {
        self.inner.process_prefiltered(clean, band, learn_ok, out);
    }
    fn latency_samples(&self) -> usize {
        self.inner.latency_samples()
    }
    fn set_input_delay(&mut self, samples: f64) {
        self.inner.set_input_delay(samples);
    }
    fn state(&self) -> Option<DetectorState> {
        Some(self.inner.state())
    }
    fn on_gap(&mut self, unobserved: u64) {
        self.inner.on_gap(unobserved);
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

pub struct Beats {
    id: &'static str,
    bank: BeatBank,
}

impl Beats {
    pub fn new(id: &'static str, bank: BeatBank) -> Self {
        Beats { id, bank }
    }
}

impl BeatStage for Beats {
    fn id(&self) -> &'static str {
        self.id
    }
    fn classify(&mut self, obs: &BeatObservation, context: BeatContext) -> BeatVerdict {
        self.bank.classify_in(obs, context)
    }
}

pub struct Af {
    id: &'static str,
    inner: AfDetector,
}

impl Af {
    pub fn new(id: &'static str, fs: f64, cfg: AfConfig) -> Self {
        Af {
            id,
            inner: AfDetector::new(fs, cfg),
        }
    }
}

impl AfStage for Af {
    fn id(&self) -> &'static str {
        self.id
    }
    fn push(&mut self, interval: &RrSample) -> Option<AfWindow> {
        self.inner.push(interval)
    }
    fn in_af(&self) -> bool {
        self.inner.in_af()
    }
    fn sustained(&self, now: u64) -> bool {
        self.inner.sustained(now)
    }
    fn fragmentation(&self) -> (u64, u64) {
        self.inner.fragmentation()
    }
    fn on_gap(&mut self) {
        self.inner.on_gap();
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

pub struct Vf {
    id: &'static str,
    inner: VfDetector,
}

impl Vf {
    pub fn new(id: &'static str, cfg: VfConfig) -> Self {
        Vf {
            id,
            inner: VfDetector::new(cfg),
        }
    }
}

impl VfStage for Vf {
    fn id(&self) -> &'static str {
        self.id
    }
    #[inline]
    fn process(&mut self, clean: f32) -> Option<VfWindow> {
        self.inner.process(clean)
    }
    fn in_vf(&self) -> bool {
        self.inner.in_vf()
    }
    fn on_gap(&mut self, unobserved: u64) {
        self.inner.on_gap(unobserved);
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

pub struct SvRuns {
    id: &'static str,
    inner: SvRunDetector,
}

impl SvRuns {
    pub fn new(id: &'static str, cfg: SvRunConfig) -> Self {
        SvRuns {
            id,
            inner: SvRunDetector::new(cfg),
        }
    }
}

impl SvRunStage for SvRuns {
    fn id(&self) -> &'static str {
        self.id
    }
    fn push(
        &mut self,
        rr_ms: f32,
        sample: u64,
        usable: bool,
        fibrillating: bool,
        called: bool,
    ) -> Option<SvRun> {
        self.inner.push(rr_ms, sample, usable, fibrillating, called)
    }
    fn finish(&mut self) -> Option<SvRun> {
        self.inner.finish()
    }
}

// ---- the registry -------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    Qrs,
    Beats,
    Af,
    Vf,
    SvRun,
}

impl StageKind {
    pub const ALL: [StageKind; 5] = [
        StageKind::Qrs,
        StageKind::Beats,
        StageKind::Af,
        StageKind::Vf,
        StageKind::SvRun,
    ];

    /// The name's prefix, and the key in a selection string.
    pub fn key(self) -> &'static str {
        match self {
            StageKind::Qrs => "qrs",
            StageKind::Beats => "beats",
            StageKind::Af => "af",
            StageKind::Vf => "vf",
            StageKind::SvRun => "svrun",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StageInfo {
    pub kind: StageKind,
    pub name: &'static str,
    pub about: &'static str,
}

/// Every implementation the engine carries.
pub const STAGES: [StageInfo; 10] = [
    StageInfo {
        kind: StageKind::Qrs,
        name: "qrs.pt@1",
        about: "band-passed energy with adaptive thresholds and search-back",
    },
    StageInfo {
        kind: StageKind::Beats,
        name: "beats.clinical@4",
        about: "N/S/V/F detector bank; ventricular ensemble with wide-beat features",
    },
    StageInfo {
        kind: StageKind::Beats,
        name: "beats.clinical@3",
        about: "the same bank before the wide-beat features (fifteen-feature ventricular ensemble)",
    },
    StageInfo {
        kind: StageKind::Beats,
        name: "beats.patch@3",
        about: "patch bank: truth-trained ventricular ensemble at its bar, S reported only when near-certain",
    },
    StageInfo {
        kind: StageKind::Af,
        name: "af.logistic@1",
        about: "interval irregularity and atrial coherence, logistic, with episode confirmation",
    },
    StageInfo {
        kind: StageKind::Vf,
        name: "vf.spectral@2",
        about: "twelve waveform and spectral features, logistic",
    },
    StageInfo {
        kind: StageKind::Vf,
        name: "vf.linear@1",
        about: "the seven waveform features before the spectral ones",
    },
    StageInfo {
        kind: StageKind::SvRun,
        name: "svrun.off@1",
        about: "no rhythm-level supraventricular runs",
    },
    StageInfo {
        kind: StageKind::SvRun,
        name: "svrun.rate@2",
        about: "rate step into a regular tachycardia faster than 100/min",
    },
    StageInfo {
        kind: StageKind::SvRun,
        name: "svrun.rate@1",
        about: "rate step into a regular run, at any rate",
    },
];

pub fn available() -> &'static [StageInfo] {
    &STAGES
}

pub fn find(name: &str) -> Option<&'static StageInfo> {
    STAGES.iter().find(|s| s.name == name)
}

/// Which implementation each stage uses. `None` builds the stage from the
/// pipeline's configuration, which is what every preset does; a name replaces
/// it with that registered implementation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageSelection {
    pub qrs: Option<&'static str>,
    pub beats: Option<&'static str>,
    pub af: Option<&'static str>,
    pub vf: Option<&'static str>,
    pub sv_run: Option<&'static str>,
}

impl StageSelection {
    /// Parse `kind=name;kind=name`, e.g. `vf=vf.linear@1;svrun=svrun.off@1`.
    /// Every name must be registered and of the kind it is given for.
    pub fn parse(spec: &str) -> Result<StageSelection, String> {
        let mut s = StageSelection::default();
        for part in spec
            .split([';', ','])
            .map(str::trim)
            .filter(|p| !p.is_empty())
        {
            let (key, name) = part
                .split_once('=')
                .ok_or_else(|| format!("{part:?} is not kind=name"))?;
            let info = find(name.trim()).ok_or_else(|| format!("no stage named {name:?}"))?;
            if info.kind.key() != key.trim() {
                return Err(format!(
                    "{} is a {} stage, not {key}",
                    info.name,
                    info.kind.key()
                ));
            }
            let slot = match info.kind {
                StageKind::Qrs => &mut s.qrs,
                StageKind::Beats => &mut s.beats,
                StageKind::Af => &mut s.af,
                StageKind::Vf => &mut s.vf,
                StageKind::SvRun => &mut s.sv_run,
            };
            *slot = Some(info.name);
        }
        Ok(s)
    }
}

/// The name a stage built from configuration answers to: a registered name
/// when the configuration is exactly that implementation's, otherwise the
/// kind with `.configured`, so a tuned stage is never mistaken for a
/// registered one.
fn same_detector(a: &ecg_beats::BinaryDetector, b: &ecg_beats::BinaryDetector) -> bool {
    use ecg_beats::detectors::Model;
    a.threshold == b.threshold
        && match (&a.model, &b.model) {
            (Model::Gbdt(x), Model::Gbdt(y)) => std::ptr::eq(x.nodes, y.nodes) && x.bias == y.bias,
            (Model::Linear(x), Model::Linear(y)) => x.bias == y.bias && x.w == y.w,
            _ => false,
        }
}

fn same_bank(a: &BeatBank, b: &BeatBank) -> bool {
    same_detector(&a.ventricular, &b.ventricular)
        && same_detector(&a.supraventricular, &b.supraventricular)
        && same_detector(&a.fusion, &b.fusion)
        && a.supraventricular_in_af == b.supraventricular_in_af
        && a.supraventricular_report == b.supraventricular_report
}

fn beats_name(bank: &BeatBank) -> &'static str {
    let same = |b: BeatBank| same_bank(&b, bank);
    if same(BeatBank::default()) {
        "beats.clinical@4"
    } else if same(BeatBank::clinical_v3()) {
        "beats.clinical@3"
    } else if same(BeatBank::patch()) {
        "beats.patch@3"
    } else {
        "beats.configured"
    }
}

fn same_weights(a: &VfWeights, b: &VfWeights) -> bool {
    a.bias == b.bias && a.w == b.w
}

fn vf_name(cfg: &VfConfig) -> &'static str {
    let stock = VfConfig::new(cfg.fs);
    // Everything but the model has to be the stock configuration; the model
    // is what tells the registered implementations apart.
    let same_but_model = |_: ecg_rhythm::vf::VfModel| {
        let mut c = *cfg;
        c.model = stock.model;
        format!("{:?}", c) == format!("{:?}", stock)
    };
    match cfg.model {
        ecg_rhythm::vf::VfModel::Linear(w)
            if same_weights(&w, &VfWeights::BASELINE) && same_but_model(cfg.model) =>
        {
            "vf.spectral@2"
        }
        ecg_rhythm::vf::VfModel::Linear(w)
            if same_weights(&w, &VfWeights::LINEAR7) && same_but_model(cfg.model) =>
        {
            "vf.linear@1"
        }
        _ => "vf.configured",
    }
}

fn svrun_name(cfg: &SvRunConfig) -> &'static str {
    let stock = SvRunConfig::default();
    if !cfg.enabled {
        "svrun.off@1"
    } else if format!(
        "{:?}",
        SvRunConfig {
            enabled: true,
            ..stock
        }
    ) == format!("{cfg:?}")
    {
        "svrun.rate@2"
    } else if format!(
        "{:?}",
        SvRunConfig {
            enabled: true,
            max_interval_ms: f32::INFINITY,
            ..stock
        }
    ) == format!("{cfg:?}")
    {
        "svrun.rate@1"
    } else {
        "svrun.configured"
    }
}

pub(crate) fn build_qrs(sel: Option<&'static str>, cfg: QrsConfig) -> Box<dyn QrsStage> {
    // One implementation today; the name is carried so a second can join.
    let _ = sel;
    Box::new(Qrs::new("qrs.pt@1", cfg))
}

pub(crate) fn build_beats(sel: Option<&'static str>, bank: BeatBank) -> Box<dyn BeatStage> {
    let bank = match sel {
        Some("beats.clinical@4") => BeatBank::default(),
        Some("beats.clinical@3") => BeatBank::clinical_v3(),
        Some("beats.patch@3") => BeatBank::patch(),
        _ => bank,
    };
    Box::new(Beats::new(beats_name(&bank), bank))
}

pub(crate) fn build_af(sel: Option<&'static str>, fs: f64, cfg: AfConfig) -> Box<dyn AfStage> {
    let _ = sel;
    Box::new(Af::new("af.logistic@1", fs, cfg))
}

pub(crate) fn build_vf(sel: Option<&'static str>, cfg: VfConfig) -> Box<dyn VfStage> {
    let mut cfg = cfg;
    match sel {
        Some("vf.spectral@2") => cfg.model = ecg_rhythm::vf::VfModel::Linear(VfWeights::BASELINE),
        Some("vf.linear@1") => cfg.model = ecg_rhythm::vf::VfModel::Linear(VfWeights::LINEAR7),
        _ => {}
    }
    Box::new(Vf::new(vf_name(&cfg), cfg))
}

pub(crate) fn build_sv_run(sel: Option<&'static str>, cfg: SvRunConfig) -> Box<dyn SvRunStage> {
    let mut cfg = cfg;
    match sel {
        Some("svrun.off@1") => cfg.enabled = false,
        Some("svrun.rate@2") => {
            cfg = SvRunConfig {
                enabled: true,
                ..SvRunConfig::default()
            }
        }
        Some("svrun.rate@1") => {
            cfg = SvRunConfig {
                enabled: true,
                max_interval_ms: f32::INFINITY,
                ..SvRunConfig::default()
            }
        }
        _ => {}
    }
    Box::new(SvRuns::new(svrun_name(&cfg), cfg))
}

/// Stages a Rust host supplies itself, replacing the pipeline's own. Any left
/// `None` are built as usual.
#[derive(Default)]
pub struct CustomStages {
    pub qrs: Option<Box<dyn QrsStage>>,
    pub beats: Option<Box<dyn BeatStage>>,
    pub af: Option<Box<dyn AfStage>>,
    pub vf: Option<Box<dyn VfStage>>,
    pub sv_run: Option<Box<dyn SvRunStage>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_is_unique_and_well_formed() {
        for (i, s) in STAGES.iter().enumerate() {
            assert!(
                s.name.starts_with(&format!("{}.", s.kind.key())),
                "{}",
                s.name
            );
            assert!(s.name.contains('@'), "{} has no version", s.name);
            assert!(
                STAGES[i + 1..].iter().all(|o| o.name != s.name),
                "{} twice",
                s.name
            );
        }
        for k in StageKind::ALL {
            assert!(STAGES.iter().any(|s| s.kind == k), "no {} stage", k.key());
        }
    }

    #[test]
    fn a_selection_is_parsed_strictly() {
        let s = StageSelection::parse("vf=vf.linear@1; svrun=svrun.off@1").unwrap();
        assert_eq!(s.vf, Some("vf.linear@1"));
        assert_eq!(s.sv_run, Some("svrun.off@1"));
        assert!(s.beats.is_none());
        assert!(StageSelection::parse("vf=vf.nonexistent@9").is_err());
        assert!(StageSelection::parse("beats=vf.linear@1").is_err());
        assert!(StageSelection::parse("vf").is_err());
        assert_eq!(
            StageSelection::parse("").unwrap(),
            StageSelection::default()
        );
    }

    #[test]
    fn presets_answer_to_registered_names() {
        let c = crate::PipelineConfig::new(250.0);
        assert_eq!(beats_name(&c.bank), "beats.clinical@4");
        assert_eq!(vf_name(&c.vf), "vf.spectral@2");
        assert_eq!(svrun_name(&c.sv_run), "svrun.off@1");
        let p = crate::PipelineConfig::patch(250.0);
        assert_eq!(beats_name(&p.bank), "beats.patch@3");
        assert_eq!(svrun_name(&p.sv_run), "svrun.rate@2");
        let mut tuned = c;
        tuned.bank.ventricular.threshold = 0.5;
        assert_eq!(beats_name(&tuned.bank), "beats.configured");
    }
}
