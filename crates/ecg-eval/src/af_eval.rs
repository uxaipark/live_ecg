//! Atrial fibrillation evaluation.
//!
//! Three questions, deliberately separated, because pooling them hides which
//! part of the engine is responsible for a number:
//!
//! 1. **Does the rhythm logic work?** Run it on the reference beat annotations.
//!    Detector errors cannot reach the result.
//! 2. **Does the product work?** Run it on our own detections, end to end. This
//!    is the number that ships.
//! 3. **How often does it cry wolf?** Alarm rate on a corpus with no AF at all.
//!    Sensitivity is easy; an AF detector that fires on normal sinus rhythm is
//!    unusable regardless of it.

use crate::manifest::RecordEntry;
use crate::rhythm_ref::{self, AfLabel, FlutterPolicy};
use crate::Opts;
use ecg_pipeline::{ChannelOutput, ChannelPipeline, Preprocessor};
use ecg_qrs::QrsEvent;
use ecg_rhythm::{
    AfDetector, AfFeatures, AfWindow, EpisodeConfig, EpisodeTracker, RrConfig, RrSample, RrStream,
};
use ecg_wfdb::{read_signal, AnnotationFile, Header};
use rayon::prelude::*;
use std::path::Path;

/// Which beat stream feeds the rhythm logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeatSource {
    /// The corpus's own beat annotations: isolates the rhythm logic.
    Reference,
    /// Our detector: the end-to-end path.
    Detected,
}

pub struct AfRecord {
    pub source: String,
    pub record: String,
    pub seconds: usize,
    /// Per-second reference label and predicted state, same length.
    pub truth: Vec<AfLabel>,
    pub predicted: Vec<bool>,
    /// One entry per decision window, with the label at its closing beat.
    pub windows: Vec<(AfFeatures, f32, AfLabel)>,
    pub error: Option<String>,
}

fn ann_ext(source: &str) -> &'static str {
    if source == "afdb" {
        "qrs"
    } else {
        "atr"
    }
}

/// Beats and the rhythm timeline for one record.
/// The AF decision windows for one record, plus its reference timeline.
///
/// The two beat sources take deliberately different paths. `Detected` runs the
/// real `ChannelPipeline` and takes the AF windows it emits, so the quality gate
/// that drops intervals spanning noise is exercised exactly as deployed.
/// `Reference` builds the RR stream from the corpus annotations with every
/// interval trusted, which is the point: it isolates the rhythm logic.
fn load(
    entry: &RecordEntry,
    opts: &Opts,
    policy: FlutterPolicy,
) -> std::io::Result<(Header, Vec<AfLabel>, Vec<AfWindow>)> {
    let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    let hdr = Header::read(&entry.hea_path()).map_err(|e| err(e.to_string()))?;

    // Rhythm labels live in `.atr` for every corpus here, including afdb, whose
    // `.qrs` file carries beats only.
    let af_free = opts
        .get_str("assume-af-free")
        .map(|v| {
            v.split(',')
                .any(|slug| slug.trim() == entry.source || slug.trim() == "ALL")
        })
        .unwrap_or(false);
    let truth = if af_free {
        // Corpora that carry no rhythm annotations but are AF-free by
        // construction - the Normal Sinus Rhythm Database is 24-hour Holters
        // selected for having no significant arrhythmia. Every alarm on them is
        // a false one, which makes them the sharpest test of deployability that
        // exists here: sensitivity is easy, and an AF detector that fires on
        // normal sinus rhythm is unusable whatever its sensitivity.
        vec![AfLabel::NonAf; (hdr.n_samples as f64 / hdr.fs).floor() as usize]
    } else {
        let rhythm = AnnotationFile::read(Path::new(&entry.ann_path("atr")))
            .map_err(|e| err(e.to_string()))?;
        let spans = rhythm_ref::spans(&rhythm, hdr.n_samples as i64, policy);
        rhythm_ref::per_second(&spans, hdr.n_samples as i64, hdr.fs)
    };

    let cfg = crate::qrs_eval::config_from(opts, hdr.fs);
    let windows = match beat_source(opts) {
        BeatSource::Reference => {
            let a = AnnotationFile::read(Path::new(&entry.ann_path(ann_ext(&entry.source))))
                .map_err(|e| err(e.to_string()))?;
            let beats: Vec<i64> = a.beat_samples();
            // The signal is read here even though the beat positions come from
            // the annotations. One of the window's features is a measurement of
            // the atrial segment, not of the intervals, and this mode isolates
            // the rhythm logic from *detection* - it is not a claim that rhythm
            // can be judged without ever looking at the trace.
            let lead = opts.lead.min(hdr.n_sig - 1);
            let sig = read_signal(&hdr, lead, 0, hdr.n_samples).map_err(|e| err(e.to_string()))?;
            let mut pre = Preprocessor::new(cfg.preprocess);
            let mut an = ecg_beats::BeatAnalyzer::new(cfg.beats);
            let mut delin = ecg_beats::delineate::Delineator::new(cfg.delineate);
            let d = cfg.delineate;
            delin.set_delays(
                pre.group_delay_samples(d.qrs_ref_hz) + pre.qrs_group_delay_samples(d.qrs_ref_hz),
                pre.group_delay_samples(d.p_ref_hz) + pre.pt_group_delay_samples(d.p_ref_hz),
                pre.group_delay_samples(d.t_ref_hz) + pre.pt_group_delay_samples(d.t_ref_hz),
            );
            let mut recent: [Option<u64>; 3] = [None; 3];
            let mut rr = RrStream::new(RrConfig::new(hdr.fs));
            let mut af = AfDetector::new(hdr.fs, cfg.af);
            let mut windows = Vec::new();
            let mut pending: Option<RrSample> = None;
            let mut next = 0usize;
            for (i, &x) in sig.iter().enumerate() {
                let b = pre.process(x);
                an.push_sample(b.clean, b.qrs, true);
                delin.push_sample(b.qrs, b.pt);
                while next < beats.len() && beats[next] <= i as i64 {
                    next += 1;
                    let ev = QrsEvent {
                        sample: i as u64,
                        amplitude: 0.0,
                        energy: 0.0,
                        margin: 1.0,
                        recovered: false,
                        interval_energy: 0.0,
                    };
                    rr.observe_quality(true);
                    let interval = rr.push(&ev);
                    recent = [recent[1], recent[2], Some(ev.sample)];
                    let wave = match (recent[0], recent[1], recent[2]) {
                        (Some(p), Some(m), Some(c)) => delin.delineate(m, Some(m - p), Some(c - m)),
                        _ => None,
                    };
                    // Same one-beat lag as the pipeline: the interval is held
                    // until the beat that closes it has been analysed.
                    if let Some(obs) = an.push_beat(&ev, wave.as_ref()) {
                        if let Some(mut held) = pending.take() {
                            held.atrial_coherence = obs.features.p_ncc_prev;
                            if let Some(w) = af.push(&held) {
                                windows.push(w);
                            }
                        }
                    }
                    if interval.is_some() {
                        pending = interval;
                    }
                }
            }
            windows
        }
        BeatSource::Detected => {
            let lead = opts.lead.min(hdr.n_sig - 1);
            let sig = read_signal(&hdr, lead, 0, hdr.n_samples).map_err(|e| err(e.to_string()))?;
            let mut pipe = ChannelPipeline::new(cfg);
            let mut out = ChannelOutput::default();
            let block = ((hdr.fs * 0.25) as usize).max(1);
            let mut windows = Vec::new();
            for chunk in sig.chunks(block) {
                out.clear();
                pipe.push(chunk, &mut out);
                windows.extend_from_slice(&out.af);
            }
            windows
        }
    };
    Ok((hdr, truth, windows))
}

pub fn beat_source(opts: &Opts) -> BeatSource {
    match opts.get_str("beats") {
        Some("reference") | Some("ref") => BeatSource::Reference,
        _ => BeatSource::Detected,
    }
}

fn flutter_policy(opts: &Opts) -> FlutterPolicy {
    match opts.get_str("flutter") {
        Some("af") => FlutterPolicy::AsAf,
        _ => FlutterPolicy::Exclude,
    }
}

pub fn analyse(entry: &RecordEntry, opts: &Opts) -> AfRecord {
    let mut r = AfRecord {
        source: entry.source.clone(),
        record: entry.record.clone(),
        seconds: 0,
        truth: Vec::new(),
        predicted: Vec::new(),
        windows: Vec::new(),
        error: None,
    };
    let policy = flutter_policy(opts);
    let (hdr, truth, windows) = match load(entry, opts, policy) {
        Ok(v) => v,
        Err(e) => {
            r.error = Some(e.to_string());
            return r;
        }
    };

    let fs = hdr.fs;
    // The reported state is *confirmed episodes*, not raw window verdicts. That
    // is what the product emits, so it is what gets scored: a per-window metric
    // would flatter the detector by counting bursts it would never report.
    let n_sec = truth.len();
    let mut ep_cfg = EpisodeConfig::default();
    if let Some(v) = opts.get_f64("af-bridge") {
        ep_cfg.bridge_s = v as f32;
    }
    if let Some(v) = opts.get_f64("af-min-episode") {
        ep_cfg.min_episode_s = v as f32;
    }
    let mut tracker = EpisodeTracker::new(fs, ep_cfg);
    let mut episodes = Vec::new();
    for w in windows.iter() {
        if let Some(e) = tracker.update(w.sample, w.in_af) {
            episodes.push(e);
        }
    }
    if let Some(e) = tracker.finish() {
        episodes.push(e);
    }
    let mut predicted = vec![false; n_sec];
    for e in &episodes {
        let a = ((e.start as f64 / fs).floor() as usize).min(n_sec);
        let b = ((e.end as f64 / fs).ceil() as usize + 1).min(n_sec);
        for v in predicted.iter_mut().take(b).skip(a) {
            *v = true;
        }
    }

    r.windows = windows
        .iter()
        .map(|w| {
            let sec = ((w.sample as f64 / fs).floor() as usize).min(n_sec.saturating_sub(1));
            let label = truth.get(sec).copied().unwrap_or(AfLabel::Excluded);
            (w.features, w.probability, label)
        })
        .collect();
    r.seconds = n_sec;
    r.truth = truth;
    r.predicted = predicted;
    r
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Confusion {
    pub tp: u64,
    pub fp: u64,
    pub tn: u64,
    pub fn_: u64,
}

impl Confusion {
    pub fn add(&mut self, truth: AfLabel, pred: bool) {
        match (truth, pred) {
            (AfLabel::Af, true) => self.tp += 1,
            (AfLabel::Af, false) => self.fn_ += 1,
            (AfLabel::NonAf, true) => self.fp += 1,
            (AfLabel::NonAf, false) => self.tn += 1,
            (AfLabel::Excluded, _) => {}
        }
    }
    pub fn merge(&mut self, o: &Confusion) {
        self.tp += o.tp;
        self.fp += o.fp;
        self.tn += o.tn;
        self.fn_ += o.fn_;
    }
    pub fn sensitivity(&self) -> f64 {
        ratio(self.tp, self.tp + self.fn_)
    }
    pub fn specificity(&self) -> f64 {
        ratio(self.tn, self.tn + self.fp)
    }
    pub fn ppv(&self) -> f64 {
        ratio(self.tp, self.tp + self.fp)
    }
    pub fn f1(&self) -> f64 {
        let (se, pp) = (self.sensitivity(), self.ppv());
        if se + pp == 0.0 {
            0.0
        } else {
            2.0 * se * pp / (se + pp)
        }
    }
}

fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

/// Reference AF episodes of at least `min_s` seconds, and how many of them were
/// touched by a reported episode. Shared with the regression tests so the guard
/// measures the same thing the report does.
pub fn episode_recall(r: &AfRecord, min_s: usize) -> (usize, usize) {
    let truth_eps = rhythm_ref::episodes(&r.truth, AfLabel::Af, min_s);
    let found = truth_eps
        .iter()
        .filter(|&&(a, b)| r.predicted[a..b].iter().any(|&p| p))
        .count();
    (found, truth_eps.len())
}

/// Reported episodes per 24 hours. On an AF-free record every one is false.
pub fn alarms_per_day(r: &AfRecord, min_s: usize) -> f64 {
    let labels: Vec<AfLabel> = r
        .predicted
        .iter()
        .map(|&p| if p { AfLabel::Af } else { AfLabel::NonAf })
        .collect();
    let episodes = rhythm_ref::episodes(&labels, AfLabel::Af, min_s).len();
    let days = r.seconds as f64 / 86400.0;
    if days > 0.0 {
        episodes as f64 / days
    } else {
        f64::NAN
    }
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    if entries.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }
    let src = beat_source(opts);
    let policy = flutter_policy(opts);
    eprintln!(
        "AF evaluation: {} records  beats={:?}  flutter={:?}  zones={:?}",
        entries.len(),
        src,
        policy,
        opts.zones
    );

    let results: Vec<AfRecord> = entries.par_iter().map(|e| analyse(e, opts)).collect();
    report(&results, opts);
    Ok(())
}

pub fn report(results: &[AfRecord], opts: &Opts) {
    let min_ep = opts.get_usize("min-episode-s").unwrap_or(30);
    let mut total = Confusion::default();
    let mut ep_ref = 0usize;
    let mut ep_hit = 0usize;
    let mut ep_pred = 0usize;
    let mut ep_true_pred = 0usize;
    let mut af_free_seconds = 0u64;
    let mut af_free_alarms = 0u64;
    let mut per_record: Vec<(String, Confusion, f64, usize, f64)> = Vec::new();
    // AF burden - the share of monitored time spent in fibrillation - is what
    // gets reported clinically, and unlike an episode count it does not move
    // when one long episode is split by a momentary dip.
    let mut burden_err: Vec<f64> = Vec::new();

    for r in results {
        if r.error.is_some() {
            continue;
        }
        let mut c = Confusion::default();
        for (t, p) in r.truth.iter().zip(&r.predicted) {
            c.add(*t, *p);
        }
        total.merge(&c);

        // Episode agreement: a reference episode counts as found when any second
        // of it was called AF, and a predicted episode counts as correct when it
        // overlaps a reference one. Overlap, not exact boundaries - a detector
        // that needs a window of beats cannot mark a transition to the second.
        let pred_labels: Vec<AfLabel> = r
            .predicted
            .iter()
            .map(|&p| if p { AfLabel::Af } else { AfLabel::NonAf })
            .collect();
        let pred_eps = rhythm_ref::episodes(&pred_labels, AfLabel::Af, min_ep);
        let (hit, total) = episode_recall(r, min_ep);
        ep_ref += total;
        ep_pred += pred_eps.len();
        ep_hit += hit;
        for &(a, b) in &pred_eps {
            if r.truth[a..b].contains(&AfLabel::Af) {
                ep_true_pred += 1;
            }
        }

        // Records with no reference AF at all: the alarm rate there is the
        // number that decides whether this can be deployed.
        if !r.truth.contains(&AfLabel::Af) {
            af_free_seconds += r.truth.iter().filter(|&&t| t == AfLabel::NonAf).count() as u64;
            af_free_alarms += pred_eps.len() as u64;
        }

        let scored = (c.tp + c.fp + c.tn + c.fn_) as f64;
        if scored > 0.0 {
            let truth_burden = (c.tp + c.fn_) as f64 / scored;
            let pred_burden = (c.tp + c.fp) as f64 / scored;
            burden_err.push(pred_burden - truth_burden);
        }
        let hours = scored / 3600.0;
        per_record.push((
            format!("{}/{}", r.source, r.record),
            c,
            c.f1(),
            pred_eps.len(),
            hours,
        ));
    }

    if opts.per_record {
        println!(
            "{:<16} {:>6} {:>7} {:>9} {:>9} {:>9} {:>9} {:>10}",
            "record", "hours", "AF %", "Se %", "Sp %", "PPV %", "F1 %", "alarms/24h"
        );
        // Sorted by alarm rate: on AF-free records sensitivity is undefined, and
        // the alarm rate is the only thing worth ranking them by.
        per_record.sort_by(|a, b| {
            let ra = a.3 as f64 / (a.4 / 24.0).max(1e-9);
            let rb = b.3 as f64 / (b.4 / 24.0).max(1e-9);
            rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (name, c, _, eps, hours) in &per_record {
            let af_frac = 100.0 * ratio(c.tp + c.fn_, c.tp + c.fn_ + c.tn + c.fp);
            println!(
                "{:<16} {:>6.1} {:>7.1} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>10.2}",
                name,
                hours,
                af_frac,
                100.0 * c.sensitivity(),
                100.0 * c.specificity(),
                100.0 * c.ppv(),
                100.0 * c.f1(),
                *eps as f64 / (hours / 24.0).max(1e-9)
            );
        }
        println!();
    }

    let n_ok = results.iter().filter(|r| r.error.is_none()).count();
    let scored = total.tp + total.fp + total.tn + total.fn_;
    println!("── AF detection ──────────────────────────────────────────────");
    println!("records              {n_ok}");
    println!(
        "scored               {:.1} h  ({:.1}% atrial fibrillation)",
        scored as f64 / 3600.0,
        100.0 * ratio(total.tp + total.fn_, scored)
    );
    println!("duration-weighted, per second:");
    println!("  sensitivity        {:.3} %", 100.0 * total.sensitivity());
    println!("  specificity        {:.3} %", 100.0 * total.specificity());
    println!("  PPV                {:.3} %", 100.0 * total.ppv());
    println!("  F1                 {:.3} %", 100.0 * total.f1());
    println!("episodes (>= {min_ep} s):");
    println!(
        "  reference found    {ep_hit} / {ep_ref}  ({:.1} %)",
        100.0 * ratio(ep_hit as u64, ep_ref as u64)
    );
    println!(
        "  reported correct   {ep_true_pred} / {ep_pred}  ({:.1} %)",
        100.0 * ratio(ep_true_pred as u64, ep_pred as u64)
    );
    if af_free_seconds > 0 {
        println!(
            "false alarms         {af_free_alarms} over {:.1} h of AF-free signal  ({:.2} per 24 h)",
            af_free_seconds as f64 / 3600.0,
            af_free_alarms as f64 / (af_free_seconds as f64 / 86400.0)
        );
    }

    if !burden_err.is_empty() {
        let mean_abs = burden_err.iter().map(|e| e.abs()).sum::<f64>() / burden_err.len() as f64;
        let bias = burden_err.iter().sum::<f64>() / burden_err.len() as f64;
        let worst = burden_err
            .iter()
            .cloned()
            .fold(0.0f64, |a, b| if b.abs() > a.abs() { b } else { a });
        println!(
            "AF burden error      mean |error| {:.2} pp   bias {:+.2} pp   worst {:+.2} pp",
            100.0 * mean_abs,
            100.0 * bias,
            100.0 * worst
        );
    }

    let mut f1s: Vec<f64> = per_record
        .iter()
        .map(|(_, _, f, _, _)| *f)
        .filter(|f| f.is_finite())
        .collect();
    f1s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !f1s.is_empty() {
        let q = |p: f64| f1s[((p * (f1s.len() - 1) as f64).round() as usize).min(f1s.len() - 1)];
        println!(
            "per-record F1        min {:.2}  p10 {:.2}  median {:.2}  mean {:.2}",
            100.0 * f1s[0],
            100.0 * q(0.10),
            100.0 * q(0.50),
            100.0 * f1s.iter().sum::<f64>() / f1s.len() as f64
        );
    }
    let failed: Vec<&AfRecord> = results.iter().filter(|r| r.error.is_some()).collect();
    if !failed.is_empty() {
        println!("\nfailed ({}):", failed.len());
        for f in failed.iter().take(10) {
            println!(
                "  {}/{}: {}",
                f.source,
                f.record,
                f.error.as_deref().unwrap_or("")
            );
        }
    }
}
