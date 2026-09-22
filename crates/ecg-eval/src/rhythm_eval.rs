//! Episode detection evaluation.
//!
//! Each condition is scored on its own, against its own reference label, in
//! duration-weighted seconds and in episodes. Pooling them would be meaningless:
//! they occur at wildly different prevalences, and an average over eleven
//! conditions is a number about the corpus rather than about the engine.
//!
//! # Two references, because there are two questions
//!
//! **Against the rule on reference beats.** Every condition here is *defined* —
//! a pause is an interval over two seconds, bigeminy is every second beat being
//! ventricular. So the reference is the same rule run over the corpus's own beat
//! annotations and beat labels. What that measures is whether our detection and
//! classification reproduce the rule, which is the question this engine can
//! actually be wrong about.
//!
//! **Against the annotator's rhythm spans.** Secondary, and only where such a
//! span exists. It answers a different question — whether the rule agrees with a
//! human — and the disagreements are mostly definitional rather than errors.
//!
//! Scoring rate conditions against rhythm spans alone would be meaningless, and
//! the first run of this harness showed why: tachycardia scored 6.6% precision
//! because these corpora label rhythm *origin*, not rate. Sinus tachycardia is
//! annotated `(N`, so every correctly detected fast sinus rhythm counted as a
//! false positive. The detector was right and the reference was the wrong one.

use crate::manifest::RecordEntry;
use crate::rhythm_ref;
use crate::Opts;
use ecg_pipeline::{ChannelOutput, ChannelPipeline};
/// Conditions the bank reports, so these arrays cannot fall out of step with
/// it the way they just did: adding three conditions left the scoring arrays at
/// eight and the evaluation panicked on the ninth.
const NC: usize = Condition::ALL.len();

use ecg_rhythm::{Beat, Condition, RhythmBank, RhythmConfig, RrConfig, RrStream};
use ecg_wfdb::{is_beat_symbol, read_signal, AnnotationFile, Header};
use rayon::prelude::*;
use std::path::Path;

/// Reference rhythm names that count as each condition.
///
/// * `SBR` is sinus bradycardia; there is no separate "slow" label.
/// * Tachycardia has no generic label, so supraventricular tachycardia and
///   ventricular tachycardia both count — the engine reports rate without
///   claiming an origin.
/// * A ventricular run and ventricular tachycardia share `(VT`. The engine
///   separates them by rate; the annotation does not.
fn reference_labels(c: Condition) -> &'static [&'static str] {
    match c {
        Condition::Bradycardia => &["SBR"],
        Condition::Tachycardia => &["SVTA", "VT"],
        Condition::VentricularRun | Condition::VentricularTachycardia => &["VT"],
        Condition::Bigeminy => &["B"],
        Condition::Trigeminy => &["T"],
        Condition::Asystole => &["ASYS"],
        // No span label exists; scored as point events against `PSE`.
        Condition::Idioventricular => &["IVR"],
        Condition::Pause => &[],
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Score {
    pub tp: u64,
    pub fp: u64,
    pub tn: u64,
    pub fn_: u64,
    /// Reference episodes, and how many were touched by a detection.
    pub ref_episodes: u64,
    pub ref_found: u64,
    /// Reported episodes, and how many overlapped a reference one.
    pub rep_episodes: u64,
    pub rep_correct: u64,
}

impl Score {
    fn merge(&mut self, o: &Score) {
        self.tp += o.tp;
        self.fp += o.fp;
        self.tn += o.tn;
        self.fn_ += o.fn_;
        self.ref_episodes += o.ref_episodes;
        self.ref_found += o.ref_found;
        self.rep_episodes += o.rep_episodes;
        self.rep_correct += o.rep_correct;
    }
    pub fn sensitivity(&self) -> f64 {
        div(self.tp, self.tp + self.fn_)
    }
    pub fn ppv(&self) -> f64 {
        div(self.tp, self.tp + self.fp)
    }
    pub fn specificity(&self) -> f64 {
        div(self.tn, self.tn + self.fp)
    }
}

fn div(a: u64, b: u64) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

pub struct RecordResult {
    pub record: String,
    pub source: String,
    pub hours: f64,
    /// Against the same rule applied to the corpus's beat annotations.
    pub scores: [Score; NC],
    /// Against the annotator's rhythm spans, where one exists.
    pub annotator: [Score; NC],
    pub error: Option<String>,
}

/// Run the rule over the corpus's own beats and beat labels.
///
/// Identical detector, different input: the only thing that changes is whether
/// the beats and their classes came from this engine or from the annotator.
fn reference_episodes(
    ann: &AnnotationFile,
    fs: f64,
    n_sec: usize,
    cfg: RhythmConfig,
) -> [Vec<bool>; NC] {
    let mut rr = RrStream::new(RrConfig::new(fs));
    let mut bank = RhythmBank::new(cfg);
    let mut episodes = Vec::new();
    for a in ann
        .annotations
        .iter()
        .filter(|a| a.sample >= 0 && is_beat_symbol(a.symbol))
    {
        let ev = ecg_qrs::QrsEvent {
            sample: a.sample as u64,
            amplitude: 0.0,
            energy: 0.0,
            margin: 1.0,
            recovered: false,
            interval_energy: 0.0,
        };
        rr.observe_quality(true);
        if let Some(s) = rr.push(&ev) {
            let beat = match crate::beat_eval::aami(a.symbol) {
                Some(crate::beat_eval::Aami::V) => Beat::Ventricular,
                Some(crate::beat_eval::Aami::S) => Beat::Supraventricular,
                Some(crate::beat_eval::Aami::N) => Beat::Normal,
                _ => Beat::Unknown,
            };
            bank.push(&s, beat, &mut episodes);
        }
    }
    bank.finish(&mut episodes);

    let mut out: [Vec<bool>; NC] = std::array::from_fn(|_| vec![false; n_sec]);
    for e in &episodes {
        let i = Condition::ALL
            .iter()
            .position(|c| *c == e.condition)
            .unwrap();
        let a = ((e.start as f64 / fs).floor() as usize).min(n_sec);
        let b = ((e.end as f64 / fs).ceil() as usize + 1).min(n_sec);
        for v in out[i].iter_mut().take(b).skip(a) {
            *v = true;
        }
    }
    out
}

fn runs(v: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, &x) in v.iter().enumerate() {
        if x {
            start.get_or_insert(i);
        } else if let Some(s) = start.take() {
            out.push((s, i));
        }
    }
    if let Some(s) = start {
        out.push((s, v.len()));
    }
    out
}

pub fn analyse(entry: &RecordEntry, opts: &Opts) -> RecordResult {
    let mut r = RecordResult {
        record: entry.record.clone(),
        source: entry.source.clone(),
        hours: 0.0,
        scores: [Score::default(); NC],
        annotator: [Score::default(); NC],
        error: None,
    };
    let err = |e: String| e;
    let hdr = match Header::read(&entry.hea_path()) {
        Ok(h) => h,
        Err(e) => {
            r.error = Some(err(e.to_string()));
            return r;
        }
    };
    let ann = match AnnotationFile::read(Path::new(&entry.ann_path("atr"))) {
        Ok(a) => a,
        Err(e) => {
            r.error = Some(err(e.to_string()));
            return r;
        }
    };
    let lead = opts.lead.min(hdr.n_sig.saturating_sub(1));
    let sig = match read_signal(&hdr, lead, 0, hdr.n_samples) {
        Ok(s) => s,
        Err(e) => {
            r.error = Some(err(e.to_string()));
            return r;
        }
    };

    let fs = hdr.fs;
    let n_sec = (hdr.n_samples as f64 / fs).floor() as usize;
    r.hours = n_sec as f64 / 3600.0;

    let spans = rhythm_ref::named_spans(&ann, hdr.n_samples as i64);
    let cfg = crate::qrs_eval::config_from(opts, fs);
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let block = ((fs * 0.25) as usize).max(1);
    let mut episodes = Vec::new();
    for chunk in sig.chunks(block) {
        out.clear();
        pipe.push(chunk, &mut out);
        episodes.extend_from_slice(&out.episodes);
    }
    out.clear();
    pipe.finish(&mut out);
    episodes.extend_from_slice(&out.episodes);

    let reference = reference_episodes(&ann, fs, n_sec, cfg.rhythm);
    let mut predicted: [Vec<bool>; NC] = std::array::from_fn(|_| vec![false; n_sec]);
    for e in &episodes {
        let i = Condition::ALL
            .iter()
            .position(|c| *c == e.condition)
            .unwrap();
        let a = ((e.start as f64 / fs).floor() as usize).min(n_sec);
        let b = ((e.end as f64 / fs).ceil() as usize + 1).min(n_sec);
        for v in predicted[i].iter_mut().take(b).skip(a) {
            *v = true;
        }
    }

    for i in 0..NC {
        r.scores[i] = compare(&reference[i], &predicted[i]);
    }

    // Secondary view: the annotator's own rhythm spans, where one exists.
    for (i, c) in Condition::ALL.iter().enumerate() {
        let labels = reference_labels(*c);
        if labels.is_empty() {
            continue;
        }
        // Seconds outside any annotated span are left out. These corpora
        // annotate rhythm continuously, but a stretch nobody labelled is not
        // evidence of absence.
        let mut truth = vec![None; n_sec];
        for (a, b, name) in &spans {
            let lo = ((*a as f64 / fs).floor() as usize).min(n_sec);
            let hi = ((*b as f64 / fs).ceil() as usize).min(n_sec);
            let is = labels.contains(&name.as_str());
            for t in truth.iter_mut().take(hi).skip(lo) {
                *t = Some(is);
            }
        }
        let sc = &mut r.annotator[i];
        for (t, p) in truth.iter().zip(&predicted[i]) {
            match (t, p) {
                (Some(true), true) => sc.tp += 1,
                (Some(true), false) => sc.fn_ += 1,
                (Some(false), true) => sc.fp += 1,
                (Some(false), false) => sc.tn += 1,
                (None, _) => {}
            }
        }
        let truth_bool: Vec<bool> = truth.iter().map(|t| *t == Some(true)).collect();
        episode_counts(sc, &truth_bool, &predicted[i]);
    }
    r
}

/// Duration-weighted confusion plus episode agreement between two timelines.
fn compare(truth: &[bool], pred: &[bool]) -> Score {
    let mut s = Score::default();
    for (t, p) in truth.iter().zip(pred) {
        match (t, p) {
            (true, true) => s.tp += 1,
            (true, false) => s.fn_ += 1,
            (false, true) => s.fp += 1,
            (false, false) => s.tn += 1,
        }
    }
    episode_counts(&mut s, truth, pred);
    s
}

fn episode_counts(s: &mut Score, truth: &[bool], pred: &[bool]) {
    for (a, b) in runs(truth) {
        s.ref_episodes += 1;
        if pred[a..b].iter().any(|&p| p) {
            s.ref_found += 1;
        }
    }
    for (a, b) in runs(pred) {
        s.rep_episodes += 1;
        if truth[a..b].iter().any(|&t| t) {
            s.rep_correct += 1;
        }
    }
}

/// The same, against the annotator's own rhythm spans.
pub fn annotator_scores(opts: &Opts) -> Option<[Score; NC]> {
    let entries = opts.select().ok()?;
    if entries.is_empty() {
        return None;
    }
    let results: Vec<RecordResult> = entries.par_iter().map(|e| analyse(e, opts)).collect();
    let mut total = [Score::default(); NC];
    for r in results.iter().filter(|r| r.error.is_none()) {
        for (t, s) in total.iter_mut().zip(&r.annotator) {
            t.merge(s);
        }
    }
    Some(total)
}

/// Duration-weighted sensitivity and precision per condition, against the rule
/// on the corpus's own beats. Shared with the regression tests.
pub fn summarise(opts: &Opts) -> Option<[Score; NC]> {
    let entries = opts.select().ok()?;
    if entries.is_empty() {
        return None;
    }
    let results: Vec<RecordResult> = entries.par_iter().map(|e| analyse(e, opts)).collect();
    let mut total = [Score::default(); NC];
    for r in results.iter().filter(|r| r.error.is_none()) {
        for (t, s) in total.iter_mut().zip(&r.scores) {
            t.merge(s);
        }
    }
    Some(total)
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    if entries.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }
    eprintln!("episode detection: {} records", entries.len());
    let results: Vec<RecordResult> = entries.par_iter().map(|e| analyse(e, opts)).collect();

    let mut total = [Score::default(); NC];
    let mut annotator = [Score::default(); NC];
    let mut hours = 0.0;
    for r in &results {
        if r.error.is_some() {
            continue;
        }
        hours += r.hours;
        for i in 0..NC {
            total[i].merge(&r.scores[i]);
            annotator[i].merge(&r.annotator[i]);
        }
    }

    println!("\n── episode detection ─────────────────────────────────────────");
    println!(
        "records {}   {:.1} h",
        results.iter().filter(|r| r.error.is_none()).count(),
        hours
    );
    let table = |title: &str, scores: &[Score; NC]| {
        println!("\n{title}");
        println!(
            "{:<26} {:>8} {:>8} {:>8} {:>8} {:>15} {:>15}",
            "condition", "ref %", "Se %", "Sp %", "PPV %", "episodes found", "reported ok"
        );
        for (i, c) in Condition::ALL.iter().enumerate() {
            let s = &scores[i];
            let scored = s.tp + s.fp + s.tn + s.fn_;
            if scored == 0 || (s.tp + s.fn_ == 0 && s.rep_episodes == 0) {
                continue;
            }
            println!(
                "{:<26} {:>8.3} {:>8.2} {:>8.2} {:>8.2} {:>15} {:>15}",
                c.name(),
                100.0 * div(s.tp + s.fn_, scored),
                100.0 * s.sensitivity(),
                100.0 * s.specificity(),
                100.0 * s.ppv(),
                format!("{} / {}", s.ref_found, s.ref_episodes),
                format!("{} / {}", s.rep_correct, s.rep_episodes),
            );
        }
    };
    table(
        "against the same rule on the corpus's beat annotations:",
        &total,
    );
    table("against the annotator's rhythm spans:", &annotator);

    let failed: Vec<&RecordResult> = results.iter().filter(|r| r.error.is_some()).collect();
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
    Ok(())
}
