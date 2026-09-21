//! Ventricular fibrillation evaluation and model fitting.
//!
//! # On the absence of a sealed set
//!
//! The two corpora that carry fibrillation — the MIT-BIH Malignant Ventricular
//! Arrhythmia database and the Creighton University Ventricular Tachyarrhythmia
//! database — are **entirely in the TRAIN zone** of the subject-level split this
//! project inherited. There is no held-out fibrillation data to score against.
//!
//! So the split here is positional and carved from TRAIN: every third record
//! validates, the rest fit. Every number this produces is a held-out-within-
//! training estimate and should be read as one. It is not comparable to the
//! sealed figures elsewhere in these reports, and saying so is the point of this
//! paragraph.
//!
//! # Ground truth
//!
//! * `vfdb` annotates rhythm as spans; `(VF` and `(VFL` count as fibrillation.
//!   Ventricular *tachycardia* does not — it is organised, it has beats, and the
//!   beat path already detects it.
//! * `cudb` marks fibrillation with bracket annotations, `[` for onset and `]`
//!   for offset, rather than with rhythm spans.

use crate::manifest::RecordEntry;
use crate::rhythm_ref;
use crate::Opts;
use ecg_pipeline::{ChannelOutput, ChannelPipeline};
use ecg_rhythm::VfFeatures;
use ecg_wfdb::{read_signal, AnnotationFile, Header};
use rayon::prelude::*;
use std::path::Path;

pub struct Second {
    pub truth: bool,
    pub score: f32,
    pub features: VfFeatures,
    /// Reported after episode confirmation.
    pub reported: bool,
}

/// Per-second fibrillation labels for one record.
fn truth_seconds(ann: &AnnotationFile, n_samples: i64, n_sec: usize, fs: f64) -> Vec<bool> {
    let mut v = vec![false; n_sec];
    // Rhythm spans, where the corpus uses them.
    for (s, e, name) in rhythm_ref::named_spans(ann, n_samples) {
        if name == "VF" || name == "VFL" {
            let a = ((s as f64 / fs).floor() as usize).min(n_sec);
            let b = ((e as f64 / fs).ceil() as usize).min(n_sec);
            for x in v.iter_mut().take(b).skip(a) {
                *x = true;
            }
        }
    }
    // Bracket annotations, where it uses those instead.
    let mut open: Option<i64> = None;
    for a in ann.annotations.iter().filter(|a| a.sample >= 0) {
        match a.symbol {
            '[' => open = Some(a.sample),
            ']' => {
                if let Some(s) = open.take() {
                    let lo = ((s as f64 / fs).floor() as usize).min(n_sec);
                    let hi = ((a.sample as f64 / fs).ceil() as usize).min(n_sec);
                    for x in v.iter_mut().take(hi).skip(lo) {
                        *x = true;
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(s) = open {
        let lo = ((s as f64 / fs).floor() as usize).min(n_sec);
        for x in v.iter_mut().skip(lo) {
            *x = true;
        }
    }
    v
}

pub fn analyse(entry: &RecordEntry, opts: &Opts) -> std::io::Result<Vec<Second>> {
    let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    let hdr = Header::read(&entry.hea_path()).map_err(|e| err(e.to_string()))?;
    let ann =
        AnnotationFile::read(Path::new(&entry.ann_path("atr"))).map_err(|e| err(e.to_string()))?;
    let lead = opts.lead.min(hdr.n_sig.saturating_sub(1));
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples).map_err(|e| err(e.to_string()))?;

    let fs = hdr.fs;
    let n_sec = (hdr.n_samples as f64 / fs).floor() as usize;
    let truth = truth_seconds(&ann, hdr.n_samples as i64, n_sec, fs);

    let cfg = crate::qrs_eval::config_from(opts, fs);
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let block = ((fs * 0.25) as usize).max(1);
    let mut windows = Vec::new();
    let mut episodes: Vec<(u64, u64)> = Vec::new();
    for chunk in sig.chunks(block) {
        out.clear();
        pipe.push(chunk, &mut out);
        windows.extend_from_slice(&out.vf);
        episodes.extend_from_slice(&out.vf_episodes);
    }
    out.clear();
    pipe.finish(&mut out);
    episodes.extend_from_slice(&out.vf_episodes);

    let mut reported = vec![false; n_sec];
    for (s, e) in &episodes {
        let a = ((*s as f64 / fs).floor() as usize).min(n_sec);
        let b = ((*e as f64 / fs).ceil() as usize + 1).min(n_sec);
        for x in reported.iter_mut().take(b).skip(a) {
            *x = true;
        }
    }

    // One row per decision window, labelled by the second it ends in.
    let mut rows = Vec::with_capacity(windows.len());
    for w in &windows {
        let sec = ((w.sample as f64 / fs).floor() as usize).min(n_sec.saturating_sub(1));
        if n_sec == 0 {
            break;
        }
        rows.push(Second {
            truth: truth[sec],
            score: w.probability,
            features: w.features,
            reported: reported[sec],
        });
    }
    Ok(rows)
}

fn auc(rows: &[&Second], value: impl Fn(&Second) -> f32) -> f64 {
    let mut all: Vec<(f32, bool)> = rows.iter().map(|r| (value(r), r.truth)).collect();
    let n_pos = all.iter().filter(|&&(_, p)| p).count() as f64;
    let n_neg = all.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return f64::NAN;
    }
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && all[j + 1].0 == all[i].0 {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &(_, p) in &all[i..=j] {
            if p {
                rank_sum += avg;
            }
        }
        i = j + 1;
    }
    (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

type Probe = (&'static str, fn(&Second) -> f32);
const PROBES: [Probe; 7] = [
    ("score", |r| r.score),
    ("tcsc", |r| r.features.tcsc),
    ("-leakage", |r| -r.features.leakage),
    ("-peak_to_mean", |r| -r.features.peak_to_mean),
    ("-kurtosis", |r| -r.features.kurtosis),
    ("dominant_hz", |r| r.features.dominant_hz),
    ("amplitude_rel", |r| r.features.amplitude_rel),
];

fn collect(opts: &Opts) -> std::io::Result<Vec<Second>> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    Ok(entries
        .par_iter()
        .filter_map(|e| analyse(e, opts).ok())
        .flatten()
        .collect())
}

/// Share of non-fibrillation seconds correctly left alone. Shared with the
/// regression tests.
pub fn specificity(opts: &Opts) -> Option<f64> {
    let rows = collect(opts).ok()?;
    if rows.is_empty() {
        return None;
    }
    let (mut tn, mut fp) = (0u64, 0u64);
    for r in rows.iter().filter(|r| !r.truth) {
        if r.reported {
            fp += 1;
        } else {
            tn += 1;
        }
    }
    (tn + fp > 0).then(|| tn as f64 / (tn + fp) as f64)
}

/// Samples for which beat-derived analysis was withheld on one record.
pub fn suppressed_samples(entry: &RecordEntry, opts: &Opts) -> std::io::Result<u64> {
    let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    let hdr = Header::read(&entry.hea_path()).map_err(|e| err(e.to_string()))?;
    let lead = opts.lead.min(hdr.n_sig.saturating_sub(1));
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples).map_err(|e| err(e.to_string()))?;
    let cfg = crate::qrs_eval::config_from(opts, hdr.fs);
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let block = ((hdr.fs * 0.25) as usize).max(1);
    let mut total = 0u64;
    for chunk in sig.chunks(block) {
        out.clear();
        pipe.push(chunk, &mut out);
        total += out.suppressed_samples;
    }
    Ok(total)
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    let rows = collect(opts)?;
    if rows.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }
    let refs: Vec<&Second> = rows.iter().collect();
    let pos = rows.iter().filter(|r| r.truth).count();

    println!("\n── ventricular fibrillation ──────────────────────────────────");
    println!(
        "windows {}  ({:.1} h)   fibrillation {:.2} %",
        rows.len(),
        rows.len() as f64 / 3600.0,
        100.0 * pos as f64 / rows.len() as f64
    );
    println!("\n{:<16} {:>10}", "feature", "AUC");
    let mut aucs: Vec<(&str, f64)> = PROBES.iter().map(|(n, f)| (*n, auc(&refs, f))).collect();
    aucs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (n, a) in &aucs {
        println!("{n:<16} {a:>10.4}");
    }

    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0u64, 0u64, 0u64);
    for r in &rows {
        match (r.truth, r.reported) {
            (true, true) => tp += 1,
            (true, false) => fn_ += 1,
            (false, true) => fp += 1,
            (false, false) => tn += 1,
        }
    }
    let d = |a: u64, b: u64| {
        if b == 0 {
            f64::NAN
        } else {
            a as f64 / b as f64
        }
    };
    println!("\nconfirmed episodes, per second:");
    println!("  sensitivity   {:.2} %", 100.0 * d(tp, tp + fn_));
    println!("  specificity   {:.2} %", 100.0 * d(tn, tn + fp));
    println!("  PPV           {:.2} %", 100.0 * d(tp, tp + fp));
    Ok(())
}

/// Fit the logistic model on the records selected, and print the coefficients.
pub fn fit(opts: &Opts) -> std::io::Result<()> {
    let rows = collect(opts)?;
    let n = rows.len();
    let pos = rows.iter().filter(|r| r.truth).count();
    if pos == 0 || pos == n {
        eprintln!("need both classes present");
        return Ok(());
    }
    println!(
        "windows {n} ({pos} fibrillation, {:.1} %)",
        100.0 * pos as f64 / n as f64
    );

    let nf = ecg_rhythm::vf::NF;
    let x: Vec<[f32; ecg_rhythm::vf::NF]> = rows.iter().map(|r| r.features.vector()).collect();
    let y: Vec<f64> = rows
        .iter()
        .map(|r| if r.truth { 1.0 } else { 0.0 })
        .collect();
    let mut mu = vec![0.0f64; nf];
    let mut sd = vec![0.0f64; nf];
    for v in &x {
        for k in 0..nf {
            mu[k] += v[k] as f64;
        }
    }
    for m in mu.iter_mut() {
        *m /= n as f64;
    }
    for v in &x {
        for k in 0..nf {
            sd[k] += (v[k] as f64 - mu[k]).powi(2);
        }
    }
    for s in sd.iter_mut() {
        *s = (*s / n as f64).sqrt().max(1e-9);
    }
    let w_pos = 0.5 / pos as f64;
    let w_neg = 0.5 / (n - pos) as f64;

    let l2 = opts.get_f64("l2").unwrap_or(0.01);
    let iters = opts.get_usize("iters").unwrap_or(800);
    let mut w = vec![0.0f64; nf];
    let mut b = 0.0f64;
    let mut lr = 0.5;
    let mut prev = f64::INFINITY;
    for _ in 0..iters {
        let mut gw = vec![0.0f64; nf];
        let mut gb = 0.0f64;
        let mut loss = 0.0f64;
        let mut wsum = 0.0f64;
        for (v, &yi) in x.iter().zip(&y) {
            let mut z = b;
            for k in 0..nf {
                z += w[k] * (v[k] as f64 - mu[k]) / sd[k];
            }
            let p = 1.0 / (1.0 + (-z).exp());
            let cw = if yi > 0.5 { w_pos } else { w_neg };
            let e = cw * (p - yi);
            for k in 0..nf {
                gw[k] += e * (v[k] as f64 - mu[k]) / sd[k];
            }
            gb += e;
            loss -= cw * (yi * (p + 1e-12).ln() + (1.0 - yi) * (1.0 - p + 1e-12).ln());
            wsum += cw;
        }
        let nz = wsum.max(1e-12);
        for k in 0..nf {
            w[k] -= lr * (gw[k] / nz + l2 * w[k]);
        }
        b -= lr * gb / nz;
        let loss = loss / nz;
        if loss > prev {
            lr *= 0.5;
        }
        prev = loss;
    }

    println!("\nstandardised influence:");
    let mut order: Vec<usize> = (0..nf).collect();
    order.sort_by(|&a, &c| w[c].abs().partial_cmp(&w[a].abs()).unwrap());
    for k in order {
        println!("  {:<16} {:+.4}", VfFeatures::NAMES[k], w[k]);
    }
    let mut bias = b;
    let mut wf = vec![0.0f64; nf];
    for k in 0..nf {
        wf[k] = w[k] / sd[k];
        bias -= w[k] * mu[k] / sd[k];
    }
    if opts.raw.iter().any(|(k, _)| k == "gbdt") {
        let mut tcfg = crate::gbdt_train::TrainConfig::default();
        if let Some(v) = opts.get_usize("trees") {
            tcfg.trees = v;
        }
        if let Some(v) = opts.get_usize("depth") {
            tcfg.depth = v;
        }
        let xs: Vec<Vec<f32>> = rows.iter().map(|r| r.features.vector().to_vec()).collect();
        let ys: Vec<f32> = rows
            .iter()
            .map(|r| if r.truth { 1.0 } else { 0.0 })
            .collect();
        let ws = vec![1.0f64; rows.len()];
        eprintln!("training fibrillation ensemble ...");
        let m = crate::gbdt_train::train(&xs, &ys, &ws, &(0..nf).collect::<Vec<_>>(), &tcfg);
        eprintln!("  {} nodes, {} trees", m.nodes.len(), m.roots.len());
        let src = format!(
            "// Generated by `ecg-eval fit-vf --gbdt`. Do not edit by hand.\n{}",
            crate::gbdt_train::emit("VENTRICULAR_FIBRILLATION", &m)
        );
        let path = opts
            .get_str("emit")
            .unwrap_or("crates/ecg-rhythm/src/vf_trees_generated.rs");
        std::fs::write(path, src)?;
        eprintln!("wrote {path}");
    }

    println!("\n    pub const BASELINE: VfWeights = VfWeights {{");
    println!("        bias: {bias:.6},");
    print!("        w: [");
    for (k, v) in wf.iter().enumerate() {
        print!("{}{:.6}", if k > 0 { ", " } else { "" }, v);
    }
    println!("],");
    println!("    }};");
    Ok(())
}
