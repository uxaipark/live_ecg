//! Fit the AF logistic model on TRAIN, and report what each feature is worth.
//!
//! The model is eight coefficients. That is not a concession - it is the point.
//! A linear model over hand-built RR statistics runs in nanoseconds on any
//! target, cannot memorise a patient, and its failure modes are inspectable one
//! coefficient at a time. Anything larger has to earn its place against this.
//!
//! Selection happens on TRAIN. The coefficients printed here are pasted into
//! `AfWeights::BASELINE` and TEST is scored once afterwards.

use crate::af_eval::{self, AfRecord};
use crate::rhythm_ref::AfLabel;
use crate::Opts;
use ecg_rhythm::AfFeatures;
use rayon::prelude::*;

/// Every feature the extractor produces, including the ones not in the model, so
/// a feature can be judged before being adopted rather than after.
/// A named accessor over one window's features.
type FeatureProbe = (&'static str, fn(&AfFeatures) -> f32);

const PROBES: [FeatureProbe; 16] = [
    ("rmssd_norm", |f| f.rmssd_norm),
    ("mad_norm", |f| f.mad_norm),
    ("rmssd_over_mad", |f| f.rmssd_over_mad),
    ("pnn_norm", |f| f.pnn_norm),
    ("shannon_rr", |f| f.shannon_rr),
    ("shannon_drr", |f| f.shannon_drr),
    ("cosen", |f| f.cosen),
    ("tpr", |f| f.tpr),
    ("sd1_norm", |f| f.sd1_norm),
    ("sd1_sd2", |f| f.sd1_sd2),
    ("iqr_norm", |f| f.iqr_norm),
    ("hr", |f| f.hr),
    ("drr_acf1", |f| f.drr_acf1),
    ("frac_near_median", |f| f.frac_near_median),
    ("rr_acf1", |f| f.rr_acf1),
    ("atrial_coherence", |f| f.atrial_coherence),
];

/// Mann-Whitney U as an AUC. Larger `value` must mean "more likely AF".
fn auc(rows: &[(AfFeatures, f32, AfLabel)], value: impl Fn(&AfFeatures) -> f32) -> f64 {
    let mut all: Vec<(f32, bool)> = Vec::with_capacity(rows.len());
    for (f, _, l) in rows {
        match l {
            AfLabel::Af => all.push((value(f), true)),
            AfLabel::NonAf => all.push((value(f), false)),
            AfLabel::Excluded => {}
        }
    }
    let n_pos = all.iter().filter(|&&(_, p)| p).count() as f64;
    let n_neg = all.len() as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return f64::NAN;
    }
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank_sum = 0.0f64;
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && all[j + 1].0 == all[i].0 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for &(_, is_pos) in &all[i..=j] {
            if is_pos {
                rank_sum += avg_rank;
            }
        }
        i = j + 1;
    }
    (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    if entries.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }
    eprintln!("collecting windows from {} records ...", entries.len());
    let results: Vec<AfRecord> = entries
        .par_iter()
        .map(|e| af_eval::analyse(e, opts))
        .collect();

    // Subsample: consecutive windows overlap by all but one interval, so they are
    // far from independent. Thinning costs almost no information and keeps the
    // fit honest about how much data there really is.
    let stride = opts.get_usize("stride").unwrap_or(8).max(1);
    // `--without <feature>` zeroes one input before fitting, so a feature can be
    // added and its contribution measured against the *same* fit rather than
    // against whatever weights happened to be shipped. Without it, "the model
    // got worse" cannot be told apart from "refitting made it worse".
    let dropped = opts.get_str("without").map(|s| s.to_string());
    if let Some(name) = dropped.as_deref() {
        assert!(
            AfFeatures::NAMES.contains(&name),
            "unknown feature {name:?} in --without"
        );
        eprintln!("fitting without {name}");
    }
    let drop_idx = dropped
        .as_deref()
        .and_then(|n| AfFeatures::NAMES.iter().position(|&m| m == n));
    let mut rows: Vec<(AfFeatures, f32, AfLabel)> = Vec::new();
    // Per-record weights, so a 24-hour recording does not outvote a 10-hour one
    // and a corpus of 84 records does not decide the model on its own. Without
    // this the fit is dominated by LTAFDB, whose negative class is rhythm in AF
    // patients - the model learns that population and then misreads the healthy
    // subjects it will actually spend its life on.
    let mut weights: Vec<f64> = Vec::new();
    for r in &results {
        let kept: Vec<_> = r
            .windows
            .iter()
            .step_by(stride)
            .filter(|w| w.2 != AfLabel::Excluded)
            .collect();
        if kept.is_empty() {
            continue;
        }
        let per = 1.0 / kept.len() as f64;
        for w in kept {
            rows.push(*w);
            weights.push(per);
        }
    }
    let n_pos = rows.iter().filter(|r| r.2 == AfLabel::Af).count();
    let n_neg = rows.len() - n_pos;
    println!(
        "windows: {} ({} AF, {} non-AF, {:.1}% positive, stride {stride})",
        rows.len(),
        n_pos,
        n_neg,
        100.0 * n_pos as f64 / rows.len().max(1) as f64
    );
    if n_pos == 0 || n_neg == 0 {
        eprintln!("need both classes present");
        return Ok(());
    }

    println!("\nper-feature AUC (TRAIN):");
    let mut aucs: Vec<(&str, f64)> = PROBES.iter().map(|(n, f)| (*n, auc(&rows, f))).collect();
    aucs.sort_by(|a, b| {
        (b.1 - 0.5)
            .abs()
            .partial_cmp(&(a.1 - 0.5).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (name, a) in &aucs {
        println!("  {name:<16} {a:.4}");
    }

    // Standardise, fit, then fold the standardisation back into the coefficients
    // so inference needs no per-feature statistics at runtime.
    let d = ecg_rhythm::af::NF;
    let x: Vec<[f32; ecg_rhythm::af::NF]> = rows
        .iter()
        .map(|(f, _, _)| {
            let mut v = f.vector();
            if let Some(i) = drop_idx {
                v[i] = 0.0;
            }
            v
        })
        .collect();
    let y: Vec<f32> = rows
        .iter()
        .map(|(_, _, l)| if *l == AfLabel::Af { 1.0 } else { 0.0 })
        .collect();
    let mut mu = [0.0f64; ecg_rhythm::af::NF];
    let mut sd = [0.0f64; ecg_rhythm::af::NF];
    for v in &x {
        for k in 0..d {
            mu[k] += v[k] as f64;
        }
    }
    for m in mu.iter_mut() {
        *m /= x.len() as f64;
    }
    for v in &x {
        for k in 0..d {
            sd[k] += (v[k] as f64 - mu[k]).powi(2);
        }
    }
    for s in sd.iter_mut() {
        *s = (*s / x.len() as f64).sqrt().max(1e-9);
    }

    // Class weights: AF is a minority of the corpus but not of the clinical
    // question. Balancing stops the fit from buying accuracy with silence.
    // Class balance on top of the record weights: AF is a minority of the corpus
    // but not of the clinical question.
    let sum_pos: f64 = rows
        .iter()
        .zip(&weights)
        .filter(|(r, _)| r.2 == AfLabel::Af)
        .map(|(_, w)| w)
        .sum();
    let sum_neg: f64 = rows
        .iter()
        .zip(&weights)
        .filter(|(r, _)| r.2 != AfLabel::Af)
        .map(|(_, w)| w)
        .sum();
    let w_pos = 0.5 / sum_pos.max(1e-12);
    let w_neg = 0.5 / sum_neg.max(1e-12);
    let _ = (n_pos, n_neg);

    let l2 = opts.get_f64("l2").unwrap_or(1e-4);
    let iters = opts.get_usize("iters").unwrap_or(400);
    let mut w = [0.0f64; ecg_rhythm::af::NF];
    let mut b = 0.0f64;
    let mut lr = 0.5f64;
    let mut prev_loss = f64::INFINITY;
    for it in 0..iters {
        let mut gw = [0.0f64; ecg_rhythm::af::NF];
        let mut gb = 0.0f64;
        let mut loss = 0.0f64;
        let mut wsum = 0.0f64;
        for ((v, &yi), &rw) in x.iter().zip(&y).zip(&weights) {
            let mut z = b;
            for k in 0..d {
                z += w[k] * (v[k] as f64 - mu[k]) / sd[k];
            }
            let p = 1.0 / (1.0 + (-z).exp());
            let cw = rw * if yi > 0.5 { w_pos } else { w_neg };
            let e = cw * (p - yi as f64);
            for k in 0..d {
                gw[k] += e * (v[k] as f64 - mu[k]) / sd[k];
            }
            gb += e;
            let eps = 1e-12;
            loss -= cw * (yi as f64 * (p + eps).ln() + (1.0 - yi as f64) * (1.0 - p + eps).ln());
            wsum += cw;
        }
        let n = wsum.max(1.0);
        for k in 0..d {
            gw[k] = gw[k] / n + l2 * w[k];
            w[k] -= lr * gw[k];
        }
        b -= lr * gb / n;
        let loss = loss / n;
        // Back off when a step overshoots; the objective is convex, so a rising
        // loss can only mean the step was too long.
        if loss > prev_loss {
            lr *= 0.5;
        }
        prev_loss = loss;
        if it % 100 == 99 {
            eprintln!("  iter {:>4}  loss {:.6}  lr {:.4}", it + 1, loss, lr);
        }
    }

    let mut wf = [0.0f32; ecg_rhythm::af::NF];
    let mut bias = b;
    for k in 0..d {
        wf[k] = (w[k] / sd[k]) as f32;
        bias -= w[k] * mu[k] / sd[k];
    }

    println!("\ncoefficients (standardised magnitude = influence):");
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&a, &c| w[c].abs().partial_cmp(&w[a].abs()).unwrap());
    for k in order {
        println!("  {:<16} {:+.4}", AfFeatures::NAMES[k], w[k]);
    }

    println!("\npaste into AfWeights::BASELINE:");
    println!("    pub const BASELINE: AfWeights = AfWeights {{");
    println!("        bias: {:.6},", bias);
    print!("        w: [");
    for (k, v) in wf.iter().enumerate() {
        print!("{}{:.6}", if k > 0 { ", " } else { "" }, v);
    }
    println!("],");
    println!("    }};");
    Ok(())
}

/// Feature distributions split by outcome, for one or more records.
///
/// The false positives are what matter: an aggregate alarm rate says a record is
/// a problem, never which property of the rhythm is fooling the model.
pub fn dump(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    let results: Vec<AfRecord> = entries
        .par_iter()
        .map(|e| af_eval::analyse(e, opts))
        .collect();

    let thr = opts.get_f64("af-enter").unwrap_or(0.95) as f32;
    type Window = (AfFeatures, f32, AfLabel);
    let mut groups: Vec<(&str, Vec<&Window>)> = vec![
        ("true AF", Vec::new()),
        ("false alarm", Vec::new()),
        ("correct reject", Vec::new()),
    ];
    for r in &results {
        for w in &r.windows {
            let slot = match (w.2, w.1 >= thr) {
                (AfLabel::Af, _) => 0,
                (AfLabel::NonAf, true) => 1,
                (AfLabel::NonAf, false) => 2,
                _ => continue,
            };
            groups[slot].1.push(w);
        }
    }

    println!(
        "{:<18} {:<16} {:>8} {:>9} {:>9} {:>9} {:>9}",
        "feature", "group", "n", "p10", "p25", "median", "p90"
    );
    for (name, get) in PROBES.iter() {
        for (gname, rows) in groups.iter() {
            if rows.is_empty() {
                continue;
            }
            let mut v: Vec<f32> = rows.iter().map(|w| get(&w.0)).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
            println!(
                "{:<18} {:<16} {:>8} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
                name,
                gname,
                v.len(),
                q(0.10),
                q(0.25),
                q(0.50),
                q(0.90)
            );
        }
    }
    Ok(())
}
