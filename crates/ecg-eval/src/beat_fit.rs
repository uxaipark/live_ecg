//! Fit each binary beat detector independently on TRAIN.
//!
//! Each detector gets its own fit over the same shared features, its own class
//! balance and its own regularisation path. That is the whole point of the bank:
//! the ventricular question and the supraventricular question have different
//! positive rates, different informative features and opposite cost
//! asymmetries, and a joint objective would average those away.
//!
//! Records are weighted equally rather than beats, for the reason Phase 2
//! documented: without it a handful of long recordings decide the model.

use crate::beat_eval::{self, Aami};
use crate::gbdt_train;
use crate::Opts;
use ecg_beats::{BeatFeatures, NF};
use std::collections::HashMap;

type Row = (BeatFeatures, Aami, String);
/// A named accessor over one beat's features.
/// One feature, named and addressed by its position in the shared vector.
///
/// Addressed rather than accessed through a hand-written closure: the two
/// tables that used to list the accessors had to be kept in step with
/// `BeatFeatures` by hand, and a feature added to the struct but forgotten here
/// is invisible in exactly the report that would have shown it was useless.
/// Which features a named detector is trained on.
///
/// `--features-ventricular` and `--features-supraventricular` take a
/// comma-separated list of names, `all`, or `default` -
/// [`BeatFeatures::MODEL_FEATURES`], which is what the shipped models were
/// fitted on. The default is named in the code rather than being "everything in
/// the struct", so a feature added later joins the vector and the report
/// without silently joining the models.
fn feature_set(opts: &Opts, name: &str) -> Vec<usize> {
    let key = format!("features-{}", name.to_lowercase());
    let spec = opts.get_str(&key).unwrap_or("default");
    if spec == "all" {
        return (0..NF).collect();
    }
    let wanted: Vec<&str> = if spec == "default" {
        BeatFeatures::MODEL_FEATURES.to_vec()
    } else {
        spec.split(',').map(|s| s.trim()).collect()
    };
    for w in &wanted {
        assert!(
            BeatFeatures::NAMES.contains(w),
            "unknown feature {w:?} in --{key}"
        );
    }
    (0..NF)
        .filter(|&i| wanted.contains(&BeatFeatures::NAMES[i]))
        .collect()
}

fn probes() -> Vec<(&'static str, usize)> {
    BeatFeatures::NAMES.iter().copied().zip(0..NF).collect()
}

/// Median of the per-record AUCs, over records that contain both classes.
///
/// The pooled figure is an average over *beats*, so a single patient with
/// thousands of ectopic beats decides it. That is how a feature came to score
/// 0.155 on the training records of MIT-BIH and 0.500 on its sealed ones - not
/// because it stopped working, but because the pooled number had never been
/// about more than a couple of patients. The median over records asks the
/// question that transfers: does this feature help on a patient you have not
/// seen.
fn auc_per_record(
    rows: &[Row],
    positive: Aami,
    value: impl Fn(&BeatFeatures) -> f32,
) -> (f64, usize) {
    let mut by_record: std::collections::BTreeMap<&str, Vec<Row>> = Default::default();
    for r in rows {
        by_record.entry(r.2.as_str()).or_default().push(r.clone());
    }
    let mut v: Vec<f64> = by_record
        .values()
        .map(|rs| auc(rs, positive, &value))
        .filter(|a| a.is_finite())
        .collect();
    if v.is_empty() {
        return (f64::NAN, 0);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v.len())
}

fn auc(rows: &[Row], positive: Aami, value: impl Fn(&BeatFeatures) -> f32) -> f64 {
    let mut all: Vec<(f32, bool)> = rows
        .iter()
        .filter(|(_, t, _)| matches!(t, Aami::N | Aami::S | Aami::V))
        .map(|(f, t, _)| (value(f), *t == positive))
        .collect();
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

struct Fit {
    bias: f64,
    w: [f64; NF],
    /// Standardised coefficients, for reading off influence.
    std_w: [f64; NF],
}

fn fit_binary(rows: &[Row], weights: &[f64], positive: Aami, l2: f64, iters: usize) -> Fit {
    let n = rows.len();
    let mut mu = [0.0f64; NF];
    let mut sd = [0.0f64; NF];
    for (f, _, _) in rows {
        let v = f.vector();
        for k in 0..NF {
            mu[k] += v[k] as f64;
        }
    }
    for m in mu.iter_mut() {
        *m /= n as f64;
    }
    for (f, _, _) in rows {
        let v = f.vector();
        for k in 0..NF {
            sd[k] += (v[k] as f64 - mu[k]).powi(2);
        }
    }
    for s in sd.iter_mut() {
        *s = (*s / n as f64).sqrt().max(1e-9);
    }

    let y: Vec<f64> = rows
        .iter()
        .map(|(_, t, _)| if *t == positive { 1.0 } else { 0.0 })
        .collect();
    let sum_pos: f64 = y
        .iter()
        .zip(weights)
        .filter(|(yi, _)| **yi > 0.5)
        .map(|(_, w)| w)
        .sum();
    let sum_neg: f64 = y
        .iter()
        .zip(weights)
        .filter(|(yi, _)| **yi <= 0.5)
        .map(|(_, w)| w)
        .sum();
    let w_pos = 0.5 / sum_pos.max(1e-12);
    let w_neg = 0.5 / sum_neg.max(1e-12);

    let x: Vec<[f32; NF]> = rows.iter().map(|(f, _, _)| f.vector()).collect();
    let mut w = [0.0f64; NF];
    let mut b = 0.0f64;
    let mut lr = 0.5f64;
    let mut prev = f64::INFINITY;
    for _ in 0..iters {
        let mut gw = [0.0f64; NF];
        let mut gb = 0.0f64;
        let mut loss = 0.0f64;
        let mut wsum = 0.0f64;
        for ((v, &yi), &rw) in x.iter().zip(&y).zip(weights) {
            let mut z = b;
            for k in 0..NF {
                z += w[k] * (v[k] as f64 - mu[k]) / sd[k];
            }
            let p = 1.0 / (1.0 + (-z).exp());
            let cw = rw * if yi > 0.5 { w_pos } else { w_neg };
            let e = cw * (p - yi);
            for k in 0..NF {
                gw[k] += e * (v[k] as f64 - mu[k]) / sd[k];
            }
            gb += e;
            let eps = 1e-12;
            loss -= cw * (yi * (p + eps).ln() + (1.0 - yi) * (1.0 - p + eps).ln());
            wsum += cw;
        }
        let nz = wsum.max(1.0);
        for k in 0..NF {
            w[k] -= lr * (gw[k] / nz + l2 * w[k]);
        }
        b -= lr * gb / nz;
        let loss = loss / nz;
        if loss > prev {
            lr *= 0.5;
        }
        prev = loss;
    }

    let mut wf = [0.0f64; NF];
    let mut bias = b;
    for k in 0..NF {
        wf[k] = w[k] / sd[k];
        bias -= w[k] * mu[k] / sd[k];
    }
    Fit {
        bias,
        w: wf,
        std_w: w,
    }
}

fn emit(name: &str, f: &Fit) {
    println!("\n{name} — standardised influence:");
    let mut order: Vec<usize> = (0..NF).collect();
    order.sort_by(|&a, &b| f.std_w[b].abs().partial_cmp(&f.std_w[a].abs()).unwrap());
    for k in order {
        println!("  {:<16} {:+.4}", BeatFeatures::NAMES[k], f.std_w[k]);
    }
    println!(
        "    pub const {}: LinearBinary = LinearBinary {{",
        name.to_uppercase()
    );
    println!("        bias: {:.6},", f.bias);
    print!("        w: [");
    for (k, v) in f.w.iter().enumerate() {
        print!("{}{:.6}", if k > 0 { ", " } else { "" }, v);
    }
    println!("],");
    println!("    }};");
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    let rows = beat_eval::collect(opts)?;
    let rows: Vec<Row> = rows
        .into_iter()
        .filter(|(_, t, _)| matches!(t, Aami::N | Aami::S | Aami::V))
        .collect();
    if rows.is_empty() {
        eprintln!("no beats collected");
        return Ok(());
    }

    let mut counts: HashMap<Aami, usize> = HashMap::new();
    let mut per_record: HashMap<&str, usize> = HashMap::new();
    for (_, t, rec) in &rows {
        *counts.entry(*t).or_default() += 1;
        *per_record.entry(rec.as_str()).or_default() += 1;
    }
    println!(
        "beats: {} (N {}, S {}, V {}) over {} records",
        rows.len(),
        counts.get(&Aami::N).copied().unwrap_or(0),
        counts.get(&Aami::S).copied().unwrap_or(0),
        counts.get(&Aami::V).copied().unwrap_or(0),
        per_record.len()
    );
    let weights: Vec<f64> = rows
        .iter()
        .map(|(_, _, rec)| 1.0 / per_record[rec.as_str()] as f64)
        .collect();

    println!("\nper-feature AUC (TRAIN):");
    println!(
        "{:<16} {:>10} {:>10} {:>12} {:>12}",
        "feature", "V pooled", "S pooled", "V by record", "S by record"
    );
    let probes = probes();
    for &(name, i) in probes.iter() {
        let (vm, nv) = auc_per_record(&rows, Aami::V, |f| f.vector()[i]);
        let (sm, ns) = auc_per_record(&rows, Aami::S, |f| f.vector()[i]);
        println!(
            "{:<16} {:>10.4} {:>10.4} {:>8.4} ({:>2}) {:>8.4} ({:>2})",
            name,
            auc(&rows, Aami::V, |f| f.vector()[i]),
            auc(&rows, Aami::S, |f| f.vector()[i]),
            vm,
            nv,
            sm,
            ns
        );
    }

    let l2 = opts.get_f64("l2").unwrap_or(0.01);
    let iters = opts.get_usize("iters").unwrap_or(600);
    let v = fit_binary(&rows, &weights, Aami::V, l2, iters);
    let s = fit_binary(&rows, &weights, Aami::S, l2, iters);
    emit("ventricular", &v);
    emit("supraventricular", &s);

    if opts.raw.iter().any(|(k, _)| k == "gbdt") {
        let mut cfg = gbdt_train::TrainConfig::default();
        if let Some(n) = opts.get_usize("trees") {
            cfg.trees = n;
        }
        if let Some(n) = opts.get_usize("depth") {
            cfg.depth = n;
        }
        if let Some(v) = opts.get_f64("lr") {
            cfg.learning_rate = v;
        }
        let x: Vec<Vec<f32>> = rows.iter().map(|(f, _, _)| f.vector().to_vec()).collect();
        let mut src = String::from(
            "//! Generated by `ecg-eval fit-beats --gbdt --emit`. Do not edit by hand.\n\
             //!\n\
             //! One ensemble per binary detector, each fitted on its own question with\n\
             //! its own class balance. See `crate::detectors` for why they are separate.\n\
             use crate::gbdt::{GbdtModel, Node};\n\n",
        );
        for (name, positive) in [("VENTRICULAR", Aami::V), ("SUPRAVENTRICULAR", Aami::S)] {
            let allowed = feature_set(opts, name);
            eprintln!(
                "  {name} features: {}",
                allowed
                    .iter()
                    .map(|&i| BeatFeatures::NAMES[i])
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            let y: Vec<f32> = rows
                .iter()
                .map(|(_, t, _)| if *t == positive { 1.0 } else { 0.0 })
                .collect();
            eprintln!("training {name} ensemble ...");
            let m = gbdt_train::train(&x, &y, &weights, &allowed, &cfg);
            eprintln!("  {} nodes, {} trees", m.nodes.len(), m.roots.len());
            src.push_str(&gbdt_train::emit(name, &m));
            src.push('\n');
        }
        let path = opts
            .get_str("emit")
            .unwrap_or("crates/ecg-beats/src/trees_generated.rs");
        std::fs::write(path, src)?;
        eprintln!("wrote {path}");
    }
    Ok(())
}

/// Feature distributions by AAMI class.
///
/// A marginal AUC says a feature is informative; it does not say whether the
/// separation is where the physiology says it should be. Template correlation
/// ought to sit near 1 for conducted beats and far below it for ventricular
/// ones, and if it does not, the problem is in the measurement rather than in
/// the model that consumes it.
pub fn dump(opts: &Opts) -> std::io::Result<()> {
    let rows = beat_eval::collect(opts)?;
    let classes = [
        ("N", Aami::N),
        ("S", Aami::S),
        ("V", Aami::V),
        ("F", Aami::F),
    ];
    let probes = probes();
    println!(
        "{:<16} {:<4} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "feature", "cls", "n", "p5", "p25", "median", "p75", "p95"
    );
    for &(name, i) in probes.iter() {
        for (cname, c) in classes.iter() {
            let mut v: Vec<f32> = rows
                .iter()
                .filter(|(_, t, _)| t == c)
                .map(|(f, _, _)| f.vector()[i])
                .collect();
            if v.is_empty() {
                continue;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
            println!(
                "{:<16} {:<4} {:>8} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
                name,
                cname,
                v.len(),
                q(0.05),
                q(0.25),
                q(0.50),
                q(0.75),
                q(0.95)
            );
        }
    }
    Ok(())
}
