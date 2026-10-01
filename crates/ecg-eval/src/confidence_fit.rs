//! Fitting and checking the calibrated confidences (`ecg_pipeline::confidence`).
//!
//! * `fit-qrs-conf` - every detection the engine makes on the selected
//!   records, labelled by whether a reference beat sits within the matching
//!   tolerance, and a logistic model fitted at the natural prevalence of
//!   spurious detections.
//! * `fit-beat-conf` - every classified beat matched to a reference beat, and
//!   a softmax over N, S, V and F fitted on the detectors' scores.
//! * `conf-eval` - the shipped calibration's reliability on the records
//!   selected: predicted confidence against how often it was right.
//!
//! Each fit first holds out every fourth record of each source and reports
//! the calibration it reaches there, then fits on everything and prints the
//! coefficients as Rust. A class with far more rows than the others is
//! stride-sampled and its rows weighted back, so the prevalence the model
//! learns is the corpus's own.

use crate::beat_eval::{aami, ann_ext, Aami};
use crate::manifest::RecordEntry;
use crate::Opts;
use ecg_beats::BeatClass;
use ecg_pipeline::confidence::{beat_vector, NB, NQ};
use ecg_pipeline::{ChannelOutput, ChannelPipeline};
use ecg_wfdb::{is_beat_symbol, read_signal, AnnotationFile, Header};
use rayon::prelude::*;
use std::path::Path;

struct Run {
    record: String,
    source: String,
    /// Detections: features, confidence, and whether a reference beat matched.
    det: Vec<([f32; NQ], f32, bool)>,
    /// Classified beats matched to a reference: scores, context, reported
    /// class, confidence, and the reference class.
    beats: Vec<([f32; NB], BeatClass, f32, Aami)>,
}

fn run_record(e: &RecordEntry, opts: &Opts) -> Option<Run> {
    let hdr = Header::read(&e.hea_path()).ok()?;
    let lead = opts.lead.min(hdr.n_sig.saturating_sub(1));
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples).ok()?;
    let ann = AnnotationFile::read(Path::new(&e.ann_path(ann_ext(&e.source)))).ok()?;
    let reference: Vec<(i64, char)> = ann
        .annotations
        .iter()
        .filter(|a| a.sample >= 0 && is_beat_symbol(a.symbol))
        .map(|a| (a.sample, a.symbol))
        .collect();
    let fs = hdr.fs;
    let cfg = crate::qrs_eval::config_from(opts, fs);
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let block = ((fs * 0.25) as usize).max(1);
    let (mut dets, mut verdicts) = (Vec::new(), Vec::new());
    for chunk in sig.chunks(block) {
        out.clear();
        pipe.push(chunk, &mut out);
        dets.extend_from_slice(&out.detections);
        verdicts.extend_from_slice(&out.classes);
    }
    let tol = (opts.tol_ms * fs / 1000.0).round() as i64;
    // One-to-one matching in time order.
    let matched = |samples: &[i64]| -> Vec<Option<usize>> {
        let mut out = vec![None; samples.len()];
        let (mut i, mut j) = (0usize, 0usize);
        while i < reference.len() && j < samples.len() {
            let d = samples[j] - reference[i].0;
            if d < -tol {
                j += 1;
            } else if d > tol {
                i += 1;
            } else {
                out[j] = Some(i);
                i += 1;
                j += 1;
            }
        }
        out
    };
    let ds: Vec<i64> = dets.iter().map(|d| d.sample as i64).collect();
    let dm = matched(&ds);
    let det = dets
        .iter()
        .zip(&dm)
        .map(|(d, m)| (d.features.vector(), d.confidence, m.is_some()))
        .collect();
    let vs: Vec<i64> = verdicts.iter().map(|v| v.sample as i64).collect();
    let vm = matched(&vs);
    let beats = verdicts
        .iter()
        .zip(&vm)
        .filter_map(|(v, m)| {
            let truth = aami(reference[(*m)?].1)?;
            if !matches!(truth, Aami::N | Aami::S | Aami::V | Aami::F) {
                return None;
            }
            Some((
                beat_vector(
                    v.p_ventricular,
                    v.p_supraventricular,
                    v.p_fusion,
                    v.context.fibrillating,
                ),
                v.class,
                v.confidence,
                truth,
            ))
        })
        .collect();
    Some(Run {
        record: e.record.clone(),
        source: e.source.clone(),
        det,
        beats,
    })
}

/// Beats only, through `beat_eval::analyse`, so candidate models given with
/// `--v-model`, `--s-model` and `--f-model` score them.
fn run_record_via_beat_eval(e: &RecordEntry, opts: &Opts) -> Option<Run> {
    let r = crate::beat_eval::analyse(e, opts);
    if r.error.is_some() {
        return None;
    }
    let beats = r
        .scored
        .iter()
        .filter(|s| matches!(s.truth, Aami::N | Aami::S | Aami::V | Aami::F))
        .map(|s| {
            let v = &s.verdict;
            (
                beat_vector(
                    v.p_ventricular,
                    v.p_supraventricular,
                    v.p_fusion,
                    v.context.fibrillating,
                ),
                v.class,
                v.confidence,
                s.truth,
            )
        })
        .collect();
    Some(Run {
        record: e.record.clone(),
        source: e.source.clone(),
        det: Vec::new(),
        beats,
    })
}

fn runs(opts: &Opts) -> std::io::Result<Vec<Run>> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    let via = opts.has("via-beat-eval");
    let mut v: Vec<Run> = entries
        .par_iter()
        .filter_map(|e| {
            if via {
                run_record_via_beat_eval(e, opts)
            } else {
                run_record(e, opts)
            }
        })
        .collect();
    v.sort_by(|a, b| (&a.source, &a.record).cmp(&(&b.source, &b.record)));
    Ok(v)
}

/// Every fourth record of each source, for the calibration check.
fn held_out(runs: &[Run]) -> Vec<bool> {
    let mut seen: std::collections::HashMap<&str, usize> = Default::default();
    runs.iter()
        .map(|r| {
            let c = seen.entry(r.source.as_str()).or_default();
            *c += 1;
            (*c - 1) % 4 == 0
        })
        .collect()
}

// ---- small dense solvers -------------------------------------------------

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        a.swap(c, p);
        b.swap(c, p);
        let d = a[c][c];
        if d.abs() < 1e-12 {
            continue;
        }
        for r in c + 1..n {
            let f = a[r][c] / d;
            if f == 0.0 {
                continue;
            }
            for k in c..n {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for c in (0..n).rev() {
        let mut s = b[c];
        for k in c + 1..n {
            s -= a[c][k] * x[k];
        }
        x[c] = if a[c][c].abs() < 1e-12 {
            0.0
        } else {
            s / a[c][c]
        };
    }
    x
}

/// Weighted logistic regression by Newton's method. Returns (bias, weights).
fn fit_logistic(x: &[Vec<f64>], y: &[bool], w: &[f64]) -> (f64, Vec<f64>) {
    let d = x[0].len() + 1;
    let mut beta = vec![0.0; d];
    for _ in 0..30 {
        let mut g = vec![0.0; d];
        let mut h = vec![vec![0.0; d]; d];
        for ((xi, &yi), &wi) in x.iter().zip(y).zip(w) {
            let mut z = beta[0];
            for k in 1..d {
                z += beta[k] * xi[k - 1];
            }
            let p = 1.0 / (1.0 + (-z).exp());
            let r = wi * (p - yi as u8 as f64);
            let s = wi * p * (1.0 - p);
            let v = |k: usize| if k == 0 { 1.0 } else { xi[k - 1] };
            for a in 0..d {
                g[a] += r * v(a);
                for b in a..d {
                    h[a][b] += s * v(a) * v(b);
                }
            }
        }
        for a in 0..d {
            for b in 0..a {
                h[a][b] = h[b][a];
            }
            h[a][a] += 1e-6;
        }
        let step = solve(h, g);
        let mut moved = 0.0f64;
        for k in 0..d {
            beta[k] -= step[k];
            moved = moved.max(step[k].abs());
        }
        if moved < 1e-7 {
            break;
        }
    }
    (beta[0], beta[1..].to_vec())
}

/// Weighted multinomial logistic regression over four classes by Newton's
/// method. Returns per class (bias, weights). Class 0 is the reference and
/// stays at zero: a softmax is unchanged by adding the same thing to every
/// class, and left free the coefficients drift together to sizes that f32
/// cannot difference.
fn fit_softmax(x: &[Vec<f64>], y: &[usize], w: &[f64]) -> Vec<(f64, Vec<f64>)> {
    const K: usize = 4;
    let d = x[0].len() + 1;
    let n = K * d;
    let mut beta = vec![0.0; n];
    for _ in 0..40 {
        let mut g = vec![0.0; n];
        let mut h = vec![vec![0.0; n]; n];
        for ((xi, &yi), &wi) in x.iter().zip(y).zip(w) {
            let v: Vec<f64> = std::iter::once(1.0).chain(xi.iter().cloned()).collect();
            let mut z = [0.0f64; K];
            for c in 0..K {
                z[c] = (0..d).map(|k| beta[c * d + k] * v[k]).sum();
            }
            let m = z.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = z.iter().map(|z| (z - m).exp()).collect();
            let s: f64 = e.iter().sum();
            let p: Vec<f64> = e.iter().map(|e| e / s).collect();
            for a in 0..K {
                let r = wi * (p[a] - (yi == a) as u8 as f64);
                for k in 0..d {
                    g[a * d + k] += r * v[k];
                }
                for b in 0..K {
                    let c = wi * p[a] * ((a == b) as u8 as f64 - p[b]);
                    if c == 0.0 {
                        continue;
                    }
                    for k in 0..d {
                        for l in 0..d {
                            h[a * d + k][b * d + l] += c * v[k] * v[l];
                        }
                    }
                }
            }
        }
        for i in 0..n {
            h[i][i] += 1e-4;
            g[i] += 1e-4 * beta[i];
        }
        // Pin the reference class.
        for k in 0..d {
            g[k] = 0.0;
            for j in 0..n {
                h[k][j] = 0.0;
                h[j][k] = 0.0;
            }
            h[k][k] = 1.0;
        }
        let step = solve(h, g);
        let mut moved = 0.0f64;
        for k in 0..n {
            beta[k] -= step[k];
            moved = moved.max(step[k].abs());
        }
        if moved < 1e-7 {
            break;
        }
    }
    (0..K)
        .map(|c| (beta[c * d], beta[c * d + 1..(c + 1) * d].to_vec()))
        .collect()
}

// ---- reliability ---------------------------------------------------------

const EDGES: [f64; 9] = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0001];

/// Expected calibration error over the same bins, without a table.
pub fn ece(rows: &[(f64, bool, f64)]) -> f64 {
    let mut bins = vec![(0.0f64, 0.0f64, 0.0f64); EDGES.len() - 1];
    for &(p, ok, w) in rows {
        if !p.is_finite() {
            continue;
        }
        let b = EDGES
            .windows(2)
            .position(|e| p >= e[0] && p < e[1])
            .unwrap_or(0);
        bins[b].0 += w * p;
        bins[b].1 += w * ok as u8 as f64;
        bins[b].2 += w;
    }
    let total: f64 = bins.iter().map(|b| b.2).sum();
    bins.iter()
        .filter(|b| b.2 > 0.0)
        .map(|b| b.2 / total * (b.0 / b.2 - b.1 / b.2).abs())
        .sum()
}

/// The shipped calibration's error on the records selected: (detections,
/// beats' reported class). Shared with the regression tests.
pub fn calibration_errors(opts: &Opts) -> std::io::Result<(f64, f64)> {
    let runs = runs(opts)?;
    let q: Vec<(f64, bool, f64)> = runs
        .iter()
        .flat_map(|r| r.det.iter().map(|(_, c, ok)| (*c as f64, *ok, 1.0)))
        .collect();
    let b: Vec<(f64, bool, f64)> = runs
        .iter()
        .flat_map(|r| {
            r.beats.iter().filter_map(|(_, class, conf, truth)| {
                let k = reported_index(*class)?;
                Some((*conf as f64, k == class_index(*truth), 1.0))
            })
        })
        .collect();
    Ok((ece(&q), ece(&b)))
}

/// (predicted, correct, weight) -> a table and the expected calibration error.
fn reliability(title: &str, rows: &[(f64, bool, f64)]) -> f64 {
    let mut bins = vec![(0.0f64, 0.0f64, 0.0f64, 0u64); EDGES.len() - 1];
    for &(p, ok, w) in rows {
        if !p.is_finite() {
            continue;
        }
        let b = EDGES
            .windows(2)
            .position(|e| p >= e[0] && p < e[1])
            .unwrap_or(0);
        bins[b].0 += w * p;
        bins[b].1 += w * ok as u8 as f64;
        bins[b].2 += w;
        bins[b].3 += 1;
    }
    let total: f64 = bins.iter().map(|b| b.2).sum();
    let mut ece = 0.0;
    println!("\n{title}");
    println!(
        "{:>14} {:>10} {:>12} {:>12}",
        "confidence", "count", "predicted %", "observed %"
    );
    for (i, b) in bins.iter().enumerate() {
        if b.2 == 0.0 {
            continue;
        }
        let (pred, obs) = (b.0 / b.2, b.1 / b.2);
        ece += b.2 / total * (pred - obs).abs();
        println!(
            "{:>6.2}-{:<6.2} {:>10} {:>12.2} {:>12.2}",
            EDGES[i],
            EDGES[i + 1].min(1.0),
            b.3,
            100.0 * pred,
            100.0 * obs
        );
    }
    println!("expected calibration error: {:.2} %", 100.0 * ece);
    ece
}

fn class_index(c: Aami) -> usize {
    match c {
        Aami::N => 0,
        Aami::S => 1,
        Aami::V => 2,
        _ => 3,
    }
}

fn reported_index(c: BeatClass) -> Option<usize> {
    match c {
        BeatClass::N => Some(0),
        BeatClass::S => Some(1),
        BeatClass::V => Some(2),
        BeatClass::F => Some(3),
        BeatClass::Unknown => None,
    }
}

fn print_logistic(name: &str, b: f64, w: &[f64]) {
    println!("\npub const {name}: Option<Logistic<NQ>> = Some(Logistic {{");
    println!("    bias: {b:.6},");
    let ws: Vec<String> = w.iter().map(|v| format!("{v:.6}")).collect();
    println!("    w: [{}],", ws.join(", "));
    println!("}});");
}

fn print_softmax(name: &str, m: &[(f64, Vec<f64>)]) {
    println!("\npub const {name}: Option<Softmax<NB>> = Some(Softmax {{");
    let bias: Vec<String> = m.iter().map(|(b, _)| format!("{b:.6}")).collect();
    println!("    bias: [{}],", bias.join(", "));
    println!("    w: [");
    for (_, w) in m {
        let ws: Vec<String> = w.iter().map(|v| format!("{v:.6}")).collect();
        println!("        [{}],", ws.join(", "));
    }
    println!("    ],");
    println!("}});");
}

// ---- commands ------------------------------------------------------------

pub fn fit_qrs(opts: &Opts) -> std::io::Result<()> {
    let runs = runs(opts)?;
    let hold = held_out(&runs);
    let stride = opts.get_usize("stride").unwrap_or(4).max(1);
    let collect = |pick: &dyn Fn(usize) -> bool| {
        let (mut x, mut y, mut w) = (Vec::new(), Vec::new(), Vec::new());
        let mut k = 0usize;
        for (i, r) in runs.iter().enumerate().filter(|(i, _)| pick(*i)) {
            let _ = i;
            for (f, _, ok) in &r.det {
                // Real beats outnumber spurious detections fifty to one.
                if *ok {
                    k += 1;
                    if k % stride != 0 {
                        continue;
                    }
                }
                x.push(f.iter().map(|v| *v as f64).collect::<Vec<_>>());
                y.push(*ok);
                w.push(if *ok { stride as f64 } else { 1.0 });
            }
        }
        (x, y, w)
    };
    let n_det: usize = runs.iter().map(|r| r.det.len()).sum();
    let n_bad: usize = runs
        .iter()
        .map(|r| r.det.iter().filter(|d| !d.2).count())
        .sum();
    println!(
        "detections {n_det} on {} records, {n_bad} without a reference beat ({:.2} %)",
        runs.len(),
        100.0 * n_bad as f64 / n_det.max(1) as f64
    );
    let (x, y, w) = collect(&|i| !hold[i]);
    let (b, ws) = fit_logistic(&x, &y, &w);
    let p = |f: &[f32; NQ]| {
        let z = b + f.iter().zip(&ws).map(|(a, c)| *a as f64 * c).sum::<f64>();
        1.0 / (1.0 + (-z).exp())
    };
    let rows: Vec<(f64, bool, f64)> = runs
        .iter()
        .enumerate()
        .filter(|(i, _)| hold[*i])
        .flat_map(|(_, r)| r.det.iter().map(|(f, _, ok)| (p(f), *ok, 1.0)))
        .collect();
    reliability("held-out quarter: P(real beat)", &rows);
    if opts.has("by-source") {
        let mut sources: Vec<&str> = runs.iter().map(|r| r.source.as_str()).collect();
        sources.sort();
        sources.dedup();
        for src in sources {
            let rows: Vec<(f64, bool, f64)> = runs
                .iter()
                .enumerate()
                .filter(|(i, r)| hold[*i] && r.source == src)
                .flat_map(|(_, r)| r.det.iter().map(|(f, _, ok)| (p(f), *ok, 1.0)))
                .collect();
            reliability(&format!("held out, {src}"), &rows);
        }
    }
    let (x, y, w) = collect(&|_| true);
    let (b, ws) = fit_logistic(&x, &y, &w);
    print_logistic("QRS_PT1", b, &ws);
    Ok(())
}

/// Rows written by `internal-beats --conf-rows`, one record per run.
fn runs_from_rows(path: &str) -> std::io::Result<Vec<Run>> {
    let text = std::fs::read_to_string(path)?;
    let mut by: std::collections::BTreeMap<String, Run> = Default::default();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 7 {
            continue;
        }
        let num = |i: usize| f[i].parse::<f32>().unwrap_or(f32::NAN);
        let class = match f[5] {
            "0" => BeatClass::N,
            "1" => BeatClass::S,
            "2" => BeatClass::V,
            _ => BeatClass::F,
        };
        let truth = match f[6] {
            "0" => Aami::N,
            "1" => Aami::S,
            "2" => Aami::V,
            _ => Aami::F,
        };
        let r = by.entry(f[0].to_string()).or_insert_with(|| Run {
            record: f[0].to_string(),
            source: f[0]
                .split_once('/')
                .map(|(s, _)| s)
                .unwrap_or("rows")
                .to_string(),
            det: Vec::new(),
            beats: Vec::new(),
        });
        r.beats.push((
            beat_vector(num(1), num(2), num(3), num(4) > 0.5),
            class,
            f32::NAN,
            truth,
        ));
    }
    Ok(by.into_values().collect())
}

pub fn fit_beats(opts: &Opts) -> std::io::Result<()> {
    let runs = match opts.get_str("rows") {
        Some(paths) => {
            let mut all = Vec::new();
            for p in paths.split(',') {
                all.extend(runs_from_rows(p)?);
            }
            all.sort_by(|a, b| (&a.source, &a.record).cmp(&(&b.source, &b.record)));
            all
        }
        None => runs(opts)?,
    };
    // `--dump-rows`: write the beats out and stop, so rows from several runs
    // (cross-fitted folds, other corpora) can be fitted together.
    if let Some(path) = opts.get_str("dump-rows") {
        let mut text = String::from("record,pv,ps,pf,fib,class,truth\n");
        let p = |l: f32| 1.0 / (1.0 + (-l).exp());
        for r in &runs {
            for (f, class, _, truth) in &r.beats {
                let Some(c) = reported_index(*class).or(Some(4)) else {
                    continue;
                };
                if c == 4 {
                    continue;
                }
                text.push_str(&format!(
                    "{}/{},{},{},{},{},{},{}\n",
                    r.source,
                    r.record,
                    p(f[0]),
                    p(f[1]),
                    p(f[2]),
                    f[3],
                    c,
                    class_index(*truth)
                ));
            }
        }
        std::fs::write(path, text)?;
        eprintln!("wrote {path}");
        return Ok(());
    }
    // `--per-record`: every record weighs the same, so the calibration is for
    // a typical patient rather than for whichever recordings are longest.
    let per_record = opts.per_record;
    let hold = held_out(&runs);
    let stride = opts.get_usize("stride").unwrap_or(8).max(1);
    let collect = |pick: &dyn Fn(usize) -> bool| {
        let (mut x, mut y, mut w) = (Vec::new(), Vec::new(), Vec::new());
        let mut k = 0usize;
        for (_, r) in runs.iter().enumerate().filter(|(i, _)| pick(*i)) {
            let scale = if per_record {
                1000.0 / r.beats.len().max(1) as f64
            } else {
                1.0
            };
            for (f, _, _, truth) in &r.beats {
                let c = class_index(*truth);
                if c == 0 {
                    k += 1;
                    if k % stride != 0 {
                        continue;
                    }
                }
                x.push(f.iter().map(|v| *v as f64).collect::<Vec<_>>());
                y.push(c);
                w.push(scale * if c == 0 { stride as f64 } else { 1.0 });
            }
        }
        (x, y, w)
    };
    let mut counts = [0usize; 4];
    for r in &runs {
        for b in &r.beats {
            counts[class_index(b.3)] += 1;
        }
    }
    println!(
        "matched beats on {} records: N {} S {} V {} F {}",
        runs.len(),
        counts[0],
        counts[1],
        counts[2],
        counts[3]
    );
    let (x, y, w) = collect(&|i| !hold[i]);
    let m = fit_softmax(&x, &y, &w);
    let probs = |f: &[f32; NB]| -> [f64; 4] {
        let mut z = [0.0f64; 4];
        for c in 0..4 {
            z[c] = m[c].0
                + f.iter()
                    .zip(&m[c].1)
                    .map(|(a, b)| *a as f64 * b)
                    .sum::<f64>();
        }
        let mx = z.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
        let s: f64 = e.iter().sum();
        [e[0] / s, e[1] / s, e[2] / s, e[3] / s]
    };
    let rows: Vec<(f64, bool, f64)> = runs
        .iter()
        .enumerate()
        .filter(|(i, _)| hold[*i])
        .flat_map(|(_, r)| {
            r.beats.iter().filter_map(|(f, class, _, truth)| {
                let k = reported_index(*class)?;
                Some((probs(f)[k], k == class_index(*truth), 1.0))
            })
        })
        .collect();
    reliability("held-out quarter: P(reported class is right)", &rows);
    let (x, y, w) = collect(&|_| true);
    let m = fit_softmax(&x, &y, &w);
    print_softmax(opts.get_str("name").unwrap_or("BEATS_CLINICAL4"), &m);
    Ok(())
}

pub fn eval(opts: &Opts) -> std::io::Result<()> {
    // Rows from `internal-beats --conf-rows`: scored with the shipped patch
    // calibration, since those runs do not pass through the pipeline.
    if let Some(path) = opts.get_str("rows") {
        let model = ecg_pipeline::confidence::BEATS_PATCH3.expect("patch calibration");
        let runs = runs_from_rows(path)?;
        let rows: Vec<(f64, bool, f64)> = runs
            .iter()
            .flat_map(|r| {
                r.beats.iter().filter_map(|(f, class, _, truth)| {
                    let k = reported_index(*class)?;
                    Some((
                        model.probabilities(f)[k] as f64,
                        k == class_index(*truth),
                        1.0,
                    ))
                })
            })
            .collect();
        println!("{} recordings, {} beats", runs.len(), rows.len());
        reliability("patch beats: P(reported class is right)", &rows);
        for (k, name) in [(0usize, "N"), (1, "S"), (2, "V")] {
            let sub: Vec<(f64, bool, f64)> = runs
                .iter()
                .flat_map(|r| {
                    r.beats.iter().filter_map(move |(f, class, _, truth)| {
                        (reported_index(*class) == Some(k)).then(|| {
                            (
                                model.probabilities(f)[k] as f64,
                                class_index(*truth) == k,
                                1.0,
                            )
                        })
                    })
                })
                .collect();
            if !sub.is_empty() {
                reliability(&format!("patch beats reported {name}"), &sub);
            }
        }
        return Ok(());
    }
    let runs = runs(opts)?;
    let q: Vec<(f64, bool, f64)> = runs
        .iter()
        .flat_map(|r| r.det.iter().map(|(_, c, ok)| (*c as f64, *ok, 1.0)))
        .collect();
    let calibrated = q.iter().filter(|r| r.0.is_finite()).count();
    println!(
        "{} records, {} detections ({} with a confidence)",
        runs.len(),
        q.len(),
        calibrated
    );
    reliability("QRS: P(real beat)", &q);
    let b: Vec<(f64, bool, f64)> = runs
        .iter()
        .flat_map(|r| {
            r.beats.iter().filter_map(|(_, class, conf, truth)| {
                let k = reported_index(*class)?;
                Some((*conf as f64, k == class_index(*truth), 1.0))
            })
        })
        .collect();
    reliability("beats: P(reported class is right)", &b);
    // The same, for the beats reported as each class.
    for (k, name) in [(1usize, "S"), (2, "V"), (3, "F")] {
        let rows: Vec<(f64, bool, f64)> = runs
            .iter()
            .flat_map(|r| {
                r.beats.iter().filter_map(move |(_, class, conf, truth)| {
                    (reported_index(*class) == Some(k))
                        .then(|| (*conf as f64, class_index(*truth) == k, 1.0))
                })
            })
            .collect();
        if !rows.is_empty() {
            reliability(&format!("beats reported {name}"), &rows);
        }
    }
    Ok(())
}
