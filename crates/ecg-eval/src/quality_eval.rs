//! Noise-detection evaluation.
//!
//! Ground truth comes from the MIT-BIH Noise Stress Test protocol: records
//! `118e*` / `119e*` are clean recordings with calibrated noise added in
//! two-minute segments, alternating with two clean minutes, starting five
//! minutes in. That gives an exact per-second label at six known SNRs, which is
//! far stronger evidence than a subjective artefact annotation.
//!
//! Two things are measured, because they answer different questions:
//!
//! * **Discrimination** - can the monitor tell a noisy second from a clean one
//!   (ROC AUC over the whole record).
//! * **Usefulness** - does gating on it actually buy detector precision, and at
//!   what cost in recall. A quality signal that scores well and changes no
//!   decision is worth nothing.

use crate::manifest::RecordEntry;
use crate::metrics;
use crate::qrs_eval;
use crate::Opts;
use ecg_pipeline::Preprocessor;
use ecg_quality::{QualityMonitor, QualitySample};
use ecg_wfdb::{read_signal, AnnotationFile, Header};
use rayon::prelude::*;
use std::path::Path;

/// Seconds of clean signal before the first added-noise segment.
const NST_LEAD_IN_S: f64 = 300.0;
/// Length of each added-noise segment and of each clean gap between them.
const NST_SEGMENT_S: f64 = 120.0;

/// Is second `t` inside an added-noise segment of a noise-stress record?
fn nst_noisy(t: f64) -> bool {
    if t < NST_LEAD_IN_S {
        return false;
    }
    let phase = (t - NST_LEAD_IN_S) % (2.0 * NST_SEGMENT_S);
    phase < NST_SEGMENT_S
}

/// SNR in dB encoded in a noise-stress record name (`118e06` -> 6, `118e_6` -> -6).
fn nst_snr(record: &str) -> Option<i32> {
    let idx = record.find('e')?;
    let tail = &record[idx + 1..];
    if let Some(rest) = tail.strip_prefix('_') {
        rest.parse::<i32>().ok().map(|v| -v)
    } else {
        tail.parse::<i32>().ok()
    }
}

/// What one record's analysis yields: name, added-noise SNR, per-second rows,
/// and the detector's score with the quality gate open and closed.
type RecordAnalysis = (
    String,
    i32,
    Vec<SecondRow>,
    metrics::DetectionScore,
    metrics::DetectionScore,
);

/// A named accessor over a second's row, oriented so larger means more noisy.
type FeatureProbe = (&'static str, fn(&SecondRow) -> f32);

/// A named predicate selecting a subpopulation of seconds.
type Group = (&'static str, Box<dyn Fn(&SecondRow) -> bool>);

struct SecondRow {
    noisy: bool,
    /// Monitor output: lower is worse.
    score: f32,
    unusable: bool,
    /// Detector errors attributed to this second.
    errors: u32,
    beats: u32,
    feat: ecg_quality::QualityFeatures,
}

/// Each feature, oriented so that a **larger** value means "more likely noise".
/// Orienting them here keeps the AUC comparison honest: a feature that is
/// backwards shows up as an AUC below 0.5 rather than being silently flipped.
const FEATURES: [FeatureProbe; 8] = [
    ("score", |r| -r.score),
    ("-kurtosis", |r| -r.feat.kurtosis),
    ("|skewness|", |r| -r.feat.skewness.abs()),
    ("hf_ratio", |r| r.feat.hf_ratio),
    ("base_ratio", |r| r.feat.base_ratio),
    ("-qrs_ratio", |r| -r.feat.qrs_ratio),
    ("p2p_rel_dev", |r| (r.feat.p2p_rel.max(1e-6)).ln().abs()),
    ("flat_frac", |r| r.feat.flat_frac),
];

fn analyse(entry: &RecordEntry, opts: &Opts) -> Option<RecordAnalysis> {
    let hdr = Header::read(&entry.hea_path()).ok()?;
    let lead = opts.lead.min(hdr.n_sig - 1);
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples).ok()?;
    let ann = AnnotationFile::read(Path::new(&entry.ann_path("atr"))).ok()?;
    let refs = ann.beat_samples();
    let fs = hdr.fs;

    let cfg = qrs_eval::config_from(opts, fs);
    let mut pre = Preprocessor::new(cfg.preprocess);
    let mut qual = QualityMonitor::new(cfg.quality);

    let spp = fs as usize; // samples per second
    let n_sec = sig.len() / spp.max(1);
    let mut rows: Vec<SecondRow> = Vec::with_capacity(n_sec);
    let mut acc: Vec<QualitySample> = Vec::with_capacity(spp);

    for s in 0..n_sec {
        acc.clear();
        for &x in &sig[s * spp..(s + 1) * spp] {
            let b = pre.process(x);
            acc.push(qual.process(b.raw, b.clean, b.baseline, b.hf, b.qrs, b.saturated));
        }
        // Worst instant in the second: a two-hundred-millisecond artefact ruins a
        // beat, and averaging it away would hide exactly what we are looking for.
        // Locate the worst instant once and read both the score and the features
        // from that same sample; deriving them separately let a NaN-tolerant
        // `min` and a NaN-as-equal `min_by` disagree, which showed up as healthy
        // features printed next to an impossible score.
        let mut worst = acc[0];
        for q in acc.iter() {
            if q.score < worst.score {
                worst = *q;
            }
        }
        let score = worst.score;
        let unusable = acc
            .iter()
            .any(|q| q.level(&cfg.quality) == ecg_quality::Quality::Unusable);
        rows.push(SecondRow {
            noisy: nst_noisy(s as f64),
            score,
            unusable,
            errors: 0,
            beats: 0,
            feat: worst.features,
        });
    }

    // Detector behaviour with the gate off and on, so the cost of gating is
    // measured rather than assumed.
    let mut open = opts.clone_with(&[("gate", "")]);
    open.gate = false;
    let r_open = qrs_eval::run_one(entry, &open);
    let mut gated = opts.clone_with(&[]);
    gated.gate = true;
    let r_gated = qrs_eval::run_one(entry, &gated);

    // Attribute each reference beat and each error to its second.
    let tol = (opts.tol_ms * fs / 1000.0).round() as i64;
    let det: Vec<i64> = r_open.detections.clone();
    let _ = &r_open.skipped;
    let mut matched = vec![false; refs.len()];
    let mut used = vec![false; det.len()];
    let (mut i, mut j) = (0usize, 0usize);
    while i < refs.len() && j < det.len() {
        let d = det[j] - refs[i];
        if d < -tol {
            j += 1;
        } else if d > tol {
            i += 1;
        } else {
            matched[i] = true;
            used[j] = true;
            i += 1;
            j += 1;
        }
    }
    for (k, &s) in refs.iter().enumerate() {
        let sec = (s as usize / spp).min(rows.len().saturating_sub(1));
        if rows.is_empty() {
            break;
        }
        rows[sec].beats += 1;
        if !matched[k] {
            rows[sec].errors += 1;
        }
    }
    for (k, &s) in det.iter().enumerate() {
        if used[k] {
            continue;
        }
        let sec = (s as usize / spp).min(rows.len().saturating_sub(1));
        if !rows.is_empty() {
            rows[sec].errors += 1;
        }
    }

    let snr = nst_snr(&entry.record).unwrap_or(99);
    Some((entry.record.clone(), snr, rows, r_open.score, r_gated.score))
}

/// Area under the ROC curve; `value` is oriented so larger means more positive.
fn auc_by(
    rows: &[SecondRow],
    label: impl Fn(&SecondRow) -> bool,
    value: impl Fn(&SecondRow) -> f32,
) -> f64 {
    let mut pos: Vec<f32> = Vec::new();
    let mut neg: Vec<f32> = Vec::new();
    for r in rows {
        if label(r) {
            pos.push(value(r))
        } else {
            neg.push(value(r))
        }
    }
    if pos.is_empty() || neg.is_empty() {
        return f64::NAN;
    }
    // Rank-sum (Mann-Whitney U).
    let mut all: Vec<(f32, bool)> = pos
        .iter()
        .map(|&v| (v, true))
        .chain(neg.iter().map(|&v| (v, false)))
        .collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank_sum = 0.0f64;
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && all[j + 1].0 == all[i].0 {
            j += 1;
        }
        // Ties share the average of the ranks they span.
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for &(_, is_pos) in &all[i..=j] {
            if is_pos {
                rank_sum += avg_rank;
            }
        }
        i = j + 1;
    }
    let (n1, n2) = (pos.len() as f64, neg.len() as f64);
    (rank_sum - n1 * (n1 + 1.0) / 2.0) / (n1 * n2)
}

/// Pooled AUC of the quality score for predicting a detector error. Shared with
/// the regression tests.
pub fn error_auc(opts: &Opts) -> Option<f64> {
    let entries = opts.select().ok()?;
    let out: Vec<_> = entries
        .par_iter()
        .filter_map(|e| analyse(e, opts))
        .collect();
    let rows: Vec<SecondRow> = out
        .iter()
        .flat_map(|(_, _, r, _, _)| r.iter())
        .map(|r| SecondRow {
            noisy: r.noisy,
            score: r.score,
            unusable: r.unusable,
            errors: r.errors,
            beats: r.beats,
            feat: r.feat,
        })
        .collect();
    if rows.is_empty() {
        return None;
    }
    Some(auc_by(&rows, |r| r.errors > 0, |r| -r.score))
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    if entries.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }

    let out: Vec<_> = entries
        .par_iter()
        .filter_map(|e| analyse(e, opts))
        .collect();

    println!("\n── noise detection (MIT-BIH noise stress protocol) ───────────");
    println!(
        "{:<10} {:>5} {:>9} {:>9} {:>8} {:>8} {:>10} {:>10}",
        "record", "SNR", "AUC-noise", "AUC-err", "flag%", "noisy%", "PPV open", "PPV gated"
    );
    let mut by_snr: std::collections::BTreeMap<i32, (f64, f64, f64, f64, u32)> = Default::default();
    for (rec, snr, rows, open, gated) in &out {
        let a_noise = auc_by(rows, |r| r.noisy, |r| -r.score);
        let a_err = auc_by(rows, |r| r.errors > 0, |r| -r.score);
        let flagged = rows.iter().filter(|r| r.unusable).count() as f64 / rows.len() as f64;
        let noisy = rows.iter().filter(|r| r.noisy).count() as f64 / rows.len() as f64;
        println!(
            "{:<10} {:>5} {:>9.4} {:>9.4} {:>7.1}% {:>7.1}% {:>9.3}% {:>9.3}%",
            rec,
            snr,
            a_noise,
            a_err,
            100.0 * flagged,
            100.0 * noisy,
            100.0 * open.ppv(),
            100.0 * gated.ppv()
        );
        let e = by_snr.entry(*snr).or_insert((0.0, 0.0, 0.0, 0.0, 0));
        e.0 += a_noise;
        e.1 += a_err;
        e.2 += open.ppv();
        e.3 += gated.ppv();
        e.4 += 1;
    }

    // Per-feature discrimination, pooled over every record.
    let all_rows: Vec<&SecondRow> = out.iter().flat_map(|(_, _, r, _, _)| r.iter()).collect();
    let owned: Vec<SecondRow> = all_rows
        .iter()
        .map(|r| SecondRow {
            noisy: r.noisy,
            score: r.score,
            unusable: r.unusable,
            errors: r.errors,
            beats: r.beats,
            feat: r.feat,
        })
        .collect();
    println!("\n{:<14} {:>12} {:>12}", "feature", "AUC-noise", "AUC-err");
    for (name, f) in FEATURES.iter() {
        println!(
            "{:<14} {:>12.4} {:>12.4}",
            name,
            auc_by(&owned, |r| r.noisy, f),
            auc_by(&owned, |r| r.errors > 0, f)
        );
    }

    println!(
        "\n{:<10} {:>9} {:>9} {:>10} {:>10}",
        "SNR dB", "AUC-noise", "AUC-err", "PPV open", "PPV gated"
    );
    for (snr, (an, ae, po, pg, n)) in &by_snr {
        let n = *n as f64;
        println!(
            "{:<10} {:>9.4} {:>9.4} {:>9.3}% {:>9.3}%",
            snr,
            an / n,
            ae / n,
            100.0 * po / n,
            100.0 * pg / n
        );
    }
    Ok(())
}

/// Percentile table of every feature, split by label. Thresholds are set from
/// this rather than by eye: a threshold that sits inside the clean distribution
/// flags healthy signal, which is how the first version came to condemn every
/// window of a 24 dB record.
pub fn dump_features(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    let out: Vec<_> = entries
        .par_iter()
        .filter_map(|e| analyse(e, opts))
        .collect();
    let rows: Vec<&SecondRow> = out.iter().flat_map(|(_, _, r, _, _)| r.iter()).collect();

    let groups: [Group; 3] = [
        ("all", Box::new(|_: &SecondRow| true)),
        ("clean", Box::new(|r: &SecondRow| !r.noisy)),
        ("noisy", Box::new(|r: &SecondRow| r.noisy)),
    ];
    let feats: [FeatureProbe; 10] = [
        ("score", |r| r.score),
        ("kurtosis", |r| r.feat.kurtosis),
        ("skewness", |r| r.feat.skewness),
        ("hf_ratio", |r| r.feat.hf_ratio),
        ("base_ratio", |r| r.feat.base_ratio),
        ("qrs_ratio", |r| r.feat.qrs_ratio),
        ("p2p_rel", |r| r.feat.p2p_rel),
        ("flat_frac", |r| r.feat.flat_frac),
        ("sat_frac", |r| r.feat.sat_frac),
        ("p2p", |r| r.feat.p2p),
    ];

    println!(
        "{:<12} {:<7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "feature", "group", "p1", "p5", "p25", "p50", "p75", "p99"
    );
    for (fname, fget) in feats.iter() {
        for (gname, keep) in groups.iter() {
            let mut v: Vec<f32> = rows.iter().filter(|r| keep(r)).map(|r| fget(r)).collect();
            if v.is_empty() {
                continue;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
            println!(
                "{:<12} {:<7} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
                fname,
                gname,
                q(0.01),
                q(0.05),
                q(0.25),
                q(0.50),
                q(0.75),
                q(0.99)
            );
        }
    }
    Ok(())
}

/// Raw per-second rows for one record: every feature beside the score that was
/// computed from them, so a disagreement between the two is visible directly.
pub fn dump_rows(opts: &Opts) -> std::io::Result<()> {
    let entries = opts.select()?;
    let limit = opts.get_usize("rows").unwrap_or(30);
    let from = opts.get_usize("from-sec").unwrap_or(0);
    for e in &entries {
        let Some((rec, _, rows, _, _)) = analyse(e, opts) else {
            continue;
        };
        println!("# {rec}");
        println!(
            "{:>5} {:>6} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8}",
            "sec",
            "noisy",
            "score",
            "kurt",
            "skew",
            "hf",
            "base",
            "qrs_r",
            "p2p_rel",
            "kurt_r",
            "qrs_rel",
            "flat"
        );
        let n = rows.len().max(1) as f64;
        let low = rows.iter().filter(|r| r.score <= 0.5).count() as f64;
        let err_sec = rows.iter().filter(|r| r.errors > 0).count() as f64;
        let err_in_low = rows
            .iter()
            .filter(|r| r.errors > 0 && r.score <= 0.5)
            .count() as f64;
        println!(
            "# seconds {:.0}   score<=0.5 {:.1}%   seconds with a detector error {:.1}%   of those, flagged {:.1}%",
            n, 100.0 * low / n, 100.0 * err_sec / n, 100.0 * err_in_low / err_sec.max(1.0)
        );
        for (i, r) in rows.iter().enumerate().skip(from).take(limit) {
            println!(
                "{:>5} {:>6} {:>7.3} {:>9.3} {:>9.3} {:>9.4} {:>9.4} {:>9.4} {:>9.3} {:>8.3} {:>8.3} {:>8.3}",
                i, r.noisy, r.score, r.feat.kurtosis, r.feat.skewness, r.feat.hf_ratio,
                r.feat.base_ratio, r.feat.qrs_ratio, r.feat.p2p_rel, r.feat.kurtosis_rel,
                r.feat.qrs_ratio_rel, r.feat.flat_frac
            );
        }
    }
    Ok(())
}
