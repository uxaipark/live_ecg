//! Validation of the quality monitor against human quality annotations.
//!
//! # Why this corpus
//!
//! Phase 1 validated noise detection against the MIT noise-stress protocol,
//! which has exactly known labels — and exactly one kind of noise. The 'e'
//! records add *electrode motion* only. So the feature meant to catch muscle
//! artefact, `hf_ratio`, was never tested against the thing it describes, and
//! its measured AUC there (0.27, inverted) said only that electrode motion is
//! not high-frequency.
//!
//! BUT QDB is the opposite kind of evidence: real long-term recordings with no
//! synthetic noise at all, rated by four independent annotators into three
//! classes. Worse ground truth, genuinely different failure modes.
//!
//! * class 1 — full diagnostic quality
//! * class 2 — QRS reliable, other waves not
//! * class 3 — QRS not reliably detectable
//!
//! Those map onto the monitor's own three levels, which is the point: the
//! question is whether a machine and a human agree on what is unusable.

use crate::Opts;
use ecg_pipeline::{PipelineConfig, Preprocessor};
use ecg_quality::{QualityFeatures, QualityMonitor};
use ecg_wfdb::{read_signal, Header};
use rayon::prelude::*;
use std::path::Path;

/// Consensus label for one second, plus the features measured over it.
pub struct Second {
    /// 1, 2 or 3. The median of the annotators who labelled this second.
    pub truth: u8,
    /// Annotators who agreed with the consensus, out of those who labelled it.
    pub agreement: f32,
    pub score: f32,
    pub features: QualityFeatures,
    /// Median atrial evidence over the beats in this second, from the
    /// delineator. `NaN` when no beat fell here.
    ///
    /// Class 1 and class 2 differ by exactly one thing - whether anything other
    /// than the QRS complex can be read - and the monitor measures none of it.
    /// These two are the cheapest test of whether the delineator's own output
    /// can tell them apart.
    pub p_confidence: f32,
    pub p_coherence: f32,
}

/// One annotator's segmentation: `(start, end, class)`, samples, 1-based.
type Segments = Vec<(i64, i64, u8)>;

/// Read the four annotator columns of a BUT QDB annotation file.
fn read_annotations(path: &Path) -> std::io::Result<Vec<Segments>> {
    let text = std::fs::read_to_string(path)?;
    let mut out: Vec<Segments> = vec![Vec::new(); 4];
    for line in text.lines() {
        let f: Vec<&str> = line.split(',').collect();
        for (g, segs) in out.iter_mut().enumerate() {
            let (a, b, c) = (3 * g, 3 * g + 1, 3 * g + 2);
            if f.len() <= c {
                continue;
            }
            let (Ok(s), Ok(e), Ok(k)) = (
                f[a].trim().parse::<i64>(),
                f[b].trim().parse::<i64>(),
                f[c].trim().parse::<u8>(),
            ) else {
                continue;
            };
            // Class 0 marks a stretch the annotator did not rate.
            if (1..=3).contains(&k) && e >= s {
                segs.push((s, e, k));
            }
        }
    }
    Ok(out)
}

/// Per-second class from one annotator, 0 where unrated.
fn per_second(segs: &Segments, n_sec: usize, fs: f64) -> Vec<u8> {
    let mut v = vec![0u8; n_sec];
    for &(s, e, k) in segs {
        // Annotation samples are 1-based.
        let a = (((s - 1).max(0) as f64) / fs).floor() as usize;
        let b = ((e as f64 / fs).ceil() as usize).min(n_sec);
        for x in v.iter_mut().take(b).skip(a.min(n_sec)) {
            *x = k;
        }
    }
    v
}

pub fn analyse(entry: &crate::manifest::RecordEntry, opts: &Opts) -> std::io::Result<Vec<Second>> {
    let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    let hdr = Header::read(&entry.hea_path()).map_err(|e| err(e.to_string()))?;
    let ann_path = entry
        .hea_path()
        .with_file_name(format!("{}_ANN.csv", entry.record));
    let groups = read_annotations(&ann_path)?;

    let fs = hdr.fs;
    let n_sec = (hdr.n_samples as f64 / fs).floor() as usize;
    let labels: Vec<Vec<u8>> = groups.iter().map(|g| per_second(g, n_sec, fs)).collect();

    let sig = read_signal(&hdr, 0, 0, hdr.n_samples).map_err(|e| err(e.to_string()))?;
    let cfg = PipelineConfig::new(fs);
    let mut pre = Preprocessor::new(cfg.preprocess);
    let mut qual = QualityMonitor::new(cfg.quality);

    // A second pass with the whole engine, for the wave evidence. Kept apart
    // from the monitor's own pass on purpose: the monitor decides whether beats
    // may be analysed at all, so it cannot be fed anything derived from them.
    let (p_conf, p_coh) = wave_evidence(&sig, cfg, fs, n_sec);

    let spp = fs as usize;
    let mut out = Vec::with_capacity(n_sec);
    let mut worst = (f32::INFINITY, QualityFeatures::default());
    for (i, &x) in sig.iter().enumerate() {
        let b = pre.process(x);
        let q = qual.process(b.raw, b.clean, b.baseline, b.hf, b.qrs, b.saturated);
        if q.score < worst.0 {
            worst = (q.score, q.features);
        }
        if (i + 1) % spp == 0 {
            let sec = i / spp;
            // Consensus: the median of the annotators who rated this second. A
            // second nobody rated is dropped rather than assumed good.
            let mut votes: Vec<u8> = labels
                .iter()
                .filter_map(|l| Some(*l.get(sec)?).filter(|&k| k > 0))
                .collect();
            if !votes.is_empty() && sec < n_sec {
                votes.sort_unstable();
                let truth = votes[votes.len() / 2];
                let agree =
                    votes.iter().filter(|&&k| k == truth).count() as f32 / votes.len() as f32;
                out.push(Second {
                    truth,
                    agreement: agree,
                    score: worst.0,
                    features: worst.1,
                    p_confidence: p_conf.get(sec).copied().unwrap_or(f32::NAN),
                    p_coherence: p_coh.get(sec).copied().unwrap_or(f32::NAN),
                });
            }
            worst = (f32::INFINITY, QualityFeatures::default());
        }
    }
    let _ = opts;
    Ok(out)
}

/// Per-second medians of the delineator's atrial evidence.
fn wave_evidence(sig: &[f32], cfg: PipelineConfig, fs: f64, n_sec: usize) -> (Vec<f32>, Vec<f32>) {
    use ecg_pipeline::{ChannelOutput, ChannelPipeline};
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let mut conf: Vec<Vec<f32>> = vec![Vec::new(); n_sec];
    let mut coh: Vec<Vec<f32>> = vec![Vec::new(); n_sec];
    let block = ((fs * 0.25) as usize).max(1);
    for chunk in sig.chunks(block) {
        out.clear();
        pipe.push(chunk, &mut out);
        for d in &out.waves {
            let sec = (d.r as f64 / fs) as usize;
            if sec < n_sec {
                conf[sec].push(d.p_confidence);
            }
        }
        for v in &out.classes {
            let sec = (v.sample as f64 / fs) as usize;
            if sec < n_sec {
                coh[sec].push(v.features.p_ncc_prev);
            }
        }
    }
    let median = |mut v: Vec<f32>| -> f32 {
        if v.is_empty() {
            return f32::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[v.len() / 2]
    };
    (
        conf.into_iter().map(median).collect(),
        coh.into_iter().map(median).collect(),
    )
}

/// Mann-Whitney AUC; larger `value` must mean "more likely the positive class".
fn auc(
    rows: &[&Second],
    positive: impl Fn(&Second) -> Option<bool>,
    value: impl Fn(&Second) -> f32,
) -> f64 {
    let mut all: Vec<(f32, bool)> = rows
        .iter()
        .filter_map(|r| Some((value(r), positive(r)?)))
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

type Probe = (&'static str, fn(&Second) -> f32);

/// Oriented so larger means worse quality.
const PROBES: [Probe; 8] = [
    ("score", |r| -r.score),
    ("-kurtosis", |r| -r.features.kurtosis),
    ("-kurtosis_rel", |r| -r.features.kurtosis_rel),
    ("hf_ratio", |r| r.features.hf_ratio),
    ("base_ratio", |r| r.features.base_ratio),
    ("-qrs_ratio", |r| -r.features.qrs_ratio),
    ("p2p_rel_dev", |r| r.features.p2p_rel.max(1e-6).ln().abs()),
    ("flat_frac", |r| r.features.flat_frac),
];

/// Score AUC for separating class 3 (QRS not reliably detectable) from class 1
/// (full diagnostic quality). Shared with the regression tests.
/// AUC for separating class 2 (QRS reliable only) from class 1 (everything
/// readable), using the atrial coherence rather than the quality score.
pub fn legibility_auc(opts: &Opts) -> Option<f64> {
    let entries = opts.select().ok()?;
    if entries.is_empty() {
        return None;
    }
    let per_record: Vec<Vec<Second>> = entries
        .par_iter()
        .filter_map(|e| analyse(e, opts).ok())
        .collect();
    let rows: Vec<&Second> = per_record
        .iter()
        .flatten()
        .filter(|r| r.p_coherence.is_finite())
        .collect();
    if rows.is_empty() {
        return None;
    }
    Some(auc(
        &rows,
        |r| match r.truth {
            2 => Some(true),
            1 => Some(false),
            _ => None,
        },
        |r| -r.p_coherence,
    ))
}

pub fn unusable_auc(opts: &Opts) -> Option<f64> {
    let entries = opts.select().ok()?;
    if entries.is_empty() {
        return None;
    }
    let per_record: Vec<Vec<Second>> = entries
        .par_iter()
        .filter_map(|e| analyse(e, opts).ok())
        .collect();
    let rows: Vec<&Second> = per_record.iter().flatten().collect();
    if rows.is_empty() {
        return None;
    }
    Some(auc(
        &rows,
        |r| match r.truth {
            3 => Some(true),
            1 => Some(false),
            _ => None,
        },
        |r| -r.score,
    ))
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    if entries.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }
    eprintln!("BUT QDB: {} records", entries.len());

    let per_record: Vec<(String, Vec<Second>)> = entries
        .par_iter()
        .filter_map(|e| analyse(e, opts).ok().map(|s| (e.record.clone(), s)))
        .collect();
    let rows: Vec<&Second> = per_record.iter().flat_map(|(_, v)| v.iter()).collect();
    if rows.is_empty() {
        eprintln!("no annotated seconds");
        return Ok(());
    }

    let n1 = rows.iter().filter(|r| r.truth == 1).count();
    let n2 = rows.iter().filter(|r| r.truth == 2).count();
    let n3 = rows.iter().filter(|r| r.truth == 3).count();
    let unanimous = rows.iter().filter(|r| r.agreement >= 0.999).count();
    println!("\n── quality against human annotation (BUT QDB) ────────────────");
    println!("records          {}", per_record.len());
    println!(
        "seconds          {}  ({:.1} h)  class 1 {:.1}%  class 2 {:.1}%  class 3 {:.1}%",
        rows.len(),
        rows.len() as f64 / 3600.0,
        100.0 * n1 as f64 / rows.len() as f64,
        100.0 * n2 as f64 / rows.len() as f64,
        100.0 * n3 as f64 / rows.len() as f64
    );
    println!(
        "annotators agreed unanimously on {:.1}% of seconds",
        100.0 * unanimous as f64 / rows.len() as f64
    );

    // Two questions, because they are not the same question. "Unusable" is the
    // one the gate acts on; "less than full quality" is the one a morphology
    // consumer cares about.
    let q3 = |r: &Second| match r.truth {
        3 => Some(true),
        1 => Some(false),
        _ => None,
    };
    let q23 = |r: &Second| Some(r.truth >= 2);

    println!(
        "\n{:<16} {:>18} {:>18}",
        "feature", "class 3 vs 1", "class 2+3 vs 1"
    );
    for (name, f) in PROBES.iter() {
        println!(
            "{:<16} {:>18.4} {:>18.4}",
            name,
            auc(&rows, q3, f),
            auc(&rows, q23, f)
        );
    }

    // Where the annotators agree, the label is strong evidence; where they do
    // not, disagreement between a machine and a human means less.
    let strong: Vec<&Second> = rows
        .iter()
        .filter(|r| r.agreement >= 0.999)
        .copied()
        .collect();
    println!(
        "\non unanimous seconds only:  score AUC(3 vs 1) {:.4}   AUC(2+3 vs 1) {:.4}",
        auc(&strong, q3, |r| -r.score),
        auc(&strong, q23, |r| -r.score)
    );

    // Class 1 against class 2: the one distinction the monitor has never been
    // able to make, because it is about the P and T waves and the monitor
    // measures neither. These two come from the delineator instead.
    let q12 = |r: &Second| match r.truth {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    };
    let measured: Vec<&Second> = rows
        .iter()
        .filter(|r| r.p_confidence.is_finite() && r.p_coherence.is_finite())
        .copied()
        .collect();
    println!(
        "\nclass 2 vs class 1, on the {} seconds that contained a beat:",
        measured.len()
    );
    println!(
        "  quality score        {:.4}\n  P confidence         {:.4}\n  atrial coherence     {:.4}",
        auc(&measured, q12, |r| -r.score),
        auc(&measured, q12, |r| -r.p_confidence),
        auc(&measured, q12, |r| -r.p_coherence),
    );

    // What the two together are worth as a three-class verdict, which is the
    // question the corpus actually poses. The monitor decides class 3; among
    // what it passes, the atrial coherence decides 1 against 2. The threshold
    // is swept here rather than fixed, because the number that matters is
    // whether a usable operating point exists at all.
    let unusable = ecg_quality::QualityConfig::new(250.0).score_bad;
    println!("\nthree-class agreement (monitor for class 3, coherence for 1 vs 2):");
    println!(
        "{:>10} {:>9} {:>9} {:>9} {:>10}",
        "coherence", "class 1", "class 2", "class 3", "balanced"
    );
    for t in [0.70f32, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97] {
        // 0.95 is `PipelineConfig::wave_legible_ncc`, chosen from this sweep.
        let mut hit = [0usize; 3];
        let mut total = [0usize; 3];
        for r in &measured {
            let k = (r.truth as usize).clamp(1, 3) - 1;
            total[k] += 1;
            let level = if r.score < unusable {
                3
            } else if r.p_coherence >= t {
                1
            } else {
                2
            };
            if level == r.truth {
                hit[k] += 1;
            }
        }
        let rate = |k: usize| {
            if total[k] == 0 {
                f64::NAN
            } else {
                hit[k] as f64 / total[k] as f64
            }
        };
        let balanced = (0..3).map(rate).filter(|v| v.is_finite()).sum::<f64>()
            / (0..3).map(rate).filter(|v| v.is_finite()).count() as f64;
        println!(
            "{t:>10.2} {:>9.3} {:>9.3} {:>9.3} {:>10.3}",
            rate(0),
            rate(1),
            rate(2),
            balanced
        );
    }

    // Per-class distributions. An aggregate AUC says a feature is informative;
    // only the distributions say whether it is measuring what it claims.
    if opts.per_record {
        println!(
            "\n{:<16} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "feature", "class", "n", "p10", "p25", "median", "p90"
        );
        let probes: [Probe; 9] = [
            ("score", |r| r.score),
            ("kurtosis", |r| r.features.kurtosis),
            ("kurtosis_rel", |r| r.features.kurtosis_rel),
            ("hf_ratio", |r| r.features.hf_ratio),
            ("base_ratio", |r| r.features.base_ratio),
            ("qrs_ratio", |r| r.features.qrs_ratio),
            ("qrs_ratio_rel", |r| r.features.qrs_ratio_rel),
            ("p2p_rel", |r| r.features.p2p_rel),
            ("flat_frac", |r| r.features.flat_frac),
        ];
        for (name, f) in probes.iter() {
            for c in 1..=3u8 {
                let mut v: Vec<f32> = rows.iter().filter(|r| r.truth == c).map(|r| f(r)).collect();
                if v.is_empty() {
                    continue;
                }
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
                println!(
                    "{:<16} {:>5} {:>9} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
                    name,
                    c,
                    v.len(),
                    q(0.10),
                    q(0.25),
                    q(0.50),
                    q(0.90)
                );
            }
        }
    }

    println!(
        "\n{:<10} {:>9} {:>9} {:>9} {:>12}",
        "record", "hours", "class3 %", "AUC 3v1", "mean score"
    );
    for (name, v) in per_record.iter() {
        let refs: Vec<&Second> = v.iter().collect();
        let c3 = v.iter().filter(|r| r.truth == 3).count();
        println!(
            "{:<10} {:>9.1} {:>9.2} {:>9.4} {:>12.3}",
            name,
            v.len() as f64 / 3600.0,
            100.0 * c3 as f64 / v.len().max(1) as f64,
            auc(&refs, q3, |r| -r.score),
            v.iter().map(|r| r.score as f64).sum::<f64>() / v.len().max(1) as f64
        );
    }
    Ok(())
}
