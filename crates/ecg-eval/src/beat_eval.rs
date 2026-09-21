//! Beat-classification evaluation, ANSI/AAMI EC57 style.
//!
//! Two views, kept apart on purpose:
//!
//! * **Per-detector.** Each binary detector has its own ROC and its own
//!   operating point, because each answers a different question at a different
//!   clinical cost. This is the view that decides thresholds.
//! * **Arbitrated label.** One beat has one AAMI class and the field reports a
//!   confusion matrix over them, so the bank's arbitration is scored too.
//!
//! Detection and classification are also kept apart. A reference beat our
//! detector never found is a detection failure, not a misclassification, and
//! counting it as one would blame the wrong stage.

use crate::manifest::RecordEntry;
use crate::Opts;
use ecg_beats::{BeatAnalyzer, BeatClass, BeatFeatures, BeatVerdict};
use ecg_pipeline::{ChannelOutput, ChannelPipeline, Preprocessor};
use ecg_qrs::QrsEvent;
use ecg_quality::{Quality, QualityMonitor};
use ecg_wfdb::{read_signal, AnnotationFile, Header};
use rayon::prelude::*;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Aami {
    N,
    S,
    V,
    F,
    Q,
}

/// WFDB beat symbol to AAMI class (EC57 Table 1).
pub fn aami(symbol: char) -> Option<Aami> {
    Some(match symbol {
        'N' | 'L' | 'R' | 'e' | 'j' => Aami::N,
        'A' | 'a' | 'J' | 'S' => Aami::S,
        'V' | 'E' => Aami::V,
        'F' => Aami::F,
        '/' | 'f' | 'Q' => Aami::Q,
        _ => return None,
    })
}

/// Records excluded from beat classification by EC57 because they are paced.
const PACED: [&str; 4] = ["102", "104", "107", "217"];

pub fn is_paced(entry: &RecordEntry) -> bool {
    entry.source == "mitdb" && PACED.contains(&entry.record.as_str())
}

/// One matched beat: what it was, and what we said.
#[derive(Debug, Clone, Copy)]
pub struct Scored {
    pub truth: Aami,
    pub verdict: BeatVerdict,
}

pub struct BeatRecord {
    pub source: String,
    pub record: String,
    pub scored: Vec<Scored>,
    /// Reference beats our detector never found.
    pub missed: u64,
    /// Detections with no reference beat.
    pub spurious: u64,
    pub error: Option<String>,
}

fn ann_ext(source: &str) -> &'static str {
    if source == "afdb" {
        "qrs"
    } else {
        "atr"
    }
}

pub fn analyse(entry: &RecordEntry, opts: &Opts) -> BeatRecord {
    let mut r = BeatRecord {
        source: entry.source.clone(),
        record: entry.record.clone(),
        scored: Vec::new(),
        missed: 0,
        spurious: 0,
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
    let lead = opts.lead.min(hdr.n_sig.saturating_sub(1));
    let sig = match read_signal(&hdr, lead, 0, hdr.n_samples) {
        Ok(s) => s,
        Err(e) => {
            r.error = Some(err(e.to_string()));
            return r;
        }
    };
    let ann = match AnnotationFile::read(Path::new(&entry.ann_path(ann_ext(&entry.source)))) {
        Ok(a) => a,
        Err(e) => {
            r.error = Some(err(e.to_string()));
            return r;
        }
    };
    // WFDB permits an annotation before the first sample - INCART record I35
    // opens with one at -15 - and such a beat cannot be measured because its
    // window is not in the record. Dropped here rather than clamped: a beat
    // whose morphology was never recorded should not be scored as one we failed
    // to classify.
    let reference: Vec<(i64, Aami)> = ann
        .annotations
        .iter()
        .filter(|a| a.sample >= 0)
        .filter_map(|a| aami(a.symbol).map(|c| (a.sample, c)))
        .collect();

    let fs = hdr.fs;
    let cfg = crate::qrs_eval::config_from(opts, fs);
    let use_reference_beats = matches!(opts.get_str("beats"), Some("reference") | Some("ref"));

    let verdicts: Vec<BeatVerdict> = if use_reference_beats {
        // Classification isolated from detection: the analyser is driven at the
        // reference positions, so every error below is a classification error.
        //
        // The front end still runs. Feeding the raw signal here instead would
        // not be "isolating" anything - the morphology features are defined on
        // the filtered taps, and measuring QRS width on an unfiltered trace
        // makes that feature meaningless rather than neutral.
        let mut pre = Preprocessor::new(cfg.preprocess);
        let mut qual = QualityMonitor::new(cfg.quality);
        let mut an = BeatAnalyzer::new(cfg.beats);
        let bank = cfg.bank;
        let mut out = Vec::with_capacity(reference.len());
        let mut next = 0usize;
        for (i, &x) in sig.iter().enumerate() {
            let b = pre.process(x);
            let q = qual.process(b.raw, b.clean, b.baseline, b.hf, b.qrs, b.saturated);
            an.push_sample(b.clean, b.qrs, q.level(&cfg.quality) != Quality::Unusable);
            // `<=` rather than `==`: an equality test stalls permanently the
            // moment a position is passed for any reason, and a stalled loop
            // silently produces a record with no verdicts at all.
            while next < reference.len() && reference[next].0 <= i as i64 {
                let ev = QrsEvent {
                    sample: i as u64,
                    amplitude: 0.0,
                    energy: 0.0,
                    margin: 1.0,
                    recovered: false,
                };
                if let Some(obs) = an.push_beat(&ev) {
                    out.push(bank.classify(&obs));
                }
                next += 1;
            }
        }
        out
    } else {
        let mut pipe = ChannelPipeline::new(cfg);
        let mut o = ChannelOutput::default();
        let block = ((fs * 0.25) as usize).max(1);
        let mut out = Vec::new();
        for chunk in sig.chunks(block) {
            o.clear();
            pipe.push(chunk, &mut o);
            out.extend_from_slice(&o.classes);
        }
        out
    };

    // Match verdicts to reference beats in time order.
    let tol = (opts.tol_ms * fs / 1000.0).round() as i64;
    let (mut i, mut j) = (0usize, 0usize);
    while i < reference.len() && j < verdicts.len() {
        let d = verdicts[j].sample as i64 - reference[i].0;
        if d < -tol {
            r.spurious += 1;
            j += 1;
        } else if d > tol {
            r.missed += 1;
            i += 1;
        } else {
            r.scored.push(Scored {
                truth: reference[i].1,
                verdict: verdicts[j],
            });
            i += 1;
            j += 1;
        }
    }
    r.missed += (reference.len() - i) as u64;
    r.spurious += (verdicts.len() - j) as u64;
    r
}

/// Reference class by predicted class.
#[derive(Debug, Default, Clone, Copy)]
pub struct Confusion {
    /// `[truth][predicted]`, predicted indexed N, S, V, Unknown.
    pub m: [[u64; 4]; 5],
}

fn truth_index(a: Aami) -> usize {
    match a {
        Aami::N => 0,
        Aami::S => 1,
        Aami::V => 2,
        Aami::F => 3,
        Aami::Q => 4,
    }
}

fn pred_index(c: BeatClass) -> usize {
    match c {
        BeatClass::N => 0,
        BeatClass::S => 1,
        BeatClass::V => 2,
        BeatClass::Unknown => 3,
    }
}

impl Confusion {
    pub fn add(&mut self, s: &Scored) {
        self.m[truth_index(s.truth)][pred_index(s.verdict.class)] += 1;
    }

    pub fn merge(&mut self, o: &Confusion) {
        for (a, b) in self.m.iter_mut().flatten().zip(o.m.iter().flatten()) {
            *a += b;
        }
    }

    /// Sensitivity and positive predictivity for one class, over the N/S/V
    /// population only.
    ///
    /// Fusion and paced/unclassifiable beats are left out of both numerator and
    /// denominator. EC57 permits counting a fusion beat either way, and the
    /// inter-patient literature this will be compared against excludes them; the
    /// full matrix is printed so the choice can be undone by the reader.
    pub fn class_metrics(&self, class: usize) -> (f64, f64) {
        let tp = self.m[class][class] as f64;
        let actual: f64 = (0..3).map(|p| self.m[class][p] as f64).sum();
        let predicted: f64 = (0..3).map(|t| self.m[t][class] as f64).sum();
        let se = if actual > 0.0 { tp / actual } else { f64::NAN };
        let pp = if predicted > 0.0 {
            tp / predicted
        } else {
            f64::NAN
        };
        (se, pp)
    }

    pub fn coverage(&self) -> f64 {
        let unknown: f64 = (0..3).map(|t| self.m[t][3] as f64).sum();
        let total: f64 = (0..3)
            .flat_map(|t| (0..4).map(move |p| (t, p)))
            .map(|(t, p)| self.m[t][p] as f64)
            .sum();
        if total > 0.0 {
            1.0 - unknown / total
        } else {
            f64::NAN
        }
    }
}

/// Mann-Whitney AUC for one detector against one positive class.
fn auc(scored: &[Scored], positive: Aami, score: impl Fn(&BeatVerdict) -> f32) -> f64 {
    let mut all: Vec<(f32, bool)> = Vec::with_capacity(scored.len());
    for s in scored {
        // Only N, S and V take part, matching `class_metrics`.
        if !matches!(s.truth, Aami::N | Aami::S | Aami::V) {
            continue;
        }
        all.push((score(&s.verdict), s.truth == positive));
    }
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

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let mut entries = opts.select()?;
    let keep_paced = opts.raw.iter().any(|(k, _)| k == "include-paced");
    let dropped: Vec<String> = entries
        .iter()
        .filter(|e| !keep_paced && is_paced(e))
        .map(|e| format!("{}/{}", e.source, e.record))
        .collect();
    if !keep_paced {
        entries.retain(|e| !is_paced(e));
    }
    if entries.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }
    eprintln!(
        "beat classification: {} records, beats={}",
        entries.len(),
        opts.get_str("beats").unwrap_or("detected")
    );

    let results: Vec<BeatRecord> = entries.par_iter().map(|e| analyse(e, opts)).collect();

    let mut total = Confusion::default();
    let mut all: Vec<Scored> = Vec::new();
    let (mut missed, mut spurious) = (0u64, 0u64);
    let mut per_record: Vec<(String, Confusion, u64)> = Vec::new();
    for r in &results {
        if r.error.is_some() {
            continue;
        }
        let mut c = Confusion::default();
        for s in &r.scored {
            c.add(s);
        }
        total.merge(&c);
        per_record.push((format!("{}/{}", r.source, r.record), c, r.missed));
        all.extend_from_slice(&r.scored);
        missed += r.missed;
        spurious += r.spurious;
    }

    if opts.per_record {
        println!(
            "\n{:<14} {:>8} {:>7} {:>7} {:>7} {:>9} {:>9} {:>9} {:>9}",
            "record", "beats", "unseen", "S", "V", "S Se %", "S +P %", "V Se %", "V +P %"
        );
        // Worst first: a pooled figure hides the record that is actually broken.
        per_record.sort_by(|a, b| {
            let f = |c: &Confusion| -> f64 {
                let (vs, vp) = c.class_metrics(2);
                let (ss, sp) = c.class_metrics(1);
                [vs, vp, ss, sp]
                    .iter()
                    .filter(|x| x.is_finite())
                    .sum::<f64>()
            };
            b.2.cmp(&a.2).then_with(|| {
                f(&a.1)
                    .partial_cmp(&f(&b.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        for (name, c, unseen) in &per_record {
            let n: u64 = c.m.iter().flatten().sum();
            let (ss, sp) = c.class_metrics(1);
            let (vs, vp) = c.class_metrics(2);
            println!(
                "{:<14} {:>8} {:>7} {:>7} {:>7} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
                name,
                n,
                unseen,
                c.m[1].iter().sum::<u64>(),
                c.m[2].iter().sum::<u64>(),
                100.0 * ss,
                100.0 * sp,
                100.0 * vs,
                100.0 * vp
            );
        }
    }

    println!("\n── beat classification (AAMI EC57) ───────────────────────────");
    if !dropped.is_empty() {
        println!(
            "excluded as paced ({}): {}",
            dropped.len(),
            dropped.join(" ")
        );
    }
    println!(
        "records          {}",
        results.iter().filter(|r| r.error.is_none()).count()
    );
    println!("beats matched    {}", all.len());
    println!("detection        {missed} reference beats missed, {spurious} spurious detections");
    println!(
        "coverage         {:.3} % of N/S/V beats classified",
        100.0 * total.coverage()
    );

    println!("\nconfusion (rows = reference, columns = reported):");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10}",
        "", "N", "S", "V", "unknown"
    );
    for (name, t) in [("N", 0), ("S", 1), ("V", 2), ("F", 3), ("Q", 4)] {
        println!(
            "{:>8} {:>10} {:>10} {:>10} {:>10}",
            name, total.m[t][0], total.m[t][1], total.m[t][2], total.m[t][3]
        );
    }

    println!("\n{:>16} {:>10} {:>10}", "class", "Se %", "+P %");
    for (name, idx) in [("S (SVEB)", 1), ("V (VEB)", 2)] {
        let (se, pp) = total.class_metrics(idx);
        println!("{:>16} {:>10.3} {:>10.3}", name, 100.0 * se, 100.0 * pp);
    }

    println!("\nper-detector ROC (threshold-independent):");
    println!(
        "{:>18} {:>10}",
        "ventricular",
        format!("{:.4}", auc(&all, Aami::V, |v| v.p_ventricular))
    );
    println!(
        "{:>18} {:>10}",
        "supraventricular",
        format!("{:.4}", auc(&all, Aami::S, |v| v.p_supraventricular))
    );

    let failed: Vec<&BeatRecord> = results.iter().filter(|r| r.error.is_some()).collect();
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

/// Features of every matched beat, for fitting. Used by `fit-beats`.
pub fn collect(opts: &Opts) -> std::io::Result<Vec<(BeatFeatures, Aami, String)>> {
    opts.install_thread_pool();
    let mut entries = opts.select()?;
    entries.retain(|e| !is_paced(e));
    let results: Vec<BeatRecord> = entries.par_iter().map(|e| analyse(e, opts)).collect();
    let mut out = Vec::new();
    for r in &results {
        let key = format!("{}/{}", r.source, r.record);
        for s in &r.scored {
            if s.verdict.class == BeatClass::Unknown {
                continue;
            }
            out.push((s.verdict.features, s.truth, key.clone()));
        }
    }
    Ok(out)
}
