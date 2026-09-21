//! Delineation evaluation against manual wave annotations.
//!
//! Scored the way the field scores it: for each reference mark, the signed error
//! of the corresponding detected mark, reported as mean ± standard deviation in
//! milliseconds, alongside how often the wave was found at all. The CSE working
//! party's tolerances — the spread between expert annotators on the same
//! signals — are printed beside them, because "how close is close enough" is not
//! a question this project gets to answer for itself.
//!
//! # Corpora
//!
//! * **LUDB** — 200 records, twelve leads, every beat delineated. Split by
//!   subject into a development and a sealed half, recorded in the manifest.
//!   This is where anything gets fitted.
//! * **QT** — 105 records, a different population and a different annotator,
//!   about thirty beats marked per record. Never used to fit anything here, so
//!   it is the independent check.

use crate::manifest::RecordEntry;
use crate::Opts;
use ecg_pipeline::{ChannelOutput, ChannelPipeline};
use ecg_wfdb::{read_signal, AnnotationFile, Header};
use rayon::prelude::*;
use std::path::Path;

/// The marks a delineator is judged on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mark {
    POnset,
    PPeak,
    POffset,
    QrsOnset,
    QrsOffset,
    TPeak,
    TOffset,
}

impl Mark {
    pub const ALL: [Mark; 7] = [
        Mark::POnset,
        Mark::PPeak,
        Mark::POffset,
        Mark::QrsOnset,
        Mark::QrsOffset,
        Mark::TPeak,
        Mark::TOffset,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Mark::POnset => "P onset",
            Mark::PPeak => "P peak",
            Mark::POffset => "P offset",
            Mark::QrsOnset => "QRS onset",
            Mark::QrsOffset => "QRS offset",
            Mark::TPeak => "T peak",
            Mark::TOffset => "T offset",
        }
    }
    /// CSE working-party two-standard-deviation tolerance, milliseconds.
    /// These are the spread *between expert annotators*, not a target.
    pub fn tolerance_ms(self) -> f64 {
        match self {
            Mark::POnset => 10.2,
            Mark::PPeak => 10.2,
            Mark::POffset => 12.7,
            Mark::QrsOnset => 6.5,
            Mark::QrsOffset => 11.6,
            Mark::TPeak => 30.6,
            Mark::TOffset => 30.6,
        }
    }
}

/// Reference marks parsed from a `( peak )` annotation stream.
fn reference(ann: &AnnotationFile) -> Vec<(Mark, i64)> {
    let mut out = Vec::new();
    let a = &ann.annotations;
    let mut i = 0;
    while i < a.len() {
        if a[i].symbol == '(' {
            // The WFDB convention is onset, peak, offset in sequence; the peak
            // symbol names the wave.
            if let (Some(peak), Some(off)) = (a.get(i + 1), a.get(i + 2)) {
                if off.symbol == ')' && a[i].sample >= 0 {
                    let marks = match peak.symbol {
                        'p' => Some((Mark::POnset, Mark::PPeak, Mark::POffset)),
                        'N' | 'A' | 'V' | 'B' | 'Q' => {
                            Some((Mark::QrsOnset, Mark::QrsOnset, Mark::QrsOffset))
                        }
                        't' => Some((Mark::TPeak, Mark::TPeak, Mark::TOffset)),
                        _ => None,
                    };
                    if let Some((on, pk, of)) = marks {
                        match peak.symbol {
                            'p' => {
                                out.push((on, a[i].sample));
                                out.push((pk, peak.sample));
                                out.push((of, off.sample));
                            }
                            't' => {
                                // The QT and LUDB corpora do not mark a T onset.
                                out.push((pk, peak.sample));
                                out.push((of, off.sample));
                            }
                            _ => {
                                out.push((Mark::QrsOnset, a[i].sample));
                                out.push((Mark::QrsOffset, off.sample));
                            }
                        }
                    }
                    i += 3;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

#[derive(Debug, Default, Clone)]
pub struct MarkScore {
    /// Signed errors, detected minus reference, in milliseconds.
    pub errors: Vec<f32>,
    /// Reference marks with no detected counterpart within the search radius.
    pub missed: u64,
}

impl MarkScore {
    pub fn sensitivity(&self) -> f64 {
        let n = self.errors.len() as f64;
        n / (n + self.missed as f64).max(1.0)
    }
    pub fn mean(&self) -> f64 {
        if self.errors.is_empty() {
            return f64::NAN;
        }
        self.errors.iter().map(|&e| e as f64).sum::<f64>() / self.errors.len() as f64
    }
    pub fn sd(&self) -> f64 {
        if self.errors.len() < 2 {
            return f64::NAN;
        }
        let m = self.mean();
        (self
            .errors
            .iter()
            .map(|&e| (e as f64 - m).powi(2))
            .sum::<f64>()
            / (self.errors.len() - 1) as f64)
            .sqrt()
    }
    /// Median signed error, and the share of marks inside the tolerance the
    /// field's own annotators disagree by. A mean and a standard deviation over
    /// a mixture of good marks and gross ones describe neither population.
    pub fn median(&self) -> f64 {
        if self.errors.is_empty() {
            return f64::NAN;
        }
        let mut v: Vec<f64> = self.errors.iter().map(|&e| e as f64).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    pub fn within(&self, tol: f64) -> f64 {
        if self.errors.is_empty() {
            return f64::NAN;
        }
        self.errors
            .iter()
            .filter(|e| (e.abs() as f64) <= tol)
            .count() as f64
            / self.errors.len() as f64
    }

    /// 95th percentile of |error|: where the tail sits, which a mean and a
    /// standard deviation over a mixture of good marks and gross ones hide.
    pub fn p95(&self) -> f64 {
        if self.errors.is_empty() {
            return f64::NAN;
        }
        let mut v: Vec<f64> = self.errors.iter().map(|&e| (e as f64).abs()).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(0.95 * (v.len() - 1) as f64) as usize]
    }

    fn merge(&mut self, o: &MarkScore) {
        self.errors.extend_from_slice(&o.errors);
        self.missed += o.missed;
    }
}

/// Annotation file extension carrying the delineation, per corpus.
fn delineation_ext(entry: &RecordEntry, opts: &Opts) -> String {
    if let Some(e) = opts.get_str("ann") {
        return e.to_string();
    }
    match entry.source.as_str() {
        // One file per lead; lead II is what a chest patch approximates.
        "ludb" => "ii".to_string(),
        // Manual cardiologist annotations, as distinct from the automatic ones.
        _ => "q1c".to_string(),
    }
}

/// Which signal lead corresponds to the annotation being scored.
fn lead_for(entry: &RecordEntry, opts: &Opts) -> usize {
    if let Some(l) = opts.get_usize("lead") {
        return l;
    }
    if entry.source == "ludb" {
        1 // lead ii
    } else {
        0
    }
}

pub fn analyse(entry: &RecordEntry, opts: &Opts) -> std::io::Result<[MarkScore; 7]> {
    let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    let hdr = Header::read(&entry.hea_path()).map_err(|e| err(e.to_string()))?;
    let ext = delineation_ext(entry, opts);
    let ann_path = entry.hea_path().with_extension(&ext);
    if !ann_path.exists() {
        return Err(err(format!("no {ext} annotations")));
    }
    let ann = AnnotationFile::read(Path::new(&ann_path)).map_err(|e| err(e.to_string()))?;
    let refs = reference(&ann);
    if refs.is_empty() {
        return Err(err("no delineation marks".into()));
    }

    let lead = lead_for(entry, opts).min(hdr.n_sig.saturating_sub(1));
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples).map_err(|e| err(e.to_string()))?;
    let fs = hdr.fs;
    let cfg = crate::qrs_eval::config_from(opts, fs);
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let block = ((fs * 0.25) as usize).max(1);
    let mut waves = Vec::new();
    for chunk in sig.chunks(block) {
        out.clear();
        pipe.push(chunk, &mut out);
        waves.extend_from_slice(&out.waves);
    }

    // Only the stretch the delineator was actually able to cover is scored.
    // It lags the detector by one beat and needs an interval on each side, so
    // the first beats of a record have no delineation by construction - on
    // LUDB, whose records are ten seconds long, that startup alone accounts for
    // a fifth of every reference mark. Counting it as a miss would measure the
    // corpus, not the method; the interval is stated instead.
    let Some(first) = waves.first().map(|d| d.r) else {
        return Err(err("no delineation produced".into()));
    };
    let last = waves.last().map(|d| d.r).unwrap_or(first);
    let guard = if waves.len() > 2 {
        let mut rr: Vec<u64> = waves.windows(2).map(|w| w[1].r - w[0].r).collect();
        rr.sort_unstable();
        rr[rr.len() / 2] / 2
    } else {
        (fs * 0.5) as u64
    };
    let lo_t = first as i64 - guard as i64;
    let hi_t = (last + guard) as i64;

    // Detected marks, flattened and sorted so each reference can take the
    // nearest one of its own kind.
    let mut detected: Vec<Vec<i64>> = vec![Vec::new(); 7];
    for d in &waves {
        detected[Mark::QrsOnset as usize].push(d.qrs.onset as i64);
        detected[Mark::QrsOffset as usize].push(d.qrs.offset as i64);
        if let Some(p) = d.p {
            detected[Mark::POnset as usize].push(p.onset as i64);
            detected[Mark::PPeak as usize].push(p.peak as i64);
            detected[Mark::POffset as usize].push(p.offset as i64);
        }
        if let Some(t) = d.t {
            detected[Mark::TPeak as usize].push(t.peak as i64);
            detected[Mark::TOffset as usize].push(t.offset as i64);
        }
    }
    for v in detected.iter_mut() {
        v.sort_unstable();
    }

    // A reference mark counts as found when a detected mark of the same kind is
    // within the search radius. Generous on purpose: the question here is the
    // error distribution, and a radius tight enough to exclude gross errors
    // would flatter it by turning them into misses.
    let radius = (0.15 * fs) as i64;
    let to_ms = 1000.0 / fs as f32;
    let mut scores: [MarkScore; 7] = Default::default();
    for (mark, sample) in refs {
        if sample < lo_t || sample > hi_t {
            continue;
        }
        let candidates = &detected[mark as usize];
        let nearest = candidates
            .binary_search(&sample)
            .map(|i| candidates[i])
            .unwrap_or_else(|i| {
                let before = i.checked_sub(1).map(|j| candidates[j]);
                let after = candidates.get(i).copied();
                match (before, after) {
                    (Some(b), Some(a)) => {
                        if (sample - b).abs() <= (a - sample).abs() {
                            b
                        } else {
                            a
                        }
                    }
                    (Some(b), None) => b,
                    (None, Some(a)) => a,
                    (None, None) => i64::MIN / 2,
                }
            });
        let d = nearest - sample;
        if d.abs() <= radius {
            scores[mark as usize].errors.push(d as f32 * to_ms);
        } else {
            scores[mark as usize].missed += 1;
        }
    }
    Ok(scores)
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    if entries.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }
    eprintln!("delineation: {} records", entries.len());
    let per: Vec<[MarkScore; 7]> = entries
        .par_iter()
        .filter_map(|e| analyse(e, opts).ok())
        .collect();
    if per.is_empty() {
        eprintln!("no records produced marks");
        return Ok(());
    }
    let mut total: [MarkScore; 7] = Default::default();
    for p in &per {
        for (t, s) in total.iter_mut().zip(p) {
            t.merge(s);
        }
    }

    println!("\n── delineation ───────────────────────────────────────────────");
    println!("records {}", per.len());
    println!(
        "\n{:<12} {:>8} {:>7} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8}",
        "mark", "found %", "n", "median", "mean ms", "sd ms", "p95 ms", "in tol %", "CSE 2SD"
    );
    for m in Mark::ALL {
        let s = &total[m as usize];
        if s.errors.is_empty() && s.missed == 0 {
            continue;
        }
        println!(
            "{:<12} {:>8.1} {:>7} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>9.1} {:>8.1}",
            m.name(),
            100.0 * s.sensitivity(),
            s.errors.len(),
            s.median(),
            s.mean(),
            s.sd(),
            s.p95(),
            100.0 * s.within(m.tolerance_ms()),
            m.tolerance_ms(),
        );
    }
    Ok(())
}
