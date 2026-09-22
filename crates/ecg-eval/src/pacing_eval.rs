//! Pacemaker detection, scored where it can be and bounded where it cannot.
//!
//! The corpora label paced beats (`/`) and paced-normal fusions (`f`), so this
//! is one of the two new detectors that *can* be scored. What it cannot be is
//! fitted: exactly one record in the training zone contains paced beats, so
//! every threshold in the rule is argued rather than tuned. See
//! [`ecg_beats::Cluster::paced`].
//!
//! Two figures matter and they are not symmetric. On records that contain
//! pacing, sensitivity and precision. On records that do not - which is almost
//! everybody - the false-call rate, because this detector will run on them all.

use crate::manifest::RecordEntry;
use crate::Opts;
use ecg_pipeline::{ChannelOutput, ChannelPipeline};
use ecg_wfdb::{read_signal, AnnotationFile, Header};
use rayon::prelude::*;

#[derive(Debug, Default, Clone)]
pub struct PacingScore {
    pub record: String,
    /// Beats annotated `/` or `f`.
    pub reference: usize,
    /// Beats the engine assigned to a morphology it called paced.
    pub called: usize,
    pub tp: usize,
    pub fp: usize,
    /// Beats matched to a reference annotation of any kind.
    pub judged: usize,
    /// Morphologies called paced, and the share of beats they hold.
    pub clusters: usize,
    pub share: f32,
}

pub fn analyse(entry: &RecordEntry, opts: &Opts) -> std::io::Result<PacingScore> {
    let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    let hdr = Header::read(&entry.hea_path()).map_err(|e| err(e.to_string()))?;
    let ann = AnnotationFile::read(std::path::Path::new(
        &entry.ann_path(crate::beat_eval::ann_ext(&entry.source)),
    ))
    .map_err(|e| err(e.to_string()))?;
    let paced: Vec<i64> = ann
        .annotations
        .iter()
        .filter(|a| a.sample >= 0 && (a.symbol == '/' || a.symbol == 'f'))
        .map(|a| a.sample)
        .collect();

    let lead = opts.lead.min(hdr.n_sig - 1);
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples).map_err(|e| err(e.to_string()))?;
    let cfg = crate::qrs_eval::config_from(opts, hdr.fs);
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let mut assigned: Vec<(u64, u32)> = Vec::new();
    for c in sig.chunks((hdr.fs * 0.25) as usize) {
        out.clear();
        pipe.push(c, &mut out);
        for v in &out.classes {
            if v.cluster != 0 {
                assigned.push((v.sample, v.cluster));
            }
        }
    }
    let (share, ids): (f32, Vec<u32>) = match pipe.pacing() {
        Some((s, cs)) => (s, cs.iter().map(|c| c.id).collect()),
        None => (0.0, Vec::new()),
    };

    let w = (0.15 * hdr.fs) as i64;
    let mut s = PacingScore {
        record: format!("{}/{}", entry.source, entry.record),
        reference: paced.len(),
        judged: assigned.len(),
        clusters: ids.len(),
        share,
        ..Default::default()
    };
    // A binary search would be tidier; the paced lists are short and the beat
    // lists are not sorted against them in lockstep because cluster assignment
    // can lag, so the straightforward scan is the honest one.
    for (sample, cluster) in &assigned {
        if !ids.contains(cluster) {
            continue;
        }
        s.called += 1;
        let t = *sample as i64;
        if paced
            .binary_search_by(|r| {
                if (r - t).abs() <= w {
                    std::cmp::Ordering::Equal
                } else {
                    r.cmp(&t)
                }
            })
            .is_ok()
        {
            s.tp += 1;
        } else {
            s.fp += 1;
        }
    }
    Ok(s)
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    if entries.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }
    eprintln!("pacemaker detection: {} records", entries.len());
    let all: Vec<PacingScore> = entries
        .par_iter()
        .filter_map(|e| analyse(e, opts).ok())
        .collect();
    if all.is_empty() {
        return Ok(());
    }

    let (paced, plain): (Vec<&PacingScore>, Vec<&PacingScore>) =
        all.iter().partition(|s| s.reference > 0);

    println!("\n── pacemaker detection ───────────────────────────────────────");
    if !paced.is_empty() {
        println!("\nrecords that contain pacing:");
        println!(
            "{:<16} {:>10} {:>9} {:>8} {:>8} {:>8} {:>8} {:>7}",
            "record", "reference", "called", "TP", "FP", "Se %", "+P %", "share"
        );
        let (mut r, mut c, mut tp, mut fp) = (0usize, 0usize, 0usize, 0usize);
        for s in &paced {
            r += s.reference;
            c += s.called;
            tp += s.tp;
            fp += s.fp;
            println!(
                "{:<16} {:>10} {:>9} {:>8} {:>8} {:>8.2} {:>8.2} {:>6.1}%",
                s.record,
                s.reference,
                s.called,
                s.tp,
                s.fp,
                100.0 * s.tp as f64 / s.reference.max(1) as f64,
                100.0 * s.tp as f64 / s.called.max(1) as f64,
                100.0 * s.share,
            );
        }
        println!(
            "{:<16} {:>10} {:>9} {:>8} {:>8} {:>8.2} {:>8.2}",
            "pooled",
            r,
            c,
            tp,
            fp,
            100.0 * tp as f64 / r.max(1) as f64,
            100.0 * tp as f64 / c.max(1) as f64
        );
    }
    if !plain.is_empty() {
        let judged: usize = plain.iter().map(|s| s.judged).sum();
        let called: usize = plain.iter().map(|s| s.called).sum();
        let touched = plain.iter().filter(|s| s.called > 0).count();
        println!("\nrecords that contain none — the false-call bound:");
        println!("  records                    {}", plain.len());
        println!("  beats judged               {judged}");
        println!(
            "  beats called paced         {called}  ({:.4} %)",
            100.0 * called as f64 / judged.max(1) as f64
        );
        println!("  records with any call      {touched} of {}", plain.len());
        if opts.per_record && touched > 0 {
            for s in plain.iter().filter(|s| s.called > 0) {
                println!(
                    "    {:<16} {} beats in {} morphologies ({:.1} % of the record)",
                    s.record,
                    s.called,
                    s.clusters,
                    100.0 * s.share
                );
            }
        }
    }
    Ok(())
}
