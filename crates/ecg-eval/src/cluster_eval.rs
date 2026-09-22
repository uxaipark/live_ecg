//! Evaluation of the morphology review queue.
//!
//! The per-beat and per-episode figures answer "how often is the engine right".
//! That is the wrong question for a finding whose precision is bounded by its
//! prevalence: ventricular runs occupy 0.034 % of the long-term corpus, and at
//! any specificity a single-lead classifier can reach, the false positives
//! outnumber the true ones.
//!
//! The question a review queue has to answer instead is **how much of a
//! recording's ventricular activity a reviewer can confirm, and how many
//! decisions it costs them**. So that is what is measured here: rank the
//! morphologies by score, have a notional reviewer label the top `k`, and report
//! the beat-level sensitivity and precision that results.
//!
//! A reviewer labelling clusters is idealised as labelling them *correctly* -
//! which is the assumption the whole design rests on, and it is the reasonable
//! one: they are shown one representative complex per cluster with its width,
//! its atrial evidence and its prematurity, which is the same evidence a
//! cardiologist reads a strip with.

use crate::beat_eval::{aami, is_paced, Aami};
use crate::manifest::RecordEntry;
use crate::Opts;
use ecg_beats::BeatClass;
use ecg_pipeline::{ChannelOutput, ChannelPipeline};
use ecg_wfdb::{read_signal, AnnotationFile, Header};
use rayon::prelude::*;
use std::collections::HashMap;

/// One record's clusters, each with what it actually contained.
#[derive(Debug, Default, Clone)]
pub struct RecordClusters {
    pub record: String,
    /// Clusters ranked by ventricular score, each as `(score, per-class counts)`
    /// where the classes are N, S, V, F.
    pub ranked: Vec<(f32, [u64; 4])>,
    /// Reference beats of each class that were matched to a detected beat.
    pub totals: [u64; 4],
    /// Beats the classifier would not judge, and so never joined a cluster.
    pub unjudged: u64,
    pub merges: u64,
}

fn class_index(a: Aami) -> Option<usize> {
    Some(match a {
        Aami::N => 0,
        Aami::S => 1,
        Aami::V => 2,
        Aami::F => 3,
        Aami::Q => return None,
    })
}

pub fn analyse(entry: &RecordEntry, opts: &Opts) -> std::io::Result<RecordClusters> {
    let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    let hdr = Header::read(&entry.hea_path()).map_err(|e| err(e.to_string()))?;
    let ann = AnnotationFile::read(std::path::Path::new(
        &entry.ann_path(crate::beat_eval::ann_ext(&entry.source)),
    ))
    .map_err(|e| err(e.to_string()))?;
    let reference: Vec<(i64, Aami)> = ann
        .annotations
        .iter()
        .filter(|a| a.sample >= 0)
        .filter_map(|a| aami(a.symbol).map(|c| (a.sample, c)))
        .collect();
    if reference.is_empty() {
        return Err(err("no beat annotations".into()));
    }

    let lead = opts.lead.min(hdr.n_sig - 1);
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples).map_err(|e| err(e.to_string()))?;
    let cfg = crate::qrs_eval::config_from(opts, hdr.fs);
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    // (sample, cluster id) for every beat the classifier judged.
    let mut judged: Vec<(u64, u32)> = Vec::new();
    let mut unjudged = 0u64;
    for c in sig.chunks((hdr.fs * 0.25) as usize) {
        out.clear();
        pipe.push(c, &mut out);
        for v in &out.classes {
            if v.class == BeatClass::Unknown || v.cluster == 0 {
                unjudged += 1;
            } else {
                judged.push((v.sample, v.cluster));
            }
        }
    }

    // Match each judged beat to the nearest reference beat within 150 ms, and
    // credit its class to the cluster it landed in.
    let window = (0.15 * hdr.fs) as i64;
    let mut by_cluster: HashMap<u32, [u64; 4]> = HashMap::new();
    let mut totals = [0u64; 4];
    let mut next = 0usize;
    for (sample, cluster) in &judged {
        let s = *sample as i64;
        while next + 1 < reference.len() && reference[next + 1].0 < s - window {
            next += 1;
        }
        let hit = reference[next..]
            .iter()
            .take(4)
            .find(|(r, _)| (r - s).abs() <= window);
        if let Some((_, class)) = hit {
            if let Some(i) = class_index(*class) {
                by_cluster.entry(*cluster).or_default()[i] += 1;
                totals[i] += 1;
            }
        }
    }

    let bank = pipe.morphology_bank();
    let mut ranked: Vec<(f32, [u64; 4])> = bank
        .clusters()
        .iter()
        .map(|c| {
            (
                c.score(),
                by_cluster.get(&c.id).copied().unwrap_or_default(),
            )
        })
        .filter(|(_, counts)| counts.iter().sum::<u64>() > 0)
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    Ok(RecordClusters {
        record: format!("{}/{}", entry.source, entry.record),
        ranked,
        totals,
        unjudged,
        merges: bank.merges,
    })
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let mut entries = opts.select()?;
    entries.retain(|e| !is_paced(e));
    if entries.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }
    eprintln!("morphology clustering: {} records", entries.len());
    let results: Vec<RecordClusters> = entries
        .par_iter()
        .filter_map(|e| analyse(e, opts).ok())
        .collect();
    if results.is_empty() {
        eprintln!("no records produced clusters");
        return Ok(());
    }

    if opts.per_record {
        println!("\n── clusters per record ───────────────────────────────────────");
        for r in &results {
            println!(
                "\n{}   V beats {}  unjudged {}  merges {}",
                r.record, r.totals[2], r.unjudged, r.merges
            );
            println!(
                "   {:>7} {:>8} {:>8} {:>8} {:>8} {:>8}",
                "score", "beats", "N", "S", "V", "F"
            );
            for (score, c) in r.ranked.iter().take(12) {
                println!(
                    "   {:>7.3} {:>8} {:>8} {:>8} {:>8} {:>8}",
                    score,
                    c.iter().sum::<u64>(),
                    c[0],
                    c[1],
                    c[2],
                    c[3]
                );
            }
        }
    }

    println!("\n── morphology review queue ───────────────────────────────────");
    println!(
        "records {}   clusters per record: median {}, max {}",
        results.len(),
        {
            let mut n: Vec<usize> = results.iter().map(|r| r.ranked.len()).collect();
            n.sort_unstable();
            n[n.len() / 2]
        },
        results.iter().map(|r| r.ranked.len()).max().unwrap_or(0)
    );
    let merged = results.iter().filter(|r| r.merges > 0).count();
    println!(
        "records that reached the cluster cap: {merged} of {}",
        results.len()
    );

    // Purity first, because it is the reviewer's error rate. Labelling a
    // cluster labels every beat in it, so a cluster that is 70 % ventricular
    // makes the reviewer wrong about the other 30 % - and they cannot tell,
    // because they were shown one representative complex.
    let mut pure = 0u64;
    let mut impure = 0u64;
    let mut v_in_pure = 0u64;
    let mut v_total = 0u64;
    for r in &results {
        v_total += r.totals[2];
        for (_, c) in &r.ranked {
            let n: u64 = c.iter().sum();
            let dominant = *c.iter().max().unwrap_or(&0);
            if n >= 3 {
                if dominant * 100 >= n * 95 {
                    pure += 1;
                    if c[2] == dominant {
                        v_in_pure += c[2];
                    }
                } else {
                    impure += 1;
                }
            }
        }
    }
    println!(
        "\nclusters at least 95 % one class: {pure} of {} ({:.1} %)",
        pure + impure,
        100.0 * pure as f64 / (pure + impure).max(1) as f64
    );
    println!(
        "ventricular beats inside a cluster that is predominantly ventricular: {:.1} %",
        100.0 * v_in_pure as f64 / v_total.max(1) as f64
    );

    // A reviewer reads the clusters that score above a bar - not the top k of
    // every record. Taking the top k regardless of score charges them for
    // reading the normal morphologies of every patient who has no ectopy at
    // all, which is not what anyone would do and made the first version of
    // this table report 13 % precision on clusters that were 99 % pure.
    let hours: f64 = results.len() as f64; // placeholder replaced below
    let _ = hours;
    println!(
        "\n{:>8} {:>10} {:>12} {:>9} {:>9} {:>16} {:>18}",
        "bar", "clusters", "V beats", "Se %", "+P %", "clusters/record", "records with none"
    );
    for bar in [0.99f32, 0.95, 0.90, 0.80, 0.60, 0.40] {
        let (mut tp, mut fp, mut total_v, mut n, mut silent) = (0u64, 0u64, 0u64, 0usize, 0usize);
        for r in &results {
            total_v += r.totals[2];
            let mut k = 0usize;
            for (score, counts) in &r.ranked {
                if *score >= bar {
                    k += 1;
                    tp += counts[2];
                    fp += counts[0] + counts[1] + counts[3];
                }
            }
            n += k;
            if k == 0 {
                silent += 1;
            }
        }
        if total_v == 0 {
            continue;
        }
        println!(
            "{:>8.2} {:>10} {:>12} {:>9.2} {:>9.2} {:>16.2} {:>13} / {:<4}",
            bar,
            n,
            tp,
            100.0 * tp as f64 / total_v as f64,
            100.0 * tp as f64 / (tp + fp).max(1) as f64,
            n as f64 / results.len() as f64,
            silent,
            results.len(),
        );
    }

    // And the same question the other way round: how many clusters must be
    // reviewed to reach a given share of the record's ventricular beats?
    println!(
        "\nclusters a reviewer must read to reach a given share of each record's own V beats:"
    );
    println!(
        "{:>10} {:>14} {:>14} {:>14}",
        "share", "median", "p90", "worst"
    );
    for want in [0.5f64, 0.8, 0.9, 0.95, 1.0] {
        let mut need: Vec<usize> = Vec::new();
        for r in &results {
            if r.totals[2] == 0 {
                continue;
            }
            let target = (want * r.totals[2] as f64).ceil() as u64;
            let mut got = 0u64;
            let mut k = 0usize;
            for (_, counts) in &r.ranked {
                k += 1;
                got += counts[2];
                if got >= target {
                    break;
                }
            }
            need.push(if got >= target { k } else { r.ranked.len() });
        }
        if need.is_empty() {
            continue;
        }
        need.sort_unstable();
        println!(
            "{:>9.0}% {:>14} {:>14} {:>14}",
            100.0 * want,
            need[need.len() / 2],
            need[(0.9 * (need.len() - 1) as f64) as usize],
            need[need.len() - 1]
        );
    }
    Ok(())
}
