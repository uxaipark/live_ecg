//! Per-record error anatomy.
//!
//! Aggregate rates say a record is bad; they never say why. This prints the
//! misses and false alarms with the context that distinguishes the causes that
//! need different fixes: which beat type was missed, whether misses cluster in
//! time (an amplitude or noise episode) or spread evenly (a systematic rule
//! problem), and what the local RR and amplitude were.

use crate::manifest::RecordEntry;
use crate::qrs_eval;
use crate::Opts;
use ecg_pipeline::{ChannelOutput, ChannelPipeline};
use ecg_wfdb::{is_beat_symbol, read_signal, AnnotationFile, Header};
use std::collections::BTreeMap;
use std::path::Path;

pub fn run(opts: &Opts) -> std::io::Result<()> {
    let entries = opts.select()?;
    for e in &entries {
        one(e, opts)?;
    }
    Ok(())
}

fn ann_ext(source: &str) -> &'static str {
    if source == "afdb" {
        "qrs"
    } else {
        "atr"
    }
}

fn one(entry: &RecordEntry, opts: &Opts) -> std::io::Result<()> {
    let hdr = Header::read(&entry.hea_path())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let lead = opts.lead.min(hdr.n_sig - 1);
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let ann = AnnotationFile::read(Path::new(&entry.ann_path(ann_ext(&entry.source))))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

    let fs = hdr.fs;
    let cfg = qrs_eval::config_from(opts, fs);
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let block = ((fs * 0.25) as usize).max(1);
    for chunk in sig.chunks(block) {
        pipe.push(chunk, &mut out);
    }

    let tol = (opts.tol_ms * fs / 1000.0).round() as i64;
    let refs: Vec<(i64, char)> = ann
        .annotations
        .iter()
        .filter(|a| is_beat_symbol(a.symbol))
        .map(|a| (a.sample, a.symbol))
        .collect();
    let det: Vec<i64> = out.beats.iter().map(|e| e.sample as i64).collect();

    // Mark which references were matched, with the same greedy rule as scoring.
    let mut matched = vec![false; refs.len()];
    let mut det_used = vec![false; det.len()];
    let (mut i, mut j) = (0usize, 0usize);
    while i < refs.len() && j < det.len() {
        let d = det[j] - refs[i].0;
        if d < -tol {
            j += 1;
        } else if d > tol {
            i += 1;
        } else {
            matched[i] = true;
            det_used[j] = true;
            i += 1;
            j += 1;
        }
    }

    let by_symbol_total = tally(refs.iter().map(|&(_, s)| s));
    let by_symbol_miss = tally(
        refs.iter()
            .zip(&matched)
            .filter(|(_, &m)| !m)
            .map(|(&(_, s), _)| s),
    );

    println!(
        "\n══ {}/{}  lead {} ({})  fs {} Hz",
        entry.source, entry.record, lead, hdr.signals[lead].description, fs
    );
    println!(
        "   reference {}   detected {}   missed {}   false {}",
        refs.len(),
        det.len(),
        matched.iter().filter(|m| !**m).count(),
        det_used.iter().filter(|u| !**u).count()
    );

    println!("   misses by annotated beat type:");
    for (sym, total) in &by_symbol_total {
        let miss = by_symbol_miss.get(sym).copied().unwrap_or(0);
        if miss > 0 {
            println!(
                "     '{sym}'  {miss:>6} / {total:<6}  ({:.1}%)",
                100.0 * miss as f64 / *total as f64
            );
        }
    }

    // Minute-by-minute clustering: a few bad minutes and an even spread call for
    // completely different fixes.
    let minute = (fs * 60.0) as i64;
    let mut miss_per_min: BTreeMap<i64, u32> = BTreeMap::new();
    let mut ref_per_min: BTreeMap<i64, u32> = BTreeMap::new();
    for (k, &(s, _)) in refs.iter().enumerate() {
        *ref_per_min.entry(s / minute).or_default() += 1;
        if !matched[k] {
            *miss_per_min.entry(s / minute).or_default() += 1;
        }
    }
    let mut fp_per_min: BTreeMap<i64, u32> = BTreeMap::new();
    for (k, &s) in det.iter().enumerate() {
        if !det_used[k] {
            *fp_per_min.entry(s / minute).or_default() += 1;
        }
    }
    let worst: Vec<_> = {
        let mut v: Vec<_> = miss_per_min.iter().map(|(&m, &c)| (m, c)).collect();
        v.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
        v.truncate(8);
        v.sort();
        v
    };
    if !worst.is_empty() {
        println!("   worst minutes (miss / beats / false):");
        for (m, c) in worst {
            println!(
                "     {:>3}:00  {c:>5} / {:<5} {:>5}",
                m,
                ref_per_min.get(&m).copied().unwrap_or(0),
                fp_per_min.get(&m).copied().unwrap_or(0)
            );
        }
    }

    // Local amplitude around missed beats vs matched ones: separates "the beat got
    // small" from "the rule rejected it".
    let w = (fs * 0.1) as usize;
    let amp = |s: i64| -> f32 {
        let a = (s as usize).saturating_sub(w);
        let b = ((s as usize) + w).min(sig.len());
        if b <= a {
            return 0.0;
        }
        let sl = &sig[a..b];
        sl.iter().cloned().fold(f32::MIN, f32::max) - sl.iter().cloned().fold(f32::MAX, f32::min)
    };
    let (mut am, mut nm, mut ah, mut nh) = (0.0f64, 0u32, 0.0f64, 0u32);
    for (k, &(s, _)) in refs.iter().enumerate() {
        if matched[k] {
            ah += amp(s) as f64;
            nh += 1;
        } else {
            am += amp(s) as f64;
            nm += 1;
        }
    }
    if nm > 0 && nh > 0 {
        println!(
            "   peak-to-peak around beats:  matched {:.3} mV   missed {:.3} mV   ratio {:.2}",
            ah / nh as f64,
            am / nm as f64,
            (am / nm as f64) / (ah / nh as f64)
        );
    }

    if opts.per_record {
        println!("   first 25 misses (time, symbol, prev RR ms):");
        let mut shown = 0;
        for (k, &(s, sym)) in refs.iter().enumerate() {
            if matched[k] || shown >= 25 {
                continue;
            }
            let prev_rr = if k > 0 {
                (s - refs[k - 1].0) as f64 * 1000.0 / fs
            } else {
                f64::NAN
            };
            println!(
                "     {:>8.2}s  '{sym}'  rr {:>6.0} ms  p2p {:.3} mV",
                s as f64 / fs,
                prev_rr,
                amp(s)
            );
            shown += 1;
        }
    }
    Ok(())
}

/// Per-second trace of the detector's adaptive state over a time range.
pub fn trace(opts: &Opts) -> std::io::Result<()> {
    let entries = opts.select()?;
    let entry = entries
        .first()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no record selected"))?;
    let hdr = Header::read(&entry.hea_path())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let lead = opts.lead.min(hdr.n_sig - 1);
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let fs = hdr.fs;
    let from = opts.get_usize("from-sec").unwrap_or(0);
    let rows = opts.get_usize("rows").unwrap_or(20);

    let cfg = qrs_eval::config_from(opts, fs);
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let spp = fs as usize;
    println!(
        "{:>8} {:>6} {:>11} {:>11} {:>11} {:>8} {:>9} {:>7}",
        "sec", "beats", "threshold", "spk", "npk", "spkscale", "rr_stable", "inpeak"
    );
    for s in 0..(sig.len() / spp) {
        out.clear();
        pipe.push(&sig[s * spp..(s + 1) * spp], &mut out);
        if s >= from && s < from + rows {
            let st = pipe.detector_state();
            println!(
                "{:>8} {:>6} {:>11.3e} {:>11.3e} {:>11.3e} {:>8.3} {:>9.1} {:>7}",
                s,
                out.beats.len(),
                st.threshold,
                st.spk,
                st.npk,
                st.spk_scale,
                st.rr_stable,
                st.in_peak
            );
        }
        if s >= from + rows {
            break;
        }
    }
    Ok(())
}

fn tally(it: impl Iterator<Item = char>) -> BTreeMap<char, u32> {
    let mut m = BTreeMap::new();
    for c in it {
        *m.entry(c).or_insert(0u32) += 1;
    }
    m
}
