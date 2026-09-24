//! Why an asystole was not reported.
//!
//! On the only long ambulatory corpus available, the engine reports 14 of the
//! 153 asystoles that the corpus's own beat annotations define. That is the
//! worst sensitivity in the whole evaluation, on the most urgent finding there
//! is, so it is worth knowing which step loses them rather than how many.
//!
//! There are only four ways to lose one, and they are distinguishable:
//!
//! * **A beat was detected inside the silence.** Then our interval never
//!   reaches four seconds and the condition was never a candidate.
//! * **The interval was gated on the electrode.** A flat trace and a detached
//!   lead look alike, and the tie is broken towards the electrode.
//! * **The interval was gated on its energy.** The interval carried something
//!   beat-shaped, so the rule declined to call it quiet.
//! * **Neither**: the bounding beats were missed too, or nothing explains it.
//!
//! # What it found
//!
//! The gates lose none. Not one asystole in 1,961 hours is lost to the energy
//! rule or to the electrode rule, which is the opposite of what a reading of
//! the sensitivity figure alone would suggest.
//!
//! 136 of the 139 are lost to a beat detected inside the silence, and those
//! beats are not noise. None came from search-back, their amplitude is the
//! record's own typical beat amplitude to within a tenth, and there is exactly
//! one of them per silence in a gap three and a half to four and a half times
//! the interval either side of it. 107 of the 153 reference silences are in two
//! records, one of which contains a 515-second stretch holding 1,024 detections
//! and no annotations at all. In the other direction, eight of the nine
//! asystoles reported that the reference lacks fall in spans where the
//! annotator placed no beats either.
//!
//! So the figure is a measurement of the reference. The control says so
//! directly: on MIT-BIH, which is annotated beat by beat, the same census reads
//! 14 of 14 with nothing lost to anything.
//!
//! This does not make the engine right about those intervals - an unannotated
//! stretch is not evidence in either direction, which is the whole point. It
//! makes the 9.2 % a number about the corpus, and it means the fix is to say so
//! rather than to loosen a rule that is losing nothing.

use crate::manifest::RecordEntry;
use crate::Opts;
use ecg_pipeline::{ChannelOutput, ChannelPipeline};
use ecg_wfdb::{is_beat_symbol, read_signal, AnnotationFile, Header};
use rayon::prelude::*;
use std::path::Path;

#[derive(Debug, Default, Clone)]
pub struct Census {
    pub reference: u64,
    pub reported: u64,
    /// Beats were detected inside the silence.
    pub beats_inside: u64,
    /// No interval of ours spans it, and no beat of ours is inside it either:
    /// the bounding beats were missed, so the gap merged into a longer one.
    pub no_interval: u64,
    pub gated_lead: u64,
    pub gated_energy: u64,
    /// An interval of ours spans it, long enough, ungated - and no episode.
    pub unexplained: u64,
    /// How many detections intruded, how many of those came from search-back,
    /// and how big each was against the record's typical beat.
    pub intruders: u64,
    pub recovered: u64,
    pub amp_ratio: Vec<f32>,
    /// The other direction: asystoles we reported that the reference does not
    /// have, split by whether the reference has anything there to disagree
    /// with. A span the annotator left unannotated is not a false positive;
    /// it is an absence of evidence in the only place it matters.
    pub extra: u64,
    pub extra_unannotated: u64,
}

impl Census {
    pub fn merge(&mut self, o: &Census) {
        self.reference += o.reference;
        self.reported += o.reported;
        self.beats_inside += o.beats_inside;
        self.no_interval += o.no_interval;
        self.gated_lead += o.gated_lead;
        self.gated_energy += o.gated_energy;
        self.unexplained += o.unexplained;
        self.intruders += o.intruders;
        self.recovered += o.recovered;
        self.amp_ratio.extend_from_slice(&o.amp_ratio);
        self.extra += o.extra;
        self.extra_unannotated += o.extra_unannotated;
    }
}

pub fn analyse(entry: &RecordEntry, opts: &Opts) -> Option<(String, Census)> {
    let hdr = Header::read(&entry.hea_path()).ok()?;
    let ann = AnnotationFile::read(Path::new(&entry.ann_path("atr"))).ok()?;
    let lead = opts.lead.min(hdr.n_sig.saturating_sub(1));
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples).ok()?;
    let fs = hdr.fs;
    let cfg = crate::qrs_eval::config_from(opts, fs);

    // Reference silences: consecutive annotated beats at least `asystole_ms`
    // apart. The same rule the evaluation scores against.
    let beats: Vec<u64> = ann
        .annotations
        .iter()
        .filter(|a| a.sample >= 0 && is_beat_symbol(a.symbol))
        .map(|a| a.sample as u64)
        .collect();
    let bar = (cfg.rhythm.asystole_ms as f64 * fs / 1000.0) as u64;
    let gaps: Vec<(u64, u64)> = beats
        .windows(2)
        .filter(|w| w[1] - w[0] >= bar)
        .map(|w| (w[0], w[1]))
        .collect();
    if gaps.is_empty() {
        return Some((entry.record.clone(), Census::default()));
    }

    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let block = ((fs * 0.25) as usize).max(1);
    let (mut ours, mut intervals, mut episodes) = (Vec::new(), Vec::new(), Vec::new());
    for chunk in sig.chunks(block) {
        out.clear();
        pipe.push(chunk, &mut out);
        ours.extend(out.beats.iter().copied());
        intervals.extend_from_slice(&out.intervals);
        episodes.extend_from_slice(&out.episodes);
    }
    // What a beat looks like on this channel, so an intruding detection can be
    // compared with one.
    let mut amps: Vec<f32> = ours.iter().map(|b| b.amplitude).collect();
    amps.sort_by(f32::total_cmp);
    let typical = amps.get(amps.len() / 2).copied().unwrap_or(0.0);
    out.clear();
    pipe.finish(&mut out);
    episodes.extend_from_slice(&out.episodes);

    let mut c = Census {
        reference: gaps.len() as u64,
        ..Census::default()
    };
    // The episode is anchored at the beat that closes the interval, and our
    // beat lands a few milliseconds from the annotated one, so the match needs
    // the same tolerance every other comparison here uses. Without it a
    // perfectly matched asystole is counted twice: once as missed and once as
    // invented.
    let tol = (opts.tol_ms * fs / 1000.0) as u64;
    for &(a, b) in &gaps {
        let reported = episodes.iter().any(|e| {
            e.condition == ecg_rhythm::Condition::Asystole && e.end + tol >= a && e.start <= b + tol
        });
        if opts.records.len() == 1 {
            println!(
                "  reference gap {:.1} s .. {:.1} s  ({:.2} s){}",
                a as f64 / fs,
                b as f64 / fs,
                (b - a) as f64 / fs,
                if reported { "  reported" } else { "" }
            );
        }
        if reported {
            c.reported += 1;
            continue;
        }
        // Our detections strictly inside the silence, allowing each bounding
        // beat a tolerance so that the bounding detections themselves do not
        // count as intrusions.
        let intruders: Vec<&ecg_qrs::QrsEvent> = ours
            .iter()
            .filter(|e| e.sample > a + tol && e.sample + tol < b)
            .collect();
        if !intruders.is_empty() {
            if opts.records.len() == 1 {
                let near: Vec<f64> = beats
                    .windows(2)
                    .filter(|w| w[1] + 10 * bar > a && w[0] < b + 10 * bar && w[1] - w[0] < bar)
                    .map(|w| (w[1] - w[0]) as f64 / fs)
                    .collect();
                let neighbour = if near.is_empty() {
                    f64::NAN
                } else {
                    near.iter().sum::<f64>() / near.len() as f64
                };
                println!(
                    "  gap at {:>9.1} s  {:>5.2} s  = {:>4.1} x the {:.2} s around it, \
                     {} detections, |amp| median {:.2} x typical",
                    a as f64 / fs,
                    (b - a) as f64 / fs,
                    (b - a) as f64 / fs / neighbour,
                    neighbour,
                    intruders.len(),
                    {
                        let mut v: Vec<f32> = intruders
                            .iter()
                            .map(|e| (e.amplitude / typical).abs())
                            .collect();
                        v.sort_by(f32::total_cmp);
                        v[v.len() / 2]
                    }
                );
            }
            c.beats_inside += 1;
            c.intruders += intruders.len() as u64;
            c.recovered += intruders.iter().filter(|e| e.recovered).count() as u64;
            for e in &intruders {
                if typical > 0.0 {
                    c.amp_ratio.push(e.amplitude / typical);
                }
            }
            continue;
        }
        // The interval of ours that covers this silence: it closes inside the
        // gap's far end and opens at or before its near end.
        let covering = intervals
            .iter()
            .filter(|s| s.sample > a && s.sample <= b + tol)
            .max_by(|x, y| x.rr_ms.total_cmp(&y.rr_ms));
        match covering {
            None => c.no_interval += 1,
            Some(s) if !s.lead_ok => c.gated_lead += 1,
            Some(s)
                if cfg.rhythm.pause_max_energy > 0.0
                    && s.interval_energy > cfg.rhythm.pause_max_energy =>
            {
                c.gated_energy += 1
            }
            Some(s) if s.rr_ms < cfg.rhythm.asystole_ms => c.beats_inside += 1,
            Some(_) => c.unexplained += 1,
        }
    }
    for e in episodes
        .iter()
        .filter(|e| e.condition == ecg_rhythm::Condition::Asystole)
    {
        if gaps
            .iter()
            .any(|(a, b)| e.end + tol >= *a && e.start <= *b + tol)
        {
            continue;
        }
        c.extra += 1;
        if opts.records.len() == 1 {
            println!(
                "  extra: reported {:.1} s .. {:.1} s",
                e.start as f64 / fs,
                e.end as f64 / fs
            );
        }
        // Beats the annotator placed inside the span we called silent. None at
        // all means the annotator was not looking here, which is not the same
        // as the annotator disagreeing.
        let annotated = beats.iter().filter(|&&s| s > e.start && s < e.end).count();
        if annotated == 0 {
            c.extra_unannotated += 1;
        }
    }
    Some((entry.record.clone(), c))
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    let entries = opts.select()?;
    println!("asystole census: {} records", entries.len());
    let mut rows: Vec<(String, Census)> = entries
        .par_iter()
        .filter_map(|e| analyse(e, opts))
        .filter(|(_, c)| c.reference > 0)
        .collect();
    rows.sort_by_key(|(_, c)| std::cmp::Reverse(c.reference));

    let mut total = Census::default();
    if opts.per_record {
        println!(
            "\n{:<16} {:>6} {:>6} {:>8} {:>8} {:>6} {:>7} {:>6}",
            "record", "ref", "rep", "beats in", "no intvl", "lead", "energy", "??"
        );
    }
    for (name, c) in &rows {
        total.merge(c);
        if opts.per_record {
            println!(
                "{:<16} {:>6} {:>6} {:>8} {:>8} {:>6} {:>7} {:>6}",
                name,
                c.reference,
                c.reported,
                c.beats_inside,
                c.no_interval,
                c.gated_lead,
                c.gated_energy,
                c.unexplained
            );
        }
    }
    let pct = |n: u64| {
        if total.reference == 0 {
            0.0
        } else {
            100.0 * n as f64 / total.reference as f64
        }
    };
    println!("\n── where the asystoles went ──────────────────────────────────");
    println!("reference silences      {:>8}", total.reference);
    for (name, n) in [
        ("reported", total.reported),
        ("a beat was detected inside", total.beats_inside),
        ("no interval covers it", total.no_interval),
        ("gated on the electrode", total.gated_lead),
        ("gated on interval energy", total.gated_energy),
        ("unexplained", total.unexplained),
    ] {
        println!("{:<26}{:>8} {:>7.1} %", name, n, pct(n));
    }
    println!(
        "\nasystoles reported that the reference has not: {}, of which {} fall in \
         a span the annotator left with no beats at all",
        total.extra, total.extra_unannotated
    );
    if !total.amp_ratio.is_empty() {
        let mut v = total.amp_ratio.clone();
        v.sort_by(f32::total_cmp);
        let q = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
        println!(
            "\nthe intruding detections: {} of them, {:.1} % from search-back",
            total.intruders,
            100.0 * total.recovered as f64 / total.intruders.max(1) as f64
        );
        println!(
            "amplitude against the record's own typical beat: \
             p5 {:.2}  p25 {:.2}  median {:.2}  p75 {:.2}  p95 {:.2}",
            q(0.05),
            q(0.25),
            q(0.50),
            q(0.75),
            q(0.95)
        );
    }
    Ok(())
}
