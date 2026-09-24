//! The fibrillation detector as an alarm, against fibrillation onsets that were
//! never used to fit or tune it.
//!
//! # Where the truth comes from
//!
//! * **Sudden Death** (`sddb`), 100 % TEST. Each header carries the onset of the
//!   fibrillation that ended the recording as a comment, `#vfon: HH:MM:SS`,
//!   taken from the database's own documentation. Only the onset is known,
//!   so the scoring is onset detection and false alarms before it - which is
//!   the question an alarm is asked in any case.
//! * Any corpus whose rhythm annotations carry `(VF` or `(VFL` spans, where the
//!   span starts are the onsets.
//!
//! # What is scored
//!
//! * **Onset found**: an alarm raised between `lead_s` before the onset and
//!   `find_s` after it. The latency is from the onset to the moment the alarm
//!   is raised - the episode start plus the confirmation time - because that
//!   is when a monitor would sound.
//! * **False alarms**: alarms raised earlier than `prodrome_s` before an onset,
//!   per 24 hours of that time. The stretch just before an onset is counted
//!   separately: it is often ventricular tachycardia degenerating, and whether
//!   alarming there is wrong depends on what the alarm is for.
//! * Everything after an onset is left out: the recording's truth ends there.

use crate::manifest::RecordEntry;
use crate::rhythm_ref;
use crate::Opts;
use ecg_pipeline::{ChannelOutput, ChannelPipeline};
use ecg_wfdb::{read_signal, AnnotationFile, Header};
use rayon::prelude::*;
use std::path::Path;

/// Onsets from the header comment `#vfon: HH:MM:SS`, as seconds from the
/// start of the record; or, with `clock`, as a time of day against the
/// header's base time.
fn header_onsets(entry: &RecordEntry, clock: bool) -> Vec<f64> {
    let Ok(text) = std::fs::read_to_string(entry.hea_path()) else {
        return Vec::new();
    };
    let hms = |s: &str| -> Option<f64> {
        let p: Vec<f64> = s
            .trim()
            .split(':')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        (p.len() == 3).then(|| p[0] * 3600.0 + p[1] * 60.0 + p[2])
    };
    let base = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(4))
        .and_then(hms)
        .unwrap_or(0.0);
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("#vfon:"))
        .filter_map(hms)
        .map(|t| {
            if clock {
                (t - base).rem_euclid(86_400.0)
            } else {
                t
            }
        })
        .collect()
}

/// Onsets from `(VF` / `(VFL` rhythm spans, or CUDB's `[` marks.
fn span_onsets(entry: &RecordEntry, n_samples: i64, fs: f64) -> Vec<f64> {
    let Ok(ann) = AnnotationFile::read(Path::new(&entry.ann_path("atr"))) else {
        return Vec::new();
    };
    let mut out: Vec<f64> = rhythm_ref::named_spans(&ann, n_samples)
        .into_iter()
        .filter(|(_, _, n)| n == "VF" || n == "VFL")
        .map(|(a, _, _)| a as f64 / fs)
        .collect();
    // CUDB marks fibrillation with brackets rather than rhythm spans.
    out.extend(
        ann.annotations
            .iter()
            .filter(|a| a.symbol == '[' && a.sample >= 0)
            .map(|a| a.sample as f64 / fs),
    );
    out.sort_by(f64::total_cmp);
    // Adjacent spans of the two names are one episode.
    out.dedup_by(|b, a| *b - *a < 60.0);
    out
}

pub struct RecordAlarm {
    pub name: String,
    pub hours: f64,
    pub onsets: Vec<f64>,
    /// Per decision window: its time in seconds and the detector's probability.
    pub windows: Vec<(f64, f32)>,
    /// Alarms the engine raised: when each sounded, and when it was cleared.
    pub raised: Vec<(f64, f64)>,
}

pub fn analyse(entry: &RecordEntry, opts: &Opts) -> std::io::Result<RecordAlarm> {
    let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    let hdr = Header::read(&entry.hea_path()).map_err(|e| err(e.to_string()))?;
    let fs = hdr.fs;
    let mut onsets = header_onsets(entry, opts.has("vfon-clock"));
    if onsets.is_empty() {
        onsets = span_onsets(entry, hdr.n_samples as i64, fs);
    }
    let lead = opts.lead.min(hdr.n_sig.saturating_sub(1));
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples).map_err(|e| err(e.to_string()))?;
    let cfg = crate::qrs_eval::config_from(opts, fs);
    let confirm = cfg.vf.min_episode_s as f64;
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let block = ((fs * 0.25) as usize).max(1);
    let mut windows = Vec::new();
    let mut episodes: Vec<(u64, u64)> = Vec::new();
    for chunk in sig.chunks(block) {
        out.clear();
        pipe.push(chunk, &mut out);
        windows.extend(out.vf.iter().map(|w| (w.sample as f64 / fs, w.probability)));
        episodes.extend_from_slice(&out.vf_episodes);
    }
    out.clear();
    pipe.finish(&mut out);
    episodes.extend_from_slice(&out.vf_episodes);
    Ok(RecordAlarm {
        name: format!("{}/{}", entry.source, entry.record),
        hours: hdr.n_samples as f64 / fs / 3600.0,
        onsets,
        windows,
        raised: episodes
            .iter()
            .map(|(s, e)| (*s as f64 / fs + confirm, *e as f64 / fs))
            .collect(),
    })
}

/// Alarms under a simple rule over the windows, so an operating point can be
/// swept without re-running the signal: raised once `k` consecutive windows
/// reach `thr`, re-armed after one that does not.
fn rule_alarms(windows: &[(f64, f32)], thr: f32, k: usize) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::new();
    let mut run = 0usize;
    for &(t, p) in windows {
        if p >= thr {
            run += 1;
            if run == k {
                out.push((t, t));
            } else if run > k {
                if let Some(last) = out.last_mut() {
                    last.1 = t;
                }
            }
        } else {
            run = 0;
        }
    }
    out
}

#[derive(Default, Debug, Clone, Copy)]
pub struct AlarmScore {
    pub onsets: u64,
    pub found: u64,
    pub false_alarms: u64,
    pub false_hours: f64,
    pub prodromal: u64,
    pub latency_sum: f64,
}

/// An onset counts as found when an alarm sounds in the window around it, or
/// when one is already sounding as it arrives - recordings with several
/// onsets in a row are one alarm, which is what a monitor would give.
pub fn score(r: &RecordAlarm, raised: &[(f64, f64)], opts: &Opts) -> (AlarmScore, Vec<f64>) {
    let lead = opts.get_f64("lead-s").unwrap_or(30.0);
    let find = opts.get_f64("find-s").unwrap_or(120.0);
    let prodrome = opts.get_f64("prodrome-s").unwrap_or(600.0);
    let mut s = AlarmScore::default();
    let mut latencies = Vec::new();
    let first = r.onsets.first().copied();
    for &on in &r.onsets {
        s.onsets += 1;
        if let Some(&(t, _)) = raised
            .iter()
            .find(|&&(t, _)| t >= on - lead && t <= on + find)
        {
            s.found += 1;
            latencies.push(t - on);
            s.latency_sum += t - on;
        } else if raised.iter().any(|&(t, e)| t < on - lead && e >= on) {
            s.found += 1;
            latencies.push(0.0);
        }
    }
    // Only the time before the first onset is known to be free of it. A
    // recording with no onset has no prodrome: every alarm in it is false.
    let (horizon, prodrome, lead) = match first {
        Some(on) => (on, prodrome, lead),
        None => (r.hours * 3600.0, 0.0, 0.0),
    };
    for &(t, _) in raised {
        if t < horizon - prodrome {
            s.false_alarms += 1;
        } else if t < horizon - lead {
            s.prodromal += 1;
        }
    }
    s.false_hours = ((horizon - prodrome).max(0.0)) / 3600.0;
    (s, latencies)
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    let rows: Vec<RecordAlarm> = entries
        .par_iter()
        .filter_map(|e| {
            analyse(e, opts)
                .map_err(|x| eprintln!("{}: {x}", e.record))
                .ok()
        })
        .collect();
    let report = |label: &str, alarms: &dyn Fn(&RecordAlarm) -> Vec<(f64, f64)>| {
        let mut t = AlarmScore::default();
        let mut lat = Vec::new();
        for r in &rows {
            let (s, l) = score(r, &alarms(r), opts);
            t.onsets += s.onsets;
            t.found += s.found;
            t.false_alarms += s.false_alarms;
            t.false_hours += s.false_hours;
            t.prodromal += s.prodromal;
            lat.extend(l);
            if opts.per_record && label == "engine" {
                println!(
                    "  {:<14} {:>6.1} h  onsets {:?}  found {}  false {}  prodromal {}",
                    r.name,
                    r.hours,
                    r.onsets
                        .iter()
                        .map(|x| format!("{:.0}", x))
                        .collect::<Vec<_>>(),
                    s.found,
                    s.false_alarms,
                    s.prodromal
                );
            }
        }
        lat.sort_by(f64::total_cmp);
        let med = lat.get(lat.len() / 2).copied().unwrap_or(f64::NAN);
        println!(
            "{label:<22} onsets found {:>3} / {:<3} ({:>5.1} %)  median latency {:>5.1} s   \
             false alarms {:>4} over {:>6.0} h = {:>6.2} per 24 h   just before onset {}",
            t.found,
            t.onsets,
            100.0 * t.found as f64 / t.onsets.max(1) as f64,
            med,
            t.false_alarms,
            t.false_hours,
            24.0 * t.false_alarms as f64 / t.false_hours.max(1e-9),
            t.prodromal
        );
    };
    println!(
        "fibrillation alarm: {} records, {:.0} h, {} onsets",
        rows.len(),
        rows.iter().map(|r| r.hours).sum::<f64>(),
        rows.iter().map(|r| r.onsets.len()).sum::<usize>()
    );
    report("engine", &|r| r.raised.clone());
    if let Some(list) = opts.get_str("vf-sweep") {
        // "thr:k,thr:k,..."
        for item in list.split(',') {
            let mut it = item.split(':');
            let thr: f32 = it.next().and_then(|x| x.parse().ok()).unwrap_or(0.7);
            let k: usize = it.next().and_then(|x| x.parse().ok()).unwrap_or(4);
            report(&format!("rule p>={thr} x{k}"), &|r| {
                rule_alarms(&r.windows, thr, k)
            });
        }
    }
    if opts.has("vf-around") {
        // The detector's probability around each onset, to check what the
        // onset time means before anything is scored against it.
        for r in &rows {
            for &on in &r.onsets {
                let near: Vec<String> = r
                    .windows
                    .iter()
                    .filter(|(t, _)| (*t - on).abs() <= 60.0)
                    .step_by(10)
                    .map(|(t, p)| format!("{:+.0}:{:.2}", t - on, p))
                    .collect();
                println!("  {} onset {:.0}: {}", r.name, on, near.join(" "));
            }
        }
    }
    Ok(())
}
