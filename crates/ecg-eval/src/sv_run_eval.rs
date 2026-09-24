//! The supraventricular run detector against an analyst's review, from beat
//! positions alone.
//!
//! The detector reads intervals and nothing else, so it can be swept without
//! decoding a sample: the reference beat positions are the positions the full
//! evaluation drives the engine at anyway. The full path - with the
//! fibrillation gate and the per-beat detector beside it - is
//! `internal-beats`; this exists so a parameter can be chosen in seconds.
//!
//! Truth is the analyst's label on the exhaustively reviewed recordings, and a
//! beat inside a reported run is scored as a supraventricular call.

use crate::internal_beats::{read_beats, Score};
use crate::Opts;
use crate::beat_eval::Aami;
use ecg_rhythm::{SvRunConfig, SvRunDetector};
use rayon::prelude::*;

/// Overrides on top of `base`; `--svrun on|off` switches it.
pub fn config(opts: &Opts, base: SvRunConfig) -> SvRunConfig {
    let mut c = base;
    match opts.get_str("svrun") {
        Some("on") => c.enabled = true,
        Some("off") => c.enabled = false,
        _ => {}
    }
    if let Some(v) = opts.get_usize("svr-window") {
        c.window = v;
    }
    if let Some(v) = opts.get_f64("svr-step") {
        c.step = v as f32;
    }
    if let Some(v) = opts.get_f64("svr-cv") {
        c.max_cv = v as f32;
    }
    if let Some(v) = opts.get_f64("svr-band") {
        c.band = v as f32;
    }
    if let Some(v) = opts.get_usize("svr-grace") {
        c.grace = v as u32;
    }
    if let Some(v) = opts.get_usize("svr-min") {
        c.min_beats = v as u32;
    }
    if opts.has("svr-onset-call") {
        c.require_onset_call = true;
    }
    c
}

struct Result {
    score: Score,
    runs: u64,
    hours: f64,
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    let entries: Vec<_> = opts
        .select()?
        .into_iter()
        .filter(|e| e.flag("expert_eval"))
        .collect();
    // This command exists to measure the detector, so it is on here whatever
    // the default says.
    let cfg = SvRunConfig {
        enabled: true,
        ..config(opts, SvRunConfig::default())
    };
    let rows: Vec<Result> = entries
        .par_iter()
        .filter_map(|e| {
            let path = e
                .signal_path()
                .with_file_name(format!("{}.beats.parquet", e.record));
            let beats = read_beats(&path).ok()?;
            let mut d = SvRunDetector::new(cfg);
            let mut runs = Vec::new();
            for w in beats.windows(2) {
                let rr = (w[1].sample - w[0].sample) as f32 * 1000.0 / e.fs as f32;
                let usable = (200.0..3000.0).contains(&rr);
                if let Some(r) = d.push(rr, w[1].sample, usable, false, false) {
                    runs.push(r);
                }
            }
            runs.extend(d.finish());
            let mut s = Score::default();
            let mut k = 0usize;
            for b in &beats {
                let Some(truth) = b.truth else { continue };
                while k < runs.len() && runs[k].end < b.sample {
                    k += 1;
                }
                let called = runs.get(k).is_some_and(|r| r.start <= b.sample && b.sample <= r.end);
                s.add_pub(truth == Aami::S, called);
            }
            Some(Result {
                score: s,
                runs: runs.len() as u64,
                hours: e.n_samples as f64 / e.fs / 3600.0,
            })
        })
        .collect();
    let mut s = Score::default();
    let (mut runs, mut hours) = (0u64, 0.0f64);
    for r in &rows {
        s.merge(&r.score);
        runs += r.runs;
        hours += r.hours;
    }
    println!(
        "supraventricular runs, {} records, {:.0} h:  window {} step {} cv {} band {} grace {} min {}",
        rows.len(),
        hours,
        cfg.window,
        cfg.step,
        cfg.max_cv,
        cfg.band,
        cfg.grace,
        cfg.min_beats
    );
    println!(
        "  beats inside a run as supraventricular:  Se {:.1} %  +P {:.1} %  F1 {:.3}  \
         false beats per 1000 {:.2}   runs {} ({:.1} per 24 h)",
        100.0 * s.se(),
        100.0 * s.pp(),
        s.f1(),
        s.fp_per_1000(),
        runs,
        24.0 * runs as f64 / hours.max(1e-9)
    );
    Ok(())
}
