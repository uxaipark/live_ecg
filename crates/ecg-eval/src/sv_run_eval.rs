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

use crate::beat_eval::Aami;
use crate::internal_beats::{read_beats, Score};
use crate::Opts;
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
    if let Some(v) = opts.get_f64("svr-max-rr") {
        c.max_interval_ms = v as f32;
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
    features: Vec<String>,
}

fn med(v: &[f32]) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    let mut w = v.to_vec();
    w.sort_by(f32::total_cmp);
    let n = w.len();
    if n % 2 == 1 {
        w[n / 2]
    } else {
        0.5 * (w[n / 2 - 1] + w[n / 2])
    }
}

/// What a run looks like from its intervals: how it began, how it held, and
/// how it ended. `rr[i]` is the interval closed by `at[i]`.
pub fn run_features(rr: &[f32], at: &[u64], start: u64, end: u64) -> [f32; 9] {
    let s = at.partition_point(|&a| a < start);
    let e = at.partition_point(|&a| a <= end).max(s + 1);
    let pre = &rr[s.saturating_sub(8)..s];
    let inside = &rr[s..e];
    let post = &rr[e..(e + 4).min(rr.len())];
    let rate = med(inside);
    let pre_m = med(pre);
    let mean = inside.iter().sum::<f32>() / inside.len() as f32;
    let cv = (inside.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / inside.len() as f32)
        .sqrt()
        / mean.max(1.0);
    // Least-squares slope over the run, as a fraction of its rate per beat.
    let n = inside.len() as f32;
    let xm = (n - 1.0) / 2.0;
    let (mut sxy, mut sxx) = (0.0f32, 0.0f32);
    for (i, &y) in inside.iter().enumerate() {
        sxy += (i as f32 - xm) * (y - mean);
        sxx += (i as f32 - xm) * (i as f32 - xm);
    }
    let trend = if sxx > 0.0 {
        sxy / sxx / mean.max(1.0)
    } else {
        0.0
    };
    // How abruptly it began: the first interval of the run, and the largest
    // single-interval drop among the first few.
    let onset1 = rr.get(s).copied().unwrap_or(f32::NAN) / pre_m;
    let mut biggest = 1.0f32;
    for i in s..(s + 4).min(e) {
        if i > 0 {
            biggest = biggest.min(rr[i] / rr[i - 1]);
        }
    }
    let pause = post.first().copied().unwrap_or(f32::NAN) / rate;
    let ret = med(post) / rate;
    [
        rate,
        pre_m / rate,
        onset1,
        biggest,
        cv,
        trend,
        pause,
        ret,
        inside.len() as f32,
    ]
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
            let mut rrs = Vec::with_capacity(beats.len());
            let mut ats = Vec::with_capacity(beats.len());
            for w in beats.windows(2) {
                let rr = (w[1].sample - w[0].sample) as f32 * 1000.0 / e.fs as f32;
                rrs.push(rr);
                ats.push(w[1].sample);
                let usable = (200.0..3000.0).contains(&rr);
                if let Some(r) = d.push(rr, w[1].sample, usable, false, false) {
                    runs.push(r);
                }
            }
            runs.extend(d.finish());
            let mut features = Vec::new();
            for r in &runs {
                let lo = beats.partition_point(|b| b.sample < r.start);
                let hi = beats.partition_point(|b| b.sample <= r.end);
                let (mut n, mut sv) = (0u32, 0u32);
                for b in &beats[lo..hi] {
                    if let Some(t) = b.truth {
                        n += 1;
                        sv += (t == Aami::S) as u32;
                    }
                }
                let f = run_features(&rrs, &ats, r.start, r.end);
                let f: Vec<String> = f.iter().map(|x| format!("{x:.4}")).collect();
                features.push(format!(
                    "{},{},{},{},{},{}",
                    e.record,
                    r.start,
                    r.end,
                    n,
                    sv,
                    f.join(",")
                ));
            }
            let mut s = Score::default();
            let mut k = 0usize;
            for b in &beats {
                let Some(truth) = b.truth else { continue };
                while k < runs.len() && runs[k].end < b.sample {
                    k += 1;
                }
                let called = runs
                    .get(k)
                    .is_some_and(|r| r.start <= b.sample && b.sample <= r.end);
                s.add_pub(truth == Aami::S, called);
            }
            Some(Result {
                score: s,
                runs: runs.len() as u64,
                hours: e.n_samples as f64 / e.fs / 3600.0,
                features,
            })
        })
        .collect();
    if let Some(path) = opts.get_str("svr-features") {
        let mut text = String::from(
            "record,start,end,n_truth,n_sv,rate,pre_ratio,onset1,biggest_drop,cv,trend,end_pause,return_ratio,intervals\n",
        );
        for r in &rows {
            for l in &r.features {
                text.push_str(l);
                text.push('\n');
            }
        }
        std::fs::write(path, text)?;
    }
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
