//! The internal patch corpus: what the signal is, and what one count is worth.
//!
//! The public corpora arrive in millivolts because their headers carry a gain.
//! This one does not — `adc_gain` is null for all 24,043 recordings, so the
//! samples are raw converter counts and nothing says how many of them make a
//! millivolt. That matters for exactly one number in the engine, and it is not
//! a number anyone would guess at: the saturation level, which decides the
//! electrode-failure flag, which in turn gates asystole. Fed raw counts, a
//! twenty-count QRS is read as twenty millivolts, every sample is called
//! saturated, and the channel reports a detached lead for its whole length.
//!
//! # The gain is recoverable, because the detector does not need it
//!
//! Every threshold in the QRS detector is a fraction of a running estimate or a
//! duration — `thr_frac`, `spk_quantile`, `searchback_factor`, the refractory
//! period. So it runs correctly on counts, and the amplitude it reports for
//! each beat is in counts. The gain is then whatever makes those beats the size
//! a heart makes, which is what a WFDB header's gain field already does for the
//! public corpora: it is not a free parameter, it is the missing one.
//!
//! # Where the sampling matters
//!
//! Calibrating on the opening minutes would be wrong. A patch's first stretch
//! is electrode settling, and a fourteen-day recording has hours of rubbing and
//! sleep in it. So the estimate pools beats from windows spread evenly across
//! the whole recording and takes the median, which is unmoved by the stretches
//! where there is nothing to measure.

use crate::manifest::RecordEntry;
use crate::Opts;
use ecg_pipeline::{PipelineConfig, Preprocessor};
use ecg_qrs::{QrsDetector, QrsEvent};
use ecg_quality::{Quality, QualityMonitor};
use ecg_zarr::ZarrArray;
use rayon::prelude::*;

/// What a normal QRS is taken to measure. One millivolt is the round number
/// the public corpora sit at, which is the point: it puts both corpora on one
/// scale so a threshold argued on one means the same on the other.
pub const TARGET_MV: f32 = 1.0;

#[derive(Debug, Clone)]
pub struct Calibration {
    pub record: String,
    pub zone: String,
    pub hours: f64,
    /// Median QRS amplitude in converter counts.
    pub counts: f32,
    /// Counts per millivolt, from that median.
    pub gain: f32,
    /// Interquartile spread of the beat amplitudes, over the median. A record
    /// whose beats are all one size gives a small number; one that wanders
    /// gives a large one, and the gain means less there.
    pub spread: f32,
    pub beats: usize,
    /// Beats a minute over the windows that were probed.
    pub rate: f32,
    /// What the device said the average rate was, when it said.
    pub device_rate: Option<f64>,
    pub probed_s: f64,
    /// Seconds the quality monitor was willing to believe, at unit gain and
    /// then at the estimated gain. The pair is the point: the first is what
    /// happens when counts are read as millivolts, and it is not a property of
    /// the signal.
    pub usable_s: f64,
    pub usable_scaled_s: f64,
    /// Share of samples the front end called saturated, at each scale.
    pub sat: f64,
    pub sat_scaled: f64,
    pub error: Option<String>,
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// Run the front end and the detector over one window at a given scale.
fn run_window(
    raw: &[i32],
    cfg: &PipelineConfig,
    scale: f32,
    amps: &mut Vec<f32>,
    usable: &mut u64,
    saturated: &mut u64,
) -> Result<usize, String> {
    let mut pre = Preprocessor::new(cfg.preprocess);
    let mut qual = QualityMonitor::new(cfg.quality);
    let mut qrs = QrsDetector::new(cfg.qrs);
    let mut out: Vec<QrsEvent> = Vec::new();
    let mut n = 0;
    for &x in raw {
        let b = pre.process(x as f32 * scale);
        let q = qual.process(b.raw, b.clean, b.baseline, b.hf, b.qrs, b.saturated);
        let ok = q.level(&cfg.quality) != Quality::Unusable;
        if ok {
            *usable += 1;
        }
        if b.saturated {
            *saturated += 1;
        }
        out.clear();
        qrs.process_prefiltered(b.clean, b.qrs, ok, &mut out);
        for e in &out {
            // The detector's own warm-up beats are not evidence about
            // amplitude: the running peak estimate has not settled.
            n += 1;
            if n > 4 {
                amps.push(e.amplitude.abs());
            }
        }
    }
    Ok(n)
}

pub fn calibrate(entry: &RecordEntry, opts: &Opts, cfg: &PipelineConfig) -> Calibration {
    let probes = opts.get_usize("probes").unwrap_or(30).max(1);
    let window_s = opts.get_f64("probe-s").unwrap_or(60.0).max(1.0);
    let mut c = Calibration {
        record: entry.record.clone(),
        zone: entry.zone.clone(),
        hours: entry.n_samples as f64 / entry.fs / 3600.0,
        counts: 0.0,
        gain: 0.0,
        spread: 0.0,
        beats: 0,
        rate: 0.0,
        device_rate: entry.number("hr_avg"),
        probed_s: 0.0,
        usable_s: 0.0,
        usable_scaled_s: 0.0,
        sat: 0.0,
        sat_scaled: 0.0,
        error: None,
    };
    let mut array = match ZarrArray::open(&entry.signal_path()) {
        Ok(a) => a,
        Err(e) => {
            c.error = Some(e.to_string());
            return c;
        }
    };
    let fs = array.fs();
    let span = (window_s * fs) as u64;
    let n = array.n_samples();
    if n < span {
        c.error = Some("shorter than one probe window".into());
        return c;
    }
    let mut amps = Vec::new();
    let (mut usable, mut sat) = (0u64, 0u64);
    let stride = (n - span) / probes.max(1) as u64;
    // Held so the second pass measures the same seconds as the first. Reading
    // them twice would be the same samples; keeping them is simply cheaper.
    let mut windows: Vec<Vec<i32>> = Vec::with_capacity(probes);
    for k in 0..probes as u64 {
        let from = k * stride;
        let raw = match array.read_lead(0, from, from + span) {
            Ok(r) => r,
            Err(e) => {
                c.error = Some(e.to_string());
                return c;
            }
        };
        match run_window(&raw, cfg, 1.0, &mut amps, &mut usable, &mut sat) {
            Ok(b) => {
                c.beats += b;
                c.probed_s += window_s;
            }
            Err(e) => {
                c.error = Some(e);
                return c;
            }
        }
        windows.push(raw);
    }
    c.usable_s = usable as f64 / fs;
    c.sat = sat as f64 / (c.probed_s * fs).max(1.0);
    c.rate = if c.probed_s > 0.0 {
        60.0 * c.beats as f32 / c.probed_s as f32
    } else {
        0.0
    };
    if amps.is_empty() {
        c.error = Some("no beats to calibrate on".into());
        return c;
    }
    c.counts = median(&mut amps);
    let q = |p: f64| amps[((p * (amps.len() - 1) as f64).round() as usize).min(amps.len() - 1)];
    c.spread = if c.counts > 0.0 {
        (q(0.75) - q(0.25)) / c.counts
    } else {
        0.0
    };
    c.gain = c.counts / TARGET_MV;

    // Second pass, at the gain the first one found. This is the measurement
    // that says whether the corpus is readable; the first pass is only the
    // means of getting here.
    if c.gain > 0.0 {
        let (mut usable, mut sat) = (0u64, 0u64);
        let mut scrap = Vec::new();
        for raw in &windows {
            let _ = run_window(raw, cfg, 1.0 / c.gain, &mut scrap, &mut usable, &mut sat);
        }
        c.usable_scaled_s = usable as f64 / fs;
        c.sat_scaled = sat as f64 / (c.probed_s * fs).max(1.0);
    }
    c
}

/// Calibrate every selected record.
pub fn summarise(opts: &Opts) -> std::io::Result<Vec<Calibration>> {
    let entries = opts.select()?;
    let cfg = crate::qrs_eval::config_from(opts, 250.0);
    let mut rows: Vec<Calibration> = entries
        .par_iter()
        .map(|e| calibrate(e, opts, &cfg))
        .collect();
    rows.sort_by(|a, b| a.counts.total_cmp(&b.counts));
    Ok(rows)
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    let rows = summarise(opts)?;
    println!("patch calibration: {} records", rows.len());

    if opts.per_record {
        println!(
            "\n{:<18} {:>5} {:>7} {:>9} {:>7} {:>7} {:>7} {:>9} {:>9}",
            "record",
            "zone",
            "hours",
            "counts/mV",
            "spread",
            "bpm",
            "device",
            "usable@1",
            "usable@g"
        );
        for r in &rows {
            if let Some(e) = &r.error {
                println!("{:<18} {:>5} {:>7.1}   {e}", r.record, r.zone, r.hours);
                continue;
            }
            println!(
                "{:<18} {:>5} {:>7.1} {:>9.1} {:>7.2} {:>7.1} {:>7} {:>8.1}% {:>8.1}%",
                r.record,
                r.zone,
                r.hours,
                r.gain,
                r.spread,
                r.rate,
                r.device_rate.map(|d| format!("{d:.0}")).unwrap_or_default(),
                100.0 * r.usable_s / r.probed_s.max(1e-9),
                100.0 * r.usable_scaled_s / r.probed_s.max(1e-9)
            );
        }
    }

    let ok: Vec<&Calibration> = rows.iter().filter(|r| r.error.is_none()).collect();
    println!("\n── what one count is worth ───────────────────────────────────");
    println!("records calibrated       {} of {}", ok.len(), rows.len());
    if ok.is_empty() {
        return Ok(());
    }
    let mut gains: Vec<f32> = ok.iter().map(|r| r.gain).collect();
    gains.sort_by(f32::total_cmp);
    let q = |v: &[f32], p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
    println!(
        "counts per millivolt     p5 {:.1}  p25 {:.1}  median {:.1}  p75 {:.1}  p95 {:.1}",
        q(&gains, 0.05),
        q(&gains, 0.25),
        q(&gains, 0.50),
        q(&gains, 0.75),
        q(&gains, 0.95)
    );
    println!(
        "  spread, largest / smallest              {:.0}x",
        q(&gains, 0.95) / q(&gains, 0.05).max(1e-6)
    );
    let mut spreads: Vec<f32> = ok.iter().map(|r| r.spread).collect();
    spreads.sort_by(f32::total_cmp);
    println!(
        "beat-amplitude spread within a record     median {:.2}  p95 {:.2}",
        q(&spreads, 0.50),
        q(&spreads, 0.95)
    );
    let frac = |f: &dyn Fn(&Calibration) -> f64| {
        let mut v: Vec<f64> = ok.iter().map(|r| f(r)).collect();
        v.sort_by(f64::total_cmp);
        v
    };
    let qf = |v: &[f64], p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
    let u1 = frac(&|r| 100.0 * r.usable_s / r.probed_s.max(1e-9));
    let ug = frac(&|r| 100.0 * r.usable_scaled_s / r.probed_s.max(1e-9));
    let s1 = frac(&|r| 100.0 * r.sat);
    let sg = frac(&|r| 100.0 * r.sat_scaled);
    println!(
        "\nsignal the quality monitor believes   at unit gain   median {:.1} %  p90 {:.1} %",
        qf(&u1, 0.50),
        qf(&u1, 0.90)
    );
    println!(
        "                                      at the gain    median {:.1} %  p90 {:.1} %",
        qf(&ug, 0.50),
        qf(&ug, 0.90)
    );
    println!(
        "samples called saturated              at unit gain   median {:.1} %   at the gain {:.3} %",
        qf(&s1, 0.50),
        qf(&sg, 0.50)
    );

    let paired: Vec<(f32, f64)> = ok
        .iter()
        .filter_map(|r| r.device_rate.map(|d| (r.rate, d)))
        .collect();
    if !paired.is_empty() {
        let mut err: Vec<f32> = paired.iter().map(|(a, b)| (a - *b as f32).abs()).collect();
        err.sort_by(f32::total_cmp);
        println!(
            "rate against the device   median |error| {:.1} bpm  p90 {:.1}  over {} records",
            q(&err, 0.50),
            q(&err, 0.90),
            paired.len()
        );
    }

    // The gain is derived, so it is not written back into the manifest - a
    // derived number in a hand-made file is a number that goes stale without
    // saying so. It is emitted beside the other results instead, for whatever
    // reads the corpus next.
    if let Some(path) = opts.get_str("emit-gains") {
        let map: std::collections::BTreeMap<&str, f32> =
            ok.iter().map(|r| (r.record.as_str(), r.gain)).collect();
        std::fs::write(
            path,
            serde_json::to_string_pretty(&map)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
        )?;
        println!("\nwrote {} gains to {path}", map.len());
    }
    Ok(())
}
