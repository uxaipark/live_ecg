use crate::manifest::RecordEntry;
use crate::metrics::{self, DetectionScore};
use crate::Opts;
use ecg_pipeline::{ChannelOutput, ChannelPipeline, Mains, PipelineConfig};
use ecg_wfdb::{read_signal, AnnotationFile, Header};
use rayon::prelude::*;
use std::path::Path;

/// Build a pipeline configuration for `fs`, applying CLI overrides.
pub fn config_from(opts: &Opts, fs: f64) -> PipelineConfig {
    let mut c = PipelineConfig::new(fs);
    if let Some(v) = opts.get_f64("hp-hz") {
        c.preprocess.hp_hz = v;
    }
    if let Some(v) = opts.get_f64("lp-hz") {
        c.preprocess.lp_hz = v;
    }
    if let Some(v) = opts.get_str("mains") {
        c.preprocess.mains = match v {
            "off" => Mains::Off,
            "50" => Mains::Fixed50,
            "60" => Mains::Fixed60,
            _ => Mains::Auto,
        };
    }
    if let Some(v) = opts.get_f64("bp-lo") {
        c.qrs.bp_lo = v;
    }
    if let Some(v) = opts.get_f64("bp-hi") {
        c.qrs.bp_hi = v;
    }
    if let Some(v) = opts.get_usize("bp-order") {
        c.qrs.bp_order = v;
    }
    if let Some(v) = opts.get_f64("integ-ms") {
        c.qrs.integ_ms = v;
    }
    if let Some(v) = opts.get_f64("refractory-ms") {
        c.qrs.refractory_ms = v;
    }
    if let Some(v) = opts.get_f64("twave-ms") {
        c.qrs.twave_ms = v;
    }
    if let Some(v) = opts.get_f64("thr-frac") {
        c.qrs.thr_frac = v as f32;
    }
    if let Some(v) = opts.get_f64("spk-q") {
        c.qrs.spk_quantile = v as f32;
    }
    if let Some(v) = opts.get_f64("peak-drop") {
        c.qrs.peak_drop = v as f32;
    }
    if let Some(v) = opts.get_f64("searchback") {
        c.qrs.searchback_factor = v as f32;
    }
    if let Some(v) = opts.get_f64("refine-ms") {
        c.qrs.refine_halfwidth_ms = v;
    }
    if let Some(v) = opts.get_f64("bias-ms") {
        c.qrs.fiducial_bias_ms = v;
    }
    if let Some(v) = opts.get_f64("v-thr") {
        c.bank.ventricular.threshold = v as f32;
    }
    if let Some(v) = opts.get_f64("s-thr") {
        c.bank.supraventricular.threshold = v as f32;
    }
    if let Some(v) = opts.get_usize("af-window") {
        c.af.window_beats = v;
    }
    if let Some(v) = opts.get_f64("af-enter") {
        c.af.enter_prob = v as f32;
    }
    if let Some(v) = opts.get_f64("af-exit") {
        c.af.exit_prob = v as f32;
    }
    if let Some(v) = opts.get_f64("score-bad") {
        c.quality.score_bad = v as f32;
    }
    if let Some(v) = opts.get_f64("score-warn") {
        c.quality.score_warn = v as f32;
    }
    c.suppress_unusable = opts.gate;
    c
}

pub struct RecordResult {
    pub source: String,
    pub record: String,
    pub zone: String,
    pub fs: f64,
    pub seconds: f64,
    pub score: DetectionScore,
    pub good_frac: f64,
    pub unusable_frac: f64,
    pub cpu_seconds: f64,
    pub error: Option<String>,
    /// Set when the record cannot be scored for a known, benign reason.
    pub skipped: Option<String>,
    /// Detected R positions, kept so quality analysis can attribute errors to
    /// the second they happened in without running the pipeline twice.
    pub detections: Vec<i64>,
}

/// Annotation extension to score against, per dataset.
fn ann_ext(source: &str) -> &'static str {
    match source {
        // afdb ships rhythm labels in `.atr`; the beat reference is `.qrs`.
        "afdb" => "qrs",
        _ => "atr",
    }
}

pub fn run_one(entry: &RecordEntry, opts: &Opts) -> RecordResult {
    let mut res = RecordResult {
        source: entry.source.clone(),
        record: entry.record.clone(),
        zone: entry.zone.clone(),
        fs: entry.fs,
        seconds: 0.0,
        score: DetectionScore::default(),
        good_frac: 0.0,
        unusable_frac: 0.0,
        cpu_seconds: 0.0,
        error: None,
        skipped: None,
        detections: Vec::new(),
    };

    let hdr = match Header::read(&entry.hea_path()) {
        Ok(h) => h,
        Err(e) => {
            res.error = Some(e.to_string());
            return res;
        }
    };
    let lead = opts.lead.min(hdr.n_sig.saturating_sub(1));
    let sig = match read_signal(&hdr, lead, 0, hdr.n_samples) {
        Ok(s) => s,
        Err(e) => {
            res.error = Some(e.to_string());
            return res;
        }
    };
    let ann_path = entry.ann_path(ann_ext(&entry.source));
    if !ann_path.exists() {
        // No full-record reference. Some corpora ship only a partial one - QTDB's
        // `.man` files annotate about thirty beats per record for QT measurement
        // - and scoring a whole-record detector against those would count every
        // unannotated beat as a false positive. Skipping is the honest option;
        // substituting the partial file would manufacture a number.
        res.skipped = Some("no full-record beat reference".to_string());
        return res;
    }
    let reference = match AnnotationFile::read(Path::new(&ann_path)) {
        Ok(a) => a.beat_samples(),
        Err(e) => {
            res.error = Some(format!("{}: {e}", ann_path.display()));
            return res;
        }
    };

    let fs = hdr.fs;
    let cfg = config_from(opts, fs);
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();

    let t0 = std::time::Instant::now();
    // Blocked feed, as the server does: one 250 ms packet at a time.
    let block = ((fs * 0.25) as usize).max(1);
    for chunk in sig.chunks(block) {
        pipe.push(chunk, &mut out);
    }
    res.cpu_seconds = t0.elapsed().as_secs_f64();

    let skip = (opts.skip_sec * fs) as i64;
    let tol = (opts.tol_ms * fs / 1000.0).round() as i64;
    let det: Vec<i64> = out
        .beats
        .iter()
        .map(|e| e.sample as i64)
        .filter(|&s| s >= skip)
        .collect();
    let refs: Vec<i64> = reference.into_iter().filter(|&s| s >= skip).collect();

    res.score = metrics::score(&refs, &det, tol, fs, true);
    res.detections = det;
    res.seconds = sig.len() as f64 / fs;
    let total = (out.good_samples + out.acceptable_samples + out.unusable_samples).max(1) as f64;
    res.good_frac = out.good_samples as f64 / total;
    res.unusable_frac = out.unusable_samples as f64 / total;
    res
}

pub fn run_all(entries: &[RecordEntry], opts: &Opts) -> Vec<RecordResult> {
    entries.par_iter().map(|e| run_one(e, opts)).collect()
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    if entries.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }
    eprintln!(
        "scoring {} records  zones={:?} sources={:?} lead={} tol={}ms gate={}",
        entries.len(),
        opts.zones,
        opts.sources,
        opts.lead,
        opts.tol_ms,
        opts.gate
    );

    let t0 = std::time::Instant::now();
    let results = run_all(&entries, opts);
    let wall = t0.elapsed().as_secs_f64();

    report(&results, wall, opts);
    Ok(())
}

pub fn report(results: &[RecordResult], wall: f64, opts: &Opts) {
    let mut total = DetectionScore::default();
    let mut signal_seconds = 0.0;
    let mut cpu_seconds = 0.0;
    let mut failed = Vec::new();
    let mut skipped: std::collections::BTreeMap<String, Vec<String>> = Default::default();

    if opts.per_record {
        println!(
            "{:<10} {:>8} {:>6} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8} {:>7}",
            "source", "record", "zone", "nref", "TP", "FP", "FN", "Se%", "PPV%", "bad%"
        );
    }
    let mut rows: Vec<&RecordResult> = results.iter().collect();
    rows.sort_by(|a, b| {
        a.score
            .f1()
            .partial_cmp(&b.score.f1())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for r in &rows {
        if let Some(why) = &r.skipped {
            skipped
                .entry(why.clone())
                .or_default()
                .push(format!("{}/{}", r.source, r.record));
            continue;
        }
        if let Some(e) = &r.error {
            failed.push(format!("{}/{}: {e}", r.source, r.record));
            continue;
        }
        total.merge(&r.score);
        signal_seconds += r.seconds;
        cpu_seconds += r.cpu_seconds;
        if opts.per_record {
            println!(
                "{:<10} {:>8} {:>6} {:>7} {:>7} {:>7} {:>7} {:>8.3} {:>8.3} {:>7.2}",
                r.source,
                r.record,
                r.zone,
                r.score.n_ref,
                r.score.tp,
                r.score.fp,
                r.score.fn_,
                100.0 * r.score.sensitivity(),
                100.0 * r.score.ppv(),
                100.0 * r.unusable_frac,
            );
        }
    }

    let mut offs = total.offsets.clone();
    offs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_ok = results
        .iter()
        .filter(|r| r.error.is_none() && r.skipped.is_none())
        .count();

    println!("\n── QRS detection ─────────────────────────────────────────────");
    println!(
        "records          {n_ok}  ({:.2} h of signal)",
        signal_seconds / 3600.0
    );
    println!("reference beats  {}", total.n_ref);
    println!("detected         {}", total.n_det);
    println!(
        "TP / FP / FN     {} / {} / {}",
        total.tp, total.fp, total.fn_
    );
    println!("Sensitivity      {:.4} %", 100.0 * total.sensitivity());
    println!("PPV (+P)         {:.4} %", 100.0 * total.ppv());
    println!("F1               {:.4} %", 100.0 * total.f1());
    println!("DER              {:.4} %", 100.0 * total.der());
    if !offs.is_empty() {
        println!(
            "fiducial offset  mean {:+.2} ms  sd {:.2} ms  p5 {:+.1} / p50 {:+.1} / p95 {:+.1} ms  (rates {:.0}-{:.0} Hz)",
            metrics::mean(&offs),
            metrics::stddev(&offs),
            metrics::percentile(&offs, 0.05),
            metrics::percentile(&offs, 0.50),
            metrics::percentile(&offs, 0.95),
            results.iter().filter(|r| r.error.is_none()).map(|r| r.fs).fold(f64::INFINITY, f64::min),
            results.iter().filter(|r| r.error.is_none()).map(|r| r.fs).fold(0.0, f64::max),
        );
    }

    // Per-record aggregate: the pooled figure hides a record that collapses.
    let mut f1s: Vec<f64> = rows
        .iter()
        .filter(|r| r.error.is_none() && r.skipped.is_none())
        .map(|r| r.score.f1())
        .collect();
    f1s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !f1s.is_empty() {
        let q = |p: f64| f1s[((p * (f1s.len() - 1) as f64).round() as usize).min(f1s.len() - 1)];
        let mean_f1 = f1s.iter().sum::<f64>() / f1s.len() as f64;
        println!(
            "per-record F1    min {:.4}  p10 {:.4}  median {:.4}  mean {:.4}",
            100.0 * f1s[0],
            100.0 * q(0.10),
            100.0 * q(0.50),
            100.0 * mean_f1,
        );
    }

    println!(
        "\nthroughput       {:.0}x realtime single-thread-equivalent  ({:.2} s CPU for {:.0} s signal)",
        signal_seconds / cpu_seconds.max(1e-9),
        cpu_seconds,
        signal_seconds
    );
    println!("wall             {wall:.2} s");
    for (why, recs) in &skipped {
        println!("\nskipped {} ({why}):", recs.len());
        println!("  {}", recs.join(" "));
    }
    if !failed.is_empty() {
        println!("\nfailed ({}):", failed.len());
        for f in failed.iter().take(20) {
            println!("  {f}");
        }
    }
}
