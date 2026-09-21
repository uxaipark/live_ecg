//! Parameter sweep over the detector's front-end axes.
//!
//! Selection happens on TRAIN only. TEST is scored once, after the choice is
//! frozen, so the sweep cannot quietly become the thing that fits the test set.

use crate::metrics::DetectionScore;
use crate::qrs_eval;
use crate::Opts;
use rayon::prelude::*;

/// One grid point: `(bp_lo, bp_hi, integ_ms, thr_frac, spk_quantile)`.
type Point = (f64, f64, f64, f64, f64);
/// A scored grid point.
type Row = (f64, f64, f64, f64, f64, DetectionScore, Macro);

fn axis(opts: &Opts, key: &str, default: &[f64]) -> Vec<f64> {
    match opts.get_str(key) {
        Some(s) => s.split(',').filter_map(|v| v.trim().parse().ok()).collect(),
        None => default.to_vec(),
    }
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    if entries.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }

    let los = axis(opts, "sweep-lo", &[3.0, 5.0, 8.0]);
    let his = axis(opts, "sweep-hi", &[15.0, 20.0, 25.0, 35.0]);
    let integs = axis(opts, "sweep-integ", &[100.0, 120.0, 150.0]);
    let thrs = axis(opts, "sweep-thr", &[0.25]);
    let quants = axis(opts, "sweep-spkq", &[0.5]);

    let mut grid: Vec<Point> = Vec::new();
    for &lo in &los {
        for &hi in &his {
            if hi <= lo + 5.0 {
                continue;
            }
            for &ig in &integs {
                for &th in &thrs {
                    for &q in &quants {
                        grid.push((lo, hi, ig, th, q));
                    }
                }
            }
        }
    }
    eprintln!(
        "sweeping {} points over {} records ({:?})",
        grid.len(),
        entries.len(),
        opts.zones
    );

    let mut rows: Vec<Row> = grid
        .par_iter()
        .map(|&(lo, hi, ig, th, q)| {
            let mut sub = Opts {
                manifest: opts.manifest.clone(),
                zones: opts.zones.clone(),
                sources: opts.sources.clone(),
                records: opts.records.clone(),
                lead: opts.lead,
                limit: opts.limit,
                tol_ms: opts.tol_ms,
                skip_sec: opts.skip_sec,
                threads: Some(1),
                per_record: false,
                json: None,
                gate: opts.gate,
                raw: opts.raw.clone(),
            };
            sub.raw.retain(|(k, _)| {
                !matches!(
                    k.as_str(),
                    "bp-lo" | "bp-hi" | "integ-ms" | "thr-frac" | "spk-q"
                )
            });
            sub.raw.push(("bp-lo".into(), lo.to_string()));
            sub.raw.push(("bp-hi".into(), hi.to_string()));
            sub.raw.push(("integ-ms".into(), ig.to_string()));
            sub.raw.push(("thr-frac".into(), th.to_string()));
            sub.raw.push(("spk-q".into(), q.to_string()));

            let results: Vec<_> = entries.iter().map(|e| qrs_eval::run_one(e, &sub)).collect();
            let mut total = DetectionScore::default();
            let mut f1s = Vec::with_capacity(results.len());
            for r in &results {
                if r.error.is_some() {
                    continue;
                }
                total.merge(&r.score);
                f1s.push(r.score.f1());
            }
            f1s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            (lo, hi, ig, th, q, total, Macro::from(&f1s))
        })
        .collect();

    // Ranked by the per-record mean, not the pooled figure: pooled is dominated
    // by the longest records, so a configuration that ruins a short record can
    // still come out on top. Ties are broken on the tenth percentile, which is
    // what a fleet of patches actually feels.
    rows.sort_by(|a, b| {
        (b.6.mean, b.6.p10)
            .partial_cmp(&(a.6.mean, a.6.p10))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "{:>6} {:>6} {:>8} {:>6} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "lo", "hi", "integ", "thr", "spkq", "Se%", "PPV%", "pooledF1", "meanF1", "p10F1", "worstF1"
    );
    for (lo, hi, ig, th, q, s, m) in &rows {
        println!(
            "{:>6.1} {:>6.1} {:>8.0} {:>6.2} {:>6.2} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
            lo,
            hi,
            ig,
            th,
            q,
            100.0 * s.sensitivity(),
            100.0 * s.ppv(),
            100.0 * s.f1(),
            100.0 * m.mean,
            100.0 * m.p10,
            100.0 * m.worst
        );
    }
    Ok(())
}

/// Per-record summary of a configuration.
#[derive(Debug, Clone, Copy, Default)]
struct Macro {
    mean: f64,
    p10: f64,
    worst: f64,
}

impl Macro {
    /// `f1s` must be sorted ascending.
    fn from(f1s: &[f64]) -> Macro {
        if f1s.is_empty() {
            return Macro::default();
        }
        let idx = ((0.10 * (f1s.len() - 1) as f64).round() as usize).min(f1s.len() - 1);
        Macro {
            mean: f1s.iter().sum::<f64>() / f1s.len() as f64,
            p10: f1s[idx],
            worst: f1s[0],
        }
    }
}
