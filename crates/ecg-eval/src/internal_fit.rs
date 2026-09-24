//! Fitting the ventricular detector on the patch corpus.
//!
//! The shipped ensemble was fitted on MIT-BIH and the supraventricular corpus:
//! clinical electrodes, inpatients, thirty minutes each. On the patch corpus it
//! ranks well - AUC 0.975 against an exhaustive review - and calls 2.9 % of
//! normal beats ventricular where the device calls 0.19 %. At a ventricular
//! prevalence of 1.75 % that is 33 % precision, and the review queue recovers
//! only the part of it that clusters.
//!
//! # What is trained on - and what is not
//!
//! Only labels a person decided. The corpus's ordinary labels are the device's
//! own calls with an analyst's corrections on top, and in a recording that was
//! not reviewed exhaustively an untouched beat is the device's opinion, not a
//! finding: training on it teaches the model to agree with the device, which
//! is a different thing from being right. So the rows are:
//!
//! * beats an analyst changed, moved or added (`reviewed`), outside the
//!   device's noise stretches - the corpus's analyst-determined set;
//! * every beat of the few training-zone recordings that were reviewed
//!   exhaustively, where an untouched beat was seen and agreed with;
//! * every beat of the public corpora's training zones, whose annotations are
//!   independent of this device altogether.
//!
//! The analyst-determined set is chosen by what the device got wrong, so it is
//! short of the easy cases - in it the device agrees with the analyst 2.77 % of
//! the time. The public annotations are what supply them, and the two sources
//! are weighted equally so that neither decides the model alone.
//!
//! Ventricular beats are all kept; others are stride-sampled and weighted back.
//! Within a source, records are weighted equally.
//!
//! # What is not touched
//!
//! Nothing here reads the development or sealed zones. The threshold is chosen
//! on the development zone's exhaustively reviewed recordings with
//! `internal-beats --v-model`, and the sealed zone is scored once.

use crate::beat_eval::Aami;
use crate::internal_beats::{self, read_beats};
use crate::patch_eval;
use crate::Opts;
use ecg_beats::{BeatClass, BeatFeatures, NF};
use ecg_zarr::ZarrArray;
use rayon::prelude::*;

/// One training beat: its features, its reviewed class, its weight before the
/// per-record normalisation, and which record it came from.
#[derive(Clone)]
struct Row {
    x: [f32; NF],
    class: Aami,
    weight: f64,
    record: u32,
    /// 0 for the patch corpus, 1 for the public corpora.
    source: u8,
}

fn collect(
    entry: &crate::manifest::RecordEntry,
    record: u32,
    opts: &Opts,
    cfg: &ecg_pipeline::PipelineConfig,
) -> Result<Vec<Row>, String> {
    let hours = opts.get_f64("hours").unwrap_or(24.0);
    let exhaustive = entry.flag("expert_reviewed");
    let stride = opts.get_usize("stride").unwrap_or(50).max(1);

    let beats_path = entry
        .signal_path()
        .with_file_name(format!("{}.beats.parquet", entry.record));
    let mut beats = read_beats(&beats_path)?;
    // The first day of the labelled range. More patients rather than more of
    // each: a model is judged on patients it has not seen.
    if let Some(first) = beats.first().map(|b| b.sample) {
        let end = first + (hours * 3600.0 * entry.fs) as u64;
        beats.retain(|b| b.sample < end);
    }
    let cal = patch_eval::calibrate(entry, opts, cfg);
    if let Some(e) = cal.error {
        return Err(e);
    }
    let mut array = ZarrArray::open(&entry.signal_path()).map_err(|e| e.to_string())?;
    let judged = internal_beats::classify(&mut array, cfg, 1.0 / cal.gain, &beats, None)?;

    let mut out = Vec::new();
    let mut j = 0usize;
    let mut seen = 0usize;
    for b in &beats {
        let Some(truth) = b.truth else { continue };
        if !b.qf_valid {
            continue;
        }
        // A person's label or none: the device's untouched call is not truth.
        if !(exhaustive || b.reviewed) {
            continue;
        }
        while j < judged.verdicts.len() && judged.verdicts[j].0.sample < b.sample {
            j += 1;
        }
        let Some((v, _, _)) = judged.verdicts.get(j).filter(|v| v.0.sample == b.sample) else {
            continue;
        };
        if v.class == BeatClass::Unknown {
            continue;
        }
        let weight = if truth == Aami::V {
            1.0
        } else {
            seen += 1;
            if !seen.is_multiple_of(stride) {
                continue;
            }
            stride as f64
        };
        out.push(Row {
            x: v.features.vector(),
            class: truth,
            weight,
            record,
            source: 0,
        });
    }
    Ok(out)
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let mut entries: Vec<_> = opts
        .select()?
        .into_iter()
        .filter(|e| e.zone == "TRAIN" && e.flag("labels_usable") && e.flag("has_beats"))
        .collect();
    // Spread across the zone rather than taking its first records: the list is
    // sorted by a hash, so every k-th is an unbiased sample of patients.
    let want = opts.get_usize("records-max").unwrap_or(400);
    if entries.len() > want {
        let k = entries.len() / want;
        entries = entries.into_iter().step_by(k.max(1)).take(want).collect();
    }
    println!("ventricular fit on the patch corpus: {} training records", entries.len());

    let cfg = crate::qrs_eval::config_from(opts, 250.0);
    let results: Vec<Result<Vec<Row>, String>> = entries
        .par_iter()
        .enumerate()
        .map(|(i, e)| collect(e, i as u32, opts, &cfg))
        .collect();
    let failed = results.iter().filter(|r| r.is_err()).count();
    let mut rows: Vec<Row> = results.into_iter().filter_map(|r| r.ok()).flatten().collect();

    // The public corpora's training zones, annotated independently of the
    // device. Collected with the same analyser driven at the reference beats.
    let public = opts.get_str("with-public").unwrap_or("mitdb,svdb");
    if public != "none" {
        let manifest = opts
            .get_str("public-manifest")
            .unwrap_or("manifests/records.json")
            .to_string();
        let args: Vec<String> = [
            "--manifest",
            &manifest,
            "--zone",
            "TRAIN",
            "--sources",
            public,
            "--beats",
            "reference",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let po = Opts::parse(&args);
        let base = entries.len() as u32;
        let mut ids = std::collections::HashMap::<String, u32>::new();
        for (f, class, rec) in crate::beat_eval::collect(&po)? {
            if class == Aami::Q {
                continue;
            }
            let next = base + ids.len() as u32;
            let record = *ids.entry(rec).or_insert(next);
            rows.push(Row {
                x: f.vector(),
                class,
                weight: 1.0,
                record,
                source: 1,
            });
        }
        println!("public training records: {}", ids.len());
    }
    if rows.is_empty() {
        println!("no rows");
        return Ok(());
    }

    // Records weighted equally within a source, and the two sources equally:
    // each record sums to one, then each source to one.
    let mut total = std::collections::HashMap::<u32, f64>::new();
    for r in &rows {
        *total.entry(r.record).or_default() += r.weight;
    }
    let mut per_source = [0usize; 2];
    let mut seen_rec = std::collections::HashSet::<u32>::new();
    for r in &rows {
        if seen_rec.insert(r.record) {
            per_source[r.source as usize] += 1;
        }
    }
    for r in rows.iter_mut() {
        r.weight /= total[&r.record] * per_source[r.source as usize].max(1) as f64;
    }
    for src in 0..2u8 {
        let n = rows.iter().filter(|r| r.source == src).count();
        let v = rows
            .iter()
            .filter(|r| r.source == src && r.class == Aami::V)
            .count();
        println!(
            "{}: {} rows ({} ventricular) from {} records",
            if src == 0 { "patch, analyst-determined" } else { "public, expert-annotated" },
            n,
            v,
            per_source[src as usize]
        );
    }
    println!("patch records failed: {failed}");

    let allowed: Vec<usize> = (0..NF)
        .filter(|&i| BeatFeatures::VENTRICULAR_FEATURES.contains(&BeatFeatures::NAMES[i]))
        .collect();
    let mut tc = crate::gbdt_train::TrainConfig::default();
    if let Some(n) = opts.get_usize("trees") {
        tc.trees = n;
    }
    if let Some(n) = opts.get_usize("depth") {
        tc.depth = n;
    }
    if let Some(v) = opts.get_f64("lr") {
        tc.learning_rate = v;
    }
    let x: Vec<Vec<f32>> = rows.iter().map(|r| r.x.to_vec()).collect();
    let y: Vec<f32> = rows
        .iter()
        .map(|r| if r.class == Aami::V { 1.0 } else { 0.0 })
        .collect();
    let w: Vec<f64> = rows.iter().map(|r| r.weight).collect();
    println!("training: {} trees, depth {}", tc.trees, tc.depth);
    let model = crate::gbdt_train::train(&x, &y, &w, &allowed, &tc);
    println!("  {} nodes over {} trees", model.nodes.len(), model.roots.len());

    // A training-set AUC, only as a check that the fit is not degenerate.
    let pairs: Vec<(f32, bool)> = rows
        .iter()
        .step_by(7)
        .map(|r| (model.probability(&r.x), r.class == Aami::V))
        .collect();
    println!(
        "training AUC (sampled, in sample): {:.4}",
        crate::beat_eval::rank_auc(pairs)
    );

    let path = opts.get_str("out").unwrap_or("target/models/ventricular_patch.json");
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(
        path,
        serde_json::to_string(&model)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
    )?;
    println!("wrote {path}");
    Ok(())
}

/// Emit a fitted model as engine source, with its scores divided by a
/// temperature.
///
/// The ensemble fitted on the patch corpus is confident enough that its useful
/// thresholds sit at a raw score of about 20 - a probability within two parts
/// in a billion of 1.0, which an `f32` rounds to exactly 1.0 and a threshold
/// cannot then be placed under. Dividing every leaf and the bias by a constant
/// leaves the ranking untouched - it is a monotone map of the same score - and
/// brings the operating point back to where a probability can hold it. The
/// arbitration between detectors compares how far each clears its own bar as a
/// share of the room above it, so it needs that room to exist.
pub fn emit(opts: &Opts) -> std::io::Result<()> {
    let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    let path = opts.get_str("model").ok_or_else(|| err("--model is required".into()))?;
    let name = opts.get_str("name").unwrap_or("VENTRICULAR_PATCH");
    let t = opts.get_f64("temperature").unwrap_or(1.0) as f32;
    let out = opts
        .get_str("out")
        .ok_or_else(|| err("--out is required".into()))?;
    let mut m: crate::gbdt_train::Model = serde_json::from_str(&std::fs::read_to_string(path)?)
        .map_err(|e| err(e.to_string()))?;
    m.bias /= t;
    for n in m.nodes.iter_mut() {
        if n.feature == crate::gbdt_train::LEAF {
            n.value /= t;
        }
    }
    let src = format!(
        "//! Generated by `ecg-eval emit-model`. Do not edit by hand.\n\
         //!\n\
         //! The ventricular ensemble fitted on the internal patch corpus - see\n\
         //! `ecg-eval internal-fit` - with its scores divided by a temperature of\n\
         //! {t}, which changes where the operating point sits and not the ranking.\n\
         use crate::gbdt::{{GbdtModel, Node}};\n\n{}",
        crate::gbdt_train::emit(name, &m)
    );
    std::fs::write(out, src)?;
    println!("wrote {out}: {} nodes, temperature {t}", m.nodes.len());
    Ok(())
}
