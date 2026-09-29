//! One record through the engine, written out as JSON for the introduction's
//! figures: the signal over a chosen stretch, what the engine found there and
//! what the annotators marked, and the whole record's AF and VF timelines.
//!
//! Public corpora only. The patch corpus is private, and none of its signal
//! belongs in material meant to be published.

use crate::beat_eval::{aami, ann_ext};
use crate::rhythm_ref;
use crate::Opts;
use ecg_beats::BeatClass;
use ecg_pipeline::{ChannelOutput, ChannelPipeline};
use ecg_quality::Quality;
use ecg_wfdb::{read_signal, AnnotationFile, Header};
use serde_json::{json, Value};
use std::path::Path;

fn class_name(c: BeatClass) -> &'static str {
    match c {
        BeatClass::N => "N",
        BeatClass::S => "S",
        BeatClass::V => "V",
        BeatClass::F => "F",
        BeatClass::Unknown => "Q",
    }
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    let entry = opts
        .select()?
        .into_iter()
        .next()
        .ok_or_else(|| err("no record selected".into()))?;
    let hdr = Header::read(&entry.hea_path()).map_err(|e| err(e.to_string()))?;
    let fs = hdr.fs;
    let lead = opts.lead.min(hdr.n_sig.saturating_sub(1));
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples).map_err(|e| err(e.to_string()))?;
    let from = (opts.get_f64("from-s").unwrap_or(0.0) * fs) as usize;
    let len = (opts.get_f64("seconds").unwrap_or(10.0) * fs) as usize;
    let to = (from + len).min(sig.len());
    let in_strip = |s: u64| (from as u64..to as u64).contains(&s);

    let cfg = crate::qrs_eval::config_from(opts, fs);
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let block = ((fs * 0.25) as usize).max(1);
    let (mut beats, mut af, mut vf, mut episodes, mut quality, mut vf_eps) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut all_beats: Vec<(u64, BeatClass, u32)> = Vec::new();
    for (k, chunk) in sig.chunks(block).enumerate() {
        out.clear();
        pipe.push(chunk, &mut out);
        let at = (k * block) as u64;
        if let (Some(q), Some(level)) = (out.quality, out.quality_level) {
            if in_strip(at) {
                let level = match level {
                    Quality::Good => 0,
                    Quality::Acceptable => 1,
                    Quality::Unusable => 2,
                };
                quality.push(json!([at, level, q.score]));
            }
        }
        for v in &out.classes {
            all_beats.push((v.sample, v.class, v.cluster));
            if in_strip(v.sample) {
                beats.push(json!({
                    "s": v.sample, "c": class_name(v.class),
                    "pv": v.p_ventricular, "ps": v.p_supraventricular, "k": v.cluster
                }));
            }
        }
        for w in &out.af {
            af.push(json!([w.start_sample, w.sample, w.probability, w.in_af]));
        }
        for w in &out.vf {
            vf.push(json!([w.sample, w.probability]));
        }
        for &(s, e) in &out.vf_episodes {
            vf_eps.push(json!([s, e]));
        }
        for e in &out.episodes {
            episodes.push(json!([e.condition.name(), e.start, e.end]));
        }
    }
    out.clear();
    pipe.finish(&mut out);
    for e in &out.episodes {
        episodes.push(json!([e.condition.name(), e.start, e.end]));
    }

    // Reference: beat symbols in the strip, rhythm spans over the whole record.
    let (mut reference, mut spans) = (vec![], vec![]);
    let mut ref_beats: Vec<(i64, char)> = Vec::new();
    if let Ok(ann) = AnnotationFile::read(Path::new(&entry.ann_path(ann_ext(&entry.source)))) {
        for a in &ann.annotations {
            if aami(a.symbol).is_some() {
                ref_beats.push((a.sample, a.symbol));
                if a.sample >= 0 && in_strip(a.sample as u64) {
                    reference.push(json!([a.sample, a.symbol.to_string()]));
                }
            }
        }
    }
    // Rhythm is annotated in `.atr` even where the beats are elsewhere (AFDB).
    if let Ok(ann) = AnnotationFile::read(Path::new(&entry.ann_path("atr"))) {
        for (s, e, n) in rhythm_ref::named_spans(&ann, hdr.n_samples as i64) {
            spans.push(json!([s, e, n]));
        }
    }

    // The review queue: each morphology, its exemplar's waveform, and what the
    // annotators called the beats that joined it.
    let tol = (0.15 * fs) as i64;
    let mut by_cluster: std::collections::HashMap<u32, [u64; 5]> = Default::default();
    let mut j = 0usize;
    for &(s, _, k) in &all_beats {
        while j < ref_beats.len() && ref_beats[j].0 < s as i64 - tol {
            j += 1;
        }
        let sym = ref_beats
            .get(j)
            .filter(|r| (r.0 - s as i64).abs() <= tol)
            .and_then(|r| aami(r.1));
        let slot = match sym {
            Some(crate::beat_eval::Aami::N) => 0,
            Some(crate::beat_eval::Aami::S) => 1,
            Some(crate::beat_eval::Aami::V) => 2,
            Some(crate::beat_eval::Aami::F) => 3,
            _ => 4,
        };
        by_cluster.entry(k).or_default()[slot] += 1;
    }
    let half = (0.35 * fs) as i64;
    let clusters: Vec<Value> = pipe
        .morphologies()
        .iter()
        .take(opts.get_usize("clusters").unwrap_or(8))
        .map(|c| {
            let e = c.exemplar as i64;
            let wave: Vec<f32> = (e - half..e + half)
                .map(|i| sig.get(i.max(0) as usize).copied().unwrap_or(0.0))
                .collect();
            json!({
                "id": c.id, "count": c.count, "classed": c.classed,
                "reference": by_cluster.get(&c.id).copied().unwrap_or_default(),
                "exemplar": c.exemplar, "wave": wave
            })
        })
        .collect();

    let doc = json!({
        "record": format!("{}/{}", entry.source, entry.record),
        "fs": fs,
        "n_samples": hdr.n_samples,
        "from": from, "to": to,
        "signal": &sig[from..to],
        "beats": beats,
        "reference": reference,
        "rhythm_spans": spans,
        "quality": quality,
        "af": af,
        "vf": vf,
        "vf_episodes": vf_eps,
        "episodes": episodes,
        "clusters": clusters,
        "stages": pipe.stage_ids().iter().map(|(k, n)| format!("{k}={n}")).collect::<Vec<_>>(),
    });
    let path = opts.get_str("out").unwrap_or("intro.json");
    std::fs::write(
        path,
        serde_json::to_string(&doc).map_err(|e| err(e.to_string()))?,
    )?;
    eprintln!("wrote {path}");
    Ok(())
}
