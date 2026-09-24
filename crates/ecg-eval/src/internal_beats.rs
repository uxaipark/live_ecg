//! Beat classification on the internal patch corpus, against the analyst's
//! exhaustive review.
//!
//! This is the only evaluation here whose recordings come from the domain the
//! engine is built for: a commercial single-lead patch, worn for days, on
//! outpatients. Everything else in `PERFORMANCE.md` is clinical electrodes on
//! inpatients.
//!
//! # Why this reference is different from the corpus's other labels
//!
//! The corpus's ordinary labels are the device's output with an analyst's
//! corrections layered on, and an *uncorrected* beat in that arrangement is a
//! beat that may never have been looked at. It is therefore not evidence that
//! the device was right, which means specificity and false-positive rates
//! cannot be computed from it at all — only performance on the beats a human
//! chose to touch, which is a sample selected by its answer.
//!
//! Twenty-two sealed recordings were reviewed **exhaustively**. There, an
//! unedited beat means "seen and agreed", so every beat is evidence and the
//! ordinary rates are defined. That is the whole reason this file exists.
//!
//! # What it cannot measure
//!
//! Detection. The analyst reviewed the beats the *device* marked; a beat the
//! device never marked is absent from the list and invisible to review. So
//! these are classification figures, driven at the reference positions, and no
//! sensitivity here includes a beat the device missed. Claiming detection would
//! need an independent R-peak reference on the raw waveform, and there is none.
//!
//! The device's own pre-review judgement travels in the same file, so it is
//! scored beside ours on exactly the same beats. It is the number worth
//! beating: it is what the patients' reports were actually built from.

use crate::beat_eval::Aami;
use crate::manifest::RecordEntry;
use crate::patch_eval;
use crate::Opts;
use ecg_beats::{AtrialRun, BeatAnalyzer, BeatBank, BeatClass, BeatContext, BeatVerdict};
use ecg_pipeline::{PipelineConfig, Preprocessor};
use ecg_qrs::QrsEvent;
use ecg_quality::{Quality, QualityMonitor};
use ecg_rhythm::{AfDetector, RrConfig, RrSample, RrStream};
use ecg_zarr::ZarrArray;
use parquet::basic::Type as PhysicalType;
use parquet::column::reader::get_typed_column_reader;
use parquet::data_type::{BoolType, ByteArrayType, Int64Type};
use parquet::file::reader::{FileReader, RowGroupReader, SerializedFileReader};
use rayon::prelude::*;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct InternalBeat {
    pub sample: u64,
    /// The class after the analyst's exhaustive review. `None` is the device's
    /// artifact mark, which is not one of the AAMI classes and is scored in
    /// neither direction.
    pub truth: Option<Aami>,
    /// What the device said before anyone reviewed it.
    pub device: Option<Aami>,
    /// Outside a stretch the device called noise. The clinical report counts
    /// only these, so they are reported separately.
    pub qf_valid: bool,
    /// An analyst touched this beat - changed its class, moved it, or added
    /// it. Where a recording was not reviewed exhaustively, these are the only
    /// beats whose label is a person's rather than the device's.
    pub reviewed: bool,
}

fn class_of(symbol: &str) -> Option<Aami> {
    match symbol {
        "N" => Some(Aami::N),
        "S" => Some(Aami::S),
        "V" => Some(Aami::V),
        _ => None,
    }
}

/// Read one column of a row group, whole, as one entry per row.
///
/// The columns here are declared optional even where no row is actually null,
/// so the definition levels have to be read and the values expanded against
/// them. Taking the values buffer as-is is the trap: it holds only the present
/// entries, so one null anywhere shifts every later row against the other
/// columns and yields a clean-looking table of beats wearing each other's
/// labels.
fn column<T: parquet::data_type::DataType>(
    group: &dyn RowGroupReader,
    index: usize,
    rows: usize,
    expect: PhysicalType,
) -> Result<Vec<Option<T::T>>, String>
where
    T::T: Default + Clone,
{
    let descr = group.metadata().column(index).column_descr_ptr();
    if descr.max_rep_level() != 0 {
        return Err(format!("column {} repeats", descr.name()));
    }
    if descr.physical_type() != expect {
        return Err(format!(
            "column {} is {:?}, expected {expect:?}",
            descr.name(),
            descr.physical_type()
        ));
    }
    let max_def = descr.max_def_level();
    let reader = group.get_column_reader(index).map_err(|e| e.to_string())?;
    let mut typed = get_typed_column_reader::<T>(reader);
    // These are appended to, not written into. Handing over a pre-filled buffer
    // gets the call back with a plausible count and the original contents
    // untouched, which is a silent wrong answer rather than an error.
    let mut values: Vec<T::T> = Vec::with_capacity(rows);
    let mut defs: Vec<i16> = Vec::with_capacity(rows);
    let (read, n_values, _) = typed
        .read_records(
            rows,
            if max_def > 0 { Some(&mut defs) } else { None },
            None,
            &mut values,
        )
        .map_err(|e| e.to_string())?;
    if read != rows {
        return Err(format!(
            "column {} gave {read} of {rows} rows",
            descr.name()
        ));
    }
    if max_def == 0 {
        return Ok(values.into_iter().map(Some).collect());
    }
    let mut out = Vec::with_capacity(rows);
    let mut v = 0usize;
    for &d in defs.iter().take(rows) {
        if d == max_def {
            out.push(values.get(v).cloned());
            v += 1;
        } else {
            out.push(None);
        }
    }
    if v != n_values {
        return Err(format!(
            "column {}: {n_values} values against {v} present rows",
            descr.name()
        ));
    }
    Ok(out)
}

pub fn read_beats(path: &Path) -> Result<Vec<InternalBeat>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let reader = SerializedFileReader::new(file).map_err(|e| e.to_string())?;
    let schema = reader.metadata().file_metadata().schema_descr_ptr();
    let index = |name: &str| {
        (0..schema.num_columns())
            .find(|&i| schema.column(i).name() == name)
            .ok_or_else(|| format!("no column {name}"))
    };
    // `aami` is null on artifact beats, and a nullable column would have to be
    // expanded against its definition levels to stay aligned with the others.
    // `symbol_native` carries the same classes and is required, so the whole
    // question is avoided by reading that instead.
    let (i_sample, i_truth, i_device, i_qf, i_rev) = (
        index("sample")?,
        index("symbol_native")?,
        index("symbol_auto")?,
        index("qf_valid")?,
        index("reviewed")?,
    );

    let mut out = Vec::with_capacity(reader.metadata().file_metadata().num_rows() as usize);
    for g in 0..reader.num_row_groups() {
        let group = reader.get_row_group(g).map_err(|e| e.to_string())?;
        let rows = group.metadata().num_rows() as usize;
        let sample = column::<Int64Type>(&*group, i_sample, rows, PhysicalType::INT64)?;
        let truth = column::<ByteArrayType>(&*group, i_truth, rows, PhysicalType::BYTE_ARRAY)?;
        let device = column::<ByteArrayType>(&*group, i_device, rows, PhysicalType::BYTE_ARRAY)?;
        let qf = column::<BoolType>(&*group, i_qf, rows, PhysicalType::BOOLEAN)?;
        let rev = column::<BoolType>(&*group, i_rev, rows, PhysicalType::BOOLEAN)?;
        for i in 0..rows {
            // A beat with no position cannot be placed against the signal at
            // all, and guessing one would put a label on someone else's
            // complex.
            let Some(at) = sample[i] else {
                return Err(format!("row {i} of {} has no sample", path.display()));
            };
            let text = |v: &Option<parquet::data_type::ByteArray>| {
                v.as_ref()
                    .and_then(|b| b.as_utf8().ok().map(|s| s.to_string()))
                    .unwrap_or_default()
            };
            out.push(InternalBeat {
                sample: at.max(0) as u64,
                truth: class_of(&text(&truth[i])),
                device: class_of(&text(&device[i])),
                // A missing noise flag is not evidence that the stretch was
                // clean, so it counts as noise.
                qf_valid: qf[i].unwrap_or(false),
                reviewed: rev[i].unwrap_or(false),
            });
        }
    }
    out.sort_by_key(|b| b.sample);
    Ok(out)
}

/// Sensitivity, precision, specificity and false alarms for one class.
#[derive(Debug, Default, Clone, Copy)]
pub struct Score {
    pub tp: u64,
    pub fp: u64,
    pub fn_: u64,
    pub tn: u64,
}

impl Score {
    /// A beat the analyser declined to judge is not counted in either
    /// direction.
    ///
    /// This is the convention the public-corpus tables already use - EC57
    /// sensitivity over the beats that received a class, with coverage
    /// reported beside it - and the two tables are only comparable if they
    /// agree. Counting a withheld verdict as a miss reads 72 % ventricular
    /// sensitivity where the other convention reads 84 %, and the difference
    /// is one recording whose gain was never found.
    fn add(&mut self, truth: bool, called: bool) {
        match (truth, called) {
            (true, true) => self.tp += 1,
            (true, false) => self.fn_ += 1,
            (false, true) => self.fp += 1,
            (false, false) => self.tn += 1,
        }
    }

    /// For evaluations outside this module that score calls of their own.
    pub fn add_pub(&mut self, truth: bool, called: bool) {
        self.add(truth, called);
    }

    pub fn merge(&mut self, o: &Score) {
        self.tp += o.tp;
        self.fp += o.fp;
        self.fn_ += o.fn_;
        self.tn += o.tn;
    }

    pub fn n(&self) -> u64 {
        self.tp + self.fp + self.fn_ + self.tn
    }

    pub fn se(&self) -> f64 {
        ratio(self.tp, self.tp + self.fn_)
    }

    pub fn pp(&self) -> f64 {
        ratio(self.tp, self.tp + self.fp)
    }

    pub fn sp(&self) -> f64 {
        ratio(self.tn, self.tn + self.fp)
    }

    /// False positives per thousand beats, which is the number a reviewer
    /// experiences: a rate per beat is the same fact in units nobody feels.
    pub fn fp_per_1000(&self) -> f64 {
        1000.0 * self.fp as f64 / self.n().max(1) as f64
    }

    pub fn f1(&self) -> f64 {
        let (se, pp) = (self.se(), self.pp());
        if se + pp > 0.0 {
            2.0 * se * pp / (se + pp)
        } else {
            f64::NAN
        }
    }
}

fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

#[derive(Debug, Default, Clone)]
pub struct RecordResult {
    pub record: String,
    pub hours: f64,
    pub gain: f32,
    pub beats: u64,
    pub scored: u64,
    /// `[ours, device]` for V and for S, over all beats and over `qf_valid`.
    pub ours: [Score; 2],
    pub device: [Score; 2],
    pub ours_qf: [Score; 2],
    pub device_qf: [Score; 2],
    pub unclassified: u64,
    /// `[truth][reported]` over N, S, V and then "not classified", so the
    /// shape of the disagreement is visible rather than inferred from two
    /// rates.
    pub confusion: [[u64; 4]; 3],
    /// Beats judged while the rhythm detector held sustained fibrillation, and
    /// how many of those the review calls supraventricular. The class is not
    /// reported there by design, so this is the size of that decision on this
    /// corpus rather than a defect.
    pub fibrillating: u64,
    pub fibrillating_s: u64,
    /// Threshold-independent ranking, per record. Kept per record rather than
    /// pooled because a pooled figure over recordings of wildly different
    /// length is a statement about the longest patient: on one corpus here a
    /// pooled 0.155 and a per-record 0.500 came from the same beats.
    pub auc_v: f64,
    pub auc_s: f64,
    /// `(score, is_supraventricular)` for every scored beat, stride-sampled,
    /// so the operating curve can be drawn without re-running the corpus.
    pub curve: Vec<(f32, bool)>,
    /// Beats the analyser declined to judge, split by whether the *device*
    /// thought the stretch was clean. Where it did, the disagreement is ours.
    /// Morphologies, credited with the reviewed classes of their members.
    pub clusters: crate::cluster_eval::RecordClusters,
    /// Beats whose morphology the capacity bound dropped, by reviewed class.
    pub dropped_by_class: [u64; 3],
    /// Supraventricular runs the rhythm detector reported.
    pub sv_runs: u64,
    /// Onset evidence for reported runs the analyst disagrees with (0) and
    /// agrees with (1): whether the per-beat detector called one of the first
    /// three beats supraventricular, and the first beat's P-wave match with
    /// the beat before it.
    pub run_onsets: [Vec<(bool, Option<f32>)>; 2],
    /// The same queue with the dropped morphologies put back in it, as a
    /// consumer that keeps what the bank hands out would see it.
    pub clusters_with_dropped: crate::cluster_eval::RecordClusters,
    pub dropped_v_by_size: [u64; 5],
    pub unknown_s_clean: u64,
    pub unknown_s: u64,
    /// Which of the two conditions withheld the verdict.
    pub unknown_s_quality: u64,
    pub unknown_s_template: u64,
    /// Ventricular beats by whether the device called the stretch clean
    /// (`[noise, clean]`): total, our V, our N or F, our S, our unclassified,
    /// the device's V, analyst-touched, analyst-touched and our N.
    pub v_zone: [[u64; 8]; 2],
    /// Normal beats the same way: total, our V, the device's V, analyst-touched.
    pub n_zone: [[u64; 4]; 2],
    /// Per-record ventricular AUC over each zone's classified beats.
    pub auc_v_zone: [f64; 2],
    /// Ventricular score of each ventricular beat we called normal, by zone.
    pub v_missed_pv: [Vec<f32>; 2],
    /// Supraventricular beats by their place in a consecutive run: how many
    /// runs of each length, and how we do on the beat that opens a run against
    /// the ones that continue it.
    ///
    /// The distinction is the whole question. A premature atrial beat is
    /// premature *to* the rhythm it interrupts, so only the first beat of a
    /// run has an interval that says anything; inside a run the beats are
    /// regular with respect to each other and prematurity is not evidence any
    /// more. If the class is mostly runs, it is a rhythm rather than ectopy
    /// and a per-beat detector is being asked the wrong question.
    pub run_hist: [u64; 8],
    /// How well each beat's P wave matches the previous beat's, split by
    /// whether the two beats are the same class. If the representation carries
    /// which focus made the wave, a beat inside a run matches its predecessor
    /// and a beat at the edge of one does not.
    pub p_match: [Vec<f32>; 4],
    /// How well each beat's P wave fits a running template of P waves, by
    /// class. This is what the existing `p_ncc` feature measures, except on
    /// the P-anchored window instead of the rate's - and `p_ncc` separates the
    /// two classes at the median not at all (0.900 against 0.905).
    pub p_fit: [Vec<f32>; 3],
    /// The two-cluster template's margin - how much more this beat's P wave
    /// looks like the recording's rival atrial shape than its dominant one -
    /// by class, and how often a margin was available at all.
    pub p_margin: [Vec<f32>; 3],
    pub p_margin_absent: u64,
    pub onset: Score,
    pub inside: Score,
    /// What holding the onset's verdict across the run would buy, measured
    /// rather than assumed. An upper bound: it uses the reference's own run
    /// boundaries, so a real implementation has to find the end itself and
    /// cannot do better than this.
    pub propagated: Score,
    /// A stride-sampled slice of the feature vectors, by class, for comparing
    /// this domain against the one the models were fitted on.
    pub sample: Vec<(u8, ecg_beats::BeatFeatures)>,
    pub error: Option<String>,
}

/// One beat's verdict, with the two conditions that could have withheld it,
/// and the P wave it was judged on.
pub(crate) struct Judged {
    pub(crate) verdicts: Vec<(BeatVerdict, bool, bool)>,
    pub(crate) p_shapes: Vec<Option<ecg_beats::BeatVector>>,
    /// Every morphology the bank ended the recording with, as `(id, score)`.
    /// Scores are read at the end because a cluster's score is the median of
    /// its members, and the members are only all known then.
    pub(crate) clusters: Vec<(u32, f32)>,
    pub(crate) merges: u64,
    /// Where each morphology the capacity bound removed went: folded into
    /// another, or dropped outright.
    pub(crate) redirect: std::collections::HashMap<u32, Option<u32>>,
    /// Runs of supraventricular rhythm the rhythm detector reported.
    pub(crate) sv_runs: Vec<ecg_rhythm::SvRun>,
}

/// A ventricular model to score with in place of the one compiled in, and the
/// bar it has to clear. Used to judge a candidate fitted on this corpus before
/// anything is emitted into the engine.
pub struct VModel {
    pub model: crate::gbdt_train::Model,
    pub threshold: f32,
    /// A bar on the ensemble's raw score instead of its probability, when set.
    ///
    /// Needed because a model fitted with its negatives stride-sampled can be
    /// confident enough that the useful thresholds sit within a few parts in a
    /// hundred thousand of 1.0, where an `f32` probability runs out of room.
    /// The raw score is the same ranking with the room still in it.
    pub logit: Option<f32>,
}

impl VModel {
    pub fn load(path: &str, threshold: f32) -> std::io::Result<VModel> {
        let text = std::fs::read_to_string(path)?;
        let model = serde_json::from_str(&text)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(VModel {
            model,
            threshold,
            logit: None,
        })
    }
}

/// How far a score clears its bar, as a share of the room above it - the
/// bank's own arbitration, repeated here so a substituted model is judged by
/// the same rule as the shipped one.
fn margin(p: f32, t: f32) -> f32 {
    (p - t) / (1.0 - t).max(1e-6)
}

/// Drive the classifier at the reference beat positions, streaming the record.
///
/// Stops once every reference beat has been judged, so a caller can hand in
/// the first day of a recording and not pay for the other thirteen.
pub(crate) fn classify(
    array: &mut ZarrArray,
    cfg: &PipelineConfig,
    scale: f32,
    beats: &[InternalBeat],
    v_model: Option<&VModel>,
) -> Result<Judged, String> {
    let fs = array.fs();
    let mut pre = Preprocessor::new(cfg.preprocess);
    let mut qual = QualityMonitor::new(cfg.quality);
    let mut analyser = BeatAnalyzer::new(cfg.beats);
    let mut delin = ecg_beats::delineate::Delineator::new(cfg.delineate);
    let d = cfg.delineate;
    delin.set_delays(
        pre.group_delay_samples(d.qrs_ref_hz) + pre.qrs_group_delay_samples(d.qrs_ref_hz),
        pre.group_delay_samples(d.p_ref_hz) + pre.pt_group_delay_samples(d.p_ref_hz),
        pre.group_delay_samples(d.t_ref_hz) + pre.pt_group_delay_samples(d.t_ref_hz),
    );
    let mut rr = RrStream::new(RrConfig::new(fs));
    let mut af = AfDetector::new(fs, cfg.af);
    let mut pending: Option<RrSample> = None;
    let (mut prev_v, mut prev_s) = (false, false);
    let bank: BeatBank = cfg.bank;
    let mut atrial_run = AtrialRun::new();
    let mut sv_run = ecg_rhythm::SvRunDetector::new(cfg.sv_run);
    let mut sv_runs: Vec<ecg_rhythm::SvRun> = Vec::new();
    let mut morphology = ecg_beats::MorphologyBank::new(cfg.clusters);
    let mut redirect: std::collections::HashMap<u32, Option<u32>> =
        std::collections::HashMap::new();
    // Every morphology the recording produced: handed out when the bank was
    // full, or still held at the end.
    let mut produced: Vec<ecg_beats::Cluster> = Vec::new();

    let mut recent: [Option<u64>; 3] = [None; 3];
    let mut out = Vec::with_capacity(beats.len());
    let mut p_shapes: Vec<Option<ecg_beats::BeatVector>> = Vec::with_capacity(beats.len());
    let mut next = 0usize;
    let mut chunk: Vec<i32> = Vec::new();
    let mut n: u64 = 0;

    for c in 0..array.n_chunks() {
        if next >= beats.len() {
            break;
        }
        array.read_chunk(c, &mut chunk).map_err(|e| e.to_string())?;
        for &x in &chunk {
            let b = pre.process(x as f32 * scale);
            let q = qual.process(b.raw, b.clean, b.baseline, b.hf, b.qrs, b.saturated);
            let learn_ok = q.level(&cfg.quality) != Quality::Unusable;
            analyser.push_sample(b.clean, b.qrs, learn_ok);
            delin.push_sample(b.qrs, b.pt);
            rr.observe_quality(learn_ok);
            rr.observe_lead(q.flags & ecg_quality::flags::SATURATION == 0);

            // `<=` and not `==`: an equality test stalls for good the moment a
            // position is stepped over, and a stalled loop produces a record
            // with no verdicts rather than an error.
            while next < beats.len() && beats[next].sample <= n {
                let ev = QrsEvent {
                    sample: n,
                    amplitude: 0.0,
                    energy: 0.0,
                    margin: 1.0,
                    recovered: false,
                    interval_energy: 0.0,
                };
                recent = [recent[1], recent[2], Some(ev.sample)];
                let wave = match (recent[0], recent[1], recent[2]) {
                    (Some(a), Some(m), Some(z)) => delin.delineate(m, Some(m - a), Some(z - m)),
                    _ => None,
                };
                let interval = rr.push(&ev);
                if let Some(obs) = analyser.push_beat(&ev, wave.as_ref()) {
                    p_shapes.push(obs.p_shape);
                    let context = BeatContext {
                        fibrillating: af.sustained(ev.sample),
                    };
                    let mut verdict = bank.classify_in(&obs, context);
                    if let Some(vm) = v_model {
                        let x = verdict.features.vector();
                        verdict.p_ventricular = vm.model.probability(&x);
                        if let (Some(l), true) = (vm.logit, verdict.class != BeatClass::Unknown) {
                            // Ventricular first, as the bank breaks ties: a
                            // missed ventricular beat costs more than a
                            // mislabelled one of the other classes.
                            if vm.model.raw(&x) >= l {
                                verdict.class = BeatClass::V;
                            } else if verdict.class == BeatClass::V {
                                verdict.class = BeatClass::N;
                            }
                        } else if verdict.class != BeatClass::Unknown {
                            let candidates = [
                                (BeatClass::V, verdict.p_ventricular, vm.threshold),
                                (BeatClass::F, verdict.p_fusion, bank.fusion.threshold),
                                (
                                    BeatClass::S,
                                    verdict.p_supraventricular,
                                    bank.supraventricular.threshold,
                                ),
                            ];
                            let mut best: Option<(BeatClass, f32)> = None;
                            for (class, p, t) in candidates {
                                if p < t {
                                    continue;
                                }
                                let m = margin(p, t);
                                if best.map(|(_, bm)| m > bm).unwrap_or(true) {
                                    best = Some((class, m));
                                }
                            }
                            verdict.class = match best.map(|(c, _)| c) {
                                Some(BeatClass::S)
                                    if !bank.reports_supraventricular(
                                        verdict.p_supraventricular,
                                        context,
                                    ) =>
                                {
                                    BeatClass::N
                                }
                                Some(c) => c,
                                None => BeatClass::N,
                            };
                        }
                    }
                    if atrial_run.push(
                        verdict.class == BeatClass::S,
                        obs.p_shape.as_ref(),
                        &cfg.atrial_run,
                    ) && verdict.class == BeatClass::N
                    {
                        verdict.class = BeatClass::S;
                    }
                    // Every beat joins a morphology, as in the pipeline, with
                    // the delineated width of this beat when the delineation
                    // describes it.
                    let qrs_ms = wave
                        .as_ref()
                        .filter(|d| d.r == obs.sample)
                        .map(|d| d.qrs.duration_samples() as f32 * 1000.0 / fs as f32)
                        .unwrap_or(0.0);
                    verdict.cluster = morphology.push(&obs.vector, &verdict, qrs_ms).unwrap_or(0);
                    if let Some((gone, into)) = morphology.last_capacity_event {
                        redirect.insert(gone, into);
                    }
                    morphology.take_dropped(&mut produced);
                    let v = verdict.class == BeatClass::V;
                    let s = verdict.class == BeatClass::S;
                    if let Some(mut held) = pending.take() {
                        held.ventricular = prev_v || v;
                        held.supraventricular = prev_s || s;
                        held.atrial_coherence = verdict.features.p_ncc_prev;
                        held.p_axis = verdict.features.p_polarity;
                        af.push(&held);
                        if let Some(r) = sv_run.push(
                            held.rr_ms,
                            held.sample,
                            held.usable(),
                            af.sustained(held.sample),
                            s,
                        ) {
                            sv_runs.push(r);
                        }
                    }
                    prev_v = v;
                    prev_s = s;
                    out.push((verdict, obs.quality_ok, obs.template_ready));
                }
                if interval.is_some() {
                    pending = interval;
                }
                next += 1;
            }
            n += 1;
        }
    }
    produced.extend(morphology.clusters().iter().cloned());
    sv_runs.extend(sv_run.finish());
    Ok(Judged {
        verdicts: out,
        p_shapes,
        clusters: produced.iter().map(|c| (c.id, c.score())).collect(),
        merges: morphology.merges,
        redirect,
        sv_runs,
    })
}

pub fn analyse(entry: &RecordEntry, opts: &Opts, cfg: &PipelineConfig) -> RecordResult {
    analyse_with(entry, opts, cfg, None)
}

pub fn analyse_with(
    entry: &RecordEntry,
    opts: &Opts,
    cfg: &PipelineConfig,
    v_model: Option<&VModel>,
) -> RecordResult {
    let mut r = RecordResult {
        record: entry.record.clone(),
        hours: entry.n_samples as f64 / entry.fs / 3600.0,
        ..Default::default()
    };
    let beats_path = entry
        .signal_path()
        .with_file_name(format!("{}.beats.parquet", entry.record));
    let beats = match read_beats(&beats_path) {
        Ok(b) => b,
        Err(e) => {
            r.error = Some(e);
            return r;
        }
    };
    r.beats = beats.len() as u64;

    // The gain is not in the container, so it is recovered first. Everything
    // downstream of the front end is scale-free; the saturation flag is not.
    let cal = patch_eval::calibrate(entry, opts, cfg);
    if let Some(e) = cal.error {
        r.error = Some(e);
        return r;
    }
    r.gain = cal.gain;

    let mut array = match ZarrArray::open(&entry.signal_path()) {
        Ok(a) => a,
        Err(e) => {
            r.error = Some(e.to_string());
            return r;
        }
    };
    let judged = match classify(&mut array, cfg, 1.0 / cal.gain, &beats, v_model) {
        Ok(v) => v,
        Err(e) => {
            r.error = Some(e);
            return r;
        }
    };
    // A beat inside a reported supraventricular run is a supraventricular
    // call, unless it was called ventricular: the run is found from intervals
    // and says nothing about a beat whose own shape says ventricular.
    let mut verdicts = judged.verdicts.clone();
    {
        let runs = &judged.sv_runs;
        let mut k = 0usize;
        for (v, _, _) in verdicts.iter_mut() {
            while k < runs.len() && runs[k].end < v.sample {
                k += 1;
            }
            if v.class == BeatClass::N
                && runs
                    .get(k)
                    .is_some_and(|r| r.start <= v.sample && v.sample <= r.end)
            {
                v.class = BeatClass::S;
            }
        }
    }
    r.sv_runs = judged.sv_runs.len() as u64;
    // What each run looked like at its start, split by whether the analyst
    // agrees it was supraventricular: the per-beat detector's call on its
    // first beats, and how the first beat's P wave matched the one before.
    {
        let truth_at = |sample: u64| -> Option<Aami> {
            beats
                .binary_search_by_key(&sample, |b| b.sample)
                .ok()
                .and_then(|i| beats[i].truth)
        };
        for run in &judged.sv_runs {
            let i = judged.verdicts.partition_point(|v| v.0.sample < run.start);
            let inside: Vec<Aami> = judged.verdicts[i..]
                .iter()
                .take_while(|v| v.0.sample <= run.end)
                .filter_map(|v| truth_at(v.0.sample))
                .collect();
            if inside.is_empty() {
                continue;
            }
            let s_share =
                inside.iter().filter(|t| **t == Aami::S).count() as f64 / inside.len() as f64;
            let real = usize::from(s_share >= 0.5);
            let called = judged.verdicts[i..]
                .iter()
                .take(3)
                .any(|v| v.0.class == BeatClass::S);
            let p_match = match (
                i.checked_sub(1).and_then(|k| judged.p_shapes.get(k)),
                judged.p_shapes.get(i),
            ) {
                (Some(Some(a)), Some(Some(b))) => Some(a.ncc(b)),
                _ => None,
            };
            r.run_onsets[real].push((called, p_match));
        }
    }
    let verdicts = &verdicts;
    let p_shapes = &judged.p_shapes;
    let mut by_cluster: std::collections::HashMap<u32, [u64; 4]> = std::collections::HashMap::new();
    let mut cluster_totals = [0u64; 4];
    let mut dropped_members: std::collections::HashMap<u32, [u64; 3]> =
        std::collections::HashMap::new();
    // Matched by sample, not by position in the list. The analyser withholds a
    // verdict whenever it has no template or is missing a neighbouring
    // interval, and those gaps are scattered rather than confined to the start,
    // so counting from either end slides the labels along the beats. Measured:
    // the two are the same table when it is right and read 7 % ventricular
    // sensitivity when it is not, against a device that reads 99 % on the same
    // beats - which is what a misalignment looks like, and not what a
    // classifier looks like.
    let mut pairs_v: Vec<(f32, bool)> = Vec::new();
    let mut pairs_v_zone: [Vec<(f32, bool)>; 2] = [Vec::new(), Vec::new()];
    let mut pairs_s: Vec<(f32, bool)> = Vec::new();
    let stride = opts.get_usize("feature-stride").unwrap_or(500).max(1);
    let mut seen = 0usize;
    let mut prev_was_s = false;
    let mut prev_shape: Option<ecg_beats::BeatVector> = None;
    let mut p_template: Option<ecg_beats::BeatVector> = None;
    let mut p_two_cfg = crate::atrial_shapes::AtrialTemplateConfig::default();
    if let Some(v) = opts.get_f64("p-admit") {
        p_two_cfg.admit_ncc = v as f32;
    }
    if let Some(v) = opts.get_usize("p-bootstrap") {
        p_two_cfg.bootstrap_beats = v as u32;
    }
    let mut p_two = crate::atrial_shapes::AtrialTemplate::new();
    let mut held = false;
    let mut run = 0u64;
    let mut j = 0usize;
    for b in beats.iter() {
        let Some(truth) = b.truth else { continue };
        while j < verdicts.len() && verdicts[j].0.sample < b.sample {
            j += 1;
        }
        let Some((v, quality_ok, template_ready)) =
            verdicts.get(j).filter(|v| v.0.sample == b.sample)
        else {
            r.unclassified += 1;
            continue;
        };
        r.scored += 1;
        if v.class == BeatClass::Unknown {
            r.unclassified += 1;
        }
        let classified = v.class != BeatClass::Unknown;
        if let Some(Some(now)) = p_shapes.get(j) {
            // A running template of the patient's P wave, built the way the
            // engine would build it - from every beat with a readable P, not
            // from the ones the reference calls normal. Using the labels here
            // would measure a template no detector could have.
            let k = match truth {
                Aami::N => 0,
                Aami::S => 1,
                _ => 2,
            };
            if let Some(t) = p_template.as_ref() {
                r.p_fit[k].push(t.ncc(now));
            }
            match p_two.rival_margin(now, &p_two_cfg) {
                Some(m) => r.p_margin[k].push(m),
                None => r.p_margin_absent += 1,
            }
            p_two.update(now, *quality_ok, &p_two_cfg);
            p_template = Some(match p_template {
                None => *now,
                Some(mut t) => {
                    for (x, y) in t.v.iter_mut().zip(now.v.iter()) {
                        *x += 0.02 * (*y - *x);
                    }
                    t
                }
            });
        }
        if let (Some(now), Some(before)) = (p_shapes.get(j).and_then(|x| *x), prev_shape) {
            let k = usize::from(prev_was_s) * 2 + usize::from(truth == Aami::S);
            r.p_match[k].push(before.ncc(&now));
        }
        if let Some(Some(sh)) = p_shapes.get(j) {
            prev_shape = Some(*sh);
        }
        let called = v.class == BeatClass::S;
        if classified {
            let opens = truth == Aami::S && !prev_was_s;
            if truth == Aami::S {
                if opens {
                    r.onset.add(true, called);
                } else {
                    r.inside.add(true, called);
                }
            }
            // Held from the onset across the reference's own run.
            if opens {
                held = called;
            } else if truth != Aami::S {
                held = false;
            }
            r.propagated.add(
                truth == Aami::S,
                if truth == Aami::S { held } else { called },
            );
        }
        prev_was_s = truth == Aami::S;
        let ti = match truth {
            Aami::N => 0,
            Aami::S => 1,
            _ => 2,
        };
        let pi = match v.class {
            BeatClass::N | BeatClass::F => 0,
            BeatClass::S => 1,
            BeatClass::V => 2,
            BeatClass::Unknown => 3,
        };
        r.confusion[ti][pi] += 1;
        // Credit the reviewed class to the morphology the beat joined, which
        // is what a reviewer labelling that morphology would be labelling.
        if classified && v.cluster != 0 {
            // Follow the beat to the morphology that holds it now. A cluster
            // merged into another took its members with it; one that was
            // dropped took them nowhere, and those are counted as lost rather
            // than quietly left out.
            let resolved = crate::cluster_eval::resolve(&judged.redirect, v.cluster);
            match resolved {
                Some(id) => by_cluster.entry(id).or_default()[ti] += 1,
                None => {
                    r.dropped_by_class[ti] += 1;
                    // Keyed by the last id in the chain, which is the
                    // morphology that was actually dropped - and which the
                    // bank now hands out, so its members are credited to it.
                    let mut last = v.cluster;
                    while let Some(Some(n)) = judged.redirect.get(&last) {
                        last = *n;
                    }
                    dropped_members.entry(last).or_default()[ti] += 1;
                    by_cluster.entry(last).or_default()[ti] += 1;
                }
            }
        }
        if classified {
            cluster_totals[ti] += 1;
        }
        pairs_v.push((v.p_ventricular, truth == Aami::V));
        {
            let z = usize::from(b.qf_valid);
            if classified {
                pairs_v_zone[z].push((v.p_ventricular, truth == Aami::V));
            }
            match truth {
                Aami::V => {
                    let c = &mut r.v_zone[z];
                    c[0] += 1;
                    c[match v.class {
                        BeatClass::V => 1,
                        BeatClass::N | BeatClass::F => 2,
                        BeatClass::S => 3,
                        BeatClass::Unknown => 4,
                    }] += 1;
                    c[5] += u64::from(b.device == Some(Aami::V));
                    c[6] += u64::from(b.reviewed);
                    let ours_n = matches!(v.class, BeatClass::N | BeatClass::F);
                    c[7] += u64::from(b.reviewed && ours_n);
                    if ours_n {
                        r.v_missed_pv[z].push(v.p_ventricular);
                    }
                }
                Aami::N => {
                    let c = &mut r.n_zone[z];
                    c[0] += 1;
                    c[1] += u64::from(v.class == BeatClass::V);
                    c[2] += u64::from(b.device == Some(Aami::V));
                    c[3] += u64::from(b.reviewed);
                }
                _ => {}
            }
        }
        pairs_s.push((v.p_supraventricular, truth == Aami::S));
        seen += 1;
        if seen.is_multiple_of(stride) {
            r.sample.push((ti as u8, v.features));
            r.curve.push((v.p_supraventricular, truth == Aami::S));
        }
        if truth == Aami::S {
            run += 1;
        } else if run > 0 {
            let k = (run as usize).min(r.run_hist.len());
            r.run_hist[k - 1] += run;
            run = 0;
        }
        if truth == Aami::S && v.class == BeatClass::Unknown {
            r.unknown_s += 1;
            if b.qf_valid {
                r.unknown_s_clean += 1;
            }
            if !*quality_ok {
                r.unknown_s_quality += 1;
            }
            if !*template_ready {
                r.unknown_s_template += 1;
            }
        }
        if v.context.fibrillating {
            r.fibrillating += 1;
            if truth == Aami::S {
                r.fibrillating_s += 1;
            }
        }
        for (k, class) in [(0usize, Aami::V), (1, Aami::S)] {
            let want = truth == class;
            let ours = v.class
                == match class {
                    Aami::V => BeatClass::V,
                    _ => BeatClass::S,
                };
            let device = b.device == Some(class);
            // The device is scored over the same beats, so a beat we withheld
            // is left out of its figures too. Otherwise the comparison would
            // be between two different populations.
            if classified {
                r.ours[k].add(want, ours);
                r.device[k].add(want, device);
                if b.qf_valid {
                    r.ours_qf[k].add(want, ours);
                    r.device_qf[k].add(want, device);
                }
            }
        }
    }
    let dropped_ids: std::collections::HashSet<u32> = judged
        .redirect
        .iter()
        .filter(|(_, into)| into.is_none())
        .map(|(id, _)| *id)
        .collect();
    let rank = |include_dropped: bool| {
        let mut v: Vec<(f32, [u64; 4])> = judged
            .clusters
            .iter()
            .filter(|(id, _)| include_dropped || !dropped_ids.contains(id))
            .filter_map(|(id, score)| by_cluster.get(id).map(|c| (*score, *c)))
            .filter(|(_, c)| c.iter().sum::<u64>() > 0)
            .collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        v
    };
    let ranked = rank(false);
    r.clusters_with_dropped = crate::cluster_eval::RecordClusters {
        record: entry.record.clone(),
        ranked: rank(true),
        totals: cluster_totals,
        unjudged: r.unclassified,
        merges: judged.merges,
    };
    r.clusters = crate::cluster_eval::RecordClusters {
        record: entry.record.clone(),
        ranked,
        totals: cluster_totals,
        unjudged: r.unclassified,
        merges: judged.merges,
    };
    // Ventricular beats lost with a dropped morphology, by how many judged
    // beats that morphology held: a lone beat that never recurred is
    // fragmentation, a morphology of dozens is a shape the bank threw away.
    for c in dropped_members.values() {
        let n: u64 = c.iter().sum();
        let k = match n {
            1 => 0,
            2 => 1,
            3..=5 => 2,
            6..=20 => 3,
            _ => 4,
        };
        r.dropped_v_by_size[k] += c[2];
    }
    r.auc_v = crate::beat_eval::rank_auc(pairs_v);
    for (a, p) in r.auc_v_zone.iter_mut().zip(pairs_v_zone) {
        *a = crate::beat_eval::rank_auc(p);
    }
    r.auc_s = crate::beat_eval::rank_auc(pairs_s);
    r
}

fn line(name: &str, s: &Score) {
    println!(
        "{:<22} {:>9.1} {:>9.1} {:>9.3} {:>9.2} {:>9.3}",
        name,
        100.0 * s.se(),
        100.0 * s.pp(),
        100.0 * s.sp(),
        s.fp_per_1000(),
        s.f1()
    );
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    let entries: Vec<RecordEntry> = opts
        .select()?
        .into_iter()
        .filter(|e| opts.has("all-records") || e.flag("expert_eval"))
        .collect();
    println!(
        "internal beat classification: {} exhaustively reviewed records",
        entries.len()
    );
    let cfg = crate::qrs_eval::config_from(opts, 250.0);
    let v_model = match opts.get_str("v-model") {
        Some(path) => {
            let t = opts.get_f64("v-model-thr").unwrap_or(0.5) as f32;
            let mut vm = VModel::load(path, t)?;
            vm.logit = opts.get_f64("v-model-logit").map(|l| l as f32);
            // The same division `emit-model` applies, so a candidate is judged
            // with the scores it would ship with.
            if let Some(temp) = opts.get_f64("v-model-temperature") {
                let temp = temp as f32;
                vm.model.bias /= temp;
                for n in vm.model.nodes.iter_mut() {
                    if n.feature == crate::gbdt_train::LEAF {
                        n.value /= temp;
                    }
                }
            }
            match vm.logit {
                Some(l) => println!("ventricular model: {path} at raw score {l}"),
                None => println!("ventricular model: {path} at {t}"),
            }
            Some(vm)
        }
        None => None,
    };
    let mut rows: Vec<RecordResult> = entries
        .par_iter()
        .map(|e| analyse_with(e, opts, &cfg, v_model.as_ref()))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.beats));

    if opts.per_record {
        println!(
            "\n{:<18} {:>7} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}",
            "record", "hours", "beats", "gain", "unk %", "V Se", "V +P", "S Se", "S +P"
        );
        for r in &rows {
            if let Some(e) = &r.error {
                println!("{:<18} {:>7.1}   {e}", r.record, r.hours);
                continue;
            }
            println!(
                "{:<18} {:>7.1} {:>8} {:>8.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
                r.record,
                r.hours,
                r.beats,
                r.gain,
                100.0 * r.unclassified as f64 / r.scored.max(1) as f64,
                100.0 * r.ours[0].se(),
                100.0 * r.ours[0].pp(),
                100.0 * r.ours[1].se(),
                100.0 * r.ours[1].pp()
            );
        }
    }

    let failed: Vec<&RecordResult> = rows.iter().filter(|r| r.error.is_some()).collect();
    if !failed.is_empty() {
        println!("\nfailed ({}):", failed.len());
        for r in &failed {
            println!("  {:<18} {}", r.record, r.error.as_deref().unwrap_or(""));
        }
    }
    let ok: Vec<&RecordResult> = rows.iter().filter(|r| r.error.is_none()).collect();
    if ok.is_empty() {
        println!("no records scored");
        return Ok(());
    }
    let mut total = RecordResult::default();
    for r in &ok {
        total.beats += r.beats;
        total.scored += r.scored;
        total.unclassified += r.unclassified;
        total.fibrillating += r.fibrillating;
        total.fibrillating_s += r.fibrillating_s;
        total.hours += r.hours;
        for (a, b) in total
            .confusion
            .iter_mut()
            .flatten()
            .zip(r.confusion.iter().flatten())
        {
            *a += b;
        }
        total.sample.extend_from_slice(&r.sample);
        total.curve.extend_from_slice(&r.curve);
        total.unknown_s += r.unknown_s;
        total.unknown_s_clean += r.unknown_s_clean;
        total.unknown_s_quality += r.unknown_s_quality;
        total.unknown_s_template += r.unknown_s_template;
        for z in 0..2 {
            for (a, b) in total.v_zone[z].iter_mut().zip(r.v_zone[z].iter()) {
                *a += b;
            }
            for (a, b) in total.n_zone[z].iter_mut().zip(r.n_zone[z].iter()) {
                *a += b;
            }
            total.v_missed_pv[z].extend_from_slice(&r.v_missed_pv[z]);
        }
        total.onset.merge(&r.onset);
        total.inside.merge(&r.inside);
        total.propagated.merge(&r.propagated);
        for (a, b) in total.run_hist.iter_mut().zip(r.run_hist.iter()) {
            *a += b;
        }
        for (a, b) in total.p_match.iter_mut().zip(r.p_match.iter()) {
            a.extend_from_slice(b);
        }
        for (a, b) in total.p_fit.iter_mut().zip(r.p_fit.iter()) {
            a.extend_from_slice(b);
        }
        for (a, b) in total.p_margin.iter_mut().zip(r.p_margin.iter()) {
            a.extend_from_slice(b);
        }
        total.p_margin_absent += r.p_margin_absent;
        for k in 0..2 {
            total.ours[k].merge(&r.ours[k]);
            total.device[k].merge(&r.device[k]);
            total.ours_qf[k].merge(&r.ours_qf[k]);
            total.device_qf[k].merge(&r.device_qf[k]);
        }
    }
    println!("\n── against an exhaustive review ──────────────────────────────");
    println!(
        "records {}   {:.0} h   beats {}   matched {}   classified {:.2} %",
        ok.len(),
        total.hours,
        total.beats,
        total.scored,
        100.0 * (total.scored - total.unclassified) as f64 / total.scored.max(1) as f64
    );
    let runs: u64 = ok.iter().map(|r| r.sv_runs).sum();
    {
        println!("\nwhat reported runs looked like at their start:");
        for (k, name) in [(1usize, "analyst agrees"), (0, "analyst disagrees")] {
            let all: Vec<&(bool, Option<f32>)> =
                ok.iter().flat_map(|r| r.run_onsets[k].iter()).collect();
            if all.is_empty() {
                continue;
            }
            let called = all.iter().filter(|x| x.0).count();
            let mut pm: Vec<f32> = all.iter().filter_map(|x| x.1).collect();
            pm.sort_by(f32::total_cmp);
            let q = |p: f64| {
                pm.get(((p * (pm.len().max(1) - 1) as f64).round()) as usize)
                    .copied()
                    .unwrap_or(f32::NAN)
            };
            let below = |b: f32| {
                100.0 * pm.iter().filter(|x| **x < b).count() as f64 / pm.len().max(1) as f64
            };
            println!(
                "  {:<18} runs {:>6}   S called in first 3 beats {:>5.1} %   \
                 P match at onset p25 {:.2} median {:.2}   below 0.3: {:.1} %",
                name,
                all.len(),
                100.0 * called as f64 / all.len() as f64,
                q(0.25),
                q(0.5),
                below(0.3)
            );
        }
    }
    println!(
        "supraventricular runs reported  {}  ({:.1} per 24 h){}",
        runs,
        24.0 * runs as f64 / total.hours.max(1e-9),
        if cfg.sv_run.enabled {
            ""
        } else {
            "  - detector off (--domain patch or --svrun on)"
        }
    );
    println!(
        "prevalence   V {:.2} %   S {:.2} %",
        100.0 * (total.ours[0].tp + total.ours[0].fn_) as f64 / total.scored.max(1) as f64,
        100.0 * (total.ours[1].tp + total.ours[1].fn_) as f64 / total.scored.max(1) as f64
    );
    let med = |mut v: Vec<f64>| {
        v.retain(|x| x.is_finite());
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            f64::NAN
        } else {
            v[v.len() / 2]
        }
    };
    println!(
        "per-record AUC (threshold-independent)   ventricular {:.4}   supraventricular {:.4}",
        med(ok.iter().map(|r| r.auc_v).collect()),
        med(ok.iter().map(|r| r.auc_s).collect())
    );

    {
        println!("\nventricular beats, inside the device's noise and outside it:");
        println!(
            "{:>8} {:>9} {:>7} {:>7} {:>7} {:>9} {:>9} {:>9} {:>10} {:>9}",
            "zone",
            "V beats",
            "ours V",
            "ours N",
            "ours S",
            "withheld",
            "device V",
            "touched",
            "touched&N",
            "AUC"
        );
        for (z, name) in [(0usize, "noise"), (1, "clean")] {
            let c = &total.v_zone[z];
            let pc = |k: usize| 100.0 * c[k] as f64 / c[0].max(1) as f64;
            let aucs: Vec<f64> = ok
                .iter()
                .map(|r| r.auc_v_zone[z])
                .filter(|a| a.is_finite())
                .collect();
            println!(
                "{:>8} {:>9} {:>6.1}% {:>6.1}% {:>6.1}% {:>8.1}% {:>8.1}% {:>8.1}% {:>9.1}% {:>9.4}",
                name, c[0], pc(1), pc(2), pc(3), pc(4), pc(5), pc(6), pc(7), med(aucs)
            );
        }
        for (z, name) in [(0usize, "noise"), (1, "clean")] {
            let c = &total.n_zone[z];
            println!(
                "  normal beats in {name}: {}   ours V {:.3} %   device V {:.3} %   touched {:.1} %",
                c[0],
                100.0 * c[1] as f64 / c[0].max(1) as f64,
                100.0 * c[2] as f64 / c[0].max(1) as f64,
                100.0 * c[3] as f64 / c[0].max(1) as f64
            );
        }
        for (z, name) in [(0usize, "noise"), (1, "clean")] {
            let mut v = total.v_missed_pv[z].clone();
            v.sort_by(f32::total_cmp);
            if v.is_empty() {
                continue;
            }
            let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
            println!(
                "  ventricular score of V beats we called normal, {name}: p25 {:.3} median {:.3} p75 {:.3} p90 {:.3}",
                q(0.25), q(0.5), q(0.75), q(0.9)
            );
        }
    }

    println!(
        "supraventricular beats the analyser declined to judge  {}  \
         ({} where the device called the stretch clean; \
         {} for signal quality, {} for no template)",
        total.unknown_s, total.unknown_s_clean, total.unknown_s_quality, total.unknown_s_template
    );

    {
        let total_s: u64 = total.run_hist.iter().sum();
        println!("\nsupraventricular beats by the length of the run they sit in:");
        for (i, n) in total.run_hist.iter().enumerate() {
            let label = if i + 1 == total.run_hist.len() {
                format!("{}+", i + 1)
            } else {
                format!("{}", i + 1)
            };
            println!(
                "{:>6} beat run {:>12} beats  {:>6.1} %",
                label,
                n,
                100.0 * *n as f64 / total_s.max(1) as f64
            );
        }
        println!(
            "opens a run   Se {:.1} %   of {} beats\ncontinues one Se {:.1} %   of {} beats",
            100.0 * total.onset.se(),
            total.onset.tp + total.onset.fn_,
            100.0 * total.inside.se(),
            total.inside.tp + total.inside.fn_
        );
        println!(
            "holding the onset's verdict across the run would read \
             Se {:.1} %  +P {:.1} %  against {:.1} / {:.1} now",
            100.0 * total.propagated.se(),
            100.0 * total.propagated.pp(),
            100.0 * total.ours[1].se(),
            100.0 * total.ours[1].pp()
        );
    }

    {
        println!(
            "\nhow well a beat's P wave matches the one before it:\n\
             {:<22} {:>10} {:>8} {:>8} {:>8} {:>8}",
            "previous -> this", "n", "p10", "p25", "median", "p75"
        );
        for (k, name) in [
            (0usize, "normal -> normal"),
            (1, "normal -> supravent."),
            (2, "supravent. -> normal"),
            (3, "supravent. -> supra."),
        ] {
            let mut v = total.p_match[k].clone();
            if v.is_empty() {
                continue;
            }
            v.sort_by(f32::total_cmp);
            let q = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
            println!(
                "{:<22} {:>10} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                name,
                v.len(),
                q(0.10),
                q(0.25),
                q(0.50),
                q(0.75)
            );
        }
    }

    {
        println!(
            "\nhow well a beat's P wave fits the patient's running P template:\n\
             {:<22} {:>10} {:>8} {:>8} {:>8} {:>8}",
            "class", "n", "p10", "p25", "median", "p75"
        );
        let mut pairs: Vec<(f32, bool)> = Vec::new();
        for (k, name) in [
            (0usize, "normal"),
            (1, "supraventricular"),
            (2, "ventricular"),
        ] {
            let mut v = total.p_fit[k].clone();
            if v.is_empty() {
                continue;
            }
            if k < 2 {
                pairs.extend(v.iter().map(|&x| (x, k == 1)));
            }
            v.sort_by(f32::total_cmp);
            let q = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
            println!(
                "{:<22} {:>10} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                name,
                v.len(),
                q(0.10),
                q(0.25),
                q(0.50),
                q(0.75)
            );
        }
        if !pairs.is_empty() {
            println!(
                "  as a lone predictor of supraventricular, AUC {:.4}",
                1.0 - crate::beat_eval::rank_auc(pairs)
            );
        }
    }

    {
        println!(
            "\nhow much more this beat's P wave looks like the recording's rival\n\
             atrial shape than its dominant one ({} beats had no second shape):\n\
             {:<22} {:>10} {:>8} {:>8} {:>8} {:>8}",
            total.p_margin_absent, "class", "n", "p10", "p25", "median", "p75"
        );
        let mut pairs: Vec<(f32, bool)> = Vec::new();
        for (k, name) in [
            (0usize, "normal"),
            (1, "supraventricular"),
            (2, "ventricular"),
        ] {
            let mut v = total.p_margin[k].clone();
            if v.is_empty() {
                continue;
            }
            if k < 2 {
                pairs.extend(v.iter().map(|&x| (x, k == 1)));
            }
            v.sort_by(f32::total_cmp);
            let q = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
            println!(
                "{:<22} {:>10} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
                name,
                v.len(),
                q(0.10),
                q(0.25),
                q(0.50),
                q(0.75)
            );
        }
        if !pairs.is_empty() {
            println!(
                "  as a lone predictor of supraventricular, AUC {:.4}  \
                 (the one-template fit reads 0.398)",
                crate::beat_eval::rank_auc(pairs)
            );
        }
    }

    // What the ranking can buy, whatever the threshold. If the curve is poor
    // the operating point is not the problem and moving it only trades one
    // failure for the other.
    if !total.curve.is_empty() {
        let mut c = total.curve.clone();
        c.sort_by(|a, b| b.0.total_cmp(&a.0));
        let positives = c.iter().filter(|&&(_, p)| p).count() as f64;
        println!(
            "\nwhat the supraventricular ranking can buy, over {} sampled beats:",
            c.len()
        );
        println!(
            "{:>10} {:>9} {:>9} {:>9}",
            "score", "Se %", "+P %", "called"
        );
        let (mut tp, mut called) = (0.0f64, 0.0f64);
        let mut marks = [0.10, 0.25, 0.40, 0.53, 0.70, 0.90].to_vec();
        marks.reverse();
        for &(score, pos) in &c {
            called += 1.0;
            if pos {
                tp += 1.0;
            }
            while let Some(&m) = marks.last() {
                if tp / positives >= m {
                    println!(
                        "{:>10.4} {:>9.1} {:>9.1} {:>9.0}",
                        score,
                        100.0 * tp / positives,
                        100.0 * tp / called,
                        called
                    );
                    marks.pop();
                } else {
                    break;
                }
            }
            if marks.is_empty() {
                break;
            }
        }
    }

    println!("\nconfusion (rows = review, columns = reported):");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>12}",
        "", "N", "S", "V", "unclassified"
    );
    for (i, name) in [(0usize, "N"), (1, "S"), (2, "V")] {
        let row = total.confusion[i];
        println!(
            "{:>8} {:>10} {:>10} {:>10} {:>12}",
            name, row[0], row[1], row[2], row[3]
        );
    }
    println!(
        "\nbeats judged inside sustained fibrillation  {} ({:.1} %), \
         of which the review calls {} supraventricular",
        total.fibrillating,
        100.0 * total.fibrillating as f64 / total.scored.max(1) as f64,
        total.fibrillating_s
    );

    println!(
        "\n{:<22} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "every beat", "Se %", "+P %", "Sp %", "FP/1000", "F1"
    );
    line("V, this engine", &total.ours[0]);
    line("V, the device", &total.device[0]);
    line("S, this engine", &total.ours[1]);
    line("S, the device", &total.device[1]);
    println!(
        "\n{:<22} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "outside noise only", "Se %", "+P %", "Sp %", "FP/1000", "F1"
    );
    line("V, this engine", &total.ours_qf[0]);
    line("V, the device", &total.device_qf[0]);
    line("S, this engine", &total.ours_qf[1]);
    line("S, the device", &total.device_qf[1]);

    // The same evidence, published as shapes instead of beats. Per-beat
    // ventricular precision here is bounded by prevalence - 1.75 % - however
    // good the ranking, and the ranking is good (AUC 0.975).
    println!("\n── the ventricular review queue, patch corpus ─────────────────");
    let queue: Vec<crate::cluster_eval::RecordClusters> =
        ok.iter().map(|r| r.clusters.clone()).collect();
    crate::cluster_eval::report(&queue, false);
    println!("\n── the same, keeping what the bank hands out when it is full ──");
    let kept: Vec<crate::cluster_eval::RecordClusters> =
        ok.iter().map(|r| r.clusters_with_dropped.clone()).collect();
    crate::cluster_eval::report(&kept, false);
    let mut lost = [0u64; 3];
    for r in &ok {
        for (a, b) in lost.iter_mut().zip(r.dropped_by_class.iter()) {
            *a += b;
        }
    }
    let v_all: u64 = ok.iter().map(|r| r.clusters.totals[2]).sum();
    let mut by_size = [0u64; 5];
    for r in &ok {
        for (a, b) in by_size.iter_mut().zip(r.dropped_v_by_size.iter()) {
            *a += b;
        }
    }
    println!(
        "ventricular beats lost, by the size of the morphology dropped with them:\n  \
         1 beat {}   2 beats {}   3-5 {}   6-20 {}   more {}",
        by_size[0], by_size[1], by_size[2], by_size[3], by_size[4]
    );
    let n_all: u64 = ok.iter().map(|r| r.clusters.totals[0]).sum();
    println!(
        "\nlost with a dropped morphology: {:.1} % of ventricular beats, {:.2} % of normal ones",
        100.0 * lost[2] as f64 / v_all.max(1) as f64,
        100.0 * lost[0] as f64 / n_all.max(1) as f64
    );
    println!(
        "\nfor comparison, per beat: this engine V +P {:.1} %, the device {:.1} %",
        100.0 * total.ours[0].pp(),
        100.0 * total.device[0].pp()
    );

    if opts.has("dump-features") {
        println!("\nfeature percentiles by reviewed class, this corpus:");
        println!(
            "{:<16} {:<4} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "feature", "cls", "n", "p5", "p25", "median", "p75"
        );
        for (fi, name) in ecg_beats::BeatFeatures::NAMES.iter().enumerate() {
            for (ci, cname) in [(0u8, "N"), (1, "S"), (2, "V")] {
                let mut v: Vec<f32> = total
                    .sample
                    .iter()
                    .filter(|(c, _)| *c == ci)
                    .map(|(_, f)| f.vector()[fi])
                    .collect();
                if v.is_empty() {
                    continue;
                }
                v.sort_by(f32::total_cmp);
                let q = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
                println!(
                    "{:<16} {:<4} {:>9} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
                    name,
                    cname,
                    v.len(),
                    q(0.05),
                    q(0.25),
                    q(0.50),
                    q(0.75)
                );
            }
        }
    }
    Ok(())
}
