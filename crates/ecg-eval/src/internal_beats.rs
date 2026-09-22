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
use ecg_beats::{BeatAnalyzer, BeatBank, BeatClass, BeatContext, BeatVerdict};
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
        return Err(format!("column {} gave {read} of {rows} rows", descr.name()));
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
    let (i_sample, i_truth, i_device, i_qf) = (
        index("sample")?,
        index("symbol_native")?,
        index("symbol_auto")?,
        index("qf_valid")?,
    );

    let mut out = Vec::with_capacity(reader.metadata().file_metadata().num_rows() as usize);
    for g in 0..reader.num_row_groups() {
        let group = reader.get_row_group(g).map_err(|e| e.to_string())?;
        let rows = group.metadata().num_rows() as usize;
        let sample = column::<Int64Type>(&*group, i_sample, rows, PhysicalType::INT64)?;
        let truth = column::<ByteArrayType>(&*group, i_truth, rows, PhysicalType::BYTE_ARRAY)?;
        let device = column::<ByteArrayType>(&*group, i_device, rows, PhysicalType::BYTE_ARRAY)?;
        let qf = column::<BoolType>(&*group, i_qf, rows, PhysicalType::BOOLEAN)?;
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
    fn add(&mut self, truth: bool, called: bool) {
        match (truth, called) {
            (true, true) => self.tp += 1,
            (true, false) => self.fn_ += 1,
            (false, true) => self.fp += 1,
            (false, false) => self.tn += 1,
        }
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
    pub error: Option<String>,
}

/// Drive the classifier at the reference beat positions, streaming the record.
fn classify(
    array: &mut ZarrArray,
    cfg: &PipelineConfig,
    scale: f32,
    beats: &[InternalBeat],
) -> Result<Vec<BeatVerdict>, String> {
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

    let mut recent: [Option<u64>; 3] = [None; 3];
    let mut out = Vec::with_capacity(beats.len());
    let mut next = 0usize;
    let mut chunk: Vec<i32> = Vec::new();
    let mut n: u64 = 0;

    for c in 0..array.n_chunks() {
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
                    let verdict = bank.classify_in(
                        &obs,
                        BeatContext {
                            fibrillating: af.sustained(ev.sample),
                        },
                    );
                    let v = verdict.class == BeatClass::V;
                    let s = verdict.class == BeatClass::S;
                    if let Some(mut held) = pending.take() {
                        held.ventricular = prev_v || v;
                        held.supraventricular = prev_s || s;
                        held.atrial_coherence = verdict.features.p_ncc_prev;
                        held.p_axis = verdict.features.p_polarity;
                        af.push(&held);
                    }
                    prev_v = v;
                    prev_s = s;
                    out.push(verdict);
                }
                if interval.is_some() {
                    pending = interval;
                }
                next += 1;
            }
            n += 1;
        }
    }
    Ok(out)
}

pub fn analyse(entry: &RecordEntry, opts: &Opts, cfg: &PipelineConfig) -> RecordResult {
    let mut r = RecordResult {
        record: entry.record.clone(),
        hours: entry.n_samples as f64 / entry.fs / 3600.0,
        ..Default::default()
    };
    let beats_path = entry.signal_path().with_file_name(format!(
        "{}.beats.parquet",
        entry.record
    ));
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
    let verdicts = match classify(&mut array, cfg, 1.0 / cal.gain, &beats) {
        Ok(v) => v,
        Err(e) => {
            r.error = Some(e);
            return r;
        }
    };
    // Matched by sample, not by position in the list. The analyser withholds a
    // verdict whenever it has no template or is missing a neighbouring
    // interval, and those gaps are scattered rather than confined to the start,
    // so counting from either end slides the labels along the beats. Measured:
    // the two are the same table when it is right and read 7 % ventricular
    // sensitivity when it is not, against a device that reads 99 % on the same
    // beats - which is what a misalignment looks like, and not what a
    // classifier looks like.
    let mut j = 0usize;
    for b in beats.iter() {
        let Some(truth) = b.truth else { continue };
        while j < verdicts.len() && verdicts[j].sample < b.sample {
            j += 1;
        }
        let Some(v) = verdicts.get(j).filter(|v| v.sample == b.sample) else {
            r.unclassified += 1;
            continue;
        };
        r.scored += 1;
        if v.class == BeatClass::Unknown {
            r.unclassified += 1;
        }
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
            r.ours[k].add(want, ours);
            r.device[k].add(want, device);
            if b.qf_valid {
                r.ours_qf[k].add(want, ours);
                r.device_qf[k].add(want, device);
            }
        }
    }
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
    let mut rows: Vec<RecordResult> = entries
        .par_iter()
        .map(|e| analyse(e, opts, &cfg))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.beats));

    if opts.per_record {
        println!(
            "\n{:<18} {:>7} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "record", "hours", "beats", "gain", "V Se", "V +P", "S Se", "S +P"
        );
        for r in &rows {
            if let Some(e) = &r.error {
                println!("{:<18} {:>7.1}   {e}", r.record, r.hours);
                continue;
            }
            println!(
                "{:<18} {:>7.1} {:>8} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1}",
                r.record,
                r.hours,
                r.beats,
                r.gain,
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
        for (a, b) in total.confusion.iter_mut().flatten().zip(r.confusion.iter().flatten()) {
            *a += b;
        }
        for k in 0..2 {
            total.ours[k].merge(&r.ours[k]);
            total.device[k].merge(&r.device[k]);
            total.ours_qf[k].merge(&r.ours_qf[k]);
            total.device_qf[k].merge(&r.device_qf[k]);
        }
    }
    println!("\n── against an exhaustive review ──────────────────────────────");
    println!(
        "records {}   {:.0} h   beats {}   scored {}   unclassified {}",
        ok.len(),
        total.hours,
        total.beats,
        total.scored,
        total.unclassified
    );
    println!(
        "prevalence   V {:.2} %   S {:.2} %",
        100.0 * (total.ours[0].tp + total.ours[0].fn_) as f64 / total.scored.max(1) as f64,
        100.0 * (total.ours[1].tp + total.ours[1].fn_) as f64 / total.scored.max(1) as f64
    );
    println!("\nconfusion (rows = review, columns = reported):");
    println!("{:>8} {:>10} {:>10} {:>10} {:>12}", "", "N", "S", "V", "unclassified");
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

    Ok(())
}
