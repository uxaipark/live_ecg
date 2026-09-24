//! Metric regression guards.
//!
//! These run the real pipeline over real records and assert lower bounds on the
//! numbers the reports quote. They exist because three of the defects found
//! while building this engine were silent: a patch that stopped matching after a
//! reformat, a loop that stalled on one negative annotation, a threshold that
//! was being swallowed by the argument parser. Every one of them left the code
//! compiling and the unit tests passing, and was caught only by someone noticing
//! a number had moved.
//!
//! Bounds sit a little below the measured values - far enough not to trip on
//! ordinary variation, close enough that a real regression cannot hide. They are
//! floors on a sealed test set, so tightening them means re-running the report.
//!
//! The corpora are not in the repository. Without them every test here reports
//! that it was skipped rather than passing vacuously, which is the distinction
//! that matters: a green suite must not mean "measured nothing".

use ecg_eval::beat_eval::Aami;
use ecg_eval::{af_eval, beat_eval, delin_eval, manifest, metrics, qrs_eval, Opts};

/// Build options from the same strings the command line would take.
fn opts(args: &[&str]) -> Opts {
    let mut v: Vec<String> = vec!["--manifest".into(), manifest_path()];
    v.extend(args.iter().map(|s| s.to_string()));
    Opts::parse(&v)
}

/// The internal patch corpus has its own manifest, and its own root: zarr
/// under `data/canonical` rather than WFDB under `data/raw`.
fn internal_opts(args: &[&str]) -> Opts {
    let mut v: Vec<String> = vec![
        "--manifest".into(),
        format!(
            "{}/../../manifests/internal.json",
            env!("CARGO_MANIFEST_DIR")
        ),
    ];
    v.extend(args.iter().map(|s| s.to_string()));
    Opts::parse(&v)
}

fn manifest_path() -> String {
    // Tests run with the crate as the working directory.
    format!(
        "{}/../../manifests/records.json",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// `None` when the corpora are not present, so a test can say it was skipped.
fn require_data(o: &Opts) -> Option<Vec<manifest::RecordEntry>> {
    match o.select() {
        Ok(v) if !v.is_empty() => Some(v),
        _ => {
            eprintln!("SKIPPED: no WFDB corpora. Set DEEP_ECG_RAW to enable.");
            None
        }
    }
}

#[test]
fn qrs_detection_holds_on_mitdb_test() {
    let o = opts(&["--zone", "TEST", "--sources", "mitdb"]);
    let Some(entries) = require_data(&o) else {
        return;
    };

    let mut total = metrics::DetectionScore::default();
    for r in qrs_eval::run_all(&entries, &o) {
        assert!(
            r.error.is_none(),
            "{}/{}: {:?}",
            r.source,
            r.record,
            r.error
        );
        total.merge(&r.score);
    }
    let (se, pp) = (100.0 * total.sensitivity(), 100.0 * total.ppv());
    eprintln!("mitdb TEST: Se {se:.3} %, +P {pp:.3} %");
    // Measured 99.496 / 99.717.
    assert!(se >= 99.30, "QRS sensitivity regressed to {se:.3} %");
    assert!(pp >= 99.50, "QRS precision regressed to {pp:.3} %");
}

#[test]
fn noise_detection_predicts_detector_error() {
    let o = opts(&["--zone", "ALL", "--sources", "nstdb"]);
    if require_data(&o).is_none() {
        return;
    }
    let auc = quality_auc(&o);
    eprintln!("nstdb: AUC for predicting a detector error = {auc:.4}");
    // Measured 0.960. This is the number that justifies the quality monitor
    // running ahead of the detector rather than beside it.
    assert!(auc >= 0.93, "quality/error AUC regressed to {auc:.4}");
}

fn quality_auc(o: &Opts) -> f64 {
    ecg_eval::quality_eval::error_auc(o).unwrap_or(f64::NAN)
}

#[test]
fn quality_agrees_with_human_annotation() {
    // Against four human annotators on real long-term recordings, rather than
    // against synthetic noise with known amplitude. The two corpora fail
    // differently, which is the reason for having both: this one is what showed
    // the flat-fraction term was scoring pristine signal worst.
    let o = opts(&["--zone", "DEV", "--sources", "butqdb", "--threads", "4"]);
    let Some(auc) = ecg_eval::butqdb::unusable_auc(&o) else {
        eprintln!("SKIPPED: no BUT QDB. Set DEEP_ECG_RAW to enable.");
        return;
    };
    eprintln!("butqdb DEV: score AUC for class 3 (unusable) against class 1 = {auc:.4}");
    // Measured 0.820 on the development half, 0.997 on the sealed half.
    assert!(
        auc >= 0.75,
        "agreement with human annotation regressed to {auc:.4}"
    );
}

#[test]
fn wave_legibility_separates_the_two_usable_classes() {
    // The distinction the quality monitor was never able to make, because it is
    // about the P and T waves and the monitor measures neither: on the same
    // seconds its own score separates class 2 from class 1 at 0.446, which is
    // worse than chance.
    let o = opts(&["--zone", "DEV", "--sources", "butqdb", "--threads", "4"]);
    let Some(auc) = ecg_eval::butqdb::legibility_auc(&o) else {
        eprintln!("SKIPPED: no BUT QDB. Set DEEP_ECG_RAW to enable.");
        return;
    };
    eprintln!("butqdb DEV: atrial coherence AUC for class 2 against class 1 = {auc:.4}");
    assert!(auc >= 0.72, "wave legibility regressed to {auc:.4}");
}

#[test]
fn af_detection_holds_end_to_end() {
    let o = opts(&["--zone", "TEST", "--sources", "afdb", "--beats", "detected"]);
    let Some(entries) = require_data(&o) else {
        return;
    };

    let mut c = af_eval::Confusion::default();
    let mut episodes_found = 0usize;
    let mut episodes_total = 0usize;
    for r in entries.iter().map(|e| af_eval::analyse(e, &o)) {
        assert!(
            r.error.is_none(),
            "{}/{}: {:?}",
            r.source,
            r.record,
            r.error
        );
        for (t, p) in r.truth.iter().zip(&r.predicted) {
            c.add(*t, *p);
        }
        let (found, total) = af_eval::episode_recall(&r, 30);
        episodes_found += found;
        episodes_total += total;
    }
    let (se, pp) = (100.0 * c.sensitivity(), 100.0 * c.ppv());
    eprintln!("afdb TEST end to end: Se {se:.3} %, +P {pp:.3} %, episodes {episodes_found}/{episodes_total}");
    // Measured 90.5 / 98.8, 33 of 34 episodes, at 2.22 false alarms per 24 h of
    // normal sinus signal. Without `atrial_coherence` the same records read
    // 86.1 / 98.8 at 6.39 - the guard moved on both axes at once, so the bound
    // here is only half of it and the false-alarm test below is the other half.
    assert!(se >= 87.0, "AF sensitivity regressed to {se:.3} %");
    assert!(pp >= 96.0, "AF precision regressed to {pp:.3} %");
    assert!(
        episodes_found * 100 >= episodes_total * 90,
        "AF episode recall regressed"
    );
}

#[test]
fn af_does_not_cry_wolf_on_normal_sinus() {
    let o = opts(&[
        "--zone",
        "TEST",
        "--sources",
        "nsrdb",
        "--beats",
        "detected",
        "--assume-af-free",
        "nsrdb",
    ]);
    let Some(entries) = require_data(&o) else {
        return;
    };

    // The median subject, not the mean: the distribution is heavy-tailed and one
    // subject with marked respiratory sinus arrhythmia dominates the average.
    let mut rates: Vec<f64> = entries
        .iter()
        .map(|e| af_eval::analyse(e, &o))
        .map(|r| af_eval::alarms_per_day(&r, 30))
        .collect();
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = rates[rates.len() / 2];
    eprintln!(
        "nsrdb TEST: median {median:.2} false alarms per 24 h over {} subjects",
        rates.len()
    );
    // Measured 0.00.
    assert!(
        median <= 1.0,
        "median false-alarm rate regressed to {median:.2} per 24 h"
    );

    // And the worst subject, which the median cannot see. This is where the
    // atrial feature actually shows: 14.64 false alarms per 24 h against 52.72
    // without it, on the one subject whose sinus arrhythmia is marked enough to
    // look like fibrillation on timing alone. The median was 0.00 both before
    // and after, so a guard watching only the median would have called a
    // three-and-a-half-fold change invisible. Two statistics, because one of
    // them is blind by design.
    let worst = *rates.last().unwrap();
    eprintln!("nsrdb TEST: worst subject {worst:.2} false alarms per 24 h");
    assert!(
        worst <= 20.0,
        "worst-subject false-alarm rate regressed to {worst:.2} per 24 h"
    );
}

#[test]
fn beat_classification_holds() {
    let o = opts(&[
        "--zone",
        "TEST",
        "--sources",
        "mitdb,svdb",
        "--beats",
        "reference",
    ]);
    if require_data(&o).is_none() {
        return;
    }
    let (m, auc_v, auc_s) = beat_eval::summarise(&o).expect("evaluation ran");
    let (v_se, v_pp) = m.class_metrics(2);
    let (s_se, s_pp) = m.class_metrics(1);
    eprintln!(
        "beats TEST: VEB {:.2}/{:.2}, SVEB {:.2}/{:.2}, AUC {auc_v:.4}/{auc_s:.4}",
        100.0 * v_se,
        100.0 * v_pp,
        100.0 * s_se,
        100.0 * s_pp
    );
    // Measured VEB 94.82/81.26, SVEB 53.03/47.62, AUC 0.9940 / 0.8525.
    //
    // The ventricular figures moved when the template stopped taking the most
    // frequent morphology for the conducted one: 93.80/78.44 before. The gain
    // is much larger on INCART, which is where the failure lives and which this
    // guard does not cover - it is 12-lead, and one `--lead` cannot be right
    // for three corpora at once.
    //
    // The supraventricular figures moved when the class stopped being reported
    // inside sustained fibrillation: 54.85/37.22 before. Both directions are
    // guarded, because the change trades one for the other and a guard on the
    // precision alone would be satisfied by a detector that reported nothing.
    assert!(
        100.0 * v_se >= 92.0,
        "VEB sensitivity regressed to {:.2} %",
        100.0 * v_se
    );
    assert!(
        100.0 * v_pp >= 77.0,
        "VEB precision regressed to {:.2} %",
        100.0 * v_pp
    );
    assert!(
        100.0 * s_se >= 52.0,
        "SVEB sensitivity regressed to {:.2} %",
        100.0 * s_se
    );
    assert!(
        100.0 * s_pp >= 45.0,
        "SVEB precision regressed to {:.2} %",
        100.0 * s_pp
    );
    // The detector ROCs are the threshold-independent guard: a change that moves
    // only the operating point is a decision, a change that moves these is a bug.
    assert!(
        auc_v >= 0.990,
        "ventricular detector AUC regressed to {auc_v:.4}"
    );
    assert!(
        auc_s >= 0.83,
        "supraventricular detector AUC regressed to {auc_s:.4}"
    );
}

#[test]
fn episode_detection_holds() {
    use ecg_rhythm::Condition;
    let o = opts(&["--zone", "TEST", "--sources", "mitdb"]);
    let Some(scores) = ecg_eval::rhythm_eval::summarise(&o) else {
        eprintln!("SKIPPED: no WFDB corpora. Set DEEP_ECG_RAW to enable.");
        return;
    };
    let get = |c: Condition| {
        let i = Condition::ALL.iter().position(|x| *x == c).unwrap();
        (100.0 * scores[i].sensitivity(), 100.0 * scores[i].ppv())
    };
    for c in [
        Condition::Pause,
        Condition::Asystole,
        Condition::Bradycardia,
        Condition::Tachycardia,
        Condition::Bigeminy,
    ] {
        let (se, pp) = get(c);
        eprintln!("episodes, {:<24} Se {se:.2} %  +P {pp:.2} %", c.name());
    }

    // Asystole is the one that has to hold. It was structurally unreportable
    // twice - once gated behind "physiological interval", once behind the
    // quality monitor calling its own flatness a dead lead - and both times
    // every other number looked fine.
    let (se, pp) = get(Condition::Asystole);
    assert!(se >= 95.0, "asystole sensitivity regressed to {se:.2} %");
    assert!(pp >= 90.0, "asystole precision regressed to {pp:.2} %");

    let (se, pp) = get(Condition::Pause);
    assert!(se >= 95.0, "pause sensitivity regressed to {se:.2} %");
    assert!(pp >= 88.0, "pause precision regressed to {pp:.2} %");

    for (c, min_se, min_pp) in [
        (Condition::Bradycardia, 96.0, 92.0),
        (Condition::Tachycardia, 94.0, 96.0),
        (Condition::Bigeminy, 65.0, 92.0),
    ] {
        let (se, pp) = get(c);
        assert!(
            se >= min_se,
            "{} sensitivity regressed to {se:.2} %",
            c.name()
        );
        assert!(
            pp >= min_pp,
            "{} precision regressed to {pp:.2} %",
            c.name()
        );
    }
}

#[test]
fn fibrillation_detection_leaves_normal_rhythm_alone() {
    // The fibrillation detector is not accurate enough to alarm on (see
    // `ecg_rhythm::vf`), but it must not fire on ordinary rhythm - otherwise the
    // "do not trust the beats here" flag it exists to raise is worthless.
    let o = opts(&["--zone", "TEST", "--sources", "nsrdb"]);
    let Some(sp) = ecg_eval::vf_eval::specificity(&o) else {
        eprintln!("SKIPPED: no WFDB corpora. Set DEEP_ECG_RAW to enable.");
        return;
    };
    eprintln!(
        "fibrillation on 270 h of normal sinus: specificity {:.3} %",
        100.0 * sp
    );
    assert!(
        sp >= 0.998,
        "fibrillation specificity on normal rhythm regressed to {sp:.5}"
    );
}

#[test]
fn fibrillation_suppression_leaves_other_rhythms_alone() {
    // Withholding the beat-derived analysis deletes true findings, so it has to
    // fire only on fibrillation. Driving it from the reporting threshold cost
    // 95% of one AFDB record's atrial-fibrillation windows; this pins that it
    // touches nothing on corpora that contain no fibrillation at all.
    for source in ["mitdb", "afdb", "nsrdb", "svdb"] {
        let o = opts(&["--zone", "TEST", "--sources", source]);
        let Some(entries) = require_data(&o) else {
            return;
        };
        let mut suppressed = 0u64;
        let mut total = 0u64;
        for e in entries.iter().take(4) {
            suppressed += ecg_eval::vf_eval::suppressed_samples(e, &o).unwrap_or(0);
            total += e.n_samples as u64;
        }
        let share = suppressed as f64 / total.max(1) as f64;
        eprintln!("{source}: {:.4} % of samples withheld", 100.0 * share);
        // Measured: zero on MIT-BIH, AFDB and the Supraventricular corpus, and
        // 0.018% on Normal Sinus. None of them contains fibrillation, so any
        // material amount here is the detector misfiring on a rhythm it should
        // have left alone - which is what took AFDB sensitivity from 86% to 34%.
        assert!(
            share < 0.001,
            "{source} had {:.3} % of its analysis withheld",
            100.0 * share
        );
    }
}

#[test]
fn every_aami_symbol_maps() {
    // EC57 Table 1. A symbol silently failing to map would shrink the reference
    // population and flatter every rate computed from it.
    for (sym, want) in [
        ('N', Aami::N),
        ('L', Aami::N),
        ('R', Aami::N),
        ('e', Aami::N),
        ('j', Aami::N),
        ('A', Aami::S),
        ('a', Aami::S),
        ('J', Aami::S),
        ('S', Aami::S),
        ('V', Aami::V),
        ('E', Aami::V),
        ('F', Aami::F),
        ('/', Aami::Q),
        ('f', Aami::Q),
        ('Q', Aami::Q),
    ] {
        assert_eq!(beat_eval::aami(sym), Some(want), "symbol {sym:?}");
    }
    for sym in ['+', '~', '|', '[', ']', '!', 'x', 'p', 't'] {
        assert_eq!(
            beat_eval::aami(sym),
            None,
            "non-beat symbol {sym:?} must not map"
        );
    }
}

/// The patch corpus carries no gain, and the engine has exactly one threshold
/// that needs one.
///
/// This holds both halves, because only the pair is evidence. At unit gain the
/// quality monitor must reject nearly everything - a twenty-count QRS read as
/// twenty millivolts saturates the front end on every beat - and at the gain
/// recovered from the detector's own beat amplitudes it must accept nearly
/// everything. A guard on the second alone would be satisfied by a monitor
/// that had stopped checking.
#[test]
fn the_patch_corpus_is_readable_once_its_gain_is_recovered() {
    use ecg_eval::patch_eval;

    let o = internal_opts(&[
        "--sources",
        "atheart-backup",
        "--zone",
        "TEST",
        "--limit",
        "12",
        "--probes",
        "6",
        "--probe-s",
        "600",
    ]);
    let Ok(rows) = patch_eval::summarise(&o) else {
        eprintln!("SKIPPED: no internal corpus. Set DEEP_ECG_CANONICAL to enable.");
        return;
    };
    if rows.is_empty() {
        eprintln!("SKIPPED: no internal corpus.");
        return;
    }
    let ok: Vec<&patch_eval::Calibration> = rows.iter().filter(|r| r.error.is_none()).collect();
    assert_eq!(ok.len(), rows.len(), "a record failed to calibrate");

    let share = |f: &dyn Fn(&patch_eval::Calibration) -> f64| {
        let mut v: Vec<f64> = ok.iter().map(|r| f(r)).collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let at_one = share(&|r| 100.0 * r.usable_s / r.probed_s.max(1e-9));
    let at_gain = share(&|r| 100.0 * r.usable_scaled_s / r.probed_s.max(1e-9));
    let mut gains: Vec<f32> = ok.iter().map(|r| r.gain).collect();
    gains.sort_by(f32::total_cmp);
    eprintln!(
        "patch TEST: usable {at_one:.1} % at unit gain, {at_gain:.1} % at the gain; \
         counts per millivolt {:.1} to {:.1}",
        gains[0],
        gains[gains.len() - 1]
    );
    assert!(
        at_one <= 25.0,
        "the front end accepted {at_one:.1} % of raw counts read as millivolts, \
         so this test is no longer measuring the thing it was written for"
    );
    assert!(
        at_gain >= 85.0,
        "only {at_gain:.1} % of the patch corpus survives its own gain"
    );
    // A single constant would be wrong by the width of this spread, which is
    // why the gain is estimated per record rather than configured.
    assert!(
        gains[gains.len() - 1] / gains[0].max(1e-6) >= 3.0,
        "the gain stopped varying between records; a constant would now do"
    );
}

/// On the patch corpus the ventricular class is found and not precise: per
/// beat, 33 % precision against the device's 90 %, at a ranking AUC of 0.975.
/// That is the arithmetic of a 1.75 % prevalence, and the review queue is the
/// answer to it. This holds the queue's precision near the device's, and holds
/// it well clear of the per-beat figure, because the gap between the two is
/// the entire reason the queue exists.
#[test]
fn the_patch_review_queue_holds() {
    let o = internal_opts(&[
        "--sources",
        "atheart-backup",
        "--zone",
        "TEST",
        "--probes",
        "6",
        "--probe-s",
        "600",
    ]);
    let Ok(entries) = o.select() else {
        eprintln!("SKIPPED: no internal corpus. Set DEEP_ECG_CANONICAL to enable.");
        return;
    };
    let entries: Vec<_> = entries
        .into_iter()
        .filter(|e| e.flag("expert_eval"))
        .collect();
    if entries.is_empty() {
        eprintln!("SKIPPED: no internal corpus.");
        return;
    }
    let cfg = ecg_eval::qrs_eval::config_from(&o, 250.0);
    use rayon::prelude::*;
    let rows: Vec<_> = entries
        .par_iter()
        .map(|e| ecg_eval::internal_beats::analyse(e, &o, &cfg))
        .filter(|r| r.error.is_none())
        .collect();
    let (mut tp, mut fp, mut total_v, mut beat_tp, mut beat_fp) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for r in &rows {
        total_v += r.clusters.totals[2];
        beat_tp += r.ours[0].tp;
        beat_fp += r.ours[0].fp;
        for (score, c) in &r.clusters.ranked {
            if *score >= 0.99 {
                tp += c[2];
                fp += c[0] + c[1] + c[3];
            }
        }
    }
    let se = 100.0 * tp as f64 / total_v.max(1) as f64;
    let pp = 100.0 * tp as f64 / (tp + fp).max(1) as f64;
    let beat_pp = 100.0 * beat_tp as f64 / (beat_tp + beat_fp).max(1) as f64;
    eprintln!(
        "patch TEST review queue: Se {se:.2} %, +P {pp:.2} % against {beat_pp:.2} % per beat"
    );
    // Measured 59.38 / 83.82 against 33.4 per beat.
    assert!(pp >= 80.0, "review-queue precision fell to {pp:.2} %");
    assert!(se >= 55.0, "review-queue sensitivity fell to {se:.2} %");
    assert!(
        pp >= beat_pp + 40.0,
        "the queue is no longer much better than asking per beat: {pp:.2} against {beat_pp:.2}"
    );
}

/// The patch preset: the patch bank's ventricular ensemble, and
/// supraventricular runs found by the rhythm. The ensemble takes per-beat
/// ventricular false positives from thirty per thousand beats to under two and
/// pays in sensitivity; the runs take supraventricular sensitivity from 8 % to
/// 48 % and pay in false calls. Both directions are held for both classes, so
/// neither guard is satisfied by a detector that has stopped firing.
#[test]
fn the_patch_preset_holds() {
    let o = internal_opts(&[
        "--sources",
        "atheart-backup",
        "--zone",
        "TEST",
        "--probes",
        "6",
        "--probe-s",
        "600",
        "--domain",
        "patch",
    ]);
    let Ok(entries) = o.select() else {
        eprintln!("SKIPPED: no internal corpus. Set DEEP_ECG_CANONICAL to enable.");
        return;
    };
    let entries: Vec<_> = entries
        .into_iter()
        .filter(|e| e.flag("expert_eval"))
        .collect();
    if entries.is_empty() {
        eprintln!("SKIPPED: no internal corpus.");
        return;
    }
    let cfg = ecg_eval::qrs_eval::config_from(&o, 250.0);
    use rayon::prelude::*;
    let rows: Vec<_> = entries
        .par_iter()
        .map(|e| ecg_eval::internal_beats::analyse(e, &o, &cfg))
        .filter(|r| r.error.is_none())
        .collect();
    let (mut v, mut sv) = (
        ecg_eval::internal_beats::Score::default(),
        ecg_eval::internal_beats::Score::default(),
    );
    for r in &rows {
        v.merge(&r.ours[0]);
        sv.merge(&r.ours[1]);
    }
    let (se, pp, fp) = (100.0 * v.se(), 100.0 * v.pp(), v.fp_per_1000());
    let (sse, spp) = (100.0 * sv.se(), 100.0 * sv.pp());
    eprintln!(
        "patch preset TEST: V Se {se:.2} % +P {pp:.2} % ({fp:.2} false per 1000); \
         S Se {sse:.2} % +P {spp:.2} %"
    );
    // Measured V 55.7 / 84.7 / 1.83 and S 48.2 / 49.8, against the device's
    // 88.1 / 90.1 / 1.76 and 62.1 / 91.5.
    assert!(pp >= 80.0, "patch ventricular precision fell to {pp:.2} %");
    assert!(
        fp <= 2.5,
        "patch ventricular false positives rose to {fp:.2} per 1000"
    );
    assert!(
        se >= 52.0,
        "patch ventricular sensitivity fell to {se:.2} %"
    );
    assert!(
        sse >= 44.0,
        "patch supraventricular sensitivity fell to {sse:.2} %"
    );
    assert!(
        spp >= 45.0,
        "patch supraventricular precision fell to {spp:.2} %"
    );
}

/// The rules that decide when *not* to report an asystole must never be the
/// reason one is missed.
///
/// Two of them exist - the interval must be quiet, and the electrode must be
/// attached - and both were added to suppress false alarms, which is exactly
/// the kind of change that pays for itself in the currency of missed findings
/// without anyone noticing. This holds them to zero on 1,961 hours, and holds
/// the control at 14 of 14 on the corpus that is annotated beat by beat.
///
/// It deliberately does not guard the long-term corpus's *sensitivity*. That
/// number is 9.2 % and it is a measurement of the annotations rather than of
/// the engine: see the census's own header.
#[test]
fn nothing_is_lost_to_the_rules_that_suppress_asystole() {
    use ecg_eval::asystole_eval;

    let control = opts(&["--zone", "TEST", "--sources", "mitdb"]);
    if require_data(&control).is_none() {
        return;
    }
    let census = |o: &ecg_eval::Opts| {
        let mut total = asystole_eval::Census::default();
        for e in o.select().expect("manifest") {
            if let Some((_, c)) = asystole_eval::analyse(&e, o) {
                total.merge(&c);
            }
        }
        total
    };

    let c = census(&control);
    eprintln!(
        "mitdb TEST asystole: {} of {} reported",
        c.reported, c.reference
    );
    assert_eq!(
        c.reported, c.reference,
        "an asystole went missing on the corpus that is annotated beat by beat"
    );

    let long = opts(&["--zone", "TRAIN", "--sources", "ltafdb"]);
    if require_data(&long).is_none() {
        return;
    }
    let c = census(&long);
    eprintln!(
        "ltafdb: {} silences, {} reported, {} with a beat detected inside, \
         {} gated on the electrode, {} gated on energy",
        c.reference, c.reported, c.beats_inside, c.gated_lead, c.gated_energy
    );
    assert_eq!(
        c.gated_energy, 0,
        "the quiet-interval rule started suppressing asystoles"
    );
    assert_eq!(
        c.gated_lead, 0,
        "the electrode rule started suppressing asystoles"
    );
}

#[test]
fn wave_delineation_holds() {
    // LUDB's sealed half. The guard is on the *median* error and on how often
    // each wave is found at all, not on the mean and standard deviation: a
    // handful of gross mismatches moves those two without changing what the
    // delineator does on the beats it gets right, and a bound that a few
    // outliers can trip is a bound that gets loosened rather than investigated.
    let o = opts(&["--zone", "TEST", "--sources", "ludb"]);
    let Some(entries) = require_data(&o) else {
        return;
    };

    let mut total: [delin_eval::MarkScore; 7] = Default::default();
    for e in &entries {
        if let Ok(s) = delin_eval::analyse(e, &o) {
            for (t, r) in total.iter_mut().zip(s.iter()) {
                t.errors.extend_from_slice(&r.errors);
                t.missed += r.missed;
            }
        }
    }

    // Measured on 66 records: found 93.7-100 %, |median| 0-12 ms, inside the
    // CSE tolerance 54.4-86.4 %.
    //
    // The share inside tolerance is guarded beside the median because the two
    // boundary marks are now calibrated against the annotators' definition of
    // where a wave ends. A calibration can be undone two ways - by the bias
    // drifting, which the median sees, and by the spread widening around a
    // bias that still reads zero, which only this sees.
    let bounds: [(delin_eval::Mark, f64, f64, f64); 7] = [
        (delin_eval::Mark::POnset, 88.0, 12.0, 52.0),
        (delin_eval::Mark::PPeak, 88.0, 12.0, 80.0),
        (delin_eval::Mark::POffset, 88.0, 12.0, 62.0),
        (delin_eval::Mark::QrsOnset, 98.0, 8.0, 60.0),
        (delin_eval::Mark::QrsOffset, 98.0, 8.0, 48.0),
        (delin_eval::Mark::TPeak, 90.0, 20.0, 76.0),
        (delin_eval::Mark::TOffset, 88.0, 12.0, 65.0),
    ];
    for (mark, min_found, max_median, min_within) in bounds {
        let s = &total[mark as usize];
        let found = 100.0 * s.sensitivity();
        let median = s.median();
        let within = 100.0 * s.within(mark.tolerance_ms());
        eprintln!(
            "ludb TEST {:<11} found {found:.1} %, median {median:.1} ms, \
             inside tolerance {within:.1} % over {} marks",
            mark.name(),
            s.errors.len()
        );
        assert!(
            found >= min_found,
            "{} found on only {found:.1} % of reference marks",
            mark.name()
        );
        assert!(
            median.abs() <= max_median,
            "{} median error moved to {median:.1} ms",
            mark.name()
        );
        assert!(
            within >= min_within,
            "{} inside tolerance fell to {within:.1} %",
            mark.name()
        );
    }
}

#[test]
fn the_dominant_beat_is_not_just_the_most_frequent_one() {
    // INCART's sealed half contains what the training corpora do not: patients
    // whose recording is half ventricular, where "most frequent" anchors the
    // template on the ectopic beat and inverts every morphology feature at
    // once. Record I43 is 51 % ventricular and the detector used to find 2.6 %
    // of it. Lead II, because a chest patch approximates lead II and the
    // pooled figure at lead I measures the lead choice.
    let o = opts(&[
        "--zone",
        "TEST",
        "--sources",
        "incartdb",
        "--beats",
        "reference",
        "--lead",
        "1",
    ]);
    if require_data(&o).is_none() {
        return;
    }
    let (m, auc_v, _) = beat_eval::summarise(&o).expect("evaluation ran");
    let (v_se, v_pp) = m.class_metrics(2);
    eprintln!(
        "incart TEST lead II: VEB {:.2}/{:.2}, AUC {auc_v:.4}",
        100.0 * v_se,
        100.0 * v_pp
    );
    // Measured 88.57/88.30, AUC 0.9733. Before the width rule: 77.80/79.58,
    // AUC 0.9575.
    assert!(
        100.0 * v_se >= 85.0,
        "VEB sensitivity on INCART regressed to {:.2} %",
        100.0 * v_se
    );
    assert!(
        100.0 * v_pp >= 84.0,
        "VEB precision on INCART regressed to {:.2} %",
        100.0 * v_pp
    );
    assert!(
        auc_v >= 0.965,
        "INCART ventricular AUC regressed to {auc_v:.4}"
    );
}

#[test]
fn the_fusion_detector_knows_something() {
    // Guarded on the AUC rather than on sensitivity, and the distinction is the
    // finding. A fusion beat fires the ventricular detector too - it is half a
    // ventricular beat - and clears its threshold by a wider margin, so
    // arbitration hands it to V whatever this detector says. The ranking is
    // what this class can be held to today; the operating point is not.
    let o = opts(&[
        "--zone",
        "TEST",
        "--sources",
        "mitdb",
        "--beats",
        "reference",
    ]);
    if require_data(&o).is_none() {
        return;
    }
    let (m, _, _) = beat_eval::summarise(&o).expect("evaluation ran");
    let (f_se, f_pp) = m.metrics_over(3, 4);
    let auc = beat_eval::fusion_auc(&o).expect("evaluation ran");
    eprintln!(
        "mitdb TEST: fusion AUC {auc:.4}, at the shipped threshold {:.2}/{:.2}",
        100.0 * f_se,
        100.0 * f_pp
    );
    // Measured AUC 0.8328, 22.22/33.21 at threshold 0.95.
    assert!(auc >= 0.78, "fusion detector AUC regressed to {auc:.4}");
}

#[test]
fn idioventricular_rhythm_is_found_where_it_is_annotated() {
    // Episode recall against the annotator's own `(IVR` spans, not
    // second-by-second agreement and not precision. Both of those are bounded
    // by the ventricular detector: this condition is a subset of the
    // ventricular runs, so it cannot be more precise than they are, and on
    // long-term ambulatory data they are five per cent precise. What this guard
    // holds is that the episodes are found at all.
    let o = opts(&["--zone", "TRAIN", "--sources", "ltafdb", "--threads", "8"]);
    if require_data(&o).is_none() {
        return;
    }
    let Some(scores) = ecg_eval::rhythm_eval::annotator_scores(&o) else {
        eprintln!("SKIPPED: no long-term corpus.");
        return;
    };
    let i = ecg_rhythm::Condition::ALL
        .iter()
        .position(|c| *c == ecg_rhythm::Condition::Idioventricular)
        .unwrap();
    let s = &scores[i];
    eprintln!(
        "ltafdb TRAIN: idioventricular episodes found {} / {}",
        s.ref_found, s.ref_episodes
    );
    // Measured 78 of 135. Before the run was rated over its own intervals it
    // was 0 of 135: the rate could not be computed across the pause the escape
    // rhythm was escaping from, so the condition was unreportable outright.
    assert!(
        s.ref_episodes > 0 && s.ref_found * 100 >= s.ref_episodes * 40,
        "idioventricular episode recall regressed to {} / {}",
        s.ref_found,
        s.ref_episodes
    );
}

#[test]
fn no_subject_appears_on_both_sides_of_the_split() {
    // The zone column is per record, and several of these corpora are the same
    // recordings under different names: 104 of QT's 105 records are taken from
    // seven other databases, the noise-stress records are MIT-BIH 118 and 119
    // with noise added, and BUT QDB's identifiers are subject and session. A
    // record-level split can be perfectly disjoint and still put the same
    // patient on both sides, which is the leak that does not look like one.
    //
    // This checks the identity that actually matters: wherever two corpora hold
    // the same recording, both copies must be in the same zone.
    let o = opts(&["--zone", "ALL", "--sources", "ALL"]);
    let Some(entries) = require_data(&o) else {
        return;
    };
    use std::collections::HashMap;
    let zone: HashMap<(String, String), String> = entries
        .iter()
        .map(|e| ((e.source.clone(), e.record.clone()), e.zone.clone()))
        .collect();

    // QT names its records `sel<id>` or `sele<id>`, where `<id>` is the
    // original. The noise-stress records are `<id>e<snr>`.
    let mut checked = 0usize;
    let mut leaks: Vec<String> = Vec::new();
    for e in &entries {
        let bodies: Vec<String> = match e.source.as_str() {
            "qtdb" => {
                let b = e.record.trim_start_matches("sel");
                vec![
                    b.to_string(),
                    b.trim_start_matches('e').to_string(),
                    format!("e{b}"),
                    b.trim_start_matches('0').to_string(),
                ]
            }
            "nstdb" => vec![e.record.split('e').next().unwrap_or("").to_string()],
            _ => continue,
        };
        for src in ["mitdb", "nsrdb", "edb", "sddb", "stdb", "svdb", "ltdb"] {
            if let Some(b) = bodies
                .iter()
                .find(|b| zone.contains_key(&(src.to_string(), (*b).clone())))
            {
                checked += 1;
                let theirs = &zone[&(src.to_string(), b.clone())];
                if *theirs != e.zone {
                    leaks.push(format!(
                        "{}/{} is {} but {}/{} is {}",
                        e.source, e.record, e.zone, src, b, theirs
                    ));
                }
                break;
            }
        }
    }
    eprintln!(
        "cross-corpus recording reuse: {checked} pairs checked, {} leaks",
        leaks.len()
    );
    assert!(
        checked > 100,
        "only {checked} shared recordings found; the name matching broke"
    );
    assert!(
        leaks.is_empty(),
        "the same recording is on both sides of the split:\n{}",
        leaks.join("\n")
    );
}

#[test]
fn a_pause_is_not_reported_for_a_beat_we_missed() {
    // Episode precision on the long-term corpus, which is the only one long
    // enough to show this: at 1.25 beats a second, 99.9 % detection sensitivity
    // means one missed beat every thirteen minutes, and every one of them
    // presents as a doubled interval. Before the interval was required to be
    // quiet, that cost 117 false pauses per patient-day over 1,961 hours.
    let o = opts(&["--zone", "TRAIN", "--sources", "ltafdb", "--threads", "8"]);
    if require_data(&o).is_none() {
        return;
    }
    let Some(scores) = ecg_eval::rhythm_eval::summarise(&o) else {
        eprintln!("SKIPPED: no long-term corpus.");
        return;
    };
    let i = ecg_rhythm::Condition::ALL
        .iter()
        .position(|c| *c == ecg_rhythm::Condition::Pause)
        .unwrap();
    let s = &scores[i];
    let ppv = 100.0 * s.ppv();
    let se = 100.0 * s.sensitivity();
    let false_per_day = (s.rep_episodes - s.rep_correct) as f64 / 1960.6 * 24.0;
    eprintln!(
        "ltafdb TRAIN: pause Se {se:.2} %, +P {ppv:.2} %, {false_per_day:.1} false per patient-day"
    );
    // Measured 95.25 / 89.18 and 4.9 per patient-day; 96.37 / 25.50 and 116.9
    // without the silence requirement.
    assert!(se >= 92.0, "pause sensitivity regressed to {se:.2} %");
    assert!(ppv >= 85.0, "pause precision regressed to {ppv:.2} %");
    assert!(
        false_per_day <= 10.0,
        "false pauses rose to {false_per_day:.1} per patient-day"
    );
}

#[test]
fn the_review_queue_beats_the_alarm_it_replaces() {
    // The ventricular findings cannot be made precise as alarms: runs occupy
    // 0.034 % of the long-term corpus, and at any specificity a single-lead
    // classifier reaches, the false positives outnumber the true ones. What can
    // be made precise is the question asked a few dozen times instead of eight
    // million: is this *shape* ventricular.
    //
    // Guarded on the sealed arrhythmia corpus. Against the episode alarm on the
    // same records - 65.9 % sensitivity at 9.2 % precision, 220 false episodes
    // per patient-day - the queue reads 96.1 % at 87.2 % for nine clusters.
    let o = opts(&["--zone", "TEST", "--sources", "mitdb", "--threads", "8"]);
    if require_data(&o).is_none() {
        return;
    }
    let results: Vec<ecg_eval::cluster_eval::RecordClusters> = o
        .select()
        .unwrap()
        .iter()
        .filter(|e| !ecg_eval::beat_eval::is_paced(e))
        .filter_map(|e| ecg_eval::cluster_eval::analyse(e, &o).ok())
        .collect();
    assert!(!results.is_empty(), "no records produced clusters");

    let bar = 0.99f32;
    let (mut tp, mut fp, mut total_v, mut n) = (0u64, 0u64, 0u64, 0usize);
    for r in &results {
        total_v += r.totals[2];
        for (score, c) in &r.ranked {
            if *score >= bar {
                n += 1;
                tp += c[2];
                fp += c[0] + c[1] + c[3];
            }
        }
    }
    let se = 100.0 * tp as f64 / total_v.max(1) as f64;
    let pp = 100.0 * tp as f64 / (tp + fp).max(1) as f64;
    let per_record = n as f64 / results.len() as f64;
    eprintln!(
        "mitdb TEST review queue: Se {se:.2} %, +P {pp:.2} %, {per_record:.1} clusters per record"
    );
    // Measured 96.05 / 87.16 over 8.7 clusters per record.
    assert!(
        se >= 92.0,
        "review-queue sensitivity regressed to {se:.2} %"
    );
    assert!(pp >= 80.0, "review-queue precision regressed to {pp:.2} %");
    assert!(
        per_record <= 16.0,
        "the queue grew to {per_record:.1} clusters per record"
    );
}

#[test]
fn electrode_failure_is_not_reported_on_attached_electrodes() {
    // There is no lead-off label in any corpus here, so this guard is one-sided
    // by necessity: it bounds the false positives and says nothing about the
    // sensitivity, which is measured by construction in the pipeline's own
    // tests instead.
    //
    // These are clinical recordings with electrodes attached. A detector that
    // claims minutes of electrode failure on them is wrong whatever it does on
    // a patch.
    let o = opts(&[
        "--zone",
        "TEST",
        "--sources",
        "mitdb,nsrdb,svdb",
        "--threads",
        "8",
    ]);
    let Some(entries) = require_data(&o) else {
        return;
    };
    let all: Vec<ecg_eval::leadoff_eval::Census> = entries
        .iter()
        .filter_map(|e| ecg_eval::leadoff_eval::analyse(e, &o).ok())
        .collect();
    assert!(!all.is_empty());
    let hours: f64 = all.iter().map(|c| c.hours).sum();
    let claimed: f64 = all.iter().map(|c| c.rail_s + c.open_s).sum();
    let share = 100.0 * claimed / (hours * 3600.0);
    let touched = all.iter().filter(|c| c.episodes > 0).count();
    eprintln!(
        "lead-off on {:.0} h of attached-electrode signal: {share:.4} % claimed, {touched} of {} records",
        hours,
        all.len()
    );
    // Measured 0.1693 % over 288.8 h, all of it in one 25-hour ambulatory
    // record (Normal Sinus 16272) that this project already knows is the
    // hardest one it has.
    assert!(
        share <= 0.5,
        "lead-off claims {share:.4} % of clinical signal"
    );
    assert!(
        touched <= 3,
        "lead-off reports on {touched} of {} clinical records",
        all.len()
    );
}

#[test]
fn pacing_is_found_without_crying_wolf() {
    // Two figures and they are not symmetric. On records that contain pacing,
    // precision; on the overwhelming majority that do not, the false-call rate,
    // because this runs on all of them.
    //
    // The rule is argued rather than fitted: exactly one record in the training
    // zone contains paced beats, so a threshold tuned on it would be tuned on
    // one patient.
    let o = opts(&[
        "--zone",
        "TEST",
        "--sources",
        "mitdb",
        "--include-paced",
        "--threads",
        "8",
    ]);
    let Some(entries) = require_data(&o) else {
        return;
    };
    let all: Vec<ecg_eval::pacing_eval::PacingScore> = entries
        .iter()
        .filter_map(|e| ecg_eval::pacing_eval::analyse(e, &o).ok())
        .collect();
    assert!(!all.is_empty());
    let (paced, plain): (Vec<_>, Vec<_>) = all.iter().partition(|s| s.reference > 0);

    let tp: usize = paced.iter().map(|s| s.tp).sum();
    let called: usize = paced.iter().map(|s| s.called).sum();
    let reference: usize = paced.iter().map(|s| s.reference).sum();
    let pp = 100.0 * tp as f64 / called.max(1) as f64;
    let se = 100.0 * tp as f64 / reference.max(1) as f64;

    let judged: usize = plain.iter().map(|s| s.judged).sum();
    let false_calls: usize = plain.iter().map(|s| s.called).sum();
    let rate = 100.0 * false_calls as f64 / judged.max(1) as f64;
    eprintln!(
        "mitdb TEST pacing: Se {se:.2} %, +P {pp:.2} % on {} paced records; \
         {rate:.4} % of {judged} unpaced beats called paced",
        paced.len()
    );
    // Measured 52.56 / 100.00, and 0.1262 % of 46,743 unpaced beats.
    //
    // Precision is guarded and sensitivity is not, because sensitivity is set
    // by how wide the paced complex happens to be: the same rule at a 110 ms
    // bar reads 81 % here and calls 9.3 % of unpaced beats paced, and the
    // records it fires on are the bundle branch blocks. Without the pacing
    // spike - half a millisecond wide, against 2.8 ms per sample - a paced
    // complex and a bundle-branch-block complex are the same object.
    assert!(pp >= 95.0, "pacing precision regressed to {pp:.2} %");
    assert!(
        rate <= 1.0,
        "pacing calls {rate:.4} % of unpaced beats, which is crying wolf"
    );
}
