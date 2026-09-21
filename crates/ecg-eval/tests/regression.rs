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
    // Measured VEB 93.80/78.44, SVEB 55.69/37.36, AUC 0.9936 / 0.8486. The
    // same models without the atrial features read 93.88/79.64 and
    // 55.27/35.58, AUC 0.9941 / 0.8373: the P wave buys the supraventricular
    // detector about a point of AUC and two of precision, and costs the
    // ventricular one a little precision on this pooled set.
    assert!(
        100.0 * v_se >= 90.0,
        "VEB sensitivity regressed to {:.2} %",
        100.0 * v_se
    );
    assert!(
        100.0 * v_pp >= 74.0,
        "VEB precision regressed to {:.2} %",
        100.0 * v_pp
    );
    assert!(
        100.0 * s_se >= 52.0,
        "SVEB sensitivity regressed to {:.2} %",
        100.0 * s_se
    );
    // The detector ROCs are the threshold-independent guard: a change that moves
    // only the operating point is a decision, a change that moves these is a bug.
    assert!(
        auc_v >= 0.985,
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

    // Measured on 66 records: found 93.7-100 %, |median| 0-26 ms.
    let bounds: [(delin_eval::Mark, f64, f64); 7] = [
        (delin_eval::Mark::POnset, 88.0, 12.0),
        (delin_eval::Mark::PPeak, 88.0, 12.0),
        (delin_eval::Mark::POffset, 88.0, 24.0),
        (delin_eval::Mark::QrsOnset, 98.0, 8.0),
        (delin_eval::Mark::QrsOffset, 98.0, 8.0),
        (delin_eval::Mark::TPeak, 90.0, 20.0),
        (delin_eval::Mark::TOffset, 88.0, 36.0),
    ];
    for (mark, min_found, max_median) in bounds {
        let s = &total[mark as usize];
        let found = 100.0 * s.sensitivity();
        let median = s.median();
        eprintln!(
            "ludb TEST {:<11} found {found:.1} %, median {median:.1} ms over {} marks",
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
    }
}
