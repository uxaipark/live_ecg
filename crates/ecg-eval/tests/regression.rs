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
use ecg_eval::{af_eval, beat_eval, manifest, metrics, qrs_eval, Opts};

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
    // Measured 86.2 / 98.8, 33 of 34 episodes.
    assert!(se >= 82.0, "AF sensitivity regressed to {se:.3} %");
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
    // Measured VEB 81.4/66.1, SVEB 53.9/35.3, AUC 0.985 / 0.839.
    assert!(
        100.0 * v_se >= 78.0,
        "VEB sensitivity regressed to {:.2} %",
        100.0 * v_se
    );
    assert!(
        100.0 * v_pp >= 62.0,
        "VEB precision regressed to {:.2} %",
        100.0 * v_pp
    );
    assert!(
        100.0 * s_se >= 50.0,
        "SVEB sensitivity regressed to {:.2} %",
        100.0 * s_se
    );
    // The detector ROCs are the threshold-independent guard: a change that moves
    // only the operating point is a decision, a change that moves these is a bug.
    assert!(
        auc_v >= 0.97,
        "ventricular detector AUC regressed to {auc_v:.4}"
    );
    assert!(
        auc_s >= 0.80,
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
