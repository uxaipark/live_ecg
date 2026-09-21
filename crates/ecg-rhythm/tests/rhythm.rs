//! The rhythm detectors are defined by rules, so they are tested against
//! hand-built beat sequences where the right answer is not in doubt. A corpus
//! tells you the rate is wrong; a sequence tells you which rule is wrong.

use ecg_rhythm::{Beat, Condition, RhythmBank, RhythmConfig, RhythmEpisode, RrSample};

const FS: f64 = 250.0;

fn sample(t_ms: f32, rr_ms: f32) -> RrSample {
    RrSample {
        sample: (t_ms as f64 * FS / 1000.0) as u64,
        rr_ms,
        physiological: (200.0..=3000.0).contains(&rr_ms),
        quality_ok: true,
        lead_ok: true,
        amplitude: 1.0,
        ventricular: false,
        supraventricular: false,
        atrial_coherence: 0.0,
        qrs_ms: 90.0,
        p_axis: 1.0,
        continuous: true,
    }
}

/// Feed `(rr_ms, class)` beats and collect the episodes that close.
fn run(beats: &[(f32, Beat)]) -> Vec<RhythmEpisode> {
    let mut bank = RhythmBank::new(RhythmConfig::new(FS));
    let mut out = Vec::new();
    let mut t = 0.0f32;
    for &(rr, c) in beats {
        t += rr;
        bank.push(&sample(t, rr), c, &mut out);
    }
    bank.finish(&mut out);
    out
}

fn saw(out: &[RhythmEpisode], c: Condition) -> bool {
    out.iter().any(|e| e.condition == c)
}

/// `n` normal beats at `bpm`.
fn sinus(n: usize, bpm: f32) -> Vec<(f32, Beat)> {
    vec![(60_000.0 / bpm, Beat::Normal); n]
}

#[test]
fn a_long_interval_is_a_pause_and_a_longer_one_is_asystole() {
    let mut b = sinus(12, 70.0);
    b.push((2500.0, Beat::Normal));
    let out = run(&b);
    assert!(saw(&out, Condition::Pause));
    assert!(
        !saw(&out, Condition::Asystole),
        "2.5 s is a pause, not asystole"
    );

    let mut b = sinus(12, 70.0);
    b.push((5000.0, Beat::Normal));
    let out = run(&b);
    assert!(saw(&out, Condition::Asystole));
    // Asystole is also a pause: a consumer filtering for pauses must see it.
    assert!(saw(&out, Condition::Pause));
}

#[test]
fn a_rate_must_last_to_be_reported() {
    // Forty beats per minute for a minute.
    assert!(saw(&run(&sinus(40, 40.0)), Condition::Bradycardia));
    // The same rate for four beats is not an episode.
    let mut b = sinus(20, 70.0);
    b.extend(sinus(4, 40.0));
    b.extend(sinus(20, 70.0));
    assert!(!saw(&run(&b), Condition::Bradycardia));
}

#[test]
fn tachycardia_is_reported_and_normal_sinus_is_not() {
    assert!(saw(&run(&sinus(60, 130.0)), Condition::Tachycardia));
    let out = run(&sinus(60, 70.0));
    assert!(!saw(&out, Condition::Tachycardia));
    assert!(!saw(&out, Condition::Bradycardia));
    assert!(!saw(&out, Condition::Pause));
}

#[test]
fn three_ventricular_beats_are_a_run_and_a_fast_run_is_ventricular_tachycardia() {
    let mut b = sinus(12, 70.0);
    b.extend(vec![(400.0, Beat::Ventricular); 3]); // 150 bpm
    b.extend(sinus(6, 70.0));
    let out = run(&b);
    assert!(saw(&out, Condition::VentricularRun));
    assert!(saw(&out, Condition::VentricularTachycardia));

    // Two is a couplet, not a run.
    let mut b = sinus(12, 70.0);
    b.extend(vec![(400.0, Beat::Ventricular); 2]);
    b.extend(sinus(6, 70.0));
    assert!(!saw(&run(&b), Condition::VentricularRun));

    // A slow run is a run but not tachycardia.
    let mut b = sinus(12, 70.0);
    b.extend(vec![(1000.0, Beat::Ventricular); 4]); // 60 bpm
    b.extend(sinus(6, 70.0));
    let out = run(&b);
    assert!(saw(&out, Condition::VentricularRun));
    assert!(!saw(&out, Condition::VentricularTachycardia));
}

#[test]
fn bigeminy_is_a_pattern_not_a_count() {
    let mut b = sinus(8, 70.0);
    for _ in 0..12 {
        b.push((600.0, Beat::Normal));
        b.push((500.0, Beat::Ventricular));
    }
    let out = run(&b);
    assert!(saw(&out, Condition::Bigeminy));
    assert!(!saw(&out, Condition::Trigeminy));

    // The same number of ventricular beats, scattered: not bigeminy.
    let mut b = sinus(8, 70.0);
    for _ in 0..12 {
        b.extend(sinus(5, 70.0));
        b.push((500.0, Beat::Ventricular));
    }
    assert!(!saw(&run(&b), Condition::Bigeminy));
}

#[test]
fn trigeminy_is_every_third_beat() {
    let mut b = sinus(8, 70.0);
    for _ in 0..12 {
        b.push((600.0, Beat::Normal));
        b.push((600.0, Beat::Normal));
        b.push((500.0, Beat::Ventricular));
    }
    let out = run(&b);
    assert!(saw(&out, Condition::Trigeminy));
    assert!(!saw(&out, Condition::Bigeminy));
}

#[test]
fn morphology_conditions_go_quiet_without_a_class() {
    // Unreadable morphology must not be reported as "no ectopy".
    let mut b = sinus(8, 70.0);
    for _ in 0..12 {
        b.push((600.0, Beat::Unknown));
        b.push((500.0, Beat::Unknown));
    }
    let out = run(&b);
    assert!(!saw(&out, Condition::Bigeminy));
    assert!(!saw(&out, Condition::VentricularRun));
}

#[test]
fn asystole_is_reportable_at_all() {
    // The interval that defines asystole is longer than any believable heartbeat
    // interval, so a detector gated on "physiological" can never report one.
    let mut b = sinus(12, 70.0);
    b.push((6000.0, Beat::Normal));
    b.extend(sinus(6, 70.0));
    let out = run(&b);
    assert!(
        saw(&out, Condition::Asystole),
        "a six-second gap is asystole"
    );
}

#[test]
fn a_pause_does_not_invent_a_bradycardia() {
    // Averaging one long interval into the rate would read as a slow rhythm.
    let mut b = sinus(20, 75.0);
    b.push((3500.0, Beat::Normal));
    b.extend(sinus(20, 75.0));
    let out = run(&b);
    assert!(saw(&out, Condition::Pause));
    assert!(
        !saw(&out, Condition::Bradycardia),
        "one pause is not bradycardia"
    );
}

/// A slow ventricular rhythm following a pause is an escape rhythm, and it is
/// the commonest way idioventricular rhythm actually appears.
#[test]
fn a_slow_ventricular_run_after_a_pause_is_idioventricular() {
    let mut beats: Vec<(f32, Beat)> = vec![(800.0, Beat::Normal); 10];
    beats.push((3500.0, Beat::Ventricular)); // the pause, then the escape
    for _ in 0..12 {
        beats.push((1400.0, Beat::Ventricular)); // 43 a minute
    }
    beats.extend(vec![(800.0, Beat::Normal); 10]);
    let episodes = run(&beats);
    let kinds: Vec<Condition> = episodes.iter().map(|e| e.condition).collect();
    assert!(
        kinds.contains(&Condition::Idioventricular),
        "no idioventricular episode in {kinds:?}"
    );
    assert!(
        !kinds.contains(&Condition::VentricularTachycardia),
        "a 43-a-minute rhythm was called tachycardia: {kinds:?}"
    );
}
