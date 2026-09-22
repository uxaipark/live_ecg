//! Electrode failure has no labelled corpus anywhere in this project, and it is
//! the one finding here that does not need one: it is defined by physics rather
//! than by a rhythm, so it can be constructed. These tests splice a known
//! failure into a known-good signal and ask whether it is found, how late, and
//! whether it clears.
//!
//! The third test is the important one. A flat trace is what asystole looks
//! like, and calling that an electrode failure would silently reclassify every
//! asystole as an artefact - so the detector must *not* fire on one.

use ecg_pipeline::{ChannelOutput, ChannelPipeline, PipelineConfig};
use ecg_quality::LeadOffKind;

const FS: f64 = 250.0;

/// Synthetic sinus rhythm: a narrow complex and a T wave on a wandering
/// baseline, so the QRS band is not the only thing in the signal.
fn sinus(seconds: f64, bpm: f64) -> Vec<f32> {
    let n = (seconds * FS) as usize;
    let period = (60.0 / bpm * FS) as usize;
    (0..n)
        .map(|i| {
            let phase = i % period;
            let beat = match phase {
                0 | 1 => 0.5,
                2 | 3 => 1.0,
                4 | 5 => -0.4,
                20..=30 => 0.15,
                _ => 0.0,
            };
            let breath = 0.05 * (2.0 * std::f32::consts::PI * 0.25 * i as f32 / FS as f32).sin();
            beat + breath
        })
        .collect()
}

fn run(sig: &[f32]) -> ChannelOutput {
    let mut pipe = ChannelPipeline::new(PipelineConfig::new(FS));
    let mut out = ChannelOutput::default();
    let mut all = ChannelOutput::default();
    for c in sig.chunks(62) {
        out.clear();
        pipe.push(c, &mut out);
        all.lead_off.extend_from_slice(&out.lead_off);
        all.episodes.extend_from_slice(&out.episodes);
    }
    out.clear();
    pipe.finish(&mut out);
    all.lead_off.extend_from_slice(&out.lead_off);
    all.episodes.extend_from_slice(&out.episodes);
    all
}

/// Replace `[from, to)` seconds of `sig` using `f`.
fn splice(sig: &mut [f32], from: f64, to: f64, mut f: impl FnMut(usize) -> f32) {
    let n = sig.len();
    let (a, b) = ((from * FS) as usize, ((to * FS) as usize).min(n));
    for (i, x) in sig[a..b].iter_mut().enumerate() {
        *x = f(i);
    }
}

#[test]
fn an_electrode_pulled_to_the_rail_is_reported() {
    let mut sig = sinus(120.0, 60.0);
    splice(
        &mut sig,
        40.0,
        70.0,
        |i| if i % 2 == 0 { 40.0 } else { 38.0 },
    );
    let out = run(&sig);
    let found: Vec<LeadOffKind> = out.lead_off.iter().map(|e| e.kind).collect();
    assert!(
        found.contains(&LeadOffKind::RailContact),
        "a 30 s excursion to 40 mV was not reported as an electrode failure: {found:?}"
    );
    let e = out
        .lead_off
        .iter()
        .find(|e| e.kind == LeadOffKind::RailContact)
        .unwrap();
    let start_s = e.start as f64 / FS;
    let end_s = e.end as f64 / FS;
    assert!(
        (38.0..48.0).contains(&start_s),
        "reported the failure starting at {start_s:.1} s, not 40"
    );
    assert!(
        (68.0..82.0).contains(&end_s),
        "reported the failure ending at {end_s:.1} s, not 70"
    );
}

#[test]
fn an_open_input_picking_up_mains_is_reported() {
    let mut sig = sinus(120.0, 60.0);
    // Ten times the patient's amplitude, all of it at 50 Hz, none of it a QRS.
    splice(&mut sig, 40.0, 70.0, |i| {
        10.0 * (2.0 * std::f32::consts::PI * 50.0 * i as f32 / FS as f32).sin()
    });
    let out = run(&sig);
    let found: Vec<LeadOffKind> = out.lead_off.iter().map(|e| e.kind).collect();
    assert!(
        !found.is_empty(),
        "a high-impedance input at ten times the patient's amplitude was not reported"
    );
}

#[test]
fn an_asystole_is_not_called_an_electrode_failure() {
    let mut sig = sinus(120.0, 60.0);
    splice(&mut sig, 60.0, 68.0, |_| 0.0);
    let out = run(&sig);
    assert!(
        out.lead_off.is_empty(),
        "eight seconds of asystole were reported as an electrode failure: {:?}",
        out.lead_off.iter().map(|e| e.kind).collect::<Vec<_>>()
    );
}

#[test]
fn a_clean_recording_reports_nothing() {
    let out = run(&sinus(300.0, 60.0));
    assert!(
        out.lead_off.is_empty(),
        "five minutes of clean signal produced {} electrode failures",
        out.lead_off.len()
    );
}
