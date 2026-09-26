//! The single-file engine is the workspace's engine, not a copy of it that
//! might have drifted: fed the same samples, every output is identical,
//! compared through its full `Debug` form - probabilities and features
//! included, to the last bit.

use std::path::PathBuf;

fn record(name: &str) -> Option<(f64, Vec<f32>)> {
    let roots = [
        std::env::var("DEEP_ECG_RAW").ok().map(PathBuf::from),
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../deep_ecg/data/raw")),
    ];
    for root in roots.into_iter().flatten() {
        let hea = root.join("mitdb").join(format!("{name}.hea"));
        if let Ok(h) = ecg_wfdb::Header::read(&hea) {
            let n = h.n_samples.min((h.fs * 600.0) as usize);
            return ecg_wfdb::read_signal(&h, 0, 0, n).ok().map(|s| (h.fs, s));
        }
    }
    None
}

macro_rules! same {
    ($a:expr, $b:expr, $what:expr) => {{
        let (a, b) = (format!("{:?}", $a), format!("{:?}", $b));
        assert!(
            a == b,
            "{} differs:\n workspace {}\n single    {}",
            $what,
            a,
            b
        );
    }};
}

fn compare(name: &str, patch: bool) {
    let Some((fs, sig)) = record(name) else {
        eprintln!("SKIPPED: no MIT-BIH record {name}. Set DEEP_ECG_RAW to enable.");
        return;
    };
    let (mut ws, mut single) = if patch {
        (
            ecg_pipeline::ChannelPipeline::new(ecg_pipeline::PipelineConfig::patch(fs)),
            ecg::ecg_pipeline::ChannelPipeline::new(ecg::ecg_pipeline::PipelineConfig::patch(fs)),
        )
    } else {
        (
            ecg_pipeline::ChannelPipeline::new(ecg_pipeline::PipelineConfig::new(fs)),
            ecg::ecg_pipeline::ChannelPipeline::new(ecg::ecg_pipeline::PipelineConfig::new(fs)),
        )
    };
    let mut a = ecg_pipeline::ChannelOutput::default();
    let mut b = ecg::ecg_pipeline::ChannelOutput::default();
    let mut beats = 0usize;
    let mut check = |a: &ecg_pipeline::ChannelOutput, b: &ecg::ecg_pipeline::ChannelOutput| {
        same!(a.classes, b.classes, "beat verdicts");
        same!(a.episodes, b.episodes, "rhythm episodes");
        same!(a.af, b.af, "AF windows");
        same!(a.vf, b.vf, "VF windows");
        same!(a.vf_episodes, b.vf_episodes, "VF episodes");
        same!(a.sv_runs, b.sv_runs, "supraventricular runs");
        same!(a.lead_off, b.lead_off, "lead-off");
        same!(a.waves, b.waves, "delineation");
        beats += a.classes.len();
    };
    for chunk in sig.chunks(90) {
        a.clear();
        b.clear();
        ws.push(chunk, &mut a);
        single.push(chunk, &mut b);
        check(&a, &b);
    }
    a.clear();
    b.clear();
    ws.finish(&mut a);
    single.finish(&mut b);
    check(&a, &b);
    assert!(beats > 500, "only {beats} beats compared");
    eprintln!(
        "{name} ({}): {beats} beats identical",
        if patch { "patch" } else { "clinical" }
    );
}

#[test]
fn the_single_file_engine_is_the_workspace_engine() {
    // Ventricular ectopy, supraventricular ectopy, and atrial fibrillation.
    for name in ["119", "232", "203"] {
        compare(name, false);
        compare(name, true);
    }
}

#[test]
fn the_standard_interface_reports_what_the_engine_found() {
    let Some((fs, sig)) = record("119") else {
        eprintln!("SKIPPED: no MIT-BIH data.");
        return;
    };
    use ecg::ecg_ffi::*;
    let cfg = EcgConfig::new(fs, ECG_PRESET_CLINICAL);
    let mut e = Engine::new(&cfg).unwrap();
    let mut pipe = ecg_pipeline::ChannelPipeline::new(ecg_pipeline::PipelineConfig::new(fs));
    let mut out = ecg_pipeline::ChannelOutput::default();
    let mut buf = vec![EcgEvent::default(); 1024];
    let (mut ours, mut theirs) = (Vec::new(), Vec::new());
    for chunk in sig.chunks(250) {
        e.push(chunk);
        loop {
            let n = e.poll(&mut buf);
            ours.extend(
                buf[..n]
                    .iter()
                    .filter(|x| x.kind == ECG_EV_BEAT)
                    .map(|x| (x.start, x.code)),
            );
            if n < buf.len() {
                break;
            }
        }
        out.clear();
        pipe.push(chunk, &mut out);
        theirs.extend(out.classes.iter().map(|v| (v.sample, v.class as u32)));
    }
    assert_eq!(ours.len(), theirs.len());
    // The interface's class numbering is its own; it agrees with the engine's
    // enum order today, and this pins the mapping rather than the coincidence.
    for ((sa, ca), (sb, cb)) in ours.iter().zip(&theirs) {
        assert_eq!(sa, sb);
        assert_eq!(ca, cb);
    }
}
