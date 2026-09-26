//! What every replaceable stage promises, checked for every implementation
//! the engine carries: selected by name it is the one that runs, the same
//! input gives the same output, and outputs come back in time order.

use ecg_pipeline::stages::{self, StageKind, VfStage};
use ecg_pipeline::{ChannelOutput, ChannelPipeline, CustomStages, PipelineConfig, StageSelection};
use ecg_rhythm::VfWindow;

/// Narrow complexes at 72 a minute with an early wide one every seventh beat.
fn signal(fs: f64, seconds: f64) -> Vec<f32> {
    let n = (fs * seconds) as usize;
    let period = 60.0 / 72.0;
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            let k = (t / period) as u64;
            let ph = t - k as f64 * period;
            let v = if k % 7 == 6 {
                -1.2 * (-((ph - 0.18) / 0.035).powi(2)).exp()
            } else {
                (-((ph - 0.25) / 0.012).powi(2)).exp()
                    + 0.25 * (-((ph - 0.55) / 0.06).powi(2)).exp()
            };
            v as f32
        })
        .collect()
}

fn run(cfg: PipelineConfig, sig: &[f32]) -> (String, [(&'static str, &'static str); 5]) {
    let mut p = ChannelPipeline::new(cfg);
    let ids = p.stage_ids();
    let mut out = ChannelOutput::default();
    let mut trace = String::new();
    let mut last = 0u64;
    for chunk in sig.chunks(113) {
        out.clear();
        p.push(chunk, &mut out);
        for v in &out.classes {
            assert!(v.sample >= last, "beats out of order");
            last = v.sample;
        }
        for (s, e) in &out.vf_episodes {
            assert!(e >= s);
        }
        for e in &out.episodes {
            assert!(e.end >= e.start);
        }
        trace.push_str(&format!(
            "{:?}{:?}{:?}{:?}{:?}",
            out.classes, out.episodes, out.af, out.vf, out.sv_runs
        ));
    }
    out.clear();
    p.finish(&mut out);
    trace.push_str(&format!("{:?}{:?}", out.episodes, out.sv_runs));
    (trace, ids)
}

#[test]
fn every_registered_stage_keeps_the_contract() {
    let sig = signal(250.0, 90.0);
    for info in stages::available() {
        let spec = format!("{}={}", info.kind.key(), info.name);
        let mut cfg = PipelineConfig::new(250.0);
        cfg.stages = StageSelection::parse(&spec).unwrap();
        let (a, ids) = run(cfg, &sig);
        let (b, _) = run(cfg, &sig);
        assert!(a == b, "{} is not deterministic", info.name);
        let running = ids.iter().find(|(k, _)| *k == info.kind.key()).unwrap().1;
        assert_eq!(
            running, info.name,
            "asked for {} and got {running}",
            info.name
        );
    }
    assert!(StageKind::ALL.len() == 5);
}

/// A stage of the caller's own replaces the engine's, and is the one reported.
#[test]
fn a_callers_stage_is_used_in_place_of_the_engines() {
    struct Silent;
    impl VfStage for Silent {
        fn id(&self) -> &'static str {
            "vf.silent-test@1"
        }
        fn process(&mut self, _clean: f32) -> Option<VfWindow> {
            None
        }
        fn in_vf(&self) -> bool {
            false
        }
        fn on_gap(&mut self, _unobserved: u64) {}
        fn reset(&mut self) {}
    }
    let custom = CustomStages {
        vf: Some(Box::new(Silent)),
        ..CustomStages::default()
    };
    let mut p = ChannelPipeline::with_stages(PipelineConfig::new(250.0), custom);
    assert!(p.stage_ids().contains(&("vf", "vf.silent-test@1")));
    let mut out = ChannelOutput::default();
    p.push(&signal(250.0, 30.0), &mut out);
    assert!(out.vf.is_empty(), "the engine's own VF stage ran");
    assert!(out.classes.len() > 20, "the rest of the pipeline stopped");
}
