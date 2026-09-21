use ecg_rhythm::{EpisodeConfig, EpisodeTracker};

const FS: f64 = 100.0;

fn cfg(bridge_s: f32, min_s: f32) -> EpisodeConfig {
    EpisodeConfig {
        bridge_s,
        min_episode_s: min_s,
    }
}

/// Drive the tracker with one verdict per second.
fn run(c: EpisodeConfig, states: &[bool]) -> Vec<(u64, u64)> {
    let mut t = EpisodeTracker::new(FS, c);
    let mut out = Vec::new();
    for (i, &p) in states.iter().enumerate() {
        if let Some(e) = t.update(i as u64 * FS as u64, p) {
            out.push((e.start / FS as u64, e.end / FS as u64));
        }
    }
    if let Some(e) = t.finish() {
        out.push((e.start / FS as u64, e.end / FS as u64));
    }
    out
}

#[test]
fn short_runs_are_not_reported() {
    let mut s = vec![false; 60];
    s[20..30].fill(true); // 10 s against a 30 s minimum
    assert!(run(cfg(15.0, 30.0), &s).is_empty());
}

#[test]
fn sustained_run_is_reported_with_its_true_start() {
    let mut s = vec![false; 200];
    s[20..120].fill(true);
    assert_eq!(run(cfg(15.0, 30.0), &s), vec![(20, 119)]);
}

#[test]
fn a_short_gap_does_not_split_an_episode() {
    let mut s = vec![false; 200];
    s[20..120].fill(true);
    s[60..65].fill(false); // 5 s dip, under the 15 s bridge
    assert_eq!(run(cfg(15.0, 30.0), &s), vec![(20, 119)]);
}

#[test]
fn a_long_gap_does_split_an_episode() {
    let mut s = vec![false; 300];
    s[20..120].fill(true);
    s[140..260].fill(true);
    assert_eq!(run(cfg(15.0, 30.0), &s), vec![(20, 119), (140, 259)]);
}

#[test]
fn an_episode_still_open_at_the_end_is_reported() {
    let mut s = vec![false; 100];
    s[20..100].fill(true);
    assert_eq!(run(cfg(15.0, 30.0), &s), vec![(20, 99)]);
}
