//! Runs of supraventricular rhythm, found by the rhythm rather than the beat.
//!
//! # Why this is a rhythm detector
//!
//! On the patch corpus 90 % of the supraventricular beats an analyst marked sit
//! in runs of eight or more, and a per-beat detector finds the beat that opens a
//! run (54.8 %) and almost none of the beats that continue it (2.1 %): after the
//! first beat there is nothing to be early against. Per-beat atrial evidence
//! does not rescue it either - a P wave there is a tenth of a millivolt, and
//! three ways of carrying the label across a run on it were refuted.
//!
//! What a run does have is a rhythm. Joining runs an analyst split by a few
//! normal beats, 57 % of the episodes on the development zone begin with the
//! interval dropping to under 0.85 of what came before - median 0.73 - and a
//! step that size appears in 0.32 % of stretches of sinus rhythm. It is the
//! classical sign of an ectopic focus taking over: sinus rhythm accelerates
//! over many beats, a focus switches on between two.
//!
//! # The rule
//!
//! Streaming, with the lag a step needs to be seen: the median of the latest
//! `window` intervals has to fall below `step` times the median of the
//! `window` before them, and those latest intervals have to be regular. The run
//! is then held while intervals stay within `band` of its own rate, which
//! follows slow drift, and ends when more than `grace` intervals in a row fall
//! outside it. Runs shorter than `min_beats` are not reported.
//!
//! It opens no run inside sustained fibrillation, which is irregular by
//! definition and would otherwise supply steps without end; the caller says
//! when that is.

#[derive(Debug, Clone, Copy)]
pub struct SvRunConfig {
    /// Off unless a deployment turns it on, and the measurement is why.
    ///
    /// On the patch it is the difference between finding the class and not:
    /// against the development zone's exhaustive review, supraventricular F1
    /// goes from 0.12 to 0.47 with the engine's own beats and fibrillation
    /// gate. On the public corpora, against annotations independent of any
    /// device, it is mixed - MIT-BIH training records 60.1 % at 52.5 % become
    /// 76.3 % at 52.4 %, the supraventricular corpus's 78.6 % at 60.8 % become
    /// 82.5 % at 51.3 %, and on long recordings with little ectopy the
    /// precision roughly halves, because sinus rhythm does sometimes step. So
    /// it is part of the patch configuration, `PipelineConfig::patch`, and not
    /// of the default.
    pub enabled: bool,
    /// Intervals in each of the two windows compared for a step.
    pub window: usize,
    /// The later window's median must be below this times the earlier one's.
    pub step: f32,
    /// Coefficient of variation the later window may have: a step into an
    /// irregular stretch is noise or fibrillation, not a focus.
    pub max_cv: f32,
    /// How far an interval may be from the run's rate and still belong to it.
    pub band: f32,
    /// Consecutive intervals outside the band a run survives.
    pub grace: u32,
    /// Beats a run needs to be reported.
    pub min_beats: u32,
    /// Whether the per-beat detector has to have called one of the run's first
    /// beats supraventricular before the run is opened.
    ///
    /// A focus switching on usually announces itself with a premature beat;
    /// sinus rhythm stepping up after an arousal does not. On the development
    /// zone the per-beat call is present at the start of 63.9 % of the runs an
    /// analyst agrees with and 37.0 % of the ones they do not - too weak a
    /// separation to pay for itself: requiring it takes patch F1 from 0.47 to
    /// 0.39, and buys three points of precision on the public corpora. Off.
    pub require_onset_call: bool,
}

impl Default for SvRunConfig {
    /// Chosen on the development zone's 25 exhaustively reviewed patch
    /// recordings, by best F1 of the beats inside reported runs against the
    /// analyst's supraventricular label, from beat positions alone. F1 is flat
    /// between 0.54 and 0.55 across the neighbourhood of this point: at it,
    /// 64.0 % sensitivity at 48.9 % precision, 11.5 runs per day. The per-beat
    /// detector on the same recordings reads 8.3 % at 21.4 %.
    fn default() -> Self {
        SvRunConfig {
            enabled: false,
            window: 4,
            step: 0.75,
            max_cv: 0.12,
            band: 0.17,
            grace: 3,
            min_beats: 16,
            require_onset_call: false,
        }
    }
}

/// One run, on the input time base: the beat that opened it and the last beat
/// that belonged to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SvRun {
    pub start: u64,
    pub end: u64,
    pub beats: u32,
}

const MAX_WINDOW: usize = 16;

#[derive(Debug, Clone)]
pub struct SvRunDetector {
    cfg: SvRunConfig,
    /// The last `2 * window` intervals and the beats that closed them.
    rr: [f32; 2 * MAX_WINDOW],
    at: [u64; 2 * MAX_WINDOW],
    /// Whether the per-beat detector called each of those beats
    /// supraventricular.
    called: [bool; 2 * MAX_WINDOW],
    n: usize,
    idx: usize,
    run: Option<Open>,
}

#[derive(Debug, Clone, Copy)]
struct Open {
    start: u64,
    last: u64,
    rate: f32,
    beats: u32,
    miss: u32,
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

impl SvRunDetector {
    pub fn new(cfg: SvRunConfig) -> Self {
        assert!(cfg.window >= 2 && cfg.window <= MAX_WINDOW);
        SvRunDetector {
            cfg,
            rr: [0.0; 2 * MAX_WINDOW],
            at: [0; 2 * MAX_WINDOW],
            called: [false; 2 * MAX_WINDOW],
            n: 0,
            idx: 0,
            run: None,
        }
    }

    /// Feed one interval (a no-op unless the detector is enabled): its length, the beat that closed it, whether it is
    /// believable, whether the rhythm is in sustained fibrillation, and whether
    /// the per-beat detector called the closing beat supraventricular. A run
    /// that ends is returned.
    pub fn push(
        &mut self,
        rr_ms: f32,
        sample: u64,
        usable: bool,
        fibrillating: bool,
        called: bool,
    ) -> Option<SvRun> {
        if !self.cfg.enabled {
            return None;
        }
        if let Some(mut open) = self.run {
            let inside = usable && (rr_ms - open.rate).abs() <= self.cfg.band * open.rate;
            if inside {
                // Follow slow drift in the run's own rate.
                open.rate += 0.1 * (rr_ms - open.rate);
                open.last = sample;
                open.beats += 1;
                open.miss = 0;
                self.run = Some(open);
                return None;
            }
            open.miss += 1;
            if open.miss <= self.cfg.grace {
                self.run = Some(open);
                return None;
            }
            self.run = None;
            self.reset_window();
            return (open.beats >= self.cfg.min_beats).then_some(SvRun {
                start: open.start,
                end: open.last,
                beats: open.beats,
            });
        }

        if !usable {
            // An interval that cannot be believed breaks the comparison: the
            // two windows would no longer be adjacent stretches of one record.
            self.reset_window();
            return None;
        }
        let w = self.cfg.window;
        let cap = 2 * w;
        self.rr[self.idx] = rr_ms;
        self.at[self.idx] = sample;
        self.called[self.idx] = called;
        self.idx = (self.idx + 1) % cap;
        self.n = (self.n + 1).min(cap);
        if self.n < cap || fibrillating {
            return None;
        }
        // Oldest first.
        let mut earlier = [0.0f32; MAX_WINDOW];
        let mut later = [0.0f32; MAX_WINDOW];
        for k in 0..w {
            earlier[k] = self.rr[(self.idx + k) % cap];
            later[k] = self.rr[(self.idx + w + k) % cap];
        }
        let mean = later[..w].iter().sum::<f32>() / w as f32;
        let var = later[..w].iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / w as f32;
        let cv = var.sqrt() / mean.max(1.0);
        let a = median(&mut earlier[..w]);
        let b = median(&mut later[..w]);
        let announced =
            !self.cfg.require_onset_call || (0..w).any(|k| self.called[(self.idx + w + k) % cap]);
        if b < self.cfg.step * a && cv <= self.cfg.max_cv && announced {
            // Backdated to the beat that closed the first short interval.
            let start = self.at[(self.idx + w) % cap];
            self.run = Some(Open {
                start,
                last: sample,
                rate: b,
                beats: w as u32,
                miss: 0,
            });
        }
        None
    }

    /// Close whatever is open, at the end of a stream.
    pub fn finish(&mut self) -> Option<SvRun> {
        let open = self.run.take()?;
        self.reset_window();
        (open.beats >= self.cfg.min_beats).then_some(SvRun {
            start: open.start,
            end: open.last,
            beats: open.beats,
        })
    }

    fn reset_window(&mut self) {
        self.n = 0;
        self.idx = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(d: &mut SvRunDetector, rr: &[f32]) -> Vec<SvRun> {
        let mut t = 0u64;
        let mut out = Vec::new();
        for &x in rr {
            t += (x * 0.25) as u64;
            if let Some(r) = d.push(x, t, true, false, false) {
                out.push(r);
            }
        }
        out.extend(d.finish());
        out
    }

    #[test]
    fn an_abrupt_regular_run_is_found_and_ends_where_the_rate_returns() {
        let mut d = SvRunDetector::new(SvRunConfig { enabled: true, ..SvRunConfig::default() });
        let mut rr = vec![800.0; 20];
        rr.extend(vec![500.0; 30]); // a focus switches on: 0.625 of what came before
        rr.extend(vec![800.0; 20]);
        let runs = feed(&mut d, &rr);
        assert_eq!(runs.len(), 1, "{runs:?}");
        assert!((28..=31).contains(&runs[0].beats), "{runs:?}");
    }

    /// Sinus rhythm accelerates over many beats; that is not a focus.
    #[test]
    fn a_gradual_acceleration_is_not_a_run() {
        let mut d = SvRunDetector::new(SvRunConfig { enabled: true, ..SvRunConfig::default() });
        let rr: Vec<f32> = (0..200).map(|i| 900.0 - 2.0 * i as f32).collect();
        assert!(feed(&mut d, &rr).is_empty());
    }

    /// Fibrillation supplies steps without end, and is reported by its own
    /// detector.
    #[test]
    fn nothing_opens_inside_fibrillation() {
        let mut d = SvRunDetector::new(SvRunConfig { enabled: true, ..SvRunConfig::default() });
        let mut t = 0;
        let mut rr = vec![800.0f32; 20];
        rr.extend(vec![500.0; 30]);
        for x in rr {
            t += 200;
            assert!(d.push(x, t, true, true, false).is_none());
        }
        assert!(d.finish().is_none());
    }

    #[test]
    fn a_short_burst_is_not_reported() {
        let mut d = SvRunDetector::new(SvRunConfig { enabled: true, ..SvRunConfig::default() });
        let mut rr = vec![800.0; 20];
        rr.extend(vec![500.0; 5]);
        rr.extend(vec![800.0; 20]);
        assert!(feed(&mut d, &rr).is_empty());
    }
}
