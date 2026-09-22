//! A bank of binary rhythm detectors over beats and their classes.
//!
//! Same shape as the beat bank and for the same reason: each condition answers a
//! different question from different evidence, and each carries its own
//! operating point because the cost of missing one is not the cost of missing
//! another. A missed asystole is lethal; a missed run of trigeminy is a note in
//! a report.
//!
//! What they share is the input - intervals and beat classes - and the episode
//! machinery that turns a per-beat verdict into something worth reporting. What
//! they do not share is a threshold, a minimum duration, or a model.
//!
//! # Evidence, per condition
//!
//! | condition | evidence |
//! |---|---|
//! | pause, asystole | one interval |
//! | bradycardia, tachycardia | rate over a short run of intervals |
//! | ventricular run, ventricular tachycardia | consecutive ventricular beats, and their rate |
//! | bigeminy, trigeminy | the *pattern* of classes, not their count |
//!
//! Only the last two need the beat classifier at all, which is why the first
//! four keep working when morphology is unreadable.

use crate::episode::{EpisodeConfig, EpisodeTracker};
use crate::rr::RrSample;

/// Beat class as the rhythm layer needs it. Kept local so this crate does not
/// depend on the classifier, only on its verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Beat {
    Normal,
    Supraventricular,
    Ventricular,
    /// A conducted beat and a ventricular one arriving together.
    ///
    /// Counted as ventricular by every rule here that looks for ventricular
    /// activity, because that is what it contains: a fusion beat at the edge of
    /// a run is part of the run, and the convention the field reports against
    /// treats it that way. It is kept as its own value so that a consumer
    /// counting ectopic *beats* can tell the two apart, which is a different
    /// question from whether the ventricle fired.
    Fusion,
    Unknown,
}

impl Beat {
    /// Whether this beat carries a ventricular activation.
    #[inline]
    pub fn is_ventricular(self) -> bool {
        matches!(self, Beat::Ventricular | Beat::Fusion)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Condition {
    Pause,
    Asystole,
    Bradycardia,
    Tachycardia,
    VentricularRun,
    VentricularTachycardia,
    Bigeminy,
    Trigeminy,
    /// A ventricular pacemaker slow enough not to be tachycardia. The same
    /// evidence as a ventricular run, separated by rate, because the two mean
    /// different things: this one is usually an escape rhythm and the treatment
    /// for tachycardia would be actively harmful.
    Idioventricular,
}

impl Condition {
    pub const ALL: [Condition; 9] = [
        Condition::Pause,
        Condition::Asystole,
        Condition::Bradycardia,
        Condition::Tachycardia,
        Condition::VentricularRun,
        Condition::VentricularTachycardia,
        Condition::Bigeminy,
        Condition::Trigeminy,
        Condition::Idioventricular,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Condition::Pause => "pause",
            Condition::Asystole => "asystole",
            Condition::Bradycardia => "bradycardia",
            Condition::Tachycardia => "tachycardia",
            Condition::VentricularRun => "ventricular run",
            Condition::VentricularTachycardia => "ventricular tachycardia",
            Condition::Bigeminy => "bigeminy",
            Condition::Trigeminy => "trigeminy",
            Condition::Idioventricular => "idioventricular rhythm",
        }
    }

    /// Needs the beat classifier, so it stops when morphology is unreadable.
    pub fn needs_beat_class(self) -> bool {
        matches!(
            self,
            Condition::VentricularRun
                | Condition::VentricularTachycardia
                | Condition::Bigeminy
                | Condition::Trigeminy
                | Condition::Idioventricular
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RhythmConfig {
    pub fs: f64,
    /// An interval longer than this is a pause.
    pub pause_ms: f32,
    /// Longer still, and it is asystole. Reported as both, deliberately: an
    /// asystole is a pause, and a consumer filtering for pauses should see it.
    pub asystole_ms: f32,
    pub brady_bpm: f32,
    pub tachy_bpm: f32,
    /// Intervals the rate is averaged over before calling brady- or tachycardia.
    /// One short interval is not a rate.
    pub rate_beats: usize,
    /// Consecutive ventricular beats that make a run.
    pub run_beats: usize,
    /// A run at or above this rate is ventricular tachycardia.
    pub vt_bpm: f32,
    /// Repeats of the pattern before bigeminy or trigeminy is called.
    pub pattern_cycles: usize,
    /// A ventricular rhythm slower than this is idioventricular rather than
    /// tachycardia; slower than the floor it is not a rhythm at all.
    pub ivr_bpm_lo: f32,
    /// Beat-like energy a pause may carry and still be a pause, as a share of
    /// what a beat looks like on this channel. Zero disables the test.
    ///
    /// Chosen on the long-term corpus, which is the only one long enough to
    /// show the problem, by the rule this project uses for alarm thresholds:
    /// hold a false-alarm budget and take the sensitivity. Between 0.013 and
    /// 0.018 sensitivity moves less than a point while the false-alarm rate
    /// doubles, so the value sits at the elbow: 116.9 false pauses per
    /// patient-day become 4.9, for 1.1 points of sensitivity.
    ///
    /// The interval alone cannot distinguish "the heart did not beat" from "it
    /// beat and we missed it" - they produce exactly the same interval. What
    /// separates them is what the trace was doing in between: a real pause is
    /// quiet in the band the detector works in, and a dropout is not, which is
    /// why the detector failed there in the first place.
    pub pause_max_energy: f32,
    /// How much faster than the rhythm it replaced this one may be.
    ///
    /// Set loosely on purpose. "Slower than what it replaced" is the classical
    /// escape rhythm and it is not what these corpora label: `(IVR` covers
    /// accelerated idioventricular rhythm too, which is by definition faster
    /// than the escape rate and often faster than the rhythm around it. Tested
    /// at 0.9 the criterion doubled precision on the long-term corpus and found
    /// nothing at all on MIT-BIH, where the underlying rhythm is itself slow.
    /// At 1.3 it trims some of the false positives and costs no detections on
    /// either.
    pub escape_frac: f32,
    /// Episode shaping, per condition.
    pub episodes: [EpisodeConfig; 9],
}

impl RhythmConfig {
    pub fn new(fs: f64) -> Self {
        // Minimum durations differ by condition because the conditions differ.
        // A pause is a single interval and reporting it needs no duration at
        // all; a rate is not a rate until it has lasted.
        let event = EpisodeConfig {
            bridge_s: 0.0,
            min_episode_s: 0.0,
        };
        let sustained = EpisodeConfig {
            bridge_s: 5.0,
            min_episode_s: 15.0,
        };
        let pattern = EpisodeConfig {
            bridge_s: 3.0,
            min_episode_s: 6.0,
        };
        // An escape rhythm is short by nature - it lasts until the rhythm it is
        // escaping from comes back - and its definition is already a beat
        // count: three consecutive ventricular beats. A duration minimum on top
        // of that is the same requirement stated twice, and at fifty a minute
        // the two contradict each other. Three beats at that rate is about
        // three seconds of *run* but only one second of episode, because the
        // condition does not become true until the third beat. A three-second
        // minimum rejected every idioventricular rhythm in the corpus.
        //
        // The bridge stays: one misclassified beat in the middle of a run
        // should not split it into two findings.
        let escape = EpisodeConfig {
            bridge_s: 2.0,
            min_episode_s: 0.0,
        };
        RhythmConfig {
            fs,
            pause_ms: 2000.0,
            asystole_ms: 4000.0,
            brady_bpm: 50.0,
            tachy_bpm: 100.0,
            rate_beats: 8,
            run_beats: 3,
            vt_bpm: 100.0,
            pattern_cycles: 3,
            ivr_bpm_lo: 15.0,
            escape_frac: 1.3,
            pause_max_energy: 0.014,
            episodes: [
                event,     // pause
                event,     // asystole
                sustained, // bradycardia
                sustained, // tachycardia
                event,     // ventricular run
                event,     // ventricular tachycardia
                pattern,   // bigeminy
                pattern,   // trigeminy
                escape,    // idioventricular rhythm
            ],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RhythmEpisode {
    pub condition: Condition,
    pub start: u64,
    pub end: u64,
}

const HISTORY: usize = 16;

pub struct RhythmBank {
    cfg: RhythmConfig,
    trackers: Vec<EpisodeTracker>,
    /// Recent intervals and classes, oldest first within the filled prefix.
    rr: [f32; HISTORY],
    class: [Beat; HISTORY],
    /// Whether each interval was within physiological bounds.
    physiological: [bool; HISTORY],
    /// Wave evidence for the beat that closed each interval.
    qrs_ms: [f32; HISTORY],
    p_axis: [f32; HISTORY],
    n: usize,
    idx: usize,
}

impl RhythmBank {
    pub fn new(cfg: RhythmConfig) -> Self {
        RhythmBank {
            trackers: cfg
                .episodes
                .iter()
                .map(|e| EpisodeTracker::new(cfg.fs, *e))
                .collect(),
            cfg,
            rr: [0.0; HISTORY],
            class: [Beat::Unknown; HISTORY],
            physiological: [true; HISTORY],
            qrs_ms: [0.0; HISTORY],
            p_axis: [0.0; HISTORY],
            n: 0,
            idx: 0,
        }
    }

    pub fn config(&self) -> &RhythmConfig {
        &self.cfg
    }

    /// Feed one classified beat. Completed episodes are appended to `out`.
    pub fn push(&mut self, rr: &RrSample, class: Beat, out: &mut Vec<RhythmEpisode>) {
        self.rr[self.idx] = rr.rr_ms;
        self.class[self.idx] = class;
        self.physiological[self.idx] = rr.physiological;
        self.qrs_ms[self.idx] = rr.qrs_ms;
        self.p_axis[self.idx] = rr.p_axis;
        self.idx = (self.idx + 1) % HISTORY;
        self.n = (self.n + 1).min(HISTORY);

        for (i, c) in Condition::ALL.iter().enumerate() {
            // Each condition is gated by what it actually needs.
            //
            // Pause and asystole are gated on signal quality alone. Gating them
            // on the interval being *physiological* - as every other condition
            // is - filters out exactly the intervals that define them: the
            // upper bound on a believable heartbeat interval is three seconds,
            // so asystole, which starts at four, could never be reported at all.
            //
            // A condition that needs morphology goes quiet without it. Absence
            // of evidence is not evidence of absence, and reporting "no ectopy"
            // because the beat could not be classified would be a lie.
            let gated = match c {
                // Gated on electrode integrity, not on signal quality: a flat
                // trace is what asystole looks like, so the quality monitor
                // condemns it as a dead lead. What distinguishes the two is
                // whether the electrode is attached.
                //
                // And not while the interval carries beat-like energy. Gating
                // on general signal quality instead would suppress the thing
                // being looked for, but this is not a quality test: it asks
                // whether *this interval* was quiet, which is what a pause is.
                Condition::Pause | Condition::Asystole => {
                    !rr.lead_ok
                        || (self.cfg.pause_max_energy > 0.0
                            && rr.interval_energy > self.cfg.pause_max_energy)
                }
                _ if c.needs_beat_class() => !rr.usable() || class == Beat::Unknown,
                _ => !rr.usable(),
            };
            let present = if gated { false } else { self.holds(*c) };
            if let Some(e) = self.trackers[i].update(rr.sample, present) {
                out.push(RhythmEpisode {
                    condition: *c,
                    start: e.start,
                    end: e.end,
                });
            }
        }
    }

    /// Most recent `k` entries, newest last.
    fn recent(&self, k: usize) -> impl Iterator<Item = (f32, Beat)> + '_ {
        let k = k.min(self.n);
        (0..k).map(move |j| {
            let i = (self.idx + HISTORY - k + j) % HISTORY;
            (self.rr[i], self.class[i])
        })
    }

    fn last_rr(&self) -> f32 {
        self.rr[(self.idx + HISTORY - 1) % HISTORY]
    }

    fn holds(&self, c: Condition) -> bool {
        match c {
            Condition::Pause => self.last_rr() >= self.cfg.pause_ms,
            Condition::Asystole => self.last_rr() >= self.cfg.asystole_ms,
            Condition::Bradycardia => self
                .mean_rate(self.cfg.rate_beats)
                .is_some_and(|bpm| bpm < self.cfg.brady_bpm),
            Condition::Tachycardia => self
                .mean_rate(self.cfg.rate_beats)
                .is_some_and(|bpm| bpm > self.cfg.tachy_bpm),
            Condition::VentricularRun => self.ventricular_run() >= self.cfg.run_beats,
            Condition::VentricularTachycardia => {
                let run = self.ventricular_run();
                run >= self.cfg.run_beats
                    && self.run_rate(run).is_some_and(|b| b >= self.cfg.vt_bpm)
            }
            Condition::Bigeminy => self.alternating(2),
            Condition::Trigeminy => self.alternating(3),
            Condition::Idioventricular => self.idioventricular(),
        }
    }

    /// Mean rate over the last `k` intervals, `None` until there are that many
    /// and they are all believable as heartbeat intervals.
    ///
    /// One implausible interval - a missed beat, a pause - would drag the mean
    /// far enough to invent a bradycardia, so the rate is simply not reported
    /// across one.
    fn mean_rate(&self, k: usize) -> Option<f32> {
        if self.n < k || k == 0 {
            return None;
        }
        let mut sum = 0.0f32;
        for j in 0..k {
            let i = (self.idx + HISTORY - k + j) % HISTORY;
            if !self.physiological[i] {
                return None;
            }
            sum += self.rr[i];
        }
        (sum > 0.0).then(|| 60_000.0 * k as f32 / sum)
    }

    /// A ventricular rhythm slow enough not to be tachycardia, and not much
    /// faster than the rhythm it replaced.
    ///
    /// The second half is what keeps this from being a synonym for "ventricular
    /// run below a hundred a minute", which any three beats misclassified as
    /// ventricular at an ordinary rate would satisfy. It helps and it does not
    /// solve the problem: this condition is a subset of the ventricular runs,
    /// so its precision cannot exceed theirs, and on long-term ambulatory data
    /// theirs is five per cent.
    fn idioventricular(&self) -> bool {
        let run = self.ventricular_run();
        if run < self.cfg.run_beats {
            return false;
        }
        let Some(rate) = self.run_rate(run) else {
            return false;
        };
        if !(self.cfg.ivr_bpm_lo..self.cfg.vt_bpm).contains(&rate) {
            return false;
        }
        match self.rate_before(run) {
            Some(before) => rate <= self.cfg.escape_frac * before,
            // Nothing to compare against - the run reaches back past everything
            // remembered - so the rate bound stands on its own.
            None => run >= HISTORY - 2,
        }
    }

    /// Rate of the beats immediately preceding a run of `run` beats.
    fn rate_before(&self, run: usize) -> Option<f32> {
        let k = self.cfg.rate_beats.min(self.n.saturating_sub(run));
        if k < 2 {
            return None;
        }
        let mut sum = 0.0f32;
        for j in 0..k {
            let i = (self.idx + HISTORY - run - k + j) % HISTORY;
            if !self.physiological[i] {
                return None;
            }
            sum += self.rr[i];
        }
        (sum > 0.0).then(|| 60_000.0 * k as f32 / sum)
    }

    /// Rate of a ventricular run, over the intervals *inside* it.
    ///
    /// A run of `run` beats has `run - 1` intervals of its own; the one before
    /// that is the coupling interval, which belongs to the rhythm the run
    /// interrupted. Including it was not a rounding error, it made
    /// idioventricular rhythm unreportable outright. An escape rhythm is what
    /// follows a pause, `mean_rate` refuses to average across an interval too
    /// long to be a heartbeat - which is right, one of those would invent a
    /// bradycardia - and the pause that the escape rhythm is escaping from is
    /// exactly that interval. So the run was detected, and then asking how fast
    /// it was returned nothing, every time.
    ///
    /// This is the third condition in this engine to have been defined by the
    /// very thing a general-purpose guard filtered out, after asystole and the
    /// physiological gate. The pattern has a name here now: a rule about what
    /// happens *after* an abnormal interval cannot be evaluated over a window
    /// that still contains it.
    fn run_rate(&self, run: usize) -> Option<f32> {
        let k = run.saturating_sub(1).min(self.cfg.rate_beats);
        if k < 2 || self.n < k {
            return None;
        }
        let mut sum = 0.0f32;
        for j in 0..k {
            let i = (self.idx + HISTORY - k + j) % HISTORY;
            if !self.physiological[i] {
                return None;
            }
            sum += self.rr[i];
        }
        (sum > 0.0).then(|| 60_000.0 * k as f32 / sum)
    }

    /// Length of the run of ventricular beats ending at the newest one.
    fn ventricular_run(&self) -> usize {
        let mut run = 0;
        for (_, c) in self.recent(self.n).collect::<Vec<_>>().into_iter().rev() {
            if c.is_ventricular() {
                run += 1;
            } else {
                break;
            }
        }
        run
    }

    /// Is every `period`-th beat ventricular and the rest not, for
    /// `pattern_cycles` repeats ending at the newest beat?
    ///
    /// The pattern is what defines bigeminy, not the count: a patient with many
    /// isolated ventricular beats is not bigeminal, and counting ectopy would
    /// call them the same thing.
    fn alternating(&self, period: usize) -> bool {
        let need = period * self.cfg.pattern_cycles;
        if self.n < need {
            return false;
        }
        let window: Vec<Beat> = self.recent(need).map(|(_, c)| c).collect();
        for (i, c) in window.iter().enumerate() {
            // Counting back from the newest beat, which is ventricular.
            let ventricular_slot = (window.len() - 1 - i).is_multiple_of(period);
            match (ventricular_slot, *c) {
                (true, b) if b.is_ventricular() => {}
                (false, Beat::Normal) | (false, Beat::Supraventricular) => {}
                _ => return false,
            }
        }
        true
    }

    /// Samples were lost. The interval history no longer describes a
    /// continuous stretch, so it is discarded rather than spliced.
    pub fn on_gap(&mut self) {
        self.n = 0;
        self.idx = 0;
    }

    /// Close any open episodes at the end of a stream.
    pub fn finish(&mut self, out: &mut Vec<RhythmEpisode>) {
        for (i, c) in Condition::ALL.iter().enumerate() {
            if let Some(e) = self.trackers[i].finish() {
                out.push(RhythmEpisode {
                    condition: *c,
                    start: e.start,
                    end: e.end,
                });
            }
        }
    }

    pub fn reset(&mut self) {
        for t in self.trackers.iter_mut() {
            t.reset();
        }
        self.n = 0;
        self.idx = 0;
        self.physiological = [true; HISTORY];
    }
}
