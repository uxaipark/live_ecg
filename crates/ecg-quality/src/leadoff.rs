//! Electrode failure, reported as an episode rather than inferred from silence.
//!
//! # Why this is not a rhythm detector
//!
//! Every other finding in this engine is driven by beats. Electrode failure is
//! the one condition that is *defined* by there being none, so a detector fed
//! beats cannot see it: during a lead-off there is nothing to push. It is driven
//! by the quality monitor's hop instead, four times a second, whether or not a
//! beat ever arrives.
//!
//! # What can and cannot be claimed
//!
//! A detached electrode presents in three ways, and only two of them are
//! unambiguous.
//!
//! * **Rail contact.** The input excursion sits at the amplifier's limit for a
//!   large share of the window. No patient produces that.
//! * **Open input.** A high-impedance input picks up mains and body
//!   capacitance: amplitude many times the patient's own, with none of it in the
//!   QRS band. No patient produces that either.
//! * **A flat trace** is not claimed on its own, and that is deliberate. A flat
//!   trace is what asystole looks like. Phase 3 of this project established
//!   that gating pause and asystole on signal quality suppresses exactly the
//!   thing being looked for, and the same argument applies in reverse: calling
//!   a flat trace an electrode failure would silently reclassify every
//!   asystole as an artefact.
//!
//! So a patch whose electrode peels off is caught, because that is an open
//! input; a patch on a patient in asystole is not called a lead-off, because
//! the waveform cannot tell those apart and the wrong answer is fatal.
//!
//! # A flat trace means different things depending on what preceded it
//!
//! Rail contact is only visible while it is *arriving*. The analysis signal is
//! high-passed at half a hertz, so a standing forty-millivolt offset is gone
//! within about two seconds and what is left is a flat line - which is why the
//! first version of this detector reported a thirty-second electrode failure as
//! a three-second one. Measuring the excursion on the raw sample instead is not
//! the answer: electrode half-cell potential puts tens of millivolts of
//! standing offset on a perfectly good input, which is why Phase 1 moved that
//! test onto the high-passed signal in the first place.
//!
//! What resolves it is the prior. A stretch with no beats that arrives out of
//! normal rhythm is asystole. A stretch with no beats that arrives out of a
//! rail excursion or a mains-dominated input is the same electrode, still off.
//! So an absence of beats *holds an episode open* and never opens one, and the
//! two readings of the same evidence are separated by which got there first.
//!
//! The absence of beats is the right evidence for this and the window's
//! amplitude is not, which took a measurement to establish. While the
//! high-pass is still shedding a forty-millivolt step the window looks
//! *normal* by every statistic the monitor has - a peak-to-peak of 5 mV against
//! a slow reference the same step has just dragged up to 44, so the relative
//! amplitude reads exactly 1.0 and the QRS-band share reads exactly 1.0. Four
//! hops of that closed the episode and the flat stretch that followed could not
//! reopen it, because a flat stretch is not allowed to open one.
//!
//! The steady state of a very long lead-off is still not distinguishable from a
//! very long asystole on an AC-coupled waveform alone; that needs electrode
//! impedance, which the front end knows and the waveform does not.

use crate::{flags, QualityConfig, QualitySample};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeadOffKind {
    /// The input excursion is sitting at the amplifier's limit.
    RailContact,
    /// Amplitude far above the patient's own, with nothing in the QRS band:
    /// a high-impedance input picking up its surroundings.
    OpenInput,
}

impl LeadOffKind {
    pub fn name(self) -> &'static str {
        match self {
            LeadOffKind::RailContact => "rail contact",
            LeadOffKind::OpenInput => "open input",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LeadOffEpisode {
    pub kind: LeadOffKind,
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct LeadOffConfig {
    /// Share of a window at the rail that counts as rail contact.
    pub sat_frac: f32,
    /// Amplitude, against this channel's own slow median, above which the input
    /// is producing more than the patient can.
    pub open_p2p_rel: f32,
    /// Share of in-band power in the QRS sub-band, relative to this channel's
    /// own median, below which the amplitude is not a complex.
    pub open_qrs_rel: f32,
    /// How long without a beat counts as not beating, milliseconds.
    ///
    /// Not "no beat in this window", which was the first version and does not
    /// work: at sixty a minute against a 250 ms hop a beat lands in one hop out
    /// of four, so three quarters of a perfectly normal recording reads as not
    /// beating. The bound is the longest interval a heart plausibly produces,
    /// so past it an absence of beats is an absence of beats.
    pub quiet_ms: f64,
    /// Hops the evidence must hold before an episode opens, and before it
    /// closes. Asymmetric: an electrode comes off in one sample and goes back
    /// on over several, and a flapping report is worse than a late one.
    pub enter_hops: u32,
    pub exit_hops: u32,
}

impl Default for LeadOffConfig {
    fn default() -> Self {
        LeadOffConfig {
            sat_frac: 0.25,
            open_p2p_rel: 4.0,
            open_qrs_rel: 0.35,
            enter_hops: 2,
            quiet_ms: 3000.0,
            exit_hops: 4,
        }
    }
}

pub struct LeadOffDetector {
    cfg: LeadOffConfig,
    /// Consecutive hops of evidence, and of its absence.
    holding: u32,
    clearing: u32,
    open: Option<(LeadOffKind, u64)>,
    /// The kind currently accumulating evidence, so two presentations of the
    /// same failure do not each start their own count.
    pending: Option<LeadOffKind>,
}

impl LeadOffDetector {
    pub fn new(cfg: LeadOffConfig) -> Self {
        LeadOffDetector {
            cfg,
            holding: 0,
            clearing: 0,
            open: None,
            pending: None,
        }
    }

    /// Which failure this window looks like, if any.
    fn judge(&self, q: &QualitySample, _cfg: &QualityConfig) -> Option<LeadOffKind> {
        let f = &q.features;
        if f.sat_frac >= self.cfg.sat_frac || q.flags & flags::SATURATION != 0 {
            return Some(LeadOffKind::RailContact);
        }
        if f.p2p_rel >= self.cfg.open_p2p_rel && f.qrs_ratio_rel <= self.cfg.open_qrs_rel {
            return Some(LeadOffKind::OpenInput);
        }
        None
    }

    /// Feed one quality hop, at sample position `now`, with how long it has
    /// been since the last detected beat.
    pub fn push(
        &mut self,
        now: u64,
        q: &QualitySample,
        cfg: &QualityConfig,
        samples_since_beat: u64,
    ) -> Option<LeadOffEpisode> {
        let beating = (samples_since_beat as f64) < self.cfg.quiet_ms * cfg.fs / 1000.0;
        let judged = self.judge(q, cfg).or_else(|| {
            // Already off, and still no beats: the same electrode, now settled.
            // The prior is what makes this readable, and it is why an absence
            // of beats can hold an episode open and can never open one.
            self.open.map(|(k, _)| k).filter(|_| !beating)
        });
        match judged {
            Some(kind) => {
                self.clearing = 0;
                if self.pending != Some(kind) {
                    self.pending = Some(kind);
                    self.holding = 1;
                } else {
                    self.holding = self.holding.saturating_add(1);
                }
                if self.open.is_none() && self.holding >= self.cfg.enter_hops {
                    self.open = Some((kind, now));
                }
                None
            }
            None => {
                self.holding = 0;
                self.pending = None;
                self.clearing = self.clearing.saturating_add(1);
                if self.clearing >= self.cfg.exit_hops {
                    if let Some((kind, start)) = self.open.take() {
                        return Some(LeadOffEpisode {
                            kind,
                            start,
                            end: now,
                        });
                    }
                }
                None
            }
        }
    }

    /// Whether an electrode is currently believed to be off.
    pub fn off(&self) -> Option<LeadOffKind> {
        self.open.map(|(k, _)| k)
    }

    /// Close an episode still open at the end of the stream.
    pub fn finish(&mut self, now: u64) -> Option<LeadOffEpisode> {
        self.open.take().map(|(kind, start)| LeadOffEpisode {
            kind,
            start,
            end: now,
        })
    }

    pub fn reset(&mut self) {
        self.holding = 0;
        self.clearing = 0;
        self.open = None;
        self.pending = None;
    }
}
