//! Episode confirmation.
//!
//! A per-window verdict is not a clinical finding. Atrial fibrillation is
//! reported when it is *sustained* - thirty seconds is the usual threshold - and
//! a run of AF is not ended by one window that happened to look regular. Two
//! rules turn a noisy state signal into episodes:
//!
//! * **Bridge** short gaps. A few regular-looking windows inside a long
//!   fibrillating stretch are a property of the measurement, not of the rhythm.
//! * **Confirm** only runs that reach the minimum duration. Everything shorter is
//!   dropped.
//!
//! This costs latency, unavoidably: an episode cannot be confirmed before it has
//! lasted long enough to qualify. The confirmed episode is reported with its true
//! start, so the record is right even though the alarm is late.

#[derive(Debug, Clone, Copy)]
pub struct EpisodeConfig {
    /// Gaps shorter than this do not end an episode.
    pub bridge_s: f32,
    /// Runs shorter than this are never reported.
    pub min_episode_s: f32,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        EpisodeConfig {
            bridge_s: 15.0,
            min_episode_s: 30.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Episode {
    /// First and last sample of the episode, on the input time base.
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, Copy)]
struct Open {
    start: u64,
    /// Most recent sample that was positive.
    last_positive: u64,
}

pub struct EpisodeTracker {
    cfg: EpisodeConfig,
    bridge: u64,
    minimum: u64,
    open: Option<Open>,
    confirmed: bool,
}

impl EpisodeTracker {
    pub fn new(fs: f64, cfg: EpisodeConfig) -> Self {
        EpisodeTracker {
            cfg,
            bridge: (cfg.bridge_s as f64 * fs) as u64,
            minimum: (cfg.min_episode_s as f64 * fs) as u64,
            open: None,
            confirmed: false,
        }
    }

    pub fn config(&self) -> &EpisodeConfig {
        &self.cfg
    }

    /// True once the currently open run has lasted long enough to report.
    pub fn confirmed(&self) -> bool {
        self.confirmed
    }

    /// Feed one verdict. Returns an episode when a confirmed run ends.
    pub fn update(&mut self, sample: u64, positive: bool) -> Option<Episode> {
        match self.open.as_mut() {
            Some(o) => {
                if positive {
                    o.last_positive = sample;
                } else if sample.saturating_sub(o.last_positive) > self.bridge {
                    let o = self.open.take().unwrap();
                    let was = self.confirmed;
                    self.confirmed = false;
                    if was {
                        return Some(Episode {
                            start: o.start,
                            end: o.last_positive,
                        });
                    }
                    return None;
                }
                if !self.confirmed && o.last_positive.saturating_sub(o.start) >= self.minimum {
                    self.confirmed = true;
                }
                None
            }
            None => {
                if positive {
                    self.open = Some(Open {
                        start: sample,
                        last_positive: sample,
                    });
                    // A zero-length minimum means confirm immediately.
                    self.confirmed = self.minimum == 0;
                }
                None
            }
        }
    }

    /// Close any open episode at the end of the stream.
    pub fn finish(&mut self) -> Option<Episode> {
        let o = self.open.take()?;
        let was = self.confirmed;
        self.confirmed = false;
        was.then_some(Episode {
            start: o.start,
            end: o.last_positive,
        })
    }

    pub fn reset(&mut self) {
        self.open = None;
        self.confirmed = false;
    }
}
