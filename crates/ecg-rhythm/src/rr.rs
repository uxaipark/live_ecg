//! RR interval stream.
//!
//! Turns the detector's beat events into intervals, and marks the ones that
//! downstream rhythm logic must not believe. Two things make an interval
//! untrustworthy, and they are kept separate because they mean different things:
//!
//! * **Non-physiological** — outside 200 ms..3 s. Almost always a missed or
//!   spurious detection rather than a heartbeat.
//! * **Poor signal quality** — the beat may be real, but the interval spanning a
//!   noise burst cannot be used as evidence about rhythm.
//!
//! Nothing is interpolated or repaired here. An interval that cannot be trusted
//! is marked, not invented: a fabricated interval is indistinguishable from a
//! measured one downstream, and AF detection is precisely a judgement about how
//! irregular the measured intervals are.

use ecg_qrs::QrsEvent;

#[derive(Debug, Clone, Copy)]
pub struct RrConfig {
    pub fs: f64,
    pub min_rr_ms: f32,
    pub max_rr_ms: f32,
}

impl RrConfig {
    pub fn new(fs: f64) -> Self {
        RrConfig {
            fs,
            min_rr_ms: 200.0,
            max_rr_ms: 3000.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RrSample {
    /// Sample index of the beat that closes this interval.
    pub sample: u64,
    /// Interval to the previous beat, milliseconds.
    pub rr_ms: f32,
    /// Within physiological bounds.
    pub physiological: bool,
    /// Signal quality was acceptable across the interval.
    pub quality_ok: bool,
    /// Amplitude of the closing beat, mV. Carried for later ectopy work.
    pub amplitude: f32,
}

impl RrSample {
    /// Usable as evidence about rhythm.
    #[inline]
    pub fn usable(&self) -> bool {
        self.physiological && self.quality_ok
    }
}

#[derive(Debug, Clone)]
pub struct RrStream {
    cfg: RrConfig,
    last: Option<u64>,
    /// Whether every sample since the previous beat passed the quality gate.
    clean_since_last: bool,
}

impl RrStream {
    pub fn new(cfg: RrConfig) -> Self {
        RrStream {
            cfg,
            last: None,
            clean_since_last: true,
        }
    }

    pub fn config(&self) -> &RrConfig {
        &self.cfg
    }

    /// Record the quality verdict for the sample just consumed. The interval
    /// inherits the worst verdict seen across its whole span, not just the one
    /// at the beat: a noise burst in the middle is what corrupts an interval.
    #[inline]
    pub fn observe_quality(&mut self, ok: bool) {
        self.clean_since_last &= ok;
    }

    /// Close an interval on a new beat.
    pub fn push(&mut self, ev: &QrsEvent) -> Option<RrSample> {
        let prev = self.last.replace(ev.sample);
        let clean = std::mem::replace(&mut self.clean_since_last, true);
        let prev = prev?;
        if ev.sample <= prev {
            return None;
        }
        let rr_ms = (ev.sample - prev) as f32 * 1000.0 / self.cfg.fs as f32;
        Some(RrSample {
            sample: ev.sample,
            rr_ms,
            physiological: rr_ms >= self.cfg.min_rr_ms && rr_ms <= self.cfg.max_rr_ms,
            quality_ok: clean,
            amplitude: ev.amplitude,
        })
    }

    pub fn reset(&mut self) {
        self.last = None;
        self.clean_since_last = true;
    }
}
