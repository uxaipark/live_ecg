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
    /// The electrode was attached and unsaturated across the interval.
    ///
    /// Kept apart from `quality_ok` because asystole and a detached lead look
    /// alike: both are flat. Gating asystole on general quality therefore
    /// suppresses exactly the thing it is meant to catch, and reporting "signal
    /// lost" when a patient is in asystole is the wrong failure to prefer.
    /// Electrode integrity is evidence a flat trace *is* the patient.
    pub lead_ok: bool,
    /// Amplitude of the closing beat, mV.
    pub amplitude: f32,
    /// A beat bounding this interval was classified ventricular.
    ///
    /// An ectopic beat distorts the interval before it (early) and the one after
    /// it (the pause), so both are evidence about the ectopy rather than about
    /// the underlying rhythm. Set by the pipeline once the beat classifier has
    /// ruled, which is why the rhythm path lags the detector.
    pub ventricular: bool,
    /// A beat bounding this interval was classified supraventricular.
    ///
    /// Kept apart from `ventricular` because the two cannot be used the same
    /// way. Ventricular beats are identified from morphology, which is
    /// independent of the timing that rhythm analysis measures. Supraventricular
    /// beats are identified from prematurity - the same evidence - so filtering a
    /// rhythm series by them is circular.
    pub supraventricular: bool,
    /// Correlation between the atrial segments of this interval's closing beat
    /// and the beat before it.
    ///
    /// High when consecutive P waves come from the same sinus node, near zero
    /// when there is no organised atrial activity to repeat. Set by the
    /// pipeline from the beat analyser, so it arrives on the same one-beat lag
    /// as the classification does; 0.0 means "not measured", which is why the
    /// window takes a median over the intervals that carried a value rather
    /// than over all of them.
    pub atrial_coherence: f32,
    /// This interval is immediately adjacent in time to the previous one the
    /// consumer accepted.
    ///
    /// Successive-difference features are only meaningful across an adjacent
    /// pair. Once intervals can be withheld - for quality, and now for ectopy -
    /// two neighbouring entries in a window need not be neighbours in time, and
    /// differencing them invents an irregularity that never happened.
    pub continuous: bool,
}

impl RrSample {
    /// Usable as evidence about rhythm.
    #[inline]
    pub fn usable(&self) -> bool {
        self.physiological && self.quality_ok
    }

    /// Usable as evidence about the underlying rhythm under a given exclusion
    /// policy.
    #[inline]
    pub fn usable_excluding(&self, ventricular: bool, supraventricular: bool) -> bool {
        self.usable()
            && !(ventricular && self.ventricular)
            && !(supraventricular && self.supraventricular)
    }
}

#[derive(Debug, Clone)]
pub struct RrStream {
    cfg: RrConfig,
    last: Option<u64>,
    /// Whether every sample since the previous beat passed the quality gate.
    clean_since_last: bool,
    /// Whether the electrode stayed attached across the interval.
    lead_since_last: bool,
    /// Whether the previous beat produced an interval at all.
    emitted_last: bool,
}

impl RrStream {
    pub fn new(cfg: RrConfig) -> Self {
        RrStream {
            cfg,
            last: None,
            clean_since_last: true,
            lead_since_last: true,
            emitted_last: false,
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

    /// Record electrode integrity for the sample just consumed.
    #[inline]
    pub fn observe_lead(&mut self, ok: bool) {
        self.lead_since_last &= ok;
    }

    /// Close an interval on a new beat.
    pub fn push(&mut self, ev: &QrsEvent) -> Option<RrSample> {
        let prev = self.last.replace(ev.sample);
        let clean = std::mem::replace(&mut self.clean_since_last, true);
        let lead = std::mem::replace(&mut self.lead_since_last, true);
        let emitted = std::mem::replace(&mut self.emitted_last, true);
        let prev = prev?;
        if ev.sample <= prev {
            self.emitted_last = false;
            return None;
        }
        let rr_ms = (ev.sample - prev) as f32 * 1000.0 / self.cfg.fs as f32;
        Some(RrSample {
            sample: ev.sample,
            rr_ms,
            physiological: rr_ms >= self.cfg.min_rr_ms && rr_ms <= self.cfg.max_rr_ms,
            quality_ok: clean,
            lead_ok: lead,
            amplitude: ev.amplitude,
            ventricular: false,
            supraventricular: false,
            atrial_coherence: 0.0,
            continuous: emitted,
        })
    }

    pub fn reset(&mut self) {
        self.last = None;
        self.clean_since_last = true;
        self.lead_since_last = true;
        self.emitted_last = false;
    }
}
