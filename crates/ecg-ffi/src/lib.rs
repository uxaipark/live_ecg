//! The engine's standard interface.
//!
//! # Why there is one
//!
//! The engine is meant to be replaced often: a better model, a new detector,
//! a retuned bar. What must not change each time is everything around it - the
//! server that feeds hundreds of channels, the phone app, the Pi gateway. So
//! the boundary between the two is fixed here, and it is small: create a
//! channel, push samples, poll events, read a status, destroy it. Everything
//! the engine finds comes back as one record shape, [`EcgEvent`], told apart by
//! `kind` and `code`, so a new engine can report something new without the old
//! host breaking - a host skips a `kind` it does not know.
//!
//! The same boundary is exposed twice: as a safe Rust API, [`Engine`], and as
//! a C ABI over it (`ecg_*`, declared in `include/ecg.h`). A host in any
//! language links the C library or loads it at run time; replacing the file
//! replaces the engine.
//!
//! # The rules that keep engines interchangeable
//!
//! * `ecg_abi_version()` is `major << 16 | minor`. A host refuses an engine
//!   whose major differs from the one it was written for. Within a major,
//!   engines only ever *add*: new event kinds, new codes, new fields at the end
//!   of [`EcgConfig`] and [`EcgStatus`]. Nothing is renumbered or removed.
//! * Every struct a host passes in or reads back starts with `struct_size`,
//!   so an older host and a newer engine agree on how much of it exists.
//! * Codes are numbered here and in the header, and mapped from the engine's
//!   own types by an explicit table. Renaming or reordering an enum inside the
//!   engine does not move a number the host depends on.
//! * Nothing unwinds across the boundary. A panic inside the engine becomes
//!   [`ECG_ERR_INTERNAL`], and the channel it happened in refuses further work
//!   with [`ECG_ERR_POISONED`] rather than continue from a state nobody
//!   checked.

use ecg_beats::{BeatClass, BeatVerdict};
use ecg_pipeline::{ChannelOutput, ChannelPipeline, PipelineConfig};
use ecg_quality::{LeadOffKind, Quality};
use ecg_rhythm::Condition;
use std::collections::VecDeque;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub const ABI_MAJOR: u32 = 1;
pub const ABI_MINOR: u32 = 0;

/// Identifies the engine behind the interface. Replaced with the source
/// revision when the single-file engine is generated.
pub const ENGINE_ID: &str = "live-ecg (workspace build)\0";

// ---- result codes -------------------------------------------------------

pub const ECG_OK: i32 = 0;
pub const ECG_ERR_NULL: i32 = -1;
pub const ECG_ERR_CONFIG: i32 = -2;
pub const ECG_ERR_INTERNAL: i32 = -3;
pub const ECG_ERR_POISONED: i32 = -4;

// ---- presets ------------------------------------------------------------

/// Clinical electrodes: the default configuration, fitted and tuned on the
/// public corpora.
pub const ECG_PRESET_CLINICAL: u32 = 0;
/// A single-lead patch worn for days: `PipelineConfig::patch`.
pub const ECG_PRESET_PATCH: u32 = 1;

// ---- event kinds and codes ----------------------------------------------

/// One classified beat. `start == end` is its R peak. `code` is the class,
/// `score` holds the ventricular, supraventricular and fusion detectors'
/// probabilities, `aux` the morphology the beat joined, and flag bit 0 says
/// the beat was judged inside sustained atrial fibrillation.
pub const ECG_EV_BEAT: u32 = 1;
pub const ECG_BEAT_N: u32 = 0;
pub const ECG_BEAT_S: u32 = 1;
pub const ECG_BEAT_V: u32 = 2;
pub const ECG_BEAT_F: u32 = 3;
/// The analyser declined to judge it: signal quality, or no template yet.
pub const ECG_BEAT_UNKNOWN: u32 = 4;
pub const ECG_BEAT_FLAG_FIBRILLATING: u32 = 1;

/// A rhythm episode that has ended. `code` is the condition.
pub const ECG_EV_RHYTHM: u32 = 2;
pub const ECG_RHYTHM_PAUSE: u32 = 1;
pub const ECG_RHYTHM_ASYSTOLE: u32 = 2;
pub const ECG_RHYTHM_BRADYCARDIA: u32 = 3;
pub const ECG_RHYTHM_TACHYCARDIA: u32 = 4;
pub const ECG_RHYTHM_VENTRICULAR_RUN: u32 = 5;
pub const ECG_RHYTHM_VENTRICULAR_TACHYCARDIA: u32 = 6;
pub const ECG_RHYTHM_BIGEMINY: u32 = 7;
pub const ECG_RHYTHM_TRIGEMINY: u32 = 8;
pub const ECG_RHYTHM_IDIOVENTRICULAR: u32 = 9;

/// One atrial fibrillation decision window, `start..end`. `score[0]` is the
/// probability; flag bit 0 is the detector's state after it.
pub const ECG_EV_AF_WINDOW: u32 = 3;
pub const ECG_AF_FLAG_IN_AF: u32 = 1;

/// A ventricular fibrillation episode that has ended.
pub const ECG_EV_VF: u32 = 4;

/// An electrode failure that has ended. `code` is its kind.
pub const ECG_EV_LEAD_OFF: u32 = 5;
pub const ECG_LEAD_OFF_RAIL: u32 = 1;
pub const ECG_LEAD_OFF_OPEN: u32 = 2;

/// A run of supraventricular rhythm that has ended; `aux` is its beat count.
/// A consumer reports a beat inside it that was classified N as S.
pub const ECG_EV_SV_RUN: u32 = 6;

// ---- status -------------------------------------------------------------

pub const ECG_QUALITY_GOOD: u32 = 0;
pub const ECG_QUALITY_ACCEPTABLE: u32 = 1;
pub const ECG_QUALITY_UNUSABLE: u32 = 2;
pub const ECG_QUALITY_UNKNOWN: u32 = 3;

pub const ECG_STATE_IN_AF: u32 = 1;
pub const ECG_STATE_IN_VF: u32 = 2;
pub const ECG_STATE_LEAD_OFF: u32 = 4;
/// Beat-derived findings are being withheld because fibrillation is suspected.
pub const ECG_STATE_SUPPRESSING: u32 = 8;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EcgConfig {
    pub struct_size: u32,
    pub preset: u32,
    /// Sampling rate, Hz.
    pub fs: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EcgEvent {
    pub kind: u32,
    pub code: u32,
    pub flags: u32,
    pub aux: u32,
    /// Sample indices on this channel's own time base, counting from zero at
    /// the first sample pushed and advanced by gaps.
    pub start: u64,
    pub end: u64,
    pub score: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EcgStatus {
    pub struct_size: u32,
    pub quality: u32,
    pub state: u32,
    pub reserved: u32,
    pub samples: u64,
    /// Signal quality, 0..1, 1 pristine; negative before the first window.
    pub quality_score: f32,
}

/// One channel of the engine, behind the standard interface.
pub struct Engine {
    pipe: ChannelPipeline,
    out: ChannelOutput,
    queue: VecDeque<EcgEvent>,
    quality: u32,
    quality_score: f32,
}

impl Engine {
    pub fn new(cfg: &EcgConfig) -> Result<Engine, i32> {
        if !(cfg.fs.is_finite() && (50.0..=4000.0).contains(&cfg.fs)) {
            return Err(ECG_ERR_CONFIG);
        }
        let pc = match cfg.preset {
            ECG_PRESET_CLINICAL => PipelineConfig::new(cfg.fs),
            ECG_PRESET_PATCH => PipelineConfig::patch(cfg.fs),
            _ => return Err(ECG_ERR_CONFIG),
        };
        Ok(Engine {
            pipe: ChannelPipeline::new(pc),
            out: ChannelOutput::default(),
            queue: VecDeque::new(),
            quality: ECG_QUALITY_UNKNOWN,
            quality_score: -1.0,
        })
    }

    /// Samples in millivolts.
    pub fn push(&mut self, samples: &[f32]) {
        self.out.clear();
        self.pipe.push(samples, &mut self.out);
        self.drain();
    }

    /// `samples` were lost; the time base advances past them.
    pub fn gap(&mut self, samples: u64) {
        self.pipe.mark_gap(samples);
    }

    /// End of stream: close whatever is open.
    pub fn finish(&mut self) {
        self.out.clear();
        self.pipe.finish(&mut self.out);
        self.drain();
    }

    /// Move up to `out.len()` queued events into `out`, oldest first.
    pub fn poll(&mut self, out: &mut [EcgEvent]) -> usize {
        let n = out.len().min(self.queue.len());
        for slot in out.iter_mut().take(n) {
            *slot = self.queue.pop_front().expect("counted");
        }
        n
    }

    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    pub fn status(&self) -> EcgStatus {
        let mut state = 0;
        if self.pipe.in_af() {
            state |= ECG_STATE_IN_AF;
        }
        if self.pipe.in_vf() {
            state |= ECG_STATE_IN_VF;
        }
        if self.pipe.lead_off().is_some() {
            state |= ECG_STATE_LEAD_OFF;
        }
        if self.pipe.suppressing() {
            state |= ECG_STATE_SUPPRESSING;
        }
        EcgStatus {
            struct_size: std::mem::size_of::<EcgStatus>() as u32,
            quality: self.quality,
            state,
            reserved: 0,
            samples: self.pipe.samples_processed(),
            quality_score: self.quality_score,
        }
    }

    fn drain(&mut self) {
        let o = &self.out;
        if let Some(q) = o.quality {
            self.quality = match q.level(&self.pipe.config().quality) {
                Quality::Good => ECG_QUALITY_GOOD,
                Quality::Acceptable => ECG_QUALITY_ACCEPTABLE,
                Quality::Unusable => ECG_QUALITY_UNUSABLE,
            };
            self.quality_score = q.score;
        }
        for v in &o.classes {
            self.queue.push_back(beat_event(v));
        }
        for e in &o.episodes {
            self.queue.push_back(EcgEvent {
                kind: ECG_EV_RHYTHM,
                code: rhythm_code(e.condition),
                start: e.start,
                end: e.end,
                ..EcgEvent::default()
            });
        }
        for w in &o.af {
            self.queue.push_back(EcgEvent {
                kind: ECG_EV_AF_WINDOW,
                flags: if w.in_af { ECG_AF_FLAG_IN_AF } else { 0 },
                start: w.start_sample,
                end: w.sample,
                score: [w.probability, 0.0, 0.0, 0.0],
                ..EcgEvent::default()
            });
        }
        for &(s, e) in &o.vf_episodes {
            self.queue.push_back(EcgEvent {
                kind: ECG_EV_VF,
                start: s,
                end: e,
                ..EcgEvent::default()
            });
        }
        for l in &o.lead_off {
            self.queue.push_back(EcgEvent {
                kind: ECG_EV_LEAD_OFF,
                code: match l.kind {
                    LeadOffKind::RailContact => ECG_LEAD_OFF_RAIL,
                    LeadOffKind::OpenInput => ECG_LEAD_OFF_OPEN,
                },
                start: l.start,
                end: l.end,
                ..EcgEvent::default()
            });
        }
        for r in &o.sv_runs {
            self.queue.push_back(EcgEvent {
                kind: ECG_EV_SV_RUN,
                aux: r.beats,
                start: r.start,
                end: r.end,
                ..EcgEvent::default()
            });
        }
    }
}

fn beat_event(v: &BeatVerdict) -> EcgEvent {
    EcgEvent {
        kind: ECG_EV_BEAT,
        code: match v.class {
            BeatClass::N => ECG_BEAT_N,
            BeatClass::S => ECG_BEAT_S,
            BeatClass::V => ECG_BEAT_V,
            BeatClass::F => ECG_BEAT_F,
            BeatClass::Unknown => ECG_BEAT_UNKNOWN,
        },
        flags: if v.context.fibrillating {
            ECG_BEAT_FLAG_FIBRILLATING
        } else {
            0
        },
        aux: v.cluster,
        start: v.sample,
        end: v.sample,
        score: [v.p_ventricular, v.p_supraventricular, v.p_fusion, 0.0],
    }
}

/// The numbering is the interface's, not the engine's: a condition added to
/// the engine gets the next free number here, and none is ever reused.
fn rhythm_code(c: Condition) -> u32 {
    match c {
        Condition::Pause => ECG_RHYTHM_PAUSE,
        Condition::Asystole => ECG_RHYTHM_ASYSTOLE,
        Condition::Bradycardia => ECG_RHYTHM_BRADYCARDIA,
        Condition::Tachycardia => ECG_RHYTHM_TACHYCARDIA,
        Condition::VentricularRun => ECG_RHYTHM_VENTRICULAR_RUN,
        Condition::VentricularTachycardia => ECG_RHYTHM_VENTRICULAR_TACHYCARDIA,
        Condition::Bigeminy => ECG_RHYTHM_BIGEMINY,
        Condition::Trigeminy => ECG_RHYTHM_TRIGEMINY,
        Condition::Idioventricular => ECG_RHYTHM_IDIOVENTRICULAR,
    }
}

// ---- the C ABI ----------------------------------------------------------

/// What a C handle points at: the engine, and whether it is still trusted.
pub struct EcgChannel {
    engine: Engine,
    poisoned: bool,
}

fn guarded<T>(ch: &mut EcgChannel, f: impl FnOnce(&mut Engine) -> T) -> Result<T, i32> {
    if ch.poisoned {
        return Err(ECG_ERR_POISONED);
    }
    match catch_unwind(AssertUnwindSafe(|| f(&mut ch.engine))) {
        Ok(v) => Ok(v),
        Err(_) => {
            ch.poisoned = true;
            Err(ECG_ERR_INTERNAL)
        }
    }
}

#[no_mangle]
pub extern "C" fn ecg_abi_version() -> u32 {
    (ABI_MAJOR << 16) | ABI_MINOR
}

#[no_mangle]
pub extern "C" fn ecg_engine_id() -> *const c_char {
    ENGINE_ID.as_ptr() as *const c_char
}

/// # Safety
/// `cfg` must point at a readable `EcgConfig` of at least `cfg.struct_size`
/// bytes; `err` may be null.
#[no_mangle]
pub unsafe extern "C" fn ecg_channel_create(
    cfg: *const EcgConfig,
    err: *mut i32,
) -> *mut EcgChannel {
    let set = |code: i32| {
        if !err.is_null() {
            *err = code;
        }
    };
    if cfg.is_null() {
        set(ECG_ERR_NULL);
        return std::ptr::null_mut();
    }
    // Only the fields the caller says it has; this version needs all three.
    let size = std::ptr::read_unaligned(cfg as *const u32) as usize;
    if size < std::mem::size_of::<EcgConfig>() {
        set(ECG_ERR_CONFIG);
        return std::ptr::null_mut();
    }
    let c = std::ptr::read_unaligned(cfg);
    match catch_unwind(|| Engine::new(&c)) {
        Ok(Ok(engine)) => {
            set(ECG_OK);
            Box::into_raw(Box::new(EcgChannel {
                engine,
                poisoned: false,
            }))
        }
        Ok(Err(code)) => {
            set(code);
            std::ptr::null_mut()
        }
        Err(_) => {
            set(ECG_ERR_INTERNAL);
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `ch` must be null or a handle from `ecg_channel_create` not yet destroyed.
#[no_mangle]
pub unsafe extern "C" fn ecg_channel_destroy(ch: *mut EcgChannel) {
    if !ch.is_null() {
        drop(Box::from_raw(ch));
    }
}

/// # Safety
/// `ch` a live handle; `samples` readable for `n` floats (may be null if `n`
/// is zero).
#[no_mangle]
pub unsafe extern "C" fn ecg_channel_push(
    ch: *mut EcgChannel,
    samples: *const f32,
    n: usize,
) -> i32 {
    let Some(ch) = ch.as_mut() else {
        return ECG_ERR_NULL;
    };
    if n == 0 {
        return ECG_OK;
    }
    if samples.is_null() {
        return ECG_ERR_NULL;
    }
    let s = std::slice::from_raw_parts(samples, n);
    guarded(ch, |e| e.push(s)).map_or_else(|c| c, |_| ECG_OK)
}

/// # Safety
/// `ch` a live handle.
#[no_mangle]
pub unsafe extern "C" fn ecg_channel_gap(ch: *mut EcgChannel, samples: u64) -> i32 {
    let Some(ch) = ch.as_mut() else {
        return ECG_ERR_NULL;
    };
    guarded(ch, |e| e.gap(samples)).map_or_else(|c| c, |_| ECG_OK)
}

/// # Safety
/// `ch` a live handle.
#[no_mangle]
pub unsafe extern "C" fn ecg_channel_finish(ch: *mut EcgChannel) -> i32 {
    let Some(ch) = ch.as_mut() else {
        return ECG_ERR_NULL;
    };
    guarded(ch, |e| e.finish()).map_or_else(|c| c, |_| ECG_OK)
}

/// Copies up to `cap` events into `out` and returns how many; a negative
/// value is an error code.
///
/// # Safety
/// `ch` a live handle; `out` writable for `cap` events.
#[no_mangle]
pub unsafe extern "C" fn ecg_channel_poll(
    ch: *mut EcgChannel,
    out: *mut EcgEvent,
    cap: usize,
) -> i64 {
    let Some(ch) = ch.as_mut() else {
        return ECG_ERR_NULL as i64;
    };
    if cap == 0 {
        return 0;
    }
    if out.is_null() {
        return ECG_ERR_NULL as i64;
    }
    let slots = std::slice::from_raw_parts_mut(out, cap);
    guarded(ch, |e| e.poll(slots)).map_or_else(|c| c as i64, |n| n as i64)
}

/// Fills `out` up to the smaller of its `struct_size` and this engine's.
///
/// # Safety
/// `ch` a live handle; `out` writable for `out.struct_size` bytes, which the
/// caller sets before the call.
#[no_mangle]
pub unsafe extern "C" fn ecg_channel_status(ch: *mut EcgChannel, out: *mut EcgStatus) -> i32 {
    let Some(ch) = ch.as_mut() else {
        return ECG_ERR_NULL;
    };
    if out.is_null() {
        return ECG_ERR_NULL;
    }
    let want = std::ptr::read_unaligned(out as *const u32) as usize;
    match guarded(ch, |e| e.status()) {
        Ok(s) => {
            let n = want.min(std::mem::size_of::<EcgStatus>());
            std::ptr::copy_nonoverlapping(&s as *const EcgStatus as *const u8, out as *mut u8, n);
            // The size written back is what was filled.
            std::ptr::write_unaligned(out as *mut u32, n as u32);
            ECG_OK
        }
        Err(c) => c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(fs: f64, seconds: f64) -> Vec<f32> {
        // One narrow complex a second on a quiet baseline, with a T wave.
        let n = (fs * seconds) as usize;
        (0..n)
            .map(|i| {
                let t = (i as f64 / fs) % 1.0;
                let qrs = (-((t - 0.3) / 0.012).powi(2)).exp();
                let tw = 0.25 * (-((t - 0.6) / 0.06).powi(2)).exp();
                (qrs + tw) as f32
            })
            .collect()
    }

    #[test]
    fn beats_come_back_through_the_interface() {
        let cfg = EcgConfig {
            struct_size: std::mem::size_of::<EcgConfig>() as u32,
            preset: ECG_PRESET_CLINICAL,
            fs: 250.0,
        };
        let mut e = Engine::new(&cfg).unwrap();
        for chunk in synthetic(250.0, 60.0).chunks(250) {
            e.push(chunk);
        }
        e.finish();
        let mut buf = vec![EcgEvent::default(); 4096];
        let n = e.poll(&mut buf);
        let beats: Vec<_> = buf[..n].iter().filter(|x| x.kind == ECG_EV_BEAT).collect();
        assert!((50..=62).contains(&beats.len()), "{} beats", beats.len());
        assert!(beats.windows(2).all(|w| w[0].start < w[1].start));
    }

    #[test]
    fn a_bad_configuration_is_refused_not_trusted() {
        for (preset, fs) in [
            (9, 250.0),
            (ECG_PRESET_CLINICAL, 0.0),
            (ECG_PRESET_PATCH, f64::NAN),
        ] {
            let cfg = EcgConfig {
                struct_size: std::mem::size_of::<EcgConfig>() as u32,
                preset,
                fs,
            };
            assert_eq!(Engine::new(&cfg).err(), Some(ECG_ERR_CONFIG));
        }
    }

    #[test]
    fn a_short_status_struct_is_filled_only_as_far_as_it_goes() {
        let cfg = EcgConfig {
            struct_size: std::mem::size_of::<EcgConfig>() as u32,
            preset: ECG_PRESET_CLINICAL,
            fs: 250.0,
        };
        let mut err = 0;
        let ch = unsafe { ecg_channel_create(&cfg, &mut err) };
        assert_eq!(err, ECG_OK);
        // An older host that knows only the first three fields.
        let mut buf = [0xAAu8; 32];
        buf[..4].copy_from_slice(&12u32.to_ne_bytes());
        let rc = unsafe { ecg_channel_status(ch, buf.as_mut_ptr() as *mut EcgStatus) };
        assert_eq!(rc, ECG_OK);
        assert_eq!(u32::from_ne_bytes(buf[..4].try_into().unwrap()), 12);
        assert!(
            buf[12..].iter().all(|&b| b == 0xAA),
            "wrote past what the host has"
        );
        unsafe { ecg_channel_destroy(ch) };
    }
}
