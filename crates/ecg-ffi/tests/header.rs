//! `include/ecg.h` and this crate describe one interface. Every number the
//! header defines must be the number the engine uses, and every struct the
//! size the engine reads and writes - otherwise a host compiled against the
//! header and an engine built from this crate disagree without either noticing.

use ecg_ffi::*;
use std::collections::HashMap;

fn defines() -> HashMap<String, i64> {
    let text = include_str!("../include/ecg.h");
    let mut out = HashMap::new();
    for line in text.lines() {
        let t = line.trim_start();
        let Some(rest) = t.strip_prefix("#define ") else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let (Some(name), Some(value)) = (it.next(), it.next()) else {
            continue;
        };
        if let Ok(v) = value.trim_end_matches('u').parse::<i64>() {
            out.insert(name.to_string(), v);
        }
    }
    out
}

#[test]
fn the_header_and_the_engine_number_things_the_same_way() {
    let d = defines();
    let expect: &[(&str, i64)] = &[
        ("ECG_ABI_MAJOR", ABI_MAJOR as i64),
        ("ECG_ABI_MINOR", ABI_MINOR as i64),
        ("ECG_OK", ECG_OK as i64),
        ("ECG_ERR_NULL", ECG_ERR_NULL as i64),
        ("ECG_ERR_CONFIG", ECG_ERR_CONFIG as i64),
        ("ECG_ERR_INTERNAL", ECG_ERR_INTERNAL as i64),
        ("ECG_ERR_POISONED", ECG_ERR_POISONED as i64),
        ("ECG_PRESET_CLINICAL", ECG_PRESET_CLINICAL as i64),
        ("ECG_PRESET_PATCH", ECG_PRESET_PATCH as i64),
        ("ECG_EV_BEAT", ECG_EV_BEAT as i64),
        ("ECG_BEAT_N", ECG_BEAT_N as i64),
        ("ECG_BEAT_S", ECG_BEAT_S as i64),
        ("ECG_BEAT_V", ECG_BEAT_V as i64),
        ("ECG_BEAT_F", ECG_BEAT_F as i64),
        ("ECG_BEAT_UNKNOWN", ECG_BEAT_UNKNOWN as i64),
        (
            "ECG_BEAT_FLAG_FIBRILLATING",
            ECG_BEAT_FLAG_FIBRILLATING as i64,
        ),
        ("ECG_EV_RHYTHM", ECG_EV_RHYTHM as i64),
        ("ECG_RHYTHM_PAUSE", ECG_RHYTHM_PAUSE as i64),
        ("ECG_RHYTHM_ASYSTOLE", ECG_RHYTHM_ASYSTOLE as i64),
        ("ECG_RHYTHM_BRADYCARDIA", ECG_RHYTHM_BRADYCARDIA as i64),
        ("ECG_RHYTHM_TACHYCARDIA", ECG_RHYTHM_TACHYCARDIA as i64),
        (
            "ECG_RHYTHM_VENTRICULAR_RUN",
            ECG_RHYTHM_VENTRICULAR_RUN as i64,
        ),
        (
            "ECG_RHYTHM_VENTRICULAR_TACHYCARDIA",
            ECG_RHYTHM_VENTRICULAR_TACHYCARDIA as i64,
        ),
        ("ECG_RHYTHM_BIGEMINY", ECG_RHYTHM_BIGEMINY as i64),
        ("ECG_RHYTHM_TRIGEMINY", ECG_RHYTHM_TRIGEMINY as i64),
        (
            "ECG_RHYTHM_IDIOVENTRICULAR",
            ECG_RHYTHM_IDIOVENTRICULAR as i64,
        ),
        ("ECG_EV_AF_WINDOW", ECG_EV_AF_WINDOW as i64),
        ("ECG_AF_FLAG_IN_AF", ECG_AF_FLAG_IN_AF as i64),
        ("ECG_EV_VF", ECG_EV_VF as i64),
        ("ECG_EV_LEAD_OFF", ECG_EV_LEAD_OFF as i64),
        ("ECG_LEAD_OFF_RAIL", ECG_LEAD_OFF_RAIL as i64),
        ("ECG_LEAD_OFF_OPEN", ECG_LEAD_OFF_OPEN as i64),
        ("ECG_EV_SV_RUN", ECG_EV_SV_RUN as i64),
        ("ECG_QUALITY_GOOD", ECG_QUALITY_GOOD as i64),
        ("ECG_QUALITY_ACCEPTABLE", ECG_QUALITY_ACCEPTABLE as i64),
        ("ECG_QUALITY_UNUSABLE", ECG_QUALITY_UNUSABLE as i64),
        ("ECG_QUALITY_UNKNOWN", ECG_QUALITY_UNKNOWN as i64),
        ("ECG_STATE_IN_AF", ECG_STATE_IN_AF as i64),
        ("ECG_STATE_IN_VF", ECG_STATE_IN_VF as i64),
        ("ECG_STATE_LEAD_OFF", ECG_STATE_LEAD_OFF as i64),
        ("ECG_STATE_SUPPRESSING", ECG_STATE_SUPPRESSING as i64),
    ];
    for (name, value) in expect {
        assert_eq!(d.get(*name), Some(value), "{name} in ecg.h");
    }
    // Nothing numbered in the header that the engine does not define.
    let known: std::collections::HashSet<&str> = expect.iter().map(|(n, _)| *n).collect();
    for name in d.keys() {
        assert!(
            known.contains(name.as_str()),
            "{name} is in ecg.h but not checked here"
        );
    }
}

#[test]
fn the_structs_have_the_layout_the_header_declares() {
    // ecg_config: u32, u32, f64, and since 1.1 a pointer; 1.0 ends at 16.
    assert_eq!(std::mem::offset_of!(EcgConfig, stages), 16);
    assert_eq!(std::mem::size_of::<EcgConfig>(), 24);
    // ecg_event: four u32, two u64, four f32.
    assert_eq!(std::mem::size_of::<EcgEvent>(), 48);
    assert_eq!(std::mem::align_of::<EcgEvent>(), 8);
    // ecg_status: four u32, u64, f32, padded to 8.
    assert_eq!(std::mem::size_of::<EcgStatus>(), 32);
}
