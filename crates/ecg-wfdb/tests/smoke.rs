use ecg_wfdb::{read_signal, AnnotationFile, Header};

/// The corpora are not in the repository. Resolution matches the evaluation
/// harness: `$DEEP_ECG_RAW`, else a sibling checkout next to this one.
fn mitdb() -> Option<std::path::PathBuf> {
    let root = std::env::var("DEEP_ECG_RAW")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../deep_ecg/data/raw")
        });
    let d = root.join("mitdb");
    d.join("100.hea").exists().then_some(d)
}

#[test]
fn reads_mitdb_100() {
    let Some(mitdb) = mitdb() else {
        eprintln!("SKIPPED: no WFDB corpora. Set DEEP_ECG_RAW to enable.");
        return;
    };
    let hdr = Header::read(&mitdb.join("100.hea")).unwrap();
    assert_eq!(hdr.n_sig, 2);
    assert_eq!(hdr.fs, 360.0);
    assert_eq!(hdr.n_samples, 650_000);
    assert_eq!(hdr.signals[0].description, "MLII");

    let sig = read_signal(&hdr, 0, 0, hdr.n_samples).unwrap();
    assert_eq!(sig.len(), 650_000);
    // header init value for MLII is 995 adu with baseline 1024, gain 200
    assert!(
        (sig[0] - (995.0 - 1024.0) / 200.0).abs() < 1e-6,
        "first sample {}",
        sig[0]
    );

    let ann = AnnotationFile::read(&mitdb.join("100.atr")).unwrap();
    let beats = ann.beat_samples();
    // MIT-BIH record 100 carries 2273 beats (2239 N + 33 A + 1 V).
    assert_eq!(beats.len(), 2273, "beat count");
    assert_eq!(beats[0], 77);
}
