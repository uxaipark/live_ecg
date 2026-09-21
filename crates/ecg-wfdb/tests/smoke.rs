use ecg_wfdb::{read_signal, AnnotationFile, Header};
use std::path::Path;

const MITDB: &str = "/Users/elliotpark/dev/deep_ecg/data/raw/mitdb";

#[test]
fn reads_mitdb_100() {
    let hdr = Header::read(&Path::new(MITDB).join("100.hea")).unwrap();
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

    let ann = AnnotationFile::read(&Path::new(MITDB).join("100.atr")).unwrap();
    let beats = ann.beat_samples();
    // MIT-BIH record 100 carries 2273 beats (2239 N + 33 A + 1 V).
    assert_eq!(beats.len(), 2273, "beat count");
    assert_eq!(beats[0], 77);
}
