//! Dump annotations as TSV, so the reader can be diffed against the reference
//! WFDB implementation instead of being trusted.

use ecg_wfdb::AnnotationFile;
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: dump-ann <annotation file> [limit]");
    let limit: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let f = AnnotationFile::read(Path::new(&path)).expect("read");
    for a in f.annotations.iter().take(limit) {
        println!(
            "{}\t{}\t{}",
            a.sample,
            a.symbol,
            a.aux.as_deref().unwrap_or("")
        );
    }
}
