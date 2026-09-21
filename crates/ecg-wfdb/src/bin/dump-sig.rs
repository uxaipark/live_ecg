//! Dump a lead as text, for differencing against the reference WFDB reader.

use ecg_wfdb::{read_signal, Header};
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let hea = args
        .next()
        .expect("usage: dump-sig <header> [lead] [start] [count]");
    let lead: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let start: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let count: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let hdr = Header::read(Path::new(&hea)).expect("header");
    let sig = read_signal(&hdr, lead, start, count).expect("signal");
    for v in sig {
        println!("{v:.6}");
    }
}
