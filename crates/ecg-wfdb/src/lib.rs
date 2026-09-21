//! Minimal, fast WFDB reader: `.hea` headers, `.dat` signal formats (212/16/80/61/24/32),
//! and MIT annotation files (`.atr`, `.qrs`, ...).
//!
//! Scope is deliberately narrow: what the evaluation harness needs to read the
//! PhysioNet corpora in `deep_ecg/data/raw` without a Python round-trip.

mod annot;
mod header;
mod signal;

pub use annot::{code_to_symbol, is_beat_symbol, Annotation, AnnotationFile};
pub use header::{Header, SignalSpec};
pub use signal::read_signal;

use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Parse(String),
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Parse(s) => write!(f, "parse: {s}"),
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
