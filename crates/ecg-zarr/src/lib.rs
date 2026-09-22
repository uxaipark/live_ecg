//! Minimal zarr v3 reader for the internal patch corpus.
//!
//! Scope is as narrow as `ecg-wfdb`'s and for the same reason: read what the
//! evaluation harness needs without a Python round-trip. What it needs is one
//! shape of array — a single-lead recording stored as `int32` in a zarr v3
//! `ZipStore`, compressed with blosc over zstd.
//!
//! Three properties of that store make a small reader possible:
//!
//! * **The zip entries are stored, not deflated.** Every chunk sits contiguously
//!   in the file, so the container costs a directory walk and nothing else, and
//!   chunks can be read straight out of a memory map.
//! * **The containers are small.** The largest is 170 MB with 4,039 entries,
//!   far under the 4 GB and 65,535 that would need ZIP64. ZIP64 is detected and
//!   refused rather than misread, because a reader that quietly returns the
//!   wrong bytes is worse than one that stops.
//! * **Chunks are independent.** A recording is 300-second chunks, so a
//!   fourteen-day record streams through the engine without ever being
//!   materialised.
//!
//! # What this reader does not do
//!
//! Sharding, non-default chunk key encodings, codecs other than blosc/zstd,
//! dtypes other than `int32`, and writing. Each is refused explicitly.

mod array;
mod blosc;
mod zip;

pub use array::{Attrs, ZarrArray};

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
