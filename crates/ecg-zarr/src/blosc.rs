//! Blosc1 frame decoding.
//!
//! Blosc is a container around an ordinary codec rather than a codec itself: it
//! splits a chunk into cache-sized blocks, optionally applies a byte-transpose
//! filter, and compresses each block with zstd. Decoding it is therefore mostly
//! bookkeeping, and the bookkeeping is worth writing out because every field
//! here is a place to be silently wrong by a few bytes.
//!
//! The one thing that is not obvious from the format description: **each
//! compressed stream carries its own length in front of it**, and a stream
//! whose length equals its uncompressed size is stored rather than compressed.
//! That is what makes the split and no-split layouts the same code, and it is
//! why nothing here has to know where a zstd frame ends.

use crate::{Error, Result};

const HEADER: usize = 16;

// Flag bits, from c-blosc's `blosc.h`.
const DO_SHUFFLE: u8 = 0x01;
const MEMCPYED: u8 = 0x02;
const DO_BITSHUFFLE: u8 = 0x04;
const DO_DELTA: u8 = 0x08;
/// Set when the encoder chose *not* to divide each block into one stream per
/// byte of the type. zstd is compressed whole by default, so this is the case
/// the corpus actually takes.
const NO_SPLIT: u8 = 0x10;
const COMPRESSOR_SHIFT: u8 = 5;
const COMPRESSOR_ZSTD: u8 = 4;

fn u32_at(b: &[u8], i: usize) -> Result<usize> {
    b.get(i..i + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
        .ok_or_else(|| Error::Parse(format!("blosc frame truncated at {i}")))
}

/// Decode one blosc frame into `out`, which is resized to the frame's size.
pub fn decode(src: &[u8], out: &mut Vec<u8>) -> Result<()> {
    if src.len() < HEADER {
        return Err(Error::Parse("blosc frame shorter than its header".into()));
    }
    let flags = src[2];
    let typesize = src[3] as usize;
    let nbytes = u32_at(src, 4)?;
    let blocksize = u32_at(src, 8)?;
    let cbytes = u32_at(src, 12)?;

    if cbytes > src.len() {
        return Err(Error::Parse(format!(
            "blosc frame claims {cbytes} bytes of {}",
            src.len()
        )));
    }
    if flags & DO_BITSHUFFLE != 0 || flags & DO_DELTA != 0 {
        return Err(Error::Unsupported("blosc bitshuffle or delta filter".into()));
    }
    let compressor = flags >> COMPRESSOR_SHIFT;
    if flags & MEMCPYED == 0 && compressor != COMPRESSOR_ZSTD {
        return Err(Error::Unsupported(format!(
            "blosc compressor code {compressor}; only zstd is handled"
        )));
    }

    out.clear();
    out.resize(nbytes, 0);

    if flags & MEMCPYED != 0 {
        let body = src
            .get(HEADER..HEADER + nbytes)
            .ok_or_else(|| Error::Parse("stored blosc frame is short".into()))?;
        out.copy_from_slice(body);
        return Ok(());
    }
    if blocksize == 0 {
        return Err(Error::Parse("blosc block size of zero".into()));
    }

    let n_blocks = nbytes.div_ceil(blocksize);
    let streams = if flags & NO_SPLIT != 0 { 1 } else { typesize };
    if streams == 0 {
        return Err(Error::Parse("blosc type size of zero".into()));
    }
    let shuffled = flags & DO_SHUFFLE != 0;
    let mut tmp = Vec::new();

    for b in 0..n_blocks {
        let start = u32_at(src, HEADER + 4 * b)?;
        let bsize = blocksize.min(nbytes - b * blocksize);
        let block = &mut out[b * blocksize..b * blocksize + bsize];
        // The filter runs over the whole block, so a shuffled block has to be
        // assembled somewhere else first: unshuffling in place would read bytes
        // it has already overwritten.
        if shuffled {
            tmp.clear();
            tmp.resize(bsize, 0);
            decode_block(src, start, bsize, streams, &mut tmp)?;
            unshuffle(typesize, &tmp, block);
        } else {
            decode_block(src, start, bsize, streams, block)?;
        }
    }
    Ok(())
}

/// One block: `streams` length-prefixed pieces, laid out end to end.
fn decode_block(src: &[u8], start: usize, bsize: usize, streams: usize, dest: &mut [u8]) -> Result<()> {
    let each = bsize / streams;
    let mut p = start;
    for s in 0..streams {
        let n = u32_at(src, p)?;
        p += 4;
        let piece = src
            .get(p..p + n)
            .ok_or_else(|| Error::Parse("blosc stream runs past the frame".into()))?;
        // The last stream takes the remainder, which is not `each` when the
        // block does not divide evenly.
        let span = if s + 1 == streams {
            bsize - s * each
        } else {
            each
        };
        let target = &mut dest[s * each..s * each + span];
        if n == span {
            // Incompressible: blosc stores it rather than growing it.
            target.copy_from_slice(piece);
        } else {
            zstd_into(piece, target)?;
        }
        p += n;
    }
    Ok(())
}

fn zstd_into(src: &[u8], dest: &mut [u8]) -> Result<()> {
    use std::io::Read;
    let mut d = ruzstd::decoding::StreamingDecoder::new(src)
        .map_err(|e| Error::Parse(format!("zstd frame: {e}")))?;
    let mut n = 0;
    while n < dest.len() {
        match d.read(&mut dest[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) => return Err(Error::Parse(format!("zstd: {e}"))),
        }
    }
    if n != dest.len() {
        return Err(Error::Parse(format!(
            "zstd produced {n} bytes, expected {}",
            dest.len()
        )));
    }
    Ok(())
}

/// Inverse of blosc's byte transpose: byte `j` of every element is stored
/// together, so element `i`'s byte `j` sits at `j * elements + i`.
///
/// The tail that does not fill a whole element is not shuffled by the encoder
/// either, so it is copied straight across.
fn unshuffle(typesize: usize, src: &[u8], dest: &mut [u8]) {
    let n = src.len();
    let elements = n / typesize;
    if typesize <= 1 || elements == 0 {
        dest.copy_from_slice(src);
        return;
    }
    for j in 0..typesize {
        let plane = &src[j * elements..(j + 1) * elements];
        for (i, &byte) in plane.iter().enumerate() {
            dest[i * typesize + j] = byte;
        }
    }
    let tail = elements * typesize;
    dest[tail..].copy_from_slice(&src[tail..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The filter is its own inverse only for a two-byte type, so this checks
    /// the general case against an explicit layout.
    #[test]
    fn unshuffle_puts_the_planes_back() {
        // Two elements of four bytes: 00 01 02 03 / 10 11 12 13.
        let shuffled = [0x00, 0x10, 0x01, 0x11, 0x02, 0x12, 0x03, 0x13];
        let mut out = [0u8; 8];
        unshuffle(4, &shuffled, &mut out);
        assert_eq!(out, [0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13]);
    }

    #[test]
    fn a_tail_that_does_not_fill_an_element_is_copied() {
        let shuffled = [0x00, 0x10, 0x01, 0x11, 0xaa];
        let mut out = [0u8; 5];
        unshuffle(2, &shuffled, &mut out);
        assert_eq!(out, [0x00, 0x01, 0x10, 0x11, 0xaa]);
    }
}
