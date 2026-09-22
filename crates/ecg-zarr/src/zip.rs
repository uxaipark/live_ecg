//! The container: a directory of stored entries over a memory map.
//!
//! Only what a zarr `ZipStore` produces is handled — entries written with no
//! compression, no encryption, no ZIP64. Everything else is refused by name.

use crate::{Error, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

const EOCD_SIG: u32 = 0x0605_4b50;
const CD_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;
const ZIP64_EOCD_LOCATOR_SIG: u32 = 0x0706_4b50;
/// Both the entry count and the size fields use this as "look in the ZIP64
/// record instead".
const ZIP64_SENTINEL_32: u32 = 0xffff_ffff;
const ZIP64_SENTINEL_16: u16 = 0xffff;

pub struct Archive {
    map: Mmap,
    /// Name to (offset of the entry's data, length).
    entries: HashMap<String, (usize, usize)>,
}

fn u16_at(b: &[u8], i: usize) -> Result<u16> {
    b.get(i..i + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| Error::Parse(format!("truncated at {i}")))
}

fn u32_at(b: &[u8], i: usize) -> Result<u32> {
    b.get(i..i + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| Error::Parse(format!("truncated at {i}")))
}

impl Archive {
    pub fn open(path: &Path) -> Result<Archive> {
        let file = File::open(path)?;
        // SAFETY: the corpus is read-only during evaluation. A concurrent
        // truncation would be undefined, and the same is already true of every
        // WFDB record this harness maps.
        let map = unsafe { Mmap::map(&file)? };
        let eocd = find_eocd(&map)?;

        // The comment length is the only variable-length field in the record,
        // so a ZIP64 locator - if there is one - sits immediately before it.
        if eocd >= 20 && u32_at(&map, eocd - 20)? == ZIP64_EOCD_LOCATOR_SIG {
            return Err(Error::Unsupported(
                "ZIP64 container; this reader would misread the offsets".into(),
            ));
        }
        let n_entries = u16_at(&map, eocd + 10)?;
        let cd_offset = u32_at(&map, eocd + 16)? as usize;
        if n_entries == ZIP64_SENTINEL_16 || cd_offset as u32 == ZIP64_SENTINEL_32 {
            return Err(Error::Unsupported("ZIP64 fields in a plain record".into()));
        }

        let mut entries = HashMap::with_capacity(n_entries as usize);
        let mut p = cd_offset;
        for _ in 0..n_entries {
            if u32_at(&map, p)? != CD_SIG {
                return Err(Error::Parse(format!("no central header at {p}")));
            }
            let method = u16_at(&map, p + 10)?;
            let size = u32_at(&map, p + 24)? as usize;
            let name_len = u16_at(&map, p + 28)? as usize;
            let extra_len = u16_at(&map, p + 30)? as usize;
            let comment_len = u16_at(&map, p + 32)? as usize;
            let local = u32_at(&map, p + 42)? as usize;
            let name = std::str::from_utf8(
                map.get(p + 46..p + 46 + name_len)
                    .ok_or_else(|| Error::Parse("truncated name".into()))?,
            )
            .map_err(|e| Error::Parse(e.to_string()))?
            .to_string();

            if method != 0 {
                return Err(Error::Unsupported(format!(
                    "{name} is compressed with method {method}; \
                     a zarr ZipStore stores entries"
                )));
            }
            // The local header's extra field is allowed to differ in length
            // from the central one, so the data offset has to come from the
            // local header. Trusting the central copy is the classic way to
            // read a zip a few bytes off.
            if u32_at(&map, local)? != LOCAL_SIG {
                return Err(Error::Parse(format!("no local header for {name}")));
            }
            let l_name = u16_at(&map, local + 26)? as usize;
            let l_extra = u16_at(&map, local + 28)? as usize;
            let start = local + 30 + l_name + l_extra;
            if start + size > map.len() {
                return Err(Error::Parse(format!("{name} runs past the file")));
            }
            entries.insert(name, (start, size));
            p += 46 + name_len + extra_len + comment_len;
        }
        Ok(Archive { map, entries })
    }

    pub fn get(&self, name: &str) -> Option<&[u8]> {
        let &(start, len) = self.entries.get(name)?;
        Some(&self.map[start..start + len])
    }
}

/// The end-of-central-directory record, searched from the back because a
/// trailing comment may follow it.
fn find_eocd(map: &[u8]) -> Result<usize> {
    let min = 22usize;
    if map.len() < min {
        return Err(Error::Parse("file is shorter than a zip record".into()));
    }
    let scan = map.len().saturating_sub(min + u16::MAX as usize);
    for i in (scan..=map.len() - min).rev() {
        if u32_at(map, i)? == EOCD_SIG {
            // The comment length has to agree with what is left of the file,
            // otherwise this is a signature that happened to appear in data.
            let comment = u16_at(map, i + 20)? as usize;
            if i + min + comment == map.len() {
                return Ok(i);
            }
        }
    }
    Err(Error::Parse("no end-of-central-directory record".into()))
}
