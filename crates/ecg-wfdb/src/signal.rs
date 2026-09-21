use crate::header::Header;
use crate::{Error, Result};
use memmap2::Mmap;
use std::fs::File;

/// Read one lead of a record as physical units (mV), `[start, start+count)` in frames.
///
/// All signals of a record share one interleaved `.dat` stream, so a lead is read
/// with a stride rather than by materialising every channel.
pub fn read_signal(hdr: &Header, lead: usize, start: usize, count: usize) -> Result<Vec<f32>> {
    let spec = hdr
        .signals
        .get(lead)
        .ok_or_else(|| Error::Parse(format!("lead {lead} out of range")))?;

    // Every lead in the records we read lives in a single .dat file. Signals split
    // across several files would need a per-file frame layout; reject them loudly.
    if hdr
        .signals
        .iter()
        .any(|s| s.file_name != hdr.signals[0].file_name)
    {
        return Err(Error::Unsupported("signals split across files".into()));
    }

    let path = hdr.dir.join(&spec.file_name);
    let file = File::open(&path)?;
    let map = unsafe { Mmap::map(&file)? };
    let bytes: &[u8] = &map;

    let n_sig = hdr.n_sig;
    let end = start.saturating_add(count);
    let mut out = Vec::with_capacity(count);

    let gain = spec.gain as f32;
    let baseline = spec.baseline as f32;
    let scale = 1.0f32 / gain;

    match spec.format {
        212 => {
            for f in start..end {
                let g = f * n_sig + lead;
                let tri = g >> 1;
                let off = tri * 3;
                if off + 2 >= bytes.len() {
                    break;
                }
                let b0 = bytes[off] as i32;
                let b1 = bytes[off + 1] as i32;
                let b2 = bytes[off + 2] as i32;
                let raw = if g & 1 == 0 {
                    b0 | ((b1 & 0x0F) << 8)
                } else {
                    b2 | ((b1 >> 4) << 8)
                };
                out.push((sext(raw, 12) as f32 - baseline) * scale);
            }
        }
        16 | 61 => {
            let be = spec.format == 61;
            for f in start..end {
                let off = (f * n_sig + lead) * 2;
                if off + 1 >= bytes.len() {
                    break;
                }
                let raw = if be {
                    i16::from_be_bytes([bytes[off], bytes[off + 1]])
                } else {
                    i16::from_le_bytes([bytes[off], bytes[off + 1]])
                };
                out.push((raw as f32 - baseline) * scale);
            }
        }
        80 => {
            for f in start..end {
                let off = f * n_sig + lead;
                if off >= bytes.len() {
                    break;
                }
                let raw = bytes[off] as i32 - 128;
                out.push((raw as f32 - baseline) * scale);
            }
        }
        24 => {
            for f in start..end {
                let off = (f * n_sig + lead) * 3;
                if off + 2 >= bytes.len() {
                    break;
                }
                let raw = (bytes[off] as i32)
                    | ((bytes[off + 1] as i32) << 8)
                    | ((bytes[off + 2] as i32) << 16);
                out.push((sext(raw, 24) as f32 - baseline) * scale);
            }
        }
        32 => {
            for f in start..end {
                let off = (f * n_sig + lead) * 4;
                if off + 3 >= bytes.len() {
                    break;
                }
                let raw = i32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]);
                out.push((raw as f32 - baseline) * scale);
            }
        }
        310 | 311 => {
            // Three 10-bit samples bit-packed into four bytes. Indexing is by
            // global sample number, so a lead is still read with a stride.
            let bits = spec.format;
            for f in start..end {
                let g = f * n_sig + lead;
                let quartet = g / 3;
                let within = g % 3;
                let off = quartet * 4;
                if off + 3 >= bytes.len() {
                    break;
                }
                let (b0, b1, b2, b3) = (
                    bytes[off] as i32,
                    bytes[off + 1] as i32,
                    bytes[off + 2] as i32,
                    bytes[off + 3] as i32,
                );
                let raw = if bits == 310 {
                    match within {
                        0 => (b0 >> 1) + 128 * (b1 & 0x07),
                        1 => (b2 >> 1) + 128 * (b3 & 0x07),
                        _ => ((b1 >> 3) & 0x1F) + 32 * ((b3 >> 3) & 0x1F),
                    }
                } else {
                    match within {
                        0 => b0 + 256 * (b1 & 0x03),
                        1 => (b1 >> 2) + 64 * (b2 & 0x0F),
                        _ => (b2 >> 4) + 16 * (b3 & 0x7F),
                    }
                };
                out.push((sext(raw, 10) as f32 - baseline) * scale);
            }
        }
        f => return Err(Error::Unsupported(format!("dat format {f}"))),
    }

    Ok(out)
}

#[inline(always)]
fn sext(v: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (v << shift) >> shift
}
