use crate::Result;
use std::path::Path;

const SKIP: u16 = 59;
const NUM: u16 = 60;
const SUB: u16 = 61;
const CHN: u16 = 62;
const AUX: u16 = 63;

#[derive(Debug, Clone)]
pub struct Annotation {
    pub sample: i64,
    pub code: u16,
    pub symbol: char,
    pub subtype: i8,
    pub chan: u8,
    pub num: i8,
    /// `aux` payload; rhythm labels such as `(AFIB` arrive here.
    pub aux: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AnnotationFile {
    pub annotations: Vec<Annotation>,
}

impl AnnotationFile {
    /// Read a MIT-format annotation file (`.atr`, `.qrs`, ...).
    pub fn read(path: &Path) -> Result<AnnotationFile> {
        let buf = std::fs::read(path)?;
        let mut out = Vec::with_capacity(buf.len() / 2);
        let mut i = 0usize;
        let mut time: i64 = 0;
        let mut chan: u8 = 0;
        let mut num: i8 = 0;

        let w = |b: &[u8], i: usize| -> u16 { u16::from_le_bytes([b[i], b[i + 1]]) };

        while i + 1 < buf.len() {
            let a = w(&buf, i);
            i += 2;
            if a == 0 {
                break; // end of file
            }
            let mut code = a >> 10;
            if code == SKIP {
                if i + 3 >= buf.len() {
                    break;
                }
                // 32-bit interval, high word first, and **signed**: files written
                // by recent WFDB open with a `-1` skip ahead of a time-resolution
                // note, and reading it unsigned puts every later annotation 2^32
                // samples into the future.
                let hi = w(&buf, i) as u32;
                let lo = w(&buf, i + 2) as u32;
                i += 4;
                time += (((hi << 16) | lo) as i32) as i64;
                if i + 1 >= buf.len() {
                    break;
                }
                // The word carrying the code still contributes its own time field.
                let a2 = w(&buf, i);
                i += 2;
                code = a2 >> 10;
                time += (a2 & 0x3FF) as i64;
            } else {
                time += (a & 0x3FF) as i64;
            }

            let mut ann = Annotation {
                sample: time,
                code,
                symbol: code_to_symbol(code),
                subtype: 0,
                chan,
                num,
                aux: None,
            };

            // Consume the modifier words that belong to this annotation.
            while i + 1 < buf.len() {
                let m = w(&buf, i);
                let mcode = m >> 10;
                let data = m & 0x3FF;
                match mcode {
                    SUB => {
                        ann.subtype = (data as u8) as i8;
                        i += 2;
                    }
                    CHN => {
                        chan = data as u8;
                        ann.chan = chan;
                        i += 2;
                    }
                    NUM => {
                        num = (data as u8) as i8;
                        ann.num = num;
                        i += 2;
                    }
                    AUX => {
                        i += 2;
                        let len = data as usize;
                        let end = (i + len).min(buf.len());
                        ann.aux = Some(
                            String::from_utf8_lossy(&buf[i..end])
                                .trim_end_matches('\0')
                                .to_string(),
                        );
                        i = end;
                        if len % 2 == 1 {
                            i += 1; // aux payloads are padded to an even byte count
                        }
                    }
                    _ => break,
                }
            }

            out.push(ann);
        }

        Ok(AnnotationFile { annotations: out })
    }

    /// Samples of beat annotations only (the QRS reference used for detector
    /// scoring). Annotations before the first sample are excluded: WFDB permits
    /// them, and they refer to a beat the record does not contain.
    pub fn beat_samples(&self) -> Vec<i64> {
        self.annotations
            .iter()
            .filter(|a| is_beat_symbol(a.symbol) && a.sample >= 0)
            .map(|a| a.sample)
            .collect()
    }
}

/// MIT annotation code table (see WFDB `ecgcodes.h`).
pub fn code_to_symbol(code: u16) -> char {
    const TABLE: [char; 50] = [
        ' ', 'N', 'L', 'R', 'a', 'V', 'F', 'J', 'A', 'S', // 0-9
        'E', 'j', '/', 'Q', '~', '?', '|', '?', 's', 'T', // 10-19
        '*', 'D', '"', '=', 'p', 'B', '^', 't', '+', 'u', // 20-29
        '?', '!', '[', ']', 'e', 'n', '@', 'x', 'f', '(', // 30-39
        ')', 'r', '?', '?', '?', '?', '?', '?', '?', '?', // 40-49
    ];
    *TABLE.get(code as usize).unwrap_or(&'?')
}

/// The WFDB `isqrs` set: annotation symbols that mark a heartbeat.
pub fn is_beat_symbol(s: char) -> bool {
    matches!(
        s,
        'N' | 'L'
            | 'R'
            | 'B'
            | 'A'
            | 'a'
            | 'J'
            | 'S'
            | 'V'
            | 'r'
            | 'F'
            | 'e'
            | 'j'
            | 'n'
            | 'E'
            | '/'
            | 'f'
            | 'Q'
            | '?'
    )
}
