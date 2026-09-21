use crate::{Error, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct SignalSpec {
    pub file_name: String,
    pub format: u16,
    /// ADC units per physical unit (mV). 0 in the header means "default 200".
    pub gain: f64,
    pub baseline: i64,
    pub units: String,
    pub adc_res: u32,
    pub adc_zero: i64,
    pub init_value: i64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct Header {
    pub record: String,
    pub dir: PathBuf,
    pub n_sig: usize,
    pub fs: f64,
    pub n_samples: usize,
    pub signals: Vec<SignalSpec>,
}

impl Header {
    pub fn read(hea_path: &Path) -> Result<Header> {
        let text = std::fs::read_to_string(hea_path)?;
        let dir = hea_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let mut lines = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'));

        let rec_line = lines
            .next()
            .ok_or_else(|| Error::Parse("empty header".into()))?;
        let mut it = rec_line.split_whitespace();
        let record = it
            .next()
            .ok_or_else(|| Error::Parse("no record name".into()))?
            // a record line may carry a segment count as `name/count` (multi-segment)
            .split('/')
            .next()
            .unwrap()
            .to_string();
        if rec_line.split_whitespace().next().unwrap().contains('/') {
            return Err(Error::Unsupported("multi-segment record".into()));
        }
        let n_sig: usize = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| Error::Parse("no nsig".into()))?;
        // `fs` may appear as `fs/counter(base)`; we only need the sampling frequency.
        let fs: f64 = it
            .next()
            .map(|s| s.split('/').next().unwrap().parse().unwrap_or(250.0))
            .unwrap_or(250.0);
        let n_samples: usize = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);

        let mut signals = Vec::with_capacity(n_sig);
        for _ in 0..n_sig {
            let l = lines
                .next()
                .ok_or_else(|| Error::Parse("truncated signal lines".into()))?;
            signals.push(parse_signal_line(l)?);
        }

        Ok(Header {
            record,
            dir,
            n_sig,
            fs,
            n_samples,
            signals,
        })
    }

    pub fn lead_index(&self, name: &str) -> Option<usize> {
        self.signals.iter().position(|s| s.description == name)
    }
}

fn parse_signal_line(line: &str) -> Result<SignalSpec> {
    let mut it = line.split_whitespace();
    let file_name = it
        .next()
        .ok_or_else(|| Error::Parse("no signal file".into()))?
        .to_string();

    // format may carry `fmt xN` (samples per frame), `fmt:T` (skew), `fmt+O` (byte offset)
    let fmt_field = it.next().ok_or_else(|| Error::Parse("no format".into()))?;
    let fmt_num: String = fmt_field
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let format: u16 = fmt_num
        .parse()
        .map_err(|_| Error::Parse(format!("bad format {fmt_field}")))?;
    if fmt_field.contains('x') {
        return Err(Error::Unsupported("multi-sample-per-frame signal".into()));
    }

    // gain field: `gain(baseline)/units`
    let gain_field = it.next().unwrap_or("200");
    let (gain_part, units) = match gain_field.split_once('/') {
        Some((g, u)) => (g, u.to_string()),
        None => (gain_field, "mV".to_string()),
    };
    let (gain_str, baseline) = match gain_part.split_once('(') {
        Some((g, b)) => (g, b.trim_end_matches(')').parse::<i64>().unwrap_or(0)),
        None => (gain_part, 0),
    };
    let mut gain: f64 = gain_str.parse().unwrap_or(200.0);
    if gain == 0.0 {
        gain = 200.0; // WFDB convention: 0 means "unspecified", use the default
    }

    let adc_res: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(12);
    let adc_zero: i64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let init_value: i64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(adc_zero);
    let _checksum = it.next();
    let _blocksize = it.next();
    let description = it.collect::<Vec<_>>().join(" ");

    // When the baseline is absent the WFDB convention is baseline == adc_zero.
    let baseline = if gain_part.contains('(') {
        baseline
    } else {
        adc_zero
    };

    Ok(SignalSpec {
        file_name,
        format,
        gain,
        baseline,
        units,
        adc_res,
        adc_zero,
        init_value,
        description,
    })
}
