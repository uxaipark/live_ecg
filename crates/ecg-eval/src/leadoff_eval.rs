//! Electrode-failure census.
//!
//! There is no lead-off label in any corpus here, so this cannot be scored the
//! way a beat detector is. What it can do is bound the two things that matter:
//! how much of each corpus the detector claims, and whether what it claims
//! lines up with signal a human already called unusable.
//!
//! The first is a false-positive bound. These are clinical recordings with
//! attached electrodes; a detector that reports minutes of electrode failure on
//! them is wrong, whatever it does on a patch. The second is the only positive
//! evidence available: BUT QDB's class 3 is signal its annotators judged
//! unusable, which includes electrode failure among other things, so overlap
//! with it is a weak confirmation and non-overlap is a strong complaint.

use crate::manifest::RecordEntry;
use crate::Opts;
use ecg_pipeline::{ChannelOutput, ChannelPipeline};
use ecg_quality::LeadOffKind;
use ecg_wfdb::{read_signal, Header};
use rayon::prelude::*;

#[derive(Debug, Default, Clone)]
pub struct Census {
    pub record: String,
    pub hours: f64,
    pub episodes: usize,
    /// Seconds claimed, by kind.
    pub rail_s: f64,
    pub open_s: f64,
    pub longest_s: f64,
}

pub fn analyse(entry: &RecordEntry, opts: &Opts) -> std::io::Result<Census> {
    let err = |e: String| std::io::Error::new(std::io::ErrorKind::InvalidData, e);
    let hdr = Header::read(&entry.hea_path()).map_err(|e| err(e.to_string()))?;
    let lead = opts.lead.min(hdr.n_sig - 1);
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples).map_err(|e| err(e.to_string()))?;
    let cfg = crate::qrs_eval::config_from(opts, hdr.fs);
    let mut pipe = ChannelPipeline::new(cfg);
    let mut out = ChannelOutput::default();
    let mut c = Census {
        record: format!("{}/{}", entry.source, entry.record),
        hours: hdr.n_samples as f64 / hdr.fs / 3600.0,
        ..Default::default()
    };
    let mut take = |out: &ChannelOutput, c: &mut Census| {
        for e in &out.lead_off {
            let s = (e.end.saturating_sub(e.start)) as f64 / hdr.fs;
            c.episodes += 1;
            c.longest_s = c.longest_s.max(s);
            match e.kind {
                LeadOffKind::RailContact => c.rail_s += s,
                LeadOffKind::OpenInput => c.open_s += s,
            }
        }
    };
    for chunk in sig.chunks((hdr.fs * 0.25) as usize) {
        out.clear();
        pipe.push(chunk, &mut out);
        take(&out, &mut c);
    }
    out.clear();
    pipe.finish(&mut out);
    take(&out, &mut c);
    Ok(c)
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    if entries.is_empty() {
        eprintln!("no records selected");
        return Ok(());
    }
    eprintln!("electrode-failure census: {} records", entries.len());
    let all: Vec<Census> = entries
        .par_iter()
        .filter_map(|e| analyse(e, opts).ok())
        .collect();
    if all.is_empty() {
        return Ok(());
    }

    let hours: f64 = all.iter().map(|c| c.hours).sum();
    let episodes: usize = all.iter().map(|c| c.episodes).sum();
    let rail: f64 = all.iter().map(|c| c.rail_s).sum();
    let open: f64 = all.iter().map(|c| c.open_s).sum();
    let touched = all.iter().filter(|c| c.episodes > 0).count();

    println!("\n── electrode failure ─────────────────────────────────────────");
    println!("records {}   {:.1} h", all.len(), hours);
    println!("records with any report      {touched} of {}", all.len());
    println!(
        "episodes                     {episodes}  ({:.2} per 24 h)",
        episodes as f64 / hours * 24.0
    );
    println!(
        "signal claimed               {:.1} s of {:.0} s  ({:.4} %)",
        rail + open,
        hours * 3600.0,
        100.0 * (rail + open) / (hours * 3600.0).max(1.0)
    );
    println!("  rail contact               {rail:.1} s");
    println!("  open input                 {open:.1} s");
    println!(
        "longest single episode       {:.1} s",
        all.iter().map(|c| c.longest_s).fold(0.0, f64::max)
    );

    if opts.per_record {
        let mut v: Vec<&Census> = all.iter().filter(|c| c.episodes > 0).collect();
        v.sort_by(|a, b| {
            (b.rail_s + b.open_s)
                .partial_cmp(&(a.rail_s + a.open_s))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        println!(
            "\n{:<18} {:>7} {:>9} {:>9} {:>9} {:>9}",
            "record", "hours", "episodes", "rail s", "open s", "% claimed"
        );
        for c in v.iter().take(20) {
            println!(
                "{:<18} {:>7.2} {:>9} {:>9.1} {:>9.1} {:>9.3}",
                c.record,
                c.hours,
                c.episodes,
                c.rail_s,
                c.open_s,
                100.0 * (c.rail_s + c.open_s) / (c.hours * 3600.0).max(1.0)
            );
        }
    }
    Ok(())
}
