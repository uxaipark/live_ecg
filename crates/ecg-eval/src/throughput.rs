//! Throughput benchmark: how many patch channels one host can carry.
//!
//! The server case is many independent channels, so the figure that matters is
//! aggregate realtime factor under a realistic packet cadence, not the speed of
//! one long batch. Each simulated channel keeps its own pipeline state and is
//! fed in 250 ms packets, interleaved, which is what the deployed loop does.

use crate::qrs_eval;
use crate::Opts;
use ecg_pipeline::{ChannelOutput, ChannelPipeline};
use ecg_wfdb::{read_signal, Header};
use rayon::prelude::*;

pub fn run(opts: &Opts) -> std::io::Result<()> {
    opts.install_thread_pool();
    let entries = opts.select()?;
    let entry = entries
        .first()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no records selected"))?;

    let fs_target = opts.get_f64("fs").unwrap_or(250.0);
    let channels = opts.get_usize("channels").unwrap_or(256);
    let minutes = opts.get_f64("minutes").unwrap_or(1.0);

    let hdr = Header::read(&entry.hea_path())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let src = read_signal(&hdr, opts.lead.min(hdr.n_sig - 1), 0, hdr.n_samples)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

    // Resample the source record to the target patch rate by linear interpolation.
    // This only feeds the benchmark; detection quality is measured elsewhere.
    let n_out = (minutes * 60.0 * fs_target) as usize;
    let ratio = hdr.fs / fs_target;
    let mut signal = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let p = (i as f64 * ratio) % (src.len() - 1) as f64;
        let k = p as usize;
        let f = (p - k as f64) as f32;
        signal.push(src[k] * (1.0 - f) + src[k + 1] * f);
    }

    let cfg = qrs_eval::config_from(opts, fs_target);
    let packet = ((fs_target * 0.25) as usize).max(1);
    let threads = rayon::current_num_threads();

    eprintln!(
        "bench: {channels} channels x {minutes:.1} min @ {fs_target} Hz, {packet}-sample packets, {threads} threads"
    );

    let mut states: Vec<(ChannelPipeline, ChannelOutput)> = (0..channels)
        .map(|_| (ChannelPipeline::new(cfg), ChannelOutput::default()))
        .collect();

    // Warm the filters so the measurement excludes the learning phase.
    for (p, o) in states.iter_mut() {
        p.push(&signal[..packet.min(signal.len())], o);
        o.clear();
    }

    let t0 = std::time::Instant::now();
    let beats: u64 = states
        .par_iter_mut()
        .map(|(pipe, out)| {
            let mut n = 0u64;
            for chunk in signal.chunks(packet) {
                out.clear();
                pipe.push(chunk, out);
                n += out.beats.len() as u64;
            }
            n
        })
        .sum();
    let wall = t0.elapsed().as_secs_f64();

    let signal_seconds = channels as f64 * signal.len() as f64 / fs_target;
    let packets = channels as f64 * (signal.len() / packet) as f64;
    let cpu_us_per_packet = wall * threads as f64 * 1e6 / packets;
    let ns_per_sample = wall * threads as f64 * 1e9 / (channels as f64 * signal.len() as f64);

    println!("\n── throughput ────────────────────────────────────────────────");
    println!("threads              {threads}");
    println!("channels             {channels}");
    println!("sample rate          {fs_target} Hz");
    println!(
        "signal processed     {:.1} channel-hours",
        signal_seconds / 3600.0
    );
    println!("wall                 {wall:.3} s");
    println!("aggregate realtime   {:.0}x", signal_seconds / wall);
    println!("cost per sample      {ns_per_sample:.1} ns of CPU");
    println!("cost per 250 ms pkt  {cpu_us_per_packet:.1} us of CPU");
    println!(
        "capacity, 1 core     {:.0} channels @ {fs_target} Hz",
        signal_seconds / wall / threads as f64
    );
    println!(
        "capacity, {threads} cores   {:.0} channels @ {fs_target} Hz",
        signal_seconds / wall
    );
    println!("beats emitted        {beats}");
    Ok(())
}

/// Stage-by-stage cost breakdown on a single core.
///
/// Each row adds one stage to the previous row, so the difference between
/// consecutive rows is that stage's marginal cost. Run single-threaded on one
/// channel: this measures the algorithm, not the scheduler.
pub fn stages(opts: &Opts) -> std::io::Result<()> {
    let entries = opts.select()?;
    let entry = entries
        .first()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no records selected"))?;
    let hdr = Header::read(&entry.hea_path())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let sig = read_signal(&hdr, opts.lead.min(hdr.n_sig - 1), 0, hdr.n_samples)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let fs = hdr.fs;
    let cfg = qrs_eval::config_from(opts, fs);
    let reps = opts.get_usize("reps").unwrap_or(3);
    let n = sig.len() as f64;

    let time = |label: &str, mut f: Box<dyn FnMut()>| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let ns = best * 1e9 / n;
        println!("{label:<34} {ns:>8.1} ns/sample");
        ns
    };

    println!("\n── stage cost, single core, {} Hz ──────────────────", fs);
    use std::hint::black_box;

    let t_pre = {
        let sig = &sig;
        let mut pre = ecg_pipeline::Preprocessor::new(cfg.preprocess);
        time(
            "filter bank (4 taps)",
            Box::new(move || {
                for &x in sig.iter() {
                    black_box(pre.process(x));
                }
            }),
        )
    };

    let t_q = {
        let sig = &sig;
        let mut pre = ecg_pipeline::Preprocessor::new(cfg.preprocess);
        let mut qual = ecg_quality::QualityMonitor::new(cfg.quality);
        time(
            "  + quality monitor",
            Box::new(move || {
                for &x in sig.iter() {
                    let b = pre.process(x);
                    black_box(qual.process(b.raw, b.clean, b.baseline, b.hf, b.qrs, b.saturated));
                }
            }),
        )
    };

    let t_all = {
        let sig = &sig;
        let mut pipe = ChannelPipeline::new(cfg);
        let mut out = ChannelOutput::default();
        time(
            "  + QRS detection (full pipeline)",
            Box::new(move || {
                for c in sig.chunks(64) {
                    out.clear();
                    pipe.push(c, &mut out);
                    black_box(out.beats.len());
                }
            }),
        )
    };

    println!("\nmarginal:");
    println!("{:<34} {:>8.1} ns/sample", "filter bank", t_pre);
    println!("{:<34} {:>8.1} ns/sample", "quality monitor", t_q - t_pre);
    println!("{:<34} {:>8.1} ns/sample", "QRS detection", t_all - t_q);
    Ok(())
}
