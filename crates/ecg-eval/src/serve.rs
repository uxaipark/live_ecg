//! Multi-channel runtime benchmark.
//!
//! Measures what the server case actually costs: many independent channels fed
//! in packets, each shard on its own thread, with a configurable share of the
//! packets lost so the gap path is exercised rather than assumed free.
//!
//! The figure that matters is not how fast one long recording can be pushed
//! through. It is how many channels one host can carry in real time while
//! nothing shares state.

use crate::Opts;
use ecg_pipeline::{ChannelOutput, PipelineConfig};
use ecg_server::{Packet, Server};
use ecg_wfdb::{read_signal, Header};

pub fn run(opts: &Opts) -> std::io::Result<()> {
    let entries = opts.select()?;
    let entry = entries
        .first()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no records selected"))?;
    let hdr = Header::read(&entry.hea_path())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let src = read_signal(&hdr, opts.lead.min(hdr.n_sig - 1), 0, hdr.n_samples)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

    let fs = opts.get_f64("fs").unwrap_or(250.0);
    let channels = opts.get_usize("channels").unwrap_or(512) as u64;
    let minutes = opts.get_f64("minutes").unwrap_or(2.0);
    let threads = opts
        .get_usize("threads")
        .unwrap_or_else(rayon::current_num_threads);
    let loss = opts.get_f64("loss").unwrap_or(0.0);
    let packet = ((fs * 0.25) as usize).max(1);

    // Resample the source record to the patch rate. Only the cost is being
    // measured here; detection quality is measured elsewhere.
    let n_out = (minutes * 60.0 * fs) as usize;
    let ratio = hdr.fs / fs;
    let mut signal = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let p = (i as f64 * ratio) % (src.len() - 1) as f64;
        let k = p as usize;
        let f = (p - k as f64) as f32;
        signal.push(src[k] * (1.0 - f) + src[k + 1] * f);
    }

    let cfg = PipelineConfig::new(fs);
    let mut server = Server::new(threads);
    for ch in 0..channels {
        server.open(ch, cfg, packet);
    }
    let packets_per_channel = signal.len() / packet;

    eprintln!(
        "serve: {channels} channels x {minutes:.1} min @ {fs} Hz, {threads} shards, \
         {packet}-sample packets, {:.1}% packet loss",
        100.0 * loss
    );

    let signal = &signal;
    let t0 = std::time::Instant::now();
    let totals: Vec<(u64, u64)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (shard_index, shard) in server.split().enumerate() {
            handles.push(scope.spawn(move || {
                let mut out = ChannelOutput::default();
                let mut beats = 0u64;
                let mut gaps = 0u64;
                // Deterministic pseudo-loss, so a run is repeatable.
                let mut rng = 0x9E3779B97F4A7C15u64 ^ (shard_index as u64 + 1);
                let owned: Vec<u64> = (0..channels)
                    .filter(|ch| (ch % threads as u64) as usize == shard_index)
                    .collect();
                for seq in 0..packets_per_channel {
                    let chunk = &signal[seq * packet..(seq + 1) * packet];
                    for &ch in &owned {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        if loss > 0.0 && ((rng >> 11) as f64 / (1u64 << 53) as f64) < loss {
                            gaps += 1;
                            continue;
                        }
                        out.clear();
                        shard.ingest(
                            Packet {
                                channel: ch,
                                sequence: seq as u64,
                                samples: chunk,
                            },
                            &mut out,
                        );
                        beats += out.beats.len() as u64;
                    }
                }
                (beats, gaps)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let wall = t0.elapsed().as_secs_f64();

    let beats: u64 = totals.iter().map(|t| t.0).sum();
    let gaps: u64 = totals.iter().map(|t| t.1).sum();
    let signal_seconds = channels as f64 * signal.len() as f64 / fs;
    let ns_per_sample = wall * threads as f64 * 1e9 / (channels as f64 * signal.len() as f64);

    println!("\n── multi-channel runtime ─────────────────────────────────────");
    println!("shards / threads     {threads}");
    println!("channels             {channels}");
    println!("packets lost         {gaps}");
    println!(
        "signal processed     {:.1} channel-hours",
        signal_seconds / 3600.0
    );
    println!("wall                 {wall:.3} s");
    println!("aggregate realtime   {:.0}x", signal_seconds / wall);
    println!("cost per sample      {ns_per_sample:.1} ns of CPU");
    println!(
        "capacity, 1 shard    {:.0} channels @ {fs} Hz",
        signal_seconds / wall / threads as f64
    );
    println!(
        "capacity, {threads} shards   {:.0} channels @ {fs} Hz",
        signal_seconds / wall
    );
    println!("beats emitted        {beats}");
    Ok(())
}
