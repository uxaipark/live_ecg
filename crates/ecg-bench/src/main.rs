//! Standalone capacity benchmark.
//!
//! Self-contained on purpose: no corpus, no data files, no threading crate. It
//! is meant to be cross-compiled, copied onto a Raspberry Pi, a phone or
//! whatever the deployment actually runs on, and executed there — because the
//! numbers in this project's reports are all from a development workstation and
//! do not transfer. Per-shard capacity already turns out to depend on memory
//! bandwidth rather than core count, so extrapolating from one machine to
//! another is guesswork.
//!
//! The signal is synthesised rather than read. That measures cost, which is what
//! this is for; detection quality is measured against real corpora elsewhere and
//! is a property of the algorithm, not of the host.

use ecg_pipeline::PipelineConfig;
use ecg_server::{Packet, Server};
use std::time::Instant;

/// A synthetic beat train: P wave, QRS, T wave, on a wandering baseline with
/// noise. Crude, but it exercises every path — detection, classification, the
/// rhythm bank — at a realistic beat density, which is what determines cost.
fn synthesise(seconds: f64, fs: f64, bpm: f64) -> Vec<f32> {
    let n = (seconds * fs) as usize;
    let period = (60.0 / bpm * fs) as usize;
    let mut v = Vec::with_capacity(n);
    let mut rng: u64 = 0x2545F4914F6CDD1D;
    let ms = |t: f64| (t * fs / 1000.0) as usize;
    for i in 0..n {
        let phase = i % period;
        let t = i as f64 / fs;

        let wave = |centre: usize, width: usize, amp: f32| -> f32 {
            if width == 0 {
                return 0.0;
            }
            let d = phase as isize - centre as isize;
            if d.unsigned_abs() > width {
                0.0
            } else {
                let x = d as f32 / width as f32;
                amp * (1.0 - x * x) * (1.0 - x * x)
            }
        };

        // P wave, then the complex, then repolarisation.
        let p = wave(ms(0.0).max(1), ms(40.0), 0.12);
        let q = wave(ms(160.0), ms(10.0), -0.10);
        let r = wave(ms(180.0), ms(14.0), 1.10);
        let s = wave(ms(200.0), ms(12.0), -0.25);
        let tw = wave(ms(330.0), ms(70.0), 0.25);

        // Respiration, and a little broadband noise.
        let baseline = 0.05 * (2.0 * std::f32::consts::PI * 0.25 * t as f32).sin();
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let noise = ((rng >> 40) as f32 / 8_388_608.0 - 1.0) * 0.01;

        v.push(p + q + r + s + tw + baseline + noise);
    }
    v
}

struct Args {
    channels: u64,
    seconds: f64,
    fs: f64,
    threads: usize,
    loss: f64,
    bpm: f64,
}

fn parse() -> Args {
    let mut a = Args {
        channels: 256,
        seconds: 120.0,
        fs: 250.0,
        threads: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        loss: 0.0,
        bpm: 72.0,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i + 1 < argv.len() {
        let v = &argv[i + 1];
        match argv[i].trim_start_matches("--") {
            "channels" => a.channels = v.parse().unwrap_or(a.channels),
            "seconds" => a.seconds = v.parse().unwrap_or(a.seconds),
            "fs" => a.fs = v.parse().unwrap_or(a.fs),
            "threads" => a.threads = v.parse().unwrap_or(a.threads),
            "loss" => a.loss = v.parse().unwrap_or(a.loss),
            "bpm" => a.bpm = v.parse().unwrap_or(a.bpm),
            "help" | "h" => {}
            _ => {}
        }
        i += 2;
    }
    a
}

fn main() {
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        println!(
            "usage: ecg-bench [--channels N] [--seconds S] [--fs HZ] [--threads N] \
             [--loss FRACTION] [--bpm N]"
        );
        return;
    }
    let args = parse();
    let packet = ((args.fs * 0.25) as usize).max(1);
    let signal = synthesise(args.seconds, args.fs, args.bpm);
    let packets = signal.len() / packet;

    let cfg = PipelineConfig::new(args.fs);
    let mut server = Server::new(args.threads);
    for ch in 0..args.channels {
        server.open(ch, cfg, packet);
    }

    println!(
        "ecg-bench  {} channels x {:.0} s @ {} Hz, {} shards, {}-sample packets, {:.1}% loss",
        args.channels,
        args.seconds,
        args.fs,
        args.threads,
        packet,
        100.0 * args.loss
    );
    println!("target     {}", std::env::consts::ARCH);

    let signal = &signal;
    let threads = args.threads;
    let channels = args.channels;
    let loss = args.loss;

    let start = Instant::now();
    let results: Vec<(u64, u64)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (index, shard) in server.split().enumerate() {
            handles.push(scope.spawn(move || {
                let mut out = ecg_pipeline::ChannelOutput::default();
                let (mut beats, mut dropped) = (0u64, 0u64);
                let mut rng = 0x9E3779B97F4A7C15u64 ^ (index as u64 + 1);
                let owned: Vec<u64> = (0..channels)
                    .filter(|ch| (ch % threads as u64) as usize == index)
                    .collect();
                for seq in 0..packets {
                    let chunk = &signal[seq * packet..(seq + 1) * packet];
                    for &ch in &owned {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        if loss > 0.0 && ((rng >> 11) as f64 / (1u64 << 53) as f64) < loss {
                            dropped += 1;
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
                (beats, dropped)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let wall = start.elapsed().as_secs_f64();

    let beats: u64 = results.iter().map(|r| r.0).sum();
    let dropped: u64 = results.iter().map(|r| r.1).sum();
    let signal_seconds = args.channels as f64 * signal.len() as f64 / args.fs;
    let ns_per_sample = wall * threads as f64 * 1e9 / (args.channels as f64 * signal.len() as f64);
    let per_shard = signal_seconds / wall / threads as f64;

    println!();
    println!("wall                 {wall:.3} s");
    println!("aggregate realtime   {:.0}x", signal_seconds / wall);
    println!("cost per sample      {ns_per_sample:.1} ns of CPU");
    println!(
        "capacity, 1 shard    {per_shard:.0} channels @ {} Hz",
        args.fs
    );
    println!(
        "capacity, {threads} shards   {:.0} channels @ {} Hz",
        signal_seconds / wall,
        args.fs
    );
    println!("beats                {beats}   packets dropped {dropped}");

    // Beats are the sanity check: a target where the arithmetic differs enough
    // to change detection would show up here rather than in a wrong capacity.
    let expected = (args.channels as f64 * args.seconds * args.bpm / 60.0) as u64;
    let found = beats + dropped / 4;
    let ratio = found as f64 / expected.max(1) as f64;
    println!(
        "detection sanity     {:.3} of the {expected} beats present  ({})",
        ratio,
        if (0.9..=1.05).contains(&ratio) {
            "ok"
        } else {
            "CHECK THIS TARGET"
        }
    );
}
