//! The runtime is defined by what it does when the stream misbehaves, so that
//! is what is tested: packets out of order, duplicated, missing, and a channel
//! that cannot keep up. The headline property is that none of those becomes a
//! clinical finding.

use ecg_pipeline::{ChannelOutput, PipelineConfig};
use ecg_rhythm::Condition;
use ecg_server::{Ingest, Packet, Server};

const FS: f64 = 250.0;
const PACKET: usize = 62; // ~250 ms

/// A synthetic sinus rhythm at `bpm`: a narrow triangular complex on a flat
/// baseline. Crude, but it gives the detector something unambiguous to find.
fn sinus(seconds: f64, bpm: f64) -> Vec<f32> {
    let n = (seconds * FS) as usize;
    let period = (60.0 / bpm * FS) as usize;
    let mut v = vec![0.0f32; n];
    for (i, x) in v.iter_mut().enumerate() {
        *x = match i % period {
            0 | 1 => 0.5,
            2 | 3 => 1.0,
            4 | 5 => -0.4,
            _ => 0.0,
        };
    }
    v
}

fn open_one(server: &mut Server, channel: u64) {
    server.open(channel, PipelineConfig::new(FS), PACKET);
}

/// Feed `signal` as packets, optionally losing the packets whose index is in
/// `lose`. Returns the collected findings.
fn feed(server: &mut Server, channel: u64, signal: &[f32], lose: &[usize]) -> ChannelOutput {
    let mut out = ChannelOutput::default();
    for (i, chunk) in signal.chunks(PACKET).enumerate() {
        if lose.contains(&i) {
            continue;
        }
        server.ingest(
            Packet {
                channel,
                sequence: i as u64,
                samples: chunk,
            },
            &mut out,
        );
    }
    out
}

#[test]
fn a_clean_stream_is_analysed() {
    let mut s = Server::new(2);
    open_one(&mut s, 7);
    let out = feed(&mut s, 7, &sinus(60.0, 60.0), &[]);
    let stats = s.stats(7).unwrap();
    assert_eq!(stats.missing_packets, 0);
    assert_eq!(stats.stale_packets, 0);
    // A minute at 60 bpm, minus the detector's learning period.
    assert!(out.beats.len() > 50, "found {} beats", out.beats.len());
}

#[test]
fn a_lost_packet_does_not_move_the_clock() {
    // The property the whole design exists for, and it is not the one you might
    // expect. Splicing across a gap does *not* present an enormous interval -
    // the skipped samples are never fed, so the pipeline's clock never advances
    // and the two sides join seamlessly. Ten seconds of signal simply vanish and
    // everything after them is reported ten seconds early, with nothing saying
    // so. A wrong timeline is worse than a missing one.
    let signal = sinus(120.0, 60.0);
    let lose: Vec<usize> = (200..240).collect(); // ten seconds

    let mut s = Server::new(1);
    open_one(&mut s, 11);
    let out = feed(&mut s, 11, &signal, &lose);

    let last = out.beats.last().expect("beats found").sample as f64 / FS;
    let expected = signal.len() as f64 / FS;
    assert!(
        (last - expected).abs() < 3.0,
        "last beat reported at {last:.1} s of a {expected:.1} s recording: \
         the clock lost the gap"
    );
}

#[test]
fn a_lost_packet_does_not_become_a_pause() {
    // And having advanced the clock, the gap must not then be presented as one
    // enormous interval - an interval of the right size is reported as asystole.
    let signal = sinus(120.0, 60.0);
    let lose: Vec<usize> = (200..240).collect(); // ten seconds of silence

    let mut s = Server::new(1);
    open_one(&mut s, 1);
    let out = feed(&mut s, 1, &signal, &lose);

    let stats = s.stats(1).unwrap();
    assert_eq!(stats.missing_packets, lose.len() as u64);
    assert!(stats.loss_rate() > 0.0);

    let invented: Vec<Condition> = out
        .episodes
        .iter()
        .map(|e| e.condition)
        .filter(|c| {
            matches!(
                c,
                Condition::Pause | Condition::Asystole | Condition::Bradycardia
            )
        })
        .collect();
    assert!(
        invented.is_empty(),
        "a ten-second packet loss was reported as {invented:?}"
    );
}

#[test]
fn a_duplicate_packet_is_discarded() {
    let mut s = Server::new(1);
    open_one(&mut s, 3);
    let signal = sinus(10.0, 60.0);
    let mut out = ChannelOutput::default();
    let chunks: Vec<&[f32]> = signal.chunks(PACKET).collect();

    for (i, c) in chunks.iter().enumerate() {
        s.ingest(
            Packet {
                channel: 3,
                sequence: i as u64,
                samples: c,
            },
            &mut out,
        );
    }
    // Replay the middle of the stream.
    for (i, c) in chunks.iter().enumerate().take(20).skip(10) {
        let v = s.ingest(
            Packet {
                channel: 3,
                sequence: i as u64,
                samples: c,
            },
            &mut out,
        );
        assert!(
            matches!(v, Ingest::Stale { .. }),
            "duplicate accepted: {v:?}"
        );
    }
    assert_eq!(s.stats(3).unwrap().stale_packets, 10);
}

#[test]
fn a_gap_is_reported_as_a_gap() {
    let mut s = Server::new(1);
    open_one(&mut s, 5);
    let signal = sinus(10.0, 60.0);
    let mut out = ChannelOutput::default();
    let chunks: Vec<&[f32]> = signal.chunks(PACKET).collect();

    s.ingest(
        Packet {
            channel: 5,
            sequence: 0,
            samples: chunks[0],
        },
        &mut out,
    );
    let v = s.ingest(
        Packet {
            channel: 5,
            sequence: 4,
            samples: chunks[4],
        },
        &mut out,
    );
    assert_eq!(v, Ingest::AcceptedAfterGap { missing: 3 });
    assert_eq!(s.stats(5).unwrap().missing_packets, 3);
}

#[test]
fn an_unknown_channel_is_refused_rather_than_created() {
    let mut s = Server::new(1);
    let mut out = ChannelOutput::default();
    let v = s.ingest(
        Packet {
            channel: 99,
            sequence: 0,
            samples: &[0.0; PACKET],
        },
        &mut out,
    );
    assert_eq!(v, Ingest::Unknown);
    assert_eq!(s.channels(), 0);
}

#[test]
fn channels_are_isolated_from_each_other() {
    // One channel losing most of its packets must not change what another sees.
    let signal = sinus(60.0, 60.0);
    let mut solo = Server::new(4);
    open_one(&mut solo, 2);
    let alone = feed(&mut solo, 2, &signal, &[]);

    let mut both = Server::new(4);
    open_one(&mut both, 2);
    open_one(&mut both, 6); // same shard: 2 % 4 == 6 % 4
    assert_eq!(both.shard_of(2), both.shard_of(6));
    let mut out = ChannelOutput::default();
    for (i, chunk) in signal.chunks(PACKET).enumerate() {
        both.ingest(
            Packet {
                channel: 2,
                sequence: i as u64,
                samples: chunk,
            },
            &mut out,
        );
        if i % 3 == 0 {
            both.ingest(
                Packet {
                    channel: 6,
                    sequence: i as u64,
                    samples: chunk,
                },
                &mut out,
            );
        }
    }
    let shared = both.stats(2).unwrap();
    assert_eq!(shared.beats, alone.beats.len() as u64);
    assert_eq!(
        shared.missing_packets, 0,
        "a neighbour's loss leaked across"
    );
}

#[test]
fn a_dropped_packet_is_counted_and_not_silently_lost() {
    let mut s = Server::new(1);
    open_one(&mut s, 8);
    let signal = sinus(30.0, 60.0);
    let mut out = ChannelOutput::default();
    for (i, chunk) in signal.chunks(PACKET).enumerate() {
        if (40..60).contains(&i) {
            // Back-pressure: there is nowhere to put it.
            s.shard_mut(0).record_drop(8, 1);
            continue;
        }
        s.ingest(
            Packet {
                channel: 8,
                sequence: i as u64,
                samples: chunk,
            },
            &mut out,
        );
    }
    let stats = s.stats(8).unwrap();
    assert_eq!(stats.dropped_packets, 20);
    // Dropped, not miscounted as received-and-missing.
    assert_eq!(stats.missing_packets, 0);
    assert!(stats.loss_rate() > 0.15);
}

#[test]
fn a_channel_keeps_its_shard_for_life() {
    let s = Server::new(8);
    for ch in 0..1000u64 {
        assert_eq!(s.shard_of(ch), s.shard_of(ch));
        assert!(s.shard_of(ch) < 8);
    }
}
