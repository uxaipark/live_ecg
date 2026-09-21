//! Multi-channel runtime.
//!
//! Receives packets from many patches and runs one [`ChannelPipeline`] per
//! channel. The engineering content is almost entirely in what happens when the
//! stream is *not* clean, because a stream from a body-worn radio never is.
//!
//! # No shared state on the hot path
//!
//! A channel's state is owned by exactly one shard, and a shard is owned by one
//! thread. Channels are routed to shards by identifier, so no lock is taken to
//! process a packet and no channel can be touched by two threads at once. Adding
//! a channel takes a lock; carrying one does not.
//!
//! This also bounds the blast radius: a channel that behaves badly costs its own
//! shard's time and nothing else's.
//!
//! # A lost packet must not become a clinical finding
//!
//! This is the property the whole design is arranged around. Packets arrive out
//! of order, twice, or not at all. If a gap is closed by splicing the two sides
//! together, the engine sees one enormous interval and reports a pause that
//! never happened — and by Phase 4 a pause of the right size is reported as
//! asystole. Feeding silence instead is no better: it manufactures a flatline.
//!
//! So a gap is declared as a gap. [`ChannelPipeline::mark_gap`] discards the
//! windowed state that spans it and keeps the adaptive state that still
//! describes the patient.
//!
//! # Back-pressure is accounted, never silent
//!
//! When a shard cannot keep up, packets are dropped — there is nowhere to put
//! them and a growing queue only moves the failure later. What matters is that
//! a drop is *counted* and surfaces as a gap, so the analysis knows it is
//! missing data rather than believing it has continuous signal.

use ecg_pipeline::{ChannelOutput, ChannelPipeline, PipelineConfig};
use std::collections::HashMap;

pub type ChannelId = u64;

/// One packet of samples from one patch.
#[derive(Debug, Clone, Copy)]
pub struct Packet<'a> {
    pub channel: ChannelId,
    /// Monotonic per channel, counting packets rather than samples.
    pub sequence: u64,
    pub samples: &'a [f32],
}

/// What the runtime did with a packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ingest {
    /// Processed in order.
    Accepted,
    /// Processed, but packets before it never arrived. The pipeline was told.
    AcceptedAfterGap { missing: u64 },
    /// Already seen, or older than one already processed. Discarded.
    ///
    /// Late packets are not reordered back into place: the pipeline is a
    /// streaming one and cannot rewind, and holding a buffer long enough to
    /// reorder would add that latency to every channel to rescue a few.
    Stale { expected: u64 },
    /// The channel is not registered.
    Unknown,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ChannelStats {
    pub packets: u64,
    pub samples: u64,
    /// Packets that never arrived, inferred from sequence numbers.
    pub missing_packets: u64,
    /// Packets discarded for arriving late or twice.
    pub stale_packets: u64,
    /// Packets dropped because the shard could not keep up.
    pub dropped_packets: u64,
    pub beats: u64,
    pub episodes: u64,
    pub good_samples: u64,
    pub unusable_samples: u64,
}

impl ChannelStats {
    /// Share of expected packets that never reached the analysis.
    pub fn loss_rate(&self) -> f64 {
        let expected = self.packets + self.missing_packets + self.dropped_packets;
        if expected == 0 {
            0.0
        } else {
            (self.missing_packets + self.dropped_packets) as f64 / expected as f64
        }
    }
}

struct Channel {
    pipeline: ChannelPipeline,
    /// Sequence number the next in-order packet will carry.
    next_sequence: u64,
    samples_per_packet: usize,
    stats: ChannelStats,
}

/// A set of channels processed by one thread.
///
/// Public so a host can drive shards from its own executor; the runtime does not
/// impose a threading model.
pub struct Shard {
    channels: HashMap<ChannelId, Channel>,
    scratch: ChannelOutput,
}

impl Default for Shard {
    fn default() -> Self {
        Self::new()
    }
}

impl Shard {
    pub fn new() -> Self {
        Shard {
            channels: HashMap::new(),
            scratch: ChannelOutput::default(),
        }
    }

    pub fn open(&mut self, channel: ChannelId, cfg: PipelineConfig, samples_per_packet: usize) {
        self.channels.insert(
            channel,
            Channel {
                pipeline: ChannelPipeline::new(cfg),
                next_sequence: 0,
                samples_per_packet,
                stats: ChannelStats::default(),
            },
        );
    }

    pub fn close(&mut self, channel: ChannelId) -> Option<ChannelStats> {
        self.channels.remove(&channel).map(|c| c.stats)
    }

    pub fn len(&self) -> usize {
        self.channels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    pub fn stats(&self, channel: ChannelId) -> Option<ChannelStats> {
        self.channels.get(&channel).map(|c| c.stats)
    }

    /// Record that a packet was dropped before reaching the analysis.
    ///
    /// Kept separate from ordinary loss so the two can be told apart: one is the
    /// radio's fault and one is ours.
    pub fn record_drop(&mut self, channel: ChannelId, packets: u64) {
        if let Some(c) = self.channels.get_mut(&channel) {
            c.stats.dropped_packets += packets;
            c.next_sequence = c.next_sequence.saturating_add(packets);
            c.pipeline.mark_gap(packets * c.samples_per_packet as u64);
        }
    }

    /// Process one packet. Findings are appended to `out`.
    pub fn ingest(&mut self, packet: Packet<'_>, out: &mut ChannelOutput) -> Ingest {
        let Some(c) = self.channels.get_mut(&packet.channel) else {
            return Ingest::Unknown;
        };

        let verdict = if packet.sequence < c.next_sequence {
            c.stats.stale_packets += 1;
            return Ingest::Stale {
                expected: c.next_sequence,
            };
        } else if packet.sequence > c.next_sequence {
            let missing = packet.sequence - c.next_sequence;
            c.stats.missing_packets += missing;
            // Tell the pipeline before it sees the next sample, so nothing is
            // measured across the hole.
            c.pipeline.mark_gap(missing * c.samples_per_packet as u64);
            Ingest::AcceptedAfterGap { missing }
        } else {
            Ingest::Accepted
        };

        self.scratch.clear();
        c.pipeline.push(packet.samples, &mut self.scratch);
        c.next_sequence = packet.sequence + 1;
        c.stats.packets += 1;
        c.stats.samples += packet.samples.len() as u64;
        c.stats.beats += self.scratch.beats.len() as u64;
        c.stats.episodes += self.scratch.episodes.len() as u64;
        c.stats.good_samples += self.scratch.good_samples;
        c.stats.unusable_samples += self.scratch.unusable_samples;

        append(out, &self.scratch);
        verdict
    }

    /// Close every channel's open episodes, at shutdown.
    pub fn finish(&mut self, out: &mut ChannelOutput) {
        for c in self.channels.values_mut() {
            self.scratch.clear();
            c.pipeline.finish(&mut self.scratch);
            append(out, &self.scratch);
        }
    }
}

fn append(out: &mut ChannelOutput, src: &ChannelOutput) {
    out.beats.extend_from_slice(&src.beats);
    out.intervals.extend_from_slice(&src.intervals);
    out.af.extend_from_slice(&src.af);
    out.classes.extend_from_slice(&src.classes);
    out.episodes.extend_from_slice(&src.episodes);
    out.vf.extend_from_slice(&src.vf);
    out.vf_episodes.extend_from_slice(&src.vf_episodes);
    out.good_samples += src.good_samples;
    out.acceptable_samples += src.acceptable_samples;
    out.unusable_samples += src.unusable_samples;
    if src.quality.is_some() {
        out.quality = src.quality;
    }
}

/// The whole runtime: a fixed set of shards and a routing rule.
pub struct Server {
    shards: Vec<Shard>,
}

impl Server {
    pub fn new(shards: usize) -> Self {
        Server {
            shards: (0..shards.max(1)).map(|_| Shard::new()).collect(),
        }
    }

    /// Which shard owns a channel. A channel's state never moves, so this is
    /// fixed for its lifetime and no rebalancing can race with processing.
    #[inline]
    pub fn shard_of(&self, channel: ChannelId) -> usize {
        (channel % self.shards.len() as u64) as usize
    }

    pub fn shards(&self) -> usize {
        self.shards.len()
    }

    pub fn shard_mut(&mut self, i: usize) -> &mut Shard {
        &mut self.shards[i]
    }

    /// Hand out the shards so a host can drive each on its own thread.
    pub fn split(&mut self) -> std::slice::IterMut<'_, Shard> {
        self.shards.iter_mut()
    }

    pub fn open(&mut self, channel: ChannelId, cfg: PipelineConfig, samples_per_packet: usize) {
        let i = self.shard_of(channel);
        self.shards[i].open(channel, cfg, samples_per_packet);
    }

    pub fn close(&mut self, channel: ChannelId) -> Option<ChannelStats> {
        let i = self.shard_of(channel);
        self.shards[i].close(channel)
    }

    pub fn stats(&self, channel: ChannelId) -> Option<ChannelStats> {
        self.shards[self.shard_of(channel)].stats(channel)
    }

    pub fn channels(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Single-threaded ingestion, for tests and small deployments.
    pub fn ingest(&mut self, packet: Packet<'_>, out: &mut ChannelOutput) -> Ingest {
        let i = self.shard_of(packet.channel);
        self.shards[i].ingest(packet, out)
    }
}
