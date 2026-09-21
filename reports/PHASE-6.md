# live_ecg — Phase 6: the multi-channel runtime

Receiving from many patches at once. Almost all of the engineering here is in
what happens when the stream is *not* clean, because a stream from a body-worn
radio never is.

---

## 1. Capacity

Apple M1 Ultra, 250 Hz, 250 ms packets, one shard per thread.

| | |
|---|---|
| Cost per sample | **132–185 ns** of CPU |
| Capacity, one shard | **~22,500–30,700 channels** @ 250 Hz |
| Capacity, 4 shards | ~123,000 channels |
| Memory | 97 MB resident for 1,024 channels — **~88 KB per channel** |
| Under 10% packet loss | no cost; the gap path is cheaper than the signal path |

The requirement was hundreds of channels per server. One shard covers that by
about two orders of magnitude, so the constraint on this design is not
throughput.

Per-shard capacity is *higher* at 4 threads than at 20 (30,700 against 22,500).
That is memory bandwidth, not scheduling: every channel is walking its own ring
buffers, and twenty of them contend for the same cache. It is worth knowing
before anyone sizes a deployment by multiplying a single-core figure by the
core count.

---

## 2. No shared state on the hot path

A channel's state belongs to exactly one shard, and a shard to one thread.
Channels route to shards by identifier, so processing a packet takes no lock and
no channel is ever touched by two threads. Opening a channel takes a lock;
carrying one does not.

A channel keeps its shard for life. Nothing rebalances, so nothing can race with
processing, and the cost is that a shard's load depends on which identifiers
landed on it. For hundreds of channels against a capacity of tens of thousands
that does not matter.

It also bounds the blast radius: a channel that misbehaves costs its own shard's
time and nothing else's. There is a test for exactly that — one channel losing
two thirds of its packets does not change a single beat reported on a neighbour
sharing its shard.

---

## 3. The failure this is built around

**Splicing across a lost packet does not produce a false pause. It silently
moves the clock.**

That was the surprise. The obvious worry is that joining the two sides of a gap
presents one enormous interval, which by Phase 4 is reported as asystole. It
does not: the skipped samples are never fed, so the pipeline's sample counter
never advances and the two sides join seamlessly. Ten seconds of signal simply
vanish, and every finding after them is reported ten seconds early with nothing
saying so.

A wrong timeline is worse than a missing one. A missing stretch is visible; a
compressed one is not, and every burden figure, episode duration and time
stamp downstream inherits the error.

The first test written for this asserted the obvious property — "a lost packet
does not become a pause" — and **passed without the fix in place.** It was
reassuring rather than informative. The test that replaced it checks the clock:
with gap handling removed, the last beat of a 120-second recording is reported
at 109.1 seconds.

So a gap is declared as one. `ChannelPipeline::mark_gap` advances the clock by
the lost duration and discards the windowed state that would otherwise span it —
the filter memories, the detector's retained traces, the quality windows, the RR
history, the fibrillation window. It keeps the adaptive state, because a dropped
packet is not a new patient: the thresholds, the beat template and the slow
per-channel references all still describe the same person.

Every stateful component grew an `on_gap` for this, and the distinction between
what it discards and what it keeps is the whole content of each one.

---

## 4. What the runtime does with a misbehaving stream

| situation | behaviour |
|---|---|
| in order | processed |
| gap in sequence | clock advanced, windowed state discarded, missing packets counted |
| duplicate, or late | discarded and counted, never replayed into a streaming pipeline that cannot rewind |
| shard cannot keep up | dropped, counted **separately** from radio loss, and surfaced as a gap |
| unknown channel | refused, not silently created |

Late packets are not reordered back into place. The pipeline is streaming and
cannot rewind, and holding a buffer long enough to reorder would add that
latency to every channel to rescue a few.

Dropped and missing are counted apart on purpose: one is the radio's fault and
one is ours, and a deployment that cannot tell them apart cannot fix either.

---

## 5. Limitations

- **No transport.** This is the runtime, not a network stack: it takes packets
  from a caller. Sockets, framing, authentication and persistence are the host's.
- **Back-pressure policy is the caller's.** `record_drop` accounts for a drop;
  deciding *when* to drop needs a queue, and the queue belongs to whoever owns
  the transport.
- **Shard assignment is static.** A skewed identifier distribution skews load.
  At hundreds of channels against a capacity of tens of thousands this is not
  worth solving.
- **Every figure is from an M1 Ultra.** Nothing here has been measured on a
  Raspberry Pi 5, a phone or a low-end PC, which is where this is meant to run.
  That is the next thing to do and it is not done.
- **The fibrillation flag is not consumed.** `in_vf()` is exposed and nothing
  acts on it, so beat-based conclusions are still emitted during fibrillation
  where they mean nothing (Phase 5 §6).
