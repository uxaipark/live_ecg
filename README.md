# live_ecg

Real-time analysis engine for a single-lead wearable ECG patch. Rust, streaming,
allocation-free in the steady state. Built to run identically on a server
carrying hundreds of channels and on the patch-side host (Raspberry Pi 5, phone,
low-end PC).

**Phase 1** — preprocessing filter bank, noise detection, QRS detection.
[`reports/PHASE-1.md`](reports/PHASE-1.md)
**Phase 2** — RR intervals, AF features, AF detection, episode confirmation.
[`reports/PHASE-2.md`](reports/PHASE-2.md)
**Phase 3** — per-beat morphology and a bank of binary N/S/V detectors.
[`reports/PHASE-3.md`](reports/PHASE-3.md)

| | TRAIN (selection) | TEST (sealed) |
|---|---|---|
| Records / hours | 122 / 236 h | 318 / 885 h |
| QRS sensitivity | 99.862% | 99.192% |
| QRS precision | 99.630% | 97.973% |
| Noise detection | — | AUC 0.94 for predicting detector error |
| AF sensitivity / precision (end to end) | — | 86.2% / 98.8% |
| AF episodes found (≥30 s) | — | 33 / 34 |
| AF false alarms, 270 h with no AF | — | **0.00 per 24 h median subject** (mean 6.5) |
| VEB sensitivity / precision | — | 85.2% / 69.2% (87% on lead II) |
| SVEB sensitivity / precision | — | 62.0% / 29.8% |
| Throughput | ~117 ns per sample per channel — **~28,700 channels per core @ 250 Hz** | |

---

## Layout

```
crates/
  ecg-wfdb/      WFDB reader: headers, formats 16/24/32/61/80/212/310/311,
                 MIT annotations. Verified against the reference implementation.
  ecg-dsp/       Streaming primitives: biquad cascades with computed group
                 delay, ring buffers, moving window statistics.
  ecg-pipeline/  Front-end filter bank and the per-channel pipeline.
  ecg-quality/   Noise and signal-quality detection.
  ecg-qrs/       QRS detector.
  ecg-rhythm/    RR interval stream, AF features and detector, episode
                 confirmation.
  ecg-beats/     Per-beat morphology features, a running beat template, and a
                 bank of independent binary detectors (N/S/V).
  ecg-eval/      Evaluation, sweeps, diagnostics and benchmarks.
manifests/
  records.json   Record list with TRAIN/DEV/TEST zones, carried over from
                 deep_ecg's subject-level split.
reports/
  PHASE-1.md     Results and method.
  results/       Raw harness output.
tools/
  run_evaluation.sh   Regenerates every number in the report.
```

## Design

**One pass, four taps.** The filter bank emits `baseline` (<0.5 Hz), `clean`
(0.5–40 Hz, mains-notched), `qrs` (5–20 Hz) and `hf` (>40 Hz), all sample
aligned. The detector and the quality monitor share the `qrs` tap; nothing is
filtered twice.

**Quality runs ahead of detection, not beside it.** The quality verdict gates
the detector's threshold adaptation and decides whether a latched threshold is
allowed to come back down. That coupling is load-bearing — see §6 of the Phase 1
report.

**Arrhythmias are a bank of binary detectors over a shared feature layer, not
one multi-class model.** Each condition has its own evidence base and its own
cost asymmetry — a missed ventricular run can kill, a false premature atrial
beat costs a reviewer ten seconds — and a single `argmax` has nowhere to put
that. Each detector carries its own threshold, can be revalidated alone, and
publishes its own score so the episode layer sees ambiguity rather than a label.
Measured, the ventricular and supraventricular detectors lean on almost disjoint
features. See Phase 3 §1.

**No allocation in the steady state.** A `ChannelPipeline` is `Send`, owns all
its state, allocates only at construction, and costs ~48 KB. A server scales by
handing each worker thread a slice of the channels; nothing is shared.

**Fixed-point and accelerator targets.** Everything is `f32` sample data with
`f64` filter state, block-processed, with no data-dependent control flow in the
hot path. The rule-based and small-model stages are the ones running here; a
deep model, if one is needed later for beat classification, belongs on the Hailo
accelerator behind this front end, which already reduces a sample stream to
beats plus quality.

## Running

Needs the PhysioNet corpora at the paths in `manifests/records.json`
(`deep_ecg/data/raw`).

```bash
cargo build --release

# Detection metrics
./target/release/ecg-eval qrs --zone TEST --sources mitdb --per-record

# Noise detection against the MIT noise-stress protocol
./target/release/ecg-eval quality --zone ALL --sources nstdb

# AF: rhythm logic isolated, then end to end, then false alarms
./target/release/ecg-eval af --zone TEST --sources afdb --beats reference
./target/release/ecg-eval af --zone TEST --sources afdb --beats detected
./target/release/ecg-eval af --zone TEST --sources nsrdb --beats detected \
    --assume-af-free nsrdb --per-record

# Refit the AF model (TRAIN only) and see each feature's AUC
./target/release/ecg-eval fit-af --zone TRAIN --sources afdb,ltafdb,nsrdb \
    --assume-af-free nsrdb --beats reference --l2 0.015

# Beat classification: AAMI confusion matrix and per-detector ROC
./target/release/ecg-eval beats --zone TEST --sources mitdb,svdb,incartdb \
    --beats reference --per-record

# Refit both beat detectors and regenerate their tree ensembles
./target/release/ecg-eval fit-beats --zone TRAIN --sources mitdb,svdb \
    --beats reference --gbdt --depth 4 --trees 120

# Parameter sweep (TRAIN only)
./target/release/ecg-eval sweep --zone TRAIN --sources mitdb,svdb,nsrdb \
    --sweep-lo 5,8 --sweep-hi 15,20,25 --sweep-thr 0.10,0.15,0.20

# Why a record scores badly
./target/release/ecg-eval diag  --zone TEST --sources nsrdb --records 16272
./target/release/ecg-eval trace --zone TEST --sources nsrdb --records 16272 --from-sec 35578

# Throughput
./target/release/ecg-eval stages --zone ALL --sources mitdb --records 100
./target/release/ecg-eval bench  --zone ALL --sources mitdb --records 100 \
    --channels 256 --minutes 5 --fs 250

# Everything in the report
./tools/run_evaluation.sh
```

`ecg-eval` with no arguments lists every option.

## Next

- Wire beat classes back into the AF path and re-measure Phase 2's false-alarm
  rate — the one concrete prediction Phase 3 left open.
- Arrhythmia episode detection: pause and asystole, bradycardia and tachycardia,
  ventricular runs and VT/VF, bigeminy and trigeminy. Each is another binary
  detector in the same bank.
- The multi-channel server supervisor.

On deep learning and the Hailo accelerator: not needed so far, and Phase 3 §3
shows the current models are capacity-saturated rather than starved — depth 6
and 300 trees bought nothing. The case for a deep model should be made against
these numbers, not against an untested assumption.
