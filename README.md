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
**Phase 4** — arrhythmia episodes: pause, asystole, brady/tachycardia,
ventricular runs, bigeminy. [`reports/PHASE-4.md`](reports/PHASE-4.md)
**Phase 5** — ventricular fibrillation, and what it cannot yet do.
[`reports/PHASE-5.md`](reports/PHASE-5.md)
**Phase 6** — the multi-channel runtime. [`reports/PHASE-6.md`](reports/PHASE-6.md)
**Phase 7** — cross-building for the deployment targets.
[`reports/PHASE-7.md`](reports/PHASE-7.md)
**Phase 8** — acting on the fibrillation flag.
[`reports/PHASE-8.md`](reports/PHASE-8.md)
**Phase 9** — wave delineation, and the atrial evidence it gives the beat layer.
[`reports/PHASE-9.md`](reports/PHASE-9.md)
**Phase 10** — the beat layer, fusion, idioventricular rhythm, and four things
that did not work. [`reports/PHASE-10.md`](reports/PHASE-10.md)
**Phase 11** — the patch corpus, and what counts as truth.
[`reports/PHASE-11.md`](reports/PHASE-11.md)

Every current figure, with the places the train/test split does not hold, is in
[`reports/PERFORMANCE.md`](reports/PERFORMANCE.md). A few of them:

| | public corpora, sealed | patch corpus, sealed |
|---|---|---|
| Records / hours | 318 / 885 h | 22 / 3,274 h, exhaustively reviewed |
| QRS sensitivity / precision | 99.20 % / 97.96 % | not measurable (§ below) |
| AF sensitivity / precision, end to end | 90.5 % / 98.8 % (AFDB) | — |
| AF false alarms on AF-free rhythm | 2.22 per 24 h | — |
| Ventricular Se / +P per beat | 95.1 % / 83.3 % (MIT-BIH) | 65.2 % / 88.4 % (patch bank, outside noise); device 84.3 % / 87.1 % |
| Ventricular review queue | 96.1 % / 89.2 %, 8.3 clusters per record | 59.4 % / 83.8 %, 14.8 clusters |
| Supraventricular Se / +P per beat | 24.9 % / 26.0 % (MIT-BIH) | 44.7 % / 69.6 % with rhythm runs, outside noise; device 58.9 % / 90.4 % |
| Asystole / pause | 100 % / 99.2 % (MIT-BIH) | — |
| Fibrillation alarm: onsets found, latency, false alarms | 19 / 20, 9 s, 4.97 per 24 h on Sudden Death (0.61 without its record 38); 4 in 616 h of other sealed Holter | — |
| Throughput | 211 ns per sample per channel, **~19,000 channels per core @ 250 Hz** | |

The patch column is agreement with an analyst who corrected the device's own
labels, outside the stretches the device called noise - inside them the labels
are the device's own, untouched by review; no patch recording has yet been read afresh by an expert, and detection
cannot be measured there because a beat the device never marked is invisible to
review. Truth, throughout, is either an annotation made independently of the
device or a label an analyst decided; the device's automatic calls are a
competitor, never truth.

---

## Layout

```
crates/
  ecg-wfdb/      WFDB reader: headers, formats 16/24/32/61/80/212/310/311,
                 MIT annotations. Verified against the reference implementation.
  ecg-zarr/      zarr v3 reader for the internal patch corpus: stored-zip
                 containers, blosc over zstd. Checked byte-exactly against zarr.
  ecg-dsp/       Streaming primitives: biquad cascades with computed group
                 delay, ring buffers, moving window statistics.
  ecg-pipeline/  Front-end filter bank and the per-channel pipeline.
  ecg-quality/   Noise and signal-quality detection.
  ecg-qrs/       QRS detector.
  ecg-rhythm/    RR interval stream, AF features and detector, episode
                 confirmation.
  ecg-beats/     Per-beat morphology features, a running beat template, a
                 bank of independent binary detectors (N/S/V/F), the morphology
                 bank behind the ventricular review queue, and the patch bank.
  ecg-ffi/       The standard interface: a safe Rust API and a versioned C ABI
                 (include/ecg.h) that stay fixed while the engine is replaced.
  ecg-single/    Builds dist/ecg_engine.rs as a crate, to test it against the
                 workspace it was generated from.
  ecg-server/    Multi-channel runtime: sharding, packet ordering, gap
                 handling, back-pressure accounting.
  ecg-bench/     Self-contained capacity benchmark for a deployment target:
                 no corpus, no data files, cross-compiles to 0.8 MB.
  ecg-eval/      Evaluation, sweeps, diagnostics and benchmarks.
dist/
  ecg_engine.rs  The whole engine as one generated source file (committed).
  ecg.h          The standard interface's header (committed).
manifests/
  records.json   Public record list with TRAIN/DEV/TEST zones, carried over
                 from deep_ecg's subject-level split.
  internal.json  The patch corpus's list. Generated, not committed: it is
                 per-exam metadata from a private corpus.
reports/
  PERFORMANCE.md Every current figure, and where the split does not hold.
  PHASE-N.md     What each phase did, including what was refuted.
  results/       Raw harness output.
tools/
  run_evaluation.sh           Regenerates every number in the report.
  build_engine.sh             Regenerates dist/, builds the C library from the
                              single file, and runs the conformance check.
  ecg_conformance.c           Loads an engine library by path and checks it
                              keeps the interface.
  build_internal_manifest.py  Builds manifests/internal.json.
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

The PhysioNet corpora are not in this repository. Point the harness at them with
`$DEEP_ECG_RAW` or `--data-root`; with no setting it looks for a `deep_ecg`
checkout beside this one. `manifests/records.json` carries the record list,
subject-level zone assignment and paths relative to that root.

```bash
export DEEP_ECG_RAW=/path/to/deep_ecg/data/raw
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

# Arrhythmia episodes, and the quality monitor against human annotation
./target/release/ecg-eval episodes --zone TEST --sources mitdb
./target/release/ecg-eval butqdb   --zone TEST --sources butqdb --threads 4

# Ventricular fibrillation: the model held out within TRAIN, and the alarm
# against the sealed Sudden Death onsets
./target/release/ecg-eval vf --zone TRAIN --sources vfdb,cudb \
    --holdout-every 3 --holdout-take
./target/release/ecg-eval vf-alarm --zone TEST --sources sddb

# Multi-channel runtime, with packet loss
./target/release/ecg-eval serve --zone ALL --sources mitdb --records 100 \
    --channels 512 --minutes 2 --fs 250 --threads 4 --loss 0.02

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

### The internal patch corpus

Private, and not in this repository. The harness reads it where `deep_ecg`
keeps it — `$DEEP_ECG_CANONICAL`, or a `deep_ecg` checkout beside this one —
and finds it by looking, so the same `--manifest` flag switches corpora.

```bash
# Build the record list (needs pyarrow; decodes no samples)
python3 tools/build_internal_manifest.py

# Recover each recording's gain: the containers do not carry one
./target/release/ecg-eval patch --manifest manifests/internal.json \
    --sources atheart-backup --zone TEST

# Beat classification against the 22 exhaustively reviewed sealed recordings,
# scored beside the device's own calls; --domain patch for the patch preset
./target/release/ecg-eval internal-beats --manifest manifests/internal.json \
    --sources atheart-backup --zone TEST --domain patch

# The supraventricular run detector alone, from beat positions, in seconds
./target/release/ecg-eval svrun --manifest manifests/internal.json \
    --sources atheart-backup --zone DEV

# Refit the patch ventricular ensemble on truth only - analyst-decided beats
# and the public corpora - then emit it as engine source
./target/release/ecg-eval internal-fit --manifest manifests/internal.json \
    --sources atheart-backup --zone TRAIN --records-max 400 --hours 24 \
    --out target/models/ventricular_patch.json
./target/release/ecg-eval emit-model --model target/models/ventricular_patch.json \
    --name VENTRICULAR_PATCH --temperature 4 \
    --out crates/ecg-beats/src/trees_patch_generated.rs
```

A deployment on the patch uses `PipelineConfig::patch(fs)`: the patch bank's
ventricular ensemble and supraventricular runs found by the rhythm. Both were
chosen against the patch corpus's exhaustive review and both cost the clinical
corpora, so `PipelineConfig::new(fs)` stays the default. In the harness,
`--domain patch` selects it.

`ecg-eval` with no arguments lists every option.

## Tests

```bash
cargo test --release -- --nocapture
```

Two kinds. Unit tests pin arithmetic that a corpus cannot check — the quality
score on a hand-built window, episode confirmation against hand-built state.
Metric regression tests run the real pipeline over real records and assert
floors on the numbers the reports quote: QRS sensitivity and precision, the
quality monitor's AUC for predicting a detector error, AF sensitivity, precision
and episode recall, the median false-alarm rate on normal sinus rhythm, and both
beat detectors' ROC.

They exist because three of the defects found while building this engine were
silent — a patch that stopped matching after a reformat, a loop that stalled on
one negative annotation, an argument the parser was swallowing. All three left
the code compiling and the unit tests passing.

Without the corpora those tests print `SKIPPED` rather than passing vacuously,
and `--nocapture` shows the value each one measured. A green suite that measured
nothing is the failure mode worth guarding against.

## Swapping the engine, or one stage of it

The engine is meant to be replaced often, so the boundary around it is fixed
and small, and the engine itself travels as one file. Two grains of upgrade are
supported, and a host needs no change for either:

- **the whole engine** - load a different `libecg`, or build a different
  `ecg_engine.rs`;
- **one stage inside it** - QRS detection, beat classification, atrial
  fibrillation, ventricular fibrillation and supraventricular runs each sit
  behind a trait (`ecg_pipeline::stages`), and the engine carries several
  implementations of some of them, each named `kind.variant@version`. A
  channel is told which to use, so a new stage can run beside the old one on
  the same signal before it is adopted, and a deployment can fall back without
  a new engine. From Rust a host can also supply a stage of its own
  (`ChannelPipeline::with_stages`).

| stage | implementations | default (clinical / patch) |
|---|---|---|
| `qrs` | `qrs.pt@1` | `qrs.pt@1` |
| `beats` | `beats.clinical@4`, `beats.clinical@3` (before the wide-beat features), `beats.patch@3` | `beats.clinical@4` / `beats.patch@3` |
| `af` | `af.logistic@1` | `af.logistic@1` |
| `vf` | `vf.spectral@2`, `vf.linear@1` (before the spectral features) | `vf.spectral@2` |
| `svrun` | `svrun.off@1`, `svrun.rate@2`, `svrun.rate@1` (no tachycardia rule) | `svrun.off@1` / `svrun.rate@2` |

A version moves whenever the stage's output would, so a name identifies
behaviour: every channel reports the stages it runs, and a finding can be
traced to the stage that made it. A stage built from a tuned configuration
reports itself as `kind.configured`, never as a registered name.

**One file.** `dist/ecg_engine.rs` is every engine crate - signal processing,
detection, classification, rhythm, the pipeline and the interface - generated
from the workspace by `ecg-eval amalgamate`. It depends on nothing but the
standard library and builds with plain `rustc`:

```bash
./tools/build_engine.sh    # regenerate dist/, build libecg, run the conformance check
# or by hand:
rustc --edition 2021 -O -C panic=unwind --crate-name ecg \
      --crate-type cdylib --crate-type staticlib dist/ecg_engine.rs
```

Its first line names it - `live-ecg 0.1.0 src <hash>`, the hash of the generated
source - and `ecg_engine_id()` returns the same string, so a host can log which
engine produced a finding. The workspace remains the source of truth: a test
fails when the committed file is not what the workspace generates, and another
checks that the single-file engine's output is bit-identical to the
workspace's on MIT-BIH records under both presets.

**One interface.** `dist/ecg.h` (source: `crates/ecg-ffi/include/ecg.h`), ABI
1.1: create a channel, push samples in millivolts, poll events, read a status,
destroy it; since 1.1, `ecg_config.stages` selects stages by name
(`"vf=vf.linear@1;beats=beats.clinical@3"`), `ecg_engine_stages()` lists what
the engine carries and `ecg_channel_stages()` what a channel runs. Everything the engine finds comes back as one record, `ecg_event`,
told apart by `kind` and `code` - beats, rhythm episodes, AF windows, VF
episodes, electrode failures, supraventricular runs. The rules that let
engines be swapped:

- `ecg_abi_version()` is `major << 16 | minor`. A host refuses a different
  major. Within a major, engines only add - kinds, codes, struct fields at the
  end - and never renumber or remove.
- A host skips any `kind` or `code` it does not know.
- Every struct passed across carries `struct_size`, so an older host and a
  newer engine agree on how much of it exists.
- Codes are the interface's own, mapped from the engine's types by an explicit
  table, so a refactor inside the engine does not move them. A test holds the
  header and the Rust constants to the same numbers and struct layouts.
- Nothing unwinds across the boundary: an internal failure returns
  `ECG_ERR_INTERNAL`, and the channel then refuses further work.

**Checking a new engine before trusting it.** `tools/ecg_conformance.c` is a
host that knows the engine only by the path it is given: it loads the library
at run time, checks the ABI version, and exercises the contract - bad
configurations refused, beats in time order, spans well formed, scores in
range, two channels agreeing, the time base advancing through gaps, partial
polls, null arguments, the patch preset, a 1.0-sized configuration, and every
stage the engine lists selected in turn. It checks the contract, not accuracy;
accuracy is `tools/run_evaluation.sh`.

```bash
dist/ecg_conformance path/to/new/libecg.so
```

From Rust, build the file as its own crate (a `[lib] path` pointing at it,
named `ecg`) and use `ecg::ecg_ffi::Engine`, the same interface without the C.

## Deployment targets

Every crate is pure Rust, so cross-compiling needs only a linker and `rust-lld`
ships with the toolchain — `rustup target add` is the whole setup. The Linux
targets link static musl binaries, which removes the glibc version question.

```bash
./tools/cross-build.sh
```

| target | deployment | binary |
|---|---|---|
| `aarch64-unknown-linux-musl` | Raspberry Pi 5, 64-bit ARM Linux | 0.82 MB |
| `armv7-unknown-linux-musleabihf` | 32-bit ARM Linux | 0.81 MB |
| `x86_64-unknown-linux-musl` | low-end PC, servers | 0.84 MB |
| `aarch64-apple-ios` | iPhone, iPad | 0.6 MB |
| `aarch64-apple-darwin` / `x86_64-apple-darwin` | Mac | 0.6 / 0.7 MB |

Copy `ecg-bench` to the target and run it there — it synthesises its own signal
and reports capacity plus a detection sanity check:

```bash
./ecg-bench --channels 64 --seconds 120 --threads 4
```

**No figure in these reports was measured on a deployment target.** All of them
are from an Apple M1 Ultra, and Phase 6 found capacity is bounded by memory
bandwidth rather than core count, so they do not transfer. That is what the
benchmark is for.

## Next

- **Run `ecg-bench` on a Raspberry Pi 5, a phone and a low-end PC.** The engine
  cross-builds for all of them (Phase 7); nothing has been measured on one.
- **Fibrillation detection is not accurate enough to alarm on** (Phase 5). It
  needs spectral, complexity and phase-space features, and a sealed set that
  does not exist in the inherited split. It now bounds how much nonsense can be
  suppressed during fibrillation too (Phase 8).
- **PQRST delineation**, which is the single thing that would unblock
  supraventricular detection, the atrial-fibrillation false-alarm tail and the
  quality monitor's inability to separate full-diagnostic from QRS-only signal —
  three limitations that turn out to be one missing piece of evidence.
- **Beat-classification precision**, which is what bounds ventricular run
  detection (Phase 4 §5), and **supraventricular detection**, which needs P-wave
  evidence and so delineation.

On deep learning and the Hailo accelerator: not needed so far, and Phase 3 §3
shows the current models are capacity-saturated rather than starved — depth 6
and 300 trees bought nothing. The case for a deep model should be made against
these numbers, not against an untested assumption.
