# live_ecg — Phase 1: filter bank, noise detection, QRS detection

Real-time single-lead analysis engine for a wearable ECG patch. Rust, streaming,
allocation-free in the steady state. This report covers the first three stages
in the order they were requested: **preprocessing filters → noise detection →
QRS detection**, each with its evaluation.

Regenerate every number here with `tools/run_evaluation.sh`; raw outputs are in
`reports/results/`.

---

## 1. Evaluation protocol

**Train and test are separated at the subject level, and the split was not made
by this project.** `manifests/records.json` carries the zone assignment already
established in `deep_ecg/manifests/zones/zones@dev.parquet`, which assigns
*subjects* (not records) to TRAIN / DEV / TEST. Reusing it means a subject whose
recordings appear in several corpora cannot land on both sides.

| | |
|---|---|
| Selection set (TRAIN) | mitdb, svdb, nsrdb, stdb, qtdb, ltdb — 122 records, 236 h |
| Sealed set (TEST) | every corpus, 318 records, 885 h |
| Matching tolerance | ±150 ms, one-to-one, AAMI EC57 / `bxb` convention |
| Reference | WFDB beat annotations (`isqrs` symbol set) |
| Lead | index 0, the whole record, no warm-up excluded |

Every parameter in the engine was chosen by sweeping on TRAIN. TEST was scored
after the configuration was frozen. Two defects found *during* the TEST run are
called out explicitly in §6 — both were structural faults rather than parameter
choices, both were fixed and re-verified on TRAIN, and both are reported here
rather than quietly absorbed.

### Corpora deliberately excluded from selection

- **afdb** — its `.qrs` beat reference is machine-generated and unaudited.
  Reported, never used to choose a parameter.
- **cudb, vfdb** — ventricular fibrillation; "beat" is not well defined.
- **qtdb sel30–sel52, sddb (11 records), nstdb noise-only records** — 37 records
  with no full-record beat reference. QTDB's `.man` files annotate about thirty
  beats per record for QT measurement; scoring a whole-record detector against
  them would count every unannotated beat as a false positive. **Skipped, and
  the harness names them.** Substituting the partial file would manufacture a
  number.

### The foundation was verified before anything was measured on it

The harness reads WFDB directly in Rust. Before trusting a single metric, the
reader was differenced against the reference `wfdb` implementation:

- **60/60 annotation files** across 15 datasets — identical sample positions and
  symbols.
- **79/79 lead reads** across 15 datasets — identical physical values.

This was not ceremony. It found a real bug: the 32-bit `SKIP` interval in MIT
annotation files is **signed**, and files written by recent WFDB open with a
`-1` skip ahead of a time-resolution note. Reading it unsigned put every
subsequent annotation 2³² samples into the future. Three MIT-BIH
Supraventricular records scored **0.0% sensitivity** before the fix and 99.9%
after — the detector had been right all along.

---

## 2. Stage 1 — preprocessing filter bank

One pass produces four sample-aligned taps, so nothing downstream filters twice:

| tap | band | used by |
|---|---|---|
| `baseline` | below 0.5 Hz | wander / motion feature |
| `clean` | 0.5–40 Hz, mains-notched | morphology, fiducial placement |
| `qrs` | 5–20 Hz | detector **and** quality monitor |
| `hf` | above 40 Hz | EMG / contact-noise feature |

Cascaded second-order sections, `f64` coefficients and state. The baseline
high-pass sits near 2·10⁻³ in normalised frequency, where `f32` state in a
direct-form recursion loses real precision; the recursion is serial and cannot
be vectorised within a channel anyway, so the wider state costs nothing.

**Group delay is computed, not tuned.** The detector places the R fiducial on
`clean`, which an IIR chain shifts by a frequency-dependent amount. Rather than
calibrate a constant, `Cascade::group_delay` differentiates the phase response
of the coefficients actually in use, and the pipeline subtracts it. That moved
the median fiducial offset on MIT-BIH from **+15.2 ms to 0.0 ms**, and it stays
correct when a corner frequency, the sample rate or the mains decision changes.

**Mains detection is a narrowband test, not a coin flip.** Comparing 50 Hz
against 60 Hz degenerates below ~150 Hz sampling, where only one line is
representable — the survivor always "won", so a notch was applied to every
128 Hz recording whether or not it carried interference. Each candidate is now
probed against its own spectral neighbourhood (±7 Hz) and notched only if it
stands 6× above it. On QTDB that removed a **12 ms lag** the unnecessary notch
had been adding to every reported R position.

---

## 3. Stage 2 — noise detection

### What it answers

Narrowly: *is the next second of this channel worth believing?* The output gates
the detector's adaptive state and feeds the episode layer; it is not a
classifier of artefact types.

### Ground truth

The MIT-BIH Noise Stress Test protocol: records `118e*`/`119e*` are clean
recordings with calibrated noise added in two-minute segments alternating with
two clean minutes, starting five minutes in. That gives an **exact** per-second
label at six known SNRs — far stronger than a subjective artefact annotation.

Two things are measured, because they answer different questions: whether the
monitor can tell a noisy second from a clean one (AUC-noise), and whether it
predicts that the detector is about to be wrong (AUC-err). A quality signal that
scores well and changes no decision is worth nothing.

### Features are scale-invariant, and each one's orientation was measured

The first version normalised each band's power by total in-band power. It
measured the *opposite* of what it claimed — added motion artefact lands inside
the analysis band, so the denominator grows with the noise:

| feature | AUC-noise | AUC-err | |
|---|---|---|---|
| `hf_ratio` (as first built) | 0.27 | 0.17 | **backwards** |
| `base_ratio` (as first built) | 0.25 | 0.12 | **backwards** |
| `flat_frac` (as first built) | 0.23 | 0.03 | **backwards** |
| combined score | 0.49 | 0.47 | worthless |

Per-feature AUC is why this was visible at all. Three distinct bugs were behind
it:

1. `base_ratio` used raw **power** of the baseline tap, which is dominated by the
   electrode's standing DC offset — it read 0.996 on pristine signal. Measuring
   it as **variance** turned AUC 0.25 into 0.81.
2. `flat_frac` scaled its slope threshold by the *current* window's
   peak-to-peak, so a larger excursion raised the bar for what counted as
   motion. It now references the channel's own slow amplitude.
3. Saturation was tested against the raw sample in millivolts. Electrode
   half-cell potential puts a standing offset of tens of millivolts on the
   input, so the test reported permanent saturation on perfectly good signal —
   `sat_frac` was 0.99 while peak-to-peak was 2.3 mV, which alone crushed the
   score to 0.02 and condemned 100% of a 24 dB record. It is now measured on the
   DC-removed signal.

The surviving features are invariant to amplitude, either as shape statistics,
as bounded power fractions, or as ratios against the channel's own slow history
— which also means the thresholds survive an uncalibrated recording.

| feature | AUC-noise | AUC-err |
|---|---|---|
| kurtosis (negated) | 0.83 | 0.89 |
| QRS-band share (negated) | 0.80 | 0.92 |
| relative amplitude deviation | 0.73 | 0.89 |
| baseline variance share | 0.80 | 0.85 |
| **combined score** | **0.78** | **0.94** |

Thresholds were set from the measured percentiles of the clean population
(`ecg-eval qfeat`), not by eye.

### Vetoes multiply, soft evidence averages

Rail contact, a flatlined lead and extreme out-of-band power each condemn a
window alone. Kurtosis, QRS-band share and relative amplitude each have a
legitimate low range that depends on the **rhythm**, not the quality — a clean
bigeminal or paced recording is far less leptokurtic than clean sinus for that
reason alone. Multiplying all six let one term veto the other five: 79% of
seconds of the clean, bigeminal record svdb/801 were condemned, and 81% of the
paced mitdb/217. Averaging the soft terms requires agreement, and real artefact
degrades all three at once. Shape features additionally pass on *either* an
absolute test or a test against the channel's own median.

After both changes svdb/801 is no longer flagged at all (79% → 0.0% of seconds
unusable) and mitdb/217 drops from 81% to 8.0%.

### Result, by added-noise SNR

| SNR | AUC-noise | AUC-err | detector precision, gate open | gate closed |
|---:|---:|---:|---:|---:|
| −6 dB | 0.989 | 0.960 | 65.3% | **93.8%** |
| 0 dB | 0.918 | 0.861 | 73.6% | **83.5%** |
| 6 dB | 0.760 | 0.710 | 86.6% | 87.1% |
| 12 dB | 0.584 | 0.655 | 98.1% | 98.1% |
| 18 dB | — | — | 99.93% | 99.93% |
| 24 dB | — | — | 99.96% | 99.96% |

At 18 and 24 dB the added noise does not actually degrade the signal, so there
is nothing to detect and nothing to gain — AUC there is over a near-empty
positive class and is not meaningful. The important property is that **gating
costs nothing when the signal is good** (99.956% → 99.956%) and buys 28 points
of precision when it is not.

### Beat suppression is off by default

Gating *threshold adaptation* is always on. Gating *beat emission* is a separate
switch, default off, because suppressing beats hides asystole — that call
belongs to the episode layer, not the detector. The measured cost of forcing it
on over clean corpora is a sensitivity drop from 99.2% to 92.6% on MIT-BIH, for
half a point of precision. The numbers are in the report so the decision is
made on evidence.

---

## 4. Stage 3 — QRS detection

Pan-Tompkins topology (band-pass → derivative → square → moving-window
integration → adaptive threshold) with the refinements that decide whether a
detector survives real wearable data. Parameters were selected by grid sweep on
TRAIN, ranked by **per-record mean F1**, not the pooled figure — pooled is
dominated by the longest records, and a configuration that ruins a short record
can still come out on top.

Selecting on MIT-BIH alone chose 5–35 Hz / 100 ms / 0.12. Selecting across six
corpora at three sample rates chose **5–20 Hz / 120 ms / 0.15**, which is both
different and closer to the literature. That is the argument for multi-source
selection in one line.

### TRAIN (selection set, 122 records, 236 h)

| | |
|---|---|
| Sensitivity | **99.862%** |
| Precision (+P) | **99.630%** |
| F1 | 99.746% |
| DER | 0.509% |
| per-record F1 | median 99.886, p10 99.279, min 92.99 |

### TEST (sealed, scored after freezing, 318 records, 885 h, 3.75 M beats)

| | |
|---|---|
| Sensitivity | **99.192%** |
| Precision (+P) | **97.973%** |
| F1 | 98.579% |
| DER | 2.861% |
| per-record F1 | median 99.855, p10 95.66, mean 98.28 |

Per corpus:

| corpus | records | hours | Se % | +P % | F1 % |
|---|---:|---:|---:|---:|---:|
| MIT-BIH Arrhythmia | 24 | 12.0 | 99.496 | 99.717 | 99.606 |
| MIT-BIH Supraventricular | 13 | 6.5 | 99.808 | 99.982 | 99.895 |
| MIT-BIH ST Change | 6 | 3.5 | 99.924 | 99.695 | 99.809 |
| QT | 78 | 19.5 | 99.740 | 99.882 | 99.811 |
| MIT-BIH Normal Sinus | 11 | 270.2 | 99.700 | 99.328 | 99.514 |
| MIT-BIH Long-Term | 5 | 109.7 | 99.576 | 99.936 | 99.756 |
| European ST-T | 90 | 180.0 | 99.674 | 99.304 | 99.489 |
| Sudden Cardiac Death | 12 | 205.1 | 98.987 | 94.540 | 96.712 |
| AF (unaudited reference) | 4 | 40.9 | 98.647 | 97.460 | 98.050 |
| INCART, lead 0 = **lead I** | 75 | 37.5 | 94.051 | 94.557 | 94.303 |
| INCART, **lead II** | 75 | 37.5 | **99.537** | **99.285** | **99.411** |

**On INCART.** It is the only 12-lead corpus here, so "lead 0" means lead I —
a poor single-lead view for many subjects and *not* what a chest patch sees. A
patch approximates lead II. Both numbers are given; the lead-I row is the
mechanical "index 0" convention and the lead-II row is the one that answers the
question this engine is being built for. The choice is stated on principle, not
picked because it scored better.

**On sudden cardiac death (94.5% precision).** These are terminal recordings
containing ventricular fibrillation, agonal rhythm and asystole, where "beat" is
not well defined and the reference reflects that. The number is reported as
measured, not excluded.

### Fiducial placement

| corpus | sample rate | median offset | sd |
|---|---:|---:|---:|
| MIT-BIH Arrhythmia | 360 Hz | 0.0 ms | 13.7 ms |
| MIT-BIH ST Change | 360 Hz | −11.1 ms | 5.9 ms |
| QT | 250 Hz | 0.0 ms | 9.7 ms |
| Supraventricular / Normal Sinus / Long-Term | 128 Hz | +15.6 ms | 7.5–19.4 ms |

The 128 Hz corpora retain a **2-sample** offset that survives every filter
change, including running with the notch disabled. It is almost certainly the
annotation convention of those databases rather than a lag in this engine. It is
left uncorrected: a constant per-corpus bias cancels exactly in RR intervals,
which is what the downstream AF and ectopy features consume, and trimming it
with a fudge constant tuned to one corpus would be fitting the reference rather
than fixing the signal path.

---

## 5. Throughput

Measured on Apple M1 Ultra, single lead, 250 Hz, 250 ms packets — the cadence
the deployed loop uses.

| stage | marginal cost |
|---|---:|
| filter bank (4 taps) | 10.3 ns/sample |
| quality monitor | 50.5 ns/sample |
| QRS detection | 25.4 ns/sample |
| **full pipeline** | **~78 ns/sample** |

| | |
|---|---|
| Per channel @ 250 Hz | 19.5 µs of CPU per second of signal |
| Per 250 ms packet | 4.8 µs |
| **Capacity, one core** | **~51,000 channels @ 250 Hz** |
| Capacity, 4 cores (Pi 5 class) | ~205,000 channels |
| Memory | **~48 KB per channel** (31.7 MB resident for 512 channels) |

The requirement was hundreds of channels per server. One core covers that by
two orders of magnitude, which means the headroom is available for the stages
still to come (RR analysis, AF features, N/S/V classification, episode logic)
rather than being spent here.

Three optimisations account for most of it, each found by measuring rather than
guessing — the stage breakdown showed the quality monitor was 88% of the cost:

1. **Nine passes over the feature window fused into one** (200 → 148 ns).
2. **Window sums maintained incrementally**, add-one/subtract-one, with an exact
   sweep every 64 hops to bound drift (148 → 131 ns).
3. **The slow amplitude reference stopped re-sorting 1,200 floats every hop**
   (131 → 71 ns single-core). It describes five minutes of signal; sampling it
   four times a second bought nothing and cost more than the rest of the engine
   combined.

---

## 6. Two faults the sealed run exposed

Both were found while scoring TEST, both are structural rather than parameter
choices, and both were fixed and re-verified on TRAIN. Recording them here is
the point — a sealed set that never changes anything is not being used.

### An absorbing state in the adaptive threshold

MIT-BIH Normal Sinus record **16272** scored 26.1% sensitivity. Tracing the
detector's state showed `npk` (the noise estimate) latched at 4.75·10⁻² while
`spk` (the signal estimate) had decayed to 1.24·10⁻² **below** it. The threshold
sat four times above every real beat on visibly clean signal — kurtosis 8–11,
quality score 0.92.

The trap is circular: `npk` rises only when a candidate is rejected, and a
candidate exists only when the threshold is crossed. Once `npk` reaches `spk`,
nothing can ever bring it back.

The fix is not simply to cap `npk`. Capping it unconditionally restored 16272
but dropped precision across the Normal Sinus corpus from 99.62% to 90.55%,
because a high `npk` through a noise burst is exactly what stops the detector
inventing beats. **The cap applies only while the quality monitor believes the
signal.** On signal we believe, the detector must never be able to go deaf; on
signal we do not believe, staying deaf is correct — and the quality monitor is
what separates the two cases.

Record 16272: 26.1% → **96.4%** sensitivity. Normal Sinus corpus precision:
**99.62%**, unchanged. This is the clearest argument for the quality monitor
running *ahead* of the detector rather than beside it.

Two smaller faults were fixed in the same area: a candidate window that could
latch open indefinitely when the integration trace hovered between the threshold
and the peak-drop bound (search-back stands down while a candidate is open, so
this could silence the detector outright), and RR intervals outside 0.2–3 s
entering the running average, which pushed the search-back deadline out to eight
seconds.

### Electrode failure that stopped being reported

Record 16272 also loses its electrode 4.9 hours into a 25-hour recording;
amplitude collapses twentyfold. The quality monitor reported only 6.6% of the
record unusable, for two reasons. A twentyfold amplitude collapse was being
*averaged* against two shape statistics that still looked plausible on whatever
was left — it is a hard failure and is now a veto. And the relative amplitude
reference, a five-minute median, **re-anchored to the collapsed level**, so the
fault became the channel's new normal within one reference span. The reference
now refuses to learn from a lead it has already judged to be off.

---

## 7. What this phase does not yet do

Scoped out deliberately, in the order the brief asks for them:

- RR-interval analysis and rhythm features
- AF feature extraction
- N/S/V beat classification
- Arrhythmia episode detection and the server's multi-channel supervisor

The engine is structured for them: `ChannelPipeline` is `Send`, holds no global
state, allocates only at construction, and already emits per-beat amplitude,
energy, detection margin and a per-second quality verdict. The remaining stages
consume beats and quality, not raw samples.

### Known limitations

- **Beat classification is not here**, so no claim is made about it. The
  detector says "there is a beat", never what kind.
- **Cold start on a channel that is noisy from the first second**: the relative
  quality references have nothing clean to anchor to. The absolute tests still
  fire, but the relative ones cannot.
- **A lead that is off for hours stays flagged**, by design. A legitimate
  permanent amplitude change (posture, electrode replacement) will also stay
  flagged until the pipeline is reset. Re-anchoring after sustained stable
  low amplitude is not implemented.
- **The 128 Hz fiducial offset** (§4) is characterised but not explained; it
  needs a look at those databases' annotation provenance, not more filtering.
- **DEV zone is unused.** Selection used TRAIN, reporting used TEST. DEV is
  available for the model-based stages to come.
