# live_ecg — detection and classification performance

Every number here is copied from `reports/results/`, regenerated in one run by
`tools/run_evaluation.sh`. Nothing is quoted from memory and nothing is rounded
in this engine's favour.

**Zones.** TRAIN selects, TEST scores, and the two are never mixed: no threshold
or coefficient in the engine was chosen by looking at a TEST result. Where a
corpus has no sealed half — the fibrillation corpora, BUT QDB — that is stated
on the line, because it changes what the number means.

**Lead.** INCART is twelve-lead. A chest patch approximates lead II, so INCART is
reported in lead II and never pooled with the single-lead corpora: one `--lead`
cannot be right for three corpora at once, and pooling it at lead I produces a
figure about the lead choice rather than about the engine.

---

## 1. QRS detection — sealed, 318 records, 885 hours

| corpus | records | hours | Se % | +P % |
|---|---:|---:|---:|---:|
| MIT-BIH Arrhythmia | 24 | 12.0 | 99.50 | 99.72 |
| Supraventricular | 13 | 6.5 | 99.81 | 99.98 |
| Normal Sinus | 11 | 270.2 | 99.73 | 99.31 |
| European ST-T | 90 | 180.0 | 99.68 | 99.30 |
| Long-Term | 5 | 109.7 | 99.58 | 99.94 |
| QT | 78 | 19.5 | 99.74 | 99.88 |
| ST Change | 6 | 3.5 | 99.92 | 99.69 |
| INCART, lead II | 75 | 37.5 | 99.54 | 99.29 |
| INCART, lead I | 75 | 37.5 | 94.04 | 94.56 |
| Atrial Fibrillation | 4 | 40.9 | 98.65 | 97.46 |
| Sudden Death | 12 | 205.1 | 99.00 | 94.52 |
| **pooled** | **318** | **884.8** | **99.20** | **97.96** |

The pooled figure is dragged by the last two, and deliberately so: the sudden-death
corpus contains ventricular fibrillation, where there are no beats to detect, and
its 94.5 % precision is the engine reporting detections in a rhythm that has none.
Fibrillation suppression acts on that downstream (§5), not here.

---

## 2. Beat classification (ANSI/AAMI EC57)

Figures are over the N/S/V population, which is what the inter-patient literature
reports; fusion is scored separately over N/S/V/F because a class cannot be
scored against a population it is excluded from.

### 2a. Classification isolated from detection — driven at the reference beat positions

| corpus | beats | cov % | V Se/+P | S Se/+P | F Se/+P | AUC V / S / F |
|---|---:|---:|---|---|---|---|
| MIT-BIH | 47,551 | 98.5 | **95.78 / 80.83** | 29.16 / 18.30 | 22.22 / 33.21 | 0.9937 / 0.7456 / 0.8328 |
| Supraventricular | 27,015 | 99.4 | **90.09 / 83.62** | 89.53 / 68.28 | 5.26 / 5.00 | 0.9934 / 0.9868 / 0.7041 |
| INCART lead II | 175,778 | 96.5 | **88.14 / 88.63** | 91.29 / 23.15 | 13.33 / 6.15 | 0.9754 / 0.9872 / 0.9023 |

### 2b. End to end — our own detector, our own classifier

| corpus | beats | missed / spurious | V Se/+P | S Se/+P | F Se/+P | AUC V / S / F |
|---|---:|---|---|---|---|---|
| MIT-BIH + Supraventricular | 74,255 | 350 / 106 | **94.20 / 74.71** | 52.90 / 36.01 | 23.87 / 36.12 | 0.9918 / 0.8542 / 0.8524 |
| INCART lead II | 174,975 | 892 / 1,291 | **86.22 / 85.01** | 95.11 / 22.85 | 27.54 / 10.65 | 0.9721 / 0.9840 / 0.9118 |

### 2c. Confusion, MIT-BIH sealed (reference positions)

| reference ↓ / reported → | N | S | V | F | unclassified |
|---|---:|---:|---:|---:|---:|
| **N** | 38,539 | 2,262 | 602 | 123 | 652 |
| **S** | 1,131 | 515 | 120 | 0 | 12 |
| **V** | 46 | 38 | 3,044 | 50 | 23 |
| **F** | 276 | 1 | 24 | 86 | 0 |
| **Q** | 2 | 0 | 2 | 0 | 3 |

The supraventricular class is the weak one, and the matrix says why: 2,262 normal
beats are called supraventricular against 515 that really are. In one lead an
atrial premature beat is identified by prematurity plus a P wave that is usually
buried in the preceding T wave.

---

## 3. Atrial fibrillation

| | scored | Se % | Sp % | +P % | F1 | episodes ≥30 s | burden error |
|---|---:|---:|---:|---:|---:|---|---|
| AFDB TEST, end to end | 40.8 h | **90.52** | 99.21 | 98.81 | **94.48** | 33 / 34 found, 144 / 145 correct | 3.53 pp |
| AFDB TEST, rhythm logic isolated | 40.8 h | 89.11 | 99.36 | 99.02 | 93.81 | 33 / 34, 146 / 147 | 4.21 pp |
| AFDB DEV | 29.7 h | 89.40 | 97.89 | 78.68 | 83.70 | 42 / 43, 49 / 65 | 1.64 pp |

Per-record median F1 on the sealed half is 92.7.

### False alarms on rhythm that contains none

| | AF-free signal | alarms | per 24 h |
|---|---:|---:|---:|
| Normal Sinus TEST | 270.2 h | 25 | **2.22** |
| Normal Sinus TRAIN | 143.5 h | 3 | **0.50** |

Worst single subject on the sealed half: 14.6 per 24 h. That subject has
respiratory sinus arrhythmia marked enough to be as irregular as fibrillation,
and the atrial-coherence feature is what brought it down from 52.7.

---

## 4. Rhythm episodes

Two references, because there are two questions. **Against the rule on reference
beats** asks whether our detection and classification reproduce a rule that is
already a definition — a pause is an interval over two seconds. That is the
question this engine can be wrong about. Against the annotator's own rhythm spans
is a different question and mostly a definitional disagreement.

### MIT-BIH, sealed, 24 records, 12 hours

| condition | prevalence | Se % | Sp % | +P % | episodes found | reported correct |
|---|---:|---:|---:|---:|---|---|
| pause | 0.60 % | **100.00** | 99.97 | 95.57 | 92 / 92 | 92 / 98 |
| asystole | 0.07 % | **100.00** | 100.00 | **100.00** | 14 / 14 | 14 / 14 |
| bradycardia | 3.90 % | 99.53 | 99.82 | 95.68 | 40 / 40 | 40 / 42 |
| tachycardia | 8.74 % | 97.57 | 100.00 | **100.00** | 14 / 14 | 22 / 22 |
| bigeminy | 1.60 % | 75.58 | 99.96 | 96.67 | 22 / 26 | 26 / 28 |
| ventricular run | 0.10 % | 65.91 | 99.33 | **9.15** | 15 / 20 | 15 / 125 |
| ventricular tachycardia | 0.09 % | 50.00 | 99.75 | **15.08** | 10 / 17 | 10 / 52 |
| idioventricular rhythm | 0.01 % | 100.00 | 99.60 | **3.31** | 3 / 3 | 3 / 73 |

### Long-Term AF, 84 records, 1,961 hours (development corpus, not sealed)

| condition | prevalence | Se % | Sp % | +P % | episodes found |
|---|---:|---:|---:|---:|---|
| bradycardia | 3.85 % | 98.95 | 99.87 | 96.71 | 3,639 / 3,691 |
| tachycardia | 13.61 % | 92.95 | 99.23 | 94.98 | 7,909 / 8,836 |
| pause | 0.17 % | 96.37 | 99.53 | 25.50 | 5,500 / 5,666 |
| bigeminy | 0.06 % | 29.55 | 99.99 | 70.58 | 76 / 278 |
| trigeminy | 0.02 % | 34.55 | 100.00 | 64.65 | 33 / 119 |
| idioventricular rhythm | 0.01 % | 38.70 | 99.91 | **5.51** | 161 / 346 |
| ventricular run | 0.03 % | 43.72 | 99.61 | **3.65** | 456 / 943 |
| ventricular tachycardia | 0.02 % | 44.32 | 99.75 | **3.51** | 280 / 593 |
| asystole | 0.004 % | 9.54 | 99.88 | **0.35** | 16 / 153 |

### Normal Sinus TEST, 270 hours — what fires where nothing should

| condition | reported episodes | correct |
|---|---:|---:|
| ventricular run | 103 | 0 |
| ventricular tachycardia | 73 | 0 |
| idioventricular rhythm | 14 | 0 |
| bradycardia | 235 | 234 |
| tachycardia | 414 | 401 |

Rate conditions are near-perfect on healthy subjects; every morphology-dependent
condition has a false-positive floor set by the ventricular detector's precision
on long recordings. That is the single binding limitation in this table.

---

## 5. Ventricular fibrillation

| | windows | Se % | Sp % | +P % | AUC |
|---|---:|---:|---:|---:|---:|
| Held out within TRAIN, model fitted on 3/4 | 17,127 | 83.57 | 87.07 | 32.26 | **0.9014** |
| — same fit without the phase-space feature | 17,127 | 80.94 | 84.55 | 27.85 | 0.8966 |
| `vf_heldout.txt` (in sample — see note) | 20,739 | 79.87 | 89.55 | 42.78 | 0.9190 |
| Normal Sinus TEST, 270 h | 972,751 | — | **100.00** | — | — |

There is no sealed fibrillation corpus, so the shipped model is fitted on all of
TRAIN and `vf_heldout.txt` scores it on a third of the same records. That file is
a fair comparison of two shipped models against each other and **is not an
estimate of field performance**; the first two rows, fitted on three quarters and
scored on the held-out quarter, are.

Precision near 30 % is honest for this detector and is why fibrillation drives a
*flag* rather than an alarm, with a separate and higher bar (0.75) before it is
allowed to withhold beat-derived findings.

---

## 6. Signal quality

### Against four human annotators — BUT QDB

| | seconds | unanimous | AUC, class 3 vs 1 | AUC, class 2 vs 1 |
|---|---:|---:|---:|---:|
| sealed half | 159,548 | 80.9 % | **0.9964** | quality score 0.427 |
| | | | | atrial coherence **0.844** |
| development half | 198,256 | 60.2 % | 0.8520 | quality score 0.450 |
| | | | | atrial coherence **0.818** |

"Unusable" is separated from "full diagnostic quality" almost perfectly. The
distinction between *full diagnostic quality* and *QRS reliable only* is about
the P and T waves, which the quality monitor does not measure — it scores 0.43
there, worse than chance — and is reported instead by the wave-legibility output,
which scores 0.84.

### Against known noise — MIT noise-stress protocol

| SNR | AUC vs noise | AUC vs detector error | QRS +P, ungated | QRS +P, gated |
|---|---:|---:|---:|---:|
| −6 dB | 0.9928 | 0.9710 | 65.0 % | **92.2 %** |
| 0 dB | 0.9900 | 0.9409 | 73.6 % | **85.1 %** |
| 6 dB | 0.9224 | 0.8300 | 86.5 % | 87.0 % |
| 12 dB | 0.8014 | 0.7670 | 98.1 % | 98.1 % |
| 18 dB | 0.6473 | 0.4881 | 99.9 % | 99.9 % |
| 24 dB | 0.5230 | 0.4992 | 100.0 % | 100.0 % |

The monitor predicts detector error (AUC 0.96 at −6 dB) better than it predicts
the noise itself, which is the property that matters: the point is not to notice
noise but to know when to stop believing the output.

---

## 7. Wave delineation

Median error and the share inside the tolerance that expert annotators disagree
by (CSE working party, two standard deviations).

| mark | LUDB sealed | | | QT (independent) | | | CSE 2SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| | found % | median | in tol % | found % | median | in tol % | |
| P onset | 94.2 | 0 ms | 59.8 | 77.8 | +4 ms | 33.6 | 10.2 ms |
| P peak | 93.9 | −2 ms | 86.5 | 74.4 | +4 ms | 58.5 | 10.2 ms |
| P offset | 93.7 | −16 ms | 35.2 | 73.7 | −4 ms | 49.3 | 12.7 ms |
| QRS onset | **100.0** | **0 ms** | 67.4 | **100.0** | **0 ms** | 38.9 | 6.5 ms |
| QRS offset | **100.0** | **0 ms** | 54.4 | **100.0** | **0 ms** | 33.8 | 11.6 ms |
| T peak | 98.2 | −12 ms | 82.6 | 86.2 | −12 ms | 65.5 | 30.6 ms |
| T offset | 95.5 | −26 ms | 49.6 | 89.7 | −28 ms | 28.9 | 30.6 ms |

QT was never used to fit anything here. Its medians hold and its spread widens,
which is what a corpus sampled at 250 Hz — 4 ms per sample — should do against a
6.5 ms tolerance.

---

## 8. Cost

| | ns / sample | channels, 1 core @ 250 Hz |
|---|---:|---:|
| 4 shards | 182.3 | 21,945 |
| 20 shards | 239.9 | 16,671 |

Per stage, single core: filter bank 10.8, quality monitor 51.2, everything
downstream of it 128.5 ns/sample. Per-shard cost rises with shard count because
the limit is memory bandwidth, not cores.

Measured: 1,000 channels at 250 Hz cost **3.3 % of one core** and 110 MB on this
workstation; at 500 Hz, 6.0 % and 173 MB. A Raspberry Pi 5 is estimated at three
to six times slower — that estimate has not been measured on the device.

---

## 9. What this table does not say

- **Ventricular precision on long ambulatory recordings** is the single binding
  limitation. It sets a floor under three conditions at once: ventricular runs
  (3.7 %), ventricular tachycardia (3.5 %) and idioventricular rhythm (5.5 %) on
  1,961 hours.
- **Supraventricular detection** is 0.75 AUC on MIT-BIH and 0.99 on the two
  corpora built around supraventricular ectopy. The difference is not the
  detector; it is that MIT-BIH's atrial beats are the hard ones.
- **Fusion** has a usable ranking (AUC 0.83–0.91) and no usable operating point.
- **Atrial flutter and junctional rhythm are absent**, having been built,
  measured and removed. See `PHASE-10.md` §7 and §8.
- **No corpus here is a wearable patch.** Every record was taken with clinical
  electrodes, mostly on inpatients. The engine is designed for single-lead patch
  data and has never been measured on any.
