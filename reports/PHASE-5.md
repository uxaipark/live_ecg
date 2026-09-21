# live_ecg — Phase 5: ventricular fibrillation

The most dangerous condition this engine is asked about, and the one it detects
least well. This report says so plainly and explains what it is nonetheless
useful for.

---

## 1. Why it needed a separate detector

Every stage below QRS detection assumes there are beats. In fibrillation there
are none: the trace is a continuous, roughly sinusoidal oscillation with no
isoelectric line, no P wave and no QRS to find. A beat detector fed with it
still produces detections — and the RR series, the AF features, the beat
classes and every rhythm episode built on those detections are then describing
an artefact.

So this detector reads the analysis signal directly and never asks what the
beats were. It runs *beside* the beat path rather than after it, which means it
keeps working in precisely the situation where the rest of the engine has
quietly stopped being meaningful.

---

## 2. There is no sealed test set, and that is not fixable here

Both corpora that carry fibrillation — the MIT-BIH Malignant Ventricular
Arrhythmia database and the Creighton University Ventricular Tachyarrhythmia
database, 57 records between them — are **entirely in the TRAIN zone** of the
subject-level split this project inherited.

So the split is positional and carved from TRAIN: every third record validates,
the rest fit. **Every figure below is a held-out-within-training estimate.** It
is not comparable to the sealed numbers in the other reports, and it should not
be quoted as though it were.

Ground truth differs by corpus and both forms are read: `vfdb` annotates rhythm
as spans, where `(VF` and `(VFL` count; `cudb` marks onset and offset with `[`
and `]` brackets instead. Ventricular *tachycardia* is not counted as
fibrillation — it is organised, it has beats, and the beat path already detects
it.

---

## 3. Result

Held-out third, 5.8 hours, 8.9% fibrillation:

| feature | AUC |
|---|---:|
| **combined score** | **0.889** |
| VF filter leakage | 0.883 |
| excess kurtosis (negated) | 0.838 |
| threshold-crossing sample count | 0.831 |
| peak-to-mean ratio (negated) | 0.828 |
| dominant frequency | 0.446 |
| relative amplitude | 0.399 |

At the shipped operating point: **sensitivity 85.6%, specificity 78.9%,
precision 28.4%.**

On 270 hours of normal sinus rhythm, specificity is **99.95%** — the detector
does leave ordinary rhythm alone, which is the property the guard in the test
suite pins.

The negative class here is not normal rhythm. These corpora are made of
malignant arrhythmia, so the 78.9% specificity is measured against ventricular
tachycardia, asystole and noise — the hard cases, not the easy ones. That makes
the number lower and more honest than the same metric computed against sinus
rhythm, which is what much of the published comparison does.

---

## 4. What it is, and is not, good enough for

**Not good enough to raise a fibrillation alarm on.** 28% precision means three
false alarms for every true one, and there is no threshold that fixes it:
raising it to 0.8 buys 44% precision at 53% sensitivity, which is worse in the
direction that matters for a lethal rhythm.

**Good enough for the job the rest of the engine needs**, which is different:
flagging that the beat path's assumptions have failed. Over-calling "do not
trust the beats here" costs a suppressed conclusion. Over-calling "this patient
is in fibrillation" costs something else entirely. `ChannelPipeline::in_vf()` is
meant to be read in the first sense, and the operating point was chosen for it —
high enough that normal rhythm is untouched, low enough to catch most
fibrillation.

---

## 5. The model is not the limit

A gradient-boosted ensemble was fitted over the same features and measured
against the linear model on the same held-out records: **AUC 0.876 against
0.889.** Not better. It is not shipped, and the fitting path stays so the
comparison can be repeated.

That is worth stating because trees *were* the answer for beat classification,
where they took the ventricular detector from 0.956 to 0.977. Here they are not,
which says the six features are what bounds this — not the model's ability to
combine them.

What would plausibly move it: spectral concentration, a complexity measure, and
phase-space or Hilbert-transform features, all of which the VF literature uses
and none of which are implemented. More training data would help too, and there
is none to be had from these corpora without giving up the held-out third.

---

## 6. Limitations

- **No sealed evaluation** (§2). The honest ceiling on confidence here is lower
  than anywhere else in this project.
- **Atrial flutter, junctional and idioventricular rhythms** are annotated in
  these corpora and have no detector.
- **The beat path is not yet suppressed during fibrillation.** `in_vf()` is
  exposed and nothing consumes it. Wiring it — so that RR, AF and beat-class
  conclusions are withheld while the flag is raised — is the obvious next step
  and is not done.
- **Transitions are smeared.** Features are measured over a four-second window
  and scored against per-second labels, so onset and offset are approximate by
  construction. At 8.9% prevalence that costs more than it would at 50%.
