# live_ecg — Phase 3: beat classification (N / S / V)

Builds on [`PHASE-1.md`](PHASE-1.md) (filter bank, noise, QRS) and
[`PHASE-2.md`](PHASE-2.md) (RR, AF). Adds per-beat morphology features and a
bank of independent binary detectors.

Regenerate with `tools/run_evaluation.sh`; raw output in `reports/results/`.

---

## 1. Architecture: a bank of binary detectors, not one multi-class model

The conditions this engine reports do not share an evidence base or a cost
function. AF is decided from timing alone; a ventricular beat from morphology; a
pause from an interval; ventricular tachycardia from all three plus duration.

The decisive reason is the **operating point**. A missed ventricular run can
kill; a false premature atrial beat costs a reviewer ten seconds. Those call for
thresholds on opposite sides, and a single softmax with one `argmax` has nowhere
to put that asymmetry. Separate detectors each carry their own threshold, each
can be revalidated or replaced alone, and each publishes its own score so the
episode layer can weigh ambiguity instead of inheriting a decision already
collapsed to a label.

**The measurements confirm the premise rather than merely permitting it.** Each
detector's top features are almost disjoint:

| | strongest features (TRAIN AUC) |
|---|---|
| **Ventricular** | `ncc_template` 0.92, `rr_prev_rel` 0.92, `ncc_prev` 0.89 — *morphology* |
| **Supraventricular** | `rr_prev_rel` 0.85, `rr_ratio` 0.82, `rr_sum_rel` 0.71 — *timing* |

`ncc_template` is worth 0.92 to the ventricular question and 0.64 to the
supraventricular one. Forcing both through one shared representation would
average that away.

**What is not separate is the feature layer.** Morphology and interval context
are computed once per beat and shared; the bank costs one extraction plus one
model evaluation per detector.

**Where exclusivity is real.** One beat has one AAMI label, and EC57 wants a
confusion matrix over them. So the bank publishes both the per-detector scores
*and* an arbitrated label. The arbitration rule is explicit — the detector
clearing its own threshold by the larger relative margin wins, ventricular takes
an exact tie — rather than implied by an `argmax` no one can inspect.

---

## 2. Headline

Sealed TEST. Lead 0 throughout except where noted.

| | reference beats | end to end |
|---|---|---|
| **VEB (V)** sensitivity / precision | **85.2% / 69.2%** | 78.9% / 58.5% |
| **SVEB (S)** sensitivity / precision | 62.0% / 29.8% | 61.2% / 24.6% |
| ventricular detector AUC | 0.967 | 0.945 |
| supraventricular detector AUC | 0.888 | 0.885 |
| coverage (beats classified, not deferred) | 87.1% | 83.6% |

Per corpus, classification isolated from detection:

| corpus | lead | VEB Se / +P | SVEB Se / +P | AUC V / S |
|---|---|---|---|---|
| MIT-BIH Arrhythmia | 0 | 80.0 / 63.6 | 27.1 / 15.2 | 0.979 / **0.722** |
| MIT-BIH Supraventricular | 0 | 88.2 / 80.1 | 90.1 / 75.8 | 0.995 / 0.988 |
| INCART | 0 (= lead I) | 86.3 / 70.1 | 74.9 / 25.2 | 0.960 / 0.980 |
| INCART | 1 (= **lead II**) | **86.1 / 86.7** | 89.1 / 24.3 | 0.976 / 0.985 |

INCART is the only 12-lead corpus, so "lead 0" is lead I — not what a chest
patch sees. On lead II, which a patch approximates, ventricular precision goes
from 70% to 87%. Same protocol decision as Phase 1 §4, stated on principle
rather than picked for the number.

---

## 3. A linear model could not do this, and trees could

The first bank used logistic regression over the same ten features. It reached
an AUC of 0.956 for the ventricular detector and **could not convert that into
usable precision**: 53% sensitivity at 35% precision.

The reason is structural. A ventricular beat is abnormal in shape **and** early
**and** followed by a compensatory pause. A supraventricular beat is early with
*normal* shape and *no* compensatory pause. Those are conjunctions, and a
weighted sum cannot represent a conjunction — the fitted linear model leaned on
the RR features and gave `width_rel` a coefficient of −0.06 despite its
standalone AUC of 0.74.

Replacing each detector's model with a small gradient-boosted ensemble — depth 4,
120 trees, ~3,500 nodes, about 12 KB — moved the ventricular detector from
AUC 0.956 to 0.977 and sensitivity from 53% to 88% on the same validation
records. Both model forms are kept in the code so the choice stays a measured
one.

This is still a small model that runs anywhere without an accelerator, which was
the constraint. Depth 6 and 300 trees were also measured and bought nothing
(VEB 84.4/58.1 against 87.9/59.9), so the capacity is saturated, not starved —
worth knowing before anyone reaches for a deep network.

---

## 4. Four defects the measurements found

### QRS width was zero for a quarter of all beats

`width_rel` and `amp_rel` read **0.000 at both the 5th and 25th percentile of
every class**. The QRS-band tap is band-passed and therefore lags the fiducial,
which is placed on the analysis signal and already delay-compensated. Walking
outward from the R peak to find the complex boundaries started at a point where
the band signal had not risen yet, and the complex collapsed to a single sample.

Anchoring the boundary search on the band's own local peak fixed it. Ventricular
AUC went from 0.916 to 0.956 on that change alone, and the feature distributions
became physiological: width 0.99 for normal beats against 1.46 for ventricular.

### Every tree refused to split

The first ensemble reported "120 nodes, 120 trees" — each tree a bare leaf. The
class weights were normalised to sum to one, so the total Hessian mass of the
whole dataset was about 0.25, while `min_child_weight` was 8. No split could
ever pass. Both are in units of Hessian mass; putting the weights on a
sample-count scale fixed it.

### One negative annotation silently deleted seven records

Seven INCART records produced **zero** verdicts — 17,896 beats, 10% of the
corpus. WFDB permits an annotation before the first sample, and record I35 opens
with one at −15. Casting that to `usize` wrapped it to a huge number, the
equality test that advanced the reference cursor never matched again, and the
entire record was skipped in silence.

Two fixes: pre-record annotations are dropped rather than clamped (a beat whose
morphology was never recorded should not be scored as one we failed to
classify), and the cursor advances on `<=` so it cannot stall. INCART went from
17,896 unseen beats to 89.

### A bare CLI flag swallowed the option after it

`--holdout-take --v-thr 0.9` parsed as `holdout-take = "--v-thr"`, and the
threshold was discarded. It presented as the threshold having no effect at all —
identical results across the whole sweep — which is indistinguishable from a
model whose scores are saturated. Flags whose next token is another flag are now
boolean.

---

## 5. Supraventricular detection is the weak class, and the reason is specific

Two different problems wear the same number.

**On MIT-BIH the discrimination itself is poor** — AUC 0.722, against 0.988 on
the Supraventricular database and 0.985 on INCART. A single lead sees atrial
activity badly, and MIT-BIH's atrial premature beats overlap heavily with
ordinary sinus variation. This is a real limit, not a tuning problem.

**On INCART the discrimination is excellent and the threshold is wrong** —
AUC 0.985 but precision 24%. Supraventricular beats are 1.2% of that corpus, and
at that prevalence a specificity of 97% still yields three false positives for
every true one. Precision at a fixed threshold is a function of prevalence, and
the populations differ.

That second case is exactly what the per-detector threshold exists for, and it
argues for calibrating it per deployment rather than shipping one number. The
frozen thresholds — ventricular 0.85, supraventricular 0.93, deliberately
asymmetric — were chosen on held-out TRAIN records and are the right shape, but
the curve there was flat (VEB 87.9/59.9 at 0.5 against 87.1/66.1 at 0.97), so
they are not strongly evidenced. The knob matters more than this corpus shows.

---

## 6. Protocol

| | |
|---|---|
| **TRAIN** — fits both models | MIT-BIH Arrhythmia 21, Supraventricular 65 (194,503 beats) |
| internal validation | every 5th TRAIN record, for model form and capacity |
| **TEST** — scored once | MIT-BIH 21, Supraventricular 13, INCART 75 |

Paced records (102, 104, 107, 217) are excluded as EC57 requires, and the
harness names them. Fusion and paced/unclassifiable beats appear in the
confusion matrix but are outside the Se/+P denominators, which is what the
inter-patient literature this will be compared against does; the full matrix is
printed so a reader can undo that choice.

Hyperparameters were selected on the internal split and the models then refitted
on all of TRAIN. TEST was scored afterwards.

**Coverage is reported, not hidden.** 13% of beats are returned as `Unknown` —
poor signal quality, or no dominant template established yet. A wrong label is
worse than an absent one, and a classifier that quietly guesses on noise would
show a better confusion matrix and be less useful.

---

## 7. Cost

| stage | marginal cost |
|---|---:|
| filter bank | 10.2 ns/sample |
| quality monitor | 50.4 ns/sample |
| QRS + RR + AF + beat classification | 55.8 ns/sample |
| **full pipeline** | **~117 ns/sample single core** |

| | Phase 1 | Phase 2 | Phase 3 |
|---|---:|---:|---:|
| Capacity, one core @ 250 Hz | ~51,000 | ~32,700 | **~28,700 channels** |

Two tree ensembles of ~3,500 nodes each cost about 15 ns/sample, and they run
once per beat rather than per sample. Still two orders of magnitude beyond the
stated requirement of hundreds of channels per server, and the models are static
tables that link into the binary with no loader and no allocation — the same
code on the server and on the patch.

---

## 8. Limitations

- **SVEB on MIT-BIH is weak** (Se 27%, AUC 0.72) and this is the class the field
  struggles with. A second lead would help most; from one lead, P-wave evidence
  is the missing ingredient and delineation is not implemented.
- **Thresholds are prevalence-sensitive** (§5). They should be set per
  deployment population, not inherited from this corpus.
- **Fusion beats are not a class.** They are measured and reported in the
  matrix — 429 of 625 land in N — but no detector targets them.
- **Ectopy is still not fed back into AF.** The stated Phase 3 payoff for Phase 2
  was removing ectopic beats from the RR series before the AF features are
  computed. The classifier now exists; the wiring does not. Phase 2 §5 predicted
  this would reduce the ectopy-driven share of AF false alarms and not the
  respiratory-sinus-arrhythmia share, and that prediction is still untested.
- **Classification lags detection by one beat.** The interval *after* a beat is
  part of the evidence for what it was. Inherent to the evidence, not the
  implementation.
- **No internal DEV zone exists for these corpora**, so the validation split is
  carved out of TRAIN by position. For MIT-BIH that is 5 records, and one of them
  (207, which carries both bundle-branch morphologies and ventricular flutter)
  dominated the error. Conclusions drawn from it are weakly evidenced.

---

## 9. Next

- Wire beat classes back into the AF path and re-measure Phase 2's false-alarm
  rate — the one concrete prediction this phase left open.
- Arrhythmia episode detection: pause and asystole, bradycardia and tachycardia,
  ventricular runs and VT/VF, bigeminy and trigeminy. These consume beat classes
  and quality, and each is another binary detector in the same bank.
- The multi-channel server supervisor.
