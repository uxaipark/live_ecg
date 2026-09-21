# live_ecg — Phase 2: RR intervals and atrial fibrillation

Builds on [`PHASE-1.md`](PHASE-1.md) (filter bank, noise detection, QRS
detection). Adds the RR interval stream, AF feature extraction, an AF detector
and episode confirmation. Rule-based statistics with an eleven-coefficient
logistic model on top; no deep learning, no accelerator.

Regenerate with `tools/run_evaluation.sh`; raw output in `reports/results/`.

---

## 1. Headline

| | reference beats | end to end |
|---|---|---|
| **TEST** (AFDB, 4 records, 40.8 h, 42% AF) | | |
| duration-weighted sensitivity | 83.0% | **86.2%** |
| specificity | 99.36% | **99.26%** |
| PPV | 98.94% | **98.83%** |
| episodes found (≥30 s) | 33 / 34 | **33 / 34** |
| AF burden error | 6.8 pp | **5.4 pp** |
| **False alarms** (Normal Sinus, TEST, 270 h, no AF) | | |
| median subject | — | **0.00 per 24 h** |
| mean | — | 6.48 per 24 h |

**The end-to-end path beats the reference-beat path.** Running on our own
detections gives 86.2% sensitivity against 83.0% on AFDB's own `.qrs`
annotations. Those annotations are machine-generated and unaudited (noted in
Phase 1 §1), so this is not a paradox — it says our detector's RR series is the
cleaner of the two.

**Six of eleven Normal Sinus subjects produce zero false alarms.** One subject
produces 72% of all of them. That distribution, not the mean, is the finding;
§5 explains why.

---

## 2. Protocol

Same subject-level zones as Phase 1, and the three roles are kept apart:

| | |
|---|---|
| **TRAIN** — fits the model | AFDB 18, LTAFDB 84, Normal Sinus 6 (106 records, 640k windows) |
| **DEV** — sets the operating point | AFDB 3 records, 29.7 h, 8% AF |
| **TEST** — scored once, after freezing | AFDB 4 records, 40.8 h; Normal Sinus 11 records, 270 h |

Three questions are answered separately, because pooling them hides which part
of the engine owns a number:

1. **Does the rhythm logic work?** Feed it the corpus's own beat annotations.
   Detector errors cannot reach the result.
2. **Does the product work?** Feed it our detections through the real
   `ChannelPipeline`, including the quality gate.
3. **How often does it cry wolf?** Alarm rate on 24-hour Holters with no AF at
   all. Sensitivity is easy; an AF detector that fires on normal sinus rhythm is
   unusable whatever its sensitivity.

**Atrial flutter is excluded from scoring, not counted either way.** Flutter is
organised, so its RR series can be perfectly regular; an RR-only detector has no
way to call it AF and no reason to. `--flutter af` gives the pooled convention
many published AFDB results use, and the report says which one it is.

**The scored state is confirmed episodes, not raw window verdicts.** Clinical AF
is a sustained finding; a per-window metric would flatter the detector by
counting bursts it would never report. That costs latency — an episode cannot be
confirmed before it has lasted 30 s — and the confirmed episode is reported with
its true start, so the record is right even though the alarm is late.

---

## 3. What the features are, and why each one

Every feature is dimensionless and computed over a sliding window of 20 usable
RR intervals, so none of them depends on heart rate or sampling rate. No
interval is ever interpolated or repaired: a fabricated interval is
indistinguishable from a measured one downstream, and AF detection is precisely
a judgement about how irregular the measured intervals are. Unusable intervals
are dropped, and a window that reaches across a blind stretch gets no vote.

Per-feature AUC on TRAIN, oriented so larger means more likely AF:

| feature | AUC | what it answers |
|---|---:|---|
| `cosen` | 0.978 | Coefficient of sample entropy. Does a matching pair of intervals stay matched one beat later? |
| `frac_near_median` | 0.946 | Is there still a dominant interval? Sinus with ectopy keeps one; AF has none. |
| `pnn_norm` | 0.944 | Share of successive differences above 5% of the median. |
| `iqr_norm` | 0.919 | Interquartile spread over the median. |
| `mad_norm` | 0.912 | Median absolute successive difference — ignores a handful of outliers. |
| `shannon_drr` | 0.877 | Entropy of the difference histogram. |
| `rmssd_norm`, `sd1_norm` | 0.872 | Classic dispersion of successive differences. |
| `rr_acf1` | 0.673 | Is the series *smooth*? Respiratory variation is correlated beat to beat; AF is not. |
| `tpr` | 0.692 | Turning-point ratio: an independent random series gives 2/3. |
| `drr_acf1` | 0.604 | Ectopy alternates short-long, so its differences alternate in sign. |
| `rmssd_over_mad` | 0.500 | Is the irregularity diffuse or carried by a few beats? |

Two things imitate AF, and each has its own answer. **Frequent ectopy** makes RR
irregular too, but sparsely and with structure — `rmssd_over_mad`, `drr_acf1`
and `frac_near_median` test that. **Respiratory sinus arrhythmia** is smooth and
correlated between neighbouring intervals — `rr_acf1` and `tpr` test that.

---

## 4. Four defects the measurements found

### The training negatives were the wrong population

Fitted on AFDB and LTAFDB alone, the negative class is almost entirely
AF-adjacent rhythm in AF patients. That is not what the model will meet: a patch
spends its life on sinus rhythm with occasional ectopy. The first model raised
**17 false alarms per 24 h** of normal sinus signal. Adding the Normal Sinus
TRAIN records to the fit is what fixed it — making the negative class
representative, not making the model bigger.

### Sample entropy was rate-dependent, and night-time is where it broke

`cosen` is the strongest feature by a wide margin, and it used the fixed 30 ms
tolerance from the paper that introduced it. At 110 bpm that is 5.5% of the RR
interval; at 54 bpm it is 2.7%. So on a sleeping subject almost no template
matches, sample entropy rises, and ordinary bradycardia with deep respiratory
variation reads as fibrillation.

The false-alarm windows on one Normal Sinus record made this visible directly:
median heart rate **54 bpm**, `cosen` **−0.05** against **−1.87** for the same
record's correctly-rejected windows — higher than genuine AF's −0.52. Scaling
the tolerance with the median interval (`max(20 ms, 4% of median)`) keeps the
question the same one at every heart rate.

### Collinear features with a weak penalty

The fitted weights had `mad_norm` at −8.2 against `rmssd_norm` at +1.6 — two
statistics measuring nearly the same quantity, with large cancelling
coefficients. That is textbook multicollinearity under a near-zero penalty, and
it does not survive a new subject.

Raising L2 from 1e-4 to 0.015 was the single largest improvement in the whole
phase:

| L2 | worst subject, alarms/24 h | DEV sensitivity | DEV episodes |
|---:|---:|---:|---:|
| 0.0001 | 29.1 | 89.0% | 42/43 |
| 0.005 | 11.4 | 87.8% | 42/43 |
| 0.010 | 7.3 | 84.6% | 42/43 |
| **0.015** | **3.1** | **81.7%** | **41/43** |
| 0.030 | 1.0 | 70.2% | 40/43 |

Chosen by an explicit criterion — the lowest false-alarm rate that still finds at
least 41 of 43 reference episodes — not by maximising an accuracy figure.

### The pipeline wiring was never actually connected

The end-to-end path reported zero AF windows on every record. Not a gating
effect: an earlier edit to `ChannelPipeline::push` had silently failed to match
after reformatting, so `out.intervals` and `out.af` were never populated. The
first "end-to-end" numbers came from the harness rebuilding the RR stream from
beats with quality forced on — which was not the product path. Worth recording
because it looked exactly like a real result: a plausible number in one place and
a suspicious zero in another.

---

## 5. The false-alarm distribution is the finding

Normal Sinus, TEST zone, 270 hours, no AF anywhere, end to end:

| subject | alarms / 24 h |
|---|---:|
| 16539 | 52.7 |
| 16272 | 9.6 |
| 16273 | 3.9 |
| 16795 | 3.1 |
| 16773 | 2.0 |
| **six others** | **0.0** |

Median 0.00, mean 6.48. **One subject accounts for 72% of all false alarms.**

The cause is not ectopy — these records carry 0.02% ectopic beats. It is marked
respiratory sinus arrhythmia, which is common in the young healthy subjects this
corpus is made of, and which an RR-only detector has genuinely limited means to
distinguish from AF. `rr_acf1` separates them in the data (AF median 0.004
against 0.227 for the false-alarm windows) but not cleanly enough on its own.

Reporting the mean alone would have hidden this. It also means the mean is the
wrong number to quote: a deployment sees per-subject rates, and most subjects
see none.

The same heavy tail exists in TRAIN (one record at 23.9 alarms/24 h against
0–1 for the rest), so the fix was developed and validated without touching TEST.

---

## 6. Cost

| stage | marginal cost |
|---|---:|
| filter bank | 10.6 ns/sample |
| quality monitor | 52.1 ns/sample |
| QRS + RR + AF | 40.6 ns/sample |
| **full pipeline** | **~103 ns/sample single core** |

| | Phase 1 | Phase 2 |
|---|---:|---:|
| Capacity, one core @ 250 Hz | ~51,000 channels | **~32,700 channels** |
| Memory per channel | ~48 KB | ~49 KB |

Rhythm analysis costs about 16 ns/sample, almost all of it the O(N²) sample
entropy, which runs once per beat rather than per sample. Capacity is still two
orders of magnitude beyond the stated requirement of hundreds of channels.

---

## 7. Limitations

- **A second operating point exists and may be the right one.** At
  `enter_prob = 0.92` the detector finds 42 of 43 DEV episodes and reaches 89%
  duration sensitivity at 1.8 false alarms per 24 h, against 0.50 at the frozen
  setting. Which is correct depends on whether a reviewer sees every alarm.
- **AF burden is under-reported by about 5 percentage points**, consistently
  negative. Episode confirmation deliberately drops everything shorter than 30 s
  and the window needs 20 intervals to fill, so short runs are missed by
  construction.
- **Episode counts fragment.** 149 reported episodes against 34 reference ones on
  TEST, at 98.7% precision — a long episode is split by momentary dips rather
  than false episodes being invented. AF burden, which is what gets reported
  clinically, is unaffected; a presentation-level merge would fix the count, and
  lengthening the bridge trades precision away (measured: 15 s is the best point).
- **Atrial flutter is not detected**, by design (§2).
- **Ectopy is not yet excluded from the RR series.** The standard mitigation is
  to drop ectopic beats before computing the features, which needs beat
  classification — Phase 3. Expect the ectopy-driven share of false alarms to
  fall then; the respiratory-sinus-arrhythmia share will not.
- **Sensitivity is bounded by the 20-interval window.** AF lasting under about
  20 beats cannot be seen at all.

---

## 8. Next

Phase 3 — **N/S/V beat classification** (AAMI classes) on MIT-BIH Arrhythmia,
Supraventricular and INCART, inter-patient. This is the point at which deep
learning and the Hailo accelerator get judged: a feature-based light model gets
built first, and a deep model has to beat it before it earns the power budget.
It also feeds back into this phase by letting ectopic beats be removed from the
RR series before AF features are computed.
