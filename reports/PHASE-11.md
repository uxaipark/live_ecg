# live_ecg — Phase 11: the patch corpus, and what counts as truth

Two halves. The first finishes work on the public corpora: supraventricular
calls in fibrillation, the ends of the P and T waves, and an asystole figure
that turned out to be about its annotations. The second is the first contact
with the domain the engine exists for — 24,043 single-lead patch recordings,
3.58 million hours — and most of it is about what can and cannot be claimed
from that corpus's labels.

Five things shipped. Seven were built, measured and removed or left off. As in
Phase 10, the refutations get the same space as the results: they took the same
work, and they are what stops the next attempt repeating them.

Every figure is from `reports/results/` or from a run quoted in the commit that
made it. `PERFORMANCE.md` is the table; this is the account.

---

## 1. Result

| | before | after | |
|---|---|---|---|
| Supraventricular precision, MIT-BIH + supraventricular, sealed | 37.2 % | **47.7 %** | §2, §3 |
| T offset inside the CSE tolerance, LUDB sealed | 49.6 % | **72.0 %** | §3 |
| P offset inside the CSE tolerance, LUDB sealed | 35.2 % | **69.1 %** | §3 |
| Asystole lost to the suppression rules, 2,200 h | not measured | **0** | §4 |
| Patch signal the quality monitor believes | 4.9 % | **97.5 %** | §5 |
| Patch ventricular false positives per 1,000 beats | 30.7 | **1.83** (patch bank) | §8 |
| Long-term AF review-queue precision, as published | 56.1 % | **49.4 %** — a correction, §7 | |

Refuted and removed, or left off: holding the supraventricular label across a
run on the rate's atrial window; the same on a P-anchored window; a two-cluster
atrial template; three eviction policies for the morphology bank; epochs for
the morphology bank; a patch ventricular ensemble trained on the device's own
calls.

---

## 2. No supraventricular call inside sustained fibrillation

A premature atrial beat is premature *to* a sinus rhythm. In fibrillation there
is none, so the two strongest pieces of supraventricular evidence — prematurity
and a missing P wave — are not weak there but undefined. On the training half of
MIT-BIH, 893 beats inside sustained fibrillation were called supraventricular
and 15 were.

Three things had to be right. *Sustained*, not the window's instantaneous call:
dense atrial ectopy makes the intervals look fibrillating for a few beats, and
gating on that suppresses the ectopy that caused it. *A withdrawn claim must not
become another one*: dropping the class from the ballot handed the beat to the
runner-up and cost two points of ventricular precision for nothing, so
arbitration runs first and the context applies to its winner. *A bar of 1.0 must
mean never*: a confident ensemble saturates to exactly 1.0 in `f32`.

Raising the bar instead of closing it recovered 1.2 points. The model is not
marginally wrong in fibrillation but confidently wrong, and a threshold cannot
fix a confident error.

## 3. The ends of the P and T waves

P offset and T offset sat at −16 and −26 ms on the LUDB sealed half while P
onset and both QRS marks sat at zero. A one-sided error on half the marks is not
a delay error — a delay moves every mark on a tap together. It is a
disagreement about where a wave *ends*: a threshold or a tangent declares it over
before the trace rejoins the baseline, which is where a person marks it.

The boundary fraction and the T reference frequency were tried first. The first
is shared with P onset; the second would cancel the bias while pretending to be
physics. So the correction is named for what it is and fitted on the LUDB
development half. It transfers to QT on T offset (28.9 → 45.4 % inside
tolerance) and not on P offset (49.3 → 43.1 %): the two annotator groups
disagree with each other by the whole tolerance on that one mark.

## 4. The asystole figure was a measurement of the annotations

Asystole sensitivity on the long-term AF corpus read 9.5 %. There are four ways
to lose one and each was counted: a beat detected in the silence, the electrode
rule, the quiet-interval rule, or nothing covering it.

**The two suppression rules lose none, in 2,200 hours.** 136 of the 153 were lost
to a beat detected inside the silence — full amplitude, never from search-back,
exactly one per gap three to four and a half times the surrounding interval. On
the normal sinus corpus the reference contains 426 four-second asystoles in 270
hours of healthy volunteers, and every one of them contains beats. On MIT-BIH,
annotated beat by beat, the census reads 14 of 14. A guard now holds both
suppression rules at zero losses.

## 5. Reading the patch corpus

**A reader rather than a copy.** Exporting to WFDB would have duplicated 250 GB
of a corpus that should not be copied. The containers are zarr v3 in stored zip
entries, blosc over zstd, 300-second chunks; `ecg-zarr` reads them from a memory
map, is checked byte-exactly against zarr 3.3.0, and refuses ZIP64 by name
rather than misreading it.

**The gain was missing and recoverable.** `adc_gain` is null for every
recording. Almost nothing in the engine is absolute — one threshold is, the
saturation level, and it gates the electrode flag, which gates asystole. Fed raw
counts, a twenty-count QRS reads as twenty millivolts and the quality monitor
believes 4.9 % of the corpus. The QRS detector needs no gain, so it runs on
counts and the gain is whatever makes its beats the size a heart makes. Per
record, because counts per millivolt span sevenfold: at that gain the monitor
believes 97.5 % and 0.14 % of samples read saturated.

The manifest is generated (`tools/build_internal_manifest.py`) and not
committed: it is per-exam metadata from a private corpus.

## 6. The supraventricular class on the patch is a rhythm

Against 22 recordings an analyst reviewed exhaustively — the only reference in
the corpus on which specificity is defined — supraventricular sensitivity read
7.9 % against the device's 62.1 %. Split by position:

| | sensitivity |
|---|---|
| beat that opens a run | 54.8 % |
| beat that continues one | 2.1 % |

**90.2 % of the class sits in runs of eight beats or more.** After the first beat
there is nothing to be early against. Holding the onset's verdict across the
reference's own runs would read 32.1 % at 66.4 % — which is why three ways of
finding the end of a run were built:

1. **On the rate's atrial window.** Refuted: that window is mostly baseline, and
   two of them correlate highly whatever atrium produced them (0.900 normal,
   0.905 supraventricular).
2. **On a P wave cut on its own peak.** It separates *changes* of class on both
   halves of the corpus, but not the match between two beats of the same class
   (0.758 on development, 0.436 sealed), and propagation multiplies the onset
   call's errors: the onset is 21–33 % precise. Shipped off, with its ceiling
   and its sweep recorded where the default is set.
3. **A two-cluster atrial template.** A single running P template predicts the
   class worse than chance (AUC 0.398): a run hundreds of beats long becomes the
   template. Two clusters kept apart plateau at chance (0.526), the second
   cluster filling with noise rather than a second focus.

A P wave here is a tenth of a millivolt; per-beat atrial identity on a
single-lead patch is below the noise floor. The remaining explanation for the
device's figure is that its supraventricular label is assigned by rhythm and
carried to the beats, which no per-beat classifier can reproduce. That is a
hypothesis about the reference, not something this repository can settle.

## 7. The ventricular review queue on the patch, and a count that flattered

Per beat, ventricular calls on the patch run at 33 % precision — the arithmetic
of a 1.75 % prevalence, from a ranking with AUC 0.975. Published as morphologies
instead, the queue reads 59.4 % at 83.8 % over 14.8 clusters per recording.

**A correction to published figures.** When the bank is full two morphologies can
merge, and the beats of the one that goes now belong to the one that stays. Both
evaluations credited each beat to its original id, so merged beats pointed at
nothing and silently left the table. On the long-term AF corpus the published
queue moves from 64.0 % at 56.1 % to 67.4 % at 49.4 % — precision had been 6.7
points too high. The pipeline now reports merges and drops as they happen
(`ChannelOutput::morphology_events`), because a reviewer's tool has the same
problem.

**What the lost fifth is.** The bank drops ventricular beats at six times the
rate of normal ones. Three eviction policies and bank epochs of 24, 6 and 2 hours
were built; all lost to the shipped policy, and two-hour epochs still filled the
bank in every recording. Counted directly, **80 % of the ventricular beats lost
were in a morphology of one beat** that resembled nothing else in its recording.
The bank now hands those out (`ChannelOutput::dropped_morphologies`); kept, they
lift sensitivity to 77.1 % and cost 5,540 decisions per recording at 62.5 %. A
beat like no other is a per-beat decision, and a queue does not change that.

## 8. What counts as truth, and a ventricular detector fitted on the patch

**Truth is two things**: annotations made independently of the device — the
public corpora — and labels an analyst decided. The device's automatic calls are
a competitor, scored beside the engine and never used as truth. Evaluation
followed that from the start. Training, at first, did not.

The first patch ensemble was fitted on the corpus's ordinary labels, which
outside the exhaustively reviewed recordings are mostly the device's untouched
calls. It ranked at AUC 0.979 on the development zone against 0.960 for the
compiled-in ensemble. **Trained on truth only — the beats an analyst changed,
moved or added, plus the public corpora — it ranks at 0.963.** The advantage was
the device's label lineage, which the exhaustive review itself starts from. That
ensemble was withdrawn.

The truth-trained one is `BeatBank::patch()`. On the sealed 22:

| | Se % | +P % | false / 1,000 |
|---|---|---|---|
| compiled-in ensemble | 84.4 | 33.4 | 30.7 |
| **patch bank** | 55.7 | 84.7 | 1.83 |
| patch bank, outside noise | 65.2 | 88.4 | 1.22 |
| the device | 88.1 | 90.1 | 1.76 |

It is a different operating point on a similar ranking (sealed AUC 0.968 against
0.975), with a high-specificity end the compiled-in ensemble lacks. Against the
public corpora's independent annotations the ensembles rank alike and the patch
bar is too high for them — MIT-BIH sensitivity 50.7 % against 95.4 % — so the
compiled-in ensemble stays the default and the patch bank is for the patch.

The ensemble is confident enough that its useful bar sits where an `f32`
probability rounds to 1.0, so it is emitted with its scores divided by a
temperature of 4 (`ecg-eval emit-model`), which leaves the ranking alone.

## 9. Disclosures

- **The sealed patch set has been scored more than once.** Twice for the
  withdrawn ensemble — once before the bank's arbitration was accounted for,
  once after — and twice for the patch bank, once at the wrong bar when a
  chained command failed on a missing formatter and left the previous bar in
  place (34.7 % at 91.0 %). No choice was taken from a sealed result, but the set
  is no longer untouched.
- **The patch truth descends from the device.** Even the exhaustive review began
  as the device's beat list. Every patch figure is agreement with an analyst who
  was shown the device's answer first.
- **Patch detection is not measured.** A beat the device never marked is
  invisible to review.
- **The wave-boundary corrections are calibrated to one annotator group** and
  only one of the two transfers to the other (§3).

## 10. What would move this most

A few dozen patch recordings, sampled at random and read afresh from the raw
waveform by an expert who is not shown the device's labels. It would give the
patch corpus its first reference independent of the device, measure detection
there for the first time, and replace a sealed set that has now been looked at.
It is data to acquire, not code to write.

## 11. Afterwards: the rhythm detector, and more data

**The supraventricular class, found by its rhythm.** §6 said the question has to
be asked of the run rather than the beat. Joining runs an analyst split by a few
normal beats, 57 % of the development zone's episodes start with the interval
dropping below 0.85 of what came before, and a step that size appears in 0.32 %
of stretches of sinus rhythm. `SvRunDetector` finds such steps into regular
stretches and holds them while the rate stays near its own; tuned on the
development zone from beat positions alone, and scored once on the sealed set:

| patch, sealed 22 | Se % | +P % | F1 |
|---|---|---|---|
| per-beat only | 7.9 | 32.7 | 0.13 |
| with runs | **48.2** | **49.8** | **0.49** |
| the device | 62.1 | 91.5 | 0.74 |

Against the public corpora it is mixed. MIT-BIH's supraventricular sensitivity
goes from 24.9 % to 84.0 % - record 232's ectopic atrial rhythm is found at last
- while on long recordings with little ectopy, where sinus rhythm sometimes
steps, precision roughly halves. So runs and the patch bank together are
`PipelineConfig::patch()`, and the default is unchanged. Requiring the per-beat
detector's call at the start of a run was measured and left off: the analyst's
runs carry one 63.9 % of the time and the others 37.0 %.

**More data did not help the ventricular ensemble.** Refitted on 1,875 training
recordings at 72 hours each - 1.89 million analyst-determined rows - it ranks at
AUC 0.958 on the development zone against 0.963 for 342 recordings at 24 hours,
with the same best F1 (0.751 against 0.753). The analyst-determined set is
selected by what the device got wrong, and more of it is more of the same hard
cases. A larger ensemble on the same rows - 240 trees of depth 6, eight times the
nodes - overfits: AUC 0.938 on the development zone against 0.975 in sample, best
F1 0.721. The smaller ensemble stays. What would change the ventricular figures is
§10's data, not more of this.

## 12. Afterwards: where the supraventricular false calls came from, and the ventricular run alarm

**The runs were catching sinus rhythm stepping from slow to normal.** A dump of
every run the detector reported on the development zone (`ecg-eval svrun
--svr-features`), each labelled by whether the analyst's review agrees, gives an
unusually clean split. The agreed runs beat at a median of 376 ms and the
disputed ones at 830 ms. Most of the dispute is sinus rhythm stepping up after
an arousal and then held for minutes. A supraventricular *tachycardia* is by
definition faster than 100 a minute, so that became the rule
(`SvRunConfig::max_interval_ms`, 600 ms): from beat positions, the beats inside
runs go from 64.0 % at 48.9 % to 55.2 % at 88.4 %. F1 reads the same from 600 to
700 ms, and 600 was taken because it is the definition. With the full pipeline,
the disputed runs fall from 832 to 146 and the agreed ones from 602 to 558.
Loosening the other parameters to win the sensitivity back was not stable: a
longer grace helps at 5 and falls off a cliff at 6, because one recording
dominates. They stayed.

**Most of what was left was the per-beat call.** With runs off, the per-beat
supraventricular detector on the development zone is right 87,764 times and
wrong 235,833 times; with the rate rule in place, the runs add only about 12,000
false calls to that. The per-beat call is kept on the ballot - a beat it wins
is kept from the ventricular detector - but its win is reported only where the
detector is all but certain (`BeatBank::supraventricular_report`). Chosen by F1
on the development zone, and the ventricular figures are identical in every row:

| bar | S Se % | +P % | false / 1000 | F1 |
|---|---|---|---|---|
| none | 42.6 | 51.5 | 18.8 | 0.466 |
| 0.999 | 41.2 | 65.6 | 10.1 | 0.506 |
| 0.9999 | 40.4 | 72.8 | 7.1 | 0.519 |
| **0.99999** | 39.5 | 79.5 | 4.8 | **0.528** |
| 1.0, never | 31.8 | 93.9 | 1.0 | 0.475 |

Lowering the detector's threshold instead - taking it off the ballot - was
measured first: at 1.0 it reads 37.4 % at 89.3 %, but ventricular precision
falls from 86.7 % to 82.9 %. The beats it had won went to the ventricular
detector, the failure `classify_in` already names.

Scored once on the sealed set:

| patch, sealed 22 | S Se % | +P % | false / 1000 | F1 |
|---|---|---|---|---|
| runs, first version | 48.2 | 49.8 | 23.1 | 0.49 |
| **runs, current** | **43.0** | **67.5** | **9.84** | **0.53** |
| the device | 62.1 | 91.5 | 2.75 | 0.74 |

The public corpora read the same way except one. Under the patch preset,
end to end, supraventricular precision is two to six times the first version's
(supraventricular corpus 72.3 → 88.2 %, INCART 7.2 → 19.1 %, Long-Term 2.7 →
15.5 %), at five to seven points of sensitivity. On MIT-BIH it collapses, from
88.0 % at 21.4 % to 12.5 % at 10.4 %: record 232's ectopic atrial rhythm is
slower than 100 a minute. The rule reports what it is defined to report, and
that patient is outside it. The default configuration, which does not run it,
is unchanged. Both tables are in `PERFORMANCE.md` §4b.

**The ventricular run alarm cannot be fixed after the fact.** MIT-BIH's alarm is
right 9 % of the time and the Long-Term AF corpus's 4 %. `ecg-eval vruns` dumps
every run of three or more beats the engine called ventricular, with what the
reference says they were and fourteen features of the run: morphology
consistency, clusters, rate, regularity, coupling, the shortest interval, the
scores. On the public training and development zones, 12,003 runs, 528 real:

- A third of the false runs contain a detection with no reference beat, mostly
  a wide ventricular beat counted twice. Most of the rest are normal beats
  called ventricular next to a real one, or in bursts of noise.
- The best single rule, at most two morphologies in the run, takes MIT-BIH from
  23 % to 36 % and loses a quarter of the real runs. The shortest interval
  (dropping runs with an interval under 240 ms) buys three points.
- A gradient-boosted classifier over all fourteen features, cross-validated by
  record, sets the ceiling: at the operating point that doubles MIT-BIH's
  precision it keeps 17 of 41 real runs.

The alarm's errors are the per-beat detector's errors, correlated in time, and a
filter on the run cannot see past them. The review queue (§4b of
`PERFORMANCE.md`) remains the way to present ventricular runs, and nothing was
changed. The diagnostic stays for the next attempt at the per-beat detector.

**Patient-specific morphology is already in.** The third item on the list was
to add distance to the patient's own normal template as features. They are
already there - `ncc_template`, `width_rel`, `amp_rel` and the rest are all
relative to this recording's running template - so there is nothing to add
under that name. On the development zone, outside the stretches the device
called noise, the patch bank reads 87.2 % at 92.0 % against the device's
96.3 % at 91.9 %. Counting every beat it is 66.4 % against 97.6 %: across the
development zone the engine declines 77,324 of the analyst's ventricular beats
for signal quality and calls 119,503 normal, and the difference between the two
rows says most of that is inside noise.

## 13. Afterwards: the fibrillation alarm, scored sealed for the first time

**There was sealed fibrillation all along.** The reports said no sealed corpus
carries fibrillation, because VFDB and CUDB are entirely TRAIN. The Sudden
Death corpus - 100 % TEST, 23 records, 447 hours - has no rhythm annotations,
but each header carries the onset of the fibrillation that ended the recording
as a comment, `#vfon: HH:MM:SS`. Whether that is a time of day or an offset
from the start was settled by the signal rather than assumed: read as an
offset, the detector's probability rises within ten seconds of all twenty
onsets; read against the header's base time, it rises at none.

`ecg-eval vf-alarm` scores the detector the way an alarm is used: an onset is
found if the alarm sounds from 30 s before to 120 s after it, or is already
sounding; latency runs to the moment it sounds; false alarms are counted more
than ten minutes before an onset, or anywhere in a recording without one. The
ten minutes before are kept apart, because they are often tachycardia
degenerating. (`rhythm-census` lists what each corpus's annotators wrote, which
is how the question was asked.)

**Choosing the bar on the training zones.** At the shipped 0.70 the training
zones read 38 of 38 onsets on VFDB and 43 of 44 on CUDB, with false alarms of
0.18 a day on the Long-Term AF corpus's 1,947 hours, 6.5 on MIT-BIH's training
records and 101 on the noise-stress records. At 0.80: 38 of 38 and 41 of 44,
with 0.02, none and 60. At 0.85 CUDB falls to 38. The bar is now 0.80.

Refitting the model with the noise-stress records among the negatives was
tried and did not help: on the held-out third its onsets and false alarms were
within a record of the shipped fit's. The features were built to separate
fibrillation from other rhythms, and noise at -6 dB is not a rhythm.

**Scored once on the sealed set**, after the choice:

| sealed | onsets found | median latency | false alarms per 24 h |
|---|---|---:|---:|
| Sudden Death, bar 0.70 | 19 / 20 | 8 s | 6.75 |
| **Sudden Death, bar 0.80** | **19 / 20** | **9 s** | **3.96** |
| Normal Sinus 270 h, European ST-T 180 h, Long-Term 110 h | — | — | **0** |
| MIT-BIH 12 h / INCART 38 h | — | — | 3.99 / 2.56 |

Sensitivity and latency are where a monitor needs them. Four false alarms a
day on patients at risk of sudden death is not.

**The ventricular tachycardia alarm stays where it was.** Requiring longer runs
helps on MIT-BIH's training records - runs of five or more at 100 a minute are
right 35 % of the time against 23 % for three - but not on the Long-Term AF
corpus, where aberrantly conducted runs in fibrillation are read as ventricular
and stay at 4 %. Adding interval regularity, morphology consistency or both
trades half the real runs for that precision. It is §12's conclusion again: the
alarm's errors are the per-beat detector's, and telling a wide aberrant beat
from a ventricular one on a single lead is the problem to solve, not the run
rule. Nothing was changed.
