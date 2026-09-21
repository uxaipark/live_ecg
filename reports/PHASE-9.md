# live_ecg — Phase 9: the waves either side of the complex

Three separate limitations in this engine turned out to be the same missing
piece of evidence.

| limitation | measured before |
|---|---|
| supraventricular beats are found from prematurity alone | AUC 0.711 on sealed MIT-BIH |
| the atrial-fibrillation false-alarm tail is sinus arrhythmia | 6.39 alarms per 24 h of normal sinus signal, worst subject 52.7 |
| the quality monitor cannot tell "all waves readable" from "QRS only" | AUC 0.446 — worse than chance |

All three are about the P wave, and the engine measured nothing between one QRS
complex and the next. This phase delineates the P, QRS and T waves, validates
the marks against manual annotations, and feeds the result to all three.

---

## 1. Result

| | before | after |
|---|---:|---:|
| SVEB detector AUC, sealed MIT-BIH | 0.7111 | **0.7384** |
| SVEB sensitivity, sealed INCART | 73.1 % | **79.8 %** |
| AF sensitivity, sealed AFDB | 86.11 % | **90.52 %** |
| AF F1, sealed AFDB | 92.03 | **94.48** |
| AF burden error, sealed AFDB | 5.41 pp | **3.53 pp** |
| AF false alarms per 24 h, sealed NSRDB | 6.39 | **2.22** |
| — worst subject | 52.72 | **14.64** |
| class 1 vs class 2 agreement, BUT QDB | 0.446 | **0.823** |
| QRS detection, sealed MIT-BIH | 99.4975 / 99.7223 | **99.4975 / 99.7223** |
| full pipeline | 185.8 ns/sample | **226.8 ns/sample** |
| capacity, one core at 250 Hz | 21,525 channels | **17,639 channels** |

Delineation and the atrial features cost about 22 % of the engine's total, and
buy all three. A single core still carries seventeen thousand channels.

---

## 2. Where the marks land

Fitted on the development half of LUDB, confirmed on its sealed half. The
tolerance column is the spread *between* expert annotators from the CSE working
party — not a target this project set, and not one it meets on every landmark.

LUDB, sealed half, 66 records:

| mark | found | median error | in CSE tolerance | CSE 2SD |
|---|---:|---:|---:|---:|
| P onset | 94.2 % | 0 ms | 59.8 % | 10.2 ms |
| P peak | 93.9 % | −2 ms | 86.5 % | 10.2 ms |
| P offset | 93.7 % | −16 ms | 35.2 % | 12.7 ms |
| QRS onset | 100.0 % | 0 ms | 67.4 % | 6.5 ms |
| QRS offset | 100.0 % | 0 ms | 54.4 % | 11.6 ms |
| T peak | 98.2 % | −12 ms | 82.6 % | 30.6 ms |
| T offset | 95.5 % | −26 ms | 49.6 % | 30.6 ms |

QT is an independent corpus — different annotators, different leads, 250 Hz,
never used to fit anything here. Medians hold between 0 and −28 ms with no
collapse; the spread widens, which is what a corpus quantised at 4 ms per sample
should do against a 6.5 ms tolerance.

The two offsets are where this is weakest. A P wave's end and a T wave's end are
where a slow deflection flattens into baseline, and an amplitude threshold cuts
both short.

---

## 3. The first measurement was a clock problem, not a detection problem

QRS onset came out 71 ms late with a standard deviation of 20. A bias that tight
with that little spread is not a detector failing to find something; it is a
detector finding it and reporting the wrong time.

Marks were being found on filtered taps and reported as if found on the input.
Every tap lags by its own group delay, the QRS tap and the P/T tap lag by
different amounts, and the R position handed in had already been corrected for a
third. Ring indices are now converted at the accessor, so no search in the file
can touch one.

Three things the measurement then refuted, each of which had looked reasonable:

- **The P/T tap was a band-pass applied to an already high-passed signal.** A
  second 0.5 Hz corner buys no further baseline rejection and costs 155 ms of
  group delay at 1 Hz against 24 ms at 10 Hz — on exactly the two waves the tap
  exists to locate. It is a low-pass now, and its delay is flat across the band.
- **Boundaries were walked on |x| of a band-passed complex.** A QRS in 5–20 Hz is
  a damped oscillation with several zero crossings inside it, so the walk stopped
  at the first one: a complex 58 ms too narrow. It walks an envelope.
- **P presence was scored against the window's median |x|.** A baseline offset
  dominates that. Against the window's median *and* its median absolute
  deviation — a robust z-score, invariant to the offset — P waves found went from
  57 % to 95 %.

The reference frequencies at which each tap's delay is evaluated were fitted
against the *peak* of each wave, never against a boundary. A boundary also
carries whatever the threshold criterion does, and absorbing that into a delay
would make the delay wrong everywhere else the tap is used.

---

## 4. The pooled AUC was measuring two patients

The atrial template match `p_ncc` scored 0.155 for supraventricular beats on
MIT-BIH's training records and 0.500 on its sealed ones. That looks like a
feature that stopped working. It is a pooled average over *beats* being decided
by one patient who has thousands of them.

The per-record median tells the transferable story, and by it `p_ncc` is the
strongest feature in the whole table: 0.088 for S, against 0.041 for prematurity.
Both numbers are now printed.

---

## 5. Two features measured and rejected

**`p_present`** — the largest deflection in the atrial window over that window's
noise floor. The search returns the largest deflection whatever the window
contains, so it fires before almost every beat, ventricular ones included. On
205,000 beats it separated conducted from ventricular at an AUC of 0.48. It
measures that the window exists.

**`p_ncc`** — the raw correlation with the atrial template, the strongest feature
in the table by per-record AUC, and adding it cost 0.033 of sealed AUC. Both are
true at once. A correlation is not comparable between patients: 0.75 means "no
atrial activity" for a patient whose P wave is clean and "entirely normal" for
one whose P wave is small, and a single global threshold cannot be right for
both. Expressed against the patient's own running median — which is how every
other feature in that vector was already written — the same evidence helps.

That is the third time in this project a feature has failed for being absolute
rather than relative.

---

## 6. Fibrillation: the feature that is not about timing

Every input the AF model had measured how irregular the rhythm was. The false
alarms that survived were rhythms that are genuinely irregular and genuinely
sinus — marked respiratory arrhythmia in a healthy young subject is as variable
as fibrillation, and no interval statistic can say otherwise.

The window now carries the median correlation between *consecutive beats'* atrial
segments. Deliberately consecutive rather than against a template: a template
adapts, and in sustained fibrillation it would re-anchor on the fibrillatory
baseline and report that everything matches. Two consecutive segments have
nothing to adapt to.

By AUC it is the third strongest of the fifteen features measured (0.074,
inverted), behind sample entropy and the share of intervals near the median.

Compared at matched false-alarm rates, because the probability scale moved:

| alarms / 24 h of normal sinus | AFDB sensitivity |
|---|---|
| 2.22 → **1.07** | 78.0 % → **82.5 %** |
| 6.39 → **3.82** | 86.1 % → **93.2 %** |
| 14.39 → **9.86** | 94.1 % → **96.4 %** |

The operating point moved from 0.95 to 0.92 against the same development budget
the old one was chosen against — hold the false-alarm rate, take the sensitivity.

---

## 7. Comparing a model change against itself

The first comparison of the AF model was confounded: the refit with the new
feature was measured against *the weights that happened to be shipped*, which
were not produced by the same command. "The model got worse" and "refitting made
it worse" are not the same claim.

`fit-af` now takes `--without <feature>`, which zeroes one input before fitting.
The control and the candidate then differ by one thing. The control reproduced
the shipped model to three decimals, which is what made the comparison usable.

The beat trainer gained the same facility from the other direction: a
per-detector feature set, which is the bank's own argument carried one step
further. Separate detectors already have separate thresholds because the costs
differ; they do not always share a feature either, and a feature that is noise
for one question is not free for it to ignore — it is something its trees can
overfit. The default set is named in the code, so a feature added to the vector
joins the report without silently joining the models.

---

## 8. A guard that was blind by design

The false-alarm guard watched the median subject, because the distribution is
heavy-tailed and one subject dominates the mean. That reasoning was right and the
guard was still useless here: the median was 0.00 before and after, so a
three-and-a-half-fold improvement on the one subject it was written for was
invisible to it.

It now watches the worst subject as well. Two statistics, because one of them is
blind by construction.

---

## 9. Legibility is reported beside quality, not inside it

The quality monitor decides whether beats may be analysed at all, so it cannot
consume anything derived from beats without the decision feeding itself. The
atrial coherence is published as a separate output.

Together they are the three classes a human rater uses: the monitor says whether
the complex can be trusted, legibility says whether anything else can. Swept
against BUT QDB's raters, a threshold of 0.95 agrees with 70.5 % of their class 1
and 72.9 % of their class 2 — a balanced split of a distinction that previously
had no machine side at all.

One caveat, stated in the code because it will look like a defect: atrial
activity is also incoherent in fibrillation, so a patient in AF on a clean trace
is reported "QRS only". That is the right answer. The P wave is not readable,
because there is not one.

---

## 10. Limitations

- **P and T offsets** sit outside the CSE tolerance about half the time. Both are
  where a slow wave flattens into baseline; the tangent construction helps the T
  offset's bias but widens its tail.
- **Supraventricular detection is still the weak class.** 0.738 AUC on sealed
  MIT-BIH. The P wave helps and does not solve it: an atrial premature beat's P
  wave is often buried inside the preceding T wave, which is precisely where this
  delineator does not look.
- **The atrial features were selected on a training-internal holdout**, then
  reported once on the sealed sets. The sealed sets were looked at once during
  that process, before the holdout protocol was adopted; the numbers above are
  from the holdout-selected model.
- **Wave legibility is not gated on anything.** It is published and nothing
  consumes it yet.
- **LUDB is ten-second records of mostly clean signal.** The delineator's
  behaviour on hours of ambulatory noise is measured only indirectly, through
  what the downstream detectors do with it.
