# live_ecg — Phase 4: arrhythmia episode detection

Eight conditions, each a binary detector in the same bank as the beat
classifiers, over the same shared evidence: intervals, beat classes, and signal
quality. Builds on Phases 1–3.

Regenerate with `tools/run_evaluation.sh`; raw output in `reports/results/`.

---

## 1. Result

Sealed TEST, MIT-BIH Arrhythmia, 24 records. Scored against the same rule
applied to the corpus's own beat annotations — see §2 for why that is the right
reference and what the alternative measured instead.

| condition | prevalence | Se % | Sp % | +P % | episodes found |
|---|---:|---:|---:|---:|---|
| **asystole** | 0.07% | **100.0** | 100.0 | **100.0** | 14 / 14 |
| **pause** | 0.60% | **100.0** | 100.0 | 95.6 | 92 / 92 |
| **bradycardia** | 3.9% | **99.5** | 99.8 | 95.7 | 40 / 40 |
| **tachycardia** | 8.7% | **97.6** | 100.0 | **100.0** | 14 / 14 |
| bigeminy | 1.6% | 71.2 | 100.0 | 98.0 | 20 / 26 |
| ventricular run | 0.10% | 38.6 | 99.4 | **6.4** | 9 / 20 |
| ventricular tachycardia | 0.09% | 44.7 | 99.8 | **17.9** | 9 / 17 |

Over 1,961 hours of the Long-Term AF corpus the timing conditions hold at scale:
bradycardia 99.0% / 96.7%, tachycardia 93.0% / 95.0%. The ventricular conditions
do not (§5).

On the sick-sinus record 232 — 14 documented asystolic pauses, and the reason
that record exists — every pause and every asystole is found, with no false
ones: 89/89 and 14/14.

---

## 2. Two references, because there are two questions

Each of these conditions is *defined*: a pause is an interval over two seconds,
bigeminy is every second beat being ventricular. So the primary reference is the
**same rule run over the corpus's own beat annotations**. What that measures is
whether our detection and classification reproduce the definition, which is the
thing this engine can be wrong about.

The annotator's rhythm spans are reported as a secondary view. They answer a
different question — whether the rule agrees with a human — and the
disagreements are mostly definitional.

**The first run of this harness made the case for the distinction.** Scored
against rhythm spans alone, tachycardia came out at **6.6% precision**. The
detector was right and the reference was wrong for the purpose: these corpora
annotate rhythm *origin*, not rate. Sinus tachycardia is written `(N`, so every
correctly detected fast sinus rhythm counted as a false positive. Against the
same definition applied to the reference beats, tachycardia is 97.6% / 100%.

---

## 3. Asystole was structurally unreportable, twice

Both times every other number looked fine, and both times the cause was a gate
that was individually reasonable.

**First: gated behind "physiological".** Every condition was gated on
`RrSample::usable()`, which requires the interval be a believable heartbeat
interval — at most three seconds. Asystole begins at four. The filter meant to
reject detection artefacts removed exactly the intervals that define the
condition, so asystole could never be reported at all, at any threshold, on any
record. It read as a clean 0.00% sensitivity against 14 reference episodes.

Each condition is now gated on what it actually needs: pause and asystole on
electrode integrity, rate conditions on the intervals being believable,
morphology conditions on a beat class existing.

**Second: the quality monitor called it a dead lead.** With the first gate
fixed, asystole was still 0 / 14. A flat trace is what asystole looks like, so
the `LEAD_OFF` veto — added in Phase 2 and correct on its own terms — fired on
the asystole itself.

From one lead, without electrode impedance, there is no clean way to separate a
detached electrode from a heart that has stopped. Both are flat. So the tie is
broken by consequence rather than by likelihood: **a false asystole alarm is
reviewed and dismissed; a missed one is not.** Asystole and pause are now gated
on rail contact only, and the quality flags travel with the episode so a
consumer can see that `LEAD_OFF` was also raised and weigh it.

Record 232 before and after: 0 / 14 asystoles, then 14 / 14; 39% of pauses, then
100%.

Both defects have regression guards now. They are the ones most worth having: a
condition that is structurally impossible to report looks exactly like a
condition that never occurred.

---

## 4. A third absorbing state, in the beat template

Phase 3's morphology conditions read **0.00% on 1,961 hours** of the Long-Term AF
corpus. Not weak — nothing at all. Every beat came back `Unknown`, so every
detector that needs a beat class went quiet, correctly, on a verdict that was
itself wrong.

The beat template admits only beats that already resemble it, at a correlation
of 0.90 or better. That is what stops ectopy from polluting it. It also means
that if it anchors on an unrepresentative first beat, **nothing can ever match
and it is stuck for the rest of the recording** — and these are 21-hour
recordings. Short records survived it by luck: MIT-BIH Supraventricular records
are half an hour, and their coverage was 99%.

The template now re-anchors after 50 consecutive non-matching beats. The
threshold was chosen on held-out training records, where accuracy is flat from
12 to 75 and coverage plateaus around 50.

This turned out to matter well beyond the corpus that exposed it, because the
same trap was springing partially everywhere:

| | before | after |
|---|---:|---:|
| beats classified, Long-Term AF | 0.0% | 98.7% |
| beats classified, MIT-BIH | 98.6% | 97.6% |
| **VEB sensitivity / precision, sealed TEST** | 85.2 / 69.2 | **89.1 / 76.0** |
| ventricular detector AUC | 0.967 | **0.992** on held-out training |
| SVEB sensitivity / precision | 62.0 / 29.8 | 62.1 / 28.1 |

That is the third absorbing state in this engine — after the QRS threshold in
Phase 1 and this one, the pattern is worth naming: **every component that learns
a reference from its own accepted inputs needs a way back.** The gate that keeps
it clean is the same gate that can lock it onto a mistake.

---

## 5. Ventricular runs are the weak condition

Ventricular run and ventricular tachycardia score 6.4% and 17.9% precision:
108 reported runs against 20 real ones on MIT-BIH, 8,586 against 943 on the
Long-Term corpus.

The cause is arithmetic rather than mysterious. A run requires **three
consecutive** correct ventricular classifications. At 76% precision per beat,
false ventricular beats do not need to be common to make spurious triples — they
only need to cluster, and they do, because the beats a classifier gets wrong are
the ones where morphology is odd for several beats together.

This is the condition most exposed to beat-classification precision, and it will
not improve by tuning the run rule. It improves when VEB precision does.

Bigeminy behaves differently for the same reason in reverse — 98% precision,
71% sensitivity — because a *pattern* is far harder to produce by accident than
a run is. Three consecutive errors make a false run; making a false bigeminy
needs errors that alternate.

---

## 6. What is not covered

- **Ventricular fibrillation and flutter are not detected.** They are the most
  dangerous thing on this list and they need a different kind of detector: in VF
  there are no beats, so every stage downstream of QRS detection is built on an
  assumption that has failed. The two corpora that carry it — `vfdb` and `cudb` —
  are also entirely in the TRAIN zone, so there is no sealed set to score
  against without carving one. Both are on the list; neither is done.
- **Long intervals on 24-hour Holters are not physiology.** On the Normal Sinus
  corpus the reference contains 550 "pauses" and 426 "asystoles" over 270 hours,
  and the engine scores 5% and 0.5% against them. Those reference intervals are
  stretches where the annotator's beat labels stop — electrode dropout — not
  stopped hearts. Neither side is measuring the patient there. MIT-BIH 232, which
  has genuine documented pauses, is 100% on both.
- **Atrial flutter, junctional and idioventricular rhythms** have annotations in
  these corpora and no detector here.
- **Search-back works against pause detection.** The QRS detector is built not to
  go deaf, and in a genuine pause it will look for a beat. That tension is
  currently resolved in favour of detection and has not been measured
  separately.

---

## 7. Cost

| | Phase 3 | Phase 4 |
|---|---:|---:|
| Capacity, one core @ 250 Hz | ~28,700 channels | ~28,000 channels |

The rhythm bank is eight state machines over a sixteen-beat history, evaluated
once per beat rather than per sample. It does not move the budget.
