# Phase 2B — feeding beat classes back into AF

Phase 3 left one concrete prediction open. Phase 2 §5 said the AF false-alarm
rate was dominated by a handful of subjects, and that removing ectopic beats
from the RR series — the textbook mitigation, which needs a beat classifier —
would cut the ectopy-driven share of it. Phase 3 built the classifier. This is
the measurement.

**The prediction is refuted, and Phase 2's own data had already said why.**

## What was measured

Sealed TEST. AF on AFDB, false alarms on the AF-free Normal Sinus records,
end to end through the real pipeline.

| intervals withheld | Se % | +P % | episodes found | Normal Sinus false alarms /24 h (median / worst) |
|---|---:|---:|---:|---|
| **nothing** | **86.0** | 98.8 | **33 / 34** | 0.00 / 52.7 |
| ventricular beats | 74.0 | 99.0 | 29 / 34 | 0.00 / **51.7** |
| ventricular and supraventricular | 11.0 | 99.4 | 16 / 34 | 0.00 / 8.8 |

Excluding ventricular beats costs **12 points of sensitivity and four episodes**
and moves the worst subject's false-alarm rate from 52.7 to 51.7 per 24 hours —
nothing.

## Why it could not have worked

Phase 2 §5 already measured it: the subjects that generate the false alarms
carry **0.02% ectopic beats**. There was no ectopy to remove. The cause was
identified there as marked respiratory sinus arrhythmia, and no amount of ectopy
filtering touches that.

The prediction was reasonable when written and wrong, and the evidence that it
was wrong was sitting in the report that made it. The lesson is narrow and
practical: a mitigation aimed at a mechanism should be checked against the
measured prevalence of that mechanism in the failing cases before it is built.

## The supraventricular result is a design constraint, not a tuning outcome

Withholding supraventricular beats takes AF sensitivity from 86% to 11%.

That is not a threshold that needs adjusting. The supraventricular detector
identifies a beat as ectopic from **prematurity**, and atrial fibrillation is
the condition in which *every* beat is early against the running median. The two
read the same evidence, so filtering the AF detector's input by the
supraventricular detector removes the rhythm being looked for.

This is worth stating as a rule rather than a result: **a detector may only
filter another detector's input when their evidence is independent.**
Ventricular ectopy qualifies — it is identified from morphology, which is
orthogonal to timing. Supraventricular ectopy does not.

It also sharpens the argument for the detector bank. Two detectors that share an
evidence base cannot be composed naively, and a single multi-class model would
have hidden the dependency inside one softmax where it could not be inspected at
all.

## What was kept

Both switches remain (`exclude_ventricular`, `exclude_supraventricular`), default
off, so the measurement is repeatable and so a deployment population with
frequent ectopy can turn the first one on.

One change was kept unconditionally, because it is a correctness fix rather than
a policy: **successive-difference features are now computed only across pairs of
intervals that are genuinely adjacent in time.** Once intervals can be withheld —
for quality, and now optionally for ectopy — two neighbouring entries in a
window need not be neighbours in time, and differencing them invents an
irregularity that never happened. Previously the AF features differenced across
quality gaps silently. With the model refitted, AF sensitivity is 86.0% against
86.2% before, so this costs nothing and removes a latent source of false
irregularity.

That refit is itself the point of a procedure note: changing a feature's
definition invalidates the coefficients fitted against the old one. The first
run after this change read 11% sensitivity and looked like a catastrophic
regression. It was a stale model. The same thing happened in Phase 2 with the
sample-entropy tolerance.

## Status

AF detection is unchanged from Phase 2 at 86.0% / 98.8% with 33 of 34 episodes,
now with correct gap handling. The false-alarm tail remains what Phase 2 said it
was: one subject with marked respiratory sinus arrhythmia, six of eleven
subjects at zero, and an RR-only detector has limited means to separate smooth
respiratory variation from fibrillation.

Closing that gap needs different evidence — P-wave presence, which means
delineation — not a better filter on the same evidence.
