# Phase 1B — the quality monitor against human annotation

Phase 1 validated noise detection against the MIT noise-stress protocol, which
has exactly known labels and exactly one kind of noise: the `118e*`/`119e*`
records add *electrode motion*. So `hf_ratio`, the feature meant to catch muscle
artefact, had never been tested against the thing it describes. Phase 1 §7 listed
that as an open axis. This closes it.

BUT QDB is the opposite kind of evidence — 18 long-term recordings, no synthetic
noise at all, each rated second by second by **four independent annotators** into
three classes: full diagnostic quality, QRS-reliable-only, and QRS not reliably
detectable. Worse ground truth, genuinely different failure modes.

Split by subject (the leading digits of a BUT QDB record name identify the
person, so `103001`–`103003` are one subject and must not straddle the line),
alternating subjects, recorded in the manifest rather than inferred.

---

## 1. It found a defect that inverted the score

| | class 3 vs 1 | class 2+3 vs 1 |
|---|---:|---:|
| score, as Phase 1 shipped it | 0.604 | **0.226** |
| score, after the fix | **0.820** | 0.446 |

An AUC of 0.226 is not weak, it is backwards: the monitor ranked degraded
seconds as *better* than pristine ones. Every individual feature was correctly
oriented at the same time — `-kurtosis` 0.955, `p2p_rel_dev` 0.963 — which is
what made it worth looking at the distributions rather than the summary.

**The cause was `flat_frac` entering the score as a linear penalty.** A clean
ECG spends most of each beat on the isoelectric line, so its flat fraction is
high *by nature*:

| annotator class | median `flat_frac` |
|---|---:|
| 1 — full diagnostic quality | **0.688** |
| 2 — QRS reliable only | 0.393 |
| 3 — QRS not detectable | 0.357 |

Multiplying the score by `1 - flat_frac` charged the best signal a 69% penalty
and the worst a 36% one. The flag threshold was 0.90 and correct; the
*continuous* term underneath it was not gated by anything.

Phase 1 had already seen this feature read backwards on the noise-stress corpus
(AUC 0.23) and treated it as a measurement bug in the feature — the threshold
was rescaled to reference the channel's own slow amplitude. That fixed how
`flat_frac` was computed and left how it was *used* untouched, which is why the
orientation survived.

Every veto is now a threshold rather than a proportional demerit: flatline and
saturation ramp to zero near their limits instead of subtracting their fraction
continuously.

## 2. Everything else improved, nothing regressed

The fix touches the whole engine, so it was re-measured everywhere:

| | before | after |
|---|---:|---:|
| nstdb, AUC for predicting a detector error | 0.913 | **0.960** |
| nstdb, AUC-noise at −6 / 0 / 6 / 12 dB | 0.989 / 0.982 / 0.918 / 0.730 | **0.993 / 0.990 / 0.922 / 0.801** |
| MIT-BIH QRS sensitivity / precision | 99.496 / 99.717 | 99.497 / 99.722 |
| AF sensitivity / precision, end to end | 86.1 / 98.8 | 86.1 / 98.8 |
| VEB sensitivity / precision | 81.5 / 66.0 | 81.5 / 66.0 |
| Normal Sinus median false alarms per 24 h | 0.00 | 0.00 |

## 3. The answer on `hf_ratio`

| | development half | sealed half |
|---|---:|---:|
| `hf_ratio`, class 3 vs 1 | 0.507 | **0.9992** |

On the development half the feature is worth nothing. On the sealed half it is
almost a perfect separator. The two halves are not in conflict — they contain
different artefact. The sealed half is dominated by one 38-hour recording that is
34% unusable, and whatever degrades it is high-frequency; the development half's
bad seconds are not.

So the honest answer to the question Phase 1 left open is: **the feature is real,
and whether it matters depends entirely on the recording.** That is the case for
keeping it as an extreme-value veto — which is how it is already used — and the
case against ever having trusted the single number the noise-stress corpus gave
it. One corpus with one kind of noise cannot retire a feature.

The same records make the point in reverse: on the sealed half `base_ratio`
scores 0.002 and `-qrs_ratio` 0.157, both strongly inverted, where on the
development half they are 0.603 and 0.895. Different recordings fail in
different directions, and a score built on any single feature would follow them.

## 4. Sealed result

| | records | hours | class 3 | score AUC (3 vs 1) |
|---|---:|---:|---:|---:|
| development half | 9 | 55.1 | 3.5% | 0.820 |
| **sealed half** | 9 | 44.3 | 29.8% | **0.9965** |

On seconds where all four annotators agreed, the sealed figure is 0.9964.

**The sealed number is one recording.** Record 105001 is 38.7 of the 44.3 hours
and carries 87% of the class-3 seconds; the remaining eight records are under an
hour each. It should be read as "the monitor agrees with four humans on one long,
genuinely bad recording", not as a corpus-wide result.

## 5. What is still not measured

The score does not separate class 1 from class 2 (AUC 0.45 on the development
half, 0.71 on the sealed one). That is a different question and the features
saturate before reaching it: at kurtosis 15 and 24 both ramps are already at
their ceiling, so nothing distinguishes "QRS reliable" from "every wave
readable".

This is a limitation rather than a defect. The score's job is to say whether
detection can be believed, and it does that. Whether P and T waves are readable
needs evidence about P and T waves, which means delineation — and a consumer
that needs the Good-versus-Acceptable distinction should not rely on this number
until that exists.
