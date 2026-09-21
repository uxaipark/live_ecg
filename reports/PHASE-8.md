# live_ecg — Phase 8: acting on the fibrillation flag

Phase 5 built a fibrillation detector and exposed `in_vf()`. Nothing read it, so
the engine kept emitting beat-derived findings through fibrillation, where there
are no beats and every one of them describes an artefact.

Measured on the fibrillation corpus before this phase, across 12.8 hours:

| | reported | real |
|---|---:|---:|
| ventricular runs | 982 | 0 |
| ventricular tachycardias | 707 | 0 |
| pauses | 303 | 0 |
| **asystoles** | **60** | 0 |
| tachycardias | 157 | 0 |
| bradycardias | 67 | 0 |
| atrial fibrillation alarms | 135 per 24 h | 0 |

---

## 1. Result

| | before | after |
|---|---:|---:|
| ventricular runs on the fibrillation corpus | 982 | **216** |
| ventricular tachycardias | 707 | **162** |
| pauses | 303 | **95** |
| asystoles | 60 | **10** |
| AF alarms per 24 h | 134.7 | **59.8** |
| AF sensitivity, sealed AFDB | 86.11% | **86.11%** |
| asystole, sealed MIT-BIH | 100% | **100%** |
| analysis withheld on MIT-BIH / AFDB / Supraventricular | — | **0.000%** |
| analysis withheld on Normal Sinus | — | 0.018% |

Roughly three quarters of the nonsense is gone, and nothing that contains no
fibrillation is touched.

---

## 2. Reporting and suppressing are not the same decision

The first attempt drove suppression from the *faster* signal — the raw
per-window fibrillation state rather than the confirmed episode — on the
reasoning that suppression is cheap: it costs a suspended conclusion, while an
alarm costs a clinician's attention.

That was backwards, and the guards said so immediately: **atrial fibrillation
sensitivity fell from 86.0% to 28.7%** and asystole from 100% to 14%. Suppression
*deletes true findings*, which makes it the more consequential decision, not the
less.

Moving to the confirmed episode fixed asystole but not AF, because the
fibrillation detector still fires occasionally on atrial fibrillation — it runs
at 28% precision (Phase 5), and 1.74% of one AFDB record was flagged. That 1.74%
cost **95% of that record's AF windows**, because every interruption both
withholds intervals and invalidates the twenty-interval window that has to refill
afterwards.

So the two decisions now have their own thresholds. Reporting fibrillation uses
0.60, where sensitivity is 85.6%. Withholding uses **0.75**, measured as the
point where suppression stops touching AFDB at all while still removing more than
half the spurious findings on the fibrillation corpus. Below it, AFDB starts to
suffer; above it, the benefit shrinks.

A guard now pins that suppression touches essentially nothing on four corpora
that contain no fibrillation.

---

## 3. A fourth absorbing state, self-inflicted

The first working version placed the suppression `continue` *before* the
fibrillation stage. While suppressing, the detector stopped receiving samples —
so it could never observe the fibrillation ending, and suppression became
permanent.

That is the fourth of these in this project, after the QRS threshold, the beat
template, and the physiological gate on asystole. The pattern now has a name in
these reports: **anything that decides when to stop must keep receiving the
evidence that would tell it to.**

---

## 4. Gap handling was corrupting the time base

Chasing the AF regression turned up an older defect. All four `on_gap` handlers,
added in Phase 6, reset their sample counter to zero.

The counter *is* the time base. Resetting it restarts every later reported
position from zero, so after any gap — a lost packet, a fibrillation episode —
every beat, interval and episode carried a time stamp wrong by the length of the
recording so far. Nothing downstream would notice.

`on_gap` now takes the number of unobserved samples and advances the counter
*through* the gap, invalidating history by raising a floor instead. Positions
stay true and nothing already reported moves.

The server's clock test caught this the moment the handlers stopped resetting —
it began failing at 109.1 seconds of a 120-second recording, which is exactly
what a gap the clock has not accounted for looks like. That test was written in
Phase 6 to catch splicing and ended up catching the fix's own incompleteness.

---

## 5. What is still emitted during fibrillation

Raw beat detections and the quality verdict still go out, deliberately: a
reviewer needs to see what the detector did, and `suppressed_samples` counts the
silence so a consumer can tell "nothing happened" from "we stopped believing the
beats".

Suppression starts late by construction — a four-second window plus a
ten-second confirmation, so roughly fourteen seconds of artefact still escapes at
the start of an episode. Withholding retroactively would mean holding every
channel's output in a delay line, which costs that latency on every channel to
clean up a few.

---

## 6. Limitations

- **The residue is the detector.** What remains — 216 ventricular runs, 95 pauses
  on 12.8 hours — is fibrillation that the detector does not flag, at 85.6%
  sensitivity. It shrinks when Phase 5's feature set improves, not by tuning
  this.
- **0.018% of the Normal Sinus corpus is withheld**, which is 66 seconds across
  a hundred hours. Immaterial, and still a false positive.
- **The suppression threshold was chosen against AFDB and the fibrillation
  corpora only.** It has not been checked against ventricular tachycardia or
  paced rhythms, which are the other things that might plausibly trip it.
