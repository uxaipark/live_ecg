# live_ecg — Phase 10: the beat layer, and four things that did not work

Six items: supraventricular accuracy, ventricular precision, fibrillation
precision, a fusion beat class, and detectors for atrial flutter, junctional
rhythm and idioventricular rhythm.

Three shipped and improved, one shipped as a class without a usable operating
point, and two were measured and abandoned. This report gives the same weight to
all six, because the refutations took the same work and are the more useful half
of it.

---

## 1. Result

| | before | after |
|---|---:|---:|
| VEB precision, sealed MIT-BIH | 77.74 % | **80.65 %** |
| VEB precision, sealed INCART (lead II) | 79.58 % | **87.14 %** |
| VEB sensitivity, sealed INCART | 77.80 % | **88.29 %** |
| Ventricular detector AUC, INCART | 0.9575 | **0.9737** |
| SVEB AUC, sealed MIT-BIH | 0.7384 | 0.7431 |
| Fibrillation AUC, held out | 0.8966 | **0.9014** |
| Fibrillation precision at a matched operating point | 27.85 % | **32.26 %** |
| Fibrillation specificity, 270 h of normal sinus | 99.945 % | **100.000 %** |
| Fibrillation, `reports/results` (in-sample, see §6) | 0.8895 / 28.35 % | **0.9190 / 42.78 %** |
| Fusion class | not reported at all | AUC **0.8328** |
| Idioventricular episodes found, long-term corpus | 0 of 135 | **78 of 135** |
| Atrial flutter | — | refuted, not shipped |
| Junctional rhythm | — | refuted, not shipped |

---

## 2. The dominant beat was the most frequent one

Every morphology feature in this engine is expressed relative to the patient's
dominant beat, and "dominant" meant "whatever arrived first and kept matching" —
which is the same as "most frequent". In a patient whose recording is half
ventricular that is the wrong beat, and the failure is not graceful: anchoring on
the ectopic morphology inverts all of those features at once.

INCART record I43 is 51 % ventricular. The detector found **2.6 %** of it and was
right about **4.8 %** of what it did report. It had learned the ectopy as normal
and was flagging the conducted beats.

The template now keeps a rival cluster and promotes it when the dominant is too
wide to be a conducted beat and the rival is not. Almost nothing here is an
absolute threshold; this is the exception that earns one, because QRS duration is
set by how the impulse travels, not by the electrode, the gain or the patient.
120 ms, with a 40 ms margin so it cannot flip between two morphologies that are
both wide — which is what bundle branch block looks like, and where flipping
helps nothing.

The width it tests is the **delineated** duration from Phase 9, not the
band-energy proxy the `width_rel` feature uses. That proxy runs at about half the
true duration and is only ever read as a ratio; the delineated one lands within a
millisecond of manual annotation, which is what makes it comparable to a
physiological constant at all.

### The threshold was chosen by proving the rule harmless

The training corpora contain 26 records with 200 or more ventricular beats and
**not one of them exhibits this failure**. A selection set that lacks the
phenomenon cannot be used to tune against it. So it was used for the only thing
it can honestly settle: at a 40 ms margin the rule leaves every training number
exactly where it was — 93.800/81.348 becomes 93.807/81.354 — and the sealed
improvement is the rule firing on a condition the training set does not contain.

---

## 3. A reported number that was measuring the lead choice

The pooled end-to-end figure ran three corpora under one `--lead`, which meant
INCART at lead I — a lead in which our own detector misses half the beats. The
"61.9 % ventricular precision" that came out of it was a statement about the lead
choice. INCART is now reported on its own, in the lead a chest patch
approximates: **88.12 / 85.17** end to end.

---

## 4. Fusion: a class, and an operating point that is not worth having

Fusion beats had been a row in the confusion matrix with no column since the bank
was built — the engine could be wrong about them but never right. There is now a
third detector on the ventricular detector's evidence, and `Beat::Fusion` reaches
the rhythm bank, where every rule that looks for ventricular activity counts it.

| | AUC | at the shipped threshold |
|---|---:|---|
| MIT-BIH | 0.8328 | 22.2 / 33.2 |
| INCART | 0.9023 | 13.3 / 6.2 |
| Supraventricular DB | 0.7041 | 5.3 / 5.0 (19 fusion beats in the zone) |

The guard is on the AUC, not on sensitivity, and that is the report rather than a
convenience. A fusion beat fires the ventricular detector too — it is half a
ventricular beat — and clears that threshold by a wider margin, so arbitration
hands it to V whatever this detector says. Out of sample the class is selected
0.27 % of the time at *every* threshold from 0.05 to 0.98, which is the signature
of a decision that does not belong to the threshold at all. Buying the operating
point would mean paying in missed ventricular beats.

### A metric that would have flattered itself

Sensitivity counted predictions over N, S and V only. Adding a fourth column
meant a ventricular beat called *fusion* silently left the ventricular
denominator, and reported ventricular sensitivity rose from 95.8 % to 97.3 % for
no reason but the new column existing. Sensitivity now counts every class a beat
could have been given; precision still counts only the reference classes in the
population, which is what keeps the N/S/V figures comparable with the literature.

---

## 5. Idioventricular rhythm was unreportable for two independent reasons

A ventricular run's rate was averaged over the whole run, and the rate is not
reported across an interval too long to be a heartbeat — which is right, one of
those would invent a bradycardia. But an escape rhythm is by definition what
follows a pause, so the interval it refuses to cross is the one the rhythm is
escaping from. The run was detected, asking how fast it was returned nothing, and
the condition could never hold. It is rated over the intervals *inside* the run
now.

**That is the third condition in this engine defined by exactly the thing a
general-purpose guard filtered out**, after asystole and the physiological gate
on pauses. The pattern has a name in the code now: a rule about what happens
*after* an abnormal interval cannot be evaluated over a window that still
contains it.

The second reason was a six-second minimum episode, right for a pattern like
bigeminy and wrong here: the condition is already a beat count, and at fifty a
minute the two requirements contradict each other.

| | episodes found | seconds |
|---|---|---|
| long-term AF corpus | 78 of 135 | 20.4 % |
| MIT-BIH | 3 of 4 | 90.9 % precision on record 124 |

Precision on the long-term corpus is 2 %, and that is structural: this condition
is a subset of the ventricular runs, so it cannot be more precise than they are,
and there they are 5 %.

---

## 6. Fibrillation: the shape of the trajectory

Every feature the fibrillation detector had was a statement about how big or how
fast the signal is, and all of them can be satisfied by a large slow artefact.
The phase-space plot is about neither: scale the window by its own peak, plot it
against itself half a second later, and count how much of a 40 by 40 grid the
trajectory visits. A rhythm with beats occupies a thin figure; fibrillation
fills the plane.

Fitted on three quarters of the training corpus, scored on the held-out quarter,
against a control fitted the same way with the feature zeroed:

| | AUC | Se | Sp | PPV |
|---|---:|---:|---:|---:|
| without | 0.8966 | 80.94 | 84.55 | 27.85 |
| with | **0.9014** | **83.57** | **87.07** | **32.26** |

Spectral moment and in-band amplitude share were added alongside and make it
*worse* — 0.8936 with all three, 0.8826 with the spectral pair alone. Two weak
correlated inputs degrade a regularised fit. They are gone, and with them a
128-point transform per window. Tree ensembles were refitted on the larger
feature set and are worse again, 0.864 against 0.894, which is the second time
they have been tried and the second time they have lost.

One caveat on the recorded result. `reports/results/vf_heldout.txt` reads
0.8895 to 0.9190 and 28.35 % to 42.78 %, and that comparison is **in sample**:
there is no sealed fibrillation corpus, the shipped model is fitted on all of
TRAIN, and the file scores it on a third of the same records. It is a fair
comparison of two shipped models against each other and it is not an estimate of
field performance. The held-out table above is.

---

## 7. Atrial flutter: two measurements, both refuted

Flutter is organised atrial activity at 240–340 a minute under a ventricular
rhythm that is often perfectly regular — so every interval statistic the
fibrillation detector owns would call it sinus. What identifies it is the atria,
and the engine had no measurement of them between complexes.

**First attempt: autocorrelation with the complexes blanked.** It scored *higher*
on MIT-BIH 100, which is ordinary sinus rhythm, than on 202, which is flutter.
The holes left by blanking repeat at the ventricular rate and correlate with each
other, and normalising against the whole window's energy caps the score at the
fraction of the window that survives.

**Second attempt: subtract an averaged QRST instead of blanking it.** This is the
right method and it fixed the artefact — the rate estimate on 202 moved into the
flutter band — and it still did not separate the two populations. Flutter records
sat at 0.14 regularity, sinus records at 0.15 to 0.27.

The measurement cost 20 ns a sample, 13 % of the whole engine, and has been
removed. What it needs is a lead in which flutter waves are visible; in MLII, on
these corpora, they largely are not.

---

## 8. Junctional rhythm: no evidence to detect it with

Junctional rhythm is a narrow complex at 40–100 a minute with no P wave in front
of it, or an inverted one. The engine can measure the first two and not the
third.

- **Polarity adapts.** The reference for "this patient's P wave direction" is a
  running mean, so a sustained junctional rhythm becomes its own normal within
  about ten beats and the inversion disappears.
- **Amplitude does not separate.** P amplitude against the beat's own R amplitude
  runs 0.021 to 0.166 across junctional records and 0.045 to 0.076 across sinus
  ones — completely overlapping.
- Written against *absence of evidence* instead, the detector fired 1,686 times
  on the fibrillation database and was right none of them. Absence of evidence is
  not evidence of absence, which this engine already knew in three other places
  and this detector had to relearn.

---

## 9. What the supraventricular class got, and did not

The atrial interval — the PP interval against the patient's own running median —
survives at 0.386 by record and buys about half a point of ventricular precision.

Its ratio to the RR interval does not. That ratio is the textbook distinction
between the two kinds of premature beat: an atrial ectopic beat is early because
the atrium fired early, so the P wave moves with the complex and the ratio stays
near one; a ventricular one leaves the atrium alone and the ratio rises. **Its
AUC is 0.520.** It cannot work here, because the sinus P wave that marches
through a ventricular beat is buried in the complex and a single lead does not
see it. The textbook picture assumes twelve.

Supraventricular detection remains the weak class at 0.743 on MIT-BIH.

---

## 10. Limitations

- **Ventricular precision on ambulatory data** is still the binding constraint,
  and it now bounds two conditions rather than one: ventricular runs at 5 % and
  idioventricular rhythm at 2 % on the long-term corpus.
- **Supraventricular detection** at 0.743. The P wave helps and does not solve
  it; an atrial premature beat's P wave is often inside the preceding T wave,
  which is exactly where this delineator does not look.
- **Fusion** has a usable ranking and no usable operating point.
- **Flutter and junctional rhythm are not detectable** with what a single lead
  gives this engine, on these corpora. Both would need either a second lead or a
  P-wave measurement that does not adapt away.
- **Record 207's idioventricular rhythm** is silenced by fibrillation
  suppression, which is working as designed: the record contains ventricular
  flutter and the two are adjacent.
