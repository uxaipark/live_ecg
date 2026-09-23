//! Holding the supraventricular label across an ectopic atrial rhythm.
//!
//! # The class is mostly a rhythm, not a beat
//!
//! A premature atrial beat is premature *to* the rhythm it interrupts, so its
//! interval is evidence about the first beat and about nothing after it. On
//! 3,274 sealed hours of patch recording that is not an edge case: **90.2 % of
//! the reviewed supraventricular beats sit in runs of eight or more** and 7.6 %
//! are isolated. Split by position, the beat detector reads 54.8 % sensitivity
//! on the beat that opens a run and 2.1 % on the beats that continue one. It is
//! not failing to recognise ectopy; after the first beat it is being asked a
//! question that has no answer.
//!
//! # What continues a run, and what does not
//!
//! The atrium, because that is the only part of the beat that changed - an
//! ectopic focus takes a different path through the atria while the ventricles
//! are still reached through the normal conduction system, which is why the QRS
//! is unchanged and why this cannot reuse the ventricular machinery.
//!
//! Which P wave is compared matters more than the comparison. Cut on the rate's
//! window - several hundred milliseconds of which the P wave is a small part -
//! any two atrial segments correlate highly whatever atrium produced them, and
//! a run held on that basis runs straight through ordinary sinus beats. On the
//! patch corpus that window reads 0.900 on normal beats and 0.905 on
//! supraventricular ones: no separation at all.
//!
//! Cut on the P peak at a fixed width it does separate, but only in one
//! respect, and that distinction is the whole reason this is not switched on:
//!
//! | previous -> this | development | sealed |
//! |---|---|---|
//! | normal -> normal | 0.558 | 0.659 |
//! | normal -> supraventricular | **-0.035** | **-0.039** |
//! | supraventricular -> normal | **-0.279** | **-0.159** |
//! | supraventricular -> supraventricular | 0.758 | 0.436 |
//!
//! **A change of class is visible on both halves** - either boundary sits at or
//! below zero while same-class pairs stay well above it. **How well two beats of
//! the same class match is not**: it reads 0.758 on the development half and
//! 0.436 on the sealed one, and on the sealed half it is *lower* than the
//! normal-to-normal figure. A fixed bar chosen on one half does not describe
//! the other, which is the second reason a run cannot be held on it yet.
//!
//! # What it is allowed to do
//!
//! Extend a claim the beat detector already made, forward only. It cannot open
//! a run, cannot revive a closed one, and cannot be held open by beats with no
//! readable P wave - the same prior the electrode detector uses, because an
//! absence of evidence that keeps a claim alive keeps it alive for ever.

use crate::template::BeatVector;

#[derive(Debug, Clone, Copy)]
pub struct AtrialRunConfig {
    /// How well a beat's P wave must match the one that opened the run for the
    /// rhythm to be the same rhythm.
    pub hold_ncc: f32,
    /// Consecutive beats with no readable P wave that a run survives.
    pub max_blind: u32,
}

impl AtrialRunConfig {
    /// Never holds. The mechanism is off unless a bar is set.
    pub const OFF: AtrialRunConfig = AtrialRunConfig {
        hold_ncc: 2.0,
        max_blind: 0,
    };
}

impl Default for AtrialRunConfig {
    /// Off, and the measurement is why.
    ///
    /// Holding the onset's verdict across the reference's *own* runs would
    /// read Se 32.1 % and +P 66.4 % against 7.9 / 32.7 - four times the
    /// sensitivity and twice the precision. Holding it across the runs this
    /// finds costs precision at every bar swept on the development zone:
    ///
    /// | hold at | Se % | +P % | false calls / 1000 |
    /// |---|---|---|---|
    /// | off | 8.3 | 21.4 | 14.3 |
    /// | 0.80 | 9.0 | 13.7 | 26.7 |
    /// | 0.50 | 9.8 | 10.6 | 38.6 |
    /// | 0.20 | 10.9 | 9.9 | 46.7 |
    ///
    /// The gap between the two has two causes and neither is the threshold.
    /// **Propagation multiplies whatever the onset call got wrong** - the
    /// onset call is 21-33 % precise, so each false one becomes a run of false
    /// ones, while the oracle only ever propagates inside true runs. And the
    /// bar that decides where a run ends does not transfer between halves of
    /// the corpus; see the table above.
    ///
    /// So this waits on a precise onset, and the thing standing in the way of
    /// one is measured too: a beat's fit to a running P template is a *worse*
    /// than chance predictor of the class (AUC 0.398), because with runs
    /// hundreds of beats long the template becomes the ectopic P wave and the
    /// run fits it better than sinus does. A P template has to carry a
    /// dominant and a rival the way the QRS template already does.
    fn default() -> Self {
        AtrialRunConfig::OFF
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AtrialRun {
    onset: Option<BeatVector>,
    blind: u32,
}

impl AtrialRun {
    pub fn new() -> Self {
        AtrialRun::default()
    }

    /// Feed one beat. `called` is what the beat detector decided on its own.
    /// True when the beat belongs to an ectopic atrial rhythm.
    pub fn push(&mut self, called: bool, p: Option<&BeatVector>, cfg: &AtrialRunConfig) -> bool {
        if called {
            // The detector's own call stands, and re-anchors the run on the
            // most recent beat it was sure about.
            self.onset = p.copied().or(self.onset);
            self.blind = 0;
            return true;
        }
        let Some(onset) = self.onset.as_ref() else {
            return false;
        };
        let Some(p) = p else {
            self.blind += 1;
            if self.blind > cfg.max_blind {
                self.reset();
                return false;
            }
            return true;
        };
        self.blind = 0;
        if onset.ncc(p) >= cfg.hold_ncc {
            true
        } else {
            self.reset();
            false
        }
    }

    pub fn reset(&mut self) {
        self.onset = None;
        self.blind = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::TEMPLATE_LEN;

    /// An explicit working configuration. The shipped default is `OFF`, and
    /// the logic still has to be right for the day the onset call is precise
    /// enough to switch it on.
    const ON: AtrialRunConfig = AtrialRunConfig {
        hold_ncc: 0.5,
        max_blind: 2,
    };

    fn wave(seed: f32) -> BeatVector {
        let mut v = [0.0f32; TEMPLATE_LEN];
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i as f32 * 0.1) + seed).sin();
        }
        BeatVector { v, scale: 1.0 }
    }

    #[test]
    fn a_run_holds_while_the_atrium_repeats_and_ends_when_it_changes() {
        let cfg = ON;
        let mut run = AtrialRun::new();
        let (ectopic, sinus) = (wave(0.0), wave(3.1));
        assert!(run.push(true, Some(&ectopic), &cfg), "the detector's own call");
        for _ in 0..5 {
            assert!(run.push(false, Some(&ectopic), &cfg), "same atrium");
        }
        assert!(!run.push(false, Some(&sinus), &cfg), "the atrium changed back");
        assert!(!run.push(false, Some(&ectopic), &cfg), "a closed run revived");
    }

    /// Without this a run with no readable P waves runs to the end of the
    /// recording.
    #[test]
    fn a_run_cannot_be_held_open_by_missing_evidence() {
        let cfg = ON;
        let mut run = AtrialRun::new();
        assert!(run.push(true, Some(&wave(0.0)), &cfg));
        assert!(run.push(false, None, &cfg));
        assert!(run.push(false, None, &cfg));
        assert!(!run.push(false, None, &cfg), "held open by nothing");
    }

    #[test]
    fn nothing_happens_without_a_call_to_extend() {
        let cfg = ON;
        let mut run = AtrialRun::new();
        for _ in 0..10 {
            assert!(!run.push(false, Some(&wave(0.0)), &cfg));
        }
    }

    #[test]
    fn the_shipped_default_never_holds() {
        let cfg = AtrialRunConfig::default();
        let mut run = AtrialRun::new();
        let p = wave(0.0);
        assert!(run.push(true, Some(&p), &cfg), "the detector's call still stands");
        assert!(!run.push(false, Some(&p), &cfg));
    }
}
