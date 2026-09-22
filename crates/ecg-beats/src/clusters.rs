//! Morphology clustering, and why the ventricular findings need it.
//!
//! # The problem is arithmetic, not discrimination
//!
//! The ventricular detector separates at an AUC of 0.97 to 0.99. Its episode
//! precision on 1,961 hours of ambulatory signal is 3.7 %. Both are true, and
//! the second is not a symptom of the first.
//!
//! Ventricular runs occupy 0.034 % of that corpus. At 99.6 % specificity — which
//! is what an AUC of 0.99 buys at a sensible operating point — the false
//! positives outnumber the true ones eleven to one, because there are three
//! thousand times more seconds to be wrong about. Reaching 50 % precision at
//! that prevalence needs 99.9966 % specificity, a hundredfold improvement, and
//! no single-lead beat classifier reaches it. Tightening the threshold trades
//! the sensitivity away and leaves the precision where it was.
//!
//! So the output is wrong, not the detector. Eight million beats produce eight
//! million independent chances to be wrong, and a reviewer cannot audit eight
//! million decisions. What a reviewer *can* audit is thirty.
//!
//! # What clustering changes
//!
//! Beats of the same origin have the same shape. Group them, and a recording's
//! millions of beats collapse into a few dozen morphologies; the question stops
//! being "is this beat ventricular", asked millions of times, and becomes "is
//! this *shape* ventricular", asked a few dozen times.
//!
//! That is not a presentation trick, it is a statistical one. A cluster's score
//! is the median of its members', and the median of 317 beats is determined far
//! better than any one of them. The engine goes from eight million decisions per
//! record to about thirty, and the base-rate arithmetic that made precision
//! hopeless applies to the thirty instead.
//!
//! It is also what a Holter reading room does, which is not a coincidence — it
//! is where the constraint was discovered.
//!
//! # Streaming, not batch
//!
//! Clusters accumulate from the start of the recording and can be read at any
//! time; a patch records for days and there is no end of file to wait for.
//! Memory is fixed: at capacity the two most similar clusters merge, so the
//! count never grows and the state stays a few kilobytes per channel.

use crate::detectors::{BeatClass, BeatVerdict};
use crate::template::BeatVector;

/// Ceiling on the number of morphologies, whatever a caller asks for.
pub const CLUSTER_CEILING: usize = 256;

/// Running median over the last sixteen values, for a cluster's summary.
#[derive(Debug, Clone, Copy)]
struct Median16 {
    buf: [f32; 16],
    n: usize,
    idx: usize,
}

impl Median16 {
    fn new() -> Self {
        Median16 {
            buf: [0.0; 16],
            n: 0,
            idx: 0,
        }
    }
    fn push(&mut self, v: f32) {
        self.buf[self.idx] = v;
        self.idx = (self.idx + 1) & 15;
        self.n = (self.n + 1).min(16);
    }
    fn median(&self) -> f32 {
        if self.n == 0 {
            return f32::NAN;
        }
        let mut t = [0.0f32; 16];
        t[..self.n].copy_from_slice(&self.buf[..self.n]);
        t[..self.n].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        t[self.n / 2]
    }
}

/// One morphology, and the evidence accumulated about it.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Stable identifier. Survives merges into this cluster; a cluster merged
    /// *away* takes its members' counts with it and its own id disappears.
    pub id: u32,
    centroid: BeatVector,
    /// Beats assigned to this morphology.
    pub count: u64,
    /// First and last beat assigned, on the input time base.
    pub first: u64,
    pub last: u64,
    /// The member that sits closest to the centroid, which is what a reviewer
    /// should be shown. Held as a position, not a waveform: the caller has the
    /// signal and this object is meant to stay small.
    pub exemplar: u64,
    exemplar_ncc: f32,
    /// Running medians of the evidence a reviewer would otherwise have to
    /// reconstruct: how wide the complex is, whether a P wave precedes it, how
    /// premature it is, and what the detectors made of it.
    qrs_ms: Median16,
    p_ncc_rel: Median16,
    rr_prev_rel: Median16,
    p_ventricular: Median16,
    p_supraventricular: Median16,
    /// Beats in this cluster by arbitrated class, so a cluster that the beat
    /// layer itself disagrees about is visible as such.
    pub classed: [u64; 4],
}

impl Cluster {
    fn new(id: u32, v: &BeatVector, sample: u64) -> Self {
        Cluster {
            id,
            centroid: *v,
            count: 0,
            first: sample,
            last: sample,
            exemplar: sample,
            exemplar_ncc: -1.0,
            qrs_ms: Median16::new(),
            p_ncc_rel: Median16::new(),
            rr_prev_rel: Median16::new(),
            p_ventricular: Median16::new(),
            p_supraventricular: Median16::new(),
            classed: [0; 4],
        }
    }

    /// Median QRS duration in milliseconds, zero when never measured.
    pub fn qrs_ms(&self) -> f32 {
        let v = self.qrs_ms.median();
        if v.is_nan() {
            0.0
        } else {
            v
        }
    }

    /// How badly the atrial segment matches this patient's own, at the median.
    /// Around one means a P wave as usual; far above means none.
    pub fn atrial_mismatch(&self) -> f32 {
        let v = self.p_ncc_rel.median();
        if v.is_nan() {
            1.0
        } else {
            v
        }
    }

    pub fn prematurity(&self) -> f32 {
        let v = self.rr_prev_rel.median();
        if v.is_nan() {
            1.0
        } else {
            v
        }
    }

    /// The cluster's ventricular score: the median of its members'.
    ///
    /// The median, not the mean, and not the centroid's own score. A cluster
    /// gathers beats that share a shape, not beats that share a context, so a
    /// few of its members will have been measured across a noisy interval or a
    /// missed neighbour. The median ignores them; a mean would not, and the
    /// centroid's score would throw away the n that makes this worth doing.
    pub fn score(&self) -> f32 {
        let v = self.p_ventricular.median();
        if v.is_nan() {
            0.0
        } else {
            v
        }
    }

    pub fn supraventricular_score(&self) -> f32 {
        let v = self.p_supraventricular.median();
        if v.is_nan() {
            0.0
        } else {
            v
        }
    }

    fn fold(&mut self, v: &BeatVector, d: &BeatVerdict, a: f32) {
        let (f, class, sample) = (&d.features, d.class, d.sample);
        let ncc = self.centroid.ncc(v);
        for (c, x) in self.centroid.v.iter_mut().zip(v.v.iter()) {
            *c += a * (*x - *c);
        }
        let norm = self
            .centroid
            .v
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-6);
        for c in self.centroid.v.iter_mut() {
            *c /= norm;
        }
        if ncc > self.exemplar_ncc {
            self.exemplar_ncc = ncc;
            self.exemplar = sample;
        }
        if f.width_rel > 0.0 {
            self.rr_prev_rel.push(f.rr_prev_rel);
            self.p_ncc_rel.push(f.p_ncc_rel);
        }
        self.p_ventricular.push(d.p_ventricular);
        self.p_supraventricular.push(d.p_supraventricular);
        self.count += 1;
        self.last = sample;
        let i = match class {
            BeatClass::N => 0,
            BeatClass::S => 1,
            BeatClass::V => 2,
            BeatClass::F | BeatClass::Unknown => 3,
        };
        self.classed[i] += 1;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ClusterConfig {
    /// A beat joins a cluster when it correlates at least this well with it.
    ///
    /// Looser than the dominant template's admission bar, and for the opposite
    /// reason: that gate exists to keep ectopy *out* of one reference, this one
    /// exists to gather ectopy together. Too tight and one morphology splits
    /// across several clusters, which costs the reviewer the very thing
    /// clustering buys.
    pub admit_ncc: f32,
    /// How fast a centroid follows its members.
    pub alpha: f32,
    /// Beats a cluster needs before it is worth showing anyone.
    pub min_count: u64,
    /// Morphologies tracked at once.
    ///
    /// Not a reviewer's budget - that is set by how many clusters score above
    /// the bar, which is far fewer. This is how many *distinct shapes* the
    /// recording is allowed to contain, and a day of ambulatory signal contains
    /// many: posture, electrode drift and rate all move the complex. Set to 32
    /// it bound on 76 of 84 long-term records, and the forced merges blended
    /// ventricular morphologies into normal ones until only 17.8 % of the
    /// ventricular beats were left in a predominantly ventricular cluster.
    pub max_clusters: usize,
    /// How similar two clusters must be before they may be merged to make room.
    ///
    /// Without this the merge is unconditional: at capacity the closest pair
    /// joins whatever their similarity, and "closest" in a full bank can still
    /// be two shapes a reviewer would never have called the same. When no pair
    /// is this similar the smallest cluster is dropped instead, which loses a
    /// few beats rather than corrupting a morphology.
    pub merge_ncc: f32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        ClusterConfig {
            admit_ncc: 0.90,
            alpha: 0.02,
            min_count: 3,
            max_clusters: 64,
            merge_ncc: 0.95,
        }
    }
}

pub struct MorphologyBank {
    cfg: ClusterConfig,
    clusters: Vec<Cluster>,
    next_id: u32,
    /// Beats that arrived before any cluster could take them.
    pub unassigned: u64,
    /// Merges forced by the capacity bound, so a caller can tell whether the
    /// cap was ever reached.
    pub merges: u64,
    /// Beats lost with a dropped morphology, when no two were alike enough to
    /// merge. Published because a silent loss is worse than a counted one.
    pub dropped: u64,
}

impl MorphologyBank {
    pub fn new(cfg: ClusterConfig) -> Self {
        MorphologyBank {
            cfg,
            clusters: Vec::with_capacity(cfg.max_clusters.min(CLUSTER_CEILING)),
            next_id: 1,
            unassigned: 0,
            merges: 0,
            dropped: 0,
        }
    }

    pub fn clusters(&self) -> &[Cluster] {
        &self.clusters
    }

    /// Clusters worth a reviewer's time, most ventricular first.
    pub fn ranked(&self) -> Vec<&Cluster> {
        let mut v: Vec<&Cluster> = self
            .clusters
            .iter()
            .filter(|c| c.count >= self.cfg.min_count)
            .collect();
        v.sort_by(|a, b| {
            b.score()
                .partial_cmp(&a.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }

    /// Assign one beat. Returns the cluster id it landed in.
    pub fn push(&mut self, v: &BeatVector, d: &BeatVerdict) -> Option<u32> {
        if d.class == BeatClass::Unknown {
            // A beat the classifier would not judge tells us nothing about a
            // morphology, and folding it in would blur whichever centroid it
            // happened to resemble.
            self.unassigned += 1;
            return None;
        }
        let best = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.centroid.ncc(v)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((i, ncc)) = best {
            if ncc >= self.cfg.admit_ncc {
                let a = self.cfg.alpha;
                self.clusters[i].fold(v, d, a);
                return Some(self.clusters[i].id);
            }
        }

        if self.clusters.len() >= self.cfg.max_clusters.min(CLUSTER_CEILING) {
            self.make_room();
        }
        let id = self.next_id;
        self.next_id += 1;
        let mut c = Cluster::new(id, v, d.sample);
        c.fold(v, d, self.cfg.alpha);
        self.clusters.push(c);
        Some(id)
    }

    /// Make room for a new morphology, by merging two that are the same shape
    /// or, failing that, by dropping the smallest.
    ///
    /// Similarity first: dropping the smallest as a matter of course would
    /// throw away the rare morphology, and the rare morphology is the one a
    /// reviewer needs. But merging two shapes that are *not* the same is worse
    /// than losing a handful of beats, so the merge has a bar and there is a
    /// fallback for when nothing clears it.
    fn make_room(&mut self) {
        let mut best = (0usize, 1usize, -2.0f32);
        for i in 0..self.clusters.len() {
            for j in (i + 1)..self.clusters.len() {
                let s = self.clusters[i].centroid.ncc(&self.clusters[j].centroid);
                if s > best.2 {
                    best = (i, j, s);
                }
            }
        }
        let (i, j, sim) = best;
        if sim < self.cfg.merge_ncc {
            // Nothing here is the same shape as anything else. Give up the
            // least-evidenced morphology rather than corrupt two.
            let smallest = self
                .clusters
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| c.count)
                .map(|(k, _)| k)
                .unwrap_or(0);
            self.dropped += self.clusters[smallest].count;
            self.clusters.remove(smallest);
            return;
        }
        let (keep, drop) = if self.clusters[i].count >= self.clusters[j].count {
            (i, j)
        } else {
            (j, i)
        };
        let gone = self.clusters.remove(drop);
        let keep = if drop < keep { keep - 1 } else { keep };
        let k = &mut self.clusters[keep];
        k.count += gone.count;
        k.first = k.first.min(gone.first);
        k.last = k.last.max(gone.last);
        for (a, b) in k.classed.iter_mut().zip(gone.classed.iter()) {
            *a += b;
        }
        if gone.exemplar_ncc > k.exemplar_ncc {
            k.exemplar = gone.exemplar;
            k.exemplar_ncc = gone.exemplar_ncc;
        }
        self.merges += 1;
    }

    pub fn reset(&mut self) {
        self.clusters.clear();
        self.unassigned = 0;
        self.merges = 0;
        self.dropped = 0;
    }
}
