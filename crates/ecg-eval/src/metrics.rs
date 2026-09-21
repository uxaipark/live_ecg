//! Detector scoring.
//!
//! One-to-one matching inside a tolerance window, as in AAMI EC57 / `bxb`.
//! Greedy in time order: with a tolerance far below the refractory period the
//! greedy assignment and the optimal assignment coincide, and the greedy pass
//! is O(n) rather than O(n log n).

#[derive(Debug, Clone, Default)]
pub struct DetectionScore {
    pub tp: u64,
    pub fp: u64,
    pub fn_: u64,
    pub n_ref: u64,
    pub n_det: u64,
    /// Signed detection - reference offsets for matched beats, in milliseconds.
    /// Milliseconds rather than samples so records at different rates pool.
    pub offsets: Vec<f32>,
}

impl DetectionScore {
    pub fn sensitivity(&self) -> f64 {
        div(self.tp, self.tp + self.fn_)
    }

    pub fn ppv(&self) -> f64 {
        div(self.tp, self.tp + self.fp)
    }

    pub fn f1(&self) -> f64 {
        let (se, pp) = (self.sensitivity(), self.ppv());
        if se + pp == 0.0 {
            0.0
        } else {
            2.0 * se * pp / (se + pp)
        }
    }

    /// Detection error rate: the gross error count over the reference count.
    pub fn der(&self) -> f64 {
        div(self.fp + self.fn_, self.n_ref)
    }

    pub fn merge(&mut self, other: &DetectionScore) {
        self.tp += other.tp;
        self.fp += other.fp;
        self.fn_ += other.fn_;
        self.n_ref += other.n_ref;
        self.n_det += other.n_det;
        self.offsets.extend_from_slice(&other.offsets);
    }
}

fn div(a: u64, b: u64) -> f64 {
    if b == 0 {
        0.0
    } else {
        a as f64 / b as f64
    }
}

/// Match `det` against `reference` within `tol` samples.
pub fn score(
    reference: &[i64],
    det: &[i64],
    tol: i64,
    fs: f64,
    keep_offsets: bool,
) -> DetectionScore {
    let to_ms = 1000.0 / fs as f32;
    let mut s = DetectionScore {
        n_ref: reference.len() as u64,
        n_det: det.len() as u64,
        ..Default::default()
    };
    let (mut i, mut j) = (0usize, 0usize);
    while i < reference.len() && j < det.len() {
        let d = det[j] - reference[i];
        if d < -tol {
            s.fp += 1;
            j += 1;
        } else if d > tol {
            s.fn_ += 1;
            i += 1;
        } else {
            // Prefer the closer of the two candidate detections for this reference.
            if j + 1 < det.len() {
                let d2 = det[j + 1] - reference[i];
                if d2.abs() < d.abs() && d2.abs() <= tol {
                    s.fp += 1;
                    j += 1;
                    continue;
                }
            }
            s.tp += 1;
            if keep_offsets {
                s.offsets.push(d as f32 * to_ms);
            }
            i += 1;
            j += 1;
        }
    }
    s.fn_ += (reference.len() - i) as u64;
    s.fp += (det.len() - j) as u64;
    s
}

pub fn percentile(sorted: &[f32], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)] as f64
}

pub fn mean(v: &[f32]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
}

pub fn stddev(v: &[f32]) -> f64 {
    if v.len() < 2 {
        return f64::NAN;
    }
    let m = mean(v);
    (v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}
