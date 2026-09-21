//! Fixed-capacity streaming windows. Every one of these is O(1) per sample and
//! allocates only at construction — nothing in the steady-state path allocates.

/// Power-of-two ring buffer used as a delay line and look-back store.
#[derive(Debug, Clone)]
pub struct Ring {
    buf: Vec<f32>,
    mask: usize,
    /// Total samples ever pushed; `pos & mask` is the next write slot.
    pub pos: u64,
}

impl Ring {
    pub fn with_capacity(min_cap: usize) -> Self {
        let cap = min_cap.next_power_of_two().max(2);
        Ring {
            buf: vec![0.0; cap],
            mask: cap - 1,
            pos: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, x: f32) {
        self.buf[(self.pos as usize) & self.mask] = x;
        self.pos += 1;
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Sample at absolute index `i` (must be within the last `capacity()` pushes).
    #[inline(always)]
    pub fn at(&self, i: u64) -> f32 {
        self.buf[(i as usize) & self.mask]
    }

    /// `delay` samples back from the most recent push (`delay = 0` is the newest).
    #[inline(always)]
    pub fn back(&self, delay: usize) -> f32 {
        self.buf[((self.pos as usize).wrapping_sub(1 + delay)) & self.mask]
    }

    /// Index and value of the maximum over the absolute range `[from, to)`.
    pub fn argmax(&self, from: u64, to: u64) -> (u64, f32) {
        let mut bi = from;
        let mut bv = f32::NEG_INFINITY;
        for i in from..to {
            let v = self.at(i);
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        (bi, bv)
    }

    /// Same as `argmax` but on |x|, for fiducial placement on a band-passed signal
    /// where the dominant deflection may be negative.
    pub fn argmax_abs(&self, from: u64, to: u64) -> (u64, f32) {
        let mut bi = from;
        let mut bv = f32::NEG_INFINITY;
        for i in from..to {
            let v = self.at(i).abs();
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        (bi, bv)
    }

    pub fn reset(&mut self) {
        self.buf.iter_mut().for_each(|v| *v = 0.0);
        self.pos = 0;
    }
}

/// Moving-window integrator: a running sum over the last `n` samples, divided by `n`.
///
/// The running sum is kept in `f64`; an `f32` accumulator drifts badly over the
/// hours-long records this engine is built for.
#[derive(Debug, Clone)]
pub struct MovingAverage {
    buf: Vec<f32>,
    idx: usize,
    sum: f64,
    n: usize,
    filled: usize,
}

impl MovingAverage {
    pub fn new(n: usize) -> Self {
        let n = n.max(1);
        MovingAverage {
            buf: vec![0.0; n],
            idx: 0,
            sum: 0.0,
            n,
            filled: 0,
        }
    }

    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        let old = self.buf[self.idx];
        self.buf[self.idx] = x;
        self.idx += 1;
        if self.idx == self.n {
            self.idx = 0;
        }
        self.sum += x as f64 - old as f64;
        if self.filled < self.n {
            self.filled += 1;
        }
        (self.sum / self.filled as f64) as f32
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn reset(&mut self) {
        self.buf.iter_mut().for_each(|v| *v = 0.0);
        self.idx = 0;
        self.sum = 0.0;
        self.filled = 0;
    }
}

/// Five-point central derivative, the Pan-Tompkins slope stage.
/// `y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) / 8`, scaled to be sample-rate aware.
#[derive(Debug, Clone, Default)]
pub struct Derivative5 {
    z: [f32; 4],
    scale: f32,
}

impl Derivative5 {
    pub fn new(fs: f64) -> Self {
        // The classic kernel is defined at 200 Hz; rescale so the slope units are
        // comparable across sampling rates.
        Derivative5 {
            z: [0.0; 4],
            scale: (fs / 200.0) as f32 / 8.0,
        }
    }

    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = (2.0 * x + self.z[0] - self.z[2] - 2.0 * self.z[3]) * self.scale;
        self.z[3] = self.z[2];
        self.z[2] = self.z[1];
        self.z[1] = self.z[0];
        self.z[0] = x;
        y
    }

    pub fn reset(&mut self) {
        self.z = [0.0; 4];
    }

    /// Group delay in samples (the kernel is centred on `x[n-2]`).
    pub const fn delay(&self) -> usize {
        2
    }
}

/// Exponentially weighted mean and variance, for adaptive thresholds.
#[derive(Debug, Clone, Copy)]
pub struct Ewma {
    pub mean: f32,
    alpha: f32,
    primed: bool,
}

impl Ewma {
    pub fn new(alpha: f32) -> Self {
        Ewma {
            mean: 0.0,
            alpha,
            primed: false,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, x: f32) -> f32 {
        if self.primed {
            self.mean += self.alpha * (x - self.mean);
        } else {
            self.mean = x;
            self.primed = true;
        }
        self.mean
    }

    pub fn is_primed(&self) -> bool {
        self.primed
    }

    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.primed = false;
    }
}

/// Running sum of squares over a fixed window; the basis of every band-power feature.
#[derive(Debug, Clone)]
pub struct MovingPower {
    inner: MovingAverage,
}

impl MovingPower {
    pub fn new(n: usize) -> Self {
        MovingPower {
            inner: MovingAverage::new(n),
        }
    }

    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        self.inner.process(x * x)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Sliding min/max over a fixed window in amortised O(1) via monotonic deques.
/// Used for peak-to-peak amplitude, which is far more robust than variance when
/// a single motion spike dominates the window.
#[derive(Debug, Clone)]
pub struct MovingExtrema {
    n: usize,
    t: u64,
    // (index, value), decreasing for max, increasing for min
    max_d: std::collections::VecDeque<(u64, f32)>,
    min_d: std::collections::VecDeque<(u64, f32)>,
}

impl MovingExtrema {
    pub fn new(n: usize) -> Self {
        let n = n.max(1);
        MovingExtrema {
            n,
            t: 0,
            max_d: std::collections::VecDeque::with_capacity(n.min(1024)),
            min_d: std::collections::VecDeque::with_capacity(n.min(1024)),
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> (f32, f32) {
        while self.max_d.back().is_some_and(|&(_, v)| v <= x) {
            self.max_d.pop_back();
        }
        self.max_d.push_back((self.t, x));
        while self.min_d.back().is_some_and(|&(_, v)| v >= x) {
            self.min_d.pop_back();
        }
        self.min_d.push_back((self.t, x));

        let floor = self.t.saturating_sub(self.n as u64 - 1);
        while self.max_d.front().is_some_and(|&(i, _)| i < floor) {
            self.max_d.pop_front();
        }
        while self.min_d.front().is_some_and(|&(i, _)| i < floor) {
            self.min_d.pop_front();
        }
        self.t += 1;
        (self.min_d.front().unwrap().1, self.max_d.front().unwrap().1)
    }

    pub fn reset(&mut self) {
        self.t = 0;
        self.max_d.clear();
        self.min_d.clear();
    }
}
