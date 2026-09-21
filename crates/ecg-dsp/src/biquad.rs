//! Second-order IIR sections.
//!
//! Coefficients and state are `f64`: the baseline-wander high-pass sits at
//! ~0.5 Hz, i.e. a normalised frequency near 2e-3, where `f32` state in a
//! direct-form recursion loses audible precision. The recursion is serial and
//! cannot be vectorised within a channel anyway, so the wider state costs
//! nothing that matters. Samples cross the boundary as `f32`.

#[derive(Debug, Clone, Copy, Default)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    // transposed direct form II
    s1: f64,
    s2: f64,
}

impl Biquad {
    pub fn new(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        let inv = 1.0 / a0;
        Biquad {
            b0: b0 * inv,
            b1: b1 * inv,
            b2: b2 * inv,
            a1: a1 * inv,
            a2: a2 * inv,
            s1: 0.0,
            s2: 0.0,
        }
    }

    /// Low-pass, `q` = 1/sqrt(2) for a single Butterworth section.
    pub fn low_pass(fs: f64, f0: f64, q: f64) -> Self {
        let (w0, cw, alpha) = prewarp(fs, f0, q);
        let _ = w0;
        let b0 = (1.0 - cw) * 0.5;
        Biquad::new(b0, 1.0 - cw, b0, 1.0 + alpha, -2.0 * cw, 1.0 - alpha)
    }

    pub fn high_pass(fs: f64, f0: f64, q: f64) -> Self {
        let (_, cw, alpha) = prewarp(fs, f0, q);
        let b0 = (1.0 + cw) * 0.5;
        Biquad::new(b0, -(1.0 + cw), b0, 1.0 + alpha, -2.0 * cw, 1.0 - alpha)
    }

    /// Band-stop with a -3 dB width of `bw` Hz around `f0` (mains rejection).
    pub fn notch(fs: f64, f0: f64, bw: f64) -> Self {
        let q = f0 / bw;
        let (_, cw, alpha) = prewarp(fs, f0, q);
        Biquad::new(1.0, -2.0 * cw, 1.0, 1.0 + alpha, -2.0 * cw, 1.0 - alpha)
    }

    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = x as f64;
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y as f32
    }

    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    /// Prime the state so a constant input of `x` produces no startup transient.
    pub fn prime(&mut self, x: f32) {
        let x = x as f64;
        // steady-state y for a DC input x
        let y = (self.b0 + self.b1 + self.b2) / (1.0 + self.a1 + self.a2) * x;
        self.s1 = self.b1 * x - self.a1 * y + (self.b2 * x - self.a2 * y);
        self.s2 = self.b2 * x - self.a2 * y;
    }

    /// Phase response in radians at `f` Hz.
    pub fn phase(&self, fs: f64, f: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * f / fs;
        let (c1, s1) = (w.cos(), -w.sin());
        let (c2, s2) = ((2.0 * w).cos(), -(2.0 * w).sin());
        let nr = self.b0 + self.b1 * c1 + self.b2 * c2;
        let ni = self.b1 * s1 + self.b2 * s2;
        let dr = 1.0 + self.a1 * c1 + self.a2 * c2;
        let di = self.a1 * s1 + self.a2 * s2;
        ni.atan2(nr) - di.atan2(dr)
    }

    /// Group delay in samples at `f` Hz, by central difference of the phase.
    ///
    /// The fiducial placement downstream needs this: an IIR chain shifts the R
    /// wave by a frequency-dependent amount, and leaving that as a tuned constant
    /// would silently break whenever a corner frequency or the sample rate moves.
    pub fn group_delay(&self, fs: f64, f: f64) -> f64 {
        let df = (fs / 4096.0).min(f.max(1e-6) * 0.05).max(1e-6);
        let (a, b) = (self.phase(fs, f - df), self.phase(fs, f + df));
        let mut d = b - a;
        // Keep the difference on the principal branch before differentiating.
        while d > std::f64::consts::PI {
            d -= 2.0 * std::f64::consts::PI;
        }
        while d < -std::f64::consts::PI {
            d += 2.0 * std::f64::consts::PI;
        }
        let dw = 2.0 * std::f64::consts::PI * (2.0 * df) / fs;
        -d / dw
    }

    /// Magnitude response at `f` Hz, for tests and filter documentation.
    pub fn magnitude(&self, fs: f64, f: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * f / fs;
        let (c1, s1) = (w.cos(), -w.sin());
        let (c2, s2) = ((2.0 * w).cos(), -(2.0 * w).sin());
        let nr = self.b0 + self.b1 * c1 + self.b2 * c2;
        let ni = self.b1 * s1 + self.b2 * s2;
        let dr = 1.0 + self.a1 * c1 + self.a2 * c2;
        let di = self.a1 * s1 + self.a2 * s2;
        ((nr * nr + ni * ni) / (dr * dr + di * di)).sqrt()
    }
}

#[inline]
fn prewarp(fs: f64, f0: f64, q: f64) -> (f64, f64, f64) {
    let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
    let cw = w0.cos();
    let alpha = w0.sin() / (2.0 * q);
    (w0, cw, alpha)
}

/// Butterworth section Q factors for an order-`n` (even) filter.
pub fn butterworth_qs(n: usize) -> Vec<f64> {
    assert!(n.is_multiple_of(2) && n >= 2, "even order required");
    (0..n / 2)
        .map(|k| {
            let theta = std::f64::consts::PI * (2.0 * k as f64 + 1.0) / (2.0 * n as f64);
            1.0 / (2.0 * theta.cos())
        })
        .collect()
}

/// A cascade of second-order sections.
#[derive(Debug, Clone, Default)]
pub struct Cascade {
    sections: Vec<Biquad>,
}

impl Cascade {
    pub fn new(sections: Vec<Biquad>) -> Self {
        Cascade { sections }
    }

    pub fn butter_low_pass(fs: f64, f0: f64, order: usize) -> Self {
        Cascade::new(
            butterworth_qs(order)
                .into_iter()
                .map(|q| Biquad::low_pass(fs, f0, q))
                .collect(),
        )
    }

    pub fn butter_high_pass(fs: f64, f0: f64, order: usize) -> Self {
        Cascade::new(
            butterworth_qs(order)
                .into_iter()
                .map(|q| Biquad::high_pass(fs, f0, q))
                .collect(),
        )
    }

    pub fn band_pass(fs: f64, lo: f64, hi: f64, order: usize) -> Self {
        let mut s = Cascade::butter_high_pass(fs, lo, order).sections;
        s.extend(Cascade::butter_low_pass(fs, hi, order).sections);
        Cascade::new(s)
    }

    pub fn push(&mut self, b: Biquad) {
        self.sections.push(b);
    }

    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    #[inline(always)]
    pub fn process(&mut self, x: f32) -> f32 {
        let mut y = x;
        for s in self.sections.iter_mut() {
            y = s.process(y);
        }
        y
    }

    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        // Section-major: each section sweeps the whole block, so its coefficients
        // and state stay in registers instead of being reloaded per sample.
        output.copy_from_slice(input);
        for s in self.sections.iter_mut() {
            for y in output.iter_mut() {
                *y = s.process(*y);
            }
        }
    }

    pub fn reset(&mut self) {
        for s in self.sections.iter_mut() {
            s.reset();
        }
    }

    pub fn prime(&mut self, x: f32) {
        let mut v = x;
        for s in self.sections.iter_mut() {
            s.prime(v);
            v = s.process(v);
        }
    }

    pub fn magnitude(&self, fs: f64, f: f64) -> f64 {
        self.sections.iter().map(|s| s.magnitude(fs, f)).product()
    }

    /// Group delay in samples at `f` Hz; delays of cascaded sections add.
    pub fn group_delay(&self, fs: f64, f: f64) -> f64 {
        self.sections.iter().map(|s| s.group_delay(fs, f)).sum()
    }
}
