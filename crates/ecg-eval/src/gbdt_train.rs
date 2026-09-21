//! Gradient-boosted tree training, for the binary beat detectors.
//!
//! Newton boosting on the logistic loss with histogram split finding. Small and
//! self-contained on purpose: the model has to ship inside this binary with no
//! runtime and no loader, so the trainer emits Rust source rather than a format
//! something else has to parse.

/// A node in the form the runtime consumes.
#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub feature: u8,
    pub threshold: f32,
    pub left: u32,
    pub right: u32,
    pub value: f32,
}

pub const LEAF: u8 = 0xFF;

pub struct Model {
    pub bias: f32,
    pub nodes: Vec<Node>,
    pub roots: Vec<u32>,
}

impl Model {
    pub fn raw(&self, x: &[f32]) -> f32 {
        let mut sum = self.bias;
        for &root in &self.roots {
            let mut i = root as usize;
            loop {
                let n = self.nodes[i];
                if n.feature == LEAF {
                    sum += n.value;
                    break;
                }
                i = if x[n.feature as usize] <= n.threshold {
                    n.left as usize
                } else {
                    n.right as usize
                };
            }
        }
        sum
    }
}

pub struct TrainConfig {
    pub trees: usize,
    pub depth: usize,
    pub learning_rate: f64,
    pub bins: usize,
    /// Newton-step damping; also the leaf shrinkage.
    pub lambda: f64,
    /// A split must leave at least this much Hessian mass on each side.
    pub min_child_weight: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            trees: 120,
            depth: 4,
            learning_rate: 0.15,
            bins: 48,
            lambda: 1.0,
            min_child_weight: 8.0,
        }
    }
}

/// Per-feature bin edges, from quantiles of the training data.
struct Binner {
    edges: Vec<Vec<f32>>,
}

impl Binner {
    fn fit(x: &[Vec<f32>], n_features: usize, bins: usize) -> Binner {
        let mut edges = Vec::with_capacity(n_features);
        for k in 0..n_features {
            let mut v: Vec<f32> = x.iter().map(|r| r[k]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut e = Vec::with_capacity(bins);
            for b in 1..bins {
                let idx = (b as f64 / bins as f64 * (v.len() - 1) as f64) as usize;
                let t = v[idx];
                if e.last().map(|&l: &f32| l < t).unwrap_or(true) {
                    e.push(t);
                }
            }
            edges.push(e);
        }
        Binner { edges }
    }

    fn bin(&self, k: usize, v: f32) -> u8 {
        // Quantile edges are sorted; a linear scan over at most `bins` of them is
        // faster than a branchy binary search at this size.
        let e = &self.edges[k];
        let mut b = 0usize;
        while b < e.len() && v > e[b] {
            b += 1;
        }
        b as u8
    }
}

struct Split {
    feature: usize,
    bin: usize,
    gain: f64,
}

/// Train one binary ensemble. `w` weights each row; `y` is 0 or 1.
/// Train one binary ensemble over `allowed` features only.
///
/// A detector's evidence is its own. The bank exists because the questions do
/// not share a cost function; they do not always share a feature either, and a
/// feature that is noise for one question is something for its trees to overfit
/// rather than something they can ignore for free.
pub fn train(x: &[Vec<f32>], y: &[f32], w: &[f64], allowed: &[usize], cfg: &TrainConfig) -> Model {
    let n_features = x.first().map(|r| r.len()).unwrap_or(0);
    let n = x.len();
    let binner = Binner::fit(x, n_features, cfg.bins);
    let binned: Vec<Vec<u8>> = (0..n)
        .map(|i| (0..n_features).map(|k| binner.bin(k, x[i][k])).collect())
        .collect();

    // Class-balanced weights, so the ensemble is not bought with silence.
    let sum_pos: f64 = y
        .iter()
        .zip(w)
        .filter(|(yi, _)| **yi > 0.5)
        .map(|(_, wi)| wi)
        .sum();
    let sum_neg: f64 = y
        .iter()
        .zip(w)
        .filter(|(yi, _)| **yi <= 0.5)
        .map(|(_, wi)| wi)
        .sum();
    let mut cw: Vec<f64> = y
        .iter()
        .zip(w)
        .map(|(yi, wi)| {
            wi * if *yi > 0.5 {
                0.5 / sum_pos.max(1e-12)
            } else {
                0.5 / sum_neg.max(1e-12)
            }
        })
        .collect();
    // Rescale so the weights sum to the row count. `lambda` and
    // `min_child_weight` are both in units of Hessian mass, so leaving the
    // weights normalised to one would put the whole tree's mass below any
    // sensible minimum and every split would be rejected - which is exactly what
    // happened: 120 trees, 120 nodes, not one of them split.
    let total: f64 = cw.iter().sum();
    let scale = n as f64 / total.max(1e-12);
    for v in cw.iter_mut() {
        *v *= scale;
    }

    let bias = 0.0f32; // balanced weights put the prior at even odds
    let mut f: Vec<f64> = vec![bias as f64; n];
    let mut nodes: Vec<Node> = Vec::new();
    let mut roots: Vec<u32> = Vec::new();

    let mut idx: Vec<u32> = (0..n as u32).collect();
    for _ in 0..cfg.trees {
        let mut g = vec![0.0f64; n];
        let mut h = vec![0.0f64; n];
        for i in 0..n {
            let p = 1.0 / (1.0 + (-f[i]).exp());
            g[i] = cw[i] * (p - y[i] as f64);
            h[i] = cw[i] * p * (1.0 - p);
        }
        let root = nodes.len() as u32;
        roots.push(root);
        nodes.push(Node {
            feature: LEAF,
            threshold: 0.0,
            left: 0,
            right: 0,
            value: 0.0,
        });
        idx.sort_unstable();
        grow(
            &mut nodes,
            root as usize,
            &mut idx,
            0,
            n,
            &binned,
            &binner,
            &g,
            &h,
            allowed,
            cfg,
            0,
        );
        // Apply the new tree.
        let tree = Model {
            bias: 0.0,
            nodes: nodes.clone(),
            roots: vec![root],
        };
        for i in 0..n {
            f[i] += cfg.learning_rate * tree.raw(&x[i]) as f64;
        }
    }

    Model { bias, nodes, roots }
}

/// Grow one node over `idx[lo..hi]`, partitioning in place.
#[allow(clippy::too_many_arguments)]
fn grow(
    nodes: &mut Vec<Node>,
    node: usize,
    idx: &mut Vec<u32>,
    lo: usize,
    hi: usize,
    binned: &[Vec<u8>],
    binner: &Binner,
    g: &[f64],
    h: &[f64],
    allowed: &[usize],
    cfg: &TrainConfig,
    depth: usize,
) {
    let (gs, hs) = sums(idx, lo, hi, g, h);
    let leaf_value = -gs / (hs + cfg.lambda);
    nodes[node] = Node {
        feature: LEAF,
        threshold: 0.0,
        left: 0,
        right: 0,
        value: leaf_value as f32,
    };
    if depth >= cfg.depth || hi - lo < 2 {
        return;
    }

    let Some(split) = best_split(idx, lo, hi, binned, g, h, allowed, cfg, gs, hs) else {
        return;
    };
    if split.gain <= 0.0 {
        return;
    }

    // Partition idx[lo..hi] by the chosen bin.
    let (k, b) = (split.feature, split.bin);
    let mut left = lo;
    let mut right = hi;
    while left < right {
        if (binned[idx[left] as usize][k] as usize) <= b {
            left += 1;
        } else {
            right -= 1;
            idx.swap(left, right);
        }
    }
    if left == lo || left == hi {
        return;
    }

    let threshold = binner.edges[k].get(b).copied().unwrap_or(f32::MAX);
    let l = nodes.len() as u32;
    nodes.push(Node {
        feature: LEAF,
        threshold: 0.0,
        left: 0,
        right: 0,
        value: 0.0,
    });
    let r = nodes.len() as u32;
    nodes.push(Node {
        feature: LEAF,
        threshold: 0.0,
        left: 0,
        right: 0,
        value: 0.0,
    });
    nodes[node] = Node {
        feature: k as u8,
        threshold,
        left: l,
        right: r,
        value: 0.0,
    };

    grow(
        nodes,
        l as usize,
        idx,
        lo,
        left,
        binned,
        binner,
        g,
        h,
        allowed,
        cfg,
        depth + 1,
    );
    grow(
        nodes,
        r as usize,
        idx,
        left,
        hi,
        binned,
        binner,
        g,
        h,
        allowed,
        cfg,
        depth + 1,
    );
}

fn sums(idx: &[u32], lo: usize, hi: usize, g: &[f64], h: &[f64]) -> (f64, f64) {
    let mut gs = 0.0;
    let mut hs = 0.0;
    for &i in &idx[lo..hi] {
        gs += g[i as usize];
        hs += h[i as usize];
    }
    (gs, hs)
}

#[allow(clippy::too_many_arguments)]
fn best_split(
    idx: &[u32],
    lo: usize,
    hi: usize,
    binned: &[Vec<u8>],
    g: &[f64],
    h: &[f64],
    allowed: &[usize],
    cfg: &TrainConfig,
    gs: f64,
    hs: f64,
) -> Option<Split> {
    let parent = gs * gs / (hs + cfg.lambda);
    let mut best: Option<Split> = None;
    let mut gh = vec![0.0f64; cfg.bins];
    let mut hh = vec![0.0f64; cfg.bins];
    for &k in allowed {
        gh.iter_mut().for_each(|v| *v = 0.0);
        hh.iter_mut().for_each(|v| *v = 0.0);
        for &i in &idx[lo..hi] {
            let b = binned[i as usize][k] as usize;
            gh[b] += g[i as usize];
            hh[b] += h[i as usize];
        }
        let (mut gl, mut hl) = (0.0, 0.0);
        for b in 0..cfg.bins - 1 {
            gl += gh[b];
            hl += hh[b];
            let (gr, hr) = (gs - gl, hs - hl);
            if hl < cfg.min_child_weight || hr < cfg.min_child_weight {
                continue;
            }
            let gain = gl * gl / (hl + cfg.lambda) + gr * gr / (hr + cfg.lambda) - parent;
            if best.as_ref().map(|s| gain > s.gain).unwrap_or(gain > 0.0) {
                best = Some(Split {
                    feature: k,
                    bin: b,
                    gain,
                });
            }
        }
    }
    best
}

/// Shortest decimal that round-trips to the same `f32`.
///
/// Printing a fixed number of places emits digits an `f32` cannot hold, which is
/// both noise in the diff and a lint the generated file would otherwise have to
/// suppress.
fn f32s(v: f32) -> String {
    let s = format!("{v:?}");
    if s.contains('.') || s.contains('e') || s.contains("NaN") || s.contains("inf") {
        s
    } else {
        format!("{s}.0")
    }
}

/// Emit the ensemble as Rust source for `ecg_beats::gbdt`.
pub fn emit(name: &str, m: &Model) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "    pub const {name}: GbdtModel = GbdtModel {{\n        bias: {},\n        nodes: &{}_NODES,\n        roots: &{}_ROOTS,\n    }};\n",
        f32s(m.bias), name, name
    ));
    s.push_str(&format!(
        "    static {name}_NODES: [Node; {}] = [\n",
        m.nodes.len()
    ));
    for n in &m.nodes {
        if n.feature == LEAF {
            s.push_str(&format!("        Node::leaf({}),\n", f32s(n.value)));
        } else {
            s.push_str(&format!(
                "        Node::split({}, {}, {}, {}),\n",
                n.feature,
                f32s(n.threshold),
                n.left,
                n.right
            ));
        }
    }
    s.push_str("    ];\n");
    s.push_str(&format!(
        "    static {name}_ROOTS: [u32; {}] = [",
        m.roots.len()
    ));
    for (i, r) in m.roots.iter().enumerate() {
        s.push_str(&format!("{}{}", if i > 0 { ", " } else { "" }, r));
    }
    s.push_str("];\n");
    s
}
