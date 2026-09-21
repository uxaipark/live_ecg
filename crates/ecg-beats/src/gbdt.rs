//! Compact gradient-boosted decision trees, inference only.
//!
//! # Why trees here and a linear model for AF
//!
//! The AF question is close to linear in its features: every statistic says
//! "more irregular" in the same direction, and a weighted sum expresses that
//! well. The beat question is not. A ventricular beat is abnormal in shape
//! **and** early **and** followed by a compensatory pause; a supraventricular
//! beat is early with *normal* shape and *no* compensatory pause. Those are
//! conjunctions, and a linear model cannot represent a conjunction - measured,
//! it reached an AUC of 0.956 for the ventricular detector and could not convert
//! that into usable precision at a 6% positive rate.
//!
//! Trees represent conjunctions natively, need no feature scaling, and cost a
//! handful of comparisons. This is still a small model - a few hundred nodes -
//! that runs on any target without an accelerator, which is the point.
//!
//! The model is a static table so it links into the binary with no loading step
//! and no allocation, on the server and on the patch alike.

/// One node. A leaf is marked by [`Node::LEAF`] in `feature`.
#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub feature: u8,
    pub threshold: f32,
    /// Index of the child taken when `value <= threshold`.
    pub left: u32,
    pub right: u32,
    /// Leaf contribution; unused in an internal node.
    pub value: f32,
}

impl Node {
    pub const LEAF: u8 = 0xFF;

    pub const fn leaf(value: f32) -> Node {
        Node {
            feature: Node::LEAF,
            threshold: 0.0,
            left: 0,
            right: 0,
            value,
        }
    }

    pub const fn split(feature: u8, threshold: f32, left: u32, right: u32) -> Node {
        Node {
            feature,
            threshold,
            left,
            right,
            value: 0.0,
        }
    }
}

/// An additive ensemble over a fixed-length feature vector.
#[derive(Debug, Clone, Copy)]
pub struct GbdtModel {
    /// Log-odds the ensemble starts from.
    pub bias: f32,
    pub nodes: &'static [Node],
    /// Index of each tree's root in `nodes`.
    pub roots: &'static [u32],
}

impl GbdtModel {
    pub const EMPTY: GbdtModel = GbdtModel {
        bias: 0.0,
        nodes: &[],
        roots: &[],
    };

    /// Sum of the leaves reached in every tree, as a log-odds.
    #[inline]
    pub fn raw(&self, x: &[f32]) -> f32 {
        let mut sum = self.bias;
        for &root in self.roots {
            let mut i = root as usize;
            // Bounded by the node count so a malformed table cannot spin.
            for _ in 0..64 {
                let n = &self.nodes[i];
                if n.feature == Node::LEAF {
                    sum += n.value;
                    break;
                }
                let v = x[n.feature as usize];
                i = if v <= n.threshold {
                    n.left as usize
                } else {
                    n.right as usize
                };
            }
        }
        sum
    }

    #[inline]
    pub fn probability(&self, x: &[f32]) -> f32 {
        1.0 / (1.0 + (-self.raw(x)).exp())
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }
}
