use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Where the WFDB corpora live.
///
/// The manifest stores paths relative to this, so the repository is not bound to
/// one machine. Resolution order: `--data-root`, then `$DEEP_ECG_RAW`, then a
/// sibling checkout next to the repository.
///
/// The fallback is resolved against the manifest's own location, not the working
/// directory. `cargo test` runs with the package directory as the working
/// directory while the binary runs from the repository root, so a
/// cwd-relative default silently found nothing under test - and a test that
/// finds no data skips, which looks exactly like a test that passed.
pub fn data_root(explicit: Option<&str>, manifest: &Path) -> PathBuf {
    data_roots(explicit, manifest).remove(0)
}

/// Every place the records might be, best first.
///
/// There are two corpora now and they live in different trees: the public ones
/// under `data/raw` as WFDB, the internal patch corpus under `data/canonical`
/// as zarr. A manifest names records, not where they are kept, so the root is
/// resolved by looking - the first candidate that actually holds the manifest's
/// first record wins. Asking the caller to know which tree a manifest refers to
/// is a question with one right answer that the program can find itself.
pub fn data_roots(explicit: Option<&str>, manifest: &Path) -> Vec<PathBuf> {
    if let Some(p) = explicit {
        return vec![PathBuf::from(p)];
    }
    let mut out = Vec::new();
    for var in ["DEEP_ECG_RAW", "DEEP_ECG_CANONICAL"] {
        if let Ok(p) = std::env::var(var) {
            out.push(PathBuf::from(p));
        }
    }
    for sibling in ["deep_ecg/data/raw", "deep_ecg/data/canonical"] {
        out.push(sibling_root(manifest, sibling));
    }
    out
}

fn sibling_root(manifest: &Path, under: &str) -> PathBuf {
    // <repo>/manifests/records.json -> <repo>/../deep_ecg/data/raw.
    // Canonicalised first: `Path::parent` strips a trailing component without
    // resolving `..`, so walking up a path that contains one lands in the wrong
    // place.
    let manifest = manifest.canonicalize();
    let manifest = manifest
        .as_deref()
        .unwrap_or(Path::new("manifests/records.json"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .map(|p| p.join(under))
        .unwrap_or_else(|| PathBuf::from("..").join(under))
}

#[derive(Debug, Clone, Deserialize)]
pub struct RecordEntry {
    pub record_uid: String,
    pub subject_uid: String,
    pub source: String,
    pub record: String,
    pub fs: f64,
    pub n_samples: usize,
    pub n_leads: usize,
    pub leads: Vec<String>,
    pub zone: String,
    /// Header path relative to the data root.
    pub path: String,
    /// Whatever else the manifest carried. Corpora differ in what is known
    /// about a record - the internal one brings its zone flags and the
    /// device's own summary of the recording - and putting those here keeps
    /// them available without giving every corpus every other corpus's fields.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_json::Value>,
    #[serde(skip)]
    root: PathBuf,
}

impl RecordEntry {
    /// Where the record's signal is. Named for WFDB because that is what it
    /// was, and kept because every caller that reads a header uses it; the
    /// internal corpus's entries point at a `.zarr.zip` through the same field.
    pub fn hea_path(&self) -> PathBuf {
        self.root.join(&self.path)
    }

    pub fn signal_path(&self) -> PathBuf {
        self.root.join(&self.path)
    }

    pub fn flag(&self, key: &str) -> bool {
        self.extra.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
    }

    pub fn number(&self, key: &str) -> Option<f64> {
        self.extra.get(key).and_then(|v| v.as_f64())
    }

    pub fn ann_path(&self, ext: &str) -> PathBuf {
        self.hea_path().with_extension(ext)
    }
}

pub fn load(path: &str, root: &Path) -> std::io::Result<Vec<RecordEntry>> {
    load_from(path, std::slice::from_ref(&root.to_path_buf()))
}

pub fn load_from(path: &str, roots: &[PathBuf]) -> std::io::Result<Vec<RecordEntry>> {
    let text = std::fs::read_to_string(path)?;
    let mut v: Vec<RecordEntry> = serde_json::from_str(&text)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    // Records with no signal file (annotation-only) cannot be scored.
    v.retain(|r| r.n_samples > 0 && r.n_leads > 0);
    let Some(first) = v.first().cloned() else {
        return Ok(v);
    };
    let root = roots
        .iter()
        .find(|r| r.join(&first.path).exists())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "no data for {} under {}. Set --data-root, $DEEP_ECG_RAW or \
                     $DEEP_ECG_CANONICAL.",
                    first.path,
                    roots
                        .iter()
                        .map(|r| r.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )
        })?;
    for r in v.iter_mut() {
        r.root = root.clone();
    }
    Ok(v)
}
