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
    if let Some(p) = explicit {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("DEEP_ECG_RAW") {
        return PathBuf::from(p);
    }
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
        .map(|p| p.join("deep_ecg/data/raw"))
        .unwrap_or_else(|| PathBuf::from("../deep_ecg/data/raw"))
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
    #[serde(skip)]
    root: PathBuf,
}

impl RecordEntry {
    pub fn hea_path(&self) -> PathBuf {
        self.root.join(&self.path)
    }

    pub fn ann_path(&self, ext: &str) -> PathBuf {
        self.hea_path().with_extension(ext)
    }
}

pub fn load(path: &str, root: &Path) -> std::io::Result<Vec<RecordEntry>> {
    let text = std::fs::read_to_string(path)?;
    let mut v: Vec<RecordEntry> = serde_json::from_str(&text)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    // Records with no signal file (annotation-only) cannot be scored.
    v.retain(|r| r.n_samples > 0 && r.n_leads > 0);
    for r in v.iter_mut() {
        r.root = root.to_path_buf();
    }
    if let Some(first) = v.first() {
        if !first.hea_path().exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "no WFDB data under {}. Set --data-root or $DEEP_ECG_RAW.",
                    root.display()
                ),
            ));
        }
    }
    Ok(v)
}
