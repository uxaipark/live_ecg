//! The array itself: zarr v3 metadata, and chunks addressed through it.

use crate::blosc;
use crate::zip::Archive;
use crate::{Error, Result};
use serde::Deserialize;
use std::path::Path;

/// What the corpus records about a recording beside the samples.
///
/// `units` is `adc` for every record in the internal corpus and the gain is
/// not stored, so these counts are not millivolts and nothing here pretends
/// otherwise. Every threshold in the engine but one is dimensionless; the one
/// that is not is the saturation level, and it is measured rather than
/// converted.
#[derive(Debug, Clone, Deserialize)]
pub struct Attrs {
    pub fs: f64,
    #[serde(default)]
    pub units: String,
    #[serde(default)]
    pub lead_names: Vec<String>,
    #[serde(default)]
    pub n_samples: u64,
}

#[derive(Deserialize, Default)]
struct ChunkGridConfig {
    chunk_shape: Vec<u64>,
}

#[derive(Deserialize)]
struct Named<C> {
    name: String,
    #[serde(default)]
    configuration: Option<C>,
}

#[derive(Deserialize, Default)]
struct ChunkKeyConfig {
    #[serde(default)]
    separator: String,
}

#[derive(Deserialize)]
struct Metadata {
    shape: Vec<u64>,
    data_type: String,
    chunk_grid: Named<ChunkGridConfig>,
    #[serde(default)]
    chunk_key_encoding: Option<Named<ChunkKeyConfig>>,
    #[serde(default)]
    codecs: Vec<serde_json::Value>,
    attributes: Attrs,
}

pub struct ZarrArray {
    zip: Archive,
    /// Samples, then leads.
    shape: [u64; 2],
    chunk: u64,
    separator: String,
    attrs: Attrs,
    scratch: Vec<u8>,
}

impl ZarrArray {
    pub fn open(path: &Path) -> Result<ZarrArray> {
        let zip = Archive::open(path)?;
        let meta = zip
            .get("zarr.json")
            .ok_or_else(|| Error::Parse("no zarr.json in the container".into()))?;
        let meta: Metadata =
            serde_json::from_slice(meta).map_err(|e| Error::Parse(e.to_string()))?;

        if meta.data_type != "int32" {
            return Err(Error::Unsupported(format!("dtype {}", meta.data_type)));
        }
        if meta.chunk_grid.name != "regular" {
            return Err(Error::Unsupported(format!(
                "chunk grid {}; sharded stores are not handled",
                meta.chunk_grid.name
            )));
        }
        // A codec this reader does not implement would otherwise be discovered
        // as corrupt samples rather than as an error.
        for c in &meta.codecs {
            let name = c.get("name").and_then(|n| n.as_str()).unwrap_or("?");
            if !matches!(name, "bytes" | "blosc") {
                return Err(Error::Unsupported(format!("codec {name}")));
            }
            if name == "bytes" {
                let endian = c
                    .get("configuration")
                    .and_then(|k| k.get("endian"))
                    .and_then(|e| e.as_str())
                    .unwrap_or("little");
                if endian != "little" {
                    return Err(Error::Unsupported(format!("{endian}-endian samples")));
                }
            }
        }
        if meta.shape.len() != 2 {
            return Err(Error::Unsupported(format!(
                "{}-dimensional array; this reader expects samples by leads",
                meta.shape.len()
            )));
        }
        let grid = meta
            .chunk_grid
            .configuration
            .ok_or_else(|| Error::Parse("chunk grid has no shape".into()))?;
        if grid.chunk_shape.len() != 2 || grid.chunk_shape[1] != meta.shape[1] {
            return Err(Error::Unsupported(
                "chunks divide the lead axis; this reader expects whole leads".into(),
            ));
        }
        let separator = meta
            .chunk_key_encoding
            .as_ref()
            .and_then(|k| {
                if k.name == "default" {
                    k.configuration.as_ref().map(|c| c.separator.clone())
                } else {
                    None
                }
            })
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "/".to_string());

        Ok(ZarrArray {
            zip,
            shape: [meta.shape[0], meta.shape[1]],
            chunk: grid.chunk_shape[0],
            separator,
            attrs: meta.attributes,
            scratch: Vec::new(),
        })
    }

    pub fn attrs(&self) -> &Attrs {
        &self.attrs
    }

    pub fn fs(&self) -> f64 {
        self.attrs.fs
    }

    pub fn n_samples(&self) -> u64 {
        self.shape[0]
    }

    pub fn n_leads(&self) -> u64 {
        self.shape[1]
    }

    /// Samples per chunk. The engine is fed a chunk at a time, so a fourteen-day
    /// recording never exists in memory at once.
    pub fn chunk_samples(&self) -> u64 {
        self.chunk
    }

    pub fn n_chunks(&self) -> u64 {
        self.shape[0].div_ceil(self.chunk)
    }

    /// Decode chunk `i` into `out`, interleaved by lead as it is stored.
    ///
    /// A chunk that is absent from the container is not an error: zarr writes
    /// nothing for a chunk that is entirely the fill value, so the absence
    /// means zeros and is filled in as such.
    pub fn read_chunk(&mut self, i: u64, out: &mut Vec<i32>) -> Result<()> {
        if i >= self.n_chunks() {
            return Err(Error::Parse(format!("chunk {i} past the end")));
        }
        let rows = self.chunk.min(self.shape[0] - i * self.chunk);
        let values = (rows * self.shape[1]) as usize;
        out.clear();
        out.resize(values, 0);

        let key = format!("c{sep}{i}{sep}0", sep = self.separator);
        // Borrowed as two disjoint fields rather than through `self`, so the
        // mapped bytes and the decode buffer can be held at once.
        let (zip, scratch) = (&self.zip, &mut self.scratch);
        let Some(raw) = zip.get(&key) else {
            return Ok(());
        };
        blosc::decode(raw, scratch)?;

        // A chunk at the end of the array is stored full and padded, so the
        // decoded frame is allowed to be longer than the rows that exist.
        if scratch.len() < values * 4 {
            return Err(Error::Parse(format!(
                "chunk {i} decoded to {} bytes, needed {}",
                scratch.len(),
                values * 4
            )));
        }
        for (v, b) in out.iter_mut().zip(scratch.chunks_exact(4)) {
            *v = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        Ok(())
    }

    /// One lead of a sample range, as the engine wants it.
    pub fn read_lead(&mut self, lead: usize, from: u64, to: u64) -> Result<Vec<i32>> {
        let to = to.min(self.shape[0]);
        if lead as u64 >= self.shape[1] || from > to {
            return Err(Error::Parse(format!("lead {lead}, samples {from}..{to}")));
        }
        let leads = self.shape[1] as usize;
        let mut out = Vec::with_capacity((to - from) as usize);
        let mut chunk = Vec::new();
        for c in (from / self.chunk)..=(to.saturating_sub(1) / self.chunk) {
            self.read_chunk(c, &mut chunk)?;
            let base = c * self.chunk;
            let lo = from.saturating_sub(base) as usize;
            let hi = ((to - base) as usize).min(chunk.len() / leads);
            out.extend(
                chunk[lo * leads..hi * leads]
                    .iter()
                    .skip(lead)
                    .step_by(leads),
            );
        }
        Ok(out)
    }
}
