//! Evaluation and benchmarking harness for the live ECG engine.
//!
//! Exposed as a library as well as a binary so the regression tests can call the
//! same code paths the reports are generated from. A test that re-implements the
//! measurement is testing itself.

pub mod af_eval;
pub mod af_fit;
pub mod beat_eval;
pub mod beat_fit;
pub mod butqdb;
pub mod diag;
pub mod gbdt_train;
pub mod manifest;
pub mod metrics;
pub mod qrs_eval;
pub mod quality_eval;
pub mod rhythm_eval;
pub mod rhythm_ref;
pub mod sweep;
pub mod throughput;

pub const DEFAULT_MANIFEST: &str = "manifests/records.json";

/// Flat option bag. A CLI crate would buy little here and the harness is the one
/// place where a dependency has to be justified against build time on the Pi.
pub struct Opts {
    pub manifest: String,
    pub zones: Vec<String>,
    pub sources: Vec<String>,
    pub records: Vec<String>,
    pub lead: usize,
    pub limit: Option<usize>,
    pub tol_ms: f64,
    pub skip_sec: f64,
    pub threads: Option<usize>,
    pub per_record: bool,
    pub json: Option<String>,
    pub gate: bool,
    pub raw: Vec<(String, String)>,
}

impl Opts {
    pub fn parse(args: &[String]) -> Opts {
        let mut o = Opts {
            manifest: DEFAULT_MANIFEST.to_string(),
            zones: vec!["TRAIN".into()],
            sources: vec!["mitdb".into()],
            records: vec![],
            lead: 0,
            limit: None,
            tol_ms: 150.0,
            skip_sec: 0.0,
            threads: None,
            per_record: false,
            json: None,
            gate: false,
            raw: vec![],
        };
        let mut i = 0;
        while i < args.len() {
            let key = args[i].trim_start_matches("--").to_string();
            if key == "per-record" {
                o.per_record = true;
                i += 1;
                continue;
            }
            if key == "gate" {
                o.gate = true;
                i += 1;
                continue;
            }
            // A flag whose next token is another flag (or nothing) is boolean.
            // Consuming that token unconditionally silently swallowed the option
            // that followed any bare flag, which looks exactly like the swallowed
            // option having no effect.
            let next_is_flag = args.get(i + 1).map(|a| a.starts_with("--")).unwrap_or(true);
            if next_is_flag {
                o.raw.push((key, String::new()));
                i += 1;
                continue;
            }
            let val = args.get(i + 1).cloned().unwrap_or_default();
            match key.as_str() {
                "manifest" => o.manifest = val.clone(),
                "zone" => o.zones = split(&val),
                "sources" => o.sources = split(&val),
                "records" => o.records = split(&val),
                "lead" => o.lead = val.parse().unwrap_or(0),
                "limit" => o.limit = val.parse().ok(),
                "tol-ms" => o.tol_ms = val.parse().unwrap_or(150.0),
                "skip-sec" => o.skip_sec = val.parse().unwrap_or(0.0),
                "threads" => o.threads = val.parse().ok(),
                "json" => o.json = Some(val.clone()),
                _ => {}
            }
            o.raw.push((key, val));
            i += 2;
        }
        o
    }

    /// Copy these options with `overrides` replacing matching keys.
    pub fn clone_with(&self, overrides: &[(&str, &str)]) -> Opts {
        let mut o = Opts {
            manifest: self.manifest.clone(),
            zones: self.zones.clone(),
            sources: self.sources.clone(),
            records: self.records.clone(),
            lead: self.lead,
            limit: self.limit,
            tol_ms: self.tol_ms,
            skip_sec: self.skip_sec,
            threads: Some(1),
            per_record: false,
            json: None,
            gate: self.gate,
            raw: self.raw.clone(),
        };
        for (k, v) in overrides {
            o.raw.retain(|(rk, _)| rk != k);
            o.raw.push((k.to_string(), v.to_string()));
        }
        o
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.raw
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.parse().ok())
    }

    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.raw
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.parse().ok())
    }

    pub fn get_i32(&self, key: &str) -> Option<i32> {
        self.raw
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| v.parse().ok())
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.raw
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn select(&self) -> std::io::Result<Vec<manifest::RecordEntry>> {
        let root = manifest::data_root(
            self.get_str("data-root"),
            std::path::Path::new(&self.manifest),
        );
        let all = manifest::load(&self.manifest, &root)?;
        let zone_all = self.zones.iter().any(|z| z == "ALL");
        let src_all = self.sources.iter().any(|s| s == "ALL");
        let mut v: Vec<_> = all
            .into_iter()
            .filter(|r| zone_all || self.zones.contains(&r.zone))
            .filter(|r| src_all || self.sources.contains(&r.source))
            .filter(|r| self.records.is_empty() || self.records.contains(&r.record))
            .collect();
        v.sort_by(|a, b| (&a.source, &a.record).cmp(&(&b.source, &b.record)));
        // Optional internal split, for corpora whose zone assignment has no DEV
        // records. Deterministic by position in the sorted list, applied only
        // within whatever zone was already selected - it never reaches across
        // the TRAIN/TEST boundary.
        if let Some(every) = self.get_usize("holdout-every").filter(|&n| n > 1) {
            let take_holdout = self.raw.iter().any(|(k, _)| k == "holdout-take");
            v = v
                .into_iter()
                .enumerate()
                .filter(|(i, _)| (i % every == 0) == take_holdout)
                .map(|(_, r)| r)
                .collect();
        }
        if let Some(n) = self.limit {
            v.truncate(n);
        }
        Ok(v)
    }

    pub fn install_thread_pool(&self) {
        if let Some(n) = self.threads {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global();
        }
    }
}

fn split(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}
