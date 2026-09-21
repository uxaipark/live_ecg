//! Reference rhythm timeline from WFDB annotations.
//!
//! A rhythm annotation marks a *change*: `(AFIB` at sample X means the rhythm is
//! atrial fibrillation from X until the next rhythm annotation. The timeline is
//! therefore built by pairing consecutive markers, not by treating each as an
//! event.

use ecg_wfdb::AnnotationFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfLabel {
    Af,
    NonAf,
    /// Neither a positive nor a negative for an AF detector; left out of scoring.
    Excluded,
}

#[derive(Debug, Clone, Copy)]
pub struct RhythmSpan {
    pub start: i64,
    pub end: i64,
    pub label: AfLabel,
}

/// How atrial flutter is treated.
///
/// Flutter is organised, not fibrillating, so its RR series can be regular - an
/// RR-only detector has no way to call it AF and no reason to. But clinically
/// the two are often managed together and many published AFDB results pool them.
/// Both conventions are supported so the reported number says which one it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlutterPolicy {
    /// Primary: flutter is scored as neither positive nor negative.
    Exclude,
    /// Secondary: flutter counts as AF, matching the pooled convention.
    AsAf,
}

fn classify(aux: &str, policy: FlutterPolicy) -> Option<AfLabel> {
    // Only aux payloads starting with '(' are rhythm changes. Records also carry
    // aux notes such as `MISSB`, `PSE` and `M`, which mark events, not rhythms.
    let name = aux.strip_prefix('(')?.trim_end_matches('\0').trim();
    Some(match name {
        "AFIB" => AfLabel::Af,
        "AFL" => match policy {
            FlutterPolicy::Exclude => AfLabel::Excluded,
            FlutterPolicy::AsAf => AfLabel::Af,
        },
        _ => AfLabel::NonAf,
    })
}

/// Rhythm-change markers as the annotator wrote them, paired into spans.
///
/// The AF view above is one reading of these; the episode detectors need the
/// rest, so the raw names are exposed rather than collapsed at the source.
pub fn named_spans(ann: &AnnotationFile, n_samples: i64) -> Vec<(i64, i64, String)> {
    let mut marks: Vec<(i64, String)> = ann
        .annotations
        .iter()
        .filter(|a| a.sample >= 0)
        .filter_map(|a| {
            let name = a.aux.as_deref()?.strip_prefix('(')?;
            Some((a.sample, name.trim_end_matches('\0').trim().to_string()))
        })
        .collect();
    marks.sort_by_key(|(s, _)| *s);
    marks.dedup_by_key(|(s, _)| *s);
    let mut out = Vec::with_capacity(marks.len());
    for (i, (start, name)) in marks.iter().enumerate() {
        let end = marks.get(i + 1).map(|(s, _)| *s).unwrap_or(n_samples);
        if end > *start {
            out.push((*start, end, name.clone()));
        }
    }
    out
}

/// Point annotations that are events rather than rhythms - `PSE` for a pause,
/// `MISSB` for a missed beat. They carry no span, so they are matched by time.
pub fn point_events(ann: &AnnotationFile, name: &str) -> Vec<i64> {
    ann.annotations
        .iter()
        .filter(|a| a.sample >= 0)
        .filter(|a| a.aux.as_deref().map(|x| x.trim_end_matches('\0').trim()) == Some(name))
        .map(|a| a.sample)
        .collect()
}

/// Build the rhythm timeline covering `[0, n_samples)`.
pub fn spans(ann: &AnnotationFile, n_samples: i64, policy: FlutterPolicy) -> Vec<RhythmSpan> {
    let mut marks: Vec<(i64, AfLabel)> = ann
        .annotations
        .iter()
        .filter_map(|a| {
            let aux = a.aux.as_deref()?;
            classify(aux, policy).map(|l| (a.sample, l))
        })
        .collect();
    marks.sort_by_key(|&(s, _)| s);
    marks.dedup_by_key(|&mut (s, _)| s);

    let mut out = Vec::with_capacity(marks.len());
    for (i, &(start, label)) in marks.iter().enumerate() {
        let end = marks.get(i + 1).map(|&(s, _)| s).unwrap_or(n_samples);
        if end > start {
            out.push(RhythmSpan { start, end, label });
        }
    }
    out
}

/// Per-second label array over the whole record.
///
/// Seconds are the scoring unit throughout: they are long enough that the exact
/// sample of a rhythm transition does not matter, short enough that a
/// thirty-second clinical threshold stays meaningful, and they make the
/// duration-weighted totals that AF work is normally reported in straightforward.
pub fn per_second(spans: &[RhythmSpan], n_samples: i64, fs: f64) -> Vec<AfLabel> {
    let n_sec = (n_samples as f64 / fs).floor() as usize;
    // Anything before the first marker is unlabelled, not implicitly normal.
    let mut out = vec![AfLabel::Excluded; n_sec];
    for s in spans {
        let a = (s.start as f64 / fs).floor().max(0.0) as usize;
        let b = ((s.end as f64 / fs).ceil() as usize).min(n_sec);
        for v in out.iter_mut().take(b).skip(a) {
            *v = s.label;
        }
    }
    out
}

/// Contiguous runs of `wanted` lasting at least `min_s` seconds.
pub fn episodes(labels: &[AfLabel], wanted: AfLabel, min_s: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &l) in labels.iter().enumerate() {
        if l == wanted {
            start.get_or_insert(i);
        } else if let Some(s) = start.take() {
            if i - s >= min_s {
                out.push((s, i));
            }
        }
    }
    if let Some(s) = start {
        if labels.len() - s >= min_s {
            out.push((s, labels.len()));
        }
    }
    out
}
