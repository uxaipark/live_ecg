// throwaway: delineated QRS duration by reference class, on one record
use ecg_beats::delineate::Delineator;
use ecg_pipeline::{PipelineConfig, Preprocessor};
use ecg_qrs::QrsEvent;
use ecg_wfdb::{read_signal, AnnotationFile, Header};

fn main() {
    let rec = std::env::args().nth(1).unwrap();
    let lead: usize = std::env::args()
        .nth(2)
        .unwrap_or("1".into())
        .parse()
        .unwrap();
    let hdr = Header::read(std::path::Path::new(&format!("{rec}.hea"))).unwrap();
    let ann = AnnotationFile::read(std::path::Path::new(&format!("{rec}.atr"))).unwrap();
    let sig = read_signal(&hdr, lead, 0, hdr.n_samples).unwrap();
    let cfg = PipelineConfig::new(hdr.fs);
    let mut pre = Preprocessor::new(cfg.preprocess);
    let d = cfg.delineate;
    let mut del = Delineator::new(d);
    del.set_delays(
        pre.group_delay_samples(d.qrs_ref_hz) + pre.qrs_group_delay_samples(d.qrs_ref_hz),
        pre.group_delay_samples(d.p_ref_hz) + pre.pt_group_delay_samples(d.p_ref_hz),
        pre.group_delay_samples(d.t_ref_hz) + pre.pt_group_delay_samples(d.t_ref_hz),
    );
    let marks: Vec<(i64, char)> = ann
        .annotations
        .iter()
        .filter(|a| a.sample >= 0)
        .map(|a| (a.sample, a.symbol))
        .collect();
    let mut recent: [Option<u64>; 3] = [None; 3];
    let mut sym: [char; 3] = ['?'; 3];
    let mut by: std::collections::BTreeMap<char, Vec<f32>> = Default::default();
    let mut next = 0usize;
    let to_ms = 1000.0 / hdr.fs as f32;
    for (i, &x) in sig.iter().enumerate() {
        let b = pre.process(x);
        del.push_sample(b.qrs, b.pt);
        while next < marks.len() && marks[next].0 <= i as i64 {
            let s = marks[next].1;
            next += 1;
            if !"NLRejAaJSVEFQ/f".contains(s) {
                continue;
            }
            recent = [recent[1], recent[2], Some(i as u64)];
            sym = [sym[1], sym[2], s];
            if let (Some(a), Some(m), Some(c)) = (recent[0], recent[1], recent[2]) {
                if let Some(w) = del.delineate(m, Some(m - a), Some(c - m)) {
                    by.entry(sym[1])
                        .or_default()
                        .push(w.qrs.duration_samples() as f32 * to_ms);
                }
            }
        }
    }
    print!("{}  ", rec.rsplit('/').next().unwrap());
    for (s, mut v) in by {
        if v.len() < 30 {
            continue;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        print!("{s}: n={} med={:.0}ms  ", v.len(), v[v.len() / 2]);
    }
    println!();
}
