//! Cross-check against the zarr library on a real container.
//!
//! A format reader fails by returning plausible numbers, not by crashing: a
//! wrong shuffle or an off-by-one block offset yields samples that still look
//! like a signal. So the check is byte-exact against an independent decoder -
//! `zarr` 3.3.0 with `numcodecs` 0.16.5 - on four ranges chosen to exercise the
//! parts that differ: the first chunk, a whole interior chunk, a range inside a
//! chunk, and the short chunk at the end.
//!
//! Skips when the corpus is absent, the way the evaluation guards do.

use ecg_zarr::ZarrArray;
use std::path::PathBuf;

/// `$DEEP_ECG_CANONICAL`, else the sibling checkout the harness assumes.
fn canonical() -> PathBuf {
    if let Ok(p) = std::env::var("DEEP_ECG_CANONICAL") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../deep_ecg/data/canonical")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("../deep_ecg/data/canonical"))
}

const RECORD: &str = "atheart-backup/00/00d60597e47a8002.zarr.zip";

#[test]
fn decodes_the_same_samples_as_the_zarr_library() {
    let path = canonical().join(RECORD);
    if !path.exists() {
        eprintln!("SKIPPED: no internal corpus. Set DEEP_ECG_CANONICAL to enable.");
        return;
    }
    let mut z = ZarrArray::open(&path).expect("open");
    assert_eq!(z.n_samples(), 302_305_200);
    assert_eq!(z.n_leads(), 1);
    assert_eq!(z.fs(), 250.0);
    assert_eq!(z.chunk_samples(), 75_000);
    assert_eq!(z.attrs().lead_names, vec!["PATCH1".to_string()]);

    /// A range and everything the other decoder said about it: the first six
    /// samples, the last three, the sum, the smallest and the largest.
    struct Probe {
        from: u64,
        to: u64,
        first: [i32; 6],
        last: [i32; 3],
        sum: i64,
        lo: i32,
        hi: i32,
    }
    let p = |from, to, first, last, sum, lo, hi| Probe {
        from,
        to,
        first,
        last,
        sum,
        lo,
        hi,
    };
    let probes = [
        p(
            0,
            75_000,
            [1, -94, -92, -93, -92, -93],
            [659, 657, 657],
            47_873_289,
            -99,
            685,
        ),
        p(
            75_000,
            150_000,
            [658, 656, 657, 656, 656, 656],
            [645, 645, 644],
            48_089_925,
            603,
            695,
        ),
        p(
            1_000_000,
            1_000_500,
            [7286, 7286, 7286, 7286, 7285, 7285],
            [7285, 7285, 7283],
            3_643_204,
            7268,
            7301,
        ),
        p(
            302_200_000,
            302_305_200,
            [10915895, 10915893, 10915894, 10915894, 10915893, 10915892],
            [10916429, 10916434, 10916435],
            1_148_407_665_806,
            10_915_849,
            10_916_508,
        ),
    ];

    for k in probes {
        let from = k.from;
        let v = z.read_lead(0, from, k.to).expect("read");
        assert_eq!(v.len(), (k.to - from) as usize, "length at {from}");
        assert_eq!(&v[..6], &k.first, "first samples at {from}");
        assert_eq!(&v[v.len() - 3..], &k.last, "last samples at {from}");
        assert_eq!(
            v.iter().map(|&x| x as i64).sum::<i64>(),
            k.sum,
            "sum at {from}"
        );
        assert_eq!(*v.iter().min().unwrap(), k.lo, "min at {from}");
        assert_eq!(*v.iter().max().unwrap(), k.hi, "max at {from}");
    }
}

/// A range that starts and ends inside different chunks has to stitch them,
/// and the stitch is where an off-by-one lives. Checked against the whole-chunk
/// reads rather than against another library, so it holds without the corpus's
/// exact numbers.
#[test]
fn a_range_across_a_chunk_boundary_matches_the_chunks_it_spans() {
    let path = canonical().join(RECORD);
    if !path.exists() {
        return;
    }
    let mut z = ZarrArray::open(&path).expect("open");
    let n = z.chunk_samples();
    let whole = z.read_lead(0, 0, 2 * n).expect("read");
    let spanning = z.read_lead(0, n - 5, n + 5).expect("read");
    assert_eq!(spanning, whole[(n - 5) as usize..(n + 5) as usize]);
}
