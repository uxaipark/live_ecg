//! Command-line front end for the evaluation harness.

use ecg_eval::{
    af_eval, af_fit, asystole_eval, beat_eval, beat_fit, butqdb, cluster_eval, delin_eval, diag,
    leadoff_eval, patch_eval,
    pacing_eval, qrs_eval, quality_eval, rhythm_eval, serve, sweep, throughput, vf_eval, Opts,
    DEFAULT_MANIFEST,
};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    let opts = Opts::parse(&args[1..]);

    let result = match cmd {
        "qrs" => qrs_eval::run(&opts),
        "sweep" => sweep::run(&opts),
        "bench" => throughput::run(&opts),
        "serve" => serve::run(&opts),
        "stages" => throughput::stages(&opts),
        "diag" => diag::run(&opts),
        "trace" => diag::trace(&opts),
        "quality" => quality_eval::run(&opts),
        "butqdb" => butqdb::run(&opts),
        "episodes" => rhythm_eval::run(&opts),
        "vf" => vf_eval::run(&opts),
        "fit-vf" => vf_eval::fit(&opts),
        "af" => af_eval::run(&opts),
        "beats" => beat_eval::run(&opts),
        "delineate" => delin_eval::run(&opts),
        "clusters" => cluster_eval::run(&opts),
        "leadoff" => leadoff_eval::run(&opts),
        "pacing" => pacing_eval::run(&opts),
        "asystole" => asystole_eval::run(&opts),
        "patch" => patch_eval::run(&opts),
        "fit-beats" => beat_fit::run(&opts),
        "beat-dump" => beat_fit::dump(&opts),
        "fit-af" => af_fit::run(&opts),
        "af-dump" => af_fit::dump(&opts),
        "qfeat" => quality_eval::dump_features(&opts),
        "qrows" => quality_eval::dump_rows(&opts),
        _ => {
            eprintln!(
                "usage: ecg-eval <qrs|sweep|bench|diag|quality|af|beats|fit-af|fit-beats> [options]\n\
                 \n\
                 common options\n  \
                   --manifest PATH     record manifest (default {DEFAULT_MANIFEST})\n  \
                   --data-root PATH    WFDB corpora root (or $DEEP_ECG_RAW)\n  \
                   --zone Z[,Z]        TRAIN | DEV | TEST | ALL      (default TRAIN)\n  \
                   --sources S[,S]     dataset slugs, or ALL         (default mitdb)\n  \
                   --records R[,R]     explicit native record names\n  \
                   --lead N            lead index                    (default 0)\n  \
                   --limit N           cap the record count\n  \
                   --tol-ms MS         match tolerance               (default 150)\n  \
                   --skip-sec S        drop the first S seconds from scoring\n  \
                   --threads N         worker threads\n  \
                   --per-record        print a line per record\n  \
                   --holdout-every N   keep every Nth record out (internal split)\n  \
                   --holdout-take      select those held-out records instead\n  \
                   --json PATH         write the full result as JSON\n  \
                 \n\
                 detector overrides\n  \
                   --bp-lo HZ  --bp-hi HZ  --bp-order N  --integ-ms MS\n  \
                   --refractory-ms MS  --twave-ms MS  --thr-frac F\n  \
                   --searchback F  --refine-ms MS  --bias N\n  \
                 \n\
                 front-end overrides\n  \
                   --hp-hz HZ  --lp-hz HZ  --mains off|50|60|auto  --gate\n"
            );
            return ExitCode::from(2);
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
