#!/usr/bin/env python3
"""Build `manifests/internal.json` from the internal patch corpus.

The manifest is not in the repository and should not be: it is 24,043 rows of
per-exam metadata from a private corpus, and it is derived, so a copy here
would be a second place for the zone assignment to be wrong. The corpora
themselves are not in the repository either, for the same reason.

What it carries is what the harness needs and nothing else - the hashed record
and subject identifiers, the zone, where the signal is, and the device's own
summary of the recording so a detected rate has something to be checked
against. Sample counts come from each container's own metadata rather than
from the statistics table, because that is the number the reader will verify
against and the two have been known to disagree by a day.

    python3 tools/build_internal_manifest.py [--canonical DIR] [--out PATH]

Needs `pyarrow` and nothing else; it does not decode a single sample.
"""

import argparse
import concurrent.futures as futures
import json
import os
import pathlib
import zipfile

SLUG = "atheart-backup"
# Device-derived summary columns worth carrying. Everything else in
# `record_stats.parquet` stays there.
STATS = [
    "recording_hours",
    "analysis_hours",
    "hr_avg",
    "hr_min",
    "hr_max",
    "max_rr_sec",
    "beats_normal",
    "beats_ventricular",
    "beats_supraventricular",
]


def probe(job):
    root, rec = job
    try:
        with zipfile.ZipFile(root / rec["path"]) as z:
            meta = json.loads(z.read("zarr.json"))
        rec["n_samples"] = int(meta["shape"][0])
        rec["n_leads"] = int(meta["shape"][1])
        rec["leads"] = list(meta["attributes"].get("lead_names") or [])
        rec["fs"] = float(meta["attributes"].get("fs", 250.0))
    except Exception as exc:  # a container that cannot be opened is dropped
        rec["n_samples"] = 0
        rec["n_leads"] = 0
        rec["leads"] = []
        rec["error"] = str(exc)[:120]
    return rec


def main():
    here = pathlib.Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--canonical",
        default=os.environ.get("DEEP_ECG_CANONICAL", here / ".." / "deep_ecg" / "data" / "canonical"),
    )
    ap.add_argument("--out", default=here / "manifests" / "internal.json")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    root = pathlib.Path(args.canonical).resolve()
    corpus = root / SLUG
    splits = pq.read_table(corpus / "splits.parquet").to_pydict()
    table = pq.read_table(corpus / "record_stats.parquet")
    columns = [c for c in STATS if c in table.schema.names]
    stats = {
        uid: {c: table.column(c)[i].as_py() for c in columns}
        for i, uid in enumerate(table.column("record_uid").to_pylist())
    }

    jobs, absent = [], 0
    for i, uid in enumerate(splits["record_uid"]):
        rel = f"{SLUG}/{uid[:2]}/{uid}.zarr.zip"
        if not (root / rel).exists():
            absent += 1
            continue
        rec = {
            "record_uid": uid,
            "subject_uid": splits["subject_uid"][i],
            "source": SLUG,
            "record": uid,
            "fs": 250.0,
            "zone": splits["zone"][i],
            "path": rel,
            "labels_usable": bool(splits["labels_usable"][i]),
            "expert_reviewed": bool(splits["expert_reviewed"][i]),
            "expert_eval": bool(splits["expert_eval"][i]),
            "has_beats": (corpus / f"{uid[:2]}/{uid}.beats.parquet").exists(),
        }
        rec.update({k: v for k, v in (stats.get(uid) or {}).items() if v is not None})
        jobs.append((root, rec))

    with futures.ThreadPoolExecutor(16) as pool:
        records = list(pool.map(probe, jobs))
    unreadable = [r for r in records if r["n_samples"] == 0]

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=1))
    hours = sum(r["n_samples"] for r in records) / 250 / 3600
    zones = {}
    for r in records:
        zones[r["zone"]] = zones.get(r["zone"], 0) + 1
    print(f"{len(records)} records, {hours:,.0f} h ({hours / 8766:.1f} years)")
    print("zones", zones)
    print("signal absent", absent, " unreadable", len(unreadable))
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
