#!/usr/bin/env bash
# Regenerate every number in reports/PHASE-1.md.
#
# TRAIN selects, TEST scores. The two are never mixed: no parameter in the
# engine was chosen by looking at a TEST result.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=target/release/ecg-eval
OUT=reports/results
mkdir -p "$OUT"
command -v cargo >/dev/null || . "$HOME/.cargo/env"
cargo build --release >/dev/null

SEL="mitdb,svdb,nsrdb,stdb,qtdb,ltdb"     # parameter-selection corpora

echo "==> QRS detection, TRAIN (selection set)"
$BIN qrs --zone TRAIN --sources "$SEL" --per-record > "$OUT/qrs_train.txt" 2>&1

echo "==> QRS detection, TEST (sealed)"
$BIN qrs --zone TEST --sources ALL --per-record > "$OUT/qrs_test.txt" 2>&1
for s in mitdb svdb incartdb edb stdb qtdb ltdb nsrdb sddb afdb; do
  $BIN qrs --zone TEST --sources "$s" > "$OUT/qrs_test_$s.txt" 2>&1
done
# INCART is 12-lead; a chest patch approximates lead II, not lead I.
$BIN qrs --zone TEST --sources incartdb --lead 1 > "$OUT/qrs_test_incartdb_leadII.txt" 2>&1

echo "==> noise detection, MIT noise-stress protocol"
$BIN quality --zone ALL --sources nstdb > "$OUT/quality_nstdb.txt" 2>&1
$BIN qfeat   --zone ALL --sources nstdb > "$OUT/quality_features.txt" 2>&1

echo "==> AF, rhythm logic isolated (reference beats)"
$BIN af --zone DEV  --sources afdb --beats reference --per-record > "$OUT/af_dev_reference.txt" 2>&1
$BIN af --zone TEST --sources afdb --beats reference --per-record > "$OUT/af_test_reference.txt" 2>&1

echo "==> AF, end to end (our detector, real pipeline)"
$BIN af --zone TEST --sources afdb --beats detected --per-record > "$OUT/af_test_detected.txt" 2>&1

echo "==> AF false alarms on AF-free normal sinus"
$BIN af --zone TRAIN --sources nsrdb --beats detected --assume-af-free nsrdb --per-record > "$OUT/af_falsealarm_train.txt" 2>&1
$BIN af --zone TEST  --sources nsrdb --beats detected --assume-af-free nsrdb --per-record > "$OUT/af_falsealarm_test.txt"  2>&1

echo "==> AF model fit (TRAIN only; prints per-feature AUC and coefficients)"
$BIN fit-af --zone TRAIN --sources afdb,ltafdb,nsrdb --assume-af-free nsrdb \
    --beats reference --stride 16 --iters 800 --l2 0.015 > "$OUT/af_fit.txt" 2>&1

echo "==> quality against human annotation (BUT QDB)"
$BIN butqdb --zone DEV  --sources butqdb --threads 4 --per-record > "$OUT/butqdb_dev.txt"  2>&1
$BIN butqdb --zone TEST --sources butqdb --threads 4 --per-record > "$OUT/butqdb_test.txt" 2>&1

echo "==> beat classification, classification isolated (reference beats)"
$BIN beats --zone TEST --sources mitdb,svdb,incartdb --beats reference --per-record > "$OUT/beats_test_reference.txt" 2>&1
for s in mitdb svdb incartdb; do
  $BIN beats --zone TEST --sources "$s" --beats reference > "$OUT/beats_test_$s.txt" 2>&1
done
# INCART is 12-lead; a chest patch approximates lead II, not lead I.
$BIN beats --zone TEST --sources incartdb --beats reference --lead 1 > "$OUT/beats_test_incartdb_leadII.txt" 2>&1

echo "==> beat classification, end to end"
# INCART is left out of the pooled figure and reported on its own below. It is
# 12-lead, and one `--lead` cannot be right for three corpora at once: pooling it
# at lead I means pooling a lead in which our own detector misses half the beats,
# and the resulting "ventricular precision" measures the lead choice.
$BIN beats --zone TEST --sources mitdb,svdb --per-record > "$OUT/beats_test_detected.txt" 2>&1
$BIN beats --zone TEST --sources incartdb --lead 1 --per-record > "$OUT/beats_test_detected_incartdb_leadII.txt" 2>&1

echo "==> beat model fit (TRAIN only; per-feature AUC and the tree ensembles)"
$BIN fit-beats --zone TRAIN --sources mitdb,svdb --beats reference --gbdt \
    --depth 4 --trees 120 --emit /dev/null > "$OUT/beats_fit.txt" 2>&1

echo "==> wave delineation (fitted on the LUDB development half)"
$BIN delineate --zone DEV  --sources ludb > "$OUT/delineate_dev_ludb.txt"   2>&1
$BIN delineate --zone TEST --sources ludb > "$OUT/delineate_test_ludb.txt"  2>&1
$BIN delineate --zone TEST --sources qtdb > "$OUT/delineate_test_qtdb.txt"  2>&1

echo "==> pacemaker detection (one training record; rule is argued, not fitted)"
$BIN pacing --zone TEST --sources mitdb --include-paced --per-record > "$OUT/pacing_test_mitdb.txt" 2>&1
$BIN pacing --zone TEST --sources sddb --per-record > "$OUT/pacing_test_sddb.txt" 2>&1
$BIN pacing --zone TEST --sources nsrdb,svdb --per-record > "$OUT/pacing_unpaced.txt" 2>&1

echo "==> electrode failure (false-positive bound; no corpus labels it)"
$BIN leadoff --zone TEST --sources mitdb,nsrdb,svdb --per-record > "$OUT/leadoff_clinical.txt" 2>&1
$BIN leadoff --zone DEV  --sources butqdb  > "$OUT/leadoff_butqdb_dev.txt"  2>&1
$BIN leadoff --zone TEST --sources butqdb  > "$OUT/leadoff_butqdb_test.txt" 2>&1

echo "==> ventricular morphology review queue"
$BIN clusters --zone TEST  --sources mitdb  > "$OUT/clusters_test_mitdb.txt"  2>&1
$BIN clusters --zone TEST  --sources svdb   > "$OUT/clusters_test_svdb.txt"   2>&1
$BIN clusters --zone TRAIN --sources ltafdb > "$OUT/clusters_train_ltafdb.txt" 2>&1

# The internal patch corpus is private and may not be present. It is the only
# data here from the deployment domain, so it is run when it is there and
# skipped in a sentence when it is not.
if [ -f manifests/internal.json ]; then
  echo "==> patch corpus: recovering the gain the containers do not carry"
  $BIN patch --manifest manifests/internal.json --sources atheart-backup \
      --zone TEST --probes 10 --probe-s 600 \
      --emit-gains "$OUT/patch_gain_test.json" > "$OUT/patch_gain_test.txt" 2>&1
else
  echo "==> patch corpus: absent, skipped"
fi

echo "==> asystole census (which step loses one, on the corpus that has them)"
$BIN asystole --zone TRAIN --sources ltafdb --per-record > "$OUT/asystole_ltafdb.txt" 2>&1
$BIN asystole --zone TEST  --sources mitdb  --per-record > "$OUT/asystole_mitdb.txt"  2>&1
$BIN asystole --zone TEST  --sources nsrdb  --per-record > "$OUT/asystole_nsrdb.txt"  2>&1

echo "==> episode detection"
$BIN episodes --zone TEST  --sources mitdb  > "$OUT/episodes_test_mitdb.txt"  2>&1
$BIN episodes --zone TEST  --sources nsrdb  > "$OUT/episodes_test_nsrdb.txt"  2>&1
$BIN episodes --zone TRAIN --sources ltafdb > "$OUT/episodes_train_ltafdb.txt" 2>&1

echo "==> ventricular fibrillation (held out within TRAIN; no sealed set exists)"
$BIN vf --zone TRAIN --sources vfdb,cudb --holdout-every 3 --holdout-take > "$OUT/vf_heldout.txt" 2>&1
$BIN vf --zone TEST  --sources nsrdb > "$OUT/vf_normal_sinus.txt" 2>&1

echo "==> throughput"
$BIN stages --zone ALL --sources afdb --records 04936 > "$OUT/stages.txt" 2>&1
$BIN serve  --zone ALL --sources mitdb --records 100 --channels 512 --minutes 2 --fs 250 --threads 4 > "$OUT/serve_4core.txt" 2>&1
$BIN bench  --zone ALL --sources mitdb --records 100 --channels 256 --minutes 5 --fs 250 > "$OUT/bench_20core.txt" 2>&1
$BIN serve  --zone ALL --sources mitdb --records 100 --channels 512 --minutes 2 --fs 250 --threads 4 > "$OUT/serve_4core.txt" 2>&1
$BIN bench  --zone ALL --sources mitdb --records 100 --channels 256 --minutes 5 --fs 250 --threads 4 > "$OUT/bench_4core.txt" 2>&1

echo "results in $OUT"
