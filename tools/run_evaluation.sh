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

echo "==> beat classification, classification isolated (reference beats)"
$BIN beats --zone TEST --sources mitdb,svdb,incartdb --beats reference --per-record > "$OUT/beats_test_reference.txt" 2>&1
for s in mitdb svdb incartdb; do
  $BIN beats --zone TEST --sources "$s" --beats reference > "$OUT/beats_test_$s.txt" 2>&1
done
# INCART is 12-lead; a chest patch approximates lead II, not lead I.
$BIN beats --zone TEST --sources incartdb --beats reference --lead 1 > "$OUT/beats_test_incartdb_leadII.txt" 2>&1

echo "==> beat classification, end to end"
$BIN beats --zone TEST --sources mitdb,svdb,incartdb --per-record > "$OUT/beats_test_detected.txt" 2>&1

echo "==> beat model fit (TRAIN only; per-feature AUC and the tree ensembles)"
$BIN fit-beats --zone TRAIN --sources mitdb,svdb --beats reference --gbdt \
    --depth 4 --trees 120 --emit /dev/null > "$OUT/beats_fit.txt" 2>&1

echo "==> throughput"
$BIN stages --zone ALL --sources afdb --records 04936 > "$OUT/stages.txt" 2>&1
$BIN bench  --zone ALL --sources mitdb --records 100 --channels 256 --minutes 5 --fs 250 > "$OUT/bench_20core.txt" 2>&1
$BIN bench  --zone ALL --sources mitdb --records 100 --channels 256 --minutes 5 --fs 250 --threads 4 > "$OUT/bench_4core.txt" 2>&1

echo "results in $OUT"
