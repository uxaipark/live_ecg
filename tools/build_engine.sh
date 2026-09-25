#!/usr/bin/env bash
# Build the swappable engine: the single source file, the C library built from
# it, the header, and a conformance run of the result.
#
#   tools/build_engine.sh          regenerate dist/ecg_engine.rs, then build
#   tools/build_engine.sh --check  fail if dist/ecg_engine.rs is stale instead
#
# Outputs in dist/: ecg_engine.rs and ecg.h (committed), libecg.{a,so|dylib}
# and ecg_conformance (built).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"

cargo build --release -q -p ecg-eval
if [ "${1:-}" = "--check" ]; then
  ./target/release/ecg-eval amalgamate --check
else
  ./target/release/ecg-eval amalgamate
fi
cp crates/ecg-ffi/include/ecg.h dist/ecg.h

rustc --edition 2021 -O -C panic=unwind -C codegen-units=1 \
  --crate-name ecg --crate-type cdylib --crate-type staticlib \
  dist/ecg_engine.rs --out-dir dist

case "$(uname -s)" in
  Darwin) LIB=dist/libecg.dylib; LDL="" ;;
  *)      LIB=dist/libecg.so;    LDL="-ldl" ;;
esac
cc -O2 -Wall -Wextra -I dist -o dist/ecg_conformance tools/ecg_conformance.c -lm $LDL
dist/ecg_conformance "$LIB"
