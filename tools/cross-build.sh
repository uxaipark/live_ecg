#!/usr/bin/env bash
# Build the standalone benchmark for every deployment target.
#
# No cross toolchain is required. Every crate here is pure Rust, so a foreign
# target needs only a linker, and `rust-lld` ships with the toolchain. The Linux
# targets link static musl binaries, which also removes the glibc version
# question: one file runs on a Raspberry Pi OS image and on whatever the
# customer already has.
#
#   rustup target add <target>
#   ./tools/cross-build.sh [--strip]
set -euo pipefail
cd "$(dirname "$0")/.."
command -v cargo >/dev/null || . "$HOME/.cargo/env"

TARGETS=(
  aarch64-unknown-linux-musl        # Raspberry Pi 5, and any 64-bit ARM Linux
  armv7-unknown-linux-musleabihf    # 32-bit ARM Linux, older Pi images
  x86_64-unknown-linux-musl         # low-end PC, and most servers
  aarch64-apple-ios                 # iPhone, iPad
  aarch64-apple-darwin              # Apple silicon
  x86_64-apple-darwin               # Intel Mac
)

STRIP=""
[ "${1:-}" = "--strip" ] && STRIP=1

printf "%-34s %-8s %s\n" target result size
for t in "${TARGETS[@]}"; do
  if ! rustup target list --installed | grep -qx "$t"; then
    printf "%-34s %-8s %s\n" "$t" skipped "rustup target add $t"
    continue
  fi
  if out=$(cargo build --release -p ecg-bench --target "$t" 2>&1); then
    bin="target/$t/release/ecg-bench"
    [ -n "$STRIP" ] && strip "$bin" 2>/dev/null || true
    printf "%-34s %-8s %s\n" "$t" ok "$(ls -l "$bin" | awk '{printf "%.1f MB", $5/1048576}')"
  else
    printf "%-34s %-8s %s\n" "$t" FAILED "$(echo "$out" | grep -m1 '^error' | cut -c1-50)"
  fi
done

cat <<'NOTE'

Android needs the NDK, which does not ship with Rust:
  rustup target add aarch64-linux-android
  export NDK=$ANDROID_HOME/ndk/<version>/toolchains/llvm/prebuilt/<host>/bin
  CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$NDK/aarch64-linux-android24-clang \
    cargo build --release -p ecg-bench --target aarch64-linux-android

Then copy the binary to the target and run it there:
  ./ecg-bench --channels 64 --seconds 120 --threads 4
NOTE
