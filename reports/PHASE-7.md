# live_ecg — Phase 7: cross-building for the deployment targets

## 1. What this phase does and does not deliver

**Delivered:** the engine builds and links for every deployment target, from a
workstation with no cross toolchain installed, and there is a self-contained
benchmark that can be copied onto a target and run there.

**Not delivered: any measurement on a target.** There is no Raspberry Pi, no
phone and no low-end PC attached to this machine, and no emulation available —
this host has no Rosetta and no container runtime. Every performance figure in
every report here is still from an Apple M1 Ultra.

That gap matters more than it might sound. Phase 6 found that per-shard capacity
is higher at 4 threads than at 20 — 30,700 channels against 22,500 — because
every channel walks its own ring buffers and twenty of them contend for the same
cache. **Capacity is bounded by memory bandwidth, not by core count.** A
Raspberry Pi 5 has four cores and a fraction of this machine's memory bandwidth,
so the honest position is that its capacity is unknown, and multiplying a
single-core figure by four would be guessing.

The benchmark exists so that someone with the hardware can close that gap in a
minute. It has not been closed.

---

## 2. Cross-building needs nothing installed

Every crate here is pure Rust. No C is compiled, so a foreign target needs only
a linker — and `rust-lld` ships with the toolchain. `rustup target add` is the
whole setup.

| target | deployment | binary |
|---|---|---|
| `aarch64-unknown-linux-musl` | Raspberry Pi 5, any 64-bit ARM Linux | 0.82 MB |
| `armv7-unknown-linux-musleabihf` | 32-bit ARM Linux, older Pi images | 0.81 MB |
| `x86_64-unknown-linux-musl` | low-end PC, most servers | 0.84 MB |
| `aarch64-apple-ios` | iPhone, iPad | 0.6 MB |
| `aarch64-apple-darwin` | Apple silicon | 0.6 MB |
| `x86_64-apple-darwin` | Intel Mac | 0.7 MB |

The Linux targets link **static musl** binaries. That is deliberate beyond
convenience: it removes the glibc version question, so one file runs on a
Raspberry Pi OS image and on whatever the customer already has, with no runtime
dependency to discover in the field.

The whole workspace cross-builds, not only the benchmark — including the
evaluation harness, which pulls in memory mapping and a thread pool.

Android is the exception and needs the NDK, which does not ship with Rust.
`tools/cross-build.sh` prints the incantation.

CI builds every Linux target on Linux and both Apple targets on macOS. If that
job ever needs a C compiler, something has acquired a C dependency, and that is
worth finding out from a build failure rather than from a deployment.

---

## 3. The benchmark carries its own sanity check

`ecg-bench` synthesises its own signal — a P wave, a complex, a T wave, on a
respiratory baseline with noise — so it needs no corpus and no data files. It
measures cost, which is what changes between hosts; detection quality is a
property of the algorithm and is measured against real corpora elsewhere.

It also reports whether it found the beats it put in:

```
detection sanity     0.972 of the 4608 beats present  (ok)
```

That line is there because a target where the arithmetic differs enough to
change detection would otherwise show up as a plausible-looking capacity number
and nothing else. It is a cheap way for the number to refuse to look fine when
it is not.

```
./ecg-bench --channels 64 --seconds 120 --threads 4
```

---

## 4. Numerical consistency

Detection is identical across optimisation levels on the host — sensitivity,
precision and the fiducial offset all agree to the digit at `opt-level` 0, 2 and
3, with and without link-time optimisation. Rust does not contract
multiply-adds by default, so the arithmetic is the same arithmetic everywhere.

**What is not guaranteed** is the platform maths library. `exp`, `ln`, `sqrt`,
`sin` and `atan2` come from the target's libm and may differ in the last unit in
the last place. Those appear in the logistic models, the biquad designer and the
entropy features, where a last-bit difference is far below any threshold in the
engine. It is a reason to run the sanity check on a new target, not a reason to
expect trouble.

---

## 5. Memory

About 88 KB of state per channel, measured on the host: 97 MB resident for 1,024
channels against 7.3 MB for one. Dominated by the ring buffers — the quality
monitor's four taps, the detector's three retained traces, the beat template,
the fibrillation window.

A Raspberry Pi 5 with 8 GB could hold the state for far more channels than its
memory bandwidth will let it process, so memory is not expected to be the
binding constraint. That expectation is also unmeasured.

---

## 6. What is left

- **Run the benchmark on a Raspberry Pi 5, a phone and a low-end PC.** This is
  the actual deliverable of the phase and it needs hardware.
- **Android** needs the NDK wired into CI.
- **iOS builds, but nothing links it into an app.** A static library and a C
  header would be the next step for a phone deployment; the binary target proves
  the code compiles and links, not that it is packaged.
- **Deployment capacity should be stated per target**, not inherited from a
  workstation figure, once there is a target to state it for.
