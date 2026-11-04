# History

## 2026-11-03 – Session 1: sim core + calibration harness
- Built the Cargo workspace (sim, calib, sim-wasm), params.json (all placeholder), the 2.5D physics, the calib CLI and a Nelder-Mead fitter.
- Synthetic slide/bounce scenarios: the fitter recovers friction and restitution to about 1e-9.
- Determinism verified across runs and across native vs wasm. Benchmark about 4.4M steps/s (2 players).
- Installed Rust on the Windows machine. Used the GNU toolchain override because MSVC lacks the Windows SDK.
