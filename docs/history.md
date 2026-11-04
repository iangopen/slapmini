# History

## 2026-11-03 – Session 1: sim core + calibration harness
- Built the Cargo workspace (sim, calib, sim-wasm), params.json (all placeholder), the 2.5D physics, the calib CLI and a Nelder-Mead fitter.
- Synthetic slide/bounce scenarios: the fitter recovers friction and restitution to about 1e-9.
- Determinism verified across runs and across native vs wasm. Benchmark about 4.4M steps/s (2 players).
- Installed Rust on the Windows machine. Used the GNU toolchain override because MSVC lacks the Windows SDK.

## 2026-11-03 – Session 2: feel lab
- Added web/ (Vite + TS) with a fixed-step loop, minimap renderer, live param panel, practice tools (approximate), record/replay in calib formats, and the ghost overlay.
- sim-wasm gained set_params/get_params/reset. params.json gained 5 display-only params.
- Playwright verify passes. A lab recording replays byte-identically and evaluates to about 0 error in native calib.
- README.md converted from UTF-16 to UTF-8.
