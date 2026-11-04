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

## 2026-11-04 – Session 3: tunneling fix, then measure/
- Tunneling stress test + swept net collision (see CLAUDE.md).
- Added measure/ (Python): map / track / qa / synth, the coordinate convention, synthetic-video tests, QA sheets.
- Found: the real rink's length/width ratio is ~1.57 (params 2.31); one "fixed" clip's camera moves; the recording duplicates frames.
- Feel lab gained a game view (rotated like the real camera) used to check a real track as a ghost.

## 2026-11-04 – Session 4: first real fits
- Ice-plane scale check (board outline is at ice level), puck ground reprojection (height parallax), measured rink markings and net_x.
- Clip-held-out staged fits of board_restitution, board_friction, ice_linear_drag, ice_friction_decel → params.fitted.json.
- Drag-law comparison: no quadratic term; an unexplained medium-speed dip in decel/v is real. Segmenter fixes found via overlays.
