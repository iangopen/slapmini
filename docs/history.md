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

## 2026-11-04 – Session 6: fitted defaults, puck/player size, figures
- Goal-speed clip not on clip-data: the absolute scale stays a placeholder.
- params.json carries the fitted puck params (params.placeholder.json keeps the old file); segment initial speeds now from a
  30-row fit (the 10-row fit had ~8% noise), which made the clip-held-out fits agree.
- puck_radius / player_radius measured from frames; a Coulomb board-friction model was tested and not adopted.
- Fit figures on the sim's own rink in measure/results/fits/ (README figure included).

## 2026-11-04 – Integration of s7 + s7b: fitted player/stick defaults, absolute scale applied
- Merged origin/s7 and origin/s7b; merged their fits into params.json (`measure.merge_models`); applied the goal-readout
  scale ×1.0873 to params and all real/skating data (`measure.rescale`); bot shot constants re-measured; `verify-skate` added.
- CLAUDE.md was restructured into current-state sections. The detailed per-session records below were moved here verbatim
  (meters in them are PLACEHOLDER meters unless stated: multiply by 1.0873 for real meters).

### Detailed session records (moved from CLAUDE.md)

#### Verified (2026-11-03, Windows, rustc 1.99.1 GNU toolchain)
- `cargo build --release` and `cargo test --workspace` pass (14 tests).
- Fitter recovery: friction 1.0 (start 1.4) recovered as 1.0000000008, board restitution 0.7 (start 0.49) as 0.7000000002, both in 66 Nelder-Mead iterations.
- Determinism: two 100k-step runs give byte-identical states. **Native and wasm (Node) hashes match after 100k steps** (`c93c94a203a58d2d`).
- Benchmark (release, single thread): about 4.4M steps/s with 2 players and about 1.7M steps/s with 6.
- `wasm-pack build sim-wasm --release --target web` succeeds. wasm-opt is disabled in Cargo metadata to avoid the binaryen download.
- Feel lab (`npm run build` passes; `npm run verify` reports VERIFY OK, headless Chromium):
  - The stick chases the mouse: a gap of 1.56 rad on the step the target moved, closing over about 40 steps with a small overshoot.
  - The same stick hit (launch 20.63 m/s both times) slides 32.94 m at friction 1.0 and 26.52 m at 2.5, with friction changed through the DOM panel.
  - F held 1.5 s raised the spawn to about 2.3 m. After release the puck landed after 82 steps (0.683 s), against 0.692 s predicted for free fall, then bounced.
  - A recorded run and its replay produced byte-identical trajectory CSVs (same sha256).
  - `calib eval` on the lab-recorded scenario and CSV gives pos RMSE 2e-16, so lab recordings and calib formats need no conversion and native Rust reproduces the wasm run.
  - Screenshots are written to `web/verify-out/` (gitignored). They were checked by eye: rink, puck at rest, mid-slide, high spawn (red puck, big offset shadow), landed, ghost.

#### Verified (2026-11-04, Linux cloud container)
- Puck tunneling stress test (`sim/tests/tunneling.rs`): 30,180 net shots (all 6 segments, both sides, 11 points per segment, 35 angles, 3 start distances) and 23,376 board shots (88 targets on straights and corners), each at all three speed caps (30/35/45 m/s). It checks every step that the puck never crossed a net segment, never overlaps one, and never left the rink.
  Before the fix, 88 net shots at 45 m/s tunneled through the back wall near the net corners (the per-segment penetration push near a corner pushed the puck out the far side). With swept net collision: 0 failures. The test also passed in a one-off run at 90 and 120 m/s.
- `cargo test --workspace` passes (16 tests). The native and wasm crosscheck hashes are unchanged (`c93c94a203a58d2d`), because that run never touches a net. `npm run build` passes.

#### Verified (2026-11-04, measure/ session, Linux cloud container)
- Clips on `clip-data` (sizes checked): justpuckphysicshittingitatfixedpoint (3970 frames), justhittingwallphysicsfixedpoint (3778),
  justliftedshots (1663), all 1920×1080 at 60 fps. staticrotatingstick skipped (moving camera, stick unreadable).
- Map: inner board ≈913–914 × 581–582 px, corner radius ≈133 px, center dot within 0.5 px of the board center.
  **Measured length/width = 1.569–1.573 against params 60/26 = 2.308**, so the real rink is relatively much wider (or shorter)
  than params.json. With the placeholder length, the width comes out at ≈38.2 m and the corner radius at ≈8.7 m.
- Camera drift (corner landmark, superseded by the 2026-11-04 calibration-prep session below): **justhittingwallphysicsfixedpoint's
  camera moves up to ~15 px for whole stretches**; that session dropped 1391 of its 3776 frames (now corrected instead).
- Track: puck clip 99.7% of frames (11 tracklets), lifted 99.8% (4), wall clip 99.7% of camera-stable frames (63.0% of all).
  No speed outliers above 45 m/s (at the placeholder scale) and no back-and-forth spikes. QA sheets in `measure/results/qa/` checked by eye.
- The recording repeats a frame now and then (11 in the puck clip, 8 in the wall clip). Step lengths in steady glides vary with a
  median of 4–6% and a p90 of 37–51% (capture timing jitter, the game renders at ~180 fps). Fit with that noise in mind.
- Ghost check (Playwright, game view): the puck clip's track lies inside the sim rink's length everywhere (x −0.73..29.96 m) and goes
  to the top-left corner as in the video. In y it reaches 18.95 m against a 13 m half-width (the rink-shape mismatch above).
  Screenshot: `measure/results/qa/ghost-game-view.png`.
- `cargo test --workspace` (16 tests incl. both tunneling tests; they read all three speed caps from params.json), `npm run build`,
  `npm run verify` (VERIFY OK) and the measure tests all pass.

#### Verified (2026-11-04, calibration-prep session, Linux cloud container)
- **Camera correction (wall clip)**: raw drift up to 14.99 px; frames kept 2389/3776 without correction → 3776/3776 with it; residual after
  correction max 0.098 px (378 re-measured frames). Puck/lifted clips: raw drift ≤ 0.50 px, residual ≤ 0.35 px. Detection 99.7 / 99.6 / 99.9%.
- **Circles are round → top-down camera**: vertical/horizontal diameter ratio 0.9955–0.9966 for all 5 circles in all 3 clips (fit rms ≈ 0.07 px,
  tolerance 2%). One scale for x and y; the measured board length/width **1.571 is the real rink shape**. The ~0.4% flattening is the same
  everywhere in the image (not tilt, which would vary across the frame); left uncorrected.
- **Perspective camera** (not orthographic), seen two ways: (1) the puck blob grows with distance r from the image center (≈44.5 + 0.079·r px²):
  more of the puck's side shows off-axis, so the blob centroid sits up to ~2 px (≈0.5% of r) further out than the ground position (NOT corrected);
  (2) the net roofs show parallax: with the camera panned 240 px the roof appears 20 px further out (k = 0.084), used to correct the net size.
- cargo test --workspace passes (17 tests incl. both tunneling tests, re-run against the new measured-shape rink and nets, and the noisy
  recovery test). Native and wasm crosscheck hashes match after 100k steps: `b3e351378683a6f4` (changed from `c93c94a2…` only because
  params.json's defaults changed; sim code untouched). `npm run build` and `npm run verify` pass; the real ghost now lies inside the sim
  rink box for 3956/3956 samples (was 3367/3960 with the old 26 m width).
- Noisy synthetic recovery (`noisy_recovery_path_objective`): 0.3 px noise at 0.0657 m/px, 0.28% duplicated + 0.28% skipped frames, 25% of
  frames ±1/180 s off. Friction 1.0 / restitution 0.7 from 1.4 / 0.49: time objective → 0.99867 / 0.70142, path objective → 0.99989 / 0.70022.

#### Measured rink shape (params.json provenance `measured-shape`)
- **puck_radius 0.2298 m** (half-max edge of the resting puck's bright face near the image center, ÷ (1 + k_top); 0.231 / 0.218 /
  0.241 m per clip; the dark-green rim reaches ~0.32 m, the tracker blob ~0.245 m) and **player_radius 0.4557 m** (the pink disc,
  circle fit to its half-max edge; 0.459 / 0.458 / 0.451 m; the blue rim reaches ~0.53 m; the disc may be drawn at the player's
  height, so possibly a few % large): `python -m measure sizes`. Check: at bounces the sim's puck center reaches 0.230 m from the
  board (0.235 m resampled at 60 Hz), the real tracks 0.254 m median (0.199–0.276). Tunneling with this puck: 22,092 net + 21,768
  board shots, 0 failures.
- rink_width 38.198 m, rink_corner_radius 8.724 m, net_width 5.921 m, net_depth 2.226 m, **net_x 22.817 m** (goal line),
  blue_line_x 8.050, faceoff_x 16.305, faceoff_y 10.426, faceoff_circle_radius 4.708 m (the last four display only): ratios to
  rink_length are measured, **meters assume the placeholder rink_length = 60 m** (still provenance `placeholder`).
  Made by `python -m measure params` (markings: ice-plane scale, mean of 3 clips, sd ≤ 0.015 m).
- Nets: roof outline parallax-corrected with k = 0.084 (camera-centered clips); depth assumes a flat roof (uncorrected roof outline: 4.34 m).
- Tunneling re-run against this layout: 30,552 net shots + 24,552 board shots, 0 failures.
- Any later absolute scale fix multiplies all of these together; the shape stays.

#### Physics defaults (params.json, provenance `fitted`)
- **params.json now carries the fitted puck params** (merged from params.fitted.json by `python -m measure merge`):
  **board_restitution 0.343, board_friction 0, ice_linear_drag 0.4725 1/s, ice_friction_decel 0** — all scale-free.
  `params.fitted.json` stays: `measure holdout` writes it and `measure merge` reads it (it carries the same rule keys as
  params.json). `params.placeholder.json` keeps the pre-merge file (placeholder physics, old 0.15 m puck) for tests that need it
  (`puck_slides_to_stop_on_ice`); `calib/data/params.synthetic.json` stays the pin for the synthetic fitter data.
- Fit (`python -m measure holdout`, staged, path objective, held out by clip; base = placeholder physics): A = ice_linear_drag
  (+ ice_friction_decel) on glides, B = board_restitution + board_friction on wall bounces with `--objective path-only`.
  Per direction (wall→puck / puck→wall): restitution 0.339 / 0.362, drag 0.468 / 0.473, board friction 0 / 0, constant friction
  0 / 0 (so fixed at exactly 0). Held-out score 307.0 → 3.06 and 51.5 → 0.46. The joint 4-parameter fit now agrees too
  (e 0.31 / 0.29, k 0.484 / 0.485).
- **What changed the fit in session 6**: the segments' initial speed. A 10-row quadratic at t = 0 has ~8% speed noise under the
  capture jitter (and starts moved to a noisy speed peak selected high values); 30 rows (before any bounce) has ~1.6%
  (400 synthetic glides with the real capture faults: quad10 sd 8.2%, quad20 2.8%, quad30 1.6%, a linear-drag model fit 0.4% but
  circular). With clean starts the directions agree and restitution matches the direct bounce estimate (0.36).
- Re-fitting with the measured puck radius (0.23 m instead of 0.15) gave values within the direction spread, so the radius
  change alone didn't require a refit (`calib/data/real/holdout_radius_check.json`).
- Drag model: no quadratic term (held-out, session 5: linear 18.02 / 1.36, + quadratic 42.50 / 2.41). The sim is unchanged.

#### Board friction model (session 6: not adopted)
- Question: does the real puck lose extra speed sliding along / riding the boards that the per-contact model (restitution +
  proportional tangential loss) can't give? Candidate: Coulomb-style loss, tangential speed −= μ_c (1 + e) |v_n| per contact
  (total loss ∝ how much the board turns the puck). `python -m measure boardmodel` on 9 board-contact segments, held out by clip
  (needs `docs/experiments/board_coulomb_friction.patch`, not in the sim).
- With the corrected initial speeds (`calib/data/real/boardmodel.json`), held-out score wall→puck / puck→wall:
  defaults 2.02 / 0.55; current model refit 1.94 / 0.59 (friction → 0); Coulomb 1.94 / 0.58 (μ_c 0.000 / 0.035);
  both 1.94 / 0.58; Coulomb with e fixed 2.02 / 0.61. No candidate beats the defaults in both directions; μ_c fits ≈ 0.
- Before the initial-speed fix Coulomb looked better (9.37 / 2.55 vs defaults 12.95 / 2.98) but only by lowering e to 0.28 —
  the speed-vs-geometry confound; with e fixed the models tied. The "extra loss along the boards" (puck_08) was mostly the
  noisy initial speed. **Evidence does not support a sim change**; the sim and params.rs are unchanged.

#### Verified (2026-11-04, first-fits session, Linux cloud container)
- `cargo test --workspace` (17 tests; tunneling 30,552 + 24,552 shots against the measured layout, 0 failures), the measure tests
  (15), `npm run build`, `npm run verify` (VERIFY OK). Native and wasm crosscheck hashes match after 100k steps:
  `b3e351378683a6f4` (no sim or params.rs change this session; that run never touches a net, so the net_x change doesn't move it).
- verify.mjs: the F-spawn check found first ground contact by `z === 0` on per-step samples, which only works when contact lands
  on a step's last substep (it failed once by chance). Now it uses the first step where a falling puck's vz turns ≥ 0 near the ice:
  first contact after 84 steps (0.700 s) vs 0.707 s predicted for free fall from 2.45 m.

#### Verified (2026-11-04, session 6: defaults + sizes, Linux cloud container)
- **Absolute scale: NOT measured.** The goal-speed clip (clips/goalspeed.mkv) is not on clip-data (still commit 8e77faa with the
  original four clips; no other branch has it), so the placeholder scale stays (rink_length 60 m). Nothing was rescaled.
- `cargo test --workspace` (18 tests incl. both tunneling tests against puck_radius 0.23 m: 22,092 + 21,768 shots, 0 failures,
  and `fitted_default_glide_is_linear_drag`), the measure tests (15), `npm run build`, `npm run verify` (VERIFY OK). Sim code
  and params.rs unchanged (the Coulomb experiment was reverted); native and wasm hashes still match after 100k steps:
  `93105bebdd8a0d19` (changed from `b3e351…` only because the default params changed).

#### Placeholder / unmeasured – EVERYTHING physical
- **Still unmeasured** (provenance `placeholder`): timestep, substeps, gravity, rink_length (the absolute scale), puck mass, air drag,
  ice restitution, net restitution, player restitution/accel/speed/drag/turn/height/mass, stick length/mass/angular limits/z-range,
  all three puck speed caps, dash and lift numbers, speed_display_scale. Measured shape: rink, nets, markings, puck and player
  radius (meters at the placeholder scale). Fitted: board_restitution, board_friction, ice_linear_drag, ice_friction_decel.
- **Dash logic is a placeholder**: on the press edge it adds a velocity impulse along the move input (or facing) and opens a `dash_duration` window in which hits count as dash shots, then a cooldown applies.
- **Lift logic is a placeholder**: holding lift at stick contact sets vz = `lift_vz_factor`·speed (capped), and dash + lift caps the hit at `chip_max_speed` (the "chip").
  The real mechanic (dash + lift + slight rotation → low-speed chip) is unknown.
- **Key bindings are UNVERIFIED GUESSES**: dash = Shift and lift = Space (`web/src/bindings.ts`). The real game's bindings have not been checked.
- **Practice tools are APPROXIMATE**, written from patch notes and not checked against the real game:
  - **F**: sets the puck spawn at the cursor (z = 0). Holding F raises the spawn at 1.5 m/s up to 6 m, with the puck pinned there; release drops it.
  - **G**: sets a target marker at the cursor. The panel shows the puck's closest approach.
  - **R**: puts the puck back at the spawn, at rest.
  - **Q**: slides the puck along the ice toward a "forehand" point near the blade on the CCW side of the stick (handedness unknown), sized to arrive at about 3 m/s and capped at 20 m/s.
- Display-only placeholders: `blue_line_x`, `faceoff_x`, `faceoff_y`, `faceoff_circle_radius` and `speed_display_scale` (the real-game speed per sim m/s used for the mph/kph readout; 1.0 is a guess).
- Model simplifications that are still open: boards are infinitely tall, the net has no top bar, there is no puck spin, and no stick-on-stick or stick-on-player collision. Facing simply follows velocity. Goals only increment the score (no reset).

#### Open
- Real calibration: 4 scale-free puck params fitted and now the default. **The meter scale is the main open measurement**
  (the goal-speed clip was not on clip-data in session 6).
- Model gaps (not fixed): (1) the extra speed loss along the boards seen in session 5 was mostly noisy initial speeds; with clean
  starts a Coulomb board friction doesn't help (see "Board friction model"); (2) decel/v is not
  constant: ≈ 0.50–0.55 1/s above ~8 m/s, 0.33–0.39 at 3–7 m/s, ≈ 0.55 near the stop (placeholder scale). Not a smoothing or
  jitter artifact (persists for windows 4–16 rows; synthetic linear drag through the same capture faults reads 0.42–0.46 in every
  band) and not quadratic (non-monotonic; a v² term overfits held-out). One linear k is a compromise (fast glides like puck_02
  slow faster than k = 0.41).

##### Recording checklist for Ian (what the data cannot give yet)
1. **Absolute scale (most important)**: practice-mode goals where the in-game speed readout (mph/kph) shows, ideally 5–10 shots at
   different distances and speeds, camera fixed and top-down as before, the whole rink in view. The readout gives m/s for a
   tracked speed, which fixes meters per pixel and with it every absolute value (friction in m/s², speeds, sizes). A clip with any
   known distance (e.g. if the game states rink or net dimensions) works too.
2. **Net restitution**: shots that hit the posts, crossbar side and the back of the net without scoring, from several angles (no
   net hits in the current clips; also note whether the puck goes under the roof and disappears).
3. **Stick and player collisions**: the player standing still while the puck is shot into the body and the stick blade; a slow
   puck sliding into a stationary stick; record from the same camera so the player can be tracked later (player tracking is not
   built yet).
4. **Lift height**: lifted shots with a side reference (or the in-game shadow and landing ring visible), plus a few lifts that
   land on open ice away from the boards, so the airborne time and landing point can be measured; the top-down view alone gives
   no height.
5. **Board sliding**: slow pucks sent along a straight board and around a corner at a few speeds (the sim's board-contact speed
   loss is the largest remaining error).
6. **Medium-speed glides**: long free glides from ~10 m/s down to rest across open ice (no board contact), to pin down the
   speed-dependence of the ice drag.
  Clips live on the `clip-data` branch; `clips/` is gitignored so clips never land on main.
- Menu (later session). Scripted bots and the training pipeline exist ("Bots and training"); real training runs have not started.
- Opponent player is idle in the lab. Panel edits mark a row as changed but keep the provenance tag (they are experiments, not fits).
- Practice-tool actions are not recordable (they are state edits). Set up the drill first, then record.

#### Fidelity log
(Each later session records per-scenario error versus the real game here.)

##### 2026-11-04 – session 6: fitted params are the default; puck/player radius measured; scale still unmeasured
- **Scale**: not measured — the goal-speed clip is not on clip-data. Meters stay at the PLACEHOLDER scale (rink_length 60 m,
  0.0657 m/px); nothing rescaled; ice_friction_decel stays at 0 (fitted, scale-free) rather than a value in real m/s².
- **New defaults** (params.json): board_restitution 0.7 → 0.343, board_friction 0.1 → 0, ice_linear_drag 0.05 → 0.4725 1/s,
  ice_friction_decel 1.0 → 0 (all `fitted`); puck_radius 0.15 → 0.2298 m, player_radius 0.45 → 0.4557 m (`measured-shape`).
- Residuals after the merge (`calib/data/real/residuals.txt`; old = params.placeholder.json, new = params.json, held-out = staged
  fit on the other clip), medians:

| clip | label | n | pos_rmse old → new (held-out), m | path_rmse | v2_rmse, m²/s² |
|---|---|---|---|---|---|
| puck | glide | 7 | 2.10 → 0.91 (0.87) | 0.08 → 0.47 (0.44) | 15.1 → 7.3 (7.2) |
| puck | wall bounce | 3 | 19.01 → 0.82 (0.94) | 1.56 → 0.07 (0.07) | 268.4 → 13.9 (15.4) |
| wall | glide | 8 | 1.06 → 0.43 (0.44) | 0.13 → 0.22 (0.22) | 8.0 → 4.7 (4.7) |
| wall | wall bounce | 3 | 2.04 → 0.50 (0.61) | 1.32 → 0.31 (0.35) | 10.6 → 3.8 (3.7) |
| lifted (excluded) | glide | 1 | 4.00 → 1.78 | 0.16 → 1.09 | 21.4 → 7.0 |
| lifted (excluded) | wall bounce | 3 | 23.11 → 2.56 | 1.11 → 0.16 | 507.7 → 108.8 |

  All 21 ground segments: pos_rmse median 2.10 → 0.50 m, v2_rmse 15.1 → 6.6 m²/s², mean score 175.2 → 1.63. Glide path_rmse
  rises (0.08 → 0.47 m in the puck clip): with one linear drag the slow glides stop short of the real ones (the medium-speed
  dip in decel/v — the real puck coasts further at 3–7 m/s).
- Worst three by the new default (`measure/results/fits/worst_<seg>.png`, inspected): **puck_01** (score 16.0) — path exact, the
  v² error is the ±1.5 m/s capture-jitter wobble at ~25 m/s (measurement noise); **puck_09** (2.2) and **puck_08** (2.0) — corner
  rides now followed closely (path 0.07 / 0.18 m); before the bounce the real puck slows ~1 m/s faster at high speed (drag law).
- **Still placeholder**: everything about players (accel, speed, drag, turn, mass, height, restitution), sticks (length, mass,
  angular limits, z-range) and puck–stick impulses, the three puck speed caps, nets (restitution; nets are static segments
  without a top bar), lift (lift_vz_factor, max vz, chip) and dash (speed, duration, cooldown), air drag, ice restitution,
  gravity, puck mass, timestep/substeps, and the absolute scale.

##### 2026-11-04 – first real fits (params.fitted.json) vs placeholder, ice-plane scale, 25 segments
Meters at the PLACEHOLDER scale (rink_length 60 m). Positions are puck ground points (height-parallax corrected). Per-segment
table: `calib/data/real/residuals.txt` (placeholder / fitted on all ground segments / held-out = fitted on the other clip).
Score per segment = path_rmse² + 0.01·v2_rmse² (the path objective's term).

Hold-out by clip (staged fit, ground segments), held-out clip score (lower is better):

| train → test | baseline test score | fitted test score | test pos_rmse median | test path_rmse median | values (e / bf / k / μ) |
|---|---|---|---|---|---|
| wall (11) → puck (10) | 283.85 | 12.81 | 3.97 → 0.83 m | 0.40 → 0.42 m | 0.403 / 0 / 0.468 / 0 |
| puck (10) → wall (11) | 56.10 | 2.30 | 1.99 → 0.98 m | 0.45 → 0.32 m | 0.368 / 0 / 0.396 / 0 |

All 21 ground segments, final values: score 164.55 → 7.27, pos_rmse median 2.62 → 0.59 m, path_rmse median 0.45 → 0.33 m,
v2_rmse median 21.2 → 5.3 m²/s².

| clip | label | n | pos_rmse median placeholder → fitted (held-out) | path_rmse | v2_rmse |
|---|---|---|---|---|---|
| puck | glide | 7 | 2.68 → 0.48 (0.75) m | 0.33 → 0.33 (0.47) | 22.3 → 3.5 (4.6) |
| puck | wall bounce | 3 | 19.37 → 1.93 (1.33) m | 2.22 → 0.14 (0.16) | 276.1 → 37.7 (20.3) |
| wall | glide | 8 | 1.36 → 0.86 (0.77) m | 0.43 → 0.39 (0.39) | 11.9 → 4.6 (4.3) |
| wall | wall bounce | 3 | 2.62 → 1.09 (1.25) m | 0.93 → 0.42 (0.23) | 14.6 → 6.1 (6.5) |
| lifted (excluded) | glide | 1 | 3.83 → 0.82 m | 0.05 → 0.30 | 20.4 → 3.9 |
| lifted (excluded) | wall bounce | 3 | 21.98 → 3.69 m | 0.98 → 0.19 | 437.6 → 99.1 |

Worst three after the fit (overlays `measure/results/qa/compare_<seg>.png`, inspected):
- **puck_01** (score 74): path and timing match; the v² error is capture-timing jitter — at ~25 m/s the measured speed wobbles
  ±1.5 m/s around the trend (a tracker/capture limit, not physics).
- **puck_02** (16.6): clean 27 m/s glide; the real puck slows faster than k = 0.41 at this speed (decel/v ≈ 0.5): the single
  linear drag is a compromise with the slower glides (drag-law gap).
- **puck_08** (15.3): left-board bounce + corner ride + slide along the top board; the fitted path follows it, but the real puck
  loses more speed while riding/sliding along the boards than the sim (board-contact model gap) and slows faster at high speed.
- Fixed because of the overlays (before this table): shots straight into the board next to the player had the bounce inside the
  initial-state fit (wall_01, wall_09, wall_11 → wrong initial velocity, fake restitution); stretches that ended at a reset
  included the board impact (puck_01). The segmenter now handles both.

Scale-free estimates on the final segments (`measure estimate`): normal restitution 0.363 (median; sd 0.041, n = 8),
tangential ratio 0.97, decel/v per stretch median 0.423 1/s (sd 0.078, n = 19), pooled decel = 0.50·v − 5.4 px/s² (constant ≈ 0).

##### 2026-11-04 – baseline: placeholder physics vs 26 real segments (no fit yet)
Params: params.json with the measured rink shape; all physics placeholders (ice_friction_decel 1.0 m/s², ice_linear_drag 0.05 1/s,
board_restitution 0.7, board_friction 0.1). Meters at the PLACEHOLDER scale (rink_length 60 m / ~913.5 board px = 0.0657 m/px).
Full table: `calib/data/real/baseline_eval.txt` (pos_rmse = time-aligned, path_rmse/v2_rmse = timing-free).

| clip | label | n | pos_rmse median (range), m | path_rmse median, m | v2_rmse median, m²/s² |
|---|---|---|---|---|---|
| puck | glide | 7 | 1.60 (0.99–6.41) | 0.17 | 27 |
| puck | wall bounce | 3 | 18.15 (14.21–18.57) | 2.35 | 248 |
| wall | glide | 8 | 1.50 (0.06–9.18) | 0.87 | 8 |
| wall | wall bounce | 4 | 8.52 (2.59–21.58) | 4.35 | 140 |
| lifted (possibly airborne) | glide | 1 | 3.87 | 0.05 | 21 |
| lifted (possibly airborne) | wall bounce | 3 | 22.19 (19.06–29.26) | 1.12 | 441 |

All 22 ground segments: pos_rmse median 2.62 m, path_rmse median 0.63 m. The placeholder sim slides much too far
(constant 1 m/s² friction vs the real speed-proportional slowdown) and bounces too hard.

Scale-free estimates (`python -m measure estimate` → `calib/data/real/estimates.json`; lifted clip excluded; NOT written to params.json):
- **Board restitution**: normal-velocity ratio **0.36 (median; mean 0.358, sd 0.040, IQR 0.34–0.38), n = 8 bounces** in 6 segments
  (puck clip 5, wall clip 3; 4 at corners, same values). Tangential ratio 0.98 (sd 0.14) → board friction ≈ 0. Speed ratio |v_after|/|v_before|
  0.69 median (sd 0.14) — it depends on the incidence angle (19–72°), as normal restitution predicts.
- **Glide deceleration shape**: proportional to speed (linear drag), not constant. decel/v per stretch: median **0.43 1/s** (mean 0.44, sd 0.08,
  IQR 0.40–0.51), n = 19 stretches from 17 segments (puck + wall clips). Pooled fit over 2615 local fits: decel = 0.50 1/s · v − 5.4 px/s²
  (constant term ≈ 0; −0.36 m/s² at the placeholder scale), a v² term adds nothing. Pure linear drag fits better than pure constant
  deceleration in 18 of 19 stretches. Unexplained: decel/v dips to 0.34–0.39 1/s for 50–112 px/s (≈3.3–7.4 m/s placeholder) vs ≈0.52 elsewhere.
  k (1/s) is scale-free; any constant friction term needs the meter scale.

#### Player (session s6, 2026-11-04, Linux cloud container) – video-only estimates, PROVISIONAL
**Provisional**: estimated from video without input logs; params.player.json is a reference only and is NOT merged into
params.json (a better estimate from input-logged clips is coming). params.json keeps the placeholder player values.
- Code (new files only): `measure/player.py` (`python -m measure.player {track,qa,estimate}`), `measure/player_motion.py`
  (events, estimates, params.player.json, sim check), `measure/player_synth.py` + `measure/tests/test_player.py` (synthetic video:
  moving pink disc with a stick crossing it, pink board lights + HUD badge, camera-follow stretch, duplicated frame; tracked < 1 px),
  `sim/examples/player_start.rs` (sim player from rest, full +x input for PUSH_S then none; prints t,speed).
- Outputs: `measure/results/NAME.player.csv` (t,x,y; same convention/placeholder scale as the puck), `NAME.player.json` (per-frame
  u/v ref px, raw px, camera shift, area, roundness, conf, summary), `player_estimates.json`, QA `qa/NAME.player_{overview,sheet}.png`.
- **The game camera follows the player.** The disc stays within 1.4–1.6 px (p99) of the image center in every clip; the player's
  motion IS the camera motion, measured on the faceoff-dot landmarks (search centered on the predicted shift, so large travel works).
  Player position = disc centroid (ellipse fit to the convex hull of the pink blob, robust to the stick's notch) − camera shift.
  The big blue glow is a marker on the ice at the spawn point (it stays behind when the player skates), not part of the player.
- **Height correction: not applied, by the data.** The disc never leaves the image center, so k/2·r < 0.08 px even with k = 10× the
  puck's; the area-vs-r model can't even be fitted (r range < 3 px). The rule (apply if the blob grows ≥ 25% with r, like the puck) is in the code.
- Tracking: detection 100% in all 4 clips, longest gap 0 frames (puck 3962/3962 non-duplicate frames, wall 3758/3758, lifted 1653/1653,
  staticrotatingstick 2224/2224); duplicates dropped 8/18/8/15. Rest noise (30-frame std) 0.02–0.07 px in the fixed clips, 0.3 px in
  staticrotatingstick (the stick rotates over the disc the whole time). Landmark residual ≤ 0.57 px.
- **staticrotatingstick was added** (the 3 "fixed-camera" clips are fixed only because the player stands still: puck and lifted clips
  move < 2 px, the wall clip only nudges ≤ 25 px). It has no map (camera travels 134 px); it is registered against the puck clip's
  reference frame (same camera/zoom): constant +20.1 px horizontal offset = this clip's spawn, residual ≤ 0.07 px.
- Events: speed (local quadratic fit ±3 samples) > 8 px/s for ≥ 6 frames; **locomotion only if the camera moved ≥ 3 px** over the
  event (21 of 30 candidates were disc wobble from the stick, camera net 0.0–0.6 px; the 9 real ones moved it 4–210 px).
  Overlays inspected (sheets at the starts/stops/teleport); fixes from inspection: the blue glow is a static marker (pink-only
  detection), interpolation across a teleport (wall frame 263) is now forbidden, overview labels the reference background.

##### Estimates (px are reference ice px; placeholder m = px × 0.0657, rink_length 60 m)
| quantity | value | n | events |
|---|---|---|---|
| top speed (flat plateau ≥ 0.25 s) | **180.3 px/s = 11.84 m/s**, dead flat for ~1 s (likely a hard cap) | 1 | rotstick_04 |
| peak speed, all locomotion | median 37.9, 25–201 px/s | 9 | |
| acceleration from rest (10→90% ramp, normal starts) | **241 px/s² = 15.8 m/s²** (sd 44 px/s², 146–277) | 7 | wall_01/03/05/06/09/10, rotstick_01 |
| time rest → top speed | 0.75 s extrapolated (180/241); rotstick_01 reached 144 px/s after 0.62 s; the wall taps reach their own ~35 px/s plateau in 0.12 s | | |
| stop deceleration (v0²/2d from the stop distance) | median **396 px/s² = 26.0 m/s²** (sd 260, 29–907); stops take ~0.07–0.25 s; exp-rate view: median 20 1/s (2.3–24) | 8 | |
- Neither constant nor exponential decay fits all stops (decel grows with speed but not proportionally). The 2 fast runs stop in
  0.13–0.22 s from 144/180 px/s. Several runs hold a flat speed before stopping (wall taps ~0.13 s at ~35 px/s, rotstick_04 1 s at 180),
  i.e. no visible coasting decay at all: the stops are probably **active braking**, and the real release drag may be much smaller.
- **Dash**: no burst above a plateau (> 1.2×) was found. Two **near-instant starts** (6× the normal ramp): wall_08 (0→94 px/s in ~0.03 s,
  gone again within ~0.1 s, 7 px hop) and rotstick_04 (0→180 px/s in 0.13 s, ~1500 px/s², then the flat 180 plateau; a 1.12× bump to
  201 px/s for ~0.1 s just before its stop). They are dash candidates; spacing can't be measured (one per clip). No dash model was fitted.
- Limits: inputs unknown → "pushing" = speed rising or flat near its plateau; a partial/tapped input looks like a low plateau (wall taps),
  a brake looks like drag; top speed is n = 1; all motion is along the rink's long axis except the wall taps (directions −100…90°).

##### params.player.json (provenance `fitted-no-input-log`; params.json untouched)
player_max_speed 11.84 m/s, player_accel 15.85 m/s², player_ice_drag 26.02 m/s² (placeholder scale; rescale all three with the meter scale).
Held-out by clip (fit on the other clip, unfitted names stay placeholder; sim = `examples/player_start`, from rest, full input until the
measured last pushing frame), speed RMSE px/s, whole event / rising phase:
| test | held-out fit | all-clip fit | placeholder (8 / 12 / 2) |
|---|---|---|---|
| rotstick_01 (train wall) | 19.9 / 6.8 | 34.3 / 7.3 | 37.5 / 24.2 |
| wall_01,03,05,06,09,10 (train rotstick) | 8.4–15.6 / 3.1–7.3 | 4.9–21.9 / 2.3–6.6 | 6.8–17.8 / 6.5–10.3 |
| bursts wall_08 / rotstick_04 | 69 / 76 (rising 71 / 97) | 68 / 66 | 69 / 87 |
The ramp is clearly better than the placeholder in every normal start (rising phase); whole-event errors on the wall taps are dominated
by their unknown short inputs (the sim keeps accelerating through the measured flat ~35 px/s). The bursts can't be reproduced (no dash model).

##### Recording guide
`docs/recording-guide.md`: OBS + Input Overlay (univrsal/input-overlay; NohBoard fallback), overlay over the stands, 1080p, 120 fps if
possible; 8 clips (W start/release/brake, A/S/D, diagonal, full turn, dash from rest/at speed/repeated, stick sweeps at closer zoom
with known angles, stick hits at several swing speeds, long coast), each with what it measures. Keep runs between the end-zone circles
so ≥ 3 faceoff dots stay visible (the tracker follows the camera on them).

##### Verified (s8, Linux cloud container)
- `cargo test --workspace`: 25 tests incl. 8 new in `sim/tests/match_rules.rs` (new match starts in countdown with a full clock,
  mercy as a 5-goal lead, clock end with a winner, tie → overtime → next goal wins, goal pause frozen then reset to spawn,
  inputs ignored in countdown and goal pause, 100k-step match determinism, native match crosscheck hash).
- Determinism: a scripted 100k-step match (goals, pauses, mercy end) gives byte-identical states (hash `f3e066e94d96a5fb`).
  **Native and wasm match hashes match after 100k steps: `22f04837353ecb0b`** (`scripts/wasm_crosscheck.mjs` now runs both);
  free play still `b3e351378683a6f4` (both hashes are pre-merge; see "Integration of s6, s8, s9" for the current ones). A lab-recorded match segment (goal pause → reset → countdown → play) replays byte-identically
  in the lab and through native `calib replay` (same CSV sha256).
- `npm run build`, `npm run verify` (VERIFY OK) and `npm run verify-match` (VERIFY MATCH OK: mercy 5-0, clock end 0-1, 1-1 at
  00:00 → overtime → 1-2). Screenshots `web/verify-out/match-*.png` checked by eye.

##### Verified (2026-11-04, s9, Linux cloud container, 4 cores, CPU only)
- `cargo test --workspace`: 31 tests pass (bots 8, sim-ffi 6, plus the 17 existing). Python: `python -m unittest discover
  -s train/tests -t .` 9 tests pass. `npm run build` passes. sim, sim-wasm and params.json are unchanged.
- Env throughput with obs + reward + scripted chaser opponent (`cargo run --release -p sim-ffi --example bench`), sim steps/s:
  1 env 1.16M; 8 envs 1.13M (1 thread) / 1.20M (rayon); 64 envs 1.12M (1 thread) / 2.68M (rayon, 4 cores).
  From Python through ctypes (random actions): 1 env 152k, 8 envs 324k, 64 envs 1.44M sim steps/s.
- Determinism: same seed → identical batch hash and observation bits with 1 thread and with rayon (Rust: 64 envs,
  2000 decisions, 930 episodes, hash `cc2d16d781790bbf`; Python: 64 envs, 1500 decisions, `7d190a29d2b55bdb`); another seed differs.
- Mirror: the team-1 observation equals the team-0 observation of the rotated, team-swapped state (200 random states,
  |Δ| < 1e-6), and a chaser-vs-chaser game played from a state and from its rotation stays rotated for 240 steps (< 1e-6 m).
- Smoke training (CPU, 6 min, 64 envs, only the chaser as opponent, 823 updates, 6.7M decisions): rolling goal diff per
  episode −1.00 → −0.55, win rate 0 → 12% (training starts). Eval at standard faceoffs, 400 episodes each vs chaser:
  random policy −0.998 (0 wins), untrained policy −0.985 (0.5% wins), trained −0.34 (20.5% wins, 55% losses, 25% draws;
  deterministic −0.40). Curve: `train/results/smoke_chaser_learning_curve.png` (inspected: steady rise after ~2.5 min),
  metrics `train/results/smoke_chaser_metrics.csv`.
- ONNX export of that checkpoint: onnxruntime vs PyTorch max |Δ| 6.0e-7 over 1000 random observations (314 KB file).
- Scripted tournament (`tournament 100 60`: 100 matches × 60 s per pairing, jittered faceoffs, sides alternate):

| A | B | A goals/match | B goals/match | A wins | draws | B wins |
|---|---|---|---|---|---|---|
| chaser | chaser | 0.79 | 0.65 | 29 | 46 | 25 |
| chaser | defender | 1.23 | 0.07 | 70 | 27 | 3 |
| chaser | cautious | 14.10 | 0.00 | 100 | 0 | 0 |
| chaser | idle | 14.13 | 0.00 | 100 | 0 | 0 |
| defender | defender / cautious / idle | 0 | 0 | 0 | 100 | 0 |
| cautious | cautious / idle | 0 | 0 | 0 | 100 | 0 |

  The defender cuts the chaser from 14.1 to 1.2 goals per minute. The cautious bot does not defend (it waits at 15 m and
  the chaser's 20 m/s shots go past it), and defender/cautious never attack a puck that does not come to them, so their
  mutual games are 0–0. A first test (`tournament` with consecutive-seed LCG jitter) showed a false 15–3 side bias; the
  jitter now uses SplitMix64.

#### Integration of s6, s8, s9 into main (2026-11-04, Linux cloud container)
- Merges (normal merge commits, no rebase): `0830666` origin/s6 (clean), `0e7806c` origin/s8, `fa525dd` origin/s9.
- Conflicts: CLAUDE.md (s6/s8/s9 each appended a section: all kept, in that order); sim/tests/sim_tests.rs (main's FITTED
  list + placeholder pin and s8's MATCH_RULES list: both kept, provenance check accepts measured-shape / fitted / rule /
  placeholder). params.json, params.fitted.json, calib/data/params.synthetic.json auto-merged (main's values, s8's five rule
  keys appended); params.placeholder.json (main-only) got the same five keys by hand so it still loads (`to_params` is strict).
  params.player.json stays a separate provisional reference (not merged).
- Fixes after the merge: `bots::rotate_swap` swaps `match_state.winner/last_goal`; the sim-ffi test now requires ≥ 75% of
  scoring episodes to have a shot on net (with the 0.23 m puck the chaser's post-side aims sometimes go in off the post,
  which the detector rightly does not count); bot shot constants re-measured (see below). No code builds State field by field.
- Checks: `cargo test --workspace` 41 tests pass (sim 12 + match rules 8 + tunneling 2, calib 4, bots 9, sim-ffi 6);
  measure tests 19 pass; train tests 9 pass (incl. ONNX parity); a fresh 30-update checkpoint on the merged physics exports
  with onnxruntime vs PyTorch max |Δ| 2.4e-7; old checkpoints are refused by the physics stamp (`215d58e4…` → `33d0972b55efca23`).
  All four params files load (`calib eval`). `npm run build`, `npm run verify` (VERIFY OK) and `npm run verify-match`
  (VERIFY MATCH OK) pass.
- **Crosscheck hashes after 100k steps (native = wasm), merged default params: free play `93105bebdd8a0d19`, match
  `78ac344e15f2dff7`** (both changed from `b3e351…` / `22f04837…` because the default physics changed).
- Lab in match mode with the default params (Playwright, real inputs + one physics-driven goal): `docs/integration-midmatch.png`,
  scoreboard BLUE 1 – 02:40 – 0 ORANGE, phase playing, lab params ice_linear_drag 0.47254 / board_restitution 0.342796, no page errors.
- Scripted bots on the fitted physics (`tournament 100 60`). Old placeholder physics (s9): chaser vs idle 14.13, vs defender
  1.23–0.07. Fitted physics, s9 constants (before) → re-measured constants (after, SHOT_OVERTURN 0.054, SHOT_SPEED 18.9):

| A | B | before: A–B goals/match (A wins/draws/B wins) | after: A–B goals/match (A wins/draws/B wins) |
|---|---|---|---|
| chaser | chaser | 0.50–0.44 (25/52/23) | 0.66–0.52 (27/53/20) |
| chaser | defender | 0.30–0.01 (27/73/0) | 0.59–0.03 (39/60/1) |
| chaser | cautious | 13.00–0.00 (100/0/0) | 13.00–0.00 (100/0/0) |
| chaser | idle | 13.00–0.00 (100/0/0) | 13.00–0.00 (100/0/0) |
| defender / cautious / idle pairings | 0–0 (all draws) | 0–0 (all draws) |

  Scoring did not collapse (shots still reach: 18.9 m/s with k = 0.47 1/s travels up to 40 m); shotlab's systematic aim error
  was 0.013 rad (signed with the swing) before the re-measure and 0.000 after.

#### Session s7: skating clips with a key display, moving-camera registration, player/dash fits, absolute scale (2026-11-04)
New files only (`measure/camreg.py`, `keys.py`, `skate.py`, `skatefit.py`, `goalscale.py`, `pan_synth.py`,
`key_templates.json`, `tests/test_camreg.py`, `calib/tests/player_track.rs`, `calib/data/skate/`, `measure/results/skate/`,
`measure/results/scale.json`) plus one additive calib change: a scenario may set `"track": "player0"` (default: the puck) so
replay/eval/fit compare player 0's path. params.json is untouched; player values go to params.player.json.

##### Clips (clip-data; 1920x1080, 60 fps, decoded frame counts)
| clip | MB | frames | what it shows (checked on frames) |
|---|---|---|---|
| wkeymovement | 38.5 | 2146 | 4 standing starts holding W 3.15-3.95 s (R reset before each), coast to rest |
| diagonalkeypresses | 80.8 | 3764 | W+D (3x), W+A (3x) held 2.3-3.6 s from rest, one S tap |
| whilemovingturning | 58.1 | 2394 | W to top speed, then D added (turn to the diagonal at speed), 5 runs |
| dashatrest | 28.4 | 2137 | Shift alone 3x (no motion), then W+Shift tapped together from rest 6x (R between) |
| dashwhilerunning | 46.8 | 2736 | W to top speed, Shift (+ W release), 4-5 runs; one dash from 2.6 m/s |
| goalspeeds_part00-03 | 66.6/57.6/64.9/7.3 | 1681/1412/1441/126 | one match recording (4660 frames, stream-copied parts, joined in order); 6 goals with "SHOT SPEED: xx.xx MPH" under the goal banner; no goal is cut at a part boundary |
| hittingpuck, rotatinganddizzy | | | stick clips: skipped (parallel session) |
The s7 clips use a new arena look (dark ice) and a closer zoom than the s6 clips (board-to-board 646 px vs 581). The player is a
grey disc (~19 px) with a red team rim and a black stick; the puck is green (on ice) and drawn above a white ground ring when
airborne. A Keyviz-style key display sits bottom right; an "Alt" cap is held through all of wkeymovement (a stuck key in the
display, ignored).

##### Camera registration (`measure/camreg.py`)
- **The camera follows the player in both axes** (the disc stays at the screen centre, ~(959.5, 541) +- 0.6 px, except where the
  camera stops at the rink ends) and looks straight down: line spacing is constant across the image and frames (center to blue
  line 136.5 px, center to goal line 386.8 px, ratio 2.835 vs params 2.834), board-to-board width constant (646.6 +- 0.2 px), so
  the ice moves by a pure translation.
- Per frame: **u** from the inner side-board edges (ice-colour mask, subpixel threshold crossing, median over the rows whose run
  is board-to-board wide; the side boards are always in view); **v** from the five full-width painted lines (per-row median
  "non-ice" profile over the ice columns, subpixel centroid), identified by their spacing and the blue center line, fitted as one
  translation (the anchor). Fallback when < 2 lines are identified: template-matching odometry (phase correlation was dropped:
  it returned a spurious 0.5 px horizontal shift between near-identical frames). px per placeholder meter s from the line spacing
  over the clip: **16.951-16.953 (skating clips), 16.940 (goal arena)**. World: x = (v_c - v)/s, y = (u_c - u)/s.
- Real clips: **every frame anchored** (0 odometry frames). Line-fit residual median 0.11-0.15 px, p95 0.20-0.21 px, max 0.26-2.13 px;
  3-5 lines per frame; board-width deviation median <= 0.35 px. Odometry alone vs the anchors (how error would grow without
  landmarks), median / p95 px after n frames: 1: 0.05-0.11 / 0.25-0.48; 15: 0.72-1.66 / 3.1-7.0; 30: 1.5-3.5 / 5.4-14; 120: 6.6-14 /
  15-43 (it underestimates motion near the screen-fixed HUD).
- Synthetic panning test (`measure/pan_synth.py`, exact area-coverage rendering, camera following in both axes, HUD box, stick
  over the disc): registration error max 0.078 px (u) / 0.028 px (v), player world error max 0.40 px, puck 0.27 px (< 1 px required).
- Player disc: neutral-grey fill as the seed (saturation < 24; ice ~53, painted lines ~33), centre and radius from 72 rays to the
  outer edge of fill+red rim and a robust circle fit (drops rays stopped by the stick or running into the red end board at the
  spawn), radius ~9.3 px. Candidates must lie inside the rink (0.3 m margin), outside the net boxes (+1.2 m; the nets and corner
  mesh structures look like discs); acquired nearest the screen centre, then followed. Detected in 97-99% of frames. Puck: green
  blob 20-160 px, not the green center dot, not behind/beside a net (a static green ornament sits on the net mesh).
- Limits: no height correction for the player disc or an airborne puck (the camera keeps the player near the image centre, so
  parallax is small for the player); a lifted puck is drawn displaced above its ground ring. Frames with the darkened ice of a
  goal celebration are not registered. QA: `measure/results/skate/qa/{diagonalkeypresses,whilemovingturning}.path.png` (the
  registered world path projected back into frames through each frame's own registration; inspected: the path passes through
  the detections and stays put on the ice while the camera pans, the origin marker sits on the painted center dot).

##### Key display (`measure/keys.py`)
- Each press adds a white keycap at the right end of a row (region y 760-900, x 900-1800 at 1080p). Verified frame by frame: a
  new cap appears RAISED (face top y = 29 in the region), sinks FLAT (y = 37) within 3-4 frames while held, rises back to raised
  on release and stays ~1.0-1.1 s before it disappears; a re-press while it lingers adds a new cap. So held = from the frame a
  cap appears (or starts sinking again) to the frame it starts rising. Labels from glyph templates (24x24 dark-pixel maps of 15
  reference caps, `REFS`): W A S D Shift R Alt Ctrl C 4 5 CapsLock (no Space: lift was never pressed). Unit test on a scripted
  cap sequence (press, hold, release, linger, re-press).
- Readings vs the protocol: wkeymovement W held 3.15 / 3.95 / 3.85 / 3.30 s (protocol "W ~4 s"), each after an R reset
  (0.10-0.12 s taps); dashatrest: Shift alone 0.42 / 0.23 / 0.88 s (no motion), then W+Shift pressed within 1 frame of each other
  for 4-7 frames, 6x; diagonal: D/A and W pressed within 1 frame, held 2.25-3.55 s. Checked on frames (W 528-765, Shift 80-105,
  W+Shift 161-168).
- **Key -> motion lag: 0.000 s median** (mean -0.009 s, sd 0.031 s, n = 12 standing starts; fit 0.5 a (t - L)^2 to the first
  0.35 s): the display and the game agree to within capture timing. Scenarios use the median (0).
- Mapping: move_x = W - S, move_y = A - D (screen up = +x, left = +y), dash = Shift **only with a move key** (Shift alone does
  nothing in the game; the sim would dash along `facing`), stick target constant, initial state = measured position, v = 0
  (runs start at rest), puck parked in a far corner. 25 runs (`measure/results/skate/runs.json`, `calib/data/skate/`).

##### Player fit (`python -m measure.skatefit fit`; calib Nelder-Mead, time objective; params.player.json, provenance `fitted`)
Hold-out by clip: fit on the other two skating clips, test on the held-out one. pos_rmse median per run (m) / speed_rmse (m/s);
placeholder = params.json (accel 12, max 8, drag 2), s6 = video-only (15.85 / 11.84 / 26.02).
| test clip (runs) | fitted on the others: accel / max / drag | placeholder | s6 video-only | held-out fit | all-clip fit |
|---|---|---|---|---|---|
| wkeymovement (4) | 16.51 / 10.65 / 6.04 | 5.69 / 2.73 | 2.45 / 2.69 | **0.48 / 1.91** | 0.46 / 1.91 |
| diagonalkeypresses (6) | 18.02 / 10.53 / 5.93 | 5.23 / 2.79 | 2.69 / 2.95 | **0.53 / 2.23** | 0.51 / 2.24 |
| whilemovingturning (5) | 14.55 / 10.75 / 6.04 | 7.28 / 3.34 | 2.64 / 2.89 | **0.75 / 1.97** | 0.51 / 1.98 |
Final (all three clips), placeholder-scale m and px at s = 16.951 px/m (real-scale m = x 1.087, see scale):
| param | s7 fitted | px | real-scale estimate | placeholder | s6 video-only |
|---|---|---|---|---|---|
| player_accel | **16.47 m/s^2** | 279.1 px/s^2 | 17.9 m/s^2 | 12 | 15.85 |
| player_max_speed | **10.62 m/s** (model-free plateau 10.755, sd 0.035, n = 15, dead flat) | 180.0 px/s | 11.5 m/s | 8 | 11.84 |
| player_ice_drag | **5.97 m/s^2** | 101.2 px/s^2 | 6.5 m/s^2 | 2 | 26.02 |
| dash_speed | **9.25 m/s (compromise)** | 156.7 px/s | 10.1 m/s | 6 | - |
- Speed RMSE (~1.9-2.2 m/s) is dominated by frame-difference noise (capture jitter, as for the puck); position RMSE is the
  meaningful number. Curves: `measure/results/skate/qa/player_speed_fits.png` (inspected).
- Diagonal input has the same top speed as a single key (normalized, like the sim). Turning at speed (W -> W+D): the speed stays
  at the cap while the velocity rotates over ~0.5-1 s, which the sim's direct acceleration reproduces (held-out 0.75 m on the
  turning clip); no separate turn parameter was fitted (player_turn_rate only rotates `facing`, which is not observed).
- **Coasting is not a constant deceleration**: after release the speed decays roughly in proportion to speed (rate 0.52-1.34 1/s,
  median ~0.85; better than a constant in all 11 coasts, coast ~3 s from 10 m/s). The sim's constant player_ice_drag (5.97) is a
  compromise; the s6 value 26 m/s^2 (stops in 0.07-0.25 s) must have been active braking.
- **Dash = a speed boost to ~12 m/s, not an impulse**: W+Shift from rest peaks at 11.0-12.5 m/s within 0.12-0.22 s (10-90% rise
  0.07-0.15 s), from top speed it adds only 0.45-3.3 m/s (peak 12.2-13.3), from 2.6 m/s it reaches 11.5 (fits.json `dash_bursts`).
  The sim's additive impulse can only compromise (dash_speed 9.25 on both clips; held out it fails both ways: fitted on
  dashwhilerunning (6.41) it gives 4.71 m on dashatrest vs 5.19 for the placeholder, fitted on dashatrest (9.82) it gives 3.35 m on
  dashwhilerunning vs 2.61 for the placeholder). dash_duration fits to 0 (the sim's raised-cap window doesn't match) and
  no clip has two dashes within a second, so neither dash_duration nor dash_cooldown is written (placeholders stay).

##### Absolute scale (`python -m measure.goalscale scale` -> measure/results/scale.json)
Readouts read from the banner text: 13.42, 65.63, 67.14, 35.94, 11.24, 88.05 mph. The banner appears in the frame the puck
crosses the goal line. Puck speed (px/s, camera motion removed) over the last <= 0.2 s of the flight before the line; flight
start = earliest frame from which the track still fits a free puck under linear drag (k = 0.4725). Goal-clip scale: 16.940 px
per placeholder m (placeholder m_per_px 0.05903).
| goal (mph) | ground/air | flight s | px/s at goal | m/px at goal | px/s at hit (drag-corrected) | m/px at hit |
|---|---|---|---|---|---|---|
| 13.42 | ground | 1.37 | 94.2 | 0.06368 | 171.4 | 0.03500 |
| 65.63 | airborne | 0.50 | 501.1 | (0.05855) | 605.4 | (0.04846) |
| 67.14 | airborne | 0.38 | 515.9 | (0.05818) | 589.8 | (0.05089) |
| 35.94 | ground | 0.60 | 252.0 | 0.06375 | 319.2 | 0.05034 |
| 11.24 | ground | 1.37 | 77.8 | 0.06462 | 141.5 | 0.03551 |
| 88.05 | ground | 0.08 | 585.7 | 0.06721 | 597.3 | 0.06590 |
- **Estimate: 0.06418 m/px (median of the 4 ground goals; mean 0.06481, sd 0.00165, CV 2.5%) = 1.087 x the placeholder scale**
  (implied rink_length 65.2 m). Corrected back to the hit with the fitted drag the same goals scatter by 31% (0.035-0.066), so the
  readout most likely measures the puck speed **at the goal line**, not at the hit, despite the "SHOT SPEED" label. Airborne goals
  are excluded (the drawn puck is displaced by height; they read 0.058).
- ASSUMPTION: the readout is in real units (mph of the game's own world). If it is, the real/placeholder factor is 1.087 for every
  length, speed and acceleration (times and rates unchanged). params.json is NOT rescaled (the next integration step applies it
  to everything at once).

##### Verified (s7, Linux cloud container)
- `python -m unittest discover -s measure/tests -t .` 23 tests (incl. the synthetic panning test and the key-tracker test),
  `cargo test --workspace` 43 tests (incl. 2 new calib player-track tests: replay samples the player, the fitter recovers
  player_accel 9.0 from 13.0), `npm run build` pass.

##### Open (s7)
- Apply the scale (1.087) together with the rink size; then rescale params.player.json and the puck numbers in one step.
- Sim model gaps found: coasting is speed-proportional (the sim: constant), the dash is a speed boost to ~12 m/s (the sim:
  additive impulse + raised cap), Shift without a move key does nothing (the sim dashes along `facing`). Changing these is a
  sim change for a later session; then refit dash and coast from these runs (`calib/data/skate/`).
- Not measured: dash cooldown (record repeated Shift presses 0.2-1.0 s apart while holding W), braking with the opposite key at
  speed (record W -> S at top speed), backward/sideways speed differences, player height parallax.
- More goals on the ice (not lifted), with long flights, would pin down what the readout measures beyond 4 goals.

#### Stick (session s7b, 2026-11-04, Linux cloud container) – video-only stick tracker, chase and hit fits
New files only: `measure/stick.py` (`python -m measure.stick {track,qa,scale,shape,chase,hits,params}`), `stick_scale.py`,
`stick_shape.py`, `stick_chase.py`, `stick_hits.py`, `stick_params.py`, `stick_synth.py` + `measure/tests/test_stick.py`,
`sim/examples/stick_step.rs` (sim stick step response; the Python chase emulation matches it to 1e-15 rad),
`calib/data/stick/hit_01..07` (scenario + CSV per hit), **`params.stick.json`** (params.json untouched, not merged).

##### Clips (clip-data, all 1920x1080, 60 fps)
- **rotatinganddizzy.mkv** (7.4 MB, 2683 frames, 44.7 s) = the **sweeps**: camera zoomed in (~4.2x on the ice), player standing on the centre
  dot, stick turned by the mouse: slow circling, several multi-turn spins at the speed cap, stops, reversals, small wiggles at the end.
- **hittingpuck.mkv** (11.9 MB, 4883 frames, 81.4 s) = the **hits**, at the normal zoom (whole rink, ~4% closer than the old clips: board
  949 px, 0.06322 m/px; `measure map` + `measure track` run on it): puck put at the forehand spot (practice reset), 7 swings slow/medium/fast.
- staticrotatingstick: not usable for the stick (pink skin, 14 px stick, camera travel; the tracker finds the body in 2.9% of frames).
  Also new on clip-data, not used here: dashatrest, dashwhilerunning, diagonalkeypresses, whilemovingturning, wkeymovement, goalspeeds_part00-03.
- The 2026-11 clips show the player as a **grey body** (S = 0) in a blue glow, not the pink disc; the stick is a black shaft along the
  side of the body ending in a white blade. **The body markings turn with the body (facing), not with the stick.** With the left mouse
  button held (green dot in the mouse-hint icon) the stick takes another pose (shaft along the blade).

##### Tracker
- Pivot = centre of an ellipse fitted to the convex hull of the grey body (opened with a 0.3 body-radius disc so the shaft can't join it).
  Angle = circular cross-correlation of a polar unwrap (black shaft inside the body; black + white outside it; each radius ring minus its
  mean) against a template built from the clip (3 passes); features that stay put on screen are masked; direction pivot -> blade.
  Duplicated capture frames (zero step followed by a ~2x catch-up step) are dropped; a hard stop (zero step, no catch-up) is kept.
- Synthetic test (cap-speed spins, hard stops, reversal, slow sweep, separately turning markings, puck, screen-fixed icon, HUD, dup frame):
  max error 0.24 deg, median 0.04 deg; tip radius and pivot within 1 px.
- rotatinganddizzy: **100.00% (2681/2682), longest gap 0**, 1 duplicate dropped, conf median 0.93 (0.5 in the left-button pose, angle
  still on the blade). hittingpuck: **99.80% (4870/4882), longest gap 6 frames** (2452-2457), conf median 0.75; there the angle carries a
  constant ~15-20 deg offset from the blade (the blurry template mixes in the shaft): rotation rates are right, absolute angles are not.
- QA (inspected): `measure/results/qa/{rotatinganddizzy,hittingpuck}.stick_sheet.png`. Fixed because of them: body markings removed from
  the template (they turn with the body), body blob merging with the stick (10 px pivot error at frames 982-995), the 39.6 rad/s "peak"
  was a duplicated frame, stick snaps at practice resets are not swings.

##### Scale
- Disc scale (as asked): grey body radius by the same half-max edge at both zooms (8.82 px at hittingpuck = 0.558 m at its map scale;
  49.72 px zoomed) -> 0.01122 m/px zoomed. **Cross-check with the centre circle (an ice marking, 314.5 px): 0.01498 m/px — 1.34x apart.**
  The player model is drawn 1.34x larger relative to the ice when zoomed: the camera zooms by moving closer and the body top is raised
  (dolly model: top at 0.305 of the zoomed camera height, ~8% magnification at the normal zoom; ground radius 0.518 m vs params
  player_radius 0.4557 m, which is the pink skin's half-max disc). So: **ice scale for things on the ice, disc scale only for the body.**
  params player_radius cannot set the zoomed scale (different skin): 0.009166 m/px would be 39% off. Angles need no scale.

##### Chase (rotatinganddizzy; step model with a free onset per event)
- **stick_max_ang_speed = 26.17 rad/s** (hard cap: flat plateaus in every cap-limited move, 344 frames, per-frame sd 1.65 = capture
  jitter; slow mouse-driven moves peak <= 20.7). Placeholder 14.
- Held out by event (4 step events: +134 deg, and spins of -1971 (the fast sweep, 5.5 turns), -1480, -1021 deg): angle RMSE
  **3.16 / 2.34 / 1.87 / 3.01 deg** vs placeholder (14, 120) **7.9 / 363 / 251 / 151 deg**.
- **stick_max_ang_accel = 514 rad/s^2** from the 3 flick starts (rest -> cap within 2-4 frames; RMSE 3.7 deg at 400 vs 4.2 at >= 1500).
  The joint step fit gives ~380 (biased low by mouse-limited starts: 6-11 frame ramps). **Stops are harder: flat above ~1500 rad/s^2**
  (60 fps can't resolve more), and the real stick **overshoots the final angle by 0.7-12 deg and settles back over 2-5 frames**; the
  sim's symmetric bang-bang braking cannot do either (stop residuals -5..-8 deg then +3..+7 deg). Placeholder 120.
- The game kept spinning the same way for 5.5 turns while the mouse circled: the fit uses the unwrapped final angle as the target (the sim
  wraps the target error, which is equivalent while the mouse leads by < half a turn).

##### Shape (measured-shape, placeholder meters)
- **stick_length = 1.088 m** (pivot -> blade tip on the ice plane): half-max tip 17.62 px at the normal zoom (1.113 m at its scale) and 80.74 px
  zoomed (1.210 m at the ice scale); the dolly model puts the tip at 0.10 of the zoomed camera height and gives 1.088 m on the ground.
  Consistent with the hits (resting-puck contact radii 0.95-1.06 m, perpendicular exits). Template-threshold tips (0.89-0.91 m) under-read
  (the 1.4 px blade fades at the normal zoom); the sim with 0.90 m hit the puck with the tip and sent it 50 deg off.
- **stick_thickness = 0.054 m** (half the blade FWHM, 8 px zoomed, height-corrected). Placeholder 0.05.
- Not written: stick_inner (the white blade starts at the body edge, ~0.57 m at the disc scale; the shaft crosses the body off-centre).
- **Not measurable from top-down video: stick_z_min / stick_z_max** (the height range the stick can hit) and the shaft's real 3D shape.

##### Hits (hittingpuck, placeholder ice-plane scale 0.06322 m/px; w = stick angle change over the frame interval containing the contact)
| frame | group | w rad/s | r_c m | v_head m/s | v_exit m/s | exit/head | exit dir vs perpendicular |
|---|---|---|---|---|---|---|---|
| 584 | slow | -8.6 | 0.959 | 8.3 | 7.3 (115 px/s) | 0.88 | -2.3 |
| 4289 | slow | +9.2 | 0.965 | 8.9 | 6.8 (108 px/s) | 0.77 | -4.0 |
| 3910 | slow | +10.8 | 1.002 | 10.8 | 17.7 (279 px/s) | 1.64 | -7.2 |
| 3471 | medium | +20.7 | 0.952 | 19.7 | 19.8 (314 px/s) | 1.01 | -7.2 |
| 861 | medium | -23.6 | 0.965 | 22.8 | 27.4 (434 px/s) | 1.20 | +6.3 |
| 3159 | medium | +25.2 | 1.056 | 26.6 | 31.4 (497 px/s) | 1.18 | -2.0 |
| 280 | fast | -31.3 | 0.951 | 29.8 | 28.6 (452 px/s) | 0.96 | +3.7 |
- Exit speed: drag-model fit (k = params ice_linear_drag) over 31 rows from 2 frames after contact, extrapolated to the contact.
- **Linear**: v_exit = 1.09 v_head through the origin (rms 3.3 m/s; with intercept 1.06 v_head + 0.7). Single-frame w noise 0.5 rad/s at rest,
  ~3-5 rad/s while swinging (capture jitter), so the ratio scatters 0.77-1.64. **Saturation: not established** — the two fastest exits
  are 31.4 and 28.6 m/s (ratio 1.18 and 0.96); a cap near ~30 m/s (placeholder scale) is possible but rests on 1-2 hits. Head speeds are
  limited by the 26 rad/s cap x ~1 m, so faster swings can't be recorded at this contact radius.
- **Exit direction is within -7..+6 deg (mean -1.8) of perpendicular to pivot->puck**, as if the stick met the puck centre-on. The sim's
  contact (reach = puck_radius + stick_thickness = 0.28 m) touches it ~17 deg earlier and sends it ~15 deg outward: **a contact-model gap**
  (path RMSE 0.8-3 m in every hit, for any mass/restitution). Effective game reach from the direction: ~0.05-0.12 m.
- calib (base: params.json + chase fit + measured length/thickness, puck speed caps lifted to 200 m/s; the sim stick starts one substep
  short of the puck at the measured w, target set so the chase law holds w): **stick_mass and stick_restitution are degenerate** — the
  joint fit runs to a rigid stick (mass 1e12-1e13 kg, e = 0.136-0.138 in every hold-out fold): only exit/head ~1.14 is measured, i.e.
  (1+e) / (1 + 3 m_puck r^2 / (m_stick L^2)). With **stick_mass fixed at the placeholder 2 kg: stick_restitution = 0.360** (held out by
  swing speed: slow 0.361, medium 0.348, fast 0.364; placeholder 0.4). Held-out speed RMSE fitted / placeholder (m/s): slow 3.6/3.8,
  4.5/4.8, 3.6/3.3; medium 3.4/4.2, 1.2/1.3, 2.0/1.2; fast 1.1/1.4 (5 of 7 better). stick_mass is NOT written (placeholder stays).
- Not separable here: the stick's angular-speed drop at contact (it would fix the mass; 60 fps + the chase re-accelerating hide it).

##### params.stick.json
stick_max_ang_speed 26.17 rad/s (fitted), stick_max_ang_accel 514 rad/s^2 (fitted), stick_length 1.088 m and stick_thickness 0.054 m
(measured-shape, placeholder meters: rescale with the absolute scale), stick_restitution 0.360 (fitted, conditional on stick_mass 2 kg
and puck_mass 0.17 kg, both placeholders).

##### Stick: still open
- Contact model: the game's exit direction says the stick meets the puck centre-on (or pushes along the swing); the sim's 0.28 m reach
  turns shots ~15 deg outward. Needs a sim change (out of scope here) before mass/restitution can be fitted on paths.
- Asymmetric chase (fast stops with overshoot) is not representable with one accel; a separate brake decel + overshoot would be.
- stick_mass needs a measurement that sees the stick slow at contact (higher fps, or hits on a heavy/stationary puck at several radii);
  stick_inner, z range and the left-button pose's geometry are unmeasured; the hit table has only 7 hits (more fast swings at
  different contact radii would test saturation and the puck speed cap).

## s10 (2026-11-04): player coast, dash, stick braking and hit contact models
- Replaced the constant-drag coast (now speed-proportional, player_coast_rate), the additive dash (now a ramp toward dash_speed, move key required), the symmetric stick chase (separate stick_max_ang_decel) and the stick impulse direction (centre-on). Refit held out by clip / event / hit, with the old models refit the same way for comparison. An overshoot term and a blade-normal impulse were tested and rejected. Hit scenarios rescaled to real meters; bot shot constants re-measured. Details: CLAUDE.md "s10 models".
