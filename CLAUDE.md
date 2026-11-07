# slapmini – project facts

A compressed, top-down 2.5D browser version of Slapshot: Rebound (WASD to move, the mouse angle rotates the stick around the player),
built to train RL bots. The physics must eventually match the real game. Ian is recording practice-mode clips for calibration.

## Current state (m4, 2026-11-06; Windows, RTX 3070 Ti, 12 cores)
This section wins over anything older below. Session narratives and old verified lists are in docs/history.md.

### Hashes and stamp (final code, native = wasm)
| check | hash |
|---|---|
| free play | `562f7223be45fe73` |
| match | `6abaa84c491b49a1` |
| s10 shots | `0e7d2a60a9fcc55d` |
| raised stick | `f3acae2492133ab0` |
| physics stamp | `e750f45cff93780c` |
| style zero = default, Rust | first_goal `1dd3d7ddac18af58`, match `f877960d1338cb10` |
| style zero = default, Python | first_goal `2e5ada33d675b1ad`, match `95f38f2c07175173` |
| threads 1 = 4 | first_goal `e65794f73e11b878`, match `8516b7fd23e9fb7c` |

Formats: **ACT v4** (6 floats: move index, stick angle, dash, lift, stick_up, brake), **OBS v3** (42 floats), **ABI v4**. Team rotation leaves
stick_up and brake alone. Scripted bots never brake or raise the stick. Older checkpoints and sidecars are refused.
Passing: cargo 78, measure 31, train 16, `npm run build`, 12 verify scripts (each picks a free port, `web/scripts/free_port.mjs`; `VERIFY_PORT` overrides).
Obs crosscheck 684 observations, 28,728 fields, 0 mismatches; 684 actions (174 braking), 0 mismatches.

### Sizes (`python -m measure.size_ratios --clips DIR`)
Radius / inner board length in one camera, so scale-free; height corrected with the two-zoom camera model.
| ratio | as seen, normal zoom | height corrected | sim |
|---|---|---|---|
| body / rink | 0.00889 | 0.00817 ± 0.00023 | 0.00817 (player_radius 0.5329 m) |
| puck / rink | 0.00431 (zoomed 0.00417, no measurable height) | 0.00435 ± 0.00022 | 0.00417 (puck_radius 0.2721 m) |
| body / puck | 2.064 | 1.88 ± 0.11 | 1.958 |
The real game draws the body at its height, 9% above ground size at normal zoom; the sim draws ground size. rink_length is unchanged (radius/rink ratios cannot test the scale).
Calib, old radii → new, path objective, 25 real segments: all 0.5583 → 0.5620 m (+0.7%), 9 bounces 0.2734 → 0.2767 (+1.2%), 6 ground bounces 0.2627 → 0.2578 (−1.8%). Under the 10% limit: board_restitution and ice_linear_drag not refit.

### Stick and hit rule
Three capsules in the stick frame (+x = stick_angle), `python -m measure.stick_outline`, held-out IoU 0.472 / 0.834 / **0.890** for 1 / 2 / 3 capsules: hand (−0.0721, 0.5051),
kink (0.4324, 0.1252), heel (0.7725, 0.0053), toe (1.1035, −0.0186); half-widths 0.0496 / 0.0754 / 0.0567 m. `step.rs` and `render.ts` read the same params; figure `measure/results/stick_chain.png` is from params only.
Hit rule: centre-on (`stick_hit_rule` 0): direction RMS 5.1° against 16.9° for the surface normal, exit ratio 1.07–1.11. The early touch (19–27°) is not explained by the shape.

### Contact rules (`sim/src/step.rs`, test `sim/tests/contact_stress.rs`)
- Stick and body are swept per substep. `resolve_overlaps` pushes along contact normals; boards and nets win a pinch: the stick turns back and rebounds, the player is pushed off.
- **stick_ghost** (Player field, serde default false, hashed only when set): a stick that is on the puck at the start of a substep by more than 0.02 m without a swing (lowered onto it, a puck dropping into its height range, a reset) ignores the puck until they are apart. Trace that led here: step 1752, stick up → down, gap −0.307 m, puck jumped 0.37 m.
- A puck that falls into a body's height range while inside it slides out at 6 m/s.
- Stress run, 100,000 steps: 564 contacts, max penetration 0.1595 m (limit 0.2), end-of-step overlaps 0, puck outside the rink 0, ghost 50 cases / 211 player-steps, body settling 23 player-steps, 265 pinch stops, 448 push-offs.
- Longest contact run 6 steps in open ice (limit 7: stick spin-up is 26.17 / 513.9 s = 6.1 steps, pushing on each) and 19 against a board, net or pinch (limit 20: the stick leans on a pinned puck and the rebounds break it up slowly).

### Skating and brake (w5b; comparison in docs/history.md)
Along the key player_accel 16.9508, against it player_brake_accel, across it player_turn_accel 3.2792 (turns the velocity, keeps the speed), coast 1.0751 1/s, cap 11.6935 m/s. Space = slow walk (`PlayerInput.brake`, brake_speed, brake_decel). Panel: F2 in the game, same panel in the lab; edits are stored in localStorage and reach training only through params.json.

### Tuning to the repo
`npm run adopt-params -- exported.json [--dry-run]` (in web/): writes changed values, sets provenance `tuned-by-feel` with a dated note, prints a diff; unknown names, negatives (chain coordinates excepted) and values over 10× the current one write nothing. The pin test accepts the tag only with that note.

### Bots
`hit_radius` = stick_inner + 0.13 = 0.90 m, SHOT_SPEED 26.5, SHOT_OVERTURN 0.020; shotlab 48/48, median |err| 0.002 rad. A stall breaker (1 s still beside a still puck → 0.75 s punt) ends the deadlock where two chasers pressed together beside the puck for a whole match.
| A–B goals/match, 100 × 60 s | main | merge before bot fixes | final |
|---|---|---|---|
| chaser–chaser | 0.53–0.39 | 0.56–0.41 | 1.16–1.03 |
| chaser–defender | 0.54–0.01 | 0.36–0.00 | 0.71–0.01 |
| chaser–cautious | 16.98 | 17.00 | 16.00 |
The chaser–defender drop was the bot aiming at the blade toe (1.09 m) with the wrong SHOT_SPEED, not body size (old radii 0.34). The w5b chaser–cautious drop was the turn term on that branch (16.85 without it, 16.03 with); 16.00 now goes with the slower shots at the new radius (inferred, not tested).

### Smoke runs
- Smoke bot (`web/public/bots/smoke.onnx`, 326 KB, SMOKE TEST; 10 min CPU, 16.6M decisions, chaser only): eval vs chaser 64 matches win 0.03, loss 0.97, goals 1.50–5.27. In the browser mean stick_up 0.339, brake 0.195. It is weaker than the t2 smoke bot, which is expected of one short run and not investigated.
- 3 min match smoke (`train/results/m4_match3_*`): goal diff stays −5.00 and return −6.99, so results barely move; match length 17.9 → 32.2 s, goals for 0 → 0.13, entropy 6.79 → 4.65, stick_up 0.48 → 0.20, brake 0.51 → 0.16.
- Long-run command, GPU, 2 min then resume to 3 min: 11.8M → 16.7M decisions, ~105k decisions/s, metrics.csv with stick_up and brake columns, learning_curve.png, latest.pt; loss 0.30 → 0.26, stick_up 0.54 → 0.60, brake 0.36 → 0.28.

### Placeholders (invented or chosen, not measured)
stick_mass (2 kg); max_puck_speed_air, max_puck_speed_ice_normal, max_puck_speed_ice_dash; stick_up_z_min, stick_up_z_max; stick_z_min, stick_z_max; brake_speed, brake_decel, player_brake_accel; dash_duration, dash_cooldown; the pinch rebounds PINCH_REBOUND_OMEGA 8 rad/s and PINCH_REBOUND_SPEED 0.6 m/s, BODY_SETTLE_SPEED 6 m/s and GHOST_DEPTH 0.02 m (constants in step.rs); camera zoom and the follow camera; the stick lead limit; plus the rest of "What the clips could not give".

### Open
- The first long run has not started: README "Long run on Ian's machine".
- Whether a lowered stick really ignores the puck in the game is unmeasured; the ghost rule is a choice.
- Display-only body magnification; toe height; stick vs board and stick vs stick do not collide; the defender and cautious bots never score.

## Architecture
- Cargo workspace with `sim`, `calib`, `sim-wasm`, `bots` and `sim-ffi` (see "Bots and training"), plus `train/` (Python PPO), plus `web/` (the feel lab: Vite + TypeScript, no framework) and `measure/` (Python: real clips → trajectory CSVs). Commands are in README.md.
- **sim** (pure, deterministic): `step(&mut State, &[PlayerInput], &Params)`. No I/O, no clock, no RNG. Trig goes through `libm`, so native and wasm give identical bits.
  Deps: serde, serde_json, libm. `params.json` is embedded with `include_str!` as `Params::default()`.
  - `State { tick, puck: Puck{x,y,z,vx,vy,vz,in_net}, players: Vec<Player>, score }`. The player count is not hardcoded (`State::new_default(p, n)`, tested with 6).
  - `Player { team, x, y, vx, vy, facing, stick_angle, stick_omega, dash_timer, dash_cooldown, dash_prev }`.
  - `PlayerInput { move_x, move_y, stick_target_angle (world rad), dash, lift }`.
  - Each step: controls update once, then `substeps` collision substeps (players → board → player-player → puck integrate → body/stick → nets → boards), then goal detection.
  - Rink: a rounded rectangle centered at the origin. Nets are 3 static segments each (sides + back) at ±`net_x`, mouths facing center. A puck in net 0 (+x) scores for team 0.
  - Stick: a rod pivoting at the player center, I = m L²/3. The angle chases the target with a braking bang-bang profile (max ang speed + accel). The impulse uses stick-head velocity (v_player + ω×r) and feeds back into ω. It only collides when `stick_z_min ≤ z ≤ stick_z_max`.
  - Speed caps after a stick hit: air / ice-normal / ice-dash, kept separate.
  - `State::to_bytes()` / `hash()` (FNV-1a) are for determinism checks.
- **params**: a flat `Params` struct generated by a macro. `get/set` by name for the fitter, and `ParamFile` keeps provenance/unit/note.
- **calib**: library + CLI (`eval`, `fit`, `synth`, `replay`). It replays a scenario, interpolates the sim at the CSV times, and reports pos RMSE (3D, m), speed RMSE (finite differences, m/s) and an optional stick RMSE.
  The fitter is Nelder-Mead in start-value-scaled coordinates. Default objective (`--objective time`) = Σ pos_rmse² + 0.1·speed_rmse².
  `--objective path` (timing-robust) = Σ path_rmse² + 0.01·v2_rmse²: both paths are smoothed identically (local quadratic fits, ±5 samples;
  the sim is first sampled at the measured times) and compared at equal distance travelled, by position and by v² (v² falls linearly
  with distance under constant friction). Frame times never align the two paths, so capture jitter and dropped/duplicated frames are
  only zero-mean noise. `eval` prints both sets of metrics. Fitted names get provenance `fitted` in the output file.
  Synthetic self-check data (generated from the sim itself) is in `calib/data/`, made with the PINNED `calib/data/params.synthetic.json`
  (the original all-placeholder params, old 60×26 m rink); `calib/tests/fit_recovery.rs` uses that file, so params.json edits don't move it.
  Real measured segments are in `calib/data/real/` (see "Segment manifest").
- **sim-wasm**: a `Sim` class with `new(params_json, n_players)`, `step(inputs_json)`, `get_state()`, `set_state(json)`, `set_params(json)`, `get_params()`, `reset(scenario_json)` (loads `initial_state`), `start_match`, `state_hash()`, and the bots crate compiled in: `observe(player, style)`, `action_input(...)`, `set_scripted`, `scripted_input` (see "Game (w1)"). JSON crosses the boundary in the same shapes as params.json and scenario files.
- **web/**: the game at `index.html` (see "Game (w1)") and the feel lab at **`lab.html`** (described below). `npm run wasm` builds sim-wasm into `web/wasm-pkg/` (gitignored). `npm run dev` serves both, and `npm run build` (wasm + tsc + vite, two pages) writes a static `web/dist/` with relative paths. `npm run verify` is the lab's Playwright check.
  - The app imports the repo-root `params.json` directly, so param edits show up without a wasm rebuild.
  - `src/lab.ts`: sim owner with no DOM. Covers stepping, record/replay and practice tools.
  - `src/loop.ts`: fixed-step accumulator, capped at 5 steps per advance; the overflow is dropped and counted. It doesn't render, so it can run headless.
  - `src/render.ts`: pure drawing. `src/panel.ts`: param inputs. `src/main.ts`: wiring, plus the only wall-clock reads (`performance.now`).
  - `src/bindings.ts`: every key binding in one file.
  - Rendering: rink from params (blue lines, faceoff spots and the speed scale are DISPLAY ONLY params).
    The puck goes blue → red as z goes from 0 to 2 m, and its shadow grows and offsets with z. The puck is drawn at a 6 px minimum radius (display only).
    The stick is drawn at its actual angle, with a faint dashed line at the target angle.
    "Rink vertical" checkbox (default on, like the real camera and the game page): +x up (net 0 at the top), +y left; off (or `?view=horizontal`) = the old horizontal view. WASD stays screen-relative. Mouse: relative (default) or cursor aim (`?mouse=cursor`). See "View, mouse and Space (w2)".
    The ghost path breaks at time gaps and at jumps faster than 3× the largest speed cap (teleports in measured tracks).
  - Record: snapshots `initial_state` and logs an input event whenever a player's input changes. Samples t,x,y,z,stick_angle each step in the calib CSV format.
    Replay: `reset(scenario)` and then feeds the events. Keyboard and mouse are ignored during replay.
  - Practice tools change the state only in play mode; they are disabled while recording or replaying, so recordings stay pure input streams.
  - Ghost: loads a CSV, draws its path faded, and the scrub slider moves a marker along it.

- **measure/** (Python 3, numpy/scipy/opencv-python-headless, venv in `.venv/`, gitignored): `python -m measure {map,track,qa,synth}`.
  Outputs in `measure/results/` (committed: map/track JSON, traj CSV, QA PNGs). Cached median frames go to `clips/NAME.bg.png`.
  - `map`: median of 61 frames → inner board outline (the ice region enclosed by the light-grey boards and their yellow inner strip),
    straight-side extents, corner radius (from the area deficit), center line (saturation peak), center dot (the dark-ring brightness dip pair
    on the center line). Shape block: ellipse fits to the center + 4 faceoff circles (ray sampling, outlier rejection), grey lines (goal/blue
    lines), net roofs. Camera: 4 faceoff-dot patches (81×81, painted on the ice) matched once per second; raw drift = median shift, residual =
    what remains after undoing it by a translation. The map fails only if > 10% of samples can't be corrected to ≤ 2 px.
  - `track`: green HSV blobs inside the rink outline (+8 px), minus the net roofs (a large textured patch near each end, found from the median
    frame). Blobs that are green in the median frame are tagged static and may only extend an existing track at its prediction.
    Greedy linking with a gate of `2 × max speed cap / fps / m_per_px` (≈23 px/frame). Across gaps the allowance grows only with the
    track's own speed, so teleports start a new segment. Only segments that travel (≥12 px extent, ≥5 moving frames) count as the puck.
    Duplicated capture frames are dropped (compared with the previous KEPT row, so runs of repeats go too); gaps of ≤6 frames inside a
    segment are interpolated. **Camera correction**: every frame's translation is measured from the faceoff landmarks and undone
    (subpixel warp) before detection; every 10th corrected frame is re-measured (residual); frames that still exceed 2 px are dropped
    together with the stretch between them. Rows store the frame's shift and the blob area. `--no-camera-correction` = old behaviour.
    (Board-corner landmarks were tried first: the boards stand above the ice and their parallax left ~1.3 px when the camera panned.)
  - `scale`: board-based vs ice-plane scale per clip (both stored in the map JSON, `extra.scales`). `reproject`: puck blob
    centroids → ground points (`ug_px`/`vg_px` in track.json; the CSV uses them); `track` runs it automatically.
  - `segment`: free-puck stretches → `calib/data/real/` scenario + CSV pairs and `manifest.json` (see "Segment manifest").
    `segqa`: per-clip segment sheets and frame strips. `estimate`: scale-free physics → `calib/data/real/estimates.json`.
    `params`: writes the measured rink shape + markings into params.json (provenance `measured-shape`).
    `holdout`: clip-held-out fits → `params.fitted.json` + `calib/data/real/holdout.json` (needs `cargo build --release -p calib`).
    `dragmodel`: held-out drag-law comparison + smoothing/synthetic checks → `calib/data/real/dragmodel.json`.
    `compare SEG --params A --params B`: overlay of a real segment and sim replays (path + timing dots + v vs s) → `qa/compare_SEG.png`.
  - s7 modules (moving camera + key display): `camreg.py` (camera registration on the painted lines and side boards), `keys.py`
    (key-display reader → input events), `skate.py` / `skatefit.py` (player runs → `calib/data/skate/`, player/dash fits →
    params.player.json), `goalscale.py` (absolute scale → `measure/results/scale.json`). A scenario may set `"track": "player0"`.
  - s7b modules: `python -m measure.stick {track,qa,scale,shape,chase,hits,params}` (stick tracker, chase and hit fits →
    params.stick.json, hits in `calib/data/stick/`).
  - Integration tools: `python -m measure.merge_models` (player/stick fits → params.json), `python -m measure.rescale` (scale).
  - Tests: `python -m unittest discover -s measure/tests -t .` (31 tests: synthetic videos with camera pan / following camera,
    key display, stick; geometry round-trips; rescale; no real clips needed).

## Coordinate convention (all measure/ outputs; matches the sim)
- Meters. Origin = the center dot (center ice). +x along the rink's long axis, toward the net at the TOP of the game screen.
  +y such that x then y turns counterclockwise as seen on screen, i.e. +y = screen LEFT. This is the sim's frame
  (+x = net 0, team 0 attacks +x); the game and the lab draw it vertically by default (+x up, +y left, as the game shows it);
  the lab's old horizontal view (+x right) is this rotated 90° clockwise.
- Pixel mapping: x = m_per_px · (v0 − v), y = m_per_px · (u0 − u), with (u0, v0) the center dot and v pointing down.
- **Scale**: m_per_px = params.json `rink_length` / measured inner board length in pixels; rink_length is now 65.2358 m, the
  ESTIMATED real length (see "Absolute scale"). Every output records this as `scale_source`. Outputs written before the rescale
  were in placeholder meters (rink_length 60); the committed ones have been rescaled. (s7 clips: the following camera is
  registered on the painted lines, `measure/camreg.py`, at px per placeholder m from the line spacing.)
- **Ice-plane scale** (`measure scale`): the board outline we measure is at ice level, not the board top. In the panned clip an
  elevated outline would widen the far-side gap between board edge and faceoff dots; measured k_board = −0.0024 ± 0.0014 (not
  significant, unphysical sign), and board length / faceoff spacing agrees across clips to 0.11% (1.8386–1.8406). So
  m_per_px_ice = m_per_px_board (both stored, `extra.scales`; `m_per_px` = the ice-plane one).
- **Puck ground position** (`reproject`): the blob centroid sits above the ice (top face + visible side wall), displaced outward
  from the principal point (image center) by k_top/2·r, k_top = (area slope)/(blob diameter) = 0.0104 / 0.0106 (puck / wall clip),
  0.0082 (lifted). Ground = pp + (c − pp)/(1 + k_top/2): up to 2.7 px (≈0.5% radial). Applied only if the blob grows ≥ 25% across
  the radii seen. Check: at bounces the ground center is 3.0–4.8 px from the board (puck radius 3.76 px), the raw blob was as close
  as 0.5 px. Effect on speeds: −0.5% (v0 −0.31…−0.59%, mean path speed −0.40%).
- t = frame_index / fps from the clip's first frame. z is always 0 (a top-down view cannot see height).


## Absolute scale (applied 2026-11-04, s7/s7b integration) – and its assumption
- **Sim meters are real meters now**, at an ESTIMATED scale: 0.06418 m/px at the goal clip's zoom = **1.0873 × the old
  placeholder meter** (rink_length 60 → **65.24 m**). Source: `python -m measure.goalscale scale` → `measure/results/scale.json`:
  the in-game "SHOT SPEED: xx.xx MPH" banner of 4 ground goals vs the tracked puck speed over the last ≤ 0.2 s before the goal
  line (median; mean 0.06481, sd 0.00165, **CV 2.5%**). Read at the hit (drag-extrapolated) the same goals scatter 31%, so the
  readout most likely measures the speed **at the goal line**. Two airborne goals are excluded (drawn displaced by height).
- **ASSUMPTION: the mph readout is in real units of the game's own world.** If the game scales its readout, every length, speed
  and acceleration below is off by one common factor; shapes, ratios and rates are not affected.
- Applied in one step by **`python -m measure.rescale`** (tests: `measure/tests/test_rescale.py`). Unit rule: a value scales by
  factor^p, p = the power of meters in its unit (m, m/s, m/s^2 → 1; s, 1/s, ratio, kg, rad, rad/s, rad/s^2, count, goals → 0;
  any other unit is an error). Rewritten: params.json (clip-derived values + rink_length; see below), `calib/data/real/`
  (25 scenarios, CSVs, manifest `meter_scale`), `calib/data/skate/` (25 runs), `measure/results/*.map.json|*.track.json`
  (m_per_px*, *_m, max_speed_m_s, `scale_source`) and `*.traj.csv`. It refuses to run twice (marker "REAL METERS").
  Not rescaled: `params.placeholder.json` and `calib/data/params.synthetic.json` (pinned by tests), `params.fitted.json`
  (old puck-fit output, placeholder meters), the references `params.player.json` / `params.stick.json` (placeholder meters),
  `calib/data/stick/` (hit scenarios), `measure/results/skate/` and the stick results.
- **Placeholder lengths were NOT scaled** (gravity 9.81, heights, puck speed caps, lift_max_vz, chip_max_speed,
  player_spawn_x): they were invented in real-world terms, not read off the placeholder scale. speed_display_scale = 1.0.
- Checks: fitted ratios and rates are bit-identical (board_restitution, board_friction, ice_linear_drag, stick angular values,
  stick_restitution). Every real segment's pos_rmse scaled by 1.08726 (ratio/factor 0.99999–1.00073, path_rmse 0.9947–1.00005,
  v2_rmse by factor² within 2e-5); the 25 skating runs the same (0.99995–1.00005).
- `measure` pipeline: `geometry.py` derives m_per_px from params rink_length, so re-running `map`/`track`/`segment` now writes
  real meters directly. Do NOT re-run `measure.rescale` on such output.

## Defaults (params.json) and provenance
Provenance tags: `measured` (model-free measurement), `measured-shape` (geometry from video), `fitted` (calib fit on real clips),
`rule` (Ian's match rules), `placeholder` (invented). Values in real meters (above). `sim/tests/sim_tests.rs` pins the tag per name.
| param | value | provenance | source |
|---|---|---|---|
| rink_length | 65.2358 m | measured | the absolute scale (60 × 1.0873) |
| rink_width / rink_corner_radius | 41.5311 / 9.4851 m | measured-shape | s6 board outline (length/width 1.571) |
| net_x / net_width / net_depth | 24.8082 / 6.4375 / 2.4206 m | measured-shape | goal line, parallax-corrected net roof (depth assumes a flat roof) |
| blue_line_x, faceoff_x/y, faceoff_circle_radius | 8.7525, 17.7281 / 11.3362, 5.1193 m | measured-shape | display only |
| puck_radius | 0.2721 m | measured-shape | w5a: puck / rink 0.00417 in the sharp zoomed clip (see "Geometry and stick contact (w5a)") |
| player_radius | 0.5329 m | measured-shape | w5a: height-corrected body / rink 0.00817 (`measure/size_ratios.py`) |
| board_restitution / board_friction | 0.342796 / 0 | fitted | s5/s6 wall bounces, held out by clip |
| ice_linear_drag / ice_friction_decel | 0.47254 1/s / 0 m/s^2 | fitted | glides; decel refit in real m/s^2 (below) |
| player_accel | 16.9508 m/s^2 | fitted | w5b refit with the turn term (was 17.3594); see docs/history.md |
| player_turn_accel | 3.2792 m/s^2 | fitted | w5b, turning clip (leave-one-run-out 2.9–3.7) |
| player_brake_accel | 16.9508 m/s^2 | **placeholder** | w5b, = player_accel: no clip presses against the velocity |
| brake_speed / brake_decel | 1.5 m/s / 25 m/s^2 | **placeholder** | w5b slow walk (Space) |
| player_max_speed | 11.6935 m/s | **measured** | model-free hard cap 10.755 (sd 0.035, n = 15) × 1.0873; the fit's 10.62 not used |
| player_coast_rate / player_ice_drag | 1.0751 1/s / 0 m/s^2 | fitted | coast dv/dt = −rate·v − ice_drag; rate refit in w5b (was 1.08146) |
| dash_speed / dash_ramp_time | 15.4803 m/s / 0.11631 s | fitted | s10 set-speed dash ramp (held out by clip) |
| dash_duration / dash_cooldown | 0.25 s / 1 s | **placeholder** | not identifiable from the clips |
| stick_max_ang_speed / accel / decel | 26.173 rad/s / 513.9 / 2316.1 rad/s^2 | fitted | s7b sweeps (cap plateau; flick starts); s10 stop decel |
| stick chain (stick_hand/kink/heel/toe_x/y, stick_shaft_radius, stick_heel_radius, stick_thickness = blade half-width) | see w5a | measured-shape | w5a: three capsules; stick_length 1.1036 m (pivot to toe; inertia and bots only), stick_inner 0.7725 m (bots only) |
| stick_restitution | 0.3597 | fitted, **conditional** | 7 hits, holds only for stick_mass 2 kg (placeholder) |
| match_length_s / mercy_lead | 300 s / 5 | rule | Ian |
| everything else | | placeholder | see "What the clips could not give" |
- Merge is reproducible: `python -m measure.merge_models` (from params.player.json + params.stick.json, before the rescale).
- **ice_friction_decel in real m/s^2**: refit jointly with ice_linear_drag on the 15 ground glides, path objective, start
  0.3 m/s^2 (start 0 can't move: the fitter scales by the start value), held out by clip: train wall (8) → 0 (drag 0.4691),
  train puck (7) → 0 (drag 0.4742). Both agree on 0 (the non-negative bound), so 0 stays.
- Real segments after the rescale (`calib eval`, default params): puck_02 pos 0.0567 → 0.0616 m, path 0.0166 → 0.0181 m;
  wall_03 0.366 → 0.398 m, 0.142 → 0.155 m; puck_08 0.820 → 0.891 m, 0.183 → 0.199 m (all × 1.087: same fit, bigger meters).

## Sim-model mismatches still open
- The player coast, the dash, the stick stop and the hit direction now follow the s10 models (see "Sim models (s10)"); their
  remaining gaps (dash peak/fade shape, stick overshoot, contact detection still ~18 deg early) are listed there.
- Puck: decel/v is not constant (≈0.50–0.55 1/s above ~8 m/s, 0.33–0.39 at 3–7 m/s placeholder speeds); one linear k is a
  compromise. Board friction: a Coulomb term did not help (docs/experiments/board_coulomb_friction.patch, not in the sim).
- Other model simplifications: infinitely tall boards, no net top bar, no puck spin, no stick-on-stick or stick-on-player
  collision, facing follows velocity.

## What the clips could not give (still placeholder or unknown)
**Still placeholder, by name: stick_mass (2 kg), the three puck speed caps (max_puck_speed_air 30, _ice_normal 35, _ice_dash 45
m/s), the stick height range (stick_z_min 0 / stick_z_max 0.25 m), dash_duration (0.25 s) and dash_cooldown (1 s).**
- **stick_mass** (degenerate with stick_restitution; needs the stick's slowdown at contact: higher fps or a heavy target),
  **stick_inner**, **stick_z_min / stick_z_max** (height range; not visible top-down), the left-button stick pose.
- **dash_duration, dash_cooldown** (no two dashes within a second in any clip; the fit drives duration to 0).
- Player: mass, height, restitution, turn rate (facing is not observed), braking with the opposite key, backward/sideways
  speeds, collision radius of the new body art.
- Puck: mass, the three speed caps (a ~30 m/s placeholder-scale cap is possible from 2 hits), air drag, ice restitution,
  lift (lift_vz_factor, lift_max_vz, chip_max_speed), net restitution (no net hits recorded).
- gravity, timestep/substeps, countdown/goal-pause lengths and spawn positions, the readout's unit (the scale assumption).
- Recording wishes: more ground goals with long flights (pins down the readout), repeated Shift presses 0.2–1.0 s apart while
  holding W (cooldown), W → S at top speed (braking), net hits, stick/body collisions, lifts landing on open ice,
  fast swings at several contact radii (`docs/recording-guide.md`).

## Other open items
- `params.fitted.json` (old puck holdout output) still has placeholder-meter shape values; `measure holdout` would rewrite it.
- Menu; the opponent player is idle in the lab; panel edits keep the provenance tag; practice-tool actions are not recordable.
- Real RL training runs have not started: the long run in README "Long run on Ian's machine" is next (see "Bots and training").

## Figures (measure/results/fits/, `python -m measure figures`)
- Drawn on the SIM's own rink from params.json (game orientation, +x up), no game footage / logos / HUD; dots every 0.25 s;
  speed vs distance panels. `readme_fit.png` (README), `restitution_wall_07.png` (e = 0.2 / fitted / placeholder: the rebound
  angle matches the fitted e, the post-bounce speed sits lower), `drag_puck_06.png`, `corner_puck_08.png`,
  `worst_<seg>.png` (the three worst segments by the new default). ~0.6 MB together.
- Tracked images (all *.png in git) after session 6: 11.52 MB (limit 30 MB).

## Segment manifest (calib/data/real/manifest.json)
- 25 free-puck stretches, one scenario + CSV pair each (t from 0 at the stretch start, x/y = puck GROUND position in REAL meters
  (ice-plane scale, rescaled ×1.0873; `meter_scale` in the manifest), z = 0), initial_state from a quadratic fit to the first ≤ 30 samples before any bounce, no players.
  - puck clip: 10 (7 glide, 3 wall bounce); wall clip: 11 (8 glide, 3 wall bounce); lifted clip: 4 (1 glide, 3 wall bounce). Net hits: 0.
  - Start: after the hit's contact window; moved to the speed peak if the puck is still accelerating (> 3% within 15 rows); moved
    to 4 rows after a bounce that comes within 12 rows (shots straight into the board next to the player; the initial state
    must not straddle a bounce). End: teleport/reset (tracklet end; trimmed where the puck already reached the board), stop
    (< 0.2 px/frame), or contact: a speed-up anywhere, or a sudden slowdown not within 8 rows of a board/net. Bounce = local
    minimum of the distance to the boards where the board-normal velocity flips (approach rows may precede the stretch).
    Skipped: < 0.3 s, and stretches that start from rest without a detected hit.
  - `possibly_airborne`: every lifted-clip stretch. On screen a lift shows as a speed signature (parallax), not reliably as blob size,
    so the whole clip is flagged and excluded from the estimates.
- QA: `measure/results/qa/segments_<clip>.png`, `segment_wall_09_frames.png`, `segment_puck_08_frames.png`, and
  `compare_<seg>.png` overlays of the worst fitted segments (checked by eye).


## Tracker limits (measure/)
- z is not observable. A lifted puck is drawn displaced on screen, so its x/y are wrong while airborne. The game also draws a
  ground shadow (blue dot) and a landing ring that are not used yet.
- The scale rests on the goal-readout estimate (see "Absolute scale"), and speed caps are placeholders, so the gate and the
  "speed outlier" threshold are guesses.
- A puck under a net roof, under the player or merged with another green blob is missed (a gap) or biased. The puck overlapping the
  board graphics near the boards can bias the centroid by a pixel or two.
- One CSV holds many shots. Practice-tool teleports split segments (`segments` in track.json), but the CSV itself is continuous.
  Split per segment before fitting. Confidence is a heuristic (blob area vs typical, saturation).
- Puck height: blob centroids are reprojected to the ground with a per-clip parallax factor from the blob-area model
  (principal point assumed at the image center). Lifted pucks are not handled (z unknown).
- Capture timing jitter limits speed measurements at high speed: at ~25 m/s the measured speed wobbles ±1.5 m/s (±1/180 s ≈ 2 px),
  which dominates v² error on fast glides (puck_01).
- Camera motion is undone as a pure translation (verified to 0.1 px with ice-level landmarks). A rotating/zooming camera is not handled
  (it would show up as a large landmark residual and those frames are dropped).
- Contacts with the stick are inferred from the puck alone (player untracked): a gentle push or a deflection that doesn't change the
  speed by ~15% is missed; segments start ~0.1–0.15 s after the hit.


## Match rules (s8 session, 2026-11-04)
- **Rules (Ian)**: 1v1, no goalie, a 5:00 clock, the team with more goals wins, mercy rule 5, a tie at 5:00 is decided by the next goal.
- **Mercy interpretation — UNCONFIRMED (Ian, please confirm)**: `mercy_lead` = 5 is read as a **5-goal lead**: the match ends as soon as
  |score0 − score1| ≥ 5 (5-0, 6-1, …). It is not "first to 5 goals": 5-4 or 5-1 play on. `mercy_lead ≤ 0` disables it.
- **Values** (appended at the end of params.json, params.fitted.json and the pinned calib/data/params.synthetic.json, because
  `to_params` is strict; no physics value changed): `match_length_s` 300 and `mercy_lead` 5 have provenance **`rule`** (Ian's).
  `countdown_s` 3, `goal_pause_s` 3, `player_spawn_x` 15 m are **`placeholder`** (invented; the spawn x matches
  `new_default`'s rink_length/4). `rule` shows as a purple badge in the panel.
- **Sim** (`sim/src/rules.rs`, opt-in): `State.match_state: Option<MatchState>` (serde `match_state`, default null).
  `None` = free play, exactly the old behaviour (all old tests, calib, lab play mode; free-play `to_bytes`/hash unchanged by
  the rules). `State::new_match(p, n)` turns the rules on. `step`'s signature is unchanged; it calls
  `rules::step_match` when a match state is present, which wraps `physics_step` (the old step body; `update_goals` now also
  returns the scoring team).
  - `MatchState { phase: countdown|playing|goal_pause|finished, phase_ticks, clock_ticks, overtime, winner, last_goal }`,
    all in `to_bytes`/`hash` (appended after the score only when present).
  - countdown / goal_pause / finished: **no physics at all** (players, sticks, puck frozen; inputs ignored; the clock stopped);
    `tick` still advances. playing: physics + inputs, the clock counts down one step per step (not in overtime).
  - Goal: score +1 → mercy check → (overtime: the scorer wins) → (a goal on the last regulation step: the leader wins; an
    equalizer there sets overtime) → goal pause (puck left in the net) → reset → countdown → play.
  - Reset: puck at center at rest (`in_net` cleared); players at ±`player_spawn_x` (team 0 at −x, attacking +x; team 1 at +x),
    teammates spread in y like `new_default`, at rest, facing center, stick along the facing, dash timers/cooldowns cleared.
  - Clock reaches 0: a leader wins; tied → `overtime = true`, play continues with the clock at 0 until the next goal.
- **sim-wasm**: `start_match(n_players)`; `get_state`/`set_state`/`reset(scenario)` carry `match_state`.
- **Feel lab**: "Match mode" checkbox (or `?mode=match`), "New match" button, Enter (bindings.ts `rematch`, + NumpadEnter) starts a
  rematch after game over. Canvas HUD (`drawMatchHud` in render.ts, screen space so it works in game view): scoreboard
  BLUE score | mm:ss | score ORANGE (clock counts down, rounded up like a game clock, yellow in overtime), countdown digits,
  GOAL! banner, red OVERTIME / NEXT GOAL WINS banner, game-over overlay (winner, final score, "Press Enter for a rematch").
  Practice tools (F/G/R/Q) are off in match mode; toggling the mode, New match and Enter work only in play mode (not while
  recording/replaying). Record/replay works in matches (the scenario's initial_state carries `match_state`); `Recording.endHash`
  = state hash after the last recorded/replayed step. `?speed=N` / `window.lab.setSpeed(N)` runs the loop N× real time.
- **Team names/colours** BLUE (team 0) / ORANGE (team 1) follow the lab's existing player colours; the real game's HUD layout
  was approximated (score boxes either side of a mm:ss clock), not measured.

### Match rules still open
- Goalie, player-player contact changes (checks, body contact rules), stick-on-stick / stick-on-player contact, head/body
  hitboxes: not added (physics and contact model unchanged).
- Mercy reading (lead vs. total) to be confirmed by Ian. Countdown and goal-pause lengths, spawn positions (and whether the real
  game spawns at a faceoff, or lets players move during the countdown), and what the real clock shows in overtime (it stays at
  00:00 here) are unverified.
- Offside/icing or any other real-game rule, faceoff mechanics, team sizes above 1v1 (the code supports any count, untested in
  matches beyond 2 players), a goal exactly at 0:00 (counted: goals on the last step score first), the opponent player still idle.

## Bots and training (bots/, sim-ffi/, train/)
1v1, no goalie. Formats are versioned: **ACT v4, OBS_VERSION 3, ABI_VERSION 4** (see "Current state"); checkpoints record the physics stamp (sha256 of
the params.json values), the obs version and the act version, and refuse to load across any change (old checkpoints from
before the s10 + t1 integration are refused by the physics stamp: intended, do not resume them).

### bots crate (pure Rust, deterministic, no I/O; deps sim + libm)
- **Canonical frame** (`frame.rs`): every player sees the world as if it attacks +x. Team 0 = identity; team 1 = the world
  rotated 180° ((x, y) → (−x, −y), angles + π); a rotation, not a mirror, so one policy plays both sides.
  `rotate_swap(state)` rotates the whole state and swaps teams, scores and the match state's team indices (tests).
- **Observation** (`obs.rs`, OBS_DIM 42, OBS_VERSION 3, f32, fixed normalization constants, `observe(state, me, style, params, out)`):
  0–1 own x/30, y/20; 2–3 own v/14; 4–5 cos/sin facing; 6–7 cos/sin stick angle; 8 stick ω/14; 9 dash window/0.25 s;
  10 dash cooldown/1 s; 11–12 puck x/30, y/20; 13 puck z/2; 14–15 puck v/30; 16 puck vz/10; 17–18 (puck − own)/10;
  19–20 opponent x/30, y/20; 21–22 opponent v/14; 23–24 cos/sin opponent stick; 25 opponent ω/14; 26–27 (opponent − own)/10;
  **28 (own − opponent goals)/5; 29 overtime flag; 30 regulation time left/300 s** (0 in free play); 31 dash ready (0/1);
  32–39 style slot (STYLE_DIM 8); **40 own stick_up, 41 opponent stick_up (0/1, OBS v3)**. Change any field or constant →
  bump OBS_VERSION.
- **Action**: the 9-way WASD format plus stick_up and brake (ACT v4), see "Current state".
- **Scripted bots** (`scripted.rs`, trait `Bot`, `Scripted` by `Kind` id 0 chaser, 1 defender, 2 cautious, 3 idle). Each bot
  plans an analog `Wish` (move, stick); `Scripted::act` turns the move into 9-way key presses with a sigma-delta dither (t1).
  - shared `Striker`: picks a hitting spot beside the (predicted) puck, holds the stick cocked and swings through the puck
    only when the swing sends it at the target. Shot model for the s10 centre-on contact: the puck leaves perpendicular to
    player → puck, plus `SHOT_OVERTURN` **0.020 rad**, from-rest speed `SHOT_SPEED` **26.5 m/s** (shotlab on the merged code:
    48/48 hits, median |err| 0.001 rad, median speed 27.7 m/s; history 0.041/19.7 → 0.054/18.9 → 0.110/27.2 → 0.018/27.6).
    The bots are 120 Hz controllers: driven through the env at 30 decisions/s they can stall with the puck (sim-ffi test drives
    them at action_repeat 1).
  - chaser: striker toward the opponent net (net center or the post side the opponent's body does not block).
  - defender: guards the own net→puck line (0.4 × distance, 2–9 m out), steps onto a shot's path; clears when it wins the race.
  - cautious: waits at x = −rink_length/4 on the puck→own-net line, engages within 6 m, lets go past 9 m.
- `game.rs`: `faceoff`, `play`, `play_with_faceoffs`. Examples: `tournament`, `shotlab`, `trace`.

### sim-ffi crate (cdylib + rlib; deps sim, bots, serde, rayon)
- `Batch` of N envs (`env.rs`): State, SplitMix64 RNG (resets and style sampling only), learner slot, opponent (scripted or
  external), base + effective reward weights, style. Stepped with rayon in tasks of ≥ `chunk` envs; results do not depend on
  the thread count (tests). Auto-reset: the returned obs is the next episode's first; `final_obs` holds the pre-reset one.
- **Config JSON** (all optional, unknown keys rejected): `mode` "first_goal" | "match", `max_steps` (0 = auto: 30 s / match
  + `overtime_cap_s` 120), `action_repeat` 4, `learner_team` −1, `jitter_puck` 1.5, `jitter_player` 1.0,
  `jitter_every_faceoff` true, `opponent`, `reward` {weights}, `style` [floats], `style_mode` "off" | "fixed" | "sample",
  `threads` 0, `chunk` 16, `profile` false.
- **C ABI v3** (`lib.rs`; `slap_act_version` added in v3): `slap_batch_new`, `_free`, `_reset(obs, opp_obs)`, `_step(actions, opp_actions, obs, opp_obs,
  final_obs, rewards, dones, info)`, `_set_opponent`, `_set_style`, `_set_reward_weights` (base weights), `_hash`, `_env_hash`,
  `_get_state/_set_state`, `_profile`, dims/version getters, `slap_last_error`. Info row (INFO_DIM 14): end (0 running, 1 win,
  2 loss, 3 truncated), playing sim steps, return, learner team, goals for, goals against, opponent possession s, 7 raw sums.

### Reward (`reward.rs`; reward = Σ weight × raw, per decision)
| term | raw value | default weight |
|---|---|---|
| goal_for | goals scored (0/1) | 1.0 |
| goal_against | goals conceded (0/1) | −1.0 |
| puck_progress | Δ puck x / net_x | 0.2 |
| shot_on_net | 1 per learner contact leaving the puck at ≥ 5 m/s on a line into the opponent mouth (0.5 s cooldown) | 0.05 |
| possession | seconds the learner is the closest player with the puck within stick reach (+0.2 m) | 0.02 |
| defensive_position | seconds goal-side, ≤ 1.5 m off the own-net→puck line | 0.0 |
| match_win | match mode: +1 win / −1 loss at the end | 1.0 |

### train/ (Python 3, numpy, torch; ctypes)
- `slapenv.py` (VecEnv; returns views of its buffers), `ppo.py` (PPO, self-play pool, eval, plot), `league.py` (ladder),
  `export_onnx.py`. Details, defaults and numbers: "Training on the real game" below; run guide: README.
- Tests: `python -m unittest discover -s train/tests -t .` (16).

## Environment notes
- Several sessions have run in parallel on this machine: use a separate `git worktree` per session (e.g.
  `git worktree add ..\slapmini-NAME main`, then `rustup override set stable-x86_64-pc-windows-gnu` in it) instead of a shared folder.
- On Ian's Windows machine the MSVC toolchain can't link (no Windows SDK), so the repo has a local `rustup override set stable-x86_64-pc-windows-gnu`. That override lives in rustup settings, not the repo.
  The minimal GNU toolchain has no `dlltool`, so wasm-pack came from npm (`npm i -g wasm-pack`).
- Run cargo from PowerShell. In Git Bash, GNU `/usr/bin/link` shadows MSVC `link.exe`.
- Playwright Chromium is installed with `npx playwright install chromium` (in web/). verify.mjs kills the Vite process tree with `taskkill /T` on Windows
  and the process group on POSIX. If Playwright's own browser is missing it falls back to `PW_CHROMIUM` or `/opt/pw-browsers/chromium` (cloud container).
- measure/ runs from a venv at the repo root: `.venv\Scripts\python` on Windows, `.venv/bin/python` on macOS/Linux.

## Sim models (s10): coast, dash, stick braking, hit contact (2026-11-04)
State, PlayerInput and the public sim API are unchanged; three params were appended
(params.json and the pinned params.placeholder.json, params.fitted.json, calib/data/params.synthetic.json; in the pinned
files they are set to reproduce the old coast and stick chase where possible).

### The four models (sim/src/step.rs)
| model | old form | new form | params |
|---|---|---|---|
| coast (no move key) | constant decel player_ice_drag 6.49 m/s^2 | dv/dt = −player_coast_rate·v − player_ice_drag | **player_coast_rate 1.0815 1/s** (new, fitted); player_ice_drag is now the constant part, fitted **0** |
| skating / cap | accel to the cap, decay above it at 2·accel (already effectively hard) | unchanged: player_accel up to the hard cap player_max_speed | player_accel **17.359** (was 17.904); cap 11.6935 (measured) |
| dash | Shift edge: +dash_speed impulse along move (or `facing` with no key), cap raised to max + dash_speed for dash_duration | Shift edge **with a move key** opens the window and ramps the velocity toward dash_speed along the move key (along the velocity after release) at dash_speed / dash_ramp_time for dash_ramp_time (never slows a faster player); afterwards the excess above the cap fades at 2·player_accel. Shift alone does nothing. dash_timer > 0 still means "dash shot" for the puck caps | **dash_speed 15.480** (was 10.05), **dash_ramp_time 0.1163 s** (new) |
| stick chase | bang-bang with one limit (stick_max_ang_accel) for speeding up and braking | speeding up at stick_max_ang_accel, slowing down (and the brake curve) at stick_max_ang_decel; no overshoot term | **stick_max_ang_decel 2316 rad/s^2** (new); accel 513.9 kept |
| stick→puck impulse | along the separation normal (closest blade point → puck), at the closest blade point | **centre-on**: perpendicular to pivot → puck centre (toward the puck's side of the stick), at the stick point under the puck centre (radius = pivot → puck). Detection and the separation push are unchanged | stick_restitution 0.3597 kept, stick_mass 2 kg (placeholder) kept |

### Evidence
Tools: `python -m measure.s10_skate --old-calib <calib.exe built from the pre-s10 code>`, `python -m measure.s10_stick`,
`cargo run --release -p sim --example hitlab -- params.json measure/results/stick_hits.json`.
- Skating, held out by clip (fit on the other two skating clips, real meters, median per-run RMSE; `measure/results/skate/s10_fits.json`):

  | held-out clip | placeholder pos | old model pos / speed | new model pos / speed |
  |---|---|---|---|
  | wkeymovement | 6.20 m | 0.577 m / 2.08 m/s | **0.179 m / 1.87 m/s** |
  | diagonalkeypresses | 7.50 m | 0.650 m / 2.45 m/s | **0.276 m / 2.24 m/s** |
  | whilemovingturning | 10.07 m | 0.827 m / 2.14 m/s | **0.746 m / 2.11 m/s** |

  Folds: accel 16.46–18.02, coast rate 1.055–1.104, ice_drag 0 in all three (old model: accel 15.77–17.80, ice_drag
  6.59–7.01). Speed RMSE is dominated by finite-difference noise of the tracked player.
- Dash, held out by clip (fit on the other dash clip, skating params fixed):

  | held-out clip | placeholder pos | old model pos / speed | new model pos / speed |
  |---|---|---|---|
  | dashatrest | 1.07 m | 4.99 m / 2.70 m/s | **0.143 m / 1.55 m/s** |
  | dashwhilerunning | 3.07 m | 4.03 m / 3.15 m/s | **0.341 m / 2.19 m/s** |

  Folds: dash_speed 15.47 / 14.72, ramp 0.108 / 0.133 s (old: 7.34 / 11.02 m/s, duration 0 / 0.14 s; the old fits failed both ways).
- Stick chase, held out by event (4 cap-reaching step events of rotatinganddizzy, angle RMSE in deg; `measure/results/s10_stick_chase.json`):

  | model | ev 0 | ev 1 | ev 2 | ev 3 | mean |
  |---|---|---|---|---|---|
  | old symmetric (one limit, fitted) | 3.71 | 2.31 | 1.95 | 3.01 | 2.75 |
  | decel, accel fixed 513.9 (**chosen**) | 3.91 | 2.51 | 0.55 | 3.05 | 2.51 |
  | decel + accel both fitted | 3.01 | 2.49 | 0.96 | 2.07 | 2.13 |
  | + overshoot term, accel 513.9 | 3.93 | 2.51 | 0.70 | 3.06 | 2.55 |
  | + overshoot term, accel fitted | 3.59 | 2.23 | 0.84 | 2.08 | 2.19 |

  The overshoot term (the brake aims over·ω/ω_max past the target) does not lower the held-out error, so it is not in the sim.
  Accel stays at the s7b flick-start value 513.9: the joint fits move it between 223 and 383 depending on the start (these 4
  events determine it weakly; the flick starts measure it directly). Python emulation = Rust to 1.3e-15 rad. The discrete hard
  stop still overshoots ~1–2 deg and settles in 2–3 steps (the measured 0.7–12 deg is not reproduced).
- Hits (the 7 recorded hits; `calib/data/stick/` is now in REAL meters, ×1.0872638; the stick turns at the measured ω with
  the chase limits off, 16 contact phases; `measure/results/s10_hits_before.csv` / `s10_hits_after.csv`). Offset = exit
  direction − the perpendicular to pivot → resting puck (+ = ahead), deg; ratio = exit speed / (|ω|·r_c):

  | hit | measured offset | measured ratio | old offset | old ratio | new offset | new ratio |
  |---|---|---|---|---|---|---|
  | hit_01 | −2.30 | 0.884 | −16.10 | 1.104 | 0.00 | 1.135 |
  | hit_02 | −4.02 | 0.768 | −16.02 | 1.103 | 0.00 | 1.133 |
  | hit_03 | −7.18 | 1.637 | −15.18 | 1.092 | 0.00 | 1.118 |
  | hit_04 | −7.18 | 1.008 | −14.57 | 1.112 | 0.00 | 1.138 |
  | hit_05 | +6.29 | 1.202 | −14.19 | 1.109 | 0.00 | 1.132 |
  | hit_06 | −2.05 | 1.183 | −12.50 | 1.079 | 0.00 | 1.096 |
  | hit_07 | +3.72 | 0.959 | −13.21 | 1.117 | 0.00 | 1.138 |

  Direction RMS vs measured: old 13.5 deg, new 5.1 deg (the measured scatter itself; the measured mean −1.8 ± 1.9 deg s.e. is
  not different from 0). The old offsets come from the blade edge touching the puck 18–21 deg before the stick points at it
  (hitlab's contact_lead_deg). The blade-normal variant gives the same −12..−16 deg (the closest point is mid-blade), so only
  the centre-on direction fixes it. The model has no free direction parameter, so holding out by hit changes nothing for the
  direction. Exit-speed ratio: refitting stick_restitution with each hit held out (e 0.267–0.398) gives a held-out exit-speed
  RMS of 4.56 m/s vs 3.88 m/s for the kept 0.3597 (sim ratio 1.10–1.14; the measured through-origin fit is 1.09). The 7 ratios
  scatter 0.77–1.64, so the data pin the ratio only to about ±0.1, and mass vs restitution not at all.
- **Puck speed cap**: the measured exits in real meters are 7.9, 7.4, 19.2, 21.6, 29.8, **34.2** (hit_06, head 28.9 m/s) and
  31.1 m/s (hit_07, head 32.4 m/s, ratio 0.96). One ordinary ice shot at 34.2 m/s rules out a ~30 m/s cap for ordinary shots;
  hit_07's low ratio hints at saturation near 31–34 m/s but is a single hit. The 88 mph goal (39 m/s) is above the 35 m/s
  max_puck_speed_ice_normal, so either dash shots go faster (max_puck_speed_ice_dash 45, placeholder) or the normal cap is
  higher. **Caps unchanged** (no evidence to set them).

### Remaining gaps
- **dash_duration 0.25 s, dash_cooldown 1 s**: placeholders, not identifiable from the clips (no two dashes within a second).
  The window is the dash-shot window and also clips the ramp. Ian: set them from play.
- Dash shape: the sim peaks at 15.5 m/s for an instant and fades at 2·accel (34.7 m/s^2); the game peaks at 12–13.6 m/s and,
  from rest with the key released, drops to ~8.5 m/s within 0.15 s before coasting. At top speed the sim adds 3.8 m/s, the
  game 0.5–3.6. The fade rate (2·accel) was not fitted, and a state-free model cannot tell the boost from the base speed.
- Stick: the real overshoot (0.7–12 deg, settling over 2–5 frames) is not modelled; the decel is barely resolved at 60 fps.
- stick_mass (placeholder 2 kg) and restitution are degenerate; the puck speed caps, stick_z_min / stick_z_max (height range)
  and stick_inner are unmeasured. Contact DETECTION is still the old capsule (the puck is touched ~18 deg early); only the
  impulse direction and point moved.
- measure/merge_models.py and measure/skatefit.py still describe the old models (their `params` writers would put the old
  compromise notes back); use measure/s10_skate.py for refits.

## Training on the real game (session t1, 2026-11-04; Ian's Windows machine: RTX 3070 Ti, 12 cores)
Measured on the t1 branch (before the s10 physics); the merged numbers are in "Verified (s10 + t1 integration)".

### Match mode (sim-ffi env Config `mode`)
- `"first_goal"` (default; the curriculum): jittered free-play faceoff, ends at the first goal or after 30 s (truncated).
- `"match"`: `State::new_match`, full 5:00 match with the sim's rules (goal resets, mercy at a 5-goal lead, next-goal
  overtime). Countdown and goal pause are **fast-forwarded inside the env** (no decision spent, no reward tracked across the
  reset, the scripted opponent is reset). Step limit = playing sim steps, default regulation + `overtime_cap_s` 120 s
  (truncated). **Every faceoff after a goal is jittered too** (`jitter_every_faceoff`, default true): the first match smoke
  run without it learned to score from the one fixed post-goal faceoff (99% match wins vs the chaser in training, 95% in
  match eval, but 5% wins from jittered first-goal starts; curve `train/results/t1_smoke_match_nojitter_curve.png`).
- End reasons (info 0): 1 win, 2 loss (first goal / match result), 3 truncated. Info row (INFO_DIM 14): end, playing sim
  steps, return, learner team, goals for, goals against, opponent possession s, then 7 raw reward sums.
- Reward term `match_win` (raw ±1 at the match end, default weight 1.0); the other defaults are unchanged.
- `final_obs` (new step argument): the learner's observation of the last state of an episode that ended (pre-reset).
- `python -m train.ppo train --mode match` or `--curriculum-minutes M` (first_goal for M minutes, then match; a new env,
  the rolling stats reset, the plot marks the switch). `eval` defaults to full matches (`--mode match`).
- Scripted matches in the env (test `match_mode_endings`; current values in "Verified (s10 + t1 integration)"): mercy,
  clock end and an overtime goal each occur; some 0-0 games are truncated at the overtime cap. Determinism across thread
  counts is asserted for both modes (the test checks equality, the hashes are printed with `--nocapture`).

### Training details
- **Truncation**: a truncated episode adds γ·V(final_obs) to the last reward and then ends the GAE trace (a real end does
  not bootstrap). `--no-bootstrap` = the old behaviour, for comparisons.
- **Running observation normalization** (`--obs-norm`, off by default): mean/var of fields 0–31 (not the style slot) as
  buffers in the policy (saved in checkpoints, baked into ONNX), updated after each PPO update from that rollout, clip ±10.
  Test: matches numpy mean/var over 3 batches; resume keeps the statistics.
- Checkpoints: physics stamp and obs version checks kept; also `act_version` 2 and `obs_norm`. Ctrl+C finishes the
  update, saves latest.pt and exits. `--keep-every N` keeps `ckpts/upd_N.pt` for the ladder.
- ONNX outputs: `move` [N] (argmax index), `move_probs` [N, 9], `stick_dir` [N, 2], `dash_prob`, `lift_prob`, `stick_up_prob` (t2); team 1:
  move k>0 → ((k−1+4) mod 8)+1, negate stick_dir. Parity vs PyTorch on 2000 random obs: 3.3e-6 (3.8e-6 with obs norm), move
  index identical.

### Throughput (profiled with `--profile`, 64 envs × 128 steps, update on the GPU; `train/results/t1_profile_*.md`)
| section | before (old layout: GPU actor, rayon all, no chunks), µs/decision | after (defaults), µs/decision |
|---|---|---|
| env step in Rust (sim + bot + reward + obs) | 4.79 | 3.57 |
| ctypes + wrapper | 0.74 | 0.40 |
| obs → device | 2.27 | 0.14 |
| policy forward + sampling | **49.92** | 8.27 |
| actions → host | 2.20 | 0.40 |
| bookkeeping (buffers, dones) | 10.94 | 1.25 |
| GAE | 1.18 | 0.56 |
| update H2D | 0.01 | 0.24 |
| PPO update | 13.52 | 9.35 |
| total | 85.93 (11.6k dec/s, synced) | 24.53 (40.8k dec/s, synced) |
Rust CPU split: sim ~48%, scripted opponent ~42%, reward ~5.5%, observe ~5%. Unprofiled training rate (metrics.csv,
first update excluded): **origin/main 12.3k decisions/s (18.6k with --device cpu) → t1 42.9k** (3.5×, same learning setup);
match mode 47.5k; with 50% pool self-play 24.5k (each frozen snapshot runs separately). More envs: 128 → 72k, 256 → 110–117k,
512 → 157–176k (bigger batch per update = a learning-behaviour change; the README run guide uses 512 + 8 minibatches).
Fixes: rollouts act with a CPU copy of the policy (synced each update); sampling/log-prob/entropy by hand (same
distributions; tested against torch.distributions; ~3× less per-call overhead); thread defaults rayon 1 for ≤ 128 envs
(else min(8, cores)) and torch 4 (idle rayon workers spinning slowed torch); rayon tasks of ≥ 16 envs (`chunk`; 64-env
rayon step 217k → 356k dec/s in the Rust bench); the reward tracker no longer clones the State each sim step; PPO stats
stay on the device. Learning behaviour unchanged except the RNG stream of the samples.
What limits it now: Python per-step overhead (≈ 8 µs/decision inference at 64 envs, mostly fixed per call) and the
GPU update's per-minibatch launches; both shrink per decision with more envs. The Rust env alone does 350k–1.3M dec/s.

### Styles (sim-ffi/src/reward.rs `style_weights`, env `style_mode` off | fixed | sample)
| style entry | meaning | reward weight |
|---|---|---|
| s0 | aggression | puck_progress × (1 + s0) |
| s1 | shot-taking | shot_on_net × (1 + s1) |
| s2 | possession | possession × (1 + s2) |
| s3 | defense | defensive_position + 0.03 × max(s3, 0) |
goal_for, goal_against, match_win never change. `sample` draws s0..s3 uniform in [−1, 1) per episode (rest 0) and writes it
into obs 32–35; `train.ppo --styles`. **Style zero reproduces the default env exactly** (tests: Rust and Python, both
modes, current hashes in "Verified (s10 + t1 integration)"). Self-play opponents see a zero style.

### Ladder (`python -m train.league`)
Checkpoints vs scripted bots (env opponent) and vs each other (external opponent), full matches by default, win 1 / draw
0.5; Bradley-Terry fit on the Elo scale with one virtual draw per pairing, anchor chaser = 1000; per-style stats vs
`--style-opponent` (wins, goals for/against, shots per match, possession share = learner / (learner + opponent)
possession seconds). JSON output.

## Game (session w1, 2026-11-04, branch w1)
A playable browser game around the sim: menu → match vs a bot → game over, plus the lab, as plain static files for GitHub Pages.
Nothing in sim/, bots/, sim-ffi/, train/ or the params files changed; sim-wasm gained the bots crate (crosscheck hashes unchanged:
free play `ed2b9c3b2c52da91`, match `ef5c44a7fc4de600`, s10 `b19f6906467b50c1`).

### Game flow (`web/index.html`, `web/src/game/`)
- **Menu**: title, opponent select, skill slider 0–100 (with the handicap it means), Play vs Bot, Practice (opens `lab.html`;
  the lab has a "← menu" link back), Controls (bindings; dash and lift marked UNVERIFIED). Keyboard: focus starts on Play,
  ArrowUp/Down and Tab move, Enter activates, Esc leaves Controls.
- **Match**: `Sim.start_match(2)`, the sim's rules (5:00, mercy at a 5-goal lead, next-goal overtime) and the lab's canvas
  scoreboard/countdown/banners (`drawMatchHud`). The human is BLUE (player 0, team 0) with WASD (screen-relative) and the mouse
  (relative stick control since w2), drawn with the rink vertical (BLUE attacks the top net). Esc releases a captured mouse;
  Esc again returns to the menu.
- **Game over**: "You win!" / "<bot> wins", the score and how it ended (mercy / clock / overtime), Rematch (Enter) and Menu (Esc).
- `?speed=N` and `window.game` (summary, bot stats, frame stats, `setSpeed`, `setScore` test hook) for verification.

### Same observation as training (`sim-wasm`)
- `Sim.observe(player, style)` = `bots::observe` (Float32Array, OBS_DIM 42, obs v3); `Sim.action_input(player, move, angle,
  dash, lift, stick_up)` = `bots::Action::to_input` (team rotation in Rust); `Sim.set_scripted(player, kind)` + `Sim.scripted_input(player)`
  = `bots::Scripted` (reset at goal resets / a new match, as in training). The browser never re-implements the observation.
- **Equality test**: `cargo run --release -p sim-wasm --example obs_fixture > target/obs_fixture.jsonl` (states from scripted free
  play and matches with goals, dashes, lifts, three style vectors) then `node scripts/obs_crosscheck.mjs target/obs_fixture.jsonl`:
  **684 observations (456 in match mode), 27,360 fields, 0 bit mismatches**.

### Profiles and skill (`web/public/bots/`, `web/src/game/profiles.ts`, `skill.ts`)
- `index.json` lists profile files; a profile is `{name, description, policy: "x.onnx" | null, scripted: "chaser" | ... | null,
  style: [≤ 8 floats], skill: 0..100, fallback: "chaser", smoke_test?}` (exactly one of policy / scripted). Built-in scripted
  profiles (chaser, defender, cautious) are added even if index.json is missing.
- Skill → handicaps, ONE file (`skill.ts`), h = 1 − skill/100: perception delay round(18·h) sim steps (≤ 150 ms; policies see an
  older observation, scripted bots' inputs are delayed instead because they read the live state in Rust), stick-angle noise
  N(0, 0.5·h rad) redrawn per decision, mistakes 0.6·h per second lasting 0.25 s (skate a random direction, no dash/lift).
  Skill 100 = no handicap. **UNCALIBRATED**: calibrate with the rating ladder (`train.league` at several skills) before trusting
  that skill numbers mean comparable strength.

### Bot runtime (`web/src/game/botRuntime.ts`)
- onnxruntime-web **1.31.1** (pinned, `onnxruntime-web/wasm` entry, single-threaded: no cross-origin isolation needed). Vite
  emits its runtime (`ort-wasm-simd-threaded-*.wasm`, 14.8 MB, 3.8 MB gzipped) into `dist/assets/` with a relative URL: no CDN.
- One decision per 4 sim steps (30 Hz), action held in between, deterministic heads (argmax move, mean stick direction,
  dash/lift/stick_up/brake prob > 0.5); the sidecar `.json` is checked (obs v3 / 42 / act v4 with `brake_prob`). `run()` is async: a decision's action applies
  from the next sim step after it resolves (≈ one frame); if one is still pending at the next decision it is skipped and counted.
- **Measured (Playwright, headless Chromium on Ian's machine, smoke policy)**: inference per decision mean 0.82 ms, p50 0.70,
  p95 0.90 ms (one 40 ms first-call warm-up); 326 decisions in ~10.9 s at real time, 0 skipped. Frame interval p50/p95 16.7 ms;
  work per frame (sim steps + bot + draw) mean 0.37 ms, p95 0.50 ms, max 1.0 ms. **No Web Worker needed.** At 6× loop speed
  many decisions are skipped (962 of 1771): bots play worse in sped-up verification runs only.
- Fallback: a missing (HTTP 404) or corrupt model (protobuf parse error), or a layout mismatch, shows an amber banner
  ("Could not load the … model (…). Playing the scripted chaser instead.") and plays the profile's scripted `fallback`.

### Opponents shipped
chaser, defender, cautious (scripted, no model file) and **Smoke-test bot** (`smoke.onnx`, 317 KB, + `smoke.json` sidecar,
ACT v4 / OBS v3, physics stamp `e750f45cff93780c`): a SMOKE TEST, labelled so in the menu. See "Current state".

### Adding a policy
`python -m train.export_onnx --ckpt runs/NAME/latest.pt --out web/public/bots/NAME.onnx` (writes NAME.json), write
`NAME.profile.json`, add it to `index.json`'s `profiles`. A policy trained under other physics still loads (the game does not
check the physics stamp; it is in the sidecar for reference).

### Deployment (`.github/workflows/pages.yml`)
- On push to main (or manual run): checkout main only (depth 1; clip-data is never fetched), Rust + wasm32, wasm-pack 0.13.1,
  Node 22, `npm ci && npm run build`, a guard that fails on any image/video file in dist, upload-pages-artifact, deploy-pages.
- Ian must: (1) move the clips off the repo (clip-data branch) and deal with the 30 footage-derived images tracked on main
  (`measure/results/*.map.png`, `measure/results/qa/*` except ghost-game-view.png, `measure/results/skate/qa/*.path.png`; also
  in history: rewrite with git filter-repo or publish a fresh footage-free repo); (2) make the repo public; (3) Settings → Pages →
  Source: GitHub Actions. Nothing was switched on in this session. README "Publishing on GitHub Pages" has the steps.
- Relative paths: `npm run verify-game` serves `web/dist` under `/slapmini/` with a server that 404s anything outside it:
  0 requests escaped; the game, the ONNX model, the ort wasm and `lab.html` all load from the subfolder.

## View, mouse and Space (session w2, 2026-11-05, branch w2)
Web only (web/src, web/*.html, web/scripts, scripts/lift_demo.mjs). sim/, params and bots untouched: crosscheck hashes
unchanged (free play `ed2b9c3b2c52da91`, match `ef5c44a7fc4de600`, s10 `b19f6906467b50c1`, native = wasm).

### The view transform (one place: `web/src/render.ts`)
- `toScreen` / `toWorld` plus `screenDirToWorld`, `screenAngleToWorld`, `worldAngleToScreen`, `keysToMove` (all derived
  from toScreen/toWorld). `View.rot = true` = rink vertical: +x up (BLUE attacks the top net, defends the bottom one), +y left;
  `rot = false` = the old horizontal view. The sim never sees it (+x stays along the rink length).
- WHO MUST USE IT: everything that converts between screen and world: the canvas drawing (rink, players, sticks, the stick
  target line, ghost, spawn and target markers), `fitView` (camera/zoom), WASD (`keysToMove`: W = up the screen), the mouse aim
  (relative and cursor), practice-tool cursor positions (F, G, Q via `toWorld`), and the verify hooks (`worldToClient`).
  The scoreboard (`drawMatchHud`) is screen-space and orientation-free. Never branch on `view.rot` elsewhere.
- Game: always vertical. Lab: "Rink vertical" checkbox, default on; `?view=horizontal` (or `?view=lab`) = the old view,
  for earlier recordings and scripts (recordings store world inputs, so they replay identically in either view).
- `fitView` now never returns a scale ≤ 0: the 1×1 placeholder canvas before the first layout gave a negative scale, which
  flipped the transform and started the lab's relative aim pointing backwards (found in w2, fixed).

### Mouse model (`web/src/stick.ts`, shared by the game and the lab)
- **Relative (default)**: the first click on the rink (in a match, or anywhere in the lab) captures the pointer (Pointer Lock
  API). Only `movementX` turns the stick: right = clockwise on screen (the stick swings right), left = counterclockwise;
  `movementY` is ignored. The angle is kept in SCREEN radians, starts forward (toward the opponent's net: up), is never clamped,
  and goes to the sim as `stick_target_angle` in world radians via `screenAngleToWorld`. The stick still turns only as fast as
  the sim's chase limits (26.17 rad/s, 513.9 up / 2316 down rad/s²).
- **Cursor aim**: the old model (stick toward the cursor). Lab option and `?mouse=cursor`; the older verify scripts use it.
- Settings (Controls screen in the game, Mouse section in the lab): mode and sensitivity, default 0.005 rad per mouse unit
  (100 units ≈ 28.6°), range 0.001–0.02; stored in localStorage `slapmini.mouse.v1` inside try/catch (blocked storage =
  defaults for the page). A "Click the rink to capture the mouse" hint shows whenever relative mode is on and the pointer is
  free. Esc releases the pointer (the page also calls exitPointerLock); in a match the match goes on, and a second Esc
  returns to the menu. Game over releases it. In the lab, while captured, F/G use the last cursor position.
- The game resets the aim to forward at every match start (not at goal resets: unknown what the real game does).
- Recording/replay log `stick_target_angle`, not mouse events: a relative-mouse recording replays byte-identically.

## Stick lead and brake (session w3, 2026-11-05, branch w3)
Web only (web/src, web/index.html, web/scripts). sim/, params and bots untouched: crosscheck hashes unchanged (free play
`ed2b9c3b2c52da91`, match `ef5c44a7fc4de600`, s10 `b19f6906467b50c1`, native = wasm). The sim's stick chase is unchanged.

### "The stick throttles and doesn't move": cause as found
- `npm run verify-stick` (`web/scripts/repro_stick.mjs`): REAL pointer lock, bursts of movementX (20 / 60 / 150 / 400 units per
  frame for 30 frames, both directions, then 1 s still), per sim step the stick angle, ω and target. Before the fix, at 400
  units per frame (2 rad per frame, far above the 26.17 rad/s stick cap) the stick turned only **164° of 3438°**, its mean ω was
  −3 rad/s (against the mouse), with **28 reversal steps while the target was more than 90° away** (up to 177° wrapped), 8–9
  stall steps, and the unbounded target 3600° (10 turns) ahead when the mouse stopped. Cause confirmed: StickAim kept an
  unbounded target; the sim turns the stick the short way (wrap(target − stick)); once the target ran more than half a turn
  ahead, the short way flipped and the stick turned back and stalled. (The leftover whole-turn "debt" is invisible to the
  wrapped chase, so after the mouse stopped the stick simply stopped on a wrapped point, not where the mouse aimed.)
  At 20–150 units per frame the stick kept up (lead ≤ 56°) and nothing went wrong.
- Fix (`web/src/stick.ts`): **`MAX_LEAD = 1.2` rad (69°)**, one constant. Every sim step `StickAim.worldAngle(view,
  stickWorld)` converts the sim's current stick_angle into screen space through the view transform, pulls the aim back to
  within MAX_LEAD of it (mouse movement beyond that is dropped, not owed) and stores the aim wrapped into one turn. Callers:
  `humanInput` (game) and `readHumanInput` (lab) pass `players[0].stick_angle`; cursor-aim mode is unchanged.
- After (same script): **0 flip-reversals and 0 stalls at every speed**; at 400 units per frame the stick turns at the cap
  (mean ω 25.7–26.2 rad/s, 1484–1536° in the burst); max lead 66.7° (< 69°); after the mouse stops it settles in 0.025 s
  (residual lead 18.8°). Occasional 1–2 reverse steps 0.3–0.6° from the target are the sim's hard-stop overshoot when mouse
  events pause briefly (also seen before the fix at 150 units); the script counts them separately.
- Slow movement (100 units over ~1 s, 2 units per frame): target = integrated mouse angle exactly (max |target − mouse|
  0.000°), the stick on it, no lag or deadzone.
- Lab readout: "lead" = target − stick (wrapped, after the step) with "AT LIMIT" in amber when the aim hit MAX_LEAD within the
  last 8 sim steps (`StickAim.clamped`), plus the mouse mode and sensitivity.

### Brake (Space) and lift (E)
**Superseded in w5b:** Space is now the sim input `PlayerInput.brake` (a slow walk); `web/src/brake.ts` is gone. Lift on E stands.
- Bindings (`web/src/bindings.ts`, all UNVERIFIED guesses): **lift moved from Space to E**, **Space = brake**. The Controls
  screen and the lab's key list show both, marked unverified.
- `web/src/brake.ts` builds the brake from the sim's own controls (the sim has no brake input; adding one would change the
  physics before the first training run): while Space is held the move input points opposite to the velocity with length
  m = min(1, speed / (player_accel · timestep)), so the player decelerates at the skating acceleration and stops on the step it
  would overshoot, never reversing; below 0.05 m/s no move is sent. Space overrides WASD; dash and lift pass through.
  **The stopping time is a PLACEHOLDER** (the sim's acceleration run backwards; the real game's braking is unmeasured: the
  "W then S at top speed" recording is still missing). **Bots do not brake.** LIFT and BRAKE badges show while held (game and lab).
- Measured (`npm run verify-w3`, lab, real keys): from top speed 11.6935 m/s with W held, Space stops the player in **0.667 s
  over 3.794 m** (max/accel predicts 0.674 s), speed falling linearly, min along-velocity 0, sideways 0; at rest it sends no
  move; Space released mid-stop with W held → speed back up to 11.69 (move (1, 0)); Space 0.06 s into a dash: the dash still
  fires (peak 15.48 m/s), then the brake stops the player 0.858 s after the dash; E → lift true, Space → lift false.
  `node scripts/lift_demo.mjs`: lift at contact still raises the puck to 1.822 m (0 without).
- Record → replay with the relative mouse (clamped aim) and the brake: byte-identical CSVs (sha256 fc3a94d7…, recordings log
  the inputs, including the brake's move and the clamped stick_target_angle).
- Verify-script changes: `verify_w2.mjs` holds E instead of Space for the lift in its recording; `diag_space.mjs`
  (`npm run verify-space`) holds E for the lift checks and now also checks that Space with a focused checkbox neither lifts
  nor toggles it. New: `verify-w3`, `verify-stick`.

## Scale, raised stick, follow camera (session w4, 2026-11-05, branch w4)
Physics changed (player_radius + the raised stick), so the crosscheck hashes and the physics stamp changed (listed below).
ACT/OBS were unchanged by w4; session t2 added the raised stick to them.

### Player to puck scale
- Cause of "the player looks too small": the lab's 6 px minimum puck radius drew the puck (6 px) larger than the player
  (5.41 px) at the whole-rink zoom; the sim radii were already close.
- Measured (`python -m measure.body_puck` → `measure/results/body_puck.json`): body and puck half-max edge radii in the SAME
  frames of the hittingpuck clip (resting puck near the player, < 200 px from the image centre, radius > 2 px; 18 of 39
  episodes): body 8.44 px, puck 4.25 px, **ratio 1.989 (median; mean 2.012, sd 0.078)** → player_radius = 1.989 ×
  puck_radius 0.249853 = **0.4969 m** (was 0.495466; provenance measured-shape, source in the note). The ratio is
  scale-free, so it does not depend on the absolute-scale assumption. Both blobs read ~17% larger than the sim radii at the
  map scale (soft edges); the ratio cancels that.
- Drawn radii in the game, 1280×760: follow camera 28.29 px/m → player 14.06 px, puck 7.07 px; whole rink 10.91 px/m →
  5.42 / 2.73 px (ratio 1.989 both; before w4: 5.41 / 6.00 px). Screenshots `web/verify-out/w4-scale-before.png`,
  `w4-scale-after-follow.png`, `w4-scale-after-fit.png` (`node web/scripts/scale_shot.mjs OUT.png [query]`).
- The game draws true scale (`puckMinPx: 0`); the lab keeps the 6 px minimum as a checkbox "Minimum puck size 6 px
  (display only)", default on.

### Raised stick (sim)
- `PlayerInput.stick_up` and `Player.stick_up` (bool, serde default false, so old JSON loads). The player's flag is set from
  the input every step; raising and lowering are **instant** (placeholder) and the stick keeps rotating while raised.
- Raised, the stick hits the puck only for `stick_up_z_min ≤ z ≤ stick_up_z_max` = **0.35 / 1.5 m (placeholder)** instead of
  stick_z_min / stick_z_max; a puck on the ice passes under it. Appended to params.json and the three pinned params files.
- Hash: `to_bytes` appends 0x5C + one byte per player ONLY when some player has the stick up, so runs that never raise it
  keep their encoding.
- Tests `sim/tests/stick_up.rs`: rolling puck deflected with the stick down / untouched up; a puck at z 0.8 m hit only by
  the raised stick; encoding + serde default; 100k-step determinism with random raises (hash `99709a2c0e5f2646`, 66,440 of
  200,000 player-steps raised, native = wasm).
- Bots: shotlab unchanged (48/48, median |err| 0.001 rad, 27.6–27.7 m/s), SHOT constants kept. Tournament 100×60 s,
  before → after: chaser–chaser 0.53–0.35 (27/61/12) → 0.53–0.39 (24/61/15); chaser–defender 0.50–0.02 (42/56/2) →
  0.54–0.01 (43/56/1); chaser–cautious 17.00 → 16.98, chaser–idle 17.00; all other pairings 0–0. **Scripted bots never raise
  the stick.**

### Web
- Left mouse button (bindings.ts `MOUSE_BINDINGS.stickUp`, Controls screen, marked unverified) raises the stick while held,
  in the game and the lab, both mouse modes (`StickRaise` in stick.ts). In relative mode the click that captures the pointer
  does not raise. Released on mouseup, window blur, pointer unlock, game over and back to the menu. The lab ignores it
  during a replay; recordings carry `stick_up` (sameInput compares it).
- Drawing: a raised stick is lighter, thinner, with a thin offset line; "STICK UP" badge next to LIFT / BRAKE.
- **Follow camera** (game, default): game orientation, centred on the human player (clamped to the rink), fixed zoom
  = min(height / attack zone length (rink_length/2 − blue_line_x + 3 m), width / (rink_width + 2 m)), so the whole attack zone
  fits. Controls → Camera "Whole rink" (or `?camera=fit`) is the old fitted view (localStorage `slapmini.camera.v1`).
  All screen↔world conversion stays in render.ts (`followView`, View `ox/oy`).
- `npm run verify-w4` (real mouse buttons): capture click does not raise; held raises (badge, input and sim state true);
  release lowers; Esc while held lowers; whole-rink camera available; in the lab a puck rolled across the raised stick
  passes untouched (down: deflected); record → replay with raised sticks byte-identical (end hash equal).

## Observation v3 and format guards (ACT v4 is in "Current state")

### OBS v3 (bots/src/obs.rs, OBS_DIM 42)
| idx | field |
|---|---|
| 0..31 | unchanged (see "Bots and training") |
| 32..39 | style slot, not moved |
| 40 | **own stick_up (0/1), not normalized** |
| 41 | **opponent stick_up (0/1; 0 with no opponent), not normalized** |

The running normalizer covers 0..31 only, so the style slot and both flags pass through.

### Guards
- `load_ckpt` refuses other versions. Real checkpoints: `../slapmini-int/runs/merged_match/latest.pt: observation layout v2/40
  does not match the library's v3/42; cannot load`. With only the act version old: `action format ACT v3, this trainer uses ACT v4`.
- `slapenv.lib()` checks the ABI first (an old DLL lacks `slap_act_version`), then ACT/OBS versions.
- `botRuntime.ts` rejects a sidecar that is not obs v3 / 42 / act v4 with `brake_prob`, and falls back to the scripted bot.
- `window.game.forceBotStickUp(on)` is a test hook; `window.game.bot` reports `upProbMean`, `upDecisions`, `stickUp`.

