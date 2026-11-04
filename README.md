# slapmini

A compressed, top-down 2D (2.5D: the puck has height) hockey sim in Rust, modeled on Slapshot: Rebound,
for training RL bots. `params.json` holds the rink shape measured from real clips (provenance `measured-shape`) and four puck
params fitted to them (`fitted`); everything else is still an unmeasured placeholder (`params.placeholder.json` keeps the
pre-fit values for tests).

![Real shots (dashed) vs the sim replaying them with the fitted params (solid), on the sim's own rink](measure/results/fits/readme_fit.png)

*Five real practice-mode shots (dashed, measured from screen recordings by `measure/`) and the sim replaying each one from
its measured start with the fitted puck params (solid). Right: speed against distance for the corner ride, fitted vs
placeholder physics. Meters are at a placeholder scale until a clip with the in-game speed readout pins it down.*

Crates:

- `sim` – deterministic physics core (pure: no I/O, clock or RNG; builds for wasm32).
- `calib` – CLI + library: replay scenarios, score them against trajectory CSVs, and fit params.
- `sim-wasm` – wasm-bindgen wrapper (`new`, `step`, `get_state`, `set_state`, `set_params`, `get_params`, `reset`, `start_match`, `state_hash`).
- `web/` – the feel lab: a browser client for playing the sim, tuning params live, practice drills, record/replay and ghost overlays.

## Commands (same in PowerShell and macOS/Linux shells)

```sh
cargo build --release
cargo test --workspace
cargo test --workspace --release -- --nocapture       # prints fitter recovery + determinism hashes
cargo run --release -p sim --example bench            # single-thread steps/sec
wasm-pack build sim-wasm --release --target web       # -> sim-wasm/pkg

# calibration (the synthetic self-check data in calib/data/ uses the pinned calib/data/params.synthetic.json)
cargo run --release -p calib -- synth  --params calib/data/params.synthetic.json --out-dir calib/data
cargo run --release -p calib -- eval   --params calib/data/params.synthetic.json calib/data/slide_stop.scenario.json calib/data/slide_stop.csv
cargo run --release -p calib -- fit    --params calib/data/params.synthetic.json --fit ice_friction_decel,board_restitution --out fitted.json \
    [--objective path] calib/data/slide_stop.scenario.json calib/data/slide_stop.csv calib/data/wall_bounce.scenario.json calib/data/wall_bounce.csv
cargo run --release -p calib -- replay --params params.json <scenario.json> <out.csv>
# real segments (measured): baseline error of the current params
cargo run --release -p calib -- eval --params params.json calib/data/real/wall_09.scenario.json calib/data/real/wall_09.csv

# native-vs-wasm determinism cross-check
wasm-pack build sim-wasm --release --target nodejs --out-dir pkg-node
node scripts/wasm_crosscheck.mjs
cargo test -p sim --release crosscheck_hash_native -- --nocapture
```

## Feel lab (web/)

```sh
cd web
npm install
npx playwright install chromium   # only for npm run verify
npm run wasm      # build sim-wasm into web/wasm-pkg
npm run dev       # http://localhost:5173
npm run build     # wasm + typecheck + static site in web/dist (relative paths, host as plain files)
npm run verify    # headless Playwright checks; screenshots in web/verify-out
npm run verify-match   # three full matches in match mode (mercy, clock, overtime), sped-up loop; match-*.png
```

Controls: WASD move; the mouse sets the stick target; Shift dash and Space lift (both unverified guesses).
Practice tools (approximate): F spawn (hold to raise), G target, R reset puck, Q call puck to forehand.
Match mode (checkbox, or `?mode=match`): 1v1 rules from the sim (5:00 clock, mercy lead 5, sudden-death overtime),
scoreboard + OVERTIME banner + game-over overlay, Enter = rematch, practice tools off. `?speed=N` runs the loop N× real time.
Downloaded lab recordings (`.scenario.json` + `.csv`) work directly with `calib eval` / `calib fit`.

Setup: Rust via rustup plus `rustup target add wasm32-unknown-unknown`, and wasm-pack (`npm install -g wasm-pack` or `cargo install wasm-pack`).
On Windows without the Windows SDK, use the GNU toolchain: `rustup override set stable-x86_64-pc-windows-gnu` in the repo.

## Measuring real clips (measure/)

A Python tool that turns top-down screen recordings of the real game into trajectory CSVs in calib's format.
It uses OpenCV's bundled ffmpeg, so no system ffmpeg is needed. Clips live in `clips/` (gitignored; fetch them from the
`clip-data` branch with `git fetch origin clip-data`, `git checkout FETCH_HEAD -- clips/`, `git reset -q clips/`).
Outputs go to `measure/results/` (small CSV/JSON/PNG, committed). The median background frame is cached as `clips/NAME.bg.png`.

PowerShell (Windows):

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r measure\requirements.txt
.venv\Scripts\python -m measure map   clips\justpuckphysicshittingitatfixedpoint.mkv
.venv\Scripts\python -m measure track clips\justpuckphysicshittingitatfixedpoint.mkv
.venv\Scripts\python -m measure qa    clips\justpuckphysicshittingitatfixedpoint.mkv
.venv\Scripts\python -m unittest discover -s measure\tests -t .
```

bash (macOS / Linux):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r measure/requirements.txt
.venv/bin/python -m measure map   clips/justpuckphysicshittingitatfixedpoint.mkv
.venv/bin/python -m measure track clips/justpuckphysicshittingitatfixedpoint.mkv
.venv/bin/python -m measure qa    clips/justpuckphysicshittingitatfixedpoint.mkv
.venv/bin/python -m unittest discover -s measure/tests -t .
```

Full pipeline (bash; in PowerShell use `.venv\Scripts\python` and backslashes):

```bash
C="clips/justpuckphysicshittingitatfixedpoint.mkv clips/justhittingwallphysicsfixedpoint.mkv clips/justliftedshots.mkv"
.venv/bin/python -m measure map      $C          # rink, circles (camera top-down?), lines, nets, camera check
.venv/bin/python -m measure track    $C          # puck per frame, camera motion undone
.venv/bin/python -m measure qa       $C          # QA sheets per clip
.venv/bin/python -m measure segment  $C          # free-puck stretches -> calib/data/real/*.scenario.json + .csv + manifest.json
.venv/bin/python -m measure segqa    $C --frames wall_09   # segment QA sheets (+ frame strip for a segment)
.venv/bin/python -m measure estimate             # scale-free restitution / deceleration -> calib/data/real/estimates.json
.venv/bin/python -m measure params   $C --dry-run  # measured rink shape (drop --dry-run to write params.json)
.venv/bin/python -m measure scale    $C          # board vs ice-plane scale -> map JSONs
.venv/bin/python -m measure reproject $C         # puck blob -> ground point, rewrites the traj CSVs (track does it too)
cargo build --release -p calib
.venv/bin/python -m measure holdout              # clip-held-out fits -> params.fitted.json (+ calib/data/real/holdout.json)
.venv/bin/python -m measure dragmodel            # drag-law comparison -> calib/data/real/dragmodel.json
.venv/bin/python -m measure compare puck_08 --params params.json --params params.fitted.json   # overlay -> qa/compare_puck_08.png
.venv/bin/python -m measure merge                # params.fitted.json -> params.json (the 4 fitted puck params)
.venv/bin/python -m measure sizes    $C          # puck + player radius from the frames -> params.json (measured-shape)
.venv/bin/python -m measure residuals            # per-segment residuals old vs new default vs held-out
.venv/bin/python -m measure figures              # fit figures on the sim's own rink -> measure/results/fits/
```

- `map` finds the inner board outline, center line and center dot in a median frame, fits ellipses to the center and faceoff
  circles (round = top-down camera), finds the grey lines and net roofs, writes `NAME.map.json` + `NAME.map.png`, and checks the
  camera once per second against four faceoff-dot landmarks: raw drift, and the residual after undoing it by a translation.
  It fails only if the camera motion cannot be corrected to 2 px.
- `track` undoes each frame's camera translation before detection (all positions are reference-frame pixels; frames that cannot
  be corrected are dropped). Writes `NAME.traj.csv` (`t,x,y,z`, z always 0) and `NAME.track.json` (per-row frame, pixel position,
  confidence, interpolated flag, tracklet id, blob area, camera shift; summary; `scale_source`). `--no-camera-correction` drops moved frames instead.
- `qa` writes `qa/NAME.overview.png` (whole path on the background) and `qa/NAME.sheet.png` (sampled frames with zoom insets).
- `segment` / `segqa` / `estimate` / `params`: see CLAUDE.md (segment manifest, scale-free estimates, measured rink shape).
- Several clips can be passed at once. `python -m measure synth out.mkv` renders the synthetic test video.

## File formats

- **params.json**: `{"params": {name: {"value", "provenance", "unit", "note"}}}`. Provenance is `placeholder`, `fitted`, `measured`,
  or `measured-shape` (a ratio to rink_length is measured; the meter value still assumes the placeholder scale).
- **Scenario** (`*.scenario.json`): `{name, description, steps, initial_state: State, inputs: [{step, player, input}]}`.
  Each input is held from its `step` until that player's next event.
- **Trajectory CSV**: header `t,x,y,z` plus an optional `stick_angle` (player 0) column. Puck position in meters, rink-centered.
  Measured tracks use t = frame / fps from the clip's first frame. Rows can be irregular: dropped frames are simply missing.
