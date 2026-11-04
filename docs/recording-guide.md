# Recording guide: player and stick clips

The puck is calibrated from video. The player is not, because the existing clips don't show which keys were
held. These clips fix that: the keys and mouse are visible in the recording, so every speed change can be
matched to the input that caused it.

## What the current clips showed (2026-11-04)

- **The game camera follows the player.** The pink disc stays at the image center and the ice moves under it.
  The tracker measures the player from the painted faceoff dots, so at least 3 of the 4 end-zone faceoff dots must
  stay in view. Keep runs between the two end-zone circles, starting near center ice.
- The only real skating in the old clips is two runs in `staticrotatingstick`: a flat top speed (~180 px/s) and one
  normal ramp. Without inputs it is unclear whether the stops were a key release or a brake, and whether the one
  near-instant start was a dash.

## Setup (OBS)

1. Install **Input Overlay** (OBS plugin by univrsal, Windows/Linux): github.com/univrsal/input-overlay → Releases
   for the binaries, wiki for installation and presets (the repo has a `presets/` folder and the wiki a preset
   creation tool). Add an *Input Overlay* source with a keyboard preset showing **W A S D + Shift + Space** and a
   mouse preset with the buttons and, if your version offers it, **mouse movement** (check this in a 10 s test
   clip: move the mouse in a circle and see that the overlay shows it).
   - If keys don't light up while the game has focus, try running OBS as administrator (an elevated game can
     hide its input from a non-elevated hook). This is a common fix, not something the plugin docs state.
   - Fallback if the plugin won't load or shows no mouse movement: **NohBoard** (keyboard + mouse buttons,
     captured as a window source). For the stick clips (6, 7) without a movement display, also write down
     each flick (from which direction to which) in the clip name or a text file.
2. Place the overlay in a **corner over the arena/stands**, never over the ice or the boards. Bottom-left is free
   in the current layout (the minimap is top-right, the score top-left, the stamina bar bottom-center).
3. Same capture as before: 1920×1080, the same zoom, `.mkv`. 60 fps works. **120 fps if your PC can** (the game
   renders at ~180 fps; a faster recording halves the timing jitter that limits speed measurements).
4. Pressed keys must look clearly different from released ones (default presets do this). Don't move or resize the
   overlay between clips.
5. Write down the in-game mouse sensitivity and the Windows pointer speed once (needed for the stick clips).

## Clips to record

Wait about 1 s at rest before and after each run. Don't use practice tools (F/G/R/Q) during a run, because a reset
teleports the player. 5 or more repeats of each run is good; one clip per numbered item.

| # | Clip (suggested name) | What to do | What it measures |
|---|---|---|---|
| 1 | `start_w` | From rest: hold **W** ~2 s, release, touch nothing until the player stops. Then hold W ~2 s and press **S** (no release) until stopped. | Top speed (plateau), acceleration from rest, coasting drag after a release, and braking with the opposite key. Splits the two kinds of stop the old clips could not tell apart. |
| 2 | `start_asd` | The same as clip 1 with **A**, **S** and **D** (5× each). | Whether speed or acceleration depends on direction (forward vs backward vs sideways relative to where the player/stick faces). |
| 3 | `start_diag` | From rest: hold **W+D** ~2 s, release. Also W+A. | Whether diagonal input is normalized (same top speed) or faster (√2), and the diagonal ramp. |
| 4 | `turn` | Reach top speed with W, then switch to **D** (hold), then to **S**, then to **A**: a full turn, ~1 s per key. Also: at top speed, W→S directly. | How velocity turns under a new input (turn rate vs braking + re-accelerating), and the reversal deceleration. |
| 5 | `dash` | Dash from rest with no move key, 5×. Dash from rest while holding W. Dash at top speed while holding W. Then press dash repeatedly every ~0.3 s. | Dash burst size, duration and decay back to the plateau; whether the near-instant start seen once (0→180 px/s in ~0.15 s) is a dash; the cooldown (the shortest spacing that still bursts). |
| 6 | `stick_sweep` (closer zoom) | Player still. Zoom the camera in so the stick tip is clearly visible. Move the mouse a **known angle** around the player: quarter turns and half turns, each slowly and as a fast flick, and a few full circles. Mark the start/end directions with the cursor on clear ice features (e.g. the center line) so the angle is known. | Stick angular speed and acceleration limits, the lag behind the mouse, and the stick length (tip radius). Needs the mouse movement in the overlay (or the written-down flicks). |
| 7 | `stick_hit` | Puck at rest near the blade (Q or F). Player still. Swing the stick through the puck at several speeds, from a slow push to the fastest flick, 3× each. Normal zoom (whole rink, puck tracked). | Puck exit speed vs swing speed: stick restitution, effective stick mass, and where the per-shot speed caps start. |
| 8 | `coast_long` | Optional: hold W to top speed, release, and let the player coast all the way to a stop without touching anything (repeat in both directions along the rink). | The coasting drag curve over the full speed range (constant vs speed-proportional), if clip 1 shows a slow coast. |

## Checklist before sending clips

- The overlay is visible in every frame and doesn't cover the ice, the boards or the faceoff circles.
- The camera zoom is the same as the old clips (except clip 6).
- Each clip's name says what it is; push to the `clip-data` branch as before (never to main).
