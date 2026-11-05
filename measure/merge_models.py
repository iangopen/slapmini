"""Merge the s7 player/dash and s7b stick fits into params.json (`python -m measure.merge_models`).

Sources (kept as references): params.player.json (s7) and params.stick.json (s7b). Values are copied with their
provenance; player_max_speed takes the model-free measured cap instead of the fitted one, and the entries for
models the sim does not match yet get a COMPROMISE note. Run this BEFORE `measure.rescale` (sources are at the
placeholder meter scale). Not written (the clips could not determine them): stick_mass, stick_inner,
stick_z_min/max, dash_duration, dash_cooldown.
"""

from __future__ import annotations

import json
from pathlib import Path

from .geometry import PARAMS_PATH
from .shape_params import set_entries

ROOT = PARAMS_PATH.parent
PLAYER = ("player_accel", "player_ice_drag", "dash_speed")
STICK = ("stick_max_ang_speed", "stick_max_ang_accel", "stick_length", "stick_thickness", "stick_restitution")

EXTRA_NOTES = {
    "player_ice_drag": "COMPROMISE (sim model mismatch): the game's coast decays in proportion to speed (rate ~0.85 1/s, 0.52-1.34), the sim's player_ice_drag is a constant deceleration.",
    "dash_speed": "COMPROMISE (sim model mismatch): the game SETS the speed to ~12 m/s (placeholder scale; 11.0-12.5 from rest, 12.2-13.3 from top speed), the sim ADDS an impulse of dash_speed.",
    "stick_restitution": "Holds only for the placeholder stick_mass 2 kg (and puck_mass 0.17 kg): mass and restitution are degenerate in the hit data; refit if stick_mass changes.",
}


def merged_entries(root: Path = ROOT) -> dict:
    pl = json.loads((root / "params.player.json").read_text())["params"]
    st = json.loads((root / "params.stick.json").read_text())["params"]
    out = {n: dict(pl[n]) for n in PLAYER}
    out.update({n: dict(st[n]) for n in STICK})
    for n, extra in EXTRA_NOTES.items():
        out[n]["note"] = extra + " " + out[n]["note"]
    out["player_max_speed"] = {
        "value": 10.755,
        "provenance": "measured",
        "unit": "m/s",
        "note": "MEASURED hard cap: model-free speed plateau with a move key held, 10.755 m/s (sd 0.035, n = 15 runs, dead flat; 182.3 px/s at 16.951 px/m) in wkeymovement, diagonalkeypresses, whilemovingturning (s7); diagonal input has the same cap. The calib fit gave 10.62 (compromise with the constant-accel ramp).",
    }
    return out


def main() -> None:
    e = merged_entries()
    set_entries(PARAMS_PATH, e)
    for n, v in e.items():
        print(f"[merge] {n:22s} = {v['value']} {v['unit']} ({v['provenance']})")


if __name__ == "__main__":
    main()
