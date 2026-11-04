"""`params` command: write the measured rink SHAPE into params.json (provenance "measured-shape").

Only ratios are measured; meters still use the placeholder scale (rink_length = 60 m / board px).
- rink_width, rink_corner_radius: inner board outline (ice level), mean over clips.
- net_width, net_depth: the net roof is seen from above. The net is tall, so its roof outline is
  displaced and magnified by parallax: with the camera panned off-center (justhittingwallphysicsfixedpoint,
  rink center 240 px right of the image center) the roof appears 20 px further right. That gives the
  roof-height parallax factor k = shift / offset; ground = principal point + (image - principal) / (1 + k).
  Width and depth use the camera-centered clips, corrected with k. Depth = goal line (ice level) to the
  corrected back of the roof, assuming a flat roof (if the roof slopes down to the back, the true depth
  is larger, up to the uncorrected value, which is also reported).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .geometry import PARAMS_PATH

PP = (960.0, 540.0)  # principal point: image center (assumed)
KEYS = ("rink_width", "rink_corner_radius", "net_width", "net_depth", "net_x", "blue_line_x", "faceoff_x", "faceoff_y", "faceoff_circle_radius")


def measure_shape(maps: list[dict]) -> dict:
    centered = [m for m in maps if abs(m["origin_px"][0] - PP[0]) < 40]
    offset = [m for m in maps if abs(m["origin_px"][0] - PP[0]) >= 100]
    width_m = [m["board_width_px"] * m["m_per_px"] for m in maps]
    corner_m = [m["corner_radius_px"] * m["m_per_px"] for m in maps]
    ks = [m["extra"]["shape"]["net_center_u_offset_px"] / (m["origin_px"][0] - PP[0]) for m in offset]
    if not ks:
        raise RuntimeError("need a clip with the camera panned off-center to measure the net-roof parallax")
    k = float(np.mean(ks))
    nw, nd, nd_raw, gl = [], [], [], []
    for m in centered:
        sh = m["extra"]["shape"]
        lines = sh["grey_lines_v_px"]
        (top, bot) = sorted(sh["net_roofs_px"], key=lambda z: z[1])
        mpp = m["m_per_px"]
        nw += [(top[2] - top[0]) / (1 + k) * mpp, (bot[2] - bot[0]) / (1 + k) * mpp]
        back_top = PP[1] + (top[1] - PP[1]) / (1 + k)
        back_bot = PP[1] + (bot[3] - PP[1]) / (1 + k)
        nd += [(lines[0] - back_top) * mpp, (back_bot - lines[3]) * mpp]
        nd_raw += [(lines[0] - top[1]) * mpp, (bot[3] - lines[3]) * mpp]
        gl.append(sh["goal_line_x_m"])
    # Painted markings (ice plane, so no parallax): goal + blue lines, faceoff dots and circles, all clips.
    goal, blue, fx, fy, fr = [], [], [], [], []
    for m in maps:
        sh = m["extra"]["shape"]
        mpp = m["m_per_px"]  # ice-plane scale (see measure/scale.py)
        v0 = m["origin_px"][1]
        u0 = m["origin_px"][0]
        L = sh["grey_lines_v_px"]
        if len(L) == 4:
            goal += [(v0 - L[0]) * mpp, (L[3] - v0) * mpp]
            blue += [(v0 - L[1]) * mpp, (L[2] - v0) * mpp]
        for k_, f in sh["circles"]["fits"].items():
            if k_.startswith("faceoff") and f:
                fx.append(abs(v0 - f["center_px"][1]) * mpp)
                fy.append(abs(u0 - f["center_px"][0]) * mpp)
                fr.append((f["horizontal_diameter_px"] + f["vertical_diameter_px"]) / 4 * mpp)
    ms = lambda a: (float(np.mean(a)), float(np.std(a)))
    return {
        "net_x": ms(goal),
        "blue_line_x": ms(blue),
        "faceoff_x": ms(fx),
        "faceoff_y": ms(fy),
        "faceoff_circle_radius": ms(fr),
        "rink_width": (float(np.mean(width_m)), float(np.std(width_m))),
        "rink_corner_radius": (float(np.mean(corner_m)), float(np.std(corner_m))),
        "net_width": (float(np.mean(nw)), float(np.std(nw))),
        "net_depth": (float(np.mean(nd)), float(np.std(nd))),
        "net_depth_uncorrected": float(np.mean(nd_raw)),
        "net_roof_parallax_k": k,
        "goal_line_x": float(np.mean(gl)),
        "clips_all": [m["clip"] for m in maps],
        "clips_centered": [m["clip"] for m in centered],
    }


def run_params(results: Path, clips: list[str], params_path: Path = PARAMS_PATH, write: bool = True) -> dict:
    maps = [json.loads((results / f"{c}.map.json").read_text()) for c in clips]
    s = measure_shape(maps)
    common = "absolute scale UNMEASURED (meters assume the placeholder rink_length 60 m); the ratio to rink_length is measured"
    notes = {
        "rink_width": f"MEASURED SHAPE: inner board width, mean of {len(s['clips_all'])} clips (sd {s['rink_width'][1]:.3f} m); length/width = {60.0 / s['rink_width'][0]:.4f}; {common}",
        "rink_corner_radius": f"MEASURED SHAPE: from the rounded outline's area deficit, mean of {len(s['clips_all'])} clips (sd {s['rink_corner_radius'][1]:.3f} m); {common}",
        "net_width": f"MEASURED SHAPE: net roof width seen from above, corrected for roof-height parallax (k = {s['net_roof_parallax_k']:.4f}, from the off-center clip); camera-centered clips; {common}",
        "net_depth": f"MEASURED SHAPE: goal line to the parallax-corrected back of the roof, assuming a flat roof (uncorrected roof outline gives {s['net_depth_uncorrected']:.2f} m); {common}",
    }
    n_cl = len(s["clips_all"])
    marks = f"painted ice markings, ice-plane scale, mean of {n_cl} clips"
    notes["net_x"] = f"MEASURED SHAPE: goal line |x| (net mouth), grey line nearest each net, {marks} (sd {s['net_x'][1]:.3f} m); {common}"
    notes["blue_line_x"] = f"DISPLAY ONLY. MEASURED SHAPE: |x| of the inner grey lines, {marks} (sd {s['blue_line_x'][1]:.3f} m); {common}"
    notes["faceoff_x"] = f"DISPLAY ONLY. MEASURED SHAPE: |x| of the end-zone faceoff dots, {marks} (sd {s['faceoff_x'][1]:.3f} m); {common}"
    notes["faceoff_y"] = f"DISPLAY ONLY. MEASURED SHAPE: |y| of the end-zone faceoff dots, {marks} (sd {s['faceoff_y'][1]:.3f} m); {common}"
    notes["faceoff_circle_radius"] = f"DISPLAY ONLY. MEASURED SHAPE: faceoff circle radius (ellipse fits), {marks} (sd {s['faceoff_circle_radius'][1]:.3f} m); {common}"
    print(f"[params] rink length/width measured = {60.0 / s['rink_width'][0]:.4f} (params.json had 60/26 = 2.3077)")
    for k in KEYS:
        print(f"[params] {k:20s} = {s[k][0]:.4f} m  (sd {s[k][1]:.4f})")
    print(f"[params] net roof parallax k = {s['net_roof_parallax_k']:.4f}; uncorrected net depth {s['net_depth_uncorrected']:.3f} m; goal line |x| = {s['goal_line_x']:.3f} m")
    if write:
        text = Path(params_path).read_text()
        for k in KEYS:
            pat = re.compile(r'^(\s*"' + k + r'":\s*)\{.*\},?$', re.M)
            mm = pat.search(text)
            if not mm:
                raise RuntimeError(f"{k} not found in {params_path}")
            comma = "," if mm.group(0).rstrip().endswith(",") else ""
            entry = json.dumps({"value": round(s[k][0], 4), "provenance": "measured-shape", "unit": "m", "note": notes[k]})
            text = text[: mm.start()] + mm.group(1) + entry + comma + text[mm.end() :]
        json.loads(text)  # still valid JSON
        Path(params_path).write_text(text)
        print(f"[params] wrote {len(KEYS)} entries to {params_path} with provenance measured-shape")
    return s
