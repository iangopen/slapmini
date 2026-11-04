"""`sizes` command: puck and player radius from the frames (provenance measured-shape).

Puck: frames where the puck rests within 40 px of the image center (no side wall visible) are averaged; the
greenness G - (R + B)/2 radial profile is taken within 10 px of its center, and the radius is the half-max
edge between the bright core and the ice (the puck's dark-green rim falls outside it). Divided by (1 + k_top)
for the top face's height magnification (track.json puck_parallax).
Player: the pink disc in the median frame (the camera follows the player, so it sits at the image center
and has no lateral parallax). min(R, B) - G pinkness; rays from the blob center find the half-max edge and a
circle is fitted (outlier rejection). The blue rim around it is not included. The disc may be drawn at the
player's height (magnified by an unknown parallax factor), so this is the disc as seen.
Meters use the ice-plane scale (placeholder rink_length 60 m).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import PARAMS_PATH, RinkMap
from .video import iter_frames


def puck_radius(clip: Path, results: Path) -> dict:
    name = clip.stem
    R = json.loads((results / f"{name}.track.json").read_text())
    rows, pp = R["rows"], R["puck_parallax"]
    f = np.array(rows["frame"])
    u = np.array(rows["u_px"]) + np.array(rows["shift_u"])
    v = np.array(rows["v_px"]) + np.array(rows["shift_v"])
    mv = np.r_[0, np.hypot(np.diff(u), np.diff(v))]
    sel = (mv < 0.02) & (np.hypot(u - 960, v - 540) < 40)
    if sel.sum() < 20:
        return {"clip": name, "frames": int(sel.sum()), "error": "too few resting frames near the image center"}
    uc, vc = float(u[sel].mean()), float(v[sel].mean())
    want = set(f[sel][:200].tolist())
    acc, k = None, 0
    for i, fr in iter_frames(clip):
        if i in want:
            acc = fr.astype(float) if acc is None else acc + fr
            k += 1
        if i > max(want):
            break
    F = acc / k
    g = F[..., 1] - (F[..., 0] + F[..., 2]) / 2
    ys, xs = np.mgrid[0 : F.shape[0], 0 : F.shape[1]]
    rr = np.hypot(xs - uc, ys - vc)
    core = float(g[rr < 2].mean())
    bg = float(g[(rr > 8) & (rr < 10) & (F[..., 0] < 120)].mean())
    rs = np.arange(0, 9, 0.05)
    with np.errstate(invalid="ignore"):
        prof = np.array([np.nanmean(g[(rr >= a) & (rr < a + 0.5)]) if ((rr >= a) & (rr < a + 0.5)).any() else np.nan for a in rs])
    r_half = float(rs[np.argmax(prof < bg + 0.5 * (core - bg))] + 0.25)
    m = RinkMap.load(results / f"{name}.map.json")
    r_ground = r_half / (1 + pp["k_top"])
    return {"clip": name, "frames": k, "center_px": [round(uc, 2), round(vc, 2)], "half_max_radius_px": round(r_half, 3),
            "k_top": pp["k_top"], "ground_radius_px": round(r_ground, 3), "radius_m": r_ground * m.m_per_px,
            "blob_model_radius_px": round(math.sqrt(pp["area_px2_at_center"] / math.pi), 3)}


def player_radius(clip: Path, results: Path) -> dict:
    name = clip.stem
    m = RinkMap.load(results / f"{name}.map.json")
    bg = cv2.imread(str(clip.parent / f"{name}.bg.png")).astype(float)
    pink = np.minimum(bg[..., 2], bg[..., 0]) - bg[..., 1]
    mask = (pink > 40) & (m.rink_mask(-5.0) > 0)
    n, _, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n < 2:
        return {"clip": name, "error": "no pink disc"}
    i = max(range(1, n), key=lambda j: st[j][4])
    cx, cy = cen[i]
    ys, xs = np.mgrid[0 : bg.shape[0], 0 : bg.shape[1]]
    half = float(np.percentile(pink[np.hypot(xs - cx, ys - cy) < 5], 80)) / 2
    pts = []
    for a in np.linspace(-np.pi, np.pi, 144, endpoint=False):
        rs = np.arange(0, 20, 0.1)
        px = (cx + rs * np.cos(a)).astype(np.float32).reshape(1, -1)
        py = (cy + rs * np.sin(a)).astype(np.float32).reshape(1, -1)
        val = cv2.remap(pink.astype(np.float32), px, py, cv2.INTER_LINEAR)[0]
        k = int(np.argmax(val < half))
        if k > 0:
            pts.append((px[0, k], py[0, k]))
    P = np.array(pts)
    q = least_squares(lambda z: np.hypot(P[:, 0] - z[0], P[:, 1] - z[1]) - z[2], [cx, cy, 8]).x
    keep = np.ones(len(P), bool)
    for _ in range(3):
        r = np.abs(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]) - q[2])
        keep = r < max(0.75, 2.5 * np.median(r))
        q = least_squares(lambda z: np.hypot(P[keep, 0] - z[0], P[keep, 1] - z[1]) - z[2], q).x
    return {"clip": name, "center_px": [round(float(q[0]), 2), round(float(q[1]), 2)], "radius_px": round(float(q[2]), 3),
            "edge_points_kept": int(keep.sum()), "edge_points": len(P), "radius_m": float(q[2]) * m.m_per_px}


def run_sizes(clips: list[Path], results: Path, params_path: Path = PARAMS_PATH, write: bool = True) -> dict:
    from .shape_params import set_entries

    pucks = [puck_radius(c, results) for c in clips]
    players = [player_radius(c, results) for c in clips]
    for p in pucks:
        if "error" in p:
            print(f"[sizes] puck   {p['clip'][:10]}: {p['error']}")
        else:
            print(f"[sizes] puck   {p['clip'][:10]}: {p['frames']} resting frames at {p['center_px']}; half-max radius {p['half_max_radius_px']:.3f} px / (1 + k_top {p['k_top']:.4f}) = {p['ground_radius_px']:.3f} px = {p['radius_m']:.4f} m (tracker blob model {p['blob_model_radius_px']:.3f} px)")
    for p in players:
        print(f"[sizes] player {p['clip'][:10]}: pink disc at {p['center_px']}, radius {p['radius_px']:.3f} px ({p['edge_points_kept']}/{p['edge_points']} edge points) = {p['radius_m']:.4f} m")
    pr = [p["radius_m"] for p in pucks if "radius_m" in p]
    pl = [p["radius_m"] for p in players if "radius_m" in p]
    res = {"puck": pucks, "player": players, "puck_radius_m": float(np.mean(pr)), "player_radius_m": float(np.mean(pl))}
    common = "absolute scale UNMEASURED (meters assume the placeholder rink_length 60 m); the ratio to rink_length is measured"
    entries = {
        "puck_radius": {"value": round(res["puck_radius_m"], 4), "provenance": "measured-shape", "unit": "m",
                        "note": f"MEASURED SHAPE: half-max edge of the resting puck's bright face near the image center, corrected for the top face's height magnification; {len(pr)} clip(s) ({', '.join(f'{x:.4f}' for x in pr)} m); the dark-green rim reaches ~0.32 m and the tracker blob ~0.245 m; {common}"},
        "player_radius": {"value": round(res["player_radius_m"], 4), "provenance": "measured-shape", "unit": "m",
                          "note": f"MEASURED SHAPE: the player's pink disc (half-max edge circle fit), {len(pl)} clips ({', '.join(f'{x:.4f}' for x in pl)} m); the blue rim around it reaches ~0.53 m; the disc may be drawn at the player's height (unknown parallax magnification, so possibly a few % large); {common}"},
    }
    print(f"[sizes] puck_radius = {res['puck_radius_m']:.4f} m (sd {np.std(pr):.4f}), player_radius = {res['player_radius_m']:.4f} m (sd {np.std(pl):.4f})")
    if write:
        set_entries(params_path, entries)
        print(f"[sizes] wrote puck_radius and player_radius to {params_path} (measured-shape)")
    (results / "sizes.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    return res
