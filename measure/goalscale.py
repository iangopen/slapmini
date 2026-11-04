"""Absolute scale from the goal-speed clip (session s7).

    python -m measure.goalscale track      # registration + puck for goalspeeds_part00..03 (one recording, in order)
    python -m measure.goalscale scale      # per-goal speeds -> measure/results/scale.json

The game prints "SHOT SPEED: xx.xx MPH" under the goal banner. The readouts were read by eye from the frames
(READOUTS; crops in CLAUDE.md "Session s7"). For each goal the puck's speed in reference px/s (camera motion
removed, measure/camreg.py) is fitted over the last 0.2 s before it crosses the goal line into the net, and
m_per_px = readout (m/s) / speed (px/s). Because the readout is called SHOT speed it probably measures the puck as
it leaves the stick, so a drag-corrected version extrapolates the measured speed back to the hit with the fitted
linear drag (v_hit = v_goal * exp(k * t_flight), k = ice_linear_drag, scale-free); both are reported.
ASSUMPTION: the readout is in real units (mph of the game's own world), i.e. the game's m/s are real m/s.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from .geometry import REPO, load_params
from .skate import OUT, track_clip

PARTS = [f"goalspeeds_part{k:02d}" for k in range(4)]
MPH = 0.44704  # m/s per mph
# (part, banner onset frame within the part, readout in mph, airborne), read by eye from the frames: the banner text
# "SHOT SPEED: xx.xx MPH", and whether the puck is drawn above a white ground ring (a lifted shot) before the goal.
READOUTS = [(0, 137, 13.42, False), (0, 1187, 65.63, True), (1, 160, 67.14, True), (1, 776, 35.94, False),
            (2, 447, 11.24, False), (2, 1302, 88.05, False)]
FIT_S = 0.2  # seconds before the goal line used for the speed


def load_joined() -> tuple[list[dict], list[int], float, float]:
    rows, offsets, off = [], [], 0
    s_vals = []
    fps = 60.0
    for name in PARTS:
        summ = json.loads((OUT / f"{name}.json").read_text())
        s_vals.append(summ["calibration"]["s_px_per_m"])
        fps = summ["fps"]
        with open(OUT / f"{name}.frames.csv") as fh:
            part = list(csv.DictReader(fh))
        offsets.append(off)
        for r in part:
            r["g"] = off + int(r["frame"])
        rows += part
        off += len(part)
    return rows, offsets, float(np.median(s_vals)), fps


def goal_crossings(rows: list[dict], p: dict) -> list[int]:
    """Indices where the puck crosses a goal line into the net mouth (|y| < net half-width), moving outward."""
    out = []
    prev = None
    for i, r in enumerate(rows):
        if r["puck_x"] == "":
            continue
        x, y = float(r["puck_x"]), float(r["puck_y"])
        if prev is not None:
            px, py, pi = prev
            if i - pi <= 3 and abs(px) < p["net_x"] <= abs(x) and px * x > 0 and abs(y) < p["net_width"] / 2:
                out.append(i)
        prev = (x, y, i)
    return out


def speed_before(rows: list[dict], i: int, s: float, fps: float, span: float = FIT_S) -> dict | None:
    """Linear fit of the puck's ground track (px) over [t_i - span, t_i]: speed px/s and its fit residual."""
    n = int(round(span * fps))
    pts = [(int(r["g"]), float(r["puck_x"]) * s, float(r["puck_y"]) * s) for r in rows[max(0, i - n) : i + 1] if r["puck_x"] != ""]
    if len(pts) < max(5, n // 2):
        return None
    g, X, Y = (np.array(c, float) for c in zip(*pts))
    t = g / fps
    A = np.column_stack([t - t.mean(), np.ones_like(t)])
    cx, rx, *_ = np.linalg.lstsq(A, X, rcond=None)
    cy, ry, *_ = np.linalg.lstsq(A, Y, rcond=None)
    res = math.sqrt((np.sum((A @ cx - X) ** 2) + np.sum((A @ cy - Y) ** 2)) / len(t))
    return {"speed_px_s": float(math.hypot(cx[0], cy[0])), "n": len(t), "fit_rms_px": round(res, 3),
            "t0": float(t[0]), "t1": float(t[-1])}


def hit_before(rows: list[dict], i: int, s: float, fps: float, k: float = 0.47254, max_back_s: float = 3.0,
               rms_px: float = 1.5, max_px: float = 4.0) -> int | None:
    """Start of the free flight that ends at row i: the earliest start for which the track from there to the goal
    line still fits a free puck under linear drag, p(t) = p_end - v_end (exp(k (t_end - t)) - 1) / k, with RMS
    <= rms_px and no point off by more than max_px (the puck on the stick, or before the hit, breaks the fit)."""
    lo = max(0, i - int(max_back_s * fps))
    pts = [(int(rows[j]["g"]), float(rows[j]["puck_x"]) * s, float(rows[j]["puck_y"]) * s, j) for j in range(lo, i + 1) if rows[j]["puck_x"] != ""]
    if len(pts) < 6:
        return None
    t_end = pts[-1][0] / fps
    best = None
    for start in range(len(pts) - 6, -1, -1):
        seg = pts[start:]
        t = np.array([q[0] / fps for q in seg])
        basis = (np.exp(k * (t_end - t)) - 1) / k
        A = np.column_stack([np.ones_like(t), -basis])
        res = []
        for col in (1, 2):
            y = np.array([q[col] for q in seg])
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            res.append(A @ c - y)
        r = np.hypot(res[0], res[1])
        if math.sqrt(np.mean(r * r)) <= rms_px and r.max() <= max_px:
            best = seg[0][3]
        else:
            break
    return best


def run_scale() -> dict:
    p = load_params()
    k_drag = p["ice_linear_drag"]
    rows, offsets, s, fps = load_joined()
    cross = goal_crossings(rows, p)
    gidx = {int(r["g"]): i for i, r in enumerate(rows)}
    goals = []
    for part, onset, mph, airborne in READOUTS:
        g_on = offsets[part] + onset
        entry = {"part": part, "banner_frame_in_part": onset, "banner_frame_global": g_on, "readout_mph": mph,
                 "readout_m_s": round(mph * MPH, 4), "airborne": airborne}
        # the banner appears as the puck crosses the goal line; frames after it have darkened ice (unregistered)
        cands = [i for i in cross if g_on - int(4 * fps) <= int(rows[i]["g"]) <= g_on + 3]
        if cands:
            i = cands[-1]
            entry["goal_line"] = "crossing tracked"
        else:
            back = [k for k in range(gidx[g_on], max(0, gidx[g_on] - int(0.5 * fps)), -1) if rows[k]["puck_x"] != ""]
            if not back:
                entry["status"] = "no puck track before the banner"
                goals.append(entry)
                continue
            i = back[0]
            entry["goal_line"] = "last tracked sample before the banner (crossing not tracked)"
        h = hit_before(rows, i, s, fps, k_drag)
        t_cross = int(rows[i]["g"]) / fps
        t_hit = int(rows[h]["g"]) / fps if h is not None else t_cross - 1.0
        span = min(FIT_S, max(t_cross - t_hit, 3 / fps))
        sp = speed_before(rows, i, s, fps, span)
        if sp is None:
            entry["status"] = "too few puck samples in the flight"
            goals.append(entry)
            continue
        v_goal = sp["speed_px_s"]
        t_mid = (sp["t0"] + sp["t1"]) / 2
        v_hit = v_goal * math.exp(k_drag * (t_mid - t_hit))
        entry.update({"status": "ok", "goal_line_frame_global": int(rows[i]["g"]), "hit_frame_global": int(rows[h]["g"]) if h is not None else None,
                      "flight_s": round(t_cross - t_hit, 3), "fit_window_s": round(span, 3), "fit": sp,
                      "puck_speed_px_s_at_goal": round(v_goal, 2), "puck_speed_px_s_at_hit_drag_corrected": round(v_hit, 2),
                      "m_per_px_at_goal": mph * MPH / v_goal, "m_per_px_at_hit": mph * MPH / v_hit})
        goals.append(entry)
    placeholder_mpp = 1.0 / s

    def stats(key, ground_only=True):
        v = np.array([g[key] for g in goals if g.get("status") == "ok" and (not ground_only or not g["airborne"])])
        if not len(v):
            return None
        return {"median": float(np.median(v)), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                "cv": float(np.std(v, ddof=1) / np.mean(v)) if len(v) > 1 else None, "min": float(v.min()), "max": float(v.max()), "n": int(len(v)),
                "factor_vs_placeholder_median": float(np.median(v) / placeholder_mpp),
                "rink_length_m_implied": float(p["rink_length"] * np.median(v) / placeholder_mpp)}

    at_goal, at_hit = stats("m_per_px_at_goal"), stats("m_per_px_at_hit")
    primary = "at_goal" if (at_goal and at_hit and at_goal["cv"] <= at_hit["cv"]) else "at_hit"
    out = {
        "note": ("Absolute scale from the goal-speed clip (CLAUDE.md 'Session s7'). m_per_px is per reference px at the goal "
                 "clip's zoom; factor_vs_placeholder = real meters per placeholder meter (multiply placeholder-scale lengths, "
                 "speeds and accelerations by it; times and rates such as ice_linear_drag are unchanged). ASSUMPTION: the "
                 "readout is in real units (mph). Two readings of what the readout measures are compared: the puck speed at "
                 "the goal line (last <= 0.2 s of the flight) and the speed at the hit (extrapolated back with the fitted linear "
                 "drag). The one that is more consistent across the ground goals is the primary estimate. Airborne goals "
                 "(the puck drawn above a white ground ring: its screen position is displaced by height) are listed but not "
                 "used for the estimate."),
        "readout_unit": "mph (1 mph = 0.44704 m/s)", "fps": fps,
        "goal_clip_px_per_placeholder_m": s, "placeholder_m_per_px": placeholder_mpp, "ice_linear_drag_used": k_drag,
        "primary": primary,
        "m_per_px": at_goal if primary == "at_goal" else at_hit,
        "m_per_px_at_goal_ground": at_goal, "m_per_px_at_hit_ground": at_hit,
        "m_per_px_at_goal_all": stats("m_per_px_at_goal", False), "m_per_px_at_hit_all": stats("m_per_px_at_hit", False),
        "goals": goals,
    }
    path = REPO / "measure" / "results" / "scale.json"
    path.write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m measure.goalscale")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("track")
    sub.add_parser("scale")
    a = ap.parse_args(argv)
    if a.cmd == "track":
        for name in PARTS:
            track_clip(REPO / "clips" / f"{name}.mkv", read_keys=False)
            print("tracked", name)
    else:
        out = run_scale()
        for g in out["goals"]:
            print(json.dumps({k: g.get(k) for k in ("part", "readout_mph", "airborne", "status", "goal_line", "flight_s", "fit_window_s",
                                                    "puck_speed_px_s_at_goal", "puck_speed_px_s_at_hit_drag_corrected",
                                                    "m_per_px_at_goal", "m_per_px_at_hit")}))
        print(json.dumps({k: out[k] for k in ("primary", "m_per_px_at_goal_ground", "m_per_px_at_hit_ground")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
