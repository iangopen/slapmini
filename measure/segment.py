"""`segment` command: cut each clip's track into free-puck stretches and write calib scenarios.

A stretch starts once the puck has left the stick (a few frames after the speed peak of a hit, or
where a moving track begins) and ends at the next teleport (practice-tool reset), stop, or contact
(a speed increase anywhere, or a sudden slowdown away from boards and nets). The player is not
tracked, so contacts are inferred from the puck alone. Each stretch becomes a scenario + trajectory
pair in calib/data/real/, with initial_state from a quadratic fit to the first 10 samples and no players.

Labels: "net hit" if the stretch reaches a net (roof box grown toward the mouth), else "wall bounce"
if it changes direction at the boards, else "glide". "airborne" flags stretches where the puck blob is
enlarged (lifted: x/y biased, z unknown).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .geometry import REPO, RinkMap, load_params

OUT = REPO / "calib" / "data" / "real"
MIN_DURATION_S = 0.3
STOP_PX = 0.2  # px/frame: slower than this counts as stopped
WALL_MARGIN_PX = 10.0  # puck center this close to the board outline (plus puck radius) = at the wall
PUCK_R_PX = 4.0
SHORT = {
    "justpuckphysicshittingitatfixedpoint": "puck",
    "justhittingwallphysicsfixedpoint": "wall",
    "justliftedshots": "lift",
}


def board_distance(m: RinkMap, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Distance (px) from points inside the rink to the inner board outline (rounded rectangle)."""
    b, r = m.board_px, m.corner_radius_px
    cx0, cx1 = b["left"] + r, b["right"] - r
    cy0, cy1 = b["top"] + r, b["bottom"] - r
    qx = np.clip(u, cx0, cx1)
    qy = np.clip(v, cy0, cy1)
    in_corner = ((u < cx0) | (u > cx1)) & ((v < cy0) | (v > cy1))
    d_corner = r - np.hypot(u - qx, v - qy)
    d_side = np.minimum.reduce([u - b["left"], b["right"] - u, v - b["top"], b["bottom"] - v])
    return np.where(in_corner, d_corner, d_side)


def board_normal(m: RinkMap, u: float, v: float) -> tuple[float, float]:
    """Inward unit normal (pixel frame) of the board nearest to (u, v)."""
    b, r = m.board_px, m.corner_radius_px
    cx0, cx1 = b["left"] + r, b["right"] - r
    cy0, cy1 = b["top"] + r, b["bottom"] - r
    if (u < cx0 or u > cx1) and (v < cy0 or v > cy1):
        qx, qy = min(max(u, cx0), cx1), min(max(v, cy0), cy1)
        d = math.hypot(qx - u, qy - v) or 1.0
        return ((qx - u) / d, (qy - v) / d)
    ds = {(1.0, 0.0): u - b["left"], (-1.0, 0.0): b["right"] - u, (0.0, 1.0): v - b["top"], (0.0, -1.0): b["bottom"] - v}
    return min(ds, key=ds.get)


def net_boxes(m: RinkMap, mouth_px: float = 30.0, pad: float = 6.0) -> list[list[float]]:
    """Net roof boxes grown toward center ice (the mouth and posts are in front of the roof)."""
    out = []
    v0 = m.origin_px[1]
    for x0, y0, x1, y1 in m.extra["shape"]["net_roofs_px"]:
        if y1 < v0:  # top net: mouth below
            out.append([x0 - pad, y0 - pad, x1 + pad, y1 + mouth_px])
        else:
            out.append([x0 - pad, y0 - mouth_px, x1 + pad, y1 + pad])
    return out


def near_net(boxes, u, v) -> np.ndarray:
    hit = np.zeros(np.shape(u), bool)
    for x0, y0, x1, y1 in boxes:
        hit |= (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
    return hit


def window_speed(f: np.ndarray, u: np.ndarray, v: np.ndarray, w: int = 3) -> np.ndarray:
    """Centered speed (px/frame) over +-w rows: robust to the capture's per-frame timing jitter."""
    n = len(f)
    sp = np.zeros(n)
    for k in range(n):
        a, b = max(k - w, 0), min(k + w, n - 1)
        if b > a:
            sp[k] = math.hypot(u[b] - u[a], v[b] - v[a]) / (f[b] - f[a])
    return sp


def direction(f, u, v, k, side: int, w: int = 4):
    """Mean direction over w rows before (side=-1) or after (side=+1) row k."""
    n = len(f)
    a, b = (max(k - w, 0), k) if side < 0 else (k, min(k + w, n - 1))
    if b <= a:
        return None
    du, dv = u[b] - u[a], v[b] - v[a]
    L = math.hypot(du, dv)
    return None if L < 1e-6 else (du / L, dv / L)


def quad_state(t: np.ndarray, x: np.ndarray, y: np.ndarray, at: float):
    """Position and velocity at time `at` from a quadratic least-squares fit."""
    deg = 2 if len(t) >= 6 else 1
    cx = np.polyfit(t - at, x, deg)
    cy = np.polyfit(t - at, y, deg)
    return float(cx[-1]), float(cy[-1]), float(cx[-2]), float(cy[-2])


def area_model(u_raw: np.ndarray, v_raw: np.ndarray, area: np.ndarray, center=(960.0, 540.0)):
    """Expected on-ice puck blob area vs distance r from the image center (raw frame pixels).

    The camera is a perspective camera: off-axis, more of the puck's side wall shows, so the blob grows
    with r (about 45 px^2 at the center to 83 px^2 at the boards). Fit: robust line through per-bin
    medians. Lifted pucks are closer to the camera and bigger than this."""
    r = np.hypot(u_raw - center[0], v_raw - center[1])
    ok = area > 0
    bins = np.arange(0, r[ok].max() + 50, 50)
    rc, med = [], []
    for lo in bins:
        s_ = ok & (r >= lo) & (r < lo + 50)
        if s_.sum() >= 20:
            rc.append(float(np.median(r[s_])))
            med.append(float(np.median(area[s_])))
    if len(rc) < 2:
        raise ValueError("too few radius bins for the area model")
    a1, a0 = np.polyfit(rc, med, 1)
    return float(a0), float(a1)


def analyse_tracklet(m: RinkMap, f, u, v, area, area_ref, boxes):
    """Split one tracklet into free stretches. Returns a list of dicts with row index ranges + events."""
    n = len(f)
    sp = window_speed(f, u, v)
    wall = board_distance(m, u, v) < PUCK_R_PX + WALL_MARGIN_PX
    net = near_net(boxes, u, v)
    bad = np.zeros(n, bool)
    reason = {}
    step = np.r_[0.0, np.hypot(np.diff(u), np.diff(v)) / np.maximum(np.diff(f), 1)]
    for k in range(1, n):
        if f[k] - f[k - 1] > 3:  # a real gap (dropped duplicates leave up to 3): new stretch
            bad[k] = True
            reason[k] = "gap"
        before = np.median(step[max(k - 4, 1) : k]) if k > 1 else 0.0
        after = np.median(step[k + 1 : k + 5]) if k + 1 < n else 0.0
        if step[k] > 5.0 and step[k] > 3.0 * max(before, after) + 3.0:
            bad[k] = True
            reason[k] = "teleport"
    # Contacts: compare the speed over rows [k-7, k-2] with rows [k+2, k+7] (the frames right at k are
    # skipped, so a wall reversal at k reads as "slower after", never as a speed-up).
    def span_speed(a, b):
        a, b = max(a, 0), min(b, n - 1)
        if b <= a or f[b] == f[a]:
            return None
        return math.hypot(u[b] - u[a], v[b] - v[a]) / (f[b] - f[a])

    for k in range(2, n - 2):
        lo, hi = span_speed(k - 7, k - 2), span_speed(k + 2, k + 7)
        if lo is None or hi is None:
            continue
        pre = max(lo, span_speed(k - 12, k - 7) or 0.0)  # a bounce inside [k-7, k-2] can't hide the pre-bounce speed
        if hi > 1.15 * pre + 0.4:  # speeding up: a stick contact (walls and friction only slow it)
            bad[max(k - 3, 0) : k + 4] = True
            reason.setdefault(k, "contact (speed-up)")
        elif hi < 0.7 * lo - 0.2 and not (wall[max(k - 8, 0) : k + 9].any() or net[max(k - 8, 0) : k + 9].any()):
            bad[max(k - 3, 0) : k + 4] = True  # sudden slowdown away from boards/nets: the stick
            reason.setdefault(k, "contact (slow-down)")
    stopped = sp < STOP_PX
    ok = ~bad & ~stopped
    out = []
    k = 0
    while k < n:
        if not ok[k]:
            k += 1
            continue
        j = k
        while j + 1 < n and ok[j + 1] and f[j + 1] - f[j] <= 3:
            j += 1
        # Why it starts / ends
        start_reason = "track start (moving)" if k == 0 else ("after contact" if bad[k - 1] else ("after stop" if stopped[k - 1] else "after gap"))
        if j == n - 1:
            end_reason = "track end (teleport/reset or lost)"
        else:
            nxt = j + 1
            end_reason = "stop" if stopped[nxt] and not bad[nxt] else next((reason[q] for q in range(nxt, min(nxt + 8, n)) if q in reason), "contact")
        out.append({"i0": k, "i1": j, "start_reason": start_reason, "end_reason": end_reason})
        k = j + 1
    # events inside each stretch
    for s in out:
        bounces, net_hit = [], False
        bd = board_distance(m, u, v)
        for q in range(s["i0"], s["i1"] - 1):
            if net[q]:
                d0, d1 = direction(f, u, v, q, -1), direction(f, u, v, q, +1)
                if d0 and d1 and d0[0] * d1[0] + d0[1] * d1[1] < math.cos(math.radians(15)):
                    net_hit = True
                continue
            if not wall[q] or bd[q] > bd[max(q - 5, s["i0"]) : min(q + 6, s["i1"] + 1)].min():
                continue  # bounce = local minimum of the distance to the boards
            # The approach may come from before the stretch (a shot straight into the board next to the player).
            a0, a1 = max(q - 7, 0), max(q - 2, 0)
            b0, b1 = min(q + 2, s["i1"]), min(q + 7, s["i1"])
            if a1 <= a0 or b1 <= b0:
                continue
            nx, ny = board_normal(m, u[q], v[q])
            vin = ((u[a1] - u[a0]) * nx + (v[a1] - v[a0]) * ny) / (f[a1] - f[a0])
            vout = ((u[b1] - u[b0]) * nx + (v[b1] - v[b0]) * ny) / (f[b1] - f[b0])
            if vin < -0.5 and vout > 0.1 and (not bounces or q - bounces[-1] > 6):
                bounces.append(q)  # the velocity component along the board normal flipped: a bounce
        if net[s["i0"] : s["i1"] + 1].any() or (s["i1"] + 1 < n and net[s["i1"] : min(s["i1"] + 3, n)].any()):
            net_hit = True
        s["bounces"] = bounces
        s["net_hit"] = net_hit
        big = area[s["i0"] : s["i1"] + 1] > 1.25 * area_ref[s["i0"] : s["i1"] + 1]
        s["airborne_frac"] = float(big.mean())
    return out


def run_segment(clips: list[Path], results: Path, out_dir: Path = OUT, params_path=None) -> dict:
    params = load_params(params_path) if params_path else load_params()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "Free-puck stretches from Ian's practice-mode clips (measure/segment.py). MEASURED positions at the PLACEHOLDER "
        "meter scale (params rink_length / measured board px); z unknown (0). No player in the scenarios: the puck is free from t = 0.",
        "convention": "see CLAUDE.md: meters, origin center dot, +x toward the top net, +y screen left",
        "timestep_s": params["timestep"],
        "segments": [],
    }
    for clip in clips:
        name = Path(clip).stem
        m = RinkMap.load(results / f"{name}.map.json")
        meta = json.loads((results / f"{name}.track.json").read_text())
        R = meta["rows"]
        f_all = np.array(R["frame"])
        # Positions: ground points (blob centroid reprojected for the puck's height) when available.
        u_all, v_all = np.array(R.get("ug_px", R["u_px"])), np.array(R.get("vg_px", R["v_px"]))
        u_blob, v_blob = np.array(R["u_px"]), np.array(R["v_px"])
        tid = np.array(R["tracklet"])
        area_all = np.array(R.get("area_px", [50] * len(f_all)), float)
        u_raw = u_blob + np.array(R.get("shift_u", [0] * len(f_all)))
        v_raw = v_blob + np.array(R.get("shift_v", [0] * len(f_all)))
        a0, a1 = area_model(u_raw, v_raw, area_all)
        area_exp_all = a0 + a1 * np.hypot(u_raw - 960.0, v_raw - 540.0)
        manifest.setdefault("puck_area_model", {})[name] = {"area_px2_at_center": round(a0, 2), "per_px_from_center": round(a1, 5)}
        boxes = net_boxes(m)
        short = SHORT.get(name, name[:8])
        k_seg = 0
        for t in sorted(set(tid.tolist()), key=lambda t: f_all[tid == t][0]):
            sel = tid == t
            f, u, v, area = f_all[sel], u_all[sel], v_all[sel], area_all[sel]
            # Defensive: drop rows that repeat the previous position exactly while the puck moves (stale frames).
            keep = np.ones(len(f), bool)
            for k in range(2, len(f)):
                if math.hypot(u[k] - u[k - 1], v[k] - v[k - 1]) < 0.05 and math.hypot(u[k - 1] - u[k - 2], v[k - 1] - v[k - 2]) > 1.0:
                    keep[k] = False
            f, u, v, area = f[keep], u[keep], v[keep], area[keep]
            sel_idx = np.nonzero(sel)[0][keep]
            for s in analyse_tracklet(m, f, u, v, area, area_exp_all[sel_idx], boxes):
                i0, i1 = s["i0"], s["i1"]
                if s["start_reason"] == "after stop":
                    continue  # began moving from rest with no detected hit: a slow stick push, not a free puck
                # The initial state comes from a fit to the first rows; it must not straddle a bounce. A bounce in
                # the first 12 rows moves the start to 4 rows after it (the shot into the board is not used).
                while s["bounces"] and s["bounces"][0] - i0 < 12:
                    i0 = s["bounces"].pop(0) + 4
                    s["start_reason"] = "after a bounce right after the hit"
                if i0 >= i1:
                    continue
                # Still accelerating? (the stick contact lasts longer than the speed-up test saw): start at the
                # window-speed peak if the speed keeps rising by > 3% within the first 15 rows.
                spw = window_speed(f, u, v)
                pk = i0 + int(np.argmax(spw[i0 : min(i0 + 15, i1)]))
                if pk > i0 and spw[pk] > 1.03 * spw[i0]:
                    i0 = pk
                    s["start_reason"] += " (moved to the speed peak: puck still accelerating)"
                    s["bounces"] = [q for q in s["bounces"] if q > i0 + 2]
                # Cut off at a reset: if the stretch ends at the board (the puck hit it as it was reset), trim the
                # rows from where it came within reach of the board (the bounce itself is not in the data).
                if s["end_reason"].startswith("track end"):
                    bdw = board_distance(m, u[i0 : i1 + 1], v[i0 : i1 + 1]) < PUCK_R_PX + WALL_MARGIN_PX
                    tail = [q for q in range(max(i1 - 12, i0), i1 + 1) if bdw[q - i0]]
                    if tail and (not s["bounces"] or s["bounces"][-1] < tail[0] - 6):
                        i1 = tail[0] - 1
                        s["end_reason"] += "; trimmed where the puck reached the board"
                if i1 - i0 < 5:
                    continue
                dur = (f[i1] - f[i0]) / m.fps
                if dur < MIN_DURATION_S:
                    continue
                k_seg += 1
                sid = f"{short}_{k_seg:02d}"
                ff = f[i0 : i1 + 1]
                x, y = m.px_to_m(u[i0 : i1 + 1], v[i0 : i1 + 1])
                tt = (ff - ff[0]) / m.fps
                nfit = min(10, len(tt))
                if s["bounces"]:
                    nfit = min(nfit, s["bounces"][0] - i0 - 2)
                x0, y0, vx0, vy0 = quad_state(tt[:nfit], x[:nfit], y[:nfit], 0.0)
                label = "net hit" if s["net_hit"] else ("wall bounce" if s["bounces"] else "glide")
                steps = int(math.ceil(tt[-1] / params["timestep"] - 1e-9))
                scen = {
                    "name": sid,
                    "description": f"MEASURED ({label}) from {name} frames {int(ff[0])}-{int(ff[-1])}; placeholder meter scale; z unknown",
                    "steps": steps,
                    "initial_state": {
                        "tick": 0,
                        "puck": {"x": x0, "y": y0, "z": 0.0, "vx": vx0, "vy": vy0, "vz": 0.0, "in_net": None},
                        "players": [],
                        "score": [0, 0],
                    },
                    "inputs": [],
                }
                (out_dir / f"{sid}.scenario.json").write_text(json.dumps(scen, indent=2) + "\n")
                lines = ["t,x,y,z"] + [f"{a!r},{round(float(b), 6)!r},{round(float(c), 6)!r},0" for a, b, c in zip(tt.tolist(), x, y)]
                (out_dir / f"{sid}.csv").write_text("\n".join(lines) + "\n")
                manifest["segments"].append({
                    "id": sid,
                    "scenario": f"{sid}.scenario.json",
                    "csv": f"{sid}.csv",
                    "clip": name,
                    "tracklet": int(t),
                    "start_frame": int(ff[0]),
                    "end_frame": int(ff[-1]),
                    "duration_s": round(float(dur), 4),
                    "rows": int(len(ff)),
                    "label": label,
                    "wall_bounce_frames": [int(f[q]) for q in s["bounces"]],
                    "net_hit": bool(s["net_hit"]),
                    "airborne": bool(s["airborne_frac"] > 0.2),
                    # The lifted-shot clip's shots leave the ice; on screen a lift shows only as a speed
                    # signature (parallax), not reliably as size. Treat that whole clip as possibly airborne.
                    "possibly_airborne": bool(s["airborne_frac"] > 0.2 or "lifted" in name),
                    "airborne_frac": round(s["airborne_frac"], 3),
                    "start_reason": s["start_reason"],
                    "end_reason": s["end_reason"],
                    "v0_m_s": round(math.hypot(vx0, vy0), 3),
                    "v0_px_per_frame": round(math.hypot(vx0, vy0) / m.m_per_px / m.fps, 3),
                })
    segs = manifest["segments"]
    labels = ["glide", "wall bounce", "net hit"]
    manifest["summary"] = {
        "segments": len(segs),
        "by_clip": {c: {lab: sum(1 for s in segs if s["clip"] == c and s["label"] == lab) for lab in labels} | {"total": sum(1 for s in segs if s["clip"] == c), "possibly_airborne": sum(1 for s in segs if s["clip"] == c and s["possibly_airborne"])} for c in sorted({s["clip"] for s in segs})},
        "by_label": {lab: sum(1 for s in segs if s["label"] == lab) for lab in labels},
        "skipped_rule": f"stretches shorter than {MIN_DURATION_S} s are skipped, and so are stretches that start from rest without a detected hit (slow stick pushes)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[segment] wrote {len(segs)} scenario/trajectory pairs + manifest.json to {out_dir}")
    for c, d in manifest["summary"]["by_clip"].items():
        print(f"[segment]   {c}: {d}")
    print(f"[segment]   by label: {manifest['summary']['by_label']}")
    return manifest
