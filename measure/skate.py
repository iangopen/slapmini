"""Session s7: skating clips with a key display, and the goal-speed clip.

    python -m measure.skate templates                 # keycap glyph templates from the reference caps (needs clips/)
    python -m measure.skate track clips/NAME.mkv ...  # registration + keys + player/puck -> results/skate/NAME.*
    python -m measure.skate qa NAME                   # overlay of the registered player path on frames
    (fits and scale: measure/skatefit.py, measure/goalscale.py)

Per clip, one decode pass: camera registration (measure/camreg.py), key display (measure/keys.py), player disc and
puck detection. Outputs in measure/results/skate/:
  NAME.frames.csv  frame,t,v_c,u_c,reg_source,reg_residual,n_lines,player_u,player_v,player_x,player_y,
                   puck_u,puck_v,puck_x,puck_y,keys   (x/y in meters at the PLACEHOLDER scale, px/m = s)
  NAME.json        summary: calibration (ice colour, boards, s), registration residuals and drift test,
                   key events, detection rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from . import keys as K
from .camreg import Registrar, board_edges, drift_test, odometry
from .geometry import REPO, load_params
from .video import iter_frames, open_clip

OUT = REPO / "measure" / "results" / "skate"
# Screen-fixed overlays inside the ice columns (1080p): key display, mouse indicator, stamina bar, banner text.
HUD_BOXES = [(755, 905, 1260, 1470), (930, 985, 1420, 1470), (995, 1050, 840, 1070)]
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
PLAYER_AREA = (60, 500)
CENTRE = (959.5, 540.5)  # where the following camera keeps the player
PUCK_AREA = (20, 160)


def in_hud(u: float, v: float, boxes=HUD_BOXES) -> bool:
    return any(y0 <= v <= y1 and x0 <= u <= x1 for y0, y1, x0, x1 in boxes)


def circle_fit(pts: np.ndarray, iters: int = 4, tol: float = 1.2) -> tuple[float, float, float]:
    """Robust algebraic circle fit (drops outline points > tol px off the circle, e.g. where the stick joins)."""
    keep = np.ones(len(pts), bool)
    cx = cy = r = 0.0
    for _ in range(iters):
        x, y = pts[keep, 0], pts[keep, 1]
        A = np.column_stack([x, y, np.ones_like(x)])
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
        d = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        new = d <= tol
        if new.sum() < 8 or (new == keep).all():
            break
        keep = new
    return float(cx), float(cy), float(r)


def player_masks(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fill, body) masks. fill: the neutral-grey disc fill (saturation < 24; ice ~53, painted lines ~33) - the seed.
    body: fill or the red rim - the disc's full extent (the rim also matches the red end board, see refine_disc)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    fill = (sat < 24) & (val > 65)
    rim = ((h < 10) | (h > 150)) & (sat > 30) & (val > 50)
    return fill.astype(np.uint8), (fill | rim).astype(np.float32)


_ANG = np.linspace(0, 2 * math.pi, 72, endpoint=False)
_RAD = np.arange(3.0, 17.0, 0.25)


def refine_disc(body: np.ndarray, u0: float, v0: float, iters: int = 3) -> tuple[float, float, float, int] | None:
    """Disc centre from its outer edge: along 72 rays the radius where the body mask (bilinear) drops below 0.5;
    circle fit with outlier rejection (rays stopped by the stick or running on into the red end board)."""
    cu, cv_ = u0, v0
    r = 9.5
    inl = 0
    for _ in range(iters):
        X = cu + np.cos(_ANG)[:, None] * _RAD[None, :]
        Y = cv_ + np.sin(_ANG)[:, None] * _RAD[None, :]
        vals = cv2.remap(body, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR, borderValue=0.0)
        pts = []
        for k in range(len(_ANG)):
            row = vals[k]
            inside = np.nonzero(row >= 0.5)[0]
            if len(inside) == 0 or inside[0] > 8:
                continue
            j = inside[0]
            while j + 1 < len(row) and row[j + 1] >= 0.5:
                j += 1
            if j + 1 >= len(row):
                continue  # runs off the window: board or another body
            # subpixel crossing between samples j and j+1
            a, b = row[j], row[j + 1]
            rr = _RAD[j] + (a - 0.5) / (a - b) * (_RAD[1] - _RAD[0]) if a != b else _RAD[j]
            pts.append((cu + math.cos(_ANG[k]) * rr, cv_ + math.sin(_ANG[k]) * rr))
        if len(pts) < 20:
            return None
        P = np.array(pts)
        keep = np.ones(len(P), bool)
        for _ in range(5):
            x, y = P[keep, 0], P[keep, 1]
            A = np.column_stack([x, y, np.ones_like(x)])
            sol = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
            cx, cy = sol[0] / 2, sol[1] / 2
            rf = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
            d = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - rf)
            new = d <= max(0.8, 2.5 * np.median(d[keep]))
            if new.sum() < 20 or (new == keep).all():
                break
            keep = new
        cu, cv_, r, inl = float(cx), float(cy), float(rf), int(keep.sum())
    return cu, cv_, r, inl


def player_candidates(frame: np.ndarray, ice: np.ndarray, board: tuple[float, float]) -> list[dict]:
    """Grey fill blobs of player size on the ice as seeds, centre and radius from refine_disc."""
    a, b = int(board[0]) + 2, int(board[1]) - 1
    sub = frame[:, a:b]
    fill, body = player_masks(sub)
    m = cv2.morphologyEx(fill, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if not (PLAYER_AREA[0] <= area <= PLAYER_AREA[1]) or min(w, h) / max(w, h) < 0.45 or not (8 <= max(w, h) <= 30):
            continue
        ref = refine_disc(body, float(cen[k][0]), float(cen[k][1]))
        if ref is None:
            continue
        cu, cv_, rad, inl = ref
        if not (7.0 <= rad <= 13.0) or inl < 30:
            continue
        u, v = cu + a, cv_
        if in_hud(u, v):
            continue
        out.append({"u": u, "v": v, "area": int(area), "r": rad, "inliers": inl})
    return out


def puck_candidates(frame: np.ndarray, board: tuple[float, float]) -> list[dict]:
    a, b = int(board[0]) + 2, int(board[1]) - 1
    hsv = cv2.cvtColor(frame[:, a:b], cv2.COLOR_BGR2HSV)
    g = ((hsv[..., 0] > 35) & (hsv[..., 0] < 85) & (hsv[..., 1] > 120) & (hsv[..., 2] > 120)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(g)
    out = []
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if PUCK_AREA[0] <= area <= PUCK_AREA[1] and max(w, h) <= 16:
            u, v = float(cen[k][0] + a), float(cen[k][1])
            if not in_hud(u, v):
                out.append({"u": u, "v": v, "area": int(area)})
    return out


def in_net(x: float, y: float, p: dict, margin: float = 1.2) -> bool:
    """World point inside either net box (the nets' grey frames look like a body)."""
    return p["net_x"] - margin <= abs(x) <= p["net_x"] + p["net_depth"] + margin and abs(y) <= p["net_width"] / 2 + margin


def behind_net(x: float, y: float, p: dict) -> bool:
    """On or beside a net's back/side mesh (a static green ornament sits on the mesh): past the goal line and
    either outside the mouth width or beyond the net's depth (minus 0.3 m)."""
    return abs(x) > p["net_x"] and (abs(y) > p["net_width"] / 2 or abs(x) > p["net_x"] + p["net_depth"] - 0.3)


def inside_rink(x: float, y: float, p: dict, margin: float = 0.0) -> bool:
    """World point inside the rounded-rectangle rink (shrunk by margin, m)."""
    rc = p["rink_corner_radius"]
    hx, hy = p["rink_length"] / 2 - rc, p["rink_width"] / 2 - rc
    dx, dy = max(abs(x) - hx, 0.0), max(abs(y) - hy, 0.0)
    return math.hypot(dx, dy) <= rc - margin


def pick(cands: list[dict], prev: tuple[float, float] | None, gate: float) -> dict | None:
    if not cands:
        return None
    if prev is None:
        return max(cands, key=lambda c: c["area"])
    c = min(cands, key=lambda c: math.hypot(c["u"] - prev[0], c["v"] - prev[1]))
    return c if math.hypot(c["u"] - prev[0], c["v"] - prev[1]) <= gate else None


def track_clip(path: Path, out_dir: Path = OUT, read_keys: bool = True) -> dict:
    path = Path(path)
    name = path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    _, info = open_clip(path)
    sample = [f for _, f in iter_frames(path, every=max(1, info.frame_count // 60))]
    R = Registrar()
    R.calibrate(sample)
    templates = K.load_templates() if read_keys else None
    P = load_params()
    tracker = K.KeyTracker()
    states, rows, odo_dv = [], [], []
    prev_frame = None
    p_prev = q_prev = None
    p_world_prev = q_world_prev = None
    for i, f in iter_frames(path):
        r = R.add(i, f)
        if prev_frame is not None:
            odo_dv.append(odometry(prev_frame, f, R.cols)[1])
        else:
            odo_dv.append(float("nan"))
        prev_frame = f
        st = tracker.update(K.read_caps(f, templates)) if read_keys else {}
        states.append(st)
        # Gates in screen px: the player moves <= ~6 px/frame on screen (camera follows), teleports re-acquire.
        pc = player_candidates(f, R.ice, R.board)
        if r.v_c == r.v_c:
            pc = [c for c in pc if inside_rink(*R.to_world(r, c["u"], c["v"]), P, 0.3) and not in_net(*R.to_world(r, c["u"], c["v"]), P)]
        # The camera keeps the player at the screen centre except at the rink ends: (re)acquire the candidate nearest
        # the centre, then follow the nearest one (static look-alikes: corner mesh structures, the other net).
        p = (pick(pc, p_prev, 25.0) if p_prev else None) or pick(pc, CENTRE, 120.0)
        qc = puck_candidates(f, R.board)
        # exclude the green center dot (world 0, 0)
        if r.v_c == r.v_c:
            qc = [c for c in qc if math.hypot(c["u"] - r.u_c, c["v"] - r.v_c) > 8 and not behind_net(*R.to_world(r, c["u"], c["v"]), P)]
        q = pick(qc, q_prev, 60.0) or (pick(qc, None, 0) if len(qc) == 1 else None)
        row = {"frame": i, "t": round(i / info.fps, 6), "v_c": r.v_c, "u_c": r.u_c, "reg_source": r.source,
               "reg_residual": r.residual, "n_lines": r.n_lines}
        for key, obj in (("player", p), ("puck", q)):
            if obj and r.v_c == r.v_c:
                x, y = R.to_world(r, obj["u"], obj["v"])
                row.update({f"{key}_u": round(obj["u"], 3), f"{key}_v": round(obj["v"], 3), f"{key}_x": round(x, 5), f"{key}_y": round(y, 5)})
            else:
                row.update({f"{key}_u": "", f"{key}_v": "", f"{key}_x": "", f"{key}_y": ""})
        row["keys"] = "+".join(sorted(k for k, v in st.items() if v))
        rows.append(row)
        p_prev = (p["u"], p["v"]) if p else p_prev
        q_prev = (q["u"], q["v"]) if q else q_prev
    with open(out_dir / f"{name}.frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and v == v else ("" if isinstance(v, float) else v)) for k, v in row.items()})
    reg = R.out
    anchored = [x for x in reg if x.source == "lines"]
    res = np.array([x.residual for x in anchored]) if anchored else np.array([np.nan])
    v_anchor = np.array([x.v_c if x.source == "lines" else np.nan for x in reg])
    wdev = np.array([abs(x.width - R.width) for x in reg])
    summary = {
        "clip": name, "fps": info.fps, "frames": len(rows), "frame_size": [info.width, info.height],
        "calibration": {"ice_bgr": R.ice.tolist(), "board_width_px": R.width, "s_px_per_m": R.s,
                        "board_width_dev_px": {"median": round(float(np.nanmedian(wdev)), 3), "p95": round(float(np.nanpercentile(wdev, 95)), 3),
                                               "max": round(float(np.nanmax(wdev)), 3)} if np.isfinite(wdev).any() else None,
                        "u_c_range_px": [round(float(min(x.u_c for x in reg)), 2), round(float(max(x.u_c for x in reg)), 2)]},
        "scale_source": "px per meter from the painted line spacing; meters at the PLACEHOLDER scale (params.json, rink_length 60 m)",
        "registration": {
            "anchored_frames": len(anchored), "odometry_frames": sum(x.source == "odometry" for x in reg),
            "unregistered_frames": sum(x.source == "none" for x in reg),
            "residual_px": {"median": round(float(np.nanmedian(res)), 3), "p95": round(float(np.nanpercentile(res, 95)), 3),
                             "max": round(float(np.nanmax(res)), 3)},
            "lines_per_anchored_frame": {str(k): int(v) for k, v in enumerate(np.bincount([x.n_lines for x in anchored])) if v} if anchored else {},
            "odometry_drift_vs_anchors": drift_test(v_anchor, np.array(odo_dv)),
        },
        "detection": {"player_frames": sum(1 for r in rows if r["player_u"] != ""), "puck_frames": sum(1 for r in rows if r["puck_u"] != "")},
    }
    if read_keys:
        summary["key_events"] = K.events_from_states(states)
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=1))
    return summary


def qa_path(name: str, frames: list[int] | None = None, out_dir: Path = OUT, span_s: float = 3.0) -> Path:
    """The registered player path (world coordinates, +-span_s around each frame) projected back into sampled frames
    through each frame's own registration: if registration and detection are right, the path passes through the
    disc and lies still on the ice while the camera pans."""
    summ = json.loads((out_dir / f"{name}.json").read_text())
    s = summ["calibration"]["s_px_per_m"]
    fps = summ["fps"]
    with open(out_dir / f"{name}.frames.csv") as fh:
        rows = list(csv.DictReader(fh))
    have = [int(r["frame"]) for r in rows if r["player_x"] and r["v_c"]]
    if frames is None:
        frames = [have[int(k)] for k in np.linspace(len(have) * 0.15, len(have) * 0.85, 4)]
    world = {int(r["frame"]): (float(r["player_x"]), float(r["player_y"])) for r in rows if r["player_x"]}
    reg = {int(r["frame"]): (float(r["u_c"]), float(r["v_c"])) for r in rows if r["v_c"]}
    want = set(frames)
    tiles = []
    for k, f in iter_frames(REPO / "clips" / f"{name}.mkv"):
        if k not in want:
            continue
        uc, vc = reg[k]
        img = f.copy()
        pts = [(uc - y * s, vc - x * s) for j, (x, y) in sorted(world.items()) if abs(j - k) <= span_s * fps]
        for a, b in zip(pts[:-1], pts[1:]):
            if math.hypot(b[0] - a[0], b[1] - a[1]) < 30:
                cv2.line(img, (int(a[0] * 4), int(a[1] * 4)), (int(b[0] * 4), int(b[1] * 4)), (0, 255, 255), 1, cv2.LINE_AA, 2)
        if k in world:
            x, y = world[k]
            cv2.circle(img, (int((uc - y * s) * 4), int((vc - x * s) * 4)), 13 * 4, (0, 0, 255), 1, cv2.LINE_AA, 2)
        cv2.drawMarker(img, (int(uc), int(vc)), (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(img, f"{name} frame {k}  (path +-{span_s:.0f} s yellow, detection red, world origin magenta)", (20, 40), 0, 0.9, (0, 255, 255), 2)
        tiles.append(cv2.resize(img[:, 480:1440 + 160], (0, 0), fx=0.5, fy=0.5))
        if len(tiles) == len(frames):
            break
    sheet = np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles) - len(tiles) % 2, 2)])
    qa = out_dir / "qa"
    qa.mkdir(exist_ok=True)
    path = qa / f"{name}.path.png"
    cv2.imwrite(str(path), sheet)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m measure.skate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("templates")
    t = sub.add_parser("track")
    t.add_argument("clips", nargs="+", type=Path)
    t.add_argument("--no-keys", action="store_true")
    q = sub.add_parser("qa")
    q.add_argument("names", nargs="+")
    q.add_argument("--frames", default="")
    a = ap.parse_args(argv)
    if a.cmd == "templates":
        K.build_templates(REPO / "clips")
        print(f"wrote {K.TEMPLATES}")
    elif a.cmd == "track":
        for c in a.clips:
            s = track_clip(c, read_keys=not a.no_keys)
            print(json.dumps({k: s[k] for k in ("clip", "frames", "registration", "detection")}, indent=1))
    elif a.cmd == "qa":
        fr = [int(x) for x in a.frames.split(",")] if a.frames else None
        for n in a.names:
            print("wrote", qa_path(n, fr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
