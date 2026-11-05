"""Stick tracker: pivot (player body centre) and stick angle per frame -> NAME.stick.csv + NAME.stick.json.

    python -m measure.stick track clips/NAME.mkv [...]   # angle per frame
    python -m measure.stick qa clips/NAME.mkv --frames a,b,c
    python -m measure.stick scale clips/NAME.mkv         # body radius, centre circle, puck: per-clip m/px
    python -m measure.stick shape                        # stick length / blade thickness (measure/stick_shape.py)
    python -m measure.stick chase                        # chase fit on the sweep clip (measure/stick_chase.py)
    python -m measure.stick params                       # params.stick.json (measure/stick_params.py)
    python -m measure.stick hits                         # hit physics (measure/stick_hits.py)

Model of the picture (rotatinganddizzy, hittingpuck; the 2026-11 clips): the player is a grey body (pure grey, S = 0)
inside a blue glow; the stick is a black shaft held along one side of the body, ending in a white blade outside it.
Body markings, shaft and blade turn together (the player model rotates with the stick). The camera follows the
player, so the pivot sits near the image centre.

Pivot: centre of an ellipse fitted to the convex hull of the grey body blob. Angle: the black (shaft) and white
(blade, markings) pixels in an annulus around the pivot are unwrapped to polar coordinates; the rotation of each frame
against a template is the peak of their circular cross-correlation (sub-bin by a parabola). The template is built from
the clip itself (frames de-rotated and averaged, two passes). Features that do not move on screen (the mouse-hint icon
next to the player, HUD) are masked out. The angle reported is the direction from the pivot to the blade (the white
blade pixels outside the body in the template), in the sim frame (+x = screen up, +y = screen left, CCW positive).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from .geometry import REPO
from .video import iter_frames, open_clip

RESULTS = REPO / "measure" / "results"
N_ANG = 720  # polar angle bins (0.5 deg)
R_IN, R_OUT = 0.25, 2.2  # annulus in body radii
STATIC_FRAC = 0.7  # a feature pixel lit in >= this fraction of frames is static on screen (icon, HUD)
MIN_CONF = 0.25
BODY_EDGE = 0.97  # body radii: inside, only black (shaft) pixels count


def body_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (s < 25) & (v >= 40) & (v <= 185)


def feature_image(hsv: np.ndarray) -> np.ndarray:
    """Black shaft = 2, white blade/markings = 1, everything else 0 (float32)."""
    s, v = hsv[..., 1], hsv[..., 2]
    return (2.0 * (v < 45) + 1.0 * ((v > 185) & (s < 70))).astype(np.float32)  # 2 = black, 1 = white


def find_body(frame: np.ndarray, guess: tuple[float, float], search: float, r_prev: float | None = None) -> dict | None:
    """Grey body blob nearest the guess: centre (ellipse of the convex hull), radius, roundness."""
    H, W = frame.shape[:2]
    x0, y0 = int(max(guess[0] - search, 0)), int(max(guess[1] - search, 0))
    x1, y1 = int(min(guess[0] + search, W)), int(min(guess[1] + search, H))
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = body_mask(hsv).astype(np.uint8)
    k = max(3, int(search / 25) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if r_prev:  # opening with a disc ~0.3 body radii strips the shaft/blade where they touch the body
        ko = max(3, int(0.3 * r_prev) | 1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    best = None
    for i in range(1, n):
        if st[i, 4] < 30:
            continue
        d = math.hypot(cen[i][0] + x0 - guess[0], cen[i][1] + y0 - guess[1])
        score = st[i, 4] / (1 + d)
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        return None
    comp = (lab == best[1]).astype(np.uint8)
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    if len(hull) < 5:
        return None
    (cu, cv_), (d1, d2), _ = cv2.fitEllipse(hull)
    if min(d1, d2) / max(d1, d2) < 0.6:  # not a round body (the shaft along the edge leaves ~0.7) (another skin, or stray grey pixels)
        return None
    return {"u": cu + x0, "v": cv_ + y0, "r": (d1 + d2) / 4, "roundness": min(d1, d2) / max(d1, d2), "area": int(comp.sum())}


def polar(feat: np.ndarray, centre: tuple[float, float], r_body: float) -> np.ndarray:
    """(N_ANG, n_r) polar image of the annulus; rows = screen angle atan2(dv, du) from 0 to 2 pi."""
    r_max = R_OUT * r_body
    n_r = max(16, int(round(r_max)))
    p = cv2.warpPolar(feat, (n_r, N_ANG), centre, r_max, cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR)
    p = p[:, int(round(R_IN / R_OUT * n_r)) :].copy()
    # The body's markings turn with the body (facing), not with the stick: inside the body keep only the black
    # shaft; outside it, black shaft + white blade. Each radius ring minus its mean (rings carry no angle).
    r_cols = np.linspace(R_IN, R_OUT, p.shape[1] + 1)[:-1]  # body radii of the kept columns
    inside = r_cols < BODY_EDGE
    p[:, inside] = (p[:, inside] >= 1.5).astype(np.float32)
    p[:, ~inside] = (p[:, ~inside] >= 0.5).astype(np.float32)
    return p - p.mean(axis=0, keepdims=True)


def best_rotation(F: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    """Screen-angle rotation d (rad) with F(phi) ~ T(phi - d), and a normalised peak score in [0, 1]."""
    cc = np.fft.irfft(np.fft.rfft(F, axis=0) * np.conj(np.fft.rfft(T, axis=0)), n=N_ANG, axis=0).sum(axis=1)
    k = int(np.argmax(cc))
    a, b, c = cc[(k - 1) % N_ANG], cc[k], cc[(k + 1) % N_ANG]
    den = a - 2 * b + c
    sub = 0.5 * (a - c) / den if den != 0 else 0.0
    norm = math.sqrt(float((F * F).sum()) * float((T * T).sum()))
    return (k + sub) * 2 * math.pi / N_ANG, (float(b) / norm if norm > 0 else 0.0)


def roll(P: np.ndarray, d: float) -> np.ndarray:
    """Rotate a polar image by d (rad) along the angle axis (linear interpolation)."""
    x = d / (2 * math.pi) * N_ANG
    k = int(math.floor(x))
    f = x - k
    return (1 - f) * np.roll(P, k, axis=0) + f * np.roll(P, k + 1, axis=0)


def screen_to_sim(phi: float) -> float:
    """Screen angle atan2(dv, du) (v down) -> sim angle (+x = screen up, +y = screen left)."""
    return (-phi - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi


def track_stick(clip: Path, every: int = 1, max_frames: int | None = None) -> tuple[list[dict], dict]:
    _, info = open_clip(clip)
    W, H = info.width, info.height
    guess = (W / 2, H / 2)
    # Pass 0: body (pivot + radius) and the raw feature crops.
    bodies, crops = {}, {}
    search = 0.2 * H
    n_dec = 0
    rs: list[float] = []
    r_run = None
    for i, f in iter_frames(clip, every=every):
        n_dec += 1
        if max_frames and i >= max_frames:
            break
        b = find_body(f, guess, search, r_run)
        if b is None:
            continue
        bodies[i] = b
        rs.append(b["r"])
        r_run = float(np.median(rs[-51:]))
        guess = (b["u"], b["v"])
        search = max(6 * b["r"], 40)
        rc = int(R_OUT * b["r"]) + 3
        cu, cv_ = int(round(b["u"])), int(round(b["v"]))
        x0, y0 = max(cu - rc, 0), max(cv_ - rc, 0)
        sub = f[y0 : cv_ + rc + 1, x0 : cu + rc + 1]
        crops[i] = (feature_image(cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)), b["u"] - x0, b["v"] - y0, x0, y0)
    if not bodies:
        raise RuntimeError(f"{clip}: no player body found")
    r_body = float(np.median([b["r"] for b in bodies.values()]))
    # Static-on-screen mask (in screen pixels): features lit in most frames.
    acc = np.zeros((H, W), np.float32)
    for i, (F, cu, cv_, x0, y0) in crops.items():
        acc[y0 : y0 + F.shape[0], x0 : x0 + F.shape[1]] += F > 0
    static = acc >= STATIC_FRAC * len(crops)
    static = cv2.dilate(static.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    P = {}
    for i, (F, cu, cv_, x0, y0) in crops.items():
        F = F * (~static[y0 : y0 + F.shape[0], x0 : x0 + F.shape[1]])
        P[i] = polar(F, (cu, cv_), r_body)
    frames = sorted(P)
    if len(frames) < 10:
        raise RuntimeError(f"{clip}: too few frames with a grey player body ({len(frames)}): this tracker expects the grey skin")
    # Template: first frame with features, then two passes of de-rotate-and-average.
    T = P[frames[0]].copy()
    rot = {}
    for _ in range(3):
        rot = {i: best_rotation(P[i], T) for i in frames}
        good = [i for i in frames if rot[i][1] >= MIN_CONF]
        T = np.mean([roll(P[i], -rot[i][0]) for i in good], axis=0)
    rot = {i: best_rotation(P[i], T) for i in frames}
    # Blade direction in the template: white/black feature mass beyond the body edge.
    n_r = T.shape[1]
    r_cols = np.linspace(R_IN, R_OUT, n_r + 1)[:-1] * r_body
    outside = np.clip(T[:, r_cols > 1.08 * r_body], 0, None).sum(axis=1)
    ang = np.arange(N_ANG) * 2 * math.pi / N_ANG
    phi_tip = math.atan2(float((outside * np.sin(ang)).sum()), float((outside * np.cos(ang)).sum()))
    # Tip radius: outermost radius where the template still has stick features near phi_tip.
    k_tip = int(round(phi_tip / (2 * math.pi) * N_ANG)) % N_ANG
    sector = T[[(k_tip + d) % N_ANG for d in range(-30, 31)], :].max(axis=0)
    lit = np.where(sector > 0.3)[0]
    r_tip = float(r_cols[lit.max()]) if len(lit) else float("nan")
    rows = []
    for i in frames:
        d, score = rot[i]
        b = bodies[i]
        phi = phi_tip + d
        rows.append({"frame": i, "t": i / info.fps, "angle": screen_to_sim(phi), "phi_screen": (phi + math.pi) % (2 * math.pi) - math.pi,
                     "pivot_u": b["u"], "pivot_v": b["v"], "body_r": b["r"], "conf": score, "ok": score >= MIN_CONF})
    # Duplicated capture frames: the picture repeats exactly (same angle and pivot) right after a real step.
    # A hard stop also gives one zero step; a duplicate is followed by a catch-up step (~2x the step before it).
    dups = []
    step = lambda x, y: abs(math.remainder(float(y["angle"]) - float(x["angle"]), 2 * math.pi))  # noqa: E731
    for k in range(2, len(rows) - 1):
        a, b, c, d = rows[k - 2], rows[k - 1], rows[k], rows[k + 1]
        if c["frame"] - b["frame"] == every and step(b, c) < 1e-9 and c["pivot_u"] == b["pivot_u"] \
                and step(a, b) > 0.05 and step(c, d) > 1.4 * step(a, b):
            dups.append(c["frame"])
    dset = set(dups)
    rows = [r for r in rows if r["frame"] not in dset]
    # Unwrapped angle for analysis.
    a = np.unwrap([r["angle"] for r in rows])
    for r, x in zip(rows, a):
        r["angle_unwrapped"] = float(x)
    det = [r["frame"] for r in rows if r["ok"]]
    n = n_dec
    longest, gap_at = 0, None
    marks = [-every] + det + [n * every]  # duplicates are not gaps
    det_all = set(det) | dset
    for a_, b_ in zip(marks, marks[1:]):
        g = sum(1 for q in range(a_ + every, b_, every) if q not in det_all)
        if g > longest:
            longest, gap_at = g, (a_ + every, b_ - every)
    rep = {"frames_total": n, "frames_detected": len(det), "detection_rate": len(det) / max(n - len(dups), 1),
           "duplicate_frames_dropped": len(dups), "duplicate_frames": dups, "longest_gap_frames": longest,
           "longest_gap_range": gap_at, "body_radius_px": round(r_body, 3), "tip_radius_px": round(r_tip, 2),
           "tip_radius_body_radii": round(r_tip / r_body, 4), "static_mask_px": int(static.sum()), "every": every,
           "template_tip_screen_angle_deg": round(math.degrees(phi_tip), 2),
           "conf_median": round(float(np.median([r["conf"] for r in rows])), 3)}
    return rows, rep


def write_outputs(clip: Path, rows: list[dict], rep: dict, results: Path = RESULTS) -> None:
    name = Path(clip).stem
    lines = ["t,angle"] + [f"{r['t']!r},{round(r['angle'], 6)!r}" for r in rows if r["ok"]]
    (results / f"{name}.stick.csv").write_text("\n".join(lines) + "\n")
    rd = lambda x, k=4: round(float(x), k)  # noqa: E731
    meta = {
        "clip": name, "csv": f"{name}.stick.csv",
        "angle": "sim frame (+x = screen up, +y = screen left, CCW positive), radians, direction from the pivot to the blade; "
                 "csv has rows with conf >= %.2f only" % MIN_CONF,
        "method": "polar circular cross-correlation of black+white stick/marking pixels against a self-built template (see measure/stick.py)",
        "confidence": "normalised correlation peak (cosine similarity of the frame's polar feature image and the rotated template)",
        "summary": rep,
        "rows": {"frame": [r["frame"] for r in rows], "angle": [rd(r["angle"], 5) for r in rows],
                 "angle_unwrapped": [rd(r["angle_unwrapped"], 5) for r in rows],
                 "pivot_u": [rd(r["pivot_u"], 3) for r in rows], "pivot_v": [rd(r["pivot_v"], 3) for r in rows],
                 "body_r": [rd(r["body_r"], 3) for r in rows], "conf": [rd(r["conf"], 3) for r in rows]},
    }
    (results / f"{name}.stick.json").write_text(json.dumps(meta, separators=(",", ":")) + "\n")


def run_track(clip: Path, results: Path = RESULTS, every: int = 1, max_frames: int | None = None):
    rows, rep = track_stick(clip, every, max_frames)
    write_outputs(clip, rows, rep, results)
    print(f"[stick] {Path(clip).stem}: detection {rep['detection_rate'] * 100:.2f}% ({rep['frames_detected']}/{rep['frames_total']}), "
          f"longest gap {rep['longest_gap_frames']} frames at {rep['longest_gap_range']}; body r {rep['body_radius_px']} px, "
          f"blade tip r {rep['tip_radius_px']} px ({rep['tip_radius_body_radii']} body radii); conf median {rep['conf_median']}; "
          f"static mask {rep['static_mask_px']} px")
    return rows, rep


# ---------------------------------------------------------------- QA

def _label(img, text, org, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def run_qa(clip: Path, frames: list[int], results: Path = RESULTS, half: int | None = None) -> Path:
    name = Path(clip).stem
    meta = json.loads((results / f"{name}.stick.json").read_text())
    R = meta["rows"]
    by = {f: k for k, f in enumerate(R["frame"])}
    rb = meta["summary"]["body_radius_px"]
    rt = meta["summary"]["tip_radius_px"]
    half = half or int(2.6 * rt)
    Z = max(1, int(240 / half))
    want = set(frames)
    tiles = []
    for i, f in iter_frames(clip):
        if i > max(want):
            break
        if i not in want or i not in by:
            continue
        k = by[i]
        cu, cv_ = R["pivot_u"][k], R["pivot_v"][k]
        x0, y0 = int(cu) - half, int(cv_) - half
        crop = cv2.resize(f[y0 : y0 + 2 * half, x0 : x0 + 2 * half], None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
        a = R["angle"][k]
        du, dv = -math.sin(a), -math.cos(a)  # sim angle -> screen direction
        p0 = (int((cu - x0) * Z), int((cv_ - y0) * Z))
        p1 = (int((cu - x0 + du * rt) * Z), int((cv_ - y0 + dv * rt) * Z))
        cv2.circle(crop, p0, int(rb * Z), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(crop, p0, p1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(crop, p0, (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
        _label(crop, f"frame {i}  {math.degrees(a):+.1f} deg  conf {R['conf'][k]:.2f}", (6, 18), 0.45)
        tiles.append(crop)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r : r + cols]) for r in range(0, len(tiles), cols)])
    qa = results / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    p = qa / f"{name}.stick_sheet.png"
    cv2.imwrite(str(p), sheet)
    print(f"[stick qa] wrote {p}")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m measure.stick", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("track")
    pt.add_argument("clips", nargs="+", type=Path)
    pt.add_argument("--out-dir", type=Path, default=RESULTS)
    pq = sub.add_parser("qa")
    pq.add_argument("clip", type=Path)
    pq.add_argument("--frames", required=True)
    pq.add_argument("--out-dir", type=Path, default=RESULTS)
    ps = sub.add_parser("scale")
    ps.add_argument("clips", nargs="+", type=Path)
    sub.add_parser("shape")
    sub.add_parser("params")
    sub.add_parser("chase")
    sub.add_parser("hits")
    a = ap.parse_args(argv)
    if a.cmd == "track":
        for c in a.clips:
            run_track(c, a.out_dir)
    elif a.cmd == "qa":
        run_qa(a.clip, [int(x) for x in a.frames.split(",")], a.out_dir)
    elif a.cmd == "scale":
        from .stick_scale import run_scale

        run_scale(a.clips)
    elif a.cmd == "params":
        from .stick_params import run_params

        run_params()
    elif a.cmd == "shape":
        from .stick_shape import run_shape

        run_shape()
    elif a.cmd == "chase":
        from .stick_chase import run_chase

        run_chase()
    elif a.cmd == "hits":
        from .stick_hits import run_hits

        run_hits()
    return 0


if __name__ == "__main__":
    sys.exit(main())
