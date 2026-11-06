"""Stick silhouette in the player's local frame and capsule-chain fits (`python -m measure.stick_outline --clips DIR`).

Source: rotatinganddizzy (the zoomed clip, body 49.7 px). Every usable frame is de-rotated by the tracked stick angle and
centred on the pivot, so local +X is the tracked pivot -> blade direction (the sim's stick_angle) and +Y is 90 deg CCW in
the sim frame. Only numbers leave this file; no pixels are written to the repo.

Silhouette: black shaft (V < 45) or white blade (V > 185, S < 70). The white body markings turn with the stick, so
white counts only outside the body.

Candidates: chains of 1, 2, 3 capsules (a polyline with a half-width per capsule) fitted to the mean mask of the training
frames (blocks of consecutive frames, leave one block out), scored on the held-out block's mask: IoU, and the RMS distance
from held-out silhouette edge pixels to the chain outline.

Height: the blade is near the ice and the hands near the body top, so a point at zoomed px radius r maps to the ground by
r (1 - x), x = h / H_zoom: x_tip from stick_shape.json (0.101) for the part outside the body. The shaft inside the body
never meets the puck (the body keeps it out), so its height only moves the drawing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from .geometry import REPO, load_params
from .video import iter_frames

RESULTS = REPO / "measure" / "results"
OUT = RESULTS / "stick_shape.json"
CLIP = "rotatinganddizzy"
HALF = 110  # local crop half-size, px
MAX_OMEGA = 3.0  # rad/s: slow frames only (no motion blur, no lean)
BLOCKS = 5


def local_masks(clip: Path, every: int = 2, limit: int = 500) -> tuple[np.ndarray, np.ndarray, float]:
    S = json.loads((RESULTS / f"{CLIP}.stick.json").read_text())
    R = S["rows"]
    ang = np.array(R["angle_unwrapped"], float)
    fr = np.array(R["frame"])
    w = np.gradient(ang) * 60
    ok = {int(f): k for k, f in enumerate(fr) if R["conf"][k] > 0.7 and abs(w[k]) < MAX_OMEGA}
    rb = float(S["summary"]["body_radius_px"])
    masks, frames = [], []
    for i, f in iter_frames(clip, every=every):
        if i not in ok:
            continue
        k = ok[i]
        cu, cv_ = R["pivot_u"][k], R["pivot_v"][k]
        phi = math.degrees(-R["angle"][k] - math.pi / 2)  # screen angle of the blade
        M = cv2.getRotationMatrix2D((cu, cv_), phi, 1.0)  # blade to screen +u
        M[0, 2] += HALF - cu
        M[1, 2] += HALF - cv_
        rot = cv2.warpAffine(f, M, (2 * HALF, 2 * HALF), flags=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(rot, cv2.COLOR_BGR2HSV)
        s, v = hsv[..., 1], hsv[..., 2]
        ys, xs = np.mgrid[0 : 2 * HALF, 0 : 2 * HALF]
        rr = np.hypot(xs - HALF, ys - HALF)
        black = v < 45
        white = (v > 185) & (s < 70) & (rr > 0.97 * rb)
        masks.append((black | white) & (rr < 2.1 * rb))
        frames.append(i)
        if len(masks) >= limit:
            break
    # Local image: column = +X; row down = screen v after rotation = local -Y flipped (handled in `to_local`).
    return np.array(masks, dtype=bool), np.array(frames), rb


def raster(pts: np.ndarray, rad: np.ndarray) -> np.ndarray:
    """Anti-aliased occupancy of a capsule chain; pts in image px (x right, y down)."""
    ys, xs = np.mgrid[0 : 2 * HALF, 0 : 2 * HALF].astype(np.float32)
    sd = np.full(xs.shape, 1e9, np.float32)
    for a, b, r in zip(pts[:-1], pts[1:], rad):
        e = b - a
        t = np.clip(((xs - a[0]) * e[0] + (ys - a[1]) * e[1]) / max(float(e @ e), 1e-6), 0, 1)
        sd = np.minimum(sd, np.hypot(xs - a[0] - t * e[0], ys - a[1] - t * e[1]) - r)
    return np.clip(0.5 - sd, 0, 1)


def unpack(q: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    return q[: 2 * (n + 1)].reshape(n + 1, 2), np.abs(q[2 * (n + 1) :])


def fit_chain(target: np.ndarray, n: int, starts: list[np.ndarray]) -> np.ndarray:
    def loss(q):
        pts, rad = unpack(q, n)
        return float(((raster(pts, rad) - target) ** 2).sum())

    best = None
    for q0 in starts:
        res = minimize(loss, q0, method="Powell", options={"xtol": 0.05, "ftol": 1e-4, "maxiter": 60})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def clean(mask: np.ndarray) -> np.ndarray:
    """Largest component after closing (the blade carries small dark decals; stray marking pixels sit near the rim)."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return lab == 1 + int(np.argmax(st[1:, 4]))


# Hand end, shaft kink, blade heel, toe, read off the mean mask: starting points only, the fit moves them.
SEED = np.array([[-8.75, 48.75], [28.75, 13.75], [51.0, 0.0], [85.0, -2.0]])


def initial_chains(target: np.ndarray, n: int) -> list[np.ndarray]:
    picks = {1: [[0, 3], [2, 3]], 2: [[0, 1, 3], [0, 2, 3]], 3: [[0, 1, 2, 3]]}[n]
    img = lambda p: np.c_[p[:, 0] + HALF, HALF - p[:, 1]]
    return [np.r_[img(SEED[k]).ravel(), np.full(n, 5.0)] for k in picks]


def edge_rms(pts: np.ndarray, rad: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(np.uint8)
    edge = m - cv2.erode(m, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(edge)
    sd = np.full(xs.shape, 1e9)
    for a, b, r in zip(pts[:-1], pts[1:], rad):
        e = b - a
        t = np.clip(((xs - a[0]) * e[0] + (ys - a[1]) * e[1]) / max(float(e @ e), 1e-6), 0, 1)
        sd = np.minimum(sd, np.hypot(xs - a[0] - t * e[0], ys - a[1] - t * e[1]) - r)
    return float(np.sqrt(np.mean(sd**2)))


def to_local(pts: np.ndarray) -> np.ndarray:
    """Image px -> local px (X along the blade direction, Y = 90 deg CCW in the sim frame = image up, since the sim frame
    as seen on screen is a rotation, not a mirror)."""
    return np.c_[pts[:, 0] - HALF, HALF - pts[:, 1]]


def ground_chain(c3: dict, rink_length: float) -> dict:
    """Zoomed px -> ground meters, point by point: x falls from the body top at the hand to the blade value at the heel."""
    sr = json.loads((RESULTS / "size_ratios.json").read_text())
    tip = json.loads(OUT.read_text())["tip_half_max"]["tip_height_over_zoomed_camera_height"]
    x_hand = sr["height"]["body_x"]
    k = rink_length / sr["zoom"]["rink_length_px_zoomed"]
    P = np.array(c3["points_px"])
    seg = np.hypot(*np.diff(P, axis=0).T)
    frac = np.minimum(np.r_[0, np.cumsum(seg)] / seg[:2].sum(), 1.0)
    x = x_hand + (tip - x_hand) * frac
    G = P * ((1 - x) * k)[:, None]
    rad = np.array(c3["radii_px"]) * (1 - (x[:-1] + x[1:]) / 2) * k
    names = ["hand", "kink", "heel", "toe"]
    out = {f"stick_{n}_{a}": round(float(G[i, j]), 4) for i, n in enumerate(names) for j, a in enumerate("xy")}
    out.update({"stick_shaft_radius": round(float(rad[0]), 4), "stick_heel_radius": round(float(rad[1]), 4), "stick_thickness": round(float(rad[2]), 4),
                "stick_inner": round(float(np.hypot(*G[2])), 4), "stick_length": round(float(np.hypot(*G[3])), 4),
                "height_x": [round(float(v), 3) for v in x], "m_per_px_zoomed_ice": k,
                "blade_angle_to_shaft_deg": round(math.degrees(math.atan2(G[3, 1] - G[2, 1], G[3, 0] - G[2, 0]) - math.atan2(G[1, 1] - G[0, 1], G[1, 0] - G[0, 0])), 1),
                "blade_length_m": round(float(np.hypot(*(G[3] - G[2]))), 4), "shaft_length_m": round(float(np.hypot(*(G[1] - G[0]))), 4),
                "heel_section_m": round(float(np.hypot(*(G[2] - G[1]))), 4)})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m measure.stick_outline")
    ap.add_argument("--clips", type=Path, default=REPO / "clips")
    ap.add_argument("--debug-png", type=Path, help="write the mean mask here (keep it outside the repo)")
    a = ap.parse_args(argv)
    masks, frames, rb = local_masks(a.clips / f"{CLIP}.mkv")
    print(f"{len(masks)} slow frames, body {rb:.1f} px")
    mean = masks.mean(axis=0).astype(np.float32)
    if a.debug_png:
        cv2.imwrite(str(a.debug_png), cv2.resize((mean * 255).astype(np.uint8), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    blocks = np.array_split(np.arange(len(masks)), BLOCKS)
    res = {}
    for n in (1, 2, 3):
        ious, rmss = [], []
        for b in blocks:
            tr = np.setdiff1d(np.arange(len(masks)), b)
            T = masks[tr].mean(axis=0).astype(np.float32)
            T = clean(T > 0.5).astype(np.float32)
            pts, rad = unpack(fit_chain(T, n, initial_chains(T, n)), n)
            H = clean(masks[b].mean(axis=0) > 0.5)
            C = raster(pts, rad) > 0.5
            ious.append(float((H & C).sum() / max((H | C).sum(), 1)))
            rmss.append(edge_rms(pts, rad, H))
        res[n] = {"heldout_iou": float(np.mean(ious)), "heldout_iou_sd": float(np.std(ious)), "heldout_edge_rms_px": float(np.mean(rmss)), "folds_iou": ious}
        print(f"chain of {n}: held-out IoU {res[n]['heldout_iou']:.3f} (sd {res[n]['heldout_iou_sd']:.3f}), edge RMS {res[n]['heldout_edge_rms_px']:.2f} px")
    T = clean(mean > 0.5).astype(np.float32)
    full = {}
    for n in (1, 2, 3):
        pts, rad = unpack(fit_chain(T, n, initial_chains(T, n)), n)
        full[n] = {"points_px": to_local(pts).round(2).tolist(), "radii_px": rad.round(2).tolist()}
        print(n, full[n])
    chain = ground_chain(full[3], load_params()["rink_length"])
    shape = json.loads(OUT.read_text()) if OUT.exists() else {}
    shape["chain"] = {**chain, "frames": len(masks), "heldout": res, "candidates_px": full}
    OUT.write_text(json.dumps(shape, indent=1) + "\n")
    print(json.dumps(chain, indent=1))


if __name__ == "__main__":
    main()
