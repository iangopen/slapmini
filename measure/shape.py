"""Rink shape from the reference frame: circle roundness (is the camera top-down?), lines, nets.

Circles: the center circle and the four end-zone faceoff circles are thin grey rings. For each, rays
are cast from the center; the ring's brightness peak along each ray gives an edge point (subpixel),
and an ellipse is fitted with outlier rejection (rays that hit hash marks or the center line are
dropped). A top-down camera draws them round: vertical/horizontal extent ratio 1.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

ROUND_TOL = 0.02  # |v/h - 1| below this counts as round


def _hsv(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def faceoff_dots(bgr: np.ndarray, ice: np.ndarray, origin, b: dict) -> list[tuple[float, float]]:
    """Centers of the four end-zone faceoff dots (small saturated cyan blobs)."""
    hsv = _hsv(bgr)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    inner = cv2.erode(ice, np.ones((31, 31), np.uint8)) > 0
    m = ((h >= 85) & (h <= 105) & (s > 100) & (v > 120) & inner).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    L, W = b["bottom"] - b["top"], b["right"] - b["left"]
    dots = [tuple(cen[i]) for i in range(1, n)
            if 6 <= st[i][4] <= 150 and abs(cen[i][0] - origin[0]) > 0.1 * W and abs(cen[i][1] - origin[1]) > 0.1 * L]
    return dots


def _sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return cv2.remap(img, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR)


def fit_ring(bgr: np.ndarray, center, r_min=30.0, r_max=220.0, exclude_r=12.0) -> dict | None:
    """Fit an ellipse to the grey ring around `center`. Returns ellipse stats or None."""
    hsv = _hsv(bgr).astype(np.float32)
    val, sat = hsv[..., 2], hsv[..., 1]
    cx, cy = center
    ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    # 1) radius: brightness peak of the angular mean, among low-saturation pixels
    rs = np.arange(max(r_min, exclude_r), r_max, 0.5)
    X = cx + np.outer(rs, np.cos(ang))
    Y = cy + np.outer(rs, np.sin(ang))
    Vv = _sample(val, X, Y)
    Ss = _sample(sat, X, Y)
    grey = np.where(Ss < 70, Vv, np.nan)
    prof = np.nanmedian(grey, axis=1)
    if np.all(np.isnan(prof)):
        return None
    base = np.nanmedian(prof)
    k = int(np.nanargmax(prof))
    if prof[k] - base < 8:
        return None
    r0 = rs[k]
    # 2) per-ray subpixel peak in r0 +- 8 px
    rr = np.arange(r0 - 8, r0 + 8.01, 0.25)
    X = cx + np.outer(rr, np.cos(ang))
    Y = cy + np.outer(rr, np.sin(ang))
    Vv = _sample(val, X, Y)
    Ss = _sample(sat, X, Y)
    pts = []
    for j in range(len(ang)):
        col = Vv[:, j]
        i = int(np.argmax(col))
        if col[i] - base < 8 or Ss[i, j] > 70 or i < 2 or i > len(rr) - 3:
            continue
        w = np.clip(col - (base + 0.5 * (col[i] - base)), 0, None)
        lo, hi = i, i
        while lo > 0 and w[lo - 1] > 0:
            lo -= 1
        while hi < len(w) - 1 and w[hi + 1] > 0:
            hi += 1
        if hi - lo > 24:  # a radial streak (hash mark), not a ring crossing
            continue
        r = float((rr[lo : hi + 1] * w[lo : hi + 1]).sum() / w[lo : hi + 1].sum())
        pts.append((cx + r * math.cos(ang[j]), cy + r * math.sin(ang[j])))
    if len(pts) < 60:
        return None
    P = np.array(pts, np.float32)
    keep = np.ones(len(P), bool)
    for _ in range(3):
        (ex, ey), (w_, h_), th = cv2.fitEllipse(P[keep])
        # radial residual against the fitted ellipse (approximate via normalized radius)
        t = math.radians(th)
        dx, dy = P[:, 0] - ex, P[:, 1] - ey
        a_ = dx * math.cos(t) + dy * math.sin(t)
        b_ = -dx * math.sin(t) + dy * math.cos(t)
        rho = np.sqrt((a_ / (w_ / 2)) ** 2 + (b_ / (h_ / 2)) ** 2)
        resid = np.abs(rho - 1) * (w_ + h_) / 4
        keep = resid < max(1.0, 2.5 * np.median(resid[keep]))
    (ex, ey), (w_, h_), th = cv2.fitEllipse(P[keep])
    t = math.radians(th)
    half_h = math.sqrt((w_ / 2 * math.cos(t)) ** 2 + (h_ / 2 * math.sin(t)) ** 2)
    half_v = math.sqrt((w_ / 2 * math.sin(t)) ** 2 + (h_ / 2 * math.cos(t)) ** 2)
    return {
        "center_px": [round(float(ex), 3), round(float(ey), 3)],
        "axes_px": [round(float(w_), 3), round(float(h_), 3)],
        "angle_deg": round(float(th), 2),
        "horizontal_diameter_px": round(2 * half_h, 3),
        "vertical_diameter_px": round(2 * half_v, 3),
        "v_over_h": round(half_v / half_h, 5),
        "minor_over_major": round(min(w_, h_) / max(w_, h_), 5),
        "points_used": int(keep.sum()),
        "points_total": int(len(P)),
        "fit_rms_px": round(float(np.sqrt(np.mean(resid[keep] ** 2))), 3),
    }


def circles(bgr: np.ndarray, ice: np.ndarray, origin, b: dict) -> dict:
    out = {"center": fit_ring(bgr, origin)}
    for i, d in enumerate(sorted(faceoff_dots(bgr, ice, origin, b), key=lambda p: (p[1], p[0]))):
        out[f"faceoff_{i}"] = fit_ring(bgr, d)
    fits = {k: v for k, v in out.items() if v}
    ratios = [v["v_over_h"] for v in fits.values()]
    summary = {
        "n": len(fits),
        "v_over_h_mean": round(float(np.mean(ratios)), 5) if ratios else None,
        "v_over_h_min": round(float(np.min(ratios)), 5) if ratios else None,
        "v_over_h_max": round(float(np.max(ratios)), 5) if ratios else None,
        "round_tol": ROUND_TOL,
    }
    summary["round"] = bool(ratios) and all(abs(r - 1) < ROUND_TOL for r in ratios)
    return {"fits": out, "summary": summary}


def grey_lines(bgr: np.ndarray, ice: np.ndarray, b: dict) -> list[float]:
    """Rows (subpixel) of the grey lines that cross the rink from board to board."""
    hsv = _hsv(bgr)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    m = (v >= 70) & (v <= 140) & (s < 60) & (ice > 0)
    c0, c1 = int(b["left"] + 0.15 * (b["right"] - b["left"])), int(b["right"] - 0.15 * (b["right"] - b["left"]))
    frac = m[:, c0:c1].mean(axis=1)
    rows = np.nonzero(frac > 0.6)[0]
    lines = []
    for r in rows:
        if lines and r - lines[-1][-1] <= 1:
            lines[-1].append(r)
        else:
            lines.append([r])
    return [float(np.average(g, weights=frac[g])) for g in lines]
