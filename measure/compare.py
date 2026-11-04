"""`compare` command: overlay a real segment and sim replays of it (one per params file).

Left: the reference frame (camera-corrected median) cropped to the segment, with the measured ground
path (yellow) and each replay (cyan, magenta, ...). Dots every 0.25 s on every path show timing.
Right: speed against distance travelled (v vs s) for the measurement (local quadratic fits) and replays.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .geometry import RinkMap
from .holdout import CALIB
from .segment import OUT

COLS = [(255, 255, 0), (255, 0, 255), (0, 128, 255), (0, 255, 0)]


def _load_csv(p: Path) -> np.ndarray:
    return np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)


def _vs(t, x, y, half=5):
    """speed (m/s) and arc length (m) from local quadratic fits."""
    P, V = [], []
    for k in range(len(t)):
        a, b = max(k - half, 0), min(k + half, len(t) - 1)
        tt = t[a : b + 1] - t[k]
        deg = 2 if b - a >= 4 else 1
        cx, cy = np.polyfit(tt, x[a : b + 1], deg), np.polyfit(tt, y[a : b + 1], deg)
        P.append((cx[-1], cy[-1]))
        V.append(np.hypot(cx[-2], cy[-2]))
    P = np.array(P)
    return np.array(V), np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]


def run_compare(seg_id: str, params: list[Path], results: Path, labels: list[str] | None = None, out: Path | None = None) -> Path:
    man = json.loads((OUT / "manifest.json").read_text())
    seg = next(s for s in man["segments"] if s["id"] == seg_id)
    m = RinkMap.load(results / f"{seg['clip']}.map.json")
    bg = cv2.imread(str(REPO_CLIPS / f"{seg['clip']}.bg.png"))
    meas = _load_csv(OUT / seg["csv"])
    tmp = Path(tempfile.mkdtemp())
    sims = []
    for i, p in enumerate(params):
        o = tmp / f"sim{i}.csv"
        subprocess.run([str(CALIB), "replay", "--params", str(p), str(OUT / seg["scenario"]), str(o)], check=True, capture_output=True)
        sims.append(_load_csv(o))
    labels = labels or [Path(p).stem for p in params]
    paths = [("measured", meas, (0, 230, 255))] + [(labels[i], s, COLS[i % len(COLS)]) for i, s in enumerate(sims)]
    allpx = []
    img = bg.copy()
    for name, a, col in paths:
        u, v = m.m_to_px(a[:, 1], a[:, 2])
        allpx.append(np.stack([u, v], 1))
        pts = np.stack([u, v], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
        tick = np.searchsorted(a[:, 0], np.arange(0, a[-1, 0] + 1e-9, 0.25))
        for k in tick[tick < len(a)]:
            cv2.circle(img, (int(u[k]), int(v[k])), 3, col, -1, cv2.LINE_AA)
    for bf in seg["wall_bounce_frames"]:
        k = bf - seg["start_frame"]
        tt = k / m.fps
        j = int(np.argmin(np.abs(meas[:, 0] - tt)))
        u, v = m.m_to_px(meas[j, 1], meas[j, 2])
        cv2.circle(img, (int(u), int(v)), 10, (0, 0, 255), 1, cv2.LINE_AA)
    allpx = np.concatenate(allpx)
    x0, y0 = np.floor(allpx.min(0) - 40).astype(int)
    x1, y1 = np.ceil(allpx.max(0) + 40).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    crop = img[y0:y1, x0:x1]
    sc = max(1.0, 520 / max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    # v vs s panel
    H, W = crop.shape[0], 520
    panel = np.full((H, W, 3), 25, np.uint8)
    curves = []
    for name, a, col in paths:
        if name == "measured":
            V, S = _vs(a[:, 0], a[:, 1], a[:, 2])
        else:
            d = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 2]))
            S = np.r_[0, np.cumsum(d)]
            V = np.r_[d[0], d] / np.r_[a[1, 0] - a[0, 0], np.diff(a[:, 0])]
        curves.append((name, S, V, col))
    smax = max(c[1].max() for c in curves) or 1
    vmax = max(c[2].max() for c in curves) or 1
    for name, S, V, col in curves:
        pts = np.stack([20 + S / smax * (W - 40), H - 30 - V / vmax * (H - 60)], 1).astype(np.int32)
        cv2.polylines(panel, [pts], False, col, 1, cv2.LINE_AA)
    cv2.putText(panel, f"speed vs distance (0..{vmax:.1f} m/s, 0..{smax:.1f} m)", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    sheet = np.hstack([crop, panel])
    yy = 40
    for name, a, col in paths:
        cv2.putText(sheet, name, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(sheet, name, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        yy += 18
    cv2.putText(sheet, f"{seg_id} {seg['label']} {seg['clip'][:14]} f{seg['start_frame']}-{seg['end_frame']} (dots every 0.25 s, red rings = detected bounces)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    out = out or results / "qa" / f"compare_{seg_id}.png"
    cv2.imwrite(str(out), sheet)
    print(f"[compare] wrote {out}")
    return out


from .geometry import REPO  # noqa: E402

REPO_CLIPS = REPO / "clips"
