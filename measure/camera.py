"""Camera motion: measure each frame's shift against the reference (median) frame and undo it.

Landmarks: 81x81 patches centered on the four end-zone faceoff dots (dot + cross marks), painted on
the ice, so they move exactly like the puck. (Board-corner patches were tried first; the boards stand
above the ice, and their parallax under a panning perspective camera left ~1.3 px of disagreement.)
A frame's shift is the median of the landmark shifts; if the camera only pans, all of them agree. Their spread around the median is what a pure translation cannot fix; it is reported as the
frame's residual, and frames whose residual exceeds the limit (or with < 3 landmarks found) are dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

HALF = 40  # landmark patch half-size (81x81)
SEARCH = 32  # px searched around the reference position
MIN_SCORE = 0.5
LIMIT_PX = 2.0


def landmark_points(m) -> list[tuple[float, float]]:
    """Faceoff dot centers from the map's circle fits."""
    fits = m.extra["shape"]["circles"]["fits"]
    pts = [tuple(v["center_px"]) for k, v in fits.items() if k.startswith("faceoff") and v]
    if len(pts) < 3:
        raise RuntimeError("map has fewer than 3 faceoff circles; cannot track the camera")
    return pts


def _para(a, b, c):
    d = a - 2 * b + c
    return 0.0 if d == 0 else 0.5 * (a - c) / d


@dataclass
class Shift:
    du: float  # frame content is displaced by (du, dv) relative to the reference
    dv: float
    residual: float  # max landmark deviation from (du, dv), px
    found: int

    @property
    def ok(self) -> bool:
        return self.found >= 3 and self.residual <= LIMIT_PX

    @property
    def magnitude(self) -> float:
        return math.hypot(self.du, self.dv)


class Camera:
    def __init__(self, ref_bgr: np.ndarray, points: list[tuple[float, float]]):
        g = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        self.H, self.W = g.shape
        self.points = []
        self.templates = []
        for u, v in points:
            cu, cv_ = int(round(u)), int(round(v))
            if cv_ - HALF < 0 or cu - HALF < 0 or cv_ + HALF >= self.H or cu + HALF >= self.W:
                continue
            self.points.append((cu, cv_))
            self.templates.append(g[cv_ - HALF : cv_ + HALF + 1, cu - HALF : cu + HALF + 1].copy())
        if len(self.points) < 3:
            raise RuntimeError("fewer than 3 camera landmarks fit in the frame")

    def _match(self, gray_win: np.ndarray, x0: int, y0: int, k: int):
        cu, cv_ = self.points[k]
        res = cv2.matchTemplate(gray_win, self.templates[k], cv2.TM_CCOEFF_NORMED)
        _, score, _, (px, py) = cv2.minMaxLoc(res)
        sx = _para(res[py, px - 1], res[py, px], res[py, px + 1]) if 0 < px < res.shape[1] - 1 else 0.0
        sy = _para(res[py - 1, px], res[py, px], res[py + 1, px]) if 0 < py < res.shape[0] - 1 else 0.0
        # Top-left of the template at zero shift, in window coordinates:
        zx, zy = cu - HALF - x0, cv_ - HALF - y0
        return px + sx - zx, py + sy - zy, float(score)

    def landmark_shifts(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float] | None]:
        out = []
        for k, (cu, cv_) in enumerate(self.points):
            x0, y0 = max(cu - HALF - SEARCH, 0), max(cv_ - HALF - SEARCH, 0)
            x1, y1 = min(cu + HALF + SEARCH + 1, self.W), min(cv_ + HALF + SEARCH + 1, self.H)
            win = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            du, dv, sc = self._match(win, x0, y0, k)
            out.append((du, dv, sc) if sc >= MIN_SCORE else None)
        return out

    def measure(self, frame_bgr: np.ndarray) -> Shift:
        found = [s for s in self.landmark_shifts(frame_bgr) if s is not None]
        if not found:
            return Shift(0.0, 0.0, float("inf"), 0)
        du = float(np.median([s[0] for s in found]))
        dv = float(np.median([s[1] for s in found]))
        res = max(math.hypot(s[0] - du, s[1] - dv) for s in found)
        return Shift(du, dv, res, len(found))

    @staticmethod
    def undo(frame_bgr: np.ndarray, s: Shift) -> np.ndarray:
        """Shift the frame back onto the reference (subpixel, bilinear)."""
        if abs(s.du) < 0.05 and abs(s.dv) < 0.05:
            return frame_bgr
        M = np.float32([[1, 0, -s.du], [0, 1, -s.dv]])
        return cv2.warpAffine(frame_bgr, M, (frame_bgr.shape[1], frame_bgr.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
