"""Moving-camera registration for the player-following game camera (session s7 clips).

What the clips show (CLAUDE.md "Session s7"): the camera looks straight down and follows the player, keeping the
player disc at the screen centre (959.5, 540) except where it stops at the rink ends. The painted lines keep a
constant spacing everywhere in the image (center-to-blue-line 136.5 px, center-to-goal-line 386.8 px in every
frame) and the board-to-board width is constant, so the ice plane moves by a pure translation (no zoom, no tilt).

Registration, per frame, from painted ice features:
- u (across the rink, world y): the inner board edges (ice-colour mask, subpixel threshold crossing, median over
  the rows that see board to board) give the column of the rink's long axis, u_c. The side boards are always in
  view (the rink is 647 px wide in a 1920 px frame), so u is absolute in every frame; the measured board-to-board
  width against its clip median checks it.
- v (along the rink, world x): the five full-width painted lines (center line, two blue lines, two goal lines;
  world x from params.json: 0, +-blue_line_x, +-net_x) are found as peaks of the per-row median "non-ice" profile
  over the ice columns (the median ignores circles, players and the puck, which cover few columns). The lines are
  identified by their spacing (and the center line by its colour), which gives v_c = the image row of world x = 0
  in that frame: the landmark anchor. Residual = max deviation of the identified lines from one translation.
- Between anchors (fewer than 2 lines identified), v_c is carried by frame-to-frame odometry (phase correlation of
  the ice band). `drift_test` measures how odometry error grows with the number of frames since an anchor.
World position of a pixel: x = (v_c - v) / s, y = (u_c - u) / s, s = px per (placeholder) meter fitted from the
line spacing over the whole clip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.signal import find_peaks

from .geometry import load_params

LINE_HALF_MAX = 14  # px: a painted line is narrower than this (circles/boards/banners are wider)
MATCH_TOL = 3.0  # px: a detected line within this of its predicted row counts as identified
MIN_PEAK = 20.0  # profile units (max |BGR - ice| over channels, median over columns)


def line_model(params: dict | None = None) -> dict[str, float]:
    """World x (m) of the full-width painted lines, from params.json (measured shape, placeholder scale)."""
    p = params or load_params()
    return {"goal+": p["net_x"], "blue+": p["blue_line_x"], "center": 0.0, "blue-": -p["blue_line_x"], "goal-": -p["net_x"]}


def ice_colour(frame: np.ndarray, cols: tuple[int, int] | None = None) -> np.ndarray:
    """Most common colour in the central band (the bare ice), BGR."""
    h, w = frame.shape[:2]
    a, b = cols if cols else (w // 3, 2 * w // 3)
    band = frame[h // 4 : 3 * h // 4, a:b].reshape(-1, 3).astype(np.int32) // 4
    keys, counts = np.unique(band[:, 0] * 4096 + band[:, 1] * 64 + band[:, 2], return_counts=True)
    k = int(keys[np.argmax(counts)])
    return np.array([(k // 4096) * 4 + 2, ((k // 64) % 64) * 4 + 2, (k % 64) * 4 + 2], np.float32)


def ice_mask(frame: np.ndarray, ice: np.ndarray, tol: float = 14.0) -> np.ndarray:
    return (np.abs(frame.astype(np.float32) - ice).max(axis=2) < tol).astype(np.uint8)


def board_columns(frame: np.ndarray, ice: np.ndarray) -> tuple[float, float, int] | None:
    """Inner board edges (left, right) in px with subpixel precision: median over rows whose ice run is the widest."""
    m = ice_mask(frame, ice)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lefts, rights = [], []
    for y in range(0, frame.shape[0], 8):
        xs = np.nonzero(m[y])[0]
        if len(xs) < 100:
            continue
        runs = np.split(xs, np.nonzero(np.diff(xs) > 2)[0] + 1)
        r = max(runs, key=len)
        lefts.append(r[0])
        rights.append(r[-1])
    if len(lefts) < 10:
        return None
    lefts, rights = np.array(lefts), np.array(rights)
    widths = rights - lefts
    full = widths >= np.percentile(widths, 90) - 2  # rows that see board to board (no line/circle cut)
    if full.sum() < 5:
        return None
    # The ice run ends at the last ice pixel; the edge lies half a pixel further out.
    return float(np.median(lefts[full])) - 0.5, float(np.median(rights[full])) + 0.5, int(full.sum())


def board_edges(frame: np.ndarray, ice: np.ndarray, expected: float | None = None, tol: float = 14.0, step: int = 6) -> tuple[float, float, int] | None:
    """Per-frame inner board edges (left, right) with subpixel precision: on sampled rows, the ice run's ends are
    refined to where the distance-from-ice crosses `tol`; the median over the rows whose run has the board-to-board
    width (`expected`, or the most common run width; at the rink ends dark stands can look like ice and make
    some runs wider)."""
    rows = frame[::step].astype(np.float32)
    d = np.abs(rows - ice).max(axis=2)
    ok = d < tol
    L, Rr = [], []
    for k in range(ok.shape[0]):
        xs = np.nonzero(ok[k])[0]
        if len(xs) < 200:
            continue
        runs = np.split(xs, np.nonzero(np.diff(xs) > 2)[0] + 1)
        r = max(runs, key=len)
        a, b = int(r[0]), int(r[-1])
        if a < 1 or b >= d.shape[1] - 1:
            continue
        da, db = d[k, a - 1], d[k, a]
        left = (a - 1) + (da - tol) / (da - db) if da > db else a - 0.5
        ea, eb = d[k, b], d[k, b + 1]
        right = b + (tol - ea) / (eb - ea) if eb > ea else b + 0.5
        L.append(left)
        Rr.append(right)
    if len(L) < 5:
        return None
    L, Rr = np.array(L), np.array(Rr)
    w = Rr - L
    if expected is None:
        vals, counts = np.unique(np.round(w[w > 300]), return_counts=True)
        if not len(vals):
            return None
        expected = float(vals[np.argmax(counts)])
    full = np.abs(w - expected) < 3.0
    if full.sum() < 5:
        return None
    return float(np.median(L[full])), float(np.median(Rr[full])), int(full.sum())


def row_profile(frame: np.ndarray, ice: np.ndarray, cols: tuple[int, int]) -> np.ndarray:
    """Per-row median over the ice columns of the max-channel distance from the ice colour."""
    a, b = cols
    d = np.abs(frame[:, a:b:3].astype(np.float32) - ice).max(axis=2)
    return np.median(d, axis=1)


@dataclass
class Line:
    v: float  # subpixel centre row
    width: int
    strength: float
    blue: bool


def find_lines(frame: np.ndarray, ice: np.ndarray, cols: tuple[int, int]) -> list[Line]:
    prof = row_profile(frame, ice, cols)
    pk, _ = find_peaks(prof, height=MIN_PEAK, distance=10)
    out = []
    a, b = cols
    for p in pk:
        lo, hi = p, p
        half = prof[p] / 2
        while lo > 0 and prof[lo - 1] > half:
            lo -= 1
        while hi < len(prof) - 1 and prof[hi + 1] > half:
            hi += 1
        width = hi - lo + 1
        if width > LINE_HALF_MAX:
            continue
        w = prof[lo : hi + 1] - half
        cen = float((np.arange(lo, hi + 1) * w).sum() / w.sum())
        bgr = np.median(frame[p, a:b].astype(np.float32), axis=0)
        blue = bool(bgr[0] > bgr[2] + 60 and bgr[0] > 140)
        out.append(Line(cen, width, float(prof[p]), blue))
    return out


@dataclass
class Fit:
    v_c: float | None
    residual: float
    lines: list[tuple[str, float]] = field(default_factory=list)


def identify(lines: list[Line], s: float, model: dict[str, float], v_pred: float | None) -> Fit:
    """Choose the translation v_c that explains the most detected lines (tie: closest to the prediction).
    The spacing pattern (136.5 / 250.3 px) and the blue center line make the choice unique with >= 2 lines."""
    best = None
    for L in lines:
        for name, x in model.items():
            if name == "center" and not L.blue:
                continue
            if name != "center" and L.blue:
                continue
            vc = L.v + s * x
            matched = []
            for M in lines:
                for n2, x2 in model.items():
                    if abs(M.v - (vc - s * x2)) <= MATCH_TOL and (n2 == "center") == M.blue:
                        matched.append((n2, M.v, x2))
            names = {m[0] for m in matched}
            if len(names) != len(matched):
                continue
            vcs = [m[1] + s * m[2] for m in matched]
            vc_fit = float(np.mean(vcs))
            res = float(max(abs(v - vc_fit) for v in vcs))
            key = (len(matched), -(abs(vc_fit - v_pred) if v_pred is not None else 0.0), -res)
            if best is None or key > best[0]:
                best = (key, Fit(vc_fit, res, [(m[0], m[1]) for m in matched]))
    if best is None or len(best[1].lines) < 2:
        return Fit(None, float("nan"), best[1].lines if best else [])
    return best[1]


def fit_scale(per_frame_lines: list[list[Line]], s0: float, model: dict[str, float]) -> float:
    """Least-squares px per meter from frames with >= 3 identified lines (one v_c per frame, common s)."""
    xs, vs, fid = [], [], []
    for k, lines in enumerate(per_frame_lines):
        f = identify(lines, s0, model, None)
        if f.v_c is None or len(f.lines) < 3:
            continue
        for name, v in f.lines:
            xs.append(model[name]); vs.append(v); fid.append(k)
    if len(xs) < 6:
        return s0
    xs, vs, fid = np.array(xs), np.array(vs), np.array(fid)
    # v = vc_f - s x  -> remove per-frame means, then regress.
    xm = np.array([xs[fid == f].mean() for f in fid])
    vm = np.array([vs[fid == f].mean() for f in fid])
    dx, dv = xs - xm, vs - vm
    return float(-(dx * dv).sum() / (dx * dx).sum())


def odometry(prev: np.ndarray, cur: np.ndarray, cols: tuple[int, int], search: int = 40) -> tuple[float, float, float]:
    """Frame-to-frame shift of the ice band (du, dv, score): the central part of the previous frame's band matched in
    the current frame (normalised cross-correlation of gradient images, parabolic subpixel peak). Phase correlation
    was tried first and returned a spurious 0.5 px horizontal shift between near-identical frames."""
    a, b = cols

    def prep(f):
        g = cv2.cvtColor(f[:, a:b], cv2.COLOR_BGR2GRAY).astype(np.float32)
        return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

    p, c = prep(prev), prep(cur)
    H, W = p.shape
    tpl = p[search : H - search, search : W - search]
    res = cv2.matchTemplate(c, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (px, py) = cv2.minMaxLoc(res)

    def para(m, z, q):
        d = m - 2 * z + q
        return 0.0 if d == 0 else 0.5 * (m - q) / d

    sx = para(res[py, px - 1], res[py, px], res[py, px + 1]) if 0 < px < res.shape[1] - 1 else 0.0
    sy = para(res[py - 1, px], res[py, px], res[py + 1, px]) if 0 < py < res.shape[0] - 1 else 0.0
    return float(px + sx - search), float(py + sy - search), float(score)


@dataclass
class FrameReg:
    frame: int
    u_c: float
    v_c: float
    source: str  # "lines" (anchored), "odometry", or "none"
    residual: float
    n_lines: int
    width: float = float("nan")  # measured board-to-board width (px); its deviation from the clip median checks u


class Registrar:
    """Registers every frame of one clip. Feed frames in order with `add`."""

    def __init__(self, params: dict | None = None):
        self.model = line_model(params)
        self.ice = None
        self.cols = None  # (left, right) inner board columns
        self.s = None
        self.prev = None
        self.out: list[FrameReg] = []

    def calibrate(self, frames: list[np.ndarray]) -> None:
        """Ice colour, board-to-board width and px/m from a sample of frames."""
        self.ice = ice_colour(frames[len(frames) // 2])
        self.frame_w = frames[0].shape[1]
        be = [board_edges(f, self.ice) for f in frames]
        be = [b for b in be if b]
        if not be:
            raise RuntimeError("no board edges found")
        self.width = float(np.median([b[1] - b[0] for b in be]))
        p = load_params()
        s0 = self.width / p["rink_width"]
        per = []
        for f, b in zip(frames, [board_edges(f, self.ice) for f in frames]):
            if b:
                per.append(find_lines(f, self.ice, self._cols(b)))
        self.s = fit_scale(per, s0, self.model)
        self.board = (float(np.median([b[0] for b in be])), float(np.median([b[1] for b in be])))
        self.cols = self._cols(self.board)

    def _cols(self, b) -> tuple[int, int]:
        """Ice columns inside the boards (kept inside the frame and at least half a rink wide)."""
        W = getattr(self, "frame_w", 1920)
        a, c = max(0, int(math.ceil(b[0])) + 12), min(W, int(b[1]) - 12)
        if c - a < 200:
            a, c = getattr(self, "cols", (0, W))
        return a, c

    def add(self, k: int, frame: np.ndarray) -> FrameReg:
        du = dv = None
        if self.prev is not None and self.out and self.out[-1].v_c == self.out[-1].v_c:
            du, dv, _ = odometry(self.prev, frame, self.cols)
        be = board_edges(frame, self.ice, self.width)
        if be and abs((be[1] - be[0]) - self.width) < 3.0:
            u_c, width = (be[0] + be[1]) / 2, be[1] - be[0]
            self.board = (be[0], be[1])
            self.cols = self._cols(be)
        elif du is not None and abs(du) < 30:
            u_c, width = self.out[-1].u_c + du, float("nan")
            self.board = (self.board[0] + du, self.board[1] + du)
            self.cols = self._cols(self.board)
        else:
            u_c, width = (self.board[0] + self.board[1]) / 2, float("nan")
        lines = find_lines(frame, self.ice, self.cols)
        odo = self.out[-1].v_c + dv if dv is not None else None
        fit = identify(lines, self.s, self.model, odo)
        if fit.v_c is not None:
            r = FrameReg(k, u_c, fit.v_c, "lines", fit.residual, len(fit.lines), width)
        elif odo is not None:
            r = FrameReg(k, u_c, odo, "odometry", float("nan"), len(fit.lines), width)
        else:
            r = FrameReg(k, u_c, float("nan"), "none", float("nan"), 0, width)
        self.prev = frame
        self.out.append(r)
        return r

    def to_world(self, r: FrameReg, u: float, v: float) -> tuple[float, float]:
        return (r.v_c - v) / self.s, (r.u_c - u) / self.s


def drift_test(v_anchor: np.ndarray, dv_odo: np.ndarray, gaps=(1, 5, 15, 30, 60, 120), max_step: float = 40.0) -> dict:
    """Error of pure odometry vs the line anchors after n frames without an anchor (all start frames)."""
    out = {}
    ok = np.isfinite(v_anchor)
    # windows that contain a teleport (a practice-mode reset: the anchor jumps > max_step px in one frame) are skipped
    step = np.r_[0.0, np.abs(np.diff(v_anchor))]
    jump = ~(step <= max_step)
    for n in gaps:
        errs = []
        for a in range(0, len(v_anchor) - n):
            if ok[a] and ok[a + n] and np.isfinite(dv_odo[a + 1 : a + n + 1]).all() and not jump[a + 1 : a + n + 1].any():
                errs.append(v_anchor[a] + dv_odo[a + 1 : a + n + 1].sum() - v_anchor[a + n])
        if errs:
            e = np.abs(np.array(errs))
            out[str(n)] = {"median_px": round(float(np.median(e)), 3), "p95_px": round(float(np.percentile(e, 95)), 3), "n": len(e)}
    return out
