"""`figures` command: fit figures drawn on the SIM's own rink (from params.json), no game footage.

Game orientation (+x up, +y left), like the real camera and the lab's game view. Each figure shows the
measured puck path (the segment's ground positions) and sim replays of the same scenario (calib replay),
dots every 0.25 s on every path, and a speed-vs-distance panel. All meters at the PLACEHOLDER scale.
Output: measure/results/fits/*.png (small, flat colours).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .geometry import PARAMS_PATH, REPO
from .holdout import CALIB, with_values
from .segment import OUT

FITS = REPO / "measure" / "results" / "fits"
BG, ICE, LINE, RED, BLUE, INK = (250, 250, 250), (244, 238, 233), (200, 200, 200), (70, 70, 210), (200, 130, 60), (40, 40, 40)
MEAS = (30, 30, 30)
SIMCOLS = [(40, 140, 240), (60, 170, 60), (200, 80, 160), (200, 160, 40)]


def _P(params_path):
    return {k: v["value"] for k, v in json.loads(Path(params_path).read_text())["params"].items()}


class View:
    """meters -> pixels, game orientation: screen right = -y, screen up = +x."""

    def __init__(self, P, height_px=760, margin=20, crop=None):
        L, W = P["rink_length"], P["rink_width"]
        if crop is None:
            crop = (-L / 2, L / 2, -W / 2, W / 2)  # x0, x1, y0, y1
        self.x0, self.x1, self.y0, self.y1 = crop
        self.s = (height_px - 2 * margin) / (self.x1 - self.x0)
        self.w = int(round((self.y1 - self.y0) * self.s + 2 * margin))
        self.h = height_px
        self.m = margin

    def px(self, x, y):
        return (self.m + (self.y1 - np.asarray(y)) * self.s, self.m + (self.x1 - np.asarray(x)) * self.s)

    def pt(self, x, y):
        u, v = self.px(x, y)
        return (int(round(float(u) * 16)), int(round(float(v) * 16)))


def draw_rink(P, view: View) -> np.ndarray:
    img = np.full((view.h, view.w, 3), BG, np.uint8)
    L, W, rc = P["rink_length"], P["rink_width"], P["rink_corner_radius"]
    # ice: rounded rectangle
    mask = np.zeros(img.shape[:2], np.uint8)
    sh = 4
    def rr(x0, x1, y0, y1, r, col, thick):
        pts = []
        for cx, cy, a0 in ((x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270)):
            for t in np.linspace(np.radians(a0), np.radians(a0 + 90), 24):
                pts.append(view.pt(cx + r * np.cos(t), cy + r * np.sin(t)))
        pts = np.array(pts, np.int32)
        if thick < 0:
            cv2.fillPoly(img, [pts], col, cv2.LINE_AA, sh)
        else:
            cv2.polylines(img, [pts], True, col, thick, cv2.LINE_AA, sh)
    rr(-L / 2, L / 2, -W / 2, W / 2, rc, ICE, -1)

    def line(x0, y0, x1, y1, col, th=1):
        cv2.line(img, view.pt(x0, y0), view.pt(x1, y1), col, th, cv2.LINE_AA, sh)

    hw = W / 2 - 0.02
    line(0, -hw, 0, hw, RED, 2)
    for sgn in (1, -1):
        line(sgn * P["blue_line_x"], -hw, sgn * P["blue_line_x"], hw, BLUE, 2)
        line(sgn * P["net_x"], -hw + 1.5, sgn * P["net_x"], hw - 1.5, RED, 1)
        # net: sides + back
        x0, x1, w = sgn * P["net_x"], sgn * (P["net_x"] + P["net_depth"]), P["net_width"] / 2
        for a, b in (((x0, w), (x1, w)), ((x1, w), (x1, -w)), ((x1, -w), (x0, -w))):
            line(a[0], a[1], b[0], b[1], RED, 2)
        for sy in (1, -1):
            c = view.pt(sgn * P["faceoff_x"], sy * P["faceoff_y"])
            cv2.circle(img, c, int(P["faceoff_circle_radius"] * view.s * 16), RED, 1, cv2.LINE_AA, sh)
            cv2.circle(img, c, int(0.3 * view.s * 16), RED, -1, cv2.LINE_AA, sh)
    cv2.circle(img, view.pt(0, 0), int(P["faceoff_circle_radius"] * view.s * 16), BLUE, 1, cv2.LINE_AA, sh)
    rr(-L / 2, L / 2, -W / 2, W / 2, rc, (120, 120, 120), 2)
    return img


def replay(params_path: Path, scenario: Path) -> np.ndarray:
    out = Path(tempfile.mkdtemp()) / "r.csv"
    subprocess.run([str(CALIB), "replay", "--params", str(params_path), str(scenario), str(out)], check=True, capture_output=True)
    return np.loadtxt(out, delimiter=",", skiprows=1, ndmin=2)


def draw_path(img, view, a, col, th=2, dots=True, dashed=False):
    u, v = view.px(a[:, 1], a[:, 2])
    pts = (np.stack([u, v], 1) * 16).astype(np.int32)
    if dashed:
        for i in range(0, len(pts) - 1, 4):
            cv2.polylines(img, [pts[i : i + 3]], False, col, th, cv2.LINE_AA, 4)
    else:
        cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA, 4)
    if dots:
        k = np.searchsorted(a[:, 0], np.arange(0, a[-1, 0] + 1e-9, 0.25))
        for i in k[k < len(a)]:
            cv2.circle(img, (int(pts[i, 0]), int(pts[i, 1])), 3 * 16, col, -1, cv2.LINE_AA, 4)


def vs_curve(a, measured: bool):
    t, x, y = a[:, 0], a[:, 1], a[:, 2]
    if measured:
        from .dragmodel import smooth_vs

        V, S = smooth_vs(t, x, y)
        # The first/last 5 samples use one-sided local fits (edge artefacts): leave them out of the plot.
        return (S[5:-5], V[5:-5]) if len(S) > 20 else (S, V)
    d = np.hypot(np.diff(x), np.diff(y))
    S = np.r_[0, np.cumsum(d)]
    V = np.r_[d[0], d] / np.r_[t[1] - t[0], np.diff(t)]
    return S, V


def speed_panel(curves, w, h, title):
    """curves: [(label, S, V, col, measured)] -> image with axes."""
    img = np.full((h, w, 3), BG, np.uint8)
    l, r, t, b = 58, 16, 34, 44
    smax = max(c[1].max() for c in curves) * 1.02
    vmax = max(c[2].max() for c in curves) * 1.08
    X = lambda s: l + s / smax * (w - l - r)
    Y = lambda v: h - b - v / vmax * (h - t - b)
    cv2.rectangle(img, (l, t), (w - r, h - b), (180, 180, 180), 1)
    for vt in np.arange(0, vmax, 5 if vmax > 12 else 2):
        cv2.line(img, (l - 4, int(Y(vt))), (l, int(Y(vt))), INK, 1)
        cv2.putText(img, f"{vt:.0f}", (8, int(Y(vt)) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, INK, 1, cv2.LINE_AA)
    step = 5 if smax > 20 else 2
    for st in np.arange(0, smax, step):
        cv2.line(img, (int(X(st)), h - b), (int(X(st)), h - b + 4), INK, 1)
        cv2.putText(img, f"{st:.0f}", (int(X(st)) - 6, h - b + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, INK, 1, cv2.LINE_AA)
    cv2.putText(img, "distance travelled (m)", (l + 40, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA)
    cv2.putText(img, "speed (m/s)", (8, t - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA)
    cv2.putText(img, title, (l + 70, t - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA)
    for lab, S, V, col, measured in curves:
        pts = np.stack([X(S), Y(V)], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, col, 2 if measured else 2, cv2.LINE_AA)
    return img


def legend(img, items, x, y):
    for i, (lab, col, dashed) in enumerate(items):
        yy = y + 20 * i
        if dashed:
            for k in range(0, 24, 8):
                cv2.line(img, (x + k, yy), (x + k + 5, yy), col, 2, cv2.LINE_AA)
        else:
            cv2.line(img, (x, yy), (x + 24, yy), col, 2, cv2.LINE_AA)
        cv2.putText(img, lab, (x + 32, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA)


def segment_figure(seg_id: str, variants: list[tuple[str, dict]], title: str, out: Path, params_path: Path = PARAMS_PATH, zoom=True):
    man = json.loads((OUT / "manifest.json").read_text())
    seg = next(s for s in man["segments"] if s["id"] == seg_id)
    P = _P(params_path)
    meas = np.loadtxt(OUT / seg["csv"], delimiter=",", skiprows=1, ndmin=2)
    tmp = Path(tempfile.mkdtemp())
    sims = []
    for i, (lab, vals) in enumerate(variants):
        sims.append((lab, replay(with_values(params_path, vals, tmp / f"p{i}.json"), OUT / seg["scenario"]), SIMCOLS[i % len(SIMCOLS)]))
    crop = None
    if zoom:
        allx = np.concatenate([meas[:, 1]] + [s[1][:, 1] for s in sims])
        ally = np.concatenate([meas[:, 2]] + [s[1][:, 2] for s in sims])
        L, W = P["rink_length"], P["rink_width"]
        pad = 3.0
        crop = (max(allx.min() - pad, -L / 2 - 0.5), min(allx.max() + pad, L / 2 + 0.5), max(ally.min() - pad, -W / 2 - 0.5), min(ally.max() + pad, W / 2 + 0.5))
        # keep aspect sane
        if (crop[1] - crop[0]) < 0.6 * (crop[3] - crop[2]):
            mid = (crop[0] + crop[1]) / 2
            half = 0.3 * (crop[3] - crop[2])
            crop = (mid - half, mid + half, crop[2], crop[3])
    view = View(P, height_px=560, crop=crop)
    img = draw_rink(P, view)
    for lab, a, col in sims:
        draw_path(img, view, a, col, th=2)
    draw_path(img, view, meas, MEAS, th=2, dashed=True)
    panel = speed_panel([("measured", *vs_curve(meas, True), MEAS, True)] + [(lab, *vs_curve(a, False), col, False) for lab, a, col in sims],
                        620, 560, f"{seg_id} ({seg['label']})")
    legend(panel, [("measured (dashed)", MEAS, True)] + [(lab, col, False) for lab, _, col in sims], 380, 70)
    fig = np.hstack([img, np.full((560, 10, 3), BG, np.uint8), panel])
    head = np.full((44, fig.shape[1], 3), BG, np.uint8)
    cv2.putText(head, title, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK, 1, cv2.LINE_AA)
    cv2.putText(head, "sim rink from params.json, +x up; dots every 0.25 s; meters at the PLACEHOLDER scale (rink_length 60 m)", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    fig = np.vstack([head, fig])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), fig, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"[figures] wrote {out} ({out.stat().st_size // 1024} KB)")
    return out


def readme_figure(seg_ids: list[str], out: Path, params_path: Path = PARAMS_PATH):
    man = {s["id"]: s for s in json.loads((OUT / "manifest.json").read_text())["segments"]}
    P = _P(params_path)
    view = View(P, height_px=820)
    img = draw_rink(P, view)
    for sid in seg_ids:
        seg = man[sid]
        meas = np.loadtxt(OUT / seg["csv"], delimiter=",", skiprows=1, ndmin=2)
        sim = replay(params_path, OUT / seg["scenario"])
        draw_path(img, view, sim, SIMCOLS[0], th=2, dots=False)
        draw_path(img, view, meas, MEAS, th=2, dots=False, dashed=True)
        u, v = view.px(meas[0, 1], meas[0, 2])
        cv2.circle(img, (int(u), int(v)), 4, MEAS, -1, cv2.LINE_AA)
    # speed panel for the first one
    seg = man[seg_ids[0]]
    meas = np.loadtxt(OUT / seg["csv"], delimiter=",", skiprows=1, ndmin=2)
    sim = replay(params_path, OUT / seg["scenario"])
    plc = replay(Path(REPO / "params.placeholder.json"), OUT / seg["scenario"])
    panel = speed_panel([("measured", *vs_curve(meas, True), MEAS, True), ("fitted sim", *vs_curve(sim, False), SIMCOLS[0], False),
                         ("placeholder sim", *vs_curve(plc, False), (170, 170, 170), False)], 560, 400, f"speed along {seg_ids[0]}")
    legend(panel, [("measured (real game)", MEAS, True), ("sim, fitted params", SIMCOLS[0], False), ("sim, placeholder params", (170, 170, 170), False)], 300, 70)
    right = np.vstack([panel, np.full((820 - 400, 560, 3), BG, np.uint8)])
    notes = ["Real practice-mode shots (dashed) vs the sim",
             "replaying each shot from its measured start",
             "(solid), on the sim's own rink.",
             f"{len(seg_ids)} shots; dot = start of each shot.",
             "Fitted: board restitution 0.41, linear ice",
             "drag 0.41/s, no constant friction (held out",
             "by clip). Meters at a placeholder scale."]
    for i, t in enumerate(notes):
        cv2.putText(right, t, (20, 450 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
    fig = np.hstack([img, right])
    cv2.imwrite(str(out), fig, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"[figures] wrote {out} ({out.stat().st_size // 1024} KB)")
    return out


def run_figures(params_path: Path = PARAMS_PATH) -> list[Path]:
    P = _P(params_path)
    PL = _P(REPO / "params.placeholder.json")
    e, k = P["board_restitution"], P["ice_linear_drag"]
    outs = [
        segment_figure("wall_07", [("e = 0.20", {"board_restitution": 0.2}), (f"e = {e:.2f} (fitted)", {}), (f"e = {PL['board_restitution']:.2f} (placeholder)", {"board_restitution": PL["board_restitution"]})],
                       "Board restitution: a wall bounce replayed at three restitution values", FITS / "restitution_wall_07.png"),
        segment_figure("puck_06", [(f"drag k = {k:.2f}/s (fitted)", {}), ("k = 0.20/s", {"ice_linear_drag": 0.2}),
                                   ("placeholder: 1 m/s2 + 0.05/s", {"ice_linear_drag": PL["ice_linear_drag"], "ice_friction_decel": PL["ice_friction_decel"]})],
                       "Ice drag: a free glide replayed with the fitted linear drag and two alternatives", FITS / "drag_puck_06.png"),
        segment_figure("puck_08", [("fitted", {}), ("placeholder", {k_: PL[k_] for k_ in ("board_restitution", "board_friction", "ice_linear_drag", "ice_friction_decel")})],
                       "Corner ride: left-board bounce, corner, slide along the top board (fitted vs placeholder)", FITS / "corner_puck_08.png"),
        readme_figure(["puck_08", "wall_07", "puck_06", "wall_09", "puck_02"], FITS / "readme_fit.png"),
    ]
    return outs
