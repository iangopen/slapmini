"""Synthetic panning video for measure/camreg.py + measure/skate.py (session s7 camera).

A top-down rink drawn at s px/m with the painted lines at their params.json positions, faceoff circles and dots,
striped stands outside the boards, a key-display box in the HUD corner, a grey/red player disc with a black stick
and a green puck. The camera follows the player with lag and
sub-pixel offsets; the camera follows the player along the rink and sideways (as in the real clips). Ground truth: per frame the image row of world x = 0 (v_c) and the world positions of the
player and the puck.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .geometry import load_params

W, H, FPS = 1920, 1080, 60
S = 16.95  # px per meter
U_C = 1132.0
ICE = (58, 54, 46)
N_FRAMES = 150
MARGIN = 700  # canvas rows above/below the rink


def canvas() -> tuple[np.ndarray, float]:
    """The world as one big image; returns (image, row of world x = 0)."""
    p = load_params()
    half_l, half_w = p["rink_length"] / 2, p["rink_width"] / 2
    rows = int(2 * (half_l * S + MARGIN))
    v0 = rows / 2
    rng = np.random.default_rng(3)
    img = np.zeros((rows, W, 3), np.uint8)
    for y in range(0, rows, 6):  # stands: stripes of random grey (never ice-coloured)
        img[y : y + 6] = rng.integers(90, 200)
    P = lambda x, y: (int(round((U_C - y * S) * 16)), int(round((v0 - x * S) * 16)))
    # rounded-rectangle ice
    rc = p["rink_corner_radius"] * S
    x0, x1 = U_C - half_w * S, U_C + half_w * S
    y0, y1 = v0 - half_l * S, v0 + half_l * S
    # Exact (area-coverage) rendering where the registration looks: pixel i covers [i - 0.5, i + 0.5].
    def cover(i: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip(np.minimum(i + 0.5, hi) - np.maximum(i - 0.5, lo), 0.0, 1.0)

    cols = np.arange(W, dtype=np.float64)
    rws = np.arange(rows, dtype=np.float64)
    alpha = np.zeros((rows, W), np.float64)
    alpha[:] = cover(cols, x0, x1)[None, :] * cover(rws, y0 + rc, y1 - rc)[:, None]
    mid = cover(cols, x0 + rc, x1 - rc)[None, :] * cover(rws, y0, y1)[:, None]
    alpha = np.maximum(alpha, mid)
    corner = np.zeros((rows, W), np.uint8)
    for cx, cy in ((x0 + rc, y0 + rc), (x1 - rc, y0 + rc), (x0 + rc, y1 - rc), (x1 - rc, y1 - rc)):
        cv2.circle(corner, (int(round(cx * 16)), int(round(cy * 16))), int(round(rc * 16)), 255, -1, cv2.LINE_AA, 4)
    alpha = np.maximum(alpha, corner / 255.0)
    img = (img * (1 - alpha[..., None]) + np.array(ICE) * alpha[..., None]).round().astype(np.uint8)
    mask = (alpha > 0.5).astype(np.uint8) * 255
    grey, blue = (93, 90, 81), (191, 162, 69)
    span = cover(cols, x0 + 2, x1 - 2)
    for x, col, th in ((p["net_x"], grey, 4), (p["blue_line_x"], grey, 7), (0.0, blue, 7), (-p["blue_line_x"], grey, 7), (-p["net_x"], grey, 4)):
        v = v0 - x * S
        rr = np.arange(int(v - th), int(v + th) + 2)
        a_line = cover(rr.astype(np.float64), v - th / 2, v + th / 2)[:, None] * span[None, :]
        img[rr] = (img[rr] * (1 - a_line[..., None]) + np.array(col) * a_line[..., None]).round().astype(np.uint8)
    for fx in (p["faceoff_x"], -p["faceoff_x"]):
        for fy in (p["faceoff_y"], -p["faceoff_y"]):
            cv2.circle(img, P(fx, fy), int(p["faceoff_circle_radius"] * S * 16), grey, 2, cv2.LINE_AA, 4)
            cv2.circle(img, P(fx, fy), 3 * 16, (180, 150, 60), -1, cv2.LINE_AA, 4)
    cv2.circle(img, P(0, 0), int(4.7 * S * 16), grey, 2, cv2.LINE_AA, 4)
    cv2.circle(img, P(0, 0), 3 * 16, (60, 200, 60), -1, cv2.LINE_AA, 4)
    return img, v0


def player_world(t: float) -> tuple[float, float]:
    """A run up the ice with a lateral weave, up to ~10 m/s."""
    return -18.0 + 14.0 * t + 0.0 * t * t - 3.0 * math.sin(1.3 * t), 4.0 * math.sin(0.9 * t)


def puck_world(t: float) -> tuple[float, float]:
    return 10.0 - 6.0 * t, -6.0 + 2.0 * t


def render(out: Path) -> dict:
    img, v0 = canvas()
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
    truth = {"v_c": [], "u_c": [], "player": [], "puck": [], "s": S}
    cam = camu = None
    for k in range(N_FRAMES):
        t = k / FPS
        px, py = player_world(t)
        target = v0 - px * S - H / 2  # canvas row at the top of the frame that centres the player
        cam = target if cam is None else cam + 0.25 * (target - cam)  # lagging follow
        tu = py * S + (959.5 - U_C)  # the camera also follows sideways, keeping the player at u = 959.5 like the game
        camu = tu if camu is None else camu + 0.25 * (tu - camu)
        top = cam + 0.37  # never on whole pixels
        left = camu + 0.21
        M = np.float32([[1, 0, left], [0, 1, -top]])
        f = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        vc = v0 - top
        uc = U_C + left
        # puck (green) and player (grey disc, red rim, black stick) at sub-pixel positions
        qx, qy = puck_world(t)
        q = (uc - qy * S, vc - qx * S)
        cv2.circle(f, (int(round(q[0] * 16)), int(round(q[1] * 16))), 5 * 16, (60, 220, 90), -1, cv2.LINE_AA, 4)
        pu, pv = uc - py * S, vc - px * S
        a = 0.8 * t
        cv2.line(f, (int(pu * 16), int(pv * 16)), (int((pu + 24 * math.cos(a)) * 16), int((pv - 24 * math.sin(a)) * 16)), (20, 20, 20), 3, cv2.LINE_AA, 4)
        cv2.circle(f, (int(round(pu * 16)), int(round(pv * 16))), 10 * 16, (60, 60, 190), -1, cv2.LINE_AA, 4)
        cv2.circle(f, (int(round(pu * 16)), int(round(pv * 16))), 8 * 16, (88, 88, 90), -1, cv2.LINE_AA, 4)
        # screen-fixed HUD: a key-display box with two white caps
        cv2.rectangle(f, (1380, 785), (1720, 880), (170, 170, 170), -1)
        cv2.rectangle(f, (1546, 797), (1614, 865), (255, 255, 255), -1)
        cv2.rectangle(f, (1636, 797), (1704, 865), (255, 255, 255), -1)
        wr.write(f)
        truth["v_c"].append(vc)
        truth["u_c"].append(uc)
        truth["player"].append((px, py))
        truth["puck"].append((qx, qy))
    wr.release()
    return truth
