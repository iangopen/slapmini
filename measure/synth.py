"""Synthetic test video: known puck path plus every distractor the real footage has.

Rink (vertical long axis) with grey boards, cyan center line and a ringed center dot; a green puck
that rests, then glides and bounces; an always-on green light inside the rink; a blinking green
light that jumps between two spots; a moving pink disc with a blue glow; faceoff circles (the camera landmarks); a camera pan
over frames 100-129; a HUD box with an animated
green bar and green "stamina" dots outside the rink; and one duplicated (repeated) frame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, N = 960, 540, 60.0, 150
BOARD = {"left": 330, "right": 630, "top": 30, "bottom": 510}  # inner board edges (pixel boundaries)
CORNER_R = 60
BAND = 18
ICE, BOARD_COL, ARENA = (47, 56, 58), (165, 165, 165), (40, 40, 40)
LINE_COL = (194, 168, 72)  # cyan, like the real center line
PUCK_COL = (35, 213, 11)  # BGR green
PUCK_R = 4.0
PINK, GLOW = (250, 148, 255), (230, 120, 60)
CENTER = (480.0, 270.0)
DUP_FRAME = 70  # rendered identical to frame 69
FACEOFFS = [(400.0, 110.0), (560.0, 110.0), (400.0, 430.0), (560.0, 430.0)]
RING_R = 35.0
RING_COL = (88, 90, 92)  # grey ring, like the real faceoff circles
PAN = (100, 130, 7.4, -3.2)  # frames [100, 130) are shifted by (+7.4, -3.2) px: a camera pan
SHIFT = 4
S = 1 << SHIFT


def _P(x, y):
    return (int(round(x * S)), int(round(y * S)))


def _rounded_rect(img, x0, y0, x1, y1, r, col):
    cv2.rectangle(img, _P(x0 + r, y0), _P(x1 - r, y1), col, -1, cv2.LINE_8, SHIFT)
    cv2.rectangle(img, _P(x0, y0 + r), _P(x1, y1 - r), col, -1, cv2.LINE_8, SHIFT)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, _P(cx, cy), int(r * S), col, -1, cv2.LINE_8, SHIFT)


def background() -> np.ndarray:
    img = np.full((H, W, 3), ARENA, np.uint8)
    b = BOARD
    # Edges are pixel boundaries: ice pixel centers run from left to right - 1.
    _rounded_rect(img, b["left"] - BAND, b["top"] - BAND, b["right"] - 1 + BAND, b["bottom"] - 1 + BAND, CORNER_R + BAND, BOARD_COL)
    _rounded_rect(img, b["left"], b["top"], b["right"] - 1, b["bottom"] - 1, CORNER_R, ICE)
    cv2.circle(img, _P(*CENTER), int(40 * S), RING_COL, 2, cv2.LINE_AA, SHIFT)
    for fx, fy in FACEOFFS:
        cv2.circle(img, _P(fx, fy), int(RING_R * S), RING_COL, 2, cv2.LINE_AA, SHIFT)
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):  # cross marks: texture for the camera landmarks
            cv2.line(img, _P(fx + 6 * dx, fy + 3 * dy), _P(fx + 14 * dx, fy + 3 * dy), RING_COL, 1, cv2.LINE_AA, SHIFT)
            cv2.line(img, _P(fx + 3 * dx, fy + 6 * dy), _P(fx + 3 * dx, fy + 14 * dy), RING_COL, 1, cv2.LINE_AA, SHIFT)
        cv2.circle(img, _P(fx, fy), int(3 * S), LINE_COL, -1, cv2.LINE_AA, SHIFT)
    cv2.rectangle(img, (b["left"], int(CENTER[1]) - 3), (b["right"] - 1, int(CENTER[1]) + 3), LINE_COL, -1)
    cv2.circle(img, _P(*CENTER), int(6 * S), (60, 60, 60), -1, cv2.LINE_AA, SHIFT)
    cv2.circle(img, _P(*CENTER), int(4.5 * S), (150, 150, 140), -1, cv2.LINE_AA, SHIFT)
    return img


def puck_path() -> list:
    """Truth (u, v) per frame. Rests 20 frames, then glides with drag and bounces off the boards."""
    u, v = 470.0, 380.0
    vu, vv = 0.0, 0.0
    lo_u, hi_u = BOARD["left"] + PUCK_R, BOARD["right"] - PUCK_R
    lo_v, hi_v = BOARD["top"] + PUCK_R, BOARD["bottom"] - PUCK_R
    out = []
    for f in range(N):
        if f == 20:
            vu, vv = 1.3, -7.1  # the hit: bounces off the straight top board, away from the rounded corners
        u += vu
        v += vv
        if u < lo_u or u > hi_u:
            u = 2 * (lo_u if u < lo_u else hi_u) - u
            vu = -0.7 * vu
        if v < lo_v or v > hi_v:
            v = 2 * (lo_v if v < lo_v else hi_v) - v
            vv = -0.7 * vv
        vu *= 0.993
        vv *= 0.993
        out.append((u, v))
    return out


def pink_path(f: int) -> tuple[float, float]:
    return 360.0 + 240.0 * (0.5 - 0.5 * math.cos(2 * math.pi * f / N)), 425.0  # passes 25 px from the resting puck


STATIC_LIGHT = (600.0, 470.0)
BLINK_SPOTS = [(370.0, 60.0), (402.0, 64.0)]


def render(out: Path) -> dict:
    out = Path(out)
    bg = background()
    path = puck_path()
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
    if not wr.isOpened():
        raise RuntimeError("FFV1 writer unavailable in this OpenCV build")
    truth = []
    prev = None
    min_pink = 1e9
    for f in range(N):
        if f == DUP_FRAME:
            wr.write(prev)
            truth.append(None)  # stale frame: position belongs to frame f-1
            continue
        img = bg.copy()
        # HUD outside the rink: box with an animated green bar, plus green stamina dots.
        cv2.rectangle(img, (20, 20), (220, 80), (200, 80, 40), -1)
        cv2.rectangle(img, (30, 50), (30 + (f * 7) % 180, 60), PUCK_COL, -1)
        for x in (470, 492):
            cv2.circle(img, (x, 534), 3, PUCK_COL, -1, cv2.LINE_AA)
        # Static green light inside the rink (always on).
        cv2.circle(img, _P(*STATIC_LIGHT), int(4 * S), PUCK_COL, -1, cv2.LINE_AA, SHIFT)
        # Blinking light that jumps between two spots.
        if (f // 25) % 2 == 0:
            cv2.circle(img, _P(*BLINK_SPOTS[(f // 50) % 2]), int(4 * S), PUCK_COL, -1, cv2.LINE_AA, SHIFT)
        # Pink player disc with a blue glow.
        pu, pv = pink_path(f)
        cv2.circle(img, _P(pu, pv), int(13 * S), GLOW, -1, cv2.LINE_AA, SHIFT)
        cv2.circle(img, _P(pu, pv), int(9 * S), PINK, -1, cv2.LINE_AA, SHIFT)
        u, v = path[f]
        min_pink = min(min_pink, math.hypot(u - pu, v - pv))
        cv2.circle(img, _P(u, v), int(PUCK_R * S), PUCK_COL, -1, cv2.LINE_AA, SHIFT)
        if PAN[0] <= f < PAN[1]:
            M = np.float32([[1, 0, PAN[2]], [0, 1, PAN[3]]])
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wr.write(img)
        prev = img
        truth.append([u, v])
    wr.release()
    meta = {"frames": truth, "fps": FPS, "board": BOARD, "corner_r": CORNER_R, "center": CENTER, "dup_frame": DUP_FRAME,
            "static_light": STATIC_LIGHT, "blink_spots": BLINK_SPOTS, "min_puck_pink_px": min_pink, "pan": PAN}
    out.with_suffix(".truth.json").write_text(json.dumps(meta))
    return meta
