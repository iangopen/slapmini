"""Synthetic test video for the stick tracker (measure/stick.py).

A grey player body (S = 0) in a blue glow, at the image centre like the real follow camera. A white radial blade
outside the body and a black shaft across it, turning on a known angle profile (rest, ramp to a 26 rad/s cap, stop,
reversal, slow sweep). Distractors: body markings that turn on their own (the real body faces separately from the
stick), a green puck nearby, a dark mouse-hint icon with a white glyph fixed on screen next to the player, a HUD box
with black and white text, and one duplicated capture frame. Truth = the blade direction in the sim frame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, N = 640, 360, 60.0, 150
C = (320.0, 180.0)
R = 40.0
ICE = (60, 58, 50)  # BGR, bluish grey (S ~ 50)
BODY = (88, 88, 88)
GLOW = (230, 120, 40)
DUP_FRAME = 77
SHIFT = 4
S = 1 << SHIFT


def _P(x, y):
    return (int(round(x * S)), int(round(y * S)))


def angle_profile() -> list[float]:
    """Sim-frame blade angle per frame (rad, unwrapped)."""
    a, w = 0.4, 0.0
    out = []
    for f in range(N):
        if 15 <= f < 55:
            w = max(w - 450 / FPS, -26.0)  # spin clockwise (negative) at the cap
        elif 55 <= f < 58:
            w = min(w + 900 / FPS, 0.0)  # hard stop
        elif 70 <= f < 95:
            w = min(w + 450 / FPS, 26.0)  # reversal
        elif 95 <= f < 98:
            w = max(w - 900 / FPS, 0.0)
        elif 110 <= f < 140:
            w = 3.0  # slow sweep
        else:
            w = 0.0
        a += w / FPS
        out.append(a)
    return out


def sim_to_screen(theta: float) -> tuple[float, float]:
    return -math.sin(theta), -math.cos(theta)  # (du, dv) unit vector


def render(out: Path) -> dict:
    out = Path(out)
    bg = np.full((H, W, 3), ICE, np.uint8)
    for r in range(int(2.2 * R), int(R), -1):  # glow
        k = (r - R) / (1.2 * R)
        col = tuple(int(g * (1 - k) + i * k) for g, i in zip(GLOW, ICE))
        cv2.circle(bg, _P(*C), int(r * S), col, -1, cv2.LINE_AA, SHIFT)
    truth = profile = angle_profile()
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
    prev = None
    rows = []
    for f in range(N):
        if f == DUP_FRAME:
            wr.write(prev)
            rows.append(None)
            continue
        img = bg.copy()
        cv2.circle(img, _P(*C), int(R * S), BODY, -1, cv2.LINE_AA, SHIFT)
        cv2.circle(img, _P(*C), int(R * S), (50, 50, 50), 2, cv2.LINE_AA, SHIFT)
        face = 0.3 + 0.8 * math.sin(f / 25)  # body markings turn on their own
        for k, (rr, da) in enumerate(((0.45, 0.0), (0.45, 0.6), (0.25, -0.9))):
            du, dv = sim_to_screen(face + da)
            p = (C[0] + rr * R * du, C[1] + rr * R * dv)
            cv2.circle(img, _P(*p), int(4 * S), (235, 235, 235), -1, cv2.LINE_AA, SHIFT)
        th = profile[f]
        du, dv = sim_to_screen(th)
        nu, nv = -dv, du  # perpendicular
        root = (C[0] + 0.98 * R * du, C[1] + 0.98 * R * dv)
        tip = (C[0] + 1.62 * R * du, C[1] + 1.62 * R * dv)
        inner = (C[0] + 0.55 * R * (du * math.cos(1.0) + nu * math.sin(1.0)), C[1] + 0.55 * R * (dv * math.cos(1.0) + nv * math.sin(1.0)))
        cv2.line(img, _P(*inner), _P(*root), (12, 12, 12), 7, cv2.LINE_AA, SHIFT)  # black shaft across the body
        cv2.line(img, _P(*root), _P(*tip), (250, 250, 250), 6, cv2.LINE_AA, SHIFT)  # white blade
        cv2.circle(img, _P(C[0] + 90, C[1] - 75), int(9 * S), (40, 210, 40), -1, cv2.LINE_AA, SHIFT)  # puck
        cv2.circle(img, _P(C[0] + 58, C[1] + 58), int(17 * S), (30, 30, 30), -1, cv2.LINE_AA, SHIFT)  # mouse-hint icon
        cv2.ellipse(img, _P(C[0] + 58, C[1] + 58), (int(6 * S), int(9 * S)), 0, 0, 360, (240, 240, 240), 2, cv2.LINE_AA, SHIFT)
        cv2.rectangle(img, (10, 10), (150, 40), (250, 250, 250), -1)  # HUD
        cv2.putText(img, "AWAY 0-0", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
        wr.write(img)
        prev = img
        rows.append(th)
    wr.release()
    meta = {"angle": rows, "fps": FPS, "body_r": R, "tip_r": 1.62 * R + 3.0, "dup_frame": DUP_FRAME}  # visible tip: line end + half its 6 px width
    out.with_suffix(".truth.json").write_text(json.dumps(meta))
    return meta
