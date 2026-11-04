"""Synthetic test video for the player tracker (measure/player.py).

Built on measure.synth's rink (same board, faceoff landmarks, center line). Contents:
- the player: a pink disc (r = 7 px) with a white stick that sweeps around it and crosses the disc,
  standing still, then accelerating, gliding at a plateau, stopping;
- a blue glow marker painted on the ice at the spawn point (stays put, like the real game);
- static distractors: small pink board lights inside the rink, a pink HUD badge outside it,
  and a resting green puck;
- a stretch where the camera FOLLOWS the player (disc fixed on screen, ice content shifts, as in the
  real game), plus one duplicated capture frame.
Truth = the player's position in reference (un-shifted) pixels, per frame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from . import synth

W, H, FPS, N = synth.W, synth.H, synth.FPS, 160
DISC_R = 7.0
PINK = (225, 120, 245)  # BGR; HSV hue ~150, sat ~130 like the real disc
GLOW = (230, 120, 40)
STICK = (240, 240, 240)
SPAWN = (480.0, 330.0)
FOLLOW = (100, 135)  # frames [100, 135): the camera follows the player
DUP_FRAME = 60
PINK_LIGHTS = [(338.0, 270.0), (622.0, 270.0)]  # small lights on the boards at the center line
HUD_BADGE = (100, 300)


def player_path() -> list[tuple[float, float]]:
    """Rest 20 frames, accelerate along +u for 25 frames to 2.5 px/frame, glide, brake to rest, then a
    slow diagonal creep (the camera-follow stretch)."""
    u, v = SPAWN
    vu = 0.0
    out = []
    for f in range(N):
        if 20 <= f < 45:
            vu = min(vu + 0.1, 2.5)
        elif 60 <= f < 70:
            vu = max(vu - 0.25, 0.0)
        u += vu
        if FOLLOW[0] <= f < FOLLOW[1]:
            u -= 0.35
            v += 0.3
        out.append((u, v))
    return out


def render(out: Path) -> dict:
    out = Path(out)
    bg = synth.background()
    cv2.circle(bg, synth._P(*SPAWN), int(12 * synth.S), GLOW, -1, cv2.LINE_AA, synth.SHIFT)  # spawn marker on the ice
    for lu, lv in PINK_LIGHTS:
        cv2.circle(bg, synth._P(lu, lv), int(2 * synth.S), PINK, -1, cv2.LINE_AA, synth.SHIFT)
    path = player_path()
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
    if not wr.isOpened():
        raise RuntimeError("FFV1 writer unavailable in this OpenCV build")
    truth, shifts = [], []
    prev = None
    for f in range(N):
        if f == DUP_FRAME:
            wr.write(prev)
            truth.append(None)
            shifts.append(None)
            continue
        img = bg.copy()
        cv2.circle(img, synth._P(420.0, 200.0), int(synth.PUCK_R * synth.S), synth.PUCK_COL, -1, cv2.LINE_AA, synth.SHIFT)
        pu, pv = path[f]
        cv2.circle(img, synth._P(pu, pv), int(DISC_R * synth.S), PINK, -1, cv2.LINE_AA, synth.SHIFT)
        ang = 0.15 * f
        cv2.line(img, synth._P(pu - 3 * math.cos(ang), pv - 3 * math.sin(ang)), synth._P(pu + 22 * math.cos(ang), pv + 22 * math.sin(ang)),
                 STICK, 2, cv2.LINE_AA, synth.SHIFT)
        du = dv = 0.0
        if FOLLOW[0] <= f < FOLLOW[1]:  # camera keeps the player where it was at the start of the stretch
            du, dv = path[FOLLOW[0] - 1][0] - pu, path[FOLLOW[0] - 1][1] - pv
            M = np.float32([[1, 0, du], [0, 1, dv]])
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cv2.rectangle(img, (HUD_BADGE[0] - 30, HUD_BADGE[1] - 12), (HUD_BADGE[0] + 30, HUD_BADGE[1] + 12), PINK, -1)  # HUD: screen-fixed
        wr.write(img)
        prev = img
        truth.append([pu, pv])
        shifts.append([du, dv])
    wr.release()
    meta = {"frames": truth, "shifts": shifts, "fps": FPS, "dup_frame": DUP_FRAME, "follow": FOLLOW, "spawn": SPAWN}
    out.with_suffix(".truth.json").write_text(json.dumps(meta))
    return meta
