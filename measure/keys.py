"""Key display reader for Ian's s7 clips (a Keyviz-style overlay, bottom right of the screen).

What the overlay does (checked frame by frame, CLAUDE.md "Session s7"): every key press adds a white keycap at the
right end of a row; the cap appears RAISED (dark bottom edge visible, face top at y = 29 px in the region below),
sinks FLAT (face top at y = 37) within 3-4 frames while the key is held, pops back up to RAISED when the key is
released, and stays visible ~1.0-1.1 s before it disappears. A re-press while it lingers adds a new cap.
So, per cap: held from the frame it appears (or starts sinking again) until the frame it starts rising.

Labels come from glyph templates (24x24 dark-pixel maps) cut from labelled reference caps (REFS), stored in
measure/key_templates.json (`python -m measure.skate templates`). Unknown caps are labelled "?".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry import REPO

REGION = (760, 900, 900, 1800)  # y0, y1, x0, x1 in 1080p frames
TEMPLATES = REPO / "measure" / "key_templates.json"
MIN_SCORE = 0.75
FLAT_Y = 36  # face top (region px) at or below this row = sunk/held
# Reference caps: (label, clip, frame, (x, y, w, h) in REGION px). Identified by eye on the frames (see CLAUDE.md).
REFS = [
    ("R", "wkeymovement", 0, (566, 29, 68, 68)), ("Ctrl", "wkeymovement", 0, (648, 37, 76, 68)),
    ("4", "wkeymovement", 0, (737, 37, 68, 68)), ("Alt", "wkeymovement", 0, (476, 38, 76, 67)),
    ("W", "wkeymovement", 42, (737, 29, 68, 68)), ("5", "wkeymovement", 2145, (737, 35, 68, 68)),
    ("D", "diagonalkeypresses", 57, (656, 30, 68, 67)), ("S", "diagonalkeypresses", 1716, (737, 30, 68, 68)),
    ("C", "diagonalkeypresses", 2121, (737, 29, 68, 68)), ("A", "diagonalkeypresses", 2154, (656, 30, 68, 68)),
    ("D", "whilemovingturning", 1452, (737, 30, 68, 67)), ("D", "whilemovingturning", 1506, (737, 37, 68, 67)),
    ("Shift", "dashatrest", 81, (729, 30, 76, 67)), ("Shift", "dashatrest", 453, (729, 29, 76, 68)),
    ("CapsLock", "dashwhilerunning", 1845, (693, 35, 112, 67)),
]
MOVE_KEYS = ("W", "A", "S", "D", "Shift", "Space")


def glyph(face_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    m = (g < 140).astype(np.float32)
    m[:4, :] = 0
    m[-4:, :] = 0
    m[:, :4] = 0
    m[:, -4:] = 0
    v = cv2.resize(m, (24, 24), interpolation=cv2.INTER_AREA).ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def cap_boxes(region_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 225) & (hsv[..., 1] < 35)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(white)
    return sorted((int(x), int(y), int(w), int(h)) for x, y, w, h, a in st[1:] if a >= 1500 and 55 <= h <= 80)


def load_templates(path: Path = TEMPLATES) -> list[tuple[str, int, np.ndarray]]:
    raw = json.loads(Path(path).read_text())
    return [(t["label"], t["width"], np.array(t["glyph"], np.float32)) for t in raw["templates"]]


def build_templates(clips_dir: Path, path: Path = TEMPLATES) -> None:
    from .video import iter_frames

    out = []
    for label, clip, frame, (x, y, w, h) in REFS:
        for k, f in iter_frames(clips_dir / f"{clip}.mkv"):
            if k == frame:
                y0, _, x0, _ = REGION
                out.append({"label": label, "clip": clip, "frame": frame, "width": w,
                            "glyph": [round(float(v), 5) for v in glyph(f[y0 + y : y0 + y + h, x0 + x : x0 + x + w])]})
                break
    Path(path).write_text(json.dumps({"note": "keycap glyph templates (measure/keys.py REFS)", "templates": out}))


@dataclass
class Cap:
    x: int
    y: int
    w: int
    h: int
    label: str
    score: float


def read_caps(frame: np.ndarray, templates) -> list[Cap]:
    y0, y1, x0, x1 = REGION
    reg = frame[y0:y1, x0:x1]
    caps = []
    for x, y, w, h in cap_boxes(reg):
        g = glyph(reg[y : y + h, x : x + w])
        best, score = "?", 0.0
        for label, tw, tg in templates:
            if abs(tw - w) > 10:
                continue
            sc = float(g @ tg)
            if sc > score:
                best, score = label, sc
        caps.append(Cap(x, y, w, h, best if score >= MIN_SCORE else "?", score))
    return caps


@dataclass
class _Tracked:
    label: str
    x: int
    y: int
    held: bool


@dataclass
class KeyTracker:
    """Per-frame held state of every labelled key from the cap sequence."""

    caps: list[_Tracked] = field(default_factory=list)
    first: bool = True

    def update(self, now: list[Cap]) -> dict[str, bool]:
        new: list[_Tracked] = []
        labels = {c.label for c in now} | {c.label for c in self.caps}
        for lab in labels:
            prev = sorted([c for c in self.caps if c.label == lab], key=lambda c: c.x)
            cur = sorted([c for c in now if c.label == lab], key=lambda c: c.x)
            if len(cur) >= len(prev):
                pairs = list(zip(prev, cur[: len(prev)]))
                born = cur[len(prev) :]  # a new press appends a cap at the right end
            else:
                pairs = list(zip(prev[len(prev) - len(cur) :], cur))  # the oldest (leftmost) expired
                born = []
            for p, c in pairs:
                held = p.held
                if c.y > p.y + 0.5:
                    held = True  # sinking: pressed
                elif c.y < p.y - 0.5:
                    held = False  # rising: released
                new.append(_Tracked(lab, c.x, c.y, held))
            for c in born:
                new.append(_Tracked(lab, c.x, c.y, True if not self.first else c.y >= FLAT_Y))
        self.caps = new
        self.first = False
        state: dict[str, bool] = {}
        for c in new:
            state[c.label] = state.get(c.label, False) or c.held
        return state


def events_from_states(states: list[dict[str, bool]], keys=MOVE_KEYS + ("R",)) -> list[dict]:
    """Down/up edges per key: [{"frame", "key", "down": bool}]."""
    out = []
    prev = {k: False for k in keys}
    for i, st in enumerate(states):
        for k in keys:
            v = bool(st.get(k, False))
            if v != prev[k]:
                out.append({"frame": i, "key": k, "down": v})
                prev[k] = v
    return out
