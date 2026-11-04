"""Pixel <-> meter mapping. The single coordinate convention for every measure output.

Convention (also in CLAUDE.md):
- meters, origin at center ice (the center dot),
- +x along the rink's long axis, pointing at the net at the TOP of the screen,
- +y chosen so that x then y turns counterclockwise as seen on screen, i.e. +y points to screen LEFT.
This matches the sim (+x toward net 0, y = x rotated 90 deg CCW); the lab's horizontal view is the
game view rotated 90 deg clockwise.

Scale: m_per_px = params.json rink_length (a PLACEHOLDER) / measured inner board length in pixels.
It is recorded as `scale_source` in every output, because it is unmeasured.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PARAMS_PATH = REPO / "params.json"

CONVENTION = (
    "meters; origin = center dot; +x along the long axis toward the net at the top of the screen; "
    "+y = screen left (x then y counterclockwise on screen); z not observable (always 0)"
)


def load_params(path: Path = PARAMS_PATH) -> dict[str, float]:
    raw = json.loads(Path(path).read_text())
    return {k: v["value"] for k, v in raw["params"].items()}


def ccw_on_screen(ax: tuple[float, float]) -> tuple[float, float]:
    """Rotate a pixel-space vector 90 deg counterclockwise as seen on screen (pixel v points down)."""
    return (ax[1], -ax[0])


@dataclass
class RinkMap:
    clip: str
    frame_size: tuple[int, int]
    fps: float
    frame_count: int
    origin_px: tuple[float, float]
    x_axis_px: tuple[float, float]  # unit pixel vector of +x
    board_px: dict  # inner board edges: left, right, top, bottom (pixels)
    corner_radius_px: float
    m_per_px: float
    scale_source: str
    extra: dict = field(default_factory=dict)

    @property
    def y_axis_px(self) -> tuple[float, float]:
        return ccw_on_screen(self.x_axis_px)

    @property
    def board_length_px(self) -> float:
        b = self.board_px
        return abs(self._along(b, self.x_axis_px))

    @property
    def board_width_px(self) -> float:
        b = self.board_px
        return abs(self._along(b, self.y_axis_px))

    @staticmethod
    def _along(b: dict, ax: tuple[float, float]) -> float:
        # extent of the board box along a pixel axis that is horizontal or vertical
        return (b["right"] - b["left"]) * abs(ax[0]) + (b["bottom"] - b["top"]) * abs(ax[1])

    # ---- mapping ----
    def px_to_m(self, u, v):
        u = np.asarray(u, dtype=float) - self.origin_px[0]
        v = np.asarray(v, dtype=float) - self.origin_px[1]
        ex, ey = self.x_axis_px, self.y_axis_px
        x = (u * ex[0] + v * ex[1]) * self.m_per_px
        y = (u * ey[0] + v * ey[1]) * self.m_per_px
        return x, y

    def m_to_px(self, x, y):
        x = np.asarray(x, dtype=float) / self.m_per_px
        y = np.asarray(y, dtype=float) / self.m_per_px
        ex, ey = self.x_axis_px, self.y_axis_px
        return self.origin_px[0] + x * ex[0] + y * ey[0], self.origin_px[1] + x * ex[1] + y * ey[1]

    def rink_mask(self, margin_px: float = 0.0) -> np.ndarray:
        """uint8 mask (255 inside) of the inner board outline (rounded rectangle), grown by margin_px."""
        import cv2

        w, h = self.frame_size
        b = self.board_px
        r = max(0.0, self.corner_radius_px + margin_px)
        x0, x1 = b["left"] - margin_px, b["right"] + margin_px
        y0, y1 = b["top"] - margin_px, b["bottom"] + margin_px
        S = 16  # cv2 fixed-point shift factor (4 bits)
        m = np.zeros((h, w), np.uint8)

        def P(x, y):
            return (int(round(x * S)), int(round(y * S)))

        cv2.rectangle(m, P(x0 + r, y0), P(x1 - r, y1), 255, -1, cv2.LINE_8, 4)
        cv2.rectangle(m, P(x0, y0 + r), P(x1, y1 - r), 255, -1, cv2.LINE_8, 4)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(m, P(cx, cy), int(round(r * S)), 255, -1, cv2.LINE_8, 4)
        return m

    # ---- io ----
    def to_json(self) -> dict:
        d = asdict(self)
        d["y_axis_px"] = list(self.y_axis_px)
        d["board_length_px"] = self.board_length_px
        d["board_width_px"] = self.board_width_px
        d["convention"] = CONVENTION
        return d

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "RinkMap":
        d = json.loads(Path(path).read_text())
        keys = {f for f in cls.__dataclass_fields__}
        d = {k: v for k, v in d.items() if k in keys}
        d["frame_size"] = tuple(d["frame_size"])
        d["origin_px"] = tuple(d["origin_px"])
        d["x_axis_px"] = tuple(d["x_axis_px"])
        return cls(**d)


def scale_from_params(board_length_px: float, params: dict[str, float]) -> tuple[float, str]:
    L = params["rink_length"]
    mpp = L / board_length_px
    src = (
        f"params.json rink_length = {L:g} m (PLACEHOLDER, unmeasured) / measured inner board length "
        f"{board_length_px:.2f} px"
    )
    return mpp, src


def length_width_ratio(params: dict[str, float]) -> float:
    return params["rink_length"] / params["rink_width"]


def fmt_ratio(measured: float, params: dict[str, float]) -> str:
    p = length_width_ratio(params)
    return (
        f"rink length/width: measured {measured:.3f}  vs  params.json {params['rink_length']:g}/"
        f"{params['rink_width']:g} = {p:.3f}  (ratio of ratios {measured / p:.3f})"
    )

