"""Drawing of the stick chain on the sim's own geometry (`python -m measure.stick_figure`) -> measure/results/stick_chain.png.

Everything comes from params.json: player and puck circles, the three capsules, the old straight rod for comparison.
No footage is used.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import REPO, load_params

OUT = REPO / "measure" / "results" / "stick_chain.png"
S = 420  # px per m
W, H, OX, OY = 900, 620, 300, 400


def px(x: float, y: float) -> tuple[int, int]:
    return int(OX + x * S), int(OY - y * S)


def main() -> None:
    p = load_params()
    img = np.full((H, W, 3), 245, np.uint8)
    cv2.line(img, px(-0.7, 0), px(1.4, 0), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(img, px(0, 0), int(p["player_radius"] * S), (170, 170, 170), -1, cv2.LINE_AA)
    cv2.circle(img, px(0, 0), 4, (60, 60, 60), -1, cv2.LINE_AA)
    pts = [(p[f"stick_{n}_x"], p[f"stick_{n}_y"]) for n in ("hand", "kink", "heel", "toe")]
    rad = [p["stick_shaft_radius"], p["stick_heel_radius"], p["stick_thickness"]]
    for a, b, r, col in zip(pts, pts[1:], rad, [(40, 40, 40), (120, 90, 40), (200, 130, 40)]):
        cv2.line(img, px(*a), px(*b), col, max(1, int(2 * r * S)), cv2.LINE_AA)
    cv2.line(img, px(0.4, 0), px(1.1825, 0), (60, 60, 220), 2, cv2.LINE_AA)  # the old rod, centre line
    cv2.circle(img, px(1.0, -p["puck_radius"] - p["stick_thickness"]), int(p["puck_radius"] * S), (70, 200, 70), 2, cv2.LINE_AA)
    for i, t in enumerate(["grey: body  black: shaft  brown: heel  blue: blade", "red line: old rod  green: puck touching the blade", "stick frame, +x = stick_angle, 420 px/m, from params.json"]):
        cv2.putText(img, t, (20, 28 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT), img)
    print(OUT)


if __name__ == "__main__":
    main()
