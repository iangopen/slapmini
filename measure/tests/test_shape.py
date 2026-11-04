"""Circle roundness check: round circles pass, a squashed one (tilted camera) is reported, not hidden."""

import unittest

import cv2
import numpy as np

from measure.shape import ROUND_TOL, fit_ring


def ring_image(rx: float, ry: float) -> np.ndarray:
    img = np.full((400, 400, 3), (47, 56, 58), np.uint8)
    cv2.ellipse(img, (int(200.3 * 16), int(199.6 * 16)), (int(rx * 16), int(ry * 16)), 0, 0, 360, (88, 90, 92), 2, cv2.LINE_AA, 4)
    return img


class ShapeTest(unittest.TestCase):
    def test_round_circle(self):
        f = fit_ring(ring_image(72, 72), (200.0, 200.0))
        self.assertIsNotNone(f)
        self.assertLess(abs(f["v_over_h"] - 1), 0.005)
        self.assertAlmostEqual(f["horizontal_diameter_px"], 144, delta=1.0)
        self.assertAlmostEqual(f["center_px"][0], 200.3, delta=0.3)

    def test_squashed_circle_detected(self):
        f = fit_ring(ring_image(72, 64.8), (200.0, 200.0))  # v/h = 0.9
        self.assertIsNotNone(f)
        self.assertAlmostEqual(f["v_over_h"], 0.9, delta=0.01)
        self.assertGreater(abs(f["v_over_h"] - 1), ROUND_TOL)


if __name__ == "__main__":
    unittest.main()
