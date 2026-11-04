import math
import unittest

import numpy as np

from measure.geometry import RinkMap, ccw_on_screen, load_params, scale_from_params


def make_map() -> RinkMap:
    b = {"left": 668.5, "right": 1250.5, "top": 59.5, "bottom": 972.5}
    mpp, src = scale_from_params(b["bottom"] - b["top"], {"rink_length": 60.0})
    return RinkMap("t", (1920, 1080), 60.0, 100, (959.47, 515.57), (0.0, -1.0), b, 132.8, mpp, src)


class GeometryTest(unittest.TestCase):
    def test_round_trip(self):
        m = make_map()
        rng = np.random.default_rng(1)
        u = rng.uniform(0, 1920, 1000)
        v = rng.uniform(0, 1080, 1000)
        x, y = m.px_to_m(u, v)
        u2, v2 = m.m_to_px(x, y)
        self.assertLess(np.abs(u2 - u).max(), 1e-9)
        self.assertLess(np.abs(v2 - v).max(), 1e-9)
        x2, y2 = m.px_to_m(*m.m_to_px(x, y))
        self.assertLess(np.abs(x2 - x).max(), 1e-9)
        self.assertLess(np.abs(y2 - y).max(), 1e-9)

    def test_convention(self):
        m = make_map()
        x, y = m.px_to_m(*m.origin_px)
        self.assertAlmostEqual(float(x), 0.0)
        self.assertAlmostEqual(float(y), 0.0)
        u0, v0 = m.origin_px
        x, y = m.px_to_m(u0, v0 - 100)  # up on screen = +x, toward the top net
        self.assertGreater(x, 0)
        self.assertAlmostEqual(float(y), 0.0)
        x, y = m.px_to_m(u0 - 100, v0)  # screen left = +y
        self.assertGreater(y, 0)
        self.assertAlmostEqual(float(x), 0.0)
        # x then y is counterclockwise on screen: in a y-up frame the cross product is positive.
        ex, ey = m.x_axis_px, m.y_axis_px
        self.assertEqual(ey, ccw_on_screen(ex))
        cross_y_up = ex[0] * (-ey[1]) - (-ex[1]) * ey[0]
        self.assertGreater(cross_y_up, 0)

    def test_scale_is_rink_length_over_board_px(self):
        m = make_map()
        b = m.board_px
        # The full inner board length maps to exactly rink_length.
        xt, _ = m.px_to_m(m.origin_px[0], b["top"])
        xb, _ = m.px_to_m(m.origin_px[0], b["bottom"])
        self.assertAlmostEqual(float(xt - xb), 60.0, places=9)
        self.assertIn("PLACEHOLDER", m.scale_source)

    def test_params_rink_length_used(self):
        p = load_params()
        mpp, src = scale_from_params(1000.0, p)
        self.assertTrue(math.isclose(mpp, p["rink_length"] / 1000.0))
        self.assertIn("rink_length", src)

    def test_json_round_trip(self):
        import tempfile
        from pathlib import Path

        m = make_map()
        with tempfile.TemporaryDirectory() as d:
            m.save(Path(d) / "a.map.json")
            m2 = RinkMap.load(Path(d) / "a.map.json")
        self.assertEqual(m2.origin_px, m.origin_px)
        self.assertEqual(m2.m_per_px, m.m_per_px)
        self.assertEqual(m2.px_to_m(10.0, 20.0), m.px_to_m(10.0, 20.0))


if __name__ == "__main__":
    unittest.main()
