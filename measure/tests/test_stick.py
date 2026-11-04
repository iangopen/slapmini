"""Stick tracker on a synthetic video: a stick turning on a known profile (cap-speed spins, hard stops, a slow
sweep) with separately turning body markings, a puck, a screen-fixed icon, a HUD and a duplicated frame.
The blade angle must come back within 1 degree."""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from measure import stick_synth
from measure.stick import track_stick


class SyntheticStickTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="stick-test-"))
        cls.clip = cls.dir / "sticksynth.mkv"
        cls.truth = stick_synth.render(cls.clip)
        cls.rows, cls.rep = track_stick(cls.clip)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_angle_within_1_degree(self):
        by = {r["frame"]: r for r in self.rows if r["ok"]}
        errs = []
        for f, th in enumerate(self.truth["angle"]):
            if th is None:
                self.assertNotIn(f, by, "the duplicated frame must be dropped")
                continue
            self.assertIn(f, by, f"frame {f} not tracked")
            errs.append(abs(math.degrees(math.remainder(by[f]["angle"] - th, 2 * math.pi))))
        self.assertLess(max(errs), 1.0, f"max error {max(errs):.3f} deg")

    def test_unwrapped_follows_spins(self):
        by = {r["frame"]: r for r in self.rows}
        t = self.truth["angle"]
        # The spin turns ~ -14 rad; the unwrapped track must follow it, not alias.
        d_true = t[60] - t[10]
        d_meas = by[60]["angle_unwrapped"] - by[10]["angle_unwrapped"]
        self.assertAlmostEqual(d_meas, d_true, delta=math.radians(1.0))

    def test_report_and_geometry(self):
        r = self.rep
        self.assertEqual(r["duplicate_frames_dropped"], 1)
        self.assertEqual(r["longest_gap_frames"], 0)
        self.assertAlmostEqual(r["detection_rate"], 1.0)
        self.assertAlmostEqual(r["body_radius_px"], stick_synth.R, delta=1.5)
        self.assertAlmostEqual(r["tip_radius_px"], self.truth["tip_r"], delta=3.0)
        for row in self.rows:
            self.assertAlmostEqual(row["pivot_u"], stick_synth.C[0], delta=1.0)
            self.assertAlmostEqual(row["pivot_v"], stick_synth.C[1], delta=1.0)


if __name__ == "__main__":
    unittest.main()
