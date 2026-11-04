"""Player tracker end-to-end on a synthetic video: a moving pink disc with a stick crossing it, static pink
distractors, a camera-follow stretch and a duplicated frame. The disc must come back within 1 px."""

import math
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from measure import player_synth
from measure.geometry import PARAMS_PATH
from measure.player import run_track
from measure.rinkmap import run_map


class SyntheticPlayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="player-test-"))
        cls.clip = cls.dir / "playersynth.mkv"
        cls.truth = player_synth.render(cls.clip)
        with redirect_stdout(StringIO()):
            run_map(cls.clip, cls.dir, PARAMS_PATH)
            cls.rows, cls.rep = run_track(cls.clip, cls.dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_tracked_within_1px(self):
        by = {r["frame"]: r for r in self.rows if not r["interpolated"]}
        errs = []
        for f, t in enumerate(self.truth["frames"]):
            if t is None:
                self.assertNotIn(f, by, "the duplicated frame must be dropped")
                continue
            self.assertIn(f, by, f"frame {f} not detected")
            errs.append(math.hypot(by[f]["u"] - t[0], by[f]["v"] - t[1]))
        self.assertLess(max(errs), 1.0, f"max error {max(errs):.3f} px")
        self.assertLess(sorted(errs)[len(errs) // 2], 0.3)

    def test_camera_follow_recovered(self):
        f0, f1 = self.truth["follow"]
        by = {r["frame"]: r for r in self.rows}
        for f in range(f0, f1):
            du, dv = self.truth["shifts"][f]
            self.assertAlmostEqual(by[f]["shift_u"], du, delta=0.3)
            self.assertAlmostEqual(by[f]["shift_v"], dv, delta=0.3)
            # on screen the disc stays put while the camera follows
            self.assertAlmostEqual(by[f]["raw_u"], self.truth["frames"][f0 - 1][0], delta=1.0)

    def test_report(self):
        r = self.rep
        self.assertEqual(r["duplicate_frames_dropped"], 1)
        self.assertEqual(r["longest_gap_frames"], 0)
        self.assertAlmostEqual(r["detection_rate"], 1.0)
        self.assertFalse(r["height_correction"]["applied"])

    def test_distractors_never_reported(self):
        for r in self.rows:
            for lu, lv in player_synth.PINK_LIGHTS:
                self.assertGreater(math.hypot(r["u"] - lu, r["v"] - lv), 10)


if __name__ == "__main__":
    unittest.main()
