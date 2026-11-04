"""End-to-end on a synthetic video: map, then track; the puck path must come back within 1 px
and no distractor may ever be reported as the puck. Needs no real clips."""

import json
import math
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cv2

from measure import synth
from measure.geometry import PARAMS_PATH, RinkMap
from measure.rinkmap import run_map
from measure.track import Detector, run_track, static_exclusion
from measure.video import iter_frames


class SyntheticTrackerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="measure-test-"))
        cls.clip = cls.dir / "synthetic.mkv"
        cls.truth = synth.render(cls.clip)
        with redirect_stdout(StringIO()):
            cls.map = run_map(cls.clip, cls.dir, PARAMS_PATH)
            cls.rows, cls.meta = run_track(cls.clip, cls.dir, PARAMS_PATH)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_map(self):
        m = self.map
        self.assertAlmostEqual(m.origin_px[0], synth.CENTER[0], delta=0.5)
        self.assertAlmostEqual(m.origin_px[1], synth.CENTER[1], delta=0.5)
        b = synth.BOARD
        self.assertAlmostEqual(m.board_length_px, b["bottom"] - b["top"], delta=1.0)
        self.assertAlmostEqual(m.board_width_px, b["right"] - b["left"], delta=1.0)
        self.assertTrue(m.extra["drift"]["ok"])
        self.assertLess(m.extra["drift"]["max_px"], 0.1)

    def test_path_within_1px(self):
        frames = self.truth["frames"]
        errs = []
        for r in self.rows:
            t = frames[r["frame"]]
            self.assertIsNotNone(t, f"frame {r['frame']} is the duplicated frame and must be dropped")
            errs.append(math.hypot(r["u"] - t[0], r["v"] - t[1]))
        self.assertLess(max(errs), 1.0, f"max error {max(errs):.3f} px")

    def test_detection_rate(self):
        visible = sum(1 for t in self.truth["frames"] if t is not None)
        detected = sum(1 for r in self.rows if not r["interpolated"])
        self.assertGreaterEqual(detected, visible - 1)

    def test_duplicate_frame_dropped(self):
        self.assertIn(synth.DUP_FRAME, self.meta["summary"]["duplicate_frames"])
        self.assertNotIn(synth.DUP_FRAME, [r["frame"] for r in self.rows])

    def test_distractors_visible_but_never_chosen(self):
        # The detector does see the in-rink distractors (so the tracker logic, not luck, rejects them).
        m = RinkMap.load(self.dir / "synthetic.map.json")
        det = Detector(m.rink_mask(8.0) > 0, static=static_exclusion(cv2.imread(str(self.dir / "synthetic.bg.png"))))
        lights = [synth.STATIC_LIGHT, *synth.BLINK_SPOTS]
        seen = set()
        for i, f in iter_frames(self.clip, every=10):
            for c in det.candidates(f):
                for k, (lu, lv) in enumerate(lights):
                    if math.hypot(c[0] - lu, c[1] - lv) < 2:
                        seen.add(k)
        self.assertEqual(seen, {0, 1, 2})
        for r in self.rows:
            pu, pv = synth.pink_path(r["frame"])
            self.assertGreater(math.hypot(r["u"] - pu, r["v"] - pv), 12, "pink disc reported as puck")
            for lu, lv in lights:
                self.assertGreater(math.hypot(r["u"] - lu, r["v"] - lv), 12, "green light reported as puck")
            self.assertTrue(m.rink_mask(8.0)[int(r["v"]), int(r["u"])] > 0, "HUD / outside-rink point reported")

    def test_csv_format(self):
        lines = (self.dir / "synthetic.traj.csv").read_text().splitlines()
        self.assertEqual(lines[0], "t,x,y,z")
        self.assertEqual(len(lines) - 1, len(self.rows))
        m = self.map
        for line, r in zip(lines[1:], self.rows):
            t, x, y, z = map(float, line.split(","))
            self.assertEqual(z, 0.0)
            self.assertAlmostEqual(t, r["frame"] / synth.FPS, places=12)
            tu, tv = self.truth["frames"][r["frame"]]
            tx, ty = m.px_to_m(tu, tv)
            self.assertLess(math.hypot(x - tx, y - ty), 1.0 * m.m_per_px)
        meta = json.loads((self.dir / "synthetic.track.json").read_text())
        self.assertIn("PLACEHOLDER", meta["scale_source"])
        self.assertEqual(len(meta["rows"]["conf"]), len(self.rows))


if __name__ == "__main__":
    unittest.main()
