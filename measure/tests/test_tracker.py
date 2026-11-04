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
        d = m.extra["drift"]
        self.assertTrue(d["ok"])
        self.assertTrue(d["camera_moves"])  # the 1 s sample at frame 120 is inside the pan
        self.assertAlmostEqual(d["max_px"], math.hypot(synth.PAN[2], synth.PAN[3]), delta=0.2)
        self.assertLess(d["max_residual_after_correction_px"], 0.3)
        self.assertTrue(m.extra["shape"]["circles"]["summary"]["round"])

    def test_camera_pan_corrected(self):
        s = self.meta["summary"]
        self.assertEqual(s["camera_moved_frames"], 0, "every panned frame must be corrected, not dropped")
        self.assertEqual(s["frames_kept_without_correction"], synth.N - (synth.PAN[1] - synth.PAN[0]))
        self.assertLess(s["camera_residual_after_correction_px"]["max"], 0.3)
        pan = [r for r in self.rows if synth.PAN[0] <= r["frame"] < synth.PAN[1]]
        self.assertGreaterEqual(len(pan), synth.PAN[1] - synth.PAN[0] - 1)
        for r in pan:
            self.assertAlmostEqual(r["shift_u"], synth.PAN[2], delta=0.2)
            self.assertAlmostEqual(r["shift_v"], synth.PAN[3], delta=0.2)

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

    def test_segment(self):
        """One free stretch: rest, hit at frame 20, glide, bounce off the top board, glide."""
        from measure.segment import run_segment

        dest = self.dir / "real"
        with redirect_stdout(StringIO()):
            man = run_segment([self.clip], self.dir, dest)
        segs = man["segments"]
        self.assertEqual(len(segs), 1, segs)
        s = segs[0]
        self.assertEqual(s["label"], "wall bounce")
        self.assertGreater(s["start_frame"], 20)  # starts after the hit, once the puck is free
        self.assertLess(s["start_frame"], 30)
        # truth bounce: the frame where the vertical velocity flips sign
        fr = self.truth["frames"]
        bounce = next(f for f in range(22, synth.N - 1) if fr[f - 1] and fr[f] and fr[f + 1] and (fr[f][1] - fr[f - 1][1]) < 0 < (fr[f + 1][1] - fr[f][1]))
        self.assertEqual(len(s["wall_bounce_frames"]), 1)
        self.assertLessEqual(abs(s["wall_bounce_frames"][0] - bounce), 2)
        sc = json.loads((dest / s["scenario"]).read_text())
        p = sc["initial_state"]["puck"]
        tu, tv = fr[s["start_frame"]]
        tx, ty = self.map.px_to_m(tu, tv)
        self.assertLess(math.hypot(p["x"] - tx, p["y"] - ty), 0.5 * self.map.m_per_px)
        # initial velocity: truth pixel velocity (per frame) mapped to m/s
        (u0, v0), (u1, v1) = fr[s["start_frame"]], fr[s["start_frame"] + 1]
        vx_t, vy_t = (v0 - v1) * self.map.m_per_px * synth.FPS, (u0 - u1) * self.map.m_per_px * synth.FPS
        self.assertLess(math.hypot(p["vx"] - vx_t, p["vy"] - vy_t), 0.03 * math.hypot(vx_t, vy_t))
        self.assertEqual(sc["initial_state"]["players"], [])

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
