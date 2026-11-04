"""Moving-camera registration (measure/camreg.py) and player/puck world positions (measure/skate.py) on a
synthetic panning video, and the key-display state machine (measure/keys.py) on a scripted cap sequence."""

import csv
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from measure import pan_synth
from measure.keys import Cap, KeyTracker, events_from_states
from measure.skate import track_clip


class SyntheticPanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="pan-test-"))
        cls.clip = cls.dir / "pansynth.mkv"
        cls.truth = pan_synth.render(cls.clip)
        cls.summary = track_clip(cls.clip, cls.dir, read_keys=False)
        with open(cls.dir / "pansynth.frames.csv") as fh:
            cls.rows = list(csv.DictReader(fh))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_camera_moves(self):
        v, u = self.truth["v_c"], self.truth["u_c"]
        self.assertGreater(max(v) - min(v), 200, "the synthetic camera must pan a long way")
        self.assertGreater(max(u) - min(u), 30, "and sideways")

    def test_registration_within_1px(self):
        errs = [abs(float(r["v_c"]) - t) for r, t in zip(self.rows, self.truth["v_c"])]
        erru = [abs(float(r["u_c"]) - t) for r, t in zip(self.rows, self.truth["u_c"])]
        self.assertEqual(len(errs), pan_synth.N_FRAMES)
        self.assertEqual(self.summary["registration"]["anchored_frames"], pan_synth.N_FRAMES)
        self.assertLess(max(errs), 1.0, f"max registration error along the rink {max(errs):.3f} px")
        self.assertLess(max(erru), 1.0, f"max registration error across the rink {max(erru):.3f} px")
        self.assertLess(abs(self.summary["calibration"]["s_px_per_m"] / pan_synth.S - 1), 0.002)

    def test_world_positions_within_1px(self):
        s = pan_synth.S
        for key in ("player", "puck"):
            errs = []
            for r, t in zip(self.rows, self.truth[key]):
                self.assertNotEqual(r[f"{key}_x"], "", f"{key} missed in frame {r['frame']}")
                errs.append(math.hypot(float(r[f"{key}_x"]) - t[0], float(r[f"{key}_y"]) - t[1]) * s)
            self.assertLess(max(errs), 1.0, f"{key}: max world error {max(errs):.3f} px")


class KeyTrackerTest(unittest.TestCase):
    def test_press_hold_release_linger_and_repress(self):
        # Frames: W appears raised (y29) and sinks, is held flat (y37), rises on release, lingers, disappears;
        # then a new W cap appears while A is held.
        seq = [[], [Cap(700, 29, 68, 68, "W", 1)], [Cap(700, 32, 68, 68, "W", 1)], [Cap(700, 37, 68, 68, "W", 1)],
               [Cap(700, 37, 68, 68, "W", 1)], [Cap(700, 33, 68, 68, "W", 1)], [Cap(700, 29, 68, 68, "W", 1)],
               [Cap(700, 29, 68, 68, "W", 1)], [Cap(610, 29, 68, 68, "W", 1), Cap(700, 29, 68, 68, "A", 1)],
               [Cap(520, 29, 68, 68, "W", 1), Cap(610, 37, 68, 68, "A", 1), Cap(700, 29, 68, 68, "W", 1)],
               [Cap(610, 37, 68, 68, "A", 1), Cap(700, 37, 68, 68, "W", 1)], []]
        tr = KeyTracker()
        states = [tr.update(c) for c in seq]
        ev = [(e["frame"], e["key"], e["down"]) for e in events_from_states(states)]
        self.assertEqual(ev, [(1, "W", True), (5, "W", False), (8, "A", True), (9, "W", True), (11, "W", False), (11, "A", False)])


if __name__ == "__main__":
    unittest.main()
