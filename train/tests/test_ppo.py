import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train import ppo
from train.slapenv import DEFAULT_PARAMS


class Ppo(unittest.TestCase):
    def test_action_mapping(self):
        raw = np.array([[2.0, -0.5, 0.0, 1.0, 1.0, 0.0]], np.float32)
        a = ppo.to_env_actions(raw)
        self.assertTrue(np.allclose(a[0], [1.0, -0.5, np.pi / 2, 1.0, 0.0]))

    def test_short_selfplay_run_checkpoint_and_physics_guard(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            base = ["train", "--run", str(run), "--envs", "8", "--steps", "16", "--device", "cpu",
                    "--opponents", "chaser:0.5,pool:0.5", "--snapshot-every", "1", "--print-every", "100",
                    "--save-every", "1"]
            ppo.main(base + ["--updates", "3"])
            ck = torch.load(run / "latest.pt", weights_only=False)
            self.assertEqual(ck["update"], 3)
            self.assertEqual(len(ck["physics_stamp"]), 16)
            self.assertEqual(len(list((run / "pool").glob("snap_*.pt"))), 4)  # initial + 3 updates
            rows = (run / "metrics.csv").read_text().strip().splitlines()
            self.assertEqual(len(rows), 4)
            # Resume continues the update counter.
            ppo.main(base + ["--updates", "4", "--resume", str(run / "latest.pt")])
            self.assertEqual(torch.load(run / "latest.pt", weights_only=False)["update"], 4)
            # Different physics: blocked unless allowed.
            data = json.loads(Path(DEFAULT_PARAMS).read_text())
            data["params"]["board_restitution"]["value"] = 0.123
            other = Path(d) / "params_other.json"
            other.write_text(json.dumps(data))
            with self.assertRaises(SystemExit) as cm:
                ppo.main(base + ["--updates", "5", "--resume", str(run / "latest.pt"), "--params", str(other)])
            self.assertIn("physics", str(cm.exception))
            ppo.main(base + ["--updates", "5", "--resume", str(run / "latest.pt"), "--params", str(other),
                             "--allow-physics-change"])
            self.assertEqual(torch.load(run / "latest.pt", weights_only=False)["update"], 5)

    def test_eval_random_and_untrained(self):
        r = ppo.main(["eval", "--policy", "random", "--opponent", "idle", "--episodes", "8", "--envs", "4",
                      "--episode-seconds", "2", "--device", "cpu"])
        self.assertEqual(r["episodes"], 8)
        self.assertAlmostEqual(r["win"] + r["loss"] + r["draw"], 1.0)


if __name__ == "__main__":
    unittest.main()
