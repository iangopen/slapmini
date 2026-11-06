import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train import league, ppo
from train.slapenv import DEFAULT_PARAMS, dims


class Ppo(unittest.TestCase):
    def test_action_mapping(self):
        raw = np.array([[2.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]], np.float32)   # move 2, stick (0, 1), dash, no lift, stick up, brake
        a = ppo.to_env_actions(raw)
        self.assertTrue(np.allclose(a[0], [2.0, np.pi / 2, 1.0, 0.0, 1.0, 1.0]))
        pol = ppo.Policy(dims()["obs"])
        raw, logp, v = pol.act(torch.zeros(5, dims()["obs"]))
        self.assertEqual(tuple(raw.shape), (5, ppo.RAW_DIM))
        self.assertTrue(torch.all((raw[:, 0] >= 0) & (raw[:, 0] <= 8) & (raw[:, 0] == raw[:, 0].round())))
        lp, ent, _ = pol.evaluate(torch.zeros(5, dims()["obs"]), raw)
        self.assertTrue(torch.allclose(lp, logp, atol=1e-6))
        # Near-uniform start: the move entropy is close to log 9.
        self.assertGreater(float(ent.mean()), np.log(9))

    def test_hand_written_logp_matches_torch_distributions(self):
        torch.manual_seed(3)
        pol = ppo.Policy(dims()["obs"], 64)
        with torch.no_grad():
            pol.pi[-1].weight.normal_(0, 0.5)
            pol.pi[-1].bias.normal_(0, 0.5)
            pol.log_std.copy_(torch.tensor([-0.2, 0.3]))
        obs = torch.randn(500, dims()["obs"])
        raw, logp, _ = pol.act(obs)
        mv, g, d, l, u, b = pol.dist(obs)
        want = (mv.log_prob(raw[:, 0].long()) + g.log_prob(raw[:, 1:3]).sum(-1) + d.log_prob(raw[:, 3]) + l.log_prob(raw[:, 4])
                + u.log_prob(raw[:, 5]) + b.log_prob(raw[:, 6]))
        want_ent = mv.entropy() + g.entropy().sum(-1) + d.entropy() + l.entropy() + u.entropy() + b.entropy()
        lp, ent, _ = pol.evaluate(obs, raw)
        self.assertTrue(torch.allclose(logp, want, atol=1e-5))
        self.assertTrue(torch.allclose(lp, want, atol=1e-5))
        self.assertTrue(torch.allclose(ent, want_ent, atol=1e-5))
        # Sampled moves follow the categorical distribution (chi-square-ish check on one state).
        one = obs[:1].repeat(20000, 1)
        r, _, _ = pol.act(one)
        freq = np.bincount(r[:, 0].long().numpy(), minlength=9) / 20000
        p = torch.softmax(pol.heads(obs[:1])[0], -1)[0].detach().numpy()
        self.assertLess(np.abs(freq - p).max(), 0.015)
        self.assertAlmostEqual(float(r[:, 3].mean()), float(torch.sigmoid(pol.heads(obs[:1])[2])[0]), delta=0.015)
        self.assertAlmostEqual(float(r[:, 5].mean()), float(torch.sigmoid(pol.heads(obs[:1])[4])[0]), delta=0.015)

    def test_running_obs_norm(self):
        torch.manual_seed(0)
        n = ppo.RunningNorm(42, 32)
        xs = [torch.randn(200, 42) * 3 + 5, torch.randn(50, 42) * 0.5 - 1, torch.randn(1000, 42) * 2]
        for x in xs:
            n.update(x)
        allx = torch.cat(xs).double()
        self.assertTrue(torch.allclose(n.mean[:32].double(), allx[:, :32].mean(0), atol=1e-4))
        self.assertTrue(torch.allclose(n.var[:32].double(), allx[:, :32].var(0, unbiased=False), rtol=1e-3))
        self.assertTrue(torch.all(n.mean[32:] == 0) and torch.all(n.var[32:] == 1))
        z = n(allx.float())
        self.assertTrue(torch.allclose(z[:, :32].mean(0), torch.zeros(32), atol=1e-3))
        self.assertTrue(torch.allclose(z[:, 32:], allx[:, 32:].float()))   # the style slot and the stick_up flags pass through
        # The statistics travel with the state dict (checkpoints).
        pol = ppo.Policy(42, 32, obs_norm=True)
        pol.norm.update(allx.float())
        pol2 = ppo.Policy(42, 32, obs_norm=True)
        pol2.load_state_dict(pol.state_dict())
        self.assertTrue(torch.equal(pol2.norm.mean, pol.norm.mean))

    def test_short_selfplay_run_checkpoint_and_physics_guard(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            base = ["train", "--run", str(run), "--envs", "8", "--steps", "16", "--device", "cpu",
                    "--opponents", "chaser:0.5,pool:0.5", "--snapshot-every", "1", "--print-every", "100",
                    "--save-every", "1", "--obs-norm", "--styles", "--threads", "1"]
            ppo.main(base + ["--updates", "3"])
            ck = torch.load(run / "latest.pt", weights_only=False)
            self.assertEqual(ck["update"], 3)
            self.assertEqual(len(ck["physics_stamp"]), 16)
            self.assertTrue(ck["obs_norm"])
            self.assertGreater(float(ck["policy"]["norm.count"]), 3 * 8 * 16 - 1)
            self.assertEqual(len(list((run / "pool").glob("snap_*.pt"))), 4)  # initial + 3 updates
            rows = (run / "metrics.csv").read_text().strip().splitlines()
            self.assertEqual(len(rows), 4)
            # Resume continues the update counter (and keeps the normalizer).
            ppo.main(base + ["--updates", "4", "--resume", str(run / "latest.pt")])
            ck4 = torch.load(run / "latest.pt", weights_only=False)
            self.assertEqual(ck4["update"], 4)
            self.assertGreater(float(ck4["policy"]["norm.count"]), float(ck["policy"]["norm.count"]))
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
            # An ACT v3 checkpoint (no brake head) is refused with a clear message.
            old = torch.load(run / "latest.pt", weights_only=False)
            self.assertEqual(old["act_version"], 4)
            old["act_version"] = 3
            torch.save(old, Path(d) / "old.pt")
            with self.assertRaises(SystemExit) as cm:
                ppo.load_ckpt(Path(d) / "old.pt")
            print(f"old checkpoint refused: {cm.exception}")
            self.assertIn("ACT v3", str(cm.exception))
            old["act_version"], old["obs_version"], old["obs_dim"] = 4, 2, 40
            torch.save(old, Path(d) / "old_obs.pt")
            with self.assertRaises(SystemExit) as cm:
                ppo.load_ckpt(Path(d) / "old_obs.pt")
            print(f"old observation refused: {cm.exception}")
            self.assertIn("v2/40", str(cm.exception))

    def test_match_mode_curriculum_and_profile(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            ppo.main(["train", "--run", str(run), "--envs", "8", "--steps", "32", "--device", "cpu", "--updates", "4",
                      "--curriculum-minutes", "0.0001", "--print-every", "100", "--profile", "--threads", "1",
                      "--keep-every", "2"])
            rows = list(__import__("csv").DictReader(open(run / "metrics.csv")))
            self.assertEqual(rows[-1]["mode"], "match")
            self.assertTrue((run / "profile.md").read_text().startswith("| section"))
            self.assertEqual(len(list((run / "ckpts").glob("upd_*.pt"))), 2)
            # A ladder over the kept checkpoints (first-goal games for speed).
            out = league.main(["--ckpts", str(run / "ckpts" / "*.pt"), "--bots", "chaser,idle", "--matches", "4",
                               "--mode", "first_goal", "--envs", "4", "--styles", "0,0,0,0;1,0,0,0",
                               "--style-matches", "4", "--out", str(Path(d) / "league.json")])
            self.assertEqual(set(out["ratings"]), {"ckpts/upd_000002", "ckpts/upd_000004", "chaser", "idle"})
            self.assertAlmostEqual(out["ratings"]["chaser"], 1000.0)
            self.assertEqual(len(out["style_stats"]), 4)
            for s in out["style_stats"]:
                self.assertIn("shots_per_ep", s)
                self.assertIn("possession_share", s)

    def test_elo_fit(self):
        # A beats B 75% of the time -> about 191 Elo points apart (before the virtual draw prior).
        r = league.elo_fit(["A", "B"], [("A", "B", 750.0, 1000)], "B")
        self.assertAlmostEqual(r["B"], 1000.0)
        self.assertAlmostEqual(r["A"] - r["B"], 400 * np.log10(3), delta=2.0)

    def test_eval_random_and_untrained(self):
        r = ppo.main(["eval", "--policy", "random", "--opponent", "idle", "--episodes", "8", "--envs", "4",
                      "--mode", "first_goal", "--episode-seconds", "2", "--device", "cpu"])
        self.assertEqual(r["episodes"], 8)
        self.assertAlmostEqual(r["win"] + r["loss"] + r["draw"], 1.0)
        r = ppo.main(["eval", "--policy", "untrained", "--opponent", "chaser", "--episodes", "2", "--envs", "2",
                      "--device", "cpu"])
        self.assertEqual(r["episodes"], 2)   # full matches against the chaser


if __name__ == "__main__":
    unittest.main()
