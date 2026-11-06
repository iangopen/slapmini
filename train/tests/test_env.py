import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train.slapenv import (DEFAULT_PARAMS, END_LOSS, END_TRUNCATED, I_END, I_GA, I_GF, I_RAW, VecEnv, dims,
                           physics_stamp)


def rand_actions(rng, n):
    a = np.empty((n, 6), np.float32)   # ACT v4: move, stick angle, dash, lift, stick_up, brake
    a[:, 0] = rng.integers(0, 9, n)
    a[:, 1] = rng.uniform(-3.14, 3.14, n)
    a[:, 2:4] = rng.uniform(0, 1, (n, 2)) < 0.2
    a[:, 4] = rng.uniform(0, 1, n) < 0.3
    a[:, 5] = rng.uniform(0, 1, n) < 0.1
    return a


def rollout(env, steps, seed=0):
    rng = np.random.default_rng(seed)
    obs = [env.reset().copy()]
    hashes = []
    for _ in range(steps):
        o, r, d, info = env.step(rand_actions(rng, env.n))
        obs.append(o.copy())
        hashes.append(env.hash())
    return np.stack(obs), hashes


class Env(unittest.TestCase):
    def test_dims(self):
        d = dims()
        self.assertEqual(d["obs"], 42)
        self.assertEqual(d["act"], 6)
        self.assertEqual((d["obs_version"], d["act_version"], d["abi"]), (3, 4, 4))
        self.assertEqual(d["reward_terms"], ["goal_for", "goal_against", "puck_progress", "shot_on_net", "possession",
                                             "defensive_position", "match_win"])

    def test_same_seed_is_deterministic_across_thread_counts(self):
        for mode in ("first_goal", "match"):
            a = VecEnv(16, seed=5, config={"threads": 1, "max_steps": 900, "mode": mode})
            b = VecEnv(16, seed=5, config={"threads": 4, "max_steps": 900, "mode": mode})
            c = VecEnv(16, seed=6, config={"threads": 4, "max_steps": 900, "mode": mode})
            oa, ha = rollout(a, 400)
            ob, hb = rollout(b, 400)
            oc, hc = rollout(c, 400)
            self.assertTrue(np.array_equal(oa.view(np.uint32), ob.view(np.uint32)), mode)
            self.assertEqual(ha, hb, mode)
            self.assertNotEqual(ha[-1], hc[-1], mode)

    def test_state_roundtrip_and_episode_end(self):
        env = VecEnv(2, seed=1, config={"opponent": "chaser", "threads": 1})
        env.reset()
        s = env.get_state(1)
        h = env.env_hash(1)
        env.set_state(1, s)
        self.assertEqual(env.env_hash(1), h)
        # An idle learner against the chaser concedes: done with end reason 2 (loss) and reward <= -1.
        ends = []
        for _ in range(400):
            _, r, d, info = env.step(np.zeros((2, 6), np.float32))
            ends += [(info[i, 0], r[i]) for i in np.nonzero(d)[0]]
        self.assertTrue(ends)
        for reason, r in ends:
            self.assertEqual(reason, END_LOSS)
            self.assertLessEqual(r, -0.9)

    def test_match_mode_mercy_loss_and_truncation_final_obs(self):
        # Idle learner vs the chaser in a full match: the chaser wins 5-0 by the mercy rule.
        env = VecEnv(2, seed=3, config={"opponent": "chaser", "threads": 1, "mode": "match"})
        env.reset()
        done_rows = []
        for _ in range(30 * 300):
            _, r, d, info = env.step(np.zeros((2, 6), np.float32))
            for i in np.nonzero(d)[0]:
                done_rows.append(info[i].copy())
            if len(done_rows) >= 2:
                break
        self.assertGreaterEqual(len(done_rows), 2)
        for row in done_rows:
            self.assertEqual(row[I_END], END_LOSS)
            self.assertEqual((row[I_GF], row[I_GA]), (0, 5))
            self.assertEqual(row[I_RAW + 6], -1.0)  # match_win raw term
        # Idle vs idle with a short limit: truncated, and final_obs is the pre-reset observation.
        env = VecEnv(1, seed=3, config={"opponent": "idle", "threads": 1, "max_steps": 40, "learner_team": 0})
        env.reset()
        fwd = np.array([[1, 0, 0, 0, 0, 0]], np.float32)
        for k in range(10):
            o, r, d, info = env.step(fwd)
        self.assertTrue(d[0])
        self.assertEqual(info[0, I_END], END_TRUNCATED)
        self.assertGreater(env.final_obs[0, 2], 0.1)   # still skating in the final state
        self.assertAlmostEqual(float(o[0, 2]), 0.0)    # the new episode starts at rest

    def test_style_zero_reproduces_default_hashes(self):
        for mode in ("first_goal", "match"):
            a = VecEnv(8, seed=77, config={"threads": 1, "mode": mode})
            b = VecEnv(8, seed=77, config={"threads": 1, "mode": mode, "style_mode": "fixed", "style": [0] * 8})
            oa, ha = rollout(a, 600, seed=4)
            ob, hb = rollout(b, 600, seed=4)
            print(f"style zero {mode}: default {ha[-1]:016x} style-zero {hb[-1]:016x}")
            self.assertEqual(ha, hb, mode)
            self.assertTrue(np.array_equal(oa.view(np.uint32), ob.view(np.uint32)))
            # The random actions raise sticks, so the OBS v3 flags are exercised (own 40, opponent 41).
            self.assertTrue(oa[..., 40].any() and set(np.unique(oa[..., 40:42])) <= {0.0, 1.0}, mode)

    def test_reward_weights_and_style(self):
        env = VecEnv(2, seed=1, config={"reward": {"puck_progress": 0, "shot_on_net": 0, "possession": 0},
                                         "threads": 1, "opponent": "chaser"})
        env.reset()
        env.set_style(None, [1, 2, 3])
        env.set_reward_weights(0, {"goal_against": -5.0})
        obs, r, d, info = env.step(np.zeros((2, 6), np.float32))
        self.assertTrue(np.allclose(obs[:, 32:35], [1, 2, 3]))
        self.assertTrue(np.allclose(obs[:, 35:], 0))
        got = {0: [], 1: []}
        for _ in range(400):
            _, r, d, info = env.step(np.zeros((2, 6), np.float32))
            for i in np.nonzero(d)[0]:
                got[i].append(r[i])
        self.assertTrue(got[0] and got[1])
        self.assertTrue(all(abs(x + 5.0) < 1e-5 for x in got[0]), got[0])  # only goal_against, weight -5
        self.assertTrue(all(abs(x + 1.0) < 1e-5 for x in got[1]), got[1])  # config default -1
        with self.assertRaises(KeyError):
            env.set_reward_weights(0, {"nope": 1})
        with self.assertRaises(RuntimeError):
            VecEnv(1, config={"opponent": "nobody"})

    def test_profile_counters(self):
        env = VecEnv(8, seed=1, config={"threads": 1, "profile": True})
        env.reset()
        p0 = env.profile()
        rng = np.random.default_rng(0)
        for _ in range(50):
            env.step(rand_actions(rng, 8))
        p1 = env.profile()
        self.assertTrue(all(b > a for a, b in zip(p0, p1)), (p0, p1))

    def test_physics_stamp_ignores_formatting_but_not_values(self):
        data = json.loads(Path(DEFAULT_PARAMS).read_text())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "params.json"
            for e in data["params"].values():
                e["note"] = "edited note"
            p.write_text(json.dumps(data, indent=1).replace("\n", "\r\n"))
            self.assertEqual(physics_stamp(p), physics_stamp(DEFAULT_PARAMS))
            data["params"]["ice_linear_drag"]["value"] += 0.01
            p.write_text(json.dumps(data))
            self.assertNotEqual(physics_stamp(p), physics_stamp(DEFAULT_PARAMS))


if __name__ == "__main__":
    unittest.main()
