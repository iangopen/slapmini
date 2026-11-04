import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train.slapenv import DEFAULT_PARAMS, VecEnv, dims, physics_stamp


def rollout(env, steps, seed=0):
    rng = np.random.default_rng(seed)
    obs = [env.reset()]
    hashes = []
    for _ in range(steps):
        a = rng.uniform(-1, 1, (env.n, env.act_dim)).astype(np.float32)
        o, r, d, info = env.step(a)
        obs.append(o)
        hashes.append(env.hash())
    return np.stack(obs), hashes


class Env(unittest.TestCase):
    def test_dims(self):
        d = dims()
        self.assertEqual(d["obs"], 40)
        self.assertEqual(d["act"], 5)
        self.assertEqual(d["reward_terms"][:2], ["goal_for", "goal_against"])

    def test_same_seed_is_deterministic_across_thread_counts(self):
        a = VecEnv(16, seed=5, config={"threads": 1, "max_steps": 900})
        b = VecEnv(16, seed=5, config={"threads": 4, "max_steps": 900})
        c = VecEnv(16, seed=6, config={"threads": 4, "max_steps": 900})
        oa, ha = rollout(a, 400)
        ob, hb = rollout(b, 400)
        oc, hc = rollout(c, 400)
        self.assertTrue(np.array_equal(oa.view(np.uint32), ob.view(np.uint32)))
        self.assertEqual(ha, hb)
        self.assertNotEqual(ha[-1], hc[-1])

    def test_state_roundtrip_and_episode_end(self):
        env = VecEnv(2, seed=1, config={"opponent": "chaser", "threads": 1})
        env.reset()
        s = env.get_state(1)
        h = env.env_hash(1)
        env.set_state(1, s)
        self.assertEqual(env.env_hash(1), h)
        # An idle learner against the chaser concedes: done with end reason 2 and reward <= -1.
        ends = []
        for _ in range(400):
            _, r, d, info = env.step(np.zeros((2, 5), np.float32))
            ends += [(info[i, 0], r[i]) for i in np.nonzero(d)[0]]
        self.assertTrue(ends)
        for reason, r in ends:
            self.assertEqual(reason, 2)
            self.assertLessEqual(r, -0.9)

    def test_reward_weights_and_style(self):
        env = VecEnv(2, seed=1, config={"reward": {"puck_progress": 0, "shot_on_net": 0, "possession": 0},
                                         "threads": 1, "opponent": "chaser"})
        env.reset()
        env.set_style(None, [1, 2, 3])
        env.set_reward_weights(0, {"goal_against": -5.0})
        obs, r, d, info = env.step(np.zeros((2, 5), np.float32))
        self.assertTrue(np.allclose(obs[:, 32:35], [1, 2, 3]))
        self.assertTrue(np.allclose(obs[:, 35:], 0))
        got = {0: [], 1: []}
        for _ in range(400):
            _, r, d, info = env.step(np.zeros((2, 5), np.float32))
            for i in np.nonzero(d)[0]:
                got[i].append(r[i])
        self.assertTrue(got[0] and got[1])
        self.assertTrue(all(abs(x + 5.0) < 1e-5 for x in got[0]), got[0])  # only goal_against, weight -5
        self.assertTrue(all(abs(x + 1.0) < 1e-5 for x in got[1]), got[1])  # config default -1
        with self.assertRaises(KeyError):
            env.set_reward_weights(0, {"nope": 1})
        with self.assertRaises(RuntimeError):
            VecEnv(1, config={"opponent": "nobody"})

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
