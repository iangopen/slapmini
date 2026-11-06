import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train.export_onnx import OUTPUTS, check, check_heads, export
from train.ppo import Policy
from train.slapenv import dims


class OnnxExport(unittest.TestCase):
    def _policy(self, obs_norm):
        torch.manual_seed(0)
        pol = Policy(dims()["obs"], obs_norm=obs_norm)
        with torch.no_grad():  # make the heads non-trivial (fresh init outputs are ~0)
            pol.pi[-1].weight.normal_(0, 0.5)
            pol.pi[-1].bias.normal_(0, 0.5)
        if obs_norm:
            pol.norm.update(torch.randn(500, pol.obs_dim) * 4 + 2)
        return pol

    def test_onnxruntime_matches_pytorch(self):
        for obs_norm in (False, True):
            pol = self._policy(obs_norm)
            with tempfile.TemporaryDirectory() as d:
                out = export(pol, Path(d) / "p.onnx", {"physics_stamp": "test"})
                err = check(pol, out, n=2000, seed=1)
                print(f"onnx parity (obs_norm={obs_norm}): max abs diff {err:.3e}, move index identical")
                self.assertLess(err, 1e-5)
                heads = check_heads(pol, out, n=2000, seed=1)
                print("  per output: " + ", ".join(f"{k} {v:.3e}" for k, v in heads.items()))
                self.assertLess(heads["stick_up_prob"], 1e-5)
                self.assertLess(heads["brake_prob"], 1e-5)
                self.assertIn("stick_up_prob", OUTPUTS)
                self.assertIn("brake_prob", OUTPUTS)
                import onnxruntime as ort
                sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
                res = sess.run(OUTPUTS, {"obs": np.zeros((3, pol.obs_dim), np.float32)})
                self.assertEqual([r.shape for r in res], [(3,), (3, 9), (3, 2), (3,), (3,), (3,), (3,)])
                self.assertTrue(np.all((res[0] >= 0) & (res[0] <= 8)))
                self.assertTrue(np.allclose(res[1].sum(1), 1.0, atol=1e-5))
                self.assertTrue(out.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
