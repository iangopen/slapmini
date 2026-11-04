import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train.export_onnx import OUTPUTS, check, export
from train.ppo import Policy
from train.slapenv import dims


class OnnxExport(unittest.TestCase):
    def test_onnxruntime_matches_pytorch(self):
        torch.manual_seed(0)
        pol = Policy(dims()["obs"])
        with torch.no_grad():  # make the heads non-trivial (fresh init outputs are ~0)
            pol.pi[-1].weight.normal_(0, 0.5)
            pol.pi[-1].bias.normal_(0, 0.5)
        with tempfile.TemporaryDirectory() as d:
            out = export(pol, Path(d) / "p.onnx", {"physics_stamp": "test"})
            err = check(pol, out, n=2000, seed=1)
            self.assertLess(err, 1e-5)
            import onnxruntime as ort
            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            res = sess.run(OUTPUTS, {"obs": np.zeros((3, pol.obs_dim), np.float32)})
            self.assertEqual([r.shape for r in res], [(3, 2), (3, 2), (3,), (3,)])
            self.assertTrue(np.all(np.abs(res[0]) <= 1.0))
            self.assertTrue(out.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
