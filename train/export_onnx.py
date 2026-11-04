"""Export a trained policy to ONNX for the browser (onnxruntime-web on GitHub Pages).

    python -m train.export_onnx --ckpt runs/smoke/latest.pt --out web/public/bots/smoke.onnx

The graph is the DETERMINISTIC policy (no sampling). Input "obs": float32 [N, OBS_DIM] (the
observation from bots/src/obs.rs, canonical frame). Outputs, all float32:
  "move"      [N, 2]  move_x, move_y in [-1, 1]
  "stick_dir" [N, 2]  stick direction (canonical); the action's stick angle = atan2(y, x)
  "dash_prob" [N]     press dash if > 0.5
  "lift_prob" [N]     hold lift if > 0.5
A sidecar <out>.json records the layout versions and the physics stamp.
For team 1 the caller rotates back to the world: move and stick direction negate (180 degrees).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train.ppo import Policy, load_ckpt
from train.slapenv import DEFAULT_PARAMS, dims

OUTPUTS = ["move", "stick_dir", "dash_prob", "lift_prob"]


class Deterministic(nn.Module):
    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        mean, dl, ll = self.policy.heads(obs)
        return mean[:, 0:2].clamp(-1.0, 1.0), mean[:, 2:4], torch.sigmoid(dl), torch.sigmoid(ll)


def export(policy: Policy, out: Path, meta: dict | None = None, opset: int = 17) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m = Deterministic(policy.cpu().eval())
    dummy = torch.zeros(1, policy.obs_dim)
    kw = dict(input_names=["obs"], output_names=OUTPUTS, opset_version=opset,
              dynamic_axes={"obs": {0: "batch"}, **{k: {0: "batch"} for k in OUTPUTS}})
    try:  # the classic TorchScript exporter: one self-contained file, no onnxscript needed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(m, (dummy,), str(out), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo switch
        torch.onnx.export(m, (dummy,), str(out), **kw)
    d = dims()
    side = {"obs_dim": policy.obs_dim, "obs_version": d["obs_version"], "act_dim": d["act"],
            "inputs": {"obs": ["batch", policy.obs_dim]}, "outputs": OUTPUTS, "opset": opset,
            "frame": "canonical: the acting player attacks +x; for team 1 negate move and stick_dir",
            **(meta or {})}
    out.with_suffix(".json").write_text(json.dumps(side, indent=2))
    return out


def check(policy: Policy, path: Path, n: int = 1000, seed: int = 0) -> float:
    """Max |onnxruntime - torch| over n random observations."""
    import onnxruntime as ort
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 1, (n, policy.obs_dim)).astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(OUTPUTS, {"obs": obs})
    with torch.no_grad():
        want = Deterministic(policy.cpu().eval())(torch.as_tensor(obs))
    return max(float(np.abs(g - w.numpy()).max()) for g, w in zip(got, want))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m train.export_onnx")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default=str(DEFAULT_PARAMS))
    ap.add_argument("--allow-physics-change", action="store_true")
    args = ap.parse_args(argv)
    pol, ck = load_ckpt(args.ckpt, args.params, args.allow_physics_change)
    out = export(pol, Path(args.out), {"physics_stamp": ck.get("physics_stamp"), "update": ck.get("update"),
                                       "source": str(args.ckpt)})
    err = check(pol, out)
    print(f"wrote {out} ({out.stat().st_size} bytes) + {out.with_suffix('.json').name}; "
          f"onnxruntime vs torch max abs diff on 1000 random obs: {err:.3e}")
    if err > 1e-4:
        raise SystemExit("ONNX output does not match PyTorch")


if __name__ == "__main__":
    main()
