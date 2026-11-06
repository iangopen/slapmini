"""Export a trained policy to ONNX for the browser (onnxruntime-web on GitHub Pages).

    python -m train.export_onnx --ckpt runs/smoke/latest.pt --out web/public/bots/smoke.onnx

The graph is the DETERMINISTIC policy (no sampling), with the running observation normalization
(if the policy has one) inside it. Input "obs": float32 [N, OBS_DIM] (the observation from
bots/src/obs.rs, canonical frame; the style slot obs[32:40] picks the play style, zeros = default).
Outputs, all float32:
  "move"       [N]     move index 0..8 (argmax): 0 stop, 1..8 = +x, +x+y, +y, -x+y, -x, -x-y, -y, +x-y
                       (canonical frame; the WASD key combination, bots/src/action.rs MOVE_DIRS)
  "move_probs" [N, 9]  the move distribution (for sampling or debugging)
  "stick_dir"  [N, 2]  stick direction (canonical); the action's stick angle = atan2(y, x)
  "dash_prob"  [N]     press dash if > 0.5
  "lift_prob"  [N]     hold lift if > 0.5
  "stick_up_prob" [N]  raise the stick if > 0.5 (ACT v3)
  "brake_prob" [N]     slow walk if > 0.5 (ACT v4)
A sidecar <out>.json records the layout versions and the physics stamp.
For team 1 the caller rotates back to the world (180 degrees): move index k in 1..8 becomes
((k - 1 + 4) mod 8) + 1, and the stick direction negates.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train.ppo import ACT_VERSION, Policy, load_ckpt
from train.slapenv import DEFAULT_PARAMS, dims

OUTPUTS = ["move", "move_probs", "stick_dir", "dash_prob", "lift_prob", "stick_up_prob", "brake_prob"]


class Deterministic(nn.Module):
    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        ml, sm, dl, ll, ul, bl = self.policy.heads(obs)
        probs = torch.softmax(ml, dim=-1)
        return probs.argmax(-1).to(torch.float32), probs, sm, torch.sigmoid(dl), torch.sigmoid(ll), torch.sigmoid(ul), torch.sigmoid(bl)


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
    side = {"obs_dim": policy.obs_dim, "obs_version": d["obs_version"], "act_dim": d["act"], "act_version": ACT_VERSION,
            "obs_norm_in_graph": bool(policy.obs_norm),
            "inputs": {"obs": ["batch", policy.obs_dim]}, "outputs": OUTPUTS, "opset": opset,
            "frame": "canonical: the acting player attacks +x; for team 1 rotate move index k>0 to ((k-1+4)%8)+1 "
                     "and negate stick_dir",
            **(meta or {})}
    out.with_suffix(".json").write_text(json.dumps(side, indent=2))
    return out


def check(policy: Policy, path: Path, n: int = 1000, seed: int = 0) -> float:
    """Max |onnxruntime - torch| over n random observations of the float outputs; returns inf if
    the move index differs anywhere the top two move probabilities are not a float-precision tie."""
    import onnxruntime as ort
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 1, (n, policy.obs_dim)).astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(OUTPUTS, {"obs": obs})
    with torch.no_grad():
        want = [w.numpy() for w in Deterministic(policy.cpu().eval())(torch.as_tensor(obs))]
    top2 = np.sort(want[1], axis=1)[:, -2:]
    if np.any((got[0] != want[0]) & (top2[:, 1] - top2[:, 0] > 1e-5)):
        return float("inf")
    return max(float(np.abs(g - w).max()) for g, w in zip(got[1:], want[1:]))


def check_heads(policy: Policy, path: Path, n: int = 1000, seed: int = 0) -> dict:
    """Per-output max |onnxruntime - torch| on n random observations."""
    import onnxruntime as ort
    obs = np.random.default_rng(seed).normal(0, 1, (n, policy.obs_dim)).astype(np.float32)
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(OUTPUTS, {"obs": obs})
    with torch.no_grad():
        want = [w.numpy() for w in Deterministic(policy.cpu().eval())(torch.as_tensor(obs))]
    return {k: float(np.abs(g - w).max()) for k, g, w in zip(OUTPUTS, got, want)}


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
          f"onnxruntime vs torch max abs diff on 1000 random obs: {err:.3e} (move index identical)")
    print("  per output: " + ", ".join(f"{k} {v:.3e}" for k, v in check_heads(pol, out).items()))
    if err > 1e-4:
        raise SystemExit("ONNX output does not match PyTorch")


if __name__ == "__main__":
    main()
