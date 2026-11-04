"""`boardmodel` command: does a Coulomb-style board friction explain the speed lost sliding along boards?

EXPERIMENT, NOT ADOPTED: needs the sim param board_coulomb_friction, which is not in the sim. Apply
docs/experiments/board_coulomb_friction.patch (git apply), rebuild calib (cargo build --release -p calib) and run;
revert afterwards. Result (2026-11-04): calib/data/real/boardmodel.json and CLAUDE.md "Board friction model".

Candidates (ice params fixed at the params.json defaults; full path objective, since the speed loss matters):
  current     board_restitution + board_friction (tangential speed x (1 - f) per contact)
  coulomb     board_restitution + board_coulomb_friction (tangential loss = mu (1 + e) |v_n| per contact), f = 0
  both        all three
  ..., e fixed: board_restitution kept at the default (fitted on rebound geometry), only the friction term fitted
Segments: ground segments with board contact (a wall bounce, or >= 5 rows within puck_radius + 0.15 m of the
boards). Fit on one clip, score on the other (score = mean of path_rmse^2 + 0.01 v2_rmse^2).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .geometry import PARAMS_PATH
from .holdout import evaluate, fit, score, summary, with_values
from .segment import OUT

CANDIDATES = {
    "current (restitution + proportional loss)": (["board_restitution", "board_friction"], {"board_coulomb_friction": 0.0}),
    "coulomb (restitution + mu (1+e)|v_n|)": (["board_restitution", "board_coulomb_friction"], {"board_friction": 0.0}),
    "both": (["board_restitution", "board_friction", "board_coulomb_friction"], {}),
    # Restitution kept at the default (fitted on rebound geometry); only the friction term is fitted:
    "current, e fixed (fit proportional loss only)": (["board_friction"], {"board_coulomb_friction": 0.0}),
    "coulomb, e fixed (fit mu only)": (["board_coulomb_friction"], {"board_friction": 0.0}),
}


def board_segments(params_path: Path = PARAMS_PATH):
    P = {k: v["value"] for k, v in json.loads(Path(params_path).read_text())["params"].items()}
    L, W, rc, r = P["rink_length"], P["rink_width"], P["rink_corner_radius"], P["puck_radius"]

    def bd(x, y):
        cx0, cx1, cy0, cy1 = -L / 2 + rc, L / 2 - rc, -W / 2 + rc, W / 2 - rc
        qx, qy = np.clip(x, cx0, cx1), np.clip(y, cy0, cy1)
        corner = ((x < cx0) | (x > cx1)) & ((y < cy0) | (y > cy1))
        return np.where(corner, rc - np.hypot(x - qx, y - qy), np.minimum.reduce([x + L / 2, L / 2 - x, y + W / 2, W / 2 - y]))

    man = json.loads((OUT / "manifest.json").read_text())
    out = []
    for s in man["segments"]:
        if s["possibly_airborne"]:
            continue
        a = np.loadtxt(OUT / s["csv"], delimiter=",", skiprows=1, ndmin=2)
        near = int((bd(a[:, 1], a[:, 2]) < r + 0.15).sum())
        if s["label"] == "wall bounce" or near >= 5:
            out.append({**s, "rows_at_board": near})
    return out


def run_boardmodel(params_path: Path = PARAMS_PATH, out_path: Path = OUT / "boardmodel.json") -> dict:
    segs = board_segments(params_path)
    by = {}
    for s in segs:
        by.setdefault(s["clip"], []).append(s)
    clips = sorted(by)
    tmp = Path(tempfile.mkdtemp())
    print(f"[boardmodel] board-contact segments: " + "; ".join(f"{c[:10]}: {', '.join(f'{s['id']}({s['label'][0]},{s['rows_at_board']})' for s in by[c])}" for c in clips))
    res = {"segments": {c: [s["id"] for s in by[c]] for c in clips}, "candidates": {}}
    base = {c: evaluate(params_path, by[c]) for c in clips}
    print(f"[boardmodel] {'candidate':44s} | {'train ' + clips[0][:6] + ' -> test ' + clips[1][:6]:44s} | {'train ' + clips[1][:6] + ' -> test ' + clips[0][:6]}")
    print(f"[boardmodel] {'defaults (no refit)':44s} | test {score(base[clips[1]]):7.3f}{'':35s} | test {score(base[clips[0]]):7.3f}")
    for name, (names, fixed) in CANDIDATES.items():
        start = with_values(params_path, fixed, tmp / f"start_{len(res['candidates'])}.json")
        row = []
        for train, test in ((clips[0], clips[1]), (clips[1], clips[0])):
            vals = fit(start, by[train], tmp / f"fit_{train[:6]}_{len(res['candidates'])}.json", names=names, objective="path")
            pv = with_values(start, vals, tmp / f"vals_{train[:6]}_{len(res['candidates'])}.json")
            te, tr = evaluate(pv, by[test]), evaluate(pv, by[train])
            row.append({"train": train, "test": test, "fitted": vals, "train_score": score(tr), "test_score": score(te), "test": summary(te),
                        "per_segment_test": {k: v for k, v in te.items()}})
        res["candidates"][name] = row
        cell = lambda d: f"test {d['test_score']:7.3f} (train {d['train_score']:6.3f}) " + " ".join(f"{k.replace('board_', '')}={v:.3f}" for k, v in d["fitted"].items())
        print(f"[boardmodel] {name:44s} | {cell(row[0]):44s} | {cell(row[1])}")
    res["defaults_test_score"] = {c: score(base[c]) for c in clips}
    out_path.write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(f"[boardmodel] wrote {out_path}")
    return res
