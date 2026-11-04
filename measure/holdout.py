"""`holdout` command: fit the scale-free puck params on one clip's ground segments, test on the other.

Uses the calib binary (target/release/calib; built with `cargo build --release -p calib`) and the timing-robust
path objective. Score per segment = path_rmse^2 + 0.01 * v2_rmse^2 (the fit objective's per-segment term);
a clip's score is the mean over its segments. Fitted params are written to a new file only if the
held-out score beats the placeholder baseline in BOTH directions.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .geometry import PARAMS_PATH, REPO
from .segment import OUT

CALIB = REPO / "target" / "release" / "calib"
FIT_NAMES = ["board_restitution", "board_friction", "ice_linear_drag", "ice_friction_decel"]
V2W = 0.01


def pairs(segs):
    out = []
    for s in segs:
        out += [str(OUT / s["scenario"]), str(OUT / s["csv"])]
    return out


def evaluate(params: Path, segs) -> dict:
    r = subprocess.run([str(CALIB), "eval", "--json", "1", "--params", str(params), *pairs(segs)], capture_output=True, text=True, check=True)
    return {row["name"]: row["errors"] for row in json.loads(r.stdout)}


def score(errs: dict) -> float:
    return float(np.mean([e["path_rmse"] ** 2 + V2W * e["v2_rmse"] ** 2 for e in errs.values()]))


def summary(errs: dict) -> dict:
    a = lambda k: [e[k] for e in errs.values()]
    return {"score": round(score(errs), 4), "path_rmse_median": round(float(np.median(a("path_rmse"))), 4),
            "v2_rmse_median": round(float(np.median(a("v2_rmse"))), 3), "pos_rmse_median": round(float(np.median(a("pos_rmse"))), 3), "n": len(errs)}


def fit(params: Path, segs, out: Path, names=FIT_NAMES, iters=1500, restarts=1, objective="path") -> dict:
    src = params
    for _ in range(restarts + 1):
        subprocess.run([str(CALIB), "fit", "--objective", objective, "--params", str(src), "--fit", ",".join(names), "--iters", str(iters), "--out", str(out), *pairs(segs)],
                       capture_output=True, text=True, check=True)
        src = out
    return {n: v["value"] for n, v in json.loads(out.read_text())["params"].items() if n in names}


def with_values(params: Path, values: dict, out: Path) -> Path:
    pf = json.loads(Path(params).read_text())
    for n, v in values.items():
        pf["params"][n]["value"] = v
    out.write_text(json.dumps(pf))
    return out


def staged_fit(params: Path, segs, tmp: Path, tag: str) -> dict:
    """Stage A: ice params on glides (path objective). Stage B: board params on bounce segments with
    stage A fixed, path-only objective (rebound geometry: the speed lost riding a curved board is a
    separate model gap that otherwise drags restitution down)."""
    glides = [s for s in segs if s["label"] == "glide"]
    bounces = [s for s in segs if s["label"] == "wall bounce"]
    a = fit(params, glides, tmp / f"A_{tag}.json", names=["ice_linear_drag", "ice_friction_decel"])
    pa = with_values(params, a, tmp / f"A_{tag}_only.json")
    b = fit(pa, bounces, tmp / f"B_{tag}.json", names=["board_restitution", "board_friction"], objective="path-only")
    return {**a, **b}


def run_holdout(params_path: Path = PARAMS_PATH, write_to: Path = REPO / "params.fitted.json") -> dict:
    if not CALIB.exists():
        raise FileNotFoundError(f"{CALIB} missing: run cargo build --release -p calib")
    man = json.loads((OUT / "manifest.json").read_text())
    ground = [s for s in man["segments"] if not s["possibly_airborne"]]
    by = {}
    for s in ground:
        by.setdefault(s["clip"], []).append(s)
    clips = sorted(by)
    if len(clips) != 2:
        raise RuntimeError(f"expected 2 clips with ground segments, got {clips}")
    base = {c: evaluate(params_path, by[c]) for c in clips}
    start_vals = {n: v["value"] for n, v in json.loads(Path(params_path).read_text())["params"].items() if n in FIT_NAMES}
    res = {"fit_names": FIT_NAMES, "start": start_vals, "objective": "path; score per segment = path_rmse^2 + 0.01 v2_rmse^2", "directions": []}
    tmp = Path(tempfile.mkdtemp())
    print(f"[holdout] start (placeholder) values: {start_vals}")
    for train, test in ((clips[0], clips[1]), (clips[1], clips[0])):
        d = {"train": train, "test": test, "train_segments": [s["id"] for s in by[train]], "test_segments": [s["id"] for s in by[test]],
             "train_baseline": summary(base[train]), "test_baseline": summary(base[test])}
        print(f"[holdout] train {train} ({len(by[train])} seg) -> test {test} ({len(by[test])} seg)")
        print(f"[holdout]   {'baseline':22s} train score {d['train_baseline']['score']:9.4f} | TEST score {d['test_baseline']['score']:9.4f}  path med {d['test_baseline']['path_rmse_median']:.3f} m  v2 med {d['test_baseline']['v2_rmse_median']:7.2f}  pos med {d['test_baseline']['pos_rmse_median']:.2f} m")
        for kind in ("joint", "staged"):
            if kind == "joint":
                vals = fit(params_path, by[train], tmp / f"joint_{train}.json")
            else:
                vals = staged_fit(params_path, by[train], tmp, train[:6])
            pv = with_values(params_path, vals, tmp / f"{kind}_{train}_vals.json")
            tr, te = summary(evaluate(pv, by[train])), evaluate(pv, by[test])
            d[kind] = {"fitted": vals, "train": tr, "test": summary(te), "improves_held_out": score(te) < score(base[test]),
                       "per_segment_test": {n: {"baseline": base[test][n], "fitted": te[n]} for n in te}}
            t = d[kind]["test"]
            print(f"[holdout]   {kind + ' fit':22s} train score {tr['score']:9.4f} | TEST score {t['score']:9.4f}  path med {t['path_rmse_median']:.3f} m  v2 med {t['v2_rmse_median']:7.2f}  pos med {t['pos_rmse_median']:.2f} m  improves: {d[kind]['improves_held_out']}")
            print(f"[holdout]     {kind} values: " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(vals.items())))
        res["directions"].append(d)
    for kind in ("joint", "staged"):
        res[f"{kind}_both_improve"] = all(d[kind]["improves_held_out"] for d in res["directions"])
        spread = {n: [d[kind]["fitted"][n] for d in res["directions"]] for n in FIT_NAMES}
        res[f"{kind}_values_by_direction"] = spread
        print(f"[holdout] {kind}: both directions improve held-out: {res[f'{kind}_both_improve']}; values by direction: " + ", ".join(f"{n} {v[0]:.3f}/{v[1]:.3f}" for n, v in spread.items()))
    # Write the staged fit (consistent across directions) if both directions improve.
    kind = "staged"
    mus = res[f"{kind}_values_by_direction"]["ice_friction_decel"]
    ks = res[f"{kind}_values_by_direction"]["ice_linear_drag"]
    near_zero = all(mu < 0.05 * k * 5.0 for mu, k in zip(mus, ks))
    res["ice_friction_decel_near_zero_both"] = near_zero
    print(f"[holdout] ice_friction_decel (staged) {mus} vs 5% of the drag deceleration at 5 m/s {[round(0.05 * k * 5, 4) for k in ks]} -> near zero in both: {near_zero}")
    if res[f"{kind}_both_improve"]:
        names = FIT_NAMES if near_zero else [n for n in FIT_NAMES if n != "ice_friction_decel"]
        if near_zero:
            # Both directions put it at 0: fix it at exactly 0 (a scale-free statement) and fit the rest with it fixed,
            # so the pooled fit cannot reintroduce a small, scale-dependent value.
            p0 = with_values(params_path, {"ice_friction_decel": 0.0}, tmp / "mu0.json")
            glides = [s for s in ground if s["label"] == "glide"]
            bounces = [s for s in ground if s["label"] == "wall bounce"]
            a = fit(p0, glides, tmp / "A_all.json", names=["ice_linear_drag"])
            pa = with_values(p0, a, tmp / "A_all_only.json")
            b = fit(pa, bounces, tmp / "B_all.json", names=["board_restitution", "board_friction"], objective="path-only")
            vals = {**a, **b, "ice_friction_decel": 0.0}
        else:
            vals = staged_fit(params_path, ground, tmp, "all")  # ice_friction_decel stays at the placeholder (not written)
        vals = {n: vals[n] for n in names}
        pv = with_values(params_path, vals, tmp / "final.json")
        res["final"] = {"method": "staged fit on all ground segments", "names": names, "values": vals,
                        "all_ground": summary(evaluate(pv, ground)), "all_ground_baseline": summary({**base[clips[0]], **base[clips[1]]})}
        pf = json.loads(Path(params_path).read_text())
        gl = ", ".join(s["id"] for s in ground if s["label"] == "glide")
        bo = ", ".join(s["id"] for s in ground if s["label"] == "wall bounce")
        held = "; ".join(f"train {d['train'][:10]} -> held-out {d['test'][:10]} score {d['test_baseline']['score']:.2f} -> {d[kind]['test']['score']:.2f}" for d in res["directions"])
        for n in names:
            e = pf["params"][n]
            e["value"] = round(vals[n], 6)
            e["provenance"] = "fitted"
            src = f"glide segments ({gl}), path objective" if n.startswith("ice") else f"wall-bounce segments ({bo}), path-only objective (rebound geometry), ice params fixed from the glide fit"
            unit = {"ice_linear_drag": "1/s, scale-free", "ice_friction_decel": "fitted ~0 in both hold-out directions, so unit-free in practice",
                    "board_restitution": "ratio, scale-free", "board_friction": "ratio, scale-free"}[n]
            e["note"] = f"FITTED on real {src}; held out by clip: {held}. Positions/speeds at the PLACEHOLDER meter scale; {unit}."
        pf["_comment"] = "params.json + the first real fits (python -m measure holdout; see CLAUDE.md Fidelity log). " + pf.get("_comment", "")
        Path(write_to).write_text(json.dumps(pf, indent=2) + "\n")
        print(f"[holdout] wrote {write_to}: " + ", ".join(f"{n}={vals[n]:.4f}" for n in names))
        print(f"[holdout] all ground: baseline {res['final']['all_ground_baseline']} -> fitted {res['final']['all_ground']}")
    else:
        print("[holdout] staged fit does not improve in both directions: nothing written")
    (OUT / "holdout.json").write_text(json.dumps(res, indent=2) + "\n")
    return res
