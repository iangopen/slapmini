"""s10 player fits: proportional coast + set-speed dash vs the old constant-drag + additive-impulse models.

    python -m measure.s10_skate [--old-calib PATH]     # -> measure/results/skate/s10_fits.json

Runs: calib/data/skate/ (real meters), grouped by clip as in measure/skatefit.py. calib fit (Nelder-Mead, time
objective), HELD OUT BY CLIP: stage A (skating clips: wkeymovement, diagonalkeypresses, whilemovingturning; fit on two,
evaluate on the third), stage B (dash clips: dashatrest, dashwhilerunning; fit on one, evaluate on the other, stage A
fixed). player_max_speed is fixed at the model-free measured cap in both models.
  new A: player_accel, player_coast_rate, player_ice_drag (start 0.3: the fitter scales by the start value)
  new B: dash_speed, dash_ramp_time
  old A: player_accel, player_ice_drag; old B: dash_speed, dash_duration (as s7) -- needs the OLD calib binary
         (a build of origin/main before s10; --old-calib), with the s10 keys stripped from the params.
  placeholder: params.placeholder.json physics (old binary; the new keys stripped).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import skatefit as sf
from .geometry import REPO

OUT = REPO / "measure" / "results" / "skate" / "s10_fits.json"
NEW_KEYS = ("player_coast_rate", "dash_ramp_time", "stick_max_ang_decel")
SKATE, DASH = sf.SKATE_CLIPS, sf.DASH_CLIPS


def write_params(base: dict, values: dict, path: Path, strip_new: bool = False) -> Path:
    raw = json.loads(json.dumps(base))
    for k, v in values.items():
        raw["params"][k]["value"] = float(v)
    if strip_new:
        for k in NEW_KEYS:
            raw["params"].pop(k, None)
    path.write_text(json.dumps(raw, indent=1))
    return path


def summarize(ev):
    return {"pos_rmse_median_m": round(float(np.median([e["pos_rmse"] for e in ev])), 4),
            "speed_rmse_median_m_s": round(float(np.median([e["speed_rmse"] for e in ev])), 4), "n": len(ev),
            "per_run": {e["id"]: [round(e["pos_rmse"], 3), round(e["speed_rmse"], 3)] for e in ev}}


def stage(model: str, calib_bin: Path, base: dict, ids: dict, names_a, names_b, start_a: dict, start_b: dict) -> dict:
    sf.CALIB = calib_bin
    strip = model == "old"
    tmp = sf.TMP
    tmp.mkdir(parents=True, exist_ok=True)
    res = {"A": {}, "B": {}}
    p0 = write_params(base, {**start_a, **start_b}, tmp / f"{model}_start.json", strip)
    for held in SKATE:
        vals = sf.fit([i for c in SKATE if c != held for i in ids[c]], names_a, p0, f"s10_{model}_A_{held}")
        pf = write_params(base, {**start_b, **vals}, tmp / f"{model}_A_{held}.json", strip)
        res["A"][held] = {"fitted_on_others": vals, "test": summarize(sf.evaluate(ids[held], pf))}
    final_a = sf.fit([i for c in SKATE for i in ids[c]], names_a, p0, f"s10_{model}_A_all")
    res["A"]["final"] = final_a
    pa = write_params(base, {**start_b, **final_a}, tmp / f"{model}_A_final.json", strip)
    for held in DASH:
        other = [c for c in DASH if c != held][0]
        vals = sf.fit(ids[other], names_b, pa, f"s10_{model}_B_{held}")
        pf = write_params(base, {**final_a, **vals}, tmp / f"{model}_B_{held}.json", strip)
        res["B"][held] = {"fitted_on_other": vals, "test": summarize(sf.evaluate(ids[held], pf))}
    res["B"]["final"] = sf.fit([i for c in DASH for i in ids[c]], names_b, pa, f"s10_{model}_B_all")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m measure.s10_skate")
    ap.add_argument("--old-calib", type=Path, required=True, help="calib binary built from the pre-s10 code")
    a = ap.parse_args(argv)
    man = json.loads((sf.OUT / "runs.json").read_text())
    ids = {c: [r["id"] for r in man["runs"] if r["clip"] == c] for c in SKATE + DASH}
    base = json.loads((REPO / "params.json").read_text())
    placeholder = json.loads((REPO / "params.placeholder.json").read_text())
    P = {k: v["value"] for k, v in base["params"].items()}
    out = {"note": __doc__.strip().splitlines()[0]}
    out["new"] = stage("new", sf.CALIB, base, ids, ["player_accel", "player_coast_rate", "player_ice_drag"], ["dash_speed", "dash_ramp_time"],
                       {"player_accel": P["player_accel"], "player_coast_rate": 0.85, "player_ice_drag": 0.3},
                       {"dash_speed": 13.0, "dash_ramp_time": 0.13})
    out["old"] = stage("old", a.old_calib, base, ids, ["player_accel", "player_ice_drag"], ["dash_speed", "dash_duration"],
                       {"player_accel": P["player_accel"], "player_ice_drag": P["player_ice_drag"]},
                       {"dash_speed": P["dash_speed"], "dash_duration": 0.25})
    sf.CALIB = a.old_calib
    pp = sf.TMP / "s10_placeholder.json"
    raw = json.loads(json.dumps(placeholder))
    for k in NEW_KEYS:
        raw["params"].pop(k, None)
    pp.write_text(json.dumps(raw, indent=1))
    out["placeholder"] = {c: summarize(sf.evaluate(ids[c], pp)) for c in SKATE + DASH}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for c in SKATE:
        print(f"A {c:20s} placeholder {out['placeholder'][c]['pos_rmse_median_m']:7.3f} m  old {out['old']['A'][c]['test']['pos_rmse_median_m']:7.3f} m "
              f"/ {out['old']['A'][c]['test']['speed_rmse_median_m_s']:.3f} m/s  new {out['new']['A'][c]['test']['pos_rmse_median_m']:7.3f} m "
              f"/ {out['new']['A'][c]['test']['speed_rmse_median_m_s']:.3f} m/s   new fit {out['new']['A'][c]['fitted_on_others']}")
    for c in DASH:
        print(f"B {c:20s} placeholder {out['placeholder'][c]['pos_rmse_median_m']:7.3f} m  old {out['old']['B'][c]['test']['pos_rmse_median_m']:7.3f} m "
              f"/ {out['old']['B'][c]['test']['speed_rmse_median_m_s']:.3f} m/s  new {out['new']['B'][c]['test']['pos_rmse_median_m']:7.3f} m "
              f"/ {out['new']['B'][c]['test']['speed_rmse_median_m_s']:.3f} m/s   new fit {out['new']['B'][c]['fitted_on_other']}")
    print("final new:", out["new"]["A"]["final"], out["new"]["B"]["final"])
    print("final old:", out["old"]["A"]["final"], out["old"]["B"]["final"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
