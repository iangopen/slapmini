"""w5b skating model comparison, held out by clip, plus model-free turn and coast numbers from the clips.

    python -m measure.w5b_skate direct     # model-free: turns after a key change, coast rate by method and speed band
    python -m measure.w5b_skate compare    # candidates -> measure/results/skate/w5b_fits.json

`compare` needs the sim built with docs/experiments/skate_candidates.patch (git apply it; params player_move_model and
player_steer_tau exist only there, this script adds them to its scratch params; the sim keeps only the chosen model). Fits use calib fit (Nelder-Mead, time objective) through measure/skatefit.py; each candidate is fitted
on two skating clips and scored on the third, from several starts (the best training objective wins). player_max_speed
stays at the model-free measured cap.
  current : accel along the pressed direction, proportional coast, hard cap (turn 0, brake = accel)
  split   : + player_turn_accel on the sideways velocity (+ player_brake_accel, if any run presses against the velocity)
  turnkeep: split, but the sideways shrink keeps the speed (it turns the velocity; board slides survive)
  steer   : v -> target velocity with a time constant player_steer_tau; proportional coast
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import skatefit as sf
from .geometry import REPO

OUT = REPO / "measure" / "results" / "skate" / "w5b_fits.json"
SKATE = sf.SKATE_CLIPS

CANDIDATES = {
    "current": ({"player_move_model": 0, "player_turn_accel": 0}, ["player_accel", "player_coast_rate"],
                [{"player_accel": 17.0, "player_coast_rate": 0.9}]),
    "split": ({"player_move_model": 1, "player_brake_accel": 17.36}, ["player_accel", "player_coast_rate", "player_turn_accel"],
              [{"player_accel": 17.0, "player_coast_rate": 0.9, "player_turn_accel": t} for t in (3.0, 10.0, 30.0)]),
    "turnkeep": ({"player_move_model": 3, "player_brake_accel": 17.36}, ["player_accel", "player_coast_rate", "player_turn_accel"],
                 [{"player_accel": 17.0, "player_coast_rate": 0.9, "player_turn_accel": t} for t in (1.0, 4.0, 10.0)]),
    "steer": ({"player_move_model": 2}, ["player_steer_tau", "player_coast_rate"],
              [{"player_steer_tau": t, "player_coast_rate": 0.9} for t in (0.15, 0.4)]),
}


def params_file(base: dict, values: dict, name: str):
    raw = json.loads(json.dumps(base))
    for k, v in values.items():
        raw["params"].setdefault(k, {"value": 0.0, "provenance": "placeholder"})["value"] = float(v)
    sf.TMP.mkdir(parents=True, exist_ok=True)
    p = sf.TMP / f"w5b_{name}.json"
    p.write_text(json.dumps(raw, indent=1))
    return p


def score(ev):
    return float(np.sum([e["pos_rmse"] ** 2 + 0.1 * e["speed_rmse"] ** 2 for e in ev]))


def best_fit(base, fixed, names, starts, ids, tag):
    best = None
    for i, st in enumerate(starts):
        p0 = params_file(base, {**fixed, **st}, f"{tag}_s{i}")
        vals = sf.fit(ids, names, p0, f"w5b_{tag}_{i}")
        pf = params_file(base, {**fixed, **vals}, f"{tag}_f{i}")
        sc = score(sf.evaluate(ids, pf))
        if best is None or sc < best[0]:
            best = (sc, vals, pf)
    return best


def compare() -> int:
    man = json.loads((sf.OUT / "runs.json").read_text())
    ids = {c: [r["id"] for r in man["runs"] if r["clip"] == c] for c in SKATE}
    base = json.loads((REPO / "params.json").read_text())
    out = {"note": __doc__.strip().splitlines()[0], "models": {}}
    for name, (fixed, names, starts) in CANDIDATES.items():
        res = {}
        for held in SKATE:
            train = [i for c in SKATE if c != held for i in ids[c]]
            _, vals, pf = best_fit(base, fixed, names, starts, train, f"{name}_{held}")
            ev = sf.evaluate(ids[held], pf)
            res[held] = {"fit": vals, "pos_m": round(float(np.median([e["pos_rmse"] for e in ev])), 4),
                         "speed_m_s": round(float(np.median([e["speed_rmse"] for e in ev])), 4),
                         "per_run_pos": [round(e["pos_rmse"], 3) for e in ev]}
            print(f"{name:8s} held out {held:20s} pos {res[held]['pos_m']:.3f} m  speed {res[held]['speed_m_s']:.3f} m/s  {vals}", flush=True)
        _, vals, _ = best_fit(base, fixed, names, starts, [i for c in SKATE for i in ids[c]], f"{name}_all")
        res["final"] = vals
        print(f"{name:8s} final {vals}", flush=True)
        out["models"][name] = res
    # The turning clip is the only one with real turns, so a clip-held-out fold of it trains player_turn_accel on nothing.
    # Second test: fit the turn value on the other turning runs (plus nothing else) and score the run left out.
    tv = out["models"]["turnkeep"]["final"]
    runs = ids["whilemovingturning"]
    loro = {}
    for name, fixed, names, starts in (
        ("current", {"player_move_model": 0, **{k: v for k, v in out["models"]["current"]["final"].items()}}, ["player_accel"], [{"player_accel": 17.0}]),
        ("turnkeep", {"player_move_model": 3, "player_brake_accel": 17.36, "player_accel": tv["player_accel"], "player_coast_rate": tv["player_coast_rate"]},
         ["player_turn_accel"], [{"player_turn_accel": t} for t in (2.0, 6.0)]),
    ):
        rows = []
        for held in runs:
            _, vals, pf = best_fit(base, fixed, names, starts, [r for r in runs if r != held], f"loro_{name}_{held}")
            e = sf.evaluate([held], pf)[0]
            rows.append({"run": held, "fit": vals, "pos_m": round(e["pos_rmse"], 3), "speed_m_s": round(e["speed_rmse"], 3)})
        loro[name] = {"median_pos_m": float(np.median([r["pos_m"] for r in rows])), "median_speed_m_s": float(np.median([r["speed_m_s"] for r in rows])), "rows": rows}
        print(f"leave one turning run out, {name}: median pos {loro[name]['median_pos_m']:.3f} m, speed {loro[name]['median_speed_m_s']:.3f} m/s; {[(r['pos_m'], round(list(r['fit'].values())[0], 2)) for r in rows]}", flush=True)
    out["leave_one_turning_run_out"] = loro
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    return 0


def smooth_v(t, x, y, half):
    """Local quadratic velocity over +-half samples (noise-robust; returns vx, vy)."""
    n = len(t)
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(half, n - half):
        s = slice(i - half, i + half + 1)
        tt = t[s] - t[i]
        vx[i] = np.polyfit(tt, x[s], 2)[1]
        vy[i] = np.polyfit(tt, y[s], 2)[1]
    return vx, vy


def load_run(rid):
    sc = json.loads((sf.SCN / f"{rid}.scenario.json").read_text())
    rows = np.genfromtxt(sf.SCN / f"{rid}.csv", delimiter=",", names=True)
    return sc, rows


def key_at(sc, step):
    cur = (0.0, 0.0)
    for e in sc["inputs"]:
        if e["player"] == 0 and e["step"] <= step:
            cur = (e["input"]["move_x"], e["input"]["move_y"])
    return cur


def last_key_up(sc):
    """Step of the event that releases the last move key (None if a key is still held at the end)."""
    ev = [e for e in sc["inputs"] if e["player"] == 0]
    if ev[-1]["input"]["move_x"] or ev[-1]["input"]["move_y"]:
        return None
    return ev[-1]["step"]


def coast() -> dict:
    """Coast rate by method. raw = the s7 estimator (finite differences of the tracked position, speed > 1); smooth =
    local quadratic velocity; band = ln(speed) slope inside a speed band; pooled = regression of decel on speed."""
    man = json.loads((sf.OUT / "runs.json").read_text())
    dt = 1 / 120
    rates = {k: [] for k in ("raw_gt1", "smooth_gt1", "smooth_8_11", "smooth_4_8", "smooth_1_4", "first_1s", "after_1s")}
    pool = []
    for r in man["runs"]:
        if r["kind"] != "skate":
            continue
        sc, d = load_run(r["id"])
        up = last_key_up(sc)
        if up is None:
            continue
        t, x, y = d["t"], d["x"], d["y"]
        m = t > up * dt + 0.05
        if m.sum() < 40:
            continue
        tt, xx, yy = t[m], x[m], y[m]
        raw = np.hypot(np.gradient(xx, tt), np.gradient(yy, tt))
        vx, vy = smooth_v(tt, xx, yy, 7)
        sp = np.hypot(vx, vy)
        g = np.isfinite(sp)
        ok = raw > 1.0
        rates["raw_gt1"].append(-np.polyfit(tt[ok], np.log(raw[ok]), 1)[0])
        for key, sel in (("smooth_gt1", g & (sp > 1)), ("smooth_8_11", g & (sp > 8) & (sp <= 11)), ("smooth_4_8", g & (sp > 4) & (sp <= 8)), ("smooth_1_4", g & (sp > 1) & (sp <= 4)),
                         ("first_1s", g & (tt < tt[0] + 1.0)), ("after_1s", g & (tt >= tt[0] + 1.0) & (sp > 1))):
            if sel.sum() > 8:
                rates[key].append(float(-np.polyfit(tt[sel], np.log(sp[sel]), 1)[0]))
        pool.append((sp[g], -np.gradient(sp[g], tt[g])))
    res = {}
    print("== coast rate (1/s): ln(speed) slope after the last key-up ==")
    for k, v in rates.items():
        if not v:
            continue
        res[k] = {"n": len(v), "median": round(float(np.median(v)), 3), "mean": round(float(np.mean(v)), 3), "min": round(float(min(v)), 3), "max": round(float(max(v)), 3)}
        print(f"  {k:12s} n {len(v):2d}  median {np.median(v):.3f}  mean {np.mean(v):.3f}  range {min(v):.2f}-{max(v):.2f}")
    v = np.concatenate([a for a, _ in pool])
    a = np.concatenate([b for _, b in pool])
    print("  decel / speed by speed band (pooled; a pure proportional coast is flat):")
    for lo, hi in ((8, 13), (6, 8), (4, 6), (2, 4), (1, 2), (0.5, 1)):
        b = (v > lo) & (v <= hi)
        if b.sum() > 10:
            print(f"    {lo:>4}-{hi:<3} m/s  n {b.sum():4d}  decel {np.median(a[b]):6.2f} m/s^2  decel/v {np.median(a[b] / v[b]):.3f} 1/s")
    c = np.linalg.lstsq(np.column_stack([v, np.ones_like(v)]), a, rcond=None)[0]
    print(f"  pooled dv/dt = -{c[0]:.3f} v - {c[1]:.3f}")
    res["pooled"] = {"rate": round(float(c[0]), 3), "const": round(float(c[1]), 3)}
    return res


def replay_xy(rid, pf):
    out = sf.TMP / f"w5b_replay_{rid}.csv"
    sf.calib(["replay", "--params", str(pf), str(sf.SCN / f"{rid}.scenario.json"), str(out)])
    a = np.loadtxt(out, delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1], a[:, 2]


def turns(pfs: dict) -> list:
    """Every direction change of the keys held for >= 0.5 s while moving at > 8 m/s: time until the velocity is within
    10 deg of the key direction, in the clip and in the sim replays, through the same velocity smoothing."""
    man = json.loads((sf.OUT / "runs.json").read_text())
    dt = 1 / 120
    rows = []
    for r in man["runs"]:
        if r["kind"] != "skate":
            continue
        sc, d = load_run(r["id"])
        t = d["t"]
        tracks = {"clip": (d["x"], d["y"])}
        for name, pf in pfs.items():
            ts, xs, ys = replay_xy(r["id"], pf)
            tracks[name] = (np.interp(t, ts, xs), np.interp(t, ts, ys))
        ev = [e for e in sc["inputs"] if e["player"] == 0]
        for i, (a0, b0) in enumerate(zip(ev, ev[1:])):
            k0 = (a0["input"]["move_x"], a0["input"]["move_y"])
            k1 = (b0["input"]["move_x"], b0["input"]["move_y"])
            nxt = ev[i + 2]["step"] if i + 2 < len(ev) else sc["steps"]
            if k1 == (0, 0) or k1 == k0 or k0 == (0, 0) or (nxt - b0["step"]) * dt < 0.5:
                continue
            tk = b0["step"] * dt
            j0 = int(np.searchsorted(t, tk))
            row = {"run": r["id"], "t_key": round(tk, 2), "held_s": round((nxt - b0["step"]) * dt, 2)}
            kl = float(np.hypot(*k1))
            dx, dy = k1[0] / kl, k1[1] / kl
            for name, (x, y) in tracks.items():
                vx, vy = smooth_v(t, x, y, 7)
                ang = np.degrees(np.arctan2(vx * dy - vy * dx, vx * dx + vy * dy))
                row[name] = next((round(float(t[j] - tk), 3) for j in range(j0, len(t)) if np.isfinite(ang[j]) and abs(ang[j]) < 10), None)
                if name == "clip":
                    row["speed_mid"] = round(float(np.nanmedian(np.hypot(vx, vy)[j0 : j0 + 30])), 2)
            rows.append(row)
    return rows


def rest_to_top(pfs: dict) -> None:
    """Key down from rest to 95% of the cap, clip and sim, through the same smoothing (straight and diagonal runs)."""
    man = json.loads((sf.OUT / "runs.json").read_text())
    base = json.loads((REPO / "params.json").read_text())
    cap = base["params"]["player_max_speed"]["value"]
    res = {k: [] for k in ("clip", *pfs)}
    for r in man["runs"]:
        if r["kind"] != "skate" or not r["from_rest"]:
            continue
        sc, d = load_run(r["id"])
        ev = [e for e in sc["inputs"] if e["player"] == 0 and (e["input"]["move_x"] or e["input"]["move_y"])]
        tk = ev[0]["step"] / 120
        t = d["t"]
        tracks = {"clip": (d["x"], d["y"])}
        for name, pf in pfs.items():
            ts, xs, ys = replay_xy(r["id"], pf)
            tracks[name] = (np.interp(t, ts, xs), np.interp(t, ts, ys))
        for name, (x, y) in tracks.items():
            vx, vy = smooth_v(t, x, y, 7)
            sp = np.hypot(vx, vy)
            hit = np.nonzero((sp >= 0.95 * cap) & (t > tk))[0]
            if len(hit):
                res[name].append(float(t[hit[0]] - tk))
    print("== rest to 95% of the cap (s), median: " + ", ".join(f"{k} {np.median(v):.3f} (n {len(v)})" for k, v in res.items()))


def direct() -> int:
    coast()
    base = json.loads((REPO / "params.json").read_text())
    pf = {"old": params_file(base, {"player_turn_accel": 0, "player_accel": 17.3594, "player_coast_rate": 1.08146, "player_brake_accel": 17.3594}, "t_old"),
          "new": params_file(base, {}, "t_new")}
    rest_to_top(pf)
    rows = [r for r in turns(pf) if r["speed_mid"] > 8]
    print("== 45 deg key turns at speed, held >= 0.5 s: seconds until the velocity is within 10 deg of the key ==")
    for r in rows:
        print(f"  {r['run']:24s} t {r['t_key']:5.2f} held {r['held_s']:4.2f}  speed {r['speed_mid']:5.2f}  clip {r['clip']}  old {r['old']}  new {r['new']}")
    long = [r for r in rows if r["held_s"] >= 1.5]
    print(f"  held >= 1.5 s, n {len(long)}: median clip {np.median([r['clip'] for r in long]):.3f} s, old {np.median([r['old'] for r in long]):.3f} s, new {np.median([r['new'] for r in long]):.3f} s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m measure.w5b_skate")
    ap.add_argument("cmd", choices=["direct", "compare"])
    a = ap.parse_args(argv)
    return direct() if a.cmd == "direct" else compare()


if __name__ == "__main__":
    raise SystemExit(main())
