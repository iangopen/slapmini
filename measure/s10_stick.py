"""s10 stick chase: old symmetric bang-bang vs a separate deceleration limit vs + overshoot term, held out by event.

    python -m measure.s10_stick            # -> measure/results/s10_stick_chase.json, prints the table

Same events, windows and step-target model as measure/stick_chase.py (4 cap-reaching step events of rotatinganddizzy,
target = the measured final rest angle, unknown onset per event). The chase law below emulates
sim/src/step.rs `stick_chase` (s10) exactly; `sim/examples/stick_step` checks it (rust_check). stick_max_ang_speed is
fixed at the measured plateau (26.1726 rad/s). Held out by event: the globals are fitted on the other 3 events (with
their onsets), then only the onset is fitted on the held-out event and its angle RMSE is reported.
Models: old (accel = decel, one value), decel (accel, decel), decel_acc514 (decel only, accel = the s7b flick-start
value 513.9), overshoot / overshoot_acc514 (+ an overshoot term: the brake aims over * w / w_max past the target).
Result (s10): decel models beat old on every fold mean; the overshoot term does not help, so the sim has none.
"""

from __future__ import annotations

import json
import math
import subprocess

import numpy as np
from scipy.optimize import least_squares

from .geometry import PARAMS_PATH, REPO
from .stick_chase import RESULTS, load

W_MAX = 26.172589259915693


def chase(t_eval, t0, a0, target, w_max, acc, dec, over, dt, nsub):
    tau = t_eval - t0
    n = int(math.ceil(max(float(tau.max()), 0.0) / dt)) + 3
    ang = np.empty(n + 1)
    a, w = a0, 0.0
    ang[0] = a
    h = dt / nsub
    for k in range(n):
        err = target - a + over * w / w_max
        brake = math.sqrt(2.0 * dec * abs(err))
        desired = math.copysign(min(brake, w_max, abs(err) / dt), err) if err != 0 else 0.0
        dv = desired - w
        if w == 0.0 or w * dv > 0.0:
            w = w + min(max(dv, -acc * dt), acc * dt)
        else:
            slow = min(abs(dv), dec * dt)
            if slow <= abs(w):
                w = w - math.copysign(slow, w)
            else:
                left = max(1.0 - abs(w) / (dec * dt), 0.0)
                w = math.copysign(min(abs(dv) - abs(w), acc * dt * left), dv)
        for _ in range(nsub):
            a += w * h
        ang[k + 1] = a
    ts = np.arange(n + 1) * dt
    return np.where(tau <= 0, a0, np.interp(np.clip(tau, 0, None), ts, ang))


MODELS = {
    # name: (global names, start values, lower, upper, unpack -> (acc, dec, over))
    "old": (["accel"], [math.log(500.0)], [math.log(50.0)], [math.log(20000.0)], lambda g: (math.exp(g[0]), math.exp(g[0]), 0.0)),
    "decel": (["accel", "decel"], [math.log(500.0), math.log(2500.0)], [math.log(50.0)] * 2, [math.log(20000.0)] * 2,
              lambda g: (math.exp(g[0]), math.exp(g[1]), 0.0)),
    # accel fixed at the s7b flick-start value (stick_max_ang_accel 513.9), only the deceleration fitted
    "decel_acc514": (["decel"], [math.log(2500.0)], [math.log(50.0)], [math.log(20000.0)], lambda g: (513.9, math.exp(g[0]), 0.0)),
    "overshoot_acc514": (["decel", "overshoot"], [math.log(2500.0), 0.05], [math.log(50.0), 0.0], [math.log(20000.0), 0.6],
                         lambda g: (513.9, math.exp(g[0]), g[1])),
    "overshoot": (["accel", "decel", "overshoot"], [math.log(500.0), math.log(2500.0), 0.05], [math.log(50.0)] * 2 + [0.0],
                  [math.log(20000.0)] * 2 + [0.6], lambda g: (math.exp(g[0]), math.exp(g[1]), g[2])),
}


def resid_event(e, t, a, t0, g, model, dt, nsub):
    i0, i1, a0, a1 = e["win"]
    acc, dec, over = MODELS[model][4](g)
    return chase(t[i0:i1 + 1], t0, a0, a1, W_MAX, acc, dec, over, dt, nsub) - a[i0:i1 + 1]


def fit_globals(events, t, a, model, dt, nsub):
    names, g0, lo, hi, _ = MODELS[model]
    k = len(g0)
    ons0 = [e["t0"] - 1.0 / 60 for e in events]
    best = None
    for s in ([1.0, 1.0], [0.8, 1.3], [1.2, 0.8]):
        x0 = [v + math.log(s[0] if j == 0 else s[1]) if j < 2 and v > 1.0 else v for j, v in enumerate(g0)] + ons0
        x0 = [min(max(v, l + 1e-6), u - 1e-6) for v, l, u in zip(x0, lo + [o - 0.3 for o in ons0], hi + [o + 0.2 for o in ons0])]
        r = least_squares(lambda x: np.concatenate([resid_event(e, t, a, x[k + i], x[:k], model, dt, nsub) for i, e in enumerate(events)]),
                          x0, bounds=(lo + [e["t0"] - 0.3 for e in events], hi + [e["t0"] + 0.2 for e in events]),
                          loss="soft_l1", f_scale=0.1, diff_step=1e-3)
        if best is None or r.cost < best.cost:
            best = r
    return best.x[:k]


def heldout_rmse(e, t, a, g, model, dt, nsub):
    r = least_squares(lambda x: resid_event(e, t, a, x[0], g, model, dt, nsub), [e["t0"] - 1.0 / 60],
                      bounds=([e["t0"] - 0.3], [e["t0"] + 0.2]), loss="soft_l1", f_scale=0.1, diff_step=1e-3)
    d = resid_event(e, t, a, r.x[0], g, model, dt, nsub)
    return float(np.degrees(np.sqrt(np.mean(d * d)))), float(r.x[0])


def overshoot_deg(e, t, a, g, model, t0, dt, nsub):
    i0, i1, a0, a1 = e["win"]
    acc, dec, over = MODELS[model][4](g)
    tt = np.linspace(t[i0], t[i1] + 0.3, 600)
    s = chase(tt, t0, a0, a1, W_MAX, acc, dec, over, dt, nsub)
    sg = 1.0 if a1 > a0 else -1.0
    return float(np.degrees(max(0.0, ((s - a1) * sg).max())))


def rust_check(acc, dec, over, dt, nsub, target=2.0, total=0.4) -> float:
    base = json.loads(PARAMS_PATH.read_text())
    assert over == 0.0, "the sim has no overshoot term (it was tested here and not added)"
    for k, v in (("stick_max_ang_speed", W_MAX), ("stick_max_ang_accel", acc), ("stick_max_ang_decel", dec)):
        base["params"][k]["value"] = v
    tmp = REPO / "target" / "s10_stick_check_params.json"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(json.dumps(base))
    out = subprocess.run(["cargo", "run", "-q", "--release", "-p", "sim", "--example", "stick_step", "--", str(tmp), str(target), str(total)],
                         cwd=REPO, check=True, capture_output=True, text=True).stdout
    arr = np.array([[float(v) for v in ln.split(",")] for ln in out.strip().splitlines()[1:]])
    py = chase(arr[:, 0], 0.0, 0.0, target, W_MAX, acc, dec, over, dt, nsub)
    return float(np.abs(py - arr[:, 1]).max())


def main() -> int:
    P = json.loads(PARAMS_PATH.read_text())["params"]
    dt, nsub = P["timestep"]["value"], max(1, int(round(P["substeps"]["value"])))
    t, a, _, _ = load(RESULTS)
    events = json.loads((RESULTS / "stick_chase.json").read_text())["step_events"]
    out = {"w_max": W_MAX, "events": [{"t0": e["t0"], "dtheta_deg": round(math.degrees(e["win"][3] - e["win"][2]), 1)} for e in events],
           "measured_overshoot_deg": json.loads((RESULTS / "stick_chase.json").read_text())["overshoot_deg"], "models": {}}
    for model in MODELS:
        rows = []
        for i, e in enumerate(events):
            g = fit_globals([x for j, x in enumerate(events) if j != i], t, a, model, dt, nsub)
            rm, t0 = heldout_rmse(e, t, a, g, model, dt, nsub)
            rows.append({"event": i, "train": [float(v) for v in MODELS[model][4](g)], "rmse_deg": round(rm, 3),
                         "sim_overshoot_deg": round(overshoot_deg(e, t, a, g, model, t0, dt, nsub), 2)})
        gall = fit_globals(events, t, a, model, dt, nsub)
        out["models"][model] = {"held_out": rows, "mean_rmse_deg": round(float(np.mean([r["rmse_deg"] for r in rows])), 3),
                                "fit_all": [float(v) for v in MODELS[model][4](gall)]}
        print(model, "held-out rmse deg:", [r["rmse_deg"] for r in rows], "mean", out["models"][model]["mean_rmse_deg"],
              "fit_all (acc, dec, over):", [round(v, 4) for v in out["models"][model]["fit_all"]])
    acc, dec, _ = out["models"]["decel_acc514"]["fit_all"]
    out["rust_check_max_diff_rad"] = rust_check(acc, dec, 0.0, dt, nsub)
    print("rust check max |py - rust| rad:", out["rust_check_max_diff_rad"])
    (RESULTS / "s10_stick_chase.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
