"""Stick chase fit from the sweep clip (`python -m measure.stick chase`).

The mouse is not visible, so each move is modelled as the sim's stick answering a STEP in the target: from rest at
the measured start angle, the target jumps at an unknown onset t0 to the measured final rest angle. The sim's chase
(sim/src/step.rs update_player_controls + the substep integration) is emulated in Python (checked against the Rust
sim via examples/stick_step), and stick_max_ang_speed + stick_max_ang_accel are fitted jointly over the step events
with a free onset per event (robust soft-L1 loss: capture timing jitter puts ~0.1 rad of noise on single frames at
26 rad/s). Held out by event: globals fitted on the other events, only the onset fitted on the held-out one.

Which moves are steps: moves that start and end at rest and reach the speed cap (the stick then moves at its own
limit, whatever the mouse did in between). Slow moves (the mouse circling below the cap) follow the mouse, not a
step, and are only used to check that the cap is never exceeded. Multi-turn spins: the game kept turning the stick
the same way while the mouse circled, so the target is the UNWRAPPED final angle (the sim wraps the target error to
[-pi, pi]; it reproduces such a spin only while the mouse leads by less than half a turn, which a circling mouse does).
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .geometry import PARAMS_PATH, REPO, load_params

RESULTS = REPO / "measure" / "results"
CLIP = "rotatinganddizzy"
REST_W = 0.6  # rad/s: at rest
REST_FRAMES = 6
CAP_HIT = 22.0  # rad/s: a move that reaches this is limited by the stick, not the mouse
PAD = 0.12  # s of rest included before and after each move


def chase(t_eval: np.ndarray, t0: float, a0: float, target: float, w_max: float, acc: float, dt: float, nsub: int, w0: float = 0.0) -> np.ndarray:
    """Sim stick from rest at a0, target from t0 (exact emulation of the sim's control law, unwrapped error)."""
    tau = t_eval - t0
    T = max(float(tau.max()), 0.0) + 2 * dt
    n = int(math.ceil(T / dt)) + 1
    ang = np.empty(n + 1)
    a, w = a0, w0
    ang[0] = a
    h = dt / nsub
    for k in range(n):
        err = target - a
        brake = math.sqrt(2.0 * acc * abs(err))
        desired = math.copysign(min(brake, w_max, abs(err) / dt), err) if err != 0 else 0.0
        dw = min(max(desired - w, -acc * dt), acc * dt)
        w += dw
        for _ in range(nsub):
            a += w * h
        ang[k + 1] = a
    ts = np.arange(n + 1) * dt
    return np.where(tau <= 0, a0 + w0 * tau, np.interp(np.clip(tau, 0, None), ts, ang))


def load(results: Path = RESULTS):
    m = json.loads((results / f"{CLIP}.stick.json").read_text())
    R = m["rows"]
    f = np.array(R["frame"])
    return f / 60.0, np.array(R["angle_unwrapped"]), np.array(R["conf"]), f


def find_moves(t, a):
    """Rest-to-rest moves: (i0, i1) index ranges with their peak |w| and whether they reverse."""
    w = np.gradient(a, t)
    rest = np.abs(w) < REST_W
    # rest runs of >= REST_FRAMES
    is_rest = np.zeros(len(a), bool)
    i = 0
    while i < len(a):
        if rest[i]:
            j = i
            while j < len(a) and rest[j]:
                j += 1
            if j - i >= REST_FRAMES:
                is_rest[i:j] = True
            i = j
        else:
            i += 1
    moves = []
    i = 0
    while i < len(a):
        if not is_rest[i]:
            j = i
            while j < len(a) and not is_rest[j]:
                j += 1
            if i > 0 and j < len(a):
                seg = w[i:j]
                big = np.abs(seg) > 0.5 * np.abs(seg).max()
                signs = np.sign(seg[big])
                moves.append({"i0": i, "i1": j - 1, "peak": float(np.abs(seg).max()), "reverses": bool((signs != signs[0]).any()),
                              "dtheta": float(a[j] - a[i - 1]), "t0": float(t[i]), "t1": float(t[j - 1])})
            i = j
        else:
            i += 1
    # A cap-limited move ends where the stick stops after its last cap frame (later mouse wiggles are separate).
    for m in moves:
        seg = np.abs(w[m["i0"] : m["i1"] + 1])
        if seg.max() >= CAP_HIT:
            last_cap = m["i0"] + int(np.where(seg >= CAP_HIT)[0].max())
            k = last_cap
            while k < m["i1"] and abs(w[k]) >= 4.0:
                k += 1
            if k < m["i1"]:
                m["i1"] = k
                m["t1"] = float(t[k])
                m["trimmed"] = True
                m["dtheta"] = float(np.median(a[k + 1 : k + 4]) - a[m["i0"] - 1])
    return moves, w


def window(t, a, mv):
    i0 = max(0, int(np.searchsorted(t, mv["t0"] - PAD)))
    i1 = min(len(t) - 1, int(np.searchsorted(t, mv["t1"] + PAD)))
    a0 = float(np.median(a[max(0, mv["i0"] - REST_FRAMES) : mv["i0"]]))
    a1 = float(np.median(a[mv["i1"] + 1 : mv["i1"] + 1 + (3 if mv.get("trimmed") else REST_FRAMES)]))
    if mv.get("trimmed"):
        i1 = min(i1, mv["i1"] + 3)
    return i0, i1, a0, a1


def fit(events, t, a, dt, nsub, fixed=None, x0=(26.0, 1000.0)):
    """Joint fit: globals (w_max, acc) unless fixed, plus one onset per event. Returns (w_max, acc, onsets, rmse_deg list)."""
    def unpack(x):
        if fixed:
            return fixed[0], fixed[1], x
        return x[0], x[1], x[2:]

    def resid(x):
        wm, ac, ons = unpack(x)
        out = []
        for e, t0 in zip(events, ons):
            i0, i1, a0, a1 = e["win"]
            out.append(chase(t[i0 : i1 + 1], t0, a0, a1, wm, ac, dt, nsub) - a[i0 : i1 + 1])
        return np.concatenate(out)

    ons0 = [e["t0"] - 1.0 / 60 for e in events]
    if fixed:
        xs, lo, hi = ons0, [e["t0"] - 0.3 for e in events], [e["t0"] + 0.2 for e in events]
    else:
        xs = [x0[0], math.log(x0[1])] + ons0
        lo = [5.0, math.log(50.0)] + [e["t0"] - 0.3 for e in events]
        hi = [60.0, math.log(20000.0)] + [e["t0"] + 0.2 for e in events]

        def resid_log(x):
            return resid(np.r_[x[0], math.exp(x[1]), x[2:]])

    best = None
    starts = [xs] if fixed else [xs, [20.0, math.log(300.0)] + ons0, [30.0, math.log(3000.0)] + ons0]
    for s0 in starts:
        r = least_squares(resid if fixed else resid_log, s0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1, diff_step=1e-3)
        if best is None or r.cost < best.cost:
            best = r
    x = best.x if fixed else np.r_[best.x[0], math.exp(best.x[1]), best.x[2:]]
    wm, ac, ons = unpack(x)
    rmses = []
    for e, t0 in zip(events, ons):
        i0, i1, a0, a1 = e["win"]
        d = chase(t[i0 : i1 + 1], t0, a0, a1, wm, ac, dt, nsub) - a[i0 : i1 + 1]
        rmses.append(float(np.degrees(np.sqrt(np.mean(d * d)))))
    return float(wm), float(ac), list(map(float, ons)), rmses


def rust_check(dt, nsub, w_max, acc, target=2.0, total=0.4) -> float:
    """Max |python - rust| stick angle over a step response (rad)."""
    base = json.loads(PARAMS_PATH.read_text())
    base["params"]["stick_max_ang_speed"]["value"] = w_max
    base["params"]["stick_max_ang_accel"]["value"] = acc
    tmp = REPO / "target" / "stick_check_params.json"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(json.dumps(base))
    out = subprocess.run(["cargo", "run", "-q", "--release", "-p", "sim", "--example", "stick_step", "--", str(tmp), str(target), str(total)],
                         cwd=REPO, check=True, capture_output=True, text=True).stdout
    arr = np.array([[float(v) for v in ln.split(",")] for ln in out.strip().splitlines()[1:]])
    py = chase(arr[:, 0], 0.0, 0.0, target, w_max, acc, dt, nsub)
    return float(np.abs(py - arr[:, 1]).max())


def run_chase(results: Path = RESULTS) -> dict:
    p = load_params(PARAMS_PATH)
    dt, nsub = p["timestep"], max(1, int(round(p["substeps"])))
    t, a, conf, frames = load(results)
    moves, w = find_moves(t, a)
    steps = [m for m in moves if m["peak"] >= CAP_HIT and not m["reverses"]]
    for m in steps:
        m["win"] = window(t, a, m)
    # Cap from the plateaus: frames inside step moves at >= 90% of the move's peak (frame-difference speed).
    wf = np.abs(np.diff(a)) * 60
    plate = []
    for m in moves:
        if m["peak"] < CAP_HIT:
            continue
        seg = wf[m["i0"] : m["i1"]]
        plate += seg[seg > 0.9 * np.median(seg[seg > 0.5 * seg.max()])].tolist()
    plate = np.array(plate)
    slow = [m for m in moves if m["peak"] < CAP_HIT]
    print(f"[chase] {len(moves)} rest-to-rest moves; {len(steps)} step events (reach the cap, no reversal); "
          f"{sum(1 for m in moves if m['reverses'])} with reversals; {len(slow)} slow (follow the mouse)")
    print(f"[chase] plateau speed (frame differences inside cap-limited moves): median {np.median(plate):.2f} rad/s, "
          f"mean {plate.mean():.2f}, sd {plate.std():.2f}, n = {len(plate)} frames; slow moves peak <= {max((m['peak'] for m in slow), default=0):.1f} rad/s")
    for m in steps:
        print(f"  step  t {m['t0']:6.2f}-{m['t1']:6.2f} s  d theta {math.degrees(m['dtheta']):+8.1f} deg  peak {m['peak']:.1f} rad/s")
    wm, ac, ons, rm = fit(steps, t, a, dt, nsub)
    print(f"[chase] all-event fit: stick_max_ang_speed {wm:.2f} rad/s, stick_max_ang_accel {ac:.0f} rad/s^2; per-event RMSE "
          + ", ".join(f"{x:.2f}" for x in rm) + " deg")
    # Profile of the accel: how well is it determined? (RMSE sum vs acc, onsets refit)
    prof = []
    for acc in (300, 600, 1000, 1500, 2500, 5000, 10000):
        _, _, _, r = fit(steps, t, a, dt, nsub, fixed=(wm, acc))
        prof.append({"accel": acc, "rmse_deg_median": round(float(np.median(r)), 3), "rmse_deg_mean": round(float(np.mean(r)), 3)})
    print("[chase] accel profile (w_max fixed, onsets refit): " + "; ".join(f"{q['accel']}: {q['rmse_deg_median']:.2f}/{q['rmse_deg_mean']:.2f}" for q in prof))
    # Hold out by event.
    held = []
    plh = (p["stick_max_ang_speed"], p["stick_max_ang_accel"])
    for k, e in enumerate(steps):
        tr = [x for j, x in enumerate(steps) if j != k]
        wk, akk, _, _ = fit(tr, t, a, dt, nsub)
        _, _, _, r_fit = fit([e], t, a, dt, nsub, fixed=(wk, akk))
        _, _, _, r_plh = fit([e], t, a, dt, nsub, fixed=plh)
        held.append({"event": k, "t0": round(e["t0"], 3), "dtheta_deg": round(math.degrees(e["dtheta"]), 1), "train_w_max": round(wk, 3),
                     "train_accel": round(akk, 1), "rmse_deg_fitted": round(r_fit[0], 3), "rmse_deg_placeholder": round(r_plh[0], 3)})
        print(f"[chase] held out event {k} (t {e['t0']:.2f} s, {math.degrees(e['dtheta']):+.0f} deg): train w_max {wk:.2f}, accel {akk:.0f} -> "
              f"RMSE {r_fit[0]:.2f} deg; placeholder ({plh[0]:g}, {plh[1]:g}) {r_plh[0]:.2f} deg")
    # Braking shape: residual pattern at the stop (last 6 frames before rest) for the all-event fit.
    stops = []
    for e, t0 in zip(steps, ons):
        i0, i1, a0, a1 = e["win"]
        j1 = e["i1"]
        tt = t[j1 - 5 : j1 + 4]
        sim = chase(tt, t0, a0, a1, wm, ac, dt, nsub)
        stops.append(np.degrees(a[j1 - 5 : j1 + 4] - sim).round(2).tolist())
    over = []  # overshoot past the final angle, in the move direction
    for e in steps:
        s = np.sign(e["dtheta"])
        seg = a[e["i1"] - 3 : e["i1"] + 8]
        over.append(float(np.degrees(np.max(s * (seg - e["win"][3])))))
    print("[chase] stop residuals (data - sim, deg, frames -5..+3 around the last moving frame): " + " | ".join(str(x) for x in stops))
    print("[chase] overshoot past the final angle: " + ", ".join(f"{x:.2f}" for x in over) + " deg")
    # --- Acceleration from the phases the mouse cannot shape. Stops: from 4 frames before the end of the cap run
    # to 6 frames after rest, starting at the measured angle with w = +-w_max, target = final angle; free time offset.
    capm = [m for m in moves if m["peak"] >= CAP_HIT]
    stop_ev, start_ev = [], []
    for m in capm:
        seg = np.abs(w[m["i0"] : m["i1"] + 1])
        cap_idx = m["i0"] + np.where(seg >= CAP_HIT)[0]
        lc, fc = int(cap_idx.max()), int(cap_idx.min())
        sgn = float(np.sign(w[lc]))
        k_end = min(m["i1"] + 6, len(a) - 1)
        target = float(np.median(a[m["i1"] + 1 : m["i1"] + 4]))
        stop_ev.append({"i0": lc - 4, "i1": k_end, "a0": float(a[lc - 4]), "w0": sgn * wm, "target": target, "t_start": float(t[lc - 4])})
        if not m.get("reverses") or True:
            # start: rest before the move -> 3 frames after the first cap frame; target = far ahead (step)
            j0 = max(m["i0"] - 6, 0)
            start_ev.append({"i0": j0, "i1": fc + 3, "a0": float(np.median(a[j0 : m["i0"]])), "dir": float(np.sign(w[fc])), "t_start": float(t[m["i0"]]),
                             "ramp_frames": int(fc - m["i0"])})

    def stop_resid(acc, ev, offs):
        out = []
        for e, o in zip(ev, offs):
            tt = t[e["i0"] : e["i1"] + 1]
            out.append(chase(tt, e["t_start"] + o, e["a0"], e["target"], wm, acc, dt, nsub, w0=e["w0"]) - a[e["i0"] : e["i1"] + 1])
        return np.concatenate(out)

    def fit_stop(acc_fixed=None, ev=stop_ev):
        if acc_fixed is None:
            f_ = lambda x: stop_resid(math.exp(x[0]), ev, x[1:])  # noqa: E731
            best = min((least_squares(f_, [math.log(a0_)] + [0.0] * len(ev), bounds=([math.log(50)] + [-0.05] * len(ev), [math.log(1e5)] + [0.05] * len(ev)),
                                      loss="soft_l1", f_scale=0.1, diff_step=1e-3) for a0_ in (300, 1500, 8000)), key=lambda r: r.cost)
            acc, offs = math.exp(best.x[0]), best.x[1:]
        else:
            f_ = lambda x: stop_resid(acc_fixed, ev, x)  # noqa: E731
            r = least_squares(f_, [0.0] * len(ev), bounds=([-0.05] * len(ev), [0.05] * len(ev)), loss="soft_l1", f_scale=0.1, diff_step=1e-3)
            acc, offs = acc_fixed, r.x
        d = stop_resid(acc, ev, offs)
        return acc, float(np.degrees(np.sqrt(np.mean(d * d))))

    acc_stop, rm_stop = fit_stop()
    stop_prof = {A: round(fit_stop(A)[1], 3) for A in (200, 400, 800, 1500, 3000, 6000, 20000)}
    print(f"[chase] STOPS ({len(stop_ev)} cap-limited moves): best accel {acc_stop:.0f} rad/s^2 (RMSE {rm_stop:.2f} deg); profile "
          + ", ".join(f"{k}: {v:.2f}" for k, v in stop_prof.items()))

    def start_resid(acc, ev, ons):
        out = []
        for e, o in zip(ev, ons):
            tt = t[e["i0"] : e["i1"] + 1]
            out.append(chase(tt, o, e["a0"], e["a0"] + e["dir"] * 1e3, wm, acc, dt, nsub) - a[e["i0"] : e["i1"] + 1])
        return np.concatenate(out)

    flick = [e for e in start_ev if e["ramp_frames"] <= 4]
    def fit_start(acc_fixed=None, ev=flick):
        o0 = [e["t_start"] - 1 / 60 for e in ev]
        lo, hi = [e["t_start"] - 0.1 for e in ev], [e["t_start"] + 0.05 for e in ev]
        if acc_fixed is None:
            f_ = lambda x: start_resid(math.exp(x[0]), ev, x[1:])  # noqa: E731
            best = min((least_squares(f_, [math.log(a0_)] + o0, bounds=([math.log(50)] + lo, [math.log(1e5)] + hi), loss="soft_l1", f_scale=0.1,
                                      diff_step=1e-3) for a0_ in (300, 1500, 8000)), key=lambda r: r.cost)
            acc, ons_ = math.exp(best.x[0]), best.x[1:]
        else:
            r = least_squares(lambda x: start_resid(acc_fixed, ev, x), o0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1, diff_step=1e-3)
            acc, ons_ = acc_fixed, r.x
        d = start_resid(acc, ev, ons_)
        return acc, float(np.degrees(np.sqrt(np.mean(d * d))))

    if flick:
        acc_start, rm_start = fit_start()
        start_prof = {A: round(fit_start(A)[1], 3) for A in (200, 400, 800, 1500, 3000, 6000, 20000)}
        print(f"[chase] FLICK STARTS ({len(flick)} moves reaching the cap within 4 frames of leaving rest): best accel {acc_start:.0f} rad/s^2 "
              f"(RMSE {rm_start:.2f} deg); profile " + ", ".join(f"{k}: {v:.2f}" for k, v in start_prof.items()))
    else:
        acc_start, rm_start, start_prof = None, None, {}
    print("[chase] ramp frames from rest to the cap, all cap-limited moves: " + ", ".join(str(e["ramp_frames"]) for e in start_ev))
    try:
        diff = rust_check(dt, nsub, wm, ac)
        print(f"[chase] python emulation vs Rust sim (examples/stick_step), step of 2 rad at the fitted values: max |d angle| = {diff:.2e} rad")
    except Exception as ex:  # cargo missing
        diff = None
        print(f"[chase] Rust check skipped: {ex}")
    res = {
        "clip": CLIP, "dt": dt, "substeps": nsub,
        "moves": [{k: v for k, v in m.items() if k != "win"} for m in moves],
        "step_events": [{k: (list(v) if k == "win" else v) for k, v in m.items()} for m in steps],
        "plateau_rad_s": {"median": float(np.median(plate)), "mean": float(plate.mean()), "sd": float(plate.std()), "n_frames": int(len(plate))},
        "fit_all": {"stick_max_ang_speed": wm, "stick_max_ang_accel": ac, "onsets": ons, "rmse_deg": rm},
        "accel_profile": prof, "held_out": held, "stop_residuals_deg": stops, "overshoot_deg": over,
        "rust_check_max_diff_rad": diff,
        "stops": {"n": len(stop_ev), "accel_best": acc_stop, "rmse_deg": rm_stop, "profile_rmse_deg": stop_prof},
        "flick_starts": {"n": len(flick), "accel_best": acc_start, "rmse_deg": rm_start, "profile_rmse_deg": start_prof,
                         "ramp_frames_all": [e["ramp_frames"] for e in start_ev]}, "placeholder": {"stick_max_ang_speed": plh[0], "stick_max_ang_accel": plh[1]},
    }
    (results / "stick_chase.json").write_text(json.dumps(res, indent=1) + "\n")
    return res
