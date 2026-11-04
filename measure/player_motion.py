"""Player motion from the video alone: movement events, top speed, ramp from rest, decay, dash search,
params.player.json and a held-out check against the Rust sim.

Inputs are NOT logged, so "pushing" is inferred: a run is treated as pushing while the speed rises or stays
flat near its plateau (>= PLATEAU_FRAC of the run's peak); the decay starts at the last such frame. A release
cannot be told apart from active braking (opposite key), and a run that never reaches the game's top speed
cannot be told apart from a partial input. Every estimate below carries that limit.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

from .geometry import REPO, RinkMap

RESULTS = REPO / "measure" / "results"
CLIPS = ["justpuckphysicshittingitatfixedpoint", "justhittingwallphysicsfixedpoint", "justliftedshots", "staticrotatingstick"]
SHORT = {"justpuckphysicshittingitatfixedpoint": "puck", "justhittingwallphysicsfixedpoint": "wall", "justliftedshots": "lifted",
         "staticrotatingstick": "rotstick"}
TELEPORT_PX = 8.0  # px between consecutive frames: practice-tool reset (the fastest real run moves 3.2 px/frame)
HALF_WIN = 3  # local quadratic fit over +-3 samples for velocity (keeps a 3-frame ramp sharp)
V_MOVE = 8.0  # px/s: moving
V_REST = 3.0  # px/s: at rest (velocity noise ~2-3 px/s)
MIN_EVENT_FRAMES = 6
PLATEAU_FRAC = 0.9
DASH_FRAC = 1.2  # a burst: smoothed speed > 1.2x the run's later plateau
PARAMS_OUT = REPO / "params.player.json"
CAM_CONFIRM_PX = 3.0  # locomotion moves the (player-following) camera; disc wobble from the stick does not
BURST_ACCEL = 800.0  # px/s^2: a ramp this steep is a burst (dash candidate), ~6x the normal skating ramp


def load_track(clip: str, results: Path = RESULTS):
    meta = json.loads((results / f"{clip}.player.json").read_text())
    R = meta["rows"]
    f = np.array(R["frame"], int)
    keep = np.array(R["interpolated"]) == 0
    m = RinkMap.load(results / meta["map"])
    return f[keep], np.array(R["u_px"], float)[keep], np.array(R["v_px"], float)[keep], meta["fps"], m.m_per_px


def segments(f, u, v):
    """Split at teleports (practice resets) and at gaps > 6 frames."""
    cut = [0]
    for i in range(1, len(f)):
        if f[i] - f[i - 1] > 6 or math.hypot(u[i] - u[i - 1], v[i] - v[i - 1]) / (f[i] - f[i - 1]) > TELEPORT_PX:
            cut.append(i)
    cut.append(len(f))
    return [(a, b) for a, b in zip(cut, cut[1:]) if b - a > 2 * HALF_WIN + 1]


def velocity(t, u, v, half=HALF_WIN):
    """Local quadratic fit per sample (times may be uneven: duplicated frames are dropped)."""
    n = len(t)
    vu, vv = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        if b - a < 5:
            continue
        tt = t[a:b] - t[i]
        A = np.vstack([np.ones_like(tt), tt, tt * tt]).T
        vu[i] = np.linalg.lstsq(A, u[a:b], rcond=None)[0][1]
        vv[i] = np.linalg.lstsq(A, v[a:b], rcond=None)[0][1]
    return vu, vv


def find_events(t, sp):
    """Runs with speed > V_MOVE for >= MIN_EVENT_FRAMES, widened to where the speed is back at rest."""
    mov = sp > V_MOVE
    ev = []
    i = 0
    n = len(sp)
    while i < n:
        if mov[i]:
            j = i
            while j < n and mov[j]:
                j += 1
            if j - i >= MIN_EVENT_FRAMES:
                a, b = i, j - 1
                while a > 0 and sp[a - 1] > V_REST and sp[a - 1] < sp[a] + 5:
                    a -= 1
                while b < n - 1 and sp[b + 1] > V_REST and sp[b + 1] < sp[b] + 5:
                    b += 1
                if ev and a <= ev[-1][1]:
                    ev[-1] = (ev[-1][0], max(b, ev[-1][1]))
                else:
                    ev.append((a, b))
            i = j
        else:
            i += 1
    return ev


def analyse_event(t, sp, a, b, mpp, at_rest_before: bool, ends_at_rest: bool):
    s = sp[a : b + 1]
    tt = t[a : b + 1]
    k_peak = int(np.argmax(s))
    peak = float(s[k_peak])
    near = np.where(s >= PLATEAU_FRAC * peak)[0]
    k_last = int(near.max())  # last pushing frame (speed still near the plateau)
    k_first = int(near.min())
    plateau = s[k_first : k_last + 1]
    plat = float(np.median(plateau))
    out = {
        "t_start": round(float(tt[0]), 4), "t_end": round(float(tt[-1]), 4), "frames": int(b - a + 1),
        "starts_from_rest": bool(at_rest_before), "ends_at_rest": bool(ends_at_rest),
        "peak_px_s": round(peak, 2), "plateau_px_s": round(plat, 2),
        "plateau_sd_px_s": round(float(np.std(plateau)), 2), "plateau_s": round(float(tt[k_last] - tt[k_first]), 4),
        "push_s": round(float(tt[k_last] - tt[0]), 4),
    }
    # Ramp from rest: time to 90% of the plateau, and the mean acceleration between 10% and 90%.
    if at_rest_before:
        k90 = int(np.argmax(s >= 0.9 * plat))
        k10 = int(np.argmax(s >= 0.1 * plat))
        out["t10_to_t90_s"] = round(float(tt[k90] - tt[k10]), 4)
        out["t_rest_to_90_s"] = round(float(tt[k90] - tt[0]), 4)
        if k90 - k10 >= 2:
            out["accel_px_s2"] = round(float(np.polyfit(tt[k10 : k90 + 1], s[k10 : k90 + 1], 1)[0]), 1)
        else:
            out["accel_px_s2"] = round(float((s[k90] - s[k10]) / max(tt[k90] - tt[k10], 1e-9)), 1)
    # Decay after the last pushing frame, down to 10% of the plateau.
    dec = s[k_last:]
    td = tt[k_last:]
    k_stop = int(np.argmax(dec <= 0.1 * plat)) if (dec <= 0.1 * plat).any() else len(dec) - 1
    if k_stop >= 2:
        sd, tdd = dec[: k_stop + 1], td[: k_stop + 1]
        out["decay_s"] = round(float(tdd[-1] - tdd[0]), 4)
        lin = np.polyfit(tdd, sd, 1)
        out["decay_const_px_s2"] = round(float(-lin[0]), 1)
        out["decay_mean_px_s2"] = round(float((sd[0] - sd[-1]) / (tdd[-1] - tdd[0])), 1)
        pos = sd > 0.05 * plat
        if pos.sum() >= 3:
            ex = np.polyfit(tdd[pos], np.log(sd[pos]), 1)
            out["decay_exp_rate_1_s"] = round(float(-ex[0]), 3)
            r_lin = float(np.sqrt(np.mean((np.polyval(lin, tdd) - sd) ** 2)))
            r_exp = float(np.sqrt(np.mean((np.exp(np.polyval(ex, tdd[pos])) - sd[pos]) ** 2)))
            out["decay_fit_rmse_px_s"] = {"constant": round(r_lin, 2), "exponential": round(r_exp, 2)}
    if k_stop >= 1:
        # Smoothing-free decay: the distance covered from the last pushing frame to rest. Constant decel a gives
        # d = v0^2 / (2a); exponential decay at rate k gives d = v0 / k.
        out["_stop_k"] = (k_last, k_stop)
    for k in list(out):
        if k.endswith("_px_s") and isinstance(out[k], float):
            out[k.replace("_px_s", "_m_s")] = round(out[k] * mpp, 3)
        if k.endswith("_px_s2"):
            out[k.replace("_px_s2", "_m_s2")] = round(out[k] * mpp, 2)
    return out, k_first, k_last


def dash_search(t, sp, a, b, plat):
    """Bursts: smoothed speed above DASH_FRAC x the plateau the run settles to."""
    s = sp[a : b + 1]
    tt = t[a : b + 1]
    above = s > DASH_FRAC * plat
    out = []
    i = 0
    while i < len(s):
        if above[i]:
            j = i
            while j < len(s) and above[j]:
                j += 1
            k = i + int(np.argmax(s[i:j]))
            out.append({"t_peak": round(float(tt[k]), 4), "peak_px_s": round(float(s[k]), 2), "over_plateau_px_s": round(float(s[k] - plat), 2),
                        "duration_s": round(float(tt[j - 1] - tt[i]), 4)})
            i = j
        else:
            i += 1
    return out


def clip_events(clip: str, results: Path = RESULTS):
    f, u, v, fps, mpp = load_track(clip, results)
    R = json.loads((results / f"{clip}.player.json").read_text())["rows"]
    cf, cu_, cv_ = np.array(R["frame"]), np.array(R["shift_u"], float), np.array(R["shift_v"], float)
    t = f / fps
    out = []
    series = []
    for a, b in segments(f, u, v):
        ts, us, vs = t[a:b], u[a:b], v[a:b]
        vu, vv = velocity(ts, us, vs)
        sp = np.hypot(vu, vv)
        sp = np.nan_to_num(sp)
        series.append((ts, sp, vu, vv))
        for ea, eb in find_events(ts, sp):
            rest_before = ea >= 10 and float(np.max(sp[max(0, ea - 10) : ea])) < V_MOVE
            ends_rest = eb < len(sp) - 1 and sp[eb + 1] < V_MOVE
            ev, k1, k2 = analyse_event(ts, sp, ea, eb, mpp, rest_before, ends_rest)
            ang = math.degrees(math.atan2(-np.median(vu[ea + k1 : ea + k2 + 1]), -np.median(vv[ea + k1 : ea + k2 + 1])))  # sim frame: +x = screen up
            ev["direction_deg_sim_frame"] = round(ang, 1)
            ev["distance_px"] = round(float(np.sum(np.hypot(np.diff(us[ea : eb + 1]), np.diff(vs[ea : eb + 1])))), 2)
            ev["dash_bursts"] = dash_search(ts, sp, ea, eb, ev["plateau_px_s"]) if ev["plateau_s"] > 0.1 else []
            ia, ib = ea + k2, None
            if "_stop_k" in ev:
                ib = ea + k2 + ev.pop("_stop_k")[1]
                while ib < len(sp) - 1 and sp[ib] > V_REST:
                    ib += 1
                d_stop = math.hypot(us[ib] - us[ia], vs[ib] - vs[ia])
                v0 = float(sp[ia])
                if d_stop > 0.3:
                    ev["stop_distance_px"] = round(d_stop, 2)
                    ev["decay_from_distance_px_s2"] = round(v0 * v0 / (2 * d_stop), 1)
                    ev["decay_from_distance_m_s2"] = round(v0 * v0 / (2 * d_stop) * mpp, 2)
                    ev["decay_exp_rate_from_distance_1_s"] = round(v0 / d_stop, 2)
            fr = np.round(ts * fps).astype(int)
            ja, jb = np.searchsorted(cf, fr[ea]), min(np.searchsorted(cf, fr[eb]), len(cf) - 1)
            ev["camera_net_px"] = round(float(math.hypot(cu_[jb] - cu_[ja], cv_[jb] - cv_[ja])), 2)
            ev["locomotion"] = ev["camera_net_px"] >= CAM_CONFIRM_PX
            ev["burst"] = bool(ev.get("accel_px_s2", 0) > BURST_ACCEL)
            ev["clip"] = clip
            ev["id"] = f"{SHORT[clip]}_{len(out) + 1:02d}"
            ev["_curve"] = (ts[ea : eb + 1] - ts[ea], sp[ea : eb + 1], k2, k1)
            out.append(ev)
    return out, series, mpp


def stats(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "median": round(float(np.median(v)), 3), "mean": round(float(v.mean()), 3),
            "sd": round(float(v.std(ddof=1)), 3) if len(v) > 1 else None, "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}


# ---------------------------------------------------------------- sim check

def sim_curve(params: dict, push_s: float, total_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Speed of sim player 0 from rest with full +x input for push_s, then none (Rust sim, examples/player_start)."""
    base = json.loads((REPO / "params.json").read_text())
    for k, val in params.items():
        base["params"][k]["value"] = val
    tmp = REPO / "target" / "player_check_params.json"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(json.dumps(base))
    out = subprocess.run(["cargo", "run", "-q", "--release", "-p", "sim", "--example", "player_start", "--", str(tmp), f"{push_s}", f"{total_s}"],
                         cwd=REPO, check=True, capture_output=True, text=True).stdout
    a = np.array([[float(x) for x in ln.split(",")] for ln in out.strip().splitlines()[1:]])
    return a[:, 0], a[:, 1]


def curve_error(ev, params, mpp):
    """RMSE (px/s) between the measured speed curve and the sim from rest, full input until the measured
    last pushing frame, then released. The sim start is aligned to the measured start (first moving frame)."""
    tc, sc, k_last, k_first = ev["_curve"]
    push = float(tc[k_last])
    ts, ss = sim_curve(params, push, float(tc[-1]) + 0.05)
    sim = np.interp(tc, ts, ss) / mpp
    rmse = lambda a, b: float(np.sqrt(np.mean((sim[a:b] - sc[a:b]) ** 2)))  # noqa: E731
    return rmse(0, len(sc)), rmse(0, k_first + 1)


def loco(events):
    return [e for e in events if e["locomotion"]]


def fit_from(events, mpp):
    """Sim player params from camera-confirmed locomotion events (px -> m with the placeholder scale).
    max speed: plateaus held >= 0.25 s; accel: normal ramps from rest (bursts excluded); ice drag: the
    distance-based constant decel of runs that end at rest."""
    ev = loco(events)
    full = [e for e in ev if e["plateau_s"] >= 0.25]
    starts = [e for e in ev if e["starts_from_rest"] and not e["burst"] and "accel_px_s2" in e]
    dec = [e for e in ev if e["ends_at_rest"] and "decay_from_distance_px_s2" in e]
    p = {}
    if full:
        p["player_max_speed"] = float(np.median([e["plateau_px_s"] for e in full])) * mpp
    if starts:
        p["player_accel"] = float(np.median([e["accel_px_s2"] for e in starts])) * mpp
    if dec:
        p["player_ice_drag"] = float(np.median([e["decay_from_distance_px_s2"] for e in dec])) * mpp
    return p, {"player_max_speed": [e["id"] for e in full], "player_accel": [e["id"] for e in starts], "player_ice_drag": [e["id"] for e in dec]}


def print_table(allev):
    hdr = (f"{'id':12s} {'loco':>4s} {'cam px':>6s} {'t0 s':>6s} {'dur':>5s} {'rest':>4s} {'stop':>4s} {'dir':>5s} {'dist':>6s} {'peak':>6s} {'plat':>6s} "
           f"{'plat s':>6s} {'->90%':>5s} {'accel':>6s} {'stop d':>6s} {'decel':>6s} {'k':>5s}")
    print(hdr + "   (px, s, px/s, px/s^2; decel/k from the stop distance)")
    nan = float("nan")
    for e in allev:
        print(f"{e['id']:12s} {('yes' if e['locomotion'] else 'no'):>4s} {e['camera_net_px']:6.1f} {e['t_start']:6.2f} {e['t_end'] - e['t_start']:5.2f} "
              f"{str(e['starts_from_rest'])[0]:>4s} {str(e['ends_at_rest'])[0]:>4s} {e['direction_deg_sim_frame']:5.0f} {e['distance_px']:6.1f} "
              f"{e['peak_px_s']:6.1f} {e['plateau_px_s']:6.1f} {e['plateau_s']:6.2f} {e.get('t_rest_to_90_s', nan):5.2f} {e.get('accel_px_s2', nan):6.0f} "
              f"{e.get('stop_distance_px', nan):6.2f} {e.get('decay_from_distance_px_s2', nan):6.0f} {e.get('decay_exp_rate_from_distance_1_s', nan):5.1f}"
              + ("  BURST" if e["burst"] else ""))


def summarize(allev, mpp):
    ev = loco(allev)
    normal = [e for e in ev if e["starts_from_rest"] and not e["burst"]]
    stops = [e for e in ev if e["ends_at_rest"]]
    both = lambda vals, unit: {"px": stats(vals), "placeholder_m": stats([x * mpp for x in vals if x is not None]), "unit_px": unit}  # noqa: E731
    return {
        "events_total": len(allev),
        "locomotion_events": [e["id"] for e in ev],
        "wobble_events_excluded": [e["id"] for e in allev if not e["locomotion"]],
        "top_speed_plateau_px_s": both([e["plateau_px_s"] for e in ev if e["plateau_s"] >= 0.25], "px/s"),
        "peak_speed_px_s_all_locomotion": both([e["peak_px_s"] for e in ev], "px/s"),
        "accel_from_rest_px_s2_normal": both([e["accel_px_s2"] for e in normal], "px/s^2"),
        "accel_from_rest_px_s2_bursts": both([e["accel_px_s2"] for e in ev if e["burst"]], "px/s^2"),
        "time_rest_to_90pct_of_own_plateau_s": stats([e.get("t_rest_to_90_s") for e in normal]),
        "stop_decel_from_distance_px_s2": both([e.get("decay_from_distance_px_s2") for e in stops], "px/s^2"),
        "stop_exp_rate_from_distance_1_s": stats([e.get("decay_exp_rate_from_distance_1_s") for e in stops]),
        "stop_time_s_smoothed": stats([e.get("decay_s") for e in stops]),
        "m_per_px_placeholder": mpp,
    }


def run_estimate(results: Path = RESULTS, sim: bool = True) -> dict:
    allev, per_clip, mpps = [], {}, {}
    for c in CLIPS:
        if not (results / f"{c}.player.json").exists():
            continue
        ev, _, mpp = clip_events(c, results)
        mpps[c], per_clip[c] = mpp, ev
        allev += ev
    mpp = float(np.mean(list(mpps.values())))
    print(f"[player estimate] {len(allev)} candidate events (speed > {V_MOVE} px/s for >= {MIN_EVENT_FRAMES} frames); "
          f"locomotion = camera moved >= {CAM_CONFIRM_PX} px over the event")
    print_table(allev)
    summary = summarize(allev, mpp)
    lo = loco(allev)
    summary["dash"] = {
        "bursts_above_plateau": [dict(b, event=e["id"]) for e in lo for b in e["dash_bursts"]],
        "burst_starts": [{"event": e["id"], "accel_px_s2": e["accel_px_s2"], "peak_px_s": e["peak_px_s"], "t_rest_to_90_s": e.get("t_rest_to_90_s"),
                          "t_start": e["t_start"]} for e in lo if e["burst"]],
    }
    res = {"summary": summary, "events": [{k: v for k, v in e.items() if not k.startswith("_")} for e in allev]}
    for k, val in summary.items():
        if isinstance(val, dict) and "px" in val:
            print(f"  {k}: {val['px']}  | placeholder m: {val['placeholder_m']}")
        elif k not in ("dash",):
            print(f"  {k}: {val}")
    print(f"  dash: {json.dumps(summary['dash'])}")
    fit, src = fit_from(allev, mpp)
    res["fit_all"] = {"values": fit, "from": src}
    if sim:
        res["sim_check"] = sim_check(per_clip, mpps, fit)
    (results / "player_estimates.json").write_text(json.dumps(res, indent=1) + "\n")
    write_params(fit, src, summary, res.get("sim_check"))
    return res


PLACEHOLDER_KEYS = ("player_max_speed", "player_accel", "player_ice_drag")


def sim_check(per_clip, mpps, fit_all):
    """Held out by clip: fit on the other clips' locomotion events, test on this clip's starts from rest.
    Unfitted names keep the placeholder value. Also reported: all-clip fit (in-sample)."""
    base = {k: v["value"] for k, v in json.loads((REPO / "params.json").read_text())["params"].items()}
    placeholder = {k: base[k] for k in PLACEHOLDER_KEYS}
    out = {"placeholder": placeholder, "held_out": []}
    for test, evs in per_clip.items():
        tests = [e for e in loco(evs) if e["starts_from_rest"]]
        if not tests:
            continue
        train_ev = [e for c, ev in per_clip.items() if c != test for e in ev]
        fit, src = fit_from(train_ev, mpps[test])
        params = {**placeholder, **fit}
        for e in tests:
            err_f, err_fr = curve_error(e, params, mpps[test])
            err_a, err_ar = curve_error(e, {**placeholder, **fit_all}, mpps[test])
            err_p, err_pr = curve_error(e, placeholder, mpps[test])
            row = {"test_event": e["id"], "burst": e["burst"], "train_clips": sorted({SHORT[x["clip"]] for x in loco(train_ev)}),
                   "held_out_params": {k: round(v, 3) for k, v in params.items()}, "fitted_from": src,
                   "rmse_px_s": {"held_out_fit": round(err_f, 1), "all_clip_fit": round(err_a, 1), "placeholder": round(err_p, 1)},
                   "rmse_rising_phase_px_s": {"held_out_fit": round(err_fr, 1), "all_clip_fit": round(err_ar, 1), "placeholder": round(err_pr, 1)},
                   "measured_peak_px_s": e["peak_px_s"]}
            out["held_out"].append(row)
            print(f"[sim check] {e['id']:12s}{' BURST' if e['burst'] else '      '} train {'+'.join(row['train_clips']):14s} speed RMSE px/s: held-out fit "
                  f"{err_f:6.1f} | all-clip fit {err_a:6.1f} | placeholder {err_p:6.1f}   rising phase: {err_fr:6.1f} | {err_ar:6.1f} | {err_pr:6.1f}  "
                  f"(peak {e['peak_px_s']:.0f}) held-out params {row['held_out_params']}")
    return out


def write_params(fit, src, summary, check):
    mpp = summary["m_per_px_placeholder"]
    notes = {
        "player_max_speed": ("m/s", "top speed: the one flat plateau held >= 0.25 s (rotstick_04: 180 px/s for ~1 s, dead flat, so likely a hard cap). "
                             "n = 1 run; rotstick_01 peaked at 144 px/s without reaching a plateau."),
        "player_accel": ("m/s^2", "acceleration from rest: median linear ramp (10-90% of each run's own plateau) of the normal starts; the two "
                         "near-instant 'burst' starts (wall_08, rotstick_04, ~1400-1600 px/s^2) are excluded as dash candidates. The wall-clip "
                         "starts are short taps (plateau ~35 px/s), so this is the ramp rate, not a time-to-top-speed fit."),
        "player_ice_drag": ("m/s^2", "deceleration when the speed falls to rest, from the stop distance (v0^2 / 2d). Without an input log a "
                            "release cannot be told from active braking (opposite key): the stops take 0.07-0.25 s, and where the speed held flat "
                            "for a while before the stop no decay was visible, so the real coasting drag may be much lower and this a brake."),
    }
    out = {"_comment": ("Player params estimated from video alone (python -m measure.player estimate; see CLAUDE.md 'Player (session s6)'). "
                        "Inputs were not logged: 'pushing' = speed rising or flat near its plateau. Meters at the PLACEHOLDER scale "
                        f"(rink_length 60 m, {mpp:.5f} m/px); multiply by the real/placeholder scale ratio once it is measured. "
                        "Only names listed here change; everything else stays in params.json."),
           "params": {}}
    for k, val in fit.items():
        unit, note = notes[k]
        px = val / mpp
        out["params"][k] = {"value": round(val, 4), "provenance": "fitted-no-input-log", "unit": unit,
                            "note": f"{note} {px:.1f} px/{'s' if unit == 'm/s' else 's^2'} at {mpp:.5f} m/px (PLACEHOLDER scale). "
                                    f"From events {', '.join(src[k])}."}
    PARAMS_OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[player estimate] wrote {PARAMS_OUT.name}: {', '.join(f'{k} = {v:.3f}' for k, v in fit.items())}")
