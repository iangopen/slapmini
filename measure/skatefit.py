"""Session s7: player runs with known inputs -> calib scenarios, key->motion lag, player and dash fits.

    python -m measure.skatefit scenarios     # results/skate/*.frames.csv + keys -> calib/data/skate/ pairs + runs.json
    python -m measure.skatefit fit           # held-out-by-clip fits with calib -> params.player.json + skate/fits.json
    python -m measure.skatefit plot          # measured vs sim speed curves -> results/skate/qa/

Inputs from the key display: move_x = W - S, move_y = A - D (screen up = +x, screen left = +y, the game camera's
fixed orientation), dash = Shift, lift never pressed, stick_target_angle constant (the mouse was not moved). Key
events are shifted by the measured key->motion lag. The initial state is the player's measured position and velocity
(local quadratic over the first 5 samples); the puck is parked in a far corner.
Meters are at the PLACEHOLDER scale (camreg's px/m from the painted lines; params.json rink_length 60 m).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np

from .geometry import REPO, load_params
from .skate import OUT

SCN = REPO / "calib" / "data" / "skate"
TMP = Path(tempfile.gettempdir()) / "slapmini_skatefit"  # scratch params / replays (not committed)
CALIB = REPO / "target" / "release" / ("calib.exe" if sys.platform == "win32" else "calib")
SKATE_CLIPS = ["wkeymovement", "diagonalkeypresses", "whilemovingturning"]
DASH_CLIPS = ["dashatrest", "dashwhilerunning"]
MOVE = ("W", "A", "S", "D")
REST_SPEED = 0.3  # m/s
TELEPORT_M = 1.0  # a jump larger than this between consecutive kept frames is a reset/teleport
S6 = {"player_accel": 15.8489, "player_max_speed": 11.8397, "player_ice_drag": 26.0164}  # params.player.json (s6, video only)
PLACEHOLDER = {"player_accel": 12.0, "player_max_speed": 8.0, "player_ice_drag": 2.0, "dash_speed": 6.0,
               "dash_duration": 0.25, "dash_cooldown": 1.0}


def load_clip(name: str) -> dict:
    summ = json.loads((OUT / f"{name}.json").read_text())
    with open(OUT / f"{name}.frames.csv") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    fps = summ["fps"]
    x = np.array([float(r["player_x"]) if r["player_x"] else np.nan for r in rows])
    y = np.array([float(r["player_y"]) if r["player_y"] else np.nan for r in rows])
    u = np.array([float(r["player_u"]) if r["player_u"] else np.nan for r in rows])
    v = np.array([float(r["player_v"]) if r["player_v"] else np.nan for r in rows])
    vc = np.array([float(r["v_c"]) if r["v_c"] else np.nan for r in rows])
    # duplicated capture frames: everything identical to the previous frame (registration and player pixels)
    dup = np.zeros(n, bool)
    for i in range(1, n):
        if np.isfinite(x[i]) and x[i] == x[i - 1] and y[i] == y[i - 1] and vc[i] == vc[i - 1] and u[i] == u[i - 1] and v[i] == v[i - 1]:
            dup[i] = True
    keys = {k: np.zeros(n, bool) for k in MOVE + ("Shift", "Space", "R")}
    for e in summ.get("key_events", []):
        if e["key"] in keys:
            keys[e["key"]][e["frame"] :] = e["down"]
    return {"name": name, "fps": fps, "n": n, "x": x, "y": y, "dup": dup, "keys": keys, "summary": summ}


def velocity(t: np.ndarray, x: np.ndarray, y: np.ndarray, half: int = 3) -> tuple[np.ndarray, np.ndarray]:
    vx, vy = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for i in range(len(t)):
        a, b = max(0, i - half), min(len(t), i + half + 1)
        ok = np.isfinite(x[a:b])
        if ok.sum() < 4:
            continue
        tt = t[a:b][ok] - t[i]
        A = np.column_stack([np.ones_like(tt), tt, tt * tt])
        vx[i] = np.linalg.lstsq(A, x[a:b][ok], rcond=None)[0][1]
        vy[i] = np.linalg.lstsq(A, y[a:b][ok], rcond=None)[0][1]
    return vx, vy


def clean(c: dict) -> dict:
    """Kept frames (no duplicates, player found) with times, velocities and teleport breaks."""
    idx = np.nonzero(~c["dup"] & np.isfinite(c["x"]))[0]
    t = idx / c["fps"]
    x, y = c["x"][idx], c["y"][idx]
    jump = np.r_[False, np.hypot(np.diff(x), np.diff(y)) > TELEPORT_M]
    seg = np.cumsum(jump)
    vx, vy = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for g in np.unique(seg):  # velocities never straddle a teleport
        m = seg == g
        vx[m], vy[m] = velocity(t[m], x[m], y[m])
    return {"idx": idx, "t": t, "x": x, "y": y, "vx": vx, "vy": vy, "speed": np.hypot(vx, vy), "jump": jump}


def move_input(keys: dict, f: int) -> tuple[float, float, bool]:
    """Sim input from the keys. Shift with no move key is dropped: in the clips it does nothing at rest (dashatrest
    frames 80-105, 415-429, 454-507: no motion), while the sim would dash along `facing`."""
    mx = float(keys["W"][f]) - float(keys["S"][f])
    my = float(keys["A"][f]) - float(keys["D"][f])
    return mx, my, bool(keys["Shift"][f]) and (mx != 0 or my != 0)


def runs(c: dict, d: dict, max_s: float = 8.0) -> list[dict]:
    """Runs: from a key press while the player is at rest to the next rest with no keys held, a teleport, or max_s.
    A start is a key-down edge (full frame resolution) after >= 0.25 s without move/dash keys, with the player at
    rest: median speed over the previous 10 kept frames below REST_SPEED, or a teleport (practice reset) within 0.5 s."""
    keys, fps = c["keys"], c["fps"]
    active = np.zeros(c["n"], bool)
    for k in MOVE + ("Shift",):
        active |= keys[k]
    downs = np.nonzero(active[1:] & ~active[:-1])[0] + 1
    out = []
    next_free = 0
    for fd in downs:
        if fd < next_free or active[max(0, fd - int(0.25 * fps)) : fd].any():
            continue
        j = int(np.searchsorted(d["idx"], fd))  # first kept frame at/after the key-down
        if j >= len(d["idx"]) or d["idx"][j] - fd > 6 or j == 0:
            continue
        teleported = d["jump"][max(0, j - int(0.5 * fps)) : j + 1].any()
        if not (teleported or np.nanmedian(d["speed"][max(0, j - 10) : j]) < REST_SPEED):
            continue
        start = max(0, j - 6)
        while start < j and d["jump"][start + 1 : j + 1].any():
            start += 1
        k = j + 1
        rest_since = None
        while k < len(d["idx"]):
            fk = d["idx"][k]
            if d["jump"][k] or (keys["R"][fk] and not keys["R"][fd]) or (d["t"][k] - d["t"][start]) > max_s:
                break
            if not active[fk] and d["speed"][k] < REST_SPEED:
                rest_since = rest_since if rest_since is not None else k
                if d["t"][k] - d["t"][rest_since] > 0.3:
                    break
            else:
                rest_since = None
            k += 1
        end = k - 1
        if d["t"][end] - d["t"][start] >= 0.5:
            pressed = sorted({kk for kk in MOVE + ("Shift",) if keys[kk][d["idx"][start] : d["idx"][end] + 1].any()})
            out.append({"clip": c["name"], "start": int(start), "end": int(end), "keys": pressed, "from_rest": True,
                        "key_down_frame": int(fd)})
        next_free = int(d["idx"][end]) + 1
    return out


def lag_fit(c: dict, d: dict, lags=np.arange(-0.10, 0.30001, 1 / 240)) -> list[dict]:
    """Per key-down from rest (one move key): displacement along the key's direction over the first 0.35 s fitted
    with 0.5 a (t - L)^2 (zero before L); the L with the least squared error, a by least squares."""
    keys, fps = c["keys"], c["fps"]
    dirs = {"W": (1, 0), "S": (-1, 0), "A": (0, 1), "D": (0, -1)}
    out = []
    for kname, (ex, ey) in dirs.items():
        kk = keys[kname]
        downs = np.nonzero(kk[1:] & ~kk[:-1])[0] + 1
        for fd in downs:
            others = [o for o in MOVE + ("Shift",) if o != kname and keys[o][fd : fd + int(0.35 * fps)].any()]
            if others:
                continue
            sel = (d["idx"] >= fd - int(0.15 * fps)) & (d["idx"] <= fd + int(0.35 * fps))
            jj = np.nonzero(sel)[0]
            if len(jj) and d["jump"][jj].any():  # a reset just before the press: start after it
                last = jj[np.nonzero(d["jump"][jj])[0][-1]]
                if d["idx"][last] > fd:
                    continue
                sel[: last] = False
            if sel.sum() < 15:
                continue
            j0 = np.nonzero(sel)[0][0]
            before = d["speed"][max(0, j0 - 10) : j0 + 1]
            teleported = d["jump"][max(0, j0 - int(0.5 * fps)) : j0 + 1].any()
            if not teleported and np.nanmedian(before) > REST_SPEED:
                continue
            t = d["t"][sel] - fd / fps
            s = (d["x"][sel] - d["x"][j0]) * ex + (d["y"][sel] - d["y"][j0]) * ey
            s = s - np.median(s[t < -0.03]) if (t < -0.03).sum() >= 3 else s - s[0]
            best = None
            for L in lags:
                q = 0.5 * np.clip(t - L, 0, None) ** 2
                if (q > 0).sum() < 5:
                    continue
                a = float((q * s).sum() / (q * q).sum())
                err = float(((a * q - s) ** 2).sum())
                if best is None or err < best[0]:
                    best = (err, float(L), a)
            if best:
                out.append({"clip": c["name"], "key": kname, "frame": int(fd), "lag_s": round(best[1], 4),
                            "accel_m_s2": round(best[2], 3), "rms_m": round(math.sqrt(best[0] / sel.sum()), 4)})
    return out


def scenario(c: dict, d: dict, run: dict, lag_s: float, p: dict, run_id: str) -> tuple[dict, str]:
    fps = c["fps"]
    a, b = run["start"], run["end"]
    f0 = int(d["idx"][a])
    t0 = d["t"][a]
    # initial state from a local quadratic over the first 5 samples
    tt = d["t"][a : a + 5] - t0
    A = np.column_stack([np.ones_like(tt), tt, tt * tt])
    cx = np.linalg.lstsq(A, d["x"][a : a + 5], rcond=None)[0]
    cy = np.linalg.lstsq(A, d["y"][a : a + 5], rcond=None)[0]
    vx, vy = float(cx[1]), float(cy[1])
    if run.get("from_rest", True):
        vx = vy = 0.0  # runs start from rest (the 5-sample velocity there is only detection noise)
    # facing: direction of travel; at rest, the direction of the first motion in the run (unobservable otherwise)
    if math.hypot(vx, vy) > 0.5:
        facing = math.atan2(vy, vx)
    else:
        k = min(b, a + int(0.5 * fps))
        dx, dy = d["x"][k] - d["x"][a], d["y"][k] - d["y"][a]
        facing = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.05 else 0.0
    px, py = float(cx[0]), float(cy[0])
    puck = {"x": 25.0 if px < 0 else -25.0, "y": 15.0 if py < 0 else -15.0,  # parked in the far corner
            "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "in_net": None}
    player = {"team": 0, "x": px, "y": py, "vx": vx, "vy": vy, "facing": facing, "stick_angle": 0.0, "stick_omega": 0.0,
              "dash_timer": 0.0, "dash_cooldown": 0.0, "dash_prev": False}
    state = {"tick": 0, "puck": puck, "players": [player], "score": [0, 0]}
    dt = p["timestep"]
    inputs = []
    last = None
    for f in range(f0, int(d["idx"][b]) + 1):
        mx, my, dash = move_input(c["keys"], f)
        cur = (mx, my, dash)
        if cur != last:
            step = max(0, int(round(((f - f0) / fps + lag_s) / dt)))
            inputs.append({"step": step, "player": 0, "input": {"move_x": mx, "move_y": my, "stick_target_angle": 0.0, "dash": dash, "lift": False}})
            last = cur
    steps = int(math.ceil((d["t"][b] - t0) / dt)) + 1
    sc = {"name": run_id, "description": f"s7 {c['name']} frames {f0}-{int(d['idx'][b])}, keys {'+'.join(run['keys'])}; "
          f"inputs from the key display shifted by the measured lag {lag_s:.3f} s; player 0 tracked from video "
          "(placeholder-scale meters)", "steps": steps, "initial_state": state, "inputs": inputs, "track": "player0"}
    lines = ["t,x,y,z"] + [f"{d['t'][j] - t0:.6f},{d['x'][j]:.5f},{d['y'][j]:.5f},0" for j in range(a, b + 1)]
    return sc, "\n".join(lines) + "\n"


def run_scenarios() -> dict:
    p = load_params()
    SCN.mkdir(parents=True, exist_ok=True)
    for f in SCN.glob("*"):
        f.unlink()
    all_lags, manifest = [], []
    clips = {}
    for name in SKATE_CLIPS + DASH_CLIPS:
        c = load_clip(name)
        d = clean(c)
        clips[name] = (c, d)
        all_lags += lag_fit(c, d)
    lag = float(np.median([L["lag_s"] for L in all_lags])) if all_lags else 0.0
    for name, (c, d) in clips.items():
        for k, r in enumerate(runs(c, d)):
            rid = f"{name}_{k:02d}"
            sc, csv_text = scenario(c, d, r, lag, p, rid)
            (SCN / f"{rid}.scenario.json").write_text(json.dumps(sc, indent=1))
            (SCN / f"{rid}.csv").write_text(csv_text)
            manifest.append({**r, "id": rid, "frames": [int(d["idx"][r["start"]]), int(d["idx"][r["end"]])],
                             "duration_s": round(float(d["t"][r["end"]] - d["t"][r["start"]]), 3),
                             "peak_speed_m_s": round(float(np.nanmax(d["speed"][r["start"] : r["end"] + 1])), 3),
                             "kind": "dash" if "Shift" in r["keys"] else "skate"})
    out = {"lag_s_median": lag, "lag_samples": all_lags, "runs": manifest,
           "note": "key->motion lag fitted per standing start; scenarios use the median"}
    (OUT / "runs.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------------- fits

def calib(args: list[str]) -> str:
    r = subprocess.run([str(CALIB)] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout


def pairs(ids: list[str]) -> list[str]:
    out = []
    for i in ids:
        out += [str(SCN / f"{i}.scenario.json"), str(SCN / f"{i}.csv")]
    return out


def params_with(values: dict, base: Path, out: Path) -> Path:
    raw = json.loads(base.read_text())
    for k, v in values.items():
        raw["params"][k]["value"] = float(v)
    out.write_text(json.dumps(raw, indent=1))
    return out


def evaluate(ids: list[str], params_file: Path) -> list[dict]:
    rows = json.loads(calib(["eval", "--params", str(params_file), "--json", "1"] + pairs(ids)))
    return [{"id": r["name"], "pos_rmse": r["errors"]["pos_rmse"], "speed_rmse": r["errors"]["speed_rmse"]} for r in rows]


def fit(ids: list[str], names: list[str], start: Path, tag: str) -> dict:
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / f"fit_{tag}.params.json"
    calib(["fit", "--params", str(start), "--fit", ",".join(names), "--out", str(out), "--iters", "400", "--objective", "time"] + pairs(ids))
    vals = json.loads(out.read_text())["params"]
    return {n: vals[n]["value"] for n in names}


def summarize(ev: list[dict]) -> dict:
    return {"pos_rmse_median_m": round(float(np.median([e["pos_rmse"] for e in ev])), 4),
            "speed_rmse_median_m_s": round(float(np.median([e["speed_rmse"] for e in ev])), 4), "n": len(ev)}


def dash_bursts() -> list[dict]:
    """Every dash (Shift going down together with a move key): speed just before, peak within 0.5 s, rise time."""
    out = []
    for name in SKATE_CLIPS + DASH_CLIPS:
        c = load_clip(name)
        d = clean(c)
        sh = c["keys"]["Shift"]
        anymove = c["keys"]["W"] | c["keys"]["A"] | c["keys"]["S"] | c["keys"]["D"]
        dash = sh & anymove
        for fd in np.nonzero(dash[1:] & ~dash[:-1])[0] + 1:
            j = int(np.searchsorted(d["idx"], fd))
            if j < 4 or j + 30 >= len(d["idx"]) or d["jump"][j - 4 : j + 31].any():
                continue
            v0 = float(np.nanmedian(d["speed"][j - 4 : j]))
            win = d["speed"][j : j + 31]
            k = int(np.nanargmax(win))
            vp = float(win[k])
            lo, hi = v0 + 0.1 * (vp - v0), v0 + 0.9 * (vp - v0)
            t10 = next((d["t"][j + i] for i in range(k + 1) if win[i] >= lo), np.nan)
            t90 = next((d["t"][j + i] for i in range(k + 1) if win[i] >= hi), np.nan)
            held = int((c["keys"]["W"] | c["keys"]["A"] | c["keys"]["S"] | c["keys"]["D"])[fd : fd + 120].sum())
            out.append({"clip": name, "frame": int(fd), "speed_before": round(v0, 2), "peak": round(vp, 2),
                        "gain": round(vp - v0, 2), "t_peak_s": round(float(d["t"][j + k] - fd / c["fps"]), 3),
                        "rise_10_90_s": round(float(t90 - t10), 3), "move_key_frames_after": held})
    return out


def direct_measures() -> dict:
    """Model-free numbers from the runs: plateau speed with a move key held, and the coast after the last key-up
    (fit with constant deceleration, as in the sim, and with speed-proportional decay)."""
    man = json.loads((OUT / "runs.json").read_text())
    plate, coasts = [], []
    cache = {}
    for r in man["runs"]:
        if r["kind"] != "skate":
            continue
        c, d = cache.get(r["clip"]) or cache.setdefault(r["clip"], (lambda c: (c, clean(c)))(load_clip(r["clip"])))
        a, b = r["start"], r["end"]
        idx = d["idx"][a : b + 1]
        held = np.zeros(len(idx), bool)
        for k in MOVE:
            held |= c["keys"][k][idx]
        sp = d["speed"][a : b + 1]
        t = d["t"][a : b + 1]
        if held.any():
            first = np.nonzero(held)[0][0]
            on = held & (t - t[first] > 1.2)
            if on.sum() > 20:
                plate.append(float(np.nanmedian(sp[on])))
            last = np.nonzero(held)[0][-1]
            m = (np.arange(len(t)) > last + 2) & (sp > 1.0)
            if m.sum() > 30:
                tt, vv = t[m] - t[m][0], sp[m]
                A = np.column_stack([np.ones_like(tt), tt])
                lin = np.linalg.lstsq(A, vv, rcond=None)[0]
                ex = np.linalg.lstsq(A, np.log(vv), rcond=None)[0]
                coasts.append({"id": r["id"], "v_start": round(float(vv[:5].mean()), 2), "duration_s": round(float(tt[-1]), 2),
                               "const_decel_m_s2": round(float(-lin[1]), 3), "rms_const": round(float(np.sqrt(np.mean((A @ lin - vv) ** 2))), 3),
                               "prop_rate_1_s": round(float(-ex[1]), 3), "rms_prop": round(float(np.sqrt(np.mean((np.exp(A @ ex) - vv) ** 2))), 3)})
    return {"plateau_speed_m_s": {"median": round(float(np.median(plate)), 3), "sd": round(float(np.std(plate, ddof=1)), 3), "n": len(plate)},
            "coasts": coasts}


def run_fits() -> dict:
    man = json.loads((OUT / "runs.json").read_text())
    runs_ = man["runs"]
    base = REPO / "params.json"
    tmp = TMP
    tmp.mkdir(parents=True, exist_ok=True)
    A = ["player_accel", "player_max_speed", "player_ice_drag"]
    B = ["dash_speed", "dash_duration"]  # dash_cooldown: no repeated dash inside a second in any clip -> not identifiable
    skate = {c: [r["id"] for r in runs_ if r["clip"] == c and r["kind"] == "skate"] for c in SKATE_CLIPS}
    dash = {c: [r["id"] for r in runs_ if r["clip"] == c] for c in DASH_CLIPS}
    res = {"stage_A": {}, "stage_B": {}}
    variants = {"placeholder": {}, "s6_video_only": S6}
    # Stage A: hold out by clip (fit on the other two skating clips).
    for held in SKATE_CLIPS:
        train = [i for c in SKATE_CLIPS if c != held for i in skate[c]]
        vals = fit(train, A, base, f"A_minus_{held}")
        variants[f"heldout_{held}"] = vals
        res["stage_A"][held] = {"fitted_on_others": vals}
    all_skate = [i for c in SKATE_CLIPS for i in skate[c]]
    final_A = fit(all_skate, A, base, "A_all")
    variants["fitted_all"] = final_A
    for held in SKATE_CLIPS:
        row = {}
        for vname in ("placeholder", "s6_video_only", f"heldout_{held}", "fitted_all"):
            pf = params_with(variants[vname], base, tmp / f"{vname}.json")
            row["held_out_fit" if vname.startswith("heldout") else vname] = summarize(evaluate(skate[held], pf))
        res["stage_A"][held]["test"] = row
    res["stage_A"]["final"] = final_A
    # Stage B: dash with stage A fixed, hold out by dash clip.
    baseA = params_with(final_A, base, tmp / "A_final.json")
    for held in DASH_CLIPS:
        other = [c for c in DASH_CLIPS if c != held][0]
        vals = fit(dash[other], B, baseA, f"B_minus_{held}")
        res["stage_B"][held] = {"fitted_on_other": vals}
    final_B = fit([i for c in DASH_CLIPS for i in dash[c]], B, baseA, "B_all")
    for held in DASH_CLIPS:
        row = {}
        for vname, vals in (("placeholder_dash", {}), ("held_out_fit", res["stage_B"][held]["fitted_on_other"]), ("fitted_all", final_B)):
            pf = params_with(vals, baseA, tmp / f"B_{vname}.json")
            row[vname] = summarize(evaluate(dash[held], pf))
        res["stage_B"][held]["test"] = row
    res["stage_B"]["final"] = final_B
    res["dash_bursts"] = dash_bursts()
    res["direct"] = direct_measures()
    (OUT / "fits.json").write_text(json.dumps(res, indent=1))
    return res


def replay_speed(run_id: str, params_file: Path) -> tuple[np.ndarray, np.ndarray]:
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / f"replay_{run_id}.csv"
    calib(["replay", "--params", str(params_file), str(SCN / f"{run_id}.scenario.json"), str(out)])
    a = np.loadtxt(out, delimiter=",", skiprows=1)
    t, x, y = a[:, 0], a[:, 1], a[:, 2]
    return t[1:], np.hypot(np.diff(x), np.diff(y)) / np.diff(t)


def measured_speed(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(SCN / f"{run_id}.csv", delimiter=",", skiprows=1)
    t, x, y = a[:, 0], a[:, 1], a[:, 2]
    vx, vy = velocity(t, x, y)
    return t, np.hypot(vx, vy)


def run_plot(ids: list[str] | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fits = json.loads((OUT / "fits.json").read_text())
    man = json.loads((OUT / "runs.json").read_text())
    base = REPO / "params.json"
    tmp = TMP
    tmp.mkdir(parents=True, exist_ok=True)
    final = {**fits["stage_A"]["final"], **fits["stage_B"]["final"]}
    variants = [("placeholder", params_with({}, base, tmp / "p_placeholder.json"), "#999"),
                ("s6 video-only", params_with(S6, base, tmp / "p_s6.json"), "#c80"),
                ("fitted (s7)", params_with(final, base, tmp / "p_final.json"), "#06c")]
    if ids is None:
        by = {}
        for r in man["runs"]:
            by.setdefault(r["clip"], r["id"])
        ids = list(by.values())
    fig, axes = plt.subplots(len(ids), 1, figsize=(9, 2.4 * len(ids)), dpi=110, sharex=False)
    for ax, rid in zip(np.atleast_1d(axes), ids):
        t, v = measured_speed(rid)
        ax.plot(t, v, color="k", lw=1.6, label="measured (video)")
        for name, pf, col in variants:
            ts, vs = replay_speed(rid, pf)
            ax.plot(ts, vs, color=col, lw=1, label=name)
        sc = json.loads((SCN / f"{rid}.scenario.json").read_text())
        for e in sc["inputs"]:
            ax.axvline(e["step"] * load_params()["timestep"], color="#ddd", lw=0.8, zorder=0)
        ax.set_title(f"{rid}: keys {sc['description'].split('keys ')[1].split(';')[0]}", fontsize=9)
        ax.set_ylabel("speed, m/s (placeholder scale)", fontsize=8)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=4)
    np.atleast_1d(axes)[-1].set_xlabel("time in run, s (grey: input changes)")
    fig.tight_layout()
    qa = OUT / "qa"
    qa.mkdir(exist_ok=True)
    path = qa / "player_speed_fits.png"
    fig.savefig(path)
    return path


def write_params() -> Path:
    """params.player.json: the s7 fits (provenance "fitted"), values in placeholder-scale meters, px in the notes."""
    r = json.loads((OUT / "fits.json").read_text())
    px = float(np.median([json.loads((OUT / f"{c}.json").read_text())["calibration"]["s_px_per_m"] for c in SKATE_CLIPS + DASH_CLIPS]))
    A, B, dr = r["stage_A"], r["stage_B"], r["direct"]
    held = {n: [A[c]["fitted_on_others"][n] for c in SKATE_CLIPS] for n in A["final"]}
    rng = lambda n: f"{min(held[n]):.3f}-{max(held[n]):.3f}"
    clips_a = ", ".join(SKATE_CLIPS)
    common = (f"Held-out by clip (fit on the other two of {clips_a}): {{}}; calib fitter (Nelder-Mead, time objective) on the "
              "runs of all three skating clips, inputs read from the key display (CLAUDE.md 'Session s7'). PLACEHOLDER scale "
              f"(meters from the painted line spacing, {px:.3f} px per placeholder m); multiply lengths/speeds/accelerations by "
              "the real/placeholder factor (measure/results/scale.json) when the scale is applied.")
    bursts = r["dash_bursts"]
    rest = [b["peak"] for b in bursts if b["speed_before"] < 1.0]
    run = [b for b in bursts if b["speed_before"] > 9.0]
    entries = {
        "player_accel": {"value": A["final"]["player_accel"], "unit": "m/s^2",
                         "note": f"skating acceleration with a move key held (constant ramp to the cap) = {A['final']['player_accel'] * px:.1f} px/s^2. "
                                 + common.format(rng("player_accel")) + " Standing-start ramp fits (key->motion lag model) give 14.4-22.1 m/s^2."},
        "player_max_speed": {"value": A["final"]["player_max_speed"], "unit": "m/s",
                             "note": f"speed cap = {A['final']['player_max_speed'] * px:.1f} px/s. Model-free plateau with a key held: "
                                     f"{dr['plateau_speed_m_s']['median']} m/s (sd {dr['plateau_speed_m_s']['sd']}, n = {dr['plateau_speed_m_s']['n']} runs), "
                                     "dead flat (a hard cap); diagonal input (W+D, W+A) has the same top speed (normalized). " + common.format(rng("player_max_speed"))},
        "player_ice_drag": {"value": A["final"]["player_ice_drag"], "unit": "m/s^2",
                            "note": f"coasting deceleration with no key held = {A['final']['player_ice_drag'] * px:.1f} px/s^2 (the sim's model is a constant "
                                    "deceleration). The measured coast is closer to speed-proportional decay (rate 0.52-1.34 1/s, median ~0.85; "
                                    "fits better than a constant in all 11 coasts), so this constant is a compromise; the s6 value (26 m/s^2, stops of 0.07-0.25 s) cannot "
                                    "be coasting, which takes ~3 s here: probably active braking. " + common.format(rng("player_ice_drag"))},
        "dash_speed": {"value": B["final"]["dash_speed"], "unit": "m/s",
                       "note": f"COMPROMISE for the sim's additive dash impulse = {B['final']['dash_speed'] * px:.1f} px/s, fitted on dashatrest + "
                               "dashwhilerunning with stage A fixed. The data is a SPEED BOOST to a fixed speed, not an impulse: a dash "
                               f"(Shift with a move key) from rest peaks at {min(rest):.1f}-{max(rest):.1f} m/s within 0.12-0.22 s (10-90% rise "
                               f"0.07-0.15 s), at top speed it only adds {min(b['gain'] for b in run):.2f}-{max(b['gain'] for b in run):.2f} m/s "
                               f"(peak {min(b['peak'] for b in run):.1f}-{max(b['peak'] for b in run):.1f}). Held-out fails both ways "
                               f"(fit on the other clip: {B['dashatrest']['fitted_on_other']['dash_speed']:.2f} / {B['dashwhilerunning']['fitted_on_other']['dash_speed']:.2f}). "
                               "Shift with no move key does nothing in the clips (the sim dashes along `facing`)."},
    }
    raw = {"_comment": ("Player params fitted from s7 clips with a key display (python -m measure.skatefit fit; CLAUDE.md 'Session s7'): "
                        "wkeymovement, diagonalkeypresses, whilemovingturning (skating), dashatrest, dashwhilerunning (dash). Replaces the "
                        "s6 video-only estimates (no input log): player_max_speed 11.84, player_accel 15.85, player_ice_drag 26.02. Meters "
                        "at the PLACEHOLDER scale. dash_duration and dash_cooldown are not listed: the fit drives dash_duration to 0 (the "
                        "sim's raised-cap window does not match a speed boost, and it also defines dash shots), and no clip has two dashes "
                        "within a second, so the cooldown is not identifiable. Only names listed here change; params.json is untouched."),
           "params": {k: {"value": round(float(v["value"]), 4), "provenance": "fitted", "unit": v["unit"], "note": v["note"]} for k, v in entries.items()}}
    path = REPO / "params.player.json"
    path.write_text(json.dumps(raw, indent=2) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m measure.skatefit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scenarios")
    sub.add_parser("fit")
    sub.add_parser("params")
    pl = sub.add_parser("plot")
    pl.add_argument("ids", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "scenarios":
        out = run_scenarios()
        print(json.dumps({"lag_s_median": out["lag_s_median"], "n_lag": len(out["lag_samples"]),
                          "runs": [(r["id"], r["keys"], r["duration_s"], r["peak_speed_m_s"]) for r in out["runs"]]}, indent=0))
    elif a.cmd == "fit":
        print(json.dumps(run_fits(), indent=1))
    elif a.cmd == "params":
        print("wrote", write_params())
    elif a.cmd == "plot":
        print("wrote", run_plot(a.ids or None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
