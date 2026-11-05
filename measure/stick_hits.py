"""Hit physics from hittingpuck (`python -m measure.stick hits`).

Hits: the puck rests next to the player (within 40 px of the pivot) for >= 8 frames, then leaves (>= 4 moving frames).
Practice resets (the puck jumps and is at rest again; the stick snaps with it) are not hits.

Per hit, at the placeholder ice-plane scale of the clip's map:
- contact radius r_c = distance pivot -> resting puck centre (ground point), on the ice plane;
- swing speed w = stick angle change over frames h-2 .. h+1 / 3 frames (h = last resting frame; contact is between h
  and h+1), head speed v_head = |w| r_c;
- exit speed: straight-line fit of the puck path from h+2 to the end of the free glide (<= 30 frames, before any
  board contact), distance s(t) = v0 (1 - exp(-k t)) / k with the fitted ice drag k (params.json ice_linear_drag),
  v0 extrapolated to the contact time.
calib fit of stick_mass + stick_restitution (scenarios in calib/data/stick/), held out by swing-speed group
(slow / medium / fast): the sim stick starts just short of the puck, turning at the measured w; its target is set
just far enough ahead that the chase law keeps w (err = max(w^2 / 2 acc, w dt)).
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from .geometry import PARAMS_PATH, REPO, RinkMap, load_params

RESULTS = REPO / "measure" / "results"
CLIP = "hittingpuck"
DEST = REPO / "calib" / "data" / "stick"
CAP_OFF = 200.0  # m/s: puck speed caps lifted for the fit (saturation is checked in the data instead)


def load_puck():
    T = json.loads((RESULTS / f"{CLIP}.track.json").read_text())
    R = T["rows"]
    return (np.array(R["frame"]), np.array(R["ug_px"], float), np.array(R["vg_px"], float), np.array(R["shift_u"], float),
            np.array(R["shift_v"], float))


def load_stick():
    R = json.loads((RESULTS / f"{CLIP}.stick.json").read_text())["rows"]
    return {f: (a, u, v) for f, a, u, v in zip(R["frame"], R["angle_unwrapped"], R["pivot_u"], R["pivot_v"])}


def find_hits(pf, pu, pv, su, sv, st):
    step = np.r_[0, np.hypot(np.diff(pu), np.diff(pv))] / np.r_[1, np.maximum(np.diff(pf), 1)]
    hits = []
    # Rest = frames k-12 .. k-4 still (< 0.5 px/frame); the last 3 frames before contact may creep (the blade covers
    # part of the puck blob and biases its centroid). Contact frame h = the frame before the first step > 1 px.
    for k in range(13, len(pf) - 6):
        if not (step[k] > 1.0 and step[k - 1] <= 1.0 and np.all(step[k + 1 : k + 4] > 0.8) and np.all(step[k - 12 : k - 3] < 0.5)
                and pf[k] - pf[k - 12] <= 13):
            continue
        h = int(pf[k - 1])
        if h not in st:
            continue
        rest = (float(np.median(pu[k - 12 : k - 3])), float(np.median(pv[k - 12 : k - 3])))
        piv = (st[h][1] - su[k - 1], st[h][2] - sv[k - 1])  # raw pivot -> reference pixels
        r = math.hypot(rest[0] - piv[0], rest[1] - piv[1])
        if r > 40:
            continue
        hits.append({"frame": h, "k": k - 1, "pivot_ref": piv, "r_c_px": r, "rest_ref": rest})
    return hits, step


def exit_speed(pf, pu, pv, step, k, k_drag, fps, mpp, m=None):
    """v0 (m/s) at the contact time from h+2 .. the end of the free glide: <= 30 frames, contiguous rows (gaps <= 2),
    still moving, and the puck farther than 1.5 m from the boards (straight-side approximation of the rink)."""
    i0 = k + 2
    i1 = i0
    def free(i):
        if m is None:
            return True
        x, y = m.px_to_m(pu[i], pv[i])
        return abs(float(x)) < m.board_length_px * m.m_per_px / 2 - 1.5 and abs(float(y)) < m.board_width_px * m.m_per_px / 2 - 1.5
    while i1 + 1 < len(pf) and i1 - i0 < 30 and pf[i1 + 1] - pf[i1] <= 2 and step[i1 + 1] > 0.3 and free(i1 + 1):
        i1 += 1
    P = np.c_[pu[i0 : i1 + 1], pv[i0 : i1 + 1]]
    d = P[-1] - P[0]
    d /= np.linalg.norm(d)
    s = (P - P[0]) @ d
    tt = (pf[i0 : i1 + 1] - (pf[k] + 0.5)) / fps  # time since contact (mid-frame)
    # s = s_off + v0 (1 - e^{-k t}) / k: linear least squares in (s_off, v0)
    A = np.c_[np.ones_like(tt), (1 - np.exp(-k_drag * tt)) / k_drag]
    (s_off, v0), *_ = np.linalg.lstsq(A, s, rcond=None)
    res = s - A @ [s_off, v0]
    return float(v0 * mpp), int(i1 - i0 + 1), d, float(np.sqrt(np.mean(res ** 2))), (i0, i1)


def run_hits(results: Path = RESULTS) -> dict:
    p = load_params(PARAMS_PATH)
    m = RinkMap.load(results / f"{CLIP}.map.json")
    mpp, fps = m.m_per_px, m.fps
    pf, pu, pv, su, sv = load_puck()
    st = load_stick()
    hits, step = find_hits(pf, pu, pv, su, sv, st)
    rows = []
    for hdat in hits:
        h, k = hdat["frame"], hdat["k"]
        if not all(f in st for f in (h - 3, h, h + 1)):
            continue
        # The interval [h, h+1] contains the contact; flicks are still accelerating before it, so earlier frames under-read.
        w = (st[h + 1][0] - st[h][0]) * fps
        w_before = (st[h][0] - st[h - 3][0]) / (3 / fps)
        r_c = hdat["r_c_px"] * mpp
        v0, n, d, rms, (i0, i1) = exit_speed(pf, pu, pv, step, k, p["ice_linear_drag"], fps, mpp, m)
        rows.append({"frame": h, "t": h / fps, "omega_rad_s": w, "omega_3_frames_before_rad_s": w_before, "r_c_m": r_c, "v_head_m_s": abs(w) * r_c, "v_exit_m_s": v0,
                     "ratio": v0 / max(abs(w) * r_c, 1e-9), "exit_fit_frames": n, "exit_fit_rms_px": rms, "dir_ref_px": d.tolist(),
                     "puck_ref_px": list(hdat["rest_ref"]), "pivot_ref_px": list(hdat["pivot_ref"]), "glide_rows": [int(i0), int(i1)]})
    fr_all = np.array(sorted(st))
    a_all = np.array([st[f][0] for f in fr_all])
    d1 = np.diff(a_all) * fps
    still = np.abs(d1) < 3.0
    w_noise = float(np.std(d1[still & (np.diff(fr_all) == 1)]))
    rows.sort(key=lambda r: r["v_head_m_s"])
    hs = np.array([r["v_head_m_s"] for r in rows])
    # Groups by swing speed: gaps in head speed
    # Groups by swing speed (the protocol: slow / medium / fast), cut at the two largest gaps in |w|.
    ws = np.array([abs(r["omega_rad_s"]) for r in rows])
    order = np.argsort(ws)
    gaps = np.diff(ws[order])
    cuts = sorted(np.argsort(gaps)[-2:] + 1) if len(rows) >= 3 else []
    grp = np.zeros(len(rows), int)
    for c in cuts:
        grp[order[c:]] += 1
    names = ["slow", "medium", "fast"]
    for r, g in zip(rows, grp):
        r["group"] = names[g]
    print(f"[hits] {len(rows)} hits in {CLIP} (placeholder ice-plane scale {mpp:.5f} m/px; px/s = m/s / {mpp:.5f}); single-interval w noise "
          f"at rest (sd) {w_noise:.2f} rad/s")
    print(f"{'frame':>6s} {'group':>6s} {'w rad/s':>8s} {'r_c m':>6s} {'v_head m/s':>10s} {'v_exit m/s':>10s} {'exit/head':>9s} {'v_exit px/s':>11s} {'fit n':>5s} {'rms px':>6s}")
    for r in rows:
        print(f"{r['frame']:6d} {r['group']:>6s} {r['omega_rad_s']:8.2f} {r['r_c_m']:6.3f} {r['v_head_m_s']:10.2f} {r['v_exit_m_s']:10.2f} {r['ratio']:9.3f} "
              f"{r['v_exit_m_s'] / mpp:11.1f} {r['exit_fit_frames']:5d} {r['exit_fit_rms_px']:6.2f}")
    v_e = np.array([r["v_exit_m_s"] for r in rows])
    k0 = float((hs @ v_e) / (hs @ hs))
    lin = np.polyfit(hs, v_e, 1)
    res0 = v_e - k0 * hs
    print(f"[hits] through the origin: v_exit = {k0:.3f} v_head (rms {np.sqrt(np.mean(res0 ** 2)):.2f} m/s); with intercept: "
          f"v_exit = {lin[0]:.3f} v_head {lin[1]:+.2f} m/s; ratio by group: "
          + ", ".join(f"{g} {np.mean([r['ratio'] for r in rows if r['group'] == g]):.3f}" for g in names if any(r['group'] == g for r in rows)))
    for r in rows:  # exit direction vs the perpendicular to the pivot->puck line (the stick's push direction if it met the puck centre-on)
        cx, cy = r["puck_ref_px"][0] - r["pivot_ref_px"][0], r["puck_ref_px"][1] - r["pivot_ref_px"][1]
        cdir = math.atan2(-cx, -cy)  # sim frame
        ex, ey = r["dir_ref_px"]
        edir = math.atan2(-ex, -ey)
        sg = 1.0 if r["omega_rad_s"] >= 0 else -1.0
        r["exit_dir_offset_deg"] = math.degrees(math.remainder(edir - (cdir + sg * math.pi / 2), 2 * math.pi)) * sg  # + = outward... signed so + = ahead of perpendicular
    print("[hits] exit direction minus the perpendicular to pivot->puck (deg; negative = turned back toward the stick): "
          + ", ".join(f"{r['frame']}: {r['exit_dir_offset_deg']:+.1f}" for r in rows))
    out = {"clip": CLIP, "m_per_px": mpp, "hits": rows, "linear_origin": k0, "linear_fit": lin.tolist(), "omega_noise_sd_rad_s": w_noise}
    out["calib"] = calib_fit(rows, p, m)
    (results / "stick_hits.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


# ---------------------------------------------------------------- calib scenarios + held-out fit

def write_scenarios(rows, p, m, base_params: Path):
    DEST.mkdir(parents=True, exist_ok=True)
    pf, pu, pv, su, sv = load_puck()
    fps = m.fps
    dt = p["timestep"]
    acc = p["stick_max_ang_accel"]
    files = []
    for n, r in enumerate(rows):
        px, py = m.px_to_m(*r["pivot_ref_px"])
        bx, by = m.px_to_m(*r["puck_ref_px"])
        cdir = math.atan2(by - py, bx - px)
        w = r["omega_rad_s"]
        sg = 1.0 if w >= 0 else -1.0
        # Start just short of the puck: the stick edge clears the puck by one substep of turning.
        clear = math.asin(min(0.99, (p["puck_radius"] + p["stick_thickness"]) / r["r_c_m"]))
        h_sub = dt / max(1, int(round(p["substeps"])))
        a0 = cdir - sg * (clear + abs(w) * h_sub)  # one substep short of touching the puck
        # Target just far enough ahead that the chase law asks for |w| at contact (no speed-up during the approach).
        err = max(w * w / (2 * acc), abs(w) * dt) * 1.0001 + abs(w) * h_sub
        i0, i1 = r["glide_rows"]
        t_c = (r["frame"] + 0.5) / fps
        t_start = t_c - h_sub
        rows_csv = ["t,x,y,z"]
        for i in range(i0, i1 + 1):
            x, y = m.px_to_m(pu[i], pv[i])
            rows_csv.append(f"{float(pf[i] / fps - t_start)!r},{float(x)!r},{float(y)!r},0")
        steps = int(math.ceil((pf[i1] / fps - t_start) / dt)) + 2
        player = {"team": 0, "x": float(px), "y": float(py), "vx": 0.0, "vy": 0.0, "facing": 0.0, "stick_angle": a0, "stick_omega": w,
                  "dash_timer": 0.0, "dash_cooldown": 0.0, "dash_prev": False}
        scen = {"name": f"hit_{n + 1:02d}", "description": f"MEASURED stick hit from {CLIP} frame {r['frame']} ({r['group']} swing, w {w:.2f} rad/s); "
                "placeholder meter scale; stick placed just short of the puck at the measured angular speed",
                "steps": steps,
                "initial_state": {"tick": 0, "puck": {"x": float(bx), "y": float(by), "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "in_net": None},
                                  "players": [player], "score": [0, 0]},
                "inputs": [{"step": 0, "player": 0, "input": {"move_x": 0.0, "move_y": 0.0, "stick_target_angle": a0 + sg * err,
                                                               "dash": False, "lift": False}}]}
        sp = DEST / f"hit_{n + 1:02d}.scenario.json"
        cp = DEST / f"hit_{n + 1:02d}.csv"
        sp.write_text(json.dumps(scen, indent=1) + "\n")
        cp.write_text("\n".join(rows_csv) + "\n")
        files.append((str(sp), str(cp), r["group"]))
    return files


def calib_cmd(args):
    return subprocess.run(["cargo", "run", "-q", "--release", "-p", "calib", "--"] + args, cwd=REPO, check=True, capture_output=True, text=True).stdout


def eval_score(params: Path, files) -> dict:
    out = calib_cmd(["eval", "--params", str(params)] + [x for f in files for x in f[:2]])
    return out


def path_scores(text: str) -> list[float]:
    vals = re.findall(r"path_rmse[ =:]+([0-9.eE+-]+)", text)
    return [float(v) for v in vals]


def calib_fit(rows, p, m) -> dict:
    base = json.loads(PARAMS_PATH.read_text())
    chase = json.loads((RESULTS / "stick_chase.json").read_text()) if (RESULTS / "stick_chase.json").exists() else None
    if chase:
        base["params"]["stick_max_ang_speed"]["value"] = chase["fit_all"]["stick_max_ang_speed"]
        base["params"]["stick_max_ang_accel"]["value"] = chase["flick_starts"]["accel_best"] or chase["fit_all"]["stick_max_ang_accel"]
    shape = json.loads((RESULTS / "stick_shape.json").read_text()) if (RESULTS / "stick_shape.json").exists() else None
    if shape:
        base["params"]["stick_length"]["value"] = shape["stick_length_m"]
        base["params"]["stick_thickness"]["value"] = shape["stick_thickness_m"]
    for k in ("max_puck_speed_ice_normal", "max_puck_speed_ice_dash", "max_puck_speed_air"):
        base["params"][k]["value"] = CAP_OFF
    bp = REPO / "target" / "stick_hits_base.json"
    bp.parent.mkdir(exist_ok=True)
    bp.write_text(json.dumps(base, indent=1))
    pv = {k: v["value"] for k, v in base["params"].items()}
    files = write_scenarios(rows, pv, m, bp)
    res = {"base_params": {k: pv[k] for k in ("stick_max_ang_speed", "stick_max_ang_accel", "stick_length", "stick_inner", "stick_thickness",
                                               "puck_mass", "puck_radius", "stick_mass", "stick_restitution")}, "folds": []}
    print(f"[hits] calib base: params.json + chase fit (w_max {pv['stick_max_ang_speed']:.2f}, accel {pv['stick_max_ang_accel']:.0f}) + "
          f"stick_length {pv['stick_length']:.3f} m, puck speed caps lifted to {CAP_OFF:g} m/s")
    groups = sorted({g for *_, g in files}, key=["slow", "medium", "fast"].index)
    res["one_param_folds"] = []
    for g in groups + ["all"]:  # restitution only, stick_mass fixed at the placeholder (the pair is degenerate)
        train = [f for f in files if f[2] != g] if g != "all" else files
        test = [f for f in files if f[2] == g] if g != "all" else files
        outp = REPO / "target" / f"stick_hits_fit1_{g}.json"
        calib_cmd(["fit", "--params", str(bp), "--fit", "stick_restitution", "--out", str(outp), "--objective", "path"] + [x for f in train for x in f[:2]])
        e = json.loads(outp.read_text())["params"]["stick_restitution"]["value"]
        ev_f, ev_p = eval_score(outp, test), eval_score(bp, test)
        res["one_param_folds"].append({"held_out": g, "stick_restitution": e, "stick_mass_fixed": pv["stick_mass"], "test_eval_fitted": ev_f,
                                       "test_eval_placeholder": ev_p})
        print(f"[hits] restitution only (stick_mass {pv['stick_mass']:g} kg fixed), {'held out ' + g if g != 'all' else 'all hits'}: e = {e:.4f}")
    for g in groups + ["all"]:
        train = [f for f in files if f[2] != g] if g != "all" else files
        test = [f for f in files if f[2] == g] if g != "all" else files
        outp = REPO / "target" / f"stick_hits_fit_{g}.json"
        txt = calib_cmd(["fit", "--params", str(bp), "--fit", "stick_mass,stick_restitution", "--out", str(outp), "--objective", "path"]
                        + [x for f in train for x in f[:2]])
        fitted = json.loads(outp.read_text())["params"]
        mass, e = fitted["stick_mass"]["value"], fitted["stick_restitution"]["value"]
        ev_f = eval_score(outp, test)
        ev_p = eval_score(bp, test)
        row = {"held_out": g, "train_n": len(train), "test_n": len(test), "stick_mass": mass, "stick_restitution": e,
               "test_eval_fitted": ev_f, "test_eval_placeholder": ev_p, "fit_log_tail": txt.strip().splitlines()[-6:]}
        res["folds"].append(row)
        print(f"[hits] {'held out ' + g if g != 'all' else 'all hits (in-sample)'}: train {len(train)}, test {len(test)} -> stick_mass {mass:.4f} kg, "
              f"stick_restitution {e:.4f}")
        print("   fitted     : " + " | ".join(ln.strip() for ln in ev_f.strip().splitlines() if ln.strip()))
        print("   placeholder: " + " | ".join(ln.strip() for ln in ev_p.strip().splitlines() if ln.strip()))
    return res
