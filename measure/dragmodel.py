"""`dragmodel` command: does the puck's ice deceleration need a quadratic drag term?

1. Held-out model comparison. Candidates for the glide deceleration a(v):
     linear        a = k v
     linear+const  a = mu + k v
     linear+quad   a = k v + c v^2
     all three     a = mu + k v + c v^2
     const only    a = mu                       (reference)
   Each is integrated along distance (dv/ds = -a(v)/v) from the scenario's initial speed and scored like calib's
   path objective: (stop-distance shortfall)^2 + 0.01 (v^2 error)^2 at equal distance travelled, with measured
   speeds from local quadratic fits (+-5 rows). Fit on one clip's ground glides, scored on the other's.
2. Smoothing check: decel/v by speed band with different local-fit half-windows.
3. Synthetic check: pure linear-drag glides (each real stretch's start speed and length), put through the real
   capture faults (25% of frames +-1/180 s, 0.28% duplicated, 0.28% skipped, 0.3 px noise, duplicates dropped
   like the tracker), then the same band analysis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .segment import OUT

V2W = 0.01
BANDS = (20, 50, 80, 120, 170, 230, 300, 400)
MODELS = {
    "linear (k v)": ("k",),
    "linear + const (mu + k v)": ("mu", "k"),
    "linear + quadratic (k v + c v^2)": ("k", "c"),
    "const + linear + quadratic": ("mu", "k", "c"),
    "const only (mu)": ("mu",),
}


def smooth_vs(t, x, y, half=5):
    P, V = [], []
    for k in range(len(t)):
        a, b = max(k - half, 0), min(k + half, len(t) - 1)
        tt = t[a : b + 1] - t[k]
        deg = 2 if b - a >= 4 else 1
        cx, cy = np.polyfit(tt, x[a : b + 1], deg), np.polyfit(tt, y[a : b + 1], deg)
        P.append((cx[-1], cy[-1]))
        V.append(math.hypot(cx[-2], cy[-2]))
    P = np.array(P)
    return np.array(V), np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]


def load_glides(clip_filter=None):
    man = json.loads((OUT / "manifest.json").read_text())
    out = []
    for s in man["segments"]:
        if s["possibly_airborne"] or s["label"] != "glide" or (clip_filter and s["clip"] != clip_filter):
            continue
        a = np.loadtxt(OUT / s["csv"], delimiter=",", skiprows=1, ndmin=2)
        sc = json.loads((OUT / s["scenario"]).read_text())["initial_state"]["puck"]
        V, S = smooth_vs(a[:, 0], a[:, 1], a[:, 2])
        out.append({"id": s["id"], "clip": s["clip"], "v0": math.hypot(sc["vx"], sc["vy"]), "V": V, "S": S})
    return out


def predict(v0, S, mu=0.0, k=0.0, c=0.0, ds=0.01):
    """v(s) by integrating dv/ds = -(mu + k v + c v^2) / v; returns v at S and the stop distance."""
    smax = S[-1] + ds
    n = int(smax / ds) + 2
    v = v0
    vs = np.empty(n)
    stop = math.inf
    for i in range(n):
        vs[i] = v
        if v <= 0:
            vs[i:] = 0
            stop = min(stop, i * ds)
            break
        # midpoint step in s
        f = lambda vv: -(mu + k * vv + c * vv * vv) / max(vv, 1e-9)
        vm = v + 0.5 * ds * f(v)
        v = v + ds * f(max(vm, 1e-9)) if vm > 0 else 0.0
    grid = np.arange(n) * ds
    return np.interp(S, grid, vs), stop


def seg_residuals(g, mu, k, c):
    vp, stop = predict(g["v0"], g["S"], mu, k, c)
    short = np.clip(g["S"] - stop, 0, None) if math.isfinite(stop) else np.zeros_like(g["S"])
    return np.concatenate([short, math.sqrt(V2W) * (g["V"] ** 2 - vp**2)]) / math.sqrt(len(g["S"]))


def seg_score(g, mu, k, c):
    r = seg_residuals(g, mu, k, c)
    return float(np.sum(r**2))


def fit_model(glides, names):
    x0 = {"mu": 0.1, "k": 0.4, "c": 0.002}
    def unpack(x):
        d = dict(zip(names, x))
        return d.get("mu", 0.0), d.get("k", 0.0), d.get("c", 0.0)
    fun = lambda x: np.concatenate([seg_residuals(g, *unpack(x)) for g in glides])
    r = least_squares(fun, [x0[n] for n in names], bounds=([0.0] * len(names), [np.inf] * len(names)))
    return dict(zip(names, r.x.tolist())), unpack(r.x)


def heldout_table(clips):
    data = {c: load_glides(c) for c in clips}
    rows = []
    for name, names in MODELS.items():
        row = {"model": name, "directions": []}
        for train, test in ((clips[0], clips[1]), (clips[1], clips[0])):
            vals, (mu, k, c) = fit_model(data[train], names)
            te = float(np.mean([seg_score(g, mu, k, c) for g in data[test]]))
            tr = float(np.mean([seg_score(g, mu, k, c) for g in data[train]]))
            row["directions"].append({"train": train, "test": test, "fitted": vals, "train_score": tr, "test_score": te})
        rows.append(row)
    return rows, {c: len(v) for c, v in data.items()}


def bands(V, A):
    out = []
    for lo in BANDS:
        m = (V >= lo) & (V < lo * 1.4)
        if m.sum() >= 5:
            out.append((lo, int(m.sum()), float(np.median(A[m] / V[m]))))
    return out


def synth_glides(stretches, k=0.43, seed=7, fps=60.0):
    """Linear-drag glides in px with the real capture faults; returns (f, u, v) arrays per stretch."""
    rng = np.random.default_rng(seed)
    out = []
    for st in stretches:
        v0, T = st["v_start_px_s"], st["duration_s"] + 0.3
        n = int(T * fps) + 1
        f = np.arange(n)
        tg = f / fps
        r = rng.random(n)
        jit = np.where(rng.random(n) < 0.25, rng.choice([-1, 1], n) / 180.0, 0.0)
        tg = tg + jit
        tg = np.where((r >= 0.0028) & (r < 0.0056), f / fps + 1 / fps, tg)  # skipped frame
        s = v0 / k * (1 - np.exp(-k * np.clip(tg, 0, None)))
        ang = rng.uniform(0, 2 * math.pi)
        u = 500 + s * math.cos(ang) + rng.normal(0, 0.3, n)
        v = 500 + s * math.sin(ang) + rng.normal(0, 0.3, n)
        dup = r < 0.0028
        for i in np.nonzero(dup)[0]:
            if i > 0:
                u[i], v[i] = u[i - 1], v[i - 1]
        keep = np.r_[True, ~((np.hypot(np.diff(u), np.diff(v)) < 0.05))]
        out.append((f[keep], u[keep], v[keep]))
    return out


def local_va(f, u, v, fps=60.0, half=8):
    t = f / fps
    Vs, As = [], []
    for q in range(len(t)):
        a_, b_ = max(q - half, 0), min(q + half, len(t) - 1)
        if b_ - a_ < max(5, int(1.25 * half)):
            continue
        tt = t[a_ : b_ + 1] - t[q]
        cu, cv_ = np.polyfit(tt, u[a_ : b_ + 1], 2), np.polyfit(tt, v[a_ : b_ + 1], 2)
        vel = np.array([cu[1], cv_[1]])
        sp = float(np.linalg.norm(vel))
        if sp > 20:
            Vs.append(sp)
            As.append(float(-(2 * cu[0] * vel[0] + 2 * cv_[0] * vel[1]) / sp))
    return np.array(Vs), np.array(As)


def run_dragmodel(results: Path, out_path: Path = OUT / "dragmodel.json") -> dict:
    from .estimate import glides as real_glides

    man = json.loads((OUT / "manifest.json").read_text())
    clips = sorted({s["clip"] for s in man["segments"] if not s["possibly_airborne"]})
    rows, counts = heldout_table(clips)
    print(f"[dragmodel] held-out comparison on ground glide segments: {counts} (score = mean over test segments of shortfall^2 + 0.01 * v^2 error^2, m and m^2/s^2)")
    print(f"[dragmodel] {'model':36s} | {'train ' + clips[0][:6] + ' -> test ' + clips[1][:6]:40s} | {'train ' + clips[1][:6] + ' -> test ' + clips[0][:6]:40s}")
    for r in rows:
        cells = []
        for d in r["directions"]:
            cells.append(f"test {d['test_score']:8.3f} (train {d['train_score']:7.3f}) " + " ".join(f"{k}={v:.4g}" for k, v in d["fitted"].items()))
        print(f"[dragmodel] {r['model']:36s} | {cells[0]:40s} | {cells[1]}")
    res = {"heldout": rows, "glide_counts": counts}
    # smoothing windows on the real data
    res["smoothing"] = {}
    print("[dragmodel] decel/v (1/s) by speed band (px/s), real ground glides, by local-fit half-window (rows):")
    print("[dragmodel]   half " + " ".join(f"{lo:>4d}-{round(lo * 1.4):<4d}" for lo in BANDS))
    for half in (4, 8, 12, 16):
        G = real_glides(results, man, half=half)
        V = np.concatenate([g["_v"] for g in G])
        A = np.concatenate([g["_a"] for g in G])
        b = {lo: (n, r) for lo, n, r in bands(V, A)}
        res["smoothing"][half] = [{"band": lo, "n": n, "decel_over_v": r} for lo, (n, r) in b.items()]
        print(f"[dragmodel]   {half:4d} " + " ".join(f"{b[lo][1]:9.3f}" if lo in b else "        -" for lo in BANDS))
    # synthetic: pure linear drag through the capture faults
    G8 = real_glides(results, man, half=8)
    syn = synth_glides(G8)
    res["synthetic"] = {}
    print("[dragmodel] synthetic pure linear drag k = 0.43/s with real start speeds/lengths and capture faults (truth: 0.430 in every band):")
    for half in (4, 8, 12, 16):
        V, A = [], []
        for f, u, v in syn:
            a, b_ = local_va(f, u, v, half=half)
            V.append(a)
            A.append(b_)
        b = {lo: (n, r) for lo, n, r in bands(np.concatenate(V), np.concatenate(A))}
        res["synthetic"][half] = [{"band": lo, "n": n, "decel_over_v": r} for lo, (n, r) in b.items()]
        print(f"[dragmodel]   {half:4d} " + " ".join(f"{b[lo][1]:9.3f}" if lo in b else "        -" for lo in BANDS))
    # per-clip band table (is the dip one clip's stretches?)
    res["by_clip"] = {}
    for c in clips:
        V = np.concatenate([g["_v"] for g in G8 if g["clip"] == c])
        A = np.concatenate([g["_a"] for g in G8 if g["clip"] == c])
        b = {lo: (n, r) for lo, n, r in bands(V, A)}
        res["by_clip"][c] = [{"band": lo, "n": n, "decel_over_v": r} for lo, (n, r) in b.items()]
        print(f"[dragmodel]   {c[:10]:10s} (half 8) " + " ".join(f"{b[lo][1]:6.3f}(n{b[lo][0]})" if lo in b else "   -" for lo in BANDS))
    out_path.write_text(json.dumps(res, indent=2, default=float) + "\n")
    print(f"[dragmodel] wrote {out_path}")
    return res
