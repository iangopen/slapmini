"""`estimate` command: scale-free physics from the real segments.

Restitution: for each detected wall bounce, quadratic fits of the path on rows [b-14, b-3] and
[b+3, b+14] give the velocity just before and just after contact (both extrapolated to the bounce
frame). Reported: speed ratio |v_after|/|v_before| (scale-free), the board-normal ratio
-(v_after.n)/(v_before.n) (= the sim's board_restitution) and the tangential ratio (1 - board_friction).

Glide deceleration shape: free glide stretches (no bounce, >= 10 px from the boards, >= 0.5 s, speed
falling >= 20%). The stretch is cut into 0.25 s chunks; for each chunk d(v^2)/ds and the mean speed v.
For dv/dt = -mu - k v:  d(v^2)/ds = -2 mu - 2 k v, a straight line in v. k (1/s) is scale-free; mu is in
px/s^2 (m/s^2 only at the placeholder scale). Constant friction means k ~ 0; pure drag means mu ~ 0.
Also: RMS misfit of the pure models v^2 = a - 2 mu s (constant) and v = b - k s (linear drag).
Segments flagged possibly_airborne are excluded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .geometry import RinkMap
from .segment import OUT, board_distance, board_normal

MIN_GLIDE_S = 0.5


def _seg_rows(results: Path, seg: dict):
    R = json.loads((results / f"{seg['clip']}.track.json").read_text())["rows"]
    f = np.array(R["frame"])
    sel = (np.array(R["tracklet"]) == seg["tracklet"]) & (f >= seg["start_frame"]) & (f <= seg["end_frame"])
    f, u, v = f[sel], np.array(R.get("ug_px", R["u_px"]))[sel], np.array(R.get("vg_px", R["v_px"]))[sel]
    keep = np.r_[True, np.hypot(np.diff(u), np.diff(v)) > 0.05]  # stale repeats, if any
    return f[keep], u[keep], v[keep]


def _vel_at(f, u, v, fps, at):
    t = (f - at) / fps
    deg = 2 if len(t) >= 6 else 1
    return float(np.polyfit(t, u, deg)[-2]), float(np.polyfit(t, v, deg)[-2])


def bounces(results: Path, manifest: dict) -> list[dict]:
    out = []
    for seg in manifest["segments"]:
        if seg.get("possibly_airborne") or not seg["wall_bounce_frames"]:
            continue
        m = RinkMap.load(results / f"{seg['clip']}.map.json")
        f, u, v = _seg_rows(results, seg)
        bl = seg["wall_bounce_frames"]
        for i, b in enumerate(bl):
            lo = max(b - 14, bl[i - 1] + 4 if i else -10**9)
            hi = min(b + 14, bl[i + 1] - 4 if i + 1 < len(bl) else 10**9)
            pre = (f >= lo) & (f <= b - 3)
            post = (f >= b + 3) & (f <= hi)
            if pre.sum() < 5 or post.sum() < 5:
                continue
            vb = _vel_at(f[pre], u[pre], v[pre], m.fps, b)
            va = _vel_at(f[post], u[post], v[post], m.fps, b)
            k = int(np.argmin(np.abs(f - b)))
            nx, ny = board_normal(m, u[k], v[k])
            tx, ty = -ny, nx
            vbn, van = vb[0] * nx + vb[1] * ny, va[0] * nx + va[1] * ny
            vbt, vat = vb[0] * tx + vb[1] * ty, va[0] * tx + va[1] * ty
            sb, sa = math.hypot(*vb), math.hypot(*va)
            if vbn >= 0 or sb < 60:  # must approach the board, faster than 1 px/frame
                continue
            corner = bool(board_distance(m, np.array([u[k]]), np.array([v[k]]))[0] < 30 and (
                (u[k] < m.board_px["left"] + m.corner_radius_px or u[k] > m.board_px["right"] - m.corner_radius_px)
                and (v[k] < m.board_px["top"] + m.corner_radius_px or v[k] > m.board_px["bottom"] - m.corner_radius_px)))
            out.append({
                "segment": seg["id"], "clip": seg["clip"], "frame": int(b),
                "speed_before_px_s": round(sb, 2), "speed_after_px_s": round(sa, 2),
                "speed_ratio": round(sa / sb, 4),
                "normal_ratio": round(-van / vbn, 4),
                "tangential_ratio": round(vat / vbt, 4) if abs(vbt) > 20 else None,
                "incidence_deg_from_normal": round(math.degrees(math.atan2(abs(vbt), -vbn)), 1),
                "corner": corner,
            })
    return out


def glides(results: Path, manifest: dict, half: int = 8) -> list[dict]:
    out = []
    for seg in manifest["segments"]:
        if seg.get("possibly_airborne"):
            continue
        m = RinkMap.load(results / f"{seg['clip']}.map.json")
        fps = m.fps
        f, u, v = _seg_rows(results, seg)
        bd = board_distance(m, u, v)
        free = bd > 10.0
        for b in seg["wall_bounce_frames"]:
            free &= np.abs(f - b) > 10
        # longest runs of free rows
        runs, k = [], 0
        while k < len(f):
            if not free[k]:
                k += 1
                continue
            j = k
            while j + 1 < len(f) and free[j + 1] and f[j + 1] - f[j] <= 3:
                j += 1
            runs.append((k, j))
            k = j + 1
        for k0, k1 in runs:
            if (f[k1] - f[k0]) / fps < MIN_GLIDE_S:
                continue
            ff, uu, vv = f[k0 : k1 + 1], u[k0 : k1 + 1], v[k0 : k1 + 1]
            t = (ff - ff[0]) / fps
            # Local quadratic fits over +-8 rows (~0.27 s): speed v and along-path deceleration a.
            # Capture jitter only adds zero-mean noise to these.
            P, Vs, As, Ts = [], [], [], []
            for q in range(len(t)):
                a_, b_ = max(q - half, 0), min(q + half, len(t) - 1)
                if b_ - a_ < max(5, int(1.25 * half)):
                    continue
                tt = t[a_ : b_ + 1] - t[q]
                cu, cv_ = np.polyfit(tt, uu[a_ : b_ + 1], 2), np.polyfit(tt, vv[a_ : b_ + 1], 2)
                vel = np.array([cu[1], cv_[1]])
                sp = float(np.linalg.norm(vel))
                P.append((cu[2], cv_[2]))
                Vs.append(sp)
                As.append(float(-(2 * cu[0] * vel[0] + 2 * cv_[0] * vel[1]) / sp) if sp > 0 else 0.0)
                Ts.append(t[q])
            V, A = np.array(Vs), np.array(As)
            if len(V) < 12 or V[0] < 60 or V[-1] > 0.8 * V[0]:
                continue
            P = np.array(P)
            s_ = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
            ok = V > 20  # below ~0.3 px/frame the fits are noise
            ratio = A[ok] / V[ok]
            mu, k_drag = (np.polyfit(V[ok], A[ok], 1)[::-1] if ok.sum() >= 6 and np.ptp(V[ok]) > 30 else (float("nan"), float("nan")))
            ca = np.polyfit(s_, V**2, 1)
            ra = float(np.sqrt(np.mean((np.sqrt(np.clip(np.polyval(ca, s_), 0, None)) - V) ** 2)))
            cb = np.polyfit(s_, V, 1)
            rb = float(np.sqrt(np.mean((np.polyval(cb, s_) - V) ** 2)))
            out.append({
                "segment": seg["id"], "clip": seg["clip"], "frames": [int(ff[0]), int(ff[-1])],
                "duration_s": round(float(t[-1]), 3), "v_start_px_s": round(float(V[0]), 1), "v_end_px_s": round(float(V[-1]), 1),
                "samples": int(ok.sum()),
                "decel_over_v_median_per_s": round(float(np.median(ratio)), 4),
                "mu_px_s2": round(float(mu), 2), "k_per_s": round(float(k_drag), 4),
                "rms_v_const_decel_px_s": round(ra, 3), "rms_v_linear_drag_px_s": round(rb, 3),
                "better_pure_model": "constant" if ra < rb else "linear drag",
                "m_per_px": m.m_per_px,
                "_v": V[ok].tolist(), "_a": A[ok].tolist(),
            })
    return out


def _stats(x):
    x = np.array([a for a in x if a is not None and not (isinstance(a, float) and math.isnan(a))], float)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "median": round(float(np.median(x)), 4), "mean": round(float(np.mean(x)), 4), "sd": round(float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, 4),
            "p25": round(float(np.percentile(x, 25)), 4), "p75": round(float(np.percentile(x, 75)), 4), "min": round(float(x.min()), 4), "max": round(float(x.max()), 4)}


def run_estimate(results: Path, manifest_path: Path = OUT / "manifest.json", out_path: Path = OUT / "estimates.json") -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    B = bounces(results, manifest)
    G = glides(results, manifest)
    mpp = float(np.mean([g["m_per_px"] for g in G])) if G else float("nan")
    Vp = np.concatenate([g["_v"] for g in G]) if G else np.zeros(0)
    Ap = np.concatenate([g["_a"] for g in G]) if G else np.zeros(0)
    pooled = {"samples": int(len(Vp))}
    if len(Vp) > 20:
        c1 = np.polyfit(Vp, Ap, 1)
        c2 = np.polyfit(Vp, Ap, 2)
        pooled.update({
            "fit_decel_eq_mu_plus_k_v": {"k_per_s": round(float(c1[0]), 4), "mu_px_s2": round(float(c1[1]), 2), "mu_m_s2_PLACEHOLDER_SCALE": round(float(c1[1]) * mpp, 4), "rms_px_s2": round(float(np.std(Ap - np.polyval(c1, Vp))), 2)},
            "fit_with_v2_term": {"c_per_px": round(float(c2[0]), 6), "k_per_s": round(float(c2[1]), 4), "mu_px_s2": round(float(c2[2]), 2), "rms_px_s2": round(float(np.std(Ap - np.polyval(c2, Vp))), 2)},
            "decel_over_v_by_speed": [
                {"v_px_s": [lo, round(lo * 1.4)], "n": int(((Vp >= lo) & (Vp < lo * 1.4)).sum()), "median_per_s": round(float(np.median((Ap / Vp)[(Vp >= lo) & (Vp < lo * 1.4)])), 3)}
                for lo in (20, 50, 80, 120, 170, 230, 300, 400) if ((Vp >= lo) & (Vp < lo * 1.4)).sum() >= 5
            ],
        })
    res = {
        "description": __doc__,
        "excluded": "segments flagged possibly_airborne (the lifted-shot clip)",
        "restitution": {
            "bounces": B,
            "speed_ratio": _stats([b["speed_ratio"] for b in B]),
            "normal_ratio": _stats([b["normal_ratio"] for b in B]),
            "tangential_ratio": _stats([b["tangential_ratio"] for b in B]),
            "speed_ratio_straight_boards": _stats([b["speed_ratio"] for b in B if not b["corner"]]),
            "normal_ratio_straight_boards": _stats([b["normal_ratio"] for b in B if not b["corner"]]),
            "segments": sorted({b["segment"] for b in B}),
            "clips": sorted({b["clip"] for b in B}),
        },
        "glide": {
            "stretches": [{k: v for k, v in g.items() if not k.startswith("_")} for g in G],
            "decel_over_v_per_stretch_per_s": _stats([g["decel_over_v_median_per_s"] for g in G]),
            "k_per_s_per_stretch": _stats([g["k_per_s"] for g in G]),
            "mu_px_s2_per_stretch": _stats([g["mu_px_s2"] for g in G]),
            "better_pure_model_counts": {k: sum(1 for g in G if g["better_pure_model"] == k) for k in ("constant", "linear drag")},
            "pooled": pooled,
            "segments": sorted({g["segment"] for g in G}),
            "clips": sorted({g["clip"] for g in G}),
        },
    }
    Path(out_path).write_text(json.dumps(res, indent=2) + "\n")
    r, g = res["restitution"], res["glide"]
    print(f"[estimate] wall bounces used: {len(B)} from {len(r['segments'])} segments ({', '.join(r['clips'])})")
    for b in B:
        print(f"[estimate]   {b['segment']} f{b['frame']}: speed {b['speed_before_px_s']:.0f} -> {b['speed_after_px_s']:.0f} px/s, speed ratio {b['speed_ratio']:.3f}, normal ratio {b['normal_ratio']:.3f}, tangential {b['tangential_ratio']}, incidence {b['incidence_deg_from_normal']} deg{' (corner)' if b['corner'] else ''}")
    for key in ("speed_ratio", "normal_ratio", "tangential_ratio"):
        st = r[key]
        if st["n"]:
            print(f"[estimate] {key:16s}: median {st['median']:.3f}, mean {st['mean']:.3f}, sd {st['sd']:.3f}, IQR {st['p25']:.3f}..{st['p75']:.3f}, n {st['n']}")
    print(f"[estimate] glide stretches used: {len(G)} from {len(g['segments'])} segments ({', '.join(g['clips'])})")
    for x in G:
        print(f"[estimate]   {x['segment']} f{x['frames'][0]}-{x['frames'][1]} ({x['duration_s']} s): v {x['v_start_px_s']:.0f} -> {x['v_end_px_s']:.0f} px/s; decel/v median {x['decel_over_v_median_per_s']:.3f} 1/s; fit mu {x['mu_px_s2']:.1f} px/s2 + k {x['k_per_s']:.3f} v; pure-model rms const {x['rms_v_const_decel_px_s']:.2f} vs drag {x['rms_v_linear_drag_px_s']:.2f} px/s -> {x['better_pure_model']}")
    for key in ("decel_over_v_per_stretch_per_s", "k_per_s_per_stretch", "mu_px_s2_per_stretch"):
        st = g[key]
        if st["n"]:
            print(f"[estimate] {key:31s}: median {st['median']:.4f}, mean {st['mean']:.4f}, sd {st['sd']:.4f}, IQR {st['p25']:.4f}..{st['p75']:.4f}, n {st['n']}")
    pl = g["pooled"]
    if "fit_decel_eq_mu_plus_k_v" in pl:
        f1, f2 = pl["fit_decel_eq_mu_plus_k_v"], pl["fit_with_v2_term"]
        print(f"[estimate] pooled ({pl['samples']} local fits): decel = {f1['mu_px_s2']:.2f} px/s2 ({f1['mu_m_s2_PLACEHOLDER_SCALE']:.3f} m/s2 at the placeholder scale) + {f1['k_per_s']:.4f}/s * v, rms {f1['rms_px_s2']:.1f}; adding c v^2: c = {f2['c_per_px']:.2e}/px, rms {f2['rms_px_s2']:.1f}")
        print("[estimate] decel/v by speed band: " + ", ".join(f"{b['v_px_s'][0]}-{b['v_px_s'][1]} px/s: {b['median_per_s']:.3f} (n {b['n']})" for b in pl["decel_over_v_by_speed"]))
    print(f"[estimate] better pure model: {g['better_pure_model_counts']}  (placeholder scale {mpp:.5f} m/px)")
    print(f"[estimate] wrote {out_path}")
    return res
