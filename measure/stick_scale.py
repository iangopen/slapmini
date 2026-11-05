"""Per-clip scale for the stick clips (`python -m measure.stick scale clips/A.mkv clips/B.mkv`).

The zoomed clip (rotatinganddizzy) has no board in view, so `measure map` can't run on it. Its scale comes from the
player body, calibrated at the normal zoom:
- body radius in px at both zooms, measured the same way: per-pixel median of frames (the turning stick drops out),
  saturation along rays from the pivot, half-max edge between the grey body (S = 0) and the blue glow just outside;
  circle fit with outlier rejection;
- at the normal zoom (hittingpuck, which has a map) the body radius in placeholder meters = px x the map's ice-plane
  m_per_px; the zoomed clip's m_per_px = that / its body px ("disc scale");
- cross-check with an ice marking: the centre circle (ellipse fit to the grey ring, measure.shape.fit_ring), whose
  radius in placeholder meters comes from the normal-zoom maps. If body/circle agree between the zooms, the raised body
  shows no zoom-dependent parallax and both scales agree.
params.json player_radius (0.4557 m) is the PINK disc of the old clips' skin (half-max of pinkness); the 2026-11 clips
show a grey body, a different picture of the same player, so player_radius is reported for comparison only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import PARAMS_PATH, REPO, RinkMap, load_params
from .video import iter_frames

RESULTS = REPO / "measure" / "results"
OUT = RESULTS / "stick_scale.json"
NORMAL_MAPS = ["justpuckphysicshittingitatfixedpoint", "justhittingwallphysicsfixedpoint", "justliftedshots"]


def median_crop_frame(clip: Path, n: int = 81) -> np.ndarray:
    from .video import median_frame

    return median_frame(clip, n)


def body_edge_radius(med: np.ndarray, centre: tuple[float, float], r_guess: float) -> dict:
    sat = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    pts = []
    rs = np.arange(0.4 * r_guess, 1.6 * r_guess, max(0.02, r_guess / 400))
    for a in np.linspace(-math.pi, math.pi, 360, endpoint=False):
        xs = (centre[0] + rs * math.cos(a)).astype(np.float32).reshape(1, -1)
        ys = (centre[1] + rs * math.sin(a)).astype(np.float32).reshape(1, -1)
        prof = cv2.remap(sat, xs, ys, cv2.INTER_LINEAR)[0]
        inner = float(np.median(prof[rs < 0.7 * r_guess]))
        k_pk = int(np.argmax(np.where(rs > 0.85 * r_guess, prof, -1)))
        peak = float(prof[k_pk])
        if peak - inner < 30:
            continue
        half = inner + 0.5 * (peak - inner)
        k = int(np.argmax(prof[: k_pk + 1] >= half))
        if k == 0:
            continue
        r = rs[k - 1] + (half - prof[k - 1]) / (prof[k] - prof[k - 1]) * (rs[k] - rs[k - 1])
        pts.append((centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)))
    P = np.array(pts)
    q = least_squares(lambda z: np.hypot(P[:, 0] - z[0], P[:, 1] - z[1]) - z[2], [centre[0], centre[1], r_guess]).x
    keep = np.ones(len(P), bool)
    for _ in range(3):
        res = np.abs(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]) - q[2])
        keep = res < max(0.3, 2.5 * np.median(res))
        q = least_squares(lambda z: np.hypot(P[keep, 0] - z[0], P[keep, 1] - z[1]) - z[2], q).x
    res = np.hypot(P[keep, 0] - q[0], P[keep, 1] - q[1]) - q[2]
    return {"centre_px": [round(float(q[0]), 3), round(float(q[1]), 3)], "radius_px": round(float(q[2]), 3),
            "edge_points_kept": int(keep.sum()), "edge_points": int(len(P)), "fit_rms_px": round(float(np.sqrt(np.mean(res ** 2))), 3)}


def centre_circle_m() -> dict:
    """Centre circle radius in placeholder meters from the normal-zoom maps (ice-plane scale)."""
    vals = []
    for c in NORMAL_MAPS:
        p = RESULTS / f"{c}.map.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        f = m["extra"]["shape"]["circles"]["fits"]["center"]
        d = (f["horizontal_diameter_px"] + f["vertical_diameter_px"]) / 2
        vals.append(d / 2 * m["m_per_px"])
    return {"radius_m": float(np.mean(vals)), "sd_m": float(np.std(vals)), "n_clips": len(vals)}


def run_scale(clips: list[Path]) -> dict:
    params = load_params(PARAMS_PATH)
    cc = centre_circle_m()
    out = {"centre_circle_radius_m_placeholder": cc, "params_player_radius_m": params["player_radius"], "clips": {}}
    for clip in clips:
        clip = Path(clip)
        name = clip.stem
        st = json.loads((RESULTS / f"{name}.stick.json").read_text())
        R = st["rows"]
        centre = (float(np.median(R["pivot_u"])), float(np.median(R["pivot_v"])))
        r_guess = st["summary"]["body_radius_px"]
        med_path = clip.parent / f"{name}.bg.png"
        med = cv2.imread(str(med_path)) if med_path.exists() else median_crop_frame(clip)
        if not med_path.exists():
            cv2.imwrite(str(med_path), med)
        body = body_edge_radius(med, centre, r_guess)
        # Centre circle: from the map if there is one, else fit the ring in the median frame.
        mp = RESULTS / f"{name}.map.json"
        rec = {"body": body, "tip_radius_px": st["summary"]["tip_radius_px"]}
        if mp.exists():
            m = RinkMap.load(mp)
            f = m.extra["shape"]["circles"]["fits"].get("center")
            if f is None:  # the player stands on the centre dot: fit the ring in the median frame around the map origin
                f = ring_fit(med, m.origin_px, 55, 95)
                rec["centre_circle_fit"] = f
            rec["centre_circle_radius_px"] = (f["radius_px"] if "radius_px" in f else round((f["horizontal_diameter_px"] + f["vertical_diameter_px"]) / 4, 3)) if f else None
            rec["m_per_px_map"] = m.m_per_px
        else:
            # The centre dot sits under the resting puck in the zoomed clip; search the ring around the blue line's dot.
            guess = _centre_dot_guess(med)
            ring = ring_fit(med, guess, 0.2 * med.shape[0], 0.45 * med.shape[0])
            rec["centre_circle_fit"] = ring
            rec["centre_circle_radius_px"] = ring["radius_px"] if ring else None
        out["clips"][name] = rec
    # Calibrate the body at the zoom that has a map, then scale the others.
    ref = [n for n, r in out["clips"].items() if "m_per_px_map" in r]
    if ref:
        rn = ref[0]
        r0 = out["clips"][rn]
        body_m = r0["body"]["radius_px"] * r0["m_per_px_map"]
        out["body_radius_m_placeholder"] = {"value": body_m, "from": rn}
        for n, r in out["clips"].items():
            r["m_per_px_disc"] = body_m / r["body"]["radius_px"]
            if r.get("centre_circle_radius_px"):
                r["m_per_px_circle"] = cc["radius_m"] / r["centre_circle_radius_px"]
                r["body_over_circle"] = r["body"]["radius_px"] / r["centre_circle_radius_px"]
            r["m_per_px_params_player_radius"] = params["player_radius"] / r["body"]["radius_px"]
            r["m_per_px"] = r.get("m_per_px_map", r["m_per_px_disc"])
            r["scale_source"] = ("map (ice-plane, placeholder rink_length 60 m)" if "m_per_px_map" in r else
                                 f"disc: grey body radius {body_m:.4f} m (calibrated in {rn} at the map scale) / {r['body']['radius_px']:.2f} px")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for n, r in out["clips"].items():
        print(f"[stick scale] {n}: body r {r['body']['radius_px']:.3f} px (rms {r['body']['fit_rms_px']}, {r['body']['edge_points_kept']}/{r['body']['edge_points']} pts); "
              f"centre circle r {r.get('centre_circle_radius_px')} px; body/circle {r.get('body_over_circle', float('nan')):.5f}; "
              f"m/px: map {r.get('m_per_px_map', float('nan')):.6f}, disc {r.get('m_per_px_disc', float('nan')):.6f}, circle {r.get('m_per_px_circle', float('nan')):.6f}, "
              f"params player_radius {r['m_per_px_params_player_radius']:.6f}")
    print(f"[stick scale] centre circle radius {cc['radius_m']:.4f} m (sd {cc['sd_m']:.4f}, {cc['n_clips']} normal-zoom maps); "
          f"body radius {out.get('body_radius_m_placeholder')}; params player_radius {params['player_radius']}")
    return out


def _centre_dot_guess(med: np.ndarray) -> tuple[float, float]:
    """The centre circle's centre: the gap in the cyan centre line where the dot/puck sits, else the image centre."""
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    cyan = (hsv[..., 0] > 85) & (hsv[..., 0] < 100) & (hsv[..., 1] > 120) & (hsv[..., 2] > 120)
    rows = cyan.sum(axis=1)
    v = int(np.argmax(rows))
    band = cyan[max(v - 3, 0) : v + 4].any(axis=0)
    xs = np.where(~band[50:-50])[0] + 50
    if len(xs) == 0:
        return med.shape[1] / 2, med.shape[0] / 2
    # the largest run of non-cyan along the line (the dot/puck interrupts it)
    runs, start = [], xs[0]
    for a, b in zip(xs, xs[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, xs[-1]))
    s, e = max((r for r in runs if r[1] - r[0] < 200), key=lambda r: r[1] - r[0], default=(xs[0], xs[0]))
    return (s + e) / 2, float(v)


def ring_fit(med: np.ndarray, guess, r_lo: float, r_hi: float) -> dict | None:
    """Circle fit to the grey centre ring: per ray, the subpixel brightness peak of low-saturation pixels in
    [r_lo, r_hi] (rays crossing the player glow, the blue line or the puck have no such peak and are skipped)."""
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV).astype(np.float32)
    val, sat = hsv[..., 2], hsv[..., 1]
    rs = np.arange(r_lo, r_hi, 0.25)
    pts = []
    for a in np.linspace(-math.pi, math.pi, 720, endpoint=False):
        xs = (guess[0] + rs * math.cos(a)).astype(np.float32).reshape(1, -1)
        ys = (guess[1] + rs * math.sin(a)).astype(np.float32).reshape(1, -1)
        V = cv2.remap(val, xs, ys, cv2.INTER_LINEAR, borderValue=0)[0]
        Sa = cv2.remap(sat, xs, ys, cv2.INTER_LINEAR, borderValue=255)[0]
        V = np.where(Sa < 45, V, 0)
        k = int(np.argmax(V))
        base = float(np.median(V[V > 0])) if (V > 0).any() else 0
        if V[k] - base < 12 or k < 4 or k > len(rs) - 5:
            continue
        w = np.clip(V[k - 8 : k + 9] - (base + 0.5 * (V[k] - base)), 0, None) if k >= 8 else None
        if w is None or w.sum() == 0:
            continue
        r = float((rs[k - 8 : k + 9] * w).sum() / w.sum())
        pts.append((guess[0] + r * math.cos(a), guess[1] + r * math.sin(a)))
    if len(pts) < 60:
        return None
    P = np.array(pts)
    q = least_squares(lambda z: np.hypot(P[:, 0] - z[0], P[:, 1] - z[1]) - z[2], [guess[0], guess[1], (r_lo + r_hi) / 2]).x
    keep = np.ones(len(P), bool)
    for _ in range(4):
        res = np.abs(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]) - q[2])
        keep = res < max(0.5, 3 * np.median(res))
        q = least_squares(lambda z: np.hypot(P[keep, 0] - z[0], P[keep, 1] - z[1]) - z[2], q).x
    res = np.hypot(P[keep, 0] - q[0], P[keep, 1] - q[1]) - q[2]
    return {"centre_px": [round(float(q[0]), 3), round(float(q[1]), 3)], "radius_px": round(float(q[2]), 3),
            "points_used": int(keep.sum()), "points_total": int(len(P)), "fit_rms_px": round(float(np.sqrt(np.mean(res ** 2))), 3)}
