"""Player body radius against the puck radius in the same frames (`python -m measure.body_puck CLIP.mkv`).

The 2026-11 art draws the player as a grey body (the old clips showed a pink disc, which set params player_radius).
In hittingpuck (normal zoom, grey body, the puck resting at the forehand between hits) both are measured in the
same frames, the same way as before:
- body: half-max edge of the saturation step from the grey body (S = 0) to the blue glow, rays from the centre,
  circle fit with outlier rejection (stick_scale.body_edge_radius);
- puck: half-max edge of the greenness G − (R + B)/2 radial profile around the resting puck (as `measure sizes`).
Each resting episode (the puck still for ≥ 20 frames) gives one sample: the per-pixel median of its frames, then both
radii. ratio = body / puck, both AS SEEN (no height correction for either). The sim's player_radius is then set to
ratio × the sim's measured puck_radius, so the two keep the on-screen ratio. Meters (for reference) use the clip map's
ice-plane m_per_px (real meters since the rescale; scale.json factor).
The clip is read in place (pass its path); nothing from it is copied into the repo.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .geometry import REPO
from .stick_scale import body_edge_radius
from .video import iter_frames

RESULTS = REPO / "measure" / "results"
OUT = RESULTS / "body_puck.json"


def resting_episodes(track: dict, min_len: int = 20, max_step_px: float = 0.05) -> list[list[int]]:
    rows = track["rows"]
    f = np.array(rows["frame"])
    u = np.array(rows["u_px"]) + np.array(rows["shift_u"])
    v = np.array(rows["v_px"]) + np.array(rows["shift_v"])
    still = np.r_[False, np.hypot(np.diff(u), np.diff(v)) < max_step_px] & np.r_[False, np.diff(f) == 1]
    eps, cur = [], []
    for i, s in enumerate(still):
        if s:
            cur.append(i)
        else:
            if len(cur) >= min_len:
                eps.append(cur)
            cur = []
    if len(cur) >= min_len:
        eps.append(cur)
    return eps


def puck_half_max(F: np.ndarray, uc: float, vc: float) -> float:
    g = F[..., 1].astype(float) - (F[..., 0].astype(float) + F[..., 2].astype(float)) / 2
    ys, xs = np.mgrid[0 : F.shape[0], 0 : F.shape[1]]
    rr = np.hypot(xs - uc, ys - vc)
    core = float(g[rr < 2].mean())
    ring = (rr > 8) & (rr < 10)
    bg = float(np.median(g[ring]))
    rs = np.arange(0, 9, 0.05)
    prof = np.array([np.nanmean(g[(rr >= a) & (rr < a + 0.5)]) for a in rs])
    return float(rs[np.argmax(prof < bg + 0.5 * (core - bg))] + 0.25)


def measure(clip: Path, results: Path = RESULTS, max_frames: int = 60) -> dict:
    name = clip.stem
    track = json.loads((results / f"{name}.track.json").read_text())
    mp = json.loads((results / f"{name}.map.json").read_text())
    rows = track["rows"]
    eps = resting_episodes(track)
    want = {}
    for k, ep in enumerate(eps):
        for i in ep[:max_frames]:
            want[int(rows["frame"][i])] = k
    stacks: dict[int, list[np.ndarray]] = {k: [] for k in range(len(eps))}
    last = max(want) if want else -1
    for i, fr in iter_frames(clip):
        if i in want:
            stacks[want[i]].append(fr)
        if i > last:
            break
    samples = []
    for k, ep in enumerate(eps):
        if len(stacks[k]) < 10:
            continue
        med = np.median(np.stack(stacks[k]), axis=0).astype(np.uint8)
        i0 = ep[0]
        uc = float(rows["u_px"][i0] + rows["shift_u"][i0])
        vc = float(rows["v_px"][i0] + rows["shift_v"][i0])
        body = body_edge_radius(med, (959.5, 539.5), 8.8)
        puck = puck_half_max(med, uc, vc)
        d = math.hypot(uc - body["centre_px"][0], vc - body["centre_px"][1])
        samples.append({"episode": k, "frames": len(stacks[k]), "first_frame": int(rows["frame"][i0]), "puck_centre_px": [round(uc, 2), round(vc, 2)],
                        "puck_dist_px": round(d, 2), "body_radius_px": body["radius_px"], "body_fit_rms_px": body["fit_rms_px"],
                        "body_edge_points_kept": body["edge_points_kept"], "puck_radius_px": round(puck, 3), "ratio": body["radius_px"] / puck})
    # Keep episodes whose puck rests within 200 px of the image centre (small parallax, clean ice around it). The
    # tracker also reports "resting" episodes ~425 px away by the net, where the greenness profile fails (0.25–0.75 px).
    for s in samples:
        s["used"] = s["puck_dist_px"] < 200 and s["puck_radius_px"] > 2.0
        if not s["used"]:
            s["excluded"] = "puck > 200 px from the image centre (by the net) or no puck edge"
    used = [s for s in samples if s["used"]]
    r = np.array([s["ratio"] for s in used])
    bp = np.array([s["body_radius_px"] for s in used])
    pp = np.array([s["puck_radius_px"] for s in used])
    mpp = mp["m_per_px"]
    return {"clip": name, "episodes_found": len(eps), "samples": samples, "n": len(used), "n_excluded": len(samples) - len(used),
            "ratio_median": float(np.median(r)), "ratio_mean": float(r.mean()), "ratio_sd": float(r.std(ddof=1)) if len(r) > 1 else None,
            "body_radius_px_median": float(np.median(bp)), "body_radius_px_sd": float(bp.std(ddof=1)) if len(bp) > 1 else None,
            "puck_radius_px_median": float(np.median(pp)), "puck_radius_px_sd": float(pp.std(ddof=1)) if len(pp) > 1 else None,
            "m_per_px": mpp, "scale_source": mp.get("scale_source", ""),
            "body_radius_m_as_seen": float(np.median(bp) * mpp), "puck_radius_m_as_seen": float(np.median(pp) * mpp)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m measure.body_puck")
    ap.add_argument("clip", type=Path, help="path to hittingpuck.mkv (read in place)")
    ap.add_argument("--puck-radius-sim", type=float, default=None, help="sim puck_radius (default: params.json)")
    a = ap.parse_args(argv)
    res = measure(a.clip)
    from .geometry import load_params

    pr = a.puck_radius_sim or load_params()["puck_radius"]
    res["sim_puck_radius_m"] = pr
    res["player_radius_m"] = res["ratio_median"] * pr
    for s in res["samples"]:
        print(f"  episode {s['episode']:2d} (frame {s['first_frame']:5d}, {s['frames']} frames): body {s['body_radius_px']:.3f} px (rms {s['body_fit_rms_px']}, {s['body_edge_points_kept']} pts), puck {s['puck_radius_px']:.3f} px at {s['puck_dist_px']:.1f} px from the body centre, ratio {s['ratio']:.3f}{'' if s['used'] else '  EXCLUDED: ' + s['excluded']}")
    print(f"ratio body/puck: median {res['ratio_median']:.3f}, mean {res['ratio_mean']:.3f}, sd {res['ratio_sd']:.3f}, range {min(x['ratio'] for x in res['samples'] if x['used']):.3f}–{max(x['ratio'] for x in res['samples'] if x['used']):.3f}, n {res['n']} (of {res['episodes_found']} resting episodes; {res['n_excluded']} excluded)")
    print(f"body {res['body_radius_px_median']:.3f} px (sd {res['body_radius_px_sd']:.3f}), puck {res['puck_radius_px_median']:.3f} px (sd {res['puck_radius_px_sd']:.3f})")
    print(f"as seen at {res['m_per_px']:.5f} m/px: body {res['body_radius_m_as_seen']:.4f} m, puck {res['puck_radius_m_as_seen']:.4f} m; sim puck_radius {pr} → player_radius {res['player_radius_m']:.4f} m")
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
