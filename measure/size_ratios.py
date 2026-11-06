"""Player, puck and rink size ratios in the same camera, with the height correction explicit
(`python -m measure.size_ratios --clips DIR`). Writes measure/results/size_ratios.json; clips are read in place.

Every clip gives the ice scale (px per rink length, so nothing depends on the absolute scale) from markings on the ice:
- hittingpuck: the map's inner board length;
- rotatinganddizzy (no boards in view): the centre circle, times the board length / centre circle ratio of the maps.

Sizes:
- puck: a blurred-disc fit (radius, blur, contrast, background) to the greenness G - (R + B)/2 of the per-pixel median of
  a resting stretch. The fit keeps the radius apart from the blur; a half-max read of a 3.5 px disc does not.
- body: the half-max saturation edge of the grey body (stick_scale.body_edge_radius), per-pixel median of frames.

Height: a thing at height h under a camera at height H is magnified by 1 / (1 - h/H). The zoomed clip is the same scene
from a lower camera (ice scale ratio q = H_normal / H_zoom). With px sizes a_n, a_z and A = a_z / a_n:
    x = h / H_zoom = (A - q) / (A - 1),   ground size in zoomed px = a_z (1 - x).
This is the dolly-camera model stick_shape.py uses for the blade tip.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .body_puck import resting_episodes
from .geometry import REPO, load_params
from .stick_scale import body_edge_radius
from .video import iter_frames

RESULTS = REPO / "measure" / "results"
OUT = RESULTS / "size_ratios.json"
SS = 4  # supersampling of the disc model
BOARD_OVER_CIRCLE_DIAMETER = 6.365  # the maps: 6.361-6.368


def greenness(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    return f[..., 1] - (f[..., 0] + f[..., 2]) / 2


def disc_model(shape, u, v, r, sig, c, b):
    h, w = shape
    ys, xs = np.mgrid[0 : h * SS, 0 : w * SS]
    d = np.hypot((xs + 0.5) / SS - 0.5 - u, (ys + 0.5) / SS - 0.5 - v)
    m = np.clip((r - d) * SS + 0.5, 0, 1).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), max(abs(sig), 0.05) * SS)
    return b + c * cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)


def fit_disc(g: np.ndarray, u: float, v: float, r0: float, contrast: float | None = None) -> dict:
    """Blurred disc fit; with `contrast` given it is held fixed (small discs barely separate it from the radius)."""

    def unpack(q):
        return (q[0], q[1], q[2], q[3], contrast if contrast else q[4], q[-1])

    q0 = [u, v, r0, 0.8] + ([] if contrast else [float(g.max() - np.median(g))]) + [float(np.median(g))]
    res = least_squares(lambda q: (disc_model(g.shape, *unpack(q)) - g).ravel(), q0, diff_step=0.02)
    uu, vv, r, sig, c, b = unpack(res.x)
    return {"u": float(uu), "v": float(vv), "r": float(r), "sigma": float(abs(sig)), "contrast": float(c), "bg": float(b),
            "rms": float(np.sqrt(np.mean(res.fun**2)))}


def puck_zoomed(clip: Path) -> dict:
    """Puck at the centre dot of the zoomed clip: median of frames spread over the clip; found by its round green blob."""
    fr = [f for _, f in iter_frames(clip, every=60)]
    med = np.median(np.stack(fr), axis=0).astype(np.uint8)
    g = greenness(med)
    n, _, st, cen = cv2.connectedComponentsWithStats((g > 40).astype(np.uint8))
    # HUD bars are long and sit at the bottom edge.
    best = max((i for i in range(1, n) if st[i, 4] > 200 and 0.7 < st[i, 2] / st[i, 3] < 1.4 and cen[i][1] < 900), key=lambda i: st[i, 4])
    u, v = cen[best]
    x0, y0 = int(u) - 34, int(v) - 34
    res = fit_disc(g[y0 : y0 + 68, x0 : x0 + 68], u - x0, v - y0, 17.0)
    res["bbox_px"] = [int(st[best, 2]), int(st[best, 3])]
    res["frames"] = len(fr)
    return res


def normal_zoom(clip: Path, contrast: float) -> list[dict]:
    """Per resting episode of hittingpuck: puck disc fit (contrast fixed from the zoomed clip, and free) and the body."""
    track = json.loads((RESULTS / f"{clip.stem}.track.json").read_text())
    used = {s["episode"]: s for s in json.loads((RESULTS / "body_puck.json").read_text())["samples"] if s["used"]}
    rows = track["rows"]
    eps = resting_episodes(track)
    want = {int(rows["frame"][i]): k for k, ep in enumerate(eps) if k in used for i in ep[:60]}
    stacks: dict[int, list[np.ndarray]] = {}
    last = max(want)
    for i, f in iter_frames(clip):
        if i in want:
            stacks.setdefault(want[i], []).append(f)
        if i > last:
            break
    out = []
    for k, fs in sorted(stacks.items()):
        if len(fs) < 10:
            continue
        med = np.median(np.stack(fs), axis=0).astype(np.uint8)
        u, v = used[k]["puck_centre_px"]
        x0, y0 = int(u) - 11, int(v) - 11
        g = greenness(med[y0 : y0 + 22, x0 : x0 + 22])
        fixed = fit_disc(g, u - x0, v - y0, 4.0, contrast)
        free = fit_disc(g, u - x0, v - y0, 4.0)
        body = body_edge_radius(med, (959.5, 539.5), 8.8)
        out.append({"episode": k, "puck_r_fixed_contrast": fixed["r"], "puck_r_free": free["r"], "free_contrast": free["contrast"],
                    "sigma": fixed["sigma"], "body_r": body["radius_px"]})
    return out


def stats(v) -> dict:
    a = np.array(v, float)
    return {"median": float(np.median(a)), "sd": float(a.std(ddof=1)), "n": len(a), "se": float(a.std(ddof=1) / math.sqrt(len(a)))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m measure.size_ratios")
    ap.add_argument("--clips", type=Path, default=REPO / "clips")
    a = ap.parse_args(argv)
    p = load_params()
    mp = json.loads((RESULTS / "hittingpuck.map.json").read_text())
    sc = json.loads((RESULTS / "stick_scale.json").read_text())
    L_n, W_n = mp["board_length_px"], mp["board_width_px"]
    circ_z = sc["clips"]["rotatinganddizzy"]["centre_circle_radius_px"]
    L_z = BOARD_OVER_CIRCLE_DIAMETER * 2 * circ_z
    q = L_z / L_n
    pz = puck_zoomed(a.clips / "rotatinganddizzy.mkv")
    eps = normal_zoom(a.clips / "hittingpuck.mkv", pz["contrast"])
    body_z = sc["clips"]["rotatinganddizzy"]["body"]["radius_px"]
    bn = stats([e["body_r"] for e in eps])
    pf = stats([e["puck_r_fixed_contrast"] for e in eps])
    pr = stats([e["puck_r_free"] for e in eps])

    # Monte Carlo through the height model. Zoomed px +-0.5, rink px +-1 (normal) and 0.2% (zoomed, via the circle).
    rng = np.random.default_rng(1)
    N = 20000
    Lz = L_z * (1 + 0.002 * rng.standard_normal(N))
    qq = Lz / (L_n + rng.standard_normal(N))

    def ground(a_n, sd_n, a_z):
        an = a_n + sd_n * rng.standard_normal(N)
        az = a_z + 0.5 * rng.standard_normal(N)
        A = az / an
        x = (A - qq) / (A - 1)
        return az * (1 - x) / Lz, x

    b_l, b_x = ground(bn["median"], max(bn["sd"], 0.2), body_z)
    p_l, p_x = ground(pf["median"], max(pf["sd"], 0.1), pz["r"])
    out = {
        "note": "Ratios are scale-free: radius / inner board length, both in px of one camera. 'as seen' = no height correction.",
        "zoom": {"rink_length_px_normal": L_n, "rink_width_px_normal": W_n, "rink_length_px_zoomed": L_z, "q": q, "centre_circle_r_px_zoomed": circ_z},
        "puck_zoomed": pz, "body_px_zoomed": body_z, "body_px_normal": bn, "puck_px_normal_fixed_contrast": pf, "puck_px_normal_free": pr, "episodes": eps,
        "height": {"body_x": float(np.median(b_x)), "body_x_sd": float(b_x.std()), "puck_x": float(np.median(p_x)), "puck_x_sd": float(p_x.std())},
        "ratios": {
            "body_over_rink_as_seen_normal": bn["median"] / L_n, "body_over_rink_as_seen_zoomed": body_z / L_z,
            "body_over_rink_ground": float(np.median(b_l)), "body_over_rink_ground_sd": float(b_l.std()),
            "puck_over_rink_as_seen_normal": pf["median"] / L_n, "puck_over_rink_as_seen_zoomed": pz["r"] / L_z,
            "puck_over_rink_ground": float(np.median(p_l)), "puck_over_rink_ground_sd": float(p_l.std()),
            "body_over_puck_as_seen_normal": bn["median"] / pf["median"], "body_over_puck_as_seen_zoomed": body_z / pz["r"],
            "body_over_puck_ground": float(np.median(b_l / p_l)), "body_over_puck_ground_sd": float((b_l / p_l).std()),
            "rink_length_over_width": L_n / W_n,
        },
        "sim": {"player_radius": p["player_radius"], "puck_radius": p["puck_radius"], "rink_length": p["rink_length"],
                "body_over_rink": p["player_radius"] / p["rink_length"], "puck_over_rink": p["puck_radius"] / p["rink_length"],
                "body_over_puck": p["player_radius"] / p["puck_radius"]},
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("zoom", "puck_zoomed", "body_px_normal", "puck_px_normal_fixed_contrast", "puck_px_normal_free", "height", "ratios", "sim")}, indent=1))


if __name__ == "__main__":
    main()
