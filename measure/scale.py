"""Ice-plane scale and the puck's ground position.

Two questions, both about perspective (the camera is a perspective camera looking straight down):

1. Is the board outline that sets the scale at ice level? The boards stand above the ice; if the outline
   we measure were the board TOP, parallax would push its edges outward by k_board * (distance from the
   principal point). A centered camera can't tell (uniform magnification), but a panned one can: the gap
   between a board edge and the painted faceoff dots would grow on the far side. `scale` measures, per clip,
   board lengths against ice-marking lengths (faceoff dot spacing, center circle diameter) and the edge
   gaps, pools k_board from the panned clips, and stores both scales in the map JSON. The ice-plane scale
   is m_per_px_board * (1 + k_board) if k_board is significant and positive, else the same as the board scale.

2. Where is the puck on the ice? Its blob is the top face plus the side wall the camera sees off-axis, so
   its centroid sits above the ice and is displaced outward from the principal point. From the blob-area
   model area = A0 + a1 * r: side band length = k_top * r with k_top = a1 / D (D = blob diameter on-axis), and
   the blob centroid is displaced by k_top / 2 * r. `reproject` maps each centroid to its ground point:
   g = pp + (c - pp) / (1 + k_top / 2) (in raw frame pixels; the camera shift is applied around it).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .geometry import RinkMap

PP = (960.0, 540.0)  # principal point: the image center (assumed)
EDGE_SIGMA_PX = 0.5  # board edges are medians of integer pixel edges


def board_vs_ice(m: dict) -> dict:
    b = m["board_px"]
    sh = m["extra"]["shape"]
    fo = [v for k, v in sh["circles"]["fits"].items() if k.startswith("faceoff") and v]
    us = sorted(f["center_px"][0] for f in fo)
    vs = sorted(f["center_px"][1] for f in fo)
    lcol, rcol, trow, brow = np.mean(us[:2]), np.mean(us[2:]), np.mean(vs[:2]), np.mean(vs[2:])
    cc = sh["circles"]["fits"]["center"]
    ccd = (cc["horizontal_diameter_px"] + cc["vertical_diameter_px"]) / 2
    L, W = b["bottom"] - b["top"], b["right"] - b["left"]
    return {
        "board_length_px": L,
        "board_width_px": W,
        "faceoff_spacing_v_px": round(float(brow - trow), 3),
        "faceoff_spacing_h_px": round(float(rcol - lcol), 3),
        "center_circle_diameter_px": round(float(ccd), 3),
        "board_length_over_faceoff_spacing": round(float(L / (brow - trow)), 5),
        "board_width_over_faceoff_spacing": round(float(W / (rcol - lcol)), 5),
        "board_length_over_center_circle": round(float(L / ccd), 5),
        "gaps_px": {"left": round(float(lcol - b["left"]), 3), "right": round(float(b["right"] - rcol), 3),
                    "top": round(float(trow - b["top"]), 3), "bottom": round(float(b["bottom"] - brow), 3)},
        "board_center_minus_pp_px": [round((b["left"] + b["right"]) / 2 - PP[0], 2), round((b["top"] + b["bottom"]) / 2 - PP[1], 2)],
    }


def pooled_k_board(infos: list[dict]) -> tuple[float, float]:
    """Weighted least squares of gap asymmetry = 2 k (board center - pp), over both axes and all clips."""
    num = den = 0.0
    for i in infos:
        g = i["gaps_px"]
        du, dv = i["board_center_minus_pp_px"]
        for asym, d in ((g["right"] - g["left"], du), (g["bottom"] - g["top"], dv)):
            num += asym * 2 * d
            den += (2 * d) ** 2
    if den == 0:
        return 0.0, float("inf")
    k = num / den
    sigma = math.sqrt(2) * EDGE_SIGMA_PX / math.sqrt(den)  # asym has two edges' quantization in it
    return k, sigma


def run_scale(results: Path, clips: list[str]) -> dict:
    maps = {c: json.loads((results / f"{c}.map.json").read_text()) for c in clips}
    infos = {c: board_vs_ice(m) for c, m in maps.items()}
    k, sk = pooled_k_board(list(infos.values()))
    significant = k > 2 * sk
    k_used = k if significant else 0.0
    ratios = [i["board_length_over_faceoff_spacing"] for i in infos.values()]
    print(f"[scale] board outline parallax k_board = {k:+.5f} +- {sk:.5f} (from edge-gap asymmetry; panned clip dominates) -> "
          f"{'SIGNIFICANT, used' if significant else 'not significant: the board outline is at ice level, k_board = 0 used'}")
    print(f"[scale] board length / faceoff spacing across clips: {min(ratios):.5f}..{max(ratios):.5f} (spread {100 * (max(ratios) - min(ratios)) / np.mean(ratios):.2f}%)")
    out = {}
    for c, m in maps.items():
        i = infos[c]
        L = i["board_length_px"]
        mpp_board = 60.0 / L  # placeholder rink_length
        mpp_ice = mpp_board * (1 + k_used)
        m["m_per_px"] = mpp_ice
        m["scale_source"] = (
            f"ICE-PLANE scale for puck motion: params.json rink_length = 60 m (PLACEHOLDER, unmeasured) / inner board length {L:.2f} px "
            f"x (1 + k_board {k_used:+.5f}); the board outline is at ice level within k_board = {k:+.5f} +- {sk:.5f}, so board and ice "
            f"scales agree. Puck positions are blob centroids reprojected to the ground (see track.json puck_parallax)."
        )
        m["extra"]["scales"] = {
            "m_per_px_board": mpp_board,
            "m_per_px_ice": mpp_ice,
            "k_board_measured": round(k, 6),
            "k_board_sigma": round(sk, 6),
            "k_board_used": k_used,
            "used_for_puck_motion": "m_per_px_ice",
            "board_vs_ice": i,
            "note": "absolute scale UNMEASURED either way: both assume rink_length = 60 m",
        }
        (results / f"{c}.map.json").write_text(json.dumps(m, indent=2) + "\n")
        out[c] = m["extra"]["scales"]
        print(f"[scale] {c}: board {mpp_board:.6f} m/px, ice {mpp_ice:.6f} m/px; L/faceoff {i['board_length_over_faceoff_spacing']:.5f}, "
              f"W/faceoff {i['board_width_over_faceoff_spacing']:.5f}, L/center-circle {i['board_length_over_center_circle']:.5f}; "
              f"gaps L {i['gaps_px']['left']:.2f} R {i['gaps_px']['right']:.2f} T {i['gaps_px']['top']:.2f} B {i['gaps_px']['bottom']:.2f}; "
              f"board center - pp {i['board_center_minus_pp_px']}")
    return out


MIN_AREA_GROWTH = 0.25  # apply the height correction only if the blob grows >= 25% of A0 across the radii seen


def puck_parallax(u_raw, v_raw, area, pp=PP) -> dict:
    from .segment import area_model

    try:
        a0, a1 = area_model(np.asarray(u_raw), np.asarray(v_raw), np.asarray(area, float), center=pp)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        a0, a1 = float(np.median(np.asarray(area)[np.asarray(area) > 0])), 0.0  # too few radii to fit: no correction
    if a1 < 0:
        a1 = 0.0
    D = 2 * math.sqrt(a0 / math.pi)
    r = np.hypot(np.asarray(u_raw) - pp[0], np.asarray(v_raw) - pp[1])
    growth = a1 * float(np.percentile(r, 95)) / a0
    applied = growth >= MIN_AREA_GROWTH
    k_top = a1 / D if applied else 0.0
    return {"area_px2_at_center": round(a0, 3), "area_per_px_from_center": round(a1, 6), "blob_diameter_px": round(D, 3),
            "area_growth_over_radii_seen": round(growth, 4), "applied": applied,
            "k_top": round(k_top, 6), "k_centroid": round(k_top / 2, 6), "principal_point": list(pp)}


def ground(u_ref, v_ref, su, sv, k_c, pp=PP):
    """Blob centroid (reference px, frame shift su/sv) -> ground point (reference px)."""
    ur, vr = np.asarray(u_ref) + su, np.asarray(v_ref) + sv
    ug = pp[0] + (ur - pp[0]) / (1 + k_c)
    vg = pp[1] + (vr - pp[1]) / (1 + k_c)
    return ug - su, vg - sv


def reproject(results: Path, clip: str) -> dict:
    """Recompute ground positions and meters for an existing track (no re-detection)."""
    from .track import write_csv

    m = RinkMap.load(results / f"{clip}.map.json")
    tp = results / f"{clip}.track.json"
    meta = json.loads(tp.read_text())
    R = meta["rows"]
    u, v = np.array(R["u_px"]), np.array(R["v_px"])
    su, sv = np.array(R["shift_u"]), np.array(R["shift_v"])
    area = np.array(R["area_px"], float)
    center = (m.frame_size[0] / 2, m.frame_size[1] / 2)  # principal point = image center
    pp = puck_parallax(u + su, v + sv, area, center)
    ug, vg = ground(u, v, su, sv, pp["k_centroid"], center)
    x, y = m.px_to_m(ug, vg)
    R["ug_px"] = [round(float(a), 3) for a in ug]
    R["vg_px"] = [round(float(a), 3) for a in vg]
    corr = np.hypot(ug - u, vg - v)
    pp["max_correction_px"] = round(float(corr.max()), 3)
    pp["median_correction_px"] = round(float(np.median(corr)), 3)
    meta["puck_parallax"] = pp
    meta["scale_source"] = m.scale_source
    meta["m_per_px"] = m.m_per_px
    meta["position_note"] = "CSV x/y = ground points (ug_px, vg_px): blob centroids reprojected for the puck's height (puck_parallax)"
    rows = [{"t": f / m.fps, "x": float(a), "y": float(b)} for f, a, b in zip(R["frame"], x, y)]
    write_csv(rows, results / f"{clip}.traj.csv")
    tp.write_text(json.dumps(meta, separators=(",", ":")) + "\n")
    print(f"[reproject] {clip}: puck k_top {pp['k_top']:.5f} (centroid {pp['k_centroid']:.5f}); ground correction median {pp['median_correction_px']:.2f} px, "
          f"max {pp['max_correction_px']:.2f} px; wrote {clip}.traj.csv")
    return pp
