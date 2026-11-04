"""Stick shape (measured-shape values at the placeholder scale): length and blade thickness.

- stick_length = distance pivot -> visible blade tip. Two measures: (a) the normal-zoom clip (hittingpuck): tip
  radius px x its map's ice-plane m_per_px; (b) the zoomed clip (rotatinganddizzy): tip radius / body radius (both
  sharp there) x the body radius in meters calibrated at the normal zoom (stick_scale.json). The player model
  (body + stick) is drawn 1.34x larger relative to the ice in the zoomed clip, so (b) uses the disc scale.
- stick_thickness (sim: collision HALF-thickness) = half the blade's full width at half max, measured in the zoomed
  clip on frames de-rotated by the tracked angle and averaged, between 1.15 and 1.5 body radii.
- stick_inner (where the blade starts) is reported, not written: the blade's white begins at the body edge.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .geometry import REPO
from .video import iter_frames

RESULTS = REPO / "measure" / "results"
OUT = RESULTS / "stick_shape.json"


def blade_profile(clip: Path, every: int = 7) -> dict:
    name = Path(clip).stem
    S = json.loads((RESULTS / f"{name}.stick.json").read_text())
    R = S["rows"]
    by = {f: k for k, f in enumerate(R["frame"])}
    rb = S["summary"]["body_radius_px"]
    half = int(2.0 * rb)
    acc, n = None, 0
    blade_root = []
    for i, f in iter_frames(clip, every=every):
        if i not in by or R["conf"][by[i]] < 0.8:
            continue
        k = by[i]
        cu, cv_ = R["pivot_u"][k], R["pivot_v"][k]
        a = R["angle"][k]
        phi = math.degrees(-a - math.pi / 2)  # screen angle of the blade
        M = cv2.getRotationMatrix2D((cu, cv_), phi, 1.0)  # rotate so the blade points along +u
        M[0, 2] += half - cu
        M[1, 2] += half - cv_
        rot = cv2.warpAffine(f, M, (2 * half, 2 * half), flags=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(rot, cv2.COLOR_BGR2HSV)
        white = ((hsv[..., 2] > 185) & (hsv[..., 1] < 70)).astype(np.float32)
        acc = white if acc is None else acc + white
        n += 1
    W = acc / n
    cols = np.arange(2 * half) - half  # u offset from pivot
    sel = (cols > 1.15 * rb) & (cols < 1.5 * rb)
    prof = W[:, sel].mean(axis=1)  # across the blade (v)
    pk = float(prof.max())
    above = np.where(prof >= 0.5 * pk)[0]
    fwhm = float(above.max() - above.min() + 1) if len(above) else float("nan")
    centre_off = float((np.arange(len(prof)) * prof).sum() / prof.sum() - half)
    along = W[half - 3 : half + 4, :].max(axis=0)
    lit = np.where(along > 0.3)[0]
    lit = lit[lit - half > 0.8 * rb]  # the body's own white markings sit inside it
    root = float(lit.min() - half) if len(lit) else float("nan")
    tip = float(lit.max() - half) if len(lit) else float("nan")
    return {"frames": n, "body_r_px": rb, "blade_fwhm_px": fwhm, "blade_peak_occupancy": round(pk, 3), "blade_axis_offset_px": round(centre_off, 2),
            "white_root_px": root, "white_tip_px": tip}


def tip_half_max(clip: Path, every: int = 3, min_conf: float = 0.6) -> dict:
    """Blur-robust tip radius: frames de-rotated by the tracked angle and averaged; the blade direction is the angular
    peak of whiteness outside the body (the tracked angle may carry a constant offset at low resolution); along it, the
    tip is the outer half-max point of whiteness (V above the ice, low saturation) between the blade plateau and the ice."""
    name = Path(clip).stem
    S = json.loads((RESULTS / f"{name}.stick.json").read_text())
    R = S["rows"]
    by = {f: k for k, f in enumerate(R["frame"])}
    rb = S["summary"]["body_radius_px"]
    Z = max(1, int(round(40 / rb)))  # upsample small players so the profile has sub-pixel steps
    half = int(2.6 * rb * Z)
    acc, n = None, 0
    for i, f in iter_frames(clip, every=every):
        if i not in by or R["conf"][by[i]] < min_conf:
            continue
        k = by[i]
        cu, cv_ = R["pivot_u"][k], R["pivot_v"][k]
        phi = math.degrees(-R["angle"][k] - math.pi / 2)
        M = cv2.getRotationMatrix2D((cu, cv_), phi, Z)
        M[0, 2] += half - cu
        M[1, 2] += half - cv_
        rot = cv2.warpAffine(f, M, (2 * half, 2 * half), flags=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(rot, cv2.COLOR_BGR2HSV).astype(np.float32)
        wh = np.clip(hsv[..., 2] - 70, 0, None) * (hsv[..., 1] < 90)
        acc = wh if acc is None else acc + wh
        n += 1
    W = acc / n
    rbz = rb * Z
    ang = np.radians(np.arange(-60, 60.01, 0.5))
    best = None
    for a in ang:  # blade direction: max mean whiteness on the ray between 1.15 and 1.5 body radii
        rr = np.arange(1.15 * rbz, 1.5 * rbz, 0.5)
        xs = (half + rr * math.cos(a)).astype(np.float32).reshape(1, -1)
        ys = (half + rr * math.sin(a)).astype(np.float32).reshape(1, -1)
        val = float(cv2.remap(W.astype(np.float32), xs, ys, cv2.INTER_LINEAR)[0].mean())
        if best is None or val > best[0]:
            best = (val, a)
    a = best[1]
    rr = np.arange(1.0 * rbz, 2.5 * rbz, 0.25)
    xs = (half + rr * math.cos(a)).astype(np.float32).reshape(1, -1)
    ys = (half + rr * math.sin(a)).astype(np.float32).reshape(1, -1)
    # max across a narrow band perpendicular to the ray (the blade is curved / offset by a few px)
    prof = np.max([cv2.remap(W.astype(np.float32), (xs - d * math.sin(a)).astype(np.float32), (ys + d * math.cos(a)).astype(np.float32), cv2.INTER_LINEAR)[0]
                   for d in np.arange(-0.15 * rbz, 0.151 * rbz, 0.5)], axis=0)
    plateau = float(np.percentile(prof[(rr > 1.15 * rbz) & (rr < 1.45 * rbz)], 75))
    outer = float(np.median(prof[rr > 2.2 * rbz]))
    halfv = outer + 0.5 * (plateau - outer)
    k = int(np.where(prof >= halfv)[0].max())
    r_tip = rr[k] + (prof[k] - halfv) / max(prof[k] - prof[k + 1], 1e-9) * (rr[k + 1] - rr[k])
    return {"frames": n, "body_r_px": rb, "tip_half_max_px": float(r_tip / Z), "tip_over_body": float(r_tip / rbz),
            "blade_dir_offset_deg": round(math.degrees(a), 1), "upsample": Z}


def run_shape(zoom_clip: Path = REPO / "clips" / "rotatinganddizzy.mkv") -> dict:
    sc = json.loads((RESULTS / "stick_scale.json").read_text())
    hp = sc["clips"]["hittingpuck"]
    rd = sc["clips"]["rotatinganddizzy"]
    body_m = sc["body_radius_m_placeholder"]["value"]
    st_hp = json.loads((RESULTS / "hittingpuck.stick.json").read_text())["summary"]
    st_rd = json.loads((RESULTS / "rotatinganddizzy.stick.json").read_text())["summary"]
    L_a = st_hp["tip_radius_px"] * hp["m_per_px_map"]
    L_b = st_rd["tip_radius_px"] / rd["body"]["radius_px"] * body_m
    # Half-max tips at both zooms, and the dolly-camera model: apparent radius = r0 / m_per_px_ice / (1 - h/H).
    # Ice scale ratio between the zooms q = H_normal / H_zoom; x = h_tip / H_zoom solves
    # tip_zoom / tip_normal = q (1 - x/q) / (1 - x); then r0 = tip_zoom * m_per_px_ice_zoom * (1 - x).
    th = tip_half_max(REPO / "clips" / "hittingpuck.mkv")
    tz = tip_half_max(zoom_clip)
    q = hp["m_per_px_map"] / rd["m_per_px_circle"]
    ratio = tz["tip_half_max_px"] / th["tip_half_max_px"]
    x = (ratio - q) / (ratio - 1)  # from ratio (1 - x) = q - x
    r0 = tz["tip_half_max_px"] * rd["m_per_px_circle"] * (1 - x)
    # Same model for the body edge (a check on the camera model): body top height fraction.
    rb_ratio = rd["body"]["radius_px"] / hp["body"]["radius_px"]
    xb = (rb_ratio - q) / (rb_ratio - 1)
    body_ground = rd["body"]["radius_px"] * rd["m_per_px_circle"] * (1 - xb)
    bl = blade_profile(zoom_clip)
    mpp_disc = rd["m_per_px_disc"]
    out = {
        "stick_length_m": round(float(r0), 4),
        "stick_length_method": "half-max blade tip at both zooms + dolly-camera model (tip height solved): ground-plane tip radius",
        "tip_half_max": {"normal_zoom": th, "zoomed": tz, "ice_scale_ratio_q": q, "tip_ratio": ratio, "tip_height_over_zoomed_camera_height": x,
                         "tip_radius_m_normal_zoom_at_map_scale": th["tip_half_max_px"] * hp["m_per_px_map"],
                         "tip_radius_m_zoomed_at_ice_scale": tz["tip_half_max_px"] * rd["m_per_px_circle"]},
        "body_camera_model": {"radius_ratio": rb_ratio, "top_height_over_zoomed_camera_height": xb, "ground_radius_m": body_ground},
        "stick_length_candidates_m": {"template_tip_normal_zoom_x_map_scale (under-reads: the 1.4 px blade fades)": round(L_a, 4),
                                      "template_tip_zoomed_over_body_x_body_m": round(L_b, 4)},
        "stick_thickness_m": round(bl["blade_fwhm_px"] / 2 * rd["m_per_px_circle"] * (1 - x), 4),
        "stick_thickness_method": "half the blade FWHM in the zoomed clip, at the ice scale corrected for the blade height (same camera model as the tip)",
        "blade": bl,
        "blade_fwhm_m": round(bl["blade_fwhm_px"] * rd["m_per_px_circle"] * (1 - x), 4),
        "stick_inner_m_blade_root": round(bl["white_root_px"] * mpp_disc, 4),
        "tip_from_white_px_zoomed": bl["white_tip_px"],
        "m_per_px_zoomed_disc": mpp_disc,
        "m_per_px_zoomed_ice": rd.get("m_per_px_circle"),
        "body_radius_m": body_m,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[stick shape] half-max tip: normal zoom {th['tip_half_max_px']:.2f} px (= {th['tip_half_max_px'] * hp['m_per_px_map']:.3f} m at the map scale), "
          f"zoomed {tz['tip_half_max_px']:.2f} px (= {tz['tip_half_max_px'] * rd['m_per_px_circle']:.3f} m at the ice scale); ratio {ratio:.3f} vs ice-scale ratio {q:.3f} "
          f"-> tip height {x:.3f} of the zoomed camera height, ground-plane tip radius {r0:.4f} m. Body: ratio {rb_ratio:.3f} -> top height {xb:.3f}, "
          f"ground radius {body_ground:.4f} m")
    print(f"[stick shape] stick_length {out['stick_length_m']} m (template tips: normal {L_a:.4f}, zoomed-vs-body {L_b:.4f}); blade FWHM {bl['blade_fwhm_px']:.1f} px "
          f"= {out['blade_fwhm_m']} m -> half-thickness {out['stick_thickness_m']} m; blade white from {bl['white_root_px']:.0f} to "
          f"{bl['white_tip_px']:.0f} px (= {out['stick_inner_m_blade_root']} m .. {bl['white_tip_px'] * mpp_disc:.3f} m at the disc scale); "
          f"blade axis offset {bl['blade_axis_offset_px']} px from the radial line; {bl['frames']} frames")
    return out
