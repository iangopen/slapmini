"""Player tracker: the pink disc (player body seen from above) -> NAME.player.csv + NAME.player.json.

    python -m measure.player track clips/NAME.mkv [...]      # needs measure/results/NAME.map.json, or --ref-clip
    python -m measure.player qa clips/NAME.mkv [...]         # overlay PNGs in measure/results/qa/
    python -m measure.player estimate                        # motion events, estimates, params.player.json, sim check

What the footage shows (see CLAUDE.md "Player (session s6)"): the game camera FOLLOWS the player. The disc stays
within ~2 px of the image center in every clip, and the player's motion shows up as camera motion. So the
player's position in reference (ice) pixels = disc centroid in the raw frame - the frame's camera shift, with
the shift measured on the painted faceoff dots exactly as the puck tracker does (measure/camera.py). The
landmark search is centred on the previous frame's shift, so clips with large camera travel (the player
skating) can be followed too; a clip without its own map is registered against another clip's reference frame
(--ref-clip, same game camera, same zoom).

Detection: pink pixels (HSV), closed, one blob per player; position = centre of an ellipse fitted to the blob's
convex hull (the stick crossing the disc cuts notches into the blob; the hull fills them). The blue glow seen
under the player in the fixed-camera clips is a marker painted on the ice at the spawn point (it stays behind
when the player skates away), so it is not used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from .camera import HALF, MIN_SCORE, Camera, Shift, landmark_points
from .geometry import CONVENTION, REPO, RinkMap
from .video import iter_frames, median_frame, open_clip

RESULTS = REPO / "measure" / "results"
PINK_H, PINK_S, PINK_V = (135, 172), 60, 140  # OpenCV hue 0..180; the disc is ~152-160 hue, sat 100-130
AREA_MIN, AREA_MAX = 50, 1200  # px; the real disc is ~140-175 px at 1080p
ROUND_MIN = 0.55  # ellipse minor/major axis
GATE_PX = 15.0  # max disc travel per frame (the fastest real run is 3.2 px/frame) ...
GATE_GROW = 4.0  # ... plus this per missed frame
MAX_INTERP_GAP = 6
MAX_INTERP_SPEED = 4.0  # px/frame; a faster jump across a gap is a teleport, never interpolated
SEARCH_FOLLOW = 32  # landmark search around the predicted shift (px)
SEARCH_FIRST = 160  # first frame: wide search (a clip registered against another clip's reference)
DUP_STEP = 0.3  # a frame whose position repeats exactly after a step > this is a duplicated capture frame
K_PUCK_MAX = 0.0106  # the puck's measured k_top (CLAUDE.md); used only to bound the player's parallax


def pink_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= PINK_H[0]) & (h <= PINK_H[1]) & (s >= PINK_S) & (v >= PINK_V)


KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def disc_candidates(frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> list[dict]:
    """Pink blobs in raw-frame pixels: u, v (hull-ellipse centre), area (pink px), roundness, fill."""
    x0, y0, x1, y1 = roi if roi else (0, 0, frame.shape[1], frame.shape[0])
    sub = frame[y0:y1, x0:x1]
    if sub.size == 0:
        return []
    pk = pink_mask(cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)).astype(np.uint8)
    if not pk.any():
        return []
    closed = cv2.morphologyEx(pk, cv2.MORPH_CLOSE, KERNEL)
    n, lab, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < AREA_MIN or a > AREA_MAX:
            continue
        comp = (lab[y : y + h, x : x + w] == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cs, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        if len(hull) < 5:
            continue
        (cu, cv_), (d1, d2), _ = cv2.fitEllipse(hull)
        rnd = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0
        hull_area = cv2.contourArea(hull)
        area = int((comp & pk[y : y + h, x : x + w]).sum())
        out.append({"u": cu + x + x0, "v": cv_ + y + y0, "area": area, "roundness": float(rnd),
                    "fill": float(area / hull_area) if hull_area > 0 else 0.0, "diam": float((d1 + d2) / 2)})
    return [c for c in out if c["roundness"] >= ROUND_MIN]


class FollowCamera(Camera):
    """Camera (faceoff-dot landmarks) whose search window follows the predicted shift."""

    def measure_at(self, frame_bgr: np.ndarray, pdu: float, pdv: float, search: int) -> Shift:
        found = []
        for k, (cu, cv_) in enumerate(self.points):
            cx, cy = int(round(cu + pdu)), int(round(cv_ + pdv))
            x0, y0 = max(cx - HALF - search, 0), max(cy - HALF - search, 0)
            x1, y1 = min(cx + HALF + search + 1, self.W), min(cy + HALF + search + 1, self.H)
            if x1 - x0 < 2 * HALF + 3 or y1 - y0 < 2 * HALF + 3:
                continue
            win = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            du, dv, sc = self._match(win, x0, y0, k)
            if sc >= MIN_SCORE:
                found.append((du, dv))
        if not found:
            return Shift(0.0, 0.0, float("inf"), 0)
        du = float(np.median([s[0] for s in found]))
        dv = float(np.median([s[1] for s in found]))
        return Shift(du, dv, max(math.hypot(s[0] - du, s[1] - dv) for s in found), len(found))


def load_reference(clip: Path, results: Path, ref_clip: str | None):
    """(RinkMap, reference bg frame, reference name). The clip's own map if it exists, else --ref-clip's."""
    name = Path(clip).stem
    own = results / f"{name}.map.json"
    src = name if own.exists() else ref_clip
    if src is None:
        raise FileNotFoundError(f"{own} missing; run `python -m measure map {clip}` or pass --ref-clip")
    m = RinkMap.load(results / f"{src}.map.json")
    clip_dir = Path(clip).parent
    bg_path = clip_dir / m.extra.get("background_png", f"{src}.bg.png")
    if bg_path.exists():
        bg = cv2.imread(str(bg_path))
    else:
        src_clip = clip_dir / f"{src}.mkv"
        bg = median_frame(src_clip if src_clip.exists() else clip)
        cv2.imwrite(str(bg_path), bg)
    return m, bg, src


def rest_noise(u: np.ndarray, v: np.ndarray, win: int = 30) -> dict:
    """Position scatter while the player stands still: per-window std over windows whose spread is small."""
    sd = []
    for a in range(0, len(u) - win, win):
        su, sv = np.std(u[a : a + win]), np.std(v[a : a + win])
        if np.ptp(u[a : a + win]) < 3 and np.ptp(v[a : a + win]) < 3:
            sd.append((su, sv))
    if not sd:
        return {"windows": 0}
    sd = np.array(sd)
    return {"windows": len(sd), "window_frames": win, "std_u_px_median": round(float(np.median(sd[:, 0])), 3),
            "std_v_px_median": round(float(np.median(sd[:, 1])), 3)}


def parallax_check(raw_u, raw_v, area, size) -> dict:
    """Does the disc centroid need the puck's height correction? Only if it ever sits far from the principal
    point: the correction is k/2 * r. The data decide k only if r varies enough to fit area = A0 + a1 r."""
    pp = (size[0] / 2, size[1] / 2)
    r = np.hypot(np.asarray(raw_u) - pp[0], np.asarray(raw_v) - pp[1])
    a = np.asarray(area, float)
    out = {"principal_point": list(pp), "r_px_median": round(float(np.median(r)), 3), "r_px_p99": round(float(np.percentile(r, 99)), 3),
           "r_px_max": round(float(r.max()), 3)}
    k_c = 0.0
    if np.ptp(r) > 20:
        A = np.vstack([np.ones_like(r), r]).T
        a0, a1 = np.linalg.lstsq(A, a, rcond=None)[0]
        growth = float(a1 * np.percentile(r, 95) / a0) if a0 > 0 else 0.0
        out["area_model"] = {"a0_px2": round(float(a0), 2), "a1_px2_per_px": round(float(a1), 5), "growth_over_radii_seen": round(growth, 4)}
        if growth >= 0.25:  # same rule as the puck (scale.MIN_AREA_GROWTH)
            k_c = float(a1 / (2 * math.sqrt(a0 / math.pi))) / 2
    else:
        out["area_model"] = "not fittable: the disc never leaves the image centre (r range %.1f px)" % float(np.ptp(r))
    bound = 10 * K_PUCK_MAX / 2 * float(np.percentile(r, 99))
    out["max_correction_px_if_k_10x_puck"] = round(bound, 4)
    out["k_centroid"] = round(k_c, 6)
    out["applied"] = k_c > 0
    if k_c > 0:
        out["reason"] = "the disc blob grows with distance from the image centre (a raised object): centroids reprojected like the puck's"
    else:
        out["reason"] = (f"no height correction: the disc stays within {out['r_px_p99']:.1f} px (p99) of the image centre (the camera "
                         f"follows the player), so k/2*r < {bound:.3f} px even with k = 10x the puck's, below the centroid noise; "
                         f"and where r varies the blob does not grow with r")
    return out


def track_player(clip: Path, results: Path = RESULTS, ref_clip: str | None = None) -> tuple[list[dict], dict]:
    m, bg, ref_name = load_reference(clip, results, ref_clip)
    _, info = open_clip(clip)
    cam = FollowCamera(bg, landmark_points(m))
    allowed = m.rink_mask(margin_px=8.0) > 0
    W, H = info.width, info.height
    b = m.board_px
    pdu = pdv = 0.0
    vdu = vdv = 0.0
    last = None  # (frame, u_ref, v_ref)
    raw: dict[int, dict] = {}
    shifts: dict[int, tuple] = {}
    bad_cam: list[int] = []
    n = 0
    for i, f in iter_frames(clip):
        n = i + 1
        # Narrow search at the prediction, then at the last shift, then wide (identical faceoff circles sit
        # 160-500 px apart, so a wide search is only a last resort).
        s = cam.measure_at(f, pdu + vdu, pdv + vdv, SEARCH_FOLLOW)
        for qu, qv, sr in ((pdu, pdv, SEARCH_FOLLOW), (pdu, pdv, SEARCH_FIRST)):
            if s.ok:
                break
            s = cam.measure_at(f, qu, qv, sr)
        if not s.ok:
            bad_cam.append(i)
            continue
        vdu, vdv = s.du - pdu, s.dv - pdv
        if i == 0:
            vdu = vdv = 0.0
        pdu, pdv = s.du, s.dv
        shifts[i] = (s.du, s.dv, s.residual)
        roi = (int(max(b["left"] - 40 + s.du, 0)), int(max(b["top"] - 40 + s.dv, 0)),
               int(min(b["right"] + 40 + s.du, W)), int(min(b["bottom"] + 40 + s.dv, H)))
        cands = []
        for c in disc_candidates(f, roi):
            ur, vr = c["u"] - s.du, c["v"] - s.dv
            iu, iv = int(round(ur)), int(round(vr))
            if 0 <= iv < allowed.shape[0] and 0 <= iu < allowed.shape[1] and allowed[iv, iu]:
                cands.append({**c, "ur": ur, "vr": vr})
        if not cands:
            continue
        if last is not None:
            gate = GATE_PX + GATE_GROW * (i - last[0] - 1)
            near = [c for c in cands if math.hypot(c["ur"] - last[1], c["vr"] - last[2]) <= gate]
            pick = min(near, key=lambda c: math.hypot(c["ur"] - last[1], c["vr"] - last[2])) if near else None
            if pick is None and (len(cands) == 1 or i - last[0] > 30):  # a teleport (practice reset): re-acquire
                pick = max(cands, key=lambda c: c["area"] * c["roundness"])
        else:
            pick = max(cands, key=lambda c: c["area"] * c["roundness"])
        if pick is None:
            continue
        raw[i] = pick
        last = (i, pick["ur"], pick["vr"])

    # Duplicated capture frames: the position repeats exactly right after a real step.
    dups = []
    kept: list[int] = []
    for f in sorted(raw):
        if len(kept) >= 2 and f - kept[-1] == 1:
            a, b_, c = raw[kept[-2]], raw[kept[-1]], raw[f]
            if math.hypot(c["ur"] - b_["ur"], c["vr"] - b_["vr"]) < 1e-6 and math.hypot(b_["ur"] - a["ur"], b_["vr"] - a["vr"]) > DUP_STEP:
                dups.append(f)
                continue
        kept.append(f)
    for f in dups:
        del raw[f]
    dset = set(dups)
    det = sorted(raw)
    areas = np.array([raw[f]["area"] for f in det], float)
    area_ref = float(np.median(areas)) if len(areas) else 1.0
    rows = []
    for f in det:
        c = raw[f]
        ca = math.exp(-0.5 * (math.log(max(c["area"], 1) / area_ref) / 0.25) ** 2)
        conf = ca * min(1.0, (c["roundness"] - ROUND_MIN) / (0.9 - ROUND_MIN)) * min(1.0, c["fill"] / 0.8)
        rows.append({"frame": f, "u": c["ur"], "v": c["vr"], "raw_u": c["u"], "raw_v": c["v"], "shift_u": shifts[f][0],
                     "shift_v": shifts[f][1], "area": c["area"], "roundness": c["roundness"], "conf": max(0.0, conf), "interpolated": False})
    # Interpolate short gaps (not across dropped duplicates: no real sample there).
    filled = []
    for a, b_ in zip(rows, rows[1:]):
        filled.append(a)
        g = b_["frame"] - a["frame"]
        jump = math.hypot(b_["u"] - a["u"], b_["v"] - a["v"]) / g
        if 1 < g <= MAX_INTERP_GAP + 1 and jump <= MAX_INTERP_SPEED and not all(k in dset for k in range(a["frame"] + 1, b_["frame"])):
            for k in range(a["frame"] + 1, b_["frame"]):
                if k in dset:
                    continue
                s_ = (k - a["frame"]) / g
                filled.append({"frame": k, "u": a["u"] + s_ * (b_["u"] - a["u"]), "v": a["v"] + s_ * (b_["v"] - a["v"]),
                               "raw_u": None, "raw_v": None, "shift_u": shifts.get(k, (0, 0))[0], "shift_v": shifts.get(k, (0, 0))[1],
                               "area": 0, "roundness": 0.0, "conf": 0.0, "interpolated": True})
    if rows:
        filled.append(rows[-1])
    hc = parallax_check([r["raw_u"] for r in rows], [r["raw_v"] for r in rows], [r["area"] for r in rows], (W, H)) if rows else None
    if hc and hc["applied"]:
        from .scale import ground

        for r in filled:
            r["u"], r["v"] = (float(q) for q in ground(r["u"], r["v"], r["shift_u"], r["shift_v"], hc["k_centroid"], tuple(hc["principal_point"])))
    for r in filled:
        r["t"] = r["frame"] / info.fps
        x, y = m.px_to_m(r["u"], r["v"])
        r["x"], r["y"] = float(x), float(y)

    longest = (0, None)
    marks = [-1] + det + [n]
    for a, b_ in zip(marks, marks[1:]):
        missing = [k for k in range(a + 1, b_) if k not in dset]
        if len(missing) > longest[0]:
            longest = (len(missing), (a + 1, b_ - 1))
    u = np.array([r["u"] for r in rows])
    v = np.array([r["v"] for r in rows])
    cam_travel = np.array([math.hypot(*shifts[f][:2]) for f in shifts]) if shifts else np.zeros(1)
    rep = {
        "frames_total": n,
        "frames_detected": len(det),
        "detection_rate": len(det) / max(n - len(dups), 1),
        "frames_interpolated": sum(1 for r in filled if r["interpolated"]),
        "longest_gap_frames": longest[0],
        "longest_gap_range": longest[1],
        "duplicate_frames_dropped": len(dups),
        "duplicate_frames": dups,
        "camera_uncorrectable_frames": len(bad_cam),
        "camera_shift_max_px": round(float(cam_travel.max()), 3),
        "camera_residual_max_px": round(max((s[2] for s in shifts.values()), default=0.0), 3),
        "player_extent_px": [round(float(np.ptp(u)), 2), round(float(np.ptp(v)), 2)] if len(u) else None,
        "area_ref_px": area_ref,
        "rest_noise": rest_noise(u, v),
        "height_correction": hc,
        "reference": ref_name,
    }
    return filled, rep


def write_outputs(clip: Path, rows: list[dict], rep: dict, results: Path = RESULTS) -> None:
    name = Path(clip).stem
    m = RinkMap.load(results / f"{rep['reference']}.map.json")
    lines = ["t,x,y"] + [f"{r['t']!r},{round(r['x'], 6)!r},{round(r['y'], 6)!r}" for r in rows]
    (results / f"{name}.player.csv").write_text("\n".join(lines) + "\n")
    rnd = lambda a, k=3: None if a is None else round(float(a), k)  # noqa: E731
    meta = {
        "clip": name,
        "csv": f"{name}.player.csv",
        "map": f"{rep['reference']}.map.json",
        "map_note": "the clip's own map" if rep["reference"] == name else
                    f"no map for this clip (the camera travels too far for `measure map`): registered against {rep['reference']}'s reference frame (same game camera and zoom)",
        "fps": m.fps,
        "t_origin": "t = frame_index / fps, frame 0 = first frame of the clip",
        "convention": CONVENTION,
        "scale_source": m.scale_source,
        "m_per_px": m.m_per_px,
        "what": "player body = pink disc; position = centre of an ellipse fitted to the convex hull of the pink blob, minus the frame's camera shift (reference ice pixels)",
        "camera_follows_player": "the game camera follows the player: the disc stays at the image centre and the player's motion is the camera's motion, measured on the faceoff-dot landmarks",
        "confidence": "exp(-0.5 (ln(area/area_ref)/0.25)^2) * roundness score * fill score; 0 for interpolated rows",
        "detector": {"hsv_pink": [*PINK_H, PINK_S, PINK_V], "blob_area_px": [AREA_MIN, AREA_MAX], "roundness_min": ROUND_MIN,
                     "gate_px_per_frame": GATE_PX, "max_interp_gap_frames": MAX_INTERP_GAP},
        "summary": rep,
        "rows": {
            "frame": [r["frame"] for r in rows],
            "u_px": [rnd(r["u"]) for r in rows],
            "v_px": [rnd(r["v"]) for r in rows],
            "raw_u_px": [rnd(r["raw_u"]) for r in rows],
            "raw_v_px": [rnd(r["raw_v"]) for r in rows],
            "shift_u": [rnd(r["shift_u"]) for r in rows],
            "shift_v": [rnd(r["shift_v"]) for r in rows],
            "area_px": [r["area"] for r in rows],
            "roundness": [rnd(r["roundness"]) for r in rows],
            "conf": [rnd(r["conf"]) for r in rows],
            "interpolated": [int(r["interpolated"]) for r in rows],
        },
    }
    (results / f"{name}.player.json").write_text(json.dumps(meta, separators=(",", ":")) + "\n")


def run_track(clip: Path, results: Path = RESULTS, ref_clip: str | None = None) -> tuple[list[dict], dict]:
    rows, rep = track_player(clip, results, ref_clip)
    write_outputs(clip, rows, rep, results)
    name = Path(clip).stem
    hc = rep["height_correction"] or {}
    rn = rep["rest_noise"]
    print(f"[player] {name}: detection {rep['detection_rate'] * 100:.2f}% ({rep['frames_detected']}/{rep['frames_total'] - rep['duplicate_frames_dropped']} "
          f"non-duplicate frames), {rep['frames_interpolated']} interpolated, longest gap {rep['longest_gap_frames']} frames at {rep['longest_gap_range']}; "
          f"{rep['duplicate_frames_dropped']} duplicate frames dropped; camera shift max {rep['camera_shift_max_px']:.1f} px, landmark residual max "
          f"{rep['camera_residual_max_px']:.2f} px, {rep['camera_uncorrectable_frames']} uncorrectable; player extent {rep['player_extent_px']} px; "
          f"rest noise std {rn.get('std_u_px_median')}/{rn.get('std_v_px_median')} px; disc r from centre p99 {hc.get('r_px_p99')} px "
          f"(height correction applied: {hc.get('applied')}); reference {rep['reference']}")
    return rows, rep


# ---------------------------------------------------------------- QA overlays

def _label(img, text, org, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def run_qa(clip: Path, results: Path = RESULTS, frames: list[int] | None = None) -> list[Path]:
    """Overview (whole track on the reference frame) + a sheet of camera-corrected frames with the detection
    drawn (cross = tracked position, ring = disc size) and a 6x zoom inset."""
    name = Path(clip).stem
    meta = json.loads((results / f"{name}.player.json").read_text())
    R = meta["rows"]
    pos = {f: (u, v, it) for f, u, v, it in zip(R["frame"], R["u_px"], R["v_px"], R["interpolated"])}
    shift = {f: (a, b) for f, a, b in zip(R["frame"], R["shift_u"], R["shift_v"])}
    m, bg, _ = load_reference(clip, results, meta["summary"]["reference"])
    qa = results / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    us, vs = np.array(R["u_px"]), np.array(R["v_px"])
    pad = 60
    x0, x1 = int(us.min() - pad), int(us.max() + pad)
    y0, y1 = int(vs.min() - pad), int(vs.max() + pad)
    Z = max(1, min(8, int(500 / max(x1 - x0, y1 - y0))))
    ov = cv2.resize(bg[max(y0, 0) : y1, max(x0, 0) : x1], None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
    fr = sorted(pos)
    for a, c in zip(fr, fr[1:]):
        if c == a + 1:
            p = lambda q: (int((pos[q][0] - max(x0, 0)) * Z), int((pos[q][1] - max(y0, 0)) * Z))  # noqa: E731
            cv2.line(ov, p(a), p(c), (0, 255, 255) if not pos[c][2] else (0, 0, 255), 1, cv2.LINE_AA)
    ov = cv2.copyMakeBorder(ov, 70, 0, 0, max(0, 520 - ov.shape[1]), cv2.BORDER_CONSTANT)
    s = meta["summary"]
    _label(ov, f"{name} player track (x{Z}), yellow = detected, red = interpolated", (6, 18), 0.45)
    _label(ov, f"det {s['detection_rate'] * 100:.2f}%  longest gap {s['longest_gap_frames']}f  dups {s['duplicate_frames_dropped']}", (6, 40), 0.45)
    if s["reference"] != name:
        _label(ov, f"background: reference frame of {s['reference'][:24]} (its player at spawn)", (6, 62), 0.4)
    p1 = qa / f"{name}.player_overview.png"
    cv2.imwrite(str(p1), ov)
    if frames is None:
        frames = np.linspace(0, s["frames_total"] - 1, 8).round().astype(int).tolist()
    want = set(frames)
    tiles = []
    for i, f in iter_frames(clip):
        if i > max(want):
            break
        if i not in want:
            continue
        sh = shift.get(i)
        if sh is not None:
            f = Camera.undo(f, Shift(sh[0], sh[1], 0.0, 4))
        cu, cv_ = (pos[i][0], pos[i][1]) if i in pos else (float(np.median(us)), float(np.median(vs)))
        Hh = 120
        crop = f[int(cv_) - Hh : int(cv_) + Hh, int(cu) - Hh : int(cu) + Hh].copy()
        for g in range(i - 60, i):  # recent path
            if g in pos and g + 1 in pos:
                q = lambda k: (int(round(pos[k][0] - cu + Hh)), int(round(pos[k][1] - cv_ + Hh)))  # noqa: E731
                cv2.line(crop, q(g), q(g + 1), (0, 255, 255), 1, cv2.LINE_AA)
        zh = 16
        zoom = f[int(cv_) - zh : int(cv_) + zh, int(cu) - zh : int(cu) + zh].copy()
        zoom = cv2.resize(zoom, (240, 240), interpolation=cv2.INTER_NEAREST)
        if i in pos:
            zx, zy = (pos[i][0] - (int(cv_ * 0 + cu) - zh) + 0.5) * 7.5, (pos[i][1] - (int(cv_) - zh) + 0.5) * 7.5
            cv2.drawMarker(zoom, (int(zx), int(zy)), (0, 0, 255), cv2.MARKER_CROSS, 30, 1)
            cv2.circle(zoom, (int(zx), int(zy)), int(7.5 * math.sqrt(s["area_ref_px"] / math.pi)), (0, 255, 0), 1, cv2.LINE_AA)
        tile = np.hstack([crop, zoom])
        st = "INTERP" if i in pos and pos[i][2] else ("" if i in pos else "NO DETECTION")
        _label(tile, f"frame {i} t={i / meta['fps']:.2f}s {st}", (6, 16), 0.45)
        if sh is not None:
            _label(tile, f"cam shift {sh[0]:+.1f},{sh[1]:+.1f}px", (6, 36), 0.45)
        tiles.append(tile)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r : r + cols]) for r in range(0, len(tiles), cols)])
    p2 = qa / f"{name}.player_sheet.png"
    cv2.imwrite(str(p2), sheet)
    print(f"[player qa] wrote {p1} and {p2}")
    return [p1, p2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m measure.player", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("track")
    pt.add_argument("clips", nargs="+", type=Path)
    pt.add_argument("--out-dir", type=Path, default=RESULTS)
    pt.add_argument("--ref-clip", default="justpuckphysicshittingitatfixedpoint", help="reference map/frame for clips without their own map")
    pq = sub.add_parser("qa")
    pq.add_argument("clips", nargs="+", type=Path)
    pq.add_argument("--out-dir", type=Path, default=RESULTS)
    pq.add_argument("--frames", default="", help="comma-separated frame indices for the sheet")
    pe = sub.add_parser("estimate")
    pe.add_argument("--out-dir", type=Path, default=RESULTS)
    pe.add_argument("--no-sim", action="store_true", help="skip the cargo sim check")
    a = ap.parse_args(argv)
    if a.cmd == "track":
        for c in a.clips:
            run_track(c, a.out_dir, a.ref_clip)
    elif a.cmd == "qa":
        fr = [int(x) for x in a.frames.split(",") if x] or None
        for c in a.clips:
            run_qa(c, a.out_dir, fr)
    elif a.cmd == "estimate":
        from .player_motion import run_estimate

        run_estimate(a.out_dir, sim=not a.no_sim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
