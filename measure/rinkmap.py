"""`map` command: find the rink in a median reference frame and check the camera never moves."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .geometry import RinkMap, fmt_ratio, length_width_ratio, load_params, scale_from_params
from .video import iter_frames, median_frame, open_clip

DRIFT_LIMIT_PX = 2.0


class MapError(RuntimeError):
    pass


# ---------------------------------------------------------------- detection on the reference frame

def board_mask(bgr: np.ndarray) -> np.ndarray:
    """Pixels of the boards: bright light grey, plus the yellow strip on their inner edge."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (v > 120) & ((s < 70) | ((h >= 10) & (h <= 35)))


def detect_ice(bgr: np.ndarray) -> np.ndarray:
    """Filled mask (uint8 0/255) of the region enclosed by the inner board outline."""
    nb = (~board_mask(bgr)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(nb, connectivity=4)
    H, W = nb.shape
    best, best_area = -1, 0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue  # touches the image border: arena, not ice
        if a > best_area:
            best, best_area = i, a
    if best < 0 or best_area < 0.02 * H * W:
        raise MapError("no enclosed ice region found (is the whole rink on screen?)")
    comp = (lab == best).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return filled


def board_extents(ice: np.ndarray) -> tuple[dict, float]:
    """Inner board edges (pixel-boundary coordinates) from straight stretches, plus corner radius."""
    ys, xs = np.nonzero(ice)
    top0, bot0, left0, right0 = ys.min(), ys.max(), xs.min(), xs.max()
    L, W = bot0 - top0 + 1, right0 - left0 + 1
    long_vertical = L >= W
    # Sample the straight parts of each side, away from the corners and the bench gates.
    if long_vertical:
        rows = np.r_[int(top0 + 0.2 * L) : int(top0 + 0.3 * L), int(bot0 - 0.3 * L) : int(bot0 - 0.2 * L)]
        cols = np.arange(int(left0 + 0.3 * W), int(right0 - 0.3 * W))
    else:
        rows = np.arange(int(top0 + 0.3 * L), int(bot0 - 0.3 * L))
        cols = np.r_[int(left0 + 0.2 * W) : int(left0 + 0.3 * W), int(right0 - 0.3 * W) : int(right0 - 0.2 * W)]
    lefts = [np.argmax(ice[r] > 0) for r in rows]
    rights = [ice.shape[1] - 1 - np.argmax(ice[r, ::-1] > 0) for r in rows]
    tops = [np.argmax(ice[:, c] > 0) for c in cols]
    bots = [ice.shape[0] - 1 - np.argmax(ice[::-1, c] > 0) for c in cols]
    # Edges sit half a pixel outside the first/last ice pixel centers.
    b = {
        "left": float(np.median(lefts)) - 0.5,
        "right": float(np.median(rights)) + 0.5,
        "top": float(np.median(tops)) - 0.5,
        "bottom": float(np.median(bots)) + 0.5,
    }
    bw, bh = b["right"] - b["left"], b["bottom"] - b["top"]
    area = float((ice > 0).sum())
    deficit = max(bw * bh - area, 0.0)
    r = math.sqrt(deficit / (4 - math.pi))  # rounded rectangle: area = wh - (4 - pi) r^2
    return b, r


def detect_center_line(bgr: np.ndarray, b: dict) -> float:
    """Row (subpixel) of the saturated center line across the middle of the rink."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(float)
    c0, c1 = int(b["left"] + 10), int(b["right"] - 10)
    mid = 0.5 * (b["top"] + b["bottom"])
    half = 0.15 * (b["bottom"] - b["top"])
    r0, r1 = int(mid - half), int(mid + half)
    prof = s[r0:r1, c0:c1].mean(axis=1)
    k = int(np.argmax(prof))
    base = float(np.median(prof))
    if prof[k] - base < 40:
        raise MapError("center line not found (no saturated line near mid-rink)")
    lo, hi = max(k - 8, 0), min(k + 9, len(prof))
    w = np.clip(prof[lo:hi] - base, 0, None)
    w[w < 0.3 * (prof[k] - base)] = 0
    return r0 + float((np.arange(lo, hi) * w).sum() / w.sum())


def detect_center_dot(bgr: np.ndarray, b: dict, v_line: float) -> float:
    """Column of the center dot. Along the bright center line, the dot's dark outline shows up as two
    brightness dips; the dot center is midway between the dip pair nearest the board middle."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    r = int(round(v_line))
    val = hsv[r - 1 : r + 2, :, 2].astype(float).mean(axis=0)
    umid = 0.5 * (b["left"] + b["right"])
    half = int(0.1 * (b["right"] - b["left"]))
    c0, c1 = int(umid) - half, int(umid) + half
    seg = val[c0:c1]
    line_v = float(np.median(seg))
    dark = seg < 0.75 * line_v
    runs = []  # (weighted center, start, end) of each dark run
    i = 0
    while i < len(dark):
        if dark[i]:
            j = i
            while j < len(dark) and dark[j]:
                j += 1
            w = line_v - seg[i:j]
            runs.append((c0 + float((np.arange(i, j) * w).sum() / w.sum()), i, j))
            i = j
        else:
            i += 1
    best = None
    for a in range(len(runs)):
        for c in range(a + 1, len(runs)):
            sep = runs[c][0] - runs[a][0]
            if 4 <= sep <= 30:
                mid = 0.5 * (runs[a][0] + runs[c][0])
                if best is None or abs(mid - umid) < abs(best - umid):
                    best = mid
    if best is None:
        raise MapError("center dot not found (no pair of dark dips along the center line near mid-rink)")
    return best


# ---------------------------------------------------------------- drift check

def landmark_center(b: dict, r: float) -> tuple[float, float]:
    """The 45-degree point of the top-left board corner arc: has edges in both directions."""
    k = r * (1 - 1 / math.sqrt(2))
    return b["left"] + k, b["top"] + k


def match_landmark(gray: np.ndarray, tmpl: np.ndarray, center: tuple[float, float], search: int = 40):
    """Return (du, dv, score): subpixel offset of the template relative to `center`."""
    th, tw = tmpl.shape
    cu, cv_ = int(round(center[0])), int(round(center[1]))
    x0, y0 = cu - tw // 2 - search, cv_ - th // 2 - search
    x0c, y0c = max(x0, 0), max(y0, 0)
    win = gray[y0c : cv_ + th // 2 + search + 1, x0c : cu + tw // 2 + search + 1]
    res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (px, py) = cv2.minMaxLoc(res)

    def para(a, b_, c):
        d = a - 2 * b_ + c
        return 0.0 if d == 0 else 0.5 * (a - c) / d

    sx = para(res[py, px - 1], res[py, px], res[py, px + 1]) if 0 < px < res.shape[1] - 1 else 0.0
    sy = para(res[py - 1, px], res[py, px], res[py + 1, px]) if 0 < py < res.shape[0] - 1 else 0.0
    # Template top-left in the search window that corresponds to zero drift:
    zx, zy = (cu - tw // 2) - x0c, (cv_ - th // 2) - y0c
    return px + sx - zx, py + sy - zy, float(score)


def measure_drift(path: Path, ref_gray: np.ndarray, center, fps: float, half: int = 40):
    cu, cv_ = int(round(center[0])), int(round(center[1]))
    tmpl = ref_gray[cv_ - half : cv_ + half + 1, cu - half : cu + half + 1]
    samples = []
    for i, f in iter_frames(path, every=max(1, int(round(fps)))):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        du, dv, sc = match_landmark(g, tmpl, center)
        samples.append({"frame": i, "t": round(i / fps, 3), "du": round(float(du), 3), "dv": round(float(dv), 3), "score": round(float(sc), 3)})
    return samples


# ---------------------------------------------------------------- command

def annotate(bgr: np.ndarray, m: RinkMap, lm_center, drift_max: float) -> np.ndarray:
    img = bgr.copy()
    outline = cv2.Canny(m.rink_mask(), 50, 150)
    img[outline > 0] = (0, 255, 255)
    u0, v0 = m.origin_px
    b = m.board_px
    cv2.line(img, (int(b["left"]), int(round(m.extra["center_line_v_px"]))), (int(b["right"]), int(round(m.extra["center_line_v_px"]))), (255, 0, 255), 1)

    def arrow(ax, col, label):
        p1 = (int(round(u0 + ax[0] * 120)), int(round(v0 + ax[1] * 120)))
        cv2.arrowedLine(img, (int(round(u0)), int(round(v0))), p1, col, 3, tipLength=0.2)
        cv2.putText(img, label, (p1[0] - 10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

    arrow(m.x_axis_px, (0, 0, 255), "+x")
    arrow(m.y_axis_px, (0, 255, 0), "+y")
    cv2.drawMarker(img, (int(round(u0)), int(round(v0))), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
    for lu, lv in m.extra.get("drift", {}).get("landmark_px", []):
        cv2.rectangle(img, (int(lu - 40), int(lv - 40)), (int(lu + 40), int(lv + 40)), (0, 128, 255), 2)
    for v in m.extra.get("shape", {}).get("circles", {}).get("fits", {}).values():
        if v:
            cv2.ellipse(img, ((v["center_px"][0], v["center_px"][1]), (v["axes_px"][0], v["axes_px"][1]), v["angle_deg"]), (0, 0, 255), 1, cv2.LINE_AA)
    lines = [
        m.clip,
        f"origin (center dot) = ({u0:.1f}, {v0:.1f}) px",
        f"board L x W = {m.board_length_px:.1f} x {m.board_width_px:.1f} px, corner r = {m.corner_radius_px:.1f} px",
        f"m/px = {m.m_per_px:.5f} (PLACEHOLDER rink_length)",
        f"L/W measured {m.extra['length_width_ratio']['measured']:.3f} vs params {m.extra['length_width_ratio']['params']:.3f}",
        f"raw camera drift {drift_max:.2f} px (orange boxes = camera landmarks, red = circle fits)",
        f"circles v/h {m.extra['shape']['circles']['summary']['v_over_h_mean']:.4f} (1 = top-down)",
    ]
    for k, t in enumerate(lines):
        cv2.putText(img, t, (20, 140 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(img, t, (20, 140 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def build_map(ref: np.ndarray, name: str, fps: float, frame_count: int, params: dict) -> RinkMap:
    ice = detect_ice(ref)
    b, r = board_extents(ice)
    bw, bh = b["right"] - b["left"], b["bottom"] - b["top"]
    if bh < bw:
        raise MapError("long axis is horizontal; this tool expects the nets at the top and bottom of the screen")
    v_line = detect_center_line(ref, b)
    u_dot = detect_center_dot(ref, b, v_line)
    mpp, src = scale_from_params(bh, params)
    measured_ratio = bh / bw
    m = RinkMap(
        clip=name,
        frame_size=(ref.shape[1], ref.shape[0]),
        fps=fps,
        frame_count=frame_count,
        origin_px=(round(u_dot, 3), round(v_line, 3)),
        x_axis_px=(0.0, -1.0),
        board_px={k: round(v, 3) for k, v in b.items()},
        corner_radius_px=round(r, 3),
        m_per_px=mpp,
        scale_source=src,
    )
    bc = (0.5 * (b["left"] + b["right"]), 0.5 * (b["top"] + b["bottom"]))
    m.extra = {
        "origin_source": "center dot (gap in the center line) at the center-line row",
        "center_line_v_px": round(v_line, 3),
        "board_center_px": [round(bc[0], 3), round(bc[1], 3)],
        "center_dot_minus_board_center_px": [round(u_dot - bc[0], 3), round(v_line - bc[1], 3)],
        "board_length_m": round(bh * mpp, 4),
        "board_width_m": round(bw * mpp, 4),
        "corner_radius_m": round(r * mpp, 4),
        "length_width_ratio": {"measured": round(measured_ratio, 4), "params": round(length_width_ratio(params), 4)},
    }
    return m


def shape_block(ref: np.ndarray, m: RinkMap) -> dict:
    """Circles (camera check), grey lines, net roofs, faceoff layout: pixels and placeholder-scale meters."""
    from .shape import circles, grey_lines
    from .track import net_zones

    ice = detect_ice(ref)
    circ = circles(ref, ice, m.origin_px, m.board_px)
    lines = grey_lines(ref, ice, m.board_px)
    u0, v0 = m.origin_px
    mpp = m.m_per_px
    out = {"circles": circ, "grey_lines_v_px": [round(x, 3) for x in lines]}
    if len(lines) == 4:
        out["goal_line_x_m"] = round(0.5 * ((v0 - lines[0]) + (lines[3] - v0)) * mpp, 4)
        out["blue_line_x_m"] = round(0.5 * ((v0 - lines[1]) + (lines[2] - v0)) * mpp, 4)
    zones = net_zones(ref, m, pad=0)
    out["net_roofs_px"] = zones
    if len(zones) == 2 and len(lines) == 4:
        (a, b_) = sorted(zones, key=lambda z: z[1])
        out["net_width_m"] = round(0.5 * ((a[2] - a[0]) + (b_[2] - b_[0])) * mpp, 4)
        # depth: goal line to the back of the roof (roof is above the ice: perspective may inflate it slightly)
        out["net_depth_m"] = round(0.5 * ((lines[0] - a[1]) + (b_[3] - lines[3])) * mpp, 4)
        out["net_center_u_offset_px"] = round(0.5 * ((a[0] + a[2]) + (b_[0] + b_[2])) / 2 - u0, 2)
    fo = [v for k, v in circ["fits"].items() if k.startswith("faceoff") and v]
    if len(fo) == 4:
        out["faceoff_x_m"] = round(float(np.mean([abs(v0 - f["center_px"][1]) for f in fo])) * mpp, 4)
        out["faceoff_y_m"] = round(float(np.mean([abs(u0 - f["center_px"][0]) for f in fo])) * mpp, 4)
        out["faceoff_circle_radius_m"] = round(float(np.mean([f["horizontal_diameter_px"] + f["vertical_diameter_px"] for f in fo])) / 4 * mpp, 4)
    if circ["fits"].get("center"):
        c = circ["fits"]["center"]
        out["center_circle_radius_m"] = round((c["horizontal_diameter_px"] + c["vertical_diameter_px"]) / 4 * mpp, 4)
    out["note"] = "meters use the PLACEHOLDER scale (params rink_length / board px); ratios are scale-free"
    return out


def run_map(clip: Path, out_dir: Path, params_path: Path, bg_dir: Path | None = None) -> RinkMap:
    from .camera import Camera, LIMIT_PX, landmark_points

    params = load_params(params_path)
    cap, info = open_clip(clip)
    cap.release()
    print(f"[map] {info.name}: {info.width}x{info.height} @ {info.fps:g} fps, {info.frame_count} frames")
    ref = median_frame(clip)
    bg_path = (bg_dir or clip.parent) / f"{info.name}.bg.png"
    cv2.imwrite(str(bg_path), ref)
    m = build_map(ref, info.name, info.fps, info.frame_count, params)

    m.extra["shape"] = shape_block(ref, m)
    # Camera: 4 faceoff-dot landmarks (on the ice) once per second. Raw drift = median landmark shift; the frame
    # is then shifted back and re-measured (residual = what a translation cannot fix).
    cam = Camera(ref, landmark_points(m))
    samples = []
    for i, f in iter_frames(clip, every=max(1, int(round(info.fps)))):
        s = cam.measure(f)
        rec = {"frame": i, "t": round(i / info.fps, 3), "du": round(s.du, 3), "dv": round(s.dv, 3), "landmarks": s.found, "spread_px": round(min(s.residual, 999.0), 3)}
        if s.found >= 3:
            s2 = cam.measure(Camera.undo(f, s))
            rec["residual_px"] = round(max(s2.magnitude, s2.residual) if s2.found >= 3 else 999.0, 3)
        else:
            rec["residual_px"] = 999.0
        samples.append(rec)
    raw = [math.hypot(r["du"], r["dv"]) for r in samples if r["landmarks"] >= 3]
    max_drift = max(raw) if raw else float("inf")
    bad = [r for r in samples if r["residual_px"] > LIMIT_PX]
    max_res = max((r["residual_px"] for r in samples if r["residual_px"] < 999), default=float("inf"))
    ok = bool(raw) and len(bad) <= 0.1 * len(samples)
    lm = landmark_center(m.board_px, m.corner_radius_px)
    m.extra["drift"] = {
        "landmarks": "4 end-zone faceoff dots (81x81 px patches, painted on the ice); shift = median",
        "landmark_px": [list(map(float, p)) for p in cam.points],
        "interval_s": 1.0,
        "samples": len(samples),
        "max_px": round(max_drift, 3),
        "camera_moves": bool(max_drift > LIMIT_PX),
        "max_residual_after_correction_px": round(max_res, 3),
        "uncorrectable_samples": len(bad),
        "limit_px": LIMIT_PX,
        "ok": ok,
        "correction": "track shifts every frame back by its measured translation; frames whose residual stays above the limit are dropped",
        "per_second": samples,
    }
    m.extra["background_png"] = str(bg_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    m.save(out_dir / f"{info.name}.map.json")
    cv2.imwrite(str(out_dir / f"{info.name}.map.png"), annotate(ref, m, lm, max_drift))

    u0, v0 = m.origin_px
    sh = m.extra["shape"]
    cs = sh["circles"]["summary"]
    print(f"[map] origin (center dot) = ({u0:.2f}, {v0:.2f}) px; board center = {m.extra['board_center_px']}")
    print(f"[map] inner board: L = {m.board_length_px:.2f} px (screen vertical), W = {m.board_width_px:.2f} px, corner r = {m.corner_radius_px:.1f} px")
    print(f"[map] scale: {m.m_per_px:.6f} m/px  <- {m.scale_source}")
    print(f"[map] {fmt_ratio(m.board_length_px / m.board_width_px, params)}")
    print(
        f"[map] circles ({cs['n']} fitted): vertical/horizontal diameter ratio {cs['v_over_h_min']:.4f}..{cs['v_over_h_max']:.4f} "
        f"(mean {cs['v_over_h_mean']:.4f}) -> {'ROUND: top-down camera, one scale for x and y' if cs['round'] else 'NOT ROUND: camera is tilted'}"
    )
    for k, v in sh["circles"]["fits"].items():
        if v:
            print(f"[map]   {k:9s} at ({v['center_px'][0]:.1f}, {v['center_px'][1]:.1f}): h {v['horizontal_diameter_px']:.2f} px, v {v['vertical_diameter_px']:.2f} px, v/h {v['v_over_h']:.4f}, rms {v['fit_rms_px']:.3f} px")
    print(
        f"[map] camera: raw drift max {max_drift:.3f} px over {len(samples)} one-second samples; after translation correction "
        f"max residual {max_res:.3f} px; {len(bad)} uncorrectable samples (limit {LIMIT_PX} px)"
    )
    print(f"[map] wrote {out_dir / (info.name + '.map.json')} and .map.png")
    if not cs["round"]:
        m.extra["scale_note"] = "CIRCLES NOT ROUND: separate x/y scales stored; every speed is distorted"
        print("[map] !!! WARNING: CIRCLES ARE NOT ROUND. The camera is not top-down; x and y scales differ. Nothing was corrected. !!!")
    if not ok:
        raise MapError(f"camera check FAILED for {info.name}: {len(bad)}/{len(samples)} samples cannot be corrected to <= {LIMIT_PX} px")
    return m
