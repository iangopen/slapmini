"""`track` command: find the puck in every frame and write NAME.traj.csv + NAME.track.json.

Detection = green pixels (HSV) inside the rink outline, minus anything green in the static
background (median frame). Candidates are linked into tracklets with a speed gate (a puck cannot
move farther per frame than its speed cap allows). Only tracklets that actually travel count as
the puck: the green object on the net roof sits still and jumps between spots, so it never makes
a travelling tracklet. HUD and stamina dots are outside the rink mask.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry import CONVENTION, RinkMap, load_params
from .camera import LIMIT_PX as DRIFT_LIMIT_PX, Camera, landmark_points
from .video import iter_frames, median_frame, open_clip

# Puck color: saturated green (OpenCV hue 0..180).
HUE_LO, HUE_HI, SAT_MIN, VAL_MIN = 35, 85, 90, 90
AREA_MIN, AREA_MAX = 6, 600  # px, per blob
SPEED_SAFETY = 2.0  # per-frame gate = SPEED_SAFETY * largest placeholder speed cap (scale and caps are both guesses)
STATIC_JOIN_PX = 6.0  # a candidate on a static-background green may only extend a track this close to the prediction
DUP_EPS_PX = 0.05  # a moving puck that does not move at all in one frame is a duplicated (stale) frame
MAX_LINK_GAP = 12  # frames a tracklet may go undetected and still continue
MAX_INTERP_GAP = 6  # missing frames inside a tracklet filled by linear interpolation
MOVE_EXTENT_PX = 12.0  # a puck tracklet must travel at least this far ...
MOVE_STEPS = 5  # ... with at least this many frames of real motion (> 0.5 px)


def green_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= HUE_LO) & (h <= HUE_HI) & (s >= SAT_MIN) & (v >= VAL_MIN)


def static_exclusion(bg_bgr: np.ndarray) -> np.ndarray:
    """Pixels that are green in the background frame (lights that are on most of the time), grown 4 px."""
    g = green_mask(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)).astype(np.uint8)
    return cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0


def net_zones(bg_bgr: np.ndarray, m: RinkMap, pad: int = 8) -> list[list[int]]:
    """Boxes [x0, y0, x1, y1] around each net roof. The roof's hatched mesh is the only large textured
    patch near each end of the ice; a green object sits and moves on it. A puck under the roof is
    hidden anyway, so excluding the roof loses nothing."""
    g = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    tex = cv2.boxFilter(hp * hp, -1, (9, 9))
    t = ((tex > 60) & (m.rink_mask(-6.0) > 0)).astype(np.uint8)
    t = cv2.morphologyEx(t, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    n, _, st, cen = cv2.connectedComponentsWithStats(t)
    b = m.board_px
    L, W = b["bottom"] - b["top"], b["right"] - b["left"]
    u0 = m.origin_px[0]
    zones = []
    for end, near in (("top", lambda v: v - b["top"] < 0.25 * L), ("bottom", lambda v: b["bottom"] - v < 0.25 * L)):
        # Net roofs: big (the real ones are ~100 x 45 px) and on the long axis (faceoff marks are off-axis).
        big = [i for i in range(1, n) if st[i][4] >= 1500 and near(cen[i][1]) and abs(cen[i][0] - u0) < 0.15 * W]
        # Pieces of the same roof (the texture can split) are merged if they overlap the big box's columns.
        boxes = [st[i] for i in range(1, n) if near(cen[i][1]) and st[i][4] >= 200 and any(
            st[i][0] < st[j][0] + st[j][2] + 20 and st[j][0] < st[i][0] + st[i][2] + 20 and abs(cen[i][1] - cen[j][1]) < 60 for j in big)]
        if boxes:
            zones.append([
                int(min(x for x, *_ in boxes)) - pad, int(min(y for _, y, *_ in boxes)) - pad,
                int(max(x + w for x, _, w, *_ in boxes)) + pad, int(max(y + h for _, y, _, h, _ in boxes)) + pad,
            ])
    return zones


@dataclass
class Detector:
    allowed: np.ndarray  # bool mask, full frame
    x0: int = 0
    y0: int = 0
    x1: int = 0
    y1: int = 0

    def __post_init__(self):
        ys, xs = np.nonzero(self.allowed)
        self.x0, self.x1 = int(xs.min()), int(xs.max()) + 1
        self.y0, self.y1 = int(ys.min()), int(ys.max()) + 1
        self.roi_allowed = self.allowed[self.y0 : self.y1, self.x0 : self.x1]
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    static: np.ndarray | None = None  # bool mask: green in the static background frame

    def candidates(self, frame: np.ndarray) -> list[tuple[float, float, int, float, bool]]:
        """(u, v, area, mean_sat, on_static_green) per green blob; u, v = saturation-weighted centroid."""
        roi = frame[self.y0 : self.y1, self.x0 : self.x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        g = green_mask(hsv) & self.roi_allowed
        if not g.any():
            return []
        # Merge pieces of one blob split by a stick or the puck's highlight stripe.
        lab_src = cv2.dilate(g.astype(np.uint8), self.kernel)
        n, lab, st, _ = cv2.connectedComponentsWithStats(lab_src, connectivity=8)
        out = []
        sat = hsv[..., 1].astype(np.float32)
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            sub = (lab[y : y + h, x : x + w] == i) & g[y : y + h, x : x + w]
            area = int(sub.sum())
            if area < AREA_MIN or area > AREA_MAX:
                continue
            wgt = np.where(sub, np.clip((sat[y : y + h, x : x + w] - 60.0) / 195.0, 0.05, 1.0), 0.0)
            ys, xs = np.mgrid[y : y + h, x : x + w]
            W = wgt.sum()
            u, v = float((xs * wgt).sum() / W) + self.x0, float((ys * wgt).sum() / W) + self.y0
            on_static = bool(self.static is not None and self.static[int(round(v)), int(round(u))])
            out.append((u, v, area, float(sat[y : y + h, x : x + w][sub].mean()), on_static))
        return out


@dataclass
class Tracklet:
    id: int
    frames: list = field(default_factory=list)
    pts: list = field(default_factory=list)  # (u, v, area, sat)

    @property
    def last(self):
        return self.frames[-1], self.pts[-1]

    def velocity(self):
        if len(self.frames) < 2:
            return 0.0, 0.0
        (f1, p1), (f0, p0) = (self.frames[-1], self.pts[-1]), (self.frames[-2], self.pts[-2])
        d = f1 - f0
        return (p1[0] - p0[0]) / d, (p1[1] - p0[1]) / d

    def extent(self) -> float:
        a = np.array([p[:2] for p in self.pts])
        return float(np.hypot(*(a.max(0) - a.min(0))))

    def moving_steps(self) -> int:
        a = np.array([p[:2] for p in self.pts])
        return int((np.hypot(*np.diff(a, axis=0).T) > 0.5).sum()) if len(a) > 1 else 0

    def is_puck(self) -> bool:
        return len(self.frames) >= 3 and self.extent() >= MOVE_EXTENT_PX and self.moving_steps() >= MOVE_STEPS


def link(cands_per_frame: dict[int, list], max_step_px: float) -> list[Tracklet]:
    """Greedy nearest-neighbor linking. Gate, measured from the last position (not the prediction, so
    hits and bounces that reverse velocity still link): |v| * gap + max_step + 2 px. Across a detection
    gap the allowance grows only with the track's own speed, so a resting puck never links to a
    teleport (practice-tool reset). Candidates on static-background green only join a track when they
    sit within STATIC_JOIN_PX of its prediction. Ties are broken by distance to the prediction."""
    active: list[Tracklet] = []
    done: list[Tracklet] = []
    nid = 0
    for f in sorted(cands_per_frame):
        cands = cands_per_frame[f]
        still = []
        for t in active:
            (done if f - t.last[0] > MAX_LINK_GAP else still).append(t)
        active = still
        pairs = []
        for ti, t in enumerate(active):
            lf, lp = t.last
            gap = f - lf
            vx, vy = t.velocity()
            px, py = lp[0] + vx * gap, lp[1] + vy * gap
            gate = math.hypot(vx, vy) * gap + max_step_px + 2.0
            for ci, c in enumerate(cands):
                dpred = math.hypot(c[0] - px, c[1] - py)
                if math.hypot(c[0] - lp[0], c[1] - lp[1]) <= gate and (not c[4] or dpred <= STATIC_JOIN_PX):
                    pairs.append((dpred, ti, ci))
        pairs.sort()
        used_t, used_c = set(), set()
        for _, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            active[ti].frames.append(f)
            active[ti].pts.append(cands[ci])
        for ci, c in enumerate(cands):
            if ci not in used_c:
                active.append(Tracklet(nid, [f], [c]))
                nid += 1
    return done + active


def select_puck(tracklets: list[Tracklet]) -> dict[int, tuple]:
    """frame -> (u, v, area, sat, tracklet id) from travelling tracklets; longer tracklets win overlaps."""
    chosen: dict[int, tuple] = {}
    for t in sorted((t for t in tracklets if t.is_puck()), key=lambda t: -len(t.frames)):
        for f, p in zip(t.frames, t.pts):
            if f not in chosen:
                chosen[f] = (*p, t.id)  # (u, v, area, sat, on_static, tracklet)
    return chosen


def fill_gaps(chosen: dict[int, tuple], dups: set[int] = frozenset()) -> dict[int, tuple]:
    """Linear interpolation over gaps of <= MAX_INTERP_GAP frames inside one tracklet. A gap that is
    only a dropped duplicate frame is left empty (there is no real sample to recover there)."""

    def dup_between(a, b):
        return all(f in dups for f in range(a + 1, b))

    out = {f: (*p[:2], p[5], False) for f, p in chosen.items()}
    fr = sorted(chosen)
    for a, b in zip(fr, fr[1:]):
        if 1 < b - a <= MAX_INTERP_GAP + 1 and chosen[a][5] == chosen[b][5] and not dup_between(a, b):
            for f in range(a + 1, b):
                s = (f - a) / (b - a)
                out[f] = (chosen[a][0] + s * (chosen[b][0] - chosen[a][0]), chosen[a][1] + s * (chosen[b][1] - chosen[a][1]), chosen[a][5], True)
    return out


def confidence(area: float, sat: float, area_ref: float) -> float:
    """Heuristic in [0, 1]: blob area close to the typical puck area, and strongly saturated."""
    ca = math.exp(-0.5 * (math.log(max(area, 1) / area_ref) / 0.5) ** 2)
    cs = min(max((sat - SAT_MIN) / 100.0, 0.0), 1.0)
    return ca * (0.5 + 0.5 * cs)


def step_jitter(steps: list[tuple[int, float]]) -> dict:
    """How uneven per-frame travel is while gliding: |step / median of its 9-frame window - 1|.
    Capture timing jitter (game at a higher fps than the 60 fps recording) shows up here."""
    if len(steps) < 9:
        return {"samples": 0}
    fr = np.array([f for f, _ in steps])
    st = np.array([s for _, s in steps])
    dev = []
    for i in range(4, len(st) - 4):
        if fr[i + 4] - fr[i - 4] > 12:  # window must be (nearly) contiguous
            continue
        med = np.median(st[i - 4 : i + 5])
        if med > 2.0:
            dev.append(abs(st[i] / med - 1))
    if not dev:
        return {"samples": 0}
    return {"samples": len(dev), "median_rel_dev": float(np.median(dev)), "p90_rel_dev": float(np.percentile(dev, 90))}


def runs(frames: list[int]) -> list[tuple[int, int]]:
    out = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [tuple(r) for r in out]


def track_video(path: Path, m: RinkMap, bg: np.ndarray, max_step_px: float, correct_camera: bool = True):
    """Returns (rows, report). rows: list of dicts per output frame.

    Camera: every frame's translation against the reference is measured from the faceoff landmarks and
    undone before detection, so all positions are in reference-frame pixels. Every 10th corrected frame
    is re-measured (the residual). Frames that cannot be corrected to <= DRIFT_LIMIT_PX are dropped.
    With correct_camera=False, frames are only checked and those drifting > the limit are dropped."""
    allowed = m.rink_mask(margin_px=8.0) > 0  # the puck overlaps the board edge on screen
    zones = net_zones(bg, m)
    for x0, y0, x1, y1 in zones:
        allowed[max(y0, 0) : y1, max(x0, 0) : x1] = False
    det = Detector(allowed, static=static_exclusion(bg))
    cam = Camera(bg, landmark_points(m))
    cands: dict[int, list] = {}
    drift: dict[int, float] = {}  # raw per-frame camera shift magnitude
    shifts: dict[int, tuple[float, float]] = {}
    residuals: list[float] = []  # re-measured after correction (every 10th frame)
    uncorrectable: list[int] = []
    n = 0
    for i, f in iter_frames(path):
        n = i + 1
        sh = cam.measure(f)
        drift[i] = sh.magnitude if sh.found >= 3 else float("inf")
        if correct_camera:
            if not sh.ok:
                uncorrectable.append(i)
                cands[i] = []
                continue
            f = Camera.undo(f, sh)
            shifts[i] = (sh.du, sh.dv)
            if i % 10 == 0:
                s2 = cam.measure(f)
                residuals.append(max(s2.magnitude, s2.residual) if s2.found >= 3 else float("inf"))
                if residuals[-1] > DRIFT_LIMIT_PX:
                    uncorrectable.append(i)
        elif drift[i] > DRIFT_LIMIT_PX:
            uncorrectable.append(i)
        cands[i] = det.candidates(f)
    # Drop whole uncorrectable stretches: a frame between two bad frames <= 10 apart is dropped too.
    bad = set(uncorrectable)
    ub = sorted(bad)
    for a, b in zip(ub, ub[1:]):
        if b - a <= 10:
            bad.update(range(a, b))
    for i in bad:
        cands[i] = []
    tracklets = link(cands, max_step_px)
    chosen = select_puck(tracklets)
    moved = sorted(bad)
    raw_moved = sorted(i for i, d in drift.items() if d > DRIFT_LIMIT_PX)
    # Duplicated capture frames: the recording repeats a frame now and then, so a moving puck shows
    # zero motion for one frame. That row carries a stale position and wrong time: drop it.
    # Compare with the previous KEPT row, so runs of 2+ repeated frames are all dropped.
    dups = []
    kept: list[int] = []
    for f in sorted(chosen):
        c = chosen[f]
        if len(kept) >= 2 and f - kept[-1] <= 3:
            a, b = chosen[kept[-2]], chosen[kept[-1]]
            if a[5] == b[5] == c[5] and math.hypot(c[0] - b[0], c[1] - b[1]) < DUP_EPS_PX and math.hypot(b[0] - a[0], b[1] - a[1]) > 1.0:
                dups.append(f)
                continue
        kept.append(f)
    for f in dups:
        del chosen[f]
    filled = fill_gaps(chosen, set(dups))
    areas = [p[2] for p in chosen.values()]
    area_ref = float(np.median(areas)) if areas else 1.0
    rows = []
    for f in sorted(filled):
        u, v, tid, interp = filled[f]
        x, y = m.px_to_m(u, v)
        c = 0.0 if interp else confidence(chosen[f][2], chosen[f][3], area_ref)
        shu, shv = shifts.get(f, (0.0, 0.0))
        area = 0 if interp else chosen[f][2]
        rows.append({"frame": f, "t": f / m.fps, "u": u, "v": v, "x": float(x), "y": float(y), "conf": c, "interpolated": interp, "tracklet": tid, "shift_u": shu, "shift_v": shv, "area": area})

    # ---- report
    det_frames = sorted(chosen)
    longest = (0, None)
    if det_frames:
        for a, b in zip(det_frames, det_frames[1:]):
            if b - a - 1 > longest[0]:
                longest = (b - a - 1, (a + 1, b - 1))
    dup_set = set(dups)
    steps = []
    for a, b in zip(rows, rows[1:]):
        df = b["frame"] - a["frame"]
        if a["tracklet"] == b["tracklet"] and (df == 1 or (df == 2 and a["frame"] + 1 in dup_set)):
            steps.append((b["frame"], math.hypot(b["u"] - a["u"], b["v"] - a["v"]) / df))
    spikes = []
    for a, b, c in zip(rows, rows[1:], rows[2:]):
        if c["frame"] == a["frame"] + 2 and b["frame"] == a["frame"] + 1 and not b["interpolated"]:
            d1 = math.hypot(b["u"] - a["u"], b["v"] - a["v"])
            d2 = math.hypot(c["u"] - b["u"], c["v"] - b["v"])
            d13 = math.hypot(c["u"] - a["u"], c["v"] - a["v"])
            if d1 > 4 and d2 > 4 and d13 < 0.5 * (d1 + d2):
                spikes.append(b["frame"])
    puck_tracklets = [t for t in tracklets if t.is_puck()]
    report = {
        "frames_total": n,
        "frames_detected": len(det_frames),
        "frames_interpolated": sum(1 for r in rows if r["interpolated"]),
        "detection_rate": len(det_frames) / n if n else 0.0,
        "detection_rate_stable_frames": len(det_frames) / (n - len(moved)) if n > len(moved) else 0.0,
        "longest_gap_frames": longest[0],
        "longest_gap_range": longest[1],
        "duplicate_frames_dropped": len(dups),
        "duplicate_frames": dups,
        "step_jitter": step_jitter(steps),
        "camera_correction": correct_camera,
        "camera_moved_frames_raw": len(raw_moved),
        "frames_kept_without_correction": n - len(raw_moved),
        "camera_moved_frames": len(moved),
        "camera_moved_ranges": runs(moved),
        "frames_kept": n - len(moved),
        "max_camera_drift_px": (max((d for d in drift.values() if d != float("inf")), default=None)),
        "camera_residual_after_correction_px": {
            "samples": len(residuals),
            "max": max((r for r in residuals if r != float("inf")), default=None),
            "p99": float(np.percentile([r for r in residuals if r != float("inf")], 99)) if residuals else None,
        },
        "max_step_px_gate": max_step_px,
        "max_step_px_observed": max((s for _, s in steps), default=0.0),
        "step_px_p99": float(np.percentile([s for _, s in steps], 99)) if steps else 0.0,
        "spike_frames": spikes,
        "tracklets_total": len(tracklets),
        "tracklets_puck": len(puck_tracklets),
        "segments": [
            {"tracklet": t.id, "start_frame": t.frames[0], "end_frame": t.frames[-1], "detections": len(t.frames)}
            for t in sorted(puck_tracklets, key=lambda t: t.frames[0])
        ],
        "area_ref_px": area_ref,
        "net_zones_px": zones,
        "steps": steps,
    }
    return rows, report


def write_csv(rows: list[dict], path: Path) -> None:
    lines = ["t,x,y,z"]
    for r in rows:
        lines.append(f"{r['t']!r},{round(r['x'], 6)!r},{round(r['y'], 6)!r},0")
    Path(path).write_text("\n".join(lines) + "\n")


def run_track(clip: Path, out_dir: Path, params_path: Path, max_step_px: float | None = None, correct_camera: bool = True):
    params = load_params(params_path)
    _, info = open_clip(clip)
    map_path = out_dir / f"{info.name}.map.json"
    if not map_path.exists():
        raise FileNotFoundError(f"{map_path} missing; run `python -m measure map {clip}` first")
    m = RinkMap.load(map_path)
    drift_ok = m.extra.get("drift", {}).get("ok", False)
    if not drift_ok:
        raise RuntimeError(f"{info.name}: the map's camera check failed (camera motion not correctable by translation); refusing to track.")
    bg_path = clip.parent / m.extra.get("background_png", f"{info.name}.bg.png")
    bg = cv2.imread(str(bg_path)) if bg_path.exists() else median_frame(clip)
    cap = max(params["max_puck_speed_ice_dash"], params["max_puck_speed_ice_normal"], params["max_puck_speed_air"])
    gate_src = f"{SPEED_SAFETY:g} x max speed cap {cap:g} m/s (PLACEHOLDER) / {m.fps:g} fps / {m.m_per_px:.5f} m/px (PLACEHOLDER scale)"
    if max_step_px is None:
        max_step_px = SPEED_SAFETY * cap / m.fps / m.m_per_px
    else:
        gate_src = "command line --max-step-px"
    print(f"[track] {info.name}: {info.frame_count} frames, gate {max_step_px:.1f} px/frame ({gate_src})")
    rows, rep = track_video(clip, m, bg, max_step_px, correct_camera=correct_camera)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / f"{info.name}.traj.csv")
    speed_cap = cap
    speeds = [(f, s * m.m_per_px * m.fps) for f, s in rep.pop("steps")]
    outliers = [f for f, s in speeds if s > speed_cap]
    meta = {
        "clip": info.name,
        "csv": f"{info.name}.traj.csv",
        "map": map_path.name,
        "fps": m.fps,
        "t_origin": "t = frame_index / fps, frame 0 = first frame of the clip",
        "convention": CONVENTION,
        "scale_source": m.scale_source,
        "m_per_px": m.m_per_px,
        "z": "always 0: a top-down view cannot see height (a lifted puck also appears displaced toward the camera)",
        "map_drift_ok": drift_ok,
        "camera_correction": "per-frame translation from the faceoff landmarks, undone before detection; u_px/v_px are reference-frame pixels, shift_u/shift_v the frame's measured shift" if correct_camera else "off: frames drifting > limit dropped",
        "detector": {
            "hsv_green": [HUE_LO, HUE_HI, SAT_MIN, VAL_MIN],
            "blob_area_px": [AREA_MIN, AREA_MAX],
            "gate_px_per_frame": max_step_px,
            "gate_source": gate_src,
            "max_link_gap_frames": MAX_LINK_GAP,
            "max_interp_gap_frames": MAX_INTERP_GAP,
            "puck_tracklet_rule": f"extent >= {MOVE_EXTENT_PX} px and >= {MOVE_STEPS} moving frames",
            "confidence": "exp(-0.5 (ln(area/area_ref)/0.5)^2) * (0.5 + 0.5 sat_score); 0 for interpolated rows",
        },
        "summary": {
            **{k: v for k, v in rep.items() if k != "segments"},
            "speed_cap_m_s": speed_cap,
            "speed_outlier_frames": outliers,
            "max_speed_m_s": max((s for _, s in speeds), default=0.0),
        },
        "segments": rep["segments"],
        "rows": {
            "frame": [r["frame"] for r in rows],
            "u_px": [round(r["u"], 3) for r in rows],
            "v_px": [round(r["v"], 3) for r in rows],
            "conf": [round(r["conf"], 3) for r in rows],
            "interpolated": [int(r["interpolated"]) for r in rows],
            "tracklet": [r["tracklet"] for r in rows],
            "area_px": [r["area"] for r in rows],
            "shift_u": [round(r["shift_u"], 3) for r in rows],
            "shift_v": [round(r["shift_v"], 3) for r in rows],
        },
    }
    (out_dir / f"{info.name}.track.json").write_text(json.dumps(meta, separators=(",", ":")) + "\n")
    s = meta["summary"]
    print(
        f"[track] detection rate {s['detection_rate'] * 100:.1f}% ({s['frames_detected']}/{s['frames_total']} frames), "
        f"{s['frames_interpolated']} interpolated, longest gap {s['longest_gap_frames']} frames "
        f"({s['longest_gap_frames'] / m.fps:.2f} s) at {s['longest_gap_range']}"
    )
    cr = s["camera_residual_after_correction_px"]
    print(
        f"[track] camera: raw drift max {s['max_camera_drift_px']:.2f} px; frames kept without correction {s['frames_kept_without_correction']}/{s['frames_total']}, "
        f"with correction {s['frames_kept']}/{s['frames_total']}; residual after correction max {cr['max'] if cr['max'] is None else round(cr['max'], 3)} px, "
        f"p99 {cr['p99'] if cr['p99'] is None else round(cr['p99'], 3)} px over {cr['samples']} re-measured frames"
    )
    if s["camera_moved_frames"]:
        print(f"[track] detection rate among kept frames: {s['detection_rate_stable_frames'] * 100:.1f}% ({s['camera_moved_frames']} uncorrectable frames dropped)")
    print(
        f"[track] {s['tracklets_puck']} puck segments of {s['tracklets_total']} tracklets; "
        f"steps: max {s['max_step_px_observed']:.1f} px, p99 {s['step_px_p99']:.1f} px (gate {max_step_px:.1f}); "
        f"max speed {s['max_speed_m_s']:.1f} m/s (placeholder scale)"
    )
    print(
        f"[track] speed outliers (> {speed_cap:g} m/s): {len(outliers)} {outliers[:10]}; back-and-forth spikes: "
        f"{len(s['spike_frames'])} {s['spike_frames'][:10]}; uncorrectable frames dropped: {s['camera_moved_frames']}"
    )
    print(f"[track] wrote {out_dir / meta['csv']} ({len(rows)} rows) and {info.name}.track.json")
    return rows, meta
