"""`segqa` command: QA images for the segment manifest.

segments_<clip>.png: one tile per segment: the rink (camera-corrected reference frame) with the track
in grey for context, the segment's path colored by time (green -> red), its start (white square),
detected bounces (yellow circles), and below it the windowed speed (px/frame) with the segment shaded.
segment_<id>_frames.png: camera-corrected frames along one segment, zoomed, with the path drawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera, Shift
from .geometry import RinkMap
from .segment import OUT, window_speed
from .video import iter_frames


def _txt(img, s, org, sc=0.45, col=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1)


def _load(results: Path, clip: str):
    m = RinkMap.load(results / f"{clip}.map.json")
    R = json.loads((results / f"{clip}.track.json").read_text())["rows"]
    return m, R


def overview(clip_path: Path, results: Path, manifest: dict, qa_dir: Path) -> Path:
    clip = clip_path.stem
    m, R = _load(results, clip)
    segs = [s for s in manifest["segments"] if s["clip"] == clip]
    bg = cv2.imread(str(clip_path.parent / f"{clip}.bg.png"))
    f = np.array(R["frame"])
    u, v = np.array(R["u_px"]), np.array(R["v_px"])
    tid = np.array(R["tracklet"])
    b = m.board_px
    x0, y0, x1, y1 = int(b["left"] - 20), int(b["top"] - 20), int(b["right"] + 20), int(b["bottom"] + 20)
    S = 0.45
    tiles = []
    for s in segs:
        img = bg.copy()
        sel = tid == s["tracklet"]
        ff, uu, vv = f[sel], u[sel], v[sel]
        ctx = (ff >= s["start_frame"] - 120) & (ff <= s["end_frame"] + 60)
        pts = np.stack([uu[ctx], vv[ctx]], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, (150, 150, 150), 1, cv2.LINE_AA)
        seg = (ff >= s["start_frame"]) & (ff <= s["end_frame"])
        idx = np.nonzero(seg)[0]
        for a, c in zip(idx, idx[1:]):
            t = (ff[a] - s["start_frame"]) / max(1, s["end_frame"] - s["start_frame"])
            col = (0, int(255 * (1 - t)), int(255 * t))
            cv2.line(img, (int(uu[a]), int(vv[a])), (int(uu[c]), int(vv[c])), col, 2, cv2.LINE_AA)
        cv2.rectangle(img, (int(uu[idx[0]]) - 6, int(vv[idx[0]]) - 6), (int(uu[idx[0]]) + 6, int(vv[idx[0]]) + 6), (255, 255, 255), 2)
        for bf in s["wall_bounce_frames"]:
            k = np.nonzero(ff == bf)[0]
            if len(k):
                cv2.circle(img, (int(uu[k[0]]), int(vv[k[0]])), 12, (0, 255, 255), 2)
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
        # speed strip
        H = 70
        strip = np.full((H, crop.shape[1], 3), 30, np.uint8)
        fc, uc, vc = ff[ctx], uu[ctx], vv[ctx]
        sp = window_speed(fc, uc, vc)
        if len(fc) > 1:
            fx = lambda q: int((q - fc[0]) / max(1, fc[-1] - fc[0]) * (crop.shape[1] - 1))
            cv2.rectangle(strip, (fx(s["start_frame"]), 0), (fx(s["end_frame"]), H - 1), (60, 90, 60), -1)
            top = max(1.0, float(sp.max()))
            pl = np.stack([[fx(q) for q in fc], (H - 4 - sp / top * (H - 8)).astype(int)], 1).astype(np.int32)
            cv2.polylines(strip, [pl], False, (255, 255, 255), 1, cv2.LINE_AA)
            for bf in s["wall_bounce_frames"]:
                cv2.line(strip, (fx(bf), 0), (fx(bf), H - 1), (0, 255, 255), 1)
            _txt(strip, f"max {top:.1f} px/f", (4, 14), 0.4)
        tile = np.vstack([crop, strip])
        _txt(tile, f"{s['id']}  {s['label']}{'  AIR?' if s.get('possibly_airborne') else ''}", (6, 16))
        _txt(tile, f"f {s['start_frame']}-{s['end_frame']} ({s['duration_s']:.2f}s) v0 {s['v0_px_per_frame']:.1f}px/f", (6, 34), 0.4)
        _txt(tile, f"end: {s['end_reason']}", (6, 50), 0.4)
        tiles.append(tile)
    cols = 5
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    p = qa_dir / f"segments_{clip}.png"
    cv2.imwrite(str(p), sheet)
    return p


def frames_strip(clip_path: Path, results: Path, manifest: dict, seg_id: str, qa_dir: Path, n: int = 10) -> Path:
    clip = clip_path.stem
    s = next(x for x in manifest["segments"] if x["id"] == seg_id)
    m, R = _load(results, clip)
    rows = {fr: (uu, vv, su, sv) for fr, uu, vv, su, sv in zip(R["frame"], R["u_px"], R["v_px"], R["shift_u"], R["shift_v"]) if s["start_frame"] - 30 <= fr <= s["end_frame"] + 30}
    want = sorted(set(np.linspace(s["start_frame"], s["end_frame"], n - 2).round().astype(int).tolist() + [max(s["start_frame"] - 8, 0), s["end_frame"] + 8] + s["wall_bounce_frames"]))
    path = [(fr, rows[fr][0], rows[fr][1]) for fr in sorted(rows) if s["start_frame"] <= fr <= s["end_frame"]]
    tiles = []
    for i, f in iter_frames(clip_path):
        if i > max(want):
            break
        if i not in want:
            continue
        if i in rows:
            f = Camera.undo(f, Shift(rows[i][2], rows[i][3], 0.0, 4))
        img = f.copy()
        for (fa, ua, va), (fb, ub, vb) in zip(path, path[1:]):
            if fb <= i:
                cv2.line(img, (int(ua), int(va)), (int(ub), int(vb)), (0, 200, 255), 1, cv2.LINE_AA)
        c = rows.get(i)
        if c:
            cv2.circle(img, (int(round(c[0])), int(round(c[1]))), 9, (0, 0, 255), 1, cv2.LINE_AA)
        cu, cv_ = (int(c[0]), int(c[1])) if c else (int(path[-1][1]), int(path[-1][2]))
        Hh = 90
        crop = img[max(cv_ - Hh, 0) : cv_ + Hh, max(cu - Hh, 0) : cu + Hh]
        crop = cv2.resize(crop, (2 * Hh * 2, 2 * Hh * 2), interpolation=cv2.INTER_NEAREST)
        tag = "in segment" if s["start_frame"] <= i <= s["end_frame"] else "outside"
        if i in s["wall_bounce_frames"]:
            tag = "BOUNCE"
        _txt(crop, f"{seg_id} frame {i} ({tag})", (6, 18), 0.5)
        tiles.append(crop)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    p = qa_dir / f"segment_{seg_id}_frames.png"
    cv2.imwrite(str(p), sheet)
    return p


def run_segqa(clips: list[Path], results: Path, frames_for: list[str], manifest_path: Path = OUT / "manifest.json") -> list[Path]:
    manifest = json.loads(Path(manifest_path).read_text())
    qa_dir = results / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for c in clips:
        out.append(overview(Path(c), results, manifest, qa_dir))
        for sid in frames_for:
            seg = next((x for x in manifest["segments"] if x["id"] == sid), None)
            if seg and seg["clip"] == Path(c).stem:
                out.append(frames_strip(Path(c), results, manifest, sid, qa_dir))
    for p in out:
        print(f"[segqa] wrote {p}")
    return out
