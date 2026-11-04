"""`qa` command: PNG sheets with the track drawn over sampled frames, for checking by eye."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import RinkMap
from .video import iter_frames, open_clip

SEG_COLORS = [(0, 255, 255), (255, 128, 0), (255, 0, 255), (0, 200, 0), (0, 128, 255), (255, 255, 0), (128, 0, 255)]


def _label(img, text, org, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def pick_frames(meta: dict, n_total: int, k: int = 12) -> list[int]:
    """Spread over the clip, plus the frames the report flags (outliers, gap ends, segment starts)."""
    s = meta["summary"]
    flagged = list(s.get("speed_outlier_frames", []))[:3] + list(s.get("spike_frames", []))[:2]
    if s.get("longest_gap_range"):
        a, b = s["longest_gap_range"]
        flagged += [a - 1, (a + b) // 2, b + 1]
    even = np.linspace(0, n_total - 1, k + 2)[1:-1].round().astype(int).tolist()
    picks = sorted(set(int(f) for f in flagged + even if 0 <= f < n_total))
    return picks[:16]


def run_qa(clip: Path, out_dir: Path) -> list[Path]:
    _, info = open_clip(clip)
    m = RinkMap.load(out_dir / f"{info.name}.map.json")
    meta = json.loads((out_dir / f"{info.name}.track.json").read_text())
    R = meta["rows"]
    pos = {f: (u, v, c, it, tid) for f, u, v, c, it, tid in zip(R["frame"], R["u_px"], R["v_px"], R["conf"], R["interpolated"], R["tracklet"])}
    seg_ids = sorted(set(R["tracklet"]))
    color = {t: SEG_COLORS[i % len(SEG_COLORS)] for i, t in enumerate(seg_ids)}
    b = m.board_px
    x0, y0 = int(b["left"] - 30), int(b["top"] - 30)
    x1, y1 = int(b["right"] + 30), int(b["bottom"] + 30)
    qa_dir = out_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    outs = []

    # 1. Overview: the whole path on the background frame.
    bg = cv2.imread(str(clip.parent / m.extra.get("background_png", f"{info.name}.bg.png")))
    ov = bg.copy()
    frames = sorted(pos)
    for a, c in zip(frames, frames[1:]):
        if c == a + 1 and pos[a][4] == pos[c][4]:
            cv2.line(ov, (int(pos[a][0]), int(pos[a][1])), (int(pos[c][0]), int(pos[c][1])), color[pos[a][4]], 1, cv2.LINE_AA)
    for f in meta["summary"].get("speed_outlier_frames", []) + meta["summary"].get("spike_frames", []):
        if f in pos:
            cv2.circle(ov, (int(pos[f][0]), int(pos[f][1])), 8, (0, 0, 255), 2)
    for zx0, zy0, zx1, zy1 in meta["summary"].get("net_zones_px", []):
        cv2.rectangle(ov, (zx0, zy0), (zx1, zy1), (0, 0, 255), 1)
    ov = ov[y0:y1, x0:x1]
    s = meta["summary"]
    _label(ov, f"{info.name}", (8, 20))
    _label(ov, f"det {s['detection_rate'] * 100:.1f}%  gap {s['longest_gap_frames']}f  segs {s['tracklets_puck']}", (8, 42))
    _label(ov, "red circles = speed outliers/spikes; red boxes = excluded net roofs", (8, 64))
    p = qa_dir / f"{info.name}.overview.png"
    cv2.imwrite(str(p), ov)
    outs.append(p)

    # 2. Sheet: sampled frames, recent path + current detection, with a 4x zoom inset.
    picks = pick_frames(meta, info.frame_count)
    want = set(picks)
    tiles = []
    for i, f in iter_frames(clip):
        if i not in want:
            if i > max(want):
                break
            continue
        img = f.copy()
        for g in range(i - 45, i):
            if g in pos and g + 1 in pos and pos[g][4] == pos[g + 1][4]:
                cv2.line(img, (int(pos[g][0]), int(pos[g][1])), (int(pos[g + 1][0]), int(pos[g + 1][1])), color[pos[g][4]], 2, cv2.LINE_AA)
        if i in pos:
            u, v, c, it, tid = pos[i]
            zc = (u, v)
            status = f"conf {c:.2f}" + (" INTERP" if it else "")
        else:
            prev = [g for g in frames if g < i]
            zc = (pos[prev[-1]][0], pos[prev[-1]][1]) if prev else ((x0 + x1) / 2, (y0 + y1) / 2)
            moved = any(a <= i <= b for a, b in meta["summary"].get("camera_moved_ranges", []))
            status = "CAMERA MOVED: frame dropped" if moved else "NO DETECTION (zoom at last known)"
        Z, H = 4, 24
        cu, cv_ = int(round(zc[0])), int(round(zc[1]))
        zoom = f[max(cv_ - H, 0) : cv_ + H, max(cu - H, 0) : cu + H].copy()
        zoom = cv2.resize(zoom, (2 * H * Z, 2 * H * Z), interpolation=cv2.INTER_NEAREST)
        if i in pos:
            zu = (pos[i][0] - (cu - H) + 0.5) * Z
            zv = (pos[i][1] - (cv_ - H) + 0.5) * Z
            cv2.drawMarker(zoom, (int(zu), int(zv)), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
        crop = img[y0:y1, x0:x1]
        if i in pos:
            cv2.circle(crop, (int(pos[i][0]) - x0, int(pos[i][1]) - y0), 14, (0, 0, 255), 2)
        crop = cv2.resize(crop, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
        tile = np.zeros((max(crop.shape[0], zoom.shape[0]), crop.shape[1] + zoom.shape[1], 3), np.uint8)
        tile[: crop.shape[0], : crop.shape[1]] = crop
        tile[: zoom.shape[0], crop.shape[1] :] = zoom
        _label(tile, f"frame {i}  t={i / info.fps:.2f}s", (6, 16), 0.45)
        _label(tile, status, (6, 34), 0.45)
        tiles.append(tile)
    cols = 4
    th, tw = tiles[0].shape[:2]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows_ = [np.hstack(tiles[r : r + cols]) for r in range(0, len(tiles), cols)]
    sheet = np.vstack(rows_)
    p = qa_dir / f"{info.name}.sheet.png"
    cv2.imwrite(str(p), sheet)
    outs.append(p)
    print(f"[qa] wrote {', '.join(str(o) for o in outs)} ({len(picks)} sampled frames)")
    return outs
