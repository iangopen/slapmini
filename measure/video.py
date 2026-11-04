"""Video access through OpenCV's bundled ffmpeg (no system ffmpeg needed)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class ClipInfo:
    path: Path
    name: str
    width: int
    height: int
    fps: float
    frame_count: int


def clip_name(path: Path) -> str:
    return Path(path).stem


def open_clip(path: Path) -> tuple[cv2.VideoCapture, ClipInfo]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    info = ClipInfo(
        path=path,
        name=clip_name(path),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return cap, info


def iter_frames(path: Path, every: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Sequential decode. Yields (frame_index, BGR frame) for every `every`-th frame."""
    cap, _ = open_clip(path)
    i = 0
    try:
        while True:
            if i % every == 0:
                ok, f = cap.read()
                if not ok:
                    return
                yield i, f
            elif not cap.grab():
                return
            i += 1
    finally:
        cap.release()


def median_frame(path: Path, n_samples: int = 61) -> np.ndarray:
    """Per-pixel median of n_samples frames spread evenly over the clip (removes moving things)."""
    cap, info = open_clip(path)
    n = info.frame_count
    # Sequential decode with grab() is exact; seeking in mkv can land on the wrong frame.
    want = set(np.linspace(0, max(n - 2, 0), n_samples).round().astype(int).tolist())
    frames = []
    i = 0
    while True:
        if i in want:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        elif not cap.grab():
            break
        i += 1
        if i > max(want):
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    stack = np.stack(frames)
    out = np.empty_like(frames[0])
    for r0 in range(0, out.shape[0], 120):  # row chunks keep memory bounded
        out[r0 : r0 + 120] = np.median(stack[:, r0 : r0 + 120], axis=0).astype(np.uint8)
    return out
