"""python -m measure {map,track,qa,synth} ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .geometry import PARAMS_PATH, REPO

DEFAULT_OUT = REPO / "measure" / "results"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m measure", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("clips", nargs="+", type=Path, help="video file(s), e.g. clips/NAME.mkv")
        p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help=f"default {DEFAULT_OUT.relative_to(REPO)}")
        p.add_argument("--params", type=Path, default=PARAMS_PATH)

    common(sub.add_parser("map", help="find the rink, write NAME.map.json/.png, check camera drift"))
    pt = sub.add_parser("track", help="track the puck, write NAME.traj.csv and NAME.track.json")
    common(pt)
    pt.add_argument("--max-step-px", type=float, default=None, help="gate: max puck travel per frame (default from the speed cap)")
    pt.add_argument("--stable-frames-only", action="store_true", help="track a clip whose map failed the drift check, keeping only frames with <= 2 px drift")
    pq = sub.add_parser("qa", help="write QA sheets (track drawn over sampled frames)")
    common(pq)
    ps = sub.add_parser("synth", help="render the synthetic test video (for debugging the tests)")
    ps.add_argument("out", type=Path)

    a = ap.parse_args(argv)
    failed = []
    if a.cmd == "synth":
        from .synth import render

        truth = render(a.out)
        print(f"wrote {a.out} ({len(truth['frames'])} frames)")
        return 0
    for clip in a.clips:
        try:
            if a.cmd == "map":
                from .rinkmap import run_map

                run_map(clip, a.out_dir, a.params)
            elif a.cmd == "track":
                from .track import run_track

                run_track(clip, a.out_dir, a.params, max_step_px=a.max_step_px, stable_frames_only=a.stable_frames_only)
            elif a.cmd == "qa":
                from .qa import run_qa

                run_qa(clip, a.out_dir)
        except Exception as e:  # report and continue with the next clip
            print(f"ERROR [{a.cmd}] {clip}: {e}", file=sys.stderr)
            failed.append(clip)
    if failed:
        print(f"{len(failed)} clip(s) failed: {', '.join(str(c) for c in failed)}", file=sys.stderr)
        return 1
    return 0
