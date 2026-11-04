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
    pt.add_argument("--no-camera-correction", action="store_true", help="do not undo camera motion; drop frames drifting > 2 px instead")
    pq = sub.add_parser("qa", help="write QA sheets (track drawn over sampled frames)")
    common(pq)
    pg = sub.add_parser("segment", help="cut tracks into free-puck stretches -> calib/data/real/ scenarios + manifest")
    pg.add_argument("clips", nargs="+", type=Path)
    pg.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="where the map/track results are")
    pg.add_argument("--params", type=Path, default=PARAMS_PATH)
    pg.add_argument("--dest", type=Path, default=None, help="default calib/data/real")
    pq2 = sub.add_parser("segqa", help="QA sheets for the segment manifest (+ frame strips for --frames IDs)")
    pq2.add_argument("clips", nargs="+", type=Path)
    pq2.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    pq2.add_argument("--frames", default="", help="comma-separated segment ids to draw frame strips for")
    pp = sub.add_parser("params", help="write the measured rink shape into params.json (provenance measured-shape)")
    pp.add_argument("clips", nargs="+", type=Path)
    pp.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    pp.add_argument("--params", type=Path, default=PARAMS_PATH)
    pp.add_argument("--dry-run", action="store_true")
    pe = sub.add_parser("estimate", help="scale-free restitution + glide-deceleration estimates from the segment manifest")
    pe.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    psc = sub.add_parser("scale", help="board vs ice-plane scale per clip; stores both in the map JSONs")
    psc.add_argument("clips", nargs="+", type=Path)
    psc.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    prj = sub.add_parser("reproject", help="puck blob centroids -> ground points; rewrites NAME.traj.csv (no re-detection)")
    prj.add_argument("clips", nargs="+", type=Path)
    prj.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    sub.add_parser("holdout", help="fit scale-free puck params with clip hold-out; writes params.fitted.json if both directions improve")
    pc = sub.add_parser("compare", help="overlay a real segment with sim replays (one per --params)")
    pc.add_argument("segment")
    pc.add_argument("--params", type=Path, action="append", required=True)
    pc.add_argument("--label", action="append")
    pc.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    pdm = sub.add_parser("dragmodel", help="held-out drag-model comparison + smoothing and synthetic-jitter checks")
    pdm.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    psz = sub.add_parser("sizes", help="puck and player radius from the frames -> params.json (measured-shape)")
    psz.add_argument("clips", nargs="+", type=Path)
    psz.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    psz.add_argument("--dry-run", action="store_true")
    pmg = sub.add_parser("merge", help="copy the fitted params from params.fitted.json into params.json")
    pmg.add_argument("--src", type=Path, default=REPO / "params.fitted.json")
    ps = sub.add_parser("synth", help="render the synthetic test video (for debugging the tests)")
    ps.add_argument("out", type=Path)

    a = ap.parse_args(argv)
    failed = []
    if a.cmd == "synth":
        from .synth import render

        truth = render(a.out)
        print(f"wrote {a.out} ({len(truth['frames'])} frames)")
        return 0
    if a.cmd == "sizes":
        from .sizes import run_sizes

        run_sizes(a.clips, a.out_dir, write=not a.dry_run)
        return 0
    if a.cmd == "merge":
        from .shape_params import merge_fitted

        for n, e in merge_fitted(a.src).items():
            print(f"[merge] {n} = {e['value']} ({e['provenance']})")
        return 0
    if a.cmd == "dragmodel":
        from .dragmodel import run_dragmodel

        run_dragmodel(a.out_dir)
        return 0
    if a.cmd == "compare":
        from .compare import run_compare

        run_compare(a.segment, a.params, a.out_dir, a.label)
        return 0
    if a.cmd == "holdout":
        from .holdout import run_holdout

        run_holdout()
        return 0
    if a.cmd == "scale":
        from .scale import run_scale

        run_scale(a.out_dir, [Path(x).stem for x in a.clips])
        return 0
    if a.cmd == "reproject":
        from .scale import reproject

        for x in a.clips:
            reproject(a.out_dir, Path(x).stem)
        return 0
    if a.cmd == "estimate":
        from .estimate import run_estimate

        run_estimate(a.out_dir)
        return 0
    if a.cmd == "params":
        from .shape_params import run_params

        run_params(a.out_dir, [Path(x).stem for x in a.clips], a.params, write=not a.dry_run)
        return 0
    if a.cmd == "segqa":
        from .segqa import run_segqa

        run_segqa(a.clips, a.out_dir, [x for x in a.frames.split(",") if x])
        return 0
    if a.cmd == "segment":
        from .segment import OUT, run_segment

        run_segment(a.clips, a.out_dir, a.dest or OUT, a.params)
        return 0
    for clip in a.clips:
        try:
            if a.cmd == "map":
                from .rinkmap import run_map

                run_map(clip, a.out_dir, a.params)
            elif a.cmd == "track":
                from .track import run_track

                run_track(clip, a.out_dir, a.params, max_step_px=a.max_step_px, correct_camera=not a.no_camera_correction)
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
