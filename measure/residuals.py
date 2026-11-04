"""`residuals` command: per-segment residuals, old default (params.placeholder.json) vs new default (params.json)
vs held-out (staged fit trained on the other clip, from holdout.json). Writes calib/data/real/residuals.txt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import REPO
from .holdout import V2W, evaluate
from .segment import OUT


def sc(e):
    return e["path_rmse"] ** 2 + V2W * e["v2_rmse"] ** 2


def run_residuals(old: Path = REPO / "params.placeholder.json", new: Path = REPO / "params.json") -> dict:
    segs = json.loads((OUT / "manifest.json").read_text())["segments"]
    b, f = evaluate(old, segs), evaluate(new, segs)
    held = {}
    for d in json.loads((OUT / "holdout.json").read_text())["directions"]:
        for n, v in d["staged"]["per_segment_test"].items():
            held[n] = v["fitted"]
    L = ["# per-segment residuals; meters at the PLACEHOLDER scale; score = path_rmse^2 + 0.01 v2_rmse^2",
         "# old = params.placeholder.json (placeholder physics, puck_radius 0.15); new = params.json (fitted + measured shape); held-out = staged fit on the OTHER clip",
         "segment  clip    label        air | old pos/path/v2/score              | new pos/path/v2/score            | held-out pos/path/v2/score"]
    for s in segs:
        i = s["id"]
        h = held.get(i)
        hs = "%7.3f %6.3f %7.2f %7.2f" % (h["pos_rmse"], h["path_rmse"], h["v2_rmse"], sc(h)) if h else "   (lifted: not used)"
        L.append("%-8s %-7s %-12s %-3s | %7.3f %6.3f %7.2f %8.2f | %7.3f %6.3f %7.2f %7.2f | %s" % (
            i, s["clip"][:7], s["label"], "Y" if s["possibly_airborne"] else "", b[i]["pos_rmse"], b[i]["path_rmse"], b[i]["v2_rmse"], sc(b[i]),
            f[i]["pos_rmse"], f[i]["path_rmse"], f[i]["v2_rmse"], sc(f[i]), hs))
    (OUT / "residuals.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    grp = {}
    for s in segs:
        grp.setdefault((s["clip"][:7], s["label"], s["possibly_airborne"]), []).append(s["id"])
    print()
    out = {}
    for k, ids in sorted(grp.items()):
        m = lambda d, key: float(np.median([d[i][key] for i in ids]))
        hh = [held[i] for i in ids if i in held]
        hm = lambda key: float(np.median([x[key] for x in hh])) if hh else float("nan")
        out["/".join(map(str, k))] = {key: [m(b, key), m(f, key), hm(key)] for key in ("pos_rmse", "path_rmse", "v2_rmse")}
        print(k, len(ids), "pos %.2f -> %.2f (held-out %.2f) | path %.2f -> %.2f (%.2f) | v2 %.1f -> %.1f (%.1f)" % (
            m(b, "pos_rmse"), m(f, "pos_rmse"), hm("pos_rmse"), m(b, "path_rmse"), m(f, "path_rmse"), hm("path_rmse"), m(b, "v2_rmse"), m(f, "v2_rmse"), hm("v2_rmse")))
    g = [s["id"] for s in segs if not s["possibly_airborne"]]
    allg = lambda d, key: float(np.median([d[i][key] for i in g]))
    print("all ground (%d): pos %.3f -> %.3f m, path %.3f -> %.3f m, v2 %.2f -> %.2f, mean score %.2f -> %.2f" % (
        len(g), allg(b, "pos_rmse"), allg(f, "pos_rmse"), allg(b, "path_rmse"), allg(f, "path_rmse"), allg(b, "v2_rmse"), allg(f, "v2_rmse"),
        np.mean([sc(b[i]) for i in g]), np.mean([sc(f[i]) for i in g])))
    worst = sorted(g, key=lambda i: -sc(f[i]))[:3]
    print("worst three (new default):", [(i, round(sc(f[i]), 2), round(f[i]["path_rmse"], 3), round(f[i]["v2_rmse"], 2)) for i in worst])
    return {"groups": out, "worst": worst}
