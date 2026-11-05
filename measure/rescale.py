"""Apply the absolute meter scale (`python -m measure.rescale [--dry-run]`).

Until s7 every length was in PLACEHOLDER meters (params rink_length = 60 m / measured board px). The goal-speed clip
gave the real scale (measure/results/scale.json, `factor_vs_placeholder_median` ~1.0873 real m per placeholder m).
This tool multiplies every length, speed and acceleration by that factor in one consistent step and leaves times,
rates per second (1/s), ratios, masses, counts and angles alone.

The unit rule (`unit_power`): a unit scales by factor**p where p is the power of meters in it ("m", "m/s",
"m/s^2" -> 1; "1/s", "s", "ratio", "kg", "rad", "rad/s", "rad/s^2", "count", "goals" -> 0). Anything else is an error.

What it rewrites:
- params.json: entries with a length unit whose value came from the clips (provenance measured-shape / measured /
  fitted) plus rink_length (it defined the placeholder scale; now provenance "measured"). Placeholder lengths
  (gravity, heights, speed caps, lift, spawn) were invented in real-world terms, not read off the placeholder scale,
  so they stay. speed_display_scale is set to 1.0 (sim meters are now real meters).
- calib/data/real/*.scenario.json + CSV + manifest.json, calib/data/skate/*.scenario.json + CSV: positions and
  velocities in the initial state, x/y/z in the CSVs, v0_m_s in the manifest.
- measure/results/*.map.json, *.track.json (m_per_px*, *_m, max_speed_m_s; scale_source) and *.traj.csv (x, y, z).
Not touched: params.placeholder.json and calib/data/params.synthetic.json (tests pin them), params.fitted.json,
params.player.json, params.stick.json (references at the placeholder scale), calib/data/stick/.

It refuses to run twice (each file carries the SCALE_TAG marker after rescaling).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

from .shape_params import set_entries

ROOT = Path(__file__).resolve().parent.parent
SCALE_JSON = ROOT / "measure" / "results" / "scale.json"
SCALE_TAG = "REAL METERS"
SCALED_PROVENANCE = ("measured-shape", "measured", "fitted")

_POW0 = {"s", "1/s", "ratio", "kg", "rad", "rad/s", "rad/s^2", "count", "goals"}
_POW1 = {"m", "m/s", "m/s^2"}


def unit_power(unit: str) -> int:
    if unit in _POW1:
        return 1
    if unit in _POW0:
        return 0
    raise ValueError(f"unknown unit {unit!r}")


def load_factor(path: Path = SCALE_JSON) -> tuple[float, dict]:
    s = json.loads(Path(path).read_text())
    m = s["m_per_px"]
    return float(m["factor_vs_placeholder_median"]), m


def scale_note(factor: float, m: dict) -> str:
    return (
        f"{SCALE_TAG}: rescaled x{factor:.4f} from placeholder meters (python -m measure.rescale). Scale ESTIMATED from the "
        f"in-game SHOT SPEED readouts of {m['n']} ground goals (median {m['median']:.5f} m/px, spread (CV) {100 * m['cv']:.1f}%), "
        "ASSUMING the mph readout is in real units (measure/results/scale.json)."
    )


# ---------- params.json ----------

def rescale_param_entries(params: dict, factor: float, note: str) -> dict:
    """Return the changed entries {name: entry} (does not write)."""
    out = {}
    for k, e in params.items():
        if note.split(":")[0] in e.get("note", ""):
            raise RuntimeError(f"params already rescaled ({k} carries the '{SCALE_TAG}' marker)")
        p = unit_power(e.get("unit", "ratio"))
        if k == "speed_display_scale":
            out[k] = {"value": 1.0, "provenance": e["provenance"], "unit": e["unit"],
                      "note": "real-game speed per sim m/s for the mph/kph readout: 1.0, because sim meters are now real meters (rescaled to the goal-readout scale)"}
            continue
        if p == 0:
            continue
        if k == "rink_length":
            out[k] = {"value": round(e["value"] * factor, 4), "provenance": "measured", "unit": e["unit"],
                      "note": f"board to board along x = 60 m placeholder x {factor:.4f}. {note}"}
        elif e["provenance"] in SCALED_PROVENANCE:
            v = e["value"] * factor
            new = dict(e)
            new["value"] = round(v, 4) if abs(v) >= 1 else round(v, 6)
            new["note"] = f"{note} Before: {e['value']} placeholder m-units. " + e.get("note", "")
            out[k] = new
    return out


def rescale_params(path: Path, factor: float, note: str, write: bool = True) -> dict:
    params = json.loads(Path(path).read_text())["params"]
    changed = rescale_param_entries(params, factor, note)
    if write:
        set_entries(path, changed)
    return {k: (params[k]["value"], v["value"]) for k, v in changed.items()}


# ---------- scenarios / CSVs ----------

def rescale_state(st: dict, f: float) -> None:
    pk = st["puck"]
    for k in ("x", "y", "z", "vx", "vy", "vz"):
        pk[k] *= f
    for pl in st.get("players", []):
        for k in ("x", "y", "vx", "vy"):
            pl[k] *= f


def rescale_scenario_text(text: str, f: float) -> str:
    sc = json.loads(text)
    if SCALE_TAG in sc.get("description", ""):
        raise RuntimeError(f"scenario {sc.get('name')} already rescaled")
    rescale_state(sc["initial_state"], f)
    d = sc.get("description", "")
    d = re.sub(r"placeholder[- ]scale meters|placeholder meter scale", "real meters", d)
    sc["description"] = f"{d}; {SCALE_TAG} (x{f:.4f}, measure.rescale)"
    return json.dumps(sc, indent=2) + "\n"


def rescale_csv_text(text: str, f: float, cols=("x", "y", "z")) -> str:
    rows = list(csv.reader(io.StringIO(text)))
    head = rows[0]
    idx = [head.index(c) for c in cols if c in head]
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(head)
    for r in rows[1:]:
        if not r:
            continue
        for i in idx:
            v = float(r[i]) * f
            r[i] = "0" if v == 0 else repr(round(v, 7))
        w.writerow(r)
    return out.getvalue()


def rescale_pair_dir(d: Path, f: float, write: bool = True) -> int:
    n = 0
    for sp in sorted(d.glob("*.scenario.json")):
        new_sc = rescale_scenario_text(sp.read_text(), f)
        cp = sp.with_name(sp.name.replace(".scenario.json", ".csv"))
        new_csv = rescale_csv_text(cp.read_text(), f) if cp.exists() else None
        if write:
            if new_csv is not None:
                cp.write_text(new_csv)
            sp.write_text(new_sc)
        n += 1
    return n


def rescale_manifest(path: Path, f: float, note: str, write: bool = True) -> None:
    m = json.loads(path.read_text())
    if "meter_scale" in m:
        raise RuntimeError("manifest already rescaled")
    for s in m["segments"]:
        if "v0_m_s" in s:
            s["v0_m_s"] = round(s["v0_m_s"] * f, 3)
    m["description"] = m["description"].replace("at the PLACEHOLDER meter scale", "in REAL meters (rescaled, see meter_scale)")
    m["meter_scale"] = {"factor_vs_placeholder": f, "note": note}
    if write:
        path.write_text(json.dumps(m, indent=1) + "\n")


# ---------- measure/results map / track files ----------

_SKIP_KEYS = {"speed_cap_m_s"}  # a copy of a placeholder param (not scaled in params.json)


def _is_length_key(k: str) -> bool:
    return k not in _SKIP_KEYS and (k.startswith("m_per_px") or k.endswith("_m") or k.endswith("_m_s"))


def rescale_result_obj(o, f: float):
    if isinstance(o, dict):
        for k, v in o.items():
            if _is_length_key(k) and isinstance(v, (int, float)) and not isinstance(v, bool):
                o[k] = v * f
            else:
                rescale_result_obj(v, f)
    elif isinstance(o, list):
        for v in o:
            rescale_result_obj(v, f)
    return o


def rescale_result_json(path: Path, f: float, note: str, write: bool = True) -> None:
    d = json.loads(path.read_text())
    if SCALE_TAG in d.get("scale_source", ""):
        raise RuntimeError(f"{path.name} already rescaled")
    rescale_result_obj(d, f)
    d["scale_source"] = f"{note} Before: {d.get('scale_source', '')}"
    if write:
        compact = "\n" not in path.read_text().strip()
        path.write_text(json.dumps(d) if compact else json.dumps(d, indent=1) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m measure.rescale", description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
    root, write = a.root, not a.dry_run
    f, m = load_factor(root / "measure" / "results" / "scale.json")
    note = scale_note(f, m)
    print(f"[rescale] factor {f:.6f} (real m per placeholder m)")
    for k, (old, new) in rescale_params(root / "params.json", f, note, write).items():
        print(f"[rescale] params {k:22s} {old} -> {new}")
    real = root / "calib" / "data" / "real"
    print(f"[rescale] real segments: {rescale_pair_dir(real, f, write)}")
    rescale_manifest(real / "manifest.json", f, note, write)
    print(f"[rescale] skate runs: {rescale_pair_dir(root / 'calib' / 'data' / 'skate', f, write)}")
    res = root / "measure" / "results"
    for p in sorted(res.glob("*.map.json")) + sorted(res.glob("*.track.json")):
        rescale_result_json(p, f, note, write)
        print(f"[rescale] {p.name}")
    for p in sorted(res.glob("*.traj.csv")):
        if write:
            p.write_text(rescale_csv_text(p.read_text(), f))
        print(f"[rescale] {p.name}")
    print("[rescale] done" + (" (dry run, nothing written)" if not write else ""))


if __name__ == "__main__":
    main()
