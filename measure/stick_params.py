"""Write params.stick.json from the stick results (`python -m measure.stick params`). params.json is not touched."""

from __future__ import annotations

import json

from .geometry import PARAMS_PATH, REPO

RESULTS = REPO / "measure" / "results"
OUT = REPO / "params.stick.json"


def run_params() -> dict:
    ch = json.loads((RESULTS / "stick_chase.json").read_text())
    sh = json.loads((RESULTS / "stick_shape.json").read_text())
    hi = json.loads((RESULTS / "stick_hits.json").read_text())
    base = {k: v["value"] for k, v in json.loads(PARAMS_PATH.read_text())["params"].items()}
    wmax = ch["fit_all"]["stick_max_ang_speed"]
    acc = ch["flick_starts"]["accel_best"]
    held = ch["held_out"]
    e_all = [f for f in hi["calib"]["one_param_folds"] if f["held_out"] == "all"][0]["stick_restitution"]
    e_folds = {f["held_out"]: round(f["stick_restitution"], 3) for f in hi["calib"]["one_param_folds"] if f["held_out"] != "all"}
    tip = sh["tip_half_max"]
    out = {
        "_comment": ("Stick params from video (session s7b; python -m measure.stick ...; see CLAUDE.md 'Stick (session s7b)'). Clips: "
                     "rotatinganddizzy (zoomed sweeps) and hittingpuck (hits, normal zoom). Lengths are meters at the PLACEHOLDER scale "
                     "(rink_length 60 m): multiply by the real/placeholder scale ratio when the absolute scale is applied. Angular values "
                     "are scale-free. Only the names listed here change; everything else stays in params.json (not merged)."),
        "params": {
            "stick_max_ang_speed": {"value": round(wmax, 3), "provenance": "fitted", "unit": "rad/s",
                                    "note": (f"rotatinganddizzy: hard cap seen as a flat plateau in every cap-limited move ({ch['plateau_rad_s']['n_frames']} frames, "
                                             f"median {ch['plateau_rad_s']['median']:.2f}, sd per frame {ch['plateau_rad_s']['sd']:.2f} = capture jitter); step "
                                             f"model with free onsets, held out by event: RMSE " + ", ".join(f"{h['rmse_deg_fitted']:.1f}" for h in held)
                                             + " deg vs placeholder " + ", ".join(f"{h['rmse_deg_placeholder']:.0f}" for h in held) + " deg. Scale-free.")},
            "stick_max_ang_accel": {"value": round(acc, 1), "provenance": "fitted", "unit": "rad/s^2",
                                    "note": (f"rotatinganddizzy: from the {ch['flick_starts']['n']} flick starts (rest -> cap within 4 frames). Stops are "
                                             f"harder: their fit is flat above ~1500 rad/s^2 (60 fps cannot resolve more), and the real stick overshoots "
                                             f"the final angle by {min(ch['overshoot_deg']):.1f}-{max(ch['overshoot_deg']):.1f} deg and settles back, which the "
                                             f"sim's symmetric bang-bang braking cannot reproduce. Scale-free.")},
            "stick_length": {"value": round(sh["stick_length_m"], 4), "provenance": "measured-shape", "unit": "m",
                             "note": (f"pivot (body centre) -> blade tip on the ice plane: half-max blade tip {tip['normal_zoom']['tip_half_max_px']:.2f} px "
                                      f"(hittingpuck) and {tip['zoomed']['tip_half_max_px']:.2f} px (rotatinganddizzy); the zoomed clip magnifies the "
                                      f"player model (camera moves closer), solved with a dolly-camera model (tip at {tip['tip_height_over_zoomed_camera_height']:.3f} "
                                      f"of the zoomed camera height). Consistent with the hits: resting-puck contact radii 0.95-1.06 m with perpendicular "
                                      f"exits. PLACEHOLDER meter scale.")},
            "stick_thickness": {"value": round(sh["stick_thickness_m"], 4), "provenance": "measured-shape", "unit": "m",
                                "note": ("collision HALF-thickness = half the blade's width at half max (rotatinganddizzy, de-rotated average of "
                                         f"{sh['blade']['frames']} frames, ice scale corrected for the blade height). PLACEHOLDER meter scale.")},
            "stick_restitution": {"value": round(e_all, 4), "provenance": "fitted", "unit": "ratio",
                                  "note": (f"hittingpuck, 7 hits, calib path objective with stick_mass FIXED at the params.json placeholder "
                                           f"{base['stick_mass']:g} kg and puck_mass {base['puck_mass']:g} kg (placeholders): stick_mass and restitution "
                                           f"are degenerate (a joint fit runs to a rigid stick, e = 0.138); only exit/head speed ~1.1-1.14 is measured. "
                                           f"Held out by swing speed: {e_folds}. Changing stick_mass requires refitting this.")},
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[stick params] wrote {OUT.name}: " + ", ".join(f"{k} = {v['value']}" for k, v in out["params"].items()))
    return out
