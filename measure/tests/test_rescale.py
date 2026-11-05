import json
import tempfile
import unittest
from pathlib import Path

from measure import rescale as R

F = 1.0873


def _params():
    e = lambda v, prov, unit: {"value": v, "provenance": prov, "unit": unit, "note": "n"}
    return {
        "timestep": e(0.0083, "placeholder", "s"),
        "gravity": e(9.81, "placeholder", "m/s^2"),
        "rink_length": e(60.0, "placeholder", "m"),
        "rink_width": e(38.1978, "measured-shape", "m"),
        "board_restitution": e(0.342796, "fitted", "ratio"),
        "ice_linear_drag": e(0.47254, "fitted", "1/s"),
        "puck_mass": e(0.17, "placeholder", "kg"),
        "player_accel": e(16.467, "fitted", "m/s^2"),
        "player_max_speed": e(10.755, "measured", "m/s"),
        "stick_max_ang_speed": e(26.173, "fitted", "rad/s"),
        "stick_max_ang_accel": e(513.9, "fitted", "rad/s^2"),
        "stick_thickness": e(0.0539, "measured-shape", "m"),
        "max_puck_speed_air": e(30.0, "placeholder", "m/s"),
        "speed_display_scale": e(1.0, "placeholder", "ratio"),
        "mercy_lead": e(5.0, "rule", "goals"),
    }


def _write_params(d: Path, params: dict) -> Path:
    lines = ",\n".join(f'    "{k}": {json.dumps(v)}' for k, v in params.items())
    p = d / "params.json"
    p.write_text('{\n  "params": {\n' + lines + "\n  }\n}\n")
    return p


class UnitRule(unittest.TestCase):
    def test_powers(self):
        for u in ("m", "m/s", "m/s^2"):
            self.assertEqual(R.unit_power(u), 1)
        for u in ("s", "1/s", "ratio", "kg", "rad", "rad/s", "rad/s^2", "count", "goals"):
            self.assertEqual(R.unit_power(u), 0)
        with self.assertRaises(ValueError):
            R.unit_power("furlong")


class Params(unittest.TestCase):
    def test_lengths_scale_rest_untouched(self):
        with tempfile.TemporaryDirectory() as t:
            p = _write_params(Path(t), _params())
            changed = R.rescale_params(p, F, "REAL METERS: test.")
            new = json.loads(p.read_text())["params"]
            # measured / fitted lengths, speeds, accelerations scale; rink_length becomes measured
            for k in ("rink_length", "rink_width", "player_accel", "player_max_speed", "stick_thickness"):
                self.assertAlmostEqual(new[k]["value"], _params()[k]["value"] * F, delta=1e-4, msg=k)
            self.assertEqual(new["rink_length"]["provenance"], "measured")
            # rates, ratios, masses, angles, times, rules: unchanged (bitwise)
            for k in ("timestep", "board_restitution", "ice_linear_drag", "puck_mass", "stick_max_ang_speed",
                      "stick_max_ang_accel", "mercy_lead"):
                self.assertEqual(new[k], _params()[k], k)
            # placeholder lengths invented in real-world terms stay
            self.assertEqual(new["gravity"]["value"], 9.81)
            self.assertEqual(new["max_puck_speed_air"]["value"], 30.0)
            self.assertEqual(new["speed_display_scale"]["value"], 1.0)
            self.assertIn("REAL METERS", new["rink_width"]["note"])
            self.assertNotIn("gravity", changed)
            with self.assertRaises(RuntimeError):  # refuses to run twice
                R.rescale_params(p, F, "REAL METERS: test.")


class Data(unittest.TestCase):
    def test_scenario_and_csv(self):
        sc = {"name": "a", "description": "x; placeholder-scale meters", "steps": 1,
              "initial_state": {"tick": 0, "puck": {"x": 1.0, "y": -2.0, "z": 0.5, "vx": 3.0, "vy": 0.0, "vz": 1.0, "in_net": None},
                                "players": [{"x": 4.0, "y": 5.0, "vx": 6.0, "vy": -7.0, "facing": 0.3, "stick_angle": 1.2}],
                                "score": [0, 0]},
              "inputs": [{"step": 0, "player": 0, "input": {"move_x": 1.0, "stick_target_angle": 2.0}}]}
        out = json.loads(R.rescale_scenario_text(json.dumps(sc), F))
        pk, pl = out["initial_state"]["puck"], out["initial_state"]["players"][0]
        self.assertAlmostEqual(pk["x"], F)
        self.assertAlmostEqual(pk["vz"], F)
        self.assertAlmostEqual(pl["vy"], -7.0 * F)
        self.assertEqual(pl["facing"], 0.3)
        self.assertEqual(pl["stick_angle"], 1.2)
        self.assertEqual(out["inputs"], sc["inputs"])
        self.assertIn("real meters", out["description"])
        with self.assertRaises(RuntimeError):
            R.rescale_scenario_text(json.dumps(out), F)
        c = R.rescale_csv_text("t,x,y,z,stick_angle\n0.5,1.0,-2.0,0,0.7\n", F).splitlines()
        t, x, y, z, a = c[1].split(",")
        self.assertEqual((t, z, a), ("0.5", "0", "0.7"))
        self.assertAlmostEqual(float(x), F, places=6)
        self.assertAlmostEqual(float(y), -2 * F, places=6)

    def test_result_keys(self):
        d = {"m_per_px": 0.06, "board_px": [1, 2], "extra": {"board_length_m": 60.0, "length_width_ratio": 1.57,
             "scales": {"m_per_px_ice": 0.06}}, "summary": {"max_speed_m_s": 40.0, "speed_cap_m_s": 45.0}}
        R.rescale_result_obj(d, F)
        self.assertAlmostEqual(d["m_per_px"], 0.06 * F)
        self.assertAlmostEqual(d["extra"]["board_length_m"], 60 * F)
        self.assertAlmostEqual(d["extra"]["scales"]["m_per_px_ice"], 0.06 * F)
        self.assertAlmostEqual(d["summary"]["max_speed_m_s"], 40 * F)
        self.assertEqual(d["summary"]["speed_cap_m_s"], 45.0)
        self.assertEqual(d["extra"]["length_width_ratio"], 1.57)
        self.assertEqual(d["board_px"], [1, 2])

    def test_factor_from_scale_json(self):
        f, m = R.load_factor()
        self.assertAlmostEqual(f, 1.0873, places=4)
        self.assertEqual(m["n"], 4)


if __name__ == "__main__":
    unittest.main()
