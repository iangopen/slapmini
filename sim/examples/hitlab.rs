//! Stick-hit lab (s10): replays the 7 measured hits (calib/data/stick/) with a constant swing speed and
//! reports the sim's exit direction and speed ratio next to the measured ones.
//!
//!   cargo run --release -p sim --example hitlab -- PARAMS.json HITS.json [name=value ...]
//!
//! HITS.json = measure/results/stick_hits.json (the measured table). For every hit the stick starts
//! 0.3 rad short of the puck turning at the measured omega; the chase limits are set to 0 so omega
//! stays at the measured value until contact (the clip shows the swing speed, not the mouse), and
//! ice drag is off so the exit velocity is the contact velocity. The contact phase within a substep
//! is unknown, so each hit is run at 16 start phases spread over one substep of turning.
//! Prints one line per hit: measured offset / ratio, sim mean (min..max) offset and ratio.
//! Offset = exit direction minus the perpendicular to pivot -> resting puck, + = ahead of it
//! (outward), deg; ratio = exit speed / (|omega| * r_c), r_c = pivot -> resting puck.

use sim::{step, Params, PlayerInput, State};

const PHASES: usize = 16;

fn num(v: &serde_json::Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or_else(|| panic!("missing {k}"))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut p = Params::from_json(&std::fs::read_to_string(&a[1]).expect("params file")).expect("params");
    let hits: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[2]).expect("hits file")).expect("hits json");
    for kv in &a[3..] {
        let (k, v) = kv.split_once('=').expect("name=value");
        assert!(p.set(k, v.parse().expect("number")), "unknown param {k}");
    }
    p.stick_max_ang_accel = 0.0;
    p.stick_max_ang_decel = 0.0;
    p.ice_linear_drag = 0.0;
    p.ice_friction_decel = 0.0;
    for k in ["max_puck_speed_air", "max_puck_speed_ice_normal", "max_puck_speed_ice_dash"] {
        p.set(k, 1e6);
    }
    let h = p.timestep / p.substeps_n() as f64;
    println!("hit,meas_offset_deg,meas_ratio,sim_offset_deg,sim_offset_min,sim_offset_max,sim_ratio,sim_ratio_min,sim_ratio_max,contact_lead_deg");
    for (i, _) in hits["hits"].as_array().unwrap().iter().enumerate() {
        let name = format!("hit_{:02}", i + 1);
        let sc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("calib/data/stick/{name}.scenario.json")).unwrap()).unwrap();
        let st = &sc["initial_state"];
        let pl = &st["players"][0];
        let pk = &st["puck"];
        let (px, py, bx, by) = (num(pl, "x"), num(pl, "y"), num(pk, "x"), num(pk, "y"));
        let w = num(pl, "stick_omega");
        let sg = if w >= 0.0 { 1.0 } else { -1.0 };
        let cdir = (by - py).atan2(bx - px);
        let r_c = (bx - px).hypot(by - py);
        // The measured table is in the clip order of stick_hits.json; hit_NN follow that order.
        let m = &hits["hits"][i];
        let (m_off, m_ratio) = (num(m, "exit_dir_offset_deg"), num(m, "ratio"));
        let (mut offs, mut ratios, mut leads) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..PHASES {
            let mut s = State::new_default(&p, 1);
            s.puck.x = bx;
            s.puck.y = by;
            let q = &mut s.players[0];
            q.x = px;
            q.y = py;
            q.facing = 0.0;
            q.stick_omega = w;
            q.stick_angle = cdir - sg * (0.3 + w.abs() * h * k as f64 / PHASES as f64);
            let inp = [PlayerInput { stick_target_angle: q.stick_angle, ..Default::default() }];
            let mut lead = f64::NAN;
            let mut after = 0;
            for _ in 0..400 {
                let before = s.players[0].stick_angle;
                step(&mut s, &inp, &p);
                if s.puck.vx.hypot(s.puck.vy) > 1e-6 && lead.is_nan() {
                    // stick angle at the first touching step (start of the step), relative to the puck direction
                    lead = sim_wrap(cdir - before) * sg;
                }
                if !lead.is_nan() {
                    after += 1;
                    if after == 12 {
                        break;
                    }
                }
            }
            let e = s.puck.vy.atan2(s.puck.vx);
            offs.push(sim_wrap(e - (cdir + sg * std::f64::consts::FRAC_PI_2)).to_degrees() * sg);
            ratios.push(s.puck.vx.hypot(s.puck.vy) / (w.abs() * r_c));
            leads.push(lead.to_degrees());
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let mn = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "{name},{m_off:.2},{m_ratio:.3},{:.2},{:.2},{:.2},{:.3},{:.3},{:.3},{:.1}",
            mean(&offs), mn(&offs), mx(&offs), mean(&ratios), mn(&ratios), mx(&ratios), mean(&leads)
        );
    }
}

fn sim_wrap(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let r = (a + std::f64::consts::PI).rem_euclid(t) - std::f64::consts::PI;
    if r < -std::f64::consts::PI { r + t } else { r }
}
