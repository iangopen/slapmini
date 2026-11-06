//! s10 model tests: each fails if the pre-s10 behaviour comes back.
//! - coast: speed-proportional decay (old: constant deceleration), hard cap with a key held;
//! - dash: Shift alone does nothing (old: dashed along `facing`); the dash sets the speed toward dash_speed (old: added
//!   dash_speed on top of the current speed and raised the cap); the real dash runs stay within their measured error;
//! - stick: braking is harder than speeding up (old: symmetric);
//! - hit: the 7 measured hits leave perpendicular to pivot -> puck at 1.05-1.2 x the head speed (old: ~15 deg off).

use calib::{evaluate, parse_csv, Case, Scenario};
use sim::{step, Params, PlayerInput, State};

fn speed(s: &State) -> f64 {
    s.players[0].vx.hypot(s.players[0].vy)
}

fn one_player(p: &Params) -> State {
    let mut s = State::new_default(p, 1);
    s.players[0].x = -20.0;
    s.players[0].y = 0.0;
    s.puck.y = 0.4 * p.rink_width; // out of the way
    s
}

fn run(s: &mut State, inp: PlayerInput, steps: usize, p: &Params) {
    for _ in 0..steps {
        step(s, &[inp], p);
    }
}

const W: PlayerInput = PlayerInput { move_x: 1.0, move_y: 0.0, stick_target_angle: 0.0, dash: false, lift: false, stick_up: false, brake: false };

#[test]
fn coast_decays_in_proportion_to_speed_and_cap_is_hard() {
    let p = Params::default();
    let mut s = one_player(&p);
    // Hold W: reaches the cap and never exceeds it, also on a diagonal.
    let mut top: f64 = 0.0;
    for k in 0..240 {
        let inp = if k < 120 { W } else { PlayerInput { move_y: 1.0, ..W } };
        step(&mut s, &[inp], &p);
        top = top.max(speed(&s));
    }
    assert!(top <= p.player_max_speed + 1e-9, "above the cap: {top}");
    assert!((speed(&s) - p.player_max_speed).abs() < 1e-9, "not at the cap: {}", speed(&s));
    // Release: two equal intervals lose the same FRACTION of the speed (old model: the same amount).
    let v0 = speed(&s);
    run(&mut s, PlayerInput::default(), 60, &p);
    let v1 = speed(&s);
    run(&mut s, PlayerInput::default(), 60, &p);
    let v2 = speed(&s);
    let (r1, r2) = (v1 / v0, v2 / v1);
    assert!((r1 - r2).abs() < 0.01, "not proportional: {r1} vs {r2}");
    let expect = (-p.player_coast_rate * 0.5).exp();
    assert!((r1 - expect).abs() < 0.01, "decay per 0.5 s {r1}, expected {expect}");
    assert!(v0 - v1 > 1.5 * (v1 - v2), "absolute drops must shrink: {} then {}", v0 - v1, v1 - v2);
}

#[test]
fn dash_needs_a_move_key_and_sets_the_speed() {
    let p = Params::default();
    // Shift alone at rest: nothing happens, and no dash window opens.
    let mut s = one_player(&p);
    run(&mut s, PlayerInput { dash: true, ..Default::default() }, 30, &p);
    assert_eq!(speed(&s), 0.0);
    assert_eq!(s.players[0].dash_timer, 0.0);
    // From rest with W: dash_speed after the ramp (old: dash_speed impulse at once, then the cap decay).
    let mut s = one_player(&p);
    step(&mut s, &[PlayerInput { dash: true, ..W }], &p);
    assert!(s.players[0].dash_timer > 0.0, "dash window must open");
    assert!(speed(&s) < 0.5 * p.dash_speed, "a ramp, not an impulse: {}", speed(&s));
    let ramp_steps = (p.dash_ramp_time / p.timestep).ceil() as usize;
    let mut peak: f64 = 0.0;
    for _ in 0..ramp_steps + 5 {
        step(&mut s, &[PlayerInput { dash: true, ..W }], &p);
        peak = peak.max(speed(&s));
    }
    assert!((peak - p.dash_speed).abs() < 1e-6, "peak {peak} vs dash_speed {}", p.dash_speed);
    // At top speed the dash only lifts the speed to dash_speed (old: cap + dash_speed).
    let mut s = one_player(&p);
    run(&mut s, W, 120, &p);
    let mut peak: f64 = 0.0;
    step(&mut s, &[PlayerInput { dash: true, ..W }], &p);
    for _ in 0..60 {
        step(&mut s, &[PlayerInput { dash: true, ..W }], &p);
        peak = peak.max(speed(&s));
    }
    assert!(peak <= p.dash_speed + 1e-6 && peak > p.player_max_speed, "peak at speed {peak}");
    assert!((speed(&s) - p.player_max_speed).abs() < 1e-6, "back at the cap after the dash");
}

#[test]
fn real_dash_and_skate_runs_fit() {
    // Default params on the recorded runs (calib/data/skate/, real meters): the old models gave held-out pos RMSE
    // medians of 4.0-5.0 m on the dash clips; the s10 models give 0.14-0.34 m (measure/results/skate/s10_fits.json).
    let p = Params::default();
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/data/skate");
    let mut pos = Vec::new();
    for name in ["dashatrest_00", "dashatrest_01", "dashatrest_02", "dashwhilerunning_00", "dashwhilerunning_01", "wkeymovement_00", "wkeymovement_01"] {
        let sc = Scenario::from_json(&std::fs::read_to_string(format!("{dir}/{name}.scenario.json")).unwrap()).unwrap();
        let measured = parse_csv(&std::fs::read_to_string(format!("{dir}/{name}.csv")).unwrap()).unwrap();
        let e = evaluate(&[Case { scenario: sc, measured }], &p);
        pos.push(e[0].pos_rmse);
    }
    pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = pos[pos.len() / 2];
    assert!(med < 0.6, "median pos RMSE {med} m ({pos:?})");
}

#[test]
fn stick_brakes_harder_than_it_speeds_up() {
    let p = Params::default();
    assert!(p.stick_max_ang_decel > 2.0 * p.stick_max_ang_accel);
    let mut s = one_player(&p);
    s.players[0].stick_angle = 0.0;
    // Spin up toward a far target, then set the target to the current angle: time to stop ~ w / decel.
    let far = PlayerInput { stick_target_angle: 3.0, ..Default::default() };
    let mut up = 0;
    while s.players[0].stick_omega < p.stick_max_ang_speed - 1e-9 {
        step(&mut s, &[far], &p);
        up += 1;
        assert!(up < 200);
    }
    let a = s.players[0].stick_angle;
    let stop = PlayerInput { stick_target_angle: a, ..Default::default() };
    let mut down = 0;
    while s.players[0].stick_omega.abs() > 1e-9 {
        step(&mut s, &[stop], &p);
        down += 1;
        assert!(down < 200);
    }
    // Spin-up takes w/accel (~0.05 s = 6 steps); the stop takes ~w/decel (~0.011 s) plus the settle.
    assert!(down + 3 <= up, "stop {down} steps vs spin-up {up} (old symmetric chase: about equal)");
    let overshoot = (s.players[0].stick_angle - a).abs();
    assert!(overshoot < 0.25, "overshoot {overshoot} rad");
}

/// Measured hits (measure/results/stick_hits.json; placeholder-meter speeds, the ratio is scale-free).
const HITS: [(&str, f64, f64); 7] = [
    ("hit_01", -2.30, 0.884),
    ("hit_02", -4.02, 0.768),
    ("hit_03", -7.18, 1.637),
    ("hit_04", -7.18, 1.008),
    ("hit_05", 6.29, 1.202),
    ("hit_06", -2.05, 1.183),
    ("hit_07", 3.72, 0.959),
];

/// Exit offset (deg, + = ahead of the perpendicular to pivot -> puck) and exit / head speed for one recorded hit,
/// with the swing speed held at the measured value (same set-up as sim/examples/hitlab.rs).
fn sim_hit(name: &str, p: &Params) -> (f64, f64) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/data/stick");
    let sc = Scenario::from_json(&std::fs::read_to_string(format!("{dir}/{name}.scenario.json")).unwrap()).unwrap();
    let mut p = p.clone();
    p.stick_max_ang_accel = 0.0;
    p.stick_max_ang_decel = 0.0;
    p.ice_linear_drag = 0.0;
    p.ice_friction_decel = 0.0;
    p.max_puck_speed_ice_normal = 1e6;
    let mut s = sc.initial_state.clone();
    let (pl, pk) = (&s.players[0], &s.puck);
    let (px, py, bx, by, w) = (pl.x, pl.y, pk.x, pk.y, pl.stick_omega);
    let sg = w.signum();
    let cdir = (by - py).atan2(bx - px);
    s.players[0].stick_angle = cdir - sg * 0.3;
    let inp = [PlayerInput { stick_target_angle: s.players[0].stick_angle, ..Default::default() }];
    for _ in 0..200 {
        step(&mut s, &inp, &p);
        if s.puck.vx.hypot(s.puck.vy) > 1e-6 {
            break;
        }
    }
    for _ in 0..10 {
        step(&mut s, &inp, &p);
    }
    let e = s.puck.vy.atan2(s.puck.vx);
    let d = e - (cdir + sg * std::f64::consts::FRAC_PI_2);
    let off = d.sin().atan2(d.cos()).to_degrees() * sg;
    (off, s.puck.vx.hypot(s.puck.vy) / (w.abs() * (bx - px).hypot(by - py)))
}

#[test]
fn hits_leave_centre_on() {
    let p = Params::default();
    let mut err2 = 0.0;
    for (name, m_off, _m_ratio) in HITS {
        let (off, ratio) = sim_hit(name, &p);
        assert!(off.abs() < 1.0, "{name}: exit {off:.2} deg off the perpendicular (old model: -12..-16)");
        assert!((1.05..1.2).contains(&ratio), "{name}: exit/head speed {ratio:.3}");
        err2 += (off - m_off).powi(2);
    }
    let rms = (err2 / HITS.len() as f64).sqrt();
    assert!(rms < 6.0, "exit direction RMS vs measured {rms:.2} deg (old model ~16)");
}
