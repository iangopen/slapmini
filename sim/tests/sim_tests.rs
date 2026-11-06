use sim::{step, ParamFile, Params, PlayerInput, State, DEFAULT_PARAMS_JSON};

#[test]
fn fitted_default_glide_is_linear_drag() {
    // The merged default: no constant friction, speed decays like exp(-k t).
    let p = Params::default();
    let mut s = State::new_default(&p, 0);
    s.puck.vx = 5.0;
    let steps = (1.0 / p.timestep).round() as usize;
    for _ in 0..steps {
        step(&mut s, &[], &p);
    }
    let expect = 5.0 * (-p.ice_linear_drag * steps as f64 * p.timestep).exp();
    assert!((s.puck.vx - expect).abs() < 0.01 * expect, "vx {} vs exp decay {}", s.puck.vx, expect);
}

/// Deterministic pseudo-random inputs (LCG) so the run exercises skating, dashing, lifting and stick swings.
fn scripted_inputs(tick: u64, n: usize) -> Vec<PlayerInput> {
    (0..n)
        .map(|i| {
            let mut s = tick.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407 + i as u64 * 977);
            let mut next = || {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64) / (1u64 << 31) as f64
            };
            PlayerInput {
                move_x: next() * 2.0 - 1.0,
                move_y: next() * 2.0 - 1.0,
                stick_target_angle: next() * 6.283 - 3.1415,
                dash: next() < 0.02,
                lift: next() < 0.2,
                stick_up: false,
            }
        })
        .collect()
}

fn run(steps: u64) -> State {
    let p = Params::default();
    let mut s = State::new_default(&p, 2);
    s.puck.vx = 5.0;
    s.puck.vy = 3.0;
    for _ in 0..steps {
        let inp = scripted_inputs(s.tick / 30, 2);
        step(&mut s, &inp, &p);
    }
    s
}

#[test]
fn determinism_100k_steps_byte_identical() {
    let a = run(100_000);
    let b = run(100_000);
    assert_eq!(a.to_bytes(), b.to_bytes());
    println!("determinism: 100000 steps, state hash run1={:016x} run2={:016x}, bytes={}", a.hash(), b.hash(), a.to_bytes().len());
    // Sanity: something actually happened.
    assert!(a.players[0].x.is_finite() && a.puck.x.is_finite());
}

/// Params whose SHAPE (ratio to rink_length) was measured from real clips; the meter scale is still a placeholder.
const MEASURED_SHAPE: [&str; 24] = [
    "rink_width", "rink_corner_radius", "net_width", "net_depth", "net_x", "blue_line_x", "faceoff_x", "faceoff_y", "faceoff_circle_radius",
    "puck_radius", "player_radius", "stick_length", "stick_thickness", "stick_inner",
    "stick_hand_x", "stick_hand_y", "stick_kink_x", "stick_kink_y", "stick_heel_x", "stick_heel_y", "stick_toe_x", "stick_toe_y",
    "stick_shaft_radius", "stick_heel_radius",
];

/// Fitted to real clips: puck (measure holdout), player/dash (s7 measure.skatefit), stick (s7b measure.stick).
const FITTED: [&str; 13] = [
    "board_restitution", "board_friction", "ice_linear_drag", "ice_friction_decel",
    "player_accel", "player_ice_drag", "dash_speed", "player_coast_rate", "dash_ramp_time", "stick_max_ang_decel",
    "stick_max_ang_speed", "stick_max_ang_accel", "stick_restitution",
];

/// Measured model-free (no fit): the player speed cap, and rink_length (the absolute scale from the goal readouts).
const MEASURED: [&str; 2] = ["player_max_speed", "rink_length"];

/// The pre-merge params (placeholder physics, measured shape), for tests about placeholder behaviour.
const PLACEHOLDER_JSON: &str = include_str!("../../params.placeholder.json");

/// Match rules Ian specified (provenance "rule"); the other match params are invented placeholders.
const MATCH_RULES: [&str; 2] = ["match_length_s", "mercy_lead"];

#[test]
fn params_file_complete_and_provenance_known() {
    let f = ParamFile::from_json(DEFAULT_PARAMS_JSON).unwrap();
    f.to_params().unwrap();
    assert_eq!(f.params.len(), Params::NAMES.len());
    for (k, e) in &f.params {
        let want = if MEASURED_SHAPE.contains(&k.as_str()) {
            "measured-shape"
        } else if FITTED.contains(&k.as_str()) {
            "fitted"
        } else if MEASURED.contains(&k.as_str()) {
            "measured"
        } else if MATCH_RULES.contains(&k.as_str()) {
            "rule"
        } else {
            "placeholder"
        };
        assert_eq!(e.provenance, want, "{k}: unexpected provenance");
    }
}

#[test]
fn puck_slides_to_stop_on_ice() {
    // Pinned: with the fitted (pure linear drag, no constant friction) default the puck only stops asymptotically.
    let p = Params::from_json(PLACEHOLDER_JSON).unwrap();
    let mut s = State::new_default(&p, 0);
    s.puck.vx = 5.0;
    for _ in 0..2000 {
        step(&mut s, &[], &p);
    }
    assert_eq!(s.puck.vx, 0.0);
    assert!(s.puck.x > 5.0 && s.puck.x < 13.0, "x={}", s.puck.x);
}

#[test]
fn puck_stays_in_rink_and_bounces() {
    let p = Params::default();
    let mut s = State::new_default(&p, 0);
    s.puck.y = 5.0;
    s.puck.vx = -40.0;
    s.puck.vy = 25.0;
    let mut max_r = 0.0f64;
    for _ in 0..3000 {
        step(&mut s, &[], &p);
        max_r = max_r.max(s.puck.x.abs() - p.rink_length / 2.0).max(s.puck.y.abs() - p.rink_width / 2.0);
    }
    assert!(max_r <= 0.0, "puck left rink by {max_r}");
}

#[test]
fn gravity_arc_lands() {
    let p = Params::default();
    let mut s = State::new_default(&p, 0);
    s.puck.vz = 4.0;
    let mut peak = 0.0f64;
    for _ in 0..240 {
        step(&mut s, &[], &p);
        peak = peak.max(s.puck.z);
    }
    let expected = 16.0 / (2.0 * p.gravity);
    assert!((peak - expected).abs() < 0.02, "peak {peak} vs {expected}");
    assert_eq!(s.puck.z, 0.0);
}

#[test]
fn stick_chases_target_not_snaps() {
    let p = Params::default();
    let mut s = State::new_default(&p, 1);
    s.players[0].stick_angle = 0.0;
    let inp = [PlayerInput { stick_target_angle: 2.0, ..Default::default() }];
    step(&mut s, &inp, &p);
    let a1 = s.players[0].stick_angle;
    assert!(a1 > 0.0 && a1 < 0.05, "first step moved {a1}");
    for _ in 0..240 {
        step(&mut s, &inp, &p);
    }
    assert!((s.players[0].stick_angle - 2.0).abs() < 1e-3);
    assert!(s.players[0].stick_omega.abs() < 1e-2);
}

#[test]
fn stick_swing_shoots_puck_and_caps_apply() {
    let p = Params::default();
    let mut s = State::new_default(&p, 1);
    let pl = &mut s.players[0];
    pl.x = 0.0;
    pl.y = 0.0;
    pl.stick_angle = -1.2;
    s.puck.x = 1.2;
    s.puck.y = 0.2;
    let inp = [PlayerInput { stick_target_angle: 1.2, ..Default::default() }];
    for _ in 0..60 {
        step(&mut s, &inp, &p);
    }
    let speed = (s.puck.vx.powi(2) + s.puck.vy.powi(2)).sqrt();
    assert!(speed > 3.0, "shot speed {speed}");
    assert!(speed <= p.max_puck_speed_ice_normal + 1e-9);
}

#[test]
fn lift_raises_puck() {
    let p = Params::default();
    let mut s = State::new_default(&p, 1);
    s.players[0].x = 0.0;
    s.players[0].y = 0.0;
    s.players[0].stick_angle = -1.2;
    s.puck.x = 1.2;
    s.puck.y = 0.2;
    let inp = [PlayerInput { stick_target_angle: 1.2, lift: true, ..Default::default() }];
    let mut peak = 0.0f64;
    for _ in 0..120 {
        step(&mut s, &inp, &p);
        peak = peak.max(s.puck.z);
    }
    assert!(peak > 0.05, "lift peak {peak}");
}

#[test]
fn goal_counts_once() {
    let p = Params::default();
    let mut s = State::new_default(&p, 0);
    s.puck.x = p.net_x - 2.0;
    s.puck.vx = 6.0;
    for _ in 0..600 {
        step(&mut s, &[], &p);
    }
    assert_eq!(s.score, vec![1, 0]);
}

#[test]
fn supports_six_players() {
    let p = Params::default();
    let mut s = State::new_default(&p, 6);
    for t in 0..1000 {
        step(&mut s, &scripted_inputs(t, 6), &p);
    }
    assert!(s.players.iter().all(|pl| pl.x.is_finite()));
}

/// Inputs simple enough to reproduce exactly in JS (see scripts/wasm_crosscheck.mjs).
pub fn crosscheck_inputs(t: u64) -> Vec<PlayerInput> {
    (0..2u64)
        .map(|k| PlayerInput {
            move_x: (((t / 60 + k) % 3) as f64) - 1.0,
            move_y: (((t / 90 + k) % 3) as f64) - 1.0,
            stick_target_angle: ((t + k * 100) % 628) as f64 * 0.01 - 3.14,
            dash: t % 240 == k * 7,
            lift: t % 500 < 20,
            stick_up: false,
        })
        .collect()
}

#[test]
fn crosscheck_hash_native() {
    let p = Params::default();
    let mut s = State::new_default(&p, 2);
    for t in 0..100_000u64 {
        step(&mut s, &crosscheck_inputs(t), &p);
    }
    println!("native crosscheck hash after 100000 steps: {:016x}", s.hash());
}
