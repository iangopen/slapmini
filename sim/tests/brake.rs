//! w5b brake input (Space): a slow walk, not a stop.

use sim::{step, Params, PlayerInput, State};

fn speed(s: &State) -> f64 {
    s.players[0].vx.hypot(s.players[0].vy)
}

fn at_top_speed(p: &Params) -> State {
    let mut s = State::new_default(p, 1);
    s.players[0].x = -28.0;
    s.players[0].y = 0.0;
    s.puck.y = 0.4 * p.rink_width;
    for _ in 0..120 {
        step(&mut s, &[W], p);
    }
    assert!((speed(&s) - p.player_max_speed).abs() < 1e-9);
    s
}

const W: PlayerInput = PlayerInput { move_x: 1.0, move_y: 0.0, stick_target_angle: 0.0, dash: false, lift: false, stick_up: false, brake: false };
const WB: PlayerInput = PlayerInput { brake: true, ..W };
const B: PlayerInput = PlayerInput { move_x: 0.0, move_y: 0.0, stick_target_angle: 0.0, dash: false, lift: false, stick_up: false, brake: true };

#[test]
fn brake_defaults_to_false_in_json() {
    let i: PlayerInput = serde_json::from_str(r#"{"move_x":1,"move_y":0,"stick_target_angle":0}"#).unwrap();
    assert!(!i.brake);
}

#[test]
fn settles_at_brake_speed_and_keeps_moving() {
    let p = Params::default();
    let mut s = at_top_speed(&p);
    let mut last = speed(&s);
    let need = ((p.player_max_speed - p.brake_speed) / p.brake_decel / p.timestep).ceil() as usize;
    for k in 0..need {
        step(&mut s, &[WB], &p);
        assert!(speed(&s) <= last + 1e-12, "step {k}: speed rose");
        last = speed(&s);
    }
    assert!((speed(&s) - p.brake_speed).abs() < 1e-9, "settled at {}", speed(&s));
    let x0 = s.players[0].x;
    for _ in 0..120 {
        step(&mut s, &[WB], &p);
        assert!((speed(&s) - p.brake_speed).abs() < 1e-9);
    }
    assert!((s.players[0].x - x0 - p.brake_speed).abs() < 1e-6, "moved {} m in 1 s", s.players[0].x - x0);
}

#[test]
fn keys_accelerate_from_rest_to_the_lower_cap_in_the_pressed_direction() {
    let p = Params::default();
    let mut s = State::new_default(&p, 1);
    s.players[0].x = 0.0;
    s.players[0].y = 0.0;
    s.puck.y = 0.4 * p.rink_width;
    for _ in 0..60 {
        step(&mut s, &[PlayerInput { move_x: 0.0, move_y: -1.0, ..WB }], &p);
    }
    assert!((speed(&s) - p.brake_speed).abs() < 1e-9);
    assert!(s.players[0].vy < 0.0 && s.players[0].vx.abs() < 1e-9);
}

#[test]
fn no_keys_decays_to_rest_and_release_returns_to_normal() {
    let p = Params::default();
    let mut s = at_top_speed(&p);
    for _ in 0..(8.0 / p.timestep) as usize {
        step(&mut s, &[B], &p);
    }
    assert!(speed(&s) < 0.01, "still {}", speed(&s));
    let mut r = s.clone();
    for _ in 0..120 {
        step(&mut r, &[W], &p);
    }
    assert!((speed(&r) - p.player_max_speed).abs() < 1e-9, "release must give the full cap back");
}

#[test]
fn dash_works_while_braking() {
    let p = Params::default();
    let mut s = at_top_speed(&p);
    for _ in 0..120 {
        step(&mut s, &[WB], &p);
    }
    let mut peak: f64 = 0.0;
    for k in 0..30 {
        step(&mut s, &[PlayerInput { dash: k == 0, ..WB }], &p);
        peak = peak.max(speed(&s));
    }
    assert!((peak - p.dash_speed).abs() < 1e-6, "dash peak {peak}");
    for _ in 0..120 {
        step(&mut s, &[WB], &p);
    }
    assert!((speed(&s) - p.brake_speed).abs() < 1e-9, "back to the slow walk");
}
