//! w5b skating model: separate acceleration along the key, against the velocity and across it.

use sim::{step, Params, PlayerInput, State};

fn state(p: &Params, x: f64, y: f64, vx: f64, vy: f64) -> State {
    let mut s = State::new_default(p, 1);
    let pl = &mut s.players[0];
    (pl.x, pl.y, pl.vx, pl.vy) = (x, y, vx, vy);
    s.puck.x = 0.0;
    s.puck.y = 0.45 * p.rink_width;
    s
}

fn keys(x: f64, y: f64) -> PlayerInput {
    PlayerInput { move_x: x, move_y: y, ..Default::default() }
}

#[test]
fn a_turn_keeps_the_speed_and_turn_accel_turns_faster() {
    let mut p = Params::default();
    let time = |p: &Params| {
        let mut s = state(p, -20.0, -15.0, p.player_max_speed, 0.0);
        for k in 0..600 {
            step(&mut s, &[keys(1.0, 1.0)], p);
            let pl = &s.players[0];
            assert!((pl.vx.hypot(pl.vy) - p.player_max_speed).abs() < 1e-9, "speed changed in the turn");
            if pl.vy.atan2(pl.vx).to_degrees() > 35.0 {
                return k;
            }
        }
        panic!("never turned");
    };
    let with = time(&p);
    p.player_turn_accel = 0.0;
    assert!(with < time(&p));
}

#[test]
fn opposite_key_uses_brake_accel_and_then_accel() {
    let mut p = Params::default();
    p.player_accel = 10.0;
    p.player_brake_accel = 40.0;
    let mut s = state(&p, 0.0, 0.0, 8.0, 0.0);
    step(&mut s, &[keys(-1.0, 0.0)], &p);
    assert!((s.players[0].vx - (8.0 - 40.0 * p.timestep)).abs() < 1e-9);
    for _ in 0..24 {
        step(&mut s, &[keys(-1.0, 0.0)], &p);
    }
    // 0.2 s of braking from 8 m/s, then player_accel the other way: 0.008 s later about -0.08 m/s.
    assert!(s.players[0].vx < 0.0 && s.players[0].vx > -0.2, "vx {}", s.players[0].vx);
}

#[test]
fn a_key_pressed_into_the_boards_still_slides_along_them() {
    let p = Params::default();
    let y = p.rink_width * 0.5 - p.player_radius - 0.01;
    let mut s = state(&p, -10.0, y, p.player_max_speed, 0.0);
    for _ in 0..120 {
        step(&mut s, &[keys(1.0, 1.0)], &p);
    }
    assert!(s.players[0].vx > 0.9 * p.player_max_speed, "slide lost: vx {}", s.players[0].vx);
}
