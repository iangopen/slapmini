//! Movement times from top speed and from rest (w5b).
//!     cargo run --release -p sim --example movelab
//! Prints the current params.json next to the s10 movement (no turn term, accel 17.3594, coast 1.08146, brake = accel).

use sim::{step, Params, PlayerInput, State};

fn fresh(p: &Params, vx: f64) -> State {
    let mut s = State::new_default(p, 1);
    s.players[0].x = 0.0;
    s.players[0].y = 0.0;
    s.players[0].vx = vx;
    s.puck.x = 1e3; // out of the way
    s.puck.y = 15.0;
    s
}

/// Steps until `done`, on a rink too large to reach.
fn time(p: &Params, vx: f64, inp: PlayerInput, done: impl Fn(&State) -> bool) -> f64 {
    let mut s = fresh(p, vx);
    for k in 1..20_000 {
        step(&mut s, &[inp], p);
        if done(&s) {
            return k as f64 * p.timestep;
        }
    }
    f64::NAN
}

fn row(p: &Params) -> Vec<(&'static str, f64)> {
    let top = p.player_max_speed;
    let (w, s, a) = (
        PlayerInput { move_x: 1.0, ..Default::default() },
        PlayerInput { move_x: -1.0, ..Default::default() },
        PlayerInput { move_y: 1.0, ..Default::default() },
    );
    let diag = PlayerInput { move_x: 1.0, move_y: 1.0, ..Default::default() };
    let sp = |q: &State| q.players[0].vx.hypot(q.players[0].vy);
    let off = |q: &State, ang: f64| (q.players[0].vy.atan2(q.players[0].vx) - ang).abs().to_degrees();
    let h = core::f64::consts::FRAC_PI_2;
    vec![
        ("rest to 95% of top speed", time(p, 0.0, w, |q| sp(q) >= 0.95 * top)),
        ("rest to top speed", time(p, 0.0, w, |q| sp(q) >= top - 1e-9)),
        ("stop: top speed, opposite key, to speed 0", time(p, top, s, |q| q.players[0].vx <= 0.0)),
        ("reverse: top speed to top speed the other way", time(p, top, s, |q| q.players[0].vx <= -top + 1e-9)),
        ("turn 90 deg at top speed (within 10 deg of the key)", time(p, top, a, |q| off(q, h) < 10.0)),
        ("turn 90 deg at top speed (within 1 deg)", time(p, top, a, |q| off(q, h) < 1.0)),
        ("turn 45 deg at top speed (within 10 deg)", time(p, top, diag, |q| off(q, h / 2.0) < 10.0)),
        ("coast: top speed to half speed", time(p, top, PlayerInput::default(), |q| sp(q) <= top / 2.0)),
        ("coast: top speed to 1 m/s", time(p, top, PlayerInput::default(), |q| sp(q) <= 1.0)),
    ]
}

fn main() {
    let mut p = Params::default();
    p.rink_length = 1e5;
    p.rink_width = 1e5;
    p.net_x = 4e4;
    let mut old = p.clone();
    old.player_accel = 17.3594;
    old.player_coast_rate = 1.08146;
    old.player_turn_accel = 0.0;
    old.player_brake_accel = old.player_accel;
    println!("| time (s) | old | new |\n|---|---|---|");
    for ((name, a), (_, b)) in row(&old).into_iter().zip(row(&p)) {
        println!("| {name} | {a:.3} | {b:.3} |");
    }
    // Space from top speed in the sim: W held, then no keys.
    for (label, inp) in [("W + Space", PlayerInput { move_x: 1.0, brake: true, ..Default::default() }), ("Space, no keys", PlayerInput { brake: true, ..Default::default() })] {
        let mut s = fresh(&p, p.player_max_speed);
        print!("{label}: speed (m/s) at t =");
        for k in 0..=600 {
            if k % 60 == 0 && (k <= 240 || k % 120 == 0) {
                print!(" {:.1}s {:.3}", k as f64 * p.timestep, s.players[0].vx.hypot(s.players[0].vy));
            }
            step(&mut s, &[inp], &p);
        }
        println!();
    }
}
