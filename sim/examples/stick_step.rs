//! Stick step response, for the video stick-chase fit (measure/stick_chase.py checks its Python emulation against this).
//! Run: cargo run --release -p sim --example stick_step -- PARAMS.json TARGET_RAD TOTAL_S
//! One player at rest, stick at angle 0, target angle TARGET_RAD from t = 0. Prints `t,angle,omega` per step.

use sim::{step, Params, PlayerInput, State};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = Params::from_json(&std::fs::read_to_string(&a[1]).expect("params file")).expect("params");
    let target: f64 = a[2].parse().expect("target");
    let total: f64 = a[3].parse().expect("seconds");
    let mut s = State::new_default(&p, 1);
    s.players[0].stick_angle = 0.0;
    s.players[0].facing = 0.0;
    s.puck.y = 0.4 * p.rink_width; // out of reach
    let inp = vec![PlayerInput { stick_target_angle: target, ..Default::default() }];
    println!("t,angle,omega");
    println!("0,0,0");
    for _ in 0..(total / p.timestep).ceil() as u64 {
        step(&mut s, &inp, &p);
        let pl = &s.players[0];
        println!("{},{},{}", s.time(&p), pl.stick_angle, pl.stick_omega);
    }
}
