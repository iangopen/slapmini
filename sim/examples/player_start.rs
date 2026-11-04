//! Player standing start, for the video-based player check (measure/player_motion.py).
//! Run: cargo run --release -p sim --example player_start -- PARAMS.json PUSH_S TOTAL_S
//! One player at rest, full +x move input for PUSH_S seconds, then no input. Prints `t,speed` per step.

use sim::{step, Params, PlayerInput, State};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let json = std::fs::read_to_string(&a[1]).expect("params file");
    let p = Params::from_json(&json).expect("params");
    let push: f64 = a[2].parse().expect("push seconds");
    let total: f64 = a[3].parse().expect("total seconds");
    let mut s = State::new_default(&p, 1);
    // Far end of the rink, puck out of the way, facing the push direction.
    s.players[0].x = -0.45 * p.rink_length;
    s.players[0].y = 0.0;
    s.puck.y = 0.4 * p.rink_width;
    let mut inp = vec![PlayerInput { stick_target_angle: 0.0, ..Default::default() }];
    println!("t,speed");
    println!("0,0");
    let n = (total / p.timestep).ceil() as u64;
    for _ in 0..n {
        let t = s.time(&p);
        inp[0].move_x = if t < push { 1.0 } else { 0.0 };
        step(&mut s, &inp, &p);
        let pl = &s.players[0];
        println!("{},{}", s.time(&p), (pl.vx * pl.vx + pl.vy * pl.vy).sqrt());
    }
}
