//! Debug trace of one scripted game: `cargo run -p bots --example trace -- A B seconds`.
use bots::{Bot, Kind, Scripted};
use sim::{Params, PlayerInput};
fn kind(s: &str) -> Kind { Kind::ALL.into_iter().find(|k| k.name() == s).expect("bot name") }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p = Params::default();
    let mut s = bots::game::faceoff(&p);
    let mut bs = [Scripted::new(kind(&a[1])), Scripted::new(kind(&a[2]))];
    let secs: f64 = a[3].parse().unwrap();
    // Optional puck start: x y vx vy.
    if a.len() >= 8 {
        let g = |i: usize| a[i].parse::<f64>().unwrap();
        (s.puck.x, s.puck.y, s.puck.vx, s.puck.vy) = (g(4), g(5), g(6), g(7));
    }
    let mut inputs = vec![PlayerInput::default(); 2];
    for t in 0..(secs / p.timestep) as u64 {
        for i in 0..2 { inputs[i] = bs[i].act(&s, i, &p).to_input(s.players[i].team); }
        let pv = (s.puck.vx, s.puck.vy);
        sim::step(&mut s, &inputs, &p);
        let dv = ((s.puck.vx - pv.0).powi(2) + (s.puck.vy - pv.1).powi(2)).sqrt();
        if t % 30 == 0 || dv > 1.5 {
            let q = &s.players;
            println!("t={:6.3} puck ({:6.2},{:6.2}) v ({:6.2},{:6.2}) dv {:5.2} | p0 ({:6.2},{:6.2}) st {:5.2} tgt {:5.2} | p1 ({:6.2},{:6.2}) score {:?}",
              s.time(&p), s.puck.x, s.puck.y, s.puck.vx, s.puck.vy, dv, q[0].x, q[0].y, q[0].stick_angle, inputs[0].stick_target_angle, q[1].x, q[1].y, s.score);
        }
    }
}
