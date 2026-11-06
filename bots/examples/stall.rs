//! Where do two chasers spend a goalless stretch? Prints the puck and players every 5 s.
use bots::{Bot, Kind, Scripted};
use sim::{step, Params, State};
fn main() {
    let p = Params::default();
    let mut s = State::new_default(&p, 2);
    s.puck.y = 0.7;
    let mut b = [Scripted::new(Kind::Chaser), Scripted::new(Kind::Chaser)];
    for t in 0..120 * 120 {
        let i = [b[0].act(&s, 0, &p).to_input(0), b[1].act(&s, 1, &p).to_input(1)];
        let g: u32 = s.score.iter().sum();
        step(&mut s, &i, &p);
        if s.score.iter().sum::<u32>() != g { println!("goal at {:.1}s {:?}", t as f64 / 120.0, s.score); b[0].reset(); b[1].reset(); let sc = s.score.clone(); s = State::new_default(&p, 2); s.score = sc; }
        if t % 600 == 0 {
            println!("{:4}s puck ({:.2},{:.2}) v {:.1} | a ({:.2},{:.2}) v {:.1} ghost {} | b ({:.2},{:.2}) v {:.1} ghost {}", t / 120, s.puck.x, s.puck.y, s.puck.vx.hypot(s.puck.vy), s.players[0].x, s.players[0].y, s.players[0].vx.hypot(s.players[0].vy), s.players[0].stick_ghost, s.players[1].x, s.players[1].y, s.players[1].vx.hypot(s.players[1].vy), s.players[1].stick_ghost);
        }
    }
}
