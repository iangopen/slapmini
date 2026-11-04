//! Shot accuracy of the scripted striker: puck at rest, one chaser starting around it.
//! Prints, per start, the first hit's direction error vs the aim point and the launch speed.
use bots::{Bot, Kind, Scripted};
use sim::{Params, PlayerInput, State};
fn main() {
    let p = Params::default();
    let mut errs = vec![];
    let mut misses = 0;
    for k in 0..48 {
        let ang = k as f64 / 48.0 * std::f64::consts::TAU;
        let rad = 2.0 + (k % 3) as f64 * 3.0;
        let mut s = State::new_default(&p, 1);
        s.puck.x = -5.0;
        s.puck.y = (k % 5) as f64 - 2.0;
        s.players[0].x = s.puck.x + rad * ang.cos();
        s.players[0].y = s.puck.y + rad * ang.sin();
        let mut b = Scripted::new(Kind::Chaser);
        let aim = (p.net_x + 0.5 * p.net_depth, 0.0);
        let want = (aim.1 - s.puck.y).atan2(aim.0 - s.puck.x);
        let mut hit = None;
        for t in 0..600 {
            let a = b.act(&s, 0, &p);
            let pv = (s.puck.vx, s.puck.vy);
            let side = ((aim.0 - s.puck.x) * (s.players[0].y - s.puck.y) - (aim.1 - s.puck.y) * (s.players[0].x - s.puck.x)).signum();
            let pspeed = s.players[0].vx.hypot(s.players[0].vy);
            let sw = b.swinging(); let pre = (s.players[0].stick_omega, (s.puck.x - s.players[0].x).hypot(s.puck.y - s.players[0].y), s.players[0].stick_angle, (s.puck.y - s.players[0].y).atan2(s.puck.x - s.players[0].x));
            sim::step(&mut s, &[a.to_input(0)], &p);
            let dv = ((s.puck.vx - pv.0).powi(2) + (s.puck.vy - pv.1).powi(2)).sqrt();
            if dv > 3.0 {
                // let the follow-through finish
                for _ in 0..12 { let a = b.act(&s, 0, &p); sim::step(&mut s, &[a.to_input(0)], &p); }
                hit = Some((t, s.puck.vy.atan2(s.puck.vx) - want, s.puck.vx.hypot(s.puck.vy))); println!("   side {side} player speed {pspeed:.2} swinging {} pre omega {:.2} r {:.3} stick {:.3} puckdir {:.3}", sw, pre.0, pre.1, pre.2, pre.3);
                break;
            }
        }
        match hit {
            Some((t, e, v)) => { println!("start ang {:5.2} r {:.0}: hit at step {t:3}, dir err {:+.3} rad, speed {:.1}", ang, rad, e, v); errs.push(e.abs()); }
            None => { println!("start ang {:5.2} r {:.0}: no hit", ang, rad); misses += 1; }
        }
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("hits {} misses {misses} median |err| {:.3} p90 {:.3}", errs.len(), errs[errs.len()/2], errs[errs.len()*9/10]);
    let _ = PlayerInput::default();
}
