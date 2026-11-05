//! s10: determinism with the new models (skating, coasting, dashing, stick spins and shots) and a tunneling stress
//! test for stick shots fired at the nets and boards from close range (the centre-on contact moves the impulse point
//! to the stick point under the puck, and the stick turns at up to 26 rad/s).

use sim::{step, Params, PlayerInput, State};

const DEG: f64 = std::f64::consts::PI / 180.0;

/// Two players that chase the puck, dash every few seconds, release keys to coast, and spin the stick through the
/// puck when close. Pure function of the state and tick, using only + - * / sqrt (correctly rounded in Rust and JS),
/// so scripts/wasm_crosscheck.mjs reproduces it exactly.
pub fn s10_inputs(s: &State, t: u64) -> Vec<PlayerInput> {
    s.players
        .iter()
        .enumerate()
        .map(|(k, pl)| {
            let (dx, dy) = (s.puck.x - pl.x, s.puck.y - pl.y);
            let d = (dx * dx + dy * dy).sqrt().max(1e-9);
            let phase = (t + k as u64 * 97) % 600;
            let coasting = phase >= 480; // 1 s of every 5 s with no keys
            let (mx, my) = if coasting { (0.0, 0.0) } else { (dx / d, dy / d) };
            // Near the puck: spin the stick hard (alternating sense every 2.5 s), else hold it.
            let sense = if (t / 300 + k as u64) % 2 == 0 { 1.0 } else { -1.0 };
            let stick = if d < 2.0 { pl.stick_angle + sense * 2.5 } else { pl.stick_angle };
            PlayerInput { move_x: mx, move_y: my, stick_target_angle: stick, dash: phase % 150 < 3 && !coasting, lift: t % 700 < 15, stick_up: false }
        })
        .collect()
}

fn run(steps: u64) -> (State, u64) {
    let p = Params::default();
    let mut s = State::new_default(&p, 2);
    let mut hits = 0;
    for t in 0..steps {
        let v0 = s.puck.vx.hypot(s.puck.vy);
        let inp = s10_inputs(&s, t);
        step(&mut s, &inp, &p);
        if s.puck.vx.hypot(s.puck.vy) > v0 + 3.0 {
            hits += 1;
        }
    }
    (s, hits)
}

#[test]
fn s10_determinism_100k_steps() {
    let (a, hits) = run(100_000);
    let (b, _) = run(100_000);
    assert_eq!(a.to_bytes(), b.to_bytes());
    println!("s10 determinism: 100000 steps, hash {:016x}, shots {hits}, score {:?}", a.hash(), a.score);
    assert!(hits > 50, "the run must contain shots: {hits}");
    assert!(a.players.iter().all(|pl| pl.x.is_finite() && pl.stick_omega.is_finite()) && a.puck.x.is_finite());
}

fn net_segments(p: &Params) -> Vec<(f64, f64, f64, f64)> {
    let (gx, w, d) = (p.net_x, p.net_width * 0.5, p.net_depth);
    let mut s = Vec::new();
    for sign in [1.0, -1.0] {
        let (x0, x1) = (sign * gx, sign * (gx + d));
        s.push((x0, w, x1, w));
        s.push((x0, -w, x1, -w));
        s.push((x1, -w, x1, w));
    }
    s
}

fn seg_dist(s: (f64, f64, f64, f64), qx: f64, qy: f64) -> f64 {
    let (ax, ay, bx, by) = s;
    let (ex, ey) = (bx - ax, by - ay);
    let t = (((qx - ax) * ex + (qy - ay) * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
    ((qx - ax - ex * t).powi(2) + (qy - ay - ey * t).powi(2)).sqrt()
}

fn crosses(s: (f64, f64, f64, f64), p0: (f64, f64), p1: (f64, f64)) -> bool {
    let (ax, ay, bx, by) = s;
    let o = |ox: f64, oy: f64, px: f64, py: f64, qx: f64, qy: f64| (px - ox) * (qy - oy) - (py - oy) * (qx - ox);
    let (d1, d2) = (o(ax, ay, bx, by, p0.0, p0.1), o(ax, ay, bx, by, p1.0, p1.1));
    let (d3, d4) = (o(p0.0, p0.1, p1.0, p1.1, ax, ay), o(p0.0, p0.1, p1.0, p1.1, bx, by));
    d1 * d2 <= 0.0 && d3 * d4 <= 0.0 && !(d1 == 0.0 && d2 == 0.0)
}

fn board_overshoot(p: &Params, x: f64, y: f64) -> f64 {
    let rc = p.rink_corner_radius;
    let (hx, hy) = (p.rink_length * 0.5 - rc, p.rink_width * 0.5 - rc);
    let (dx, dy) = (x - x.clamp(-hx, hx), y - y.clamp(-hy, hy));
    (dx * dx + dy * dy).sqrt() - (rc - p.puck_radius)
}

/// A player at (x, y) swings the stick at full speed through a puck at `r` from its centre toward direction `aim`
/// (the centre-on push leaves perpendicular to player -> puck), for 90 steps. Returns the first violation.
fn swing(p: &Params, nets: &[(f64, f64, f64, f64)], x: f64, y: f64, aim: f64, sense: f64, r: f64, w0: f64) -> Option<Option<String>> {
    let mut s = State::new_default(p, 1);
    let q = &mut s.players[0];
    q.x = x;
    q.y = y;
    let to_puck = aim - sense * std::f64::consts::FRAC_PI_2;
    s.puck.x = x + r * to_puck.cos();
    s.puck.y = y + r * to_puck.sin();
    if board_overshoot(p, s.puck.x, s.puck.y) > -0.01 || nets.iter().any(|&g| seg_dist(g, s.puck.x, s.puck.y) < p.puck_radius + 0.01) {
        return None; // illegal start: not counted
    }
    let q = &mut s.players[0];
    q.stick_angle = to_puck - sense * 0.9;
    q.stick_omega = sense * w0;
    let mut hit = false;
    for k in 0..90 {
        let tgt = s.players[0].stick_angle + sense * 2.0;
        let prev = (s.puck.x, s.puck.y);
        step(&mut s, &[PlayerInput { stick_target_angle: tgt, ..Default::default() }], p);
        let cur = (s.puck.x, s.puck.y);
        if !(cur.0.is_finite() && cur.1.is_finite()) {
            return Some(Some(format!("step {k}: non-finite")));
        }
        let over = board_overshoot(p, cur.0, cur.1);
        if over > 1e-6 {
            return Some(Some(format!("step {k}: outside the rink by {over:.4} m")));
        }
        if s.puck.z < p.net_height {
            for (i, &g) in nets.iter().enumerate() {
                if crosses(g, prev, cur) {
                    return Some(Some(format!("step {k}: crossed net segment {i}")));
                }
                if seg_dist(g, cur.0, cur.1) < p.puck_radius - 1e-6 {
                    return Some(Some(format!("step {k}: inside net segment {i}")));
                }
            }
        }
        if s.puck.vx.hypot(s.puck.vy) > 5.0 {
            hit = true;
        }
    }
    if hit { Some(None) } else { Some(Some("NOHIT".to_string())) }
}

#[test]
fn stick_shots_never_tunnel() {
    let p = Params::default();
    let nets = net_segments(&p);
    let mut targets = Vec::new();
    // Points on every net segment (both faces) and along the boards, the shooter 1.5-4 m away facing them.
    for &(ax, ay, bx, by) in &nets {
        for i in 0..=6 {
            let t = i as f64 / 6.0;
            targets.push((ax + (bx - ax) * t, ay + (by - ay) * t));
        }
    }
    let rc = p.rink_corner_radius;
    let (hx, hy) = (p.rink_length * 0.5 - rc, p.rink_width * 0.5 - rc);
    for i in 0..=12 {
        let u = -hx + 2.0 * hx * i as f64 / 12.0;
        targets.push((u, hy + rc));
        targets.push((u, -hy - rc));
    }
    for (cx, cy) in [(hx, hy), (-hx, hy), (hx, -hy), (-hx, -hy)] {
        let th = cy.atan2(cx);
        targets.push((cx + rc * th.cos(), cy + rc * th.sin()));
    }
    let (mut shots, mut misses, mut fails) = (0usize, 0usize, Vec::new());
    for &(tx, ty) in &targets {
        for ai in 0..24 {
            let a = ai as f64 * 15.0 * DEG; // approach direction
            for dist in [1.5, 2.5, 4.0] {
                let (x, y) = (tx - dist * a.cos(), ty - dist * a.sin());
                if board_overshoot(&p, x, y) > -(p.player_radius + 0.1) || nets.iter().any(|&g| seg_dist(g, x, y) < p.player_radius + 0.3) {
                    continue;
                }
                for sense in [1.0, -1.0] {
                    for r in [0.7, 1.0, p.stick_length] {
                        for w0 in [p.stick_max_ang_speed, 0.5 * p.stick_max_ang_speed] {
                            let Some(res) = swing(&p, &nets, x, y, a, sense, r, w0) else { continue };
                            shots += 1;
                            if res.as_deref() == Some("NOHIT") {
                                misses += 1;
                            } else if let Some(e) = res {
                                fails.push(format!("target ({tx:.2},{ty:.2}) a={:.0} dist={dist} sense={sense} r={r} w={w0}: {e}", a / DEG));
                            }
                        }
                    }
                }
            }
        }
    }
    println!("stick-shot tunneling: {} targets, {shots} swings ({misses} without a > 5 m/s shot), {} failures", targets.len(), fails.len());
    for f in fails.iter().take(10) {
        println!("  {f}");
    }
    assert!(shots > 5000, "too few swings: {shots}");
    assert!(misses * 10 < shots, "most swings must shoot the puck: {misses} of {shots} did not");
    assert!(fails.is_empty(), "{} of {shots} stick shots tunneled", fails.len());
}
