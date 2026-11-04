//! Puck tunneling stress test: fire the puck at every net segment (from both sides) and at
//! the boards (straights and corners) at each speed cap, from many angles, and check after
//! every step that it never crossed a net wall, never overlaps one, and never left the rink.
//! Bots in training will find any hole in the walls, so this has to hold at the caps.

use sim::{step, Params, State};

const DEG: f64 = std::f64::consts::PI / 180.0;
const TOL: f64 = 1e-6;

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

/// True if segment p0-p1 crosses (or touches) segment s.
fn crosses(s: (f64, f64, f64, f64), p0: (f64, f64), p1: (f64, f64)) -> bool {
    let (ax, ay, bx, by) = s;
    let orient = |ox: f64, oy: f64, px: f64, py: f64, qx: f64, qy: f64| (px - ox) * (qy - oy) - (py - oy) * (qx - ox);
    let d1 = orient(ax, ay, bx, by, p0.0, p0.1);
    let d2 = orient(ax, ay, bx, by, p1.0, p1.1);
    let d3 = orient(p0.0, p0.1, p1.0, p1.1, ax, ay);
    let d4 = orient(p0.0, p0.1, p1.0, p1.1, bx, by);
    d1 * d2 <= 0.0 && d3 * d4 <= 0.0 && !(d1 == 0.0 && d2 == 0.0)
}

/// How far a puck-sized circle at (x, y) sticks out through the boards (<= 0 means inside).
fn board_overshoot(p: &Params, x: f64, y: f64) -> f64 {
    let rc = p.rink_corner_radius;
    let hx = p.rink_length * 0.5 - rc;
    let hy = p.rink_width * 0.5 - rc;
    let (dx, dy) = (x - x.clamp(-hx, hx), y - y.clamp(-hy, hy));
    (dx * dx + dy * dy).sqrt() - (rc - p.puck_radius)
}

struct Shot {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
}

/// Fire one shot for `steps` steps; return a description of the first violation, if any.
fn fire(p: &Params, nets: &[(f64, f64, f64, f64)], shot: &Shot, steps: usize) -> Option<String> {
    let mut s = State::new_default(p, 0);
    s.puck.x = shot.x;
    s.puck.y = shot.y;
    s.puck.vx = shot.vx;
    s.puck.vy = shot.vy;
    for k in 0..steps {
        let prev = (s.puck.x, s.puck.y);
        step(&mut s, &[], p);
        let cur = (s.puck.x, s.puck.y);
        if !(cur.0.is_finite() && cur.1.is_finite()) {
            return Some(format!("step {k}: non-finite position"));
        }
        let over = board_overshoot(p, cur.0, cur.1);
        if over > TOL {
            return Some(format!("step {k}: outside rink by {over:.4} m at ({:.3},{:.3})", cur.0, cur.1));
        }
        if s.puck.z < p.net_height {
            for (i, &seg) in nets.iter().enumerate() {
                if crosses(seg, prev, cur) {
                    return Some(format!("step {k}: crossed net segment {i} ({:.3},{:.3})->({:.3},{:.3})", prev.0, prev.1, cur.0, cur.1));
                }
                let d = seg_dist(seg, cur.0, cur.1);
                if d < p.puck_radius - TOL {
                    return Some(format!("step {k}: inside net segment {i} (dist {d:.4} < r) at ({:.3},{:.3})", cur.0, cur.1));
                }
            }
        }
    }
    None
}

fn legal_start(p: &Params, nets: &[(f64, f64, f64, f64)], x: f64, y: f64) -> bool {
    board_overshoot(p, x, y) < -0.01 && nets.iter().all(|&s| seg_dist(s, x, y) > p.puck_radius + 0.01)
}

fn speeds(p: &Params) -> [f64; 3] {
    [p.max_puck_speed_air, p.max_puck_speed_ice_normal, p.max_puck_speed_ice_dash]
}

#[test]
fn puck_never_tunnels_through_nets() {
    let p = Params::default();
    let nets = net_segments(&p);
    let (mut shots, mut fails) = (0usize, Vec::new());
    for (si, &(ax, ay, bx, by)) in nets.iter().enumerate() {
        let (ex, ey) = (bx - ax, by - ay);
        let l = (ex * ex + ey * ey).sqrt();
        let (nx, ny) = (-ey / l, ex / l);
        for ti in 0..=10 {
            let t = ti as f64 / 10.0;
            let (tx, ty) = (ax + ex * t, ay + ey * t);
            for side in [1.0, -1.0] {
                for ai in -17..=17 {
                    let a = ai as f64 * 5.0 * DEG;
                    // Direction toward the wall: -side*n rotated by a.
                    let (dx0, dy0) = (-side * nx, -side * ny);
                    let (dx, dy) = (dx0 * a.cos() - dy0 * a.sin(), dx0 * a.sin() + dy0 * a.cos());
                    for dist in [0.2, 0.5, 1.0] {
                        let (sx, sy) = (tx - dx * dist, ty - dy * dist);
                        if !legal_start(&p, &nets, sx, sy) {
                            continue;
                        }
                        for v in speeds(&p) {
                            shots += 1;
                            let shot = Shot { x: sx, y: sy, vx: dx * v, vy: dy * v };
                            if let Some(e) = fire(&p, &nets, &shot, 120) {
                                fails.push(format!("seg {si} t={t} side={side} angle={:.0}deg dist={dist} v={v}: {e}", a / DEG));
                            }
                        }
                    }
                }
            }
        }
    }
    println!("net tunneling: {shots} shots, {} failures", fails.len());
    for f in fails.iter().take(10) {
        println!("  {f}");
    }
    assert!(shots > 5000, "too few shots generated: {shots}");
    assert!(fails.is_empty(), "{} of {shots} shots tunneled or ended in a net wall", fails.len());
}

#[test]
fn puck_never_leaves_rink_through_boards() {
    let p = Params::default();
    let nets = net_segments(&p);
    let rc = p.rink_corner_radius;
    let hx = p.rink_length * 0.5 - rc;
    let hy = p.rink_width * 0.5 - rc;
    // Sample points on the inner board line (the puck-center limit) with their inward normals.
    let mut targets = Vec::new();
    for i in 0..=20 {
        let u = -hx + 2.0 * hx * i as f64 / 20.0;
        targets.push((u, hy + rc, 0.0, -1.0));
        targets.push((u, -hy - rc, 0.0, 1.0));
    }
    for i in 0..=10 {
        let u = -hy + 2.0 * hy * i as f64 / 10.0;
        targets.push((hx + rc, u, -1.0, 0.0));
        targets.push((-hx - rc, u, 1.0, 0.0));
    }
    for (cx, cy) in [(hx, hy), (-hx, hy), (hx, -hy), (-hx, -hy)] {
        for i in 0..=8 {
            let base = libm::atan2(cy, cx) - 45.0 * DEG;
            let th = base + 90.0 * DEG * i as f64 / 8.0;
            let (ux, uy) = (th.cos(), th.sin());
            if ux * cx.signum() >= -1e-9 && uy * cy.signum() >= -1e-9 {
                targets.push((cx + ux * rc, cy + uy * rc, -ux, -uy));
            }
        }
    }
    let (mut shots, mut fails) = (0usize, Vec::new());
    for &(tx, ty, nx, ny) in &targets {
        for ai in -17..=17 {
            let a = ai as f64 * 5.0 * DEG;
            let (dx0, dy0) = (-nx, -ny);
            let (dx, dy) = (dx0 * a.cos() - dy0 * a.sin(), dx0 * a.sin() + dy0 * a.cos());
            for dist in [0.3, 1.0, 3.0] {
                let (sx, sy) = (tx - dx * dist, ty - dy * dist);
                if !legal_start(&p, &nets, sx, sy) {
                    continue;
                }
                for v in speeds(&p) {
                    shots += 1;
                    let shot = Shot { x: sx, y: sy, vx: dx * v, vy: dy * v };
                    if let Some(e) = fire(&p, &nets, &shot, 240) {
                        fails.push(format!("target ({tx:.2},{ty:.2}) angle={:.0}deg dist={dist} v={v}: {e}", a / DEG));
                    }
                }
            }
        }
    }
    println!("board tunneling: {} targets, {shots} shots, {} failures", targets.len(), fails.len());
    for f in fails.iter().take(10) {
        println!("  {f}");
    }
    assert!(shots > 5000, "too few shots generated: {shots}");
    assert!(fails.is_empty(), "{} of {shots} shots left the rink or hit a net wall", fails.len());
}
