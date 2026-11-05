//! The fixed-timestep physics. Pure function of (state, inputs, params).
//! Trig goes through `libm` so native and wasm builds produce identical bits.

use crate::params::Params;
use crate::state::{Player, PlayerInput, Puck, State};
use libm::{cos, sin, sqrt};

const TAU: f64 = core::f64::consts::TAU;
const PI: f64 = core::f64::consts::PI;
const EPS: f64 = 1e-9;
/// Landing vertical speeds below this stop bouncing.
const MIN_BOUNCE_VZ: f64 = 0.5;

#[inline]
fn wrap_angle(a: f64) -> f64 {
    let r = (a + PI).rem_euclid(TAU) - PI;
    if r < -PI { r + TAU } else { r }
}

#[inline]
fn len(x: f64, y: f64) -> f64 {
    sqrt(x * x + y * y)
}

#[inline]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo { lo } else if v > hi { hi } else { v }
}

/// Inward board normal and penetration for a circle of radius `r` at (x, y).
/// The rink is a rounded rectangle: a point is legal iff its distance to the
/// inner core rectangle (shrunk by the corner radius) is <= corner_radius - r.
fn board_contact(p: &Params, x: f64, y: f64, r: f64) -> Option<(f64, f64, f64)> {
    let rc = p.rink_corner_radius.max(r + EPS);
    let hx = (p.rink_length * 0.5 - rc).max(0.0);
    let hy = (p.rink_width * 0.5 - rc).max(0.0);
    let cx = clamp(x, -hx, hx);
    let cy = clamp(y, -hy, hy);
    let (dx, dy) = (x - cx, y - cy);
    let d = len(dx, dy);
    let limit = rc - r;
    if d > limit && d > EPS {
        Some((-dx / d, -dy / d, d - limit))
    } else {
        None
    }
}

/// Static segments of both nets (two sides + back each). Net 0 at +net_x, net 1 at -net_x.
fn net_segments(p: &Params) -> [(f64, f64, f64, f64); 6] {
    let (gx, w, d) = (p.net_x, p.net_width * 0.5, p.net_depth);
    let mut s = [(0.0, 0.0, 0.0, 0.0); 6];
    for (k, sign) in [1.0, -1.0].into_iter().enumerate() {
        let x0 = sign * gx;
        let x1 = sign * (gx + d);
        s[k * 3] = (x0, w, x1, w);
        s[k * 3 + 1] = (x0, -w, x1, -w);
        s[k * 3 + 2] = (x1, -w, x1, w);
    }
    s
}

/// Closest point on segment ab to point q.
#[inline]
fn closest_on_segment(ax: f64, ay: f64, bx: f64, by: f64, qx: f64, qy: f64) -> (f64, f64) {
    let (ex, ey) = (bx - ax, by - ay);
    let l2 = ex * ex + ey * ey;
    let t = if l2 > EPS { clamp(((qx - ax) * ex + (qy - ay) * ey) / l2, 0.0, 1.0) } else { 0.0 };
    (ax + ex * t, ay + ey * t)
}

fn puck_static_bounce(puck: &mut Puck, nx: f64, ny: f64, pen: f64, restitution: f64, tangential_loss: f64) {
    puck.x += nx * pen;
    puck.y += ny * pen;
    let vn = puck.vx * nx + puck.vy * ny;
    if vn < 0.0 {
        let (tx, ty) = (puck.vx - vn * nx, puck.vy - vn * ny);
        let keep_t = 1.0 - tangential_loss;
        let vn_new = -vn * restitution;
        puck.vx = tx * keep_t + vn_new * nx;
        puck.vy = ty * keep_t + vn_new * ny;
    }
}

/// Earliest fraction t in [0, 1] at which a circle of radius `r` moving from p0 by (mx, my)
/// first touches segment ab, with the contact normal (pointing from the segment to the circle).
/// Only approaching contacts count; a circle that already overlaps at t = 0 returns None
/// (the discrete overlap resolution handles it).
fn sweep_segment(
    ax: f64, ay: f64, bx: f64, by: f64, px: f64, py: f64, mx: f64, my: f64, r: f64,
) -> Option<(f64, f64, f64)> {
    let mut best: Option<(f64, f64, f64)> = None;
    let mut consider = |t: f64, nx: f64, ny: f64| {
        if (0.0..=1.0).contains(&t) && best.map_or(true, |b| t < b.0) {
            best = Some((t, nx, ny));
        }
    };
    // Flat sides of the capsule: lines offset by r along the segment normal.
    let (ex, ey) = (bx - ax, by - ay);
    let l = len(ex, ey);
    if l > EPS {
        let (ux, uy) = (ex / l, ey / l);
        let (mut nx, mut ny) = (-uy, ux);
        let mut d0 = (px - ax) * nx + (py - ay) * ny;
        if d0 < 0.0 {
            nx = -nx;
            ny = -ny;
            d0 = -d0;
        }
        let vn = mx * nx + my * ny;
        if d0 >= r && vn < -EPS {
            let t = (d0 - r) / -vn;
            let s = (px + mx * t - ax) * ux + (py + my * t - ay) * uy;
            if s >= 0.0 && s <= l {
                consider(t, nx, ny);
            }
        }
    }
    // Rounded ends: circles of radius r at both endpoints.
    for (cx, cy) in [(ax, ay), (bx, by)] {
        let (fx, fy) = (px - cx, py - cy);
        let a = mx * mx + my * my;
        let b = fx * mx + fy * my;
        let c = fx * fx + fy * fy - r * r;
        if a > EPS && c >= 0.0 && b < 0.0 {
            let disc = b * b - a * c;
            if disc >= 0.0 {
                let t = (-b - sqrt(disc)) / a;
                let (hx, hy) = (fx + mx * t, fy + my * t);
                let hl = len(hx, hy).max(EPS);
                consider(t, hx / hl, hy / hl);
            }
        }
    }
    best
}

/// Move the puck from `start` to its current position without passing through a net wall:
/// sweep the displacement against every net segment, stop at the first contact, bounce, and
/// continue the rest of the displacement along the reflected direction (a few iterations).
/// Bodies moving faster than one puck radius per substep would otherwise tunnel.
fn sweep_puck_nets(puck: &mut Puck, start: (f64, f64), nets: &[(f64, f64, f64, f64); 6], p: &Params) {
    let (mut px, mut py) = start;
    let (mut mx, mut my) = (puck.x - px, puck.y - py);
    for _ in 0..4 {
        let mut hit: Option<(f64, f64, f64)> = None;
        for &(ax, ay, bx, by) in nets {
            if let Some(h) = sweep_segment(ax, ay, bx, by, px, py, mx, my, p.puck_radius) {
                if hit.map_or(true, |b| h.0 < b.0) {
                    hit = Some(h);
                }
            }
        }
        let Some((t, nx, ny)) = hit else {
            puck.x = px + mx;
            puck.y = py + my;
            return;
        };
        px += mx * t + nx * 1e-9;
        py += my * t + ny * 1e-9;
        let rest = 1.0 - t;
        // Velocity bounce (same rule as the discrete contact), remaining displacement reflected.
        let vn = puck.vx * nx + puck.vy * ny;
        if vn < 0.0 {
            puck.vx -= (1.0 + p.net_restitution) * vn * nx;
            puck.vy -= (1.0 + p.net_restitution) * vn * ny;
        }
        let dn = mx * nx + my * ny;
        mx = (mx - (1.0 + p.net_restitution) * dn * nx) * rest;
        my = (my - (1.0 + p.net_restitution) * dn * ny) * rest;
    }
    puck.x = px;
    puck.y = py;
}

fn update_player_controls(pl: &mut Player, inp: &PlayerInput, p: &Params, dt: f64) {
    // Normalize move input to at most length 1.
    let (mut mx, mut my) = (clamp(inp.move_x, -1.0, 1.0), clamp(inp.move_y, -1.0, 1.0));
    let ml = len(mx, my);
    if ml > 1.0 {
        mx /= ml;
        my /= ml;
    }

    // Timers.
    pl.dash_timer = (pl.dash_timer - dt).max(0.0);
    pl.dash_cooldown = (pl.dash_cooldown - dt).max(0.0);

    // Dash (s10): Shift does nothing without a move key. With one, the press opens the dash window
    // (dash_timer, which the puck speed caps read as "dash shot") and starts a ramp: for the first
    // dash_ramp_time of the window the velocity is driven toward dash_speed along the move direction
    // (along the velocity once the key is released) at dash_speed / dash_ramp_time; it never slows
    // a player who is already faster. The ramp is clipped by the window (dash_duration).
    if inp.dash && !pl.dash_prev && pl.dash_cooldown <= 0.0 && ml > EPS {
        pl.dash_timer = p.dash_duration;
        pl.dash_cooldown = p.dash_cooldown;
    }
    pl.dash_prev = inp.dash;
    // Raised stick: instant raise and lower (placeholder timing); the stick keeps rotating while raised.
    pl.stick_up = inp.stick_up;
    let ramping = pl.dash_timer > 0.0 && p.dash_duration - pl.dash_timer < p.dash_ramp_time - 1e-12;

    let s0 = len(pl.vx, pl.vy);
    if ramping {
        let (dx, dy) = if ml > EPS {
            (mx / ml, my / ml)
        } else if s0 > EPS {
            (pl.vx / s0, pl.vy / s0)
        } else {
            (cos(pl.facing), sin(pl.facing))
        };
        if s0 < p.dash_speed {
            let (ex, ey) = (dx * p.dash_speed - pl.vx, dy * p.dash_speed - pl.vy);
            let el = len(ex, ey);
            let max = p.dash_speed / p.dash_ramp_time.max(dt) * dt;
            let k = if el > max { max / el } else { 1.0 };
            pl.vx += ex * k;
            pl.vy += ey * k;
        }
    } else if ml > EPS {
        // Skating: constant acceleration along the input, hard speed cap below.
        pl.vx += mx * p.player_accel * dt;
        pl.vy += my * p.player_accel * dt;
    } else if s0 > EPS {
        // Coasting (s10): speed-proportional decay (player_coast_rate, 1/s) plus a constant
        // deceleration (player_ice_drag, m/s^2) that brings the player to rest.
        let ns = (s0 - (p.player_ice_drag + p.player_coast_rate * s0) * dt).max(0.0);
        pl.vx *= ns / s0;
        pl.vy *= ns / s0;
    }
    let cap = if ramping { p.player_max_speed.max(p.dash_speed) } else { p.player_max_speed };
    let s = len(pl.vx, pl.vy);
    if s > cap {
        // A skating player is held at the cap (the per-step decay exceeds player_accel); after a dash
        // the extra speed fades at 2 x player_accel.
        let ns = (s - p.player_accel * 2.0 * dt).max(cap);
        pl.vx *= ns / s;
        pl.vy *= ns / s;
    }

    // Facing turns toward velocity.
    if s > 0.5 {
        let target = libm::atan2(pl.vy, pl.vx);
        let err = wrap_angle(target - pl.facing);
        let max = p.player_turn_rate * dt;
        pl.facing = wrap_angle(pl.facing + clamp(err, -max, max));
    }

    stick_chase(pl, inp.stick_target_angle, p, dt);
}

/// Stick chase: bang-bang profile limited by max angular speed, acceleration (speeding up) and a
/// separate deceleration (slowing down, s10), braking so it stops at the target. (An overshoot term
/// was tested and did not lower the held-out error: measure/s10_stick.py.)
fn stick_chase(pl: &mut Player, target: f64, p: &Params, dt: f64) {
    let w = pl.stick_omega;
    let err = wrap_angle(target - pl.stick_angle);
    let brake_speed = sqrt(2.0 * p.stick_max_ang_decel * err.abs());
    let desired = err.signum() * brake_speed.min(p.stick_max_ang_speed).min(err.abs() / dt);
    let dv = desired - w;
    let (acc, dec) = (p.stick_max_ang_accel * dt, p.stick_max_ang_decel * dt);
    if w == 0.0 || w * dv > 0.0 {
        pl.stick_omega = w + clamp(dv, -acc, acc);
    } else {
        // Slowing down (or reversing): the deceleration limit until omega reaches 0, then the
        // acceleration limit for the rest of the step.
        let slow = dv.abs().min(dec);
        if slow <= w.abs() {
            pl.stick_omega = w - w.signum() * slow;
        } else {
            let left = (1.0 - w.abs() / dec).max(0.0);
            let beyond = (dv.abs() - w.abs()).min(acc * left);
            pl.stick_omega = dv.signum() * beyond;
        }
    }
}

fn integrate_puck(puck: &mut Puck, p: &Params, h: f64) {
    let airborne = puck.z > 0.0 || puck.vz > 0.0;
    if airborne {
        puck.vz -= p.gravity * h;
        puck.z += puck.vz * h;
        let k = (1.0 - p.air_drag * h).max(0.0);
        puck.vx *= k;
        puck.vy *= k;
        if puck.z <= 0.0 {
            puck.z = 0.0;
            let bounce = -puck.vz * p.ice_restitution;
            puck.vz = if bounce > MIN_BOUNCE_VZ { bounce } else { 0.0 };
        }
    } else {
        puck.z = 0.0;
        puck.vz = 0.0;
        let s = len(puck.vx, puck.vy);
        if s > EPS {
            let ns = (s - (p.ice_friction_decel + p.ice_linear_drag * s) * h).max(0.0);
            puck.vx *= ns / s;
            puck.vy *= ns / s;
        } else {
            puck.vx = 0.0;
            puck.vy = 0.0;
        }
    }
    puck.x += puck.vx * h;
    puck.y += puck.vy * h;
}

fn stick_puck(pl: &mut Player, inp: &PlayerInput, puck: &mut Puck, p: &Params) {
    // A raised stick (w4) uses its own height range: it passes over a puck on the ice.
    let (z_min, z_max) = if pl.stick_up { (p.stick_up_z_min, p.stick_up_z_max) } else { (p.stick_z_min, p.stick_z_max) };
    if puck.z < z_min || puck.z > z_max {
        return;
    }
    let (ux, uy) = (cos(pl.stick_angle), sin(pl.stick_angle));
    let (ax, ay) = (pl.x + ux * p.stick_inner, pl.y + uy * p.stick_inner);
    let (bx, by) = (pl.x + ux * p.stick_length, pl.y + uy * p.stick_length);
    let (qx, qy) = closest_on_segment(ax, ay, bx, by, puck.x, puck.y);
    let (dx, dy) = (puck.x - qx, puck.y - qy);
    let d = len(dx, dy);
    let reach = p.puck_radius + p.stick_thickness;
    if d >= reach {
        return;
    }
    let (px0, py0) = (puck.x, puck.y);
    // Separation normal from stick to puck; if centered on the stick, push along the swing direction.
    let side = if pl.stick_omega >= 0.0 { 1.0 } else { -1.0 };
    let (nx, ny) = if d > EPS { (dx / d, dy / d) } else { (-uy * side, ux * side) };
    puck.x += nx * (reach - d);
    puck.y += ny * (reach - d);

    // Impulse (s10, "centre-on"): perpendicular to pivot -> puck centre, toward the puck's side of the
    // stick, at the stick point under the puck centre (radius |puck - pivot|), as if the stick had met
    // the puck centre-on. The measured exits leave within -7..+6 deg of that line; the old impulse
    // along the separation normal (above) left ~15 deg off it, because the blade's edge touches the
    // puck ~18 deg before the stick points at it.
    let sgn = if (px0 - pl.x) * -uy + (py0 - pl.y) * ux >= 0.0 { 1.0 } else { -1.0 };
    let (rx, ry) = (px0 - pl.x, py0 - pl.y);
    let rl = len(rx, ry).max(EPS);
    let (nx, ny) = (-ry / rl * sgn, rx / rl * sgn);

    // Stick velocity at contact = player velocity + omega x r.
    let (svx, svy) = (pl.vx - pl.stick_omega * ry, pl.vy + pl.stick_omega * rx);
    let vrel = (puck.vx - svx) * nx + (puck.vy - svy) * ny;
    if vrel >= 0.0 {
        return;
    }
    // Rod pivoting at the player center: I = m L^2 / 3, effective mass at the contact.
    let inertia = p.stick_mass * p.stick_length * p.stick_length / 3.0;
    let r_cross_n = rx * ny - ry * nx;
    let inv_m = 1.0 / p.puck_mass + r_cross_n * r_cross_n / inertia;
    let j = -(1.0 + p.stick_restitution) * vrel / inv_m;
    puck.vx += j / p.puck_mass * nx;
    puck.vy += j / p.puck_mass * ny;
    pl.stick_omega -= j * r_cross_n / inertia;

    // Shot classification and speed caps (air / ice-normal / ice-dash are separate in the real game).
    let dashing = pl.dash_timer > 0.0;
    let s = len(puck.vx, puck.vy);
    let mut cap;
    // PLACEHOLDER LOGIC (lift): lift held at contact launches the puck with vz proportional to
    // horizontal speed; dash + lift is capped to a low-speed chip. Real mechanic is unmeasured.
    if inp.lift {
        puck.vz = (p.lift_vz_factor * s).min(p.lift_max_vz).max(puck.vz);
        cap = p.max_puck_speed_air;
        if dashing {
            cap = cap.min(p.chip_max_speed);
        }
    } else if puck.z > 0.0 {
        cap = p.max_puck_speed_air;
    } else if dashing {
        cap = p.max_puck_speed_ice_dash;
    } else {
        cap = p.max_puck_speed_ice_normal;
    }
    if s > cap {
        puck.vx *= cap / s;
        puck.vy *= cap / s;
    }
}

fn player_puck_body(pl: &mut Player, puck: &mut Puck, p: &Params) {
    if puck.z > p.player_height {
        return;
    }
    let (dx, dy) = (puck.x - pl.x, puck.y - pl.y);
    let d = len(dx, dy);
    let reach = p.puck_radius + p.player_radius;
    if d >= reach || d < EPS {
        return;
    }
    let (nx, ny) = (dx / d, dy / d);
    puck.x += nx * (reach - d);
    puck.y += ny * (reach - d);
    let vrel = (puck.vx - pl.vx) * nx + (puck.vy - pl.vy) * ny;
    if vrel < 0.0 {
        let j = -(1.0 + p.player_restitution) * vrel / (1.0 / p.puck_mass + 1.0 / p.player_mass);
        puck.vx += j / p.puck_mass * nx;
        puck.vy += j / p.puck_mass * ny;
        pl.vx -= j / p.player_mass * nx;
        pl.vy -= j / p.player_mass * ny;
    }
}

fn player_player(a: &mut Player, b: &mut Player, p: &Params) {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let d = len(dx, dy);
    let reach = 2.0 * p.player_radius;
    if d >= reach || d < EPS {
        return;
    }
    let (nx, ny) = (dx / d, dy / d);
    let half = (reach - d) * 0.5;
    a.x -= nx * half;
    a.y -= ny * half;
    b.x += nx * half;
    b.y += ny * half;
    let vrel = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
    if vrel < 0.0 {
        let j = -(1.0 + p.player_restitution) * vrel * 0.5; // equal masses
        a.vx -= j * nx;
        a.vy -= j * ny;
        b.vx += j * nx;
        b.vy += j * ny;
    }
}

/// Goal edge detection; returns the scoring team when a goal happened this step.
fn update_goals(state: &mut State, p: &Params) -> Option<u8> {
    let pk = &mut state.puck;
    let inside = |sign: f64| {
        let x = pk.x * sign;
        x > p.net_x && x < p.net_x + p.net_depth && pk.y.abs() < p.net_width * 0.5 && pk.z < p.net_height
    };
    let now = if inside(1.0) { Some(0u8) } else if inside(-1.0) { Some(1u8) } else { None };
    if let (Some(net), None) = (now, pk.in_net) {
        // Net 0 (+x) is defended by team 1, so a puck in it scores for team 0.
        let team = net as usize;
        if state.score.len() <= team {
            state.score.resize(team + 1, 0);
        }
        state.score[team] += 1;
        pk.in_net = now;
        return Some(net);
    }
    pk.in_net = now;
    None
}

/// Advance one fixed timestep. `inputs[i]` drives `state.players[i]`; missing inputs are idle.
/// With `state.match_state` set, the match rules (sim/src/rules.rs) wrap the physics; without it
/// (the default) this is free play: physics only, goals just count.
pub fn step(state: &mut State, inputs: &[PlayerInput], p: &Params) {
    if state.match_state.is_some() {
        crate::rules::step_match(state, inputs, p);
    } else {
        physics_step(state, inputs, p);
    }
}

/// One physics step (free play). Returns the team that scored this step, if any.
pub(crate) fn physics_step(state: &mut State, inputs: &[PlayerInput], p: &Params) -> Option<u8> {
    let dt = p.timestep;
    let n = p.substeps_n();
    let h = dt / n as f64;
    let idle = PlayerInput::default();

    for (i, pl) in state.players.iter_mut().enumerate() {
        let inp = inputs.get(i).unwrap_or(&idle);
        update_player_controls(pl, inp, p, dt);
    }

    let nets = net_segments(p);
    for _ in 0..n {
        // Players and sticks.
        for pl in state.players.iter_mut() {
            pl.x += pl.vx * h;
            pl.y += pl.vy * h;
            pl.stick_angle = wrap_angle(pl.stick_angle + pl.stick_omega * h);
            if let Some((nx, ny, pen)) = board_contact(p, pl.x, pl.y, p.player_radius) {
                pl.x += nx * pen;
                pl.y += ny * pen;
                let vn = pl.vx * nx + pl.vy * ny;
                if vn < 0.0 {
                    pl.vx -= (1.0 + p.player_restitution) * vn * nx;
                    pl.vy -= (1.0 + p.player_restitution) * vn * ny;
                }
            }
        }
        for i in 0..state.players.len() {
            for j in (i + 1)..state.players.len() {
                let (left, right) = state.players.split_at_mut(j);
                player_player(&mut left[i], &mut right[0], p);
            }
        }

        // Puck. Its whole substep displacement (integration plus stick/body pushes) is swept
        // against the nets so a fast puck cannot pass through a wall.
        let start = (state.puck.x, state.puck.y);
        integrate_puck(&mut state.puck, p, h);
        for (i, pl) in state.players.iter_mut().enumerate() {
            let inp = inputs.get(i).unwrap_or(&idle);
            player_puck_body(pl, &mut state.puck, p);
            stick_puck(pl, inp, &mut state.puck, p);
        }
        if state.puck.z < p.net_height {
            sweep_puck_nets(&mut state.puck, start, &nets, p);
            for &(ax, ay, bx, by) in &nets {
                let (qx, qy) = closest_on_segment(ax, ay, bx, by, state.puck.x, state.puck.y);
                let (dx, dy) = (state.puck.x - qx, state.puck.y - qy);
                let d = len(dx, dy);
                if d < p.puck_radius && d > EPS {
                    puck_static_bounce(&mut state.puck, dx / d, dy / d, p.puck_radius - d, p.net_restitution, 0.0);
                }
            }
        }
        if let Some((nx, ny, pen)) = board_contact(p, state.puck.x, state.puck.y, p.puck_radius) {
            puck_static_bounce(&mut state.puck, nx, ny, pen, p.board_restitution, p.board_friction);
        }
    }

    let goal = update_goals(state, p);
    state.tick += 1;
    goal
}
