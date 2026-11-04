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

    // PLACEHOLDER LOGIC (dash): on press edge, add a velocity impulse along move input
    // (or facing if idle) and open a dash-shot window. Real dash behavior is unmeasured.
    if inp.dash && !pl.dash_prev && pl.dash_cooldown <= 0.0 {
        let (dx, dy) = if ml > EPS { (mx, my) } else { (cos(pl.facing), sin(pl.facing)) };
        let dl = len(dx, dy).max(EPS);
        pl.vx += dx / dl * p.dash_speed;
        pl.vy += dy / dl * p.dash_speed;
        pl.dash_timer = p.dash_duration;
        pl.dash_cooldown = p.dash_cooldown;
    }
    pl.dash_prev = inp.dash;

    // Skating.
    if ml > EPS {
        pl.vx += mx * p.player_accel * dt;
        pl.vy += my * p.player_accel * dt;
    } else {
        let s = len(pl.vx, pl.vy);
        if s > EPS {
            let ns = (s - p.player_ice_drag * dt).max(0.0);
            pl.vx *= ns / s;
            pl.vy *= ns / s;
        }
    }
    let cap = if pl.dash_timer > 0.0 { p.player_max_speed + p.dash_speed } else { p.player_max_speed };
    let s = len(pl.vx, pl.vy);
    if s > cap {
        // Decay toward the cap instead of clamping, so a dash fades out smoothly.
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

    // Stick chase: bang-bang profile limited by max angular speed and acceleration,
    // braking so it stops at the target instead of snapping or overshooting wildly.
    let err = wrap_angle(inp.stick_target_angle - pl.stick_angle);
    let brake_speed = sqrt(2.0 * p.stick_max_ang_accel * err.abs());
    let desired = err.signum() * brake_speed.min(p.stick_max_ang_speed).min(err.abs() / dt);
    let dw = clamp(desired - pl.stick_omega, -p.stick_max_ang_accel * dt, p.stick_max_ang_accel * dt);
    pl.stick_omega += dw;
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
    if puck.z < p.stick_z_min || puck.z > p.stick_z_max {
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
    // Normal from stick to puck; if centered on the stick, push along the stick's swing direction.
    let (nx, ny) = if d > EPS {
        (dx / d, dy / d)
    } else {
        let sgn = if pl.stick_omega >= 0.0 { 1.0 } else { -1.0 };
        (-uy * sgn, ux * sgn)
    };
    puck.x += nx * (reach - d);
    puck.y += ny * (reach - d);

    // Stick velocity at contact = player velocity + omega x r.
    let (rx, ry) = (qx - pl.x, qy - pl.y);
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

fn update_goals(state: &mut State, p: &Params) {
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
    }
    pk.in_net = now;
}

/// Advance one fixed timestep. `inputs[i]` drives `state.players[i]`; missing inputs are idle.
pub fn step(state: &mut State, inputs: &[PlayerInput], p: &Params) {
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

        // Puck.
        integrate_puck(&mut state.puck, p, h);
        for (i, pl) in state.players.iter_mut().enumerate() {
            let inp = inputs.get(i).unwrap_or(&idle);
            player_puck_body(pl, &mut state.puck, p);
            stick_puck(pl, inp, &mut state.puck, p);
        }
        if state.puck.z < p.net_height {
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

    update_goals(state, p);
    state.tick += 1;
}
