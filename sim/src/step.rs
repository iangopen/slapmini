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
/// Clearance left between the puck and a surface it was pushed out of.
const SKIN: f64 = 1e-6;
/// Overlap passes per substep (stick, body, nets, boards) before a pinch is declared.
const SOLVE_PASSES: usize = 6;
/// Overlap at the start of a substep beyond which a stick turns ghost; smaller ones are ordinary contact.
const GHOST_DEPTH: f64 = 0.02;
/// A puck already inside a body at the start of a substep (it came down into the body's height range) slides out at
/// this speed instead of jumping. Chosen, not measured.
const BODY_SETTLE_SPEED: f64 = 6.0;
/// Smallest rebound of a pinched stick (rad/s) and player (m/s). Chosen, not measured: enough to clear the puck for a
/// few steps against the stick and skating accelerations.
const PINCH_REBOUND_OMEGA: f64 = 8.0;
const PINCH_REBOUND_SPEED: f64 = 0.6;

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
        // Skating (w5b), split by the pressed direction. Along it: player_accel, or player_brake_accel while moving against
        // it. Sideways: player_turn_accel shrinks the velocity across the key. Moving with the key at all, that shrink
        // keeps the speed: it turns the velocity rather than slowing the player, so a player pressing a key into the
        // boards keeps sliding along them. Hard speed cap below.
        let (dx, dy) = (mx / ml, my / ml);
        let along = pl.vx * dx + pl.vy * dy;
        let (sx, sy) = (pl.vx - along * dx, pl.vy - along * dy);
        let brake = p.player_brake_accel * ml * dt;
        let mut along_new = along + if along < 0.0 { brake } else { p.player_accel * ml * dt };
        if along < 0.0 && along_new > 0.0 {
            // Crossed zero inside the step: the rest of the step speeds up.
            along_new = p.player_accel * ml * dt * (1.0 + along / brake);
        }
        let sl = len(sx, sy);
        let keep = if sl > EPS { (sl - p.player_turn_accel * ml * dt).max(0.0) / sl } else { 0.0 };
        let (plain_x, plain_y) = (dx * along_new + sx, dy * along_new + sy);
        pl.vx = dx * along_new + sx * keep;
        pl.vy = dy * along_new + sy * keep;
        let turned = len(pl.vx, pl.vy);
        if along >= 0.0 && turned > EPS {
            let k = len(plain_x, plain_y) / turned;
            pl.vx *= k;
            pl.vy *= k;
        }
    } else if s0 > EPS {
        // Coasting (s10): speed-proportional decay (player_coast_rate, 1/s) plus a constant
        // deceleration (player_ice_drag, m/s^2) that brings the player to rest.
        let ns = (s0 - (p.player_ice_drag + p.player_coast_rate * s0) * dt).max(0.0);
        pl.vx *= ns / s0;
        pl.vy *= ns / s0;
    }
    // Brake (w5b): while held the cap drops to brake_speed and speed above it decays at brake_decel, counted from the
    // speed before this step's key acceleration, so a held key steers but cannot hold the speed up. The keys still
    // accelerate up to the lower cap. A dash ramp is not limited; the brake takes over when the ramp ends.
    let slow = inp.brake && !ramping;
    let cap = if ramping {
        p.player_max_speed.max(p.dash_speed)
    } else if slow {
        p.player_max_speed.min(p.brake_speed)
    } else {
        p.player_max_speed
    };
    let s = len(pl.vx, pl.vy);
    if s > cap {
        // A skating player is held at the cap (the per-step decay exceeds player_accel); after a dash
        // the extra speed fades at 2 x player_accel.
        let ns = if slow { (s0 - p.brake_decel * dt).max(cap).min(s) } else { (s - p.player_accel * 2.0 * dt).max(cap) };
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

/// Per-step contact data for tests and tools (`step_with_stats`). Not part of the state.
#[derive(Clone, Debug, Default)]
pub struct ContactStats {
    /// Player indices whose stick touched the puck during this step.
    pub stick_contact: Vec<bool>,
    /// Deepest stick/body overlap seen before it was resolved, this step (m).
    pub max_penetration: f64,
    /// The same, for the stick alone and the body alone.
    pub max_stick_penetration: f64,
    pub max_body_penetration: f64,
    /// Sticks stopped by a pinch this step, and bodies pushed back by one.
    pub stick_blocks: u32,
    pub body_blocks: u32,
    /// Substeps a puck spent sliding out of a body it fell into (`BODY_SETTLE_SPEED`).
    pub body_settling: u32,
}

/// Stick chain in the stick's local frame (+x = the stick_angle direction, +y = 90 deg CCW):
/// hand, kink, heel, toe, and the half-width of the capsule after each of the first three.
#[inline]
fn stick_chain(p: &Params) -> ([(f64, f64); 4], [f64; 3]) {
    (
        [(p.stick_hand_x, p.stick_hand_y), (p.stick_kink_x, p.stick_kink_y), (p.stick_heel_x, p.stick_heel_y), (p.stick_toe_x, p.stick_toe_y)],
        [p.stick_shaft_radius, p.stick_heel_radius, p.stick_thickness],
    )
}

#[inline]
fn stick_active(pl: &Player, z: f64, p: &Params) -> bool {
    // A raised stick (w4) uses its own height range: it passes over a puck on the ice.
    let (z_min, z_max) = if pl.stick_up { (p.stick_up_z_min, p.stick_up_z_max) } else { (p.stick_z_min, p.stick_z_max) };
    !pl.stick_ghost && z >= z_min && z <= z_max
}

/// Sets or clears the ghost flag from the pose at the start of a substep. A solved contact ends a substep clear of the
/// stick, so a real overlap here can only come from a height change or a reset; it must not be pushed out in one jump.
fn update_stick_ghost(pl: &mut Player, puck: &Puck, p: &Params) {
    let gap = stick_puck_gap(pl, puck, p);
    if pl.stick_ghost {
        pl.stick_ghost = gap < 0.0;
    } else if stick_active(pl, puck.z, p) && gap < -GHOST_DEPTH {
        pl.stick_ghost = true;
    }
}

/// World -> stick-local coordinates of a point, for a stick pose (x, y, angle).
#[inline]
fn to_local(pose: (f64, f64, f64), x: f64, y: f64) -> (f64, f64) {
    let (c, s) = (cos(pose.2), sin(pose.2));
    let (dx, dy) = (x - pose.0, y - pose.1);
    (c * dx + s * dy, -s * dx + c * dy)
}

/// Smallest gap between a circle of radius `r` at local (lx, ly) and the chain (negative = overlap),
/// with the local unit normal from the chain to the circle. A centre exactly on the chain's axis
/// gets the capsule's left normal; callers flip it to the swing side.
fn chain_gap(p: &Params, lx: f64, ly: f64, r: f64) -> (f64, f64, f64) {
    let (pts, rad) = stick_chain(p);
    let mut best = (f64::INFINITY, 0.0, 0.0);
    for k in 0..3 {
        let (a, b) = (pts[k], pts[k + 1]);
        let (qx, qy) = closest_on_segment(a.0, a.1, b.0, b.1, lx, ly);
        let (dx, dy) = (lx - qx, ly - qy);
        let d = len(dx, dy);
        let gap = d - r - rad[k];
        if gap < best.0 {
            let (nx, ny) = if d > EPS {
                (dx / d, dy / d)
            } else {
                let (ex, ey) = (b.0 - a.0, b.1 - a.1);
                let l = len(ex, ey).max(EPS);
                (-ey / l, ex / l)
            };
            best = (gap, nx, ny);
        }
    }
    best
}

/// Gap between the puck and a player's stick at the current pose (negative = overlap). Height is not
/// checked here: see `stick_reaches`.
pub fn stick_puck_gap(pl: &Player, puck: &Puck, p: &Params) -> f64 {
    let (lx, ly) = to_local((pl.x, pl.y, pl.stick_angle), puck.x, puck.y);
    chain_gap(p, lx, ly, p.puck_radius).0
}

/// True if this player's stick can touch the puck at its current height.
pub fn stick_reaches(pl: &Player, puck: &Puck, p: &Params) -> bool {
    stick_active(pl, puck.z, p)
}

/// Swept stick contact over one substep. The stick goes from `pose0` to the player's current pose
/// while the puck goes from `q0` to its current position. In the stick's frame that is one straight
/// move of the puck centre (a chord: the arc differs by < 3 mm at the top swing speed), swept against
/// each capsule, so a fast swing cannot pass through the puck. Returns true on contact.
fn stick_puck(pl: &mut Player, pose0: (f64, f64, f64), q0: (f64, f64), inp: &PlayerInput, puck: &mut Puck, p: &Params, h: f64) -> bool {
    if !stick_active(pl, puck.z, p) {
        return false;
    }
    let pose1 = (pl.x, pl.y, pl.stick_angle);
    let (l0x, l0y) = to_local(pose0, q0.0, q0.1);
    let (l1x, l1y) = to_local(pose1, puck.x, puck.y);
    let (pts, rad) = stick_chain(p);
    let mut hit: Option<(f64, f64, f64)> = None;
    if chain_gap(p, l0x, l0y, p.puck_radius).0 >= 0.0 {
        for k in 0..3 {
            let (a, b) = (pts[k], pts[k + 1]);
            if let Some(c) = sweep_segment(a.0, a.1, b.0, b.1, l0x, l0y, l1x - l0x, l1y - l0y, p.puck_radius + rad[k]) {
                if hit.map_or(true, |o| c.0 < o.0) {
                    hit = Some(c);
                }
            }
        }
    }
    let (t, lnx, lny) = match hit {
        Some(c) => c,
        None => {
            // Already overlapping at the start (a stick lowered onto the puck, a puck dropping into the
            // height range) or at the end: treated as a contact at the end pose.
            let (g, nx, ny) = chain_gap(p, l1x, l1y, p.puck_radius);
            if g >= 0.0 {
                return false;
            }
            (1.0, nx, ny)
        }
    };

    // Contact pose.
    let da = wrap_angle(pose1.2 - pose0.2);
    let (cx, cy, ca) = (pose0.0 + (pose1.0 - pose0.0) * t, pose0.1 + (pose1.1 - pose0.1) * t, pose0.2 + da * t);
    let (qx, qy) = (q0.0 + (puck.x - q0.0) * t, q0.1 + (puck.y - q0.1) * t);
    let (co, si) = (cos(ca), sin(ca));
    let (sx, sy) = (co * lnx - si * lny, si * lnx + co * lny); // surface normal, stick -> puck
    let (rx, ry, nx, ny);
    if p.stick_hit_rule >= 0.5 {
        // Surface rule: along the contact normal, at the contact point.
        (rx, ry) = (qx - sx * p.puck_radius - cx, qy - sy * p.puck_radius - cy);
        (nx, ny) = (sx, sy);
    } else {
        // Centre-on rule (s10): perpendicular to pivot -> puck centre, on the puck's side of the surface.
        (rx, ry) = (qx - cx, qy - cy);
        let rl = len(rx, ry).max(EPS);
        let (px_, py_) = (-ry / rl, rx / rl);
        let sd = px_ * sx + py_ * sy;
        let sgn = if sd.abs() > 1e-6 { sd.signum() } else if pl.stick_omega >= 0.0 { 1.0 } else { -1.0 };
        (nx, ny) = (px_ * sgn, py_ * sgn);
    }

    // Stick velocity at contact = player velocity + omega x r.
    let (svx, svy) = (pl.vx - pl.stick_omega * ry, pl.vy + pl.stick_omega * rx);
    let vrel = (puck.vx - svx) * nx + (puck.vy - svy) * ny;
    if vrel < 0.0 {
        // Stick pivoting at the player center: I = m L^2 / 3, effective mass at the contact.
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
            // A capped puck is slower than the blade behind it: slow the stick to the puck's speed
            // along the normal, otherwise it would catch the puck again on every substep.
            let (svx, svy) = (pl.vx - pl.stick_omega * ry, pl.vy + pl.stick_omega * rx);
            let left = (puck.vx - svx) * nx + (puck.vy - svy) * ny;
            if left < 0.0 && r_cross_n.abs() > 1e-6 {
                pl.stick_omega += left / r_cross_n;
            }
        }
    }
    // The rest of the substep at the new velocity; the overlap pass places it clear of the end pose.
    puck.x = qx + puck.vx * h * (1.0 - t);
    puck.y = qy + puck.vy * h * (1.0 - t);
    true
}

/// Moves the puck out of the stick along the contact normal and removes any velocity into the
/// surface, so a puck can never stay inside or be re-hit by the same overlap. Returns the overlap.
fn stick_separate(pl: &mut Player, puck: &mut Puck, p: &Params) -> f64 {
    if !stick_active(pl, puck.z, p) {
        return 0.0;
    }
    let pose = (pl.x, pl.y, pl.stick_angle);
    let (lx, ly) = to_local(pose, puck.x, puck.y);
    let (g, lnx, lny) = chain_gap(p, lx, ly, p.puck_radius);
    if g >= 0.0 {
        return 0.0;
    }
    let (co, si) = (cos(pose.2), sin(pose.2));
    let (mut nx, mut ny) = (co * lnx - si * lny, si * lnx + co * lny);
    let (rx, ry) = (puck.x - pl.x, puck.y - pl.y);
    let (svx, svy) = (pl.vx - pl.stick_omega * ry, pl.vy + pl.stick_omega * rx);
    // A centre on the axis has no side of its own: take the side the stick is moving toward.
    if -g >= p.puck_radius + p.stick_thickness - 1e-9 && svx * nx + svy * ny < 0.0 {
        nx = -nx;
        ny = -ny;
    }
    puck.x += nx * (SKIN - g);
    puck.y += ny * (SKIN - g);
    let vrel = (puck.vx - svx) * nx + (puck.vy - svy) * ny;
    if vrel < 0.0 {
        puck.vx -= (1.0 + p.stick_restitution) * vrel * nx;
        puck.vy -= (1.0 + p.stick_restitution) * vrel * ny;
    }
    -g
}

/// Puts a puck that would pass into the body this substep at the moment of first touch (relative to the player's
/// end position), so `player_puck_body` bounces it from there instead of from deep inside.
fn body_sweep(pl: &Player, pose0: (f64, f64, f64), q0: (f64, f64), puck: &mut Puck, p: &Params) {
    if puck.z > p.player_height {
        return;
    }
    let reach = p.puck_radius + p.player_radius;
    let (fx, fy) = (q0.0 - pose0.0, q0.1 - pose0.1);
    let (mx, my) = (puck.x - pl.x - fx, puck.y - pl.y - fy);
    let (a, b, c) = (mx * mx + my * my, fx * mx + fy * my, fx * fx + fy * fy - reach * reach);
    if a < EPS || c < 0.0 || b >= 0.0 {
        return;
    }
    let disc = b * b - a * c;
    if disc < 0.0 {
        return;
    }
    let t = (-b - sqrt(disc)) / a;
    if t <= 1.0 {
        // A hair inside, so the ordinary body contact takes it.
        let k = 1.0 - 1e-9;
        puck.x = pl.x + (fx + mx * t) * k;
        puck.y = pl.y + (fy + my * t) * k;
    }
}

/// Returns the overlap it resolved.
fn player_puck_body(pl: &mut Player, puck: &mut Puck, p: &Params) -> f64 {
    if puck.z > p.player_height {
        return 0.0;
    }
    let (dx, dy) = (puck.x - pl.x, puck.y - pl.y);
    let d = len(dx, dy);
    let reach = p.puck_radius + p.player_radius;
    if d >= reach {
        return 0.0;
    }
    let (nx, ny) = if d > EPS { (dx / d, dy / d) } else { (1.0, 0.0) };
    puck.x += nx * (reach + SKIN - d);
    puck.y += ny * (reach + SKIN - d);
    let vrel = (puck.vx - pl.vx) * nx + (puck.vy - pl.vy) * ny;
    if vrel < 0.0 {
        let j = -(1.0 + p.player_restitution) * vrel / (1.0 / p.puck_mass + 1.0 / p.player_mass);
        puck.vx += j / p.puck_mass * nx;
        puck.vy += j / p.puck_mass * ny;
        pl.vx -= j / p.player_mass * nx;
        pl.vy -= j / p.player_mass * ny;
    }
    reach - d
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

/// Static surfaces: nets, then boards. They always win: the puck stays inside the rink.
fn puck_statics(puck: &mut Puck, nets: &[(f64, f64, f64, f64); 6], p: &Params) -> bool {
    let mut moved = false;
    if puck.z < p.net_height {
        for &(ax, ay, bx, by) in nets {
            let (qx, qy) = closest_on_segment(ax, ay, bx, by, puck.x, puck.y);
            let (dx, dy) = (puck.x - qx, puck.y - qy);
            let d = len(dx, dy);
            if d < p.puck_radius && d > EPS {
                puck_static_bounce(puck, dx / d, dy / d, p.puck_radius - d, p.net_restitution, 0.0);
                moved = true;
            }
        }
    }
    if let Some((nx, ny, pen)) = board_contact(p, puck.x, puck.y, p.puck_radius) {
        puck_static_bounce(puck, nx, ny, pen, p.board_restitution, p.board_friction);
        moved = true;
    }
    moved
}

/// Depth of the puck in a stick or body right now (0 = clear), for pinch detection.
fn stick_depth(pl: &Player, puck: &Puck, p: &Params) -> f64 {
    if stick_active(pl, puck.z, p) { (-stick_puck_gap(pl, puck, p)).max(0.0) } else { 0.0 }
}

fn body_depth(pl: &Player, puck: &Puck, p: &Params) -> f64 {
    if puck.z > p.player_height {
        return 0.0;
    }
    (p.puck_radius + p.player_radius - len(puck.x - pl.x, puck.y - pl.y)).max(0.0)
}

/// Pushes a player back along (nx, ny) by `d` (the puck stays) and removes its velocity into the puck.
fn push_player(pl: &mut Player, nx: f64, ny: f64, d: f64) {
    pl.x -= nx * (d + SKIN);
    pl.y -= ny * (d + SKIN);
    // Bounce off rather than lean on the pinned puck, so the contact ends instead of repeating every step.
    let vn = pl.vx * nx + pl.vy * ny;
    let out = -PINCH_REBOUND_SPEED.max(vn * 0.5);
    if vn > out {
        pl.vx += (out - vn) * nx;
        pl.vy += (out - vn) * ny;
    }
}

/// Clears every overlap of the puck. Pushes go along each contact normal; a wedge between two
/// surfaces squeezes the puck out through these passes. If the puck has nowhere to go (a pinch against
/// a board or net, or between stick and body) the puck stays and the player gives way: the stick
/// turns back to the last angle that fits and stops, and a body or a translating stick is pushed off.
fn resolve_overlaps(state: &mut State, poses: &[(f64, f64, f64)], settling: &[bool], nets: &[(f64, f64, f64, f64); 6], h: f64, p: &Params, stats: &mut ContactStats) {
    let puck = &mut state.puck;
    for (i, pl) in state.players.iter().enumerate() {
        if settling[i] {
            let (dx, dy) = (puck.x - pl.x, puck.y - pl.y);
            let d = len(dx, dy);
            let (nx, ny) = if d > EPS { (dx / d, dy / d) } else { (1.0, 0.0) };
            let push = body_depth(pl, puck, p).min(BODY_SETTLE_SPEED * h);
            puck.x += nx * push;
            puck.y += ny * push;
            stats.body_settling += 1;
        }
    }
    for _ in 0..SOLVE_PASSES {
        let mut any = false;
        for (i, pl) in state.players.iter_mut().enumerate() {
            let b = if settling[i] { 0.0 } else { player_puck_body(pl, puck, p) };
            let s = stick_separate(pl, puck, p);
            stats.max_penetration = stats.max_penetration.max(b).max(s);
            stats.max_stick_penetration = stats.max_stick_penetration.max(s);
            stats.max_body_penetration = stats.max_body_penetration.max(b);
            if s > 0.0 {
                stats.stick_contact[i] = true;
            }
            any |= b > 0.0 || s > 0.0;
        }
        // A static push may have moved the puck back into something: only a clean pass ends this.
        if !puck_statics(puck, nets, p) && !any {
            return;
        }
    }
    // Pinched: the puck stays where the statics put it, players give way. Rounds, because moving one
    // player off the puck can bring its own stick or body (or another player's) back onto it.
    for _ in 0..3 {
    for (i, pl) in state.players.iter_mut().enumerate() {
        if stick_depth(pl, puck, p) > 0.0 {
            // Largest fraction of this substep's turn that leaves the puck clear.
            let da = wrap_angle(pl.stick_angle - poses[i].2);
            let (mut lo, mut hi) = (0.0, 1.0);
            for _ in 0..12 {
                let mid = 0.5 * (lo + hi);
                pl.stick_angle = wrap_angle(poses[i].2 + da * mid);
                if stick_depth(pl, puck, p) > 0.0 { hi = mid } else { lo = mid }
            }
            pl.stick_angle = wrap_angle(poses[i].2 + da * lo);
            // The stick springs back off the pinned puck (the chase would otherwise press it again next step).
            let back = (p.stick_restitution * pl.stick_omega.abs()).max(PINCH_REBOUND_OMEGA);
            pl.stick_omega = if da >= 0.0 { -back } else { back };
            stats.stick_blocks += 1;
        }
        // Still inside at the start angle: the player's own motion did it. Push the player off.
        for _ in 0..3 {
            let d = stick_depth(pl, puck, p);
            if d <= 0.0 {
                break;
            }
            let (lx, ly) = to_local((pl.x, pl.y, pl.stick_angle), puck.x, puck.y);
            let (_, lnx, lny) = chain_gap(p, lx, ly, p.puck_radius);
            let (co, si) = (cos(pl.stick_angle), sin(pl.stick_angle));
            push_player(pl, co * lnx - si * lny, si * lnx + co * lny, d);
            stats.body_blocks += 1;
        }
        let d = if settling[i] { 0.0 } else { body_depth(pl, puck, p) };
        if d > 0.0 {
            let (dx, dy) = (puck.x - pl.x, puck.y - pl.y);
            let l = len(dx, dy).max(EPS);
            push_player(pl, dx / l, dy / l, d);
            stats.body_blocks += 1;
        }
    }
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

/// `step` for free play that also reports the contacts of this step.
pub fn step_with_stats(state: &mut State, inputs: &[PlayerInput], p: &Params, stats: &mut ContactStats) {
    physics_step_stats(state, inputs, p, stats);
}

/// One physics step (free play). Returns the team that scored this step, if any.
pub(crate) fn physics_step(state: &mut State, inputs: &[PlayerInput], p: &Params) -> Option<u8> {
    physics_step_stats(state, inputs, p, &mut ContactStats::default())
}

fn physics_step_stats(state: &mut State, inputs: &[PlayerInput], p: &Params, stats: &mut ContactStats) -> Option<u8> {
    stats.stick_contact.clear();
    stats.stick_contact.resize(state.players.len(), false);
    stats.max_penetration = 0.0;
    stats.max_stick_penetration = 0.0;
    stats.max_body_penetration = 0.0;
    stats.stick_blocks = 0;
    stats.body_blocks = 0;
    stats.body_settling = 0;
    let dt = p.timestep;
    let n = p.substeps_n();
    let h = dt / n as f64;
    let idle = PlayerInput::default();

    for (i, pl) in state.players.iter_mut().enumerate() {
        let inp = inputs.get(i).unwrap_or(&idle);
        update_player_controls(pl, inp, p, dt);
    }

    let nets = net_segments(p);
    let mut poses: Vec<(f64, f64, f64)> = Vec::with_capacity(state.players.len());
    let mut settling: Vec<bool> = Vec::with_capacity(state.players.len());
    for _ in 0..n {
        // Players and sticks.
        for pl in state.players.iter_mut() {
            update_stick_ghost(pl, &state.puck, p);
        }
        poses.clear();
        poses.extend(state.players.iter().map(|pl| (pl.x, pl.y, pl.stick_angle)));
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
        // Inside a body from the start of the substep, at the height it has now: it came down into the body's height range.
        settling.clear();
        settling.extend(poses.iter().map(|q| {
            state.puck.z <= p.player_height && p.puck_radius + p.player_radius - len(start.0 - q.0, start.1 - q.1) > GHOST_DEPTH
        }));
        for (i, pl) in state.players.iter_mut().enumerate() {
            let inp = inputs.get(i).unwrap_or(&idle);
            if !settling[i] {
                body_sweep(pl, poses[i], start, &mut state.puck, p);
            }
            if stick_puck(pl, poses[i], start, inp, &mut state.puck, p, h) {
                stats.stick_contact[i] = true;
            }
        }
        if state.puck.z < p.net_height {
            sweep_puck_nets(&mut state.puck, start, &nets, p);
        }
        resolve_overlaps(state, &poses, &settling, &nets, h, p, stats);
    }

    let goal = update_goals(state, p);
    state.tick += 1;
    goal
}
