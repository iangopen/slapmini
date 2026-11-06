//! Scripted bots. All work in the acting player's canonical frame (attacks +x, own net at −x)
//! and return an `Action`, so they play either side. Deterministic: no RNG, state only in `self`.
//!
//! - `Chaser`: skates to a hitting spot beside the puck and swings the stick through it toward the net.
//! - `Defender`: stays between the puck and its own net; when the puck comes close (or gets behind it)
//!   it clears it up the ice toward the boards.
//! - `Cautious`: waits in its own half on the puck's line; attacks only once the puck comes to it.
//! - `Idle`: does nothing (stick held forward). A baseline for tests.

use crate::action::{Action, MOVE_DIRS};
use crate::frame::{wrap_angle, Frame};
use libm::{atan2, sqrt};
use sim::{Params, Player, Puck, State};

/// One interface for scripted bots and (outside this crate) policies.
pub trait Bot {
    fn name(&self) -> &'static str;
    /// Action for player `me` this step.
    fn act(&mut self, state: &State, me: usize, p: &Params) -> Action;
    /// Forget per-episode memory.
    fn reset(&mut self) {}
}

/// Scripted bot kinds by number (the C ABI uses these ids).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Chaser = 0,
    Defender = 1,
    Cautious = 2,
    Idle = 3,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Chaser, Kind::Defender, Kind::Cautious, Kind::Idle];
    pub fn from_id(id: u32) -> Option<Kind> {
        Kind::ALL.get(id as usize).copied()
    }
    pub fn name(self) -> &'static str {
        match self {
            Kind::Chaser => "chaser",
            Kind::Defender => "defender",
            Kind::Cautious => "cautious",
            Kind::Idle => "idle",
        }
    }
}

#[derive(Clone, Debug)]
enum Inner {
    Chaser(Chaser),
    Defender(Defender),
    Cautious(Cautious),
    Idle(Idle),
}

/// What a bot wants this step, with an ANALOG move (length ≤ 1). `Scripted` turns it into keys.
#[derive(Clone, Copy, Debug, Default)]
pub struct Wish {
    pub mx: f64,
    pub my: f64,
    pub stick: f64,
}

/// A scripted bot of any kind, by value (no boxing, cheap to clone per environment).
///
/// The bots plan with an analog move, but act through the same 9-way digital move a human has:
/// a sigma-delta modulator (`acc`) emits, each step, the key combination nearest to the
/// accumulated wish and subtracts it, so over a few steps the pressed keys average to the wish
/// (like a player tapping a key to hold a spot). Deterministic; reset with the bot.
#[derive(Clone, Debug)]
pub struct Scripted {
    inner: Inner,
    acc: (f64, f64),
}

impl Scripted {
    pub fn new(kind: Kind) -> Self {
        let inner = match kind {
            Kind::Chaser => Inner::Chaser(Chaser::default()),
            Kind::Defender => Inner::Defender(Defender::default()),
            Kind::Cautious => Inner::Cautious(Cautious::default()),
            Kind::Idle => Inner::Idle(Idle),
        };
        Scripted { inner, acc: (0.0, 0.0) }
    }
    /// True while the bot is in a committed stick swing (for diagnostics).
    pub fn swinging(&self) -> bool {
        match &self.inner {
            Inner::Chaser(b) => b.striker.swing_left > 0,
            Inner::Defender(b) => b.striker.swing_left > 0,
            Inner::Cautious(b) => b.striker.swing_left > 0,
            Inner::Idle(_) => false,
        }
    }
    pub fn kind(&self) -> Kind {
        match self.inner {
            Inner::Chaser(_) => Kind::Chaser,
            Inner::Defender(_) => Kind::Defender,
            Inner::Cautious(_) => Kind::Cautious,
            Inner::Idle(_) => Kind::Idle,
        }
    }
    /// The bot's analog wish (no key quantization).
    pub fn wish(&mut self, s: &State, me: usize, p: &Params) -> Wish {
        match &mut self.inner {
            Inner::Chaser(b) => b.wish(s, me, p),
            Inner::Defender(b) => b.wish(s, me, p),
            Inner::Cautious(b) => b.wish(s, me, p),
            Inner::Idle(b) => b.wish(s, me, p),
        }
    }
}

/// Sigma-delta step: add the wish, emit the nearest key combination (stop included), keep the rest.
fn dither(acc: &mut (f64, f64), mx: f64, my: f64) -> u8 {
    let (mx, my) = if mx.is_finite() && my.is_finite() { (mx, my) } else { (0.0, 0.0) };
    let (ax, ay) = (acc.0 + mx, acc.1 + my);
    let mut best = (0usize, f64::INFINITY);
    for (i, d) in MOVE_DIRS.iter().enumerate() {
        // Keys are compared as the sim applies them: diagonals are normalized to length 1.
        let l = if i == 0 { 1.0 } else { sqrt(d.0 * d.0 + d.1 * d.1) };
        let e = (ax - d.0 / l).powi(2) + (ay - d.1 / l).powi(2);
        if e < best.1 - 1e-12 {
            best = (i, e);
        }
    }
    let d = MOVE_DIRS[best.0];
    let l = if best.0 == 0 { 1.0 } else { sqrt(d.0 * d.0 + d.1 * d.1) };
    // Bounded memory: a wish the keys can't follow (e.g. a sudden reversal) does not pile up.
    *acc = ((ax - d.0 / l).clamp(-1.0, 1.0), (ay - d.1 / l).clamp(-1.0, 1.0));
    best.0 as u8
}

impl Bot for Scripted {
    fn name(&self) -> &'static str {
        self.kind().name()
    }
    fn act(&mut self, s: &State, me: usize, p: &Params) -> Action {
        let w = self.wish(s, me, p);
        Action { move_dir: dither(&mut self.acc, w.mx, w.my), stick_angle: w.stick, dash: false, lift: false, stick_up: false }
    }
    fn reset(&mut self) {
        match &mut self.inner {
            Inner::Chaser(b) => b.reset(),
            Inner::Defender(b) => b.reset(),
            Inner::Cautious(b) => b.reset(),
            Inner::Idle(_) => {}
        }
        self.acc = (0.0, 0.0);
    }
}

// ---------------------------------------------------------------------------------------------
// Shared canonical-frame helpers.

#[derive(Clone, Copy, Debug)]
struct V2(f64, f64);

impl V2 {
    fn sub(self, o: V2) -> V2 { V2(self.0 - o.0, self.1 - o.1) }
    fn add(self, o: V2) -> V2 { V2(self.0 + o.0, self.1 + o.1) }
    fn mul(self, k: f64) -> V2 { V2(self.0 * k, self.1 * k) }
    fn dot(self, o: V2) -> f64 { self.0 * o.0 + self.1 * o.1 }
    fn cross(self, o: V2) -> f64 { self.0 * o.1 - self.1 * o.0 }
    fn len(self) -> f64 { sqrt(self.dot(self)) }
    fn norm(self) -> V2 { let l = self.len(); if l > 1e-9 { self.mul(1.0 / l) } else { V2(1.0, 0.0) } }
    fn perp(self) -> V2 { V2(-self.1, self.0) }
    fn angle(self) -> f64 { atan2(self.1, self.0) }
}

/// The state as seen by one player (canonical frame).
struct View {
    me: Player,
    opp: Option<Player>,
    puck: Puck,
}

impl View {
    fn new(s: &State, me: usize) -> Self {
        let pl = &s.players[me];
        let f = Frame::for_team(pl.team);
        let opp = crate::obs::opponent_of(s, me).map(|j| f.player(&s.players[j]));
        View { me: f.player(pl), opp, puck: f.puck(&s.puck) }
    }
    fn pos(&self) -> V2 { V2(self.me.x, self.me.y) }
    fn vel(&self) -> V2 { V2(self.me.vx, self.me.vy) }
    fn puck_pos(&self) -> V2 { V2(self.puck.x, self.puck.y) }
    fn puck_vel(&self) -> V2 { V2(self.puck.vx, self.puck.vy) }
    /// Where the puck will be in `t` seconds, ignoring boards (linear drag only, a rough guess).
    fn puck_at(&self, t: f64, p: &Params) -> V2 {
        let k = p.ice_linear_drag.max(1e-6);
        let travel = (1.0 - libm::exp(-k * t)) / k;
        self.puck_pos().add(self.puck_vel().mul(travel))
    }
}

/// Keep a target point inside the rink (a box shrunk by `margin`).
fn clamp_to_rink(q: V2, p: &Params, margin: f64) -> V2 {
    let hx = p.rink_length * 0.5 - margin;
    let hy = p.rink_width * 0.5 - margin;
    V2(q.0.clamp(-hx, hx), q.1.clamp(-hy, hy))
}

/// Move input that brings the player to `target` and stops there (P-controller on velocity).
fn move_to(v: &View, target: V2, p: &Params, max_speed: f64) -> (f64, f64) {
    let to = target.sub(v.pos());
    let d = to.len();
    // Desired speed: brake so the player can stop at the target with the skating acceleration.
    let speed = sqrt(2.0 * p.player_accel * 0.6 * d).min(max_speed).min(d * 6.0);
    let want = to.norm().mul(speed);
    let err = want.sub(v.vel());
    let gain = 1.0 / (p.player_accel * p.timestep * 4.0).max(1e-6);
    let m = err.mul(gain);
    let l = m.len();
    if l < 0.05 { (0.0, 0.0) } else if l > 1.0 { (m.0 / l, m.1 / l) } else { (m.0, m.1) }
}

/// Extra rotation of a swung shot beyond the perpendicular to (puck − player) (rad), measured with
/// `cargo run --release -p bots --example shotlab` at the current params (re-measure after physics changes).
const SHOT_OVERTURN: f64 = 0.020;
/// Launch speed of a swung shot from rest (m/s), measured the same way.
const SHOT_SPEED: f64 = 27.8;

/// Time for the stick to turn `theta` radians from rest under the bang-bang chase (accelerate,
/// then cruise at the max angular speed).
fn swing_time(theta: f64, p: &Params) -> f64 {
    let (a, w) = (p.stick_max_ang_accel.max(1e-6), p.stick_max_ang_speed.max(1e-6));
    let theta_acc = w * w / (2.0 * a);
    if theta <= theta_acc { sqrt(2.0 * theta / a) } else { w / a + (theta - theta_acc) / w }
}

/// Mid-blade distance from the player center: where the stick should meet the puck.
fn hit_radius(p: &Params) -> f64 {
    0.5 * (p.stick_inner + p.stick_length) + 0.15
}

/// Stick-swing striker shared by all bots: approach a spot beside the puck, stick cocked back,
/// then swing through the puck so the blade moves toward `goal` at contact.
#[derive(Clone, Debug, Default)]
struct Striker {
    /// +1 / −1: which side of the shot line the player stands on (kept between steps for stability).
    side: f64,
    /// Steps left in a committed swing (a swing is never aborted halfway).
    swing_left: u32,
    swing_angle: f64,
}

impl Striker {
    /// `loose`: any direction within ~0.5 rad of `goal` will do (clears), so swing early and on the move.
    fn act(&mut self, v: &View, goal: V2, p: &Params, loose: bool) -> Wish {
        let me = v.pos();
        let dist = v.puck_pos().sub(me).len();
        let lead = (dist / p.player_max_speed).min(0.8);
        let pk = v.puck_at(lead, p);
        // The shot leaves at (incoming puck velocity + SHOT_SPEED along the blade normal), so the
        // blade normal must point at d_out·SHOT_SPEED − v_in for the puck to leave along d_out.
        let d_out = goal.sub(pk).norm();
        if loose {
            // Clear: skate through the puck along the clearing direction with the stick forward;
            // the body/blade contact punts it. From the wrong side, loop around behind it first.
            let behind = pk.sub(d_out.mul(1.0));
            let ahead = me.sub(pk).dot(d_out);
            let side = if me.sub(pk).cross(d_out) >= 0.0 { -1.0 } else { 1.0 };
            let target = if ahead > -0.5 && dist > 1.0 {
                behind.add(d_out.perp().mul(side * 1.2))
            } else {
                pk.add(d_out.mul(2.0))
            };
            let target = clamp_to_rink(target, p, p.player_radius + 0.05);
            let to = target.sub(me).norm();
            let (mx, my) = (to.0, to.1);
            return Wish { mx, my, stick: d_out.angle() };
        }
        let d = d_out.mul(SHOT_SPEED).sub(v.puck_vel()).norm();
        let r = hit_radius(p);
        // Two hitting spots, one each side of the shot line; prefer the current one unless the
        // other is clearly closer.
        let left = pk.add(d.perp().mul(r));
        let right = pk.sub(d.perp().mul(r));
        let (dl, dr) = (left.sub(me).len(), right.sub(me).len());
        if self.side == 0.0 {
            self.side = if dl <= dr { 1.0 } else { -1.0 };
        } else if self.side > 0.0 && dr + 1.0 < dl {
            self.side = -1.0;
        } else if self.side < 0.0 && dl + 1.0 < dr {
            self.side = 1.0;
        }
        // Direction player → puck at contact. The sim's stick pushes the puck perpendicular to
        // (puck − player) (s10 centre-on contact), so that line must be the shot line d rotated
        // back by π/2 (plus the small measured SHOT_OVERTURN).
        let swing = self.side;
        let u_angle = d.angle() - swing * (core::f64::consts::FRAC_PI_2 + SHOT_OVERTURN);
        let u = V2(libm::cos(u_angle), libm::sin(u_angle));
        let spot = pk.sub(u.mul(r));
        let to_spot = spot.sub(me).len();

        // If the player is on the far side of the puck (between puck and goal), go around it
        // via a point behind the hitting spot so the body does not knock the puck backward.
        let ahead = me.sub(pk).dot(d);
        let mut target = spot;
        if ahead > -0.3 && to_spot > 0.8 {
            target = spot.sub(d.mul(1.2)).add(d.perp().mul(0.8 * self.side));
        }
        let target = clamp_to_rink(target, p, p.player_radius + 0.05);
        let (mx, my) = move_to(v, target, p, p.player_max_speed);

        // Swing when the puck sits on the blade and a swing from here sends it at the goal: the hit
        // pushes the puck along the stick's normal, i.e. perpendicular to (puck − player).
        // Judge the swing at the moment the blade will arrive: the puck and the player keep moving
        // while the stick turns from cocked to contact.
        let t_c = swing_time(1.3 - 0.17, p);
        let pk_now = v.puck_at(t_c, p);
        let r_now = pk_now.sub(me);
        // SHOT_OVERTURN: the blade keeps turning during the contact substeps, which rotates the
        // shot a little further in the swing sense (measured with `examples/shotlab.rs`).
        let shot_angle = r_now.angle() + swing * (core::f64::consts::FRAC_PI_2 + SHOT_OVERTURN);
        let to_goal = goal.sub(pk_now);
        let out = v.puck_vel().add(V2(libm::cos(shot_angle), libm::sin(shot_angle)).mul(SHOT_SPEED));
        let err = libm::fabs(wrap_angle(to_goal.angle() - out.angle()));
        let tol = if loose { 0.5 } else { libm::atan(0.15 * p.net_width / to_goal.len().max(1.0)).clamp(0.02, 0.2) };
        let (spot_tol, speed_tol) = if loose { (1.2, 6.0) } else { (0.6, 2.5) };
        let on_blade = libm::fabs(r_now.len() - r) < 0.2;
        // Only swing from the cocked position, so the blade always arrives at full angular speed.
        let cocked_ok = libm::fabs(wrap_angle(v.me.stick_angle - (r_now.angle() - swing * 1.3))) < 0.25;
        if self.swing_left == 0 && on_blade && cocked_ok && err < tol && to_spot < spot_tol && v.vel().len() < speed_tol {
            self.swing_left = (0.25 / p.timestep) as u32;
            // Aim the swing only a little past the puck: the bang-bang chase brakes right after contact,
            // so the faster outer blade does not catch the puck a second time and over-rotate the shot.
            self.swing_angle = r_now.angle() + swing * 0.6;
        }
        let stick = if self.swing_left > 0 {
            self.swing_left -= 1;
            self.swing_angle // swing through the puck
        } else {
            // Cocked back, ready. The sim turns the stick the short way; if the short way would
            // sweep the blade back through the puck (pushing it toward the own end), turn the long
            // way in chunks instead.
            let cocked = u.angle() - swing * 1.3;
            let cur = v.me.stick_angle;
            let to_puck = wrap_angle(v.puck_pos().sub(me).angle() - cur);
            let delta = wrap_angle(cocked - cur);
            let crosses = dist < p.stick_length + 0.6
                && to_puck.signum() == delta.signum()
                && to_puck.abs() < delta.abs() + 0.3;
            if crosses { cur - delta.signum() * 1.5 } else { cocked }
        };
        Wish { mx, my, stick: wrap_angle(stick) }
    }
}

/// Where to aim at the opponent net: its center, or a pass-out point in front when the puck is
/// behind the goal line (a straight shot would hit the back of the net).
/// Among a few aim points across the net mouth, the one whose shot line passes farthest from the
/// opponent's body.
fn attack_target(v: &View, p: &Params) -> V2 {
    let pk = v.puck_pos();
    if pk.0 > p.net_x - 0.5 {
        return V2(p.net_x - 5.0, 0.0);
    }
    let w = 0.5 * p.net_width - p.puck_radius - 0.25;
    let aims = [V2(p.net_x + 0.5 * p.net_depth, 0.0), V2(p.net_x + 0.3, 0.6 * w), V2(p.net_x + 0.3, -0.6 * w), V2(p.net_x + 0.3, w), V2(p.net_x + 0.3, -w)];
    let Some(op) = v.opp else { return aims[0] };
    let o = V2(op.x, op.y);
    let clearance = |a: V2| {
        let d = a.sub(pk);
        let l = d.len().max(1e-9);
        let t = (o.sub(pk).dot(d) / (l * l)).clamp(0.0, 1.0);
        o.sub(pk.add(d.mul(t))).len()
    };
    let mut best = aims[0];
    let mut best_c = clearance(best);
    for &a in &aims[1..] {
        let c = clearance(a);
        // Prefer the center unless an off-center line is clearly safer.
        if c > best_c + 0.4 {
            best = a;
            best_c = c;
        }
    }
    best
}

// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Chaser {
    striker: Striker,
}

impl Chaser {
    fn wish(&mut self, s: &State, me: usize, p: &Params) -> Wish {
        let v = View::new(s, me);
        self.striker.act(&v, attack_target(&v, p), p, false)
    }
    fn reset(&mut self) { *self = Self::default(); }
}

#[derive(Clone, Debug, Default)]
pub struct Defender {
    striker: Striker,
    clearing: bool,
}

impl Defender {
    fn wish(&mut self, s: &State, me: usize, p: &Params) -> Wish {
        let v = View::new(s, me);
        let net = V2(-p.net_x, 0.0);
        let pk = v.puck_pos();
        let me_p = v.pos();
        let d_puck = pk.sub(me_p).len();
        let puck_from_net = pk.sub(net).len();
        let me_from_net = me_p.sub(net).len();
        // Clear when the puck is close, or when it is nearer the own net than the defender is.
        // ...but only if it gets there first; otherwise hold the guard line.
        let opp_d = v.opp.map_or(f64::INFINITY, |o| pk.sub(V2(o.x, o.y)).len());
        let wins_race = d_puck < 2.0 || (d_puck + 1.0 < opp_d && pk.0 < -p.net_x + 11.0);
        let start = wins_race && (d_puck < 3.5 || puck_from_net + 0.5 < me_from_net || pk.0 < -p.net_x + 0.5);
        let stop = !wins_race || (d_puck > 6.0 && puck_from_net > me_from_net + 1.0);
        if start {
            self.clearing = true;
        } else if stop {
            self.clearing = false;
        }
        if self.clearing {
            // Up the ice toward the boards on the puck's side (away from the slot in front of the net).
            let side = if pk.1 >= 0.0 { 1.0 } else { -1.0 };
            let goal = V2(p.rink_length * 0.5, side * p.rink_width * 0.35);
            return self.striker.act(&v, goal, p, true);
        }
        // Guard: on the net→puck line, close enough to the net to cover the mouth.
        let dir = pk.sub(net).norm();
        let out = (0.4 * puck_from_net).clamp(2.0, 9.0);
        let mut guard = net.add(dir.mul(out));
        // Stick across the shot line (body + stick cover the mouth), on the side the puck is drifting.
        let across = dir.perp();
        let pv = v.puck_vel();
        let lean = if pv.dot(across) >= 0.0 { 1.0 } else { -1.0 };
        let mut stick = across.mul(lean).angle();
        // A puck heading for the net: step onto its path and lay the stick along the crossing.
        if pv.0 < -2.0 && pk.0 > me_p.0 {
            let t = (me_p.0 - pk.0) / pv.0;
            let y_hit = pk.1 + pv.1 * t;
            if (y_hit - me_p.1).abs() < 0.5 * p.net_width + p.stick_length {
                guard = V2(me_p.0, y_hit);
                let side = if y_hit >= me_p.1 { 1.0 } else { -1.0 };
                stick = V2(0.0, side).angle();
            }
        }
        let guard = clamp_to_rink(guard, p, p.player_radius + 0.05);
        let (mx, my) = move_to(&v, guard, p, p.player_max_speed);
        Wish { mx, my, stick: stick }
    }
    fn reset(&mut self) { *self = Self::default(); }
}

#[derive(Clone, Debug, Default)]
pub struct Cautious {
    striker: Striker,
    engaged: bool,
}

impl Cautious {
    fn wish(&mut self, s: &State, me: usize, p: &Params) -> Wish {
        let v = View::new(s, me);
        let pk = v.puck_pos();
        let me_p = v.pos();
        let d_puck = pk.sub(me_p).len();
        // Engage once the puck is close; let go when it is far again.
        if d_puck < 6.0 {
            self.engaged = true;
        } else if d_puck > 9.0 {
            self.engaged = false;
        }
        if self.engaged {
            return self.striker.act(&v, attack_target(&v, p), p, false);
        }
        // Wait at a fixed depth in the own half, on the line from the puck to the own net.
        let hx = -0.25 * p.rink_length;
        let y = pk.1 * (p.net_x + hx) / (p.net_x + pk.0).max(1.0);
        let home = V2(hx, y.clamp(-0.25 * p.rink_width, 0.25 * p.rink_width));
        let (mx, my) = move_to(&v, home, p, p.player_max_speed * 0.6);
        let stick = pk.sub(me_p).angle();
        Wish { mx, my, stick: stick }
    }
    fn reset(&mut self) { *self = Self::default(); }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Idle;

impl Idle {
    fn wish(&mut self, _s: &State, _me: usize, _p: &Params) -> Wish {
        Wish::default()
    }
}
