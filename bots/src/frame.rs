//! Canonical frame: every player sees the world as if it attacks +x.
//!
//! The rink, the nets and the physics are symmetric under a 180° rotation about center ice
//! ((x, y) → (−x, −y)). Team 0 attacks +x already, so its canonical frame is the world frame.
//! Team 1 attacks −x, so its canonical frame is the world rotated by 180°. A rotation (not a
//! mirror) keeps handedness: angular velocities and the CCW/CW sense of a stick swing are unchanged,
//! and angles shift by π.

use sim::{Player, Puck, State};

const PI: f64 = core::f64::consts::PI;
const TAU: f64 = core::f64::consts::TAU;

/// Wrap an angle into [−π, π).
#[inline]
pub fn wrap_angle(a: f64) -> f64 {
    let r = (a + PI).rem_euclid(TAU) - PI;
    if r < -PI { r + TAU } else { r }
}

/// The transform between world and one team's canonical frame. Its own inverse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    /// +1 for team 0 (identity), −1 for team 1 (180° rotation).
    pub sign: f64,
}

impl Frame {
    pub fn for_team(team: u8) -> Self {
        Frame { sign: if team == 0 { 1.0 } else { -1.0 } }
    }

    /// A point or a vector (both transform the same way under a rotation about the origin).
    #[inline]
    pub fn vec(&self, x: f64, y: f64) -> (f64, f64) {
        (self.sign * x, self.sign * y)
    }

    /// An angle, world → canonical or canonical → world (the rotation by π is its own inverse).
    #[inline]
    pub fn angle(&self, a: f64) -> f64 {
        if self.sign > 0.0 { a } else { wrap_angle(a + PI) }
    }

    /// (cos, sin) of an angle given in the other frame, without rounding through `angle`.
    #[inline]
    pub fn dir(&self, a: f64) -> (f64, f64) {
        (self.sign * libm::cos(a), self.sign * libm::sin(a))
    }

    pub fn puck(&self, p: &Puck) -> Puck {
        let (x, y) = self.vec(p.x, p.y);
        let (vx, vy) = self.vec(p.vx, p.vy);
        Puck { x, y, vx, vy, ..*p }
    }

    pub fn player(&self, p: &Player) -> Player {
        let (x, y) = self.vec(p.x, p.y);
        let (vx, vy) = self.vec(p.vx, p.vy);
        Player { x, y, vx, vy, facing: self.angle(p.facing), stick_angle: self.angle(p.stick_angle), ..*p }
    }
}

/// The whole state rotated by 180° with team labels and scores swapped: what the game looks like
/// "from the other bench". Used by the mirror test; `rotate_swap(rotate_swap(s)) ≈ s`.
/// The goal flag `in_net` is swapped too (net 0 ↔ net 1).
pub fn rotate_swap(s: &State) -> State {
    let f = Frame { sign: -1.0 };
    let mut out = s.clone();
    out.puck = f.puck(&s.puck);
    out.puck.in_net = s.puck.in_net.map(|n| 1 - n);
    for (o, p) in out.players.iter_mut().zip(&s.players) {
        *o = f.player(p);
        o.team = 1 - p.team;
    }
    if out.score.len() >= 2 {
        out.score.swap(0, 1);
    }
    out
}
