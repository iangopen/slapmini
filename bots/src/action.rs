//! Action format, in the acting player's canonical frame (attacks +x).
//!
//! Flat layout (ACT_DIM = 5 floats), the same in the C ABI and in Python:
//!
//! | idx | field | range |
//! |---|---|---|
//! | 0 | move_x | −1..1 (clamped; vectors longer than 1 are normalized by the sim) |
//! | 1 | move_y | −1..1 |
//! | 2 | stick target angle, canonical radians (any value; wrapped) | rad |
//! | 3 | dash (pressed if > 0.5) | 0/1 |
//! | 4 | lift (held if > 0.5) | 0/1 |
//!
//! `to_input` applies the inverse rotation for team 1, so a policy trained as team 0 plays both sides.

use crate::frame::{wrap_angle, Frame};
use sim::PlayerInput;

pub const ACT_DIM: usize = 5;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Action {
    pub move_x: f64,
    pub move_y: f64,
    /// Canonical-frame stick target angle (radians).
    pub stick_angle: f64,
    pub dash: bool,
    pub lift: bool,
}

impl Action {
    pub fn from_slice(a: &[f32]) -> Self {
        let g = |i: usize| a.get(i).copied().unwrap_or(0.0) as f64;
        let fin = |v: f64| if v.is_finite() { v } else { 0.0 };
        Action {
            move_x: fin(g(0)).clamp(-1.0, 1.0),
            move_y: fin(g(1)).clamp(-1.0, 1.0),
            stick_angle: wrap_angle(fin(g(2))),
            dash: g(3) > 0.5,
            lift: g(4) > 0.5,
        }
    }

    pub fn write(&self, out: &mut [f32]) {
        out[0] = self.move_x as f32;
        out[1] = self.move_y as f32;
        out[2] = self.stick_angle as f32;
        out[3] = if self.dash { 1.0 } else { 0.0 };
        out[4] = if self.lift { 1.0 } else { 0.0 };
    }

    /// World-frame input for a player of `team`.
    pub fn to_input(&self, team: u8) -> PlayerInput {
        let f = Frame::for_team(team);
        let (mx, my) = f.vec(self.move_x.clamp(-1.0, 1.0), self.move_y.clamp(-1.0, 1.0));
        PlayerInput { move_x: mx, move_y: my, stick_target_angle: f.angle(self.stick_angle), dash: self.dash, lift: self.lift }
    }
}
