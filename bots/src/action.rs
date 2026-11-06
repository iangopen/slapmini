//! Action format, in the acting player's canonical frame (attacks +x). It matches what a human has:
//! WASD (a 9-way digital move: stop or one of 8 directions), the mouse angle (stick target), and the
//! dash and lift keys, and the left mouse button (raised stick).
//!
//! Flat layout (ACT_DIM = 6 floats, ACT_VERSION 4), the same in the C ABI and in Python:
//!
//! | idx | field | range |
//! |---|---|---|
//! | 0 | move index (rounded, clamped to 0..=8; see MOVE_DIRS) | 0..8 |
//! | 1 | stick target angle, canonical radians (any value; wrapped) | rad |
//! | 2 | dash (pressed if > 0.5) | 0/1 |
//! | 3 | lift (held if > 0.5) | 0/1 |
//! | 4 | stick_up (raised if > 0.5; ACT v3) | 0/1 |
//! | 5 | brake (slow walk if > 0.5; ACT v4) | 0/1 |
//!
//! Move indices (canonical frame, +x = toward the opponent net, +y = left of that):
//! 0 stop, 1 +x, 2 +x+y, 3 +y, 4 −x+y, 5 −x, 6 −x−y, 7 −y, 8 +x−y (counterclockwise from +x).
//! A diagonal is sent to the sim as (±1, ±1); the sim normalizes move vectors longer than 1, so a
//! diagonal accelerates and tops out exactly like a straight key (tested in bots_tests).
//!
//! `to_input` applies the inverse rotation for team 1, so a policy trained as team 0 plays both sides.
//! stick_up and brake are not rotated (flags). History: ACT v1 = 5 floats with a continuous move (dropped in ABI v2);
//! ACT v2 = 4 floats (no stick_up); ACT v3 = 5 floats (t2); ACT v4 = 6 floats (m4).

use crate::frame::{wrap_angle, Frame};
use sim::PlayerInput;

pub const ACT_DIM: usize = 6;
/// Action layout version (checkpoints and ONNX sidecars carry it).
pub const ACT_VERSION: u32 = 4;
pub const N_MOVES: usize = 9;

/// Key combination per move index (canonical frame).
pub const MOVE_DIRS: [(f64, f64); N_MOVES] =
    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)];

/// Nearest digital move for an analog wish (mx, my): stop if its length is below 0.5, else the
/// nearest of the 8 directions. Used by the scripted bots, which compute an analog wish internally.
pub fn quantize_move(mx: f64, my: f64) -> u8 {
    if !(mx.is_finite() && my.is_finite()) || libm::sqrt(mx * mx + my * my) < 0.5 {
        return 0;
    }
    let k = libm::round(libm::atan2(my, mx) / core::f64::consts::FRAC_PI_4) as i64;
    (k.rem_euclid(8) + 1) as u8
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Action {
    /// Move index 0..=8 (MOVE_DIRS).
    pub move_dir: u8,
    /// Canonical-frame stick target angle (radians).
    pub stick_angle: f64,
    pub dash: bool,
    pub lift: bool,
    /// Raised stick (left mouse button), ACT v3.
    pub stick_up: bool,
    /// Slow walk (Space), ACT v4. Scripted bots do not brake.
    pub brake: bool,
}

impl Action {
    /// An action whose move is the digital snap of an analog wish (see `quantize_move`).
    pub fn digital(mx: f64, my: f64, stick_angle: f64, dash: bool, lift: bool) -> Self {
        Action { move_dir: quantize_move(mx, my), stick_angle, dash, lift, stick_up: false, brake: false }
    }

    pub fn from_slice(a: &[f32]) -> Self {
        let g = |i: usize| a.get(i).copied().unwrap_or(0.0) as f64;
        let fin = |v: f64| if v.is_finite() { v } else { 0.0 };
        Action {
            move_dir: libm::round(fin(g(0))).clamp(0.0, (N_MOVES - 1) as f64) as u8,
            stick_angle: wrap_angle(fin(g(1))),
            dash: g(2) > 0.5,
            lift: g(3) > 0.5,
            stick_up: g(4) > 0.5,
            brake: g(5) > 0.5,
        }
    }

    pub fn write(&self, out: &mut [f32]) {
        out[0] = self.move_dir as f32;
        out[1] = self.stick_angle as f32;
        out[2] = if self.dash { 1.0 } else { 0.0 };
        out[3] = if self.lift { 1.0 } else { 0.0 };
        out[4] = if self.stick_up { 1.0 } else { 0.0 };
        out[5] = if self.brake { 1.0 } else { 0.0 };
    }

    /// Canonical move vector (key combination).
    pub fn move_vec(&self) -> (f64, f64) {
        MOVE_DIRS[(self.move_dir as usize).min(N_MOVES - 1)]
    }

    /// World-frame input for a player of `team`.
    pub fn to_input(&self, team: u8) -> PlayerInput {
        let f = Frame::for_team(team);
        let (cx, cy) = self.move_vec();
        let (mx, my) = f.vec(cx, cy);
        PlayerInput { move_x: mx, move_y: my, stick_target_angle: f.angle(self.stick_angle), dash: self.dash, lift: self.lift, stick_up: self.stick_up, brake: self.brake }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_snaps_to_nearest_direction() {
        assert_eq!(quantize_move(0.0, 0.0), 0);
        assert_eq!(quantize_move(0.3, 0.2), 0);
        assert_eq!(quantize_move(1.0, 0.1), 1);
        assert_eq!(quantize_move(0.7, 0.7), 2);
        assert_eq!(quantize_move(-0.1, 1.0), 3);
        assert_eq!(quantize_move(-1.0, 0.0), 5);
        assert_eq!(quantize_move(0.6, -0.6), 8);
        for (i, d) in MOVE_DIRS.iter().enumerate().skip(1) {
            assert_eq!(quantize_move(d.0, d.1) as usize, i);
        }
    }

    #[test]
    fn flat_round_trip() {
        let a = Action { move_dir: 6, stick_angle: 1.25, dash: true, lift: false, stick_up: true, brake: true };
        let mut buf = [0f32; ACT_DIM];
        a.write(&mut buf);
        let b = Action::from_slice(&buf);
        assert_eq!(b.move_dir, 6);
        assert!((b.stick_angle - 1.25).abs() < 1e-6 && b.dash && !b.lift && b.stick_up);
        assert!(!Action::from_slice(&[0.0, 0.0, 0.0, 0.0]).stick_up, "a short (old) slice reads stick_up as false");
        assert!(!Action::from_slice(&[0.0; 5]).brake, "a short (old) slice reads brake as false");
        for team in 0..2 { assert!(b.to_input(team).stick_up && b.to_input(team).brake); }
        assert_eq!(Action::from_slice(&[12.7, 0.0, 0.0, 0.0, 0.0, 0.0]).move_dir, 8);
        assert_eq!(Action::from_slice(&[-3.0, f32::NAN, 0.0, 0.0, 0.0, 0.0]).move_dir, 0);
    }
}
