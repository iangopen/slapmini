//! Fixed-size observation vector for one player, in that player's canonical frame (attacks +x).
//!
//! Normalization uses FIXED constants (not params), so the meaning of each number stays the same
//! while the physics is recalibrated. Values are roughly in [−1, 1] for ordinary play but are not
//! clipped (a 45 m/s puck reads 1.5).
//!
//! | idx | field | scale |
//! |---|---|---|
//! | 0, 1 | own x, y (m) | / POS_X, / POS_Y |
//! | 2, 3 | own vx, vy (m/s) | / PLAYER_V |
//! | 4, 5 | cos, sin of own facing | 1 |
//! | 6, 7 | cos, sin of own stick angle | 1 |
//! | 8 | own stick angular velocity (rad/s) | / OMEGA |
//! | 9 | own dash window left (s) | / DASH_T |
//! | 10 | own dash cooldown left (s) | / DASH_CD |
//! | 11, 12 | puck x, y (m) | / POS_X, / POS_Y |
//! | 13 | puck height z (m) | / PUCK_Z |
//! | 14, 15 | puck vx, vy (m/s) | / PUCK_V |
//! | 16 | puck vz (m/s) | / PUCK_VZ |
//! | 17, 18 | puck − own position (m) | / REL |
//! | 19, 20 | opponent x, y (m) | / POS_X, / POS_Y |
//! | 21, 22 | opponent vx, vy (m/s) | / PLAYER_V |
//! | 23, 24 | cos, sin of opponent stick angle | 1 |
//! | 25 | opponent stick angular velocity (rad/s) | / OMEGA |
//! | 26, 27 | opponent − own position (m) | / REL |
//! | 28 | score difference: own goals − opponent goals (mercy ends the match at ±5) | / SCORE |
//! | 29 | 1 in overtime (next goal wins), else 0 | 1 |
//! | 30 | regulation time left (s): 1.0 at the start of a 5:00 match, 0 in overtime; 0 in free play | / TIME |
//! | 31 | 1 if the dash is ready (cooldown 0), else 0 | 1 |
//! | 32..40 | style vector (caller-supplied, zeros by default) | as given |
//!
//! Angles are encoded as (cos, sin) so the 180° rotation is an exact sign flip. Rotation keeps
//! angular velocity unchanged. With no opponent, fields 19..=27 are 0.

use crate::frame::Frame;
use sim::{Params, State};

/// Number of base fields (everything except the style slot).
pub const BASE_DIM: usize = 32;
/// Length of the style slot appended to every observation.
pub const STYLE_DIM: usize = 8;
/// Full observation length.
pub const OBS_DIM: usize = BASE_DIM + STYLE_DIM;
/// Bump when the layout or a normalization constant changes (checkpoints record it).
pub const OBS_VERSION: u32 = 2;

pub const POS_X: f64 = 30.0;
pub const POS_Y: f64 = 20.0;
pub const PLAYER_V: f64 = 14.0;
pub const OMEGA: f64 = 14.0;
pub const DASH_T: f64 = 0.25;
pub const DASH_CD: f64 = 1.0;
pub const PUCK_Z: f64 = 2.0;
pub const PUCK_V: f64 = 30.0;
pub const PUCK_VZ: f64 = 10.0;
pub const REL: f64 = 10.0;
pub const SCORE: f64 = 5.0;
pub const TIME: f64 = 300.0;

/// Index of the opponent of player `me` (the first player on the other team), if any.
pub fn opponent_of(state: &State, me: usize) -> Option<usize> {
    let team = state.players[me].team;
    state.players.iter().position(|p| p.team != team)
}

/// Write the observation of player `me` into `out` (length OBS_DIM). `style` fills the style
/// slot (missing entries are 0, extra entries ignored).
pub fn observe(state: &State, me: usize, style: &[f32], p: &Params, out: &mut [f32]) {
    assert!(out.len() >= OBS_DIM, "obs buffer too small");
    let pl = &state.players[me];
    let f = Frame::for_team(pl.team);
    let (x, y) = f.vec(pl.x, pl.y);
    let (vx, vy) = f.vec(pl.vx, pl.vy);
    let (fc, fs) = f.dir(pl.facing);
    let (sc, ss) = f.dir(pl.stick_angle);
    let (px, py) = f.vec(state.puck.x, state.puck.y);
    let (pvx, pvy) = f.vec(state.puck.vx, state.puck.vy);
    let team = pl.team as usize;
    let score = |t: usize| state.score.get(t).copied().unwrap_or(0) as f64;

    let mut o = [0f64; BASE_DIM];
    o[0] = x / POS_X;
    o[1] = y / POS_Y;
    o[2] = vx / PLAYER_V;
    o[3] = vy / PLAYER_V;
    o[4] = fc;
    o[5] = fs;
    o[6] = sc;
    o[7] = ss;
    o[8] = pl.stick_omega / OMEGA;
    o[9] = pl.dash_timer / DASH_T;
    o[10] = pl.dash_cooldown / DASH_CD;
    o[11] = px / POS_X;
    o[12] = py / POS_Y;
    o[13] = state.puck.z / PUCK_Z;
    o[14] = pvx / PUCK_V;
    o[15] = pvy / PUCK_V;
    o[16] = state.puck.vz / PUCK_VZ;
    o[17] = (px - x) / REL;
    o[18] = (py - y) / REL;
    if let Some(j) = opponent_of(state, me) {
        let op = &state.players[j];
        let (ox, oy) = f.vec(op.x, op.y);
        let (ovx, ovy) = f.vec(op.vx, op.vy);
        let (oc, os) = f.dir(op.stick_angle);
        o[19] = ox / POS_X;
        o[20] = oy / POS_Y;
        o[21] = ovx / PLAYER_V;
        o[22] = ovy / PLAYER_V;
        o[23] = oc;
        o[24] = os;
        o[25] = op.stick_omega / OMEGA;
        o[26] = (ox - x) / REL;
        o[27] = (oy - y) / REL;
    }
    o[28] = (score(team) - score(1 - team.min(1))) / SCORE;
    if let Some(m) = &state.match_state {
        o[29] = if m.overtime { 1.0 } else { 0.0 };
        o[30] = m.clock_ticks as f64 * p.timestep / TIME;
    }
    o[31] = if pl.dash_cooldown <= 0.0 { 1.0 } else { 0.0 };

    for (d, s) in out.iter_mut().zip(o.iter()) {
        *d = *s as f32;
    }
    for k in 0..STYLE_DIM {
        out[BASE_DIM + k] = style.get(k).copied().unwrap_or(0.0);
    }
}
