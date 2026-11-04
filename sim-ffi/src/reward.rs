//! Reward as configuration. The env measures a fixed set of raw terms for the learner every
//! decision step; the reward is the dot product of those terms with a weight vector from JSON.
//!
//! Raw terms (all in the learner's canonical frame, where it attacks +x):
//!
//! | idx | name | raw value per decision step |
//! |---|---|---|
//! | 0 | goal_for | number of goals the learner's team scored (0/1) |
//! | 1 | goal_against | number of goals conceded (0/1); give it a NEGATIVE weight |
//! | 2 | puck_progress | Δ(puck x) / net_x: + when the puck moves toward the opponent net |
//! | 3 | shot_on_net | 1 when a learner contact sends the puck on a line into the opponent net mouth at ≥ 5 m/s |
//! | 4 | possession | seconds the learner is the closest player and has the puck within stick reach |
//! | 5 | defensive_position | seconds the learner is goal-side: between the puck and its own net (≤ 1.5 m off that line) |

use bots::frame::Frame;
use serde::{Deserialize, Serialize};
use sim::{Params, State};

pub const N_TERMS: usize = 6;
pub const TERM_NAMES: [&str; N_TERMS] =
    ["goal_for", "goal_against", "puck_progress", "shot_on_net", "possession", "defensive_position"];

/// Weights per raw term. Missing JSON keys take these defaults.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RewardWeights {
    pub goal_for: f32,
    pub goal_against: f32,
    pub puck_progress: f32,
    pub shot_on_net: f32,
    pub possession: f32,
    pub defensive_position: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        RewardWeights {
            goal_for: 1.0,
            goal_against: -1.0,
            puck_progress: 0.2,
            shot_on_net: 0.05,
            possession: 0.02,
            defensive_position: 0.0,
        }
    }
}

impl RewardWeights {
    pub fn to_array(&self) -> [f32; N_TERMS] {
        [self.goal_for, self.goal_against, self.puck_progress, self.shot_on_net, self.possession, self.defensive_position]
    }
    pub fn from_slice(w: &[f32]) -> Self {
        let g = |i: usize| w.get(i).copied().unwrap_or(0.0);
        RewardWeights {
            goal_for: g(0),
            goal_against: g(1),
            puck_progress: g(2),
            shot_on_net: g(3),
            possession: g(4),
            defensive_position: g(5),
        }
    }
    pub fn reward(&self, raw: &[f32; N_TERMS]) -> f32 {
        self.to_array().iter().zip(raw).map(|(w, r)| w * r).sum()
    }
}

/// Per-sim-step bookkeeping for the raw terms.
#[derive(Clone, Debug, Default)]
pub struct Tracker {
    /// Steps left before another shot can be counted.
    shot_cooldown: u32,
}

impl Tracker {
    /// Accumulate the raw terms of one sim step into `raw`. `before`/`after` are the states around
    /// the step, `me` the learner's slot.
    pub fn sim_step(&mut self, before: &State, after: &State, me: usize, p: &Params, raw: &mut [f32; N_TERMS]) {
        let team = after.players[me].team as usize;
        let f = Frame::for_team(team as u8);
        let gs = |s: &State, t: usize| s.score.get(t).copied().unwrap_or(0);
        raw[0] += (gs(after, team) - gs(before, team)) as f32;
        raw[1] += (gs(after, 1 - team) - gs(before, 1 - team)) as f32;

        let (bx, _) = f.vec(before.puck.x, before.puck.y);
        let (ax, ay) = f.vec(after.puck.x, after.puck.y);
        raw[2] += ((ax - bx) / p.net_x) as f32;

        let dt = p.timestep as f32;
        let pl = &after.players[me];
        let (mx, my) = f.vec(pl.x, pl.y);
        let d_me = ((ax - mx).powi(2) + (ay - my).powi(2)).sqrt();
        let closest = after.players.iter().enumerate().all(|(j, o)| {
            j == me || {
                let (ox, oy) = f.vec(o.x, o.y);
                ((ax - ox).powi(2) + (ay - oy).powi(2)).sqrt() >= d_me
            }
        });
        let in_reach = d_me < p.stick_length + p.puck_radius + 0.2 && after.puck.z <= p.stick_z_max;

        // Shot on net: a sudden puck velocity change near the learner (who is the closest player)
        // that leaves the puck heading into the opponent net mouth.
        if self.shot_cooldown > 0 {
            self.shot_cooldown -= 1;
        }
        let (bvx, bvy) = f.vec(before.puck.vx, before.puck.vy);
        let (avx, avy) = f.vec(after.puck.vx, after.puck.vy);
        let dv = ((avx - bvx).powi(2) + (avy - bvy).powi(2)).sqrt();
        let speed = (avx * avx + avy * avy).sqrt();
        if self.shot_cooldown == 0 && dv > 2.0 && closest && in_reach && speed >= 5.0 && avx > 0.0 && ax < p.net_x {
            let t = (p.net_x - ax) / avx;
            let y_at = ay + avy * t;
            if y_at.abs() < 0.5 * p.net_width - p.puck_radius {
                raw[3] += 1.0;
                self.shot_cooldown = (0.5 / p.timestep) as u32;
            }
        }

        if closest && in_reach {
            raw[4] += dt;
        }

        // Goal side: own net center → puck segment; learner behind the puck and near that line.
        let (nx, ny) = (-p.net_x, 0.0);
        let (sx, sy) = (ax - nx, ay - ny);
        let l2 = (sx * sx + sy * sy).max(1e-9);
        let t = (((mx - nx) * sx + (my - ny) * sy) / l2).clamp(0.0, 1.0);
        let off = ((mx - (nx + sx * t)).powi(2) + (my - (ny + sy * t)).powi(2)).sqrt();
        if mx < ax && off < 1.5 {
            raw[5] += dt;
        }
    }
}
