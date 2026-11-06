use crate::params::Params;
use crate::rules::{reset_to_spawn, MatchState};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Puck {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    /// Which net (0 = +x, 1 = -x) the puck is currently inside, if any. Used for goal edge detection.
    #[serde(default)]
    pub in_net: Option<u8>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Player {
    pub team: u8,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    /// Body facing, radians, world frame.
    pub facing: f64,
    /// Stick angle, radians, world frame (around the player center).
    pub stick_angle: f64,
    /// Stick angular velocity, rad/s.
    pub stick_omega: f64,
    /// Seconds left in the dash window (stick hits count as dash shots while > 0).
    #[serde(default)]
    pub dash_timer: f64,
    #[serde(default)]
    pub dash_cooldown: f64,
    /// Dash input from the previous step, for press-edge detection.
    #[serde(default)]
    pub dash_prev: bool,
    /// Raised stick (w4): set from the input every step. A raised stick contacts the puck only inside
    /// [stick_up_z_min, stick_up_z_max] (it passes over a puck on the ice).
    #[serde(default)]
    pub stick_up: bool,
    /// The stick came to rest on the puck without a swing (lowered onto it, a puck dropping into its height range, a
    /// reset): it ignores the puck until they are apart, so the puck never jumps out of it.
    #[serde(default)]
    pub stick_ghost: bool,
}

/// One player's input for one step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlayerInput {
    /// -1..1 (WASD). Vectors longer than 1 are normalized.
    pub move_x: f64,
    pub move_y: f64,
    /// Radians, world frame (angle from player to mouse).
    pub stick_target_angle: f64,
    #[serde(default)]
    pub dash: bool,
    #[serde(default)]
    pub lift: bool,
    /// Raise the stick while held (w4; the game's left mouse button). Default false when absent in JSON.
    #[serde(default)]
    pub stick_up: bool,
    /// Slow walk while held (w5b; the game's Space): the speed cap drops to brake_speed. Default false when absent in
    /// JSON. Bots send it through ACT v4.
    #[serde(default)]
    pub brake: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub tick: u64,
    pub puck: Puck,
    pub players: Vec<Player>,
    /// Goals per team (index = team).
    #[serde(default)]
    pub score: Vec<u32>,
    /// Match rules (phase, clock, overtime, winner). None = free play (no rules; the default).
    #[serde(default)]
    pub match_state: Option<MatchState>,
}

impl State {
    /// Puck at center ice at rest; players split between two teams on their own halves.
    /// Works for any player count (1v1 now, 3v3 later).
    pub fn new_default(params: &Params, n_players: usize) -> Self {
        let mut players = Vec::with_capacity(n_players);
        for i in 0..n_players {
            let team = (i % 2) as u8;
            let slot = (i / 2) as f64;
            let side = if team == 0 { -1.0 } else { 1.0 };
            let per_team = n_players.div_ceil(2) as f64;
            let y = (slot - (per_team - 1.0) / 2.0) * 4.0;
            let facing = if team == 0 { 0.0 } else { core::f64::consts::PI };
            players.push(Player {
                team,
                x: side * params.rink_length * 0.25,
                y,
                facing,
                stick_angle: facing,
                ..Default::default()
            });
        }
        State { tick: 0, puck: Puck::default(), players, score: vec![0, 0], match_state: None }
    }

    /// A new match: players at their spawn points, puck at center, 0-0, full clock, opening countdown.
    pub fn new_match(params: &Params, n_players: usize) -> Self {
        let mut s = State::new_default(params, n_players);
        reset_to_spawn(&mut s, params);
        s.match_state = Some(MatchState::start(params));
        s
    }

    pub fn time(&self, params: &Params) -> f64 {
        self.tick as f64 * params.timestep
    }

    /// Exact byte encoding of the state (f64 bit patterns), for determinism checks.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(64 + self.players.len() * 96);
        b.extend_from_slice(&self.tick.to_le_bytes());
        let p = &self.puck;
        for v in [p.x, p.y, p.z, p.vx, p.vy, p.vz] {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        b.push(p.in_net.map_or(255, |n| n));
        for pl in &self.players {
            b.push(pl.team);
            for v in [pl.x, pl.y, pl.vx, pl.vy, pl.facing, pl.stick_angle, pl.stick_omega, pl.dash_timer, pl.dash_cooldown] {
                b.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            b.push(pl.dash_prev as u8);
        }
        for s in &self.score {
            b.extend_from_slice(&s.to_le_bytes());
        }
        // Match state only when present, so free-play encodings (and their hashes) are unchanged.
        if let Some(m) = &self.match_state {
            b.push(0xA5);
            b.push(m.phase.code());
            b.extend_from_slice(&m.phase_ticks.to_le_bytes());
            b.extend_from_slice(&m.clock_ticks.to_le_bytes());
            b.push(m.overtime as u8);
            b.push(m.winner.map_or(255, |t| t));
            b.push(m.last_goal.map_or(255, |t| t));
        }
        // Raised sticks only when some stick is up, so states without one (every free-play and match
        // state before w4, and every bot run) keep their encodings and hashes.
        if self.players.iter().any(|pl| pl.stick_up) {
            b.push(0x5C);
            for pl in &self.players {
                b.push(pl.stick_up as u8);
            }
        }
        if self.players.iter().any(|pl| pl.stick_ghost) {
            b.push(0x5D);
            for pl in &self.players {
                b.push(pl.stick_ghost as u8);
            }
        }
        b
    }

    /// FNV-1a 64 over `to_bytes`.
    pub fn hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for byte in self.to_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}
