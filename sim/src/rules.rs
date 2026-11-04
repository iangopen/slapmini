//! Match rules (1v1 for now, any player count works): countdown, a regulation clock, goal pause +
//! reset to spawn, mercy rule, and sudden-death overtime when tied at the end of regulation.
//!
//! Opt-in: they only run when `State::match_state` is `Some` (see `State::new_match`). With `None`,
//! `step` is plain free play (the lab's practice mode, calib replays and the physics tests).
//! Rule values come from Params: match_length_s and mercy_lead are Ian's rules, countdown_s,
//! goal_pause_s and player_spawn_x are invented placeholders.

use crate::params::Params;
use crate::state::{Player, PlayerInput, Puck, State};
use crate::step::physics_step;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchPhase {
    /// Frozen before play (match start and after every goal). Inputs ignored, clock stopped.
    #[default]
    Countdown,
    /// Physics runs, inputs apply, the clock counts down (not in overtime).
    Playing,
    /// Frozen right after a goal (puck stays in the net), then reset to spawn + countdown.
    GoalPause,
    /// The match is over; `winner` is set. Nothing moves.
    Finished,
}

impl MatchPhase {
    pub fn code(self) -> u8 {
        match self {
            MatchPhase::Countdown => 0,
            MatchPhase::Playing => 1,
            MatchPhase::GoalPause => 2,
            MatchPhase::Finished => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MatchState {
    pub phase: MatchPhase,
    /// Steps left in the current countdown or goal pause (0 while playing / finished).
    pub phase_ticks: u64,
    /// Regulation steps left. Stays 0 during overtime.
    pub clock_ticks: u64,
    /// Tied when the clock hit zero: the next goal wins.
    pub overtime: bool,
    pub winner: Option<u8>,
    /// Team that scored the most recent goal (for the UI's goal banner).
    #[serde(default)]
    pub last_goal: Option<u8>,
}

/// Seconds → whole steps (rounded, never negative).
pub fn secs_to_ticks(secs: f64, p: &Params) -> u64 {
    let t = libm::round(secs / p.timestep);
    if t > 0.0 { t as u64 } else { 0 }
}

impl MatchState {
    /// A fresh match: full clock, in the opening countdown.
    pub fn start(p: &Params) -> Self {
        let mut m = MatchState { clock_ticks: secs_to_ticks(p.match_length_s, p), ..Default::default() };
        m.enter_countdown(p);
        m
    }

    fn enter_countdown(&mut self, p: &Params) {
        self.phase_ticks = secs_to_ticks(p.countdown_s, p);
        self.phase = if self.phase_ticks == 0 { MatchPhase::Playing } else { MatchPhase::Countdown };
    }

    fn finish(&mut self, team: u8) {
        self.phase = MatchPhase::Finished;
        self.phase_ticks = 0;
        self.winner = Some(team);
    }

    /// Regulation clock (in seconds, rounded up the way a countdown clock shows it).
    pub fn clock_secs(&self, p: &Params) -> f64 {
        libm::ceil(self.clock_ticks as f64 * p.timestep - 1e-9).max(0.0)
    }
}

fn goals(state: &State, team: usize) -> u32 {
    state.score.get(team).copied().unwrap_or(0)
}

/// The leading team, or None when tied.
fn leader(state: &State) -> Option<u8> {
    let (a, b) = (goals(state, 0), goals(state, 1));
    if a > b { Some(0) } else if b > a { Some(1) } else { None }
}

/// Puck to center ice at rest; every player to its team's spawn (team 0 at -player_spawn_x,
/// attacking +x; team 1 mirrored), at rest, facing center, stick along the facing, dash reset.
/// Teammates are spread along y the same way as `State::new_default`.
pub fn reset_to_spawn(state: &mut State, p: &Params) {
    state.puck = Puck::default();
    let teams: Vec<u8> = state.players.iter().map(|pl| pl.team).collect();
    for (i, pl) in state.players.iter_mut().enumerate() {
        let team = pl.team;
        let slot = teams[..i].iter().filter(|&&t| t == team).count() as f64;
        let per_team = teams.iter().filter(|&&t| t == team).count() as f64;
        let side = if team == 0 { -1.0 } else { 1.0 };
        let facing = if team == 0 { 0.0 } else { core::f64::consts::PI };
        *pl = Player {
            team,
            x: side * p.player_spawn_x,
            y: (slot - (per_team - 1.0) / 2.0) * 4.0,
            facing,
            stick_angle: facing,
            ..Default::default()
        };
    }
}

/// One step under match rules. Called by `step` when `state.match_state` is Some.
pub(crate) fn step_match(state: &mut State, inputs: &[PlayerInput], p: &Params) {
    let mut m = state.match_state.expect("step_match needs a match state");
    match m.phase {
        MatchPhase::Finished => state.tick += 1,
        MatchPhase::Countdown => {
            state.tick += 1;
            m.phase_ticks = m.phase_ticks.saturating_sub(1);
            if m.phase_ticks == 0 {
                m.phase = MatchPhase::Playing;
            }
        }
        MatchPhase::GoalPause => {
            state.tick += 1;
            m.phase_ticks = m.phase_ticks.saturating_sub(1);
            if m.phase_ticks == 0 {
                reset_to_spawn(state, p);
                m.enter_countdown(p);
            }
        }
        MatchPhase::Playing => {
            let goal = physics_step(state, inputs, p);
            if !m.overtime {
                m.clock_ticks = m.clock_ticks.saturating_sub(1);
            }
            let lead = goals(state, 0).abs_diff(goals(state, 1)) as f64;
            if let Some(team) = goal {
                m.last_goal = Some(team);
                if m.overtime {
                    m.finish(team); // sudden death
                } else if p.mercy_lead > 0.0 && lead >= p.mercy_lead {
                    m.finish(team); // mercy: the scorer just reached the lead
                } else if m.clock_ticks == 0 && leader(state).is_some() {
                    m.finish(leader(state).unwrap()); // goal on the last step of regulation
                } else {
                    if m.clock_ticks == 0 {
                        m.overtime = true; // equalizer on the last step: overtime after the reset
                    }
                    m.phase = MatchPhase::GoalPause;
                    m.phase_ticks = secs_to_ticks(p.goal_pause_s, p);
                    if m.phase_ticks == 0 {
                        reset_to_spawn(state, p);
                        m.enter_countdown(p);
                    }
                }
            } else if !m.overtime && m.clock_ticks == 0 {
                match leader(state) {
                    Some(t) => m.finish(t),
                    None => m.overtime = true, // tied at 0:00: play on, next goal wins
                }
            }
        }
    }
    state.match_state = Some(m);
}
