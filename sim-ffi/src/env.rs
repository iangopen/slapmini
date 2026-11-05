//! Batched 1v1 environments. Each environment owns its State, a seeded RNG (used only at resets),
//! its opponent (a scripted bot or an external policy slot) and its reward weights and style vector.
//! Stepping is a pure function of (env, actions), so results do not depend on thread scheduling.
//!
//! Two episode modes (`Config::mode`):
//! - "first_goal" (default, the curriculum): free play from a jittered faceoff, the episode ends at
//!   the first goal or after the step limit (default 30 s).
//! - "match": a full match under the sim's rules (`State::new_match`): 5:00 clock, goal resets,
//!   mercy at a 5-goal lead, next-goal overtime. The frozen phases (countdown, goal pause) are
//!   fast-forwarded inside the env: no decision is spent on them and they don't count toward the
//!   step limit (PLAYING sim steps; default regulation + `overtime_cap_s`). The episode ends when
//!   the match is finished, or is truncated at the cap.

use crate::reward::{RewardWeights, Tracker, N_TERMS, STYLE_USED};
use bots::{observe, Action, Bot, Kind, Scripted, ACT_DIM, OBS_DIM, STYLE_DIM};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sim::{MatchPhase, Params, PlayerInput, State};

/// Per-env info written every step (meaningful for the finished episode when done = 1):
/// [0 end reason, 1 episode sim steps (playing steps), 2 episode return, 3 learner team,
///  4 goals for, 5 goals against, 6 opponent possession seconds, 7.. raw reward term sums × N_TERMS].
pub const INFO_DIM: usize = 7 + N_TERMS;
/// Index of the first raw reward-term column in the info row.
pub const INFO_RAW: usize = 7;

/// End reasons. In first-goal mode WIN/LOSS = the first goal was for/against the learner; in
/// match mode they are the match result. TRUNCATED = the step limit (not a real terminal state).
pub const END_RUNNING: f32 = 0.0;
pub const END_WIN: f32 = 1.0;
pub const END_LOSS: f32 = 2.0;
pub const END_TRUNCATED: f32 = 3.0;

/// Opponent id for "an external policy fills the opponent action buffer".
pub const OPP_EXTERNAL: i32 = -1;

/// Batch configuration (JSON). Every field has a default.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// "first_goal" or "match" (module docs).
    pub mode: String,
    /// Episode step limit in PLAYING sim steps; 0 = auto (first_goal: 3600 = 30 s; match:
    /// match_length_s + overtime_cap_s).
    pub max_steps: u64,
    /// Match mode: overtime longer than this (s) truncates the episode.
    pub overtime_cap_s: f64,
    /// Sim steps per decision (the action is held). 4 → 30 decisions per second.
    pub action_repeat: u32,
    /// Learner team: 0, 1, or −1 = random per episode (seeded).
    pub learner_team: i32,
    /// Faceoff jitter (m) at the episode start: the puck by up to ±puck, each player by up to
    /// ±player, per axis. (Match mode: only the opening faceoff; the sim resets after goals exactly.)
    pub jitter_puck: f64,
    pub jitter_player: f64,
    /// Match mode: also jitter every faceoff after a goal (default true). The sim resets to exactly
    /// the same spawn after every goal; without jitter a policy memorizes that one faceoff (t1 smoke
    /// run: 95% match wins vs the chaser, 5% wins from jittered starts).
    pub jitter_every_faceoff: bool,
    /// Default opponent for every env: "chaser", "defender", "cautious", "idle" or "external".
    pub opponent: String,
    /// Base reward weights.
    pub reward: RewardWeights,
    /// Style vector written into every env's observation (padded / truncated to STYLE_DIM).
    pub style: Vec<f32>,
    /// "off": style slot as given, reward = base weights. "fixed": reward = the style mapping of
    /// the env's style. "sample": a new style per episode (the first STYLE_USED entries uniform in
    /// [−1, 1), the rest 0), reward = its mapping.
    pub style_mode: String,
    /// Worker threads (0 = rayon default, 1 = run on the calling thread).
    pub threads: usize,
    /// Rayon work unit: at least this many envs per task (less scheduling overhead per env).
    pub chunk: usize,
    /// Accumulate per-part CPU time (sim / reward / opponent bot / observe), read with
    /// `Batch::profile`. Off by default (it adds clock reads every sim step).
    pub profile: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            mode: "first_goal".into(),
            max_steps: 0,
            overtime_cap_s: 120.0,
            action_repeat: 4,
            learner_team: -1,
            jitter_puck: 1.5,
            jitter_player: 1.0,
            jitter_every_faceoff: true,
            opponent: "chaser".into(),
            reward: RewardWeights::default(),
            style: vec![],
            style_mode: "off".into(),
            threads: 0,
            chunk: 16,
            profile: false,
        }
    }
}

impl Config {
    pub fn is_match(&self) -> bool {
        self.mode == "match"
    }
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.mode.as_str(), "first_goal" | "match") {
            return Err(format!("mode must be \"first_goal\" or \"match\", not {:?}", self.mode));
        }
        if !matches!(self.style_mode.as_str(), "off" | "fixed" | "sample") {
            return Err(format!("style_mode must be \"off\", \"fixed\" or \"sample\", not {:?}", self.style_mode));
        }
        Ok(())
    }
    /// The effective step limit (playing sim steps).
    pub fn step_limit(&self, p: &Params) -> u64 {
        if self.max_steps > 0 {
            self.max_steps
        } else if self.is_match() {
            ((p.match_length_s + self.overtime_cap_s) / p.timestep).round() as u64
        } else {
            3600
        }
    }
}

pub fn opponent_id(name: &str) -> Result<i32, String> {
    if name == "external" {
        return Ok(OPP_EXTERNAL);
    }
    Kind::ALL.iter().find(|k| k.name() == name).map(|k| *k as i32).ok_or_else(|| format!("unknown opponent '{name}'"))
}

/// SplitMix64: tiny, seedable, good enough for start jitter and style sampling.
#[derive(Clone, Copy, Debug)]
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [−1, 1).
    pub fn sym(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

#[derive(Clone, Debug)]
pub enum Opponent {
    Scripted(Scripted),
    External,
}

#[derive(Clone, Debug)]
pub struct Env {
    pub state: State,
    pub rng: Rng,
    /// Learner slot (0 or 1); the opponent is the other slot.
    pub learner: usize,
    pub opponent: Opponent,
    pub style: [f32; STYLE_DIM],
    /// Base weights (config or set_reward_weights); `weights` = the effective ones (style mapping).
    pub base: RewardWeights,
    pub weights: RewardWeights,
    pub tracker: Tracker,
    pub ep_steps: u64,
    pub ep_return: f32,
    pub ep_raw: [f32; N_TERMS],
    pub goals: [u32; 2],
    pub opp_possession: f32,
    pub episodes: u64,
    /// Profile counters (ns): sim, reward tracker, opponent bot, observe (+ reset).
    pub prof: [u64; 4],
}

const FROZEN: [PlayerInput; 2] = [
    PlayerInput { move_x: 0.0, move_y: 0.0, stick_target_angle: 0.0, dash: false, lift: false, stick_up: false },
    PlayerInput { move_x: 0.0, move_y: 0.0, stick_target_angle: 0.0, dash: false, lift: false, stick_up: false },
];

fn frozen(s: &State) -> bool {
    matches!(s.match_state.map(|m| m.phase), Some(MatchPhase::Countdown | MatchPhase::GoalPause))
}

fn score(s: &State, t: usize) -> u32 {
    s.score.get(t).copied().unwrap_or(0)
}

impl Env {
    fn new(seed: u64, index: u64, cfg: &Config, opp: i32) -> Self {
        // Distinct, decorrelated stream per env index.
        let mut r = Rng(seed ^ index.wrapping_mul(0xD1B54A32D192ED03));
        r.next_u64();
        let mut style = [0f32; STYLE_DIM];
        for (d, s) in style.iter_mut().zip(&cfg.style) {
            *d = *s;
        }
        let mut e = Env {
            state: State::default(),
            rng: r,
            learner: 0,
            opponent: make_opponent(opp),
            style,
            base: cfg.reward,
            weights: cfg.reward,
            tracker: Tracker::default(),
            ep_steps: 0,
            ep_return: 0.0,
            ep_raw: [0.0; N_TERMS],
            goals: [0; 2],
            opp_possession: 0.0,
            episodes: 0,
            prof: [0; 4],
        };
        e.refresh_weights(cfg);
        e
    }

    pub fn opp_slot(&self) -> usize {
        1 - self.learner
    }

    /// Recompute the effective weights from the base weights and the style.
    pub fn refresh_weights(&mut self, cfg: &Config) {
        self.weights = if cfg.style_mode == "off" { self.base } else { self.base.style_weights(&self.style) };
    }

    pub fn reset(&mut self, p: &Params, cfg: &Config) {
        let mut s = if cfg.is_match() { State::new_match(p, 2) } else { State::new_default(p, 2) };
        self.jitter(&mut s, cfg);
        let team = if cfg.learner_team < 0 { (self.rng.next_u64() >> 63) as usize } else { cfg.learner_team.min(1) as usize };
        if cfg.style_mode == "sample" {
            self.style = [0.0; STYLE_DIM];
            for k in 0..STYLE_USED.min(STYLE_DIM) {
                self.style[k] = self.rng.sym() as f32;
            }
        }
        self.refresh_weights(cfg);
        // Slot i has team i % 2 in State::new_default / new_match, so slot == team for 1v1.
        self.learner = team;
        self.state = s;
        self.tracker = Tracker::default();
        self.ep_steps = 0;
        self.ep_return = 0.0;
        self.ep_raw = [0.0; N_TERMS];
        self.goals = [0; 2];
        self.opp_possession = 0.0;
        if let Opponent::Scripted(b) = &mut self.opponent {
            b.reset();
        }
        self.skip_frozen(p);
    }

    /// Faceoff jitter: the puck by up to ±jitter_puck, each player by up to ±jitter_player, per axis.
    fn jitter(&mut self, s: &mut State, cfg: &Config) {
        s.puck.x = cfg.jitter_puck * self.rng.sym();
        s.puck.y = cfg.jitter_puck * self.rng.sym();
        for pl in s.players.iter_mut() {
            pl.x += cfg.jitter_player * self.rng.sym();
            pl.y += cfg.jitter_player * self.rng.sym();
        }
    }

    /// Fast-forward a countdown or goal pause (inputs are ignored in them; no decision is spent).
    fn skip_frozen(&mut self, p: &Params) {
        while frozen(&self.state) {
            sim::step(&mut self.state, &FROZEN, p);
        }
    }

    pub fn observe(&self, p: &Params, obs: &mut [f32], opp_obs: &mut [f32]) {
        observe(&self.state, self.learner, &self.style, p, obs);
        observe(&self.state, self.opp_slot(), &[], p, opp_obs);
    }

    /// One decision step. Returns (reward, done) and writes info.
    pub fn step(&mut self, act: &[f32], opp_act: &[f32], p: &Params, cfg: &Config, info: &mut [f32]) -> (f32, bool) {
        let me = self.learner;
        let op = self.opp_slot();
        let a = Action::from_slice(act).to_input(self.state.players[me].team);
        let ext = Action::from_slice(opp_act).to_input(self.state.players[op].team);
        let mut raw = [0f32; N_TERMS];
        let mut inputs = [PlayerInput::default(); 2];
        let mut end = END_RUNNING;
        let limit = cfg.step_limit(p);
        let my_team = self.state.players[me].team as usize;
        let prof = cfg.profile;
        let lap = |t: &mut std::time::Instant, acc: &mut u64| {
            if prof {
                let now = std::time::Instant::now();
                *acc += (now - *t).as_nanos() as u64;
                *t = now;
            }
        };
        for _ in 0..cfg.action_repeat.max(1) {
            let mut t = std::time::Instant::now();
            inputs[me] = a;
            inputs[op] = match &mut self.opponent {
                Opponent::Scripted(b) => b.act(&self.state, op, p).to_input(self.state.players[op].team),
                Opponent::External => ext,
            };
            lap(&mut t, &mut self.prof[2]);
            let before = (self.state.puck, [score(&self.state, 0), score(&self.state, 1)]);
            sim::step(&mut self.state, &inputs, p);
            self.ep_steps += 1;
            lap(&mut t, &mut self.prof[0]);
            self.opp_possession += self.tracker.sim_step(&before, &self.state, me, p, &mut raw);
            lap(&mut t, &mut self.prof[1]);
            let scored = score(&self.state, 0) != before.1[0] || score(&self.state, 1) != before.1[1];
            if scored {
                let for_me = score(&self.state, my_team) != before.1[my_team];
                self.goals[if for_me { 0 } else { 1 }] += 1;
                if !cfg.is_match() {
                    end = if for_me { END_WIN } else { END_LOSS };
                    break;
                }
            }
            if let Some(m) = self.state.match_state {
                if m.phase == MatchPhase::Finished {
                    let won = m.winner == Some(my_team as u8);
                    raw[6] += if won { 1.0 } else { -1.0 };
                    end = if won { END_WIN } else { END_LOSS };
                    break;
                }
                if scored {
                    // Goal pause + reset to spawn + countdown are skipped; the next decision sees the faceoff.
                    self.skip_frozen(p);
                    if cfg.jitter_every_faceoff {
                        let mut s = std::mem::take(&mut self.state);
                        self.jitter(&mut s, cfg);
                        self.state = s;
                    }
                    if let Opponent::Scripted(b) = &mut self.opponent {
                        b.reset();
                    }
                    break;
                }
            }
            if self.ep_steps >= limit {
                end = END_TRUNCATED;
                break;
            }
        }
        let r = self.weights.reward(&raw);
        self.ep_return += r;
        for (e, x) in self.ep_raw.iter_mut().zip(&raw) {
            *e += x;
        }
        info[0] = end;
        info[1] = self.ep_steps as f32;
        info[2] = self.ep_return;
        info[3] = my_team as f32;
        info[4] = self.goals[0] as f32;
        info[5] = self.goals[1] as f32;
        info[6] = self.opp_possession;
        info[INFO_RAW..INFO_RAW + N_TERMS].copy_from_slice(&self.ep_raw);
        let done = end != END_RUNNING;
        if done {
            self.episodes += 1;
        }
        (r, done)
    }
}

fn make_opponent(id: i32) -> Opponent {
    match Kind::from_id(id.max(0) as u32) {
        Some(k) if id >= 0 => Opponent::Scripted(Scripted::new(k)),
        _ => Opponent::External,
    }
}

pub struct Batch {
    pub params: Params,
    pub cfg: Config,
    pub envs: Vec<Env>,
    pool: Option<rayon::ThreadPool>,
    scratch_opp_obs: Vec<f32>,
    scratch_opp_act: Vec<f32>,
    scratch_final: Vec<f32>,
    /// Wall time spent inside `step` (ns, cumulative).
    pub step_ns: u64,
}

/// Per-env output slices for one step.
struct Out<'a> {
    obs: &'a mut [f32],
    opp_obs: &'a mut [f32],
    final_obs: &'a mut [f32],
    rew: &'a mut f32,
    done: &'a mut u8,
    info: &'a mut [f32],
}

impl Batch {
    pub fn new(params: Params, cfg: Config, n: usize, seed: u64) -> Result<Self, String> {
        if n == 0 {
            return Err("n_envs must be > 0".into());
        }
        cfg.validate()?;
        let opp = opponent_id(&cfg.opponent)?;
        let envs = (0..n).map(|i| Env::new(seed, i as u64, &cfg, opp)).collect();
        let pool = if cfg.threads > 1 {
            Some(rayon::ThreadPoolBuilder::new().num_threads(cfg.threads).build().map_err(|e| e.to_string())?)
        } else {
            None
        };
        let mut b = Batch {
            params,
            cfg,
            envs,
            pool,
            scratch_opp_obs: vec![0.0; n * OBS_DIM],
            scratch_opp_act: vec![0.0; n * ACT_DIM],
            scratch_final: vec![0.0; n * OBS_DIM],
            step_ns: 0,
        };
        let (p, c) = (&b.params, &b.cfg);
        for e in b.envs.iter_mut() {
            e.reset(p, c);
        }
        Ok(b)
    }

    pub fn len(&self) -> usize {
        self.envs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    fn parallel(&self) -> bool {
        self.cfg.threads != 1 && self.envs.len() >= 4
    }

    pub fn set_opponent(&mut self, env: usize, id: i32) {
        self.envs[env].opponent = make_opponent(id);
    }

    /// Set env `env`'s style (padded with 0 / truncated) and re-derive its weights (style_mode
    /// "fixed"; in "sample" mode the next reset draws a new style).
    pub fn set_style(&mut self, env: usize, style: &[f32]) {
        let e = &mut self.envs[env];
        for (k, d) in e.style.iter_mut().enumerate() {
            *d = style.get(k).copied().unwrap_or(0.0);
        }
        e.refresh_weights(&self.cfg);
    }

    /// Set env `env`'s BASE reward weights (the style mapping applies on top unless style_mode is "off").
    pub fn set_reward_weights(&mut self, env: usize, w: RewardWeights) {
        let e = &mut self.envs[env];
        e.base = w;
        e.refresh_weights(&self.cfg);
    }

    /// Reset every env and write first observations.
    pub fn reset_all(&mut self, obs: &mut [f32], opp_obs: Option<&mut [f32]>) {
        let (p, c) = (&self.params, &self.cfg);
        for e in self.envs.iter_mut() {
            e.reset(p, c);
        }
        let oo = match opp_obs {
            Some(o) => o,
            None => &mut self.scratch_opp_obs[..],
        };
        for (i, e) in self.envs.iter().enumerate() {
            e.observe(p, &mut obs[i * OBS_DIM..(i + 1) * OBS_DIM], &mut oo[i * OBS_DIM..(i + 1) * OBS_DIM]);
        }
    }

    /// Step every env; finished envs are reset and their obs is the new episode's first. For
    /// finished envs, `final_obs` (if given) receives the learner's observation of the final state
    /// BEFORE the reset (to bootstrap the value of a truncated episode); other rows are untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        actions: &[f32],
        opp_actions: Option<&[f32]>,
        obs: &mut [f32],
        opp_obs: Option<&mut [f32]>,
        final_obs: Option<&mut [f32]>,
        rewards: &mut [f32],
        dones: &mut [u8],
        info: &mut [f32],
    ) {
        let n = self.envs.len();
        let parallel = self.parallel();
        assert!(actions.len() >= n * ACT_DIM && obs.len() >= n * OBS_DIM && rewards.len() >= n && dones.len() >= n && info.len() >= n * INFO_DIM);
        let opp_actions: &[f32] = opp_actions.unwrap_or(&self.scratch_opp_act[..]);
        let opp_obs: &mut [f32] = match opp_obs {
            Some(o) => o,
            None => &mut self.scratch_opp_obs[..],
        };
        let final_obs: &mut [f32] = match final_obs {
            Some(o) => o,
            None => &mut self.scratch_final[..],
        };
        let (p, c) = (&self.params, &self.cfg);
        let work = |((e, (a, oa)), o): ((&mut Env, (&[f32], &[f32])), Out)| {
            let (rew, done) = e.step(a, oa, p, c, o.info);
            *o.rew = rew;
            *o.done = done as u8;
            let t0 = if c.profile { Some(std::time::Instant::now()) } else { None };
            if done {
                observe(&e.state, e.learner, &e.style, p, o.final_obs);
                e.reset(p, c);
            }
            e.observe(p, o.obs, o.opp_obs);
            if let Some(t0) = t0 {
                e.prof[3] += t0.elapsed().as_nanos() as u64;
            }
        };
        let outs = obs
            .chunks_mut(OBS_DIM)
            .zip(opp_obs.chunks_mut(OBS_DIM))
            .zip(final_obs.chunks_mut(OBS_DIM))
            .zip(rewards.iter_mut())
            .zip(dones.iter_mut())
            .zip(info.chunks_mut(INFO_DIM))
            .map(|(((((obs, opp_obs), final_obs), rew), done), info)| Out { obs, opp_obs, final_obs, rew, done, info })
            .take(n);
        let t_step = std::time::Instant::now();
        if parallel {
            let outs: Vec<Out> = outs.collect();
            let chunk = c.chunk.max(1);
            let run = || {
                self.envs
                    .par_iter_mut()
                    .zip(actions.par_chunks(ACT_DIM).zip(opp_actions.par_chunks(ACT_DIM)))
                    .zip(outs.into_par_iter())
                    .with_min_len(chunk)
                    .for_each(work)
            };
            match &self.pool {
                Some(pool) => pool.install(run),
                None => run(),
            }
        } else {
            self.envs.iter_mut().zip(actions.chunks(ACT_DIM).zip(opp_actions.chunks(ACT_DIM))).zip(outs).for_each(work);
        }
        self.step_ns += t_step.elapsed().as_nanos() as u64;
    }

    /// [wall ns inside `step` (all calls), then CPU ns summed over envs: sim, reward, opponent bot,
    /// observe + reset]. The per-part counters are only filled with `profile: true`.
    pub fn profile(&self) -> [u64; 5] {
        let mut out = [self.step_ns, 0, 0, 0, 0];
        for e in &self.envs {
            for k in 0..4 {
                out[k + 1] += e.prof[k];
            }
        }
        out
    }

    /// FNV-1a over every env's state hash, RNG state, learner slot and episode counters.
    pub fn hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut eat = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        for e in &self.envs {
            eat(e.state.hash());
            eat(e.rng.0);
            eat(e.learner as u64);
            eat(e.ep_steps);
            eat(e.episodes);
        }
        h
    }
}
