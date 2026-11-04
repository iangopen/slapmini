//! Batched 1v1 environments. Each environment owns its State, a seeded RNG (used only at resets),
//! its opponent (a scripted bot or an external policy slot) and its reward weights and style vector.
//! Stepping is a pure function of (env, actions), so results do not depend on thread scheduling.

use crate::reward::{RewardWeights, Tracker, N_TERMS};
use bots::{observe, Action, Bot, Kind, Scripted, ACT_DIM, OBS_DIM, STYLE_DIM};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sim::{Params, PlayerInput, State};

/// Per-env info written every step (meaningful for the finished episode when done = 1):
/// [end reason (0 running, 1 goal for, 2 goal against, 3 step limit), episode sim steps,
///  episode return, learner team, raw reward term sums × N_TERMS].
pub const INFO_DIM: usize = 4 + N_TERMS;

pub const END_RUNNING: f32 = 0.0;
pub const END_GOAL_FOR: f32 = 1.0;
pub const END_GOAL_AGAINST: f32 = 2.0;
pub const END_LIMIT: f32 = 3.0;

/// Opponent id for "an external policy fills the opponent action buffer".
pub const OPP_EXTERNAL: i32 = -1;

/// Batch configuration (JSON). Every field has a default.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Episode step limit in SIM steps (default 30 s at 120 Hz).
    pub max_steps: u64,
    /// Sim steps per decision (the action is held). 4 → 30 decisions per second.
    pub action_repeat: u32,
    /// Learner team: 0, 1, or −1 = random per episode (seeded).
    pub learner_team: i32,
    /// Faceoff jitter (m): the puck by up to ±puck, each player by up to ±player, per axis.
    pub jitter_puck: f64,
    pub jitter_player: f64,
    /// Default opponent for every env: "chaser", "defender", "cautious", "idle" or "external".
    pub opponent: String,
    pub reward: RewardWeights,
    /// Style vector written into every env's observation (padded / truncated to STYLE_DIM).
    pub style: Vec<f32>,
    /// Worker threads (0 = rayon default, 1 = run on the calling thread).
    pub threads: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_steps: 3600,
            action_repeat: 4,
            learner_team: -1,
            jitter_puck: 1.5,
            jitter_player: 1.0,
            opponent: "chaser".into(),
            reward: RewardWeights::default(),
            style: vec![],
            threads: 0,
        }
    }
}

pub fn opponent_id(name: &str) -> Result<i32, String> {
    if name == "external" {
        return Ok(OPP_EXTERNAL);
    }
    Kind::ALL.iter().find(|k| k.name() == name).map(|k| *k as i32).ok_or_else(|| format!("unknown opponent '{name}'"))
}

/// SplitMix64: tiny, seedable, good enough for start jitter.
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
    pub weights: RewardWeights,
    pub tracker: Tracker,
    pub ep_steps: u64,
    pub ep_return: f32,
    pub ep_raw: [f32; N_TERMS],
    pub episodes: u64,
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
        Env {
            state: State::default(),
            rng: r,
            learner: 0,
            opponent: make_opponent(opp),
            style,
            weights: cfg.reward,
            tracker: Tracker::default(),
            ep_steps: 0,
            ep_return: 0.0,
            ep_raw: [0.0; N_TERMS],
            episodes: 0,
        }
    }

    pub fn opp_slot(&self) -> usize {
        1 - self.learner
    }

    pub fn reset(&mut self, p: &Params, cfg: &Config) {
        let mut s = State::new_default(p, 2);
        s.puck.x = cfg.jitter_puck * self.rng.sym();
        s.puck.y = cfg.jitter_puck * self.rng.sym();
        for pl in s.players.iter_mut() {
            pl.x += cfg.jitter_player * self.rng.sym();
            pl.y += cfg.jitter_player * self.rng.sym();
        }
        let team = if cfg.learner_team < 0 { (self.rng.next_u64() >> 63) as usize } else { cfg.learner_team.min(1) as usize };
        // Slot i has team i % 2 in State::new_default, so slot == team for 1v1.
        self.learner = team;
        self.state = s;
        self.tracker = Tracker::default();
        self.ep_steps = 0;
        self.ep_return = 0.0;
        self.ep_raw = [0.0; N_TERMS];
        if let Opponent::Scripted(b) = &mut self.opponent {
            b.reset();
        }
    }

    pub fn observe(&self, obs: &mut [f32], opp_obs: &mut [f32]) {
        observe(&self.state, self.learner, &self.style, obs);
        observe(&self.state, self.opp_slot(), &[], opp_obs);
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
        for _ in 0..cfg.action_repeat.max(1) {
            inputs[me] = a;
            inputs[op] = match &mut self.opponent {
                Opponent::Scripted(b) => b.act(&self.state, op, p).to_input(self.state.players[op].team),
                Opponent::External => ext,
            };
            let before = self.state.clone();
            sim::step(&mut self.state, &inputs, p);
            self.ep_steps += 1;
            self.tracker.sim_step(&before, &self.state, me, p, &mut raw);
            if self.state.score != before.score {
                end = if raw[0] > 0.0 { END_GOAL_FOR } else { END_GOAL_AGAINST };
                break;
            }
            if self.ep_steps >= cfg.max_steps {
                end = END_LIMIT;
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
        info[3] = self.state.players[me].team as f32;
        info[4..4 + N_TERMS].copy_from_slice(&self.ep_raw);
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
}

impl Batch {
    pub fn new(params: Params, cfg: Config, n: usize, seed: u64) -> Result<Self, String> {
        if n == 0 {
            return Err("n_envs must be > 0".into());
        }
        let opp = opponent_id(&cfg.opponent)?;
        let envs = (0..n).map(|i| Env::new(seed, i as u64, &cfg, opp)).collect();
        let pool = if cfg.threads > 1 {
            Some(rayon::ThreadPoolBuilder::new().num_threads(cfg.threads).build().map_err(|e| e.to_string())?)
        } else {
            None
        };
        let mut b = Batch { params, cfg, envs, pool, scratch_opp_obs: vec![0.0; n * OBS_DIM], scratch_opp_act: vec![0.0; n * ACT_DIM] };
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
            e.observe(&mut obs[i * OBS_DIM..(i + 1) * OBS_DIM], &mut oo[i * OBS_DIM..(i + 1) * OBS_DIM]);
        }
    }

    /// Step every env; finished envs are reset and their obs is the new episode's first one.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        actions: &[f32],
        opp_actions: Option<&[f32]>,
        obs: &mut [f32],
        opp_obs: Option<&mut [f32]>,
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
        let (p, c) = (&self.params, &self.cfg);
        let work = |(((((((e, a), oa), o), oo), r), d), inf): (
            ((((((&mut Env, &[f32]), &[f32]), &mut [f32]), &mut [f32]), &mut f32), &mut u8),
            &mut [f32],
        )| {
            let (rew, done) = e.step(a, oa, p, c, inf);
            *r = rew;
            *d = done as u8;
            if done {
                e.reset(p, c);
            }
            e.observe(o, oo);
        };
        if parallel {
            let run = || {
                self.envs
                    .par_iter_mut()
                    .zip(actions.par_chunks(ACT_DIM))
                    .zip(opp_actions.par_chunks(ACT_DIM))
                    .zip(obs.par_chunks_mut(OBS_DIM))
                    .zip(opp_obs.par_chunks_mut(OBS_DIM))
                    .zip(rewards.par_iter_mut())
                    .zip(dones.par_iter_mut())
                    .zip(info.par_chunks_mut(INFO_DIM))
                    .for_each(work)
            };
            match &self.pool {
                Some(pool) => pool.install(run),
                None => {
                    let mut run = run;
                    run()
                }
            }
        } else {
            self.envs
                .iter_mut()
                .zip(actions.chunks(ACT_DIM))
                .zip(opp_actions.chunks(ACT_DIM))
                .zip(obs.chunks_mut(OBS_DIM))
                .zip(opp_obs.chunks_mut(OBS_DIM))
                .zip(rewards.iter_mut())
                .zip(dones.iter_mut())
                .zip(info.chunks_mut(INFO_DIM))
                .for_each(work);
        }
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
