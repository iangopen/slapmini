//! Browser bindings. State, inputs, params and scenarios cross the boundary as JSON strings,
//! in the same shapes as params.json and calib scenario files.

use serde::Deserialize;
use sim::{ParamFile, Params, PlayerInput, State, DEFAULT_PARAMS_JSON};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Sim {
    file: ParamFile,
    params: Params,
    state: State,
    /// Scripted bots per player slot (None = driven from JS).
    bots: Vec<Option<bots::Scripted>>,
    bot_goals: u32,
    bot_fresh: bool,
}

/// The part of a calib scenario file that `reset` needs; other fields are ignored.
#[derive(Deserialize)]
struct ScenarioHead {
    initial_state: State,
}

fn js<E: ToString>(e: E) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen]
impl Sim {
    /// `params_json`: contents of a params.json; empty string uses the embedded default.
    /// `n_players`: players in the default starting state.
    #[wasm_bindgen(constructor)]
    pub fn new(params_json: &str, n_players: usize) -> Result<Sim, JsError> {
        let src = if params_json.trim().is_empty() { DEFAULT_PARAMS_JSON } else { params_json };
        let file = ParamFile::from_json(src).map_err(js)?;
        let params = file.to_params().map_err(js)?;
        let state = State::new_default(&params, n_players);
        Ok(Sim { file, params, state, bots: Vec::new(), bot_goals: 0, bot_fresh: false })
    }

    /// Advance one step. `inputs_json`: array of PlayerInput objects, one per player.
    pub fn step(&mut self, inputs_json: &str) -> Result<(), JsError> {
        let inputs: Vec<PlayerInput> =
            if inputs_json.trim().is_empty() { Vec::new() } else { serde_json::from_str(inputs_json).map_err(js)? };
        sim::step(&mut self.state, &inputs, &self.params);
        Ok(())
    }

    /// Current state as JSON.
    pub fn get_state(&self) -> String {
        serde_json::to_string(&self.state).expect("state serializes")
    }

    /// Replace the state wholesale.
    pub fn set_state(&mut self, state_json: &str) -> Result<(), JsError> {
        self.state = serde_json::from_str(state_json).map_err(js)?;
        Ok(())
    }

    /// Replace all params on the running sim (full params.json shape). State is kept.
    pub fn set_params(&mut self, params_json: &str) -> Result<(), JsError> {
        let file = ParamFile::from_json(params_json).map_err(js)?;
        self.params = file.to_params().map_err(js)?;
        self.file = file;
        Ok(())
    }

    /// Current params in params.json shape (values plus provenance).
    pub fn get_params(&self) -> String {
        self.file.to_json()
    }

    /// Set the state to a calib scenario's `initial_state`. Inputs are driven by the caller.
    pub fn reset(&mut self, scenario_json: &str) -> Result<(), JsError> {
        let head: ScenarioHead = serde_json::from_str(scenario_json).map_err(js)?;
        self.state = head.initial_state;
        Ok(())
    }

    /// Start a new match (rules on): players at their spawn points, 0-0, full clock, opening countdown.
    /// The match state is part of `get_state` / `set_state` (`match_state`, null = free play).
    pub fn start_match(&mut self, n_players: usize) {
        self.state = State::new_match(&self.params, n_players);
    }

    /// Determinism hash of the current state (hex).
    pub fn state_hash(&self) -> String {
        format!("{:016x}", self.state.hash())
    }

    // ---------- bots (the same Rust code as training: bots::observe, bots::Action, bots::Scripted) ----------

    /// The training observation of player `player` (bots::observe, OBS_DIM floats, canonical frame).
    /// `style` fills the style slot (missing entries 0).
    pub fn observe(&self, player: usize, style: &[f32]) -> Result<Vec<f32>, JsError> {
        if player >= self.state.players.len() {
            return Err(JsError::new("player index out of range"));
        }
        let mut out = vec![0f32; bots::OBS_DIM];
        bots::observe(&self.state, player, style, &self.params, &mut out);
        Ok(out)
    }

    /// Observation length and layout version (checked against the ONNX sidecar).
    pub fn obs_dim() -> usize {
        bots::OBS_DIM
    }
    pub fn obs_version() -> u32 {
        bots::OBS_VERSION
    }
    /// Action layout version (ACT v4: + brake).
    pub fn act_version() -> u32 {
        bots::ACT_VERSION
    }

    /// A policy action (canonical frame, the training action format) → world PlayerInput JSON for
    /// `player` (bots::Action::to_input applies the team rotation). `move_dir` 0..8 (bots::MOVE_DIRS).
    pub fn action_input(&self, player: usize, move_dir: u32, stick_angle: f64, dash: bool, lift: bool, stick_up: bool, brake: bool) -> Result<String, JsError> {
        let pl = self.state.players.get(player).ok_or_else(|| JsError::new("player index out of range"))?;
        let a = bots::Action { move_dir: move_dir.min(8) as u8, stick_angle: bots::wrap_angle(stick_angle), dash, lift, stick_up, brake };
        Ok(serde_json::to_string(&a.to_input(pl.team)).expect("input serializes"))
    }

    /// Drive `player` with a scripted bot: "chaser", "defender", "cautious" or "idle" ("" removes it).
    pub fn set_scripted(&mut self, player: usize, kind: &str) -> Result<(), JsError> {
        if self.bots.len() <= player {
            self.bots.resize(player + 1, None);
        }
        self.bots[player] = if kind.is_empty() {
            None
        } else {
            let k = bots::Kind::ALL.iter().copied().find(|k| k.name() == kind).ok_or_else(|| JsError::new(&format!("unknown bot {kind}")))?;
            Some(bots::Scripted::new(k))
        };
        Ok(())
    }

    /// The scripted bot's input for `player` this step (world PlayerInput JSON). Bots forget their
    /// per-episode memory whenever the players were reset (a goal reset or a new match), as in training.
    pub fn scripted_input(&mut self, player: usize) -> Result<String, JsError> {
        let goals: u32 = self.state.score.iter().sum();
        let fresh = self.state.match_state.map(|m| m.phase) == Some(sim::MatchPhase::Countdown);
        if goals != self.bot_goals || (fresh && !self.bot_fresh) {
            for b in self.bots.iter_mut().flatten() {
                bots::Bot::reset(b);
            }
            self.bot_goals = goals;
        }
        self.bot_fresh = fresh;
        let team = self.state.players.get(player).ok_or_else(|| JsError::new("player index out of range"))?.team;
        let b = self.bots.get_mut(player).and_then(|b| b.as_mut()).ok_or_else(|| JsError::new("no scripted bot on this player"))?;
        let inp = bots::Bot::act(b, &self.state, player, &self.params).to_input(team);
        Ok(serde_json::to_string(&inp).expect("input serializes"))
    }
}
