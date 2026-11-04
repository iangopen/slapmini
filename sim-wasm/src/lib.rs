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
        Ok(Sim { file, params, state })
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

    /// Determinism hash of the current state (hex).
    pub fn state_hash(&self) -> String {
        format!("{:016x}", self.state.hash())
    }
}
