//! Browser bindings. State and inputs cross the boundary as JSON strings (same shapes as calib scenario files).

use sim::{Params, PlayerInput, State};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Sim {
    params: Params,
    state: State,
}

#[wasm_bindgen]
impl Sim {
    /// `params_json`: contents of a params.json; empty string uses the embedded default.
    /// `n_players`: players in the default starting state.
    #[wasm_bindgen(constructor)]
    pub fn new(params_json: &str, n_players: usize) -> Result<Sim, JsError> {
        let params = if params_json.trim().is_empty() {
            Params::default()
        } else {
            Params::from_json(params_json).map_err(|e| JsError::new(&e))?
        };
        let state = State::new_default(&params, n_players);
        Ok(Sim { params, state })
    }

    /// Advance one step. `inputs_json`: array of PlayerInput objects, one per player.
    pub fn step(&mut self, inputs_json: &str) -> Result<(), JsError> {
        let inputs: Vec<PlayerInput> = if inputs_json.trim().is_empty() {
            Vec::new()
        } else {
            serde_json::from_str(inputs_json).map_err(|e| JsError::new(&e.to_string()))?
        };
        sim::step(&mut self.state, &inputs, &self.params);
        Ok(())
    }

    /// Current state as JSON.
    pub fn get_state(&self) -> String {
        serde_json::to_string(&self.state).expect("state serializes")
    }

    /// Replace the state (e.g. from a scenario file's initial_state).
    pub fn set_state(&mut self, state_json: &str) -> Result<(), JsError> {
        self.state = serde_json::from_str(state_json).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    /// Determinism hash of the current state (hex).
    pub fn state_hash(&self) -> String {
        format!("{:016x}", self.state.hash())
    }
}
