//! slapmini sim core: deterministic 2.5D top-down hockey physics.
//!
//! Pure by design: no I/O, no wall-clock reads, no RNG. The same code runs natively
//! (training, calibration) and in the browser (wasm32-unknown-unknown).

pub mod params;
pub mod rules;
pub mod state;
pub mod step;

pub use params::{ParamEntry, ParamFile, Params, DEFAULT_PARAMS_JSON};
pub use rules::{MatchPhase, MatchState};
pub use state::{Player, PlayerInput, Puck, State};
pub use step::step;
