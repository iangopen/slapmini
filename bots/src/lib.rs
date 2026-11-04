//! slapmini bots: observation and action formats, the canonical (attacks +x) frame, and scripted bots.
//!
//! Pure and deterministic like `sim`: no I/O, no clock, no RNG.

pub mod action;
pub mod frame;
pub mod game;
pub mod obs;
pub mod scripted;

pub use action::{Action, ACT_DIM};
pub use frame::{rotate_swap, wrap_angle, Frame};
pub use obs::{observe, opponent_of, BASE_DIM, OBS_DIM, OBS_VERSION, STYLE_DIM};
pub use scripted::{Bot, Kind, Scripted};
