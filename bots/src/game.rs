//! Running scripted 1v1 games (used by the tests, the tournament example and sim-ffi).

use crate::scripted::Bot;
use sim::{Params, PlayerInput, State};

/// A 1v1 faceoff: puck at center ice, team 0 player at −x (index 0), team 1 player at +x (index 1).
pub fn faceoff(p: &Params) -> State {
    State::new_default(p, 2)
}

/// Outcome of one game segment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// Goals scored by team 0 and team 1 during the run.
    pub goals: [u32; 2],
    /// Steps simulated.
    pub steps: u64,
}

/// Step `state` with one bot per player for up to `max_steps`. If `stop_on_goal`, stop at the
/// first goal; otherwise, after a goal the state is reset to a faceoff (score kept).
pub fn play(
    state: &mut State,
    bots: &mut [&mut dyn Bot],
    p: &Params,
    max_steps: u64,
    stop_on_goal: bool,
) -> Outcome {
    play_with_faceoffs(state, bots, p, max_steps, stop_on_goal, &mut |_| faceoff(p))
}

/// Like `play`, but after the k-th goal (k = 1, 2, ...) the state is reset to `next_faceoff(k)`
/// (the score is carried over).
pub fn play_with_faceoffs(
    state: &mut State,
    bots: &mut [&mut dyn Bot],
    p: &Params,
    max_steps: u64,
    stop_on_goal: bool,
    next_faceoff: &mut dyn FnMut(u32) -> State,
) -> Outcome {
    let mut out = Outcome::default();
    let start = [state.score.first().copied().unwrap_or(0), state.score.get(1).copied().unwrap_or(0)];
    let mut inputs = vec![PlayerInput::default(); state.players.len()];
    for _ in 0..max_steps {
        for (i, b) in bots.iter_mut().enumerate() {
            let team = state.players[i].team;
            inputs[i] = b.act(state, i, p).to_input(team);
        }
        let before = state.score.clone();
        sim::step(state, &inputs, p);
        out.steps += 1;
        if state.score != before {
            if stop_on_goal {
                break;
            }
            let score = state.score.clone();
            *state = next_faceoff(score.iter().sum());
            state.score = score;
            for b in bots.iter_mut() {
                b.reset();
            }
        }
    }
    out.goals = [state.score[0] - start[0], state.score[1] - start[1]];
    out
}
