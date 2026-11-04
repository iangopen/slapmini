//! `track: "player0"` scenarios: replay samples player 0, and the fitter recovers player_accel from them.

use calib::{fit, replay, Case, InputEvent, Scenario};
use sim::{Params, PlayerInput, State};

fn skate(p: &Params) -> Scenario {
    let mut s = State::new_default(p, 1);
    s.puck.y = 15.0; // out of the way
    let w = PlayerInput { move_x: 1.0, ..Default::default() };
    Scenario {
        name: "skate".into(),
        description: "player 0 holds +x for 1.5 s, then releases".into(),
        steps: 300,
        initial_state: s,
        inputs: vec![InputEvent { step: 0, player: 0, input: w }, InputEvent { step: 180, player: 0, input: PlayerInput::default() }],
        track: "player0".into(),
    }
}

#[test]
fn replay_samples_the_player() {
    let p = Params::default();
    let sc = skate(&p);
    let tr = replay(&sc, &p);
    let x0 = sc.initial_state.players[0].x;
    assert!(tr.last().unwrap().x > x0 + 3.0, "player should have skated forward");
    assert!(tr.iter().all(|s| s.z == 0.0));
    // The default (puck) track is unchanged by the field.
    let mut puck = sc.clone();
    puck.track = String::new();
    assert!(replay(&puck, &p).iter().all(|s| s.x == 0.0));
    // Serde: the field is optional and round-trips.
    let back = Scenario::from_json(&sc.to_json()).unwrap();
    assert_eq!(back.track, "player0");
}

#[test]
fn fitter_recovers_player_accel() {
    let mut truth = Params::default();
    truth.player_accel = 9.0;
    let sc = skate(&truth);
    let measured = replay(&sc, &truth);
    let mut start = truth.clone();
    start.player_accel = 13.0;
    let r = fit(&[Case { scenario: sc, measured }], &start, &["player_accel"], 200).unwrap();
    assert!((r.params.player_accel - 9.0).abs() < 1e-3, "got {}", r.params.player_accel);
}
