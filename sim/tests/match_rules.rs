//! Match rules: mercy end, clock end, sudden-death overtime, reset after a goal, frozen phases
//! ignore inputs, determinism. Rule values come from the default params.json.

use sim::{step, MatchPhase, MatchState, Params, PlayerInput, State};

fn m(s: &State) -> MatchState {
    s.match_state.expect("match state")
}

fn idle(s: &mut State, p: &Params, n: u64) {
    for _ in 0..n {
        step(s, &[], p);
    }
}

/// Step idle until the phase is `ph` (bounded).
fn run_until(s: &mut State, p: &Params, ph: MatchPhase) {
    for _ in 0..200_000 {
        if m(s).phase == ph {
            return;
        }
        step(s, &[], p);
    }
    panic!("never reached {ph:?}: {:?}", m(s));
}

/// While playing, slide the puck into the net that `team` attacks (team 0 → +x) and step until the goal registers.
fn score(s: &mut State, p: &Params, team: u8) {
    run_until(s, p, MatchPhase::Playing);
    let sign = if team == 0 { 1.0 } else { -1.0 };
    s.puck = sim::Puck { x: sign * (p.net_x - 1.0), vx: sign * 8.0, ..Default::default() };
    let before = s.score[team as usize];
    for _ in 0..240 {
        step(s, &[], p);
        if s.score[team as usize] > before {
            return;
        }
    }
    panic!("no goal for team {team}");
}

fn wild(t: u64) -> Vec<PlayerInput> {
    (0..2u64)
        .map(|k| PlayerInput {
            move_x: if (t / 7 + k) % 2 == 0 { 1.0 } else { -1.0 },
            move_y: if (t / 11 + k) % 3 == 0 { 1.0 } else { -0.5 },
            stick_target_angle: ((t * 13 + k * 50) % 628) as f64 * 0.01 - 3.14,
            dash: t % 30 == k,
            lift: t % 50 < 10,
        })
        .collect()
}

#[test]
fn new_match_starts_in_countdown_with_full_clock() {
    let p = Params::default();
    let s = State::new_match(&p, 2);
    let ms = m(&s);
    assert_eq!(ms.phase, MatchPhase::Countdown);
    assert_eq!(ms.clock_ticks, 36_000, "300 s at 120 Hz");
    assert_eq!(ms.phase_ticks, 360, "3 s countdown at 120 Hz");
    assert_eq!(s.score, vec![0, 0]);
    assert_eq!((s.players[0].x, s.players[1].x), (-p.player_spawn_x, p.player_spawn_x));
    // Free play stays the default.
    assert!(State::new_default(&p, 2).match_state.is_none());
}

#[test]
fn mercy_rule_is_a_five_goal_lead() {
    let p = Params::default();
    let mut s = State::new_match(&p, 2);
    // 5 goals in total without a 5-goal lead does not end the match: 4-1, then team 0 to 5-1 ends it.
    for t in [1, 0, 0, 0, 0] {
        score(&mut s, &p, t);
        assert_eq!(m(&s).phase, MatchPhase::GoalPause, "score {:?}", s.score);
    }
    assert_eq!(s.score, vec![4, 1]);
    score(&mut s, &p, 0);
    assert_eq!(s.score, vec![5, 1]);
    assert_eq!(m(&s).phase, MatchPhase::GoalPause, "lead 4 is not mercy");
    score(&mut s, &p, 0);
    assert_eq!(s.score, vec![6, 1]);
    let ms = m(&s);
    assert_eq!((ms.phase, ms.winner), (MatchPhase::Finished, Some(0)), "lead 5 ends it");
    assert!(ms.clock_ticks > 0 && !ms.overtime);
    // Finished is final: nothing moves, the score stays.
    let (puck, players) = (s.puck, s.players.clone());
    let tick = s.tick;
    for t in 0..500 {
        step(&mut s, &wild(t), &p);
    }
    assert_eq!(s.tick, tick + 500);
    assert_eq!(s.score, vec![6, 1]);
    assert_eq!((s.puck, &s.players, m(&s).winner), (puck, &players, Some(0)));
}

#[test]
fn clock_end_with_a_winner() {
    let p = Params::default();
    let mut s = State::new_match(&p, 2);
    score(&mut s, &p, 1);
    let played_before_goal = 36_000 - m(&s).clock_ticks;
    // The clock is stopped in the goal pause and the countdown.
    run_until(&mut s, &p, MatchPhase::Countdown);
    assert_eq!(36_000 - m(&s).clock_ticks, played_before_goal);
    run_until(&mut s, &p, MatchPhase::Playing);
    let mut last = m(&s);
    while m(&s).phase == MatchPhase::Playing {
        last = m(&s);
        step(&mut s, &[], &p);
    }
    let ms = m(&s);
    assert_eq!(last.clock_ticks, 1, "ends on the step the clock reaches 0");
    assert_eq!((ms.phase, ms.winner, ms.clock_ticks, ms.overtime), (MatchPhase::Finished, Some(1), 0, false));
    assert_eq!(s.score, vec![0, 1]);
}

#[test]
fn tie_at_zero_goes_to_overtime_and_the_next_goal_wins() {
    let p = Params::default();
    let mut s = State::new_match(&p, 2);
    score(&mut s, &p, 0);
    score(&mut s, &p, 1);
    run_until(&mut s, &p, MatchPhase::Playing);
    while !m(&s).overtime {
        step(&mut s, &[], &p);
    }
    let ms = m(&s);
    assert_eq!((ms.phase, ms.clock_ticks, ms.winner), (MatchPhase::Playing, 0, None), "tied at 0:00 plays on");
    assert_eq!(s.score, vec![1, 1]);
    idle(&mut s, &p, 5_000); // overtime has no clock
    assert_eq!((m(&s).phase, m(&s).clock_ticks), (MatchPhase::Playing, 0));
    score(&mut s, &p, 1);
    let ms = m(&s);
    assert_eq!((ms.phase, ms.winner, ms.overtime), (MatchPhase::Finished, Some(1), true));
    assert_eq!(s.score, vec![1, 2]);
}

#[test]
fn goal_pause_freezes_then_resets_to_spawn_and_counts_down() {
    let p = Params::default();
    let mut s = State::new_match(&p, 2);
    run_until(&mut s, &p, MatchPhase::Playing);
    for t in 0..300 {
        step(&mut s, &wild(t), &p); // move everyone away from the spawn
    }
    assert!(s.players[0].x != -p.player_spawn_x);
    score(&mut s, &p, 0);
    let ms = m(&s);
    assert_eq!((ms.phase, ms.last_goal, ms.phase_ticks), (MatchPhase::GoalPause, Some(0), 360));
    let (puck, players) = (s.puck, s.players.clone());
    assert!(puck.in_net == Some(0));
    for t in 0..359 {
        step(&mut s, &wild(t), &p);
        assert_eq!((s.puck, &s.players), (puck, &players), "frozen during the goal pause");
    }
    step(&mut s, &[], &p);
    let ms = m(&s);
    assert_eq!((ms.phase, ms.phase_ticks), (MatchPhase::Countdown, 360));
    assert_eq!(s.puck, sim::Puck::default(), "puck at center, at rest");
    for pl in &s.players {
        let (x, facing) = if pl.team == 0 { (-p.player_spawn_x, 0.0) } else { (p.player_spawn_x, std::f64::consts::PI) };
        assert_eq!((pl.x, pl.y, pl.vx, pl.vy, pl.facing, pl.stick_angle, pl.stick_omega), (x, 0.0, 0.0, 0.0, facing, facing, 0.0));
        assert_eq!((pl.dash_timer, pl.dash_cooldown, pl.dash_prev), (0.0, 0.0, false));
    }
    assert_eq!(s.score, vec![1, 0]);
}

#[test]
fn inputs_are_ignored_outside_playing() {
    let p = Params::default();
    let mut a = State::new_match(&p, 2);
    let mut b = a.clone();
    // Countdown: idle vs wild inputs give byte-identical states.
    let mut t = 0;
    while m(&a).phase == MatchPhase::Countdown {
        step(&mut a, &[], &p);
        step(&mut b, &wild(t), &p);
        t += 1;
    }
    assert_eq!(t, 360);
    assert_eq!(a.to_bytes(), b.to_bytes(), "countdown ignores inputs");
    // Playing: inputs act.
    let mut c = a.clone();
    step(&mut a, &[], &p);
    step(&mut c, &wild(1), &p);
    assert_ne!(a.to_bytes(), c.to_bytes(), "inputs act while playing");
    // Goal pause (+ the countdown after it): ignored again.
    score(&mut a, &p, 1);
    let mut b = a.clone();
    let mut t = 0;
    while m(&a).phase != MatchPhase::Playing {
        step(&mut a, &[], &p);
        step(&mut b, &wild(t), &p);
        t += 1;
    }
    assert_eq!(t, 720);
    assert_eq!(a.to_bytes(), b.to_bytes(), "goal pause and countdown ignore inputs");
}

fn scripted_match(steps: u64) -> State {
    let p = Params::default();
    let mut s = State::new_match(&p, 2);
    for t in 0..steps {
        step(&mut s, &wild(t / 3), &p);
        // Every 2000 steps, put a puck on a net (alternating) so the run passes through goals and pauses.
        if t % 2000 == 1000 && m(&s).phase == MatchPhase::Playing {
            let sign = if (t / 2000) % 3 == 0 { -1.0 } else { 1.0 };
            s.puck = sim::Puck { x: sign * (p.net_x - 1.0), vx: sign * 9.0, ..Default::default() };
        }
    }
    s
}

#[test]
fn match_determinism_100k_steps_byte_identical() {
    let a = scripted_match(100_000);
    let b = scripted_match(100_000);
    assert_eq!(a.to_bytes(), b.to_bytes());
    let ms = m(&a);
    println!("match determinism: 100000 steps, hash run1={:016x} run2={:016x}, score {:?}, {:?}", a.hash(), b.hash(), a.score, ms);
    assert!(a.score.iter().sum::<u32>() >= 2, "the run scored goals");
}

/// Same inputs as sim_tests::crosscheck_inputs, in a match (see scripts/wasm_crosscheck.mjs).
#[test]
fn crosscheck_hash_native_match() {
    let p = Params::default();
    let mut s = State::new_match(&p, 2);
    for t in 0..100_000u64 {
        let inp: Vec<PlayerInput> = (0..2u64)
            .map(|k| PlayerInput {
                move_x: (((t / 60 + k) % 3) as f64) - 1.0,
                move_y: (((t / 90 + k) % 3) as f64) - 1.0,
                stick_target_angle: ((t + k * 100) % 628) as f64 * 0.01 - 3.14,
                dash: t % 240 == k * 7,
                lift: t % 500 < 20,
            })
            .collect();
        step(&mut s, &inp, &p);
    }
    println!("native match crosscheck hash after 100000 steps: {:016x} (score {:?}, {:?})", s.hash(), s.score, m(&s));
}
