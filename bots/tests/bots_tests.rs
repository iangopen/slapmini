use bots::game::{faceoff, play};
use bots::{observe, rotate_swap, Action, Bot, Kind, Scripted, OBS_DIM};
use sim::{Params, PlayerInput, State};

/// Small deterministic generator for test states (no RNG in the crates themselves).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn random_state(p: &Params, g: &mut Lcg) -> State {
    let mut s = faceoff(p);
    s.puck.x = g.next() * 25.0;
    s.puck.y = g.next() * 15.0;
    s.puck.z = (g.next() + 1.0) * 0.5;
    s.puck.vx = g.next() * 20.0;
    s.puck.vy = g.next() * 20.0;
    s.puck.vz = g.next() * 3.0;
    for pl in s.players.iter_mut() {
        pl.x = g.next() * 25.0;
        pl.y = g.next() * 15.0;
        pl.vx = g.next() * 8.0;
        pl.vy = g.next() * 8.0;
        pl.facing = g.next() * 3.14159;
        pl.stick_angle = g.next() * 3.14159;
        pl.stick_omega = g.next() * 14.0;
        pl.dash_timer = (g.next() + 1.0) * 0.1;
        pl.dash_cooldown = (g.next() + 1.0) * 0.5;
    }
    s.score = vec![2, 5];
    s
}

#[test]
fn observation_of_team1_mirrors_team0_under_rotation() {
    let p = Params::default();
    let mut g = Lcg(7);
    let style = [0.5f32, -0.25, 0.0, 1.0];
    for _ in 0..200 {
        let s = random_state(&p, &mut g);
        let r = rotate_swap(&s);
        for i in 0..2 {
            let (mut a, mut b) = ([0f32; OBS_DIM], [0f32; OBS_DIM]);
            observe(&s, i, &style, &p, &mut a);
            observe(&r, i, &style, &p, &mut b);
            assert_ne!(s.players[i].team, r.players[i].team);
            for k in 0..OBS_DIM {
                assert!((a[k] - b[k]).abs() < 1e-6, "field {k}: {} vs {}", a[k], b[k]);
            }
        }
        // The two players' observations are genuinely different views (not a trivial pass).
        let (mut o0, mut o1) = ([0f32; OBS_DIM], [0f32; OBS_DIM]);
        observe(&s, 0, &[], &p, &mut o0);
        observe(&s, 1, &[], &p, &mut o1);
        assert_ne!(o0, o1);
        // Score difference (own − opponent) / 5; free play: no clock, no overtime.
        assert_eq!(o0[28] * 5.0, -3.0);
        assert_eq!(o1[28] * 5.0, 3.0);
        assert_eq!((o0[29], o0[30]), (0.0, 0.0));
    }
}

#[test]
fn observation_carries_the_match_clock() {
    let p = Params::default();
    let mut s = State::new_match(&p, 2);
    let mut o = [0f32; OBS_DIM];
    observe(&s, 0, &[], &p, &mut o);
    assert!((o[30] as f64 - p.match_length_s / 300.0).abs() < 1e-6, "time left {}", o[30]);
    assert_eq!(o[29], 0.0);
    let m = s.match_state.as_mut().unwrap();
    m.clock_ticks = 0;
    m.overtime = true;
    observe(&s, 1, &[], &p, &mut o);
    assert_eq!((o[29], o[30]), (1.0, 0.0));
}

#[test]
fn action_maps_with_inverse_rotation() {
    let a = Action { move_dir: 8, stick_angle: 0.4, dash: true, lift: false, stick_up: false }; // +x −y
    let i0 = a.to_input(0);
    let i1 = a.to_input(1);
    assert_eq!((i0.move_x, i0.move_y, i0.stick_target_angle), (1.0, -1.0, 0.4));
    assert_eq!((i1.move_x, i1.move_y), (-1.0, 1.0));
    assert!((i1.stick_target_angle - (0.4 - std::f64::consts::PI)).abs() < 1e-12);
    assert!(i1.dash && !i1.lift);
    // Round trip through the flat format.
    let mut buf = [0f32; bots::ACT_DIM];
    a.write(&mut buf);
    let b = Action::from_slice(&buf);
    assert!(b.move_dir == 8 && b.dash && !b.lift);
}

/// WASD diagonals: the sim normalizes (±1, ±1), so a diagonal accelerates as fast as a straight key
/// and reaches the same top speed.
#[test]
fn diagonal_move_reaches_the_straight_top_speed() {
    let p = Params::default();
    let run = |dir: u8| {
        let mut s = State::new_default(&p, 1);
        s.players[0].x = -20.0;
        s.players[0].y = -10.0;
        let a = Action { move_dir: dir, ..Default::default() }.to_input(0);
        let mut at_half = 0.0;
        for k in 0..120 {
            sim::step(&mut s, &[a], &p);
            if k == 29 {
                at_half = s.players[0].vx.hypot(s.players[0].vy);
            }
        }
        (at_half, s.players[0].vx.hypot(s.players[0].vy))
    };
    let (straight_mid, straight) = run(1);
    let (diag_mid, diag) = run(2);
    eprintln!("after 0.25 s / 1 s: straight {straight_mid:.4} / {straight:.4} m/s, diagonal {diag_mid:.4} / {diag:.4} m/s (cap {})", p.player_max_speed);
    assert!((straight - p.player_max_speed).abs() < 1e-9 && (diag - straight).abs() < 1e-9);
    assert!((diag_mid - straight_mid).abs() < 1e-9);
}

/// The physics is symmetric under the rotation: a game played from a state and the same game
/// played from the rotated state (with each player's canonical action) stay rotations of each other.
#[test]
fn rotated_games_stay_rotated() {
    let p = Params::default();
    let mut g = Lcg(11);
    let mut s = random_state(&p, &mut g);
    s.puck.z = 0.0;
    s.puck.vz = 0.0;
    let mut r = rotate_swap(&s);
    let (mut a0, mut a1) = (Scripted::new(Kind::Chaser), Scripted::new(Kind::Chaser));
    let (mut b0, mut b1) = (Scripted::new(Kind::Chaser), Scripted::new(Kind::Chaser));
    for _ in 0..240 {
        let ins = [a0.act(&s, 0, &p).to_input(s.players[0].team), a1.act(&s, 1, &p).to_input(s.players[1].team)];
        let inr = [b0.act(&r, 0, &p).to_input(r.players[0].team), b1.act(&r, 1, &p).to_input(r.players[1].team)];
        sim::step(&mut s, &ins, &p);
        sim::step(&mut r, &inr, &p);
    }
    let back = rotate_swap(&r);
    let d = |a: f64, b: f64| (a - b).abs();
    assert!(d(s.puck.x, back.puck.x) < 1e-6 && d(s.puck.y, back.puck.y) < 1e-6, "{:?} vs {:?}", s.puck, back.puck);
    for (x, y) in s.players.iter().zip(&back.players) {
        assert!(d(x.x, y.x) < 1e-6 && d(x.y, y.y) < 1e-6);
    }
    assert_eq!(s.score, back.score);
}

fn steps(p: &Params, secs: f64) -> u64 {
    (secs / p.timestep) as u64
}

#[test]
fn chaser_scores_against_idle_from_either_side() {
    let p = Params::default();
    for chaser_slot in 0..2 {
        let mut s = faceoff(&p);
        let (mut c, mut i) = (Scripted::new(Kind::Chaser), Scripted::new(Kind::Idle));
        let out = if chaser_slot == 0 {
            play(&mut s, &mut [&mut c, &mut i], &p, steps(&p, 20.0), true)
        } else {
            play(&mut s, &mut [&mut i, &mut c], &p, steps(&p, 20.0), true)
        };
        let team = s.players[chaser_slot].team as usize;
        assert_eq!(out.goals[team], 1, "chaser in slot {chaser_slot}: {out:?}");
        assert_eq!(out.goals[1 - team], 0);
    }
}

/// A shot from the slot toward the empty part of the net: the idle player concedes, the defender
/// steps across and stops it.
#[test]
fn defender_blocks_a_shot_that_beats_idle() {
    let p = Params::default();
    let shoot = |kind: Kind| {
        let mut s = faceoff(&p);
        s.players[0].x = -p.net_x + 3.0;
        s.players[0].y = 4.0;
        s.players[1].x = 20.0;
        s.puck.x = -8.0;
        s.puck.y = 1.0;
        s.puck.vx = -13.0;
        s.puck.vy = -0.6;
        let (mut d, mut i) = (Scripted::new(kind), Scripted::new(Kind::Idle));
        play(&mut s, &mut [&mut d, &mut i], &p, steps(&p, 4.0), true)
    };
    let idle = shoot(Kind::Idle);
    assert_eq!(idle.goals, [0, 1], "the scenario must score against an idle player");
    let def = shoot(Kind::Defender);
    assert_eq!(def.goals[1], 0, "defender conceded: {def:?}");
}

/// Over many chaser attacks, the defender concedes clearly fewer goals than an idle player.
#[test]
fn defender_concedes_less_than_idle_against_chaser() {
    let p = Params::default();
    let conceded = |kind: Kind| {
        let mut total = 0;
        for m in 0..6 {
            let mut s = faceoff(&p);
            s.puck.y = (m as f64 - 2.5) * 1.5;
            let (mut c, mut d) = (Scripted::new(Kind::Chaser), Scripted::new(kind));
            total += play(&mut s, &mut [&mut c, &mut d], &p, steps(&p, 30.0), false).goals[0];
        }
        total
    };
    let (idle, def) = (conceded(Kind::Idle), conceded(Kind::Defender));
    assert!(def * 3 <= idle * 2, "defender conceded {def}, idle {idle}");
}

/// The cautious bot ignores a far puck, and scores once the puck comes to it.
#[test]
fn cautious_waits_then_scores() {
    let p = Params::default();
    // Far puck at rest: it stays home.
    let mut s = faceoff(&p);
    s.puck.x = 15.0;
    let (mut c, mut i) = (Scripted::new(Kind::Cautious), Scripted::new(Kind::Idle));
    let out = play(&mut s, &mut [&mut c, &mut i], &p, steps(&p, 5.0), true);
    assert_eq!(out.goals, [0, 0]);
    assert!(s.players[0].x < -10.0, "cautious left home: x = {}", s.players[0].x);
    // Puck sliding into its zone: it engages and scores.
    let mut s = faceoff(&p);
    s.puck.x = s.players[0].x + 20.0; // relative to the spawn (rink_length / 4), not a fixed x
    s.puck.vx = -7.0;
    s.puck.vy = 1.0;
    let (mut c, mut i) = (Scripted::new(Kind::Cautious), Scripted::new(Kind::Idle));
    let out = play(&mut s, &mut [&mut c, &mut i], &p, steps(&p, 20.0), true);
    assert_eq!(out.goals, [1, 0], "{out:?}");
}

#[test]
fn scripted_bots_are_deterministic() {
    let p = Params::default();
    let run = || {
        let mut s = faceoff(&p);
        let (mut a, mut b) = (Scripted::new(Kind::Chaser), Scripted::new(Kind::Defender));
        play(&mut s, &mut [&mut a, &mut b], &p, steps(&p, 20.0), false);
        s.hash()
    };
    assert_eq!(run(), run());
    let _ = PlayerInput::default();
}

/// rotate_swap keeps match states consistent: team indices inside the match state swap with the teams.
#[test]
fn rotate_swap_swaps_match_teams() {
    let p = Params::default();
    let mut s = State::new_match(&p, 2);
    if let Some(m) = s.match_state.as_mut() {
        m.last_goal = Some(0);
        m.winner = Some(1);
    }
    let r = rotate_swap(&s);
    let m = r.match_state.as_ref().unwrap();
    assert_eq!((m.last_goal, m.winner), (Some(1), Some(0)));
    let back = rotate_swap(&r);
    assert_eq!(back.match_state, s.match_state);
    assert_eq!(back.score, s.score);
}
