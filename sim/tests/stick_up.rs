//! w4: the raised stick (PlayerInput.stick_up, the game's left mouse button). A raised stick contacts the puck only
//! inside [stick_up_z_min, stick_up_z_max] (placeholders 0.35 / 1.5 m): it passes over a puck on the ice and can hit
//! a puck in the air. The flag is part of the state (Player.stick_up) and of to_bytes only when some stick is up.

use sim::{step, Params, PlayerInput, State};

/// One player at the origin, at rest, stick held along +x; puck moving +y across the blade at x0, height z0.
fn crossing(stick_up: bool, z0: f64, player_far: bool) -> (State, Vec<(f64, f64, f64, f64, f64)>) {
    let p = Params::default();
    let mut s = State::new_default(&p, 1);
    let pl = &mut s.players[0];
    *pl = sim::Player { team: 0, x: if player_far { -25.0 } else { 0.0 }, y: 0.0, facing: 0.0, stick_angle: 0.0, ..Default::default() };
    s.puck = sim::Puck { x: 0.9, y: -1.0, z: z0, vx: 0.0, vy: 10.0, vz: 0.0, in_net: None };
    let inp = [PlayerInput { stick_target_angle: 0.0, stick_up, ..Default::default() }];
    let mut tr = Vec::new();
    for _ in 0..30 {
        step(&mut s, &inp, &p);
        tr.push((s.puck.x, s.puck.y, s.puck.z, s.puck.vx, s.puck.vy));
    }
    (s, tr)
}

#[test]
fn puck_on_the_ice_passes_under_a_raised_stick() {
    let (_, down) = crossing(false, 0.0, false);
    let (_, up) = crossing(true, 0.0, false);
    let (_, free) = crossing(false, 0.0, true); // no stick in the way at all
    let d = down.last().unwrap();
    let u = up.last().unwrap();
    let f = free.last().unwrap();
    println!("rolling puck (z 0, 10 m/s across the blade at 0.9 m), after 0.25 s: stick DOWN v = ({:.3}, {:.3}); stick UP v = ({:.6}, {:.6}); no stick v = ({:.6}, {:.6})", d.3, d.4, u.3, u.4, f.3, f.4);
    assert!((d.3 - f.3).abs() > 1.0 || (d.4 - f.4).abs() > 1.0, "a lowered stick must deflect the puck: {d:?} vs {f:?}");
    assert_eq!(up, free, "a raised stick must not touch a puck on the ice (identical trajectory to no stick)");
}

#[test]
fn a_raised_stick_hits_a_puck_in_the_air() {
    let (_, down) = crossing(false, 0.8, false);
    let (_, up) = crossing(true, 0.8, false);
    let (_, free) = crossing(false, 0.8, true);
    // The puck falls from 0.8 m; it reaches the blade after ~0.1 s at z ≈ 0.75 m.
    let at_blade = free.iter().find(|q| q.1 > -0.05).unwrap();
    let u = up.last().unwrap();
    let f = free.last().unwrap();
    println!("airborne puck (z 0.8 m, 10 m/s): z at the blade {:.3} m; stick UP v = ({:.3}, {:.3}); stick DOWN passes over (same as no stick: {}); no stick v = ({:.3}, {:.3})", at_blade.2, u.3, u.4, down == free, f.3, f.4);
    assert!(at_blade.2 > 0.35 && at_blade.2 < 1.5, "the test puck must be inside the raised range at the blade");
    assert_eq!(down, free, "a lowered stick (0..0.25 m) must not touch a puck at ~0.75 m");
    assert!((u.3 - f.3).abs() > 1.0 || (u.4 - f.4).abs() > 1.0, "a raised stick must hit a puck in the air: {u:?} vs {f:?}");
}

#[test]
fn stick_up_follows_the_input_and_is_encoded_only_when_set() {
    let p = Params::default();
    let mut s = State::new_default(&p, 2);
    let base = s.to_bytes();
    step(&mut s, &[PlayerInput::default(), PlayerInput::default()], &p);
    let down = s.clone();
    assert!(!s.players[0].stick_up && !s.players[1].stick_up);
    let mut up = down.clone();
    up.players[1].stick_up = true;
    let (bd, bu) = (down.to_bytes(), up.to_bytes());
    assert_eq!(bd.len(), base.len(), "no stick up: the encoding has the pre-w4 length");
    assert_eq!(&bu[..bd.len()], &bd[..], "the raised-stick block is appended after the old encoding");
    assert_eq!(&bu[bd.len()..], &[0x5C, 0, 1]);
    // From the input, every step (instant raise / lower).
    step(&mut s, &[PlayerInput { stick_up: true, ..Default::default() }, PlayerInput::default()], &p);
    assert!(s.players[0].stick_up && !s.players[1].stick_up);
    step(&mut s, &[PlayerInput::default(), PlayerInput::default()], &p);
    assert!(!s.players[0].stick_up);
    // JSON without the field: false (old scenarios and recordings).
    let inp: PlayerInput = serde_json::from_str(r#"{"move_x":1,"move_y":0,"stick_target_angle":0.5,"dash":false,"lift":false}"#).unwrap();
    assert!(!inp.stick_up);
    let pl: sim::Player = serde_json::from_str(r#"{"team":0,"x":0,"y":0,"vx":0,"vy":0,"facing":0,"stick_angle":0,"stick_omega":0}"#).unwrap();
    assert!(!pl.stick_up);
}

/// s10_stress's chase-and-spin inputs, plus a raised stick that toggles pseudo-randomly in 1/3 s blocks (u32 integer
/// hash, reproduced exactly by scripts/wasm_crosscheck.mjs with Math.imul).
pub fn stick_up_inputs(s: &State, t: u64) -> Vec<PlayerInput> {
    s.players
        .iter()
        .enumerate()
        .map(|(k, pl)| {
            let (dx, dy) = (s.puck.x - pl.x, s.puck.y - pl.y);
            let d = (dx * dx + dy * dy).sqrt().max(1e-9);
            let phase = (t + k as u64 * 97) % 600;
            let coasting = phase >= 480;
            let (mx, my) = if coasting { (0.0, 0.0) } else { (dx / d, dy / d) };
            let sense = if (t / 300 + k as u64) % 2 == 0 { 1.0 } else { -1.0 };
            let stick = if d < 2.0 { pl.stick_angle + sense * 2.5 } else { pl.stick_angle };
            let x = ((t / 40) * 2 + k as u64) as u32;
            let up = (x.wrapping_mul(2_654_435_761) >> 16) % 3 == 0;
            PlayerInput { move_x: mx, move_y: my, stick_target_angle: stick, dash: phase % 150 < 3 && !coasting, lift: t % 700 < 15, stick_up: up }
        })
        .collect()
}

fn run(steps: u64) -> (State, u64, u64) {
    let p = Params::default();
    let mut s = State::new_default(&p, 2);
    let (mut hits, mut up_steps) = (0, 0);
    for t in 0..steps {
        let v0 = s.puck.vx.hypot(s.puck.vy);
        let inp = stick_up_inputs(&s, t);
        up_steps += inp.iter().filter(|i| i.stick_up).count() as u64;
        step(&mut s, &inp, &p);
        if s.puck.vx.hypot(s.puck.vy) > v0 + 3.0 {
            hits += 1;
        }
    }
    (s, hits, up_steps)
}

#[test]
fn stick_up_determinism_100k_steps() {
    let (a, hits, up) = run(100_000);
    let (b, _, _) = run(100_000);
    assert_eq!(a.to_bytes(), b.to_bytes());
    println!("stick_up determinism: 100000 steps, hash {:016x}, shots {hits}, player-steps with the stick up {up} of 200000, score {:?}", a.hash(), a.score);
    assert!(hits > 20 && up > 40_000 && up < 100_000, "the run must contain shots and raised sticks: {hits} {up}");
}
