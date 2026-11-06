//! Native observation fixture for the wasm cross-check (scripts/obs_crosscheck.mjs):
//! `cargo run --release -p sim-wasm --example obs_fixture > obs_fixture.jsonl`.
//! States come from scripted games (free play and match mode, with goals, resets, dashes and
//! airborne pucks); one JSON line per (state, player, style): the State as a JSON string, the
//! player, the style vector and the native observation as f32 bit patterns.
use bots::{observe, Bot, Kind, Scripted, OBS_DIM};
use sim::{Params, PlayerInput, State};

fn main() {
    let p = Params::default();
    let styles: [&[f32]; 3] = [&[], &[0.5, -1.0, 0.25, 1.0], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.75]];
    let mut n = 0;
    let mut up = [0usize; 4];
    for (g, (a, b, matched)) in [(Kind::Chaser, Kind::Defender, true), (Kind::Chaser, Kind::Chaser, false), (Kind::Cautious, Kind::Chaser, true)].into_iter().enumerate() {
        let mut s = if matched { State::new_match(&p, 2) } else { State::new_default(&p, 2) };
        let mut bots = [Scripted::new(a), Scripted::new(b)];
        for t in 0..6000u64 {
            let mut inputs = [PlayerInput::default(); 2];
            for (i, bot) in bots.iter_mut().enumerate() {
                inputs[i] = bot.act(&s, i, &p).to_input(s.players[i].team);
                // Sprinkle dashes and lifts so those fields vary.
                inputs[i].dash = (t + 37 * i as u64) % 400 == 0;
                inputs[i].lift = (t / 200 + i as u64) % 7 == 0;
                // OBS v3: raised sticks in 60-step blocks, so neither / one / both players raised all occur.
                inputs[i].stick_up = ((t / 60) * 2 + i as u64).wrapping_mul(2654435761) % 5 < 2;
            }
            let before: u32 = s.score.iter().sum();
            sim::step(&mut s, &inputs, &p);
            if s.score.iter().sum::<u32>() != before {
                for bot in bots.iter_mut() {
                    bot.reset();
                }
                if !matched {
                    s = State::new_default(&p, 2);
                }
            }
            if t % 53 == 0 {
                let js = serde_json::to_string(&s).unwrap();
                for player in 0..2 {
                    let style = styles[(t as usize / 53 + player + g) % styles.len()];
                    let mut o = [0f32; OBS_DIM];
                    observe(&s, player, style, &p, &mut o);
                    up[(o[bots::STICK_UP_IDX] as usize) + 2 * (o[bots::STICK_UP_IDX + 1] as usize)] += 1;
                    let bits: Vec<u32> = o.iter().map(|x| x.to_bits()).collect();
                    // ACT v4: a varied action and what the native rotation makes of it.
                    let k = t / 53 + player as u64;
                    let act = bots::Action { move_dir: (k % 9) as u8, stick_angle: (k % 17) as f64 * 0.37 - 3.0, dash: k % 2 == 0, lift: k % 3 == 0, stick_up: k % 5 < 2, brake: k % 4 == 1 };
                    let input = serde_json::to_string(&act.to_input(s.players[player].team)).unwrap();
                    println!("{}", serde_json::json!({"state": js, "player": player, "style": style, "bits": bits, "act": [act.move_dir, act.stick_angle, act.dash, act.lift, act.stick_up, act.brake], "input": input}));
                    n += 1;
                }
            }
        }
    }
    eprintln!("obs_fixture: {n} observations (stick_up: none {}, own only {}, opponent only {}, both {})", up[0], up[1], up[2], up[3]);
}
