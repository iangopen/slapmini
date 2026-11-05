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
                    let bits: Vec<u32> = o.iter().map(|x| x.to_bits()).collect();
                    println!("{}", serde_json::json!({"state": js, "player": player, "style": style, "bits": bits}));
                    n += 1;
                }
            }
        }
    }
    eprintln!("obs_fixture: {n} observations");
}
