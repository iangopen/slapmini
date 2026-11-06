//! Round robin of the scripted bots: `cargo run --release -p bots --example tournament [matches] [seconds]`.
//! Every match plays `seconds` of game time. Each faceoff (the opening one and the one after every
//! goal) jitters the puck by up to ±1.5 m and the players by up to ±1 m with a small seeded
//! generator (SplitMix64, seed = match index), so matches differ; sides are swapped every other match.

use bots::game::{faceoff, play_with_faceoffs};
use bots::{Kind, Scripted};
use sim::{Params, State};

/// SplitMix64 (an LCG with consecutive seeds gave parity-correlated jitter).
struct Lcg(u64);
impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn jittered(p: &Params, g: &mut Lcg) -> State {
    let mut s = faceoff(p);
    s.puck.x = 1.5 * g.uniform();
    s.puck.y = 1.5 * g.uniform();
    for pl in s.players.iter_mut() {
        pl.x += g.uniform();
        pl.y += g.uniform();
    }
    s
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let matches: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let seconds: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60.0);
    // Optional third argument: a params file (for what-if comparisons).
    let p = args.get(3).map_or_else(Params::default, |f| Params::from_json(&std::fs::read_to_string(f).expect("params file")).expect("params"));
    let steps = (seconds / p.timestep).round() as u64;
    let kinds = [Kind::Chaser, Kind::Defender, Kind::Cautious, Kind::Idle];
    println!("{matches} matches per pairing, {seconds} s each, jittered faceoffs, sides swapped every other match");
    println!("| A | B | A goals/match | B goals/match | A wins | draws | B wins |");
    println!("|---|---|---|---|---|---|---|");
    for (ia, &a) in kinds.iter().enumerate() {
        for &b in kinds.iter().skip(ia) {
            let (mut ga, mut gb, mut wa, mut wb, mut dr) = (0u64, 0u64, 0, 0, 0);
            for m in 0..matches {
                let mut g = Lcg(0x5eed_0000 + m as u64);
                let mut s = jittered(&p, &mut g);
                let (mut ba, mut bb) = (Scripted::new(a), Scripted::new(b));
                let mut next = |_k: u32| jittered(&p, &mut g);
                let out = if m % 2 == 1 {
                    let o = play_with_faceoffs(&mut s, &mut [&mut bb, &mut ba], &p, steps, false, &mut next);
                    [o.goals[1], o.goals[0]]
                } else {
                    play_with_faceoffs(&mut s, &mut [&mut ba, &mut bb], &p, steps, false, &mut next).goals
                };
                ga += out[0] as u64;
                gb += out[1] as u64;
                if out[0] > out[1] { wa += 1 } else if out[1] > out[0] { wb += 1 } else { dr += 1 }
            }
            let n = matches as f64;
            println!("| {} | {} | {:.2} | {:.2} | {} | {} | {} |", a.name(), b.name(), ga as f64 / n, gb as f64 / n, wa, dr, wb);
        }
    }
}
