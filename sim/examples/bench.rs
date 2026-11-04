//! Single-threaded throughput benchmark. Run: cargo run --release -p sim --example bench [steps]
//! (Wall-clock reads live here, outside the sim library.)

use sim::{step, Params, PlayerInput, State};
use std::time::Instant;

fn main() {
    let steps: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2_000_000);
    let p = Params::default();
    for n_players in [2usize, 6] {
        let mut s = State::new_default(&p, n_players);
        s.puck.vx = 12.0;
        s.puck.vy = 7.0;
        let mut inputs = vec![PlayerInput::default(); n_players];
        let t0 = Instant::now();
        for i in 0..steps {
            for (k, inp) in inputs.iter_mut().enumerate() {
                let ph = ((i / 60) + k as u64) % 4;
                inp.move_x = [1.0, 0.0, -1.0, 0.0][ph as usize];
                inp.move_y = [0.0, 1.0, 0.0, -1.0][ph as usize];
                inp.stick_target_angle = (i % 360) as f64 * 0.01745 - 3.14;
                inp.dash = i % 240 == 0;
                inp.lift = i % 500 < 20;
            }
            step(&mut s, &inputs, &p);
        }
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "{n_players} players: {steps} steps in {secs:.3} s = {:.0} steps/s ({:.0}x realtime at 120 Hz), hash {:016x}",
            steps as f64 / secs,
            steps as f64 / secs / 120.0,
            s.hash()
        );
    }
}
