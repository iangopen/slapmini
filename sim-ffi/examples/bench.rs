//! Batched env throughput, including observations and rewards:
//! `cargo run --release -p sim-ffi --example bench [seconds_per_case]`.
//! Opponent: the scripted chaser (its decisions are part of the cost). Learner actions: a fixed
//! pseudo-random pattern. Also runs a determinism check (same seed twice → same hash and obs).
use bots::{ACT_DIM, OBS_DIM};
use sim_ffi::env::INFO_DIM;
use sim_ffi::new_batch;
use std::time::Instant;

fn main() {
    let secs: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("cores available: {threads}; action_repeat 4 (1 decision = 4 sim steps), opponent chaser");
    println!("| envs | threads | decisions/s | sim steps/s | sim steps/s per env |");
    println!("|---|---|---|---|---|");
    for &(n, th) in &[(1usize, 1usize), (8, 1), (8, 0), (64, 1), (64, 0)] {
        let cfg = format!(r#"{{"threads": {th}}}"#);
        let mut b = new_batch("", &cfg, n as u32 as usize, 1).unwrap();
        let (mut obs, mut opp, mut r, mut d, mut info) =
            (vec![0f32; n * OBS_DIM], vec![0f32; n * OBS_DIM], vec![0f32; n], vec![0u8; n], vec![0f32; n * INFO_DIM]);
        b.reset_all(&mut obs, Some(&mut opp));
        let mut a = vec![0f32; n * ACT_DIM];
        let start = Instant::now();
        let mut decisions = 0u64;
        let mut t = 0u64;
        while start.elapsed().as_secs_f64() < secs {
            for _ in 0..50 {
                for (i, v) in a.iter_mut().enumerate() {
                    *v = (((i as u64 * 7919) ^ (t * 104729)) % 2001) as f32 / 1000.0 - 1.0;
                }
                b.step(&a, None, &mut obs, Some(&mut opp), &mut r, &mut d, &mut info);
                t += 1;
                decisions += n as u64;
            }
        }
        let el = start.elapsed().as_secs_f64();
        let dps = decisions as f64 / el;
        let label = if th == 0 { format!("{threads} (rayon)") } else { th.to_string() };
        println!("| {n} | {label} | {:.0} | {:.0} | {:.0} |", dps, dps * 4.0, dps * 4.0 / n as f64);
    }

    // Determinism: same seed, two batches, 2000 decisions each, different thread counts.
    let go = |cfg: &str| {
        let n = 64;
        let mut b = new_batch("", cfg, n, 2024).unwrap();
        let (mut obs, mut r, mut d, mut info) = (vec![0f32; n * OBS_DIM], vec![0f32; n], vec![0u8; n], vec![0f32; n * INFO_DIM]);
        b.reset_all(&mut obs, None);
        let mut a = vec![0f32; n * ACT_DIM];
        let mut eps = 0u64;
        for t in 0..2000u64 {
            for (i, v) in a.iter_mut().enumerate() {
                *v = (((i as u64 * 7919) ^ (t * 104729)) % 2001) as f32 / 1000.0 - 1.0;
            }
            b.step(&a, None, &mut obs, None, &mut r, &mut d, &mut info);
            eps += d.iter().map(|&x| x as u64).sum::<u64>();
        }
        let oh = obs.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3));
        (b.hash(), oh, eps)
    };
    let a = go(r#"{"threads": 1}"#);
    let b = go(r#"{"threads": 0}"#);
    println!("determinism (64 envs, seed 2024, 2000 decisions, {} episodes finished): threads=1 batch hash {:016x} obs hash {:016x}; rayon batch hash {:016x} obs hash {:016x} -> {}",
        a.2, a.0, a.1, b.0, b.1, if a == b { "IDENTICAL" } else { "MISMATCH" });
}
