//! Batched env throughput, including observations and rewards:
//! `cargo run --release -p sim-ffi --example bench [seconds_per_case]`.
//! Opponent: the scripted chaser (its decisions are part of the cost). Learner actions: a fixed
//! pseudo-random pattern. Also runs a determinism check (same seed twice → same hash and obs).
use bots::{ACT_DIM, OBS_DIM};
use sim_ffi::env::INFO_DIM;
use sim_ffi::new_batch;
use std::time::Instant;

/// Pseudo-random actions in the v2 format (move index 0..8, stick angle, dash, lift).
fn fill(a: &mut [f32], t: u64) {
    for (i, v) in a.iter_mut().enumerate() {
        let x = (((i as u64 * 7919) ^ (t * 104729)) % 2001) as f32 / 1000.0; // 0..2
        *v = match i % ACT_DIM { 0 => (x * 4.5).floor(), 1 => (x - 1.0) * 3.14, _ => (x > 1.8) as u8 as f32 };
    }
}

fn main() {
    let secs: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("cores available: {threads}; action_repeat 4 (1 decision = 4 sim steps), opponent chaser");
    println!("| envs | threads | decisions/s | sim steps/s | sim steps/s per env |");
    println!("|---|---|---|---|---|");
    let mode = std::env::args().nth(2).unwrap_or_else(|| "first_goal".into());
    println!("mode {mode}");
    for &(n, th) in &[(1usize, 1usize), (8, 1), (8, 0), (64, 1), (64, 0), (256, 1), (256, 0), (1024, 0)] {
        let cfg = format!(r#"{{"threads": {th}, "mode": "{mode}"}}"#);
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
                fill(&mut a, t);
                b.step(&a, None, &mut obs, Some(&mut opp), None, &mut r, &mut d, &mut info);
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
            fill(&mut a, t);
            b.step(&a, None, &mut obs, None, None, &mut r, &mut d, &mut info);
            eps += d.iter().map(|&x| x as u64).sum::<u64>();
        }
        let oh = obs.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3));
        (b.hash(), oh, eps)
    };
    let a = go(&format!(r#"{{"threads": 1, "mode": "{mode}"}}"#));
    let b = go(&format!(r#"{{"threads": 0, "mode": "{mode}"}}"#));
    println!("determinism (64 envs, seed 2024, 2000 decisions, {} episodes finished): threads=1 batch hash {:016x} obs hash {:016x}; rayon batch hash {:016x} obs hash {:016x} -> {}",
        a.2, a.0, a.1, b.0, b.1, if a == b { "IDENTICAL" } else { "MISMATCH" });
}
