use calib::{evaluate, fit, fit_with, parse_csv, replay, synthetic_scenarios, write_csv, Case, Objective};
use sim::Params;

/// The synthetic self-checks (and calib/data/*.csv) were generated with this pinned copy of the
/// original placeholder params (old 60 x 26 m rink), so measured-shape edits to params.json don't move them.
const PINNED: &str = include_str!("../data/params.synthetic.json");

fn pinned() -> Params {
    Params::from_json(PINNED).expect("pinned params")
}

const NAMES: [&str; 2] = ["ice_friction_decel", "board_restitution"];

#[test]
fn fitter_recovers_friction_and_restitution() {
    let truth = pinned();
    // "Measured" data = sim output with the true params, round-tripped through the CSV format.
    let cases: Vec<Case> = synthetic_scenarios(&truth)
        .into_iter()
        .map(|sc| {
            let csv = write_csv(&replay(&sc, &truth));
            Case { measured: parse_csv(&csv).unwrap(), scenario: sc }
        })
        .collect();

    let mut start = truth.clone();
    start.ice_friction_decel *= 1.4;
    start.board_restitution *= 0.7;

    let before = evaluate(&cases, &start);
    let r = fit(&cases, &start, &NAMES, 600).unwrap();
    let after = evaluate(&cases, &r.params);

    println!("fit: {} iterations, objective {:.3e}", r.iterations, r.objective);
    println!("{:<20} {:>12} {:>12} {:>12} {:>10}", "param", "true", "perturbed", "recovered", "rel_err");
    for n in NAMES {
        let (t, s, f) = (truth.get(n).unwrap(), start.get(n).unwrap(), r.params.get(n).unwrap());
        println!("{n:<20} {t:>12.6} {s:>12.6} {f:>12.6} {:>10.2e}", ((f - t) / t).abs());
    }
    for ((c, b), a) in cases.iter().zip(&before).zip(&after) {
        println!("{:<12} pos_rmse before {:.4e} after {:.4e}; speed_rmse before {:.4e} after {:.4e}", c.scenario.name, b.pos_rmse, a.pos_rmse, b.speed_rmse, a.speed_rmse);
    }

    for n in NAMES {
        let (t, f) = (truth.get(n).unwrap(), r.params.get(n).unwrap());
        assert!(((f - t) / t).abs() < 0.01, "{n}: recovered {f}, true {t}");
    }
    for e in &after {
        assert!(e.pos_rmse < 0.01, "pos_rmse {}", e.pos_rmse);
    }
}

#[test]
fn csv_roundtrip_is_exact() {
    let p = pinned();
    let sc = &synthetic_scenarios(&p)[1];
    let s = replay(sc, &p);
    assert_eq!(parse_csv(&write_csv(&s)).unwrap(), s);
}

#[test]
fn scenario_json_roundtrip_replays_identically() {
    let p = pinned();
    let mut sc = synthetic_scenarios(&p)[0].clone();
    sc.initial_state = sim::State::new_default(&p, 2);
    sc.initial_state.puck.vx = 3.0;
    sc.inputs.push(calib::InputEvent { step: 10, player: 0, input: sim::PlayerInput { move_x: 1.0, stick_target_angle: 0.7, ..Default::default() } });
    let back = calib::Scenario::from_json(&sc.to_json()).unwrap();
    assert_eq!(replay(&sc, &p), replay(&back, &p));
}

// ---------- noisy synthetic recovery (real capture faults) ----------

/// Deterministic RNG (xorshift64*) so the test is reproducible without a rand dependency.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Placeholder meter scale of the real clips (60 m / ~913.5 board px) and the noise/fault rates measured there.
const M_PER_PX: f64 = 0.06568;
const NOISE_PX: f64 = 0.3;
const DUP_RATE: f64 = 11.0 / 3970.0; // duplicated frames in the puck clip
const SKIP_RATE: f64 = 11.0 / 3970.0; // assumed equal to the duplicate rate
const JITTER_RATE: f64 = 0.25; // fraction of frames captured +-1/180 s off (game ~180 fps into a 60 fps recording)

/// "Measure" a replay like the real capture: 60 fps rows with nominal times, while the position
/// comes from a jittered game time; occasional duplicated (stale) and skipped frames; 0.3 px noise.
fn capture(sim: &[calib::Sample], dt: f64, rng: &mut Rng) -> Vec<calib::Sample> {
    let fps = 60.0;
    let t_end = sim.last().unwrap().t;
    let at = |t: f64| {
        let f = (t / dt).clamp(0.0, (sim.len() - 1) as f64);
        let i = (f.floor() as usize).min(sim.len() - 2);
        let a = f - i as f64;
        (sim[i].x + (sim[i + 1].x - sim[i].x) * a, sim[i].y + (sim[i + 1].y - sim[i].y) * a)
    };
    let sigma = NOISE_PX * M_PER_PX;
    let mut out: Vec<calib::Sample> = Vec::new();
    let mut j = 0;
    loop {
        let t = j as f64 / fps;
        if t > t_end {
            break;
        }
        let r = rng.uniform();
        let (x, y) = if r < DUP_RATE && !out.is_empty() {
            let p = out.last().unwrap();
            (p.x, p.y) // stale frame: same picture as the previous row (noise included)
        } else {
            let mut tg = t;
            if r < DUP_RATE + SKIP_RATE {
                tg += 1.0 / fps; // skipped frame: shows the next frame's state
            } else if rng.uniform() < JITTER_RATE {
                tg += if rng.uniform() < 0.5 { -1.0 } else { 1.0 } / 180.0;
            }
            let (x, y) = at(tg.clamp(0.0, t_end));
            (x + sigma * rng.gauss(), y + sigma * rng.gauss())
        };
        out.push(calib::Sample { t, x, y, z: 0.0, stick_angle: None });
        j += 1;
    }
    out
}

#[test]
fn noisy_recovery_path_objective() {
    let truth = pinned();
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    let cases: Vec<Case> = synthetic_scenarios(&truth)
        .into_iter()
        .map(|sc| {
            let sim = replay(&sc, &truth);
            Case { measured: capture(&sim, truth.timestep, &mut rng), scenario: sc }
        })
        .collect();
    // How jittery is the synthetic capture? Same statistic as measure/'s step_jitter (moving rows only).
    let mut dev = Vec::new();
    for c in &cases {
        let m = &c.measured;
        let st: Vec<f64> = m.windows(2).map(|w| ((w[1].x - w[0].x).powi(2) + (w[1].y - w[0].y).powi(2)).sqrt()).collect();
        for i in 4..st.len().saturating_sub(4) {
            let mut win = st[i - 4..i + 5].to_vec();
            win.sort_by(f64::total_cmp);
            let med = win[4];
            if med > 2.0 * M_PER_PX {
                dev.push((st[i] / med - 1.0).abs());
            }
        }
    }
    dev.sort_by(f64::total_cmp);
    println!(
        "synthetic capture: noise {NOISE_PX} px = {:.4} m, dup {:.2}%, skip {:.2}%, jitter {:.0}% of frames; step deviation median {:.1}%, p90 {:.1}% (real clips: 4-6%, 37-51%)",
        NOISE_PX * M_PER_PX,
        DUP_RATE * 100.0,
        SKIP_RATE * 100.0,
        JITTER_RATE * 100.0,
        dev[dev.len() / 2] * 100.0,
        dev[dev.len() * 9 / 10] * 100.0
    );

    let mut start = truth.clone();
    start.ice_friction_decel *= 1.4;
    start.board_restitution *= 0.7;
    let mut rows = Vec::new();
    for kind in [Objective::Time, Objective::Path] {
        let r = fit_with(&cases, &start, &NAMES, 800, kind).unwrap();
        println!("objective {kind:?}: {} iterations, objective {:.4e}", r.iterations, r.objective);
        println!("  {:<20} {:>10} {:>10} {:>10} {:>9}", "param", "true", "perturbed", "recovered", "rel_err");
        for n in NAMES {
            let (t, s, f) = (truth.get(n).unwrap(), start.get(n).unwrap(), r.params.get(n).unwrap());
            println!("  {n:<20} {t:>10.5} {s:>10.5} {f:>10.5} {:>8.2}%", 100.0 * ((f - t) / t).abs());
        }
        for (c, e) in cases.iter().zip(evaluate(&cases, &r.params)) {
            println!("  {:<12} pos_rmse {:.4} m, path_rmse {:.4} m, v2_rmse {:.4} m2/s2", c.scenario.name, e.pos_rmse, e.path_rmse, e.v2_rmse);
        }
        rows.push((kind, r));
    }
    let path = &rows[1].1.params;
    let tol = [("ice_friction_decel", 0.03), ("board_restitution", 0.05)];
    for (n, tol) in tol {
        let (t, f) = (truth.get(n).unwrap(), path.get(n).unwrap());
        assert!(((f - t) / t).abs() < tol, "path objective: {n} recovered {f}, true {t} (tolerance {:.0}%)", tol * 100.0);
    }
}
