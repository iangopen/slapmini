//! Calibration harness: scenario files, trajectory CSVs, replay, error metrics, and a Nelder-Mead fitter.

use serde::{Deserialize, Serialize};
use sim::{step, Params, PlayerInput, State};

// ---------- Scenario files ----------

/// An input change for one player, held from `step` until the next event for that player.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InputEvent {
    pub step: u64,
    pub player: usize,
    pub input: PlayerInput,
}

/// Initial state plus inputs over time. Stored as JSON (`*.scenario.json`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Scenario {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Number of steps to simulate.
    pub steps: u64,
    pub initial_state: State,
    #[serde(default)]
    pub inputs: Vec<InputEvent>,
}

impl Scenario {
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| format!("scenario json: {e}"))
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("scenario serializes")
    }
}

// ---------- Trajectories ----------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Player 0's stick angle, if recorded.
    pub stick_angle: Option<f64>,
}

/// CSV with header `t,x,y,z` and optional `stick_angle` column (puck position, player 0 stick).
pub fn parse_csv(text: &str) -> Result<Vec<Sample>, String> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#'));
    let header: Vec<&str> = lines.next().ok_or("csv: empty")?.split(',').map(str::trim).collect();
    let col = |n: &str| header.iter().position(|h| *h == n);
    let (it, ix, iy, iz) = (
        col("t").ok_or("csv: missing t")?,
        col("x").ok_or("csv: missing x")?,
        col("y").ok_or("csv: missing y")?,
        col("z").ok_or("csv: missing z")?,
    );
    let ia = col("stick_angle");
    let mut out = Vec::new();
    for (n, line) in lines.enumerate() {
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        let num = |i: usize| -> Result<f64, String> {
            f.get(i).ok_or(format!("csv row {}: short", n + 2))?.parse::<f64>().map_err(|e| format!("csv row {}: {e}", n + 2))
        };
        let stick_angle = match ia {
            Some(i) if f.get(i).is_some_and(|s| !s.is_empty()) => Some(num(i)?),
            _ => None,
        };
        out.push(Sample { t: num(it)?, x: num(ix)?, y: num(iy)?, z: num(iz)?, stick_angle });
    }
    Ok(out)
}

pub fn write_csv(samples: &[Sample]) -> String {
    let with_stick = samples.iter().any(|s| s.stick_angle.is_some());
    let mut s = String::from(if with_stick { "t,x,y,z,stick_angle\n" } else { "t,x,y,z\n" });
    for p in samples {
        // `{}` on f64 prints the shortest string that round-trips exactly.
        s.push_str(&format!("{},{},{},{}", p.t, p.x, p.y, p.z));
        if with_stick {
            s.push(',');
            if let Some(a) = p.stick_angle {
                s.push_str(&format!("{a}"));
            }
        }
        s.push('\n');
    }
    s
}

// ---------- Replay ----------

/// Run a scenario through sim, returning one sample per step including t = 0.
pub fn replay(sc: &Scenario, p: &Params) -> Vec<Sample> {
    let mut state = sc.initial_state.clone();
    let n = state.players.len();
    let mut held = vec![PlayerInput::default(); n];
    let mut events = sc.inputs.clone();
    events.sort_by_key(|e| e.step); // stable: same-step events keep file order
    let mut ev = events.iter().peekable();
    let sample = |s: &State, t: f64| Sample {
        t,
        x: s.puck.x,
        y: s.puck.y,
        z: s.puck.z,
        stick_angle: s.players.first().map(|pl| pl.stick_angle),
    };
    let mut out = Vec::with_capacity(sc.steps as usize + 1);
    out.push(sample(&state, 0.0));
    for k in 0..sc.steps {
        while let Some(e) = ev.next_if(|e| e.step <= k) {
            if e.player < n {
                held[e.player] = e.input;
            }
        }
        step(&mut state, &held, p);
        out.push(sample(&state, (k + 1) as f64 * p.timestep));
    }
    out
}

/// Linear interpolation of the replay at time t (clamped to its ends). Assumes uniform spacing `dt`.
fn sim_at(sim: &[Sample], dt: f64, t: f64) -> (f64, f64, f64, Option<f64>) {
    let f = (t / dt).max(0.0);
    let i = (f.floor() as usize).min(sim.len() - 1);
    let j = (i + 1).min(sim.len() - 1);
    let a = f - i as f64;
    let (s0, s1) = (&sim[i], &sim[j]);
    let lerp = |u: f64, v: f64| u + (v - u) * a.min(1.0);
    let ang = match (s0.stick_angle, s1.stick_angle) {
        (Some(u), Some(v)) => Some(u + wrap(v - u) * a.min(1.0)),
        _ => None,
    };
    (lerp(s0.x, s1.x), lerp(s0.y, s1.y), lerp(s0.z, s1.z), ang)
}

fn wrap(a: f64) -> f64 {
    let pi = std::f64::consts::PI;
    (a + pi).rem_euclid(2.0 * pi) - pi
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Errors {
    /// 3D puck position RMSE, m.
    pub pos_rmse: f64,
    /// Puck speed RMSE (finite differences between consecutive CSV rows, 3D), m/s.
    pub speed_rmse: f64,
    /// Player 0 stick angle RMSE, rad (only if the CSV has the column).
    pub stick_rmse: Option<f64>,
    pub samples: usize,
}

/// Compare a replay against measured samples.
pub fn compare(sim: &[Sample], dt: f64, meas: &[Sample]) -> Errors {
    let at: Vec<_> = meas.iter().map(|m| sim_at(sim, dt, m.t)).collect();
    let n = meas.len();
    let mut pos = 0.0;
    let mut stick = 0.0;
    let mut stick_n = 0usize;
    for (m, s) in meas.iter().zip(&at) {
        pos += (m.x - s.0).powi(2) + (m.y - s.1).powi(2) + (m.z - s.2).powi(2);
        if let (Some(a), Some(b)) = (m.stick_angle, s.3) {
            stick += wrap(a - b).powi(2);
            stick_n += 1;
        }
    }
    let mut spd = 0.0;
    let mut spd_n = 0usize;
    for k in 1..n {
        let ddt = meas[k].t - meas[k - 1].t;
        if ddt <= 0.0 {
            continue;
        }
        let ms = ((meas[k].x - meas[k - 1].x).powi(2) + (meas[k].y - meas[k - 1].y).powi(2) + (meas[k].z - meas[k - 1].z).powi(2)).sqrt() / ddt;
        let ss = ((at[k].0 - at[k - 1].0).powi(2) + (at[k].1 - at[k - 1].1).powi(2) + (at[k].2 - at[k - 1].2).powi(2)).sqrt() / ddt;
        spd += (ms - ss).powi(2);
        spd_n += 1;
    }
    Errors {
        pos_rmse: if n > 0 { (pos / n as f64).sqrt() } else { 0.0 },
        speed_rmse: if spd_n > 0 { (spd / spd_n as f64).sqrt() } else { 0.0 },
        stick_rmse: if stick_n > 0 { Some((stick / stick_n as f64).sqrt()) } else { None },
        samples: n,
    }
}

/// A scenario paired with its measured trajectory.
pub struct Case {
    pub scenario: Scenario,
    pub measured: Vec<Sample>,
}

pub fn evaluate(cases: &[Case], p: &Params) -> Vec<Errors> {
    cases.iter().map(|c| compare(&replay(&c.scenario, p), p.timestep, &c.measured)).collect()
}

/// Fit objective: sum over cases of pos_rmse^2 + 0.1 * speed_rmse^2.
pub fn objective(cases: &[Case], p: &Params) -> f64 {
    evaluate(cases, p).iter().map(|e| e.pos_rmse.powi(2) + 0.1 * e.speed_rmse.powi(2)).sum()
}

// ---------- Fitter ----------

#[derive(Clone, Debug)]
pub struct FitResult {
    pub params: Params,
    pub objective: f64,
    pub iterations: usize,
}

/// Nelder-Mead over the named params. Searches in coordinates scaled by the starting values
/// (so params of different magnitude converge together); values are kept non-negative.
pub fn fit(cases: &[Case], start: &Params, names: &[&str], max_iter: usize) -> Result<FitResult, String> {
    let x0: Vec<f64> = names
        .iter()
        .map(|n| start.get(n).ok_or(format!("fit: unknown param '{n}'")))
        .collect::<Result<_, _>>()?;
    let scale: Vec<f64> = x0.iter().map(|v| if v.abs() > 1e-6 { v.abs() } else { 1.0 }).collect();
    let dim = names.len();
    let build = |u: &[f64]| {
        let mut p = start.clone();
        for k in 0..dim {
            p.set(names[k], (u[k] * scale[k]).max(0.0));
        }
        p
    };
    let f = |u: &[f64]| objective(cases, &build(u));

    let u0: Vec<f64> = x0.iter().zip(&scale).map(|(v, s)| v / s).collect();
    let mut simplex: Vec<Vec<f64>> = vec![u0.clone()];
    for k in 0..dim {
        let mut u = u0.clone();
        u[k] += 0.1;
        simplex.push(u);
    }
    let mut vals: Vec<f64> = simplex.iter().map(|u| f(u)).collect();

    let mut iter = 0;
    while iter < max_iter {
        iter += 1;
        let mut idx: Vec<usize> = (0..=dim).collect();
        idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
        simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
        vals = idx.iter().map(|&i| vals[i]).collect();
        let spread = simplex.iter().skip(1).flat_map(|u| u.iter().zip(&simplex[0]).map(|(a, b)| (a - b).abs())).fold(0.0, f64::max);
        if spread < 1e-9 || (vals[dim] - vals[0]).abs() < 1e-16 {
            break;
        }
        let centroid: Vec<f64> = (0..dim).map(|k| simplex[..dim].iter().map(|u| u[k]).sum::<f64>() / dim as f64).collect();
        let along = |t: f64| -> Vec<f64> { (0..dim).map(|k| centroid[k] + t * (simplex[dim][k] - centroid[k])).collect() };
        let xr = along(-1.0);
        let fr = f(&xr);
        if fr < vals[0] {
            let xe = along(-2.0);
            let fe = f(&xe);
            if fe < fr { simplex[dim] = xe; vals[dim] = fe; } else { simplex[dim] = xr; vals[dim] = fr; }
        } else if fr < vals[dim - 1] {
            simplex[dim] = xr;
            vals[dim] = fr;
        } else {
            let xc = if fr < vals[dim] { along(-0.5) } else { along(0.5) };
            let fc = f(&xc);
            if fc < vals[dim].min(fr) {
                simplex[dim] = xc;
                vals[dim] = fc;
            } else {
                for i in 1..=dim {
                    simplex[i] = (0..dim).map(|k| simplex[0][k] + 0.5 * (simplex[i][k] - simplex[0][k])).collect();
                    vals[i] = f(&simplex[i]);
                }
            }
        }
    }
    let best = (0..=dim).min_by(|&a, &b| vals[a].total_cmp(&vals[b])).unwrap();
    Ok(FitResult { params: build(&simplex[best]), objective: vals[best], iterations: iter })
}

// ---------- Synthetic scenarios ----------

/// Two harness self-checks generated from the sim: a puck sliding to a stop, and a board bounce.
/// No players, so only puck physics (friction, drag, restitution) is exercised.
pub fn synthetic_scenarios(p: &Params) -> Vec<Scenario> {
    let mut slide = State::new_default(p, 0);
    slide.puck.x = -10.0;
    slide.puck.vx = 6.0;
    slide.puck.vy = 0.5;
    let mut bounce = State::new_default(p, 0);
    bounce.puck.x = 0.0;
    bounce.puck.y = -4.0;
    bounce.puck.vx = 3.0;
    bounce.puck.vy = 14.0;
    vec![
        Scenario {
            name: "slide_stop".into(),
            description: "SYNTHETIC: puck slides freely on ice until friction stops it".into(),
            steps: 1200,
            initial_state: slide,
            inputs: vec![],
        },
        Scenario {
            name: "wall_bounce".into(),
            description: "SYNTHETIC: puck hits the side boards at an angle and rebounds".into(),
            steps: 360,
            initial_state: bounce,
            inputs: vec![],
        },
    ]
}
