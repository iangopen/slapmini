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
    /// Timing-free: position RMSE at equal arc length along the path, m (see `path_errors`).
    pub path_rmse: f64,
    /// Timing-free: RMSE of v^2 compared at equal distance travelled, m^2/s^2.
    pub v2_rmse: f64,
    pub samples: usize,
}

/// Which fit objective to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    /// Default: sum of pos_rmse^2 + 0.1 * speed_rmse^2 (positions compared at equal times).
    Time,
    /// Timing-robust: sum of path_rmse^2 + PATH_V2_WEIGHT * v2_rmse^2 (compared at equal arc length).
    Path,
}

impl std::str::FromStr for Objective {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "time" => Ok(Objective::Time),
            "path" => Ok(Objective::Path),
            _ => Err(format!("unknown objective '{s}' (time | path)")),
        }
    }
}

/// Weight of the v^2 term in the path objective: an RMSE of 1 m^2/s^2 in v^2 counts like 0.1 m of path error.
pub const PATH_V2_WEIGHT: f64 = 0.01;
/// Half-width (samples) of the local quadratic fits that smooth measured positions and velocities.
pub const PATH_SMOOTH_HALF: usize = 5;

/// Local quadratic least-squares fit of the measured samples around index k: (position, velocity) at t_k.
/// The capture's timing jitter enters only as zero-mean noise here.
fn local_fit(meas: &[Sample], k: usize, half: usize) -> ([f64; 3], [f64; 3]) {
    let a = k.saturating_sub(half);
    let b = (k + half).min(meas.len() - 1);
    let t0 = meas[k].t;
    let n = b - a + 1;
    let deg2 = n >= 5;
    // Normal equations for y = c0 + c1 dt (+ c2 dt^2).
    let (mut s0, mut s1, mut s2, mut s3, mut s4) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut r = [[0.0f64; 3]; 3]; // r[axis] = [sum y, sum y dt, sum y dt^2]
    for m in &meas[a..=b] {
        let d = m.t - t0;
        s0 += 1.0;
        s1 += d;
        s2 += d * d;
        s3 += d * d * d;
        s4 += d * d * d * d;
        for (ax, y) in [m.x, m.y, m.z].into_iter().enumerate() {
            r[ax][0] += y;
            r[ax][1] += y * d;
            r[ax][2] += y * d * d;
        }
    }
    let mut pos = [meas[k].x, meas[k].y, meas[k].z];
    let mut vel = [0.0; 3];
    for ax in 0..3 {
        if deg2 {
            let m3 = [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]];
            if let Some(c) = solve3(m3, r[ax]) {
                pos[ax] = c[0];
                vel[ax] = c[1];
                continue;
            }
        }
        let det = s0 * s2 - s1 * s1;
        if det.abs() > 1e-15 {
            pos[ax] = (r[ax][0] * s2 - r[ax][1] * s1) / det;
            vel[ax] = (s0 * r[ax][1] - s1 * r[ax][0]) / det;
        }
    }
    (pos, vel)
}

fn solve3(m: [[f64; 3]; 3], r: [f64; 3]) -> Option<[f64; 3]> {
    let det = |a: [[f64; 3]; 3]| a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let d = det(m);
    if d.abs() < 1e-18 {
        return None;
    }
    let mut out = [0.0; 3];
    for c in 0..3 {
        let mut mc = m;
        for row in 0..3 {
            mc[row][c] = r[row];
        }
        out[c] = det(mc) / d;
    }
    Some(out)
}

/// Timing-free comparison. Both paths get the same smoothing (local quadratic fits over
/// +-PATH_SMOOTH_HALF samples; the sim is sampled at the measured times first, so the corner rounding
/// at a bounce is identical on both sides) and are parameterized by distance travelled s.
/// Returns (path_rmse, v2_rmse):
/// - path_rmse: distance between measured and sim positions at the same s (sim clamped at its end);
/// - v2_rmse: v^2 at the same s. For constant friction v^2 falls linearly with s.
/// The two paths are aligned by s, never by frame time, so capture jitter, duplicated and skipped
/// frames only add zero-mean noise to the measured positions and velocities.
pub fn path_errors(sim: &[Sample], dt: f64, meas: &[Sample]) -> (f64, f64) {
    if meas.len() < 3 || sim.len() < 2 {
        return (0.0, 0.0);
    }
    let simm: Vec<Sample> = meas
        .iter()
        .map(|m| {
            let (x, y, z, _) = sim_at(sim, dt, m.t);
            Sample { t: m.t, x, y, z, stick_angle: None }
        })
        .collect();
    let smooth = |ss: &[Sample]| -> (Vec<[f64; 3]>, Vec<f64>, Vec<f64>) {
        let fits: Vec<_> = (0..ss.len()).map(|k| local_fit(ss, k, PATH_SMOOTH_HALF)).collect();
        let mut s = vec![0.0; ss.len()];
        for k in 1..ss.len() {
            let (a, b) = (fits[k - 1].0, fits[k].0);
            s[k] = s[k - 1] + ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
        }
        let v2 = fits.iter().map(|f| f.1[0] * f.1[0] + f.1[1] * f.1[1] + f.1[2] * f.1[2]).collect();
        (fits.into_iter().map(|f| f.0).collect(), s, v2)
    };
    let (mp, ms, mv2) = smooth(meas);
    let (sp, ss, sv2) = smooth(&simm);
    let last = ss.len() - 1;
    let (mut pe, mut ve) = (0.0, 0.0);
    let mut j = 0usize;
    for k in 0..meas.len() {
        let s = ms[k];
        while j + 1 < ss.len() && ss[j + 1] < s {
            j += 1;
        }
        let (p, v2) = if j >= last || s >= ss[last] {
            (sp[last], sv2[last])
        } else {
            let span = ss[j + 1] - ss[j];
            let a = if span > 0.0 { ((s - ss[j]) / span).clamp(0.0, 1.0) } else { 0.0 };
            let l = |p: f64, q: f64| p + (q - p) * a;
            ([l(sp[j][0], sp[j + 1][0]), l(sp[j][1], sp[j + 1][1]), l(sp[j][2], sp[j + 1][2])], l(sv2[j], sv2[j + 1]))
        };
        pe += (mp[k][0] - p[0]).powi(2) + (mp[k][1] - p[1]).powi(2) + (mp[k][2] - p[2]).powi(2);
        ve += (mv2[k] - v2).powi(2);
    }
    let n = meas.len() as f64;
    ((pe / n).sqrt(), (ve / n).sqrt())
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
    let (path_rmse, v2_rmse) = path_errors(sim, dt, meas);
    Errors {
        pos_rmse: if n > 0 { (pos / n as f64).sqrt() } else { 0.0 },
        speed_rmse: if spd_n > 0 { (spd / spd_n as f64).sqrt() } else { 0.0 },
        stick_rmse: if stick_n > 0 { Some((stick / stick_n as f64).sqrt()) } else { None },
        path_rmse,
        v2_rmse,
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

/// Fit objective (default): sum over cases of pos_rmse^2 + 0.1 * speed_rmse^2.
pub fn objective(cases: &[Case], p: &Params) -> f64 {
    objective_with(cases, p, Objective::Time)
}

/// Fit objective of the chosen kind.
pub fn objective_with(cases: &[Case], p: &Params, kind: Objective) -> f64 {
    evaluate(cases, p)
        .iter()
        .map(|e| match kind {
            Objective::Time => e.pos_rmse.powi(2) + 0.1 * e.speed_rmse.powi(2),
            Objective::Path => e.path_rmse.powi(2) + PATH_V2_WEIGHT * e.v2_rmse.powi(2),
        })
        .sum()
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
    fit_with(cases, start, names, max_iter, Objective::Time)
}

/// `fit` with a chosen objective.
pub fn fit_with(cases: &[Case], start: &Params, names: &[&str], max_iter: usize, kind: Objective) -> Result<FitResult, String> {
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
    let f = |u: &[f64]| objective_with(cases, &build(u), kind);

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
