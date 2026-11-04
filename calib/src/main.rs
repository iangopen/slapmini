//! calib CLI.
//!
//!   calib eval  --params params.json <scenario.json> <traj.csv> [<scenario> <traj> ...]
//!   calib fit   --params params.json --fit name1,name2 --out fitted.json [--iters N] [--objective time|path] <scenario> <traj> ...
//!   calib synth --params params.json --out-dir calib/data
//!   calib replay --params params.json <scenario.json> <out.csv>

use calib::{evaluate, fit_with, parse_csv, replay, synthetic_scenarios, write_csv, Case, Objective, Scenario};
use sim::ParamFile;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  calib eval   --params P <scenario.json> <traj.csv> [...]\n  calib fit    --params P --fit a,b --out OUT [--iters N] [--objective time|path] <scenario.json> <traj.csv> [...]\n  calib synth  --params P --out-dir DIR\n  calib replay --params P <scenario.json> <out.csv>"
    );
    ExitCode::from(2)
}

struct Args {
    flags: Vec<(String, String)>,
    pos: Vec<String>,
}

impl Args {
    fn parse(raw: &[String]) -> Result<Self, String> {
        let (mut flags, mut pos) = (Vec::new(), Vec::new());
        let mut it = raw.iter();
        while let Some(a) = it.next() {
            if let Some(k) = a.strip_prefix("--") {
                let v = it.next().ok_or(format!("--{k} needs a value"))?;
                flags.push((k.to_string(), v.clone()));
            } else {
                pos.push(a.clone());
            }
        }
        Ok(Args { flags, pos })
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.flags.iter().rev().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
    }
}

fn read(p: &str) -> Result<String, String> {
    std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))
}

fn load_cases(pos: &[String]) -> Result<Vec<Case>, String> {
    if pos.is_empty() || pos.len() % 2 != 0 {
        return Err("expected pairs of <scenario.json> <traj.csv>".into());
    }
    pos.chunks(2)
        .map(|c| Ok(Case { scenario: Scenario::from_json(&read(&c[0])?)?, measured: parse_csv(&read(&c[1])?)? }))
        .collect()
}

fn print_errors(cases: &[Case], errs: &[calib::Errors]) {
    println!("{:<20} {:>8} {:>14} {:>14} {:>14} {:>14} {:>14}", "scenario", "samples", "pos_rmse(m)", "speed_rmse", "stick_rmse", "path_rmse(m)", "v2_rmse");
    for (c, e) in cases.iter().zip(errs) {
        let st = e.stick_rmse.map_or("-".to_string(), |v| format!("{v:.6e}"));
        println!("{:<20} {:>8} {:>14.6e} {:>14.6e} {:>14} {:>14.6e} {:>14.6e}", c.scenario.name, e.samples, e.pos_rmse, e.speed_rmse, st, e.path_rmse, e.v2_rmse);
    }
}

fn run(raw: &[String]) -> Result<(), String> {
    let cmd = raw.first().ok_or("missing command")?.as_str();
    let a = Args::parse(&raw[1..])?;
    let params_path = a.get("params").unwrap_or("params.json");
    let mut pfile = ParamFile::from_json(&read(params_path)?)?;
    let params = pfile.to_params()?;
    match cmd {
        "eval" => {
            let cases = load_cases(&a.pos)?;
            print_errors(&cases, &evaluate(&cases, &params));
        }
        "fit" => {
            let names: Vec<&str> = a.get("fit").ok_or("--fit required")?.split(',').map(str::trim).collect();
            let out = a.get("out").ok_or("--out required")?;
            let iters = a.get("iters").map(|s| s.parse::<usize>().map_err(|e| e.to_string())).transpose()?.unwrap_or(500);
            let kind: Objective = a.get("objective").unwrap_or("time").parse()?;
            let cases = load_cases(&a.pos)?;
            println!("objective: {kind:?}");
            println!("before:");
            print_errors(&cases, &evaluate(&cases, &params));
            let r = fit_with(&cases, &params, &names, iters, kind)?;
            println!("after ({} iterations, objective {:.3e}):", r.iterations, r.objective);
            print_errors(&cases, &evaluate(&cases, &r.params));
            for n in &names {
                println!("  {n}: {} -> {}", params.get(n).unwrap(), r.params.get(n).unwrap());
            }
            pfile.update_from(&r.params, &names);
            std::fs::write(out, pfile.to_json() + "\n").map_err(|e| format!("{out}: {e}"))?;
            println!("wrote {out}");
        }
        "synth" => {
            let dir = PathBuf::from(a.get("out-dir").ok_or("--out-dir required")?);
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            for sc in synthetic_scenarios(&params) {
                let s = dir.join(format!("{}.scenario.json", sc.name));
                let c = dir.join(format!("{}.csv", sc.name));
                std::fs::write(&s, sc.to_json() + "\n").map_err(|e| e.to_string())?;
                std::fs::write(&c, write_csv(&replay(&sc, &params))).map_err(|e| e.to_string())?;
                println!("wrote {} and {}", s.display(), c.display());
            }
        }
        "replay" => {
            if a.pos.len() != 2 {
                return Err("replay needs <scenario.json> <out.csv>".into());
            }
            let sc = Scenario::from_json(&read(&a.pos[0])?)?;
            std::fs::write(Path::new(&a.pos[1]), write_csv(&replay(&sc, &params))).map_err(|e| e.to_string())?;
            println!("wrote {}", a.pos[1]);
        }
        _ => return Err(format!("unknown command '{cmd}'")),
    }
    Ok(())
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() {
        return usage();
    }
    match run(&raw) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            usage()
        }
    }
}
