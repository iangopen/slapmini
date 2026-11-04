use calib::{evaluate, fit, parse_csv, replay, synthetic_scenarios, write_csv, Case};
use sim::Params;

const NAMES: [&str; 2] = ["ice_friction_decel", "board_restitution"];

#[test]
fn fitter_recovers_friction_and_restitution() {
    let truth = Params::default();
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
    let p = Params::default();
    let sc = &synthetic_scenarios(&p)[1];
    let s = replay(sc, &p);
    assert_eq!(parse_csv(&write_csv(&s)).unwrap(), s);
}

#[test]
fn scenario_json_roundtrip_replays_identically() {
    let p = Params::default();
    let mut sc = synthetic_scenarios(&p)[0].clone();
    sc.initial_state = sim::State::new_default(&p, 2);
    sc.initial_state.puck.vx = 3.0;
    sc.inputs.push(calib::InputEvent { step: 10, player: 0, input: sim::PlayerInput { move_x: 1.0, stick_target_angle: 0.7, ..Default::default() } });
    let back = calib::Scenario::from_json(&sc.to_json()).unwrap();
    assert_eq!(replay(&sc, &p), replay(&back, &p));
}
