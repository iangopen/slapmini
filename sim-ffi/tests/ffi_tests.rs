use bots::{Bot, Kind, Scripted, ACT_DIM, OBS_DIM};
use sim_ffi::env::{Batch, END_GOAL_AGAINST, END_GOAL_FOR, END_LIMIT, INFO_DIM};
use sim_ffi::new_batch;

/// Deterministic pseudo-random actions for a step index.
fn actions(n: usize, t: u64) -> Vec<f32> {
    let mut a = vec![0f32; n * ACT_DIM];
    for (i, v) in a.iter_mut().enumerate() {
        let x = ((i as u64 + 1).wrapping_mul(2654435761) ^ t.wrapping_mul(40503)) % 1000;
        *v = x as f32 / 500.0 - 1.0;
    }
    a
}

fn run(b: &mut Batch, steps: u64) -> (Vec<f32>, Vec<f32>, u64) {
    let n = b.len();
    let (mut obs, mut r, mut d, mut info) = (vec![0f32; n * OBS_DIM], vec![0f32; n], vec![0u8; n], vec![0f32; n * INFO_DIM]);
    let mut opp = vec![0f32; n * OBS_DIM];
    b.reset_all(&mut obs, Some(&mut opp));
    let mut rsum = vec![0f32; n];
    for t in 0..steps {
        b.step(&actions(n, t), None, &mut obs, Some(&mut opp), &mut r, &mut d, &mut info);
        for i in 0..n {
            rsum[i] += r[i];
        }
    }
    (obs, rsum, b.hash())
}

#[test]
fn same_seed_same_everything_any_thread_count() {
    let cfg1 = r#"{"threads": 1, "max_steps": 600}"#;
    let cfg4 = r#"{"threads": 4, "max_steps": 600}"#;
    let mut a = new_batch("", cfg1, 16, 42).unwrap();
    let mut b = new_batch("", cfg4, 16, 42).unwrap();
    let mut c = new_batch("", cfg4, 16, 43).unwrap();
    let (oa, ra, ha) = run(&mut a, 500);
    let (ob, rb, hb) = run(&mut b, 500);
    let (_, _, hc) = run(&mut c, 500);
    assert_eq!(ha, hb);
    assert_eq!(oa.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), ob.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    assert_eq!(ra, rb);
    assert_ne!(ha, hc, "a different seed should give different episodes");
    // Episodes did end and reset inside the run (500 decisions × 4 > 600-step limit).
    assert!(a.envs.iter().all(|e| e.episodes >= 1));
}

/// The learner driven by the scripted chaser beats an idle opponent; the episode ends on the goal
/// with the goal_for reward and END_GOAL_FOR, and the env starts a fresh episode.
#[test]
fn goal_ends_episode_with_reward() {
    let mut b = new_batch("", r#"{"opponent": "idle", "learner_team": -1, "threads": 1}"#, 4, 7).unwrap();
    let n = b.len();
    let (mut obs, mut r, mut d, mut info) = (vec![0f32; n * OBS_DIM], vec![0f32; n], vec![0u8; n], vec![0f32; n * INFO_DIM]);
    b.reset_all(&mut obs, None);
    let mut bots: Vec<Scripted> = (0..n).map(|_| Scripted::new(Kind::Chaser)).collect();
    let mut goals = 0;
    let mut shots = 0;
    let mut teams = std::collections::BTreeSet::new();
    for _ in 0..3000 {
        let mut a = vec![0f32; n * ACT_DIM];
        for i in 0..n {
            let e = &b.envs[i];
            bots[i].act(&e.state, e.learner, &b.params).write(&mut a[i * ACT_DIM..(i + 1) * ACT_DIM]);
        }
        b.step(&a, None, &mut obs, None, &mut r, &mut d, &mut info);
        for i in 0..n {
            if d[i] == 1 {
                let inf = &info[i * INFO_DIM..(i + 1) * INFO_DIM];
                assert_eq!(inf[0], END_GOAL_FOR, "info {inf:?}");
                assert!(r[i] >= 1.0, "goal reward {}", r[i]);
                assert_eq!(inf[4], 1.0); // raw goal_for
                // A goal off the post is not a shot on net (the line must clear the post), so count, don't require.
                shots += (inf[4 + 3] >= 1.0) as u32;
                teams.insert(inf[3] as u32);
                goals += 1;
                bots[i].reset();
                assert_eq!(b.envs[i].ep_steps, 0, "env must be reset after done");
            }
        }
    }
    assert!(goals >= 8, "goals {goals}");
    assert!(shots * 4 >= goals * 3, "most goals should come from a shot on net: {shots} of {goals}");
    assert_eq!(teams.len(), 2, "learner should have played both teams");
}

#[test]
fn step_limit_and_goal_against() {
    // Idle learner vs chaser: concedes. Idle vs idle: hits the step limit.
    for (opp, want) in [("chaser", END_GOAL_AGAINST), ("idle", END_LIMIT)] {
        let cfg = format!(r#"{{"opponent": "{opp}", "max_steps": 2400, "threads": 1}}"#);
        let mut b = new_batch("", &cfg, 2, 3).unwrap();
        let n = b.len();
        let (mut obs, mut r, mut d, mut info) = (vec![0f32; n * OBS_DIM], vec![0f32; n], vec![0u8; n], vec![0f32; n * INFO_DIM]);
        b.reset_all(&mut obs, None);
        let a = vec![0f32; n * ACT_DIM];
        let mut ends = vec![];
        for _ in 0..700 {
            b.step(&a, None, &mut obs, None, &mut r, &mut d, &mut info);
            for i in 0..n {
                if d[i] == 1 {
                    ends.push((info[i * INFO_DIM], r[i]));
                }
            }
        }
        assert!(!ends.is_empty(), "{opp}: no episode ended");
        for (e, rew) in ends {
            assert_eq!(e, want, "{opp}");
            if want == END_GOAL_AGAINST {
                assert!(rew <= -0.9, "goal against reward {rew}");
            }
        }
    }
}

/// An external opponent reads the opponent action buffer; the opponent observation is the
/// opponent's own canonical view.
#[test]
fn external_opponent_and_opp_obs() {
    let mut b = new_batch("", r#"{"opponent": "external", "learner_team": 0, "threads": 1, "jitter_puck": 0, "jitter_player": 0}"#, 1, 1).unwrap();
    let (mut obs, mut opp, mut r, mut d, mut info) = (vec![0f32; OBS_DIM], vec![0f32; OBS_DIM], vec![0f32; 1], vec![0u8; 1], vec![0f32; INFO_DIM]);
    b.reset_all(&mut obs, Some(&mut opp));
    // Symmetric faceoff: both players see the same canonical picture.
    for k in 0..OBS_DIM {
        assert!((obs[k] - opp[k]).abs() < 1e-6, "field {k}: {} vs {}", obs[k], opp[k]);
    }
    // Opponent skates "forward" (canonical +x = world −x for team 1) for a second.
    let a = vec![0f32; ACT_DIM];
    let oa = vec![1.0f32, 0.0, 0.0, 0.0, 0.0];
    for _ in 0..30 {
        b.step(&a, Some(&oa), &mut obs, Some(&mut opp), &mut r, &mut d, &mut info);
    }
    let op = &b.envs[0].state.players[1];
    assert!(op.vx < -5.0 && op.x < 14.0, "opponent should skate toward −x: {op:?}");
    assert!(opp[2] > 0.3, "opponent sees itself moving forward: {}", opp[2]);
    assert!(obs[21] < -0.3, "learner sees the opponent coming at it (toward −x): {}", obs[21]);
}

#[test]
fn reward_weights_and_style_are_configurable() {
    let mut b = new_batch("", r#"{"opponent": "idle", "reward": {"puck_progress": 0, "possession": 1.0, "shot_on_net": 0}, "style": [0.5, -1], "threads": 1}"#, 1, 9).unwrap();
    let mut obs = vec![0f32; OBS_DIM];
    b.reset_all(&mut obs, None);
    assert_eq!(&obs[bots::BASE_DIM..bots::BASE_DIM + 3], &[0.5, -1.0, 0.0]);
    assert_eq!(b.envs[0].weights.possession, 1.0);
    assert_eq!(b.envs[0].weights.goal_for, 1.0, "unlisted weights keep defaults");
    assert!(new_batch("", r#"{"reward": {"bogus": 1}}"#, 1, 0).is_err(), "unknown reward terms are rejected");
    assert!(new_batch("", r#"{"opponent": "nobody"}"#, 1, 0).is_err());
}

#[test]
fn c_abi_roundtrip() {
    use std::ffi::{CStr, CString};
    unsafe {
        assert_eq!(sim_ffi::slap_obs_dim() as usize, OBS_DIM);
        assert_eq!(CStr::from_ptr(sim_ffi::slap_reward_term_name(3)).to_str().unwrap(), "shot_on_net");
        let cfg = CString::new(r#"{"threads": 2}"#).unwrap();
        let h = sim_ffi::slap_batch_new(std::ptr::null(), cfg.as_ptr(), 8, 5);
        assert!(!h.is_null());
        let n = 8;
        let (mut obs, mut r, mut d, mut info) = (vec![0f32; n * OBS_DIM], vec![0f32; n], vec![0u8; n], vec![0f32; n * INFO_DIM]);
        assert_eq!(sim_ffi::slap_batch_reset(h, obs.as_mut_ptr(), std::ptr::null_mut()), 0);
        let h0 = sim_ffi::slap_batch_hash(h);
        let a = actions(n, 0);
        assert_eq!(sim_ffi::slap_batch_step(h, a.as_ptr(), std::ptr::null(), obs.as_mut_ptr(), std::ptr::null_mut(), r.as_mut_ptr(), d.as_mut_ptr(), info.as_mut_ptr()), 0);
        assert_ne!(sim_ffi::slap_batch_hash(h), h0);
        // State JSON round trip leaves the env hash unchanged.
        let need = sim_ffi::slap_batch_get_state(h, 3, std::ptr::null_mut(), 0);
        let mut buf = vec![0u8; need as usize + 1];
        assert_eq!(sim_ffi::slap_batch_get_state(h, 3, buf.as_mut_ptr() as *mut _, buf.len() as u32), need);
        let eh = sim_ffi::slap_batch_env_hash(h, 3);
        assert_eq!(sim_ffi::slap_batch_set_state(h, 3, buf.as_ptr() as *const _), 0);
        assert_eq!(sim_ffi::slap_batch_env_hash(h, 3), eh);
        // Errors come back as −1 with a message.
        assert_eq!(sim_ffi::slap_batch_set_opponent(h, 99, 0), -1);
        assert!(CStr::from_ptr(sim_ffi::slap_last_error()).to_str().unwrap().contains("out of range"));
        let bad = CString::new("{nope").unwrap();
        assert!(sim_ffi::slap_batch_new(std::ptr::null(), bad.as_ptr(), 1, 0).is_null());
        sim_ffi::slap_batch_free(h);
    }
}
