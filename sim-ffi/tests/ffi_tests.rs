use bots::{Bot, Kind, Scripted, ACT_DIM, OBS_DIM};
use sim_ffi::env::{Batch, END_LOSS, END_TRUNCATED, END_WIN, INFO_DIM, INFO_RAW};
use sim_ffi::new_batch;
use sim_ffi::reward::RewardWeights;

/// Deterministic pseudo-random actions (ACT v3: move index, stick angle, dash, lift, stick_up).
fn actions(n: usize, t: u64) -> Vec<f32> {
    let mut a = vec![0f32; n * ACT_DIM];
    for (i, v) in a.iter_mut().enumerate() {
        let x = (((i as u64 + 1).wrapping_mul(2654435761) ^ t.wrapping_mul(40503)) % 1000) as f32 / 500.0; // 0..2
        *v = match i % ACT_DIM {
            0 => (x * 4.5).floor(),
            1 => (x - 1.0) * 3.14,
            _ => (x > 1.8) as u8 as f32,
        };
    }
    a
}

struct Bufs {
    obs: Vec<f32>,
    opp: Vec<f32>,
    fin: Vec<f32>,
    r: Vec<f32>,
    d: Vec<u8>,
    info: Vec<f32>,
}

impl Bufs {
    fn new(n: usize) -> Self {
        Bufs { obs: vec![0.0; n * OBS_DIM], opp: vec![0.0; n * OBS_DIM], fin: vec![0.0; n * OBS_DIM], r: vec![0.0; n], d: vec![0; n], info: vec![0.0; n * INFO_DIM] }
    }
    fn step(&mut self, b: &mut Batch, a: &[f32], oa: Option<&[f32]>) {
        b.step(a, oa, &mut self.obs, Some(&mut self.opp), Some(&mut self.fin), &mut self.r, &mut self.d, &mut self.info);
    }
    fn info(&self, i: usize) -> &[f32] {
        &self.info[i * INFO_DIM..(i + 1) * INFO_DIM]
    }
}

fn run(b: &mut Batch, steps: u64) -> (Vec<f32>, Vec<f32>, u64) {
    let n = b.len();
    let mut s = Bufs::new(n);
    b.reset_all(&mut s.obs, Some(&mut s.opp));
    let mut rsum = vec![0f32; n];
    for t in 0..steps {
        s.step(b, &actions(n, t), None);
        for i in 0..n {
            rsum[i] += s.r[i];
        }
    }
    (s.obs, rsum, b.hash())
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn same_seed_same_everything_any_thread_count() {
    for mode in ["first_goal", "match"] {
        let cfg1 = format!(r#"{{"threads": 1, "max_steps": 600, "mode": "{mode}"}}"#);
        let cfg4 = format!(r#"{{"threads": 4, "max_steps": 600, "mode": "{mode}"}}"#);
        let mut a = new_batch("", &cfg1, 16, 42).unwrap();
        let mut b = new_batch("", &cfg4, 16, 42).unwrap();
        let mut c = new_batch("", &cfg4, 16, 43).unwrap();
        let (oa, ra, ha) = run(&mut a, 500);
        let (ob, rb, hb) = run(&mut b, 500);
        let (_, _, hc) = run(&mut c, 500);
        assert_eq!(ha, hb, "{mode}");
        assert_eq!(bits(&oa), bits(&ob), "{mode}");
        assert_eq!(ra, rb, "{mode}");
        assert_ne!(ha, hc, "{mode}: a different seed should give different episodes");
        // Episodes did end and reset inside the run (500 decisions × 4 > 600-step limit).
        assert!(a.envs.iter().all(|e| e.episodes >= 1), "{mode}");
        eprintln!("determinism {mode}: 16 envs seed 42, threads 1 vs 4: batch hash {ha:016x} / {hb:016x}; seed 43: {hc:016x}");
    }
}

/// Learner actions from a scripted bot run outside the env (as a policy would).
fn scripted_actions(b: &Batch, bots: &mut [Scripted]) -> Vec<f32> {
    let n = b.len();
    let mut a = vec![0f32; n * ACT_DIM];
    for i in 0..n {
        let e = &b.envs[i];
        bots[i].act(&e.state, e.learner, &b.params).write(&mut a[i * ACT_DIM..(i + 1) * ACT_DIM]);
    }
    a
}

/// The learner driven by the scripted chaser beats an idle opponent; the episode ends on the goal
/// with the goal_for reward and END_WIN, and the env starts a fresh episode.
#[test]
fn goal_ends_episode_with_reward() {
    // The scripted chaser is a 120 Hz controller: drive it at one decision per sim step (at the
    // training rate of 30 Hz it can hold the puck without ever meeting its swing conditions).
    let mut b = new_batch("", r#"{"opponent": "idle", "learner_team": -1, "threads": 1, "action_repeat": 1}"#, 4, 7).unwrap();
    let n = b.len();
    let mut s = Bufs::new(n);
    b.reset_all(&mut s.obs, None);
    let mut bots: Vec<Scripted> = (0..n).map(|_| Scripted::new(Kind::Chaser)).collect();
    let (mut goals, mut shots) = (0, 0);
    let mut teams = std::collections::BTreeSet::new();
    for _ in 0..12000 {
        let a = scripted_actions(&b, &mut bots);
        s.step(&mut b, &a, None);
        for i in 0..n {
            if s.d[i] == 1 {
                let inf = s.info(i).to_vec();
                assert_eq!(inf[0], END_WIN, "info {inf:?}");
                assert!(s.r[i] >= 1.0, "goal reward {}", s.r[i]);
                assert_eq!((inf[4], inf[5]), (1.0, 0.0));
                assert_eq!(inf[INFO_RAW], 1.0); // raw goal_for
                assert_eq!(inf[INFO_RAW + 6], 0.0, "no match_win term in first-goal mode");
                // A goal off the post is not a shot on net (the line must clear the post), so count, don't require.
                shots += (inf[INFO_RAW + 3] >= 1.0) as u32;
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
    // Idle learner vs chaser: concedes. Idle vs idle: hits the step limit (truncated).
    for (opp, want) in [("chaser", END_LOSS), ("idle", END_TRUNCATED)] {
        let cfg = format!(r#"{{"opponent": "{opp}", "max_steps": 2400, "threads": 1}}"#);
        let mut b = new_batch("", &cfg, 2, 3).unwrap();
        let n = b.len();
        let mut s = Bufs::new(n);
        b.reset_all(&mut s.obs, None);
        let a = vec![0f32; n * ACT_DIM];
        let mut ends = vec![];
        for _ in 0..700 {
            s.step(&mut b, &a, None);
            for i in 0..n {
                if s.d[i] == 1 {
                    ends.push((s.info(i)[0], s.r[i]));
                }
            }
        }
        assert!(!ends.is_empty(), "{opp}: no episode ended");
        for (e, rew) in ends {
            assert_eq!(e, want, "{opp}");
            if want == END_LOSS {
                assert!(rew <= -0.9, "goal against reward {rew}");
            }
        }
    }
}

/// On a truncated episode, final_obs holds the observation of the last state before the reset
/// (what the value bootstrap needs), while obs is already the new episode's faceoff.
#[test]
fn final_obs_is_the_pre_reset_observation() {
    let mut b = new_batch("", r#"{"opponent": "idle", "max_steps": 40, "threads": 1, "learner_team": 0}"#, 1, 5).unwrap();
    let mut s = Bufs::new(1);
    b.reset_all(&mut s.obs, None);
    let fwd = [1.0f32, 0.0, 0.0, 0.0, 0.0]; // skate +x
    let mut last_obs = s.obs.clone();
    for k in 0..10 {
        s.step(&mut b, &fwd, None);
        if k < 9 {
            assert_eq!(s.d[0], 0);
            last_obs = s.obs.clone();
        }
    }
    assert_eq!(s.d[0], 1);
    assert_eq!(s.info(0)[0], END_TRUNCATED);
    // The final observation continues the last one (the player kept skating), the new one is a faceoff.
    assert!(s.fin[2] > last_obs[2], "final obs speed {} vs previous {}", s.fin[2], last_obs[2]);
    assert!(s.obs[2].abs() < 1e-6, "new episode starts at rest: {}", s.obs[2]);
}

/// An external opponent reads the opponent action buffer; the opponent observation is the
/// opponent's own canonical view.
#[test]
fn external_opponent_and_opp_obs() {
    let mut b = new_batch("", r#"{"opponent": "external", "learner_team": 0, "threads": 1, "jitter_puck": 0, "jitter_player": 0}"#, 1, 1).unwrap();
    let mut s = Bufs::new(1);
    b.reset_all(&mut s.obs, Some(&mut s.opp));
    // Symmetric faceoff: both players see the same canonical picture.
    for k in 0..OBS_DIM {
        assert!((s.obs[k] - s.opp[k]).abs() < 1e-6, "field {k}: {} vs {}", s.obs[k], s.opp[k]);
    }
    // Opponent skates "forward" (canonical +x = world −x for team 1) for a second.
    let a = vec![0f32; ACT_DIM];
    let oa = vec![1.0f32, 0.0, 0.0, 0.0, 0.0];
    for _ in 0..30 {
        s.step(&mut b, &a, Some(&oa));
    }
    let op = &b.envs[0].state.players[1];
    assert!(op.vx < -5.0 && op.x < 14.0, "opponent should skate toward −x: {op:?}");
    assert!(s.opp[2] > 0.3, "opponent sees itself moving forward: {}", s.opp[2]);
    assert!(s.obs[21] < -0.3, "learner sees the opponent coming at it (toward −x): {}", s.obs[21]);
}

/// ACT v3 / OBS v3: action field 4 raises the stick; obs 40 = own, 41 = opponent raised.
#[test]
fn stick_up_action_and_obs_flags() {
    let mut b = new_batch("", r#"{"opponent": "external", "learner_team": 0, "threads": 1, "jitter_puck": 0, "jitter_player": 0}"#, 1, 1).unwrap();
    let mut s = Bufs::new(1);
    b.reset_all(&mut s.obs, Some(&mut s.opp));
    assert_eq!((s.obs[bots::STICK_UP_IDX], s.obs[bots::STICK_UP_IDX + 1]), (0.0, 0.0));
    let a = vec![0.0f32, 0.0, 0.0, 0.0, 1.0]; // learner raises
    let oa = vec![0.0f32; ACT_DIM];
    s.step(&mut b, &a, Some(&oa));
    assert!(b.envs[0].state.players[0].stick_up && !b.envs[0].state.players[1].stick_up);
    assert_eq!((s.obs[bots::STICK_UP_IDX], s.obs[bots::STICK_UP_IDX + 1]), (1.0, 0.0), "learner: own raised");
    assert_eq!((s.opp[bots::STICK_UP_IDX], s.opp[bots::STICK_UP_IDX + 1]), (0.0, 1.0), "opponent sees the learner's raised stick");
    s.step(&mut b, &oa, Some(&a)); // now only the opponent raises
    assert_eq!((s.obs[bots::STICK_UP_IDX], s.obs[bots::STICK_UP_IDX + 1]), (0.0, 1.0));
    assert_eq!(&s.obs[bots::BASE_DIM..bots::BASE_DIM + bots::STYLE_DIM], &[0.0; 8], "the style slot stays at 32..40");
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
    assert!(new_batch("", r#"{"mode": "hockey"}"#, 1, 0).is_err());
    assert!(new_batch("", r#"{"style_mode": "wild"}"#, 1, 0).is_err());
}

#[test]
fn style_mapping() {
    let base = RewardWeights::default();
    assert_eq!(base.style_weights(&[0.0; 8]), base, "style 0 = base weights");
    let w = base.style_weights(&[1.0, -1.0, 0.5, 1.0]);
    assert_eq!((w.goal_for, w.goal_against, w.match_win), (base.goal_for, base.goal_against, base.match_win));
    assert!((w.puck_progress - 2.0 * base.puck_progress).abs() < 1e-7);
    assert_eq!(w.shot_on_net, 0.0);
    assert!((w.possession - 1.5 * base.possession).abs() < 1e-7);
    assert!((w.defensive_position - sim_ffi::reward::DEF_WEIGHT).abs() < 1e-7);
    // "fixed": weights follow set_style; "sample": every episode draws a style in [−1, 1) for 4 slots.
    let mut b = new_batch("", r#"{"style_mode": "fixed", "threads": 1}"#, 2, 1).unwrap();
    b.set_style(1, &[0.0, 1.0]);
    assert_eq!(b.envs[0].weights, base);
    assert!((b.envs[1].weights.shot_on_net - 2.0 * base.shot_on_net).abs() < 1e-7);
    let b = new_batch("", r#"{"style_mode": "sample", "threads": 1}"#, 8, 1).unwrap();
    for e in &b.envs {
        assert!(e.style[..4].iter().all(|s| (-1.0..1.0).contains(s)) && e.style[4..].iter().all(|s| *s == 0.0));
        assert_eq!(e.weights, base.style_weights(&e.style));
    }
    assert_ne!(b.envs[0].style, b.envs[1].style);
}

/// A style of zeros reproduces the default environment exactly: same batch hash, observations
/// and rewards as style_mode "off" with no style (both modes).
#[test]
fn style_zero_reproduces_default_hashes() {
    for mode in ["first_goal", "match"] {
        let mut a = new_batch("", &format!(r#"{{"threads": 1, "mode": "{mode}"}}"#), 8, 77).unwrap();
        let mut b = new_batch("", &format!(r#"{{"threads": 1, "mode": "{mode}", "style_mode": "fixed", "style": [0,0,0,0,0,0,0,0]}}"#), 8, 77).unwrap();
        let (oa, ra, ha) = run(&mut a, 1500);
        let (ob, rb, hb) = run(&mut b, 1500);
        eprintln!("style zero {mode}: default hash {ha:016x}, style-zero hash {hb:016x}");
        assert_eq!(ha, hb, "{mode}");
        assert_eq!(bits(&oa), bits(&ob));
        assert_eq!(bits(&ra), bits(&rb));
    }
}

/// Full matches between scripted bots in match mode: the mercy rule, the end of the clock with a
/// leader, and an overtime goal each happen, and the episode result and match_win reward follow.
#[test]
fn match_mode_endings() {
    // Mercy: chaser (learner) vs idle ends at a 5-goal lead long before the clock.
    let (r, _) = play_match("", "idle", 11, 1);
    let m = r[0];
    assert_eq!((m.end, m.gf, m.ga), (END_WIN, 5.0, 0.0), "{m:?}");
    assert!(m.seconds < 200.0, "mercy should end early: {m:?}");
    assert_eq!(m.match_win, 1.0);
    // Clock end with a leader: chaser vs defender over the full 5:00 (low scoring).
    let (r, _) = play_match("", "defender", 12, 4);
    let clock_end = r.iter().find(|m| m.seconds >= 299.9 && !m.overtime && m.gf != m.ga);
    assert!(clock_end.is_some(), "no regulation clock end in {r:?}");
    // Overtime: chaser vs chaser on a short clock (in-memory params only) until a tie goes to overtime.
    let mut p = sim::ParamFile::from_json(sim::DEFAULT_PARAMS_JSON).unwrap();
    p.params.get_mut("match_length_s").unwrap().value = 20.0;
    let short = serde_json::to_string(&p).unwrap();
    let (r, _) = play_match(&short, "chaser", 13, 16);
    let ot = r.iter().find(|m| m.overtime && m.end != END_TRUNCATED);
    assert!(ot.is_some(), "no overtime goal in {r:?}");
    let ot = ot.unwrap();
    assert_eq!((ot.gf - ot.ga).abs(), 1.0, "overtime is decided by one goal: {ot:?}");
}

/// Match mode jitters every faceoff after a goal (the sim alone resets to the same spawn each time),
/// unless jitter_every_faceoff is false.
#[test]
fn faceoffs_after_goals_are_jittered() {
    for (jit, want_distinct) in [(true, true), (false, false)] {
        let cfg = format!(r#"{{"mode": "match", "opponent": "idle", "threads": 1, "learner_team": 0, "jitter_every_faceoff": {jit}}}"#);
        let mut b = new_batch("", &cfg, 1, 21).unwrap();
        let mut s = Bufs::new(1);
        b.reset_all(&mut s.obs, None);
        let mut bot = vec![Scripted::new(Kind::Chaser)];
        let mut faceoffs = vec![];
        let mut gf = 0.0;
        while faceoffs.len() < 3 {
            let a = scripted_actions(&b, &mut bot);
            s.step(&mut b, &a, None);
            if s.d[0] == 1 {
                break;
            }
            if s.info(0)[4] != gf {
                gf = s.info(0)[4];
                bot[0].reset();
                let st = &b.envs[0].state;
                faceoffs.push((st.puck.x, st.puck.y, st.players[0].x, st.players[0].y));
            }
        }
        assert_eq!(faceoffs.len(), 3, "{faceoffs:?}");
        let distinct = faceoffs.windows(2).all(|w| w[0] != w[1]);
        assert_eq!(distinct, want_distinct, "jitter {jit}: {faceoffs:?}");
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct MatchResult {
    end: f32,
    gf: f32,
    ga: f32,
    seconds: f32,
    overtime: bool,
    match_win: f32,
    ret: f32,
}

/// One full match per env, learner = scripted chaser vs `opp`. Returns the results and the batch hash.
fn play_match(params: &str, opp: &str, seed: u64, n: usize) -> (Vec<MatchResult>, u64) {
    let mut b = new_batch(params, &format!(r#"{{"mode": "match", "opponent": "{opp}", "threads": 1}}"#), n, seed).unwrap();
    let mut s = Bufs::new(n);
    b.reset_all(&mut s.obs, None);
    let mut bots: Vec<Scripted> = (0..n).map(|_| Scripted::new(Kind::Chaser)).collect();
    let mut out: Vec<Option<MatchResult>> = vec![None; n];
    let mut overtime = vec![false; n];
    let mut score = vec![(0u32, 0u32); n];
    while out.iter().any(|o| o.is_none()) {
        let a = scripted_actions(&b, &mut bots);
        s.step(&mut b, &a, None);
        for i in 0..n {
            let inf = s.info(i).to_vec();
            if out[i].is_none() {
                let ms = b.envs[i].state.match_state;
                if s.d[i] == 0 {
                    overtime[i] |= ms.is_some_and(|m| m.overtime);
                    // Decisions never see a frozen phase.
                    let ph = ms.unwrap().phase;
                    assert_eq!(ph, sim::MatchPhase::Playing, "env {i} decided in {ph:?}");
                }
                if (inf[4] as u32, inf[5] as u32) != score[i] {
                    score[i] = (inf[4] as u32, inf[5] as u32);
                    eprintln!("  [{opp} seed {seed} env {i}] t {:6.1} s  goal: learner {} - {} opp{}", inf[1] / 120.0, score[i].0, score[i].1, if overtime[i] { "  (overtime)" } else { "" });
                    bots[i].reset();
                }
                if s.d[i] == 1 {
                    let m = MatchResult { end: inf[0], gf: inf[4], ga: inf[5], seconds: inf[1] / 120.0, overtime: overtime[i], match_win: inf[INFO_RAW + 6], ret: inf[2] };
                    let why = if m.end == END_TRUNCATED { "truncated" } else if m.overtime { "overtime goal" } else if (m.gf - m.ga).abs() >= 5.0 { "mercy" } else { "clock end" };
                    eprintln!("  [{opp} seed {seed} env {i}] END {why}: learner {} - {} after {:.1} s of play, result {}, match_win raw {}, return {:.3}", m.gf, m.ga, m.seconds, if m.end == END_WIN { "WIN" } else if m.end == END_LOSS { "LOSS" } else { "TRUNC" }, m.match_win, m.ret);
                    out[i] = Some(m);
                }
            }
            if s.d[i] == 1 {
                bots[i].reset();
            }
        }
    }
    (out.into_iter().map(|o| o.unwrap()).collect(), b.hash())
}

#[test]
fn c_abi_roundtrip() {
    use std::ffi::{CStr, CString};
    unsafe {
        assert_eq!(sim_ffi::slap_obs_dim() as usize, OBS_DIM);
        assert_eq!(sim_ffi::slap_act_dim() as usize, 5);
        assert_eq!(sim_ffi::slap_obs_dim() as usize, 42);
        assert_eq!((sim_ffi::slap_act_version(), sim_ffi::slap_obs_version()), (3, 3));
        assert_eq!(sim_ffi::slap_abi_version(), 3);
        assert_eq!(CStr::from_ptr(sim_ffi::slap_reward_term_name(3)).to_str().unwrap(), "shot_on_net");
        assert_eq!(CStr::from_ptr(sim_ffi::slap_reward_term_name(6)).to_str().unwrap(), "match_win");
        let cfg = CString::new(r#"{"threads": 2}"#).unwrap();
        let h = sim_ffi::slap_batch_new(std::ptr::null(), cfg.as_ptr(), 8, 5);
        assert!(!h.is_null());
        let n = 8;
        let (mut obs, mut r, mut d, mut info) = (vec![0f32; n * OBS_DIM], vec![0f32; n], vec![0u8; n], vec![0f32; n * INFO_DIM]);
        assert_eq!(sim_ffi::slap_batch_reset(h, obs.as_mut_ptr(), std::ptr::null_mut()), 0);
        let h0 = sim_ffi::slap_batch_hash(h);
        let a = actions(n, 0);
        assert_eq!(
            sim_ffi::slap_batch_step(h, a.as_ptr(), std::ptr::null(), obs.as_mut_ptr(), std::ptr::null_mut(), std::ptr::null_mut(), r.as_mut_ptr(), d.as_mut_ptr(), info.as_mut_ptr()),
            0
        );
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
