//! Batched slapmini environments behind a plain C ABI, for training from Python (ctypes) or any
//! language with a C FFI. No PyO3/maturin: build with `cargo build --release -p sim-ffi` and load
//! `target/release/{sim_ffi.dll | libsim_ffi.so | libsim_ffi.dylib}`.
//!
//! Conventions: every buffer is caller-owned, row-major f32 (dones are u8); `n` = number of envs.
//! Functions returning i32 return 0 on success and −1 on error (message via `slap_last_error`).
//! A batch handle must not be used from two threads at once (the batch parallelizes internally).
//!
//! | buffer | shape |
//! |---|---|
//! | obs, opp_obs, final_obs | n × slap_obs_dim() |
//! | actions, opp_actions | n × slap_act_dim() (format: bots::action) |
//! | rewards | n |
//! | dones | n (u8: 1 = episode ended this step; the env was reset and obs is the new episode's first;
//! |  | final_obs holds the learner's observation of the final state, info[0] says win / loss / truncated) |
//! | info | n × slap_info_dim() (see env::INFO_DIM) |

pub mod env;
pub mod reward;

use env::{opponent_id, Batch, Config, INFO_DIM};
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};

pub use bots::{ACT_DIM, OBS_DIM, OBS_VERSION, STYLE_DIM};
pub use reward::{RewardWeights, N_TERMS, TERM_NAMES};

/// Bump when any exported signature or buffer layout changes.
/// v2: 4-float action (9-way move), final_obs in slap_batch_step, INFO_DIM 14, 7 reward terms.
pub const ABI_VERSION: u32 = 2;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_error(msg: impl Into<String>) {
    let s = CString::new(msg.into().replace('\0', " ")).unwrap();
    LAST_ERROR.with(|e| *e.borrow_mut() = s);
}

/// Build a batch from Rust (the C entry point wraps this).
pub fn new_batch(params_json: &str, config_json: &str, n: usize, seed: u64) -> Result<Batch, String> {
    let params = if params_json.trim().is_empty() { sim::Params::default() } else { sim::Params::from_json(params_json)? };
    let cfg: Config =
        if config_json.trim().is_empty() { Config::default() } else { serde_json::from_str(config_json).map_err(|e| format!("config json: {e}"))? };
    Batch::new(params, cfg, n, seed)
}

unsafe fn cstr<'a>(p: *const c_char) -> Result<&'a str, String> {
    if p.is_null() {
        return Ok("");
    }
    CStr::from_ptr(p).to_str().map_err(|e| format!("utf8: {e}"))
}

unsafe fn batch<'a>(h: *mut Batch) -> Result<&'a mut Batch, String> {
    h.as_mut().ok_or_else(|| "null batch handle".to_string())
}

fn ret(r: Result<(), String>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

/// Never unwind across the C boundary: a panic becomes an error return.
fn guard<T>(default: T, f: impl FnOnce() -> T) -> T {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| {
        set_error("panic inside sim-ffi");
        default
    })
}

#[no_mangle]
pub extern "C" fn slap_abi_version() -> u32 {
    ABI_VERSION
}
#[no_mangle]
pub extern "C" fn slap_obs_version() -> u32 {
    OBS_VERSION
}
#[no_mangle]
pub extern "C" fn slap_obs_dim() -> u32 {
    OBS_DIM as u32
}
#[no_mangle]
pub extern "C" fn slap_act_dim() -> u32 {
    ACT_DIM as u32
}
#[no_mangle]
pub extern "C" fn slap_style_dim() -> u32 {
    STYLE_DIM as u32
}
#[no_mangle]
pub extern "C" fn slap_info_dim() -> u32 {
    INFO_DIM as u32
}
#[no_mangle]
pub extern "C" fn slap_reward_terms() -> u32 {
    N_TERMS as u32
}

/// Name of reward term `i` (static NUL-terminated string), or null.
#[no_mangle]
pub extern "C" fn slap_reward_term_name(i: u32) -> *const c_char {
    const NAMES: [&CStr; N_TERMS] =
        [c"goal_for", c"goal_against", c"puck_progress", c"shot_on_net", c"possession", c"defensive_position", c"match_win"];
    NAMES.get(i as usize).map_or(std::ptr::null(), |s| s.as_ptr())
}

/// Last error message on this thread (valid until the next failing call on this thread).
#[no_mangle]
pub extern "C" fn slap_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Create a batch. `params_json`: a params.json text (null/empty = embedded default).
/// `config_json`: env::Config as JSON (null/empty = defaults). Returns null on error.
///
/// # Safety
/// The string pointers must be null or valid NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn slap_batch_new(params_json: *const c_char, config_json: *const c_char, n_envs: u32, seed: u64) -> *mut Batch {
    guard(std::ptr::null_mut(), || {
        let r = (|| new_batch(cstr(params_json)?, cstr(config_json)?, n_envs as usize, seed))();
        match r {
            Ok(b) => Box::into_raw(Box::new(b)),
            Err(e) => {
                set_error(e);
                std::ptr::null_mut()
            }
        }
    })
}

/// # Safety
/// `h` must come from `slap_batch_new` and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn slap_batch_free(h: *mut Batch) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

#[no_mangle]
/// # Safety
/// `h` must be a live batch handle.
pub unsafe extern "C" fn slap_batch_len(h: *mut Batch) -> u32 {
    h.as_ref().map_or(0, |b| b.len() as u32)
}

/// Reset every env. `opp_obs` may be null.
///
/// # Safety
/// Buffers must have the documented sizes.
#[no_mangle]
pub unsafe extern "C" fn slap_batch_reset(h: *mut Batch, obs: *mut f32, opp_obs: *mut f32) -> i32 {
    guard(-1, || {
        ret((|| {
            let b = batch(h)?;
            let n = b.len();
            if obs.is_null() {
                return Err("null obs".into());
            }
            let o = std::slice::from_raw_parts_mut(obs, n * OBS_DIM);
            let oo = (!opp_obs.is_null()).then(|| std::slice::from_raw_parts_mut(opp_obs, n * OBS_DIM));
            b.reset_all(o, oo);
            Ok(())
        })())
    })
}

/// Step every env by one decision (config.action_repeat sim steps). `opp_actions` (used by envs
/// whose opponent is external), `opp_obs` and `final_obs` may be null. `final_obs` rows are only
/// written for envs that finished this step (the observation before the auto-reset).
///
/// # Safety
/// Buffers must have the documented sizes.
#[no_mangle]
pub unsafe extern "C" fn slap_batch_step(
    h: *mut Batch,
    actions: *const f32,
    opp_actions: *const f32,
    obs: *mut f32,
    opp_obs: *mut f32,
    final_obs: *mut f32,
    rewards: *mut f32,
    dones: *mut u8,
    info: *mut f32,
) -> i32 {
    guard(-1, || {
        ret((|| {
            let b = batch(h)?;
            let n = b.len();
            if actions.is_null() || obs.is_null() || rewards.is_null() || dones.is_null() || info.is_null() {
                return Err("null buffer".into());
            }
            let a = std::slice::from_raw_parts(actions, n * ACT_DIM);
            let oa = (!opp_actions.is_null()).then(|| std::slice::from_raw_parts(opp_actions, n * ACT_DIM));
            let o = std::slice::from_raw_parts_mut(obs, n * OBS_DIM);
            let oo = (!opp_obs.is_null()).then(|| std::slice::from_raw_parts_mut(opp_obs, n * OBS_DIM));
            let fo = (!final_obs.is_null()).then(|| std::slice::from_raw_parts_mut(final_obs, n * OBS_DIM));
            let r = std::slice::from_raw_parts_mut(rewards, n);
            let d = std::slice::from_raw_parts_mut(dones, n);
            let inf = std::slice::from_raw_parts_mut(info, n * INFO_DIM);
            b.step(a, oa, o, oo, fo, r, d, inf);
            Ok(())
        })())
    })
}

fn each_env(b: &mut Batch, env: u32, mut f: impl FnMut(&mut Batch, usize)) -> Result<(), String> {
    if env == u32::MAX {
        for i in 0..b.len() {
            f(b, i);
        }
        Ok(())
    } else if (env as usize) < b.len() {
        f(b, env as usize);
        Ok(())
    } else {
        Err(format!("env index {env} out of range"))
    }
}

/// Opponent of env `env` (u32::MAX = all): 0 chaser, 1 defender, 2 cautious, 3 idle, −1 external.
/// Takes effect from the next step (call it right after an env reports done to switch per episode).
///
/// # Safety
/// `h` must be a live batch handle.
#[no_mangle]
pub unsafe extern "C" fn slap_batch_set_opponent(h: *mut Batch, env: u32, kind: i32) -> i32 {
    guard(-1, || {
        ret((|| {
            let b = batch(h)?;
            if kind != -1 && bots::Kind::from_id(kind.max(0) as u32).is_none() || kind < -1 {
                return Err(format!("unknown opponent id {kind}"));
            }
            each_env(b, env, |b, i| b.set_opponent(i, kind))
        })())
    })
}

/// Opponent id by name ("chaser", ..., "external"); −2 if unknown.
///
/// # Safety
/// `name` must be a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn slap_opponent_id(name: *const c_char) -> i32 {
    cstr(name).ok().and_then(|n| opponent_id(n).ok()).unwrap_or(-2)
}

/// Style vector of env `env` (u32::MAX = all), `n` floats (padded with 0 / truncated to STYLE_DIM).
/// Visible in the observation from the next step; with style_mode "fixed" it also re-derives the
/// env's reward weights (style mapping, reward.rs). With "sample" the next reset draws a new one.
///
/// # Safety
/// `style` must point to `n` floats.
#[no_mangle]
pub unsafe extern "C" fn slap_batch_set_style(h: *mut Batch, env: u32, style: *const f32, n: u32) -> i32 {
    guard(-1, || {
        ret((|| {
            let b = batch(h)?;
            let s = if style.is_null() { &[][..] } else { std::slice::from_raw_parts(style, n as usize) };
            each_env(b, env, |b, i| b.set_style(i, s))
        })())
    })
}

/// BASE reward weights of env `env` (u32::MAX = all), `n` floats in TERM_NAMES order (missing = 0).
/// The style mapping applies on top unless style_mode is "off".
///
/// # Safety
/// `w` must point to `n` floats.
#[no_mangle]
pub unsafe extern "C" fn slap_batch_set_reward_weights(h: *mut Batch, env: u32, w: *const f32, n: u32) -> i32 {
    guard(-1, || {
        ret((|| {
            let b = batch(h)?;
            let s = if w.is_null() { &[][..] } else { std::slice::from_raw_parts(w, n as usize) };
            let rw = RewardWeights::from_slice(s);
            each_env(b, env, |b, i| b.set_reward_weights(i, rw))
        })())
    })
}

/// Determinism hash of the whole batch (all states, RNGs, learner slots, episode counters).
///
/// # Safety
/// `h` must be a live batch handle.
#[no_mangle]
pub unsafe extern "C" fn slap_batch_hash(h: *mut Batch) -> u64 {
    h.as_ref().map_or(0, |b| b.hash())
}

/// Timing counters: writes 5 u64 into `out`: wall ns spent inside slap_batch_step (always on), then
/// CPU ns summed over envs for sim / reward / opponent bot / observe+reset (config `profile: true`).
///
/// # Safety
/// `out` must point to 5 writable u64.
#[no_mangle]
pub unsafe extern "C" fn slap_batch_profile(h: *mut Batch, out: *mut u64) -> i32 {
    guard(-1, || {
        ret((|| {
            let b = batch(h)?;
            if out.is_null() {
                return Err("null out".into());
            }
            std::slice::from_raw_parts_mut(out, 5).copy_from_slice(&b.profile());
            Ok(())
        })())
    })
}

/// sim::State::hash of one env.
///
/// # Safety
/// `h` must be a live batch handle.
#[no_mangle]
pub unsafe extern "C" fn slap_batch_env_hash(h: *mut Batch, env: u32) -> u64 {
    h.as_ref().and_then(|b| b.envs.get(env as usize)).map_or(0, |e| e.state.hash())
}

/// Write env `env`'s State as JSON (sim's serde shape) into `buf` (capacity `cap`, NUL-terminated).
/// Returns the JSON length in bytes (excluding NUL); if ≥ cap, nothing is written: retry with a
/// larger buffer. −1 on error.
///
/// # Safety
/// `buf` must point to `cap` writable bytes (or be null with cap 0).
#[no_mangle]
pub unsafe extern "C" fn slap_batch_get_state(h: *mut Batch, env: u32, buf: *mut c_char, cap: u32) -> i32 {
    guard(-1, || {
        let Some(e) = h.as_ref().and_then(|b| b.envs.get(env as usize)) else {
            set_error("bad handle or env index");
            return -1;
        };
        let s = serde_json::to_string(&e.state).expect("state serializes");
        if (s.len() as u64) < cap as u64 && !buf.is_null() {
            std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, s.len());
            *buf.add(s.len()) = 0;
        }
        s.len() as i32
    })
}

/// Replace env `env`'s State from JSON (the episode counters are kept; the learner slot stays).
///
/// # Safety
/// `json` must be a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn slap_batch_set_state(h: *mut Batch, env: u32, json: *const c_char) -> i32 {
    guard(-1, || {
        ret((|| {
            let b = batch(h)?;
            let s: sim::State = serde_json::from_str(cstr(json)?).map_err(|e| format!("state json: {e}"))?;
            if s.players.len() != 2 {
                return Err("state must have exactly 2 players (1v1)".into());
            }
            let e = b.envs.get_mut(env as usize).ok_or("env index out of range")?;
            e.state = s;
            Ok(())
        })())
    })
}
