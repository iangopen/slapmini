"""ctypes + numpy wrapper around the sim-ffi C ABI (no PyO3/maturin).

Build the library first:  cargo build --release -p sim-ffi
It is looked up in $SLAPMINI_FFI_LIB, then target/release/ under the repo root.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS = REPO / "params.json"

# Opponent ids (bots::Kind) and the external-policy slot.
OPPONENTS = {"chaser": 0, "defender": 1, "cautious": 2, "idle": 3, "external": -1}
SCRIPTED = ["chaser", "defender", "cautious", "idle"]
# End reasons (info[:, 0]): first-goal mode = who scored first; match mode = the match result.
END_RUNNING, END_WIN, END_LOSS, END_TRUNCATED = 0, 1, 2, 3
END_GOAL_FOR, END_GOAL_AGAINST, END_LIMIT = END_WIN, END_LOSS, END_TRUNCATED  # old names
# Info columns (sim-ffi env::INFO_DIM): end, episode playing sim steps, return, learner team, goals for,
# goals against, opponent possession seconds, then the raw reward-term sums (names in VecEnv.reward_terms).
I_END, I_STEPS, I_RET, I_TEAM, I_GF, I_GA, I_OPP_POSS, I_RAW = 0, 1, 2, 3, 4, 5, 6, 7
ABI_VERSION = 3
ACT_VERSION = 3  # bots::ACT_VERSION: move, stick angle, dash, lift, stick_up (5 floats)
OBS_VERSION = 3  # bots::OBS_VERSION: 42 floats (32 base, style 32..40, stick_up own 40 / opponent 41)
# Move indices (bots/src/action.rs): 0 stop, then 8 directions counterclockwise from +x (canonical frame).
N_MOVES = 9


def lib_path() -> Path:
    env = os.environ.get("SLAPMINI_FFI_LIB")
    if env:
        return Path(env)
    names = {"win32": ["sim_ffi.dll"], "darwin": ["libsim_ffi.dylib"]}.get(sys.platform, ["libsim_ffi.so"])
    for prof in ("release", "debug"):
        for n in names:
            p = REPO / "target" / prof / n
            if p.exists():
                return p
    raise FileNotFoundError(
        f"sim-ffi library not found under {REPO / 'target'}; run `cargo build --release -p sim-ffi` "
        "or set SLAPMINI_FFI_LIB")


_LIB = None


def lib() -> C.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    L = C.CDLL(str(lib_path()))
    f32p, u8p, vp, cp = C.POINTER(C.c_float), C.POINTER(C.c_uint8), C.c_void_p, C.c_char_p
    sig = {
        "slap_abi_version": ([], C.c_uint32), "slap_obs_version": ([], C.c_uint32),
        "slap_obs_dim": ([], C.c_uint32), "slap_act_dim": ([], C.c_uint32), "slap_act_version": ([], C.c_uint32),
        "slap_style_dim": ([], C.c_uint32), "slap_info_dim": ([], C.c_uint32),
        "slap_reward_terms": ([], C.c_uint32), "slap_reward_term_name": ([C.c_uint32], cp),
        "slap_last_error": ([], cp),
        "slap_batch_new": ([cp, cp, C.c_uint32, C.c_uint64], vp),
        "slap_batch_free": ([vp], None), "slap_batch_len": ([vp], C.c_uint32),
        "slap_batch_reset": ([vp, f32p, f32p], C.c_int32),
        "slap_batch_step": ([vp, f32p, f32p, f32p, f32p, f32p, f32p, u8p, f32p], C.c_int32),
        "slap_batch_set_opponent": ([vp, C.c_uint32, C.c_int32], C.c_int32),
        "slap_batch_set_style": ([vp, C.c_uint32, f32p, C.c_uint32], C.c_int32),
        "slap_batch_set_reward_weights": ([vp, C.c_uint32, f32p, C.c_uint32], C.c_int32),
        "slap_batch_hash": ([vp], C.c_uint64), "slap_batch_env_hash": ([vp, C.c_uint32], C.c_uint64),
        "slap_batch_get_state": ([vp, C.c_uint32, C.c_char_p, C.c_uint32], C.c_int32),
        "slap_batch_set_state": ([vp, C.c_uint32, cp], C.c_int32),
        "slap_batch_profile": ([vp, C.POINTER(C.c_uint64)], C.c_int32),
    }
    for name, (args, res) in sig.items():
        fn = getattr(L, name)
        fn.argtypes = args
        fn.restype = res
    if L.slap_abi_version() != ABI_VERSION:
        raise RuntimeError(f"{lib_path()}: ABI v{L.slap_abi_version()}, this wrapper needs v{ABI_VERSION}; "
                           "rebuild with cargo build --release -p sim-ffi")
    _LIB = L
    return L


def _ptr(a: np.ndarray | None, ctype=C.c_float):
    if a is None:
        return None
    assert a.flags["C_CONTIGUOUS"]
    return a.ctypes.data_as(C.POINTER(ctype))


def dims() -> dict:
    L = lib()
    return {
        "abi": L.slap_abi_version(), "obs_version": L.slap_obs_version(), "act_version": L.slap_act_version(), "obs": L.slap_obs_dim(),
        "act": L.slap_act_dim(), "style": L.slap_style_dim(), "info": L.slap_info_dim(),
        "reward_terms": [L.slap_reward_term_name(i).decode() for i in range(L.slap_reward_terms())],
    }


def params_values(path: Path | str = DEFAULT_PARAMS) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return {k: v["value"] for k, v in json.load(f)["params"].items()}


def physics_stamp(path: Path | str = DEFAULT_PARAMS) -> str:
    """Hash of the physics: every param name and value in params.json (provenance/notes/whitespace
    and line endings do not matter). 16 hex chars."""
    vals = params_values(path)
    canon = json.dumps({k: float(vals[k]) for k in sorted(vals)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


class VecEnv:
    """N 1v1 environments stepped together in Rust (rayon). Finished envs auto-reset.

    step(actions[N, 4], opp_actions[N, 4] | None) -> obs, rewards, dones(bool), info
    Actions: move index 0..8, stick angle (rad, canonical), dash 0/1, lift 0/1, stick_up 0/1 (ACT v3, bots/src/action.rs).
    info columns: see I_* above (end reason: 0 running, 1 win, 2 loss, 3 truncated).
    After step, .final_obs[i] holds the learner's observation of the last state of an episode that
    ended this step (before the auto-reset), for bootstrapping truncated episodes.
    The returned arrays are views of internal buffers, overwritten by the next step: copy what you keep.
    config: see sim-ffi/src/env.rs Config (mode "first_goal" | "match", style_mode, ...).
    """

    def __init__(self, n: int, seed: int = 0, config: dict | None = None,
                 params_path: Path | str = DEFAULT_PARAMS):
        self.L = lib()
        d = dims()
        self.obs_dim, self.act_dim, self.style_dim, self.info_dim = d["obs"], d["act"], d["style"], d["info"]
        self.reward_terms = d["reward_terms"]
        self.n = n
        self.config = dict(config or {})
        params_json = Path(params_path).read_text(encoding="utf-8")
        h = self.L.slap_batch_new(params_json.encode(), json.dumps(self.config).encode(), n, seed)
        if not h:
            raise RuntimeError(self.L.slap_last_error().decode())
        self.h = h
        self.obs = np.zeros((n, self.obs_dim), np.float32)
        self.opp_obs = np.zeros((n, self.obs_dim), np.float32)
        self.final_obs = np.zeros((n, self.obs_dim), np.float32)
        self.rew = np.zeros(n, np.float32)
        self.done = np.zeros(n, np.uint8)
        self.info = np.zeros((n, self.info_dim), np.float32)
        self._zero_opp = np.zeros((n, self.act_dim), np.float32)

    def _check(self, rc: int):
        if rc != 0:
            raise RuntimeError(self.L.slap_last_error().decode())

    def reset(self) -> np.ndarray:
        self._check(self.L.slap_batch_reset(self.h, _ptr(self.obs), _ptr(self.opp_obs)))
        return self.obs.copy()

    def step(self, actions: np.ndarray, opp_actions: np.ndarray | None = None):
        a = np.ascontiguousarray(actions, np.float32).reshape(self.n, self.act_dim)
        oa = self._zero_opp if opp_actions is None else np.ascontiguousarray(opp_actions, np.float32).reshape(self.n, self.act_dim)
        self._check(self.L.slap_batch_step(self.h, _ptr(a), _ptr(oa), _ptr(self.obs), _ptr(self.opp_obs),
                                           _ptr(self.final_obs), _ptr(self.rew), _ptr(self.done, C.c_uint8),
                                           _ptr(self.info)))
        return self.obs, self.rew, self.done.view(bool), self.info

    def set_opponent(self, env: int | None, kind: str | int):
        k = OPPONENTS[kind] if isinstance(kind, str) else int(kind)
        self._check(self.L.slap_batch_set_opponent(self.h, 0xFFFFFFFF if env is None else env, k))

    def set_style(self, env: int | None, style):
        s = np.ascontiguousarray(style, np.float32)
        self._check(self.L.slap_batch_set_style(self.h, 0xFFFFFFFF if env is None else env, _ptr(s), s.size))

    def set_reward_weights(self, env: int | None, weights):
        """weights: dict by term name, or a sequence in reward_terms order."""
        if isinstance(weights, dict):
            unknown = set(weights) - set(self.reward_terms)
            if unknown:
                raise KeyError(f"unknown reward terms {unknown}")
            weights = [weights.get(t, 0.0) for t in self.reward_terms]
        w = np.ascontiguousarray(weights, np.float32)
        self._check(self.L.slap_batch_set_reward_weights(self.h, 0xFFFFFFFF if env is None else env, _ptr(w), w.size))

    def profile(self) -> list[int]:
        """[wall ns inside step (cumulative), then CPU ns summed over envs: sim, reward, opponent bot,
        observe+reset (filled only with config profile: true)]."""
        out = (C.c_uint64 * 5)()
        self._check(self.L.slap_batch_profile(self.h, out))
        return list(out)

    def step_ns(self) -> int:
        return self.profile()[0]

    def hash(self) -> int:
        return int(self.L.slap_batch_hash(self.h))

    def env_hash(self, env: int) -> int:
        return int(self.L.slap_batch_env_hash(self.h, env))

    def get_state(self, env: int) -> dict:
        need = self.L.slap_batch_get_state(self.h, env, None, 0)
        if need < 0:
            raise RuntimeError(self.L.slap_last_error().decode())
        buf = C.create_string_buffer(need + 1)
        self.L.slap_batch_get_state(self.h, env, buf, need + 1)
        return json.loads(buf.value.decode())

    def set_state(self, env: int, state: dict):
        self._check(self.L.slap_batch_set_state(self.h, env, json.dumps(state).encode()))

    def close(self):
        if getattr(self, "h", None):
            self.L.slap_batch_free(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
