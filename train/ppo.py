"""Single-file PPO for slapmini 1v1, with scripted opponents and a self-play snapshot pool.

    python -m train.ppo train --run runs/smoke --minutes 5 --opponents chaser
    python -m train.ppo train --run runs/m1 --mode match --curriculum-minutes 30 --styles --obs-norm
    python -m train.ppo eval  --ckpt runs/smoke/latest.pt --opponent chaser --episodes 200
    python -m train.ppo eval  --policy random --opponent chaser
    python -m train.ppo plot  --run runs/smoke

Policy (canonical frame, see bots/src/obs.rs and bots/src/action.rs):
  obs[OBS_DIM] -> (optional running normalization) -> MLP -> move (Categorical over 9: stop + 8
  WASD directions), stick (2D Gaussian direction; the env gets atan2 of it as the stick target
  angle: no wrap-around at +-pi), dash (Bernoulli), lift (Bernoulli), stick_up (Bernoulli; ACT v3), brake (Bernoulli; ACT v4).
Episodes: --mode first_goal (ends at the first goal; the curriculum) or match (full 5:00 matches
with the sim's rules); --curriculum-minutes M trains first_goal for M minutes, then match.
Value targets: a TRUNCATED episode (step limit) bootstraps from V(final observation); a real end
(goal in first-goal mode, match result in match mode) does not.
Rollouts act with a copy of the policy on --rollout-device (default cpu: small batches run faster
there than on a GPU); the PPO update runs on --device (cuda if available).
Checkpoints carry a physics stamp (hash of params.json values) and the observation layout version;
resuming under different physics fails unless --allow-physics-change.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train.slapenv import (ACT_VERSION, DEFAULT_PARAMS, END_LOSS, END_TRUNCATED, END_WIN, I_END, I_GA, I_GF, I_OPP_POSS, I_RAW,
                           I_RET, I_STEPS, N_MOVES, OPPONENTS, SCRIPTED, VecEnv, dims, params_values, physics_stamp)

RAW_DIM = 7      # sampled action: move index, stick_x, stick_y, dash, lift, stick_up, brake
ACT_DIM = 6      # env action (bots::ACT_DIM): move index, stick angle, dash, lift, stick_up, brake
_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
BASE_DIM = 32    # observation fields before the style slot (bots::BASE_DIM); only these are normalized
                 # (the style slot 32..40 and the OBS v3 stick_up flags 40, 41 pass through unchanged)


# ----------------------------------------------------------------------------------------------
# Policy

def mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.Tanh()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class RunningNorm(nn.Module):
    """Running mean/variance of the first `n` observation fields (Chan et al. parallel update);
    forward() returns (x - mean) / sqrt(var + eps) clipped to +-clip for those fields and leaves the
    rest (the style slot) untouched. The statistics are buffers: saved in checkpoints and baked into
    the ONNX graph. They change only through update()."""

    def __init__(self, dim: int, n: int, clip: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.n, self.clip, self.eps = n, clip, eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])[:, : self.n].double()
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        m, v, c = self.mean[: self.n].double(), self.var[: self.n].double(), self.count
        tot = c + bc
        d = bm - m
        new_m = m + d * bc / tot
        m2 = v * c + bv * bc + d * d * c * bc / tot
        self.mean[: self.n] = new_m.float()
        self.var[: self.n] = (m2 / tot).float()
        self.count.fill_(float(tot))

    def forward(self, x):
        z = ((x[:, : self.n] - self.mean[: self.n]) / torch.sqrt(self.var[: self.n] + self.eps)).clamp(-self.clip, self.clip)
        return torch.cat([z, x[:, self.n:]], dim=1)


class Policy(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, obs_norm: bool = False):
        super().__init__()
        self.obs_dim, self.hidden, self.obs_norm = obs_dim, hidden, obs_norm
        self.norm = RunningNorm(obs_dim, min(BASE_DIM, obs_dim)) if obs_norm else None
        self.pi = mlp(obs_dim, hidden, N_MOVES + 6)   # move logits (9), stick mean (2), dash, lift, stick_up, brake logits
        self.vf = mlp(obs_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.5, -0.5]))
        with torch.no_grad():  # small initial outputs: near-uniform start
            self.pi[-1].weight.mul_(0.01)
            self.pi[-1].bias.zero_()
            self.vf[-1].weight.mul_(0.1)

    def prep(self, obs):
        return self.norm(obs) if self.norm is not None else obs

    def heads(self, obs):
        out = self.pi(self.prep(obs))
        return out[:, :N_MOVES], out[:, N_MOVES:N_MOVES + 2], out[:, N_MOVES + 2], out[:, N_MOVES + 3], out[:, N_MOVES + 4], out[:, N_MOVES + 5]

    def dist(self, obs):
        """The action distribution as torch.distributions objects (reference / tests; the hot paths
        below compute the same log-probabilities and entropies by hand, ~3x faster per call)."""
        ml, sm, dl, ll, ul, bl = self.heads(obs)
        std = self.log_std.clamp(-3.0, 1.0).exp().expand_as(sm)
        return (torch.distributions.Categorical(logits=ml), torch.distributions.Normal(sm, std),
                torch.distributions.Bernoulli(logits=dl), torch.distributions.Bernoulli(logits=ll),
                torch.distributions.Bernoulli(logits=ul), torch.distributions.Bernoulli(logits=bl))

    def value(self, obs):
        return self.vf(self.prep(obs)).squeeze(-1)

    def _forward(self, obs):
        o = self.prep(obs)
        out = self.pi(o)
        return (out[:, :N_MOVES], out[:, N_MOVES:N_MOVES + 2], out[:, N_MOVES + 2], out[:, N_MOVES + 3],
                out[:, N_MOVES + 4], out[:, N_MOVES + 5], self.vf(o).squeeze(-1))

    def _logp_ent(self, ml, sm, dl, ll, ul, bl, raw, want_ent: bool):
        lsm = torch.log_softmax(ml, -1)
        log_std = self.log_std.clamp(-3.0, 1.0)
        z = (raw[:, 1:3] - sm) / log_std.exp()
        lp_stick = (-0.5 * z * z - log_std - _HALF_LOG_2PI).sum(-1)
        # Bernoulli(logits l): log p(x) = x*l - softplus(l)
        lp = (lsm.gather(1, raw[:, 0:1].long()).squeeze(1) + lp_stick
              + raw[:, 3] * dl - nn.functional.softplus(dl) + raw[:, 4] * ll - nn.functional.softplus(ll)
              + raw[:, 5] * ul - nn.functional.softplus(ul) + raw[:, 6] * bl - nn.functional.softplus(bl))
        if not want_ent:
            return lp, None
        ent_move = -(lsm.exp() * lsm).sum(-1)
        ent_stick = (0.5 + _HALF_LOG_2PI + log_std).sum()
        ent_b = lambda l: nn.functional.softplus(l) - l * torch.sigmoid(l)
        return lp, ent_move + ent_stick + ent_b(dl) + ent_b(ll) + ent_b(ul) + ent_b(bl)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Returns (raw [N, 7], logp [N], value [N])."""
        ml, sm, dl, ll, ul, bl, v = self._forward(obs)
        if deterministic:
            m = ml.argmax(-1).float()
            st = sm
            dash, lift, up, brk = (dl > 0).float(), (ll > 0).float(), (ul > 0).float(), (bl > 0).float()
        else:
            m = torch.multinomial(torch.softmax(ml, -1), 1).squeeze(1).float()
            st = sm + self.log_std.clamp(-3.0, 1.0).exp() * torch.randn_like(sm)
            u = torch.rand((obs.shape[0], 4), device=obs.device)
            dash, lift = (u[:, 0] < torch.sigmoid(dl)).float(), (u[:, 1] < torch.sigmoid(ll)).float()
            up, brk = (u[:, 2] < torch.sigmoid(ul)).float(), (u[:, 3] < torch.sigmoid(bl)).float()
        raw = torch.cat([m[:, None], st, dash[:, None], lift[:, None], up[:, None], brk[:, None]], dim=1)
        return raw, self._logp_ent(ml, sm, dl, ll, ul, bl, raw, False)[0], v

    def evaluate(self, obs, raw):
        ml, sm, dl, ll, ul, bl, v = self._forward(obs)
        lp, ent = self._logp_ent(ml, sm, dl, ll, ul, bl, raw, True)
        return lp, ent, v


def to_env_actions(raw: np.ndarray) -> np.ndarray:
    """[N, 7] sampled -> [N, 6] env format (move index, stick angle, dash, lift, stick_up, brake)."""
    a = np.empty((raw.shape[0], ACT_DIM), np.float32)
    a[:, 0] = raw[:, 0]
    a[:, 1] = np.arctan2(raw[:, 2], raw[:, 1])
    a[:, 2] = raw[:, 3]
    a[:, 3] = raw[:, 4]
    a[:, 4] = raw[:, 5]
    a[:, 5] = raw[:, 6]
    return a


def random_actions(n: int, rng: np.random.Generator) -> np.ndarray:
    a = np.empty((n, ACT_DIM), np.float32)
    a[:, 0] = rng.integers(0, N_MOVES, n)
    a[:, 1] = rng.uniform(-math.pi, math.pi, n)
    a[:, 2:6] = (rng.uniform(0, 1, (n, 4)) < 0.5)
    return a


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ----------------------------------------------------------------------------------------------
# Checkpoints

def save_ckpt(path: Path, policy: Policy, opt, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"policy": policy.state_dict(), "optimizer": opt.state_dict() if opt else None,
                "hidden": policy.hidden, "obs_dim": policy.obs_dim, "obs_norm": policy.obs_norm,
                "act_version": ACT_VERSION, **state}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, params_path=DEFAULT_PARAMS, allow_physics_change=False, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    d = dims()
    if ck.get("obs_version") != d["obs_version"] or ck.get("obs_dim") != d["obs"]:
        raise SystemExit(f"{path}: observation layout v{ck.get('obs_version')}/{ck.get('obs_dim')} does not match "
                         f"the library's v{d['obs_version']}/{d['obs']}; cannot load")
    if ck.get("act_version", 1) != ACT_VERSION:
        raise SystemExit(f"{path}: action format ACT v{ck.get('act_version', 1)}, this trainer uses ACT v{ACT_VERSION} "
                         f"(6 floats, + brake); old checkpoints cannot be loaded: train a new run")
    stamp = physics_stamp(params_path)
    if ck.get("physics_stamp") != stamp:
        msg = (f"{path}: trained under physics {ck.get('physics_stamp')}, current params give {stamp}")
        if not allow_physics_change:
            raise SystemExit(msg + " (pass --allow-physics-change to continue anyway)")
        print("WARNING: " + msg, file=sys.stderr)
    pol = Policy(ck["obs_dim"], ck["hidden"], ck.get("obs_norm", False)).to(device)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    return pol, ck


def policy_from_snapshot(s: dict, device) -> Policy:
    m = Policy(s["obs_dim"], s["hidden"], s.get("obs_norm", False)).to(device)
    m.load_state_dict(s["state"])
    m.eval()
    return m


# ----------------------------------------------------------------------------------------------
# Episode statistics (shared with train.league)

def episode_row(info_row: np.ndarray, terms: list[str], timestep: float) -> dict:
    """One finished episode from its info row."""
    end = int(info_row[I_END])
    raw = {t: float(info_row[I_RAW + k]) for k, t in enumerate(terms)}
    secs = float(info_row[I_STEPS]) * timestep
    return {"result": {END_WIN: 1, END_LOSS: -1}.get(end, 0), "truncated": end == END_TRUNCATED,
            "gf": float(info_row[I_GF]), "ga": float(info_row[I_GA]), "seconds": secs,
            "return": float(info_row[I_RET]), "shots": raw.get("shot_on_net", 0.0),
            "possession_s": raw.get("possession", 0.0), "opp_possession_s": float(info_row[I_OPP_POSS])}


def summarize(rows: list[dict]) -> dict:
    """Win/loss/draw rates, goals and per-episode style statistics over finished episodes."""
    if not rows:
        return {"episodes": 0}
    r = np.array([x["result"] for x in rows])
    gf, ga = np.array([x["gf"] for x in rows]), np.array([x["ga"] for x in rows])
    pos, opp = sum(x["possession_s"] for x in rows), sum(x["opp_possession_s"] for x in rows)
    return {"episodes": len(rows), "win": float((r == 1).mean()), "loss": float((r == -1).mean()),
            "draw": float((r == 0).mean()), "truncated": float(np.mean([x["truncated"] for x in rows])),
            "goals_for": float(gf.mean()), "goals_against": float(ga.mean()), "goal_diff": float((gf - ga).mean()),
            "shots_per_ep": float(np.mean([x["shots"] for x in rows])),
            "possession_share": float(pos / (pos + opp)) if pos + opp > 0 else float("nan"),
            "ep_seconds": float(np.mean([x["seconds"] for x in rows]))}


# ----------------------------------------------------------------------------------------------
# Opponents: scripted bots in Rust, or frozen snapshots run here.

class OpponentMixer:
    """Assigns an opponent to each env at every episode start. Mix: {"chaser": w, ..., "pool": w}.
    Pool snapshots are frozen copies of the policy; at most `active` of them are loaded at a time."""

    def __init__(self, env: VecEnv, mix: dict, device, rng: random.Random, pool_dir: Path | None,
                 pool_max=20, active=4):
        self.env, self.device, self.rng = env, device, rng
        self.mix = {k: float(v) for k, v in mix.items() if float(v) > 0}
        for k in self.mix:
            if k != "pool" and k not in SCRIPTED:
                raise SystemExit(f"unknown opponent '{k}' (use {SCRIPTED + ['pool']})")
        self.pool: list[dict] = []          # state_dicts
        self.pool_dir, self.pool_max, self.n_active = pool_dir, pool_max, active
        self.active: list[Policy] = []
        self.slot = np.full(env.n, -1)      # per env: active pool index, or -1 for scripted
        self.kind = [""] * env.n
        if pool_dir and pool_dir.exists():
            for p in sorted(pool_dir.glob("snap_*.pt")):
                s = torch.load(p, map_location="cpu", weights_only=False)
                if s.get("act_version", 1) == ACT_VERSION and s.get("obs_dim") == env.obs_dim:
                    self.pool.append(s)
            self.pool = self.pool[-pool_max:]

    def rebind(self, env: VecEnv):
        """Use a new VecEnv (curriculum switch); every env gets a fresh assignment."""
        self.env = env
        self.slot = np.full(env.n, -1)
        for i in range(env.n):
            self.assign(i)

    def add_snapshot(self, policy: Policy, tag: int):
        sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        self.pool.append({"state": sd, "hidden": policy.hidden, "obs_dim": policy.obs_dim,
                          "obs_norm": policy.obs_norm, "act_version": ACT_VERSION, "tag": tag})
        self.pool = self.pool[-self.pool_max:]
        if self.pool_dir:
            self.pool_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.pool[-1], self.pool_dir / f"snap_{tag:07d}.pt")
            for old in sorted(self.pool_dir.glob("snap_*.pt"))[:-self.pool_max]:
                old.unlink()
        self.refresh_active()

    def refresh_active(self):
        """Load up to n_active snapshots (always the newest, the rest sampled)."""
        if not self.pool:
            self.active = []
            return
        idx = [len(self.pool) - 1] + self.rng.sample(range(len(self.pool) - 1), min(self.n_active - 1, len(self.pool) - 1))
        self.active = [policy_from_snapshot(self.pool[i], self.device) for i in idx]
        # Envs pointing past the new active list fall back to a valid slot.
        self.slot = np.where(self.slot >= len(self.active), 0, self.slot)

    def assign(self, i: int):
        names, w = zip(*self.mix.items())
        k = self.rng.choices(names, weights=w)[0]
        if k == "pool" and not self.active:
            k = next((s for s in names if s != "pool"), "chaser")
        if k == "pool":
            self.slot[i] = self.rng.randrange(len(self.active))
            self.env.set_opponent(i, "external")
        else:
            self.slot[i] = -1
            self.env.set_opponent(i, k)
        self.kind[i] = k

    def actions(self, opp_obs: np.ndarray) -> np.ndarray | None:
        if not (self.slot >= 0).any():
            return None
        out = np.zeros((self.env.n, ACT_DIM), np.float32)
        for s, m in enumerate(self.active):
            idx = np.nonzero(self.slot == s)[0]
            if len(idx):
                raw, _, _ = m.act(torch.as_tensor(opp_obs[idx], device=self.device))
                out[idx] = to_env_actions(raw.cpu().numpy())
        return out


# ----------------------------------------------------------------------------------------------
# Profiling

class Prof:
    """Wall-time accumulator per section. With CUDA in use, work is synchronized at every section
    boundary so GPU time lands in the section that issued it (profiling runs only)."""

    def __init__(self, enabled: bool, cuda: bool):
        self.on, self.cuda = enabled, cuda and enabled
        self.t = defaultdict(float)
        self.last = time.perf_counter()

    def mark(self, name: str | None = None):
        if not self.on:
            return
        if self.cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        if name:
            self.t[name] += now - self.last
        self.last = now

    def add(self, name: str, secs: float):
        if self.on:
            self.t[name] += secs


PROFILE_ROWS = [
    ("env_rust", "env step inside Rust (sim + opponent bot + reward + observe, rayon)"),
    ("env_ctypes", "ctypes call + Python wrapper around env.step"),
    ("act_h2d", "observation -> rollout device"),
    ("act_forward", "policy forward + sampling"),
    ("act_d2h", "actions -> host, to_env_actions"),
    ("opp_policy", "self-play opponent inference"),
    ("bookkeeping", "rollout buffers, done handling, truncation bootstrap"),
    ("gae", "returns and advantages"),
    ("update_h2d", "rollout buffers -> training device"),
    ("update", "PPO epochs (forward, backward, Adam)"),
    ("other", "obs-norm update, weight sync, logging, snapshots, checkpoints"),
]


def profile_table(prof: Prof, decisions: int, rust_parts: list[int] | None) -> str:
    tot = sum(prof.t.values())
    lines = ["| section | what | seconds | share | us per decision |", "|---|---|---|---|---|"]
    for k, what in PROFILE_ROWS:
        s = prof.t.get(k, 0.0)
        lines.append(f"| {k} | {what} | {s:.2f} | {100 * s / max(tot, 1e-9):.1f}% | {1e6 * s / max(decisions, 1):.2f} |")
    lines.append(f"| total | | {tot:.2f} | 100% | {1e6 * tot / max(decisions, 1):.2f} |")
    lines.append(f"\n{decisions} decisions, {decisions / max(tot, 1e-9):.0f} decisions/s (profiled; CUDA synced per section)")
    if rust_parts and sum(rust_parts) > 0:
        names = ["sim", "reward", "opponent bot", "observe+reset"]
        s = sum(rust_parts)
        lines.append("Rust CPU time split (summed over envs and threads): " + ", ".join(
            f"{n} {100 * v / s:.1f}%" for n, v in zip(names, rust_parts)))
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Training

def parse_mix(s: str) -> dict:
    out = {}
    for part in s.split(","):
        k, _, w = part.partition(":")
        out[k.strip()] = float(w) if w else 1.0
    return out


def make_env_config(args, mode: str) -> dict:
    cfg = {"mode": mode, "action_repeat": args.action_repeat, "threads": args.threads, "chunk": args.chunk,
           "jitter_puck": args.jitter_puck, "jitter_player": args.jitter_player,
           "style_mode": "sample" if args.styles else "off", "profile": bool(args.profile)}
    if mode == "first_goal":
        cfg["max_steps"] = int(round(args.episode_seconds / params_values(args.params)["timestep"]))
    else:
        cfg["overtime_cap_s"] = args.overtime_cap
    if args.reward:
        cfg["reward"] = json.loads(Path(args.reward).read_text()) if Path(args.reward).exists() else json.loads(args.reward)
    return cfg


def ckpt_state(args, update, global_steps, t0, t_offset, stamp, d, env_cfg):
    return {"update": update, "global_steps": global_steps, "wall_seconds": time.time() - t0 + t_offset,
            "physics_stamp": stamp, "obs_version": d["obs_version"], "args": vars(args), "env_config": env_cfg}


def train(args):
    device = pick_device(args.device)
    rdev = device if args.rollout_device == "same" else pick_device(args.rollout_device)
    # Measured on a 12-core machine (CLAUDE.md "Training throughput"): with 64 envs a rayon step is
    # ~0.2 ms and parallelism costs more than it saves; idle rayon workers spinning also slow the
    # torch threads that act next. 4 torch threads are the sweet spot for a 256-wide MLP on CPU.
    if args.threads < 0:
        args.threads = 1 if args.envs <= 128 else min(8, os.cpu_count() or 1)
    if args.torch_threads < 0:
        args.torch_threads = 4 if rdev.type == "cpu" else 0
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    stamp = physics_stamp(args.params)
    timestep = params_values(args.params)["timestep"]
    d = dims()

    policy = Policy(d["obs"], args.hidden, args.obs_norm).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    update, global_steps, t_offset = 0, 0, 0.0
    if args.resume:
        policy, ck = load_ckpt(args.resume, args.params, args.allow_physics_change, device)
        if bool(ck.get("obs_norm", False)) != bool(args.obs_norm):
            print(f"note: observation normalization follows the checkpoint ({ck.get('obs_norm', False)})")
            args.obs_norm = ck.get("obs_norm", False)
        opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
        if ck.get("optimizer") and ck.get("physics_stamp") == stamp:
            opt.load_state_dict(ck["optimizer"])
        update, global_steps, t_offset = ck.get("update", 0), ck.get("global_steps", 0), ck.get("wall_seconds", 0.0)
        print(f"resumed {args.resume} at update {update}, {global_steps} decisions")
    actor = policy if rdev == device else copy.deepcopy(policy).to(rdev)
    policy.eval()
    actor.eval()

    def mode_at(minutes):
        if args.curriculum_minutes:
            return "first_goal" if minutes < args.curriculum_minutes else "match"
        return args.mode

    mode = mode_at(t_offset / 60)
    env_cfg = make_env_config(args, mode)
    env = VecEnv(args.envs, seed=args.seed + update, config=env_cfg, params_path=args.params)
    mixer = OpponentMixer(env, parse_mix(args.opponents), rdev, rng,
                          run / "pool" if "pool" in args.opponents else None, args.pool_max)
    if "pool" in mixer.mix and not mixer.pool:
        mixer.add_snapshot(policy, update)
    mixer.refresh_active()
    for i in range(env.n):
        mixer.assign(i)

    cur = f" (curriculum: first_goal for {args.curriculum_minutes} min, then match)" if args.curriculum_minutes else ""
    print(f"device {device} (rollouts on {rdev}), {env.n} envs, mode {mode}{cur}, "
          f"obs {d['obs']}{' + running norm' if args.obs_norm else ''}, styles {'sampled' if args.styles else 'off'}, "
          f"physics {stamp}, opponents {mixer.mix}", flush=True)
    obs = env.reset().copy()
    opp_obs = env.opp_obs.copy()
    N, T = env.n, args.steps
    b_obs = torch.zeros((T, N, d["obs"]), device=rdev)
    b_raw = torch.zeros((T, N, RAW_DIM), device=rdev)
    b_logp = torch.zeros((T, N), device=rdev)
    b_val = torch.zeros((T, N), device=rdev)
    b_rew = torch.zeros((T, N), device=rdev)
    b_done = torch.zeros((T, N), device=rdev)

    window = deque(maxlen=args.window)   # finished-episode rows (episode_row + opponent kind)
    log_path = run / "metrics.csv"
    new_log = not log_path.exists() or not args.resume
    logf = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["update", "decisions", "minutes", "episodes", "win", "loss", "draw", "goal_diff",
                      "ep_seconds", "ep_return", "pi_loss", "v_loss", "entropy", "kl", "sps", "mode", "truncated",
                      "goals_for", "goals_against", "shots_per_ep", "possession_share", "stick_up_prob", "stick_up_frac", "brake_prob", "brake_frac"])
    prof = Prof(args.profile, device.type == "cuda" or rdev.type == "cuda")
    rust0 = env.profile()
    prof_decisions = 0
    t0 = time.time()
    total_eps = 0
    gamma = args.gamma
    # Ctrl+C finishes the current update, saves latest.pt and exits cleanly (a second Ctrl+C aborts).
    stop = {"flag": False}

    def _sigint(sig, frame):
        if stop["flag"]:
            raise KeyboardInterrupt
        stop["flag"] = True
        print("\nCtrl+C: finishing this update and saving latest.pt (Ctrl+C again aborts without saving)", flush=True)

    old_handler = signal.signal(signal.SIGINT, _sigint) if threading.current_thread() is threading.main_thread() else None
    while True:
        minutes = (time.time() - t0 + t_offset) / 60
        if stop["flag"]:
            break
        if args.minutes and minutes >= args.minutes:
            break
        if args.updates and update >= args.updates:
            break
        if mode_at(minutes) != mode:  # curriculum switch: a new env in match mode
            mode = mode_at(minutes)
            env.close()
            env_cfg = make_env_config(args, mode)
            env = VecEnv(args.envs, seed=args.seed + update, config=env_cfg, params_path=args.params)
            mixer.rebind(env)
            obs, opp_obs = env.reset().copy(), env.opp_obs.copy()
            rust0 = env.profile()
            window.clear()
            print(f"curriculum: switched to {mode} mode at {minutes:.1f} min (update {update})", flush=True)
        # LR annealing over the time budget when one is given.
        if args.minutes and args.anneal:
            for g in opt.param_groups:
                g["lr"] = args.lr * max(0.05, 1 - minutes / args.minutes)

        # ---- rollout
        ts = time.time()
        prof.mark()
        for t in range(T):
            o = torch.as_tensor(obs, device=rdev)
            prof.mark("act_h2d")
            raw, logp, val = actor.act(o)
            prof.mark("act_forward")
            acts = to_env_actions(raw.cpu().numpy())
            prof.mark("act_d2h")
            opp_a = mixer.actions(opp_obs)
            prof.mark("opp_policy")
            r0 = env.step_ns()
            t_env = time.perf_counter()
            nobs, rew, done, info = env.step(acts, opp_a)
            t_env = time.perf_counter() - t_env
            r_ns = (env.step_ns() - r0) * 1e-9
            prof.add("env_rust", r_ns)
            prof.add("env_ctypes", t_env - r_ns)
            prof.last = time.perf_counter()
            b_obs[t], b_raw[t], b_logp[t], b_val[t] = o, raw, logp, val
            rew_t = torch.as_tensor(rew, device=rdev).clone()
            if done.any():
                idx = np.nonzero(done)[0]
                trunc = idx[info[idx, I_END] == END_TRUNCATED]
                if len(trunc) and not args.no_bootstrap:
                    # Truncated, not terminal: add gamma * V(final state) and end the trace there.
                    with torch.no_grad():
                        v = actor.value(torch.as_tensor(env.final_obs[trunc], device=rdev))
                    rew_t[torch.as_tensor(trunc, device=rdev)] += gamma * v
                for i in idx:
                    row = episode_row(info[i], env.reward_terms, timestep)
                    row["opponent"] = mixer.kind[i]
                    window.append(row)
                    total_eps += 1
                    mixer.assign(i)
            b_rew[t] = rew_t
            b_done[t] = torch.as_tensor(done, device=rdev, dtype=torch.float32)
            obs = nobs.copy()
            opp_obs = env.opp_obs.copy()
            prof.mark("bookkeeping")
        global_steps += N * T
        prof_decisions += N * T
        sps = N * T / (time.time() - ts)

        # ---- GAE (dones end the trace; truncated ends already carry their bootstrap in the reward)
        with torch.no_grad():
            next_val = actor.value(torch.as_tensor(obs, device=rdev))
            adv = torch.zeros_like(b_rew)
            last = torch.zeros(N, device=rdev)
            for t in reversed(range(T)):
                nv = next_val if t == T - 1 else b_val[t + 1]
                nonterm = 1.0 - b_done[t]
                delta = b_rew[t] + gamma * nv * nonterm - b_val[t]
                last = delta + gamma * args.lam * nonterm * last
                adv[t] = last
            ret = adv + b_val
        prof.mark("gae")

        # ---- PPO update
        policy.train()
        g_obs = b_obs.reshape(-1, d["obs"]).to(device)
        g_raw = b_raw.reshape(-1, RAW_DIM).to(device)
        g_logp, g_adv, g_ret = (x.reshape(-1).to(device) for x in (b_logp, adv, ret))
        prof.mark("update_h2d")
        n_b = g_obs.shape[0]
        mb = max(1, n_b // args.minibatches)
        stats = []
        for _ in range(args.epochs):
            perm = torch.randperm(n_b, device=device)
            for s in range(0, n_b, mb):
                j = perm[s:s + mb]
                logp, ent, v = policy.evaluate(g_obs[j], g_raw[j])
                ratio = (logp - g_logp[j]).exp()
                a = g_adv[j]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pl = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                vl = 0.5 * (v - g_ret[j]).pow(2).mean()
                loss = pl + args.vf_coef * vl - args.ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()
                # Keep the stats on the device (one sync per update instead of four per minibatch).
                stats.append(torch.stack([pl.detach(), vl.detach(), ent.mean().detach(), (g_logp[j] - logp).mean().detach()]))
        pl_, vl_, en_, kl_ = torch.stack(stats).mean(0).tolist()
        # Raised-stick and brake usage this rollout: mean probability and the fraction of decisions sampled on.
        with torch.no_grad():
            hd = policy.heads(g_obs)
            up_p, br_p = float(torch.sigmoid(hd[4]).mean()), float(torch.sigmoid(hd[5]).mean())
        up_f, br_f = float(g_raw[:, 5].mean()), float(g_raw[:, 6].mean())
        update += 1
        policy.eval()
        prof.mark("update")
        # Observation statistics move only between updates (a rollout and its update use the same ones).
        if policy.norm is not None:
            policy.norm.update(g_obs)
        if actor is not policy:
            actor.load_state_dict(policy.state_dict())

        # ---- self-play snapshots
        if "pool" in mixer.mix and update % args.snapshot_every == 0:
            mixer.add_snapshot(policy, update)

        # ---- log
        s = summarize(list(window))
        if s["episodes"]:
            win, loss_, draw, gd, eps_s = s["win"], s["loss"], s["draw"], s["goal_diff"], s["ep_seconds"]
            er = float(np.mean([w["return"] for w in window]))
        else:
            win = loss_ = draw = gd = eps_s = er = float("nan")
        minutes = (time.time() - t0 + t_offset) / 60
        nan = float("nan")
        log.writerow([update, global_steps, f"{minutes:.3f}", total_eps, f"{win:.4f}", f"{loss_:.4f}", f"{draw:.4f}",
                      f"{gd:.4f}", f"{eps_s:.2f}", f"{er:.4f}", f"{pl_:.4f}", f"{vl_:.4f}", f"{en_:.4f}",
                      f"{kl_:.5f}", f"{sps:.0f}", mode, f"{s.get('truncated', nan):.4f}",
                      f"{s.get('goals_for', nan):.3f}", f"{s.get('goals_against', nan):.3f}",
                      f"{s.get('shots_per_ep', nan):.3f}", f"{s.get('possession_share', nan):.3f}", f"{up_p:.4f}", f"{up_f:.4f}", f"{br_p:.4f}", f"{br_f:.4f}"])
        logf.flush()
        if update % args.print_every == 0 or update == 1:
            print(f"upd {update:5d} | {minutes:5.2f} min | {global_steps:9d} dec | eps {total_eps:6d} | {mode[:5]} | "
                  f"win {win:.2f} loss {loss_:.2f} draw {draw:.2f} | goal diff/ep {gd:+.2f} | ep {eps_s:5.1f}s | "
                  f"ret {er:+.3f} | ent {en_:.2f} | up p {up_p:.2f} raised {up_f:.2f} | {sps:.0f} dec/s", flush=True)
        if update % args.save_every == 0:
            st = ckpt_state(args, update, global_steps, t0, t_offset, stamp, d, env_cfg)
            save_ckpt(run / "latest.pt", policy, opt, st)
        if args.keep_every and update % args.keep_every == 0:
            save_ckpt(run / "ckpts" / f"upd_{update:06d}.pt", policy, None,
                      ckpt_state(args, update, global_steps, t0, t_offset, stamp, d, env_cfg))
        prof.mark("other")
    if old_handler is not None:
        signal.signal(signal.SIGINT, old_handler)
    save_ckpt(run / "latest.pt", policy, opt, ckpt_state(args, update, global_steps, t0, t_offset, stamp, d, env_cfg))
    if args.profile:
        r1 = env.profile()
        table = profile_table(prof, prof_decisions, [r1[k] - rust0[k] for k in range(1, 5)])
        print(table)
        (run / "profile.md").write_text(table + "\n")
    logf.close()
    env.close()
    print(f"saved {run / 'latest.pt'}")
    if args.plot:
        plot(argparse.Namespace(run=str(run), out=None))


# ----------------------------------------------------------------------------------------------
# Evaluation

def evaluate(policy_fn, opponent: str, episodes: int, envs: int, seed: int, params, mode="match",
             episode_seconds=30.0, action_repeat=4, opp_policy=None, device="cpu", style=None, rows_out=None):
    """Play `episodes` episodes (full matches by default); returns summarize() of them. `style`
    (a list of floats) is written into the learner's style slot (style_mode "fixed")."""
    timestep = params_values(params)["timestep"]
    cfg = {"mode": mode, "action_repeat": action_repeat, "opponent": "external" if opp_policy else opponent,
           "style_mode": "fixed" if style is not None else "off"}
    if mode == "first_goal":
        cfg["max_steps"] = int(round(episode_seconds / timestep))
    if style is not None:
        cfg["style"] = list(style)
    envs = max(1, min(envs, episodes))
    env = VecEnv(envs, seed=seed, config=cfg, params_path=params)
    obs = env.reset()
    rows = []
    # Each env plays the same number of episodes (stopping at a total count would over-sample
    # short episodes).
    quota = np.full(envs, episodes // envs)
    quota[: episodes % envs] += 1
    count = np.zeros(envs, int)
    while (count < quota).any():
        oa = None
        if opp_policy is not None:
            raw, _, _ = opp_policy.act(torch.as_tensor(env.opp_obs.copy(), device=device))
            oa = to_env_actions(raw.cpu().numpy())
        obs, _, done, info = env.step(policy_fn(obs), oa)
        for i in np.nonzero(done)[0]:
            if count[i] < quota[i]:
                count[i] += 1
                rows.append(episode_row(info[i], env.reward_terms, timestep))
    env.close()
    if rows_out is not None:
        rows_out.extend(rows)
    return summarize(rows)


def policy_fn_for(policy: Policy, device, deterministic=False):
    def f(obs):
        raw, _, _ = policy.act(torch.as_tensor(obs, device=device), deterministic=deterministic)
        return to_env_actions(raw.cpu().numpy())
    return f


def eval_cmd(args):
    device = pick_device(args.device)
    if args.policy == "random":
        rng = np.random.default_rng(args.seed)
        fn = lambda obs: random_actions(obs.shape[0], rng)
        name = "random"
    elif args.policy == "untrained":
        torch.manual_seed(args.seed)
        fn = policy_fn_for(Policy(dims()["obs"]).to(device).eval(), device, args.deterministic)
        name = "untrained"
    elif args.policy in SCRIPTED:
        raise SystemExit("scripted-vs-scripted results: cargo run --release -p bots --example tournament")
    else:
        pol, ck = load_ckpt(args.ckpt or args.policy, args.params, args.allow_physics_change, device)
        fn = policy_fn_for(pol, device, args.deterministic)
        name = f"{args.ckpt or args.policy} (update {ck.get('update')})"
    opp_pol = None
    if args.opponent not in OPPONENTS:  # a checkpoint path: policy vs policy
        opp_pol, _ = load_ckpt(args.opponent, args.params, args.allow_physics_change, device)
    style = [float(x) for x in args.style.split(",")] if args.style else None
    r = evaluate(fn, args.opponent, args.episodes, args.envs, args.seed, args.params, args.mode, args.episode_seconds,
                 opp_policy=opp_pol, device=device, style=style)
    print(json.dumps({"policy": name, "opponent": args.opponent, "mode": args.mode, "style": style, **r}))
    return r


# ----------------------------------------------------------------------------------------------
# Plot

def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(args.run)
    rows = list(csv.DictReader(open(run / "metrics.csv")))
    x = np.array([float(r["minutes"]) for r in rows])
    get = lambda k: np.array([float(r.get(k) or "nan") for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=110)
    ax[0].plot(x, get("win"), label="win rate", color="#2a7")
    ax[0].plot(x, get("loss"), label="loss rate", color="#c33")
    ax[0].plot(x, get("draw"), label="draw / truncated", color="#999")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title(f"{run.name}: rolling outcomes")
    ax[0].legend(fontsize=8)
    ax[1].plot(x, get("goal_diff"), label="goal diff / episode", color="#225", lw=2)
    ax[1].plot(x, get("goals_for"), label="goals for / episode", color="#2a7", lw=1)
    ax[1].plot(x, get("goals_against"), label="goals against / episode", color="#c33", lw=1)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_title("goals per episode")
    ax[1].legend(fontsize=8)
    ax[2].plot(x, get("ep_return"), color="#a52")
    ax[2].set_title("mean episode return (shaped reward)")
    modes = [r.get("mode", "") for r in rows]
    for k in range(1, len(modes)):
        if modes[k] != modes[k - 1]:
            for a in ax:
                a.axvline(x[k], color="#a3a", ls="--", lw=1)
            ax[0].text(x[k], 1.0, f" {modes[k]}", color="#a3a", fontsize=8, va="top")
    for a in ax:
        a.grid(alpha=0.3)
        a.set_xlabel("training minutes")
    out = Path(args.out) if args.out else run / "learning_curve.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


# ----------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m train.ppo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--run", default="runs/default")
    t.add_argument("--params", default=str(DEFAULT_PARAMS))
    t.add_argument("--device", default="auto", help="PPO update device (auto = cuda if available)")
    t.add_argument("--rollout-device", default="cpu",
                   help="device that acts during rollouts: cpu (default, fastest for small batches), cuda, or 'same'")
    t.add_argument("--torch-threads", type=int, default=-1,
                   help="torch CPU threads (-1 = auto: 4 when rollouts act on the CPU, else torch default; 0 = torch default)")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--envs", type=int, default=64)
    t.add_argument("--threads", type=int, default=-1,
                   help="rayon threads in the env (-1 = auto: 1 for <= 128 envs, else min(8, cores); 0 = all cores)")
    t.add_argument("--chunk", type=int, default=16, help="envs per rayon task")
    t.add_argument("--steps", type=int, default=128, help="rollout length per env per update")
    t.add_argument("--minutes", type=float, default=0, help="wall-clock budget (0 = unlimited)")
    t.add_argument("--updates", type=int, default=0, help="update budget (0 = unlimited)")
    t.add_argument("--mode", default="first_goal", choices=["first_goal", "match"])
    t.add_argument("--curriculum-minutes", type=float, default=0,
                   help="train first_goal episodes for this many minutes, then full matches (overrides --mode)")
    t.add_argument("--overtime-cap", type=float, default=120.0, help="match mode: truncate overtime after this (s)")
    t.add_argument("--opponents", default="chaser", help="mix, e.g. chaser:0.3,defender:0.1,cautious:0.1,pool:0.5")
    t.add_argument("--snapshot-every", type=int, default=50, help="updates between self-play snapshots")
    t.add_argument("--pool-max", type=int, default=20)
    t.add_argument("--reward", default="", help="base reward weights JSON (file or inline)")
    t.add_argument("--styles", action="store_true",
                   help="sample a play style per episode (obs style slot + reward mapping, see sim-ffi/src/reward.rs)")
    t.add_argument("--obs-norm", action="store_true", help="running observation normalization (in the policy)")
    t.add_argument("--no-bootstrap", action="store_true",
                   help="treat truncated episodes as terminal (old behaviour; for comparisons only)")
    t.add_argument("--episode-seconds", type=float, default=30.0, help="first_goal mode: step limit")
    t.add_argument("--action-repeat", type=int, default=4)
    t.add_argument("--jitter-puck", type=float, default=6.0,
                   help="training start: puck uniform within +-this (m) of center ice per axis (eval uses 1.5)")
    t.add_argument("--jitter-player", type=float, default=2.0, help="training start: players +-this (m) per axis")
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--anneal", action="store_true", help="anneal LR over --minutes")
    t.add_argument("--gamma", type=float, default=0.995)
    t.add_argument("--lam", type=float, default=0.95)
    t.add_argument("--clip", type=float, default=0.2)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--minibatches", type=int, default=4)
    t.add_argument("--ent-coef", type=float, default=0.003)
    t.add_argument("--vf-coef", type=float, default=0.5)
    t.add_argument("--max-grad-norm", type=float, default=0.5)
    t.add_argument("--window", type=int, default=400, help="episodes in the rolling stats")
    t.add_argument("--print-every", type=int, default=10)
    t.add_argument("--save-every", type=int, default=25)
    t.add_argument("--keep-every", type=int, default=0,
                   help="also keep runs/NAME/ckpts/upd_N.pt every N updates (for the ladder)")
    t.add_argument("--resume", default="")
    t.add_argument("--allow-physics-change", action="store_true")
    t.add_argument("--profile", action="store_true", help="time every section (syncs CUDA) and print a table at the end")
    t.add_argument("--plot", action="store_true", help="write learning_curve.png at the end")

    e = sub.add_parser("eval")
    e.add_argument("--policy", default="ckpt", help="ckpt | random | untrained (or a checkpoint path)")
    e.add_argument("--ckpt", default="")
    e.add_argument("--opponent", default="chaser", help="scripted bot name, or a checkpoint path")
    e.add_argument("--episodes", type=int, default=200)
    e.add_argument("--envs", type=int, default=32)
    e.add_argument("--seed", type=int, default=12345)
    e.add_argument("--mode", default="match", choices=["first_goal", "match"], help="evaluate on full matches (default)")
    e.add_argument("--episode-seconds", type=float, default=30.0, help="first_goal mode only")
    e.add_argument("--style", default="", help="style vector for the learner, e.g. 1,0,0,0")
    e.add_argument("--params", default=str(DEFAULT_PARAMS))
    e.add_argument("--device", default="cpu")
    e.add_argument("--deterministic", action="store_true")
    e.add_argument("--allow-physics-change", action="store_true")

    p = sub.add_parser("plot")
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "eval" and args.policy == "ckpt" and not args.ckpt:
        ap.error("eval needs --ckpt (or --policy random|untrained)")
    return {"train": train, "eval": eval_cmd, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    main()
