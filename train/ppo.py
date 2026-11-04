"""Single-file PPO for slapmini 1v1, with scripted opponents and a self-play snapshot pool.

    python -m train.ppo train --run runs/smoke --minutes 5 --opponents chaser
    python -m train.ppo eval  --ckpt runs/smoke/latest.pt --opponent chaser --episodes 200
    python -m train.ppo eval  --policy random --opponent chaser
    python -m train.ppo plot  --run runs/smoke

Policy (canonical frame, see bots/src/obs.rs and bots/src/action.rs):
  obs[OBS_DIM] -> MLP -> move (2 Gaussian, clipped to [-1, 1] by the env), stick (2D Gaussian
  direction; the env gets atan2 of it as the stick target angle), dash (Bernoulli), lift (Bernoulli).
  The stick is a direction, not an angle, so there is no wrap-around at +-pi.
Device: CUDA if available, else CPU (override with --device).
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
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train.slapenv import (DEFAULT_PARAMS, END_GOAL_AGAINST, END_GOAL_FOR, OPPONENTS, SCRIPTED, VecEnv, dims,
                           physics_stamp)

RAW_DIM = 6  # sampled action: move_x, move_y, stick_x, stick_y, dash, lift


# ----------------------------------------------------------------------------------------------
# Policy

def mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.Tanh()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class Policy(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.obs_dim, self.hidden = obs_dim, hidden
        self.pi = mlp(obs_dim, hidden, 6)       # move mean (2), stick mean (2), dash logit, lift logit
        self.vf = mlp(obs_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.5, -0.5, -0.5, -0.5]))
        with torch.no_grad():  # small initial outputs: near-uniform start
            self.pi[-1].weight.mul_(0.01)
            self.pi[-1].bias.zero_()
            self.vf[-1].weight.mul_(0.1)

    def heads(self, obs):
        out = self.pi(obs)
        return out[:, 0:4], out[:, 4], out[:, 5]

    def dist(self, obs):
        mean, dl, ll = self.heads(obs)
        std = self.log_std.clamp(-3.0, 1.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), torch.distributions.Bernoulli(logits=dl), \
            torch.distributions.Bernoulli(logits=ll)

    def value(self, obs):
        return self.vf(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Returns (raw [N, 6], logp [N], value [N])."""
        g, d, l = self.dist(obs)
        if deterministic:
            cont = g.mean
            dash = (d.probs > 0.5).float()
            lift = (l.probs > 0.5).float()
        else:
            cont, dash, lift = g.sample(), d.sample(), l.sample()
        raw = torch.cat([cont, dash[:, None], lift[:, None]], dim=1)
        return raw, self.logp(g, d, l, raw), self.value(obs)

    @staticmethod
    def logp(g, d, l, raw):
        return g.log_prob(raw[:, :4]).sum(-1) + d.log_prob(raw[:, 4]) + l.log_prob(raw[:, 5])

    def evaluate(self, obs, raw):
        g, d, l = self.dist(obs)
        ent = g.entropy().sum(-1) + d.entropy() + l.entropy()
        return self.logp(g, d, l, raw), ent, self.value(obs)


def to_env_actions(raw: np.ndarray) -> np.ndarray:
    """[N, 6] sampled -> [N, 5] env format (move_x, move_y, stick angle, dash, lift)."""
    a = np.empty((raw.shape[0], 5), np.float32)
    a[:, 0:2] = np.clip(raw[:, 0:2], -1.0, 1.0)
    a[:, 2] = np.arctan2(raw[:, 3], raw[:, 2])
    a[:, 3] = raw[:, 4]
    a[:, 4] = raw[:, 5]
    return a


def random_actions(n: int, rng: np.random.Generator) -> np.ndarray:
    a = np.empty((n, 5), np.float32)
    a[:, 0:2] = rng.uniform(-1, 1, (n, 2))
    a[:, 2] = rng.uniform(-math.pi, math.pi, n)
    a[:, 3:5] = (rng.uniform(0, 1, (n, 2)) < 0.5)
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
                "hidden": policy.hidden, "obs_dim": policy.obs_dim, **state}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, params_path=DEFAULT_PARAMS, allow_physics_change=False, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    d = dims()
    if ck.get("obs_version") != d["obs_version"] or ck.get("obs_dim") != d["obs"]:
        raise SystemExit(f"{path}: observation layout v{ck.get('obs_version')}/{ck.get('obs_dim')} does not match "
                         f"the library's v{d['obs_version']}/{d['obs']}; cannot load")
    stamp = physics_stamp(params_path)
    if ck.get("physics_stamp") != stamp:
        msg = (f"{path}: trained under physics {ck.get('physics_stamp')}, current params give {stamp}")
        if not allow_physics_change:
            raise SystemExit(msg + " (pass --allow-physics-change to continue anyway)")
        print("WARNING: " + msg, file=sys.stderr)
    pol = Policy(ck["obs_dim"], ck["hidden"]).to(device)
    pol.load_state_dict(ck["policy"])
    return pol, ck


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
                self.pool.append(torch.load(p, map_location="cpu", weights_only=False))
            self.pool = self.pool[-pool_max:]

    def add_snapshot(self, policy: Policy, tag: int):
        sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        self.pool.append({"state": sd, "hidden": policy.hidden, "obs_dim": policy.obs_dim, "tag": tag})
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
        self.active = []
        for i in idx:
            s = self.pool[i]
            m = Policy(s["obs_dim"], s["hidden"]).to(self.device)
            m.load_state_dict(s["state"])
            m.eval()
            self.active.append(m)
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
        out = np.zeros((self.env.n, 5), np.float32)
        for s, m in enumerate(self.active):
            idx = np.nonzero(self.slot == s)[0]
            if len(idx):
                raw, _, _ = m.act(torch.as_tensor(opp_obs[idx], device=self.device))
                out[idx] = to_env_actions(raw.cpu().numpy())
        return out


# ----------------------------------------------------------------------------------------------
# Training

def parse_mix(s: str) -> dict:
    out = {}
    for part in s.split(","):
        k, _, w = part.partition(":")
        out[k.strip()] = float(w) if w else 1.0
    return out


def train(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    stamp = physics_stamp(args.params)
    d = dims()

    env_cfg = {"max_steps": int(args.episode_seconds * 120), "action_repeat": args.action_repeat,
               "threads": args.threads, "jitter_puck": args.jitter_puck, "jitter_player": args.jitter_player}
    if args.reward:
        env_cfg["reward"] = json.loads(Path(args.reward).read_text()) if Path(args.reward).exists() else json.loads(args.reward)
    env = VecEnv(args.envs, seed=args.seed, config=env_cfg, params_path=args.params)

    policy = Policy(d["obs"], args.hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    update, global_steps, t_offset = 0, 0, 0.0
    if args.resume:
        policy, ck = load_ckpt(args.resume, args.params, args.allow_physics_change, device)
        opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
        if ck.get("optimizer") and ck.get("physics_stamp") == stamp:
            opt.load_state_dict(ck["optimizer"])
        update, global_steps, t_offset = ck.get("update", 0), ck.get("global_steps", 0), ck.get("wall_seconds", 0.0)
        print(f"resumed {args.resume} at update {update}, {global_steps} decisions")

    mixer = OpponentMixer(env, parse_mix(args.opponents), device, rng,
                          run / "pool" if "pool" in args.opponents else None, args.pool_max)
    if "pool" in mixer.mix and not mixer.pool:
        mixer.add_snapshot(policy, update)
    mixer.refresh_active()
    for i in range(env.n):
        mixer.assign(i)

    print(f"device {device}, {env.n} envs, obs {d['obs']}, physics {stamp}, opponents {mixer.mix}")
    obs = env.reset()
    opp_obs = env.opp_obs.copy()
    N, T = env.n, args.steps
    buf_obs = torch.zeros((T, N, d["obs"]), device=device)
    buf_raw = torch.zeros((T, N, RAW_DIM), device=device)
    buf_logp = torch.zeros((T, N), device=device)
    buf_val = torch.zeros((T, N), device=device)
    buf_rew = torch.zeros((T, N), device=device)
    buf_done = torch.zeros((T, N), device=device)

    window = deque(maxlen=args.window)   # (outcome, opponent kind, sim steps, return)
    log_path = run / "metrics.csv"
    new_log = not log_path.exists() or not args.resume
    logf = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["update", "decisions", "minutes", "episodes", "win", "loss", "draw", "goal_diff",
                      "ep_seconds", "ep_return", "pi_loss", "v_loss", "entropy", "kl", "sps"])
    t0 = time.time()
    total_eps = 0
    while True:
        minutes = (time.time() - t0 + t_offset) / 60
        if args.minutes and minutes >= args.minutes:
            break
        if args.updates and update >= args.updates:
            break
        # LR annealing over the time budget when one is given.
        if args.minutes and args.anneal:
            for g in opt.param_groups:
                g["lr"] = args.lr * max(0.05, 1 - minutes / args.minutes)

        # ---- rollout
        policy.eval()
        ts = time.time()
        for t in range(T):
            o = torch.as_tensor(obs, device=device)
            raw, logp, val = policy.act(o)
            opp_a = mixer.actions(opp_obs)
            obs, rew, done, info = env.step(to_env_actions(raw.cpu().numpy()), opp_a)
            opp_obs = env.opp_obs.copy()
            buf_obs[t], buf_raw[t], buf_logp[t], buf_val[t] = o, raw, logp, val
            buf_rew[t] = torch.as_tensor(rew, device=device)
            buf_done[t] = torch.as_tensor(done, device=device, dtype=torch.float32)
            for i in np.nonzero(done)[0]:
                outcome = {END_GOAL_FOR: 1, END_GOAL_AGAINST: -1}.get(int(info[i, 0]), 0)
                window.append((outcome, mixer.kind[i], info[i, 1], info[i, 2]))
                total_eps += 1
                mixer.assign(i)
        global_steps += N * T
        sps = N * T / (time.time() - ts)

        # ---- GAE (time-limit ends are treated as terminal; see CLAUDE.md)
        with torch.no_grad():
            next_val = policy.value(torch.as_tensor(obs, device=device))
            adv = torch.zeros_like(buf_rew)
            last = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nv = next_val if t == T - 1 else buf_val[t + 1]
                nonterm = 1.0 - buf_done[t]
                delta = buf_rew[t] + args.gamma * nv * nonterm - buf_val[t]
                last = delta + args.gamma * args.lam * nonterm * last
                adv[t] = last
            ret = adv + buf_val

        # ---- PPO update
        policy.train()
        b_obs, b_raw = buf_obs.reshape(-1, d["obs"]), buf_raw.reshape(-1, RAW_DIM)
        b_logp, b_adv, b_ret = buf_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        n_b = b_obs.shape[0]
        mb = max(1, n_b // args.minibatches)
        stats = []
        for _ in range(args.epochs):
            perm = torch.randperm(n_b, device=device)
            for s in range(0, n_b, mb):
                j = perm[s:s + mb]
                logp, ent, v = policy.evaluate(b_obs[j], b_raw[j])
                ratio = (logp - b_logp[j]).exp()
                a = b_adv[j]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pl = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                vl = 0.5 * (v - b_ret[j]).pow(2).mean()
                loss = pl + args.vf_coef * vl - args.ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kl = (b_logp[j] - logp).mean().item()
                stats.append((pl.item(), vl.item(), ent.mean().item(), kl))
        update += 1
        pl_, vl_, en_, kl_ = np.mean(stats, axis=0)

        # ---- self-play snapshots
        if "pool" in mixer.mix and update % args.snapshot_every == 0:
            mixer.add_snapshot(policy, update)

        # ---- log
        if window:
            oc = np.array([w[0] for w in window])
            win, loss_, draw = (oc == 1).mean(), (oc == -1).mean(), (oc == 0).mean()
            gd = oc.mean()
            eps_s = np.mean([w[2] for w in window]) / 120
            er = np.mean([w[3] for w in window])
        else:
            win = loss_ = draw = gd = eps_s = er = float("nan")
        minutes = (time.time() - t0 + t_offset) / 60
        log.writerow([update, global_steps, f"{minutes:.3f}", total_eps, f"{win:.4f}", f"{loss_:.4f}", f"{draw:.4f}",
                      f"{gd:.4f}", f"{eps_s:.2f}", f"{er:.4f}", f"{pl_:.4f}", f"{vl_:.4f}", f"{en_:.4f}",
                      f"{kl_:.5f}", f"{sps:.0f}"])
        logf.flush()
        if update % args.print_every == 0 or update == 1:
            print(f"upd {update:5d} | {minutes:5.2f} min | {global_steps:9d} dec | eps {total_eps:6d} | "
                  f"win {win:.2f} loss {loss_:.2f} draw {draw:.2f} | goal diff/ep {gd:+.2f} | ep {eps_s:4.1f}s | "
                  f"ret {er:+.3f} | ent {en_:.2f} | {sps:.0f} dec/s", flush=True)
        if update % args.save_every == 0:
            ck_state = {"update": update, "global_steps": global_steps, "wall_seconds": time.time() - t0 + t_offset,
                        "physics_stamp": stamp, "obs_version": d["obs_version"], "args": vars(args),
                        "env_config": env_cfg}
            save_ckpt(run / "latest.pt", policy, opt, ck_state)
    ck_state = {"update": update, "global_steps": global_steps, "wall_seconds": time.time() - t0 + t_offset,
                "physics_stamp": stamp, "obs_version": d["obs_version"], "args": vars(args), "env_config": env_cfg}
    save_ckpt(run / "latest.pt", policy, opt, ck_state)
    logf.close()
    env.close()
    print(f"saved {run / 'latest.pt'}")
    if args.plot:
        plot(argparse.Namespace(run=str(run), out=None))


# ----------------------------------------------------------------------------------------------
# Evaluation

def evaluate(policy_fn, opponent: str, episodes: int, envs: int, seed: int, params, episode_seconds=30.0,
             action_repeat=4, opp_policy=None, device="cpu"):
    """Play `episodes` episodes; returns a dict of win/loss/draw rates and goal difference per episode."""
    cfg = {"max_steps": int(episode_seconds * 120), "action_repeat": action_repeat,
           "opponent": "external" if opp_policy else opponent}
    env = VecEnv(envs, seed=seed, config=cfg, params_path=params)
    obs = env.reset()
    outcomes, lens = [], []
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
                outcomes.append({END_GOAL_FOR: 1, END_GOAL_AGAINST: -1}.get(int(info[i, 0]), 0))
                lens.append(info[i, 1] / 120)
    env.close()
    oc = np.array(outcomes)
    return {"episodes": len(oc), "win": float((oc == 1).mean()), "loss": float((oc == -1).mean()),
            "draw": float((oc == 0).mean()), "goal_diff_per_ep": float(oc.mean()),
            "ep_seconds": float(np.mean(lens))}


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
        fn = policy_fn_for(Policy(dims()["obs"]).to(device), device, args.deterministic)
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
    r = evaluate(fn, args.opponent, args.episodes, args.envs, args.seed, args.params, args.episode_seconds,
                 opp_policy=opp_pol, device=device)
    print(json.dumps({"policy": name, "opponent": args.opponent, **r}))
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
    get = lambda k: np.array([float(r[k]) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
    ax[0].plot(x, get("win"), label="win rate", color="#2a7")
    ax[0].plot(x, get("loss"), label="loss rate", color="#c33")
    ax[0].plot(x, get("draw"), label="draw (time limit)", color="#999")
    ax[0].plot(x, get("goal_diff"), label="goal diff / episode", color="#225", lw=2)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("training minutes")
    ax[0].set_title(f"{run.name}: rolling outcomes")
    ax[0].legend(fontsize=8)
    ax[0].set_ylim(-1.05, 1.05)
    ax[1].plot(x, get("ep_return"), color="#a52")
    ax[1].set_xlabel("training minutes")
    ax[1].set_title("mean episode return (shaped reward)")
    for a in ax:
        a.grid(alpha=0.3)
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
    t.add_argument("--device", default="auto")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--envs", type=int, default=64)
    t.add_argument("--threads", type=int, default=0, help="rayon threads in the env (0 = all cores)")
    t.add_argument("--steps", type=int, default=128, help="rollout length per env per update")
    t.add_argument("--minutes", type=float, default=0, help="wall-clock budget (0 = unlimited)")
    t.add_argument("--updates", type=int, default=0, help="update budget (0 = unlimited)")
    t.add_argument("--opponents", default="chaser", help="mix, e.g. chaser:0.3,defender:0.1,cautious:0.1,pool:0.5")
    t.add_argument("--snapshot-every", type=int, default=50, help="updates between self-play snapshots")
    t.add_argument("--pool-max", type=int, default=20)
    t.add_argument("--reward", default="", help="reward weights JSON (file or inline)")
    t.add_argument("--episode-seconds", type=float, default=30.0)
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
    t.add_argument("--resume", default="")
    t.add_argument("--allow-physics-change", action="store_true")
    t.add_argument("--plot", action="store_true", help="write learning_curve.png at the end")

    e = sub.add_parser("eval")
    e.add_argument("--policy", default="ckpt", help="ckpt | random | untrained (or a checkpoint path)")
    e.add_argument("--ckpt", default="")
    e.add_argument("--opponent", default="chaser", help="scripted bot name, or a checkpoint path")
    e.add_argument("--episodes", type=int, default=200)
    e.add_argument("--envs", type=int, default=32)
    e.add_argument("--seed", type=int, default=12345)
    e.add_argument("--episode-seconds", type=float, default=30.0)
    e.add_argument("--params", default=str(DEFAULT_PARAMS))
    e.add_argument("--device", default="auto")
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
