"""Rating ladder: checkpoints against the scripted bots and against each other, Elo-style ratings,
and per-style statistics.

    python -m train.league --ckpts runs/m1/ckpts/*.pt runs/m1/latest.pt --matches 16 --out runs/m1/league.json
    python -m train.league --ckpts runs/m1/latest.pt --styles "0,0,0,0;1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1"

Games are full matches (--mode match, the default) or first-goal episodes (--mode first_goal).
Each checkpoint plays every scripted bot in --bots (the bot is the env's opponent) and every other
checkpoint (policy vs policy through the external-opponent slot), --matches games per pairing, the
learner's side random per game. A win counts 1, a loss 0, a draw (truncated game) 0.5.

Ratings: a Bradley-Terry maximum-likelihood fit on the Elo scale (400 points = 10:1 odds) over all
games, with one virtual draw per played pairing (keeps unbeaten / winless players finite), anchored
so the --anchor bot (default chaser) has rating 1000. Scripted bots don't play each other here
(their head-to-head is `cargo run --release -p bots --example tournament`).

Per-style statistics: each checkpoint plays --style-matches games against --style-opponent with each
style vector in --styles written into its observation's style slot (style 0 = default behaviour):
win/loss/draw, goals for/against, shots on net per match and possession share (learner possession /
(learner + opponent possession) seconds). Policies trained without --styles ignore the slot, so their
rows mostly show the noise level.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from train.ppo import evaluate, load_ckpt, policy_fn_for
from train.slapenv import DEFAULT_PARAMS, SCRIPTED


def elo_fit(players: list[str], games: list[tuple[str, str, float, int]], anchor: str | None, iters: int = 2000) -> dict:
    """games: (a, b, score of a in [0, 1] summed over n games, n). Returns {player: rating}."""
    idx = {p: i for i, p in enumerate(players)}
    n_p = len(players)
    w = np.zeros((n_p, n_p))   # w[i, j] = points i scored against j
    n = np.zeros((n_p, n_p))
    for a, b, s, k in games:
        i, j = idx[a], idx[b]
        w[i, j] += s
        w[j, i] += k - s
        n[i, j] += k
        n[j, i] += k
    played = n > 0
    w += 0.5 * played   # one virtual draw per played pairing
    n += 1.0 * played
    r = np.zeros(n_p)   # natural-log strengths
    for _ in range(iters):  # MM algorithm (Hunter 2004) for Bradley-Terry
        e = np.exp(r)
        denom = (n / (e[:, None] + e[None, :])).sum(1)
        new = np.log(np.maximum(w.sum(1), 1e-12) / np.maximum(denom, 1e-12))
        new -= new.mean()
        if np.max(np.abs(new - r)) < 1e-10:
            r = new
            break
        r = new
    elo = r * 400 / math.log(10)
    if anchor in idx:
        elo += 1000 - elo[idx[anchor]]
    else:
        elo += 1000 - elo.mean()
    return {p: float(elo[idx[p]]) for p in players}


def label(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}/{p.stem}" if p.parent.name not in ("", ".") else p.stem


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m train.league")
    ap.add_argument("--ckpts", nargs="+", required=True, help="checkpoint paths or globs")
    ap.add_argument("--bots", default="chaser,defender,cautious,idle")
    ap.add_argument("--matches", type=int, default=16, help="games per pairing")
    ap.add_argument("--mode", default="match", choices=["match", "first_goal"])
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--anchor", default="chaser")
    ap.add_argument("--no-cross", action="store_true", help="skip checkpoint-vs-checkpoint games")
    ap.add_argument("--styles", default="0,0,0,0", help="style vectors separated by ';' (empty = no style stats)")
    ap.add_argument("--style-opponent", default="chaser")
    ap.add_argument("--style-matches", type=int, default=16)
    ap.add_argument("--deterministic", action="store_true", help="greedy actions (default: sampled, as trained)")
    ap.add_argument("--params", default=str(DEFAULT_PARAMS))
    ap.add_argument("--allow-physics-change", action="store_true")
    ap.add_argument("--out", default="league.json")
    args = ap.parse_args(argv)

    paths = []
    for c in args.ckpts:
        hits = sorted(glob.glob(c))
        paths += hits if hits else [c]
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise SystemExit("no checkpoints")
    bots = [b for b in args.bots.split(",") if b]
    for b in bots:
        if b not in SCRIPTED:
            raise SystemExit(f"unknown bot {b}")
    torch.manual_seed(args.seed)
    pols = {}
    for p in paths:
        pol, ck = load_ckpt(p, args.params, args.allow_physics_change, "cpu")
        pols[label(p)] = (pol, ck)
    names = list(pols)
    t0 = time.time()
    games, table = [], []

    def play(a_name, opponent, opp_policy=None, style=None, seed=0):
        pol = pols[a_name][0]
        rows: list[dict] = []
        s = evaluate(policy_fn_for(pol, "cpu", args.deterministic), opponent, args.matches if style is None else args.style_matches,
                     args.envs, seed, args.params, args.mode, opp_policy=opp_policy, style=style, rows_out=rows)
        return s, rows

    k = 0
    for a in names:
        for b in bots:
            k += 1
            s, _ = play(a, b, seed=args.seed + k)
            score = s["win"] * s["episodes"] + 0.5 * s["draw"] * s["episodes"]
            games.append((a, b, score, s["episodes"]))
            table.append({"a": a, "b": b, **s})
            print(f"{a:>28s} vs {b:<28s} W {s['win']:.2f} D {s['draw']:.2f} L {s['loss']:.2f} | "
                  f"goals {s['goals_for']:.2f}-{s['goals_against']:.2f} | shots {s['shots_per_ep']:.2f} | "
                  f"poss {s['possession_share']:.2f} | {s['ep_seconds']:.0f} s/game", flush=True)
    if not args.no_cross:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                k += 1
                s, _ = play(a, "external", opp_policy=pols[b][0], seed=args.seed + k)
                score = s["win"] * s["episodes"] + 0.5 * s["draw"] * s["episodes"]
                games.append((a, b, score, s["episodes"]))
                table.append({"a": a, "b": b, **s})
                print(f"{a:>28s} vs {b:<28s} W {s['win']:.2f} D {s['draw']:.2f} L {s['loss']:.2f} | "
                      f"goals {s['goals_for']:.2f}-{s['goals_against']:.2f}", flush=True)
    players = names + bots
    ratings = elo_fit(players, games, args.anchor)
    print("\nratings (Elo scale, anchor %s = 1000):" % args.anchor)
    for p in sorted(players, key=lambda x: -ratings[x]):
        print(f"  {ratings[p]:7.1f}  {p}{'  (scripted)' if p in bots else ''}")

    styles = [[float(x) for x in s.split(",")] for s in args.styles.split(";") if s.strip()] if args.styles else []
    style_stats = []
    for a in names:
        for st in styles:
            k += 1
            s, _ = play(a, args.style_opponent, style=st, seed=args.seed + k)
            style_stats.append({"policy": a, "style": st, "opponent": args.style_opponent, **s})
            print(f"style {a} {st} vs {args.style_opponent}: W {s['win']:.2f} D {s['draw']:.2f} L {s['loss']:.2f} | "
                  f"goals for {s['goals_for']:.2f} against {s['goals_against']:.2f} | shots/match {s['shots_per_ep']:.2f} | "
                  f"possession share {s['possession_share']:.3f}", flush=True)

    out = {"mode": args.mode, "matches_per_pairing": args.matches, "anchor": args.anchor, "deterministic": args.deterministic,
           "ratings": dict(sorted(ratings.items(), key=lambda kv: -kv[1])),
           "checkpoints": {n: {"path": p, "update": pols[n][1].get("update"), "decisions": pols[n][1].get("global_steps"),
                               "physics_stamp": pols[n][1].get("physics_stamp")} for n, p in zip(names, paths)},
           "games": table, "style_stats": style_stats, "seconds": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out} ({time.time() - t0:.0f} s)")
    return out


if __name__ == "__main__":
    main()
