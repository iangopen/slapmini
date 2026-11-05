| section | what | seconds | share | us per decision |
|---|---|---|---|---|
| env_rust | env step inside Rust (sim + opponent bot + reward + observe, rayon) | 1.17 | 14.6% | 3.57 |
| env_ctypes | ctypes call + Python wrapper around env.step | 0.13 | 1.6% | 0.40 |
| act_h2d | observation -> rollout device | 0.04 | 0.6% | 0.14 |
| act_forward | policy forward + sampling | 2.71 | 33.7% | 8.27 |
| act_d2h | actions -> host, to_env_actions | 0.13 | 1.6% | 0.40 |
| opp_policy | self-play opponent inference | 0.05 | 0.7% | 0.17 |
| bookkeeping | rollout buffers, done handling, truncation bootstrap | 0.41 | 5.1% | 1.25 |
| gae | returns and advantages | 0.18 | 2.3% | 0.56 |
| update_h2d | rollout buffers -> training device | 0.08 | 1.0% | 0.24 |
| update | PPO epochs (forward, backward, Adam) | 3.06 | 38.1% | 9.35 |
| other | obs-norm update, weight sync, logging, snapshots, checkpoints | 0.06 | 0.8% | 0.19 |
| total | | 8.04 | 100% | 24.53 |

327680 decisions, 40764 decisions/s (profiled; CUDA synced per section)
Rust CPU time split (summed over envs and threads): sim 47.3%, reward 5.6%, opponent bot 42.5%, observe+reset 4.6%
