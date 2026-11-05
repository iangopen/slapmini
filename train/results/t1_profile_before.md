| section | what | seconds | share | us per decision |
|---|---|---|---|---|
| env_rust | env step inside Rust (sim + opponent bot + reward + observe, rayon) | 1.57 | 5.6% | 4.79 |
| env_ctypes | ctypes call + Python wrapper around env.step | 0.24 | 0.9% | 0.74 |
| act_h2d | observation -> rollout device | 0.74 | 2.6% | 2.27 |
| act_forward | policy forward + sampling | 16.36 | 58.1% | 49.92 |
| act_d2h | actions -> host, to_env_actions | 0.72 | 2.6% | 2.20 |
| opp_policy | self-play opponent inference | 0.08 | 0.3% | 0.26 |
| bookkeeping | rollout buffers, done handling, truncation bootstrap | 3.59 | 12.7% | 10.94 |
| gae | returns and advantages | 0.39 | 1.4% | 1.18 |
| update_h2d | rollout buffers -> training device | 0.00 | 0.0% | 0.01 |
| update | PPO epochs (forward, backward, Adam) | 4.43 | 15.7% | 13.52 |
| other | obs-norm update, weight sync, logging, snapshots, checkpoints | 0.03 | 0.1% | 0.10 |
| total | | 28.16 | 100% | 85.93 |

327680 decisions, 11637 decisions/s (profiled; CUDA synced per section)
Rust CPU time split (summed over envs and threads): sim 49.9%, reward 5.7%, opponent bot 39.3%, observe+reset 5.2%
