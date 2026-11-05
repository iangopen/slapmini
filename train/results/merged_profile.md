| section | what | seconds | share | us per decision |
|---|---|---|---|---|
| env_rust | env step inside Rust (sim + opponent bot + reward + observe, rayon) | 1.19 | 13.6% | 3.64 |
| env_ctypes | ctypes call + Python wrapper around env.step | 0.13 | 1.5% | 0.41 |
| act_h2d | observation -> rollout device | 0.05 | 0.5% | 0.14 |
| act_forward | policy forward + sampling | 2.82 | 32.2% | 8.61 |
| act_d2h | actions -> host, to_env_actions | 0.14 | 1.6% | 0.42 |
| opp_policy | self-play opponent inference | 0.06 | 0.6% | 0.17 |
| bookkeeping | rollout buffers, done handling, truncation bootstrap | 0.43 | 4.9% | 1.30 |
| gae | returns and advantages | 0.18 | 2.0% | 0.54 |
| update_h2d | rollout buffers -> training device | 0.05 | 0.6% | 0.17 |
| update | PPO epochs (forward, backward, Adam) | 3.65 | 41.6% | 11.13 |
| other | obs-norm update, weight sync, logging, snapshots, checkpoints | 0.06 | 0.7% | 0.20 |
| total | | 8.76 | 100% | 26.72 |

327680 decisions, 37425 decisions/s (profiled; CUDA synced per section)
Rust CPU time split (summed over envs and threads): sim 49.4%, reward 5.6%, opponent bot 40.6%, observe+reset 4.4%
