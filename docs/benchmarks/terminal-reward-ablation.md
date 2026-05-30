# Terminal reward ablation

Paired **500u** runs on the workstation validation profile (`reward=terminal_only` baseline vs `reward=ship_differential` candidate).

| Arm | Reward profile | overall_win_rate | average_reward | episode_reward_mean | survival_time | policy_loss | JSON |
|-----|----------------|------------------|----------------|---------------------|---------------|-------------|------|
| binary_win | reward=terminal_only | 0.6211 | -0.0027 | 1.9339 | 0.0018 | 4757.2823 | `docs/benchmarks/terminal-reward-binary-500u.json` |
| ship_differential | reward=ship_differential | 0.3403 | 1879.6009 | 1.1548 | 0.0064 | 4611.3160 | `docs/benchmarks/terminal-reward-ship-diff-500u.json` |

## Notes

- `overall_win_rate` uses binary `terminal_is_first` telemetry in both arms.
- Candidate terminal signal is graded in [-1, 1] via best-opponent normalization.
