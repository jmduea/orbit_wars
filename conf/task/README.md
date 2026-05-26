# Task Configs

Contains environment/action/feature knobs may or may not be refactored for better naming clarity and separation of concerns.

A lot of the knobs to tweak here can have a SEVERE impact on training throughput, some recommended default configurations are provided below.

Important trajectory shield modes:

```yaml
trajectory_shield_mode: off      # no trajectory shield
trajectory_shield_mode: cheap    # cheap feature-derived mask
trajectory_shield_mode: exact    # full exact shield
trajectory_shield_mode: tiered   # cheap mask + exact selected-launch validation
```

**Recommended/current default for training:**

```yaml
trajectory_shield_mode: cheap
trajectory_shield_horizon: 50
trajectory_shield_final_validate_selected: false
```

Recommended debugging/eval mode:

```yaml
trajectory_shield_mode: exact
trajectory_shield_horizon: 500
```

Recommended tiered mode:

```yaml
trajectory_shield_mode: tiered
trajectory_shield_final_validate_selected: true
trajectory_shield_horizon: 500
```

Tiered mode is a happy medium between the cheap and exact presets, but may still be too taxing depending on training setup/environment.
