# Reward Configs

 Knobs for reward shaping configuration

Common profiles:

```yaml
terminal_only         # binary terminal reward only
early_terminal_only   # terminal reward plus early terminal shaping
ship_differential     # terminal_only + normalized_ship_differential mode
```

`ship_differential` uses best-opponent normalization at episode end:
`(L - O_max) / (L + O_max)` when `L + O_max > 0`, else `0` (ties and all-zero scores).

Example usage:

```bash
# default config with reward override
uv run ow train reward=terminal_only
uv run ow train reward=early_terminal_only
uv run ow train reward=ship_differential
```
