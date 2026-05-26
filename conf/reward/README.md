# Reward Configs

 Knobs for reward shaping configuration

Common profiles:

```yaml
terminal_only         # binary terminal reward only
early_terminal_only   # terminal reward plus early terminal shaping
```

Example usage:

```bash
# default config with reward override
uv run ow train reward=terminal_only
uv run ow train reward=early_terminal_only
```
