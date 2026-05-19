# Config migration table

Legacy flat YAML files in `configs/` were removed. Use Hydra experiment selection in `conf/experiment/`.

| Removed legacy config | Hydra command replacement |
|---|---|
| `configs/full_training.yaml` | `python -m src.train experiment=full_training` |
| `configs/attention_training.yaml` | `python -m src.train experiment=attention_training` |
| `configs/shaped_reward_training.yaml` | `python -m src.train experiment=shaped_reward_training` |
| `configs/attention_shaped_reward_training.yaml` | `python -m src.train experiment=attention_shaped_reward` |
| `configs/attention_self_play_pool.yaml` | `python -m src.train experiment=attention_self_play_pool` |
| `configs/attention_candidates_16.yaml` | `python -m src.train experiment=attention_candidates_16` |
| `configs/attention_candidates_24.yaml` | `python -m src.train experiment=attention_candidates_24` |
| `configs/mixed_2p_4p_training.yaml` | `python -m src.train experiment=mixed_2p_4p_training` |
| `configs/jax_training.yaml` | `python -m src.train experiment=jax_training` |
| `configs/jax_self_play_shaped_reward_training.yaml` | `python -m src.train experiment=jax_self_play_shaped_reward` |
| `configs/jax_mixed_2p_4p_training.yaml` | `python -m src.train experiment=jax_mixed_2p_4p_training` |
| `configs/jax_entity_transformer_500k.yaml` | `python -m src.train experiment=jax_entity_transformer_500k` |
| `configs/jax_entity_transformer_700k.yaml` | `python -m src.train experiment=jax_entity_transformer_700k` |
| `configs/jax_entity_transformer_1m.yaml` | `python -m src.train experiment=jax_entity_transformer_1m` |

Default parity command (equivalent to `default_cfg.yaml`):

```bash
python -m src.train
```
