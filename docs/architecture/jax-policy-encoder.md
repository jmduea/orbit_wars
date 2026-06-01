# JAX policy encoder stage

Planet-edge policies share a composable shell: a backbone encoder maps `TurnBatch`
to per-edge candidate embeddings, then an autoregressive pointer decoder selects
up to `max_moves_k` launches per turn.

## Encoder dispatch

```mermaid
flowchart LR
    TB[TurnBatch] --> DISPATCH{model.architecture}
    DISPATCH -->|planet_graph_transformer| TX[PlanetGraphTransformerEncoder]
    TX --> OUT[PlanetEdgeEncoderOutput]
    OUT --> DEC[AutoregressivePointerDecoder]
    DEC --> ACT[target_logits + ship_logits + value]
```

## Modules

| Module | Path | Role |
|--------|------|------|
| Shared contracts/helpers | `src/jax/encoders/planet_encoder_common.py` | `PlanetEdgeEncoderOutput`, tgt-aware edge fusion, pooling |
| GNN backbone | `src/jax/policy.py` (`PlanetEdgeBackboneEncoder`) | k-NN spatial message passing |
| Transformer backbone | `src/jax/encoders/planet_graph_transformer.py` | Masked planet self-attention + spatial bias |
| Policy factory | `src/jax/policy.py` (`build_jax_policy`) | Architecture dispatch |
| Remat helper | `src/jax/encoders/remat.py` | Optional `nn.remat` wrapper for encoder blocks |
| Checkpoint guard | `src/artifacts/checkpoint_compat.py` | `encoder_backbone` metadata plane |

## Gradient checkpointing

When `training.enable_gradient_checkpointing=true`, GNN message-passing blocks
(`PlanetGnnMessageLayer`) and transformer blocks (`PlanetTransformerBlock`) are
wrapped with Flax `nn.remat` during the encoder forward pass. Parameter names and
output shapes are unchanged; only activation memory / backward recompute behavior
differs. Decoder and value heads are not checkpointed in v1.

## Checkpoint compatibility

Feature schema (`schema_version`, P/E/G dims) and encoder topology
(`encoder_backbone`) are validated independently on resume.

## Action sampling (rollout vs submission)

Factorized launches are chosen in a fixed scan over `max_moves_k` slots in
[`src/jax/action_sampling.py`](../../src/jax/action_sampling.py). Each slot samples
source, target slot, ship size, and a stop head; `sequence_active &= ~stop` ends
the scan early when the stop head fires.

| Actor | Entry | `deterministic` | Stop |
|-------|-------|-----------------|------|
| PPO learner | [`collect_rollout_jax`](../../src/jax/rollout/collect.py) | `False` (stochastic) | Bernoulli on `sigmoid(stop_logit)` |
| Self-play opponents (latest / historical) | [`_sample_policy_action*`](../../src/jax/action_sampling.py) via [`opponents/jax_actions/sampling.py`](../../src/opponents/jax_actions/sampling.py) | `cfg.opponents.self_play.deterministic` / `snapshot.deterministic` (default `true`) | `stop_prob >= 0.5` when deterministic |
| Kaggle / tournament / docker | [`compile_shielded_policy_act`](../../src/jax/submission_runtime.py) | `True` (argmax heads) | Same threshold as frozen opponents |

Submission and frozen checkpoints share greedy argmax plus the learned stop head.
They do not force all `max_moves_k` slots. Training rollouts for the learner stay
stochastic for exploration.
