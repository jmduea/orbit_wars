from __future__ import annotations

import flax
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .config import TrainConfig
from .features import candidate_feature_dim, global_feature_dim, self_feature_dim
from .jax_env import (
    JaxAction,
    JaxEnvState,
    assign_learner_players,
    batched_reset,
    batched_step,
    batched_step_multi_player,
)
from .jax_features import JaxTurnBatch, encode_turn
from .jax_policy import action_log_prob_and_entropy, sample_actions
from .opponent_pool import (
    OPPONENT_HISTORICAL,
    OPPONENT_LATEST,
    OPPONENT_RANDOM,
    OPPONENT_SCRIPTED_SNIPER,
    OpponentRegistry,
    OpponentRegistryConfig,
    sample_opponent_type_ids_jax,
)

_MIN_JAX_UPDATE_CHUNK_ROWS = 8192


class JaxTransitionBatch(NamedTuple):
    """Rollout data consumed by the JAX PPO update.

    Arrays keep rollout, environment, and source-planet dimensions until the
    update step flattens them. ``decision_mask`` identifies valid learner-owned
    source rows that should contribute to PPO losses.
    """

    self_features: jax.Array
    candidate_features: jax.Array
    global_features: jax.Array
    candidate_mask: jax.Array
    decision_mask: jax.Array
    target_index: jax.Array
    ship_bucket: jax.Array
    log_prob: jax.Array
    returns: jax.Array
    advantages: jax.Array


@flax.struct.dataclass
class JaxTrainState:
    """Minimal immutable train state for Flax parameters and Optax state."""

    params: dict
    opt_state: optax.OptState
    optimizer: optax.GradientTransformation = flax.struct.field(pytree_node=False)


def init_train_state(key: jax.Array, policy: object, cfg: TrainConfig) -> JaxTrainState:
    """Initialize policy parameters and optimizer state for JAX PPO."""

    dummy_self = jnp.zeros((1, self_feature_dim()), dtype=jnp.float32)
    dummy_candidate = jnp.zeros(
        (1, cfg.env.candidate_count, candidate_feature_dim()), dtype=jnp.float32
    )
    dummy_global = jnp.zeros((1, global_feature_dim()), dtype=jnp.float32)
    dummy_mask = jnp.ones((1, cfg.env.candidate_count), dtype=bool)
    params = policy.init(key, dummy_self, dummy_candidate, dummy_global, dummy_mask)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm), optax.adam(cfg.ppo.lr)
    )
    return JaxTrainState(
        params=params, opt_state=optimizer.init(params), optimizer=optimizer
    )


def flatten_batch(
    batch: JaxTurnBatch,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Flatten environment/source dimensions into policy decision rows."""

    return (
        batch.self_features.reshape(-1, self_feature_dim()),
        batch.candidate_features.reshape(
            -1, batch.candidate_features.shape[-2], candidate_feature_dim()
        ),
        batch.global_features.reshape(-1, global_feature_dim()),
        batch.candidate_mask.reshape(-1, batch.candidate_mask.shape[-1]),
        batch.decision_mask.reshape(-1),
    )


def ship_count_for_bucket_jax(
    available_ships: jax.Array, bucket: jax.Array, bucket_count: int
) -> jax.Array:
    """Convert discrete ship buckets into concrete launched ship counts."""

    fraction = jnp.where(
        bucket <= 0, 0.0, bucket.astype(jnp.float32) / float(max(bucket_count - 1, 1))
    )
    ships = jnp.ceil(available_ships * fraction)
    ships = jnp.minimum(available_ships, jnp.maximum(1.0, ships))
    return jnp.where((available_ships <= 0.0) | (fraction <= 0.0), 0.0, ships)


def build_action_from_batch(
    batch: JaxTurnBatch,
    target_index: jax.Array,
    ship_bucket: jax.Array,
    cfg: TrainConfig,
) -> JaxAction:
    """Build fixed-size JAX action buffers from per-source policy choices.

    Only valid source rows with non-no-op targets and positive ship buckets are
    emitted. If ``max_fleets`` is smaller than ``max_planets``, extra source rows
    are clipped so the returned arrays always match the configured fleet buffer.
    """

    env_count = batch.self_features.shape[0]
    planet_count = batch.self_features.shape[1]
    target_index = target_index.reshape(env_count, planet_count)
    ship_bucket = ship_bucket.reshape(env_count, planet_count)
    chosen_mask = jnp.take_along_axis(
        batch.candidate_mask, target_index[..., None], axis=-1
    ).squeeze(-1)
    chosen_angle = jnp.take_along_axis(
        batch.target_angles, target_index[..., None], axis=-1
    ).squeeze(-1)
    ships = ship_count_for_bucket_jax(
        batch.source_ships, ship_bucket, cfg.env.ship_bucket_count
    )
    valid = (
        batch.decision_mask
        & chosen_mask
        & (target_index > 0)
        & (ship_bucket > 0)
        & (ships > 0.0)
    )
    fleet_slots = cfg.env.max_fleets
    action_width = min(planet_count, fleet_slots)
    pad = fleet_slots - action_width
    source_id = jnp.pad(
        batch.source_ids[:, :action_width], ((0, 0), (0, pad)), constant_values=-1
    )
    angle = jnp.pad(
        chosen_angle[:, :action_width], ((0, 0), (0, pad)), constant_values=0.0
    )
    ships = jnp.pad(ships[:, :action_width], ((0, 0), (0, pad)), constant_values=0.0)
    valid = jnp.pad(valid[:, :action_width], ((0, 0), (0, pad)), constant_values=False)
    return JaxAction(source_id=source_id, angle=angle, ships=ships, valid=valid)


def build_random_action_from_batch(
    key: jax.Array,
    batch: JaxTurnBatch,
    cfg: TrainConfig,
) -> JaxAction:
    """Sample a JAX-native random opponent action for each environment."""

    env_count = batch.self_features.shape[0]
    planet_count = batch.self_features.shape[1]
    key_target, key_bucket = jax.random.split(key)
    flat_mask = batch.candidate_mask.reshape(-1, cfg.env.candidate_count)
    real_candidate = flat_mask & (
        jnp.arange(cfg.env.candidate_count, dtype=jnp.int32)[None, :] > 0
    )
    has_target = real_candidate.any(axis=-1)
    target_logits = jnp.where(real_candidate, 0.0, jnp.finfo(jnp.float32).min)
    target = jnp.where(
        has_target,
        jax.random.categorical(key_target, target_logits, axis=-1),
        jnp.zeros((env_count * planet_count,), dtype=jnp.int32),
    )
    bucket = jax.random.randint(
        key_bucket,
        (env_count * planet_count,),
        minval=1,
        maxval=max(cfg.env.ship_bucket_count, 2),
        dtype=jnp.int32,
    )
    bucket = jnp.where(has_target, bucket, jnp.zeros_like(bucket))
    return build_action_from_batch(batch, target, bucket, cfg)




def build_sniper_action_from_batch(batch: JaxTurnBatch, cfg: TrainConfig) -> JaxAction:
    """JAX-compatible scripted sniper: use nearest candidate slot aggressively."""

    real_candidate_mask = batch.candidate_mask & (
        jnp.arange(batch.candidate_mask.shape[-1], dtype=jnp.int32)[None, None, :] > 0
    )
    nearest_slot = jnp.argmax(real_candidate_mask.astype(jnp.int32), axis=-1)
    has_target = real_candidate_mask.any(axis=-1)
    target = jnp.where(has_target, nearest_slot, 0).reshape(-1)
    bucket = jnp.full_like(target, max(cfg.env.ship_bucket_count - 1, 1))
    bucket = jnp.where(has_target.reshape(-1), bucket, 0)
    return build_action_from_batch(batch, target, bucket, cfg)
def _sample_policy_action_with_params(
    key: jax.Array,
    batch: JaxTurnBatch,
    params: dict,
    policy: object,
    cfg: TrainConfig,
    *,
    deterministic: bool,
) -> JaxAction:
    """Sample a fixed-size action buffer from a JAX policy parameter set."""

    flat_self, flat_candidate, flat_global, flat_mask, _flat_decision = flatten_batch(
        batch
    )
    output = policy.apply(params, flat_self, flat_candidate, flat_global, flat_mask)
    target, bucket, _log_prob, _entropy = sample_actions(
        key, output, deterministic=deterministic
    )
    return build_action_from_batch(batch, target, bucket, cfg)


def _sample_policy_action(
    key: jax.Array,
    batch: JaxTurnBatch,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    *,
    deterministic: bool,
) -> JaxAction:
    """Sample a fixed-size action buffer from the trainable JAX policy."""

    return _sample_policy_action_with_params(
        key,
        batch,
        train_state.params,
        policy,
        cfg,
        deterministic=deterministic,
    )


def _select_env_action(
    condition: jax.Array,
    true_action: JaxAction,
    false_action: JaxAction,
) -> JaxAction:
    """Select between two batched actions independently for each environment."""

    return jax.tree.map(
        lambda true, false: jnp.where(
            condition.reshape((condition.shape[0],) + (1,) * (true.ndim - 1)),
            true,
            false,
        ),
        true_action,
        false_action,
    )


def _stack_player_actions(player_actions: tuple[JaxAction, ...]) -> JaxAction:
    """Stack per-player batched actions into batched_step_multi_player layout."""

    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *player_actions)


def collect_rollout_jax(
    key: jax.Array,
    env_state: JaxEnvState,
    turn_batch: JaxTurnBatch,
    train_state: JaxTrainState,
    policy: object,
    cfg: TrainConfig,
    opponent_params_by_player: tuple[dict, ...] | None = None,
    update: int = 0,
) -> tuple[
    jax.Array, JaxEnvState, JaxTurnBatch, JaxTransitionBatch, dict[str, jax.Array]
]:
    """Collect one fixed-length rollout entirely in JAX.

    The function is designed to be wrapped in ``jax.jit`` by the training loop.
    It samples learner actions, generates the configured opponent actions,
    advances the vectorized JAX environment, resets completed episodes, and
    returns PPO transitions plus rollout metrics.
    """

    env_indices = jnp.arange(turn_batch.self_features.shape[0], dtype=jnp.int32)

    def scan_step(carry, _):
        key, state, batch = carry
        key, learner_key, opp_key, reset_key = jax.random.split(key, 4)
        flat_self, flat_candidate, flat_global, flat_mask, flat_decision = (
            flatten_batch(batch)
        )
        output = policy.apply(
            train_state.params, flat_self, flat_candidate, flat_global, flat_mask
        )
        target, bucket, log_prob, _entropy = sample_actions(
            learner_key, output, deterministic=False
        )
        learner_action = build_action_from_batch(batch, target, bucket, cfg)

        if cfg.env.player_count == 2:
            opp_game = state.game._replace(
                player=(1 - state.learner_player).astype(jnp.int32)
            )
            opp_batch = jax.vmap(lambda game: encode_turn(game, cfg.env))(opp_game)
            if cfg.opponent == "self":
                opponent_action = _sample_policy_action(
                    opp_key,
                    opp_batch,
                    train_state,
                    policy,
                    cfg,
                    deterministic=cfg.self_play_deterministic,
                )
            elif cfg.opponent == "random":
                opponent_action = build_random_action_from_batch(
                    opp_key, opp_batch, cfg
                )
            else:
                raise ValueError(
                    "JAX training supports opponent='self' or opponent='random', "
                    f"got {cfg.opponent!r}."
                )

            next_state, result = batched_step(
                state, learner_action, opponent_action, cfg.env
            )
        elif cfg.env.player_count == 4:
            player_actions = []
            registry = OpponentRegistry(
                OpponentRegistryConfig(
                    weights=dict(cfg.opponent_mix.weights),
                    temperature=cfg.opponent_mix.temperature,
                    curriculum=list(cfg.opponent_mix.curriculum),
                )
            )
            ids_py, probs_py = registry.ids_and_probs(update)
            ids = jnp.asarray(ids_py, dtype=jnp.int32)
            probs = jnp.asarray(probs_py, dtype=jnp.float32)
            opponent_type_ids = sample_opponent_type_ids_jax(
                jax.random.fold_in(opp_key, 9973),
                state.learner_player.shape[0],
                cfg.env.player_count,
                ids=ids,
                probs=probs,
            )
            for player_id in range(cfg.env.player_count):
                player_game = state.game._replace(
                    player=jnp.full_like(state.game.step, player_id, dtype=jnp.int32)
                )
                player_batch = jax.vmap(lambda game: encode_turn(game, cfg.env))(
                    player_game
                )
                player_key = jax.random.fold_in(opp_key, player_id)
                slot_type = opponent_type_ids[:, player_id]
                if cfg.opponent == "self":
                    opponent_params = (
                        train_state.params
                        if opponent_params_by_player is None
                        else opponent_params_by_player[player_id]
                    )
                    current_action = _sample_policy_action_with_params(
                        player_key,
                        player_batch,
                        opponent_params,
                        policy,
                        cfg,
                        deterministic=cfg.self_play_deterministic,
                    )
                    historical_action = current_action
                    random_action = build_random_action_from_batch(
                        jax.random.fold_in(player_key, cfg.env.player_count),
                        player_batch,
                        cfg,
                    )
                    scripted_action = build_sniper_action_from_batch(player_batch, cfg)
                    use_latest = slot_type == OPPONENT_LATEST
                    use_historical = slot_type == OPPONENT_HISTORICAL
                    use_scripted = slot_type == OPPONENT_SCRIPTED_SNIPER
                    action = _select_env_action(use_latest, current_action, random_action)
                    action = _select_env_action(use_historical, historical_action, action)
                    opponent_action = _select_env_action(use_scripted, scripted_action, action)
                elif cfg.opponent == "random":
                    opponent_action = build_random_action_from_batch(
                        player_key, player_batch, cfg
                    )
                else:
                    raise ValueError(
                        "JAX training supports opponent='self' or opponent='random', "
                        f"got {cfg.opponent!r}."
                    )

                is_learner_player = state.learner_player == player_id
                player_actions.append(
                    _select_env_action(
                        is_learner_player, learner_action, opponent_action
                    )
                )

            multi_player_action = _stack_player_actions(tuple(player_actions))
            next_state, result = batched_step_multi_player(
                state, multi_player_action, cfg.env
            )
        else:
            raise ValueError(
                "JAX PPO rollout supports env.player_count of 2 or 4, "
                f"got {cfg.env.player_count}."
            )
        reset_keys = jax.random.split(reset_key, batch.self_features.shape[0])
        reset_states, reset_batches = batched_reset(reset_keys, cfg.env)
        reset_episode_counts = state.episode_count + result.done.astype(jnp.int32)
        reset_states, reset_batches = assign_learner_players(
            reset_states,
            env_indices,
            reset_episode_counts,
            cfg.env,
            cfg.alternate_player_sides,
        )

        def maybe_reset(new, old):
            cond = result.done.reshape(result.done.shape + (1,) * (old.ndim - 1))
            return jnp.where(cond, new, old)

        next_state = jax.tree.map(maybe_reset, reset_states, next_state)
        next_batch = jax.tree.map(maybe_reset, reset_batches, result.batch)
        transition = {
            "self_features": batch.self_features,
            "candidate_features": batch.candidate_features,
            "global_features": batch.global_features,
            "candidate_mask": batch.candidate_mask,
            "decision_mask": flat_decision.reshape(batch.decision_mask.shape),
            "target_index": target.reshape(batch.decision_mask.shape),
            "ship_bucket": bucket.reshape(batch.decision_mask.shape),
            "log_prob": log_prob.reshape(batch.decision_mask.shape),
            "value": output.value.reshape(batch.decision_mask.shape),
            "reward": result.reward,
            "done": result.done,
            "terminal_is_first": result.terminal_is_first,
            "terminal_placement": result.terminal_placement,
            "terminal_score_share": result.terminal_score_share,
            "terminal_survival_time": result.terminal_survival_time,
        }
        return (key, next_state, next_batch), transition

    (key, env_state, turn_batch), data = jax.lax.scan(
        scan_step, (key, env_state, turn_batch), None, length=cfg.ppo.rollout_steps
    )
    returns_step = discounted_returns(data["reward"], data["done"], cfg.ppo.gamma)
    returns = jnp.broadcast_to(returns_step[..., None], data["value"].shape)
    advantages = returns - data["value"]
    transitions = JaxTransitionBatch(
        self_features=data["self_features"],
        candidate_features=data["candidate_features"],
        global_features=data["global_features"],
        candidate_mask=data["candidate_mask"],
        decision_mask=data["decision_mask"],
        target_index=data["target_index"],
        ship_bucket=data["ship_bucket"],
        log_prob=data["log_prob"],
        returns=returns,
        advantages=advantages,
    )
    opponent_slots = jnp.array(
        cfg.ppo.rollout_steps
        * turn_batch.self_features.shape[0]
        * max(cfg.env.player_count - 1, 0),
        dtype=jnp.float32,
    )
    mode = (
        cfg.multi_opponent_mode.strip().lower()
        if cfg.self_play_enabled
        else "shared_current"
    )
    snapshot_share = (
        jnp.array(1.0, dtype=jnp.float32)
        if (
            cfg.opponent == "self"
            and mode == "sampled_pool"
            and opponent_params_by_player is not None
        )
        else jnp.array(0.0, dtype=jnp.float32)
    )
    current_share = (
        jnp.array(1.0, dtype=jnp.float32)
        if (
            cfg.opponent == "self"
            and (
                mode == "shared_current"
                or (mode == "sampled_pool" and opponent_params_by_player is None)
            )
        )
        else (
            jnp.array(
                min(max(cfg.self_play_latest_probability, 0.0), 1.0), dtype=jnp.float32
            )
            if cfg.opponent == "self" and mode == "mixed"
            else jnp.array(0.0, dtype=jnp.float32)
        )
    )
    random_share = (
        jnp.array(1.0, dtype=jnp.float32)
        if cfg.opponent == "random"
        else (
            (1.0 - current_share)
            if cfg.opponent == "self" and mode == "mixed"
            else jnp.array(0.0, dtype=jnp.float32)
        )
    )
    done_float = data["done"].astype(jnp.float32)
    episode_done = done_float.sum()
    episodes_2p = jnp.where(cfg.env.player_count == 2, episode_done, 0.0)
    episodes_4p = jnp.where(cfg.env.player_count == 4, episode_done, 0.0)
    first_place_sum = (data["terminal_is_first"] * done_float).sum()
    placement_4p_sum = jnp.where(
        cfg.env.player_count == 4, (data["terminal_placement"] * done_float).sum(), 0.0
    )
    survival_time_sum = (data["terminal_survival_time"] * done_float).sum()
    score_share_sum = (data["terminal_score_share"] * done_float).sum()
    metrics = {
        "env_steps": jnp.array(
            cfg.ppo.rollout_steps * turn_batch.self_features.shape[0], dtype=jnp.float32
        ),
        "samples": transitions.decision_mask.astype(jnp.float32).sum(),
        "episode_done": episode_done,
        "episodes_2p": episodes_2p,
        "episodes_4p": episodes_4p,
        "wins_2p": jnp.where(cfg.env.player_count == 2, first_place_sum, 0.0),
        "first_places_4p": jnp.where(cfg.env.player_count == 4, first_place_sum, 0.0),
        "placement_4p_sum": placement_4p_sum,
        "survival_time_sum": survival_time_sum,
        "score_share_sum": score_share_sum,
        "win_rate_2p": jnp.where(episodes_2p > 0.0, first_place_sum / episodes_2p, 0.0),
        "first_place_rate_4p": jnp.where(
            episodes_4p > 0.0, first_place_sum / episodes_4p, 0.0
        ),
        "average_placement_4p": jnp.where(
            episodes_4p > 0.0, placement_4p_sum / episodes_4p, 0.0
        ),
        "survival_time": jnp.where(
            episode_done > 0.0, survival_time_sum / episode_done, 0.0
        ),
        "score_share": jnp.where(
            episode_done > 0.0, score_share_sum / episode_done, 0.0
        ),
        "opponent_current_slots": opponent_slots * current_share,
        "opponent_random_slots": opponent_slots * random_share,
        "opponent_snapshot_slots": opponent_slots * snapshot_share,
    }
    return key, env_state, turn_batch, transitions, metrics


def concatenate_transition_batches(
    batches: tuple[JaxTransitionBatch, ...] | list[JaxTransitionBatch],
) -> JaxTransitionBatch:
    """Concatenate compatible rollout batches along the environment axis.

    Mixed-format JAX training uses one compiled collector per static player
    count. The resulting transition tensors share rollout and feature shapes,
    so PPO can consume a single larger batch by joining their independent
    environment axes.
    """

    if not batches:
        raise ValueError("At least one transition batch is required.")
    if len(batches) == 1:
        return batches[0]
    reference_shape = batches[0].self_features.shape
    for batch in batches[1:]:
        if (
            batch.self_features.shape[0] != reference_shape[0]
            or batch.self_features.shape[2:] != reference_shape[2:]
        ):
            raise ValueError(
                "Transition batches must share rollout and feature dimensions "
                "to concatenate along the environment axis."
            )
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *batches)


def discounted_returns(rewards: jax.Array, done: jax.Array, gamma: float) -> jax.Array:
    """Compute discounted returns over rollout time with terminal resets."""

    def step(carry, item):
        reward, terminal = item
        carry = reward + gamma * carry * (1.0 - terminal.astype(jnp.float32))
        return carry, carry

    _, out = jax.lax.scan(
        step, jnp.zeros_like(rewards[-1]), (rewards, done), reverse=True
    )
    return out


def ppo_update_jax(
    train_state: JaxTrainState,
    policy: object,
    batch: JaxTransitionBatch,
    cfg: TrainConfig,
) -> tuple[JaxTrainState, dict[str, jax.Array]]:
    """Apply one PPO epoch using memory-bounded minibatches.

    Rollouts can be large when benchmarking long attention-policy runs. Running
    the policy over every rollout row in a single XLA program forces the GPU to
    materialize attention intermediates for the entire rollout at once. Instead,
    flatten once, pad to static memory chunks, and scan sequential optimizer
    steps over those chunks. The chunk size honors large configured minibatches
    but does not go below ``_MIN_JAX_UPDATE_CHUNK_ROWS`` so long rollouts do not
    devolve into thousands of tiny GPU launches.
    """

    mask = batch.decision_mask.reshape(-1).astype(jnp.float32)
    self_features = batch.self_features.reshape(-1, self_feature_dim())
    candidate_features = batch.candidate_features.reshape(
        -1, cfg.env.candidate_count, candidate_feature_dim()
    )
    global_features = batch.global_features.reshape(-1, global_feature_dim())
    candidate_mask = batch.candidate_mask.reshape(-1, cfg.env.candidate_count)
    target = batch.target_index.reshape(-1)
    bucket = batch.ship_bucket.reshape(-1)
    old_log_prob = batch.log_prob.reshape(-1)
    returns = batch.returns.reshape(-1)
    advantages = batch.advantages.reshape(-1)
    advantage_mean = masked_mean(advantages, mask)
    advantages = (advantages - advantage_mean) / jnp.sqrt(
        masked_mean((advantages - advantage_mean) ** 2, mask) + 1e-8
    )

    total_rows = mask.shape[0]
    minibatch_size = min(
        max(int(cfg.ppo.minibatch_size), _MIN_JAX_UPDATE_CHUNK_ROWS), total_rows
    )
    minibatch_count = (total_rows + minibatch_size - 1) // minibatch_size
    minibatches = {
        "mask": _reshape_minibatches(mask, minibatch_count, minibatch_size, 0.0),
        "self_features": _reshape_minibatches(
            self_features, minibatch_count, minibatch_size, 0.0
        ),
        "candidate_features": _reshape_minibatches(
            candidate_features, minibatch_count, minibatch_size, 0.0
        ),
        "global_features": _reshape_minibatches(
            global_features, minibatch_count, minibatch_size, 0.0
        ),
        "candidate_mask": _reshape_minibatches(
            candidate_mask, minibatch_count, minibatch_size, False
        ),
        "target": _reshape_minibatches(target, minibatch_count, minibatch_size, 0),
        "bucket": _reshape_minibatches(bucket, minibatch_count, minibatch_size, 0),
        "old_log_prob": _reshape_minibatches(
            old_log_prob, minibatch_count, minibatch_size, 0.0
        ),
        "returns": _reshape_minibatches(returns, minibatch_count, minibatch_size, 0.0),
        "advantages": _reshape_minibatches(
            advantages, minibatch_count, minibatch_size, 0.0
        ),
    }

    def update_minibatch(carry, minibatch):
        params, opt_state = carry

        def loss_fn(params):
            output = policy.apply(
                params,
                minibatch["self_features"],
                minibatch["candidate_features"],
                minibatch["global_features"],
                minibatch["candidate_mask"],
            )
            new_log_prob, entropy = action_log_prob_and_entropy(
                output, minibatch["target"], minibatch["bucket"]
            )
            ratio = jnp.exp(new_log_prob - minibatch["old_log_prob"])
            policy_loss = -masked_mean(
                jnp.minimum(
                    minibatch["advantages"] * ratio,
                    minibatch["advantages"]
                    * jnp.clip(ratio, 1.0 - cfg.ppo.clip_coef, 1.0 + cfg.ppo.clip_coef),
                ),
                minibatch["mask"],
            )
            value_loss = 0.5 * masked_mean(
                (minibatch["returns"] - output.value) ** 2, minibatch["mask"]
            )
            entropy_loss = masked_mean(entropy, minibatch["mask"])
            loss = (
                policy_loss
                + cfg.ppo.vf_coef * value_loss
                - cfg.ppo.ent_coef * entropy_loss
            )
            return loss, {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_loss,
                "loss": loss,
                "sample_count": minibatch["mask"].sum(),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = train_state.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = dict(metrics)
        metrics["total_loss"] = loss
        return (params, opt_state), metrics

    (params, opt_state), metrics_by_minibatch = jax.lax.scan(
        update_minibatch, (train_state.params, train_state.opt_state), minibatches
    )
    metric_weights = jnp.where(metrics_by_minibatch["sample_count"] > 0.0, 1.0, 0.0)
    metric_denominator = jnp.maximum(metric_weights.sum(), 1.0)
    metrics = {
        name: (values * metric_weights).sum() / metric_denominator
        for name, values in metrics_by_minibatch.items()
        if name != "sample_count"
    }
    metrics["minibatches"] = jnp.array(minibatch_count, dtype=jnp.float32)
    return (
        JaxTrainState(
            params=params, opt_state=opt_state, optimizer=train_state.optimizer
        ),
        metrics,
    )


def _reshape_minibatches(
    value: jax.Array,
    minibatch_count: int,
    minibatch_size: int,
    padding_value: float | int | bool,
) -> jax.Array:
    """Pad and reshape a flat leading axis into static minibatches."""

    padded_rows = minibatch_count * minibatch_size
    pad_rows = padded_rows - value.shape[0]
    pad_width = [(0, pad_rows)] + [(0, 0)] * (value.ndim - 1)
    padded = jnp.pad(value, pad_width, constant_values=padding_value)
    return padded.reshape((minibatch_count, minibatch_size) + value.shape[1:])


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    """Return the mean of ``x`` over entries where ``mask`` is non-zero."""

    return (x * mask).sum() / jnp.maximum(mask.sum(), 1.0)
