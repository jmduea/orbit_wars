from __future__ import annotations

import jax.numpy as jnp

import jax
from src.config import TrainConfig
from src.features.registry import edge_k
from src.game.constants import MAX_PLANETS
from src.jax.env import JaxAction
from src.jax.features import TurnBatch
from src.jax.policy import edge_action_count
from src.jax.shield import default_edge_action_bucket_mask, ship_count_for_bucket_jax
from src.jax.ship_action import is_continuous_ship_mode, ship_count_for_action


def noop_edge_index(task_cfg) -> int:
    return MAX_PLANETS * edge_k(task_cfg)


def owned_planet_ships(game) -> jax.Array:
    player = game.player
    if player.ndim > 0:
        player = player[:, None]
    owned = game.planets.active & (game.planets.owner == player)
    return jnp.where(owned, game.planets.ships, 0.0)


def _launch_angle_for_edge(game, batch: TurnBatch, src_row, slot):
    src_x = game.planets.x[src_row]
    src_y = game.planets.y[src_row]
    tgt_id = batch.edge_tgt_ids[src_row, slot]
    match = game.planets.id == tgt_id
    tgt_x = jnp.sum(jnp.where(match, game.planets.x, 0.0))
    tgt_y = jnp.sum(jnp.where(match, game.planets.y, 0.0))
    return jnp.arctan2(tgt_y - src_y, tgt_x - src_x)


def _merge_identical_launches(
    source_id: jax.Array,
    tgt_id: jax.Array,
    angle: jax.Array,
    ships: jax.Array,
    valid: jax.Array,
    fleet_slots: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Collapse launches sharing ``(src_planet_id, tgt_planet_id)``, summing ships."""

    launch_steps = source_id.shape[0]
    out_src = jnp.full((fleet_slots,), -1, dtype=jnp.int32)
    out_tgt = jnp.full((fleet_slots,), -1, dtype=jnp.int32)
    out_angle = jnp.zeros((fleet_slots,), dtype=jnp.float32)
    out_ships = jnp.zeros((fleet_slots,), dtype=jnp.float32)
    out_valid = jnp.zeros((fleet_slots,), dtype=bool)
    out_count = jnp.array(0, dtype=jnp.int32)

    def process_step(i: int, state):
        out_src, out_tgt, out_angle, out_ships, out_valid, out_count = state
        is_valid = valid[i] & (ships[i] > 0.0)
        src = source_id[i]
        tgt = tgt_id[i]
        shp = ships[i]
        ang = angle[i]

        def try_slot(j: int, carry):
            was_merged, merged_ships = carry
            in_range = j < out_count
            match = is_valid & in_range & (out_src[j] == src) & (out_tgt[j] == tgt)
            merged_ships = merged_ships.at[j].set(
                jnp.where(match, merged_ships[j] + shp, merged_ships[j])
            )
            return was_merged | match, merged_ships

        was_merged, merged_ships = jax.lax.fori_loop(
            0,
            fleet_slots,
            try_slot,
            (jnp.array(False), out_ships),
        )

        slot = out_count
        can_append = is_valid & (~was_merged) & (slot < fleet_slots)
        out_src = out_src.at[slot].set(jnp.where(can_append, src, out_src[slot]))
        out_tgt = out_tgt.at[slot].set(jnp.where(can_append, tgt, out_tgt[slot]))
        out_angle = out_angle.at[slot].set(jnp.where(can_append, ang, out_angle[slot]))
        out_ships = merged_ships.at[slot].set(
            jnp.where(can_append, shp, merged_ships[slot])
        )
        out_valid = out_valid.at[slot].set(jnp.where(can_append, True, out_valid[slot]))
        out_count = out_count + jnp.where(can_append, 1, 0).astype(jnp.int32)
        return (out_src, out_tgt, out_angle, out_ships, out_valid, out_count)

    out_src, out_tgt, out_angle, out_ships, out_valid, _ = jax.lax.fori_loop(
        0,
        launch_steps,
        process_step,
        (out_src, out_tgt, out_angle, out_ships, out_valid, out_count),
    )
    return out_src, out_tgt, out_angle, out_ships, out_valid


def build_action_from_factored_batch(
    game,
    batch: TurnBatch,
    source_index: jax.Array,
    target_slot: jax.Array,
    ship_bucket: jax.Array,
    stop_flag: jax.Array,
    step_mask: jax.Array,
    cfg: TrainConfig,
    ship_fraction: jax.Array | None = None,
) -> JaxAction:
    """Build env actions from factorized source/slot launches gated by stop."""

    env_count = batch.planet_features.shape[0]
    source_index = source_index.reshape(env_count, -1)
    target_slot = target_slot.reshape(env_count, -1)
    ship_bucket = ship_bucket.reshape(env_count, -1)
    stop_flag = stop_flag.reshape(env_count, -1)
    step_mask = step_mask.reshape(env_count, -1)
    launch_steps = source_index.shape[-1]
    fleet_slots = cfg.task.max_fleets

    continuous = is_continuous_ship_mode(cfg)

    def build_env_action(
        game_row,
        batch_row,
        sources,
        slots,
        buckets,
        stops,
        active,
        fractions,
    ):
        def step_fn(remaining, step_inputs):
            src_row, slot, bucket, stop, step_active, fraction = step_inputs
            launch_fraction = jnp.where(continuous, fraction, jnp.zeros_like(fraction))
            valid = (
                step_active.astype(bool)
                & jnp.logical_not(stop.astype(bool))
                & (remaining[src_row] > 0.0)
            )
            if continuous:
                valid = valid & (launch_fraction > 0.0)
            else:
                valid = valid & (bucket > 0)
            requested = ship_count_for_action(
                remaining[src_row],
                bucket,
                launch_fraction if continuous else None,
                cfg,
            )
            launched = jnp.where(valid, jnp.minimum(remaining[src_row], requested), 0.0)
            remaining = remaining.at[src_row].set(
                jnp.maximum(remaining[src_row] - launched, 0.0)
            )
            src_id = batch_row.edge_src_ids[src_row]
            tgt_id = batch_row.edge_tgt_ids[src_row, slot]
            angle = _launch_angle_for_edge(game_row, batch_row, src_row, slot)
            return remaining, (src_id, tgt_id, angle, launched, valid)

        remaining = owned_planet_ships(game_row)
        _, steps = jax.lax.scan(
            step_fn,
            remaining,
            (
                jnp.moveaxis(sources, -1, 0),
                jnp.moveaxis(slots, -1, 0),
                jnp.moveaxis(buckets, -1, 0),
                jnp.moveaxis(stops, -1, 0),
                jnp.moveaxis(active, -1, 0),
                jnp.moveaxis(fractions, -1, 0),
            ),
        )
        source_id, tgt_id, angle, ships, valid = steps
        source_id = jnp.moveaxis(source_id, 0, -1).reshape(launch_steps)
        tgt_id = jnp.moveaxis(tgt_id, 0, -1).reshape(launch_steps)
        angle = jnp.moveaxis(angle, 0, -1).reshape(launch_steps)
        ships = jnp.moveaxis(ships, 0, -1).reshape(launch_steps)
        valid = jnp.moveaxis(valid, 0, -1).reshape(launch_steps)
        merged_src, _, merged_angle, merged_ships, merged_valid = (
            _merge_identical_launches(
                source_id,
                tgt_id,
                angle,
                ships,
                valid,
                fleet_slots,
            )
        )
        return JaxAction(
            source_id=merged_src,
            angle=merged_angle,
            ships=merged_ships,
            valid=merged_valid,
        )

    if ship_fraction is None:
        ship_fraction = jnp.zeros_like(ship_bucket, dtype=jnp.float32)
    return jax.vmap(build_env_action)(
        game,
        batch,
        source_index,
        target_slot,
        ship_bucket,
        stop_flag,
        step_mask,
        ship_fraction,
    )


def build_action_from_edge_batch(
    game,
    batch: TurnBatch,
    target_index: jax.Array,
    ship_bucket: jax.Array,
    cfg: TrainConfig,
) -> JaxAction:
    env_count = batch.planet_features.shape[0]
    k = edge_k(cfg.task)
    noop_idx = noop_edge_index(cfg.task)
    target_index = target_index.reshape(env_count, -1)
    ship_bucket = ship_bucket.reshape(env_count, -1)
    launch_steps = target_index.shape[-1]
    fleet_slots = cfg.task.max_fleets

    def build_env_action(game_row, batch_row, targets, buckets):
        def step_fn(remaining, step_inputs):
            flat_idx, bucket = step_inputs
            src_row = flat_idx // k
            valid = (flat_idx < noop_idx) & (bucket > 0) & (remaining[src_row] > 0.0)
            requested = ship_count_for_bucket_jax(
                remaining[src_row], bucket, cfg.task.ship_bucket_count
            )
            launched = jnp.where(valid, jnp.minimum(remaining[src_row], requested), 0.0)
            remaining = remaining.at[src_row].set(
                jnp.maximum(remaining[src_row] - launched, 0.0)
            )
            src_id = batch_row.edge_src_ids[src_row]
            angle = _launch_angle_for_edge(game_row, batch_row, src_row, flat_idx % k)
            return remaining, (src_id, angle, launched, valid)

        remaining = owned_planet_ships(game_row)
        _, steps = jax.lax.scan(
            step_fn,
            remaining,
            (jnp.moveaxis(targets, -1, 0), jnp.moveaxis(buckets, -1, 0)),
        )
        source_id, angle, ships, valid = steps
        source_id = jnp.moveaxis(source_id, 0, -1)
        angle = jnp.moveaxis(angle, 0, -1)
        ships = jnp.moveaxis(ships, 0, -1)
        valid = jnp.moveaxis(valid, 0, -1)
        flat_source = source_id.reshape(launch_steps)
        flat_angle = angle.reshape(launch_steps)
        flat_ships = ships.reshape(launch_steps)
        flat_valid = valid.reshape(launch_steps)
        action_width = min(launch_steps, fleet_slots)
        pad = fleet_slots - action_width
        return JaxAction(
            source_id=jnp.pad(flat_source[:action_width], (0, pad), constant_values=-1),
            angle=jnp.pad(flat_angle[:action_width], (0, pad), constant_values=0.0),
            ships=jnp.pad(flat_ships[:action_width], (0, pad), constant_values=0.0),
            valid=jnp.pad(flat_valid[:action_width], (0, pad), constant_values=False),
        )

    return jax.vmap(build_env_action)(game, batch, target_index, ship_bucket)


def build_random_action_from_edge_batch(
    key: jax.Array,
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxAction:
    env_count = batch.planet_features.shape[0]
    k = edge_k(cfg.task)
    edge_count = edge_action_count(cfg.task)
    key_target, key_bucket = jax.random.split(key)
    flat_mask = jnp.concatenate(
        [
            batch.edge_mask.reshape(env_count, MAX_PLANETS * k),
            jnp.ones((env_count, 1), dtype=bool),
        ],
        axis=1,
    )
    if ship_bucket_mask is None:
        flat_bucket_mask = default_edge_action_bucket_mask(
            flat_mask, cfg.task.ship_bucket_count
        )
    else:
        flat_bucket_mask = ship_bucket_mask
    real_bucket_mask = flat_bucket_mask & (
        jnp.arange(cfg.task.ship_bucket_count, dtype=jnp.int32)[None, None, :] > 0
    )
    real_edge = (
        flat_mask
        & real_bucket_mask.any(axis=-1)
        & (jnp.arange(edge_count, dtype=jnp.int32)[None, :] < noop_edge_index(cfg.task))
    )
    has_target = real_edge.any(axis=-1)
    target_logits = jnp.where(real_edge, 0.0, jnp.finfo(jnp.float32).min)
    target = jnp.where(
        has_target,
        jax.random.categorical(key_target, target_logits, axis=-1),
        jnp.full((env_count,), noop_edge_index(cfg.task), dtype=jnp.int32),
    )
    selected_bucket_mask = jnp.take_along_axis(
        flat_bucket_mask,
        target[:, None, None].repeat(cfg.task.ship_bucket_count, axis=-1),
        axis=1,
    ).squeeze(axis=1)
    bucket_logits = jnp.where(selected_bucket_mask, 0.0, jnp.finfo(jnp.float32).min)
    bucket = jax.random.categorical(key_bucket, bucket_logits, axis=-1)
    bucket = jnp.where(has_target, bucket, jnp.zeros_like(bucket))
    return build_action_from_edge_batch(
        game, batch, target[:, None], bucket[:, None], cfg
    )


def _edge_scripted_context(
    batch: TurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None,
):
    from src.features.catalog.edge import intercept_anchor_label
    from src.features.registry import edge_feature_schema

    env_count = batch.planet_features.shape[0]
    k = edge_k(cfg.task)
    flat_count = MAX_PLANETS * k
    flat_mask = batch.edge_mask.reshape(env_count, flat_count)
    edge_schema = edge_feature_schema(cfg.task)
    owner_slice = edge_schema.slice("target_owner_slot")
    owner_slots = batch.edge_features[..., owner_slice].reshape(
        env_count, flat_count, 4
    )
    anchor_speeds = tuple(float(s) for s in cfg.task.intercept_anchors)
    distance_slices = [
        edge_schema.slice(f"intercept_distance_{intercept_anchor_label(speed)}")
        for speed in anchor_speeds
    ]
    distance_stack = jnp.stack(
        [
            batch.edge_features[..., feature_slice].reshape(env_count, flat_count)
            for feature_slice in distance_slices
        ],
        axis=-1,
    )
    distance = jnp.min(distance_stack, axis=-1)
    if ship_bucket_mask is None:
        full_mask = jnp.concatenate(
            [flat_mask, jnp.ones((env_count, 1), dtype=bool)], axis=1
        )
        bucket_mask = default_edge_action_bucket_mask(
            full_mask, cfg.task.ship_bucket_count
        )
    else:
        bucket_mask = ship_bucket_mask
    real_bucket = bucket_mask[:, :flat_count, :][..., 1:].any(axis=-1)
    valid = flat_mask & real_bucket
    owner_sum = owner_slots.sum(axis=-1)
    neutral = valid & (owner_sum < 0.5)
    enemy = valid & (owner_sum > 0.5) & (owner_slots[..., 0] < 0.5)
    return {
        "env_count": env_count,
        "flat_count": flat_count,
        "valid": valid,
        "neutral": neutral,
        "enemy": enemy,
        "distance": distance,
        "bucket_mask": bucket_mask,
    }


def _bucket_for_flat_target(
    flat_target: jax.Array,
    bucket_mask: jax.Array,
    has_target: jax.Array,
    cfg: TrainConfig,
    *,
    conservative: bool,
) -> jax.Array:
    selected_bucket_mask = jnp.take_along_axis(
        bucket_mask,
        flat_target[:, None, None].repeat(cfg.task.ship_bucket_count, axis=-1),
        axis=1,
    ).squeeze(axis=1)
    bucket_ids = jnp.arange(cfg.task.ship_bucket_count, dtype=jnp.int32)
    if conservative:
        nonzero = selected_bucket_mask & (bucket_ids[None, :] > 0)
        bucket = jnp.argmax(nonzero.astype(jnp.int32), axis=-1)
        bucket = jnp.where(has_target & nonzero.any(axis=-1), bucket, 0)
    else:
        bucket = jnp.max(
            jnp.where(selected_bucket_mask, bucket_ids[None, :], 0), axis=-1
        )
        bucket = jnp.where(has_target, bucket, 0)
    return bucket


def _build_scripted_edge_action(
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    *,
    pick_mask: jax.Array,
    distance: jax.Array,
    bucket_mask: jax.Array,
    use_nearest: bool,
    conservative_bucket: bool,
) -> JaxAction:
    noop_idx = noop_edge_index(cfg.task)
    if use_nearest:
        masked_distance = jnp.where(pick_mask, distance, jnp.inf)
        flat_target = jnp.argmin(masked_distance, axis=-1)
    else:
        flat_target = jnp.argmax(pick_mask.astype(jnp.int32), axis=-1)
    has_target = pick_mask.any(axis=-1)
    bucket = _bucket_for_flat_target(
        flat_target,
        bucket_mask,
        has_target,
        cfg,
        conservative=conservative_bucket,
    )
    target = jnp.where(has_target, flat_target, noop_idx)
    bucket = jnp.where(has_target, bucket, 0)
    return build_action_from_edge_batch(
        game, batch, target[:, None], bucket[:, None], cfg
    )


def build_sniper_action_from_edge_batch(
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxAction:
    ctx = _edge_scripted_context(batch, cfg, ship_bucket_mask)
    return _build_scripted_edge_action(
        game,
        batch,
        cfg,
        pick_mask=ctx["valid"],
        distance=ctx["distance"],
        bucket_mask=ctx["bucket_mask"],
        use_nearest=True,
        conservative_bucket=False,
    )


def build_turtle_action_from_edge_batch(
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxAction:
    ctx = _edge_scripted_context(batch, cfg, ship_bucket_mask)
    return _build_scripted_edge_action(
        game,
        batch,
        cfg,
        pick_mask=ctx["neutral"],
        distance=ctx["distance"],
        bucket_mask=ctx["bucket_mask"],
        use_nearest=False,
        conservative_bucket=True,
    )


def build_opportunistic_action_from_edge_batch(
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
    ship_bucket_mask: jax.Array | None = None,
) -> JaxAction:
    ctx = _edge_scripted_context(batch, cfg, ship_bucket_mask)
    return _build_scripted_edge_action(
        game,
        batch,
        cfg,
        pick_mask=ctx["enemy"],
        distance=ctx["distance"],
        bucket_mask=ctx["bucket_mask"],
        use_nearest=False,
        conservative_bucket=False,
    )


def build_noop_action_from_edge_batch(
    game,
    batch: TurnBatch,
    cfg: TrainConfig,
) -> JaxAction:
    """Build a pass/no-op action that launches no fleets."""

    env_count = batch.planet_features.shape[0]
    noop_idx = noop_edge_index(cfg.task)
    noop_target = jnp.full((env_count, 1), noop_idx, dtype=jnp.int32)
    noop_bucket = jnp.zeros((env_count, 1), dtype=jnp.int32)
    return build_action_from_edge_batch(game, batch, noop_target, noop_bucket, cfg)
