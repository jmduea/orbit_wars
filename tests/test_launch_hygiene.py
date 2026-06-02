from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.features.registry import edge_k
from src.game.constants import MAX_PLANETS
from src.jax.action_codec import source_mask_from_bucket_mask_and_ships
from src.jax.action_sampling import _sample_factored_step_from_logits
from src.jax.env import batched_reset
from src.jax.features import encode_turn
from src.jax.launch_hygiene import (
    apply_cumulative_forbidden_to_shield,
    apply_launch_to_cumulative_forbidden,
    build_hygiene_lookups,
    compose_hygiene_with_shield_mask,
    empty_cumulative_forbidden,
    launch_valid_at_step,
)


def _task_cfg(**kwargs) -> TaskConfig:
    base = dict(candidate_count=4, ship_bucket_count=4, max_fleets=8)
    base.update(kwargs)
    return TaskConfig(**base)


def _train_cfg(**kwargs) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.max_moves_k = 3
    cfg.task = _task_cfg(**kwargs.pop("task", {}))
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


@pytest.mark.jax
def test_empty_prefix_leaves_shield_mask_unchanged() -> None:
    cfg = _train_cfg()
    _, batch = batched_reset(jax.random.split(jax.random.PRNGKey(0), 1), cfg.task)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    shield = jnp.ones((1, MAX_PLANETS, k, buckets), dtype=bool)
    source = jnp.zeros((1, 3), dtype=jnp.int32)
    slot = jnp.zeros((1, 3), dtype=jnp.int32)
    stop = jnp.zeros((1, 3), dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.ones((1, 3), dtype=jnp.int32)

    out = compose_hygiene_with_shield_mask(
        batch,
        shield[0],
        source_sequence=source,
        target_slot_sequence=slot,
        stop_flag=stop,
        step_mask=step_mask,
        ship_bucket=bucket,
        ship_fraction=None,
        cfg=cfg,
        step_idx=0,
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(shield[0]))


@pytest.mark.jax
def test_duplicate_source_slot_masked_after_prior_launch() -> None:
    cfg = _train_cfg()
    _, batch = batched_reset(jax.random.split(jax.random.PRNGKey(1), 1), cfg.task)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    shield = jnp.ones((1, MAX_PLANETS, k, buckets), dtype=bool)

    src_row, tgt_slot = 1, 2
    source = jnp.array([[src_row, 0, 0]], dtype=jnp.int32)
    slot = jnp.array([[tgt_slot, 0, 0]], dtype=jnp.int32)
    stop = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.array([[1, 0, 0]], dtype=jnp.int32)

    out = compose_hygiene_with_shield_mask(
        batch,
        shield[0],
        source_sequence=source,
        target_slot_sequence=slot,
        stop_flag=stop,
        step_mask=step_mask,
        ship_bucket=bucket,
        ship_fraction=None,
        cfg=cfg,
        step_idx=1,
    )
    assert not bool(np.asarray(out[src_row, tgt_slot, :]).any())


@pytest.mark.jax
def test_bucket_zero_prefix_does_not_apply_hygiene() -> None:
    cfg = _train_cfg()
    _, batch = batched_reset(jax.random.split(jax.random.PRNGKey(3), 1), cfg.task)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    shield = jnp.ones((1, MAX_PLANETS, k, buckets), dtype=bool)

    src_row, tgt_slot = 1, 2
    source = jnp.array([[src_row, 0, 0]], dtype=jnp.int32)
    slot = jnp.array([[tgt_slot, 0, 0]], dtype=jnp.int32)
    stop = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.array([[0, 0, 0]], dtype=jnp.int32)

    out = compose_hygiene_with_shield_mask(
        batch,
        shield[0],
        source_sequence=source,
        target_slot_sequence=slot,
        stop_flag=stop,
        step_mask=step_mask,
        ship_bucket=bucket,
        ship_fraction=None,
        cfg=cfg,
        step_idx=1,
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(shield[0]))


@pytest.mark.jax
def test_hygiene_can_empty_targets_triggering_stop_path() -> None:
    cfg = _train_cfg()
    _, batch = batched_reset(jax.random.split(jax.random.PRNGKey(2), 1), cfg.task)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    shield = jnp.zeros((1, MAX_PLANETS, k, buckets), dtype=bool)
    shield = shield.at[0, 1, 2, 1].set(True)

    source = jnp.array([[1, 0, 0]], dtype=jnp.int32)
    slot = jnp.array([[2, 0, 0]], dtype=jnp.int32)
    stop = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.array([[1, 0, 0]], dtype=jnp.int32)

    out = compose_hygiene_with_shield_mask(
        batch,
        shield[0],
        source_sequence=source,
        target_slot_sequence=slot,
        stop_flag=stop,
        step_mask=step_mask,
        ship_bucket=bucket,
        ship_fraction=None,
        cfg=cfg,
        step_idx=1,
    )
    assert not bool(np.asarray(out).any())


def _two_friendly_planet_game():
    """Minimal two-planet layout with both worlds learner-owned."""
    from tests.test_trajectory_shield_factorized import _two_planet_game

    game = _two_planet_game(x0=20.0, y0=50.0, x1=80.0, y1=50.0, source_ships=100.0)
    owner = game.planets.owner.at[1].set(0)
    planets = game.planets._replace(owner=owner)
    return game._replace(planets=planets, initial_planets=planets)


def _edge_slot_for_target(batch, src_row: int, target_id: int) -> int:
    k = batch.edge_tgt_ids.shape[-1]
    for slot in range(k):
        if int(batch.edge_tgt_ids[0, src_row, slot]) == target_id:
            return slot
    raise AssertionError(f"target {target_id} not on row {src_row}")


@pytest.mark.jax
def test_friendly_reverse_edge_masked_after_forward_friendly_pick() -> None:
    cfg = _train_cfg()
    game = _two_friendly_planet_game()
    batch = encode_turn(game, cfg.task)
    batch = jax.tree.map(lambda x: x[None, ...], batch)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count

    src_row = 0
    tgt_id = int(np.asarray(batch.edge_tgt_ids[0, src_row, 0]))
    if tgt_id < 0:
        tgt_id = int(np.asarray(batch.edge_tgt_ids[0, src_row, 1]))
    tgt_slot = _edge_slot_for_target(batch, src_row, tgt_id)
    rev_row = 1
    rev_slot = _edge_slot_for_target(batch, rev_row, int(np.asarray(batch.edge_src_ids[0, src_row])))

    shield = jnp.zeros((1, MAX_PLANETS, k, buckets), dtype=bool)
    shield = shield.at[0, src_row, tgt_slot, 1].set(True)
    shield = shield.at[0, rev_row, rev_slot, 1].set(True)

    source = jnp.array([[src_row, 0, 0]], dtype=jnp.int32)
    slot = jnp.array([[tgt_slot, 0, 0]], dtype=jnp.int32)
    stop = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.array([[1, 0, 0]], dtype=jnp.int32)

    out = compose_hygiene_with_shield_mask(
        batch,
        shield[0],
        source_sequence=source,
        target_slot_sequence=slot,
        stop_flag=stop,
        step_mask=step_mask,
        ship_bucket=bucket,
        ship_fraction=None,
        cfg=cfg,
        step_idx=1,
    )
    assert not bool(np.asarray(out[rev_row, rev_slot, :]).any())


@pytest.mark.jax
def test_reverse_ban_masks_all_slots_for_same_target() -> None:
    cfg = _train_cfg()
    game = _two_friendly_planet_game()
    batch = encode_turn(game, cfg.task)
    batch = jax.tree.map(lambda x: x[None, ...], batch)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    batch = batch._replace(
        edge_tgt_ids=batch.edge_tgt_ids.at[0, 1, 1].set(batch.edge_tgt_ids[0, 1, 0])
    )

    src_row, tgt_slot = 0, 0
    rev_row = 1
    assert int(np.asarray(batch.edge_tgt_ids[0, rev_row, 0])) == int(
        np.asarray(batch.edge_src_ids[0, src_row])
    )
    assert int(np.asarray(batch.edge_tgt_ids[0, rev_row, 1])) == int(
        np.asarray(batch.edge_src_ids[0, src_row])
    )

    shield = jnp.ones((1, MAX_PLANETS, k, buckets), dtype=bool)
    source = jnp.array([[src_row, 0, 0]], dtype=jnp.int32)
    slot = jnp.array([[tgt_slot, 0, 0]], dtype=jnp.int32)
    stop = jnp.zeros((1, 3), dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.array([[1, 0, 0]], dtype=jnp.int32)

    out = compose_hygiene_with_shield_mask(
        batch,
        shield[0],
        source_sequence=source,
        target_slot_sequence=slot,
        stop_flag=stop,
        step_mask=step_mask,
        ship_bucket=bucket,
        ship_fraction=None,
        cfg=cfg,
        step_idx=1,
    )
    assert not bool(np.asarray(out[rev_row, 0, :]).any())
    assert not bool(np.asarray(out[rev_row, 1, :]).any())


@pytest.mark.jax
def test_hygiene_empty_targets_forces_stop_in_sampler() -> None:
    cfg = _train_cfg()
    _, batch = batched_reset(jax.random.split(jax.random.PRNGKey(5), 1), cfg.task)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    num_planets = batch.planet_mask.shape[-1]

    shield = jnp.zeros((1, MAX_PLANETS, k, buckets), dtype=bool)
    shield = shield.at[0, 1, 2, 1].set(True)

    source = jnp.array([[1, 0, 0]], dtype=jnp.int32)
    slot = jnp.array([[2, 0, 0]], dtype=jnp.int32)
    stop = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.array([[1, 0, 0]], dtype=jnp.int32)

    hygiene_mask = compose_hygiene_with_shield_mask(
        batch,
        shield[0],
        source_sequence=source,
        target_slot_sequence=slot,
        stop_flag=stop,
        step_mask=step_mask,
        ship_bucket=bucket,
        ship_fraction=None,
        cfg=cfg,
        step_idx=1,
    )
    remaining = jnp.ones((1, num_planets), dtype=jnp.float32) * 50.0
    source_mask = source_mask_from_bucket_mask_and_ships(hygiene_mask[None, ...], remaining)

    source_logits = jnp.zeros((1, num_planets), dtype=jnp.float32)
    target_logits = jnp.zeros((1, k), dtype=jnp.float32)
    stop_logits = jnp.array([-10.0], dtype=jnp.float32)
    ship_logits = jnp.zeros((1, k, buckets), dtype=jnp.float32)

    _, _, _, stop_out, _, _, _ = _sample_factored_step_from_logits(
        jax.random.PRNGKey(0),
        source_logits[0],
        target_logits[0],
        stop_logits[0],
        ship_logits[0],
        source_mask[0],
        hygiene_mask,
        deterministic=True,
    )
    assert int(stop_out) == 1


@pytest.mark.jax
def test_inactive_launch_leaves_cumulative_unchanged() -> None:
    cfg = _train_cfg()
    _, batch = batched_reset(jax.random.split(jax.random.PRNGKey(7), 1), cfg.task)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    lookups = build_hygiene_lookups(batch)
    cumulative = empty_cumulative_forbidden(
        1, num_planets=MAX_PLANETS, max_k=k, buckets=buckets
    )
    unchanged = apply_launch_to_cumulative_forbidden(
        cumulative,
        batch=batch,
        lookups=lookups,
        src_row=jnp.array([1], dtype=jnp.int32),
        slot=jnp.array([2], dtype=jnp.int32),
        active=jnp.array([False]),
    )
    np.testing.assert_array_equal(np.asarray(unchanged.count), np.asarray(cumulative.count))
    np.testing.assert_array_equal(np.asarray(unchanged.rows), np.asarray(cumulative.rows))
    np.testing.assert_array_equal(np.asarray(unchanged.slots), np.asarray(cumulative.slots))


@pytest.mark.jax
def test_cumulative_carry_matches_oracle_across_steps() -> None:
    cfg = _train_cfg()
    _, batch = batched_reset(jax.random.split(jax.random.PRNGKey(8), 1), cfg.task)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    lookups = build_hygiene_lookups(batch)
    shield = jnp.ones((1, MAX_PLANETS, k, buckets), dtype=bool)

    source = jnp.array([[1, 2, 0]], dtype=jnp.int32)
    slot = jnp.array([[2, 1, 0]], dtype=jnp.int32)
    stop = jnp.zeros((1, 3), dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.array([[1, 1, 0]], dtype=jnp.int32)

    cumulative = empty_cumulative_forbidden(
        1, num_planets=MAX_PLANETS, max_k=k, buckets=buckets
    )
    for step_idx in range(3):
        carry_mask = apply_cumulative_forbidden_to_shield(shield, cumulative)[0]
        oracle_mask = compose_hygiene_with_shield_mask(
            batch,
            shield[0],
            source_sequence=source,
            target_slot_sequence=slot,
            stop_flag=stop,
            step_mask=step_mask,
            ship_bucket=bucket,
            ship_fraction=None,
            cfg=cfg,
            step_idx=step_idx,
            lookups=lookups,
        )
        np.testing.assert_array_equal(np.asarray(carry_mask), np.asarray(oracle_mask))

        active = launch_valid_at_step(
            stop_flag=stop,
            step_mask=step_mask,
            ship_bucket=bucket,
            ship_fraction=None,
            cfg=cfg,
            step_idx=step_idx,
        )
        cumulative = apply_launch_to_cumulative_forbidden(
            cumulative,
            batch=batch,
            lookups=lookups,
            src_row=source[:, step_idx],
            slot=slot[:, step_idx],
            active=active,
        )


@pytest.mark.jax
def test_tiered_reject_prefix_does_not_apply_hygiene() -> None:
    """Tiered exact reject stores stop=1; hygiene prefix must ignore that step."""

    cfg = _train_cfg()
    _, batch = batched_reset(jax.random.split(jax.random.PRNGKey(10), 1), cfg.task)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    lookups = build_hygiene_lookups(batch)
    shield = jnp.ones((1, MAX_PLANETS, k, buckets), dtype=bool)

    src_row, tgt_slot = 1, 2
    source = jnp.array([[src_row, 0, 0]], dtype=jnp.int32)
    slot = jnp.array([[tgt_slot, 0, 0]], dtype=jnp.int32)
    # Post-tiered-reject stored sequence: source/slot kept, launch invalidated.
    stop = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.array([[0, 0, 0]], dtype=jnp.int32)

    cumulative = empty_cumulative_forbidden(
        1, num_planets=MAX_PLANETS, max_k=k, buckets=buckets
    )
    active = launch_valid_at_step(
        stop_flag=stop,
        step_mask=step_mask,
        ship_bucket=bucket,
        ship_fraction=None,
        cfg=cfg,
        step_idx=0,
    )
    assert not bool(np.asarray(active)[0])
    cumulative = apply_launch_to_cumulative_forbidden(
        cumulative,
        batch=batch,
        lookups=lookups,
        src_row=source[:, 0],
        slot=slot[:, 0],
        active=active,
    )

    out = compose_hygiene_with_shield_mask(
        batch,
        shield[0],
        source_sequence=source,
        target_slot_sequence=slot,
        stop_flag=stop,
        step_mask=step_mask,
        ship_bucket=bucket,
        ship_fraction=None,
        cfg=cfg,
        step_idx=1,
        lookups=lookups,
    )
    assert bool(np.asarray(out[src_row, tgt_slot, :]).any())


@pytest.mark.jax
def test_cumulative_carry_matches_oracle_under_cheap_shield() -> None:
    from src.jax.shield import apply_configured_trajectory_shield_factorized_topk
    from src.opponents.jax_actions.builders import owned_planet_ships

    cfg = _train_cfg(task={"trajectory_shield_mode": "cheap"})
    state, batch = batched_reset(jax.random.split(jax.random.PRNGKey(11), 1), cfg.task)
    k = edge_k(cfg.task)
    buckets = cfg.task.ship_bucket_count
    lookups = build_hygiene_lookups(batch)
    batch_row = jax.tree.map(lambda x: x[0], batch)
    game_row = jax.tree.map(lambda x: x[0], state.game)
    remaining = owned_planet_ships(state.game)[0]
    shield = apply_configured_trajectory_shield_factorized_topk(
        game_row,
        batch_row,
        cfg.task,
        remaining_planet_ships=remaining,
    ).ship_bucket_mask

    source = jnp.array([[1, 2, 0]], dtype=jnp.int32)
    slot = jnp.array([[2, 1, 0]], dtype=jnp.int32)
    stop = jnp.zeros((1, 3), dtype=jnp.float32)
    step_mask = jnp.ones((1, 3), dtype=jnp.float32)
    bucket = jnp.array([[1, 1, 0]], dtype=jnp.int32)

    cumulative = empty_cumulative_forbidden(
        1, num_planets=MAX_PLANETS, max_k=k, buckets=buckets
    )
    for step_idx in range(2):
        carry_mask = apply_cumulative_forbidden_to_shield(
            shield[None, ...], cumulative
        )[0]
        oracle_mask = compose_hygiene_with_shield_mask(
            batch,
            shield,
            source_sequence=source,
            target_slot_sequence=slot,
            stop_flag=stop,
            step_mask=step_mask,
            ship_bucket=bucket,
            ship_fraction=None,
            cfg=cfg,
            step_idx=step_idx,
            lookups=lookups,
        )
        np.testing.assert_array_equal(np.asarray(carry_mask), np.asarray(oracle_mask))

        active = launch_valid_at_step(
            stop_flag=stop,
            step_mask=step_mask,
            ship_bucket=bucket,
            ship_fraction=None,
            cfg=cfg,
            step_idx=step_idx,
        )
        cumulative = apply_launch_to_cumulative_forbidden(
            cumulative,
            batch=batch,
            lookups=lookups,
            src_row=source[:, step_idx],
            slot=slot[:, step_idx],
            active=active,
        )
