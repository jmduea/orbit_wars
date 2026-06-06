"""Micro-benchmark: ``encode_turn`` / ``encode_learner_turn`` throughput."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax.numpy as jnp

import jax
from src.benchmark.jit_timing import EncodeTimingStats, measure_jitted_encodes
from src.config.schema import TaskConfig
from src.jax.env import batched_reset
from src.jax.features import empty_feature_history, encode_learner_turn, encode_turn


def _task_cfg(
    *,
    player_count: int,
    candidate_count: int,
    edge_rank_mode: str,
) -> TaskConfig:
    return TaskConfig(
        player_count=player_count,
        candidate_count=candidate_count,
        edge_rank_mode=edge_rank_mode,
        max_fleets=32,
        ship_bucket_count=8,
        ship_feature_scale=1000.0,
        feature_history_steps=1,
        trajectory_shield_mode="cheap",
    )


def _scenario_result(
    *,
    name: str,
    edge_rank_mode: str,
    player_count: int,
    candidate_count: int,
    batch_size: int,
    stats: EncodeTimingStats,
) -> dict[str, Any]:
    return {
        "name": name,
        "edge_rank_mode": edge_rank_mode,
        "player_count": player_count,
        "candidate_count": candidate_count,
        "batch_size": batch_size,
        **asdict(stats),
    }


def _run_scenarios_for_config(
    *,
    task_cfg: TaskConfig,
    batch_size: int,
    warmup: int,
    repeats: int,
    include_learner_turn: bool,
    include_4p_all_players: bool,
) -> list[dict[str, Any]]:
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    states, _ = batched_reset(keys, task_cfg)
    games = states.game
    single_game = jax.tree.map(lambda x: x[0], games)

    results: list[dict[str, Any]] = []
    meta = {
        "edge_rank_mode": task_cfg.edge_rank_mode,
        "player_count": task_cfg.player_count,
        "candidate_count": task_cfg.candidate_count,
        "batch_size": batch_size,
    }

    @jax.jit
    def single_learner():
        return encode_turn(single_game, task_cfg)

    results.append(
        _scenario_result(
            name="single_learner",
            stats=measure_jitted_encodes(
                single_learner,
                warmup=warmup,
                repeats=repeats,
                encodes_per_call=1,
            ),
            **meta,
        )
    )

    if task_cfg.player_count == 2:
        opp_game = single_game._replace(player=jnp.array(1, dtype=jnp.int32))

        @jax.jit
        def single_opponent():
            return encode_turn(opp_game, task_cfg)

        results.append(
            _scenario_result(
                name="single_opponent_2p",
                stats=measure_jitted_encodes(
                    single_opponent,
                    warmup=warmup,
                    repeats=repeats,
                    encodes_per_call=1,
                ),
                **meta,
            )
        )

    @jax.jit
    def vmap_batch_learner():
        return jax.vmap(lambda game: encode_turn(game, task_cfg))(games)

    results.append(
        _scenario_result(
            name="vmap_batch_learner",
            stats=measure_jitted_encodes(
                vmap_batch_learner,
                warmup=warmup,
                repeats=repeats,
                encodes_per_call=batch_size,
            ),
            **meta,
        )
    )

    if task_cfg.player_count == 2:
        opp_games = games._replace(player=jnp.ones_like(games.player, dtype=jnp.int32))

        @jax.jit
        def vmap_batch_opponent():
            return jax.vmap(lambda game: encode_turn(game, task_cfg))(opp_games)

        results.append(
            _scenario_result(
                name="vmap_batch_opponent_2p",
                stats=measure_jitted_encodes(
                    vmap_batch_opponent,
                    warmup=warmup,
                    repeats=repeats,
                    encodes_per_call=batch_size,
                ),
                **meta,
            )
        )

    if include_learner_turn:
        history = empty_feature_history(task_cfg)

        @jax.jit
        def single_learner_turn():
            return encode_learner_turn(single_game, task_cfg, history)

        results.append(
            _scenario_result(
                name="encode_learner_turn_single",
                stats=measure_jitted_encodes(
                    single_learner_turn,
                    warmup=warmup,
                    repeats=repeats,
                    encodes_per_call=1,
                ),
                **meta,
            )
        )

    if include_4p_all_players and task_cfg.player_count == 4:
        player_ids = jnp.arange(task_cfg.player_count, dtype=jnp.int32)

        @jax.jit
        def vmap_4p_all_players():
            player_games = jax.vmap(
                lambda player_id: games._replace(
                    player=jnp.full_like(games.step, player_id, dtype=jnp.int32)
                )
            )(player_ids)
            flat_games = jax.tree.map(
                lambda x: x.reshape(
                    (task_cfg.player_count * batch_size,) + x.shape[2:]
                ),
                player_games,
            )
            return jax.vmap(lambda game: encode_turn(game, task_cfg))(flat_games)

        encodes = task_cfg.player_count * batch_size
        results.append(
            _scenario_result(
                name="vmap_4p_all_players",
                stats=measure_jitted_encodes(
                    vmap_4p_all_players,
                    warmup=warmup,
                    repeats=repeats,
                    encodes_per_call=encodes,
                ),
                **meta,
            )
        )

    return results


def _format_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "scenario",
        "edge_rank",
        "cand",
        "batch",
        "enc/call",
        "compile_ms",
        "mean_ms",
        "std_ms",
        "ms/encode",
        "enc/s",
    )
    lines = [
        "  ".join(f"{col:>14}" for col in header),
        "  ".join("-" * 14 for _ in header),
    ]
    for row in rows:
        lines.append(
            "  ".join(
                [
                    f"{row['name']:>14}",
                    f"{row['edge_rank_mode']:>14}",
                    f"{row['candidate_count']:>14}",
                    f"{row['batch_size']:>14}",
                    f"{row['encodes_per_call']:>14}",
                    f"{row['compile_seconds'] * 1000:>14.2f}",
                    f"{row['mean_seconds'] * 1000:>14.2f}",
                    f"{row['std_seconds'] * 1000:>14.2f}",
                    f"{row['mean_seconds_per_encode'] * 1000:>14.4f}",
                    f"{row['encodes_per_second']:>14.1f}",
                ]
            )
        )
    return "\n".join(lines)


def run_encode_turn_benchmark(
    *,
    batch_size: int = 32,
    player_count: int = 2,
    candidate_count: int = 6,
    edge_rank_modes: list[str] | None = None,
    warmup: int = 3,
    repeats: int = 20,
    include_learner_turn: bool = True,
    include_4p_all_players: bool = True,
    sweep_defaults: bool = False,
    out: Path | None = None,
) -> int:
    """Run encode_turn microbench scenarios. Returns process exit code."""

    if sweep_defaults:
        edge_rank_modes = ["snapshot", "intercept_min"]
        player_counts = [2, 4]
        candidate_counts = [candidate_count]
    else:
        edge_rank_modes = edge_rank_modes or ["snapshot"]
        player_counts = [player_count]
        candidate_counts = [candidate_count]

    all_rows: list[dict[str, Any]] = []
    for rank_mode in edge_rank_modes:
        for pc in player_counts:
            for cand in candidate_counts:
                task_cfg = _task_cfg(
                    player_count=pc,
                    candidate_count=cand,
                    edge_rank_mode=rank_mode,
                )
                all_rows.extend(
                    _run_scenarios_for_config(
                        task_cfg=task_cfg,
                        batch_size=batch_size,
                        warmup=warmup,
                        repeats=repeats,
                        include_learner_turn=include_learner_turn,
                        include_4p_all_players=include_4p_all_players,
                    )
                )

    payload = {
        "benchmark": "encode_turn",
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "scenarios": all_rows,
    }

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(_format_table(all_rows))
    if out is not None:
        print(f"Wrote {out}", flush=True)
    return 0
