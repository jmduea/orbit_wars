"""Micro-benchmark: learner policy path encoder / decoder / shield breakdown."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp

import jax
from src.benchmark.jit_timing import TimingStats, measure_jitted
from src.config import TrainConfig
from src.config.schema import TaskConfig
from src.jax.action_sampling import (
    _sample_shielded_factored_sequence_with_params,
    owned_planet_ships,
)
from src.jax.env import batched_reset
from src.jax.factored_sequence_scan import (
    forward_factorized_critic,
    forward_factorized_encode,
)
from src.jax.policy import (
    build_planet_graph_transformer_policy,
    factorized_decode_advance_carry,
    factorized_decode_init_carry,
    factorized_decode_step,
)
from src.jax.shield.trajectory import apply_configured_trajectory_shield_factorized_topk

ShieldMode = Literal["off", "cheap"]


def _train_cfg(
    *,
    shield_mode: ShieldMode,
    max_moves_k: int,
    decoder_carry: bool,
    candidate_count: int,
) -> TrainConfig:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.model.pointer_decoder = "factorized_topk"
    cfg.model.hidden_size = 128
    cfg.model.max_moves_k = max_moves_k
    cfg.model.decoder_carry = decoder_carry
    cfg.model.planet_transformer_layers = 2
    cfg.model.attention_heads = 4
    cfg.task = TaskConfig(
        candidate_count=candidate_count,
        ship_bucket_count=8,
        max_fleets=32,
        trajectory_shield_mode=shield_mode,
        trajectory_shield_final_validate_selected=False,
    )
    return cfg


def _stats_row(name: str, stats: TimingStats) -> dict[str, Any]:
    return {
        "name": name,
        **asdict(stats),
        "mean_ms": stats.mean_seconds * 1000.0,
        "std_ms": stats.std_seconds * 1000.0,
        "compile_ms": stats.compile_seconds * 1000.0,
    }


def _marginal_rows(
    cumulative: dict[str, float], *, full_ms: float
) -> list[dict[str, Any]]:
    tiers = [
        ("encoder", "encoder"),
        ("+ critic", "encode_critic"),
        ("+ decoder_k_scan", "encode_critic_decoder"),
        ("+ shield_sample_misc", "full_sample"),
    ]
    prev = 0.0
    rows: list[dict[str, Any]] = []
    for label, key in tiers:
        cum = cumulative[key]
        marginal = cum - prev
        rows.append(
            {
                "tier": label,
                "cumulative_ms": cum,
                "marginal_ms": marginal,
                "pct_of_full": (100.0 * marginal / full_ms) if full_ms > 0 else 0.0,
            }
        )
        prev = cum
    return rows


def _format_comparison_table(payload: dict[str, Any]) -> str:
    lines = [
        (
            f"policy-path-profile  device={payload['device']}  "
            f"batch={payload['batch_size']}  max_moves_k={payload['max_moves_k']}  "
            f"decoder_carry={payload['decoder_carry']}"
        ),
        "",
    ]
    for block in payload["shield_modes"]:
        mode = block["shield_mode"]
        lines.append(f"=== shield_{mode} ===")
        lines.append(f"{'tier':<24} {'cum_ms':>8} {'marg_ms':>8} {'%full':>7}")
        lines.append(f"{'-' * 24} {'-' * 8} {'-' * 8} {'-' * 7}")
        for row in block["marginals"]:
            lines.append(
                f"{row['tier']:<24} {row['cumulative_ms']:8.2f} "
                f"{row['marginal_ms']:8.2f} {row['pct_of_full']:6.1f}%"
            )
        lines.append(
            f"  shield_only_k_scan     {block['shield_only_ms']:8.2f} ms  (isolated)"
        )
        lines.append("")
    if len(payload["shield_modes"]) == 2:
        by_mode = {block["shield_mode"]: block for block in payload["shield_modes"]}
        off = by_mode.get("off")
        cheap = by_mode.get("cheap")
        if off is None or cheap is None:
            off = payload["shield_modes"][0]
            cheap = payload["shield_modes"][1]
        lines.append("=== cheap - off ===")
        lines.append(
            f"  full_sample          "
            f"{cheap['cumulative_ms']['full_sample'] - off['cumulative_ms']['full_sample']:+.2f} ms"
        )
        lines.append(
            f"  shield_only          "
            f"{cheap['shield_only_ms'] - off['shield_only_ms']:+.2f} ms"
        )
    lines.append("")
    lines.append(
        "Note: negative marginals on the fused tail mean JAX fused full_sample "
        "faster than the decode stack alone; use full_sample as ground truth."
    )
    return "\n".join(lines)


def _run_shield_mode_block(
    *,
    shield_mode: ShieldMode,
    batch_size: int,
    max_moves_k: int,
    decoder_carry: bool,
    candidate_count: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    cfg = _train_cfg(
        shield_mode=shield_mode,
        max_moves_k=max_moves_k,
        decoder_carry=decoder_carry,
        candidate_count=candidate_count,
    )
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    state, batch = batched_reset(keys, cfg.task)
    policy = build_planet_graph_transformer_policy(cfg)
    params = policy.init(jax.random.PRNGKey(1), batch)
    player_count = jnp.full((batch_size,), cfg.task.player_count, dtype=jnp.int32)
    carry_hidden = jnp.zeros((batch_size, cfg.model.hidden_size), dtype=jnp.float32)
    remaining_ships = owned_planet_ships(state.game)
    sample_key = jax.random.PRNGKey(42)
    step_axis = jnp.arange(max_moves_k, dtype=jnp.int32)

    @jax.jit
    def bench_encoder():
        return forward_factorized_encode(params, policy, batch).context_query

    @jax.jit
    def bench_encode_critic():
        encoder_out = forward_factorized_encode(params, policy, batch)
        return forward_factorized_critic(
            params, policy, encoder_out, player_count=player_count
        ).value

    @jax.jit
    def bench_encode_critic_decoder():
        encoder_out = forward_factorized_encode(params, policy, batch)
        _ = forward_factorized_critic(
            params, policy, encoder_out, player_count=player_count
        )
        decode_carry = factorized_decode_init_carry(
            params,
            policy,
            encoder_out,
            decoder_hidden=carry_hidden if decoder_carry else None,
        )

        def body(carry_in, step_idx):
            _logits, proposed = factorized_decode_step(
                params,
                policy,
                encoder_out,
                carry_in,
                rng=jax.random.fold_in(sample_key, step_idx),
                deterministic=True,
            )
            advanced = factorized_decode_advance_carry(
                params,
                policy,
                encoder_out,
                proposed,
                source=jnp.zeros((batch_size,), dtype=jnp.int32),
                target_slot=jnp.zeros((batch_size,), dtype=jnp.int32),
            )
            return advanced, None

        final_carry, _ = jax.lax.scan(body, decode_carry, step_axis)
        return final_carry.state

    @jax.jit
    def bench_shield_only():
        ships = remaining_ships

        def body(ships_in, _step_idx):
            shielded = jax.vmap(
                lambda game_row, batch_row, ship_row: (
                    apply_configured_trajectory_shield_factorized_topk(
                        game_row,
                        batch_row,
                        cfg.task,
                        remaining_planet_ships=ship_row,
                    )
                )
            )(state.game, batch, ships_in)
            return ships_in, shielded.ship_bucket_mask

        _ships, masks = jax.lax.scan(body, ships, step_axis)
        return masks

    @jax.jit
    def bench_full_sample():
        return _sample_shielded_factored_sequence_with_params(
            sample_key,
            state.game,
            batch,
            params,
            policy,
            cfg,
            deterministic=True,
            decoder_hidden_in=carry_hidden if decoder_carry else None,
        ).target_index

    scenario_fns = {
        "encoder": bench_encoder,
        "encode_critic": bench_encode_critic,
        "encode_critic_decoder": bench_encode_critic_decoder,
        "shield_only_k_scan": bench_shield_only,
        "full_sample": bench_full_sample,
    }
    scenario_stats = {
        name: measure_jitted(fn, warmup=warmup, repeats=repeats)
        for name, fn in scenario_fns.items()
    }
    cumulative_ms = {
        name: stats.mean_seconds * 1000.0 for name, stats in scenario_stats.items()
    }
    full_ms = cumulative_ms["full_sample"]
    return {
        "shield_mode": shield_mode,
        "scenarios": [
            _stats_row(name, stats) for name, stats in scenario_stats.items()
        ],
        "cumulative_ms": cumulative_ms,
        "marginals": _marginal_rows(cumulative_ms, full_ms=full_ms),
        "shield_only_ms": cumulative_ms["shield_only_k_scan"],
    }


def run_policy_path_profile_benchmark(
    *,
    batch_size: int = 16,
    max_moves_k: int = 2,
    decoder_carry: bool = True,
    candidate_count: int = 6,
    shield_modes: list[str] | None = None,
    warmup: int = 5,
    repeats: int = 30,
    out: Path | None = None,
) -> int:
    """Profile encoder/decoder/shield tiers for factorized policy sampling."""

    modes: list[ShieldMode] = [m for m in (shield_modes or ["off", "cheap"])]
    blocks = [
        _run_shield_mode_block(
            shield_mode=mode,
            batch_size=batch_size,
            max_moves_k=max_moves_k,
            decoder_carry=decoder_carry,
            candidate_count=candidate_count,
            warmup=warmup,
            repeats=repeats,
        )
        for mode in modes
    ]
    payload: dict[str, Any] = {
        "benchmark": "policy_path_profile",
        "device": str(jax.devices()[0]),
        "batch_size": batch_size,
        "max_moves_k": max_moves_k,
        "decoder_carry": decoder_carry,
        "candidate_count": candidate_count,
        "warmup": warmup,
        "repeats": repeats,
        "shield_modes": blocks,
    }

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(_format_comparison_table(payload))
    if out is not None:
        print(f"Wrote {out}", flush=True)
    return 0
