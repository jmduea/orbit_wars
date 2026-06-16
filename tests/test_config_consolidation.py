"""Hydra config composition and base-YAML contract tests.

These tests intentionally avoid brittle equality against full resolved configs.
They verify Hydra composition succeeds for primary ``ow train`` / ``ow eval``
profiles and assert command-critical values as membership in acceptable sets.
Each ``conf/<group>/base.yaml`` must declare every schema leaf path.
"""

from __future__ import annotations

import random
from itertools import product
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.make_wandb_sweep import compose_sweep_gen, write_wandb_sweep
from src.benchmark.training import (
    PRIMARY_E2E_OVERRIDES,
    WORKSTATION_VALIDATION_OVERRIDES,
    compose_benchmark_config,
    resolve_benchmark_overrides,
)
from src.config import audit_responsibility_base_yaml_keys, compose_hydra_train_config
from src.config.rollout_allocation import (
    resolve_rollout_group_specs,
    validate_rollout_allocation,
)

SWEEP_COMPOSE_RECIPES = (
    "budget",
    "preflight",
    "preflight_top_33",
)

PRIMARY_TRAIN_PROFILES: dict[str, list[str]] = {
    "default": [],
    "smoke": [
        "training=smoke",
        "curriculum=noop_only",
        "telemetry=throughput_only",
        "artifacts=disabled",
    ],
    "shield_cheap": ["task=shield_cheap", "telemetry=default"],
    "workstation_mixed": ["training=workstation"],
    "opponent_recovery": [
        "curriculum=random_only",
        "telemetry=opponent_recovery",
    ],
    "opponent_recovery_floor": [
        "curriculum=noop_only",
        "telemetry=opponent_recovery",
    ],
}

PRIMARY_EVAL_PROFILES: dict[str, list[str]] = {
    "default": [],
    "tournament_ready": ["artifacts.tournament.enabled=true"],
}

SACRED_ARCHITECTURES = frozenset(
    {"planet_graph_transformer", "planet_graph_transformer_small"}
)
SACRED_POINTER_DECODERS = frozenset({"factorized_topk"})
EXPERIMENTAL_POINTER_DECODERS = frozenset({"planet_flow_target_heatmap"})
ACCEPTABLE_VALUE_HEADS = frozenset({"shared", "format_routed", "distributional"})
ACCEPTABLE_SHIP_ACTION_MODES = frozenset({"buckets", "continuous_fraction"})
ACCEPTABLE_TRAJECTORY_SHIELD_MODES = frozenset({"off", "cheap", "tiered", "exact"})
ACCEPTABLE_REPLAY_BACKENDS = frozenset({"docker", "local"})
ACCEPTABLE_TOURNAMENT_FORMATS = frozenset(
    {
        "2p_vs_baseline",
        "2p_head_to_head",
        "4p_free_for_all",
        "4p_challenger_vs_baselines",
    }
)
ACCEPTABLE_PROMOTION_STRATEGIES = frozenset({"metric", "tournament", "hybrid"})


def test_responsibility_base_yaml_declares_required_schema_keys() -> None:
    missing = audit_responsibility_base_yaml_keys()
    assert missing == []


TRAINING_PROFILES = tuple(
    path.stem
    for path in sorted(
        (Path(__file__).resolve().parents[1] / "conf" / "training").glob("*.yaml")
    )
    if path.stem != "base"
)


def test_default_train_profile_composes_and_respects_command_critical_sets() -> None:
    cfg = compose_hydra_train_config()

    specs = resolve_rollout_group_specs(cfg)
    assert {spec.player_count: spec.num_envs for spec in specs} == {2: 16, 4: 16}
    assert cfg.training.rollout_microbatch_envs == 16

    assert cfg.model.architecture in SACRED_ARCHITECTURES
    assert cfg.model.pointer_decoder in SACRED_POINTER_DECODERS
    assert cfg.model.value_head in ACCEPTABLE_VALUE_HEADS
    assert cfg.task.ship_action_mode in ACCEPTABLE_SHIP_ACTION_MODES
    assert cfg.task.trajectory_shield_mode in ACCEPTABLE_TRAJECTORY_SHIELD_MODES
    assert cfg.artifacts.artifact_pipeline.replay_backend in ACCEPTABLE_REPLAY_BACKENDS
    assert cfg.output.root in {"outputs"}
    assert cfg.output.campaign  # non-empty slug validated at compose time
    assert resolve_rollout_group_specs(cfg)
    assert not hasattr(cfg, "env")
    assert not hasattr(cfg, "ppo")
    assert not hasattr(cfg, "save_dir")

    assert cfg.curriculum.stages
    assert cfg.opponents.dispatch == "random"
    assert not cfg.opponents.self_play.enabled
    assert cfg.opponents.snapshot.pool_size == 0
    assert cfg.opponents.snapshot.interval_updates == 0


@pytest.mark.parametrize(
    "profile,dispatch,self_play,pool_size,interval_updates",
    [
        ("noop_only", "noop", False, 0, 0),
        ("random_only", "random", False, 0, 0),
        ("latest_only", "self", True, 0, 0),
        ("nearest_only", "self", False, 0, 0),
        ("self_play_staged", "self", True, 2, 10),
    ],
)
def test_curriculum_profiles_derive_private_opponent_cache(
    profile: str,
    dispatch: str,
    self_play: bool,
    pool_size: int,
    interval_updates: int,
) -> None:
    cfg = compose_hydra_train_config([f"curriculum={profile}"])

    assert cfg.curriculum.stages
    assert cfg.opponents.dispatch == dispatch
    assert cfg.opponents.self_play.enabled is self_play
    assert cfg.opponents.snapshot.pool_size == pool_size
    assert cfg.opponents.snapshot.interval_updates == interval_updates


@pytest.mark.parametrize(
    "override",
    [
        "opponents=noop_only",
        "opponents.dispatch=noop",
        "train_bundle=opponent_recovery",
        "curriculum=off",
    ],
)
def test_public_opponent_overrides_are_rejected(override: str) -> None:
    with pytest.raises(ValueError, match="curriculum-only"):
        compose_hydra_train_config([override])


@pytest.mark.parametrize("profile", TRAINING_PROFILES)
def test_training_profile_composes(profile: str) -> None:
    cfg = compose_hydra_train_config([f"training={profile}"])
    validate_rollout_allocation(cfg)
    assert resolve_rollout_group_specs(cfg)


def test_hybrid_promotion_artifacts_profile_composes() -> None:
    cfg = compose_hydra_train_config(["artifacts=hybrid_promotion"])

    assert cfg.artifacts.promotion.strategy in ACCEPTABLE_PROMOTION_STRATEGIES
    assert cfg.artifacts.promotion.strategy == "hybrid"
    assert cfg.artifacts.tournament.enabled
    assert cfg.artifacts.unified_tournament.enabled
    assert "4p_challenger_vs_baselines" in cfg.artifacts.tournament.formats
    assert cfg.artifacts.artifact_pipeline.checkpoint_eval_async
    assert not cfg.artifacts.artifact_pipeline.docker_validation_async
    assert not cfg.artifacts.artifact_pipeline.replay_async
    assert not cfg.artifacts.replay.enabled


def test_long_preview_profile_uses_bounded_colab_geometry() -> None:
    cfg = compose_hydra_train_config(
        [
            "training=long_preview",
            "curriculum=random_only",
            "artifacts=ssot_pipeline",
        ]
    )

    assert cfg.training.total_updates == 500
    assert cfg.training.rollout_steps == 256
    assert cfg.training.reseed_every_updates == 25
    assert cfg.artifacts.ssot_pipeline.enabled
    assert cfg.artifacts.artifact_pipeline.enabled
    assert cfg.artifacts.checkpoint_every == 100
    assert cfg.opponents.dispatch == "random"
    validate_rollout_allocation(cfg)


def test_planet_flow_proof_artifacts_compose_with_local_replay() -> None:
    cfg = compose_hydra_train_config(
        [
            "model=planet_flow_target_heatmap",
            "artifacts=planet_flow_proof",
            "curriculum=random_only",
        ]
    )

    assert cfg.artifacts.artifact_pipeline.enabled
    assert cfg.artifacts.artifact_pipeline.replay_async
    assert cfg.artifacts.artifact_pipeline.replay_backend == "local"
    assert cfg.artifacts.replay.enabled
    assert not cfg.artifacts.promotion.enabled
    assert not cfg.artifacts.tournament.enabled
    assert resolve_rollout_group_specs(cfg)


def test_planet_flow_target_heatmap_profile_composes_with_proof_guards() -> None:
    cfg = compose_hydra_train_config(
        [
            "model=planet_flow_target_heatmap",
            "artifacts=disabled",
            "curriculum=random_only",
        ]
    )

    assert cfg.model.pointer_decoder in EXPERIMENTAL_POINTER_DECODERS
    assert tuple(cfg.model.planet_flow.pressure_bucket_values) == (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    )
    assert not cfg.artifacts.artifact_pipeline.enabled
    assert not cfg.artifacts.replay.enabled
    assert not cfg.artifacts.promotion.enabled
    assert not cfg.artifacts.tournament.enabled
    assert not cfg.curriculum.enabled
    assert not cfg.opponents.self_play.enabled
    assert resolve_rollout_group_specs(cfg)


def test_planet_flow_rejects_default_artifact_paths() -> None:
    with pytest.raises(ValueError, match="only supports local async replay"):
        compose_hydra_train_config(
            [
                "model=planet_flow_target_heatmap",
                "curriculum=random_only",
            ]
        )


def test_planet_flow_rejects_latest_or_historical_opponent_paths() -> None:
    with pytest.raises(ValueError, match="self-play/latest"):
        compose_hydra_train_config(
            [
                "model=planet_flow_target_heatmap",
                "artifacts=disabled",
                "curriculum=latest_only",
            ]
        )


def test_planet_flow_rejects_invalid_pressure_buckets() -> None:
    with pytest.raises(ValueError, match="pressure_bucket_values"):
        compose_hydra_train_config(
            [
                "model=planet_flow_target_heatmap",
                "model.planet_flow.pressure_bucket_values=[0.25,0.5]",
                "artifacts=disabled",
                "curriculum=random_only",
            ]
        )


def test_benchmark_sanity_defaults_compose_with_even_2p4p_split() -> None:
    cfg = compose_benchmark_config(list(WORKSTATION_VALIDATION_OVERRIDES))
    specs = resolve_rollout_group_specs(cfg)
    assert not cfg.training.rotate_format_rollouts
    assert {spec.player_count: spec.num_envs for spec in specs} == {2: 16, 4: 16}
    assert cfg.training.rollout_microbatch_envs <= min(spec.num_envs for spec in specs)


def test_benchmark_primary_preset_compose_includes_shield_cheap() -> None:
    overrides = resolve_benchmark_overrides(preset="primary", overrides=None)
    assert overrides == list(PRIMARY_E2E_OVERRIDES)
    cfg = compose_benchmark_config(overrides)
    assert cfg.task.trajectory_shield_mode == "cheap"
    assert resolve_rollout_group_specs(cfg)


@pytest.mark.parametrize("name,overrides", PRIMARY_EVAL_PROFILES.items())
def test_primary_eval_profiles_compose(name: str, overrides: list[str]) -> None:
    del name
    cfg = compose_hydra_train_config(overrides)

    assert set(cfg.artifacts.tournament.formats).issubset(ACCEPTABLE_TOURNAMENT_FORMATS)
    assert cfg.artifacts.promotion.strategy in ACCEPTABLE_PROMOTION_STRATEGIES
    assert cfg.artifacts.artifact_pipeline.replay_backend in ACCEPTABLE_REPLAY_BACKENDS


def test_wandb_group_defaults_to_output_campaign_when_unset() -> None:
    cfg = compose_hydra_train_config()
    assert cfg.telemetry.wandb.group == cfg.output.campaign

    cfg_override = compose_hydra_train_config(["output.campaign=throughput_sweep"])
    assert cfg_override.output.campaign == "throughput_sweep"
    assert cfg_override.telemetry.wandb.group == "throughput_sweep"


def test_new_responsibility_overrides_compose_to_runtime_config() -> None:
    cfg = compose_hydra_train_config(
        [
            "training.total_updates=2",
            "task.candidate_count=12",
            "reward.reward_production_delta=0.01",
            "training=2p4p_32_split",
            "telemetry.wandb.group=capacity",
        ]
    )

    assert cfg.training.total_updates == 2
    assert cfg.task.candidate_count == 12
    assert cfg.reward.reward_production_delta == 0.01
    assert resolve_rollout_group_specs(cfg)[0].num_envs == 16
    assert cfg.telemetry.wandb.group == "capacity"


@pytest.mark.parametrize(
    "legacy_override",
    [
        "ppo.total_updates=3",
        "env.candidate_count=16",
        "wandb.group=legacy_override",
        "format=2p_16env",
        "training_format.rollout_groups=[]",
        "self_play_enabled=false",
        "self_play_pool_size=0",
        "self_play_snapshot_interval=0",
        "save_dir=artifacts/old",
    ],
)
def test_legacy_overrides_are_rejected(legacy_override: str) -> None:
    with pytest.raises(Exception):
        compose_hydra_train_config([legacy_override])


def test_output_campaign_slug_is_validated() -> None:
    with pytest.raises(ValueError, match="output.campaign"):
        compose_hydra_train_config(["output.campaign='bad campaign'"])


def test_output_paths_must_be_relative() -> None:
    with pytest.raises(ValueError, match="output.wandb_dir"):
        compose_hydra_train_config(["output.wandb_dir=/tmp/wandb"])


@pytest.mark.parametrize(
    "override",
    [
        "output.run_id=../escape",
        "output.root=../outputs",
        "output.wandb_dir=../wandb",
        "artifacts.artifact_pipeline.queue_dir=../jobs",
        "artifacts.artifact_pipeline.result_dir=../evals",
    ],
)
def test_output_paths_reject_traversal(override: str) -> None:
    with pytest.raises(ValueError, match=r"\.\.|run_id"):
        compose_hydra_train_config([override])


def test_planet_flow_training_profile_resolves_proof_defaults() -> None:
    from src.config import compose_hydra_train_config

    cfg = compose_hydra_train_config(
        [
            "model=planet_flow_target_heatmap",
            "training=planet_flow",
            "artifacts=planet_flow_proof",
            "curriculum=noop_only",
        ]
    )

    assert cfg.training.rollout_steps == 512
    assert cfg.training.update_chunk_rows == 2048
    assert cfg.model.max_moves_k == 1


def test_multitask_smoke_overrides_compose() -> None:
    cfg = compose_hydra_train_config(
        [
            "model.architecture=planet_graph_transformer_small",
            "task.candidate_count=3",
            "task.edge_rank_mode=intercept_min",
            "training.num_envs=2",
            "training.rollout_microbatch_envs=1",
            "training.rollout_steps=128",
            "training.total_updates=20",
            "training.update_chunk_rows=2048",
            "curriculum=noop_only",
            "output.campaign=multitask_smoke",
        ]
    )
    assert cfg.model.architecture == "planet_graph_transformer_small"
    assert cfg.opponents.dispatch == "noop"
    from src.jax.policy import build_jax_policy
    from src.opponents.constants import validate_jax_training_opponent_mode

    validate_jax_training_opponent_mode(cfg.opponents.dispatch)
    policy = build_jax_policy(cfg)
    assert policy.__class__.__name__ == "ComposableFactorizedPlanetPolicy"


def test_jax_training_opponent_mode_normalization() -> None:
    from src.opponents.constants import (
        is_noop_jax_training_opponent_mode,
        normalize_jax_training_opponent_mode,
        validate_jax_training_opponent_mode,
    )

    for raw in ("no_op", "noop", "NO_OP"):
        validate_jax_training_opponent_mode(raw)
        assert normalize_jax_training_opponent_mode(raw) == "noop"
        assert is_noop_jax_training_opponent_mode(raw)
    validate_jax_training_opponent_mode("random")
    with pytest.raises(ValueError, match="JAX training supports"):
        validate_jax_training_opponent_mode("noop_only")


def test_preflight_sweep_generates_curriculum_only_guardrails(
    tmp_path: Path,
) -> None:
    cfg = compose_sweep_gen(
        [
            "wandb_sweep=preflight",
            f"out_dir={tmp_path}",
        ]
    )

    assert cfg["name"] == "preflight"
    assert cfg["method"] == "bayes"
    assert cfg["run_cap"] == 24
    assert cfg["metric"] == {"name": "preflight_sweep_score", "goal": "maximize"}

    parameters = cfg["parameters"]
    assert parameters["curriculum"]["value"] == "noop_only"
    assert "opponents" not in parameters
    assert parameters["telemetry.metric_groups.losses"]["value"] is True

    out_path = write_wandb_sweep(cfg)
    generated = OmegaConf.to_container(OmegaConf.load(out_path), resolve=False)
    assert isinstance(generated, dict)
    assert out_path == tmp_path / "preflight.yaml"
    assert generated["metric"] == {
        "name": "preflight_sweep_score",
        "goal": "maximize",
    }
    assert generated["run_cap"] == 24
    assert (
        generated["parameters"]["telemetry.metric_groups.action_decision"]["value"]
        is True
    )


BOUNDED_SWEEP_SAMPLE_SIZE = 200
BOUNDED_SWEEP_SAMPLE_SEED = 42


@pytest.mark.slow
def test_wandb_sweep_campaign_samples_compose_bounded() -> None:
    """Deterministic sample of the full sweep grid (same intent, ~30s vs ~3+ min)."""

    cases = list(_iter_sweep_compose_cases(full_grid=True))
    assert cases, "expected at least one valid sweep compose case"
    if len(cases) > BOUNDED_SWEEP_SAMPLE_SIZE:
        cases = random.Random(BOUNDED_SWEEP_SAMPLE_SEED).sample(
            cases, BOUNDED_SWEEP_SAMPLE_SIZE
        )
    composed = 0
    for overrides in cases:
        try:
            cfg = compose_hydra_train_config(overrides)
        except ValueError:
            # Grid includes invalid allocation combos (microbatch > group envs, etc.).
            continue
        composed += 1
        assert cfg.telemetry.wandb.group
        assert isinstance(cfg.telemetry.wandb.tags, list)
    assert composed >= min(50, len(cases)), (
        "bounded sweep sample must compose a substantial subset"
    )


@pytest.mark.sweep
def test_wandb_sweep_campaign_samples_compose_full() -> None:
    for overrides in _iter_sweep_compose_cases(full_grid=True):
        cfg = compose_hydra_train_config(overrides)
        assert cfg.telemetry.wandb.group
        assert isinstance(cfg.telemetry.wandb.tags, list)


def test_wandb_sweep_fixed_scaffolding_is_discoverable() -> None:
    fixed_dir = Path("conf/wandb_sweep/fixed")
    fixed_blocks = sorted(fixed_dir.glob("*.yaml"))
    assert fixed_blocks, f"expected at least one fixed sweep block under {fixed_dir}"


def _iter_sweep_compose_cases(*, full_grid: bool):
    from hydra.core.global_hydra import GlobalHydra

    config_dir = Path(__file__).resolve().parents[1] / "conf"
    for recipe in SWEEP_COMPOSE_RECIPES:
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            sweep = OmegaConf.to_container(
                compose(
                    config_name="sweep_gen",
                    overrides=[f"wandb_sweep={recipe}"],
                ),
                resolve=True,
            )
        GlobalHydra.instance().clear()
        parameters = sweep["parameters"]
        keys = []
        value_sets = []
        for key, spec in parameters.items():
            if "value" in spec:
                values = [spec["value"]]
            elif "values" in spec:
                values = list(spec["values"])
            elif "distribution" in spec:
                # W&B bayes/uniform sweeps are not grid-enumerable; smoke with min.
                values = [spec["min"]]
            else:
                raise KeyError(
                    f"Unsupported sweep parameter spec for {key!r}: {spec!r}"
                )
            keys.append(key)
            value_sets.append(values)

        if full_grid:
            value_products = product(*value_sets)
        else:
            value_products = [tuple(values[0] for values in value_sets)]

        for values in value_products:
            yield [
                f"{key}={_hydra_value(value)}"
                for key, value in zip(keys, values, strict=True)
            ]


def _hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)
