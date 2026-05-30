from __future__ import annotations

import sys

import pytest

from src.config.schema import TrainConfig
from src.telemetry import TelemetryLogger
from src.telemetry.wandb_run_name import (
    build_sweep_run_suffix,
    compose_run_name_prefix,
    compose_sweep_display_name,
    detect_swept_keys,
    is_excluded_swept_key,
    parse_override_entries,
    resolve_sweep_display_name,
    should_apply_sweep_run_rename,
    sweep_varying_parameter_keys,
)
from src.telemetry.wandb_tags import (
    DEFAULT_TAG_CONFIG_GROUPS,
    derive_config_group_tags,
    merge_wandb_tags,
)

THROUGHPUT_2P_SPACE = {
    "training": {"values": ["2p_16", "2p_32"]},
    "training.rollout_steps": {"values": [250, 500]},
    "training.num_envs": {"value": 16},
    "seed": {"value": 42},
    "output.campaign": {"value": "2p_only_throughput"},
}


def test_build_sweep_run_suffix_is_deterministic_and_compact() -> None:
    params = {
        "training.lr": 0.0003,
        "training.rollout_steps": 250,
        "training": "2p4p_16_rotate",
    }
    assert build_sweep_run_suffix(params) == build_sweep_run_suffix(
        {
            "training": "2p4p_16_rotate",
            "training.lr": 0.0003,
            "training.rollout_steps": 250,
        }
    )
    suffix = build_sweep_run_suffix(params)
    assert "lr3e-4" in suffix
    assert "rs250" in suffix
    assert "tr2p4p16rotate" in suffix
    assert "mix_" not in suffix


def test_build_sweep_run_suffix_truncates_to_max_length() -> None:
    params = {f"training.param_{index}": index for index in range(12)}
    suffix = build_sweep_run_suffix(params, max_length=24)
    assert len(suffix) <= 24


def test_detect_swept_keys_excludes_seed_and_output_paths() -> None:
    parameters = {
        **THROUGHPUT_2P_SPACE,
        "output.wandb_dir": {"values": ["cache/wandb", "cache/wandb_alt"]},
    }
    job = parse_override_entries(
        [
            "training=2p_16",
            "training.rollout_steps=250",
            "seed=7",
            "output.campaign=other",
            "output.wandb_dir=cache/wandb_alt",
        ]
    )
    swept = detect_swept_keys(job_overrides=job, sweep_parameters=parameters)
    assert "training.rollout_steps" in swept
    assert "training" in swept
    assert "seed" not in swept
    assert "output.campaign" not in swept
    assert "output.wandb_dir" not in swept


def test_compose_run_name_prefix_omits_swept_components() -> None:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.training.total_updates = 30
    cfg.training.num_envs = 16
    cfg.seed = 42
    swept = frozenset({"training.total_updates", "training.num_envs"})
    prefix = compose_run_name_prefix(cfg, swept)
    assert "u30" not in prefix
    assert "env16" not in prefix
    assert "planet_graph_transformer" in prefix
    assert "s42" in prefix


def test_compose_sweep_display_name_joins_prefix_and_suffix() -> None:
    cfg = TrainConfig()
    cfg.model.architecture = "planet_graph_transformer"
    cfg.training.total_updates = 30
    cfg.training.num_envs = 16
    cfg.seed = 42
    swept = frozenset({"training.rollout_steps"})
    name = compose_sweep_display_name(
        cfg,
        swept_keys=swept,
        job_overrides={"training.rollout_steps": "250"},
    )
    assert name.endswith("-rs250") or "-rs250" in name
    assert "planet_graph_transformer" in name


def test_resolve_sweep_display_name_none_for_single_run(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.telemetry.wandb_run_name.is_hydra_multirun",
        lambda: False,
    )
    monkeypatch.setattr(
        "src.telemetry.wandb_run_name.is_wandb_sweep_job",
        lambda: False,
    )
    cfg = TrainConfig()
    assert resolve_sweep_display_name(cfg, sweep_parameters=THROUGHPUT_2P_SPACE) is None
    assert not should_apply_sweep_run_rename(cfg)


def test_rename_opt_out_disables_sweep_display_name(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.telemetry.wandb_run_name.is_hydra_multirun",
        lambda: True,
    )
    cfg = TrainConfig()
    cfg.telemetry.wandb.rename_from_swept_params = False
    assert resolve_sweep_display_name(cfg, sweep_parameters=THROUGHPUT_2P_SPACE) is None


def test_sweep_varying_parameter_keys_detects_distributions() -> None:
    parameters = {
        "training.lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 6e-4},
        "training.gamma": {"value": 0.99},
    }
    varying = sweep_varying_parameter_keys(parameters)
    assert "training.lr" in varying
    assert "training.gamma" not in varying


def test_is_excluded_swept_key_covers_noise_keys() -> None:
    assert is_excluded_swept_key("seed")
    assert is_excluded_swept_key("output.campaign")
    assert is_excluded_swept_key("telemetry.wandb.group")
    assert not is_excluded_swept_key("training.rollout_steps")


def test_derive_config_group_tags_from_choices() -> None:
    tags = derive_config_group_tags(
        allowlist=DEFAULT_TAG_CONFIG_GROUPS,
        choices={
            "model": "transformer_factorized",
            "training": "2p4p_32_split",
            "opponents": "random",
        },
    )
    assert tags == [
        "model:transformer_factorized",
        "opponents:random",
        "training:2p4p_32_split",
    ]


def test_derive_config_group_tags_respects_allowlist() -> None:
    tags = derive_config_group_tags(
        allowlist=["model"],
        choices={"model": "default", "training": "2p_16"},
    )
    assert tags == ["model:default"]


def test_merge_wandb_tags_sorted_dedupe() -> None:
    merged = merge_wandb_tags(
        manual=["beta", "alpha", "beta"],
        derived=["training:2p_16", "alpha"],
    )
    assert merged == ["alpha", "beta", "training:2p_16"]


class _FakeWandbRun:
    def __init__(self, **kwargs: object) -> None:
        self.config: dict[str, object] = {}
        self.name = kwargs.get("name")

    def finish(self) -> None:
        pass


class _FakeWandb:
    def __init__(self) -> None:
        self.init_kwargs: dict[str, object] = {}
        self.run: _FakeWandbRun | None = None

    def init(self, **kwargs: object) -> _FakeWandbRun:
        self.init_kwargs = kwargs
        self.run = _FakeWandbRun(**kwargs)
        return self.run


def test_telemetry_logger_merges_manual_and_derived_tags(monkeypatch) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(
        "src.telemetry.wandb_tags._hydra_runtime_choices",
        lambda: {
            "model": "transformer_factorized",
            "training": "2p4p_32_split",
        },
    )
    cfg = TrainConfig()
    cfg.telemetry.wandb.enabled = True
    cfg.telemetry.wandb.tags = ["manual", "model:legacy"]

    TelemetryLogger(cfg)

    tags = fake_wandb.init_kwargs["tags"]
    assert tags == [
        "manual",
        "model:legacy",
        "model:transformer_factorized",
        "training:2p4p_32_split",
    ]


def test_telemetry_logger_skips_derived_tags_when_disabled(monkeypatch) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(
        "src.telemetry.wandb_tags._hydra_runtime_choices",
        lambda: {"model": "transformer_factorized"},
    )
    cfg = TrainConfig()
    cfg.telemetry.wandb.enabled = True
    cfg.telemetry.wandb.tags_from_config_groups = False
    cfg.telemetry.wandb.tags = ["manual"]

    TelemetryLogger(cfg)

    assert fake_wandb.init_kwargs["tags"] == ["manual"]


def test_telemetry_logger_keeps_single_run_name_without_multirun(monkeypatch) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(
        "src.telemetry.wandb_run_name.resolve_sweep_display_name",
        lambda _cfg, **kwargs: None,
    )
    cfg = TrainConfig()
    cfg.telemetry.wandb.enabled = True
    cfg.run_name = "baseline-run-name"

    TelemetryLogger(cfg)

    assert fake_wandb.init_kwargs["name"] == "baseline-run-name"
    assert fake_wandb.run is not None
    assert fake_wandb.run.name == "baseline-run-name"


def test_telemetry_logger_renames_after_init_on_multirun(monkeypatch) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    cfg = TrainConfig()
    cfg.telemetry.wandb.enabled = True
    cfg.run_name = "baseline-run-name"
    monkeypatch.setattr(
        "src.telemetry.wandb_run_name.resolve_sweep_display_name",
        lambda _cfg, **kwargs: "sweep-display-name",
    )

    TelemetryLogger(cfg)

    assert fake_wandb.init_kwargs["name"] == "baseline-run-name"
    assert fake_wandb.run is not None
    assert fake_wandb.run.name == "sweep-display-name"
