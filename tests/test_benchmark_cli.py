"""CLI wiring for ``ow benchmark``."""

from __future__ import annotations

from pathlib import Path

from src.cli import benchmark as benchmark_cli
from src.cli.benchmark import build_parser
from src.jax.training_benchmark import TrainingBenchmarkResult


def test_benchmark_parser_has_training_sanity_and_learn_proof() -> None:
    parser = build_parser()
    training = parser.parse_args(
        ["training", "--label", "smoke", "--out", "/tmp/bench.json", "--updates", "5"]
    )
    assert training.command == "training"
    assert training.label == "smoke"
    planet_flow_training = parser.parse_args(
        [
            "training",
            "--preset",
            "planet_flow_p0",
            "--label",
            "planet_flow",
            "--out",
            "/tmp/planet_flow.json",
            "--assert-min-env-steps-per-sec",
            "4000",
        ]
    )
    assert planet_flow_training.preset == "planet_flow_p0"
    assert planet_flow_training.assert_min_env_steps_per_sec == 4000.0

    sanity = parser.parse_args(["sanity", "--out", "/tmp/sanity.json"])
    assert sanity.command == "sanity"

    learn_proof = parser.parse_args(
        [
            "learn-proof",
            "--gate",
            "beat_noop",
            "--thresholds-path",
            "/tmp/pf-calibration.json",
        ]
    )
    assert learn_proof.command == "learn-proof"
    assert learn_proof.gate == "beat_noop"
    assert learn_proof.thresholds_path == Path("/tmp/pf-calibration.json")

    calibrate = parser.parse_args(
        ["calibrate", "--analyze-only", "--analyze-campaigns"]
    )
    assert calibrate.command == "calibrate"
    assert calibrate.analyze_only is True
    assert calibrate.analyze_campaigns == "preflight_calibrate_*"

    seed_sched = parser.parse_args(
        ["calibrate-seed-scheduler", "--analyze-only", "--dry-run"]
    )
    assert seed_sched.command == "calibrate-seed-scheduler"
    assert seed_sched.analyze_only is True

    held_out = parser.parse_args(
        [
            "learn-proof",
            "--eval-checkpoint",
            "/tmp/ckpt.pkl",
            "--baselines",
            "random",
        ]
    )
    assert held_out.eval_checkpoint == Path("/tmp/ckpt.pkl")

    shortlist = parser.parse_args(
        [
            "shortlist-planet-flow-sweep",
            "--sweep-id",
            "j0epauu2",
            "--out",
            "/tmp/shortlist.json",
        ]
    )
    assert shortlist.command == "shortlist-planet-flow-sweep"
    assert shortlist.sweep_id == "j0epauu2"

    smoke = parser.parse_args(
        [
            "planet-flow-noop-smoke",
            "--shortlist",
            "/tmp/shortlist.json",
            "--top-k",
            "2",
        ]
    )
    assert smoke.command == "planet-flow-noop-smoke"
    assert smoke.top_k == 2


def test_planet_flow_training_benchmark_requires_control_metrics(
    monkeypatch, tmp_path, capsys
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "training",
            "--preset",
            "planet_flow_p0",
            "--label",
            "planet_flow",
            "--out",
            str(tmp_path / "planet_flow.json"),
            "--updates",
            "1",
        ]
    )
    result = TrainingBenchmarkResult(
        label="planet_flow",
        overrides=("model=planet_flow_target_heatmap",),
        updates=1,
        warmup=0,
        measured_updates=1,
        seconds_total=1.0,
        seconds_per_update_mean=1.0,
        compile_seconds_to_update_3=None,
        devices=("cpu",),
        default_backend="cpu",
        num_envs=1,
        rollout_steps=1,
        update_metric_means={},
        rollout_metric_means={
            "planet_flow_control_emitted_launch_count": None,
            "planet_flow_control_emitted_ship_mass_rate": None,
            "planet_flow_emitted_launch_count_delta_vs_control": None,
        },
    )

    monkeypatch.setattr(benchmark_cli, "_init_benchmark_runtime", lambda: None)
    monkeypatch.setattr(
        "src.jax.training_benchmark.run_training_benchmark",
        lambda *args, **kwargs: result,
    )

    exit_code = benchmark_cli.run_training_benchmark_cli(args)

    assert exit_code == 1
    assert "compiler-control metrics" in capsys.readouterr().err


def test_makefile_e2e_throughput_target_uses_baseline_assert() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "test-launch-hygiene-e2e-throughput:" in makefile
    assert "ow benchmark training" in makefile
    assert "docs/benchmarks/launch-hygiene-e2e-baseline.json" in makefile
    assert "--assert-within-pct" in makefile
    parser = build_parser()
    training = parser.parse_args(
        [
            "training",
            "--preset",
            "primary",
            "--label",
            "gate",
            "--out",
            "/tmp/gate.json",
            "--baseline",
            "docs/benchmarks/launch-hygiene-e2e-baseline.json",
            "--assert-within-pct",
            "10",
        ]
    )
    assert training.preset == "primary"
    assert training.updates is None
    assert training.baseline == Path("docs/benchmarks/launch-hygiene-e2e-baseline.json")
    assert training.assert_within_pct == 10.0

    training_parser = next(
        action
        for action in parser._actions
        if getattr(action, "choices", None) and "training" in action.choices
    ).choices["training"]
    training_help = training_parser.format_help()
    assert "primary" in training_help
    assert "shield_cheap" in training_help

