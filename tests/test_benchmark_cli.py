"""CLI wiring for ``ow benchmark``."""

from __future__ import annotations

from pathlib import Path

from src.cli.benchmark import build_parser


def test_benchmark_parser_has_training_sanity_and_learn_proof() -> None:
    parser = build_parser()
    training = parser.parse_args(
        ["training", "--label", "smoke", "--out", "/tmp/bench.json", "--updates", "5"]
    )
    assert training.command == "training"
    assert training.label == "smoke"

    sanity = parser.parse_args(["sanity", "--out", "/tmp/sanity.json"])
    assert sanity.command == "sanity"

    learn_proof = parser.parse_args(["learn-proof", "--gate", "beat_noop"])
    assert learn_proof.command == "learn-proof"
    assert learn_proof.gate == "beat_noop"

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
