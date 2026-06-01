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
