"""CLI dispatch tests for ``ow benchmark rollout-phase-profile``."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src.cli import benchmark as benchmark_cli
from src.cli.benchmark.common import REPO_ROOT
from src.cli.benchmark.rollout_phase_profile import run_rollout_phase_profile_cli


def test_rollout_phase_profile_cli_uses_in_process_path_when_repo_matches(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_in_process(args: Namespace) -> int:
        captured["preset"] = args.preset
        captured["updates"] = args.updates
        out_path = tmp_path / "profile.json"
        out_path.write_text(json.dumps({"preset": args.preset}), encoding="utf-8")
        args.out = out_path
        return 0

    monkeypatch.setattr(
        "src.cli.benchmark.rollout_phase_profile._run_profile_in_process",
        fake_in_process,
    )
    args = Namespace(
        preset="admission",
        updates=3,
        warmup=1,
        max_measured_update=5,
        model=None,
        json=True,
        full_geometry=False,
        train_overrides=["task=map_pool"],
        out=tmp_path / "out.json",
        repo_root=REPO_ROOT,
    )

    assert run_rollout_phase_profile_cli(args) == 0
    assert captured == {"preset": "admission", "updates": 3}


@patch("src.cli.benchmark.rollout_phase_profile.subprocess.run")
def test_rollout_phase_profile_cli_subprocess_when_repo_differs(
    mock_run: object,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    mock_run.return_value = SimpleNamespace(returncode=0)
    sibling = REPO_ROOT.parent / "other-worktree"
    monkeypatch.setattr(
        "src.cli.benchmark.rollout_phase_profile._integration_root",
        lambda _args: sibling,
    )
    args = Namespace(
        preset="admission",
        updates=3,
        warmup=1,
        max_measured_update=5,
        model=None,
        json=False,
        full_geometry=False,
        train_overrides=[],
        out=None,
        repo_root=sibling,
    )

    assert run_rollout_phase_profile_cli(args) == 0
    mock_run.assert_called_once()
    command = mock_run.call_args.args[0]
    assert command[0:2] == ["uv", "run"]
    assert "rollout_phase_profile" in command[4]


@patch("src.cli.benchmark.rollout_phase_profile._run_profile_in_process")
def test_benchmark_main_rollout_phase_profile_dispatches(mock_profile) -> None:
    mock_profile.return_value = 0
    assert (
        benchmark_cli.main(
            [
                "rollout-phase-profile",
                "--preset",
                "admission",
                "--updates",
                "2",
                "--json",
            ]
        )
        == 0
    )
    mock_profile.assert_called_once()
