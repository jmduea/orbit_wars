"""CLI tests for ``ow benchmark map-pool``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli.benchmark import build_parser, main
from src.cli.map_pool_benchmark import (
    DEFAULT_MAX_EXTRAPOLATED_SECS,
    _bake_gate_ok,
    run_bake_cli,
)
from src.jax.map_pool.bake import bake_one_entry, save_pool_npz


def test_parser_registers_map_pool_subcommands():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["map-pool"])
    args = parser.parse_args(["map-pool", "validate", "--pool", "x.npz"])
    assert args.command == "map-pool"
    assert args.map_pool_command == "validate"


def test_validate_tiny_pool(tmp_path: Path):
    entries = [bake_one_entry(0), bake_one_entry(1)]
    pool = tmp_path / "tiny.npz"
    save_pool_npz(str(pool), entries)
    rc = main(["map-pool", "validate", "--pool", str(pool)])
    assert rc == 0


def test_bake_count_zero_fails(tmp_path: Path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "map-pool",
            "bake",
            "--count",
            "0",
            "--label",
            "bad",
            "--accept-extrapolated-secs",
            "60",
        ]
    )
    assert run_bake_cli(args) == 1


def test_bake_without_profile_or_accept_blocked():
    ok, message, _ = _bake_gate_ok(
        count=500,
        profile_path=None,
        accept_extrapolated_secs=None,
    )
    assert not ok
    assert "profile" in message


def test_bake_blocks_unacceptable_extrapolation(tmp_path: Path):
    profile = tmp_path / "slow.json"
    profile.write_text(
        json.dumps({"mean_secs_per_map": 120.0, "success_rate": 1.0}) + "\n",
        encoding="utf-8",
    )
    ok, message, _ = _bake_gate_ok(
        count=500,
        profile_path=profile,
        accept_extrapolated_secs=None,
    )
    assert not ok
    assert "extrapolated" in message
    assert str(int(DEFAULT_MAX_EXTRAPOLATED_SECS)) in message or "1800" in message


def test_bake_accepts_override_extrapolation(tmp_path: Path):
    profile = tmp_path / "slow.json"
    profile.write_text(
        json.dumps({"mean_secs_per_map": 120.0, "success_rate": 1.0}) + "\n",
        encoding="utf-8",
    )
    ok, _, _ = _bake_gate_ok(
        count=500,
        profile_path=profile,
        accept_extrapolated_secs=70000.0,
    )
    assert ok
