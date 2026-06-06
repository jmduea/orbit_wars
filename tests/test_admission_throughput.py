from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmark.admission_throughput import extract_throughput_from_records
from src.benchmark.jsonl_window import ThroughputWindow, resolve_log_path_from_input
from src.cli import benchmark as benchmark_cli


def _timing_record(
    update: int,
    *,
    update_seconds: float,
    env_steps_per_sec: float,
    rollout_seconds: float | None = None,
    ppo_seconds: float | None = None,
) -> dict[str, object]:
    env_steps = env_steps_per_sec * update_seconds
    record: dict[str, object] = {
        "update": update,
        "update_seconds": update_seconds,
        "env_steps_per_sec": env_steps_per_sec,
        "env_steps": env_steps,
        "samples": env_steps * 5,
    }
    if rollout_seconds is not None:
        record["rollout_seconds"] = rollout_seconds
    if ppo_seconds is not None:
        record["ppo_seconds"] = ppo_seconds
    return record


def test_extract_throughput_skips_warmup_and_caps_at_update_20() -> None:
    records = [
        _timing_record(1, update_seconds=10.0, env_steps_per_sec=100.0),
        _timing_record(2, update_seconds=9.0, env_steps_per_sec=200.0),
        *[
            _timing_record(
                update,
                update_seconds=2.0,
                env_steps_per_sec=1000.0,
                rollout_seconds=1.2,
                ppo_seconds=0.6,
            )
            for update in range(3, 21)
        ],
        _timing_record(21, update_seconds=1.0, env_steps_per_sec=5000.0),
    ]

    payload = extract_throughput_from_records(records)

    assert payload["measured_updates"] == 18
    assert payload["updates_in_window"] == list(range(3, 21))
    assert payload["seconds_total"] == pytest.approx(36.0)
    assert payload["env_steps"] == 36000
    assert payload["env_steps_per_sec"] == pytest.approx(1000.0)
    assert payload["seconds_per_update_mean"] == pytest.approx(2.0)
    assert payload["rollout_seconds_per_update_mean"] == pytest.approx(1.2)
    assert payload["ppo_seconds_per_update_mean"] == pytest.approx(0.6)


def test_extract_throughput_requires_rows_in_window() -> None:
    records = [_timing_record(1, update_seconds=1.0, env_steps_per_sec=10.0)]
    with pytest.raises(ValueError, match="no per-update timing rows"):
        extract_throughput_from_records(records)


def test_resolve_log_path_from_gate_result(tmp_path: Path) -> None:
    log_path = tmp_path / "run_jax.jsonl"
    log_path.write_text("{}\n", encoding="utf-8")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "gate": "beat_noop",
                "stage": {"log_path": str(log_path)},
            }
        ),
        encoding="utf-8",
    )

    resolved_log, gate_result = resolve_log_path_from_input(gate_path)

    assert resolved_log == log_path
    assert gate_result == gate_path


def test_admission_throughput_cli_from_jsonl(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "preflight_beat_noop_jax.jsonl"
    records = [
        _timing_record(update, update_seconds=2.0, env_steps_per_sec=4000.0)
        for update in range(3, 21)
    ]
    log_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    assert benchmark_cli.main(["admission-throughput", str(log_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gate"] == "admission_throughput"
    assert payload["env_steps_per_sec"] == pytest.approx(4000.0)
    assert payload["seconds_per_update_mean"] == pytest.approx(2.0)


def test_admission_throughput_cli_from_gate_result(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "run_jax.jsonl"
    records = [
        _timing_record(update, update_seconds=1.0, env_steps_per_sec=5000.0)
        for update in range(3, 21)
    ]
    log_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    gate_path = tmp_path / "learn_proof.json"
    gate_path.write_text(
        json.dumps({"stage": {"log_path": str(log_path)}}),
        encoding="utf-8",
    )

    assert benchmark_cli.main(["admission-throughput", str(gate_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["log_path"] == str(log_path)
    assert payload["gate_result_path"] == str(gate_path)


def test_custom_window_bounds(tmp_path: Path) -> None:
    records = [
        _timing_record(4, update_seconds=4.0, env_steps_per_sec=250.0),
        _timing_record(5, update_seconds=2.0, env_steps_per_sec=500.0),
    ]
    payload = extract_throughput_from_records(
        records,
        window=ThroughputWindow(warmup=3, max_measured_update=5),
    )
    assert payload["measured_updates"] == 2
    assert payload["seconds_total"] == pytest.approx(6.0)
