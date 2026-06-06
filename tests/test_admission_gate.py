from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli import benchmark as benchmark_cli
from src.cli.benchmark_gates import (
    DEFAULT_ADMISSION_THROUGHPUT_BASELINE,
    list_gate_recipes,
    load_gate_recipe,
    resolve_learning_gate_id,
    resolve_throughput_options,
    run_gate_cli,
)
from src.jax.preflight_config_summary import format_gate_train_config_summary
from src.jax.preflight_gate_loader import build_gate_spec
from src.jax.admission_throughput import run_throughput_gate
from src.jax.training_benchmark import load_e2e_baseline


def _timing_record(
    update: int,
    *,
    update_seconds: float,
    env_steps_per_sec: float,
) -> dict[str, object]:
    env_steps = env_steps_per_sec * update_seconds
    return {
        "update": update,
        "update_seconds": update_seconds,
        "env_steps_per_sec": env_steps_per_sec,
        "env_steps": env_steps,
        "samples": env_steps * 5,
    }


def test_list_gate_recipes_includes_admission() -> None:
    gates = {item["gate_id"]: item for item in list_gate_recipes()}
    assert "admission" in gates
    assert gates["admission"]["learning_gate"] == "beat_noop"
    assert gates["admission"]["primitive"] == "ow benchmark gate run admission"


def test_resolve_learning_gate_id_delegates_to_beat_noop() -> None:
    from src.cli.benchmark_gates import load_gate_recipe

    recipe = load_gate_recipe("admission")
    assert resolve_learning_gate_id(recipe, "admission") == "beat_noop"


def test_learning_first_baseline_loads() -> None:
    baseline_path = DEFAULT_ADMISSION_THROUGHPUT_BASELINE
    if not baseline_path.is_file():
        pytest.skip("learning-first baseline not present in checkout")
    baseline = load_e2e_baseline(baseline_path)
    assert baseline["gate"] == "launch_hygiene_e2e_throughput"


def test_resolve_throughput_options_uses_learning_first_baseline() -> None:
    enabled, baseline, within_pct, window = resolve_throughput_options(
        "admission",
        include_throughput=False,
        throughput_baseline=None,
        throughput_within_pct=None,
    )
    assert enabled is True
    assert baseline == DEFAULT_ADMISSION_THROUGHPUT_BASELINE
    assert within_pct == pytest.approx(10.0)
    assert window.warmup == 2
    assert window.max_measured_update == 20


def test_admission_gate_recipe_includes_operator_locked_overrides() -> None:
    recipe = load_gate_recipe("admission")
    overrides = recipe.get("train_overrides")
    assert isinstance(overrides, list)
    assert "model.max_moves_k=2" in overrides
    assert "training.rollout_steps=256" in overrides
    assert "task.candidate_count=3" in overrides
    assert "telemetry.wandb.enabled=true" in overrides
    assert "telemetry.wandb.group=preflight" in overrides
    assert "artifacts.replay.enabled=false" in overrides


def test_admission_gate_dry_run_resolves_max_moves_k_two() -> None:
    recipe = load_gate_recipe("admission")
    learning_gate_id = resolve_learning_gate_id(recipe, "admission")
    spec = build_gate_spec(learning_gate_id, model="transformer_factorized_small")
    overrides = [
        "output.campaign=preflight_admission",
        "output.root=outputs",
        *spec.train_overrides,
        *recipe.get("train_overrides", []),
    ]
    summary = "\n".join(format_gate_train_config_summary(overrides))
    assert "max_moves_k=2" in summary


def test_admission_gate_dry_run_cli(capsys) -> None:
    assert benchmark_cli.main(["gate", "run", "admission", "--dry-run"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["gate"] == "admission"
    assert payload["learning_gate"] == "beat_noop"
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["throughput"]["verdict"] == "INCONCLUSIVE"
    assert payload["throughput_verdict"] == "INCONCLUSIVE"
    assert payload["admission_passed"] is False


def test_admission_gate_dry_run_writes_combined_json(tmp_path: Path, capsys) -> None:
    out_path = tmp_path / "admission.json"
    exit_code = run_gate_cli(
        "admission",
        output_root=Path("outputs"),
        dry_run=True,
        out=out_path,
    )
    assert exit_code == 1
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["gate"] == "admission"
    assert payload["learning_gate"] == "beat_noop"
    assert "throughput" in payload
    assert payload["admission_passed"] is False
    assert json.loads(capsys.readouterr().out)["gate"] == "admission"


def _write_minimal_e2e_baseline(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "gate": "launch_hygiene_e2e_throughput",
                "runs": [{}, {}, {}],
                "aggregate": {
                    "env_steps_per_sec": {"mean": 5000.0},
                    "samples_per_sec": {"mean": 25000.0},
                    "seconds_per_update_mean": {"mean": 1.65},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_throughput_gate_against_baseline(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    _write_minimal_e2e_baseline(baseline_path)

    log_path = tmp_path / "run_jax.jsonl"
    records = [
        _timing_record(update, update_seconds=1.65, env_steps_per_sec=4950.0)
        for update in range(3, 21)
    ]
    log_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    payload, exit_code = run_throughput_gate(
        log_path,
        baseline_path=baseline_path,
        within_pct=10.0,
    )
    assert exit_code == 0
    assert payload["verdict"] == "VERIFIED"
    assert payload["gate_passed"] is True
