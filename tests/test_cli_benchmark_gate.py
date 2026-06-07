from __future__ import annotations

import json

from src.cli import benchmark as benchmark_cli
from src.cli.benchmark_gates import list_gate_recipes, load_gate_recipe


def test_list_gate_recipes_includes_beat_noop() -> None:
    recipes = list_gate_recipes()
    gate_ids = {item["gate_id"] for item in recipes}
    assert "beat_noop" in gate_ids
    assert "beat_random" in gate_ids
    assert "curriculum_staged" in gate_ids


def test_load_gate_recipe_beat_noop() -> None:
    recipe = load_gate_recipe("beat_noop")
    assert recipe["gate_id"] == "beat_noop"
    assert recipe["default_model"] == "transformer_factorized_small"


def test_load_gate_recipe_beat_random() -> None:
    recipe = load_gate_recipe("beat_random")
    assert recipe["gate_id"] == "beat_random"
    assert recipe["default_model"] == "transformer_factorized_small"
    assert recipe["ladder_index"] == 1
    assert recipe["thresholds_key"] == "learning_signal"


def test_load_gate_recipe_curriculum_staged() -> None:
    recipe = load_gate_recipe("curriculum_staged")
    assert recipe["gate_id"] == "curriculum_staged"
    assert recipe["default_model"] == "transformer_factorized"
    assert recipe["ladder_index"] == 2


def test_benchmark_gate_list_cli(capsys) -> None:
    assert benchmark_cli.main(["gate", "--list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    gate_ids = {item["gate_id"] for item in payload["gates"]}
    assert gate_ids >= {"beat_noop", "beat_random", "curriculum_staged"}


def test_benchmark_gate_dry_run(capsys) -> None:
    assert benchmark_cli.main(["gate", "run", "beat_noop", "--dry-run"]) == 1
    out = capsys.readouterr().out
    start = out.index('{\n  "gate":')
    payload = json.loads(out[start:])
    assert payload["gate"] == "beat_noop"
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["stage"]["reasons"] == ["dry_run"]


def test_benchmark_gate_positional_alias_dry_run(capsys) -> None:
    assert benchmark_cli.main(["gate", "beat_noop", "--dry-run"]) == 1


def test_benchmark_gate_beat_random_dry_run(capsys) -> None:
    assert benchmark_cli.main(["gate", "beat_random", "--dry-run"]) == 1
    out = capsys.readouterr().out
    start = out.index('{\n  "gate":')
    payload = json.loads(out[start:])
    assert payload["gate"] == "beat_random"
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["stage"]["reasons"] == ["dry_run"]


def test_benchmark_gate_unknown_id_exits_2(capsys) -> None:
    assert benchmark_cli.main(["gate", "run", "not_a_real_gate", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "Unknown gate id" in err
    assert "gate list" in err


def test_benchmark_gate_curriculum_staged_dry_run(capsys) -> None:
    assert benchmark_cli.main(["gate", "curriculum_staged", "--dry-run"]) == 1
    out = capsys.readouterr().out
    start = out.index('{\n  "gate":')
    payload = json.loads(out[start:])
    assert payload["gate"] == "curriculum_staged"
    assert payload["verdict"] == "INCONCLUSIVE"
    assert payload["stage"]["reasons"] == ["dry_run"]
