"""Load and run composable preflight gate recipes from ``conf/benchmark/gates/``."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GATES_DIR = REPO_ROOT / "conf" / "benchmark" / "gates"


def gate_yaml_paths() -> list[Path]:
    if not GATES_DIR.is_dir():
        return []
    return sorted(GATES_DIR.glob("*.yaml"))


def load_gate_recipe(gate_id: str) -> dict[str, object]:
    path = GATES_DIR / f"{gate_id}.yaml"
    if not path.is_file():
        known = [item.stem for item in gate_yaml_paths()]
        raise FileNotFoundError(
            f"Unknown gate recipe {gate_id!r}. Known YAML gates: {', '.join(known) or '(none)'}"
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Gate recipe must be a mapping: {path}")
    recipe_gate_id = str(payload.get("gate_id") or path.stem)
    if recipe_gate_id != gate_id:
        raise ValueError(
            f"Gate recipe gate_id mismatch in {path}: expected {gate_id!r}, got {recipe_gate_id!r}"
        )
    payload["path"] = str(path.relative_to(REPO_ROOT))
    return payload


def list_gate_recipes() -> list[dict[str, object]]:
    recipes: list[dict[str, object]] = []
    for path in gate_yaml_paths():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        gate_id = str(payload.get("gate_id") or path.stem)
        recipes.append(
            {
                "gate_id": gate_id,
                "title": payload.get("title"),
                "default_model": payload.get("default_model"),
                "path": str(path.relative_to(REPO_ROOT)),
                "workflow": payload.get("workflow"),
                "primitive": payload.get("primitive"),
            }
        )
    return recipes


def run_gate_cli(
    gate_id: str,
    *,
    model: str | None = None,
    output_root: Path,
    dry_run: bool = False,
    thresholds_path: Path | None = None,
    profiles_path: Path | None = None,
    train_overrides: tuple[str, ...] = (),
    out: Path | None = None,
) -> int:
    from src.jax.preflight import (
        PreflightVerdict,
        gate_evaluation_to_dict,
        run_preflight_gate,
        write_report,
    )

    recipe = load_gate_recipe(gate_id)
    resolved_model = model or str(recipe.get("default_model") or "transformer_factorized_small")
    evaluation = run_preflight_gate(
        gate_id,
        model=resolved_model,
        output_root=output_root,
        repo_root=REPO_ROOT,
        dry_run=dry_run,
        thresholds_path=thresholds_path,
        profiles_path=profiles_path,
        extra_train_overrides=train_overrides,
    )
    report: dict[str, object] = {
        "gate": gate_id,
        "recipe": recipe.get("path"),
        "model": resolved_model,
        "verdict": evaluation.verdict.value,
        "stage": gate_evaluation_to_dict(evaluation),
    }
    if out is not None:
        write_report(out, report)
    print(json.dumps(report, indent=2))
    return 0 if evaluation.verdict == PreflightVerdict.VERIFIED else 1
