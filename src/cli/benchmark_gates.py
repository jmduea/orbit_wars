"""Load and run composable preflight gate recipes from ``conf/benchmark/gates/``."""

from __future__ import annotations

import json
from pathlib import Path

from src.jax.preflight_gate_loader import (
    GATES_DIR,
    REPO_ROOT,
    build_gate_spec,
    gate_yaml_paths,
    load_gate_yaml,
)

__all__ = [
    "GATES_DIR",
    "REPO_ROOT",
    "build_gate_spec",
    "gate_yaml_paths",
    "load_gate_recipe",
    "load_gate_yaml",
    "list_gate_recipes",
    "run_gate_cli",
]


def load_gate_recipe(gate_id: str) -> dict[str, object]:
    return load_gate_yaml(gate_id)


def list_gate_recipes() -> list[dict[str, object]]:
    recipes: list[dict[str, object]] = []
    for path in gate_yaml_paths():
        gate_id = path.stem
        payload = load_gate_yaml(gate_id)
        recipe_gate_id = payload.get("gate_id") or payload.get("gate") or gate_id
        primitive = payload.get("primitive")
        if gate_id == "win_proof_tournament":
            recipe_gate_id = "win_proof"
            primitive = "uv run ow benchmark tournament-proof"
        recipes.append(
            {
                "gate_id": recipe_gate_id,
                "title": payload.get("title") or payload.get("description"),
                "default_model": payload.get("default_model"),
                "path": payload.get("path"),
                "workflow": payload.get("workflow"),
                "primitive": primitive,
            }
        )
    return recipes


def run_gate_cli(
    gate_id: str,
    *,
    model: str | None = None,
    output_root: Path,
    dry_run: bool = False,
    verbose: bool = False,
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
        verbose=verbose,
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
