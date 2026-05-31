from __future__ import annotations

import ast
from pathlib import Path


def test_game_package_does_not_import_jax() -> None:
    game_dir = Path(__file__).resolve().parents[1] / "src" / "game"
    offenders: list[str] = []
    for path in sorted(game_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "jax" or alias.name.startswith("jax."):
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "jax" or module.startswith("jax."):
                    offenders.append(f"{path.name}: from {module}")
    assert not offenders, "game/ must not import JAX:\n" + "\n".join(offenders)
