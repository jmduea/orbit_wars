"""Per-model PPO overrides for preflight calibration and learn-proof gates."""

from __future__ import annotations

import json
from pathlib import Path


def default_profiles_path(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[2]
    return root / "docs" / "benchmarks" / "preflight-profiles.json"


def load_preflight_profiles(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Preflight profiles file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Preflight profiles must be a JSON object: {path}")
    return payload


def ppo_overrides_for_model(
    model: str,
    *,
    profiles_path: Path | None = None,
    repo_root: Path | None = None,
) -> tuple[str, ...]:
    """Return Hydra overrides for the promoted PPO profile of ``model``."""

    path = profiles_path or default_profiles_path(repo_root)
    payload = load_preflight_profiles(path)
    models = payload.get("models")
    if not isinstance(models, dict):
        return ()
    entry = models.get(model)
    if not isinstance(entry, dict):
        return ()
    raw = entry.get("ppo_overrides")
    if not isinstance(raw, list):
        return ()
    overrides: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            overrides.append(text)
    return tuple(overrides)
