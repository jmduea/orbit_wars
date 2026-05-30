"""Derive W&B ``group:value`` tags from Hydra config group selections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

DEFAULT_TAG_CONFIG_GROUPS: tuple[str, ...] = (
    "model",
    "format",
    "opponents",
    "curriculum",
    "reward",
)

_HYDRA_CONFIG_GROUP_KEYS: frozenset[str] = frozenset(DEFAULT_TAG_CONFIG_GROUPS)


def _hydra_runtime_choices() -> dict[str, str]:
    try:
        from hydra.core.hydra_config import HydraConfig
    except Exception:
        return {}
    if not HydraConfig.initialized():
        return {}
    choices = getattr(HydraConfig.get().runtime, "choices", None)
    if not isinstance(choices, Mapping):
        return {}
    return {str(key): str(value) for key, value in choices.items()}


def _choices_from_overrides(overrides: Sequence[str]) -> dict[str, str]:
    choices: dict[str, str] = {}
    for raw in overrides:
        if not isinstance(raw, str) or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        if "." in key:
            continue
        if key not in _HYDRA_CONFIG_GROUP_KEYS:
            continue
        choices[key] = value
    return choices


def derive_config_group_tags(
    *,
    allowlist: Sequence[str],
    choices: Mapping[str, str] | None = None,
    overrides: Sequence[str] | None = None,
) -> list[str]:
    """Build sorted ``group:value`` tags for allowlisted Hydra config groups.

    Args:
        allowlist: Config group names to emit (for example ``model``, ``format``).
        choices: Optional explicit group selections; defaults to Hydra runtime
            choices plus ``group=value`` entries from ``overrides``.
        overrides: Hydra override strings used when runtime choices are absent.

    Returns:
        Sorted tag strings such as ``model:transformer_factorized``.
    """

    merged: dict[str, str] = {}
    merged.update(_hydra_runtime_choices())
    if overrides:
        merged.update(_choices_from_overrides(overrides))
    if choices:
        merged.update({str(key): str(value) for key, value in choices.items()})

    allowed = {str(group).strip() for group in allowlist if str(group).strip()}
    tags = [
        f"{group}:{merged[group]}"
        for group in sorted(merged)
        if group in allowed and merged[group]
    ]
    return tags


def merge_wandb_tags(
    *,
    manual: Sequence[str],
    derived: Sequence[str],
) -> list[str]:
    """Return sorted deduplicated union of manual and derived W&B tags."""

    return sorted({str(tag) for tag in manual} | {str(tag) for tag in derived})
