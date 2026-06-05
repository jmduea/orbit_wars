from __future__ import annotations

OPPONENT_LATEST = 0
OPPONENT_HISTORICAL = 1
OPPONENT_NEAREST_SNIPER = 2
OPPONENT_TURTLE = 3
OPPONENT_OPPORTUNISTIC = 4
OPPONENT_RANDOM = 5
OPPONENT_NOOP = 6

OPPONENT_FAMILY_NAMES: tuple[str, ...] = (
    "latest",
    "historical",
    "nearest_sniper",
    "turtle",
    "opportunistic",
    "random",
    "noop",
)

OPPONENT_FAMILY_COUNT = len(OPPONENT_FAMILY_NAMES)
CURRICULUM_OPPONENT_FAMILIES = frozenset(OPPONENT_FAMILY_NAMES)

_JAX_TRAINING_OPPONENT_MODES = frozenset({"self", "random", "noop", "no_op"})


def normalize_jax_training_opponent_mode(opponent: str) -> str:
    """Normalize rollout opponent mode strings for JAX training."""

    key = opponent.strip().lower()
    if key in {"no_op", "noop"}:
        return "noop"
    return key


def is_noop_jax_training_opponent_mode(opponent: str) -> bool:
    return normalize_jax_training_opponent_mode(opponent) == "noop"


def validate_jax_training_opponent_mode(opponent: str) -> None:
    normalized = normalize_jax_training_opponent_mode(opponent)
    if normalized not in {"self", "random", "noop"}:
        raise ValueError(
            "JAX training supports opponent='self', opponent='random', "
            "opponent='noop', or opponent='no_op', "
            f"got {opponent!r}."
        )
