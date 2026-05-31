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
