"""Python runtime opponents used by tournament replay and packaging probes."""

from __future__ import annotations

from src.artifacts.tournament.runner import build_baseline_agent, run_match


def _opponent_move_totals(opponent_name: str, *, seed: int = 43) -> tuple[int, int]:
    baseline = build_baseline_agent(opponent_name)
    _outcome, env, _timing = run_match(
        match_id=f"probe_{opponent_name}",
        format_name="2p_vs_baseline",
        seed=seed,
        agent_ids=("learner", f"baseline:{opponent_name}"),
        agents=[lambda _obs: [], baseline],
        max_steps=500,
    )
    actions = [(step[1].action or []) for step in env.steps if len(step) > 1]
    return sum(len(action) for action in actions), sum(
        1 for action in actions if action
    )


def test_random_runtime_opponent_launches_fleets() -> None:
    total_moves, nonempty_steps = _opponent_move_totals("random")
    assert total_moves > 0
    assert nonempty_steps > 0


def test_random_runtime_opponent_differs_from_noop() -> None:
    random_moves, _ = _opponent_move_totals("random")
    noop_moves, _ = _opponent_move_totals("noop")
    assert random_moves > noop_moves


def test_nearest_sniper_runtime_opponent_launches_fleets() -> None:
    total_moves, nonempty_steps = _opponent_move_totals("nearest_sniper")
    assert total_moves > 0
    assert nonempty_steps > 0


def test_nearest_sniper_launches_on_seed_44_regression() -> None:
    """Seed 44 blocked every cheap-shield sniper candidate before baseline shield-off."""
    total_moves, nonempty_steps = _opponent_move_totals("nearest_sniper", seed=44)
    assert total_moves > 0
    assert nonempty_steps > 0
