"""Unit tests for pre-flight gate evaluation on training JSONL records."""

from __future__ import annotations

from src.jax.preflight import (
    PreflightGateSpec,
    PreflightVerdict,
    _gate_specs,
    evaluate_gate_records,
    read_jsonl_records,
)


def _records(*rows: dict[str, float]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        payload.append(
            {
                "update": index,
                "overall_win_rate": row["win_rate"],
                "mean_active_launches_per_turn": row["launches"],
                "approx_kl": row.get("approx_kl", 0.01),
                "entropy": row.get("entropy", 0.05),
            }
        )
    return payload


def _trend_spec(**kwargs: object) -> PreflightGateSpec:
    defaults = {
        "gate_id": "beat_noop",
        "train_overrides": ("model=transformer_factorized_small",),
        "min_win_rate_delta": 0.08,
        "window_updates": 10,
    }
    defaults.update(kwargs)
    return PreflightGateSpec(**defaults)  # type: ignore[arg-type]


def test_beat_noop_verified_on_learning_trend() -> None:
    spec = _trend_spec()
    records = _records(
        *[{"win_rate": 0.2, "launches": 0.1}] * 6
        + [{"win_rate": 0.9, "launches": 0.5}] * 10
    )
    evaluation = evaluate_gate_records(
        spec,
        records,
        campaign="preflight_beat_noop",
        run_dir=None,
        checkpoint=None,
    )
    assert evaluation.verdict == PreflightVerdict.VERIFIED
    assert evaluation.win_rate_delta is not None
    assert evaluation.win_rate_delta >= spec.min_win_rate_delta


def test_beat_noop_not_verified_on_flat_trend() -> None:
    spec = _trend_spec()
    records = _records(*[{"win_rate": 0.5, "launches": 0.5}] * 16)
    evaluation = evaluate_gate_records(
        spec,
        records,
        campaign="preflight_beat_noop",
        run_dir=None,
        checkpoint=None,
    )
    assert evaluation.verdict == PreflightVerdict.NOT_VERIFIED
    assert any("win_rate_delta" in reason for reason in evaluation.reasons)


def test_curriculum_requires_promotion_event() -> None:
    spec = _gate_specs("transformer_factorized_small")["curriculum_staged"]
    records: list[dict[str, object]] = [
        {
            "update": 1,
            "overall_win_rate": 0.6,
            "approx_kl": 0.01,
            "entropy": 0.05,
            "curriculum_phase_events": [{"event": "curriculum_stage_promoted"}],
        }
    ]
    evaluation = evaluate_gate_records(
        spec,
        records,
        campaign="preflight_curriculum_staged",
        run_dir=None,
        checkpoint=None,
    )
    assert evaluation.verdict == PreflightVerdict.VERIFIED
    assert evaluation.curriculum_promotions == 1


def test_read_jsonl_records_roundtrip(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"update": 1, "overall_win_rate": 0.5}\n{"update": 2, "overall_win_rate": 0.6}\n',
        encoding="utf-8",
    )
    records = read_jsonl_records(path)
    assert len(records) == 2
    assert records[1]["overall_win_rate"] == 0.6


def test_preflight_gate_overrides_compose() -> None:
    from src.config import compose_hydra_train_config

    for gate_id, spec in _gate_specs("transformer_factorized_small").items():
        compose_hydra_train_config(list(spec.train_overrides))
