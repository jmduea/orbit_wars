"""Unit tests for pre-flight gate evaluation on training JSONL records."""

from __future__ import annotations

from src.jax.preflight import (
    PreflightGateSpec,
    PreflightVerdict,
    _gate_specs,
    evaluate_gate_records,
    read_jsonl_records,
    run_preflight_gate,
)


def _records(
    *rows: dict[str, float],
    include_planet_flow_control: bool = False,
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        record = {
            "update": index,
            "overall_win_rate": row["win_rate"],
            "mean_active_launches_per_turn": row["launches"],
            "approx_kl": row.get("approx_kl", 0.01),
            "entropy": row.get("entropy", 0.05),
        }
        if include_planet_flow_control:
            record.update(
                {
                    "planet_flow_control_emitted_launch_count": 1.0,
                    "planet_flow_control_emitted_ship_mass_rate": 0.5,
                    "planet_flow_emitted_launch_count_delta_vs_control": 0.0,
                }
            )
        payload.append(record)
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


def test_planet_flow_preflight_gate_overrides_compose_with_p0_guards() -> None:
    from src.config import compose_hydra_train_config

    spec = _gate_specs("planet_flow_target_heatmap")["beat_random"]
    cfg = compose_hydra_train_config(list(spec.train_overrides))

    assert cfg.model.pointer_decoder == "planet_flow_target_heatmap"
    assert cfg.curriculum.enabled is False
    assert cfg.artifacts.artifact_pipeline.enabled is True
    assert cfg.artifacts.artifact_pipeline.replay_async is True
    assert cfg.artifacts.replay.enabled is True


def test_planet_flow_preflight_reports_needs_calibration() -> None:
    spec = _gate_specs("planet_flow_target_heatmap")["beat_noop"]
    records = _records(
        *[{"win_rate": 0.1, "launches": 0.2}] * 6
        + [{"win_rate": 0.9, "launches": 0.6}] * 10
    )

    evaluation = evaluate_gate_records(
        spec,
        records,
        campaign="preflight_beat_noop",
        run_dir=None,
        checkpoint=None,
    )

    assert evaluation.verdict == PreflightVerdict.INCONCLUSIVE
    assert any("needs-calibration" in reason for reason in evaluation.reasons)


def test_planet_flow_preflight_requires_compiler_control_metrics() -> None:
    spec = _trend_spec(
        require_planet_flow_control_metrics=True,
        min_win_rate_delta=0.08,
    )
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

    assert evaluation.verdict == PreflightVerdict.INCONCLUSIVE
    assert any("compiler-control metric" in reason for reason in evaluation.reasons)


def test_planet_flow_preflight_accepts_compiler_control_metrics() -> None:
    spec = _trend_spec(
        require_planet_flow_control_metrics=True,
        min_win_rate_delta=0.08,
    )
    records = _records(
        *(
            [{"win_rate": 0.2, "launches": 0.1}] * 6
            + [{"win_rate": 0.9, "launches": 0.5}] * 10
        ),
        include_planet_flow_control=True,
    )

    evaluation = evaluate_gate_records(
        spec,
        records,
        campaign="preflight_beat_noop",
        run_dir=None,
        checkpoint=None,
    )

    assert evaluation.verdict == PreflightVerdict.VERIFIED


def test_planet_flow_preflight_short_circuits_without_calibration(tmp_path) -> None:
    import json

    thresholds_path = tmp_path / "preflight-calibration.json"
    thresholds_path.write_text(
        json.dumps({"learning_signal": {"window_updates": 10}}),
        encoding="utf-8",
    )

    evaluation = run_preflight_gate(
        "beat_noop",
        model="planet_flow_target_heatmap",
        output_root=tmp_path / "outputs",
        repo_root=tmp_path,
        thresholds_path=thresholds_path,
    )

    assert evaluation.verdict == PreflightVerdict.INCONCLUSIVE
    assert evaluation.evaluation_mode == "needs_calibration"
    assert evaluation.run_dir is None


def test_planet_flow_preflight_uses_calibrated_thresholds(tmp_path) -> None:
    import json

    thresholds_path = tmp_path / "preflight-calibration.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "thresholds": {
                    "learning_signal": {"window_updates": 10},
                    "win_proof_tournament": {},
                    "planet_flow_learning_signal": {
                        "window_updates": 4,
                        "min_win_rate_delta": 0.2,
                        "max_approx_kl": 0.05,
                        "min_entropy": 0.01,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    spec = _gate_specs("planet_flow_target_heatmap", thresholds_path=thresholds_path)[
        "beat_noop"
    ]

    assert spec.needs_calibration_reason is None
    assert spec.window_updates == 4
    assert spec.min_win_rate_delta == 0.2
