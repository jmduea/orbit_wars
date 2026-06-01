"""Pre-flight learning gates (Gates 1–5) before long training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from src.jax.preflight_calibration import (
    PREFLIGHT_TRAIN_BASE,
    WINDOW_UPDATES,
    default_calibration_json_path,
    load_thresholds,
    run_ow_train,
)

Verdict = Literal["VERIFIED", "NOT_VERIFIED", "INCONCLUSIVE"]

GATE_ORDER: tuple[str, ...] = ("beat_noop", "beat_random", "curriculum_staged")


class PreflightVerdict(StrEnum):
    VERIFIED = "VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class PreflightGateSpec:
    """One learnability gate run via ``ow train`` and JSONL telemetry."""

    gate_id: str
    train_overrides: tuple[str, ...]
    min_win_rate_delta: float | None = None
    window_updates: int = WINDOW_UPDATES
    require_curriculum_promotion: bool = False
    max_approx_kl: float = 0.15
    min_entropy: float = 1.0e-4


@dataclass(frozen=True, slots=True)
class GateEvaluation:
    gate_id: str
    verdict: PreflightVerdict
    reasons: tuple[str, ...]
    campaign: str
    run_dir: str | None
    log_path: str | None
    checkpoint_path: str | None
    window_overall_win_rate: float | None
    window_launches: float | None
    win_rate_first_window: float | None
    win_rate_delta: float | None
    best_rolling_win_rate: float | None
    curriculum_promotions: int
    evaluation_mode: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _learning_signal_thresholds(
    thresholds_path: Path | None = None,
) -> dict[str, object]:
    path = thresholds_path or default_calibration_json_path(_repo_root())
    payload = load_thresholds(path)
    learning = payload.get("learning_signal", {})
    if not isinstance(learning, dict):
        learning = {}
    return learning


def _gate_specs(
    model: str,
    *,
    thresholds_path: Path | None = None,
) -> dict[str, PreflightGateSpec]:
    learning = _learning_signal_thresholds(thresholds_path)
    window = int(learning.get("window_updates", WINDOW_UPDATES))
    min_delta = float(learning.get("min_win_rate_delta", 0.08))
    max_kl = float(learning.get("max_approx_kl", 0.15))
    min_ent = float(learning.get("min_entropy", 1.0e-4))
    return {
        "beat_noop": PreflightGateSpec(
            gate_id="beat_noop",
            train_overrides=(
                f"model={model}",
                "training=2p_16",
                "training.rollout_steps=128",
                "training.total_updates=200",
                "opponents=noop_only",
                "curriculum=off",
                *PREFLIGHT_TRAIN_BASE,
            ),
            min_win_rate_delta=min_delta,
            window_updates=window,
            max_approx_kl=max_kl,
            min_entropy=min_ent,
        ),
        "beat_random": PreflightGateSpec(
            gate_id="beat_random",
            train_overrides=(
                f"model={model}",
                "training=2p_16",
                "training.rollout_steps=128",
                "training.total_updates=300",
                "opponents=random_only",
                "curriculum=off",
                *PREFLIGHT_TRAIN_BASE,
            ),
            min_win_rate_delta=min_delta,
            window_updates=window,
            max_approx_kl=max_kl,
            min_entropy=min_ent,
        ),
        "curriculum_staged": PreflightGateSpec(
            gate_id="curriculum_staged",
            train_overrides=(
                "model=transformer_factorized",
                "training=2p4p_16_split",
                "training.rollout_steps=128",
                "training.total_updates=500",
                "curriculum=self_play_staged",
                *PREFLIGHT_TRAIN_BASE,
            ),
            require_curriculum_promotion=True,
            window_updates=window,
            max_approx_kl=max_kl,
            min_entropy=min_ent,
        ),
    }


def preflight_campaign(gate_id: str) -> str:
    return f"preflight_{gate_id}"


def read_jsonl_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def _launches_key(records: list[dict[str, object]]) -> str | None:
    for key in ("mean_active_launches_per_turn", "stop_utilization_ratio"):
        if any(key in record for record in records):
            return key
    return None


def _window_mean(
    records: list[dict[str, object]], key: str, *, last_n: int
) -> float | None:
    tail = records[-last_n:] if last_n > 0 else records
    values = [
        float(record[key])
        for record in tail
        if key in record and record[key] is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _window_mean_first(
    records: list[dict[str, object]], key: str, *, first_n: int
) -> float | None:
    head = records[:first_n] if first_n > 0 else records
    values = [
        float(record[key])
        for record in head
        if key in record and record[key] is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _best_rolling_mean(
    records: list[dict[str, object]], key: str, *, window_n: int
) -> float | None:
    if not records or window_n <= 0:
        return None
    rolling = [
        _window_mean(records[: index + 1], key, last_n=window_n)
        for index in range(window_n - 1, len(records))
    ]
    values = [value for value in rolling if value is not None]
    return max(values) if values else None


def _count_curriculum_promotions(records: list[dict[str, object]]) -> int:
    count = 0
    for record in records:
        events = record.get("curriculum_phase_events")
        if not isinstance(events, list):
            continue
        for event in events:
            if (
                isinstance(event, dict)
                and event.get("event") == "curriculum_stage_promoted"
            ):
                count += 1
    return count


def latest_run_dir(*, campaign: str, output_root: Path) -> Path:
    runs_root = output_root / "campaigns" / campaign / "runs"
    if not runs_root.is_dir():
        raise FileNotFoundError(
            f"No runs directory for campaign {campaign!r}: {runs_root}"
        )
    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs under campaign {campaign!r}: {runs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    paths = sorted(ckpt_dir.glob("jax_ckpt_*.pkl"))
    return paths[-1] if paths else None


def evaluate_gate_records(
    spec: PreflightGateSpec,
    records: list[dict[str, object]],
    *,
    campaign: str,
    run_dir: Path | None,
    checkpoint: Path | None,
) -> GateEvaluation:
    reasons: list[str] = []
    window_n = spec.window_updates
    metric_rows = [
        record
        for record in records
        if "overall_win_rate" in record and record.get("update") is not None
    ]
    effective_window = min(window_n, len(metric_rows)) if metric_rows else window_n
    win_rate = _window_mean(metric_rows, "overall_win_rate", last_n=effective_window)
    win_rate_first = _window_mean_first(
        metric_rows, "overall_win_rate", first_n=effective_window
    )
    win_rate_delta = (
        (win_rate - win_rate_first)
        if win_rate is not None and win_rate_first is not None
        else None
    )
    best_rolling = _best_rolling_mean(
        metric_rows, "overall_win_rate", window_n=effective_window
    )
    approx_kl = _window_mean(metric_rows, "approx_kl", last_n=effective_window)
    entropy = _window_mean(metric_rows, "entropy", last_n=effective_window)
    launches_key = _launches_key(metric_rows)
    launches_tail = (
        _window_mean(metric_rows, launches_key, last_n=effective_window)
        if launches_key is not None
        else None
    )

    if not metric_rows:
        return GateEvaluation(
            gate_id=spec.gate_id,
            verdict=PreflightVerdict.INCONCLUSIVE,
            reasons=("missing training jsonl records",),
            campaign=campaign,
            run_dir=str(run_dir) if run_dir is not None else None,
            log_path=None,
            checkpoint_path=str(checkpoint) if checkpoint is not None else None,
            window_overall_win_rate=None,
            window_launches=launches_tail,
            win_rate_first_window=None,
            win_rate_delta=None,
            best_rolling_win_rate=None,
            curriculum_promotions=0,
            evaluation_mode="learning_signal",
        )

    if spec.min_win_rate_delta is not None:
        if win_rate_delta is None:
            reasons.append("missing win-rate trend metrics")
        elif win_rate_delta < spec.min_win_rate_delta:
            reasons.append(
                f"win_rate_delta {win_rate_delta:.3f} < {spec.min_win_rate_delta:.3f} "
                f"(last {effective_window} vs first {effective_window} updates)"
            )

    if approx_kl is None:
        reasons.append("missing approx_kl")
    elif approx_kl > spec.max_approx_kl:
        reasons.append(f"approx_kl {approx_kl:.4f} > {spec.max_approx_kl:.4f}")

    if entropy is None:
        reasons.append("missing entropy")
    elif entropy < spec.min_entropy:
        reasons.append(f"entropy {entropy:.6f} < {spec.min_entropy:.6f}")

    promotions = _count_curriculum_promotions(records)
    if spec.require_curriculum_promotion and promotions == 0:
        reasons.append("no curriculum_stage_promoted events in training log")

    if any("missing training jsonl" in reason for reason in reasons):
        verdict = PreflightVerdict.INCONCLUSIVE
    elif reasons:
        verdict = PreflightVerdict.NOT_VERIFIED
    else:
        verdict = PreflightVerdict.VERIFIED

    log_path = None
    if run_dir is not None:
        logs = sorted((run_dir / "logs").glob("*_jax.jsonl"))
        if logs:
            log_path = str(logs[0])

    return GateEvaluation(
        gate_id=spec.gate_id,
        verdict=verdict,
        reasons=tuple(reasons),
        campaign=campaign,
        run_dir=str(run_dir) if run_dir is not None else None,
        log_path=log_path,
        checkpoint_path=str(checkpoint) if checkpoint is not None else None,
        window_overall_win_rate=win_rate,
        window_launches=launches_tail,
        win_rate_first_window=win_rate_first,
        win_rate_delta=win_rate_delta,
        best_rolling_win_rate=best_rolling,
        curriculum_promotions=promotions,
        evaluation_mode="learning_signal",
    )


def gate_evaluation_to_dict(evaluation: GateEvaluation) -> dict[str, object]:
    return {
        "gate_id": evaluation.gate_id,
        "verdict": evaluation.verdict.value,
        "reasons": list(evaluation.reasons),
        "campaign": evaluation.campaign,
        "run_dir": evaluation.run_dir,
        "log_path": evaluation.log_path,
        "checkpoint_path": evaluation.checkpoint_path,
        "window_overall_win_rate": evaluation.window_overall_win_rate,
        "window_launches": evaluation.window_launches,
        "win_rate_first_window": evaluation.win_rate_first_window,
        "win_rate_delta": evaluation.win_rate_delta,
        "best_rolling_win_rate": evaluation.best_rolling_win_rate,
        "curriculum_promotions": evaluation.curriculum_promotions,
        "evaluation_mode": evaluation.evaluation_mode,
    }


def run_preflight_gate(
    gate_id: str,
    *,
    model: str = "transformer_factorized_small",
    output_root: Path = Path("outputs"),
    repo_root: Path | None = None,
    dry_run: bool = False,
    thresholds_path: Path | None = None,
) -> GateEvaluation:
    specs = _gate_specs(model, thresholds_path=thresholds_path)
    if gate_id not in specs:
        raise ValueError(f"Unknown preflight gate: {gate_id!r}")
    spec = specs[gate_id]
    root = repo_root or _repo_root()
    campaign = preflight_campaign(gate_id)
    overrides = [
        f"output.campaign={campaign}",
        f"output.root={output_root.as_posix()}",
        *spec.train_overrides,
    ]
    run_ow_train(
        overrides,
        repo_root=root,
        dry_run=dry_run,
        label=f"preflight gate {gate_id} campaign={campaign}",
    )
    if dry_run:
        return GateEvaluation(
            gate_id=gate_id,
            verdict=PreflightVerdict.INCONCLUSIVE,
            reasons=("dry_run",),
            campaign=campaign,
            run_dir=None,
            log_path=None,
            checkpoint_path=None,
            window_overall_win_rate=None,
            window_launches=None,
            win_rate_first_window=None,
            win_rate_delta=None,
            best_rolling_win_rate=None,
            curriculum_promotions=0,
            evaluation_mode="learning_signal",
        )

    run_dir = latest_run_dir(campaign=campaign, output_root=output_root)
    log_files = sorted((run_dir / "logs").glob("*_jax.jsonl"))
    if not log_files:
        return evaluate_gate_records(
            spec,
            [],
            campaign=campaign,
            run_dir=run_dir,
            checkpoint=find_latest_checkpoint(run_dir),
        )
    records = read_jsonl_records(log_files[0])
    return evaluate_gate_records(
        spec,
        records,
        campaign=campaign,
        run_dir=run_dir,
        checkpoint=find_latest_checkpoint(run_dir),
    )


def run_preflight_ladder(
    *,
    through: str,
    model: str = "transformer_factorized_small",
    output_root: Path = Path("outputs"),
    repo_root: Path | None = None,
    dry_run: bool = False,
    thresholds_path: Path | None = None,
) -> tuple[PreflightVerdict, list[GateEvaluation]]:
    if through not in GATE_ORDER:
        raise ValueError(f"--through must be one of {GATE_ORDER}, got {through!r}")
    stop_index = GATE_ORDER.index(through)
    selected = GATE_ORDER[: stop_index + 1]
    evaluations: list[GateEvaluation] = []
    overall = PreflightVerdict.VERIFIED
    for gate_id in selected:
        evaluation = run_preflight_gate(
            gate_id,
            model=model if gate_id != "curriculum_staged" else "transformer_factorized",
            output_root=output_root,
            repo_root=repo_root,
            dry_run=dry_run,
            thresholds_path=thresholds_path,
        )
        evaluations.append(evaluation)
        if evaluation.verdict != PreflightVerdict.VERIFIED:
            overall = evaluation.verdict
            break
    return overall, evaluations


def compare_repro_snapshots(
    left: dict[str, object],
    right: dict[str, object],
    *,
    update: int,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> tuple[PreflightVerdict, tuple[str, ...]]:
    left_snaps = {
        int(item["update"]): item
        for item in left.get("snapshots", [])
        if isinstance(item, dict)
    }
    right_snaps = {
        int(item["update"]): item
        for item in right.get("snapshots", [])
        if isinstance(item, dict)
    }
    if update not in left_snaps or update not in right_snaps:
        return PreflightVerdict.INCONCLUSIVE, (f"missing snapshot at update {update}",)
    reasons: list[str] = []
    keys = ("overall_win_rate", "approx_kl", "entropy", "policy_loss", "value_loss")
    for key in keys:
        lv = left_snaps[update].get(key)
        rv = right_snaps[update].get(key)
        if lv is None or rv is None:
            continue
        lvf, rvf = float(lv), float(rv)
        if abs(lvf - rvf) > atol + rtol * max(abs(lvf), abs(rvf), 1.0):
            reasons.append(f"{key} mismatch: {lvf} vs {rvf}")
    if reasons:
        return PreflightVerdict.NOT_VERIFIED, tuple(reasons)
    return PreflightVerdict.VERIFIED, ()


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
