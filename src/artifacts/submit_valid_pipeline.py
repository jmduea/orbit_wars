"""Submit-valid training pipeline selection (hybrid, bracket, SSOT)."""

from __future__ import annotations

from src.config.schema import TrainConfig

SubmitValidPipelineId = str

HYBRID_CHECKPOINT_EVAL = "hybrid_checkpoint_eval"
BRACKET_TRAINING = "bracket_training"
SSOT_PIPELINE = "ssot_pipeline"


def active_submit_valid_pipelines(cfg: TrainConfig) -> tuple[SubmitValidPipelineId, ...]:
    """Return enabled submit-valid pipeline ids (at most one should be active)."""

    artifacts = cfg.artifacts
    active: list[SubmitValidPipelineId] = []
    if artifacts.bracket_training.enabled:
        active.append(BRACKET_TRAINING)
    if artifacts.ssot_pipeline.enabled:
        active.append(SSOT_PIPELINE)
    strategy = str(artifacts.promotion.strategy or "metric").strip().lower()
    if strategy == "hybrid" and artifacts.artifact_pipeline.checkpoint_eval_async:
        active.append(HYBRID_CHECKPOINT_EVAL)
    return tuple(active)


def primary_submit_valid_pipeline(cfg: TrainConfig) -> SubmitValidPipelineId | None:
    """Return the sole active submit-valid pipeline, or ``None`` when disabled."""

    active = active_submit_valid_pipelines(cfg)
    if not active:
        return None
    validate_submit_valid_pipelines_exclusive(cfg)
    return active[0]


def validate_submit_valid_pipelines_exclusive(cfg: TrainConfig) -> None:
    """Reject configs that enable more than one submit-valid pipeline."""

    active = active_submit_valid_pipelines(cfg)
    if len(active) > 1:
        raise ValueError(
            "mutually exclusive submit-valid pipelines enabled: "
            f"{', '.join(active)}; use only one of artifacts=hybrid_promotion, "
            "artifacts=bracket_training, or artifacts=ssot_pipeline."
        )
