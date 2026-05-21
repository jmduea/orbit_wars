import jax
import json
import pytest
from omegaconf import OmegaConf

from src.config import TrainConfig, compose_hydra_train_config, train_config_from_omegaconf
from src.curriculum import CurriculumController
from src.jax_env import batched_reset
from src.jax_policy import build_jax_policy
from src.jax_ppo import collect_rollout_jax, init_train_state


def _curriculum_config(stages):
    cfg = TrainConfig()
    cfg.curriculum.enabled = True
    cfg.opponents.snapshot.pool_size = 2
    cfg.opponents.snapshot.interval_updates = 1
    cfg.curriculum.stages = stages
    cfg.opponents.mode.opponent = "self"
    cfg.opponents.self_play.enabled = True
    return cfg


def test_default_hydra_config_uses_new_curriculum_surface():
    cfg = compose_hydra_train_config(["training.total_updates=1"])

    assert cfg.curriculum.enabled
    assert cfg.curriculum.stages
    assert cfg.opponents.snapshot.pool_size == 5


def test_curriculum_rejects_unknown_family():
    cfg = OmegaConf.structured(TrainConfig)
    cfg.curriculum.enabled = True
    cfg.curriculum.stages = [
        {"id": "stage", "opponent_families": {"mystery": 1.0}},
    ]

    with pytest.raises(ValueError, match="unknown families"):
        train_config_from_omegaconf(cfg)


def test_curriculum_controller_promotes_on_rolling_mean():
    cfg = _curriculum_config(
        [
            {
                "id": "bootstrap",
                "min_updates": 1,
                "promote_if": {
                    "metric": "overall_win_rate",
                    "op": ">=",
                    "value": 0.5,
                    "window_updates": 2,
                },
                "opponent_families": {"random": 1.0},
            },
            {"id": "pressure", "opponent_families": {"latest": 1.0}},
        ]
    )
    controller = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)

    assert controller.update(1, {"overall_win_rate": 0.4}) is None
    event = controller.update(2, {"overall_win_rate": 0.6})

    assert event is not None
    assert event["event"] == "curriculum_stage_promoted"
    assert controller.current_stage_id() == "pressure"


def test_recent_biased_snapshot_selection_prefers_newer_updates():
    cfg = _curriculum_config(
        [{"id": "historical", "opponent_families": {"historical": 1.0}}]
    )
    cfg.opponents.snapshot.selection = "recent_biased"
    controller = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)

    view = controller.stage_view(
        10,
        snapshot_ids=jax.numpy.array([1, 2], dtype=jax.numpy.int32),
        snapshot_valid_mask=jax.numpy.array([True, True]),
        snapshot_updates=jax.numpy.array([2, 8], dtype=jax.numpy.int32),
    )

    assert float(view.historical_selection_probs[1]) > float(
        view.historical_selection_probs[0]
    )


def test_checkpoint_payload_roundtrips_curriculum_and_historical_pool():
    from src.jax_train import (
        _add_historical_snapshot,
        _checkpoint_payload_builder,
        _init_historical_snapshot_pool,
        _restore_historical_snapshot_pool,
    )

    cfg = _curriculum_config(
        [{"id": "latest", "opponent_families": {"latest": 1.0}}]
    )
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(120), policy, cfg)
    controller = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    controller.stage_index = 0
    pool = _init_historical_snapshot_pool(train_state.params, 2)
    pool, _event = _add_historical_snapshot(pool, train_state.params, update=3)

    payload = _checkpoint_payload_builder(
        train_state,
        cfg,
        key=jax.random.PRNGKey(121),
        update=3,
        total_env_steps=4,
        completed_episodes=1,
        curriculum=controller,
        historical_pool=pool,
    )()
    restored = _restore_historical_snapshot_pool(
        payload["historical_snapshot_pool"],
        _init_historical_snapshot_pool(train_state.params, 2),
    )

    assert "curriculum_state" in payload
    assert int(jax.numpy.sum(restored.valid_mask)) == 1
    assert int(restored.snapshot_ids[0]) == 1


def test_checkpoint_payload_builder_freezes_curriculum_state_for_async_jobs():
    from src.jax_train import _checkpoint_payload_builder

    cfg = _curriculum_config(
        [
            {"id": "first", "opponent_families": {"random": 1.0}},
            {"id": "second", "opponent_families": {"latest": 1.0}},
        ]
    )
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(130), policy, cfg)
    controller = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    builder = _checkpoint_payload_builder(
        train_state,
        cfg,
        key=jax.random.PRNGKey(131),
        update=1,
        total_env_steps=2,
        completed_episodes=0,
        curriculum=controller,
    )

    controller.stage_index = 1
    payload = builder()

    assert payload["curriculum_state"]["stage_index"] == 0


def test_two_player_rollout_reports_sampled_random_family_slots():
    cfg = _curriculum_config(
        [{"id": "random", "opponent_families": {"random": 1.0}}]
    )
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    reset_keys = jax.random.split(jax.random.PRNGKey(100), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(101), policy, cfg)
    controller = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    stage_view = controller.stage_view(
        1,
        snapshot_ids=jax.numpy.zeros((2,), dtype=jax.numpy.int32),
        snapshot_valid_mask=jax.numpy.zeros((2,), dtype=bool),
        snapshot_updates=jax.numpy.zeros((2,), dtype=jax.numpy.int32),
    )

    _key, _env_state, _turn_batch, _transitions, metrics = collect_rollout_jax(
        jax.random.PRNGKey(102),
        env_state,
        turn_batch,
        train_state,
        policy,
        cfg,
        stage_view=stage_view,
    )

    assert float(metrics["opponent_slots_total"]) == 2.0
    assert float(metrics["opponent_slots_random"]) == 2.0
    assert float(metrics["opponent_slots_latest"]) == 0.0


def test_historical_family_falls_back_to_latest_when_pool_empty():
    cfg = _curriculum_config(
        [{"id": "historical", "opponent_families": {"historical": 1.0}}]
    )
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.training.num_envs = 2
    cfg.training.rollout_steps = 1
    reset_keys = jax.random.split(jax.random.PRNGKey(110), cfg.training.num_envs)
    env_state, turn_batch = batched_reset(reset_keys, cfg.task)
    policy = build_jax_policy(cfg=cfg)
    train_state = init_train_state(jax.random.PRNGKey(111), policy, cfg)
    controller = CurriculumController(cfg.curriculum, cfg.opponents.snapshot)
    stage_view = controller.stage_view(
        1,
        snapshot_ids=jax.numpy.zeros((2,), dtype=jax.numpy.int32),
        snapshot_valid_mask=jax.numpy.zeros((2,), dtype=bool),
        snapshot_updates=jax.numpy.zeros((2,), dtype=jax.numpy.int32),
    )

    _key, _env_state, _turn_batch, _transitions, metrics = collect_rollout_jax(
        jax.random.PRNGKey(112),
        env_state,
        turn_batch,
        train_state,
        policy,
        cfg,
        stage_view=stage_view,
    )

    assert float(metrics["opponent_slots_total"]) == 2.0
    assert float(metrics["opponent_slots_latest"]) == 2.0
    assert float(metrics["opponent_historical_fallback_latest_slots"]) == 2.0


def test_training_loop_logs_curriculum_events_on_same_update(tmp_path, monkeypatch):
    from src.jax_train import run_jax_training

    monkeypatch.setenv("ORBIT_WARS_ALLOW_CPU_JAX_ON_NVIDIA", "1")

    cfg = _curriculum_config(
        [
            {
                "id": "bootstrap",
                "min_updates": 1,
                "promote_if": {
                    "metric": "overall_win_rate",
                    "op": ">=",
                    "value": 0.0,
                    "window_updates": 1,
                },
                "opponent_families": {"random": 1.0},
            },
            {"id": "pressure", "opponent_families": {"latest": 1.0}},
        ]
    )
    cfg.run_name = "curriculum_events"
    cfg.artifacts.save_dir = str(tmp_path)
    cfg.artifacts.artifact_pipeline.enabled = False
    cfg.artifacts.replay.enabled = False
    cfg.telemetry.wandb.enabled = False
    cfg.task.max_fleets = 16
    cfg.task.candidate_count = 4
    cfg.model.hidden_size = 16
    cfg.model.attention_heads = 2
    cfg.training.num_envs = 1
    cfg.training.rollout_steps = 1
    cfg.training.total_updates = 1
    cfg.training.epochs = 1
    cfg.training.minibatch_size = 32
    cfg.artifacts.checkpoint_every = 100

    run_jax_training(cfg)

    log_path = tmp_path / "logs" / "curriculum_events_jax.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    update_record = next(record for record in records if record.get("update") == 1)
    events = update_record["curriculum_phase_events"]

    assert any(event.get("event") == "curriculum_stage_promoted" for event in events)
    assert any(event.get("event") == "historical_snapshot_added" for event in events)
