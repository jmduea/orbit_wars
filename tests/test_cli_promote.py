from __future__ import annotations

import json
from pathlib import Path

from src.artifacts.promotion_manifest import promoted_manifest_path
from src.cli import promote as promote_cli


def _campaign_tree(
    tmp_path: Path,
    *,
    campaign: str = "cap",
) -> tuple[Path, Path]:
    output_root = tmp_path / "outputs"
    camp_dir = output_root / "campaigns" / campaign
    camp_dir.mkdir(parents=True)
    indexes = output_root / "indexes"
    indexes.mkdir(parents=True)
    return output_root, camp_dir


def _write_promoted(
    camp_dir: Path,
    *,
    checkpoint: Path,
    metric_value: float,
    run_id: str = "run-a",
) -> None:
    manifest_path = promoted_manifest_path(camp_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "campaign": camp_dir.name,
                "checkpoint_path": str(checkpoint.resolve()),
                "metric_name": "episode_reward_mean",
                "metric_value": metric_value,
                "source_run_id": run_id,
            }
        ),
        encoding="utf-8",
    )
    (camp_dir / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "campaign": camp_dir.name,
                "current_best_value": metric_value,
                "current_best_run_id": run_id,
            }
        ),
        encoding="utf-8",
    )


def _append_index(output_root: Path, record: dict[str, object]) -> None:
    path = output_root / "indexes" / "promoted.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_promote_show_and_history(tmp_path: Path, capsys) -> None:
    output_root, camp_dir = _campaign_tree(tmp_path)
    ckpt = camp_dir / "runs" / "r1" / "checkpoints" / "jax_ckpt_000001.pkl"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"ckpt")
    _write_promoted(camp_dir, checkpoint=ckpt, metric_value=0.9)
    _append_index(
        output_root,
        {
            "campaign": "cap",
            "checkpoint_path": str(ckpt),
            "metric_name": "episode_reward_mean",
            "metric_value": 0.9,
            "updated_at": "2026-06-01T00:00:00Z",
        },
    )

    assert (
        promote_cli.main(
            ["show", "--campaign", "cap", "--output-root", str(output_root)]
        )
        == 0
    )
    show_out = capsys.readouterr().out
    assert '"metric_value": 0.9' in show_out

    assert (
        promote_cli.main(
            [
                "history",
                "--campaign",
                "cap",
                "--output-root",
                str(output_root),
                "--limit",
                "5",
            ]
        )
        == 0
    )
    history_out = capsys.readouterr().out
    assert '"count": 1' in history_out


def test_promote_demote_clears_manifest(tmp_path: Path, capsys) -> None:
    output_root, camp_dir = _campaign_tree(tmp_path)
    ckpt = camp_dir / "runs" / "r1" / "checkpoints" / "jax_ckpt_000001.pkl"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"ckpt")
    _write_promoted(camp_dir, checkpoint=ckpt, metric_value=0.5)
    _append_index(
        output_root,
        {
            "campaign": "cap",
            "checkpoint_path": str(ckpt),
            "metric_value": 0.5,
        },
    )

    assert (
        promote_cli.main(
            [
                "demote",
                "--campaign",
                "cap",
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert '"action": "cleared"' in out
    assert not promoted_manifest_path(camp_dir).exists()
    campaign_manifest = json.loads(
        (camp_dir / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    assert campaign_manifest["current_best_value"] is None
    index_lines = (output_root / "indexes" / "promoted.jsonl").read_text().splitlines()
    assert json.loads(index_lines[-1])["event"] == "demoted"


def test_promote_demote_dry_run_keeps_manifest(tmp_path: Path) -> None:
    output_root, camp_dir = _campaign_tree(tmp_path)
    ckpt = camp_dir / "runs" / "r1" / "checkpoints" / "jax_ckpt_000001.pkl"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"ckpt")
    _write_promoted(camp_dir, checkpoint=ckpt, metric_value=0.5)

    assert (
        promote_cli.main(
            [
                "demote",
                "--campaign",
                "cap",
                "--output-root",
                str(output_root),
                "--dry-run",
            ]
        )
        == 0
    )
    assert promoted_manifest_path(camp_dir).is_file()


def test_promote_demote_to_previous_restores_prior(tmp_path: Path, capsys) -> None:
    output_root, camp_dir = _campaign_tree(tmp_path)
    ckpt_old = camp_dir / "runs" / "r0" / "checkpoints" / "jax_ckpt_000001.pkl"
    ckpt_new = camp_dir / "runs" / "r1" / "checkpoints" / "jax_ckpt_000002.pkl"
    ckpt_old.parent.mkdir(parents=True)
    ckpt_new.parent.mkdir(parents=True)
    ckpt_old.write_bytes(b"old")
    ckpt_new.write_bytes(b"new")
    _write_promoted(camp_dir, checkpoint=ckpt_new, metric_value=0.9, run_id="run-b")
    _append_index(
        output_root,
        {
            "campaign": "cap",
            "checkpoint_path": str(ckpt_old),
            "metric_value": 0.4,
            "run_id": "run-a",
        },
    )
    _append_index(
        output_root,
        {
            "campaign": "cap",
            "checkpoint_path": str(ckpt_new),
            "metric_value": 0.9,
            "run_id": "run-b",
        },
    )

    assert (
        promote_cli.main(
            [
                "demote",
                "--campaign",
                "cap",
                "--output-root",
                str(output_root),
                "--to-previous",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert '"action": "restored_previous"' in out
    manifest = json.loads(promoted_manifest_path(camp_dir).read_text(encoding="utf-8"))
    assert manifest["checkpoint_path"] == str(ckpt_old.resolve())
    assert manifest["metric_value"] == 0.4
