from __future__ import annotations

import json
from pathlib import Path

from src.cli import eval as eval_cli
from src.cli.run_status import list_evaluation_results, load_evaluation_result


def test_list_evaluation_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    result_dir = run_dir / "evaluations" / "tournament_u000010_abcd"
    result_dir.mkdir(parents=True)
    manifest = {
        "kind": "tournament",
        "update": 10,
        "status": "completed",
        "promoted": True,
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    rows = list_evaluation_results(run_dir)
    assert len(rows) == 1
    assert rows[0]["kind"] == "tournament"
    assert rows[0]["update"] == 10


def test_eval_results_list_cli(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "runs" / "r1"
    result_dir = run_dir / "evaluations" / "tournament_u000010_abcd"
    result_dir.mkdir(parents=True)
    (result_dir / "manifest.json").write_text(
        json.dumps({"kind": "tournament", "update": 10}),
        encoding="utf-8",
    )

    assert eval_cli.main(["results", "list", "--run", str(run_dir)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["results"]) == 1


def test_eval_results_show_cli(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "runs" / "r1"
    result_dir = run_dir / "evaluations" / "tournament_u000010_abcd"
    result_dir.mkdir(parents=True)
    (result_dir / "manifest.json").write_text(
        json.dumps({"kind": "tournament", "update": 10}),
        encoding="utf-8",
    )

    assert (
        eval_cli.main(
            [
                "results",
                "show",
                "--run",
                str(run_dir),
                "--result",
                "tournament_u000010_abcd",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"]["update"] == 10


def test_load_evaluation_result_by_manifest_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    result_dir = run_dir / "evaluations" / "tournament_u000010_abcd"
    result_dir.mkdir(parents=True)
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"kind": "tournament"}), encoding="utf-8")

    payload = load_evaluation_result(run_dir, str(manifest_path))
    assert payload["manifest"]["kind"] == "tournament"
