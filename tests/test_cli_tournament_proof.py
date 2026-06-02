from __future__ import annotations

import json

from src.cli import benchmark as benchmark_cli


def test_tournament_proof_dry_run(capsys, tmp_path) -> None:
    checkpoint = tmp_path / "jax_ckpt_last.pkl"
    checkpoint.write_bytes(b"stub")
    out = tmp_path / "tournament.json"
    assert (
        benchmark_cli.main(
            [
                "tournament-proof",
                "--eval-checkpoint",
                str(checkpoint),
                "--out",
                str(out),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["gate"] == "win_proof"
    assert payload["dry_run"] is True
    assert "uv run ow eval tournament" in capsys.readouterr().out
