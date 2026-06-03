from __future__ import annotations

import json

from src.cli import benchmark as benchmark_cli


def test_tournament_proof_dry_run_unified_plan(tmp_path) -> None:
    checkpoint = tmp_path / "jax_ckpt_last.pkl"
    checkpoint.write_bytes(b"stub")
    cal = tmp_path / "calibration.json"
    cal.write_text(
        json.dumps(
            {
                "unified_tournament": {
                    "enforcement": False,
                    "noop_min_combined": 0.7,
                    "random_min_combined": 0.58,
                    "games_per_pair": 4,
                    "prerequisite_seeds": [0, 1, 2, 3, 4],
                    "incumbent_seeds": list(range(30)),
                    "four_p_baseline_fillers": ["noop", "random", "random"],
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "tournament.json"
    assert (
        benchmark_cli.main(
            [
                "tournament-proof",
                "--eval-checkpoint",
                str(checkpoint),
                "--out",
                str(out),
                "--thresholds-path",
                str(cal),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["gate"] == "win_proof"
    assert payload["dry_run"] is True
    assert payload["unified"] is True
    assert payload["stage1"]["opponents"] == ["noop", "random"]
    assert payload["stage1"]["seeds"] == [0, 1, 2, 3, 4]
