# Orbit Wars local tools

## SSOT training pipeline flowchart

Interactive annotated flowchart for the canonical config → Kaggle submission spine. Layout follows the [flowchart diagram pattern](https://thariqs.github.io/html-effectiveness/13-flowchart-diagram.html); styling uses tokens from [`DESIGN.md`](DESIGN.md).

| File | Role |
| --- | --- |
| `ssot-training-pipeline-flowchart.html` | Click any step for commands, wall-clock estimates, and short-circuit rules |

```bash
xdg-open docs/tools/ssot-training-pipeline-flowchart.html   # or open in your browser
```

Source of truth: [`docs/brainstorms/2026-06-03-training-pipeline-ssot-requirements.md`](../brainstorms/2026-06-03-training-pipeline-ssot-requirements.md). Plan: [`docs/plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md`](../plans/2026-06-03-013-feat-ssot-training-pipeline-plan.md). Learning: [`docs/solutions/architecture-patterns/ssot-training-pipeline-config-to-kaggle-submission.md`](../solutions/architecture-patterns/ssot-training-pipeline-config-to-kaggle-submission.md). Several stages are planned ([#211](https://github.com/jmduea/orbit_wars/issues/211)) — the chart notes what is not yet implemented.

**Legacy Gate 5 / hybrid paths** (still in repo until teardown): [`gate5-unified-tournament-submit-valid-funnel.md`](../solutions/architecture-patterns/gate5-unified-tournament-submit-valid-funnel.md) — not the canonical spine; use the flowchart for production order.

## Frozen defaults picker

Interactive editor for editable leaves in the default Hydra composition (`uv run ow train print_resolved_config=true`). Omits volatile or opt-in-only keys (e.g. `output.run_id`, `task.env_parity_mode` — see `EXCLUDE_PATHS` in the build script). Layout follows the [feature-flags editor pattern](https://thariqs.github.io/html-effectiveness/19-editor-feature-flags.html): grouped panels, change diff sidebar, copy/export, reset.

| File | Role |
| --- | --- |
| `config-frozen-defaults-picker.template.html` | Hand-edited UI shell |
| `config-frozen-defaults-picker.html` | Generated picker (open in a browser) |
| `../scripts/build_config_frozen_defaults_picker.py` | Regenerates embedded config data |

```bash
uv run python scripts/build_config_frozen_defaults_picker.py
xdg-open docs/tools/config-frozen-defaults-picker.html   # or open the path in your browser
```

After you settle values, use **Copy Hydra overrides** for launch commands or **Copy YAML diff** for `conf/<group>/base.yaml` updates.
