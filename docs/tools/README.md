# Orbit Wars local tools

## Frozen defaults picker

Interactive editor for every leaf in the default Hydra composition (`uv run ow train print_resolved_config=true`). Layout follows the [feature-flags editor pattern](https://thariqs.github.io/html-effectiveness/19-editor-feature-flags.html): grouped panels, change diff sidebar, copy/export, reset.

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
