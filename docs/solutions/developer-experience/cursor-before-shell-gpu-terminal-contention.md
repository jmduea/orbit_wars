---
module: cursor-hooks
date: 2026-06-04
problem_type: developer_experience
component: tooling
severity: medium
applies_when:
  - "Multiple Cursor agents share one GPU on this repo (WSL2 or single-GPU machine)"
  - "Adding or changing Cursor beforeShellExecution hooks that read terminal snapshots"
  - "Agents still launch ow train, pytest, or make test* despite AGENTS.md terminal guidance"
symptoms:
  - "Parallel pytest or ow train jobs contend on one GPU and wall clock balloons"
  - "make agent-context warns about gpu_contention but agents still start heavy shells"
  - "Hook denied heavy commands when terminal files had command: but no running_for_ms:"
resolution_type: tooling_addition
tags:
  - cursor-hooks
  - before-shell-execution
  - gpu-contention
  - multi-agent
  - terminals-folder
  - fail-open
  - one-gpu
  - running-for-ms
related_components:
  - .cursor/hooks/terminal_contention.py
  - .cursor/hooks/before-shell-terminal-contention.sh
  - .cursor/hooks.json
  - docs/CURSOR.md
  - tests/test_cursor_hooks.py
---

# Cursor beforeShellExecution hook for GPU terminal contention

## Context

Orbit Wars runs on **one GPU**. Multiple Cursor agents can run shells in parallel; prose in `AGENTS.md` and `make agent-context` (`gpu_contention`) is advisory only. Operators and agents were still starting `ow train`, `pytest`, and `make test*` while another session had an active terminal in the same repo.

This session shipped a **project hook** that denies **GPU-heavy** incoming shell commands when another Cursor terminal snapshot shows **GPU-heavy** work still running under this repo’s `cwd`. Light background processes (e.g. `python -m http.server` for the config picker) do not trigger contention. The same branch also refactored `src/jax/action_sampling.py` shield diagnostics (#197); that refactor is documented in `docs/solutions/developer-experience/cursor-before-shell-gpu-terminal-contention.md` — not duplicated here.

(session history) Prior work documented duplicate `wandb agent` contention in `AGENTS.md` and “check terminals folder” in merge-orchestration docs, but nothing enforced policy at shell time.

## Guidance

### Wiring

| Piece | Role |
|-------|------|
| `.cursor/hooks.json` | Registers `beforeShellExecution` → `.cursor/hooks/before-shell-terminal-contention.sh` (5s timeout) |
| `before-shell-terminal-contention.sh` | `exec python3` on `terminal_contention.py` with repo root |
| `terminal_contention.py` | Policy: scan terminals, classify heavy commands, return JSON permission |
| `docs/CURSOR.md` | Operator table for sessionStart + beforeShellExecution |
| `tests/test_cursor_hooks.py` | Unit tests via `evaluate(..., home=fake_home)` |

### Discover active terminals

Do **not** derive a single slug from the repo path (`orbit_wars` vs `orbit-wars` breaks discovery). Glob all Cursor terminal snapshots and filter by repo `cwd`:

```python
projects_root = (home or Path.home()) / ".cursor" / "projects"
for terminals_dir in projects_root.glob("*/terminals"):
    ...
```

A terminal counts as **active** only when **all** hold:

1. File body does **not** contain `exit_code:` (finished session).
2. Header meta (between first `---` pair) contains **`running_for_ms:`** — not `command:` alone.
3. `cwd:` in meta resolves under `repo_root` (from hook argv or `workspace_roots`).

```python
if "exit_code:" in text:
    continue
if "running_for_ms:" not in meta:
    continue
# cwd must be under repo_root.resolve()
```

### Deny only when both sides are GPU-heavy

Block only when **both** hold:

1. Incoming command matches `HEAVY` (GPU/contention patterns below).
2. At least one other active repo terminal’s `command:` also matches `HEAVY`.

`active_heavy_terminal_commands()` filters `active_terminal_commands()` with the same `HEAVY` regex. Light running processes — `python -m http.server`, `git`, `make agent-context`, `make help`, `curl`, etc. — do **not** count as contention even when `running_for_ms:` is present.

Otherwise **allow** (fail-open on JSON parse errors).

Heavy patterns include: `ow train`, `ow benchmark`, `ow sweep`, `make test` / `make test-*`, `pytest`, `wandb agent`, calibration commands, `test-launch-hygiene-e2e`, `test-jax`, `test-full`, etc.

### Testing

- Call `evaluate(command, repo_root, home=tmp_path)` with fake `~/.cursor/projects/.../terminals/*.txt`.
- When the **live** hook is enabled, run pytest with **isolated `HOME`** so the hook does not block its own test run:

```bash
HOME=/tmp/orbit-hook-test-isolated uv run pytest tests/test_cursor_hooks.py -q
```

## Why This Matters

Advisory text does not stop an agent mid-loop from starting a second training or pytest job. Enforcing at `beforeShellExecution` turns the one-GPU invariant into a hard gate for heavy work only, without blocking routine git or `make agent-context` during contention.

Requiring `running_for_ms:` avoids **stale** terminal files that still list `command:` after a process exited — a false positive caught in code review on this branch.

## When to Apply

- Before extending the `HEAVY` regex (new calibration or Makefile test targets).
- When debugging “hook blocked my command” — read `~/.cursor/projects/*/terminals/*.txt` for `running_for_ms:` and `cwd`.
- When adding new Cursor hooks: keep policy in importable Python (`terminal_contention.py`), shell wrapper thin, tests use `home=`.

## Examples

**Deny when another session is training:**

```python
payload = evaluate("make test-fast", repo_root, home=fake_home)
# fake_home has active terminal: cwd under repo, running_for_ms present
assert payload["permission"] == "deny"
```

**Allow agent-context while pytest runs elsewhere:**

```python
# Active terminal command is pytest; incoming command is make agent-context
assert evaluate("make agent-context", repo_root, home=fake_home)["permission"] == "allow"
```

**Allow heavy work while config picker HTTP server runs elsewhere:**

```python
# Active terminal: python3 -m http.server 8765; incoming: make test-fast
assert evaluate("make test-fast", repo_root, home=fake_home)["permission"] == "allow"
```

**Stale file must not block (regression):**

```text
---
cwd: "/home/.../orbit_wars"
command: "make test-fast"
---
```

No `running_for_ms:` → not active → heavy command allowed.

## What Didn't Work

| Attempt | Why it failed |
|---------|----------------|
| Slug path `~/.cursor/projects/<slug>/terminals` | Cursor slug uses `orbit-wars`; repo dir is `orbit_wars` |
| Deny all shell when any “active” terminal | Blocked hook self-tests and light commands |
| Treat `command:` without `running_for_ms:` as active | Stale snapshots caused false GPU-contention denies |
| Broad allowlist inside deny-all mode | Replaced by heavy-only deny + smaller allow surface |

## Related

- [`docs/CURSOR.md`](../../CURSOR.md) — hook table and manual session-start alternative
- [`docs/solutions/developer-experience/agent-native-operator-cli-phase1.md`](agent-native-operator-cli-phase1.md) — operator CLI / `make agent-context` baseline; hook enforces GPU policy at shell time
- [`docs/solutions/developer-experience/seed-scheduler-calibration-agent-native-operator-phase2.md`](seed-scheduler-calibration-agent-native-operator-phase2.md) — `make agent-context` GPU hint (advisory, `pgrep`-based)
- [`docs/solutions/workflow-issues/multi-branch-agent-merge-orchestration.md`](../workflow-issues/multi-branch-agent-merge-orchestration.md) — check terminals before parallel pytest
- [`docs/solutions/developer-experience/cursor-before-shell-gpu-terminal-contention.md`](../../solutions/developer-experience/cursor-before-shell-gpu-terminal-contention.md) — #197 shield diagnostics dedupe on same branch
- GitHub #204, #189 — perf work; hook is orthogonal but reduces accidental parallel GPU load
