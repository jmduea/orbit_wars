---
title: Recover stashed WIP after post-merge main reset looks like rollback
date: 2026-06-01
category: workflow-issues
module: development-workflow
problem_type: workflow_issue
component: development_workflow
severity: medium
applies_when:
  - "Parallel feature work with git stash before switching branches"
  - "Post-merge cleanup uses git reset --hard origin/main on main"
  - "Agent or human reports in-progress work disappeared after branch delete"
tags:
  - git-stash
  - branch-cleanup
  - parallel-work
  - agent-workflow
  - recovery
---

# Recover stashed WIP after post-merge main reset looks like rollback

## Context

While shipping PR #164 (agent-native operator CLI) from a dedicated branch, in-progress seed-scheduler calibration edits on `feat/seed-scheduler-calibration` were stashed so `feat/agent-native-operator-cli` could branch from clean `main`. After the PR merged, cleanup reset local `main` to `origin/main` and deleted the feature branch. On `main`, only merged CLI work was visible — the calibration diff looked gone.

(session history)

## Guidance

**The work is usually still in git** — in the stash and/or on the original feature branch — not erased by deleting the shipped branch or hard-resetting `main`.

### Diagnose before panicking

```bash
git stash list
git stash show -p stash@{0} --stat
git branch -a | grep seed-scheduler   # or your feature name
git log --oneline -5 feat/seed-scheduler-calibration
git reflog main -10                    # local-only commits dropped from main tip
```

| What you see | Meaning |
|--------------|---------|
| Stash entry with your message | Uncommitted WIP is preserved |
| Feature branch still exists | Committed work on that branch may still be there |
| `reflog` shows reset from local commit | Doc-only or WIP commit may be recoverable via `git cherry-pick <sha>` |

### Restore pattern that worked

```bash
git checkout feat/seed-scheduler-calibration   # branch that owned the WIP
git stash pop                                   # re-apply stashed files
git status --short
```

Expect modified files to return as unstaged changes; stash should be empty after a successful pop.

### Safe post-merge cleanup checklist

1. **`git stash list`** — note any `wip-*` stashes before deleting branches.
2. **Do not assume `main` shows all active work** — parallel branches and stashes hold other streams.
3. **`git reset --hard origin/main`** drops **local-only commits on `main`**; recover via `git reflog` + cherry-pick if needed.
4. **Deleting a merged feature branch** does not remove stashes or other local branches.
5. After cleanup, **switch to the WIP branch and `stash pop`** before concluding work was lost.

## Why This Matters

Hard-resetting `main` to match GitHub is correct for a clean default branch, but it changes the working tree snapshot. Agents and humans on `main` will not see stashed or branch-only work. Without an explicit stash pop, it reads as a rollback even when PR merges landed correctly.

## When to Apply

- LFG or parallel PR workflow: stash → new branch from `main` → merge → cleanup `main`.
- User says "we removed in-progress work" immediately after merge hygiene.
- Multiple agents; one branch ships while another's edits were stashed.

## Examples

**Before (looks like disaster on `main`):**

```bash
git checkout main && git reset --hard origin/main
git branch -d feat/agent-native-operator-cli
# User on main: seed-scheduler file changes missing
```

**After (restore WIP stream):**

```bash
git checkout feat/seed-scheduler-calibration
git stash pop
# src/jax/preflight*.py, seed_scheduler_calibration.py, tests/ visible again
```

**Recover doc commit that lived only on `main` briefly:**

```bash
git cherry-pick ce6714b   # if reflog shows the lost SHA
```

## Related Issues

- Plan: `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md` (KTD5 branch isolation + stash)
- Merged PR: [#164](https://github.com/jmduea/orbit_wars/pull/164) (agent-native CLI; triggered cleanup timing)
- Operator CLI after merge: `docs/solutions/developer-experience/agent-native-operator-cli-phase1.md`
