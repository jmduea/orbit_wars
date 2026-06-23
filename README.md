# Orbit Wars

**A reinforcement-learning agent and end-to-end training system for a real-time space-strategy game, built solo in about a month.**

> Author: Jon Duea · Repo: [github.com/jmduea/orbit_wars](https://github.com/jmduea/orbit_wars) · Competition: [Kaggle — Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)

Orbit Wars is a [Kaggle](https://www.kaggle.com/) simulation competition: a 2- or 4-player real-time strategy game played on a 100×100 continuous board with a sun at the center. Planets orbit the sun, comets streak through on elliptical paths, and players grow by launching fleets to capture neutral and enemy planets. After 500 turns, whoever controls the most ships wins.

This repository is my attempt to train an agent to play it well — and, more honestly, it's a study in **building the machinery that makes reinforcement-learning experiments fast, reproducible, and trustworthy**. Several variations of the agent showed early competence that didn't hold: against a fixed scripted opponent the best run plateaued at mediocre skill, and my most recent tuned recipe reproducibly destabilized and collapsed late in training. Rather than commit the substantially larger time budget that diagnosing and fixing this would have required, I made a deliberate decision to stop. The lasting result is the engineering and the lessons, which I've written up candidly in **[RETROSPECTIVE.md](RETROSPECTIVE.md)**.

---

## What this project is

A complete, single-operator pipeline for taking an idea from a config file to a trained, evaluated, and packageable game agent:

- **A vectorized game simulator in JAX** that runs many self-play games in parallel on GPU, kept faithful to the official game rules by a separate plain-Python reference implementation and automated parity tests.
- **A PPO (Proximal Policy Optimization) self-play training loop** with a curriculum of opponents — random, scripted heuristics, and snapshots of the agent's own past selves.
- **A configuration-first design** (Hydra) where every experiment is a composition of small, swappable building blocks (model, reward, opponent mix, training budget), so runs are reproducible and easy to vary.
- **A single command-line tool (`ow`)** covering training, evaluation, tournaments, benchmarks, and Kaggle submission packaging.
- **My first attempt at working with autoregressive/factorized decoders** This project is my first attempt at working with more complex decoders that can output a distribution of actions to take and then sample from that distribution to generate a variable length action sequence while adjusting for the fact that the actions are not independent and depend on the previous actions taken. 

If I had to summarize the contribution in one line: *I'm most proud of the systems and reproducibility infrastructure — the part that turns "I have an idea" into "I have a measured answer" quickly and reliably.*

---

## Highlights

### Systems & reproducibility (the headline)

- **Composable experiment configuration.** Models, rewards, opponent mixes, and training budgets are independent building blocks. Changing an experiment means swapping a block, not editing code.
- **Campaign-based run layout.** Every run records its full resolved configuration, metrics, and checkpoints under a structured campaign directory, so any result can be traced back to exactly how it was produced.
- **One operator CLI.** Training, watching live runs, evaluation, tournaments, throughput benchmarks, and submission packaging all live behind a single `ow` command with consistent, scriptable subcommands.

### Performance engineering

- **GPU-parallel self-play.** The environment is written so that many games step forward simultaneously under JAX's `jit`/`vmap`, instead of one game at a time in Python.
- **Measured throughput.** On a single NVIDIA L4 GPU, my longest clean run sustained a median of **~3,600 environment steps/sec** end-to-end (**~4,600/sec** during rollout collection) and **~10,900 training samples/sec**, completing **~49.1M environment steps in 3.8 hours** (3,000 PPO updates). *Source: W&B run `...2p-nearest_sniper-u3000-env32-s42-20260615T165939Z` in `jmduea-jdueadev/orbit_wars`; see `[outputs/CANONICAL_RUNS.md](outputs/CANONICAL_RUNS.md)`.*
- **Pre-baked maps to rescue vectorization.** The game's procedural map and comet generation is branchy and resists parallelization — naively it erased the speedups vectorization gave everywhere else. I moved map generation *offline* into a pre-computed pool the training loop simply samples from, preserving the GPU throughput without sacrificing rule fidelity. (More on this in the retrospective; it was my favorite problem in the project.)

### Reinforcement-learning machinery

- **Faithful simulator with a parity guardrail.** A readable Python reference implementation defines ground truth; the fast JAX version is continuously checked against it so optimizations can't silently change the game.
- **Self-play curriculum.** Opponents range from random and scripted heuristics to historical snapshots of the agent, with staged progression.
- **A configurable graph/attention policy** over the planet–fleet layout (a "planet-graph transformer"), exposed in full and small capacity variants selectable by config. Earlier exploration carried several simpler architectures (plain MLPs, a pointer network), but the project deliberately consolidated onto this single family — a simplification I'd revisit (see the retrospective: re-adding a minimal MLP baseline is on my list).

---

## Tech stack


| Area                | Tools                                            |
| ------------------- | ------------------------------------------------ |
| Language            | Python 3.12                                      |
| Learning / numerics | JAX, Flax, Optax (PPO)                           |
| Configuration       | Hydra                                            |
| Experiment tracking | Weights & Biases                                 |
| Packaging / env     | uv, Docker (for Kaggle submission validation)    |
| Testing             | pytest (tiered: fast CPU, JAX, slow integration) |
| Game / competition  | `kaggle-environments` (Orbit Wars)               |


---

## How it works, end to end

1. **Compose a run** from configuration blocks (`uv run ow train model=attention training.total_updates=1000`).
2. **Collect rollouts** in the GPU-parallel simulator against the configured opponent mix.
3. **Update the policy** with PPO.
4. **Track and checkpoint** metrics and weights into the run's campaign directory; optionally log to Weights & Biases.
5. **Evaluate** against scripted opponents and in tournaments to ask "is this actually better?"
6. **Package and validate** a submission for the Kaggle environment.

```bash
uv sync --group dev                                   # install
make test-fast                                        # fast verification loop
uv run ow train model=transformer_factorized_small training.total_updates=1000
uv run ow train print_resolved_config=true            # resolve config without training
uv run ow runs list                                   # inspect past runs
```

---

## Where to look in the code


| Path            | Role                                                                   |
| --------------- | ---------------------------------------------------------------------- |
| `src/jax/`      | GPU-parallel environment, feature encoding, policy, PPO, training loop |
| `src/game/`     | Plain-Python reference game logic (ground truth for parity)            |
| `src/features/` | Observation encoding (how the board becomes model input)               |
| `src/config/`   | Typed configuration schema and composition                             |
| `src/cli/`      | The `ow` command-line tool                                             |
| `conf/`         | Composable Hydra configuration blocks                                  |
| `tests/`        | Tiered test suite, including Python↔JAX parity                         |
| `docs/`         | Architecture notes, onboarding, and documented learnings               |


A deeper tour lives in `[docs/ONBOARDING.md](docs/ONBOARDING.md)`.

---

## A note on AI-assisted development

A meaningful part of this project was an experiment in **developing with AI coding agents as a first-class part of the workflow** — not just autocomplete, but agents driving exploration, refactors, and tooling against a documented set of project conventions. It was a genuine accelerator for development velocity, and an equally genuine lesson in where that velocity helps and where it quietly hurts. I treat that honestly in the [retrospective](RETROSPECTIVE.md) rather than as a selling point: the same leverage that let me build a lot, fast, also made it easy to build the *wrong* things convincingly.

---

