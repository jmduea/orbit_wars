# Orbit Wars — Retrospective

A candid post-mortem of a solo reinforcement-learning project I built over a little more than a month, and then deliberately set down. This is written to be honest rather than flattering: the most useful thing I can offer a reader (or my future self) is a clear account of what went well, what didn't, and what I'd change.

---

## Context

I built this shortly after finishing my B.S. in Data Science. Going in, this was the second kaggle agent competition I've worked on and I was just beginning to get comfortable with reinforcement learning. I was both confident enough to take on something ambitious, while fully aware that I still have a lot to learn about RL research methodology and how it plays out in practice.

The goal was to train an agent for **Orbit Wars**, a 2–4 player real-time space-strategy game run as a Kaggle simulation competition. Concretely, I wanted to: build a fast self-play training pipeline in JAX, get an agent to learn the game through PPO, and produce a valid competition submission. The self-play training pipeline and *technically* valid competition submission goals were definitively achieved, but the agent did not perform well enough to be competitive.

I spent a little over a month on it, working solo, and made heavy use of AI coding agents as part of the workflow.

---

## The honest outcome

**The agent showed early competence that didn't hold — it either plateaued or collapsed depending on the recipe. I chose to stop.**

Pushing past this looked like it would require substantially more time than I was willing to commit including more tuning, more careful experimentation, and real debugging of training stability to disambiguate whether the cause was due to chosen hyperparameters or an issue in the ppo update logic/calculations itself. So I stopped on purpose, satisfied with what I've learned.

What I'm less satisfied with is *why* it failed: **I never fully isolated the cause before stopping, and even more importantly, I should have taken a more methodical approach to the project as a whole**. I have hypotheses as to what was the cause of the issues, but I recognize now that if I had started from a minimal baseline and methodically performed the experiments only changing one variable at a time, I'd know whether the bottleneck was the reward signal, the exploration/self-play dynamics, the policy architecture, or the PPO update stability itself immediately instead of having to go back and debug. That gap is itself one of the project's biggest lessons, and I get into why below.

A note on rigor, added when I revisited the logs to write this up: the truth was a little worse — and more interesting — than I'd remembered. My first instinct was to call it a clean "plateau," but going back to the *authoritative* run histories on Weights & Biases (rather than the misleadingly-named local files, several of which were crashed or partial) showed two distinct failure modes, described below. That gap between my remembered story and what the data actually said is, itself, a lesson.

The durable result of the project, then, isn't a leaderboard placement. It's the **engineering system** I built around the problem and the **practical lessons** about how to (and how not to) run a project like this.

---

## What went well

### Iteration speed is incredibly value as long as it isn't at the cost of sound methodology.

The single clearest takeaway: the ability to iterate quickly is what lets you actually evaluate ideas. Every part of the project that invested in faster, more reproducible experiments paid for itself, and every part that slowed iteration down hurt — regardless of how clever it was. I came away genuinely convinced that "how fast can I get a trustworthy answer to a question?" is the metric that governs progress on this kind of work. I think pairing that mentality with a more rigorous and methodical approach to experiment design will do me well in the future.

### A surprising result about environment fidelity vs. speed

For part of the project I trained against a faster environment that I *knew* wasn't a full, faithful reproduction of the official game rules, while I was still working out policy architectures and the training pipeline itself. When I later moved to the full-parity environment, the difference in the learned policy was much smaller than I expected — training on the "good enough but not exact" environment hadn't meaningfully changed the outcome at that stage.

That raised a genuinely interesting question for me about *when* fidelity matters. Early on, while you're still shaping the architecture and the pipeline, a cheaper approximate environment may be perfectly adequate, and the speed it buys is more valuable than the exactness it gives up. Knowing where on that curve you are — and not paying for precision you can't yet use — feels like an important instinct to develop.

More generally, I'd treat full parity as a cost to be justified, not a default. Depending on the environment, it's worth explicitly asking whether a *deliberate approximation* is safe to train against when achieving full parity represents a large engineering barrier for marginal or unknown gain. The full-parity build here was hard-won and I'm glad I did it, but the right call isn't always "make it exact" — sometimes it's "make it good enough, on purpose, and know exactly where the approximation could bite you."

### Being forced to think outside the box to keep JAX's benefits

Getting the full-parity environment to run fast was the most satisfying technical problem in the project. The game's map generation and comet spawning are branchy and irregular, and they completely erased the speedups that vectorizing the rest of the environment had bought me. Rather than give up the parallelism, I moved map generation *offline* into a pre-computed pool that the training loop samples from — preserving the GPU throughput without compromising rule fidelity. It was a good lesson in separating the parts of a system that *must* be exact-and-online from the parts that can be precomputed.

### AI-assisted development as a real accelerator

Building with AI coding agents substantially raised my development velocity and let me stand up far more infrastructure than I could have alone in the same window. Just as valuable was learning, firsthand, how different ways of working with these tools either help or hinder progress — which is experience I expect to keep using. (It cut both ways, which I cover in the next section.)

### The experience itself

Overfall, this was a high-value experience in RL plus ML-systems engineering, the kind of full-stack, "make the whole loop work" effort that's hard to get from a course or a tutorial, and I feel like having the chance to explore in an independent project like this was exactly what I needed to identify where I'm weakest so I can focus on remedying those gaps going forward.

---

## What didn't go well

### I over-invested in infrastructure relative to RL progress

The systems I built are the part of the project I'm proudest of but they also absorbed more of my time than the actual goal (a learning agent) justified. I let the tooling become the work. Good infrastructure was a real enabler, but I crossed the line from "infrastructure that serves the experiments" into "infrastructure as the project." and admittedly did find myself having to pull back several times when it was getting in the way more than it was helping.

### I made a bad timing choice by implementing quality gates/thresholds prematurely which turned out to have a much bigger cost than I would have guessed

During a stretch of throughput-recovery work, I was forced to reconcile with the fact that I had jumped the gun on implementing throughput/quality benchmark thresholds meant to fail training configurations that degraded performance by more than 10%; in theory this was meant to ensure quality and optimized training runs and help make sure that any changes I or a coding agent made didn't degrade the performance of the pipeline or quality of the trained agent catastrophically, but because I had put it in place BEFORE finalizing the full-parity environment, they were calibrated to unmeetable standards. This wouldn't have been that big of a deal, but I had made the choice to put in place strict guidance/guardrails for coding agents to check there work against these thresholds and until I ripped them out completely, coding agents would continually get confused about what was the appropriate threshold. It was a sharp time-and-place lesson: the throughput/quality verification work is a choice I'd still stand by just not the specific moment I had chosen to put them in place; the right move would have been to *finish the pipeline engineering first* and then measure against honest, stable thresholds.

### I accumulated complexity debt

Configurable, composable systems are powerful, but I added knobs and abstractions faster than I added understanding. Past a point, the flexibility started slowing me down: more surface area to reason about, more ways for an experiment to differ from what I thought I was running. Some of the complexity I built to "move faster later" mostly made the present harder. Something I intend to keep in mind in the future, since when using algorithms like PPO, having many knobs to tweak is inherint to the algorithm, but I could have taken more steps to keep it under control.

### AI-assisted work created rabbit holes

The flip side of the velocity: AI agents made it easy to produce work that was *plausible but wrong*, and untangling that cost real time. Convincing-looking code and confident-sounding reasoning are not the same as correct, and the faster you generate, the more carefully you have to verify. Several of my worst time sinks were chasing down or unwinding work that looked right and wasn't. It's tempting to let an agent go wild and see what comes out of that but in my experience so far doing so often results in more work and confusion than it's worth.

### I didn't establish a simple working baseline early enough

This is the root cause behind several of the others. I didn't first get a *deliberately simple* agent learning *something* before reaching for sophistication and scale. Without that baseline, I had no clean reference point, which is a large part of why, when the agent plateaued/showed training stability issues, I couldn't cleanly isolate the cause. There was too much going on at once to tell which moving part was responsible.

---

## On the plateau (and the collapse), specifically

When I went back to the logs to write this honestly, the long runs showed **two different failure modes**:

- **Plateau, against a fixed scripted opponent.** A 3,000-update run against a single scripted "sniper" opponent flatlined at roughly a 30% win rate from very early on, while the policy's entropy (its degree of exploration) collapsed within the first fifth of training. The agent committed early to a mediocre strategy and then stopped improving, I should have put in early-stopping checks for cases like this to avoid wasting compute.
- **Collapse, in my most recent tuned recipe.** My latest setup (a mixed scripted curriculum with a sparse win/lose reward) started promisingly — around a 60% win rate in the first couple hundred updates — and then *collapsed* to nearly 0% and stayed there, with the PPO update statistics going pathological. Critically, **a second run reproduced the exact same collapse**, so it was a property of the recipe, not bad luck. This is a training-*stability* failure, and more time would only have made it worse. Something that early-stopping would also mitigate the cost of.

In hindsight, a simpler, fully-understood starting point would have made these outcomes *diagnostic*. (The supporting numbers live in `[outputs/CANONICAL_RUNS.md](outputs/CANONICAL_RUNS.md)`.) and much easier to debug if I were choosing to continue working on this.

---

## What I'd do again

- **Treat reproducible experimentation as a first-class concern.** Composable configuration, structured run records, and "every result is traceable to exactly how it was produced" — I'd keep this mindset every time. (I'd just right-size it; see below.)
- **Document decisions and learnings as I go.** Writing things down while the context was fresh was consistently worth it, including for this retrospective, and I think the value of this only increases the more removed from actual code implementation you are.
- **Develop with AI assistance — but with tighter guardrails.** The leverage is real. Next time I'd pair it with stricter verification, smaller and more reviewable steps, and a clearer line between "explore" and "commit."
- **Keep building bounded-but-ambitious personal projects.** Some people may look at work like this and see that it didn't produce a competitive submission in the related competition and call it a failure, but honestly, to me the hands on experience seeing what works and what didn't is what matters and I'd only call it a failure if I somehow managed **not** to learn something valuable from it.

---

## What I'd absolutely avoid

- **Jumping straight to a complex model I didn't fully understand just because I could.** This is the one I feel most strongly about. The combination of capable tools and my own ambition made it easy to reach for sophistication I hadn't earned an understanding of yet. Next time: simplest thing that could possibly work, fully understood, first, and iterate from there.
- **Not being more rigorous about verification of AI-generated work.** Speed without verification is just faster debt.
- **Not measuring against a baseline/measuring the outcome each modification.** I could have avoided a lot of time sinks by measuring the outcome of each modification against a baseline, and been much more confident in the results of my experiments if I had maintained a dedicated ledger.

---

## If I started over tomorrow

1. **Baseline first.** Get the simplest possible agent to learn *anything* measurable against a trivial opponent, end to end, before adding a single sophistication. Even better, make use of the working reinforcement learning example notebook shared in the competitions discussions and work from there so that I don't repeat work someone else already did for me.
2. **One variable at a time.** Change the reward, or the opponent, or the architecture — not three at once — so that every plateau or jump is interpretable.
3. **Right-size the infrastructure.** Build exactly enough tooling to make the *next* experiment faster and traceable, and no more, until a concrete need pulls more into existence.
4. **Use AI as a fast, fallible collaborator.** Lean on it for velocity, but keep changes small, reviewed, and verified, and never let it carry me into a design I don't understand.
5. **Define pass/fail criteria up front.** Preferably with metrics that can be measured for statistical significance, so that I can be confident in the results of my experiments.

---

## If I picked this back up

"Start over" and "resume" are different problems. If I came back to *this* codebase later, I wouldn't throw it away — the infrastructure is real leverage, and I'd point it at the questions it was built to answer. Roughly in order:

1. **Add an early-stopping sentinel**. Targeted at the exact collapse/instability cases I observed, doing this before any further work is justified so that I don't accidently end up wasting more compute on bad runs that could have been avoided.
2. **Build the simple baseline I skipped — including a simpler model.** Smallest viable setup: a dense shaped reward and a single trivial opponent (noop, then random), run end to end until it *reliably* learns something measurable. Concretely, I'd **re-introduce a plain MLP policy** (the project consolidated onto a single graph/attention transformer family and dropped its simpler architectures). I'd also run this simple baseline using the closest config shape I can to to the runs that collapsed/showed training instability to see if it's reproduced and isolate whether the issue stems from the PPO update calculations themselves or a bad config/architecture combination. Depending on whether it's reproduced or not would inform whether a deeper audit is needed.
3. **Test the attention-based encoder against the baseline WITHOUT the complex factorized decoder**. My gut and general understanding tells me that an attention based/transformer architecture should outperform a simple mlp, what I need is the numbers to back that up before adding other architecture complexity to the mix. This also includes dropping the autoregressive style factorized decoder, until I have the evidence base to justify it's needed and can measure the value it provides.
4. **Attack the plateau with controlled, single-variable experiments.** With a baseline in hand and a numbers backed decision on mlp vs transformer, work outward one lever at a time: exploration (an entropy schedule rather than a fixed low coefficient, since entropy collapsed early), reward shaping (dense vs. sparse), curriculum pacing against scripted opponents, and only then policy/action-space capacity. Each experiment should answer exactly one question.
5. **Use the existing infrastructure for what it's genuinely good at.** This is where the over-built tooling finally pays off: composable configs, traceable campaigns, and calibrated pass/fail gates are exactly what disciplined, one-variable-at-a-time experimentation needs. I'd resist adding *more* infrastructure and instead spend the budget on experiments — with one targeted exception: **fix the run auto-naming so names actually reflect the run.** Today a run can be named `planet_graph_transformer-…-random-…` while it's really a `transformer_factorized_small` model trained against a `scripted_heavy` opponent curriculum — the name encodes stale defaults rather than the fields that matter (model, true opponent source, reward mode, key hyperparameters). That single gap cost me real time and nearly led me to a wrong conclusion when I revisited the logs; making names trustworthy is a small change that compounds across every future experiment.
6. **Stop after training an agent that proves it can learn through self-play and reliably beat nearest_sniper**. That would effectively prove I accomplished the last of the three goals I set out to do when I started, and prepare me for the next competition.

---

## Closing thought

The most valuable outcome of this project wasn't a trained agent, it was learning, concretely and a little uncomfortably, that **the discipline of how you run an experiment matters more than the cleverness of what you put in it.** That lesson paired with the experience of building the whole thing myself and watching where it broke is something I'll keep with me going forward as I work to address the gaps I identified in my knowledge and understanding of RL and build a stronger foundation for future projects.
