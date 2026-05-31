# Deep Dive Trace: Decoder Carry

## Problem
`model.decoder_carry=true` was intended to persist the decoder GRU hidden state across turns while preserving PPO replay consistency.

## Lane A - Code Path (most likely)
**Hypothesis:** The shielded sampler advanced decoder carry inside the per-action shield loop instead of treating it as turn-level state.

**Evidence for:**
- `_sample_shielded_factored_sequence_with_params` and the joint-flat path seeded their scan-local `decoder_hidden_carry` from `probe_output.decoder_hidden`.
- `probe_output.decoder_hidden` is already the hidden state after a full K-step decode.
- Each shield step then called `policy.apply` and updated `decoder_hidden_carry` again from another full K-step decode.
- Rollout transitions store the incoming pre-turn `decoder_hidden`; PPO replay uses that incoming hidden with the final sampled sequence.

**Evidence against:**
- Rollout reset logic correctly zeros rows after terminal environments.
- Policy modules correctly gate returned hidden state on `model.decoder_carry`.

**Critical unknown:** Whether any future sampler optimization should expose decoder step primitives to avoid repeated full-sequence applies.

**Discriminating probe:** Compare sampler `decoder_hidden_out` with a single replay `policy.apply` from the incoming hidden and final sampled sequence. This failed before the fix and is now covered by `tests/test_decoder_carry.py`.

## Lane B - Config / Environment
**Hypothesis:** Config enabled an incomplete runtime path.

**Evidence for:**
- `conf/model/decoder_carry.yaml` and `conf/model/transformer_factorized.yaml` can set `decoder_carry: true`.
- The flag is schema-backed and default-off for compatibility.

**Evidence against:**
- No evidence of Hydra composition or checkpoint metadata causing the mismatch; the defect is independent of config source.

**Critical unknown:** None after code-path probe.

## Lane C - Measurement / Tests
**Hypothesis:** Existing tests validated shapes and reset helpers but not sampler/replay equivalence.

**Evidence for:**
- Prior tests checked policy output shape and reset-on-done behavior.
- They did not exercise `_sample_shielded_sequence_with_params` with an incoming carry.

**Evidence against:**
- PPO on-policy KL tests cover replay consistency generally, but not this carry-enabled sampler path.

**Critical unknown:** Whether slow rollout tests should add an end-to-end carry smoke before merge.

## Convergence
The root cause was sampler misuse of `decoder_hidden` as an intra-turn scan carry. Cross-turn decoder carry must remain the pre-turn hidden for all shielded per-step policy probes, then be advanced exactly once by replaying the final sampled sequence.

## Fix
- Keep scan-local `decoder_hidden_carry` equal to the incoming hidden.
- Do not update it from each per-step `policy.apply`.
- After sampling the final sequence, compute `decoder_hidden_out` with one policy replay from `decoder_hidden_in`.
- Added factorized and joint-flat replay-alignment tests.

## Verification
- `uv run --group dev pytest tests/test_decoder_carry.py -m "not slow and not jax"`: 5 passed.
- `make test-fast`: 181 passed, 79 deselected.
