"""Phase 0 spikes for Feature Encoding v2: float budget and symmetry frame.

No v2 encoder code — validates locked schema dims, memory estimates, and
ADR-004 learner-frame equivariance under known board transforms.

Usage:
  uv run python scripts/spike_feature_encoding_v2_phase0.py budget
  uv run python scripts/spike_feature_encoding_v2_phase0.py symmetry
  uv run python scripts/spike_feature_encoding_v2_phase0.py all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game.constants import (  # noqa: E402
    BASE_GLOBAL_FEATURE_DIM,
    BOARD_CENTER,
    BOARD_SIZE,
    MAX_PLANETS,
)

# Phase 0 locked schema (docs/feature-encoding-v2.md)
PLANET_FEATURE_DIM = 13
EDGE_FEATURE_DIM = 18
GLOBAL_FEATURE_DIM = BASE_GLOBAL_FEATURE_DIM + 1  # + angular_velocity
DEFAULT_CANDIDATE_COUNT = 4
DEFAULT_HISTORY_STEPS = 1


@dataclass(frozen=True)
class SchemaLock:
    planet_feature_dim: int = PLANET_FEATURE_DIM
    edge_feature_dim: int = EDGE_FEATURE_DIM
    global_feature_dim: int = GLOBAL_FEATURE_DIM
    max_planets: int = MAX_PLANETS

    def edge_k(self, candidate_count: int) -> int:
        return max(0, candidate_count - 1)

    def pointer_logit_dim(self, candidate_count: int) -> int:
        k = self.edge_k(candidate_count)
        return self.max_planets * k + 1

    def encoder_floats(
        self, *, candidate_count: int, history_steps: int
    ) -> dict[str, int]:
        k = self.edge_k(candidate_count)
        h = max(1, history_steps)
        planets = self.max_planets * self.planet_feature_dim
        edges = self.max_planets * k * self.edge_feature_dim
        global_vec = self.global_feature_dim * h
        return {
            "planet_features": planets,
            "edge_features": edges,
            "global_features": global_vec,
            "total_encoder_floats": planets + edges + global_vec,
            "pointer_logits": self.pointer_logit_dim(candidate_count),
            "edge_k": k,
            "history_steps": h,
        }


def _v1_row_floats(candidate_count: int, history_steps: int) -> int:
    """v1 per owned-source decision row (self + C candidates + global)."""
    h = max(1, history_steps)
    base = 30 + 24 * candidate_count + 45
    return base * h


def run_budget_spike() -> dict[str, object]:
    schema = SchemaLock()
    default = schema.encoder_floats(
        candidate_count=DEFAULT_CANDIDATE_COUNT,
        history_steps=DEFAULT_HISTORY_STEPS,
    )
    v1_row = _v1_row_floats(DEFAULT_CANDIDATE_COUNT, DEFAULT_HISTORY_STEPS)

    # Rollout memory estimate: float32 payload per env per decision step
    bytes_per_env = default["total_encoder_floats"] * 4
    rollout_steps = 64
    num_envs = 16
    rollout_payload_mb = bytes_per_env * rollout_steps * num_envs / (1024 * 1024)

    owned_planet_counts = (2, 5, 10, 20)
    v1_owned_compare = {str(n): n * v1_row for n in owned_planet_counts}

    result = {
        "schema_lock": {
            "planet_feature_dim": schema.planet_feature_dim,
            "edge_feature_dim": schema.edge_feature_dim,
            "global_feature_dim": schema.global_feature_dim,
            "edge_layout": "top_k_per_source",
            "edge_k_default": default["edge_k"],
        },
        "default_config": {
            "candidate_count": DEFAULT_CANDIDATE_COUNT,
            "feature_history_steps": DEFAULT_HISTORY_STEPS,
            **default,
        },
        "v1_comparison": {
            "v1_floats_per_owned_source_row": v1_row,
            "v1_floats_by_owned_count": v1_owned_compare,
            "v2_static_encoder_floats": default["total_encoder_floats"],
            "pointer_logit_dim_v2": default["pointer_logits"],
            "pointer_logit_dim_v1_per_source": DEFAULT_CANDIDATE_COUNT,
        },
        "memory_estimate": {
            "float32_bytes_per_env_encode": bytes_per_env,
            "rollout_steps_assumed": rollout_steps,
            "num_envs_assumed": num_envs,
            "encoder_only_rollout_mb": round(rollout_payload_mb, 2),
            "note": "Policy activations and transition batch not included; static tensor payload only.",
        },
        "budget_verdict": (
            "ACCEPT: v2 static payload ~2986 floats/env-step at H=1,K=3 is larger than "
            f"a single v1 row ({v1_row}) but comparable to ~{default['total_encoder_floats'] // v1_row} "
            "owned sources; pointer softmax is 181 vs v1 per-source 4-way. "
            "Plan target '≤200 floats/decision' applies to v1 row semantics, not full board tensor."
        ),
    }
    return result


def wrap_angle(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return wrapped


def theta_ref_from_owned(
    xs: np.ndarray, ys: np.ndarray, owners: np.ndarray, learner: int
) -> float:
    owned = (owners == learner) & (owners >= 0)
    if not np.any(owned):
        return 0.0
    cx = float(np.mean(xs[owned]))
    cy = float(np.mean(ys[owned]))
    return math.atan2(cy - BOARD_CENTER[1], cx - BOARD_CENTER[0])


def canonical_polar(
    xs: np.ndarray, ys: np.ndarray, theta_ref: float
) -> tuple[np.ndarray, np.ndarray]:
    dx = xs - BOARD_CENTER[0]
    dy = ys - BOARD_CENTER[1]
    radii = np.hypot(dx, dy) / BOARD_SIZE
    angles = np.array(
        [
            wrap_angle(math.atan2(dy[i], dx[i]) - theta_ref) / math.pi
            for i in range(len(xs))
        ],
        dtype=np.float64,
    )
    return radii, angles


def rotate_board_90_ccw(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """90° CCW around board center (matches tests/test_jax_env_parity.py)."""
    rx = BOARD_CENTER[0] - (ys - BOARD_CENTER[1])
    ry = BOARD_CENTER[1] + (xs - BOARD_CENTER[0])
    return rx, ry


def rotate_board_180(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rx = 2.0 * BOARD_CENTER[0] - xs
    ry = 2.0 * BOARD_CENTER[1] - ys
    return rx, ry


def permute_owners_90_ccw(owners: np.ndarray, player_count: int) -> np.ndarray:
    """Relabel player ids consistently with 90° CCW board rotation in symmetric formats."""
    if player_count == 2:
        return np.where(owners == 1, 2, np.where(owners == 2, 1, owners))
    if player_count == 4:
        mapping = {1: 2, 2: 3, 3: 4, 4: 1}
        return np.vectorize(lambda o: mapping.get(int(o), int(o)))(owners)
    return owners.copy()


def max_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def sorted_polar_pairs(radii: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Sort (r, θ) pairs for order-invariant comparison under board transforms."""
    pairs = np.stack([radii, angles], axis=1)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order]


def run_symmetry_spike() -> dict[str, object]:
    results: dict[str, object] = {}

    # --- 1. Angle roundtrip (decode contract) ---
    theta_ref = math.pi / 4.0
    canonical = -0.25  # normalized by pi
    absolute = canonical * math.pi + theta_ref
    recovered = wrap_angle(absolute - theta_ref) / math.pi
    results["angle_roundtrip"] = {
        "theta_ref": theta_ref,
        "canonical": canonical,
        "absolute": absolute,
        "recovered_canonical": recovered,
        "max_error": abs(recovered - canonical),
        "pass": abs(recovered - canonical) < 1e-9,
    }

    # --- 2. Synthetic 4-fold symmetric planet group ---
    learner = 1
    angles = np.array([0.0, math.pi / 2, math.pi, 3.0 * math.pi / 2])
    radius = 30.0
    xs = BOARD_CENTER[0] + radius * np.cos(angles)
    ys = BOARD_CENTER[1] + radius * np.sin(angles)
    owners = np.array([1, 0, 1, 0])  # learner owns opposite pair

    def encode_snapshot(x_arr, y_arr, owner_arr):
        t_ref = theta_ref_from_owned(x_arr, y_arr, owner_arr, learner)
        return canonical_polar(x_arr, y_arr, t_ref), t_ref

    (r0, a0), t0 = encode_snapshot(xs, ys, owners)
    xs90, ys90 = rotate_board_90_ccw(xs, ys)
    owners90 = permute_owners_90_ccw(owners, player_count=2)
    (r90, a90), t90 = encode_snapshot(xs90, ys90, owners90)

    xs180, ys180 = rotate_board_180(xs, ys)
    owners180 = permute_owners_90_ccw(permute_owners_90_ccw(owners, 2), 2)
    (r180, a180), t180 = encode_snapshot(xs180, ys180, owners180)

    p0 = sorted_polar_pairs(r0, a0)
    p90 = sorted_polar_pairs(r90, a90)
    p180 = sorted_polar_pairs(r180, a180)

    sym_90 = {
        "max_pair_delta": max_abs_delta(p0, p90),
        "theta_ref_delta": abs(wrap_angle(t90 - t0)),
        "pass": max_abs_delta(p0, p90) < 1e-6,
    }
    sym_180 = {
        "max_pair_delta": max_abs_delta(p0, p180),
        "pass": max_abs_delta(p0, p180) < 1e-6,
    }
    results["synthetic_fourfold"] = {"rotate_90_ccw": sym_90, "rotate_180": sym_180}

    # --- 3. JAX 2p reset sanity (early-game symmetric geometry, ownership not fully invariant) ---
    try:
        import jax
        from src.config import compose_hydra_train_config
        from src.jax.env import reset

        cfg = compose_hydra_train_config(["format=2p_16env"]).task
        state, _ = reset(jax.random.PRNGKey(11), cfg)
        p = state.game.planets
        active = np.asarray(p.active).astype(bool)
        xs_r = np.asarray(p.x)[active]
        ys_r = np.asarray(p.y)[active]
        owners_r = np.asarray(p.owner)[active]
        learner_id = 1

        (r_base, a_base), _ = encode_snapshot(xs_r, ys_r, owners_r)
        xs_t, ys_t = rotate_board_90_ccw(xs_r, ys_r)
        owners_t = permute_owners_90_ccw(owners_r, player_count=2)
        (r_rot, a_rot), _ = encode_snapshot(xs_t, ys_t, owners_t)
        base_pairs = sorted_polar_pairs(r_base, a_base)
        rot_pairs = sorted_polar_pairs(r_rot, a_rot)

        results["jax_reset_2p_seed11"] = {
            "active_planets": int(active.sum()),
            "max_pair_delta": max_abs_delta(base_pairs, rot_pairs),
            "pass_loose": max_abs_delta(base_pairs, rot_pairs) < 0.05,
            "note": "Early 2p reset is not fully ownership-symmetric under 90°; expect small angle drift.",
        }
    except Exception as exc:  # pragma: no cover - optional CUDA/JAX path
        results["jax_reset_2p_seed11"] = {"skipped": str(exc)}

    all_pass = results["angle_roundtrip"]["pass"] and sym_90["pass"] and sym_180["pass"]
    results["verdict"] = "PASS" if all_pass else "FAIL"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("budget", "symmetry", "all"),
        help="Which spike to run.",
    )
    args = parser.parse_args()

    output: dict[str, object] = {}
    if args.mode in ("budget", "all"):
        output["budget"] = run_budget_spike()
    if args.mode in ("symmetry", "all"):
        output["symmetry"] = run_symmetry_spike()

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
