from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from .policy import PolicyOutput


@dataclass(slots=True)
class SampledAction:
    target_index: torch.Tensor
    ship_bucket: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


@dataclass(slots=True)
class TransitionBatch:
    self_features: torch.Tensor
    candidate_features: torch.Tensor
    global_features: torch.Tensor
    candidate_mask: torch.Tensor
    target_index: torch.Tensor
    ship_bucket: torch.Tensor
    log_prob: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    step_id: torch.Tensor


def sample_actions(outputs: PolicyOutput, deterministic: bool) -> SampledAction:
    target_logits = safe_target_logits(outputs.target_logits)
    target_dist = Categorical(logits=target_logits)
    if deterministic:
        target_index = target_logits.argmax(dim=-1)
        selected_ship_logits = gather_target_ship_logits(outputs.ship_logits, target_index)
        ship_bucket = selected_ship_logits.argmax(dim=-1)
    else:
        target_index = target_dist.sample()
        selected_ship_logits = gather_target_ship_logits(outputs.ship_logits, target_index)
        ship_dist = Categorical(logits=selected_ship_logits)
        ship_bucket = ship_dist.sample()

    log_prob, entropy = action_log_prob_and_entropy(
        outputs=outputs,
        target_index=target_index,
        ship_bucket=ship_bucket,
    )
    return SampledAction(target_index=target_index, ship_bucket=ship_bucket, log_prob=log_prob, entropy=entropy)


def action_log_prob_and_entropy(
    outputs: PolicyOutput,
    target_index: torch.Tensor,
    ship_bucket: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_logits = safe_target_logits(outputs.target_logits)
    target_dist = Categorical(logits=target_logits)
    selected_ship_logits = gather_target_ship_logits(outputs.ship_logits, target_index)
    ship_dist = Categorical(logits=selected_ship_logits)
    target_log_prob = target_dist.log_prob(target_index)
    ship_log_prob = ship_dist.log_prob(ship_bucket)
    target_entropy = target_dist.entropy()
    ship_entropy = ship_dist.entropy()
    return target_log_prob + ship_log_prob, target_entropy + ship_entropy


def gather_target_ship_logits(ship_logits: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
    if ship_logits.ndim == 2:
        return ship_logits
    if ship_logits.ndim != 3:
        raise ValueError(
            f"ship_logits must have shape [batch, ship_bucket_count] or "
            f"[batch, candidate_count, ship_bucket_count], got {tuple(ship_logits.shape)}"
        )
    batch_size, _, ship_bucket_count = ship_logits.shape
    gather_index = target_index.reshape(batch_size, 1, 1).expand(-1, 1, ship_bucket_count)
    return ship_logits.gather(dim=1, index=gather_index).squeeze(1)


def safe_target_logits(target_logits: torch.Tensor) -> torch.Tensor:
    invalid_rows = ~torch.isfinite(target_logits).any(dim=-1)
    if not invalid_rows.any():
        return target_logits
    safe_logits = target_logits.clone()
    safe_logits[invalid_rows, 0] = 0.0
    return safe_logits


def grouped_minibatch_indices(step_id: torch.Tensor, minibatch_size: int, device: torch.device) -> list[torch.Tensor]:
    """Build minibatches that keep all rows from the same environment step together."""

    unique_steps = step_id.unique()
    if unique_steps.numel() == 0:
        return []
    shuffled_steps = unique_steps[torch.randperm(unique_steps.numel(), device=device)]
    minibatches: list[torch.Tensor] = []
    current_indices: list[torch.Tensor] = []
    current_size = 0
    for step in shuffled_steps:
        indices = torch.nonzero(step_id == step, as_tuple=False).flatten()
        step_size = int(indices.numel())
        if current_indices and current_size + step_size > minibatch_size:
            minibatches.append(torch.cat(current_indices))
            current_indices = []
            current_size = 0
        current_indices.append(indices)
        current_size += step_size
    if current_indices:
        minibatches.append(torch.cat(current_indices))
    return minibatches


def broadcast_step_mean(values: torch.Tensor, step_id: torch.Tensor) -> torch.Tensor:
    """Average per-decision values into one critic value per step, then broadcast."""

    unique_steps, inverse = step_id.unique(return_inverse=True)
    step_sums = torch.zeros(unique_steps.shape[0], dtype=values.dtype, device=values.device)
    step_counts = torch.zeros(unique_steps.shape[0], dtype=values.dtype, device=values.device)
    step_sums.scatter_add_(0, inverse, values)
    step_counts.scatter_add_(0, inverse, torch.ones_like(values))
    step_means = step_sums / step_counts.clamp_min(1.0)
    return step_means[inverse]


def ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if batch.self_features.shape[0] == 0:
        return {
            "approx_kl": 0.0,
            "entropy_mean": 0.0,
            "clip_fraction": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
            "explained_variance": 0.0,
            "loss": 0.0,
            "entropy": 0.0,
        }
    self_features = batch.self_features.to(device)
    candidate_features = batch.candidate_features.to(device)
    global_features = batch.global_features.to(device)
    candidate_mask = batch.candidate_mask.to(device).bool()
    old_log_prob = batch.log_prob.to(device)
    target_index = batch.target_index.to(device)
    ship_bucket = batch.ship_bucket.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)
    step_id = batch.step_id.to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    size = self_features.shape[0]
    minibatch_size = min(size, max(1, minibatch_size))
    metrics = {
        "approx_kl": 0.0,
        "entropy_mean": 0.0,
        "clip_fraction": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "total_loss": 0.0,
    }
    metric_samples = 0
    for _ in range(epochs):
        for idx in grouped_minibatch_indices(step_id, minibatch_size, device):
            outputs = policy(
                self_features[idx],
                candidate_features[idx],
                global_features[idx],
                candidate_mask[idx],
            )
            new_log_prob, entropy = action_log_prob_and_entropy(
                outputs,
                target_index[idx],
                ship_bucket[idx],
            )
            log_ratio = new_log_prob - old_log_prob[idx]
            ratio = log_ratio.exp()
            policy_loss = torch.maximum(
                -advantages[idx] * ratio,
                -advantages[idx] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
            ).mean()
            centralized_values = broadcast_step_mean(outputs.value, step_id[idx])
            value_loss = 0.5 * (returns[idx] - centralized_values).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            batch_size = int(idx.numel())
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
            metrics["approx_kl"] += float(approx_kl.detach().cpu()) * batch_size
            metrics["entropy_mean"] += float(entropy_mean.detach().cpu()) * batch_size
            metrics["clip_fraction"] += float(clip_fraction.detach().cpu()) * batch_size
            metrics["policy_loss"] += float(policy_loss.detach().cpu()) * batch_size
            metrics["value_loss"] += float(value_loss.detach().cpu()) * batch_size
            metrics["total_loss"] += float(loss.detach().cpu()) * batch_size
            metric_samples += batch_size

    averaged_metrics = {key: value / max(metric_samples, 1) for key, value in metrics.items()}
    with torch.inference_mode():
        value_outputs = policy(self_features, candidate_features, global_features, candidate_mask)
        value_predictions = broadcast_step_mean(value_outputs.value, step_id)
        return_variance = returns.var(unbiased=False)
        if float(return_variance.detach().cpu()) <= 1e-12:
            explained_variance = torch.tensor(0.0, device=device)
        else:
            explained_variance = 1.0 - (returns - value_predictions).var(unbiased=False) / return_variance
    averaged_metrics["explained_variance"] = float(explained_variance.detach().cpu())
    # Backward-compatible aliases for existing callers and notebooks.
    averaged_metrics["loss"] = averaged_metrics["total_loss"]
    averaged_metrics["entropy"] = averaged_metrics["entropy_mean"]
    return averaged_metrics
