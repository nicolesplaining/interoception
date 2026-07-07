"""GDPO advantage function plug-in for prime-rl's CustomAdvantageConfig.

Implements the algorithm from "GDPO: Group reward-Decoupled Normalization Policy
Optimization for Multi-reward RL Optimization" (arXiv:2601.05242).

Per group i of G rollouts, per reward channel k:
    A_k[i,j] = (r_k[i,j] - mean_j(r_k[i,:])) / (std_j(r_k[i,:]) + eps)
    A_sum[i,j] = sum_k(w_k * A_k[i,j])
    A_hat[i,j] = (A_sum[i,j] - batch_mean) / (batch_std + eps)

This decouples the per-channel normalization (which happens per prompt group)
from the batch-level normalization of the combined advantage (which happens
across the entire batch). Compared to GRPO's single-scalar normalization,
this preserves signal distinctness across rollouts whose reward compositions
would otherwise collapse to identical advantages.

Wired to prime-rl via CustomAdvantageConfig:
    [orchestrator.advantage]
    type = "custom"
    import_path = "interoception_countdown.gdpo_advantage.gdpo_advantage_fn"
    kwargs = { channels = ["is_correct", "f_term"], weights = [1.0, 0.5], eps = 1e-6 }

The function reads `rollout["metrics"][channel]` per channel — these come from
verifiers' Rubric aggregation, which preserves each reward function's raw
scalar under its function name. Our env's rubric declares `is_correct` and
`f_term` as separate weight-0-or-nonzero functions, so both scalars are
already present in `metrics`.
"""
from __future__ import annotations

import torch
from torch import Tensor


def gdpo_advantage_fn(
    inputs,  # AdvantageInputs (not annotated to avoid prime-rl import at env load time)
    channels: list[str] | None = None,
    weights: list[float] | None = None,
    eps: float = 1e-6,
):
    """See module docstring. Args:

      inputs:   prime-rl AdvantageInputs. `inputs.rollouts[i][j]` is the j-th
                rollout of problem i. Each rollout is a dict with a "metrics"
                dict that includes the per-function raw reward values.
      channels: names of the metrics keys to treat as separate reward channels.
                Defaults to ["is_correct", "f_term"].
      weights:  per-channel weights w_k for the weighted sum step. Defaults to
                a vector of 1.0 the same length as `channels`.
      eps:      numeric floor for the divisor in both normalization steps.

    Returns:
      AdvantageOutputs with `advantages` shape [num_problems, rollouts_per_example].
    """
    # Imported here so that the env package doesn't need prime-rl at import time
    # (allows `import interoception_countdown` without prime-rl installed, e.g.
    # in the eval-only scripts).
    from prime_rl.orchestrator.advantage import AdvantageOutputs

    channels = list(channels) if channels else ["is_correct", "f_term"]
    weights = list(weights) if weights is not None else [1.0] * len(channels)
    if len(weights) != len(channels):
        raise ValueError(
            f"gdpo_advantage_fn: len(weights)={len(weights)} != len(channels)={len(channels)}"
        )

    groups = inputs.rollouts  # list[list[RolloutOutput]]
    num_problems = len(groups)
    if num_problems == 0:
        return AdvantageOutputs(advantages=torch.zeros(0, 0, dtype=torch.float32))
    G = len(groups[0])

    # Extract per-channel reward tensor: shape [K, num_problems, G]
    K = len(channels)
    per_channel = torch.zeros((K, num_problems, G), dtype=torch.float32)
    for k, chan in enumerate(channels):
        for i, group in enumerate(groups):
            for j, r in enumerate(group):
                metrics = r.get("metrics") or {}
                per_channel[k, i, j] = float(metrics.get(chan, 0.0))

    # Per-channel, per-group normalization: mean/std across G rollouts.
    means = per_channel.mean(dim=2, keepdim=True)      # [K, num_problems, 1]
    stds = per_channel.std(dim=2, keepdim=True)        # [K, num_problems, 1]
    A_per_channel = (per_channel - means) / (stds + eps)  # [K, num_problems, G]

    # Weighted sum across channels: [num_problems, G]
    w = torch.tensor(weights, dtype=torch.float32).view(K, 1, 1)
    A_sum = (w * A_per_channel).sum(dim=0)

    # Batch-level renormalization (across all num_problems * G values).
    batch_mean = A_sum.mean()
    batch_std = A_sum.std()
    A_hat = (A_sum - batch_mean) / (batch_std + eps)  # [num_problems, G]

    return AdvantageOutputs(advantages=A_hat)
