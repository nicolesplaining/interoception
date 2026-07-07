"""Unit test for gdpo_advantage_fn. Reproduces the toy collapse example from
the GDPO paper (arXiv:2601.05242) and verifies:

  1. per-channel normalization is on the group (dim=2) axis, not the wrong axis
  2. weighted sum + batch renormalization produces the expected relative ordering
  3. GDPO preserves distinct advantages where GRPO would collapse them

Runs without any prime-rl install by stubbing AdvantageInputs/Outputs.
"""
from __future__ import annotations
import sys
import types
from dataclasses import dataclass
from pathlib import Path

# Stub prime_rl.orchestrator.advantage so we don't need it installed. The GDPO
# function only imports AdvantageOutputs — provide a minimal stand-in.
_pkg_prime = types.ModuleType("prime_rl")
_pkg_orch = types.ModuleType("prime_rl.orchestrator")
_pkg_adv = types.ModuleType("prime_rl.orchestrator.advantage")

import torch


@dataclass
class _StubAdvantageOutputs:
    advantages: "torch.Tensor"


_pkg_adv.AdvantageOutputs = _StubAdvantageOutputs
sys.modules["prime_rl"] = _pkg_prime
sys.modules["prime_rl.orchestrator"] = _pkg_orch
sys.modules["prime_rl.orchestrator.advantage"] = _pkg_adv

# Make the env package importable regardless of pwd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "environments" / "interoception_countdown"))
from gdpo_advantage import gdpo_advantage_fn


@dataclass
class _StubInputs:
    rollouts: list


def _make(rewards_by_channel: dict[str, list[list[float]]]) -> _StubInputs:
    """Build a stub AdvantageInputs from per-channel per-group rewards.
    rewards_by_channel[chan][i][j] is the value for problem i, rollout j on channel `chan`."""
    channels = list(rewards_by_channel.keys())
    num_problems = len(rewards_by_channel[channels[0]])
    G = len(rewards_by_channel[channels[0]][0])
    groups = []
    for i in range(num_problems):
        group = []
        for j in range(G):
            r = {"metrics": {chan: rewards_by_channel[chan][i][j] for chan in channels}}
            group.append(r)
        groups.append(group)
    return _StubInputs(rollouts=groups)


def test_toy_collapse_example():
    """The paper's toy collapse example (Section 3.1):
      3 rollouts on 1 prompt, 2 reward channels
      rewards: (r1=0, r2=1), (r1=0, r2=2), (r1=1, r2=2)

    GRPO would combine (mean_sum baseline) and produce identical advantages.
    GDPO's per-channel normalization + weighted sum + batch renorm should
    produce distinct advantages preserving relative ordering.
    """
    inputs = _make({
        "is_correct": [[0.0, 0.0, 1.0]],
        "f_term":     [[1.0, 2.0, 2.0]],
    })
    out = gdpo_advantage_fn(inputs, channels=["is_correct", "f_term"], weights=[1.0, 1.0])

    A = out.advantages
    assert A.shape == (1, 3), f"expected shape (1, 3), got {A.shape}"

    # After batch renormalization the mean should be ~0 and std ~1
    assert abs(float(A.mean())) < 1e-4, f"batch mean should be ~0, got {A.mean()}"
    # With only 3 values and torch's default correction, unbiased std is used
    # so we expect std to be 1 (approximately).
    assert abs(float(A.std()) - 1.0) < 0.1, f"batch std should be ~1, got {A.std()}"

    # Relative ordering: rollout 2 (r1=1, r2=2) should have the highest advantage,
    # rollout 0 (r1=0, r2=1) the lowest, rollout 1 (r1=0, r2=2) in between.
    a0, a1, a2 = float(A[0, 0]), float(A[0, 1]), float(A[0, 2])
    assert a2 > a1 > a0, (
        f"expected relative ordering a2 > a1 > a0 (highest reward gets highest advantage) "
        f"got a0={a0}, a1={a1}, a2={a2}"
    )
    print(f"  toy example: A = [{a0:+.3f}, {a1:+.3f}, {a2:+.3f}]  (paper's a2 dominates a1 dominates a0)")


def test_per_channel_normalization_axis():
    """Given 2 groups of 3 rollouts with all-identical channel rewards inside a
    group, per-channel advantages should be zero in that group (std collapses,
    but eps ensures well-defined output near zero)."""
    inputs = _make({
        "is_correct": [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        "f_term":     [[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]],
    })
    out = gdpo_advantage_fn(inputs, channels=["is_correct", "f_term"], weights=[1.0, 1.0])
    # Every group has zero within-group variance on both channels → per-channel
    # advantage is 0 (within numeric precision). Batch-renorm of all-zeros is
    # also 0 (std=0, but 0/(0+eps) = 0).
    A = out.advantages
    assert torch.allclose(A, torch.zeros_like(A), atol=1e-3), f"expected all zeros for degenerate groups, got {A}"
    print(f"  degenerate-group case: A = {A.flatten().tolist()}  (all ~0 as expected)")


def test_weights_change_ordering():
    """With one channel heavily weighted, the advantages should follow that
    channel's ordering even when others disagree."""
    inputs = _make({
        "is_correct": [[1.0, 0.0, 0.0]],  # rollout 0 is uniquely correct
        "f_term":     [[0.0, 0.5, 1.0]],  # rollout 2 has best pacing
    })

    # Case A: correctness dominates → rollout 0 wins
    outA = gdpo_advantage_fn(inputs, channels=["is_correct", "f_term"], weights=[10.0, 1.0])
    a0, a1, a2 = outA.advantages.flatten().tolist()
    assert a0 > a1 and a0 > a2, f"correctness-dominant should pick rollout 0; got {[a0, a1, a2]}"

    # Case B: pacing dominates → rollout 2 wins
    outB = gdpo_advantage_fn(inputs, channels=["is_correct", "f_term"], weights=[1.0, 10.0])
    a0, a1, a2 = outB.advantages.flatten().tolist()
    assert a2 > a0 and a2 > a1, f"pacing-dominant should pick rollout 2; got {[a0, a1, a2]}"
    print(f"  weight sensitivity: OK (correctness-dominant picks r0; pacing-dominant picks r2)")


if __name__ == "__main__":
    print("Running GDPO advantage unit tests...\n")
    test_toy_collapse_example()
    test_per_channel_normalization_axis()
    test_weights_change_ordering()
    print("\nAll tests passed.")
