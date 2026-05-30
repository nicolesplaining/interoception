"""Sanity check on the f trajectory for the Qwen3-4B long-500 run.

Two questions:
  Q1. Is the training-set f curve really showing improvement, or is the stitched
      seam at step 200 hiding a discontinuity? Compare run-1's f_term in
      [195, 280] (just before the budget kill) against run-2's f_term in [200, 280]
      (just after the resume from step_200). If they're close, the stitch is safe.
  Q2. Is the held-out f|c trajectory really flat after step 100? Print the 11
      eval points alongside a 3-point rolling mean. Also compute two other
      pacing signals from the held-out data: (a) mean f over ALL rollouts (not
      just correct, matching the training metric), and (b) fraction of correct
      rollouts that land in-budget.
"""
from __future__ import annotations
import json, pathlib, statistics as st
import wandb

ENTITY = "singhh5050-stanford-university/interoception"
RUN_NAME = "ctrl0-qwen3-4b-u1-40-long"
ENV = "ctrl0-u1-40-long-qwen3-4b-train"
F_KEY = f"metrics/{ENV}/f_term"
C_KEY = f"metrics/{ENV}/is_correct"
EVAL_CACHE = pathlib.Path("analysis/eval_rollouts/long_extension/long-500.jsonl")


def ordered(run, key):
    pairs = []
    for row in run.scan_history(keys=["_step", key], page_size=10000):
        if row.get("_step") is not None and row.get(key) is not None:
            pairs.append((int(row["_step"]), float(row[key])))
    pairs.sort()
    return [v for _, v in pairs]


def mean(xs):
    return st.mean(xs) if xs else float("nan")


# --- Q1: training-set seam check ---

api = wandb.Api()
runs = sorted(api.runs(ENTITY, filters={"display_name": RUN_NAME}), key=lambda r: r.created_at)
assert len(runs) >= 2
run1, run2 = runs[0], runs[-1]

f1 = ordered(run1, F_KEY)
f2 = ordered(run2, F_KEY)
c1 = ordered(run1, C_KEY)
c2 = ordered(run2, C_KEY)

print(f"run1 (first attempt):  {len(f1)} f_term points")
print(f"run2 (resumed):        {len(f2)} f_term points")
print()
print("=" * 78)
print("Q1. Training-set f_term across the resume seam")
print("=" * 78)
print(f"  metric points correspond to TRAINING STEPS (one log per step).")
print(f"  run1 length = {len(f1)} ~ training steps 0..{len(f1)-1}")
print(f"  run2 length = {len(f2)} ~ training steps 200..{200+len(f2)-1} (resumed from step_200)")
print()
print(f"  run1 last 30 steps  (steps {max(0, len(f1)-30)}..{len(f1)-1}):  mean f = {mean(f1[-30:]):.3f}")
print(f"  run1 first 10 steps:                                          mean f = {mean(f1[:10]):.3f}")
print(f"  run1 mean over its FULL trajectory:                           mean f = {mean(f1):.3f}")
print()
print(f"  run2 first 10 steps  (~training steps 200..209):              mean f = {mean(f2[:10]):.3f}")
print(f"  run2 last 10 steps   (~training steps {200+len(f2)-10}..{200+len(f2)-1}):     mean f = {mean(f2[-10:]):.3f}")
print(f"  run2 mean over its FULL trajectory:                           mean f = {mean(f2):.3f}")
print()
# The cleanest seam check: run1's f_term around its step-200 vs run2's first values
# (both should reflect roughly the same model state — the step_200 checkpoint).
print(f"  run1 f_term at steps 195..205 (around the eventual resume point):  mean = {mean(f1[195:206]):.3f}  (n={len(f1[195:206])})")
print(f"  run2 f_term at its first 10 steps (post-resume from step_200):     mean = {mean(f2[:10]):.3f}")
print()
print(f"  → If these are close, the stitched curve is safe. (Both are post-step_200 model rollouts.)")
print()

# --- Q2: held-out f trajectory — multiple signals ---

print("=" * 78)
print("Q2. Held-out (eval-set) pacing trajectory — 3 signals over 11 checkpoints")
print("=" * 78)

recs = [json.loads(l) for l in EVAL_CACHE.open()]
steps = sorted({r["step"] for r in recs})

print(f"  {'step':>5}  {'n':>4}  {'acc':>6}  {'f|correct':>10}  {'f|all_rollouts':>15}  {'in-budget|c':>12}")
trajectory = []
for s in steps:
    rows = [r for r in recs if r["step"] == s]
    correct = [r for r in rows if r["is_correct"]]
    acc = mean(r["is_correct"] for r in rows)
    fc = mean(r["f"] for r in correct)
    # f for ALL rollouts: f is None for non-correct in our cache. For wrong rollouts in this
    # env, f is well-defined (computed from elapsed); we just dropped it because reward=0
    # already conveys is_correct. Approximate via reward (==c*f among correct, 0 among wrong).
    # Note: this is f WEIGHTED BY correctness; it's the c*f trajectory, not pure f.
    # For a closer apples-to-apples with training f_term we'd need the raw f, but we don't have it.
    cf = mean(r["reward"] for r in rows)
    in_budget = sum(1 for r in correct if r["f"] is not None and r["f"] >= 0.999) / max(len(correct), 1)
    print(f"  {s:>5}  {len(rows):>4}  {acc:>6.3f}  {fc:>10.3f}  {cf:>15.3f}  {in_budget:>12.3f}")
    trajectory.append({"step": s, "acc": acc, "fc": fc, "in_budget": in_budget})

print()
print("  caveat: 'f|all_rollouts' here is actually c·f from reward (since our cache drops f for")
print("  wrong rollouts). For a direct match with training f_term we'd need raw f from a fresh")
print("  pull — but the c·f column already covers the headline reward trend.")
print()

# Rolling 3-point smoothing of f|correct to look past noise.
print("=" * 78)
print("Smoothed (3-step) held-out f|correct trajectory")
print("=" * 78)
fc_series = [t["fc"] for t in trajectory]
for i in range(len(fc_series)):
    lo = max(0, i - 1); hi = min(len(fc_series), i + 2)
    sm = mean(fc_series[lo:hi])
    print(f"  step {trajectory[i]['step']:>3}:  raw f|c = {fc_series[i]:.3f}   3-step smoothed = {sm:.3f}")
print()

# Compare the early plateau (steps 0..152) to late plateau (steps 302..500)
early = [t["fc"] for t in trajectory if t["step"] <= 152 and t["step"] > 0]
late  = [t["fc"] for t in trajectory if t["step"] >= 302]
print(f"  early plateau (steps 52..152, n={len(early)}):   mean f|c = {mean(early):.3f}")
print(f"  late  plateau (steps 302..500, n={len(late)}):   mean f|c = {mean(late):.3f}")
print(f"  Δ = {mean(late) - mean(early):+.3f}")
