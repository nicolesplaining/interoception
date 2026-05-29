"""Smoothed training curves for the Qwen2.5-3B YOLO run (T~U(1,40), 500 steps).

Mirror of plot_long_curves.py — stitches the first attempt's training-step 0..199
metrics onto the resumed run's 200..499 metrics so the x-axis spans the full 500.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENTITY = "singhh5050-stanford-university/interoception"
RUN = "ctrl0-qwen25-3b-u1-40-yolo"
ENV = "ctrl0-u1-40-qwen25-3b-train"
P = f"metrics/{ENV}/"
SERIES = [
    ("correctness c", P + "is_correct",       "#1f77b4"),
    ("f(t, T)",       P + "f_term",            "#2ca02c"),
    ("reward c·f",    f"reward/{ENV}/mean",    "#d62728"),
]
WIN = 10

api = wandb.Api()
runs_chrono = sorted(api.runs(ENTITY, filters={"display_name": RUN}), key=lambda r: r.created_at)
assert len(runs_chrono) >= 2, f"expected ≥2 wandb runs named {RUN!r}, got {len(runs_chrono)}"
run_first, run_resumed = runs_chrono[0], runs_chrono[-1]
RESUME_STEP = 400  # yolo resumed from step_400 (ckpt interval was 100)

def _ordered(run, key):
    pairs = []
    for row in run.scan_history(keys=["_step", key], page_size=10000):
        if row.get("_step") is not None and row.get(key) is not None:
            pairs.append((int(row["_step"]), float(row[key])))
    pairs.sort()
    return [v for _, v in pairs]

def pull(key):
    pre = _ordered(run_first, key)[:RESUME_STEP]
    post = _ordered(run_resumed, key)
    ys = pre + post
    return np.arange(len(ys)), np.array(ys)

def runavg(y, w):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for ax, (label, key, color) in zip(axes, SERIES):
    x, y = pull(key)
    ax.plot(x, y, color=color, alpha=0.22, lw=1.0)
    xs = x[WIN - 1:]
    ax.plot(xs, runavg(y, WIN), color=color, lw=2.6, label=f"{WIN}-step running avg")
    ax.set_ylabel(label, fontsize=10.5)
    ymax = max(0.15, float(np.nanmax(y)) * 1.1)  # yolo is mostly near-zero — tight y-axis
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.text(0.99, 0.06, f"final≈{runavg(y, WIN)[-1]:.3f}", transform=ax.transAxes,
            ha="right", fontsize=9, color=color, fontweight="bold")

axes[-1].set_xlabel("training step", fontsize=11)
fig.suptitle("Qwen2.5-3B — T~U(1, 40), 500 steps", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "analysis/figures/29_yolo_smoothed_curves.png"
fig.savefig(out, dpi=140)
print("wrote", out)
