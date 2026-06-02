"""Smoothed training curves for the long1k run (Qwen3-4B, 1000 steps).

Mirror of plot_long_curves.py — the long1k run was killed by Modal preemption
around training step 801 and auto-resumed; we stitch the first attempt's
steps 0..800 onto the resumed attempt's 800..1000 by deduping on the
prime-rl `step` field (the real training step, monotonic across sessions).
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENTITY = "singhh5050-stanford-university/interoception"
RUN = "ctrl0-qwen3-4b-u1-40-long1k"
ENV = "ctrl0-u1-40-long1k-qwen3-4b-train"
P = f"metrics/{ENV}/"
SERIES = [
    ("correctness c", P + "is_correct",       "#1f77b4"),
    ("f(t, T)",       P + "f_term",            "#2ca02c"),
    ("reward c·f",    f"reward/{ENV}/mean",    "#d62728"),
]
WIN = 20  # wider smoothing for the 1000-step trajectory

api = wandb.Api()
runs_chrono = sorted(api.runs(ENTITY, filters={"display_name": RUN}), key=lambda r: r.created_at)
assert len(runs_chrono) >= 1, f"no wandb runs named {RUN!r}"


def pull(metric_key):
    """Pull (training_step, value) tuples across all wandb sessions, dedupe by
    training step (later sessions overwrite earlier — they share the same global
    step counter since prime-rl resume restores progress.step)."""
    # Try the `step` field first (per-step metric, monotonic across resumes);
    # for `reward/.../mean` it's logged with `step`, for `metrics/.../*` it
    # uses `_step` (per-log) and we have to fall back to ordering by _step
    # then concatenating segments.
    by_step = {}
    for run in runs_chrono:
        for row in run.scan_history(keys=["step", metric_key], page_size=10000):
            s = row.get("step"); v = row.get(metric_key)
            if s is not None and v is not None:
                by_step[int(s)] = float(v)
    if by_step:
        items = sorted(by_step.items())
        xs = np.array([s for s, _ in items])
        ys = np.array([v for _, v in items])
        return xs, ys

    # Fallback: metrics/ namespace doesn't include `step` — stitch by
    # ordering segments and taking len from each.
    seqs = []
    for run in runs_chrono:
        pairs = []
        for row in run.scan_history(keys=["_step", metric_key], page_size=10000):
            if row.get("_step") is not None and row.get(metric_key) is not None:
                pairs.append((int(row["_step"]), float(row[metric_key])))
        pairs.sort()
        seqs.append([v for _, v in pairs])
    # Use the first session's first N (where N is the resume point), then the
    # resumed session's full trajectory. For long1k we infer the resume seam
    # from the lengths — session 1 had ~802 training-step metric rows (matches
    # max_step=801 we saw in wandb).
    if len(seqs) >= 2:
        N_RESUME = min(len(seqs[0]), 800)  # session 1's first ≤800 points
        ys = seqs[0][:N_RESUME] + seqs[-1]
    else:
        ys = seqs[0]
    return np.arange(len(ys)), np.array(ys)


def runavg(y, w):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for ax, (label, key, color) in zip(axes, SERIES):
    x, y = pull(key)
    ax.plot(x, y, color=color, alpha=0.18, lw=0.9)
    if len(y) >= WIN:
        xs = x[WIN - 1:]
        ax.plot(xs, runavg(y, WIN), color=color, lw=2.6, label=f"{WIN}-step running avg")
    ax.set_ylabel(label, fontsize=10.5)
    ax.set_ylim(0, max(0.7, float(np.nanmax(y)) * 1.1))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.text(0.99, 0.06, f"final≈{runavg(y, WIN)[-1]:.2f}" if len(y) >= WIN else f"final≈{y[-1]:.2f}",
            transform=ax.transAxes, ha="right", fontsize=9, color=color, fontweight="bold")

axes[-1].set_xlabel("training step", fontsize=11)
fig.suptitle("Qwen3-4B — T~U(1, 40), 1000-step run", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "analysis/figures/34_long1k_smoothed_curves.png"
fig.savefig(out, dpi=140)
print("wrote", out)
