"""Deeper look at the Qwen3-4B long-500 result.

Pulls all 11 eval-step sample tables from the long run's wandb runs (first attempt
+ resumed) and asks four questions:

  1. Held-out eval trajectory — how do accuracy (c), pacing (f|c=1), and reward
     (c·f) move across the 11 checkpoints?
  2. Base (step 0) vs final (step 500) — per-T-bin accuracy + pacing.
  3. Logistic slope of is_correct ~ T at base vs final — is the trained model
     more or less budget-sensitive than the base model?
  4. Distribution of f(t,T) at base vs final — did training shift the model
     toward in-budget commits, or did it just trade f for c?

Outputs:
  - analysis/figures/30_long_eval_trajectory.png  (3-panel time series)
  - analysis/figures/31_long_base_vs_final.png    (acc-vs-T + f distribution)
  - printed tables + paired stats
"""
from __future__ import annotations
import json, math, pathlib, random, statistics as st

# Reuse the cached eval rollouts produced by analyze_long_extension.py
CACHE = pathlib.Path("analysis/eval_rollouts/long_extension/long-500.jsonl")
BINS = [(1, 9), (9, 17), (17, 25), (25, 33), (33, 40.001)]


def load_recs():
    if not CACHE.exists():
        raise RuntimeError(f"missing {CACHE}; run analyze_long_extension.py first to populate the wandb cache")
    return [json.loads(l) for l in CACHE.open()]


def logistic_slope(rows):
    xs = [r["target_s"] for r in rows]; ys = [r["is_correct"] for r in rows]
    if not xs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mx = sum(xs) / len(xs); sx = (sum((x - mx) ** 2 for x in xs) / len(xs)) ** 0.5
    if sx == 0:
        return 0.0, float("nan"), 0.0, 1.0
    xn = [(x - mx) / sx for x in xs]; b0 = b1 = 0.0
    for _ in range(4000):
        g0 = g1 = 0.0
        for x, y in zip(xn, ys):
            p = 1 / (1 + math.exp(-(b0 + b1 * x))); g0 += p - y; g1 += (p - y) * x
        b0 -= 0.1 * g0 / len(xn); b1 -= 0.1 * g1 / len(xn)
    W = sum((1 / (1 + math.exp(-(b0 + b1 * x)))) * (1 - 1 / (1 + math.exp(-(b0 + b1 * x)))) * x * x for x in xn)
    se = 1 / W ** 0.5; z = b1 / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return b1, se, z, p


def trajectory_table(recs):
    steps = sorted({r["step"] for r in recs})
    out = []
    for s in steps:
        rows = [r for r in recs if r["step"] == s]
        c = st.mean(r["is_correct"] for r in rows)
        correct = [r for r in rows if r["is_correct"]]
        f_given_c = st.mean(r["f"] for r in correct) if correct else float("nan")
        cf = st.mean(r["reward"] for r in rows)
        out.append({"step": s, "n": len(rows), "acc": c, "f_given_c": f_given_c, "reward_cf": cf,
                    "n_correct": len(correct)})
    return out


def per_bin(rows, field):
    out = []
    for lo, hi in BINS:
        b = [r for r in rows if lo <= r["target_s"] < hi]
        vals = [r[field] for r in b if r.get(field) is not None]
        m = st.mean(vals) if vals else float("nan")
        out.append((lo, hi, len(b), len(vals), m))
    return out


def render_trajectory(traj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG = pathlib.Path("analysis/figures"); FIG.mkdir(parents=True, exist_ok=True)
    PINK, GREEN, NAVY = "#C2185B", "#2ca02c", "#2C3E50"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    xs = [t["step"] for t in traj]

    ax = axes[0]
    ys = [t["acc"] for t in traj]
    ax.plot(xs, ys, marker="o", ms=7, color=NAVY, lw=2.2)
    ax.set_title("Held-out accuracy (c)")
    ax.set_ylim(0, max(0.6, max(ys) * 1.15))
    ax.set_ylabel("accuracy")
    ax.grid(alpha=0.25)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 6), textcoords="offset points", fontsize=8, ha="center", color=NAVY)

    ax = axes[1]
    ys = [t["f_given_c"] for t in traj]
    ax.plot(xs, ys, marker="o", ms=7, color=GREEN, lw=2.2)
    ax.set_title("Pacing — mean f(t,T) among correct")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("f(t,T) | correct")
    ax.grid(alpha=0.25)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 6), textcoords="offset points", fontsize=8, ha="center", color=GREEN)

    ax = axes[2]
    ys = [t["reward_cf"] for t in traj]
    ax.plot(xs, ys, marker="o", ms=7, color=PINK, lw=2.2)
    ax.set_title("Reward c·f (held-out)")
    ax.set_ylim(0, max(0.4, max(ys) * 1.15))
    ax.set_ylabel("c·f")
    ax.grid(alpha=0.25)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 6), textcoords="offset points", fontsize=8, ha="center", color=PINK)

    for ax in axes:
        ax.set_xlabel("training step")
    fig.suptitle("Qwen3-4B held-out eval trajectory over 500 training steps (T~U(1,40))", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG / "30_long_eval_trajectory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def render_base_vs_final(recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG = pathlib.Path("analysis/figures"); FIG.mkdir(parents=True, exist_ok=True)
    PINK, NAVY, FAINT = "#C2185B", "#2C3E50", "#C9CDD2"
    base_s = min(r["step"] for r in recs); fin_s = max(r["step"] for r in recs)
    base = [r for r in recs if r["step"] == base_s]
    fin = [r for r in recs if r["step"] == fin_s]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # Left: per-bin accuracy
    ax = axes[0]
    for rows, color, label, marker, ms in (
        (base, FAINT, f"base (step {base_s})", "o", 5),
        (fin,  PINK,  f"final (step {fin_s})", "o", 7),
    ):
        xs, ys, es = [], [], []
        for lo, hi in BINS:
            b = [r for r in rows if lo <= r["target_s"] < hi]
            if not b: continue
            p = sum(r["is_correct"] for r in b) / len(b)
            xs.append(sum(r["target_s"] for r in b) / len(b))
            ys.append(p)
            es.append((p * (1 - p) / len(b)) ** 0.5)
        ls = "--" if "base" in label else "-"
        ax.errorbar(xs, ys, yerr=es, marker=marker, ms=ms, color=color, lw=2.4 if "final" in label else 1.4,
                    capsize=3, ls=ls, label=label)
    ax.set_xlabel("Budget T (s)"); ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 40); ax.set_ylim(0, 0.7)
    ax.set_title("Accuracy vs budget — Qwen3-4B base vs long-500")
    ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.25)

    # Right: distribution of f among correct (base vs final)
    ax = axes[1]
    f_base = [r["f"] for r in base if r["is_correct"]]
    f_fin = [r["f"] for r in fin if r["is_correct"]]
    bins_f = [i / 20 for i in range(21)]
    ax.hist(f_base, bins=bins_f, color=FAINT, edgecolor="white", alpha=0.85,
            label=f"base correct (n={len(f_base)})  mean={st.mean(f_base):.2f}" if f_base else "base correct (n=0)")
    ax.hist(f_fin, bins=bins_f, color=PINK, edgecolor="white", alpha=0.75,
            label=f"final correct (n={len(f_fin)})  mean={st.mean(f_fin):.2f}" if f_fin else "final correct (n=0)")
    ax.set_xlabel("f(t,T) = min(1, T/t)"); ax.set_ylabel("# rollouts")
    ax.set_title("Distribution of pacing factor among correct rollouts")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = FIG / "31_long_base_vs_final.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    recs = load_recs()
    traj = trajectory_table(recs)

    print(f"long-500: {len(recs)} rollouts across {len({r['step'] for r in recs})} eval steps")
    print()
    print("=" * 78)
    print("Held-out eval trajectory (Qwen3-4B, T~U(1,40))")
    print("=" * 78)
    print(f"  {'step':>5}  {'n':>4}  {'n_correct':>10}  {'accuracy':>9}  {'f|correct':>10}  {'reward c·f':>11}")
    for t in traj:
        print(f"  {t['step']:>5}  {t['n']:>4}  {t['n_correct']:>10}  {t['acc']:>9.3f}  {t['f_given_c']:>10.3f}  {t['reward_cf']:>11.3f}")
    print()

    base_s = min(t["step"] for t in traj); fin_s = max(t["step"] for t in traj)
    base = [r for r in recs if r["step"] == base_s]
    fin = [r for r in recs if r["step"] == fin_s]

    print("=" * 78)
    print(f"Base (step {base_s}) vs Final (step {fin_s}) — per-bin accuracy + pacing")
    print("=" * 78)
    print(f"  {'T-bin':<10} {'acc_base':>10}  {'acc_fin':>10}  {'Δacc':>7}    {'f_base':>9}  {'f_fin':>9}  {'Δf':>7}")
    for (lo, hi) in BINS:
        bb = [r for r in base if lo <= r["target_s"] < hi]
        ff = [r for r in fin if lo <= r["target_s"] < hi]
        ab = st.mean(r["is_correct"] for r in bb) if bb else float("nan")
        af = st.mean(r["is_correct"] for r in ff) if ff else float("nan")
        cb = [r["f"] for r in bb if r["is_correct"]]
        cf = [r["f"] for r in ff if r["is_correct"]]
        fb = st.mean(cb) if cb else float("nan")
        ff_ = st.mean(cf) if cf else float("nan")
        print(f"  {lo:>3}-{int(hi):<3}    {ab:>10.3f}  {af:>10.3f}  {af-ab:>+7.3f}    {fb:>9.3f}  {ff_:>9.3f}  {ff_-fb:>+7.3f}")
    print()

    print("=" * 78)
    print("Logistic slope: is_correct ~ standardized T  (base vs final)")
    print("=" * 78)
    for label, rows in (("base", base), ("final", fin)):
        b1, se, z, p = logistic_slope(rows)
        n_correct = sum(r["is_correct"] for r in rows)
        print(f"  {label:5s}  n={len(rows)}  n_correct={n_correct:>3}  slope = {b1:+.3f}  (z={z:+.2f}, p={p:.3f})")
    print()

    # How many correct rollouts landed IN budget (f == 1) vs over budget?
    print("=" * 78)
    print("In-budget rate among correct rollouts (f == 1.0)")
    print("=" * 78)
    for label, rows in (("base", base), ("final", fin)):
        correct = [r for r in rows if r["is_correct"]]
        in_b = sum(1 for r in correct if r["f"] is not None and r["f"] >= 0.999)
        print(f"  {label:5s}  n_correct={len(correct):>3}  in-budget={in_b:>3} ({100*in_b/max(len(correct),1):.1f}%)")
    print()

    render_trajectory(traj)
    render_base_vs_final(recs)


if __name__ == "__main__":
    main()
