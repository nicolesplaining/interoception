"""Mean commit time vs accuracy scatter, one point per model.

Reads every *_remaining_budget.jsonl in the two probe dirs, extracts elapsed-at-
commit from the assistant transcript, and aggregates:
    mean_commit_time = mean over 498 rollouts (fallback 0 if no answer)
    acc              = mean(is_correct)

Lower-left = fast + inaccurate. Upper-right = slow + accurate. The best models
sit upper-left: high accuracy, low commit time.
"""
from __future__ import annotations
import json, pathlib, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
DIR_GRPO = REPO / "analysis/eval_rollouts/prompt_salience/prompt_salience"
DIR_H100 = REPO / "analysis/eval_rollouts/prompt_salience_h100"

ELAPSED_RE = re.compile(r"\[([\d.]+)s elapsed")


def elapsed_at_commit(text: str) -> float | None:
    body_start = text.find("<|im_start|>assistant")
    body = text[body_start:] if body_start != -1 else text
    ans_pos = body.find("<answer>")
    if ans_pos == -1:
        return None
    prior = [float(m.group(1)) for m in ELAPSED_RE.finditer(body) if m.start() < ans_pos]
    return prior[-1] if prior else 0.0


def compute(fname_dir: pathlib.Path, fname: str) -> tuple[float, float, float, int] | None:
    p = fname_dir / fname
    if not p.exists():
        return None
    recs = [json.loads(l) for l in p.open()]
    if not recs:
        return None
    acc = sum(r.get("is_correct", 1 if r.get("reward", 0) > 0 else 0) for r in recs) / len(recs)
    commit_times = []
    for r in recs:
        e = elapsed_at_commit(r["completion"])
        if e is not None:
            commit_times.append(e)
    if not commit_times:
        return None
    mean_ct = sum(commit_times) / len(commit_times)
    commit_rate = len(commit_times) / len(recs)
    return acc, mean_ct, commit_rate, len(recs)


# (label, dir, fname, color, marker)
CELLS = [
    # Baselines (GRPO, Modal)
    ("base",                   DIR_GRPO, "base_remaining_budget.jsonl",                 "#27ae60", "s"),
    ("v2-flat λ=0.15",         DIR_GRPO, "long-additive-v2-l15_remaining_budget.jsonl", "#1f77b4", "s"),
    ("v2-flat λ=0.30",         DIR_GRPO, "long-additive-v2-l30_remaining_budget.jsonl", "#d62728", "s"),
    ("windowed λ=0.15",        DIR_GRPO, "windowed-l15_remaining_budget.jsonl",         "#1976D2", "^"),
    ("windowed λ=0.30",        DIR_GRPO, "windowed-l30_remaining_budget.jsonl",         "#2E7D32", "^"),
    ("windowed λ=0.50",        DIR_GRPO, "windowed-l50_remaining_budget.jsonl",         "#8E24AA", "^"),
    # GDPO short (asymmetric window, 100 steps)
    ("gdpo l15",               DIR_H100, "gdpo-l15_remaining_budget.jsonl",             "#ff9900", "o"),
    ("gdpo l25",               DIR_H100, "gdpo-l25_remaining_budget.jsonl",             "#ffa500", "o"),
    ("gdpo l30",               DIR_H100, "gdpo-l30_remaining_budget.jsonl",             "#ff8c00", "o"),
    ("gdpo l40",               DIR_H100, "gdpo-l40_remaining_budget.jsonl",             "#ff7f00", "o"),
    ("gdpo l50",               DIR_H100, "gdpo-l50_remaining_budget.jsonl",             "#ff6600", "o"),
    ("gdpo l100",              DIR_H100, "gdpo-l100_remaining_budget.jsonl",            "#e65100", "o"),
    # GDPO symmetric window
    ("gdpo sym-l25",           DIR_H100, "gdpo-sym-l25_remaining_budget.jsonl",         "#ba68c8", "D"),
    ("gdpo sym-l50",           DIR_H100, "gdpo-sym-l50_remaining_budget.jsonl",         "#9c27b0", "D"),
    ("gdpo sym-l75",           DIR_H100, "gdpo-sym-l75_remaining_budget.jsonl",         "#7b1fa2", "D"),
    ("gdpo tight-l30",         DIR_H100, "gdpo-tight-l30_remaining_budget.jsonl",       "#4a148c", "D"),
    # GDPO extended asymmetric
    ("gdpo l15-ext200",        DIR_H100, "gdpo-l15-ext200_remaining_budget.jsonl",      "#ff5722", "P"),
    ("gdpo l50-ext200",        DIR_H100, "gdpo-l50-ext200_remaining_budget.jsonl",      "#bf360c", "P"),
    # Round 4 (winners are highlighted)
    ("gdpo l10-ext200",        DIR_H100, "gdpo-l10-ext200_remaining_budget.jsonl",      "#c2185b", "*"),
    ("gdpo l20-ext200",        DIR_H100, "gdpo-l20-ext200_remaining_budget.jsonl",      "#e91e63", "*"),
    ("gdpo l30-ext200",        DIR_H100, "gdpo-l30-ext200_remaining_budget.jsonl",      "#ad1457", "*"),
    ("gdpo sym-l25-ext200",    DIR_H100, "gdpo-sym-l25-ext200_remaining_budget.jsonl",  "#1a237e", "*"),
    ("gdpo sym-l30-ext200",    DIR_H100, "gdpo-sym-l30-ext200_remaining_budget.jsonl",  "#0d47a1", "*"),
    ("gdpo l15-ext300",        DIR_H100, "gdpo-l15-ext300_remaining_budget.jsonl",      "#01579b", "*"),
    # Rounds 5-9
    ("gdpo sym-l17-ext200",       DIR_H100, "gdpo-sym-l17-ext200_remaining_budget.jsonl",        "#7986cb", "*"),
    ("gdpo sym-l20-ext200",       DIR_H100, "gdpo-sym-l20-ext200_remaining_budget.jsonl",        "#5c6bc0", "*"),
    ("gdpo sym-l22-ext200 ★",     DIR_H100, "gdpo-sym-l22-ext200_remaining_budget.jsonl",        "#e64a19", "X"),
    ("gdpo sym-l25-ext300",       DIR_H100, "gdpo-sym-l25-ext300_remaining_budget.jsonl",        "#283593", "*"),
    ("gdpo sym-l30-ext300",       DIR_H100, "gdpo-sym-l30-ext300_remaining_budget.jsonl",        "#1a237e", "*"),
    ("gdpo sym-l25-wide-ext200",  DIR_H100, "gdpo-sym-l25-wide-ext200_remaining_budget.jsonl",   "#3949ab", "*"),
    ("gdpo sym-l22-lr3e5-ext200 ★★★", DIR_H100, "gdpo-sym-l22-lr3e5-ext200_remaining_budget.jsonl", "#c62828", "X"),
    ("gdpo sym-l22-lr1e5-ext200", DIR_H100, "gdpo-sym-l22-lr1e5-ext200_remaining_budget.jsonl",  "#d84315", "P"),
    ("gdpo sym-l22-lr3e5-ext300", DIR_H100, "gdpo-sym-l22-lr3e5-ext300_remaining_budget.jsonl",  "#bf360c", "P"),
    ("gdpo sym-l22-lr3e5-r16",    DIR_H100, "gdpo-sym-l22-lr3e5-r16-ext200_remaining_budget.jsonl", "#ff7043", "P"),
    ("gdpo sym-l23-lr3e5-ext200", DIR_H100, "gdpo-sym-l23-lr3e5-ext200_remaining_budget.jsonl",  "#ff5722", "P"),
]

fig, ax = plt.subplots(figsize=(12, 8))

# Draw an isoperformance reference: "budget-tracking mid-band" (mean T ≈ 20 = midpoint of [1, 40])
ax.axvline(20.0, color="grey", linestyle=":", alpha=0.4, linewidth=0.8, label="mean T ≈ 20 (midpoint)")

rows = []
for label, d, fname, color, marker in CELLS:
    r = compute(d, fname)
    if r is None:
        continue
    acc, mct, crate, n = r
    rows.append((label, acc, mct, crate, n))
    # Star = highlight; use larger size for "★" markers
    size = 240 if "★" in label else 110
    edgewidth = 2.0 if "★" in label else 0.6
    ax.scatter(mct, acc, s=size, marker=marker, c=color, edgecolors="black",
               linewidths=edgewidth, alpha=0.85, zorder=4 if "★" in label else 3)
    ax.annotate(label, (mct, acc), fontsize=8, xytext=(5, 5),
                textcoords="offset points", alpha=0.85)

# Print table
print(f"{'cell':<28} {'acc':>6} {'mean_ct':>8} {'commit_%':>9} {'n':>4}")
print("-" * 60)
for label, acc, mct, crate, n in sorted(rows, key=lambda x: -x[1]):
    print(f"{label:<28} {acc:>6.3f} {mct:>8.2f}s {crate*100:>8.1f}% {n:>4}")

ax.set_xlabel("Mean commit time (s)")
ax.set_ylabel("Accuracy")
ax.set_title("Mean commit time vs accuracy — every cell we've evaluated\n"
             "(bars are single-point aggregates over 498 uniform-T eval rollouts)")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
outp = REPO / "analysis/figures/54_commit_time_vs_acc.png"
plt.savefig(outp, dpi=140)
print(f"\nwrote {outp}")
