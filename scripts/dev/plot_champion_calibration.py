"""Calibration plot for the GDPO champion (sym-l22-lr3e5-ext200) alongside base and
the older GRPO windowed cells. Answers: does the champion actually commit at t=T?

Bins the 498 (problem, T) probe records by target_s, then plots mean elapsed-at-
commit vs mean target for each bin. Perfect calibration would sit on the y=x diagonal.
"""
import json, pathlib, re, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
DIR_GRPO = REPO / "analysis/eval_rollouts/prompt_salience/prompt_salience"
DIR_H100 = REPO / "analysis/eval_rollouts/prompt_salience_h100"

ELAPSED_RE = re.compile(r"\[([\d.]+)s elapsed")
BINS = [(1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 41)]


def elapsed_at_commit(text: str):
    body_start = text.find("<|im_start|>assistant")
    body = text[body_start:] if body_start != -1 else text
    ans_pos = body.find("<answer>")
    if ans_pos == -1:
        return None
    prior = [float(m.group(1)) for m in ELAPSED_RE.finditer(body) if m.start() < ans_pos]
    return prior[-1] if prior else 0.0


def load(directory: pathlib.Path, fname: str):
    p = directory / fname
    recs = []
    for line in p.open():
        r = json.loads(line)
        e = elapsed_at_commit(r["completion"])
        if e is not None:
            r["elapsed_at_commit"] = e
            recs.append(r)
    return recs


# (label, dir, fname, color, linestyle, linewidth)
CELLS = [
    ("base model (no RL)",         DIR_GRPO, "base_remaining_budget.jsonl",                 "#27ae60", "-",  2.0),
    ("windowed λ=0.15 (GRPO)",     DIR_GRPO, "windowed-l15_remaining_budget.jsonl",         "#1976D2", ":",  2.0),
    ("windowed λ=0.30 (GRPO)",     DIR_GRPO, "windowed-l30_remaining_budget.jsonl",         "#2E7D32", ":",  2.0),
    ("gdpo sym-l22-ext200 (r5)",   DIR_H100, "gdpo-sym-l22-ext200_remaining_budget.jsonl",  "#ff9800", "--", 2.2),
    ("★ gdpo sym-l22-lr3e5-ext200 (CHAMPION)", DIR_H100,
                                              "gdpo-sym-l22-lr3e5-ext200_remaining_budget.jsonl", "#c62828", "-",  3.4),
]

fig, ax = plt.subplots(figsize=(11, 8))

for label, d, fname, color, ls, lw in CELLS:
    recs = load(d, fname)
    bx, by, bn = [], [], []
    for lo, hi in BINS:
        b = [r for r in recs if lo <= r["target_s"] < hi]
        if not b: continue
        bx.append(st.mean(r["target_s"] for r in b))
        by.append(st.mean(r["elapsed_at_commit"] for r in b))
        bn.append(len(b))
    ax.plot(bx, by, marker="o", ms=9, color=color, lw=lw, ls=ls,
            label=f"{label}  (n={len(recs)})")
    print(f"{label:<45} bins n={bn}")

ax.plot([0, 40], [0, 40], color="#888", lw=1.5, ls=":", label="t = T (ideal calibration)")
ax.set_xlabel("Budget T (s)", fontsize=13)
ax.set_ylabel("Mean elapsed at commit (s)", fontsize=13)
ax.set_xlim(0, 40); ax.set_ylim(0, 40)
ax.set_title("Calibration: does the model commit near T?\n"
             "Champion (red) tracks the diagonal from 1s to 40s — actually obeys the budget",
             fontsize=12)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.grid(alpha=0.25)

out = REPO / "analysis/figures/55_champion_calibration_vs_T.png"
fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"\nwrote {out}")
