"""Paired held-out comparison: long-500 vs long-1000.

Headline test of Kanishk's "train longer = c keeps climbing + f starts moving"
thesis. Both runs:
  - same model (Qwen3-4B-Instruct-2507)
  - same env config (T~U(1,40), max_turns=16, hyperbolic c·f reward)
  - same eval (498 problems, dataset_seed=777 → same (problem, T) pairs)
  - differ only in max_steps (500 vs 1000)

attempt_bonus=0 in both configs means:
  reward > 0 <=> is_correct
  reward     == c*f among correct
So we recover (is_correct, f) from each eval-table row without re-scoring.

Outputs:
  - analysis/figures/35_long1k_vs_long500_acc_vs_T.png
  - analysis/figures/36_long1k_vs_long500_paired.png
  - cached per-rollout records at analysis/eval_rollouts/long500_vs_long1k/
"""
from __future__ import annotations
import json, math, os, pathlib, random, tempfile, statistics as st

ENTITY = "singhh5050-stanford-university/interoception"
RUNS = {
    "long-500":  "ctrl0-qwen3-4b-u1-40-long",   # 500-step run (final eval at step 500)
    "long-1000": "ctrl0-qwen3-4b-u1-40-long1k", # 1000-step run (final eval at step 1000)
}
EVAL_SEED, T_LO, T_HI = 777, 1.0, 40.0
CACHE = pathlib.Path("analysis/eval_rollouts/long500_vs_long1k")
BINS = [(1, 9), (9, 17), (17, 25), (25, 33), (33, 40.001)]


def target_s_for(example_id: int) -> float:
    rng = random.Random(EVAL_SEED ^ (example_id * 2654435761 & 0xFFFFFFFF))
    return rng.uniform(T_LO, T_HI)


def pull(cond: str, wandb_name: str) -> list[dict]:
    """Pull all eval-sample-table rows for the named run(s) → per-rollout records.
    Caches to avoid re-downloading. MERGES rows across multiple wandb sessions
    (the long1k run had 2 — first attempt was killed mid-train, resumed finished
    cleanly). Dedupes by (step, example_id), preferring the later wandb session."""
    out = CACHE / f"{cond}.jsonl"
    if out.exists():
        return [json.loads(l) for l in out.open()]
    import wandb
    api = wandb.Api()
    runs = sorted(api.runs(ENTITY, filters={"display_name": wandb_name}),
                  key=lambda r: r.created_at)
    if not runs:
        raise RuntimeError(f"no wandb runs named {wandb_name!r}")
    d = tempfile.mkdtemp()
    seen = {}  # (step, example_id) -> rec; later wandb runs overwrite
    for run in runs:
        tbls = [f for f in run.files() if "table/eval" in f.name and f.name.endswith(".json")]
        for f in tbls:
            f.download(root=d, replace=True)
            j = json.load(open(os.path.join(d, f.name)))
            c = j["columns"]
            si, ri, ei = c.index("step"), c.index("reward"), c.index("example_id")
            for row in j["data"]:
                ex = int(row[ei]); rew = float(row[ri]); s = int(row[si])
                seen[(s, ex)] = {"condition": cond, "wandb_run": run.id, "step": s,
                                  "example_id": ex, "target_s": target_s_for(ex),
                                  "reward": rew, "is_correct": 1 if rew > 0 else 0,
                                  "f": rew if rew > 0 else None}
    recs = list(seen.values())
    CACHE.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return recs


def logistic_slope(rows):
    xs = [r["target_s"] for r in rows]; ys = [r["is_correct"] for r in rows]
    if not xs: return float("nan"), float("nan"), float("nan"), float("nan")
    mx = sum(xs) / len(xs); sx = (sum((x - mx) ** 2 for x in xs) / len(xs)) ** 0.5
    if sx == 0: return 0.0, float("nan"), 0.0, 1.0
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


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0: return float("nan")
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p)


def step_rows(recs, which):
    if not recs: return [], None
    s = min(r["step"] for r in recs) if which == "base" else max(r["step"] for r in recs)
    return [r for r in recs if r["step"] == s], s


def render_acc_vs_T(data):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = pathlib.Path("analysis/figures"); FIG.mkdir(parents=True, exist_ok=True)
    PINK, NAVY, FAINT = "#C2185B", "#2C3E50", "#C9CDD2"

    def bpoints(rows):
        xs, ys, es = [], [], []
        for lo, hi in BINS:
            b = [r for r in rows if lo <= r["target_s"] < hi]
            if not b: continue
            p = sum(r["is_correct"] for r in b) / len(b)
            xs.append(sum(r["target_s"] for r in b) / len(b)); ys.append(p)
            es.append((p * (1 - p) / len(b)) ** 0.5)
        return xs, ys, es

    base500, s_base = step_rows(data["long-500"], "base")
    fin500, s500 = step_rows(data["long-500"], "final")
    fin1k, s1k = step_rows(data["long-1000"], "final")

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    x, y, e = bpoints(base500)
    ax.plot(x, y, marker="o", ms=5, color=FAINT, lw=1.4, ls="--",
            label=f"base (step {s_base}, untrained)")
    x, y, e = bpoints(fin500)
    ax.errorbar(x, y, yerr=e, marker="s", ms=6, color=NAVY, lw=1.9, capsize=3,
                label=f"long-500 — final (step {s500})")
    x, y, e = bpoints(fin1k)
    ax.errorbar(x, y, yerr=e, marker="o", ms=7, color=PINK, lw=2.6, capsize=3,
                label=f"long-1000 — final (step {s1k})")
    ax.set_xlabel("Budget T (s)"); ax.set_ylabel("Accuracy (held-out eval)")
    ax.set_xlim(0, 40); ax.set_ylim(0, 0.75)
    ax.set_title("long-500 vs long-1000: accuracy vs budget — Qwen3-4B, T~U(1,40)")
    ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.grid(alpha=0.25)
    out = FIG / "35_long1k_vs_long500_acc_vs_T.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def render_paired(data500, data1k, paired):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = pathlib.Path("analysis/figures"); FIG.mkdir(parents=True, exist_ok=True)
    PINK, NAVY, GRAY = "#C2185B", "#2C3E50", "#8E9BA8"

    flip_w2c = sum(1 for p in paired if p["c500"] == 0 and p["c1k"] == 1)
    flip_c2w = sum(1 for p in paired if p["c500"] == 1 and p["c1k"] == 0)
    stay_c = sum(1 for p in paired if p["c500"] == 1 and p["c1k"] == 1)
    stay_w = sum(1 for p in paired if p["c500"] == 0 and p["c1k"] == 0)
    p_mc = mcnemar_exact(flip_w2c, flip_c2w)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    cats = ["wrong→correct\n(long-1000 helped)", "correct→wrong\n(long-1000 hurt)",
            "both correct", "both wrong"]
    vals = [flip_w2c, flip_c2w, stay_c, stay_w]
    colors = [PINK, NAVY, "#888", GRAY]
    bars = ax.bar(cats, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, str(v),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("# problems")
    ax.set_title(f"Paired outcomes per problem (n={len(paired)})\n"
                 f"McNemar (mid-p) on discordant pairs: p={p_mc:.3g}")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    both_correct = [p for p in paired if p["c500"] == 1 and p["c1k"] == 1]
    f500 = [p["f500"] for p in both_correct]
    f1k = [p["f1k"] for p in both_correct]
    if f500:
        ax.scatter(f500, f1k, s=20, color=PINK, alpha=0.55, edgecolor="white", lw=0.5)
        ax.plot([0, 1], [0, 1], color="#999", lw=1.2, ls="--", label="y=x (no change)")
        df = [b - a for a, b in zip(f500, f1k)]
        mean_d = st.mean(df) if df else 0.0
        n = len(df)
        if n > 1:
            sd = (sum((x - mean_d)**2 for x in df) / (n - 1)) ** 0.5
            t = mean_d / (sd / n**0.5) if sd > 0 else 0.0
            ax.set_title(f"f(t,T) among both-correct (n={n})\n"
                         f"Δf mean = {mean_d:+.3f}   (paired t = {t:+.2f})")
        ax.set_xlabel("long-500  f(t,T)"); ax.set_ylabel("long-1000  f(t,T)")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.grid(alpha=0.25)
    out = FIG / "36_long1k_vs_long500_paired.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    data = {c: pull(c, nm) for c, nm in RUNS.items()}
    for c, recs in data.items():
        steps = sorted(set(r["step"] for r in recs))
        print(f"{c}: {len(recs)} rollouts across {len(steps)} eval steps "
              f"(min={min(steps)}, max={max(steps)})")
    print()

    fin500, s500 = step_rows(data["long-500"], "final")
    fin1k, s1k = step_rows(data["long-1000"], "final")
    by500 = {r["example_id"]: r for r in fin500}
    by1k = {r["example_id"]: r for r in fin1k}
    paired = []
    for ex in sorted(set(by500) & set(by1k)):
        a, b = by500[ex], by1k[ex]
        paired.append({"example_id": ex, "target_s": a["target_s"],
                       "c500": a["is_correct"], "c1k": b["is_correct"],
                       "f500": a.get("f"),     "f1k":  b.get("f")})
    print(f"paired n = {len(paired)} (long-500 final step={s500}, long-1000 final step={s1k})\n")

    print("=" * 82)
    print("Accuracy by budget bin — long-500 vs long-1000 (final eval)")
    print("=" * 82)
    print("  T-bin       n     long-500    long-1000   Δaccuracy")
    for lo, hi in BINS:
        b = [p for p in paired if lo <= p["target_s"] < hi]
        if not b: continue
        a500 = st.mean([p["c500"] for p in b])
        a1k  = st.mean([p["c1k"]  for p in b])
        print(f"  {lo:>3}-{int(hi):<3}    {len(b):>4}      {a500:.3f}       {a1k:.3f}     {a1k-a500:+.3f}")
    overall500 = st.mean([p["c500"] for p in paired])
    overall1k  = st.mean([p["c1k"]  for p in paired])
    print(f"  overall   {len(paired):>4}      {overall500:.3f}       {overall1k:.3f}     {overall1k-overall500:+.3f}\n")

    print("=" * 82)
    print("Paired flips (long-500 -> long-1000)")
    print("=" * 82)
    flip_w2c = sum(1 for p in paired if p["c500"] == 0 and p["c1k"] == 1)
    flip_c2w = sum(1 for p in paired if p["c500"] == 1 and p["c1k"] == 0)
    stay_c = sum(1 for p in paired if p["c500"] == 1 and p["c1k"] == 1)
    stay_w = sum(1 for p in paired if p["c500"] == 0 and p["c1k"] == 0)
    print(f"  wrong → correct  (long-1000 helped): {flip_w2c}")
    print(f"  correct → wrong  (long-1000 hurt):   {flip_c2w}")
    print(f"  both correct (stayed correct):       {stay_c}")
    print(f"  both wrong  (stayed wrong):          {stay_w}")
    print(f"  McNemar exact (mid-p, two-sided):    p = {mcnemar_exact(flip_w2c, flip_c2w):.3g}\n")

    print("=" * 82)
    print("Logistic slope: is_correct ~ standardized T (per condition)")
    print("=" * 82)
    for label, rows in (("long-500 final", fin500), ("long-1000 final", fin1k)):
        b1, se, z, p = logistic_slope(rows)
        print(f"  {label:18s}  slope = {b1:+.3f}  (z={z:+.2f}, p={p:.3f})")
    print()

    print("=" * 82)
    print("Pacing — mean f(t,T) among CORRECT rollouts, by T-bin")
    print("=" * 82)
    for label, rows in (("long-500", fin500), ("long-1000", fin1k)):
        for lo, hi in BINS:
            correct = [r for r in rows if r["is_correct"] and lo <= r["target_s"] < hi]
            if correct:
                mf = st.mean(r["f"] for r in correct)
                print(f"  {label:10s}  T {lo:>3}-{int(hi):<3}  n_correct={len(correct):>3}  mean f={mf:.3f}")
            else:
                print(f"  {label:10s}  T {lo:>3}-{int(hi):<3}  n_correct=0")
        print()

    render_acc_vs_T(data)
    render_paired(data["long-500"], data["long-1000"], paired)


if __name__ == "__main__":
    main()
