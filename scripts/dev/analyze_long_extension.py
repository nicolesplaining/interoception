"""Long-200 vs Long-500 paired comparison: did training longer help?

Both runs share the same env config except max_steps (200 vs 500), and both
evaluate on the same held-out set with dataset_seed=777, target_s_min=1,
target_s_max=40 -> (example_id, T) pairs are byte-identical across the two,
so per-rollout paired analysis is valid.

Uses the analyze_controls.py trick (attempt_bonus=0 in both configs):
    reward > 0  <=>  is_correct
    reward      ==  c * f(t, T)  among correct rollouts
So is_correct and the pacing factor f come straight from the reward column
of each run's eval sample table.

The 500-step run is actually TWO wandb runs (a first attempt the budget
killed, plus the resumed attempt that ran to completion). We merge their
eval tables — the resumed run is the one that produced the step_500 eval.

Outputs:
  - analysis/figures/27_long_extension_acc_vs_T.png
  - analysis/figures/28_long_extension_paired.png
  - printed table + paired stats
"""
from __future__ import annotations
import json, math, os, pathlib, random, tempfile, statistics as st

ENTITY = "singhh5050-stanford-university/interoception"
RUNS = {
    "long-200": "ctrl0-qwen3-4b-u1-40",       # original 200-step run (single wandb run)
    "long-500": "ctrl0-qwen3-4b-u1-40-long",  # 500-step extension (first + resumed wandb runs)
}
EVAL_SEED, T_LO, T_HI = 777, 1.0, 40.0
CACHE = pathlib.Path("analysis/eval_rollouts/long_extension")
BINS = [(1, 9), (9, 17), (17, 25), (25, 33), (33, 40.001)]


def target_s_for(example_id: int) -> float:
    """Same formula the env uses to assign T per dataset row (uniform branch)."""
    rng = random.Random(EVAL_SEED ^ (example_id * 2654435761 & 0xFFFFFFFF))
    return rng.uniform(T_LO, T_HI)


def pull(cond: str, wandb_name: str) -> list[dict]:
    """Pull all eval-sample-table rows for the named run(s) → per-rollout records (cached).
    For the long-500 condition, MERGES rows from all wandb runs sharing the display name
    (first attempt + resumed) so the step=500 eval table is included."""
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
    recs = []
    for run in runs:
        tbls = [f for f in run.files() if "table/eval" in f.name and f.name.endswith(".json")]
        for f in tbls:
            f.download(root=d, replace=True)
            j = json.load(open(os.path.join(d, f.name)))
            c = j["columns"]
            si, ri, ei = c.index("step"), c.index("reward"), c.index("example_id")
            for row in j["data"]:
                ex = int(row[ei]); rew = float(row[ri])
                recs.append({"condition": cond, "wandb_run": run.id, "step": int(row[si]),
                             "example_id": ex, "target_s": target_s_for(ex), "reward": rew,
                             "is_correct": 1 if rew > 0 else 0,
                             "f": rew if rew > 0 else None})
    CACHE.mkdir(parents=True, exist_ok=True)
    # de-duplicate: a problem×step appearing in both wandb runs (e.g. the resumed run
    # re-evaluated step_200 on entry) — prefer the later (highest created_at) wandb run.
    seen = {}
    for r in recs:
        seen[(r["step"], r["example_id"])] = r  # later runs overwrite
    recs = list(seen.values())
    with out.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return recs


def logistic_slope(rows):
    xs = [r["target_s"] for r in rows]; ys = [r["is_correct"] for r in rows]
    mx = sum(xs) / len(xs); sx = (sum((x - mx) ** 2 for x in xs) / len(xs)) ** 0.5
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


def binned_mean(rows, field):
    out = []
    for lo, hi in BINS:
        b = [r for r in rows if lo <= r["target_s"] < hi]
        vals = [r[field] for r in b if r.get(field) is not None]
        out.append((lo, hi, len(b), st.mean(vals) if vals else float("nan")))
    return out


def step_rows(recs, which):
    if not recs:
        return [], None
    if which == "base":
        s = min(r["step"] for r in recs)
    else:
        s = max(r["step"] for r in recs)
    return [r for r in recs if r["step"] == s], s


def mcnemar_exact(b: int, c: int):
    """Exact mid-p McNemar test on (b, c) discordant pairs.
    H0: P(flip wrong->correct) = P(flip correct->wrong). Returns p-value."""
    n = b + c
    if n == 0:
        return float("nan")
    # exact binomial two-sided
    k = min(b, c)
    # binomial pmf with p=0.5
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p)


def render_acc_vs_T(data200, data500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG = pathlib.Path("analysis/figures"); FIG.mkdir(parents=True, exist_ok=True)
    PINK, NAVY, FAINT = "#C2185B", "#2C3E50", "#C9CDD2"

    def bpoints(rows):
        xs, ys, es = [], [], []
        for lo, hi in BINS:
            b = [r for r in rows if lo <= r["target_s"] < hi]
            if not b: continue
            p = sum(r["is_correct"] for r in b) / len(b)
            xs.append(sum(r["target_s"] for r in b) / len(b))
            ys.append(p)
            es.append((p * (1 - p) / len(b)) ** 0.5)
        return xs, ys, es

    base200, s_base = step_rows(data200, "base")
    fin200, s_fin200 = step_rows(data200, "final")
    fin500, s_fin500 = step_rows(data500, "final")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    x, y, e = bpoints(base200)
    ax.plot(x, y, marker="o", ms=5, color=FAINT, lw=1.4, ls="--",
            label=f"base (step {s_base}, untrained)")
    x, y, e = bpoints(fin200)
    ax.errorbar(x, y, yerr=e, marker="s", ms=6, color=NAVY, lw=1.9, capsize=3,
                label=f"long-200 — final (step {s_fin200})")
    x, y, e = bpoints(fin500)
    ax.errorbar(x, y, yerr=e, marker="o", ms=7, color=PINK, lw=2.6, capsize=3,
                label=f"long-500 — final (step {s_fin500})")
    ax.set_xlabel("Budget T (s)"); ax.set_ylabel("Accuracy (held-out eval)")
    ax.set_xlim(0, 40); ax.set_ylim(0, 0.7)
    ax.set_title("Long-200 vs Long-500: accuracy vs budget — Qwen3-4B, T~U(1,40)")
    ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.grid(alpha=0.25)
    out = FIG / "27_long_extension_acc_vs_T.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def render_paired(data200, data500, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = pathlib.Path("analysis/figures"); FIG.mkdir(parents=True, exist_ok=True)
    PINK, NAVY, GRAY = "#C2185B", "#2C3E50", "#8E9BA8"

    # Flip counts
    flip_w2c = sum(1 for p in paired if p["c200"] == 0 and p["c500"] == 1)
    flip_c2w = sum(1 for p in paired if p["c200"] == 1 and p["c500"] == 0)
    stay_correct = sum(1 for p in paired if p["c200"] == 1 and p["c500"] == 1)
    stay_wrong = sum(1 for p in paired if p["c200"] == 0 and p["c500"] == 0)
    p_mc = mcnemar_exact(flip_w2c, flip_c2w)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # left: flip-count bars
    ax = axes[0]
    cats = ["wrong→correct\n(long-500 helped)", "correct→wrong\n(long-500 hurt)",
            "both correct", "both wrong"]
    vals = [flip_w2c, flip_c2w, stay_correct, stay_wrong]
    colors = [PINK, NAVY, "#888", GRAY]
    bars = ax.bar(cats, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, str(v),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("# problems")
    ax.set_title(f"Paired outcomes per problem (n={len(paired)})\n"
                 f"McNemar (mid-p) on discordant pairs: p={p_mc:.3g}")
    ax.grid(alpha=0.25, axis="y")

    # right: paired f comparison (long-200 vs long-500) among problems BOTH got right
    ax = axes[1]
    both_correct = [p for p in paired if p["c200"] == 1 and p["c500"] == 1]
    f200 = [p["f200"] for p in both_correct]
    f500 = [p["f500"] for p in both_correct]
    if f200:
        ax.scatter(f200, f500, s=20, color=PINK, alpha=0.55, edgecolor="white", lw=0.5)
        ax.plot([0, 1], [0, 1], color="#999", lw=1.2, ls="--", label="y=x (no change)")
        df = [b - a for a, b in zip(f200, f500)]
        mean_d = st.mean(df) if df else 0.0
        # paired t-stat (just for color)
        n = len(df)
        if n > 1:
            sd = (sum((x - mean_d)**2 for x in df) / (n - 1)) ** 0.5
            t = mean_d / (sd / n**0.5) if sd > 0 else 0.0
            ax.set_title(f"f(t,T) among both-correct (n={n})\n"
                         f"Δf mean = {mean_d:+.3f}   (paired t = {t:+.2f})")
        else:
            ax.set_title(f"f(t,T) among both-correct (n={n})")
        ax.set_xlabel("long-200  f(t,T)"); ax.set_ylabel("long-500  f(t,T)")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "no both-correct rollouts", ha="center", va="center")

    out = FIG / "28_long_extension_paired.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    data = {c: pull(c, nm) for c, nm in RUNS.items()}
    for c, recs in data.items():
        steps = sorted(set(r["step"] for r in recs))
        print(f"{c}: {len(recs)} rollouts, eval steps {steps}, mean T={st.mean([r['target_s'] for r in recs]):.1f}")
    print()

    fin200, s200 = step_rows(data["long-200"], "final")
    fin500, s500 = step_rows(data["long-500"], "final")

    # Build the paired view keyed by example_id (the (problem,T) pairs are identical
    # across runs because dataset_seed=777 + same T range).
    by200 = {r["example_id"]: r for r in fin200}
    by500 = {r["example_id"]: r for r in fin500}
    paired = []
    for ex in sorted(set(by200) & set(by500)):
        a, b = by200[ex], by500[ex]
        paired.append({
            "example_id": ex, "target_s": a["target_s"],
            "c200": a["is_correct"], "c500": b["is_correct"],
            "f200": a.get("f"), "f500": b.get("f"),
        })
    print(f"paired n = {len(paired)} (long-200 final step={s200}, long-500 final step={s500})\n")

    # ---- Per-bin accuracy, both conditions side by side ----
    print("=" * 78)
    print("Accuracy by budget bin — long-200 vs long-500 (final eval)")
    print("=" * 78)
    print("  T-bin       n     long-200    long-500    Δaccuracy")
    for lo, hi in BINS:
        in_bin = [p for p in paired if lo <= p["target_s"] < hi]
        if not in_bin: continue
        a200 = st.mean([p["c200"] for p in in_bin])
        a500 = st.mean([p["c500"] for p in in_bin])
        d = a500 - a200
        print(f"  {lo:>3}-{int(hi):<3}    {len(in_bin):>4}      {a200:.3f}       {a500:.3f}     {d:+.3f}")
    overall200 = st.mean([p["c200"] for p in paired])
    overall500 = st.mean([p["c500"] for p in paired])
    print(f"  overall   {len(paired):>4}      {overall200:.3f}       {overall500:.3f}     {overall500-overall200:+.3f}\n")

    # ---- Paired flip stats ----
    print("=" * 78)
    print("Paired flips (long-200 -> long-500)")
    print("=" * 78)
    flip_w2c = sum(1 for p in paired if p["c200"] == 0 and p["c500"] == 1)
    flip_c2w = sum(1 for p in paired if p["c200"] == 1 and p["c500"] == 0)
    stay_c = sum(1 for p in paired if p["c200"] == 1 and p["c500"] == 1)
    stay_w = sum(1 for p in paired if p["c200"] == 0 and p["c500"] == 0)
    print(f"  wrong → correct  (long-500 helped): {flip_w2c}")
    print(f"  correct → wrong  (long-500 hurt):   {flip_c2w}")
    print(f"  both correct (stayed correct):      {stay_c}")
    print(f"  both wrong  (stayed wrong):         {stay_w}")
    print(f"  McNemar exact (mid-p, two-sided):   p = {mcnemar_exact(flip_w2c, flip_c2w):.3g}\n")

    # ---- Logistic slope per condition ----
    print("=" * 78)
    print("Logistic slope: is_correct ~ standardized T (per condition)")
    print("=" * 78)
    for label, rows in (("long-200 final", fin200), ("long-500 final", fin500)):
        b1, se, z, p = logistic_slope(rows)
        print(f"  {label:18s}  slope = {b1:+.3f}  (z={z:+.2f}, p={p:.3f})")
    print()

    # ---- Pacing (f among correct) per condition ----
    print("=" * 78)
    print("Pacing — mean f(t,T)=min(1,T/t) among CORRECT rollouts, final")
    print("=" * 78)
    for label, rows in (("long-200", fin200), ("long-500", fin500)):
        for lo, hi in BINS:
            correct = [r for r in rows if r["is_correct"] and lo <= r["target_s"] < hi]
            if correct:
                mf = st.mean(r["f"] for r in correct)
                print(f"  {label:9s}  T {lo:>3}-{int(hi):<3}  n_correct={len(correct):>3}  mean f={mf:.3f}")
            else:
                print(f"  {label:9s}  T {lo:>3}-{int(hi):<3}  n_correct=0")
        print()

    render_acc_vs_T(data["long-200"], data["long-500"])
    render_paired(data["long-200"], data["long-500"], paired)


if __name__ == "__main__":
    main()
