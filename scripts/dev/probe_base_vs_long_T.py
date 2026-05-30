"""Base (step 0) vs long-500 (step 500) T-conditioning probe.

Same machinery as probe_long_T_conditioning.py, applied to BOTH the base model
and the trained model on identical eval data. The question: was T-blindness
caused by RL, or did the base model already not condition on T?

Step 0 in the long run's eval is the untrained Qwen3-4B-Instruct-2507 evaluated
on data/eval.jsonl with the same env + same T-assignments as step 500. So this
is a direct apples-to-apples comparison of behavior with vs without RL.

If step 0 already shows ~0 correlation between behavior and T:
  → the model fundamentally doesn't use the budget signal; reward-shape changes
    won't fix it without a prompt-format change or model swap.

If step 0 shows real T-conditioning that RL ERASED:
  → RL collapsed the policy into a fixed-time-commit habit. Reward-design issue.
"""
from __future__ import annotations
import json, math, os, pathlib, random, re, tempfile, statistics as st

ENTITY = "singhh5050-stanford-university/interoception"
RUN_NAME = "ctrl0-qwen3-4b-u1-40-long"
EVAL_SEED, T_LO, T_HI = 777, 1.0, 40.0
CACHE = pathlib.Path("analysis/eval_rollouts/long_extension/long_step0_and_500.jsonl")
STEPS = (0, 500)
BINS = [(1, 9), (9, 17), (17, 25), (25, 33), (33, 40.001)]

ELAPSED_RE = re.compile(r"\[([\d.]+)s elapsed\]")


def target_s_for(example_id: int) -> float:
    rng = random.Random(EVAL_SEED ^ (example_id * 2654435761 & 0xFFFFFFFF))
    return rng.uniform(T_LO, T_HI)


def parse_completion(text: str) -> dict:
    elapsed_matches = list(ELAPSED_RE.finditer(text))
    num_turns = len(elapsed_matches)
    final_elapsed = float(elapsed_matches[-1].group(1)) if elapsed_matches else None

    answer_pos = text.find("<answer>")
    has_answer = answer_pos != -1

    elapsed_at_commit = None
    if has_answer:
        prior = [float(m.group(1)) for m in elapsed_matches if m.start() < answer_pos]
        elapsed_at_commit = prior[-1] if prior else 0.0

    return {
        "comp_chars": len(text),
        "num_turns": num_turns,
        "has_answer": has_answer,
        "elapsed_at_commit": elapsed_at_commit,
        "final_elapsed": final_elapsed,
    }


def load_rollouts():
    if CACHE.exists():
        return [json.loads(l) for l in CACHE.open()]
    import wandb
    api = wandb.Api()
    runs = sorted(api.runs(ENTITY, filters={"display_name": RUN_NAME}), key=lambda r: r.created_at)
    d = tempfile.mkdtemp()
    out = {}  # (step, example_id) -> record
    for run in runs:
        tbls = [f for f in run.files() if "table/eval" in f.name and f.name.endswith(".json")]
        for f in tbls:
            f.download(root=d, replace=True)
            j = json.load(open(os.path.join(d, f.name)))
            c = j["columns"]
            si, ri, ei, comp_i = c.index("step"), c.index("reward"), c.index("example_id"), c.index("completion")
            for row in j["data"]:
                step = int(row[si])
                if step not in STEPS:
                    continue
                ex = int(row[ei]); rew = float(row[ri])
                sig = parse_completion(row[comp_i])
                out[(step, ex)] = {
                    "step": step, "example_id": ex, "target_s": target_s_for(ex),
                    "reward": rew, "is_correct": 1 if rew > 0 else 0,
                    "f": rew if rew > 0 else None, **sig,
                }
    recs = list(out.values())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return recs


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2: return float("nan"), float("nan")
    xs, ys = zip(*pairs); n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); syy = sum((y-my)**2 for y in ys)
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0: return float("nan"), float("nan")
    r = sxy / (sxx*syy)**0.5
    if abs(r) >= 1.0: return r, 0.0
    z = 0.5 * math.log((1+r)/(1-r)); se = 1 / (n-3)**0.5
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z/se) / math.sqrt(2))))
    return r, p


def render(by_step):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = pathlib.Path("analysis/figures"); FIG.mkdir(parents=True, exist_ok=True)
    BASE_C, FIN_C = "#8E9BA8", "#C2185B"

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    # Panel 1: completion length vs T (bin means)
    ax = axes[0]
    for step, color, label in ((0, BASE_C, "base (step 0)"), (500, FIN_C, "long-500 (step 500)")):
        recs = by_step[step]
        bx, by = [], []
        for lo, hi in BINS:
            b = [r for r in recs if lo <= r["target_s"] < hi]
            if not b: continue
            bx.append(sum(r["target_s"] for r in b)/len(b))
            by.append(mean(r["comp_chars"] for r in b))
        ls = "--" if step == 0 else "-"
        ax.plot(bx, by, marker="o", ms=9, color=color, lw=2.4, ls=ls, label=label)
    ax.set_xlabel("Budget T (s)"); ax.set_ylabel("Completion length (chars)")
    ax.set_title("Completion length vs T")
    ax.legend(frameon=False); ax.grid(alpha=0.25)

    # Panel 2: elapsed_at_commit vs T (committers only, bin means)
    ax = axes[1]
    for step, color, label in ((0, BASE_C, "base"), (500, FIN_C, "long-500")):
        recs = [r for r in by_step[step] if r["has_answer"] and r["elapsed_at_commit"] is not None]
        bx, by = [], []
        for lo, hi in BINS:
            b = [r for r in recs if lo <= r["target_s"] < hi]
            if not b: continue
            bx.append(sum(r["target_s"] for r in b)/len(b))
            by.append(mean(r["elapsed_at_commit"] for r in b))
        ls = "--" if step == 0 else "-"
        ax.plot(bx, by, marker="o", ms=9, color=color, lw=2.4, ls=ls,
                label=f"{label} (n_commit={len(recs)})")
    ax.plot([0, 40], [0, 40], color="#999", lw=1.2, ls=":", label="t = T (on-budget line)")
    ax.set_xlabel("Budget T (s)"); ax.set_ylabel("Elapsed at commit (s)")
    ax.set_title("Elapsed at commit vs T")
    ax.set_xlim(0, 40); ax.set_ylim(0, 60)
    ax.legend(frameon=False, loc="upper left", fontsize=9); ax.grid(alpha=0.25)

    # Panel 3: commit rate per T-bin
    ax = axes[2]
    for step, color, label in ((0, BASE_C, "base"), (500, FIN_C, "long-500")):
        recs = by_step[step]
        bx, by, ec = [], [], []
        for lo, hi in BINS:
            b = [r for r in recs if lo <= r["target_s"] < hi]
            if not b: continue
            rate = sum(1 for r in b if r["has_answer"]) / len(b)
            bx.append((lo+hi)/2); by.append(rate); ec.append((rate*(1-rate)/len(b))**0.5)
        ls = "--" if step == 0 else "-"
        ax.errorbar(bx, by, yerr=ec, marker="o", ms=8, color=color, lw=2.2, capsize=4, ls=ls, label=label)
    ax.set_xlabel("Budget T (s)"); ax.set_ylabel("Commit rate (any <answer>)")
    ax.set_xlim(0, 40); ax.set_ylim(0, 1.05)
    ax.set_title("Commit rate vs T")
    ax.legend(frameon=False); ax.grid(alpha=0.25)

    fig.suptitle("Base vs long-500: does either condition behavior on T? — Qwen3-4B", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG / "33_base_vs_long_T_conditioning.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    recs = load_rollouts()
    by_step = {s: [r for r in recs if r["step"] == s] for s in STEPS}
    for s in STEPS:
        n = len(by_step[s])
        nc = sum(1 for r in by_step[s] if r["has_answer"])
        print(f"step {s}: n={n}  commit_rate={nc/n:.3f}")
    print()

    for s in STEPS:
        print("=" * 95)
        print(f"step {s} — behavior vs T")
        print("=" * 95)
        print(f"  {'T-bin':<10} {'n':>4}  {'comp_chars':>11}  {'num_turns':>10}  {'commit%':>8}  "
              f"{'elapsed@commit':>15}  {'final_elapsed':>14}")
        for lo, hi in BINS:
            b = [r for r in by_step[s] if lo <= r["target_s"] < hi]
            if not b: continue
            cr = sum(1 for r in b if r["has_answer"]) / len(b)
            eac = mean([r["elapsed_at_commit"] for r in b if r["has_answer"]])
            fe = mean([r["final_elapsed"] for r in b])
            print(f"  {lo:>3}-{int(hi):<3}    {len(b):>4}  {mean(r['comp_chars'] for r in b):>11.0f}  "
                  f"{mean(r['num_turns'] for r in b):>10.2f}  {cr:>8.3f}  {eac:>15.2f}  {fe:>14.2f}")
        print()

    # Side-by-side correlations
    print("=" * 95)
    print("Pearson r (behavior ~ T)   base   vs   long-500")
    print("=" * 95)
    print(f"  {'signal':<32} {'base r':>10}  {'base p':>8}  {'fin r':>10}  {'fin p':>8}")
    sigs = [
        ("completion length",    lambda r: r["comp_chars"]),
        ("num turns",            lambda r: r["num_turns"]),
        ("commit (0/1)",         lambda r: r["has_answer"]),
        ("final elapsed",        lambda r: r["final_elapsed"]),
    ]
    for name, getter in sigs:
        rb, pb = pearson([r["target_s"] for r in by_step[0]], [getter(r) for r in by_step[0]])
        rf, pf = pearson([r["target_s"] for r in by_step[500]], [getter(r) for r in by_step[500]])
        print(f"  {name:<32} {rb:>+10.3f}  {pb:>8.3f}  {rf:>+10.3f}  {pf:>8.3f}")
    # elapsed@commit | committed
    for label, step in (("base", 0), ("long-500", 500)):
        cs = [r for r in by_step[step] if r["has_answer"] and r["elapsed_at_commit"] is not None]
    rb_c, pb_c = pearson(
        [r["target_s"] for r in by_step[0] if r["has_answer"] and r["elapsed_at_commit"] is not None],
        [r["elapsed_at_commit"] for r in by_step[0] if r["has_answer"] and r["elapsed_at_commit"] is not None],
    )
    rf_c, pf_c = pearson(
        [r["target_s"] for r in by_step[500] if r["has_answer"] and r["elapsed_at_commit"] is not None],
        [r["elapsed_at_commit"] for r in by_step[500] if r["has_answer"] and r["elapsed_at_commit"] is not None],
    )
    print(f"  {'elapsed@commit | committed':<32} {rb_c:>+10.3f}  {pb_c:>8.3f}  {rf_c:>+10.3f}  {pf_c:>8.3f}")
    print()

    print("=" * 95)
    print("Direct contrast low-T (T<10) vs high-T (T>30) — same metric, both steps")
    print("=" * 95)
    print(f"  {'metric':<27} {'base lo-T':>10}  {'base hi-T':>10}  {'base Δ':>8}  "
          f"{'fin lo-T':>10}  {'fin hi-T':>10}  {'fin Δ':>8}")
    metrics = [
        ("completion length",  lambda r: r["comp_chars"]),
        ("num turns",          lambda r: r["num_turns"]),
        ("commit rate",        lambda r: r["has_answer"]),
        ("final elapsed",      lambda r: r["final_elapsed"]),
    ]
    for name, getter in metrics:
        bL = mean([getter(r) for r in by_step[0] if r["target_s"] < 10])
        bH = mean([getter(r) for r in by_step[0] if r["target_s"] > 30])
        fL = mean([getter(r) for r in by_step[500] if r["target_s"] < 10])
        fH = mean([getter(r) for r in by_step[500] if r["target_s"] > 30])
        print(f"  {name:<27} {bL:>10.2f}  {bH:>10.2f}  {bH-bL:>+8.2f}  {fL:>10.2f}  {fH:>10.2f}  {fH-fL:>+8.2f}")
    bL = mean([r["elapsed_at_commit"] for r in by_step[0] if r["target_s"] < 10 and r["has_answer"]])
    bH = mean([r["elapsed_at_commit"] for r in by_step[0] if r["target_s"] > 30 and r["has_answer"]])
    fL = mean([r["elapsed_at_commit"] for r in by_step[500] if r["target_s"] < 10 and r["has_answer"]])
    fH = mean([r["elapsed_at_commit"] for r in by_step[500] if r["target_s"] > 30 and r["has_answer"]])
    print(f"  {'elapsed @ commit':<27} {bL:>10.2f}  {bH:>10.2f}  {bH-bL:>+8.2f}  {fL:>10.2f}  {fH:>10.2f}  {fH-fL:>+8.2f}")
    print()

    render(by_step)


if __name__ == "__main__":
    main()
