"""Probe: does the long-500 model condition its behavior on the budget T?

Uses the existing step_500 eval rollouts (498 problems, T ~ U(1,40) assigned
independently from problem). Because T-assignment is independent of problem,
this is a valid causal probe even without re-running paired rollouts: if the
model conditions on T, we should see its behavior shift with T systematically.

Per rollout, we extract:
  - total completion length (chars)
  - has_answer       — did <answer> appear?
  - elapsed_at_commit — the last [Xs elapsed] BEFORE the <answer> tag,
                        i.e. how much time the model thought it had used at commit
  - final_elapsed    — the last [Xs elapsed] anywhere in the completion
                        (where the model "thought it was" when it stopped)
  - num_turns        — count of [Xs elapsed] injections

Then we ask: do any of these vary with T?

If completion length, num_turns, elapsed_at_commit are constant across T-bins,
the model is NOT using T. If they scale with T, the model IS conditioning.
"""
from __future__ import annotations
import json, math, os, pathlib, random, re, tempfile, statistics as st

ENTITY = "singhh5050-stanford-university/interoception"
RUN_NAME = "ctrl0-qwen3-4b-u1-40-long"
EVAL_SEED, T_LO, T_HI = 777, 1.0, 40.0
CACHE = pathlib.Path("analysis/eval_rollouts/long_extension/long-500_with_completions.jsonl")
FINAL_STEP = 500
BINS = [(1, 9), (9, 17), (17, 25), (25, 33), (33, 40.001)]

ELAPSED_RE = re.compile(r"\[([\d.]+)s elapsed\]")


def target_s_for(example_id: int) -> float:
    rng = random.Random(EVAL_SEED ^ (example_id * 2654435761 & 0xFFFFFFFF))
    return rng.uniform(T_LO, T_HI)


def parse_completion(text: str) -> dict:
    """Extract behavioral signals from a completion containing assistant turns
    interleaved with [Xs elapsed] env injections."""
    elapsed_matches = list(ELAPSED_RE.finditer(text))
    num_turns = len(elapsed_matches)
    final_elapsed = float(elapsed_matches[-1].group(1)) if elapsed_matches else None

    answer_pos = text.find("<answer>")
    has_answer = answer_pos != -1

    # elapsed_at_commit: the last elapsed message BEFORE <answer> (i.e. what the
    # model perceived as its elapsed time when it decided to commit). If <answer>
    # came in turn 1, this is None — the model committed without seeing any elapsed
    # feedback. If <answer> came mid-rollout, this is the elapsed it had observed.
    elapsed_at_commit = None
    if has_answer:
        prior = [float(m.group(1)) for m in elapsed_matches if m.start() < answer_pos]
        elapsed_at_commit = prior[-1] if prior else 0.0  # 0.0 = committed on turn 1

    return {
        "comp_chars": len(text),
        "num_turns": num_turns,
        "has_answer": has_answer,
        "answer_pos_frac": (answer_pos / len(text)) if has_answer else None,
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
    out = {}
    for run in runs:
        tbls = [f for f in run.files() if "table/eval" in f.name and f.name.endswith(".json")]
        for f in tbls:
            f.download(root=d, replace=True)
            j = json.load(open(os.path.join(d, f.name)))
            c = j["columns"]
            si, ri, ei, comp_i = c.index("step"), c.index("reward"), c.index("example_id"), c.index("completion")
            for row in j["data"]:
                step = int(row[si])
                if step != FINAL_STEP:
                    continue
                ex = int(row[ei]); rew = float(row[ri])
                # Later wandb runs overwrite earlier ones if the step appears in both.
                out[ex] = {
                    "example_id": ex, "target_s": target_s_for(ex),
                    "reward": rew, "is_correct": 1 if rew > 0 else 0,
                    "f": rew if rew > 0 else None, "completion": row[comp_i],
                }
    recs = list(out.values())
    # Enrich each with parsed signals.
    for r in recs:
        sig = parse_completion(r["completion"])
        r.update(sig)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w") as fh:
        for r in recs:
            r2 = {k: v for k, v in r.items() if k != "completion"}  # don't bloat the cache
            fh.write(json.dumps(r2) + "\n")
    return [{k: v for k, v in r.items() if k != "completion"} for r in recs]


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
    # two-sided p via Fisher z, valid for moderate n
    if abs(r) >= 1.0: return r, 0.0
    z = 0.5 * math.log((1+r)/(1-r)); se = 1 / (n-3)**0.5
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z/se) / math.sqrt(2))))
    return r, p


def render(recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = pathlib.Path("analysis/figures"); FIG.mkdir(parents=True, exist_ok=True)
    PINK, NAVY, GREEN = "#C2185B", "#2C3E50", "#2ca02c"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel 1: completion length vs T (scatter + bin means)
    ax = axes[0]
    xs = [r["target_s"] for r in recs]
    ys = [r["comp_chars"] for r in recs]
    ax.scatter(xs, ys, s=10, color=NAVY, alpha=0.25, edgecolor="white", lw=0.3)
    bx, by = [], []
    for lo, hi in BINS:
        b = [r for r in recs if lo <= r["target_s"] < hi]
        if not b: continue
        bx.append(sum(r["target_s"] for r in b)/len(b))
        by.append(mean(r["comp_chars"] for r in b))
    ax.plot(bx, by, marker="o", ms=9, color=PINK, lw=2.4, label="bin mean")
    r, p = pearson(xs, ys)
    ax.set_xlabel("Budget T (s)"); ax.set_ylabel("Completion length (chars)")
    ax.set_title(f"Completion length vs T\nPearson r={r:+.3f} (p={p:.3f})")
    ax.legend(frameon=False); ax.grid(alpha=0.25)

    # Panel 2: elapsed_at_commit vs T (only rollouts that committed)
    ax = axes[1]
    committed = [r for r in recs if r["has_answer"] and r["elapsed_at_commit"] is not None]
    xs = [r["target_s"] for r in committed]
    ys = [r["elapsed_at_commit"] for r in committed]
    ax.scatter(xs, ys, s=10, color=GREEN, alpha=0.35, edgecolor="white", lw=0.3)
    bx, by = [], []
    for lo, hi in BINS:
        b = [r for r in committed if lo <= r["target_s"] < hi]
        if not b: continue
        bx.append(sum(r["target_s"] for r in b)/len(b))
        by.append(mean(r["elapsed_at_commit"] for r in b))
    ax.plot(bx, by, marker="o", ms=9, color=PINK, lw=2.4, label="bin mean")
    ax.plot([0, 40], [0, 40], color="#999", lw=1.2, ls="--", label="t = T (on-budget line)")
    r, p = pearson(xs, ys)
    ax.set_xlabel("Budget T (s)"); ax.set_ylabel("Elapsed at commit (s)")
    ax.set_title(f"Elapsed at commit vs T (committers only, n={len(committed)})\nPearson r={r:+.3f} (p={p:.3f})")
    ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.25)

    # Panel 3: commit rate per T-bin
    ax = axes[2]
    bx, by, ec = [], [], []
    for lo, hi in BINS:
        b = [r for r in recs if lo <= r["target_s"] < hi]
        if not b: continue
        rate = sum(1 for r in b if r["has_answer"]) / len(b)
        bx.append((lo+hi)/2); by.append(rate); ec.append((rate*(1-rate)/len(b))**0.5)
    ax.errorbar(bx, by, yerr=ec, marker="o", ms=8, color=PINK, lw=2.2, capsize=4)
    ax.set_xlabel("Budget T (s)"); ax.set_ylabel("Commit rate (any <answer>)")
    ax.set_xlim(0, 40); ax.set_ylim(0, 1.05)
    ax.set_title("Commit rate vs T")
    ax.grid(alpha=0.25)

    fig.suptitle("Probe: does long-500 condition behavior on T? — Qwen3-4B, step 500", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG / "32_long_T_conditioning_probe.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    recs = load_rollouts()
    print(f"n rollouts (step {FINAL_STEP}): {len(recs)}")
    n_commit = sum(1 for r in recs if r["has_answer"])
    print(f"  commit rate: {n_commit}/{len(recs)} = {n_commit/len(recs):.3f}")
    print()

    # ---- Per-bin behavioral summary ----
    print("=" * 95)
    print("Behavioral signals by T-bin — does any of this change with T?")
    print("=" * 95)
    print(f"  {'T-bin':<10} {'n':>4}  {'comp_chars':>11}  {'num_turns':>10}  {'commit%':>8}  "
          f"{'elapsed@commit':>15}  {'final_elapsed':>14}")
    for lo, hi in BINS:
        b = [r for r in recs if lo <= r["target_s"] < hi]
        if not b: continue
        commit_rate = sum(1 for r in b if r["has_answer"]) / len(b)
        e_at_commit = mean([r["elapsed_at_commit"] for r in b if r["has_answer"]])
        e_final = mean([r["final_elapsed"] for r in b])
        print(f"  {lo:>3}-{int(hi):<3}    {len(b):>4}  {mean(r['comp_chars'] for r in b):>11.0f}  "
              f"{mean(r['num_turns'] for r in b):>10.2f}  {commit_rate:>8.3f}  "
              f"{e_at_commit:>15.2f}  {e_final:>14.2f}")
    print()

    # ---- Correlations with T ----
    print("=" * 95)
    print("Pearson correlations: behavior ~ T (positive r = scales with budget)")
    print("=" * 95)
    xs_all = [r["target_s"] for r in recs]
    for name, getter in (
        ("completion length", lambda r: r["comp_chars"]),
        ("num turns",         lambda r: r["num_turns"]),
        ("commit (0/1)",      lambda r: r["has_answer"]),
        ("final elapsed",     lambda r: r["final_elapsed"]),
    ):
        ys = [getter(r) for r in recs]
        r_, p = pearson(xs_all, ys)
        print(f"  {name:22s}  r = {r_:+.3f}   p = {p:.3f}")
    # commit-time correlation only among committers
    committers = [r for r in recs if r["has_answer"] and r["elapsed_at_commit"] is not None]
    r_, p = pearson([r["target_s"] for r in committers], [r["elapsed_at_commit"] for r in committers])
    print(f"  {'elapsed @ commit | committed':22s}  r = {r_:+.3f}   p = {p:.3f}   (n={len(committers)})")
    print()

    # ---- Key contrasts ----
    print("=" * 95)
    print("Direct contrast: low-T (T<10) vs high-T (T>30) rollouts")
    print("=" * 95)
    lo_T = [r for r in recs if r["target_s"] < 10]
    hi_T = [r for r in recs if r["target_s"] > 30]
    print(f"  n_lo = {len(lo_T)} (T<10)   n_hi = {len(hi_T)} (T>30)")
    print(f"  {'metric':<25} {'low-T':>10}  {'high-T':>10}  {'Δ':>10}")
    for name, getter in (
        ("completion length (chars)", lambda r: r["comp_chars"]),
        ("num turns",                  lambda r: r["num_turns"]),
        ("commit rate",                lambda r: r["has_answer"]),
        ("final elapsed (s)",          lambda r: r["final_elapsed"]),
    ):
        a = mean([getter(r) for r in lo_T])
        b = mean([getter(r) for r in hi_T])
        print(f"  {name:<25} {a:>10.2f}  {b:>10.2f}  {b-a:>+10.2f}")
    # Commit-time among committers
    a = mean([r["elapsed_at_commit"] for r in lo_T if r["has_answer"]])
    b = mean([r["elapsed_at_commit"] for r in hi_T if r["has_answer"]])
    print(f"  {'elapsed @ commit (s)':<25} {a:>10.2f}  {b:>10.2f}  {b-a:>+10.2f}")
    print()

    render(recs)


if __name__ == "__main__":
    main()
