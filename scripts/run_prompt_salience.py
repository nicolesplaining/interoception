"""Dispatcher for the prompt-salience eval — Modal or Lambda backend.

Modal backend (default):
    python scripts/run_prompt_salience.py --backend modal

Lambda backend (runs on an A100 over SSH):
    python scripts/run_prompt_salience.py --backend lambda \\
        --lambda-host ubuntu@129.146.160.194 \\
        [--lambda-key ~/.ssh/id_rsa]

The Lambda flow rsyncs the repo, installs deps in a venv on the box, downloads
the long-500 LoRA from the Modal volume (using the Modal CLI on this machine),
ships the adapter to the box, runs the eval, and pulls results back into
analysis/eval_rollouts/prompt_salience/.

Either backend produces the same output layout:
    analysis/eval_rollouts/prompt_salience/
        base_base.jsonl
        base_remaining_budget.jsonl
        long-500_base.jsonl
        long-500_remaining_budget.jsonl
"""
from __future__ import annotations
import argparse
import os
import pathlib
import shlex
import subprocess
import sys
import textwrap

REPO = pathlib.Path(__file__).resolve().parent.parent
LOCAL_OUT = REPO / "analysis" / "eval_rollouts" / "prompt_salience"
MODAL_VOLUME = "interoception-cache"
MODAL_ADAPTER = "runs/ctrl0_u1_40_long_qwen3_4b/weights/step_500/lora_adapters"


def sh(cmd, **kw):
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=kw.pop("check", True), **kw)


def run_modal(args):
    flags = []
    if args.skip_base: flags += ["--skip-base"]
    if args.skip_long500: flags += ["--skip-long500"]
    flags += ["--num-examples", str(args.num_examples), "--base-model", args.base_model]
    cmd = ["modal", "run", "--detach", "modal_app.py::eval_prompt_salience", "--", *flags]
    print("[backend=modal] launching:")
    print(" ", " ".join(shlex.quote(c) for c in cmd))
    sh(cmd, cwd=str(REPO))
    print()
    print("When done, pull results with:")
    print(f"  modal volume get {MODAL_VOLUME} eval_rollouts/prompt_salience {LOCAL_OUT} --force")


def run_lambda(args):
    """Drive the eval over SSH on a Lambda A100 box.

    Strategy:
      1. rsync the repo (data, env package, scripts) to ~/interoception/ on the box.
         Excludes runs/, .git/, the figure cache, and the wandb caches — only the
         source/data we need.
      2. pull the long-500 LoRA from the Modal volume on THIS machine, scp it to the box.
      3. ssh in and:
           - create a venv if absent, install vllm + hwprop + our env package
           - run scripts/eval_prompt_salience.py twice (base + long-500)
      4. rsync the resulting JSONLs back into analysis/eval_rollouts/prompt_salience/.
    """
    host = args.lambda_host
    key_arg = ["-i", os.path.expanduser(args.lambda_key)] if args.lambda_key else []
    ssh_base = ["ssh", *key_arg, "-o", "StrictHostKeyChecking=accept-new", host]
    rsync_e = "ssh " + " ".join(shlex.quote(a) for a in key_arg) if key_arg else "ssh"

    remote_root = "/home/ubuntu/interoception"
    remote_adapter = f"{remote_root}/.adapters/long-500"
    remote_out = f"{remote_root}/analysis/eval_rollouts/prompt_salience"

    print(f"[backend=lambda] host={host}  remote_root={remote_root}")

    # 1. ensure target dirs exist on the box, then rsync source tree to box.
    # (macOS's bundled rsync 2.6.9 won't auto-create parent dirs on the remote.)
    print("\n--- step 1/4: rsync source tree to Lambda ---")
    sh([*ssh_base, f"mkdir -p {remote_root} {remote_adapter} {remote_out}"])
    sh([
        "rsync", "-az", "--progress", "-e", rsync_e,
        "--exclude=.git/", "--exclude=runs/", "--exclude=.venv/",
        "--exclude=analysis/figures/", "--exclude=analysis/eval_rollouts/",
        "--exclude=.lambda_artifacts/",
        "--exclude=__pycache__/", "--exclude=*.pyc",
        f"{REPO}/", f"{host}:{remote_root}/",
    ])

    # 2. pull LoRA from Modal volume locally, scp to box (only if not skipping long-500)
    if not args.skip_long500:
        print("\n--- step 2/4: pull long-500 LoRA from Modal volume + ship to Lambda ---")
        local_adapter = REPO / ".lambda_artifacts" / "long-500"
        local_adapter.mkdir(parents=True, exist_ok=True)
        sh(["modal", "volume", "get", MODAL_VOLUME, MODAL_ADAPTER,
            str(local_adapter), "--force"])
        sh(["rsync", "-az", "--progress", "-e", rsync_e,
            f"{local_adapter}/", f"{host}:{remote_adapter}/"])
    else:
        print("\n--- step 2/4: skipped (--skip-long500) ---")

    # 3. install deps + run eval on the box. Idempotent (venv is reused if present).
    print("\n--- step 3/4: install deps + run eval on Lambda ---")
    variants_flag = " ".join(args.variants)
    skip_base = "1" if args.skip_base else ""
    skip_long = "1" if args.skip_long500 else ""
    remote_script = f"""
        set -e
        cd {remote_root}
        if ! [ -d .venv ]; then
            python3 -m venv .venv
        fi
        . .venv/bin/activate
        pip install --upgrade pip wheel >/dev/null
        # vLLM brings torch + flash attn; takes a while on first install.
        pip install 'vllm>=0.6,<0.11' >/dev/null
        # hwprop for sim timing (the env's training-time mode).
        if ! python -c 'import hwprop' 2>/dev/null; then
            pip install 'git+https://github.com/singhh5050/hardware-proprioception.git' >/dev/null
        fi
        # verifiers — PyPI's 0.1.10 dropped @vf.reward (the decorator the env uses on
        # every reward function). prime-rl pins a working version as a git submodule.
        # Install that ONE first so the env-package install below doesn't pull PyPI's.
        if ! python -c 'import verifiers as vf; vf.reward' 2>/dev/null; then
            # prime-rl's .gitmodules uses `git@github.com:` (SSH) URLs; Lambda has no
            # GitHub key, so rewrite to https on the fly (Modal image does the same).
            git config --global url."https://github.com/".insteadOf "git@github.com:"
            mkdir -p .deps && cd .deps
            if ! [ -d prime-rl/.git ]; then
                rm -rf prime-rl
                git clone -q https://github.com/PrimeIntellect-ai/prime-rl.git
            fi
            cd prime-rl
            git checkout -q b22e768fc419a1e8664729fd3fdfde98d1c13766
            git submodule update --init --quiet -- deps/verifiers
            cd ../..
            # --force-reinstall alone wasn't replacing the install; uninstall first.
            # We install with the [all] extra because verifiers gates MultiTurnEnv
            # behind it (raises AttributeError otherwise) and pulls in aiolimiter etc.
            pip uninstall -y verifiers >/dev/null 2>&1 || true
            pip install -e '.deps/prime-rl/deps/verifiers[all]' 2>&1 | tail -5
            python -c 'import verifiers as vf; from verifiers import MultiTurnEnv; print("verifiers OK, v=" + vf.__version__)'
        fi
        # our env package; -e so changes from rsync take effect without re-install.
        pip install -e ./environments/interoception_countdown >/dev/null

        # verifiers[all] just pulled in transformers 5.x (huge breaking changes vs 4.x).
        # vLLM 0.10.x calls Qwen2Tokenizer.all_special_tokens_extended which 5.x removed.
        # Pin transformers back to 4.x AFTER all the above, so this isn't undone.
        pip install 'transformers>=4.55,<5' >/dev/null
        mkdir -p {remote_out}

        BASE_MODEL={shlex.quote(args.base_model)}
        N={int(args.num_examples)}

        if [ -z "{skip_base}" ]; then
            python scripts/eval_prompt_salience.py \\
                --base-model "$BASE_MODEL" \\
                --num-examples "$N" \\
                --variants {variants_flag} \\
                --output-dir {remote_out} \\
                --run-label base
        fi
        if [ -z "{skip_long}" ]; then
            python scripts/eval_prompt_salience.py \\
                --base-model "$BASE_MODEL" \\
                --adapter-path {remote_adapter} \\
                --adapter-name long-500 \\
                --num-examples "$N" \\
                --variants {variants_flag} \\
                --output-dir {remote_out} \\
                --run-label long-500
        fi
    """
    # Pipe the script via stdin instead of passing as an argv string — SSH would
    # otherwise re-join args with spaces on the remote, which mangles the
    # embedded newlines/quotes (`bash: -c: option requires an argument`). We
    # use plain `bash` (no -l) because Lambda's login shell init can break
    # under non-interactive stdin and the script sources the venv explicitly.
    script_text = textwrap.dedent(remote_script)
    print(f"$ ssh ... bash  (script via stdin, {len(script_text.encode())} bytes)")
    subprocess.run([*ssh_base, "bash"], input=script_text, text=True, check=True)

    # 4. rsync results back
    print("\n--- step 4/4: pull results back ---")
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    sh(["rsync", "-az", "--progress", "-e", rsync_e,
        f"{host}:{remote_out}/", f"{LOCAL_OUT}/"])
    print(f"\nDone. Results at {LOCAL_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["modal", "lambda"], default="modal")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--num-examples", type=int, default=498)
    ap.add_argument("--variants", nargs="+", default=["base", "remaining_budget"])
    ap.add_argument("--skip-base", action="store_true",
                    help="Skip the base-model eval (only run long-500)")
    ap.add_argument("--skip-long500", action="store_true",
                    help="Skip the long-500 eval (only run base)")
    # Lambda-only:
    ap.add_argument("--lambda-host", default="ubuntu@129.146.160.194")
    ap.add_argument("--lambda-key", default=None, help="Path to SSH private key (optional)")
    args = ap.parse_args()

    if args.skip_base and args.skip_long500:
        sys.exit("nothing to do — both --skip-base and --skip-long500 set")

    if args.backend == "modal":
        run_modal(args)
    else:
        run_lambda(args)


if __name__ == "__main__":
    main()
