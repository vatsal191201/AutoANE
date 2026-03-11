#!/usr/bin/env python3
"""AutoANE autoresearch: automated architecture search and hyperparameter optimization.

Orchestrates experiment runs via run_experiment.sh, tracks results in experiments.jsonl,
and produces summary analysis.

Usage:
    python3 autoresearch.py                    # run default architecture search
    python3 autoresearch.py --search arch      # architecture search (depth/width)
    python3 autoresearch.py --search lr        # learning rate sweep
    python3 autoresearch.py --search custom    # custom configs from stdin (JSON per line)
    python3 autoresearch.py --analyze          # analyze existing experiments.jsonl
    python3 autoresearch.py --budget 60        # seconds per experiment (default 120)
    python3 autoresearch.py --cooldown 30      # seconds between experiments (default 30)
    python3 autoresearch.py --dry-run          # print configs without running
"""

import json
import os
import subprocess
import sys
import time
import math
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_EXPERIMENT = os.path.join(SCRIPT_DIR, "run_experiment.sh")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "experiments.jsonl")


def compute_params(dim, hidden, heads, kv_heads, hd, nlayers, vocab=49152):
    """Compute total parameter count for a transformer config."""
    q_dim = heads * hd
    kv_dim = kv_heads * hd
    wq = q_dim * dim
    wk = kv_dim * dim
    wv = kv_dim * dim
    wo = dim * q_dim
    w1 = hidden * dim
    w2 = dim * hidden
    w3 = hidden * dim
    per_layer = wq + wk + wv + wo + w1 + w2 + w3 + 2 * dim
    embed = vocab * dim
    total = nlayers * per_layer + embed + dim  # +dim for final RMSNorm
    return total


def estimate_memory_gb(params):
    """Estimate total training memory: weights + adam (m+v) + activations + grads."""
    return params * 4 * 3 / 1e9  # weights + 2x adam states, rough estimate


def make_arch_config(dim, nlayers, hidden=None, heads=None, kv_heads=None, hd=64):
    """Create a valid architecture config dict."""
    if heads is None:
        heads = max(1, dim // hd)
    if kv_heads is None:
        kv_heads = max(1, heads // 4)
    # Ensure divisibility
    while heads % kv_heads != 0:
        kv_heads -= 1
    if hidden is None:
        hidden = int(dim * 2.75)
        # Round to multiple of 64 for efficiency
        hidden = ((hidden + 63) // 64) * 64

    params = compute_params(dim, hidden, heads, kv_heads, hd, nlayers)
    mem_gb = estimate_memory_gb(params)

    return {
        "dim": str(dim),
        "nlayers": str(nlayers),
        "hidden": str(hidden),
        "heads": str(heads),
        "kv_heads": str(kv_heads),
        "hd": str(hd),
        "cpu_only": "true",
        # metadata (not passed to compiler, for logging)
        "_params_M": round(params / 1e6, 1),
        "_mem_gb": round(mem_gb, 2),
    }


def generate_arch_search():
    """Generate architecture search configs: vary depth and width."""
    configs = []

    # Grid: DIM x NLAYERS
    # Each config is independent — we measure val_loss at fixed wall-clock time
    grid = [
        # Small models (~30-50M params) — fast iteration
        (512,  4),   # 30M
        (512,  8),   # 42M
        (768,  2),   # 41M
        (768,  4),   # 63M
        (768,  6),   # 85M

        # Medium models (~80-120M params) — our current sweet spot
        (1024, 2),   # 56M
        (1024, 4),   # 95M  (baseline)
        (1024, 6),   # 134M
        (1024, 8),   # 173M

        # Larger width models
        (1536, 2),   # 88M
        (1536, 4),   # 177M
    ]

    for dim, nlayers in grid:
        cfg = make_arch_config(dim, nlayers)
        if cfg["_mem_gb"] > 10:
            print(f"  SKIP: {dim}d/{nlayers}L ({cfg['_params_M']}M params, {cfg['_mem_gb']}GB) — too large")
            continue
        configs.append(cfg)

    return configs


def generate_lr_search():
    """Generate learning rate sweep across top E39 architectures."""
    configs = []
    # Test top 3 from E39: 512d/4L, 768d/2L, 1024d/2L
    top_archs = [
        (512, 4),   # E39 winner: val_loss 3.61
        (768, 2),   # E39 #2: val_loss 3.73
        (1024, 2),  # E39 #3: val_loss 3.86
    ]
    lrs = [1e-4, 3e-4, 5e-4, 1e-3, 2e-3]
    for dim, nlayers in top_archs:
        for lr in lrs:
            cfg = make_arch_config(dim, nlayers)
            cfg["lr"] = f"{lr}"
            configs.append(cfg)
    return configs


def run_experiment(config, budget=120, dry_run=False):
    """Run a single experiment via run_experiment.sh."""
    # Remove metadata keys (prefixed with _) before passing to shell
    clean_config = {k: v for k, v in config.items() if not k.startswith("_")}
    config_json = json.dumps(clean_config)

    params_m = config.get("_params_M", "?")
    dim = config.get("dim", "?")
    nlayers = config.get("nlayers", "?")
    lr = config.get("lr", "default")
    print(f"\n{'='*60}")
    print(f"Experiment: {dim}d / {nlayers}L / {params_m}M params / LR={lr}")
    print(f"Config: {config_json}")

    if dry_run:
        print("  [DRY RUN — skipped]")
        return None

    env = os.environ.copy()
    env["AUTOANE_TIME_BUDGET"] = str(budget)

    start = time.time()
    result = subprocess.run(
        [RUN_EXPERIMENT, config_json],
        cwd=SCRIPT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=budget * 3 + 60,  # generous timeout
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        print(f"  stderr: {result.stderr[-500:]}" if result.stderr else "")
        return None

    # Parse the JSON result from the last line of stdout
    lines = result.stdout.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and "final_loss" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    print(f"  WARNING: could not parse result from output")
    return None


def analyze_results(results=None):
    """Analyze experiments.jsonl and print summary."""
    if results is None:
        results = []
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    if not results:
        print("No results to analyze.")
        return

    # Filter to successful runs with val_loss
    ok = [r for r in results if r.get("status") == "ok" and r.get("val_loss") not in (None, "null")]
    if not ok:
        ok = [r for r in results if r.get("status") == "ok" and r.get("final_loss") not in (None, "null")]
        loss_key = "final_loss"
    else:
        loss_key = "val_loss"

    if not ok:
        print("No successful runs found.")
        return

    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY ({len(ok)} successful runs)")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Params':>8} {'Steps':>6} {'ms/step':>8} {'Loss':>8} {'Val':>8} {'Mode':<15}")
    print("-" * 95)

    # Sort by val_loss (or final_loss)
    ok.sort(key=lambda r: float(r.get(loss_key, 999)))

    for r in ok:
        cfg = r.get("config", {})
        dim = cfg.get("dim", "?")
        nlayers = cfg.get("nlayers", "?")
        params = r.get("num_params_M", "?")
        steps = r.get("num_steps", "?")
        train_sec = r.get("training_seconds", 0)
        steps_int = int(steps) if steps not in (None, "?", "null") else 0
        train_f = float(train_sec) if train_sec not in (None, "null") else 0
        ms_step = f"{train_f * 1000 / steps_int:.0f}" if steps_int > 0 else "?"
        loss = r.get("final_loss", "?")
        val = r.get("val_loss", "?")
        mode = r.get("mode", "?")
        loss_s = f"{float(loss):.4f}" if loss not in (None, "?", "null") else "?"
        val_s = f"{float(val):.4f}" if val not in (None, "?", "null") else "?"
        label = f"{dim}d/{nlayers}L"
        lr = cfg.get("lr", "")
        if lr:
            label += f"/lr={lr}"

        print(f"{label:<25} {params:>7}M {steps:>6} {ms_step:>7}ms {loss_s:>8} {val_s:>8} {mode:<15}")

    # Best result
    best = ok[0]
    best_cfg = best.get("config", {})
    print(f"\nBest ({loss_key}): {best_cfg.get('dim','?')}d / {best_cfg.get('nlayers','?')}L — "
          f"{loss_key}={best.get(loss_key, '?')}, {best.get('num_steps','?')} steps")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AutoANE autoresearch orchestrator")
    parser.add_argument("--search", choices=["arch", "lr", "custom"], default="arch",
                        help="Search type (default: arch)")
    parser.add_argument("--budget", type=int, default=120,
                        help="Seconds per experiment (default: 120)")
    parser.add_argument("--cooldown", type=int, default=30,
                        help="Seconds between experiments (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results only")
    args = parser.parse_args()

    if args.analyze:
        analyze_results()
        return

    # Generate configs
    if args.search == "arch":
        configs = generate_arch_search()
    elif args.search == "lr":
        configs = generate_lr_search()
    elif args.search == "custom":
        configs = []
        print("Enter configs as JSON, one per line (Ctrl-D to end):")
        for line in sys.stdin:
            line = line.strip()
            if line:
                try:
                    configs.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  Invalid JSON: {line}")
    else:
        configs = generate_arch_search()

    print(f"\nAutoANE Autoresearch")
    print(f"  Search: {args.search}")
    print(f"  Configs: {len(configs)}")
    print(f"  Budget: {args.budget}s per experiment")
    print(f"  Cooldown: {args.cooldown}s between experiments")
    print(f"  Estimated total time: {len(configs) * (args.budget + args.cooldown) / 60:.0f} minutes")
    if args.dry_run:
        print(f"  Mode: DRY RUN")

    print(f"\nConfigurations:")
    for i, cfg in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] {cfg.get('dim','?')}d / {cfg.get('nlayers','?')}L / "
              f"{cfg.get('_params_M','?')}M params / {cfg.get('_mem_gb','?')}GB")

    # Run experiments
    session_results = []
    for i, cfg in enumerate(configs):
        result = run_experiment(cfg, budget=args.budget, dry_run=args.dry_run)
        if result:
            session_results.append(result)

        # Cooldown between experiments (not after last)
        if i < len(configs) - 1 and not args.dry_run:
            print(f"\n  Cooldown {args.cooldown}s...")
            time.sleep(args.cooldown)

    # Analyze results from this session
    if session_results:
        analyze_results(session_results)
    elif not args.dry_run:
        print("\nNo successful results in this session.")


if __name__ == "__main__":
    main()
