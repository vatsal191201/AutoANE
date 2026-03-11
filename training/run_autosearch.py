#!/usr/bin/env python3
"""Autonomous hyperparameter search — no AI agent required.

Runs N experiments with random perturbations to train.py,
keeping improvements and reverting failures. Pure Python,
no dependencies beyond stdlib.

Usage:
    cd training
    python3 run_autosearch.py --experiments 100
    python3 run_autosearch.py --experiments 50 --strategy random
"""

import subprocess, os, sys, re, time, random, math, copy, argparse, json


# Search space: (min, max, type, log_scale)
SEARCH_SPACE = {
    'SEQ':          (64, 512, 'int_pow2', False),    # powers of 2
    'LR':           (1e-4, 1e-3, 'float', True),     # log-uniform
    'ACCUM_STEPS':  (5, 20, 'int', False),
    'WARMUP_STEPS': (25, 200, 'int', False),
    'WEIGHT_DECAY': (0.01, 0.3, 'float', True),
    'ADAM_B2':      (0.9, 0.999, 'float', False),
    'DEPTH':        (2, 8, 'int', False),
    'DIM':          (256, 768, 'int_mult64', False),  # multiples of 64
    'HIDDEN':       (704, 2048, 'int_mult64', False),
}

# Parameters that require architecture rebuild (new random init)
ARCH_PARAMS = {'DEPTH', 'DIM', 'HIDDEN', 'HEADS', 'KV_HEADS', 'HEAD_DIM'}


def read_train_py():
    with open('train.py', 'r') as f:
        return f.read()


def write_train_py(content):
    with open('train.py', 'w') as f:
        f.write(content)


def get_param(content, name):
    """Extract a parameter value from train.py."""
    m = re.search(rf'^{name}\s*=\s*([^\s#]+)', content, re.MULTILINE)
    if m:
        val = m.group(1)
        try:
            if '.' in val or 'e' in val.lower():
                return float(val)
            elif val in ('True', 'False'):
                return val == 'True'
            else:
                return int(val)
        except ValueError:
            return val
    return None


def set_param(content, name, value):
    """Set a parameter value in train.py, preserving comments."""
    if isinstance(value, float):
        if value < 0.01:
            val_str = f'{value:.1e}'
        else:
            val_str = f'{value}'
    elif isinstance(value, bool):
        val_str = str(value)
    else:
        val_str = str(value)

    pattern = rf'^({name}\s*=\s*)[^\s#]+(.*?)$'
    replacement = rf'\g<1>{val_str}\2'
    new_content = re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)
    return new_content


def sample_param(name, current_val):
    """Sample a new value for a parameter, biased toward the current value."""
    if name not in SEARCH_SPACE:
        return current_val

    lo, hi, ptype, log_scale = SEARCH_SPACE[name]

    if ptype == 'float':
        if log_scale:
            log_lo, log_hi = math.log(lo), math.log(hi)
            log_cur = math.log(max(current_val, lo))
            # Gaussian perturbation in log space
            new_log = log_cur + random.gauss(0, (log_hi - log_lo) * 0.15)
            new_log = max(log_lo, min(log_hi, new_log))
            return round(math.exp(new_log), 6)
        else:
            delta = (hi - lo) * 0.15
            new_val = current_val + random.gauss(0, delta)
            return round(max(lo, min(hi, new_val)), 6)

    elif ptype == 'int':
        delta = max(1, int((hi - lo) * 0.2))
        new_val = current_val + random.randint(-delta, delta)
        return max(lo, min(hi, new_val))

    elif ptype == 'int_pow2':
        powers = [2**i for i in range(6, 10) if lo <= 2**i <= hi]  # 64-512
        return random.choice(powers)

    elif ptype == 'int_mult64':
        steps = list(range(lo, hi + 1, 64))
        if not steps:
            return current_val
        # Bias toward current value
        idx = min(range(len(steps)), key=lambda i: abs(steps[i] - current_val))
        delta = max(1, len(steps) // 5)
        new_idx = idx + random.randint(-delta, delta)
        new_idx = max(0, min(len(steps) - 1, new_idx))
        return steps[new_idx]

    return current_val


def fix_dependent_params(content):
    """Ensure DIM = HEADS * HEAD_DIM and HIDDEN is reasonable."""
    dim = get_param(content, 'DIM')
    head_dim = get_param(content, 'HEAD_DIM') or 64
    heads = dim // head_dim
    kv_heads = get_param(content, 'KV_HEADS') or 2

    # Ensure HEADS divides evenly
    while heads > 0 and heads % kv_heads != 0:
        heads -= 1
    if heads <= 0:
        heads = kv_heads

    content = set_param(content, 'HEADS', heads)
    content = set_param(content, 'DIM', heads * head_dim)

    # HIDDEN should be ~2.75x DIM, rounded to 64
    actual_dim = heads * head_dim
    hidden = get_param(content, 'HIDDEN')
    ideal_hidden = int(round(actual_dim * 2.75 / 64)) * 64
    if abs(hidden - ideal_hidden) > 128:
        content = set_param(content, 'HIDDEN', ideal_hidden)

    return content


def run_experiment(description):
    """Run training and return (val_loss, steps, params_M, ms_step) or None on failure."""
    try:
        result = subprocess.run(
            ['python3', 'train.py'],
            capture_output=True, text=True, timeout=600
        )
        output = result.stdout + result.stderr

        val_loss = None
        steps = 0
        params_M = 0
        ms_step = 0

        for line in output.split('\n'):
            if line.startswith('val_loss:'):
                val_loss = float(line.split(':')[1].strip())
            elif line.startswith('num_steps:'):
                steps = int(line.split(':')[1].strip())
            elif line.startswith('num_params_M:'):
                params_M = float(line.split(':')[1].strip())
            elif line.startswith('total_seconds:') and steps > 0:
                total = float(line.split(':')[1].strip())
            elif 'ms/step' in line:
                m = re.search(r'([\d.]+)\s*ms/step', line)
                if m:
                    ms_step = float(m.group(1))

        if val_loss is None or val_loss == 0:
            return None

        if ms_step == 0 and steps > 0:
            ms_step = round(get_param(read_train_py(), 'TIME_BUDGET') * 1000 / steps, 1)

        return (val_loss, steps, params_M, ms_step)
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  CRASH: {e}")
        return None


def git_commit(msg):
    subprocess.run(['git', 'add', 'train.py', 'results.tsv'], capture_output=True)
    subprocess.run(['git', 'commit', '-m', msg], capture_output=True)
    result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                           capture_output=True, text=True)
    return result.stdout.strip()


def git_revert():
    subprocess.run(['git', 'reset', '--hard', 'HEAD~1'], capture_output=True)


def log_result(commit, val_loss, steps, params_M, ms_step, status, description):
    """Append to results.tsv."""
    with open('results.tsv', 'a') as f:
        vl = f'{val_loss:.3f}' if val_loss else 'null'
        f.write(f'{commit}\t{vl}\t{steps}\t{params_M}\t{ms_step}\t{status}\t{description}\n')


def main():
    parser = argparse.ArgumentParser(description='Autonomous hyperparameter search')
    parser.add_argument('--experiments', type=int, default=100,
                        help='Number of experiments to run')
    parser.add_argument('--strategy', choices=['random', 'local'], default='local',
                        help='Search strategy: random (uniform) or local (perturbation)')
    parser.add_argument('--branch', type=str, default=None,
                        help='Git branch name (default: autoresearch/auto-<timestamp>)')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Verify we're in the right directory
    if not os.path.exists('train.py'):
        print("ERROR: train.py not found. Run from training/ directory.")
        sys.exit(1)

    # Create branch
    branch = args.branch or f'autoresearch/auto-{int(time.time())}'
    subprocess.run(['git', 'checkout', '-b', branch], capture_output=True)
    print(f"Branch: {branch}")

    # Get baseline
    print("\n=== Running baseline ===")
    baseline_content = read_train_py()
    result = run_experiment("baseline")
    if result is None:
        print("ERROR: Baseline failed. Fix train.py first.")
        sys.exit(1)

    best_val_loss, steps, params_M, ms_step = result
    commit = git_commit("autosearch: baseline")
    log_result(commit, best_val_loss, steps, params_M, ms_step, 'keep', 'baseline')
    print(f"  Baseline: val_loss={best_val_loss:.4f}, steps={steps}, {ms_step}ms/step")

    # Search loop
    kept = 0
    discarded = 0
    crashed = 0

    for exp in range(args.experiments):
        print(f"\n=== Experiment {exp+1}/{args.experiments} "
              f"(kept={kept}, discarded={discarded}, crashed={crashed}) ===")
        print(f"  Best so far: val_loss={best_val_loss:.4f}")

        content = read_train_py()

        # Pick 1-2 parameters to modify
        n_changes = random.choice([1, 1, 1, 2])  # usually 1
        modifiable = [p for p in SEARCH_SPACE.keys() if get_param(content, p) is not None]
        params_to_change = random.sample(modifiable, min(n_changes, len(modifiable)))

        changes = []
        for param in params_to_change:
            old_val = get_param(content, param)
            new_val = sample_param(param, old_val)
            if new_val != old_val:
                content = set_param(content, param, new_val)
                changes.append(f"{param} {old_val}->{new_val}")

        if not changes:
            print("  No change sampled, skipping")
            continue

        # Fix dependent parameters
        content = fix_dependent_params(content)
        description = ', '.join(changes)
        print(f"  Changes: {description}")

        # Write and commit
        write_train_py(content)
        commit = git_commit(f"autosearch: {description}")

        # Run
        result = run_experiment(description)

        if result is None:
            print("  CRASHED — reverting")
            git_revert()
            log_result(commit, None, 0, 0, 0, 'crash', description)
            crashed += 1
            continue

        val_loss, steps, params_M, ms_step = result
        print(f"  Result: val_loss={val_loss:.4f} (best={best_val_loss:.4f})")

        if val_loss < best_val_loss:
            print(f"  KEEP (improved by {best_val_loss - val_loss:.4f})")
            best_val_loss = val_loss
            log_result(commit, val_loss, steps, params_M, ms_step, 'keep', description)
            kept += 1
        else:
            print(f"  DISCARD (worse by {val_loss - best_val_loss:.4f})")
            git_revert()
            log_result(commit, val_loss, steps, params_M, ms_step, 'discard', description)
            discarded += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"=== Autosearch Complete ===")
    print(f"  Experiments: {kept + discarded + crashed}")
    print(f"  Kept: {kept}, Discarded: {discarded}, Crashed: {crashed}")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Branch: {branch}")
    print(f"  Results: training/results.tsv")


if __name__ == '__main__':
    main()
