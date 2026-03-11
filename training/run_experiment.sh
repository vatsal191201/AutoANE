#!/bin/bash
# run_experiment.sh — Fixed-budget experiment runner for AutoANE autoresearch
#
# Compiles train.m with -D overrides for each config value, runs training
# for exactly 60 seconds, captures machine-parseable output, and appends
# the result as a JSON line to experiments.jsonl.
#
# Usage:
#   ./run_experiment.sh '{"lr": "5e-4", "wd": "0.2", "accum": "5"}'
#   ./run_experiment.sh config.json
#   ./run_experiment.sh                # (uses all defaults from train_config.h)
#
# Supported config keys (all optional, defaults from train_config.h):
#   Architecture: dim, nlayers, heads, kv_heads, hd, hidden, seq
#   Training:     lr, adam_b1, adam_b2, adam_eps, wd, accum, warmup, grad_clip,
#                 loss_scale, min_lr_frac
#   Flags:        cpu_attn_bwd (true/false), sanitize (true/false)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SRC="${SCRIPT_DIR}/train.m"
TRAIN_BIN="${SCRIPT_DIR}/train_experiment"
RESULTS_FILE="${SCRIPT_DIR}/experiments.jsonl"
TIME_BUDGET="${AUTOANE_TIME_BUDGET:-60}"

# ===== Parse config =====
CONFIG_JSON="{}"
if [ $# -ge 1 ]; then
    if [ -f "$1" ]; then
        CONFIG_JSON="$(cat "$1")"
    else
        CONFIG_JSON="$1"
    fi
fi

# Helper: extract a key from JSON (lightweight, no jq dependency)
json_get() {
    local key="$1"
    # Match "key": "value" or "key": value (number)
    echo "$CONFIG_JSON" | sed -n "s/.*\"${key}\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^,\"}\}]*\)\"\{0,1\}.*/\1/p" | head -1
}

# ===== Build -D flags from config =====
DFLAGS=""

# Architecture overrides
val="$(json_get dim)";       [ -n "$val" ] && DFLAGS="$DFLAGS -DDIM=$val"
val="$(json_get nlayers)";   [ -n "$val" ] && DFLAGS="$DFLAGS -DNLAYERS=$val"
val="$(json_get heads)";     [ -n "$val" ] && DFLAGS="$DFLAGS -DHEADS=$val"
val="$(json_get kv_heads)";  [ -n "$val" ] && DFLAGS="$DFLAGS -DKV_HEADS=$val"
val="$(json_get hd)";        [ -n "$val" ] && DFLAGS="$DFLAGS -DHD=$val"
val="$(json_get hidden)";    [ -n "$val" ] && DFLAGS="$DFLAGS -DHIDDEN=$val"
val="$(json_get seq)";       [ -n "$val" ] && DFLAGS="$DFLAGS -DSEQ=$val"

# Training overrides (compile-time defaults; runtime flags also available)
val="$(json_get lr)";           [ -n "$val" ] && DFLAGS="$DFLAGS -DMAX_LR=${val}f"
val="$(json_get adam_b1)";      [ -n "$val" ] && DFLAGS="$DFLAGS -DADAM_B1=${val}f"
val="$(json_get adam_b2)";      [ -n "$val" ] && DFLAGS="$DFLAGS -DADAM_B2=${val}f"
val="$(json_get adam_eps)";     [ -n "$val" ] && DFLAGS="$DFLAGS -DADAM_EPS=${val}f"
val="$(json_get wd)";           [ -n "$val" ] && DFLAGS="$DFLAGS -DWD=${val}f"
val="$(json_get accum)";        [ -n "$val" ] && DFLAGS="$DFLAGS -DACCUM_STEPS=$val"
val="$(json_get warmup)";       [ -n "$val" ] && DFLAGS="$DFLAGS -DWARMUP_STEPS=$val"
val="$(json_get grad_clip)";    [ -n "$val" ] && DFLAGS="$DFLAGS -DGRAD_CLIP=${val}f"
val="$(json_get loss_scale)";   [ -n "$val" ] && DFLAGS="$DFLAGS -DLOSS_SCALE=${val}f"
val="$(json_get min_lr_frac)";  [ -n "$val" ] && DFLAGS="$DFLAGS -DMIN_LR_FRAC=${val}f"

# ===== Build runtime flags =====
resume="$(json_get resume)"
if [ "$resume" = "true" ]; then
    RUN_FLAGS="--resume --time ${TIME_BUDGET} --steps 100000"
else
    RUN_FLAGS="--scratch --time ${TIME_BUDGET} --steps 100000"
fi

# Pass training params as runtime flags too (runtime overrides compile-time)
val="$(json_get lr)";         [ -n "$val" ] && RUN_FLAGS="$RUN_FLAGS --lr $val"
val="$(json_get accum)";      [ -n "$val" ] && RUN_FLAGS="$RUN_FLAGS --accum $val"
val="$(json_get warmup)";     [ -n "$val" ] && RUN_FLAGS="$RUN_FLAGS --warmup $val"
val="$(json_get grad_clip)";  [ -n "$val" ] && RUN_FLAGS="$RUN_FLAGS --clip $val"
val="$(json_get wd)";         [ -n "$val" ] && RUN_FLAGS="$RUN_FLAGS --wd $val"
val="$(json_get loss_scale)"; [ -n "$val" ] && RUN_FLAGS="$RUN_FLAGS --scale $val"

# Boolean flags
cpu_attn_bwd="$(json_get cpu_attn_bwd)"
[ "$cpu_attn_bwd" = "true" ] && RUN_FLAGS="$RUN_FLAGS --cpu-attn-bwd"

sanitize="$(json_get sanitize)"
[ "$sanitize" = "true" ] && RUN_FLAGS="$RUN_FLAGS --sanitize"

cpu_bwd="$(json_get cpu_bwd)"
[ "$cpu_bwd" = "true" ] && RUN_FLAGS="$RUN_FLAGS --cpu-bwd"

cpu_only="$(json_get cpu_only)"
[ "$cpu_only" = "true" ] && RUN_FLAGS="$RUN_FLAGS --cpu-only"

lora="$(json_get lora)"
[ "$lora" = "true" ] && RUN_FLAGS="$RUN_FLAGS --lora"

lora_rank="$(json_get lora_rank)"
[ -n "$lora_rank" ] && RUN_FLAGS="$RUN_FLAGS --lora-rank $lora_rank"

adaptive="$(json_get adaptive)"
[ -n "$adaptive" ] && RUN_FLAGS="$RUN_FLAGS --adaptive $adaptive"

adaptive_window="$(json_get adaptive_window)"
[ -n "$adaptive_window" ] && RUN_FLAGS="$RUN_FLAGS --adaptive-window $adaptive_window"

ane_matmul_only="$(json_get ane_matmul_only)"
[ "$ane_matmul_only" = "true" ] && RUN_FLAGS="$RUN_FLAGS --ane-matmul-only"

# ===== Compile =====
CC="xcrun clang"
CFLAGS="-O2 -DACCELERATE_NEW_LAPACK -framework Foundation -framework IOSurface -framework Accelerate -isysroot $(xcrun --show-sdk-path) -fobjc-arc"

echo "=== AutoANE Experiment Runner ==="
echo "Config: $CONFIG_JSON"
echo "D-flags: $DFLAGS"
echo "Run flags: $RUN_FLAGS"
echo ""

# Include train_config.h as the base config, then the model-specific -D overrides win
COMPILE_CMD="$CC $CFLAGS -include ${SCRIPT_DIR}/train_config.h $DFLAGS -o $TRAIN_BIN $TRAIN_SRC"
echo "Compiling..."
echo "  $COMPILE_CMD"

COMPILE_LOG=$(mktemp)
COMPILE_START=$(date +%s)
if ! eval "$COMPILE_CMD" > "$COMPILE_LOG" 2>&1; then
    COMPILE_ERR=$(cat "$COMPILE_LOG")
    rm -f "$COMPILE_LOG"
    echo "COMPILE FAILED:"
    echo "$COMPILE_ERR"

    # Record failure in experiments.jsonl
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    RESULT="{\"timestamp\": \"$TIMESTAMP\", \"config\": $CONFIG_JSON, \"status\": \"compile_error\", \"error\": \"compilation failed\", \"final_loss\": null}"
    echo "$RESULT" >> "$RESULTS_FILE"
    echo ""
    echo "$RESULT"
    exit 1
fi
COMPILE_END=$(date +%s)
COMPILE_SEC=$((COMPILE_END - COMPILE_START))
rm -f "$COMPILE_LOG"
echo "Compiled in ${COMPILE_SEC}s"

# ===== Run training =====
echo ""
echo "Training for ${TIME_BUDGET}s..."
echo "  $TRAIN_BIN $RUN_FLAGS"
echo ""

TRAIN_LOG=$(mktemp)
RUN_START=$(date +%s)

# Training binary handles its own time budget via --time flag.
# We just run it and check the exit code.
$TRAIN_BIN $RUN_FLAGS > "$TRAIN_LOG" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "TRAINING CRASHED (exit code $EXIT_CODE):"
    tail -50 "$TRAIN_LOG"
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    RESULT="{\"timestamp\": \"$TIMESTAMP\", \"config\": $CONFIG_JSON, \"status\": \"crash\", \"exit_code\": $EXIT_CODE, \"final_loss\": null}"
    echo "$RESULT" >> "$RESULTS_FILE"
    echo ""
    echo "$RESULT"
    rm -f "$TRAIN_LOG"
    exit 1
fi
RUN_END=$(date +%s)
RUN_SEC=$((RUN_END - RUN_START))

# ===== Parse machine-readable output =====
# The training binary prints these after "---":
#   final_loss:       <float>
#   training_seconds: <float>
#   total_seconds:    <float>
#   total_tokens_M:   <float>
#   num_steps:        <int>
#   num_params_M:     <float>
#   depth:            <int>

parse_field() {
    local field="$1"
    sed -n "s/^${field}:[[:space:]]*\(.*\)/\1/p" "$TRAIN_LOG" | tail -1
}

FINAL_LOSS="$(parse_field final_loss)"
TRAINING_SEC="$(parse_field training_seconds)"
TOTAL_SEC="$(parse_field total_seconds)"
TOTAL_TOKENS_M="$(parse_field total_tokens_M)"
NUM_STEPS="$(parse_field num_steps)"
NUM_PARAMS_M="$(parse_field num_params_M)"
DEPTH="$(parse_field depth)"
VAL_LOSS="$(parse_field val_loss)"
MODE="$(parse_field mode)"

# Fallbacks for missing fields
[ -z "$FINAL_LOSS" ] && FINAL_LOSS="null"
[ -z "$TRAINING_SEC" ] && TRAINING_SEC="null"
[ -z "$TOTAL_SEC" ] && TOTAL_SEC="null"
[ -z "$TOTAL_TOKENS_M" ] && TOTAL_TOKENS_M="null"
[ -z "$NUM_STEPS" ] && NUM_STEPS="null"
[ -z "$NUM_PARAMS_M" ] && NUM_PARAMS_M="null"
[ -z "$DEPTH" ] && DEPTH="null"
[ -z "$VAL_LOSS" ] && VAL_LOSS="null"
[ -z "$MODE" ] && MODE="unknown"

# ===== Build result JSON =====
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
STATUS="ok"
[ "$FINAL_LOSS" = "null" ] && STATUS="no_output"

RESULT=$(cat <<ENDJSON
{"timestamp": "$TIMESTAMP", "config": $CONFIG_JSON, "status": "$STATUS", "mode": "$MODE", "final_loss": $FINAL_LOSS, "val_loss": $VAL_LOSS, "training_seconds": $TRAINING_SEC, "total_seconds": $TOTAL_SEC, "total_tokens_M": $TOTAL_TOKENS_M, "num_steps": $NUM_STEPS, "num_params_M": $NUM_PARAMS_M, "depth": $DEPTH, "compile_seconds": $COMPILE_SEC, "time_budget": $TIME_BUDGET}
ENDJSON
)

# Append to experiments.jsonl
echo "$RESULT" >> "$RESULTS_FILE"

# Print summary
echo ""
echo "=== Experiment Result ==="
echo "$RESULT"
echo ""
echo "Result appended to: $RESULTS_FILE"

# Cleanup
rm -f "$TRAIN_LOG" "$TRAIN_BIN"
