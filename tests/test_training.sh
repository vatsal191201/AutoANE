#!/bin/bash
# test_training.sh — Minimal regression tests for training pipeline
# Verifies: compilation, forward pass, backward pass, Adam update, checkpoint I/O
# Usage: cd training && bash ../tests/test_training.sh

set -e

PASS=0
FAIL=0
TRAIN_DIR="$(cd "$(dirname "$0")/../training" && pwd)"
cd "$TRAIN_DIR"

DATA="../tinystories_smollm2_data00.bin"
if [ ! -f "$DATA" ]; then
    echo "SKIP: Training data not found ($DATA). Download first."
    exit 0
fi

pass() { PASS=$((PASS+1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL+1)); echo "  FAIL: $1"; }

echo "=== AutoANE Training Regression Tests ==="
echo ""

# Test 1: Compilation
echo "Test 1: Compilation"
make clean >/dev/null 2>&1
MAKE_OUT=$(make MODEL=autoresearch 2>&1)
if [ -f ./train ]; then
    pass "Compiles cleanly"
else
    echo "$MAKE_OUT"
    fail "Compilation failed"
    exit 1
fi

# Test 2: CPU-only training produces reasonable loss
echo "Test 2: CPU-only forward+backward (7 steps, seed=42)"
OUTPUT=$(./train --scratch --data "$DATA" --lr 4e-4 --warmup 10 --accum 7 \
    --clip 1.0 --steps 7 --time 30 --scale 256.0 --cpu-only --seed 42 2>&1)

STEP0_LOSS=$(echo "$OUTPUT" | grep "^step 0" | grep -oE 'loss=[0-9.]+' | cut -d= -f2)
if [ -z "$STEP0_LOSS" ]; then
    fail "No step 0 loss output"
else
    # Step 0 loss should be close to ln(compact_vocab) ~ 9.73
    # Allow range 9.0-10.5 to account for random init
    if awk "BEGIN {exit !($STEP0_LOSS > 9.0 && $STEP0_LOSS < 10.5)}"; then
        pass "Step 0 loss=$STEP0_LOSS (expected ~9.7)"
    else
        fail "Step 0 loss=$STEP0_LOSS outside expected range [9.0, 10.5]"
    fi
fi

# Test 3: Different seeds produce different losses
echo "Test 3: Seed sensitivity"
OUTPUT2=$(./train --scratch --data "$DATA" --lr 4e-4 --warmup 10 --accum 7 \
    --clip 1.0 --steps 7 --time 30 --scale 256.0 --cpu-only --seed 123 2>&1)
STEP0_LOSS2=$(echo "$OUTPUT2" | grep "^step 0" | grep -oE 'loss=[0-9.]+' | cut -d= -f2)
if [ -n "$STEP0_LOSS2" ] && [ "$STEP0_LOSS" != "$STEP0_LOSS2" ]; then
    pass "Seed 42 ($STEP0_LOSS) != Seed 123 ($STEP0_LOSS2)"
else
    fail "Seeds produced identical losses"
fi

# Test 4: Adam update runs (loss after accumulation step should differ from step 0)
echo "Test 4: Adam update (14 steps = 2 accum cycles)"
OUTPUT3=$(./train --scratch --data "$DATA" --lr 4e-4 --warmup 10 --accum 7 \
    --clip 1.0 --steps 14 --time 30 --scale 256.0 --cpu-only --seed 42 2>&1)
FINAL_LOSS=$(echo "$OUTPUT3" | grep "^final_loss:" | awk '{print $2}')
if [ -n "$FINAL_LOSS" ] && awk "BEGIN {exit !($FINAL_LOSS < $STEP0_LOSS)}"; then
    pass "Loss decreased after Adam ($STEP0_LOSS -> $FINAL_LOSS)"
else
    fail "Loss did not decrease ($STEP0_LOSS -> $FINAL_LOSS)"
fi

# Test 5: Checkpoint save/load
echo "Test 5: Checkpoint round-trip"
CKPT="ane_autoresearch_ckpt.bin"
if [ -f "$CKPT" ]; then
    CKPT_SIZE=$(stat -f%z "$CKPT" 2>/dev/null || stat -c%s "$CKPT" 2>/dev/null)
    if [ "$CKPT_SIZE" -gt 1000 ]; then
        # Resume from checkpoint -- should not crash
        RESUME_OUT=$(./train --data "$DATA" --lr 4e-4 --warmup 10 --accum 7 \
            --clip 1.0 --steps 1 --time 10 --scale 256.0 --cpu-only --seed 42 2>&1 || true)
        if echo "$RESUME_OUT" | grep -qE "Resuming|step"; then
            pass "Checkpoint loaded ($CKPT_SIZE bytes)"
        else
            fail "Checkpoint resume failed"
        fi
    else
        fail "Checkpoint too small ($CKPT_SIZE bytes)"
    fi
else
    echo "  SKIP: No checkpoint found (expected after short training)"
fi

# Test 6: ANE matmul-only compilation and execution
echo "Test 6: ANE matmul-only mode"
ANE_OUT=$(./train --scratch --data "$DATA" --lr 4e-4 --warmup 10 --accum 7 \
    --clip 1.0 --steps 7 --time 30 --scale 256.0 --ane-matmul-only --seed 42 2>&1)
if echo "$ANE_OUT" | grep -q "Compiled.*kernels"; then
    ANE_LOSS=$(echo "$ANE_OUT" | grep "^step 0" | grep -oE 'loss=[0-9.]+' | cut -d= -f2)
    if [ -n "$ANE_LOSS" ] && awk "BEGIN {exit !($ANE_LOSS > 9.0 && $ANE_LOSS < 10.5)}"; then
        pass "ANE mode: loss=$ANE_LOSS"
    else
        fail "ANE mode: loss=$ANE_LOSS outside range"
    fi
else
    fail "ANE kernel compilation failed"
fi

# Test 7: Gradient norms are finite (need accum boundary at step where (step+1)%10==0)
echo "Test 7: Gradient health (no NaN/Inf)"
OUTPUT7=$(./train --scratch --data "$DATA" --lr 4e-4 --warmup 10 --accum 10 \
    --clip 1.0 --steps 20 --time 60 --scale 256.0 --cpu-only --seed 42 2>&1)
if echo "$OUTPUT7" | grep -q "grad_norm"; then
    GNORM=$(echo "$OUTPUT7" | grep "grad_norm" | head -1 | grep -oE 'grad_norm=[0-9.]+' | cut -d= -f2)
    if [ -n "$GNORM" ] && awk "BEGIN {exit !($GNORM > 0 && $GNORM < 1000)}"; then
        pass "Gradient norm=$GNORM (finite)"
    else
        fail "Gradient norm=$GNORM (suspicious)"
    fi
else
    fail "No gradient norm output (expected with accum=10, steps=20)"
fi

# Test 8: Python wrapper (train.py)
echo "Test 8: Python wrapper"
PYOUT=$(python3 -c "
import sys; sys.path.insert(0, '.')
exec(open('train.py').read().split('def main')[0])
validate_config()
print('validate_config: OK')
" 2>&1 || true)
if echo "$PYOUT" | grep -q "validate_config: OK"; then
    pass "train.py validate_config()"
else
    fail "train.py import/validate failed"
fi

# Test 9: MeZO compilation
echo "Test 9: MeZO compilation"
make mezo MODEL=autoresearch >/dev/null 2>&1
if [ -f ./train_mezo ]; then
    pass "MeZO compiles cleanly"
else
    fail "MeZO compilation failed"
fi

# Test 10: MeZO CPU-only forward (step 0 loss)
echo "Test 10: MeZO CPU-only forward"
MEZO_OUT=$(./train_mezo --scratch --data "$DATA" --lr 1e-5 --epsilon 1e-3 \
    --steps 7 --time 30 --cpu-only --seed 42 2>&1)
MEZO_LOSS=$(echo "$MEZO_OUT" | grep "^step 0" | grep -oE 'loss_plus=[0-9.]+' | cut -d= -f2)
if [ -n "$MEZO_LOSS" ] && awk "BEGIN {exit !($MEZO_LOSS > 9.0 && $MEZO_LOSS < 10.5)}"; then
    pass "MeZO step 0 loss=$MEZO_LOSS (expected ~9.7)"
else
    fail "MeZO step 0 loss=$MEZO_LOSS outside range [9.0, 10.5]"
fi

# Test 11: MeZO training stability (loss doesn't diverge)
echo "Test 11: MeZO training stability (200 steps)"
MEZO_OUT2=$(./train_mezo --scratch --data "$DATA" --lr 1e-5 --epsilon 1e-3 \
    --steps 200 --time 60 --cpu-only --seed 42 2>&1)
MEZO_FINAL=$(echo "$MEZO_OUT2" | grep "^final_loss_plus:" | awk '{print $2}')
MEZO_INIT=$(echo "$MEZO_OUT2" | grep "^step 0" | grep -oE 'loss_plus=[0-9.]+' | cut -d= -f2)
if [ -n "$MEZO_FINAL" ] && [ -n "$MEZO_INIT" ] && awk "BEGIN {exit !($MEZO_FINAL < $MEZO_INIT * 2.0 && $MEZO_FINAL > 0)}"; then
    pass "MeZO stable ($MEZO_INIT -> $MEZO_FINAL, no divergence)"
else
    fail "MeZO diverged ($MEZO_INIT -> $MEZO_FINAL)"
fi

# Test 12: MeZO ANE matmul-only mode
echo "Test 12: MeZO ANE mode"
MEZO_ANE=$(./train_mezo --scratch --data "$DATA" --lr 1e-5 --epsilon 1e-3 \
    --steps 7 --time 30 --ane-matmul-only --seed 42 2>&1)
if echo "$MEZO_ANE" | grep -q "Compiled"; then
    MEZO_ANE_LOSS=$(echo "$MEZO_ANE" | grep "^step 0" | grep -oE 'loss_plus=[0-9.]+' | cut -d= -f2)
    if [ -n "$MEZO_ANE_LOSS" ] && awk "BEGIN {exit !($MEZO_ANE_LOSS > 9.0 && $MEZO_ANE_LOSS < 10.5)}"; then
        pass "MeZO ANE: loss=$MEZO_ANE_LOSS"
    else
        fail "MeZO ANE: loss=$MEZO_ANE_LOSS outside range"
    fi
else
    fail "MeZO ANE kernel compilation failed"
fi

# Summary
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
