#!/bin/bash
# Measure actual power draw during AutoANE training.
# Compares CPU-only vs ANE training power consumption.
#
# Requires: sudo (for powermetrics)
# Usage: sudo bash tools/power_benchmark.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_DIR="$SCRIPT_DIR/../training"
RESULTS_DIR="$SCRIPT_DIR/../docs"
DURATION=60  # seconds per measurement

echo "=== AutoANE Power Benchmark ==="
echo "Duration: ${DURATION}s per mode"
echo ""

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script requires sudo for powermetrics."
    echo "Usage: sudo bash tools/power_benchmark.sh"
    exit 1
fi

# Check powermetrics is available
if ! command -v powermetrics &> /dev/null; then
    echo "ERROR: powermetrics not found (should be available on macOS)"
    exit 1
fi

# Build training binary
echo "Building training binary..."
cd "$TRAIN_DIR"
make clean > /dev/null 2>&1
make MODEL=autoresearch > /dev/null 2>&1

# Function to measure power during training
measure_power() {
    local mode_name="$1"
    local train_flags="$2"
    local output_file="/tmp/autoane_power_${mode_name}.log"

    echo ""
    echo "--- Measuring: $mode_name ---"
    echo "  Flags: $train_flags"

    # Start powermetrics in background
    powermetrics --samplers cpu_power,gpu_power,ane_power \
        -i 1000 \
        -o "$output_file" &
    local pm_pid=$!

    # Wait for powermetrics to start
    sleep 2

    # Run training
    cd "$TRAIN_DIR"
    ./train --scratch --time $DURATION --lr 4e-4 $train_flags \
        --data ../tinystories_smollm2_data00.bin > /tmp/autoane_train_${mode_name}.log 2>&1

    # Stop powermetrics
    sleep 1
    kill $pm_pid 2>/dev/null || true
    wait $pm_pid 2>/dev/null || true

    # Parse power readings
    if [ -f "$output_file" ]; then
        echo "  Power readings from $output_file:"

        # Extract CPU power
        local cpu_power=$(grep -i "CPU Power" "$output_file" | grep -oE '[0-9]+\.?[0-9]*' | head -20)
        if [ -n "$cpu_power" ]; then
            local avg_cpu=$(echo "$cpu_power" | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
            local max_cpu=$(echo "$cpu_power" | sort -rn | head -1)
            echo "    CPU Power: avg=${avg_cpu}mW, peak=${max_cpu}mW"
        fi

        # Extract ANE power (if available)
        local ane_power=$(grep -i "ANE Power" "$output_file" | grep -oE '[0-9]+\.?[0-9]*' | head -20)
        if [ -n "$ane_power" ]; then
            local avg_ane=$(echo "$ane_power" | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
            local max_ane=$(echo "$ane_power" | sort -rn | head -1)
            echo "    ANE Power: avg=${avg_ane}mW, peak=${max_ane}mW"
        else
            echo "    ANE Power: not reported (may need --samplers ane_power)"
        fi

        # Extract package/total power
        local pkg_power=$(grep -i "Package Power\|Combined Power\|Total Power" "$output_file" | grep -oE '[0-9]+\.?[0-9]*' | head -20)
        if [ -n "$pkg_power" ]; then
            local avg_pkg=$(echo "$pkg_power" | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
            echo "    Package Power: avg=${avg_pkg}mW"
        fi

        # Training stats
        local steps=$(grep "num_steps" /tmp/autoane_train_${mode_name}.log | grep -oE '[0-9]+' | tail -1)
        local val_loss=$(grep "val_loss" /tmp/autoane_train_${mode_name}.log | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        echo "    Training: steps=$steps, val_loss=$val_loss"
    else
        echo "  ERROR: powermetrics output not found"
    fi
}

# Idle baseline (no training)
echo ""
echo "--- Measuring: idle baseline (10s) ---"
powermetrics --samplers cpu_power,gpu_power,ane_power \
    -i 1000 -n 10 \
    -o /tmp/autoane_power_idle.log 2>/dev/null
idle_power=$(grep -i "Package Power\|CPU Power" /tmp/autoane_power_idle.log | grep -oE '[0-9]+\.?[0-9]*' | head -10 | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "N/A"}')
echo "  Idle power: ~${idle_power}mW"

# CPU-only training
measure_power "cpu_only" "--cpu-only"

# ANE matmul-only training
measure_power "ane_matmul" "--ane-matmul-only"

# ANE full training
measure_power "ane_full" ""

# Summary
echo ""
echo "=== Summary ==="
echo "Raw logs saved to /tmp/autoane_power_*.log"
echo "Training logs saved to /tmp/autoane_train_*.log"
echo ""
echo "To view detailed readings:"
echo "  cat /tmp/autoane_power_cpu_only.log | grep -i power"
echo "  cat /tmp/autoane_power_ane_matmul.log | grep -i power"
echo "  cat /tmp/autoane_power_ane_full.log | grep -i power"
