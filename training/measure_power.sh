#!/bin/bash
# measure_power.sh — ANE power measurement script (requires sudo)
# Uses powermetrics to sample ANE/CPU/GPU power during training
#
# Usage:
#   sudo ./measure_power.sh [duration_seconds]
#   Default duration: 30 seconds
#
# This script:
# 1. Starts powermetrics in background, sampling every 1s
# 2. Runs benchmark (ANE matmul) for the specified duration
# 3. Stops powermetrics and parses ANE/CPU power readings
# 4. Outputs summary statistics

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DURATION="${1:-30}"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script requires sudo to access powermetrics"
    echo "Usage: sudo $0 [duration_seconds]"
    exit 1
fi

POWER_LOG=$(mktemp)
BENCH_LOG=$(mktemp)

echo "=== ANE Power Measurement ==="
echo "Duration: ${DURATION}s"
echo "Power log: $POWER_LOG"
echo ""

# Start powermetrics in background (1s sampling interval)
powermetrics -i 1000 --samplers cpu_power,ane_power,gpu_power -a /dev/null \
    -f plist -o "$POWER_LOG" &
PM_PID=$!
echo "powermetrics started (PID $PM_PID)"

# Small delay for first sample
sleep 1

# Run ANE benchmark for the duration
echo "Running ANE benchmark for ${DURATION}s..."
if [ -x "${SCRIPT_DIR}/benchmark" ]; then
    # Use the benchmark binary if available
    timeout "$DURATION" "${SCRIPT_DIR}/benchmark" --skip-thermal > "$BENCH_LOG" 2>&1 || true
else
    echo "No benchmark binary found. Building..."
    cd "$SCRIPT_DIR"
    xcrun clang -O2 -DACCELERATE_NEW_LAPACK -framework Foundation -framework IOSurface -framework Accelerate \
        -isysroot "$(xcrun --show-sdk-path)" -fobjc-arc -include dlfcn.h -o benchmark benchmark.m
    timeout "$DURATION" ./benchmark --skip-thermal > "$BENCH_LOG" 2>&1 || true
fi

# Stop powermetrics
echo "Stopping powermetrics..."
kill "$PM_PID" 2>/dev/null || true
wait "$PM_PID" 2>/dev/null || true

# Parse power readings
echo ""
echo "=== Power Readings ==="
echo "(Parsing $POWER_LOG)"

# Extract ANE power values from plist output
# powermetrics outputs XML plist with ane_energy or ane_power fields
if grep -q "ane_power" "$POWER_LOG" 2>/dev/null; then
    echo ""
    echo "ANE Power readings found:"
    grep "ane_power" "$POWER_LOG" | head -20
elif grep -q "ane_energy" "$POWER_LOG" 2>/dev/null; then
    echo ""
    echo "ANE Energy readings found:"
    grep "ane_energy" "$POWER_LOG" | head -20
else
    echo ""
    echo "No ANE-specific power readings found in this sampling mode."
    echo "Try: sudo powermetrics -i 1000 --samplers all -n 5"
    echo ""
    echo "Available fields in log:"
    grep -o '<key>[^<]*</key>' "$POWER_LOG" | sort -u | head -30
fi

# Also check CPU power
if grep -q "cpu_power" "$POWER_LOG" 2>/dev/null; then
    echo ""
    echo "CPU Power readings:"
    grep "cpu_power" "$POWER_LOG" | head -10
fi

# Cleanup
echo ""
echo "Raw power log saved at: $POWER_LOG"
echo "Benchmark log saved at: $BENCH_LOG"
echo ""
echo "For manual inspection:"
echo "  plutil -p $POWER_LOG | grep -i ane"
echo "  plutil -p $POWER_LOG | grep -i power"
