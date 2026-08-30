#!/bin/bash
# Integration test for the machine-wide memory-pressure guard (--memory-pressure).
#
# The guard is driven entirely through the probe file (MLX_SERVE_MEM_PROBE_FILE,
# a file holding available RAM in GB as a float): the test writes fake readings
# and never consumes a byte of real RAM. Two scenarios:
#   1. sustained pressure → the process exits cleanly with code 0;
#   2. recovery above the watermark+ hysteresis resets the exit countdown.
#
# Usage: ./tests/test_mem_pressure.sh [model_dir] [port]
#   Needs any loadable model dir for CPU-side setup (config.json + tokenizer).

set -u

MODEL="${1:-$HOME/.mlx-serve/models/mlx-community/gemma-4-e2b-it-8bit}"
PORT="${2:-11294}"
BINARY="${BINARY:-./zig-out/bin/mlx-serve}"
PROBE=/tmp/test_mem_pressure_probe
LOG=/tmp/test_mem_pressure.log
PASS=0
FAIL=0

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"; fi
}

if [ ! -d "$MODEL" ]; then
    echo "SKIP: model dir not found: $MODEL"
    exit 0
fi
if [ ! -x "$BINARY" ]; then
    echo "SKIP: binary not found: $BINARY (build with: zig build -Doptimize=ReleaseFast)"
    exit 0
fi

wait_gone() { # pid timeout_s → 0 if process exited within timeout
    local pid="$1" timeout="$2" i=0
    while [ $i -lt $((timeout * 2)) ]; do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.5; i=$((i + 1))
    done
    return 1
}

# ── Scenario 1: sustained pressure → clean exit(0) (probe-driven, fake RAM) ─
# The probe file holds a fake available-RAM reading (in GB); the guard never
# touches real memory, so this exercises the full exit(0) path with NO real
# RAM pressure. Bar = 10 GB; probe stays at 9.0 (under) so the 3 s countdown
# fires. The watchdog starts BEFORE Scheduler.init, so "guard ON" is logged
# before the load even begins.
echo "9.0" > "$PROBE"
MLX_SERVE_MEM_PROBE_FILE="$PROBE" "$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld --log-level info \
    --memory-pressure 10GB --memory-pressure-exit-after 3000 > "$LOG" 2>&1 &
PID=$!

# If the machine cannot hold the model, the load preflight refuses and the
# process ends before the guard can be exercised. That is an environment
# limit (no fitting model here), not a guard failure → SKIP with a hint.
# Pass a small loadable model as $1 to exercise the full path on a real host.
for i in $(seq 1 240); do
    grep -q "Insufficient memory" "$LOG" 2>/dev/null && {
        echo "SKIP: model '${MODEL}' does not fit this machine's available RAM (preflight refused)."
        echo "      Pass a small loadable model dir as \$1 to exercise the full guard path:"
        echo "        ./tests/test_mem_pressure.sh /path/to/small-model"
        kill -9 "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; rm -f "$PROBE"
        exit 0
    }
    grep -q "guard ON" "$LOG" 2>/dev/null && break
    sleep 0.5
done
grep -q "guard ON" "$LOG" 2>/dev/null
check "guard announces itself before the load ([mem-pressure] guard ON)" "$?"
wait_gone "$PID" 15
check "sustained pressure (9 GB < 10 GB bar for 3 s) exits the process" "$?"
grep -q "exiting cleanly" "$LOG" 2>/dev/null
check "exit is the guard's clean exit(0), not a crash" "$?"
kill -9 "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

# ── Scenario 2: recovery resets the countdown (probe-driven, fake RAM) ─────
# Same probe file; a successful load is NOT required — the probe drives the
# guard readings while the process is simply alive.
echo "9.0" > "$PROBE"
MLX_SERVE_MEM_PROBE_FILE="$PROBE" "$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld --log-level info \
    --memory-pressure 10GB --memory-pressure-exit-after 4000 > "$LOG" 2>&1 &
PID=$!
sleep 2          # t≈2 s: countdown running since t0 (would exit at t≈4 s)
echo "13.0" > "$PROBE"  # recovery: above watermark (10) + hysteresis (2)
sleep 4          # t≈6 s: a sticky countdown would have fired at t≈4 s
if kill -0 "$PID" 2>/dev/null; then
    check "recovery above the bar resets the exit countdown" "1"
else
    check "recovery above the bar resets the exit countdown" "0"
fi
echo "9.0" > "$PROBE"   # fresh dip: countdown restarts from here
wait_gone "$PID" 15
check "fresh dip after recovery exits again (code 0)" "$?"
kill -9 "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

rm -f "$PROBE"
echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" = "0" ]
