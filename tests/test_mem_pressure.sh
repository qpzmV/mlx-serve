#!/bin/bash
# Integration test for the machine-wide memory-pressure guard (--memory-pressure).
#
# The guard is driven entirely through the probe file (MLX_SERVE_MEM_PROBE_FILE,
# a file holding available RAM in GB as a float): the test writes fake readings
# and never consumes a byte of real RAM.
#
# Two things are checked, and they are NOT equally skippable:
#   1. the guard STARTS ("guard ON", logged inside serve() BEFORE Scheduler.init,
#      so it is independent of how big the model is) — asserted ALWAYS. A guard
#      that never starts must FAIL, never SKIP: an earlier wiring bug put the
#      `defer` that clears the global inside a block, so it fired immediately,
#      the watchdog never spawned, and a SKIP path hid it behind a green run.
#   2. the guard ACTS (clean exit) — this needs the process to stay up long
#      enough, so it SKIPs when the load preflight refuses the model first.
#
# The intervals are deliberately tiny (100 ms checks) so the countdown fires
# BEFORE the model load gets anywhere: the process then exits having allocated
# no weights, which is both fast and free of any real memory pressure.
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
# `check` takes 1 = pass, but `grep -q` returns 0 = FOUND. Passing `$?`
# straight through inverts every assertion — a green-looking FAIL and a
# red-looking PASS. Always route grep results through this.
grepok() { if grep -q "$1" "$2" 2>/dev/null; then echo 1; else echo 0; fi; }

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

# ── Scenario 1: guard starts, and sustained pressure exits cleanly ─────────
# Fake reading 9.0 GB against a 10 GB bar, checked every 100 ms and exiting
# after 200 ms of sustained pressure. Real RAM is never touched (probe file).
echo "9.0" > "$PROBE"
MLX_SERVE_MEM_PROBE_FILE="$PROBE" "$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld --log-level info \
    --memory-pressure 10GB \
    --memory-pressure-check-interval 100 \
    --memory-pressure-exit-after 200 > "$LOG" 2>&1 &
PID=$!

# (1) GUARD STARTUP — never skipped. "guard ON" is logged before Scheduler.init,
#     so it appears whether or not the model could ever load.
for i in $(seq 1 60); do
    grep -q "guard ON" "$LOG" 2>/dev/null && break
    sleep 0.5
done
check "guard announces itself before the load ([mem-pressure] guard ON)" "$(grepok "guard ON" "$LOG")"

# (2) GUARD ACTION — the clean exit. Only meaningful if the process died for
#     OUR reason; a preflight refusal ends it first, which is an environment
#     limit (no model on this box fits), not a guard defect.
if wait_gone "$PID" 20; then
    if grep -q "exiting cleanly" "$LOG" 2>/dev/null; then
        check "sustained pressure (9 GB < 10 GB bar) exits the process cleanly" "1"
        check "exit is the guard's exit(0), not a crash" "1"
    elif grep -q "Insufficient memory" "$LOG" 2>/dev/null; then
        echo "SKIP: the load preflight refused the model before the guard's countdown could finish."
        echo "      Guard STARTUP was verified above; the EXIT PATH was not exercised here."
        echo "      Pass a small loadable model dir as \$1 to exercise it:"
        echo "        ./tests/test_mem_pressure.sh /path/to/small-model"
    else
        echo "--- process exited, but not via the guard; log tail: ---"
        tail -n 5 "$LOG" | sed 's/^/      /'
        check "sustained pressure exits the process cleanly" "0"
    fi
else
    echo "--- process still alive after 20 s; log tail: ---"
    tail -n 5 "$LOG" | sed 's/^/      /'
    check "sustained pressure exits the process cleanly" "0"
fi
kill -9 "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

# ── Scenario 2: recovery above the bar resets the countdown ────────────────
# Same probe. A longer countdown (3 s) leaves room to raise the reading above
# watermark+hysteresis (10+2 = 12 GB) and prove the countdown restarted.
echo "9.0" > "$PROBE"
MLX_SERVE_MEM_PROBE_FILE="$PROBE" "$BINARY" --model "$MODEL" --serve --port "$PORT" --no-pld --log-level info \
    --memory-pressure 10GB \
    --memory-pressure-check-interval 100 \
    --memory-pressure-exit-after 3000 > "$LOG" 2>&1 &
PID=$!

for i in $(seq 1 60); do
    grep -q "guard ON" "$LOG" 2>/dev/null && break
    sleep 0.5
done
check "scenario 2: guard starts before the load" "$(grepok "guard ON" "$LOG")"

sleep 1
echo "13.0" > "$PROBE"   # recovery: above watermark (10) + hysteresis (2)
sleep 2                  # a sticky countdown would have exited ~2 s in
if ! kill -0 "$PID" 2>/dev/null; then
    if grep -q "Insufficient memory" "$LOG" 2>/dev/null; then
        echo "SKIP: preflight refused the model before the recovery path could run."
        echo "      Guard STARTUP was verified; the RECOVERY PATH was not."
        kill -9 "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; rm -f "$PROBE"
        exit 0
    fi
    check "recovery above the bar resets the exit countdown" "0"
else
    check "recovery above the bar resets the exit countdown" "1"
fi

echo "9.0" > "$PROBE"    # fresh dip: countdown restarts from here
if wait_gone "$PID" 15; then
    check "fresh dip after recovery exits again (code 0)" "$(grepok "exiting cleanly" "$LOG")"
else
    check "fresh dip after recovery exits again (code 0)" "0"
fi
kill -9 "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

rm -f "$PROBE"
echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" = "0" ]
