# Memory-pressure guard — war stories (moved out of CLAUDE.md)

Full histories: live failures, diagnosis ladders, dead ends. The distilled RULES live in the root CLAUDE.md "Rules" section — when a rule changes, update the story here too. New gotchas in this domain: add the 1-3 line rule to root, the full story here.

### The watchdog must start BEFORE the scheduler loads the model, not after

Original wiring spawned the memwatch thread AFTER `Scheduler.init` returned. That missed the single most dangerous window: loading a 100+ GB model is exactly when available RAM is most likely to dip under the watermark, and on macOS there is no OOM killer to save the box. If the guard only came online post-load, a machine already under pressure at load time would run the model with zero protection until the next tick. Fix: the memwatch thread is spawned in `server.serve` immediately before `Scheduler.init`, with the same LIFO `defer` join/stop pattern as the gauge sampler. The evict path already guards `global_scheduler orelse continue`, so ticking before/during load is safe; `std.process.exit(0)` on `.exit` is unconditional. This also lets the integration test fire mid-load without needing the model to finish loading.

### A `0` sentinel for "countdown not started" collides with `now_ms == 0`

The state machine used `below_since_ms: i64 = 0` to mean "not currently under the bar". But the monotonic clock (`std.Io.Timestamp.now`) starts at 0, so the very first tick (`now_ms == 0`) reads the sentinel as "already started at t0" and computes `now - since >= exit_after` immediately — firing `.exit` one window early, or (when the first reading was over the bar) masking a real dip by resetting to 0. Fix: `below_since_ms: ?i64 = null`. `null` = window closed; a present value is the real start. Four of the 17 unit tests caught this on revert.
- **Rule: never use a value from the domain (a 0 timestamp) as a sentinel for an unset optional — use `?T`.**

### Probe file for testing without ever touching real RAM

`MLX_SERVE_MEM_PROBE_FILE` points at a file holding available RAM in GB as a float. `PressureWatch.availableBytes` reads it instead of `status.getAvailableMemBytes()` when set. The integration test `tests/test_mem_pressure.sh` writes fake readings (9.0 / 13.0) and consumes zero bytes of real RAM; it is meant to be run on a host with a small loadable model. On a machine whose only model is larger than free RAM, the load preflight refuses before the guard can tick, so the test SKIPs with a hint — that is an environment limit, not a guard failure.
- **Rule: drive the guard through the probe in CI; never manufacture real memory pressure to test it (it shortens RAM lifetime and is indistinguishable from a genuine OOM event).**

### Defaults & actions

- watermark = `max(10 GB, RAM/8)` → 16 GB on a 128 GB box; hysteresis = `max(2 GB, RAM/64)`; grace = 30 s (`--memory-pressure-exit-after <ms>`).
- `--memory-pressure auto` restores defaults; `--memory-pressure off` disables the guard entirely.
- On `.evict`: the LRU resident model unloads via `Scheduler.unloadModel`. On `.exit`: the process calls `exit(0)` so you restart the LLM server yourself — macOS has no OOM kill, so the guard IS the server's safety net.

Guards: `src/mem_pressure.zig` (17 unit tests, including the `?i64` sentinel and the evict/exit transitions) + `tests/test_mem_pressure.sh` (probe-driven; SKIPs on oversized-only hosts with a hint to pass a small model).
