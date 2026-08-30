# Memory-pressure guard — war stories (moved out of CLAUDE.md)

Full histories: live failures, diagnosis ladders, dead ends. The distilled RULES live in the root CLAUDE.md "Rules" section — when a rule changes, update the story here too. New gotchas in this domain: add the 1-3 line rule to root, the full story here.

### `defer` binds to its enclosing BLOCK, not to the function — the watchdog that never ran

The guard was assembled as `server_mod.g_mem_pressure = .{...};` inside a `{ }` block in `main`, and `defer server_mod.g_mem_pressure = null;` was written **inside that same block**. In Zig, `defer` fires when its enclosing *block* ends, not when the function returns — so the global was nulled microseconds after being set, long before `server.serve()` read it. The result was the worst possible failure mode: no error, no crash, no log line. "Guard off" and "guard cleared by accident" are byte-for-byte indistinguishable from the outside.

It shipped because the integration test's only guard assertion sat behind a SKIP. Proof it was dead: a 25-line run log containing `Insufficient memory` (emitted inside `serve()`, so `serve()` definitely ran) and **zero** occurrences of `mem-pressure` (the `guard ON` banner, printed before `Scheduler.init`). Fix: move the `defer` outside the block — there is now a comment at the site saying exactly why it must not move back — and make the test assert `guard ON` **unconditionally**, before anything that can SKIP.
- **Rule: a `defer` that tears down a global must sit at the same scope as the code that consumes the global — and never let an integration test's first assertion be skippable.**

### The watchdog must start BEFORE the scheduler loads the model, not after

Original wiring spawned the memwatch thread AFTER `Scheduler.init` returned. That missed the single most dangerous window: loading a 100+ GB model is exactly when available RAM is most likely to dip under the watermark, and on macOS there is no OOM killer to save the box. If the guard only came online post-load, a machine already under pressure at load time would run the model with zero protection until the next tick. Fix: the memwatch thread is spawned in `server.serve` immediately before `Scheduler.init`, with the same LIFO `defer` join/stop pattern as the gauge sampler. The evict path already guards `global_scheduler orelse continue`, so ticking before/during load is safe; `std.process.exit(0)` on `.exit` is unconditional. This also lets the integration test fire mid-load without needing the model to finish loading.

### A `0` sentinel for "countdown not started" collides with `now_ms == 0`

The state machine used `below_since_ms: i64 = 0` to mean "not currently under the bar". But the monotonic clock (`std.Io.Timestamp.now`) starts at 0, so the very first tick (`now_ms == 0`) reads the sentinel as "already started at t0" and computes `now - since >= exit_after` immediately — firing `.exit` one window early, or (when the first reading was over the bar) masking a real dip by resetting to 0. Fix: `below_since_ms: ?i64 = null`. `null` = window closed; a present value is the real start. Four of the 17 unit tests caught this on revert.
- **Rule: never use a value from the domain (a 0 timestamp) as a sentinel for an unset optional — use `?T`.**

### Probe file for testing without ever touching real RAM

`MLX_SERVE_MEM_PROBE_FILE` points at a file holding available RAM in GB as a float. `PressureWatch.availableBytes` reads it instead of `status.getAvailableMemBytes()` when set. The integration test `tests/test_mem_pressure.sh` writes fake readings (9.0 / 13.0) and consumes zero bytes of real RAM; it is meant to be run on a host with a small loadable model. On a machine whose only model is larger than free RAM, the load preflight refuses before the guard can tick, so the test SKIPs with a hint — that is an environment limit, not a guard failure.
- **Rule: drive the guard through the probe in CI; never manufacture real memory pressure to test it (it shortens RAM lifetime and is indistinguishable from a genuine OOM event).**

### Throttle before you read, not after

The loop originally slept, read memory, and only then asked `tick()` whether the interval had elapsed. Every wake-up therefore performed the read (a sysctl, plus an allocation in the probe path) even on the 39 out of 40 wake-ups it would discard. `PressureWatch.checkDue(now_ms)` is `*const` — it does not advance `last_check` — so the loop can `continue` before touching memory at all. One consequence worth knowing: the exit countdown is only ever advanced on a real check, so `--memory-pressure-exit-after` is accurate to one *check* interval, not one poll interval (default 5 s / 100 ms).

### Defaults & actions

- watermark = `max(min(10 GB, RAM/4), RAM/8)`; hysteresis = `max(min(2 GB, RAM/32), RAM/64)`; grace = 30 s (`--memory-pressure-exit-after <ms>`); check interval = 5 s (`--memory-pressure-check-interval`, rejected below 100 ms).
- The watermark is **adaptive**, not a flat 16 GB: the `RAM/4` term is what keeps a small machine from reserving a watermark it can never recover from, and the `10 GB` floor stops a big machine from sitting on a watermark too small to survive a single large load.

  | total RAM | RAM/4 | RAM/8 | watermark | winner |
  |---|---|---|---|---|
  | 16 GB | 4 GB | 2 GB | **4 GB** | RAM/4 |
  | 32 GB | 8 GB | 4 GB | **8 GB** | RAM/4 |
  | 64 GB | 16 GB | 8 GB | **10 GB** | 10 GB floor |
  | 128 GB | 32 GB | 16 GB | **16 GB** | RAM/8 |

  (128 GB still lands on 16 GB, so the box this was designed on is unchanged.)
- `--memory-pressure auto` restores defaults; `--memory-pressure off` disables the guard entirely.
- On `.evict`: the LRU resident model unloads via `Scheduler.unloadModel`. On `.exit`: the process calls `exit(0)` so you restart the LLM server yourself — macOS has no OOM kill, so the guard IS the server's safety net.

Guards: `src/mem_pressure.zig` (18 unit tests, including the `?i64` sentinel, the scaled defaults, `checkDue`, and the evict/exit transitions) + `tests/test_mem_pressure.sh` (probe-driven; asserts `guard ON` with no SKIP path, and only the "guard acted" half may SKIP on a host with no loadable model).
